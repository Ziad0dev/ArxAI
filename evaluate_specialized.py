#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to evaluate a trained model on specialized evaluation sets
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import needed functions from train.py
from train import load_domain_mapping, ResearchDataset, ResearchTransformer, convert_to_serializable, CustomDataCollator
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_model(model_dir):
    """Load the trained model from checkpoint directory."""
    # Determine model path
    if os.path.exists(os.path.join(model_dir, "best")):
        model_path = os.path.join(model_dir, "best")
        logger.info(f"Loading best model from {model_path}")
    else:
        # Find the latest checkpoint
        checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {model_dir}")
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        model_path = os.path.join(model_dir, latest_checkpoint)
        logger.info(f"Loading latest checkpoint from {model_path}")
    
    # Load training state to get configuration
    training_state_path = os.path.join(model_path, "training_state.json")
    if os.path.exists(training_state_path):
        with open(training_state_path, "r") as f:
            training_state = json.load(f)
            config = training_state.get("args", {})
    else:
        # Use default configuration if training state is not available
        logger.warning(f"Training state not found at {training_state_path}, using default configuration")
        config = {
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 6,
            "d_ff": 3072,
            "dropout": 0.1
        }
    
    # Load domain mapping
    domain_file = config.get("domain_file", "data/domains/domain_mapping.json")
    domain_mapping = load_domain_mapping(domain_file)
    num_domains = get_num_domains(domain_mapping)
    
    # Load tokenizer
    tokenizer_path = os.path.join(model_path)
    if os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        # Fall back to the original tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.get("model_name", "sentence-transformers/all-mpnet-base-v2"))
    
    # Create model
    model = ResearchTransformer(
        model_name=config.get("model_name", "sentence-transformers/all-mpnet-base-v2"),
        d_model=config.get("d_model", 768),
        n_heads=config.get("n_heads", 12),
        n_layers=config.get("n_layers", 6),
        d_ff=config.get("d_ff", 3072),
        dropout=config.get("dropout", 0.1),
        num_domains=num_domains
    )
    
    # Load state dict
    model_pt_path = os.path.join(model_path, "model.pt")
    if not os.path.exists(model_pt_path):
        raise ValueError(f"Model file not found at {model_pt_path}")
    
    model_state_dict = torch.load(model_pt_path, map_location="cpu")
    model.load_state_dict(model_state_dict)
    
    return model, tokenizer, domain_mapping, config

def get_num_domains(domain_mapping):
    """Get the number of domains from the mapping."""
    domains = domain_mapping['domains']
    # Find the maximum domain ID plus 1 to ensure all IDs are covered
    max_domain_id = max(domain['id'] for domain in domains) + 1
    logger.info(f"Using {max_domain_id} domain embeddings based on maximum domain ID")
    return max_domain_id

def evaluate_model(model, tokenizer, eval_dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()
    
    # Track metrics
    total_loss = 0
    total_loss_score = 0
    total_loss_relevance = 0
    total_samples = 0
    
    # Loss functions
    loss_fn_score = torch.nn.MSELoss()
    loss_fn_relevance = torch.nn.BCEWithLogitsLoss()
    
    # Collect all predictions and targets
    all_score_preds = []
    all_score_targets = []
    all_relevance_preds = []
    all_relevance_targets = []
    all_paper_ids = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get target values
            target_score = batch["target_score"]
            target_relevance = batch["target_relevance"]

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                domain_id=batch["domain_id"]
            )
            
            # Calculate loss
            loss_score = loss_fn_score(outputs["score"], target_score)
            loss_relevance = loss_fn_relevance(outputs["relevance_logits"], target_relevance)
            loss = loss_score + loss_relevance
            
            # Update metrics
            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_loss_score += loss_score.item() * batch_size
            total_loss_relevance += loss_relevance.item() * batch_size
            total_samples += batch_size
            
            # Collect predictions and targets
            all_score_preds.extend(outputs["score"].cpu().numpy())
            all_score_targets.extend(target_score.cpu().numpy())
            all_relevance_preds.extend((outputs["relevance_logits"] >= 0).cpu().numpy().astype(int))
            all_relevance_targets.extend(target_relevance.cpu().numpy().astype(int))
            all_paper_ids.extend(batch["paper_id"])
    
    # Calculate aggregate metrics
    metrics = {}
    metrics["loss"] = total_loss / total_samples if total_samples > 0 else 0
    metrics["loss_score"] = total_loss_score / total_samples if total_samples > 0 else 0
    metrics["loss_relevance"] = total_loss_relevance / total_samples if total_samples > 0 else 0
    
    # Calculate regression metrics
    metrics["mae"] = mean_absolute_error(all_score_targets, all_score_preds)
    
    # Calculate classification metrics
    metrics["accuracy"] = accuracy_score(all_relevance_targets, all_relevance_preds)
    metrics["precision"] = precision_score(all_relevance_targets, all_relevance_preds, zero_division=0)
    metrics["recall"] = recall_score(all_relevance_targets, all_relevance_preds, zero_division=0)
    metrics["f1"] = f1_score(all_relevance_targets, all_relevance_preds, zero_division=0)
    
    # Calculate confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(all_relevance_targets, all_relevance_preds).tolist()
    
    # Return serializable metrics and predictions
    return {
        "metrics": metrics,
        "predictions": {
            "paper_ids": all_paper_ids,
            "score_preds": [float(x) for x in all_score_preds],
            "score_targets": [float(x) for x in all_score_targets],
            "relevance_preds": [int(x) for x in all_relevance_preds],
            "relevance_targets": [int(x) for x in all_relevance_targets]
        }
    }

def main():
    """Run evaluation on specialized test sets."""
    parser = argparse.ArgumentParser(description="Evaluate model on specialized test sets")
    
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the trained model")
    parser.add_argument("--eval_dir", type=str, default="data/enhanced/eval", help="Directory containing specialized evaluation sets")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluations", help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model, tokenizer, and domain mapping
    model, tokenizer, domain_mapping, config = load_model(args.model_dir)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Get list of evaluation files
    eval_files = [f for f in os.listdir(args.eval_dir) if f.endswith(".json")]
    if not eval_files:
        logger.error(f"No evaluation files found in {args.eval_dir}")
        return
    
    # Create data collator
    data_collator = CustomDataCollator(tokenizer=tokenizer)
    
    # Evaluate on each specialized test set
    all_results = {}
    for eval_file in eval_files:
        eval_name = os.path.splitext(eval_file)[0]
        logger.info(f"Evaluating on {eval_name} test set")
        
        # Load dataset
        eval_path = os.path.join(args.eval_dir, eval_file)
        eval_dataset = ResearchDataset(
            eval_path,
            tokenizer,
            max_seq_length=config.get("max_seq_length", 512),
            is_training=False,
            valid_domain_ids=[domain['id'] for domain in domain_mapping['domains']]
        )
        
        # Create data loader
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_collator
        )
        
        # Evaluate model
        logger.info(f"Evaluating model on {len(eval_dataset)} samples")
        eval_results = evaluate_model(model, tokenizer, eval_dataloader, device)
        
        # Print key metrics
        metrics = eval_results["metrics"]
        logger.info(f"Evaluation results for {eval_name}:")
        logger.info(f"  - Loss: {metrics['loss']:.4f}")
        logger.info(f"  - MAE (impact score): {metrics['mae']:.4f}")
        logger.info(f"  - Accuracy (relevance): {metrics['accuracy']:.4f}")
        logger.info(f"  - F1 Score (relevance): {metrics['f1']:.4f}")
        logger.info(f"  - Confusion Matrix: {metrics['confusion_matrix']}")
        
        # Save results
        output_path = os.path.join(args.output_dir, f"{eval_name}_results.json")
        with open(output_path, "w") as f:
            json.dump(convert_to_serializable(eval_results), f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")
        
        # Store in overall results
        all_results[eval_name] = {
            "metrics": metrics,
            "num_samples": len(eval_dataset)
        }
    
    # Save summary of all results
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    logger.info(f"Saved evaluation summary to {summary_path}")
    
    # Print comparison table
    logger.info("\n=== Evaluation Comparison ===")
    logger.info(f"{'Test Set':<20} {'Samples':<8} {'MAE':<8} {'Accuracy':<8} {'F1':<8}")
    for name, results in all_results.items():
        metrics = results["metrics"]
        samples = results["num_samples"]
        logger.info(f"{name:<20} {samples:<8} {metrics['mae']:<8.4f} {metrics['accuracy']:<8.4f} {metrics['f1']:<8.4f}")

if __name__ == "__main__":
    main() 