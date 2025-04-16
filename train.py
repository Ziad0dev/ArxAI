#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for Research Transformer model.
Supports distributed training, mixed precision, and checkpointing.
"""

import os
import json
import time
import random
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score

from transformers import (
    AutoTokenizer, 
    AutoModel,
    get_linear_schedule_with_warmup,
    set_seed,
    DataCollatorWithPadding
)

from tqdm import tqdm
import wandb

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

# Set environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Research Transformer model")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data")
    parser.add_argument("--domain_file", type=str, required=True, help="Path to domain mapping file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model checkpoints")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Cache directory for preprocessed data")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2", 
                        help="Base pretrained model to use")
    parser.add_argument("--d_model", type=int, default=768, help="Model embedding dimension")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=3072, help="Dimension of feedforward layer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenizer")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of updates steps to accumulate gradients for")
    
    # Checkpointing arguments
    parser.add_argument("--save_steps", type=int, default=1000, help="Save model every X steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate model every X steps")
    parser.add_argument("--save_total_limit", type=int, default=5, 
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume training from checkpoint")
    
    # Distributed training arguments
    parser.add_argument("--use_ddp", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    # Optimization arguments
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", help="Track experiments with Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="research-transformer", 
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    
    return parser.parse_args()

def set_environment(args):
    """Set up the training environment."""
    # Set random seed for reproducibility
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Initialize distributed training if required
    if args.use_ddp:
        if args.local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            args.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            dist.init_process_group(backend="nccl")
            args.n_gpu = 1
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    args.device = device
    
    # Initialize W&B logging
    if args.use_wandb and (not args.use_ddp or args.local_rank == 0):
        run_name = f"rt-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)
        )
    
    logger.info(f"Using device: {device}")
    if args.n_gpu > 0:
        logger.info(f"Number of GPUs: {args.n_gpu}")
    
    return args

def load_domain_mapping(domain_file):
    """Load domain mapping from file."""
    with open(domain_file, 'r', encoding='utf-8') as f:
        domain_mapping = json.load(f)
    
    logger.info(f"Loaded domain mapping with {len(domain_mapping['domains'])} domains")
    for domain in domain_mapping['domains']:
        logger.info(f"Domain {domain['id']}: {domain['name']}")
    
    # Return the full mapping
    return domain_mapping

def get_num_domains(domain_mapping):
    """Get the number of domains from the mapping."""
    domains = domain_mapping['domains']
    # Find the maximum domain ID plus 1 to ensure all IDs are covered
    max_domain_id = max(domain['id'] for domain in domains) + 1
    logger.info(f"Using {max_domain_id} domain embeddings based on maximum domain ID")
    return max_domain_id

class ResearchDataset(Dataset):
    """Dataset for research papers."""
    def __init__(self, data_path, tokenizer, max_seq_length=512, is_training=True, valid_domain_ids=None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_training = is_training
        self.valid_domain_ids = valid_domain_ids or []
        
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Filter out papers with invalid domain IDs
        if self.valid_domain_ids:
            filtered_data = []
            for item in self.data:
                domain_id = item.get("domain_id", 0)
                if domain_id in self.valid_domain_ids:
                    filtered_data.append(item)
                else:
                    logger.warning(f"Paper {item.get('id', 'unknown')} has invalid domain_id {domain_id}, using default domain 0")
                    item["domain_id"] = 0  # Set to default domain
                    filtered_data.append(item)
            
            logger.info(f"Loaded {len(self.data)} examples, {len(filtered_data)} with valid domain IDs")
            self.data = filtered_data
        else:
            logger.info(f"Loaded {len(self.data)} examples (no domain validation)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process text
        title = item.get("title", "")
        abstract = item.get("abstract", "")
        text = f"{title} {abstract}"
        
        # Tokenize (REMOVE PADDING/TRUNCATION HERE)
        inputs = self.tokenizer(
            text,
            # max_length=self.max_seq_length, # Let collator handle max length
            # padding="max_length", # Let collator handle padding
            truncation=True, # Keep truncation at tokenizer level if needed based on absolute max model len
            return_tensors="pt"
        )
        
        # Extract as individual tensors and squeeze the batch dimension
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # Get domain label if available
        domain_id = item.get("domain_id", 0)
        
        # --- Get target values (NEW) ---
        # Assume 'impact_score' is a float, default to 0.0 if missing
        impact_score = item.get("impact_score", 0.0)
        if impact_score is None: # Handle potential null values in JSON
            logger.warning(f"Paper {item.get('id', 'unknown')} has missing impact_score, using 0.0")
            impact_score = 0.0

        # Assume 'is_primary_domain' is binary (0 or 1), default to 0 if missing
        is_primary_domain = item.get("is_primary_domain", 0)
        if is_primary_domain is None: # Handle potential null values in JSON
             logger.warning(f"Paper {item.get('id', 'unknown')} has missing is_primary_domain, using 0")
             is_primary_domain = 0
        # --- End New ---
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "domain_id": torch.tensor(domain_id, dtype=torch.long),
            "paper_id": item.get("id", ""),
            # --- Add target tensors (NEW) ---
            "target_score": torch.tensor(impact_score, dtype=torch.float),
            "target_relevance": torch.tensor(is_primary_domain, dtype=torch.float) # BCEWithLogitsLoss expects float
            # --- End New ---
        }

class ResearchTransformer(nn.Module):
    """Research Transformer model for scientific paper analysis."""
    def __init__(self, 
                 model_name,
                 d_model=768,
                 n_heads=12,
                 n_layers=6,
                 d_ff=3072,
                 dropout=0.1,
                 num_domains=10):
        super().__init__()
        
        # Load base transformer model
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Get the base model's output dimension
        self.base_output_dim = self.base_model.config.hidden_size
        logger.info(f"Base model output dimension: {self.base_output_dim}")
        
        # Add projection layer if dimensions don't match
        self.needs_projection = self.base_output_dim != d_model
        if self.needs_projection:
            logger.info(f"Adding projection layer from {self.base_output_dim} to {d_model}")
            self.projection = nn.Linear(self.base_output_dim, d_model)
        
        # Domain adaptation
        self.domain_embeddings = nn.Embedding(num_domains, d_model)
        
        # Transformer layers for domain-specific processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Final output layers
        self.classifier = nn.Linear(d_model, 1)  # For regression tasks
        self.relevance_predictor = nn.Linear(d_model, 1)  # For relevance scoring
        
    def forward(self, input_ids, attention_mask, domain_id=None):
        # Get base embeddings
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, base_output_dim]
        
        # Use CLS token as representation
        cls_output = hidden_states[:, 0, :]  # [batch_size, base_output_dim]
        
        # Project if dimensions don't match
        if self.needs_projection:
            cls_output = self.projection(cls_output)  # [batch_size, d_model]
        
        # Add domain embedding if provided
        if domain_id is not None:
            domain_emb = self.domain_embeddings(domain_id)  # [batch_size, d_model]
            cls_output = cls_output + domain_emb
        
        # Apply transformer for further processing
        # Create a sequence of length 1 with cls_output
        transformer_input = cls_output.unsqueeze(1)  # [batch_size, 1, d_model]
        transformer_output = self.transformer_encoder(transformer_input)  # [batch_size, 1, d_model]
        final_embedding = transformer_output.squeeze(1)  # [batch_size, d_model]
        
        # Get predictions
        predicted_score = self.classifier(final_embedding).squeeze(-1)  # [batch_size], raw score for MSE
        relevance_logits = self.relevance_predictor(final_embedding).squeeze(-1)  # [batch_size], raw logits for BCE
        predicted_relevance_prob = torch.sigmoid(relevance_logits) # [batch_size], probabilities for evaluation
        
        return {
            "score": predicted_score,
            "relevance_prob": predicted_relevance_prob,
            "relevance_logits": relevance_logits, # Need raw logits for BCEWithLogitsLoss
            "embedding": final_embedding
        }

def train(args, model, train_dataloader, optimizer, scheduler, scaler=None):
    """Train the model for one epoch."""
    model.train()
    epoch_loss = 0
    epoch_loss_score = 0 # Track score loss
    epoch_loss_relevance = 0 # Track relevance loss
    step = 0
    
    # Define loss functions
    loss_fn_score = nn.MSELoss()
    loss_fn_relevance = nn.BCEWithLogitsLoss()

    progress_bar = tqdm(
        train_dataloader, 
        desc="Training", 
        disable=args.use_ddp and args.local_rank != 0
    )
    
    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Get target values
        target_score = batch["target_score"]
        target_relevance = batch["target_relevance"]

        # Forward pass with mixed precision if enabled
        if args.fp16:
            with autocast(device_type='cuda'):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    domain_id=batch["domain_id"]
                )
                # --- Calculate actual loss (NEW) ---
                loss_score = loss_fn_score(outputs["score"], target_score)
                loss_relevance = loss_fn_relevance(outputs["relevance_logits"], target_relevance)
                loss = loss_score + loss_relevance # Combine losses (simple sum for now)
                # --- End New ---
        else:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                domain_id=batch["domain_id"]
            )
            # --- Calculate actual loss (NEW) ---
            loss_score = loss_fn_score(outputs["score"], target_score)
            loss_relevance = loss_fn_relevance(outputs["relevance_logits"], target_relevance)
            loss = loss_score + loss_relevance # Combine losses
            # --- End New ---
        
        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        
        # Backward pass with mixed precision if enabled
        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Accumulate component losses for logging
        epoch_loss += loss.item() * args.gradient_accumulation_steps
        epoch_loss_score += loss_score.item()
        epoch_loss_relevance += loss_relevance.item()

        # Update weights if gradient accumulation steps reached
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            
            # Log training progress
            if (step + 1) % args.logging_steps == 0 and (not args.use_ddp or args.local_rank == 0):
                lr = scheduler.get_last_lr()[0]
                # Calculate average losses for the logging step
                avg_step_loss = epoch_loss / (step + 1)
                avg_step_loss_score = epoch_loss_score / (step + 1)
                avg_step_loss_relevance = epoch_loss_relevance / (step + 1)
                progress_bar.set_postfix(
                    loss=avg_step_loss,
                    loss_score=avg_step_loss_score,
                    loss_rel=avg_step_loss_relevance,
                    lr=lr
                )
                
                if args.use_wandb:
                    wandb.log({
                        "train/loss": avg_step_loss,
                        "train/loss_score": avg_step_loss_score,
                        "train/loss_relevance": avg_step_loss_relevance,
                        "train/learning_rate": lr
                    })
        
        step += 1
    
    # Return average loss over the epoch
    return epoch_loss / step

def evaluate(args, model, eval_dataloader):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_loss_score = 0
    total_loss_relevance = 0
    total_samples = 0
    
    # Define loss functions
    loss_fn_score = nn.MSELoss()
    loss_fn_relevance = nn.BCEWithLogitsLoss()
    
    all_embeddings = []
    all_scores = []
    all_relevance_probs = []
    all_paper_ids = []
    # --- Add lists for targets and predictions (NEW) ---
    all_target_scores = []
    all_target_relevance = []
    all_pred_scores = []
    all_pred_relevance_probs = []
    # --- End New ---
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.use_ddp and args.local_rank != 0):
            # Move batch to device
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get target values
            target_score = batch["target_score"]
            target_relevance = batch["target_relevance"]

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                domain_id=batch["domain_id"]
            )
            
            # --- Calculate actual loss (NEW) ---
            loss_score = loss_fn_score(outputs["score"], target_score)
            loss_relevance = loss_fn_relevance(outputs["relevance_logits"], target_relevance)
            loss = loss_score + loss_relevance # Combine losses
            # --- End New ---
            
            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_loss_score += loss_score.item() * batch_size
            total_loss_relevance += loss_relevance.item() * batch_size
            total_samples += batch_size
            
            # --- Collect outputs and targets for detailed evaluation (UPDATED) ---
            all_embeddings.append(outputs["embedding"].cpu().numpy())
            all_paper_ids.extend(batch["paper_id"])
            
            all_pred_scores.append(outputs["score"].cpu().numpy())
            all_pred_relevance_probs.append(outputs["relevance_prob"].cpu().numpy())
            all_target_scores.append(target_score.cpu().numpy())
            all_target_relevance.append(target_relevance.cpu().numpy())
            # --- End Update ---
    
    # --- Combine results and calculate metrics (UPDATED) ---
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_loss_score = total_loss_score / total_samples if total_samples > 0 else float('inf')
    avg_loss_relevance = total_loss_relevance / total_samples if total_samples > 0 else float('inf')
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_pred_scores = np.concatenate(all_pred_scores, axis=0)
    all_pred_relevance_probs = np.concatenate(all_pred_relevance_probs, axis=0)
    all_target_scores = np.concatenate(all_target_scores, axis=0)
    all_target_relevance = np.concatenate(all_target_relevance, axis=0)
    
    # Calculate MAE for score prediction
    mae = mean_absolute_error(all_target_scores, all_pred_scores)
    
    # Calculate classification metrics for relevance prediction
    # Convert probabilities to binary predictions (threshold 0.5)
    all_pred_relevance_binary = (all_pred_relevance_probs >= 0.5).astype(int)
    all_target_relevance_int = all_target_relevance.astype(int) # Ensure targets are int for sklearn

    accuracy = accuracy_score(all_target_relevance_int, all_pred_relevance_binary)
    precision = precision_score(all_target_relevance_int, all_pred_relevance_binary, zero_division=0)
    recall = recall_score(all_target_relevance_int, all_pred_relevance_binary, zero_division=0)
    f1 = f1_score(all_target_relevance_int, all_pred_relevance_binary, zero_division=0)
    
    # Update metrics dictionary
    metrics = {
        "eval/loss": avg_loss,
        "eval/loss_score": avg_loss_score,
        "eval/loss_relevance": avg_loss_relevance,
        "eval/mae": mae,
        "eval/accuracy": accuracy,
        "eval/precision": precision,
        "eval/recall": recall,
        "eval/f1": f1
    }
    # --- End Update ---
    
    # Keep returned outputs minimal, focus on metrics
    eval_outputs = {
        "embeddings": all_embeddings,
        "scores": all_pred_scores, # Return predicted scores
        "relevance": all_pred_relevance_probs, # Return predicted relevance probabilities
        "paper_ids": all_paper_ids
        # Targets are used internally for metrics now
    }
    
    return metrics, eval_outputs

def convert_to_serializable(obj):
    """Convert NumPy types and PyTorch objects to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, torch.device):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def save_checkpoint(args, model, optimizer, scheduler, tokenizer, step, epoch, metrics, is_best=False):
    """Save model checkpoint."""
    if args.use_ddp and args.local_rank != 0:
        return
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model
    if args.use_ddp:
        model_to_save = model.module
    else:
        model_to_save = model
    
    torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
    
    # Save optimizer and scheduler
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
    
    # Save tokenizer
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Prepare training state with serializable values
    args_dict = vars(args).copy()  # Create a copy to avoid modifying the original
    serializable_args = convert_to_serializable(args_dict)
    serializable_metrics = convert_to_serializable(metrics)
    
    # Save training state
    training_state = {
        "step": step,
        "epoch": epoch,
        "metrics": serializable_metrics,
        "args": serializable_args
    }
    
    with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
        json.dump(training_state, f, indent=2)
    
    logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # If this is the best model, create a symlink
    if is_best:
        best_dir = os.path.join(args.output_dir, "best")
        if os.path.exists(best_dir):
            if os.path.islink(best_dir):
                os.unlink(best_dir)
            else:
                # If it's a directory, try to remove it
                try:
                    os.rmdir(best_dir)
                except OSError:
                    # If it's not empty, use shutil.rmtree
                    import shutil
                    shutil.rmtree(best_dir)
        
        # Create the symlink with relative path
        try:
            rel_path = os.path.relpath(checkpoint_dir, os.path.dirname(best_dir))
            os.symlink(rel_path, best_dir)
            logger.info(f"Saved best model checkpoint to {best_dir}")
        except OSError as e:
            logger.warning(f"Failed to create best model symlink: {e}")
            # Continue without failing the entire training
    
    # Manage number of checkpoints
    if args.save_total_limit > 0:
        checkpoints = sorted(
            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )
        
        if len(checkpoints) > args.save_total_limit:
            checkpoints_to_delete = checkpoints[:-args.save_total_limit]
            for checkpoint in checkpoints_to_delete:
                checkpoint_path = os.path.join(args.output_dir, checkpoint)
                logger.info(f"Deleting old checkpoint: {checkpoint_path}")
                import shutil
                shutil.rmtree(checkpoint_path)

@dataclass
class CustomDataCollator:
    """Custom data collator that handles padding for specific keys and batches others.
    
    Pads 'input_ids' and 'attention_mask' using DataCollatorWithPadding.
    Batches other numerical/tensor fields directly.
    Keeps 'paper_id' as a list of strings.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Separate features that need padding from others
        padding_features = []
        other_features = {key: [] for key in features[0].keys() if key not in ["input_ids", "attention_mask", "paper_id"]}
        paper_ids = []

        for feature in features:
            # Extract features for the standard padding collator
            padding_features.append({k: feature[k] for k in ["input_ids", "attention_mask"]})
            
            # Collect other features
            for key in other_features.keys():
                other_features[key].append(feature[key])
            
            # Collect paper_ids separately
            paper_ids.append(feature.get("paper_id", "")) # Handle if paper_id is missing

        # Use DataCollatorWithPadding for input_ids and attention_mask
        batch_padding = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )(padding_features)

        # Combine padded features with other batched features
        batch = {
            "input_ids": batch_padding["input_ids"],
            "attention_mask": batch_padding["attention_mask"],
            "paper_id": paper_ids # Keep as list of strings
        }

        # Batch other features (assuming they are tensors or tensor-like)
        for key, values in other_features.items():
            try:
                # Attempt to stack if they are already tensors (from __getitem__)
                batch[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else torch.tensor(values)
            except Exception as e:
                logger.warning(f"Could not collate feature '{key}' into a tensor: {e}. Keeping as list.")
                batch[key] = values # Keep as list if tensor conversion fails

        return batch

def main():
    """Run the training process."""
    # Parse arguments
    args = parse_args()
    args = set_environment(args)
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # --- Initialize Data Collator (NEW) ---
    data_collator = CustomDataCollator(tokenizer=tokenizer)
    # --- End New ---

    # Load domain mapping
    domain_mapping = load_domain_mapping(args.domain_file)
    num_domains = get_num_domains(domain_mapping)
    
    # Get valid domain IDs
    valid_domain_ids = [domain['id'] for domain in domain_mapping['domains']]
    logger.info(f"Valid domain IDs: {valid_domain_ids}")
    
    # Create datasets
    train_dataset = ResearchDataset(
        args.train_data,
        tokenizer,
        max_seq_length=args.max_seq_length,
        is_training=True,
        valid_domain_ids=valid_domain_ids
    )
    
    val_dataset = ResearchDataset(
        args.val_data,
        tokenizer,
        max_seq_length=args.max_seq_length,
        is_training=False,
        valid_domain_ids=valid_domain_ids
    )
    
    test_dataset = ResearchDataset(
        args.test_data,
        tokenizer,
        max_seq_length=args.max_seq_length,
        is_training=False,
        valid_domain_ids=valid_domain_ids
    )
    
    # Create data samplers
    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = None
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=data_collator
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=data_collator
    )
    
    # Calculate total steps
    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    if len(train_dataloader) % args.gradient_accumulation_steps != 0:
        steps_per_epoch += 1
    total_steps = steps_per_epoch * args.num_epochs
    
    # Initialize model
    model = ResearchTransformer(
        model_name=args.model_name,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        num_domains=num_domains
    )
    model.to(args.device)
    
    # Wrap model with DDP if needed
    if args.use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
    
    # Initialize optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if args.fp16 else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        
        # Load model
        model_path = os.path.join(args.resume_from_checkpoint, "model.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=args.device)
            if args.use_ddp:
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
        
        # Load optimizer
        optimizer_path = os.path.join(args.resume_from_checkpoint, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=args.device))
        
        # Load scheduler
        scheduler_path = os.path.join(args.resume_from_checkpoint, "scheduler.pt")
        if os.path.exists(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location=args.device))
        
        # Load training state
        training_state_path = os.path.join(args.resume_from_checkpoint, "training_state.json")
        if os.path.exists(training_state_path):
            with open(training_state_path, "r") as f:
                training_state = json.load(f)
            global_step = training_state.get("step", 0)
            start_epoch = training_state.get("epoch", 0)
            best_val_loss = training_state.get("metrics", {}).get("eval/loss", float('inf'))
    
    # Log training parameters
    if not args.use_ddp or args.local_rank == 0:
        logger.info(f"Training parameters:")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num validation examples = {len(val_dataset)}")
        logger.info(f"  Num epochs = {args.num_epochs}")
        logger.info(f"  Batch size per GPU = {args.batch_size}")
        logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {total_steps}")
        logger.info(f"  Learning rate = {args.learning_rate}")
        logger.info(f"  Number of domains = {num_domains}")
    
    # Training loop
    logger.info("Starting training")
    for epoch in range(start_epoch, args.num_epochs):
        if args.use_ddp:
            train_sampler.set_epoch(epoch)
        
        # Train for one epoch
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        epoch_loss = train(args, model, train_dataloader, optimizer, scheduler, scaler)
        
        # Evaluate and save checkpoints
        if not args.use_ddp or args.local_rank == 0:
            logger.info(f"Epoch {epoch + 1} completed. Running evaluation...")
            metrics, outputs = evaluate(args, model, val_dataloader)
            logger.info(f"Validation loss: {metrics['eval/loss']:.4f}")
            
            # Log metrics to W&B
            if args.use_wandb:
                wandb.log(metrics)
                wandb.log({"train/epoch_loss": epoch_loss})
            
            # Save checkpoint
            is_best = metrics["eval/loss"] < best_val_loss
            if is_best:
                best_val_loss = metrics["eval/loss"]
            
            save_checkpoint(
                args=args,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tokenizer=tokenizer,
                step=global_step,
                epoch=epoch + 1,
                metrics=metrics,
                is_best=is_best
            )
            
            logger.info(f"Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}, Val Loss: {metrics['eval/loss']:.4f}")
        
        global_step = total_steps * (epoch + 1)
    
    # Final evaluation on test set
    if not args.use_ddp or args.local_rank == 0:
        logger.info("Training completed. Running final evaluation on test set...")
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=data_collator
        )
        
        test_metrics, test_outputs = evaluate(args, model, test_dataloader)
        logger.info(f"Test loss: {test_metrics['eval/loss']:.4f}")
        
        # Log final metrics to W&B
        if args.use_wandb:
            for k, v in test_metrics.items():
                wandb.log({k.replace("eval", "test"): v})
            wandb.finish()
        
        # Save final test results
        final_results = {
            "test_metrics": test_metrics,
            "args": vars(args)
        }
        
        # Convert to JSON serializable format
        serializable_results = convert_to_serializable(final_results)
        
        with open(os.path.join(args.output_dir, "final_results.json"), "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Final results saved to {os.path.join(args.output_dir, 'final_results.json')}")
        
        # Always save to test_results.json as well (removing the check for do_eval)
        test_results_path = os.path.join(args.output_dir, "test_results.json")
        with open(test_results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Test results saved to {test_results_path}")

if __name__ == "__main__":
    main() 