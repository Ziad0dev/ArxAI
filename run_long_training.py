#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to run a long training session using all available paper data.
Combines knowledge base data with existing training data for a more robust model.
"""

import os
import sys
import json
import logging
import argparse
import datetime
import numpy as np
import random
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/long_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def prepare_combined_dataset():
    """
    Combine all available paper data into a single large training dataset.
    Includes:
    - Original training data
    - Knowledge base data (converted to proper format)
    - Sample validation data for checking progress
    """
    logger.info("Preparing combined dataset from all available sources")
    
    # Set base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    # Create output directory
    output_dir = os.path.join(base_dir, "data", "combined")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing datasets
    try:
        with open("data/papers_train.json", "r") as f:
            train_data = json.load(f)
        logger.info(f"Loaded {len(train_data)} papers from training dataset")
        
        with open("data/papers_val.json", "r") as f:
            val_data = json.load(f)
        logger.info(f"Loaded {len(val_data)} papers from validation dataset")
        
        with open("data/papers_test.json", "r") as f:
            test_data = json.load(f)
        logger.info(f"Loaded {len(test_data)} papers from test dataset")
    except Exception as e:
        logger.error(f"Error loading standard datasets: {e}")
        raise
    
    # Load knowledge base data (which has more papers)
    try:
        with open("data/knowledge_base.json", "r") as f:
            kb_data = json.load(f)
        
        # Check if it's an array or dictionary with papers key
        if isinstance(kb_data, list):
            kb_papers = kb_data
        elif isinstance(kb_data, dict) and "papers" in kb_data:
            kb_papers = kb_data["papers"]
        else:
            kb_papers = []
            for key, value in kb_data.items():
                if isinstance(value, dict) and "title" in value:
                    # Convert to standard paper format
                    paper = {
                        "id": key,
                        "title": value.get("title", ""),
                        "abstract": value.get("abstract", ""),
                        "domain_id": value.get("domain_id", 0)
                    }
                    kb_papers.append(paper)
        
        logger.info(f"Loaded {len(kb_papers)} papers from knowledge base")
    except Exception as e:
        logger.error(f"Error loading knowledge base data: {e}")
        kb_papers = []
    
    # Combine datasets, ensuring no duplicates by paper ID
    all_papers = {}
    
    # Start with training data
    for paper in train_data:
        paper_id = paper.get("id", "unknown")
        all_papers[paper_id] = paper
    
    # Add papers from knowledge base
    for paper in kb_papers:
        paper_id = paper.get("id", "unknown")
        if paper_id not in all_papers:
            # For papers that don't have domain_id or impact_score, add defaults
            if "domain_id" not in paper:
                paper["domain_id"] = 0  # Default to computer science
            if "impact_score" not in paper:
                paper["impact_score"] = 0.5  # Default impact score
            all_papers[paper_id] = paper
    
    combined_papers = list(all_papers.values())
    logger.info(f"Combined dataset contains {len(combined_papers)} unique papers")
    
    # Shuffle the combined dataset
    random.shuffle(combined_papers)
    
    # Split into training and validation (90/10)
    split_idx = int(len(combined_papers) * 0.9)
    new_train_data = combined_papers[:split_idx]
    new_val_data = combined_papers[split_idx:]
    
    # Write new combined datasets
    with open(os.path.join(output_dir, "combined_train.json"), "w") as f:
        json.dump(new_train_data, f, indent=2)
    logger.info(f"Saved {len(new_train_data)} papers to combined training dataset")
    
    with open(os.path.join(output_dir, "combined_val.json"), "w") as f:
        json.dump(new_val_data, f, indent=2)
    logger.info(f"Saved {len(new_val_data)} papers to combined validation dataset")
    
    # Keep the original test dataset
    with open(os.path.join(output_dir, "combined_test.json"), "w") as f:
        json.dump(test_data, f, indent=2)
    
    return {
        "train": os.path.join(output_dir, "combined_train.json"),
        "val": os.path.join(output_dir, "combined_val.json"),
        "test": os.path.join(output_dir, "combined_test.json")
    }

def run_long_training(dataset_paths, args):
    """
    Run a long training session with the combined dataset.
    
    Args:
        dataset_paths: Dictionary with paths to train, val, and test datasets
        args: Command-line arguments
    """
    # Import needed here to avoid imports when just preparing data
    import torch
    
    # Configure tensor cores if available for faster training
    if torch.cuda.is_available():
        # Get total GPU memory in GB
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Found GPU with {gpu_memory:.2f} GB memory")
        
        # Adjust batch size based on available GPU memory
        # This is a simple heuristic - adjust based on your specific hardware
        if gpu_memory > 24:  # High-end GPU (A100, etc.)
            suggested_batch_size = 64
        elif gpu_memory > 16:  # Mid-range GPU (V100, A10, etc.)
            suggested_batch_size = 48
        elif gpu_memory > 8:  # Consumer GPU (RTX 3080, etc.)
            suggested_batch_size = 32
        else:  # Lower-end GPU
            suggested_batch_size = 16
            
        if args.batch_size is None:
            args.batch_size = suggested_batch_size
            logger.info(f"Setting batch size to {args.batch_size} based on GPU memory")
    else:
        logger.warning("No GPU detected, training will be slow!")
        if args.batch_size is None:
            args.batch_size = 8
    
    # Build command for running the training script
    cmd = [
        sys.executable,
        "train.py",
        "--train_data", dataset_paths["train"],
        "--val_data", dataset_paths["val"],
        "--test_data", dataset_paths["test"],
        "--domain_file", "data/domains/domain_mapping.json",
        "--output_dir", "outputs/long_run",
        "--model_name", args.model_name,
        "--batch_size", str(args.batch_size),
        "--eval_batch_size", str(args.batch_size * 2),
        "--num_epochs", str(args.epochs),
        "--learning_rate", str(args.learning_rate),
        "--warmup_steps", str(args.warmup_steps),
        "--max_seq_length", str(args.max_seq_length),
        "--gradient_accumulation_steps", str(args.gradient_accumulation),
        "--save_steps", str(args.save_steps),
        "--eval_steps", str(args.eval_steps),
        "--logging_steps", str(args.logging_steps),
        "--save_total_limit", str(args.save_total_limit),
    ]
    
    # Add optional flags
    if args.fp16:
        cmd.append("--fp16")
    
    if args.use_wandb:
        cmd.append("--use_wandb")
        cmd.append("--wandb_project")
        cmd.append(args.wandb_project)
    
    # Print the command
    logger.info(f"Running training with command: {' '.join(cmd)}")
    
    # Execute the command (directly using os.execv to replace current process)
    os.execv(sys.executable, cmd)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a long training session using all available paper data")
    
    # Dataset arguments
    parser.add_argument("--prepare_only", action="store_true", help="Only prepare the combined dataset without training")
    parser.add_argument("--skip_preparation", action="store_true", help="Skip dataset preparation and use existing combined data")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2", 
                      help="Base pretrained model to use")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size (auto-determined if not specified)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Number of warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")
    
    # Checkpointing and evaluation arguments
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every X steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep")
    
    # Performance arguments
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="arx2-long-training", help="Weights & Biases project name")
    
    # Resume arguments
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Prepare the combined dataset
    if not args.skip_preparation:
        try:
            dataset_paths = prepare_combined_dataset()
            logger.info("Dataset preparation complete!")
            
            if args.prepare_only:
                logger.info("Exiting after dataset preparation as requested")
                return
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            sys.exit(1)
    else:
        logger.info("Skipping dataset preparation as requested")
        dataset_paths = {
            "train": "data/combined/combined_train.json",
            "val": "data/combined/combined_val.json",
            "test": "data/combined/combined_test.json"
        }
    
    # Run the long training session
    try:
        run_long_training(dataset_paths, args)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 