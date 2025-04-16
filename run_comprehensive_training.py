#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to combine all available papers and run a comprehensive long training.
This maximizes data usage and training time to create the best possible model.
"""

import os
import sys
import json
import random
import logging
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/comprehensive_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_papers(file_path):
    """Load papers from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
        logger.info(f"Loaded {len(papers)} papers from {file_path}")
        return papers
    except Exception as e:
        logger.error(f"Error loading papers from {file_path}: {e}")
        return []

def load_kb_papers(kb_path):
    """Load papers from knowledge base."""
    kb_papers = []
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
        
        if isinstance(kb_data, dict):
            for paper_id, paper_data in kb_data.items():
                if isinstance(paper_data, dict):
                    paper = {
                        "id": paper_id,
                        "title": paper_data.get("title", ""),
                        "abstract": paper_data.get("abstract", ""),
                        "domain_id": paper_data.get("domain_id", 0),
                        "impact_score": paper_data.get("impact_score", 0.5)
                    }
                    kb_papers.append(paper)
        
        logger.info(f"Loaded {len(kb_papers)} papers from knowledge base")
    except Exception as e:
        logger.error(f"Error loading knowledge base: {e}")
    
    return kb_papers

def create_comprehensive_dataset():
    """Create a comprehensive dataset using all available papers."""
    # Set base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load standard datasets
    train_papers = load_papers(os.path.join(base_dir, "data/papers_train.json"))
    val_papers = load_papers(os.path.join(base_dir, "data/papers_val.json"))
    test_papers = load_papers(os.path.join(base_dir, "data/papers_test.json"))
    
    # Load knowledge base
    kb_papers = load_kb_papers(os.path.join(base_dir, "data/knowledge_base.json"))
    
    # Load any experimental datasets that might have been created
    datasets_to_check = [
        "data/experiments/balanced_domains.json",
        "data/experiments/varied_impact.json",
        "data/experiments/mixed_sources.json"
    ]
    
    experiment_papers = []
    for dataset_path in datasets_to_check:
        full_path = os.path.join(base_dir, dataset_path)
        if os.path.exists(full_path):
            papers = load_papers(full_path)
            experiment_papers.extend(papers)
    
    # Try to load any additional data sources if they exist
    additional_sources = [
        "data/additional_papers.json", 
        "data/external_papers.json",
        "data/scraped_papers.json"
    ]
    
    additional_papers = []
    for source in additional_sources:
        source_path = os.path.join(base_dir, source)
        if os.path.exists(source_path):
            papers = load_papers(source_path)
            additional_papers.extend(papers)
    
    # Combine all papers
    all_papers_dict = {}  # Use dict to avoid duplicates based on paper_id
    
    # Process each source and add to combined set
    def add_papers_to_dict(papers, source_name):
        for paper in papers:
            paper_id = paper.get("id")
            if not paper_id:
                # Generate a random ID if none exists
                paper_id = f"paper_{random.randint(10000, 99999)}"
                paper["id"] = paper_id
            
            # Add source information for tracking
            paper["source"] = source_name
            
            # Only add if not already in dict or replace with a more complete version
            if paper_id not in all_papers_dict or len(json.dumps(paper)) > len(json.dumps(all_papers_dict[paper_id])):
                all_papers_dict[paper_id] = paper
    
    # Add all paper sources
    add_papers_to_dict(train_papers, "train")
    add_papers_to_dict(val_papers, "val")
    add_papers_to_dict(test_papers, "test")
    add_papers_to_dict(kb_papers, "kb")
    add_papers_to_dict(experiment_papers, "experiments")
    add_papers_to_dict(additional_papers, "additional")
    
    # Convert to list
    all_papers = list(all_papers_dict.values())
    logger.info(f"Combined dataset contains {len(all_papers)} unique papers")
    
    # Analyze papers by domain
    domain_counts = defaultdict(int)
    for paper in all_papers:
        domain_id = paper.get("domain_id", 0)
        domain_counts[domain_id] += 1
    
    for domain_id, count in sorted(domain_counts.items()):
        logger.info(f"Domain {domain_id}: {count} papers")
    
    # Create directories
    comprehensive_dir = os.path.join(base_dir, "data/comprehensive")
    os.makedirs(comprehensive_dir, exist_ok=True)
    
    # Shuffle all papers with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(all_papers)
    
    # Split into training, validation, and test sets
    train_idx = int(len(all_papers) * 0.8)
    val_idx = int(len(all_papers) * 0.9)
    
    comprehensive_train = all_papers[:train_idx]
    comprehensive_val = all_papers[train_idx:val_idx]
    comprehensive_test = all_papers[val_idx:]
    
    # Save datasets
    with open(os.path.join(comprehensive_dir, "train.json"), "w") as f:
        json.dump(comprehensive_train, f, indent=2)
    logger.info(f"Saved {len(comprehensive_train)} papers to comprehensive training set")
    
    with open(os.path.join(comprehensive_dir, "val.json"), "w") as f:
        json.dump(comprehensive_val, f, indent=2)
    logger.info(f"Saved {len(comprehensive_val)} papers to comprehensive validation set")
    
    with open(os.path.join(comprehensive_dir, "test.json"), "w") as f:
        json.dump(comprehensive_test, f, indent=2)
    logger.info(f"Saved {len(comprehensive_test)} papers to comprehensive test set")
    
    return {
        "train": os.path.join(comprehensive_dir, "train.json"),
        "val": os.path.join(comprehensive_dir, "val.json"),
        "test": os.path.join(comprehensive_dir, "test.json"),
        "total_papers": len(all_papers)
    }

def run_comprehensive_training(dataset_paths, args):
    """Run comprehensive training with optimal parameters."""
    # Set up training command
    train_cmd = [
        sys.executable,
        "train.py",
        "--train_data", dataset_paths["train"],
        "--val_data", dataset_paths["val"],
        "--test_data", dataset_paths["test"],
        "--domain_file", "data/domains/domain_mapping.json",
        "--output_dir", args.output_dir,
        "--model_name", args.model_name,
        "--batch_size", str(args.batch_size),
        "--num_epochs", str(args.epochs),
        "--learning_rate", str(args.learning_rate),
        "--warmup_steps", str(args.warmup_steps),
        "--gradient_accumulation_steps", str(args.gradient_accumulation),
        "--save_steps", str(args.save_steps),
        "--eval_steps", str(args.eval_steps),
        "--logging_steps", str(args.logging_steps),
        "--save_total_limit", str(args.save_total_limit),
        "--weight_decay", str(args.weight_decay),
        "--dropout", str(args.dropout)
    ]
    
    # Add optional flags
    if args.fp16:
        train_cmd.append("--fp16")
    
    if args.use_wandb:
        train_cmd.append("--use_wandb")
        train_cmd.append("--wandb_project")
        train_cmd.append(args.wandb_project)
    
    # Print command
    logger.info(f"Starting comprehensive training with command: {' '.join(train_cmd)}")
    
    # Run the training process
    process = subprocess.Popen(
        train_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Stream the output
    for line in process.stdout:
        print(line, end='')
        logger.info(line.strip())
    
    # Wait for completion
    process.wait()
    
    if process.returncode == 0:
        logger.info("Comprehensive training completed successfully!")
    else:
        logger.error(f"Training failed with return code {process.returncode}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run comprehensive training with all available data")
    
    # Dataset arguments
    parser.add_argument("--prepare_only", action="store_true", help="Only prepare the dataset without training")
    
    # Training arguments
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2", 
                      help="Base pretrained model to use")
    parser.add_argument("--output_dir", type=str, default="outputs/comprehensive", 
                      help="Output directory for model checkpoints")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=75, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=3000, help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--weight_decay", type=float, default=0.03, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Evaluation and logging arguments
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every X steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    parser.add_argument("--save_total_limit", type=int, default=5, help="Max checkpoints to keep")
    
    # Performance arguments
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="comprehensive-research-transformer", 
                      help="Weights & Biases project name")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create log directory
    os.makedirs("logs", exist_ok=True)
    
    # Prepare comprehensive dataset
    logger.info("Preparing comprehensive dataset...")
    dataset_paths = create_comprehensive_dataset()
    
    if args.prepare_only:
        logger.info("Dataset preparation complete. Exiting as requested.")
        return
    
    # Run comprehensive training
    logger.info("Starting comprehensive training...")
    run_comprehensive_training(dataset_paths, args)

if __name__ == "__main__":
    main() 