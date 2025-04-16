#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to fix the dataset preparation issue and monitor training progress.
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/monitor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fix_knowledge_base():
    """
    Fix the knowledge base data format issue that's causing the error:
    'str' object has no attribute 'get'
    """
    logger.info("Fixing knowledge base data format...")
    
    # Load knowledge base
    try:
        with open("data/knowledge_base.json", "r") as f:
            kb_data = json.load(f)
        
        # Create a new knowledge base with fixed format
        fixed_kb = {}
        
        # Process each entry in the knowledge base
        if isinstance(kb_data, dict):
            for key, value in kb_data.items():
                # Convert string values to proper dictionary format
                if isinstance(value, str):
                    fixed_kb[key] = {
                        "title": value,
                        "abstract": "",
                        "domain_id": 0
                    }
                elif isinstance(value, dict):
                    fixed_kb[key] = value
                else:
                    logger.warning(f"Skipping unknown data type for key {key}: {type(value)}")
        
        # Backup original file
        backup_path = "data/knowledge_base.json.bak"
        import shutil
        shutil.copy("data/knowledge_base.json", backup_path)
        logger.info(f"Original knowledge base backed up to {backup_path}")
        
        # Save fixed knowledge base
        with open("data/knowledge_base.json", "w") as f:
            json.dump(fixed_kb, f, indent=2)
        
        logger.info(f"Fixed knowledge base with {len(fixed_kb)} entries")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing knowledge base: {e}")
        return False

def prepare_datasets():
    """
    Prepare the combined datasets with proper error handling.
    """
    logger.info("Preparing combined datasets...")
    
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
    
    # Load knowledge base data with robust error handling
    kb_papers = []
    try:
        with open("data/knowledge_base.json", "r") as f:
            kb_data = json.load(f)
        
        # Process each entry in the knowledge base
        if isinstance(kb_data, dict):
            for key, value in kb_data.items():
                try:
                    # Safe extraction with defaults
                    if isinstance(value, dict):
                        paper = {
                            "id": key,
                            "title": value.get("title", ""),
                            "abstract": value.get("abstract", ""),
                            "domain_id": value.get("domain_id", 0)
                        }
                        if "impact_score" in value:
                            paper["impact_score"] = value["impact_score"]
                        kb_papers.append(paper)
                    else:
                        logger.warning(f"Skipping non-dict entry for key {key}: {type(value)}")
                except Exception as e:
                    logger.warning(f"Error processing entry {key}: {e}")
        
        logger.info(f"Loaded {len(kb_papers)} papers from knowledge base")
    except Exception as e:
        logger.error(f"Error loading knowledge base data: {e}")
    
    # Combine datasets, ensuring no duplicates by paper ID
    all_papers = {}
    
    # Start with training data
    for paper in train_data:
        paper_id = paper.get("id", f"train_{len(all_papers)}")
        all_papers[paper_id] = paper
    
    # Add knowledge base papers
    for paper in kb_papers:
        paper_id = paper.get("id", f"kb_{len(all_papers)}")
        if paper_id not in all_papers:
            # Add default impact score if missing
            if "impact_score" not in paper:
                paper["impact_score"] = 0.5
            all_papers[paper_id] = paper
    
    combined_papers = list(all_papers.values())
    logger.info(f"Combined dataset contains {len(combined_papers)} unique papers")
    
    # Shuffle the combined dataset
    import random
    random.seed(42)
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
    logger.info(f"Saved {len(test_data)} papers to combined test dataset")
    
    return {
        "train": os.path.join(output_dir, "combined_train.json"),
        "val": os.path.join(output_dir, "combined_val.json"),
        "test": os.path.join(output_dir, "combined_test.json")
    }

def run_training_with_fixed_data():
    """
    Run the training with the fixed dataset.
    """
    logger.info("Starting training with fixed dataset...")
    
    # Build command arguments
    cmd = [
        sys.executable,
        "run_long_training.py",
        "--skip_preparation",  # Skip preparation since we just did it
        "--fp16",
        "--epochs", "50",
        "--learning_rate", "1e-5",
        "--gradient_accumulation", "8"
    ]
    
    # Print the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    import subprocess
    subprocess.run(cmd)

def monitor_training():
    """
    Monitor training progress by checking log files and checkpoints.
    """
    logger.info("Monitoring training progress...")
    
    # Initialize tracking variables
    last_checkpoint = None
    progress_data = {
        "steps": [],
        "loss": [],
        "learning_rate": []
    }
    
    # Monitor output directory
    output_dir = "outputs/long_run"
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plotting configuration
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 12})
    
    try:
        while True:
            # Check for new checkpoints
            checkpoints = glob(f"{output_dir}/checkpoint-*")
            if checkpoints:
                newest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                if newest_checkpoint != last_checkpoint:
                    last_checkpoint = newest_checkpoint
                    logger.info(f"New checkpoint found: {newest_checkpoint}")
                    
                    # Check if there's an evaluation file in the checkpoint
                    eval_results_path = os.path.join(newest_checkpoint, "eval_results.json")
                    if os.path.exists(eval_results_path):
                        with open(eval_results_path, "r") as f:
                            eval_results = json.load(f)
                        logger.info(f"Evaluation results: Loss={eval_results.get('eval/loss', 'N/A')}, "
                                   f"MAE={eval_results.get('eval/mae', 'N/A')}")
            
            # Check for training state (which indicates current step)
            training_state_path = os.path.join(output_dir, "training_state.json")
            if os.path.exists(training_state_path):
                with open(training_state_path, "r") as f:
                    training_state = json.load(f)
                
                current_step = training_state.get("global_step", 0)
                current_epoch = training_state.get("epoch", 0)
                logger.info(f"Current progress: Step {current_step}, Epoch {current_epoch:.2f}")
                
                # Track progress for visualization
                if "loss" in training_state:
                    progress_data["steps"].append(current_step)
                    progress_data["loss"].append(training_state["loss"])
                    progress_data["learning_rate"].append(training_state.get("learning_rate", 0))
                
                # Generate visualizations
                if len(progress_data["steps"]) > 1:
                    # Loss plot
                    plt.figure(figsize=(12, 6))
                    plt.plot(progress_data["steps"], progress_data["loss"], marker='o', linestyle='-')
                    plt.title('Training Loss')
                    plt.xlabel('Step')
                    plt.ylabel('Loss')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, "training_loss.png"))
                    plt.close()
                    
                    # Learning rate plot
                    plt.figure(figsize=(12, 6))
                    plt.plot(progress_data["steps"], progress_data["learning_rate"], marker='o', linestyle='-')
                    plt.title('Learning Rate')
                    plt.xlabel('Step')
                    plt.ylabel('Learning Rate')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, "learning_rate.png"))
                    plt.close()
            
            # Check for final results
            final_results_path = os.path.join(output_dir, "final_results.json")
            if os.path.exists(final_results_path):
                logger.info("Training completed! Final results available.")
                with open(final_results_path, "r") as f:
                    final_results = json.load(f)
                
                # Print test metrics
                test_metrics = final_results.get("test_metrics", {})
                for key, value in test_metrics.items():
                    logger.info(f"{key}: {value}")
                
                # Create final visualization
                metrics = []
                values = []
                for key, value in test_metrics.items():
                    if isinstance(value, (int, float)):
                        metrics.append(key)
                        values.append(value)
                
                plt.figure(figsize=(14, 8))
                bars = plt.bar(metrics, values, color='skyblue')
                plt.title('Final Test Metrics')
                plt.xlabel('Metric')
                plt.ylabel('Value')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.4f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, "final_metrics.png"))
                plt.close()
                
                # Create a summary file
                with open(os.path.join(viz_dir, "training_summary.txt"), "w") as f:
                    f.write("=== TRAINING SUMMARY ===\n\n")
                    f.write(f"Training completed at: {datetime.datetime.now()}\n")
                    f.write(f"Final step: {max(progress_data['steps']) if progress_data['steps'] else 'N/A'}\n")
                    f.write(f"Final loss: {progress_data['loss'][-1] if progress_data['loss'] else 'N/A'}\n\n")
                    f.write("TEST METRICS:\n")
                    for key, value in test_metrics.items():
                        f.write(f"{key}: {value}\n")
                
                logger.info(f"Visualizations saved to {viz_dir}")
                break
            
            # Wait before checking again
            time.sleep(30)
    
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error while monitoring: {e}")

def main():
    """Main function to fix and monitor training."""
    parser = argparse.ArgumentParser(description="Fix dataset issues and monitor training")
    parser.add_argument("--fix-only", action="store_true", help="Only fix the datasets without running training")
    parser.add_argument("--monitor-only", action="store_true", help="Only monitor existing training without fixing or starting new training")
    args = parser.parse_args()
    
    if args.monitor_only:
        monitor_training()
        return
    
    # Fix knowledge base data
    if fix_knowledge_base():
        logger.info("Knowledge base data fixed successfully")
    else:
        logger.error("Failed to fix knowledge base data, exiting")
        return
    
    # Prepare datasets
    try:
        dataset_paths = prepare_datasets()
        logger.info("Datasets prepared successfully")
    except Exception as e:
        logger.error(f"Error preparing datasets: {e}")
        return
    
    if args.fix_only:
        logger.info("Datasets fixed. Exiting without running training as requested.")
        return
    
    # Run training with fixed data
    run_training_with_fixed_data()
    
    # Monitor training progress
    monitor_training()

if __name__ == "__main__":
    main() 