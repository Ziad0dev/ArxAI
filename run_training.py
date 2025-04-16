#!/usr/bin/env python
"""
Large-Scale Training Run for Enhanced Neural Network
---------------------------------------------------
Executes a comprehensive training run with the enhanced model architecture,
larger embeddings, and improved data processing pipeline.
"""

import os
import time
import argparse
import torch
import numpy as np
import logging
from tqdm import tqdm
from datetime import datetime
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Import ARX2 components
from advanced_ai_analyzer import CONFIG, logger
from advanced_ai_analyzer_learning import LearningSystem, EnhancedClassifier
from advanced_ai_analyzer_paper_processor import PaperProcessor
from utils.embedding_manager import EmbeddingManager
from advanced_ai_analyzer_knowledge_base import KnowledgeBase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run large-scale training for ARX2 research analyzer")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"], 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=CONFIG["learning_rate"], 
                        help="Learning rate")
    parser.add_argument("--model_type", type=str, default="enhanced", 
                        choices=["enhanced", "simple"], help="Type of model to train")
    
    # Data parameters
    parser.add_argument("--data_augmentation", action="store_true", 
                        help="Use data augmentation during training")
    parser.add_argument("--max_papers", type=int, default=CONFIG["max_papers_total"], 
                        help="Maximum number of papers to process")
    parser.add_argument("--query", type=str, 
                        default="transformer OR 'large language model' OR 'deep learning'", 
                        help="Search query for papers")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="training_results", 
                        help="Directory to save training results")
    parser.add_argument("--checkpoint_interval", type=int, default=5, 
                        help="Save model checkpoint every N epochs")
    
    # Resource parameters
    parser.add_argument("--use_gpu", action="store_true", default=CONFIG["use_gpu"], 
                        help="Use GPU for training if available")
    parser.add_argument("--num_workers", type=int, default=CONFIG["num_workers"], 
                        help="Number of workers for data loading")
    
    # Advanced parameters
    parser.add_argument("--mixed_precision", action="store_true", 
                        default=CONFIG["use_mixed_precision"], 
                        help="Use mixed precision training")
    parser.add_argument("--gradient_accumulation", type=int, 
                        default=CONFIG["gradient_accumulation_steps"], 
                        help="Number of gradient accumulation steps")
    
    return parser.parse_args()

def setup_training_environment(args):
    """Set up training environment, directories, and device."""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = output_dir / "checkpoints"
    plots_dir = output_dir / "plots"
    results_dir = output_dir / "results"
    
    for directory in [checkpoints_dir, plots_dir, results_dir]:
        directory.mkdir(exist_ok=True)
    
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training")
    
    # Set random seeds for reproducibility
    seed = CONFIG.get("random_seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    return {
        "output_dir": output_dir,
        "checkpoints_dir": checkpoints_dir,
        "plots_dir": plots_dir,
        "results_dir": results_dir,
        "device": device
    }

def load_or_collect_papers(args, paper_processor):
    """Load papers from cache or collect new ones."""
    cache_path = Path("papers_cache.json")
    
    if cache_path.exists():
        logger.info(f"Loading papers from cache: {cache_path}")
        with open(cache_path, "r") as f:
            papers_metadata = json.load(f)
        
        if len(papers_metadata) >= args.max_papers:
            logger.info(f"Loaded {len(papers_metadata)} papers from cache")
            return papers_metadata[:args.max_papers]
        else:
            logger.info(f"Cache contains only {len(papers_metadata)} papers, collecting more...")
    else:
        logger.info("Paper cache not found, collecting papers from ArXiv...")
        papers_metadata = []
    
    # Collect papers to reach desired count
    remaining = args.max_papers - len(papers_metadata)
    if remaining > 0:
        new_papers = paper_processor.download_papers(
            query=args.query, 
            max_results=min(remaining, CONFIG["max_papers_per_query"])
        )
        papers_metadata.extend(new_papers)
        
        # Save updated cache
        with open(cache_path, "w") as f:
            json.dump(papers_metadata, f)
    
    logger.info(f"Collected a total of {len(papers_metadata)} papers")
    return papers_metadata[:args.max_papers]

def process_papers(args, paper_processor, papers_metadata):
    """Process papers to extract features, embeddings, and concepts."""
    logger.info(f"Processing {len(papers_metadata)} papers")
    
    # Use augmentation if specified
    processed_papers = paper_processor.process_papers_batch(
        papers_metadata, 
        with_augmentation=args.data_augmentation
    )
    
    logger.info(f"Successfully processed {len(processed_papers)} papers")
    logger.info(f"Embedding cache stats: {paper_processor.get_embedding_cache_stats()}")
    
    return processed_papers

def run_training(args, env, learning_system):
    """Execute the main training loop."""
    logger.info(f"Starting {args.model_type} model training for {args.epochs} epochs")
    
    # Set up training parameters
    training_args = {
        "model_type": args.model_type,
        "learning_rate": args.learning_rate,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "mixed_precision": args.mixed_precision,
        "gradient_accumulation_steps": args.gradient_accumulation
    }
    
    # Start training timer
    start_time = time.time()
    
    # Train the model
    training_results = learning_system.train(**training_args)
    
    # Record training time
    training_time = time.time() - start_time
    
    # Save training results
    results_path = env["results_dir"] / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        # Add training time to results
        training_results["training_time_seconds"] = training_time
        json.dump(training_results, f, indent=2)
    
    logger.info(f"Training completed in {training_time/60:.2f} minutes")
    logger.info(f"Final training accuracy: {training_results.get('accuracy', 'N/A')}")
    logger.info(f"Final validation accuracy: {training_results.get('val_accuracy', 'N/A')}")
    
    return training_results

def generate_plots(env, training_results):
    """Generate and save plots of training metrics."""
    # Check if there's history data to plot
    if "history" not in training_results:
        logger.warning("No training history found for plotting")
        return
    
    history = training_results["history"]
    
    # Create common x-axis (epochs)
    epochs = list(range(1, len(history["loss"]) + 1))
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Training Loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Loss During Training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    if "accuracy" in history:
        plt.plot(epochs, history["accuracy"], label="Training Accuracy")
    if "val_accuracy" in history:
        plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy During Training")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(env["plots_dir"] / "training_metrics.png", dpi=300)
    plt.close()
    
    logger.info(f"Training plots saved to {env['plots_dir']}")

def evaluate_model(learning_system, env):
    """Perform a comprehensive evaluation of the trained model."""
    logger.info("Starting model evaluation")
    
    # Evaluate on test set
    eval_results = learning_system.evaluate()
    
    # Save evaluation results
    eval_path = env["results_dir"] / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Model evaluation complete. Test accuracy: {eval_results.get('accuracy', 'N/A')}")
    
    return eval_results

def main():
    """Main function to run the training pipeline."""
    args = parse_args()
    logger.info("Starting large-scale training run")
    logger.info(f"Configuration: {args}")
    
    # Set up environment
    env = setup_training_environment(args)
    
    # Initialize components
    paper_processor = PaperProcessor()
    
    # Load or collect papers
    papers_metadata = load_or_collect_papers(args, paper_processor)
    
    # Process papers
    processed_papers = process_papers(args, paper_processor, papers_metadata)
    
    # Initialize knowledge base and add processed papers
    knowledge_base = KnowledgeBase()
    knowledge_base.add_papers(processed_papers)
    
    # Initialize learning system with knowledge base
    learning_system = LearningSystem(knowledge_base)
    
    # Prepare data for training
    learning_system.prepare_data()
    
    # Run training
    training_results = run_training(args, env, learning_system)
    
    # Generate plots
    generate_plots(env, training_results)
    
    # Evaluate model
    eval_results = evaluate_model(learning_system, env)
    
    logger.info("Training pipeline completed successfully")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model type: {args.model_type}")
    print(f"Papers processed: {len(processed_papers)}")
    print(f"Training epochs: {args.epochs}")
    print(f"Final training accuracy: {training_results.get('accuracy', 'N/A')}")
    print(f"Final validation accuracy: {training_results.get('val_accuracy', 'N/A')}")
    print(f"Test accuracy: {eval_results.get('accuracy', 'N/A')}")
    print(f"Training time: {training_results.get('training_time_seconds', 0)/60:.2f} minutes")
    print(f"Results saved to: {env['output_dir']}")
    print("="*50)

if __name__ == "__main__":
    main() 