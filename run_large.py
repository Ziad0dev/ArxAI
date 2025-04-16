"""
Launcher script for running large training jobs with the Research Transformer.
This script sets up the command line arguments and calls the train.py main function directly.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the correct directory to the Python path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import the main function from train.py
from train import main, parse_args

if __name__ == "__main__":
    # Set default arguments as if they were passed on the command line
    sys.argv = [
        "train.py",
        "--train_data", "data/papers_train.json",
        "--val_data", "data/papers_val.json", 
        "--test_data", "data/papers_test.json",
        "--domain_file", "data/domains/domain_mapping.json",
        "--output_dir", "outputs/large_run",
        "--model_name", "sentence-transformers/all-mpnet-base-v2",
        "--batch_size", "32",
        "--num_epochs", "10",
        "--learning_rate", "2e-5",
        "--max_seq_length", "512",
        "--fp16",
        "--gradient_accumulation_steps", "2",
        "--save_steps", "500",
        "--eval_steps", "250",
        "--logging_steps", "50"
    ]
    
    # Create output directory
    os.makedirs("outputs/large_run", exist_ok=True)
    
    # Run the main function from train.py
    print("Starting large training run...")
    main()
    print("Training complete!") 