#!/usr/bin/env python3
"""
Wrapper script to handle import paths for the inference script.
"""

import os
import sys
import subprocess

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the environment variables
    env = os.environ.copy()
    
    # Add the current directory to PYTHONPATH
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{script_dir}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = script_dir
    
    # Run the inference script
    inference_script = os.path.join(script_dir, "run_inference.py")
    result = subprocess.run(
        ["python", inference_script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print("Errors:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main()) 