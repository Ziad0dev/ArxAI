#!/usr/bin/env python
"""
Training Progress Monitor
------------------------
Monitor training progress, hardware utilization, and metrics in real-time.
"""

import os
import time
import json
import argparse
import psutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import threading
import logging
import re

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("training_monitor")

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    logger.warning("GPUtil not installed. GPU monitoring will be limited.")
    HAS_GPUTIL = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    logger.warning("Pandas not installed. Data analysis will be limited.")
    HAS_PANDAS = False

class TrainingMonitor:
    """Real-time monitor for ML training processes."""
    
    def __init__(self, log_file=None, results_dir=None, refresh_interval=5):
        """Initialize the training monitor.
        
        Args:
            log_file (str): Path to the training log file to monitor
            results_dir (str): Directory where results and plots are saved
            refresh_interval (int): Data refresh interval in seconds
        """
        self.log_file = log_file
        self.results_dir = Path(results_dir) if results_dir else Path("training_monitor")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        self.refresh_interval = refresh_interval
        
        # Monitoring data
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.timestamps = []
        
        # Extracted metrics
        self.training_loss = []
        self.validation_loss = []
        self.training_accuracy = []
        self.validation_accuracy = []
        self.epochs = []
        
        # Current state
        self.running = False
        self.start_time = None
        self.current_epoch = 0
        self.total_epochs = 0
        
        logger.info(f"Training monitor initialized with refresh interval of {refresh_interval}s")
    
    def start_monitoring(self):
        """Start the monitoring process in a background thread."""
        self.running = True
        self.start_time = time.time()
        
        # Start the monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring process."""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
            
        self.generate_summary_report()
        logger.info("Monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop that collects metrics at regular intervals."""
        while self.running:
            # Record timestamp
            current_time = time.time()
            elapsed = current_time - self.start_time
            self.timestamps.append(elapsed)
            
            # Collect system metrics
            self._collect_system_metrics()
            
            # Parse log file for training metrics if available
            if self.log_file and os.path.exists(self.log_file):
                self._parse_log_file()
            
            # Generate plots periodically
            if len(self.timestamps) % 10 == 0:
                self.generate_plots()
            
            # Sleep until next collection point
            time.sleep(self.refresh_interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics (CPU, memory, GPU)."""
        # CPU usage (percentage)
        cpu_percent = psutil.cpu_percent(interval=0.5)
        self.cpu_usage.append(cpu_percent)
        
        # Memory usage (percentage)
        memory_percent = psutil.virtual_memory().percent
        self.memory_usage.append(memory_percent)
        
        # GPU metrics if available
        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    # Take the first GPU for simplicity
                    gpu = gpus[0]
                    self.gpu_usage.append(gpu.load * 100)  # Convert to percentage
                    self.gpu_memory.append(gpu.memoryUsed / gpu.memoryTotal * 100)  # Memory usage as percentage
                else:
                    self.gpu_usage.append(0)
                    self.gpu_memory.append(0)
            except Exception as e:
                logger.error(f"Error collecting GPU metrics: {e}")
                self.gpu_usage.append(0)
                self.gpu_memory.append(0)
        else:
            # No GPU monitoring available
            self.gpu_usage.append(0)
            self.gpu_memory.append(0)
    
    def _parse_log_file(self):
        """Parse training log file to extract metrics."""
        try:
            with open(self.log_file, 'r') as f:
                content = f.read()
            
            # Extract current epoch information
            epoch_matches = re.findall(r'Epoch (\d+)/(\d+)', content)
            if epoch_matches:
                self.current_epoch = int(epoch_matches[-1][0])
                self.total_epochs = int(epoch_matches[-1][1])
            
            # Extract loss values
            train_loss_matches = re.findall(r'training loss: ([\d\.]+)', content, re.IGNORECASE)
            val_loss_matches = re.findall(r'validation loss: ([\d\.]+)', content, re.IGNORECASE)
            
            # Extract accuracy values
            train_acc_matches = re.findall(r'training accuracy: ([\d\.]+)', content, re.IGNORECASE)
            val_acc_matches = re.findall(r'validation accuracy: ([\d\.]+)', content, re.IGNORECASE)
            
            # Update if new values found
            if len(train_loss_matches) > len(self.training_loss):
                self.training_loss = [float(x) for x in train_loss_matches]
                self.epochs = list(range(1, len(self.training_loss) + 1))
            
            if len(val_loss_matches) > len(self.validation_loss):
                self.validation_loss = [float(x) for x in val_loss_matches]
            
            if len(train_acc_matches) > len(self.training_accuracy):
                self.training_accuracy = [float(x) for x in train_acc_matches]
            
            if len(val_acc_matches) > len(self.validation_accuracy):
                self.validation_accuracy = [float(x) for x in val_acc_matches]
                
        except Exception as e:
            logger.error(f"Error parsing log file: {e}")
    
    def generate_plots(self):
        """Generate real-time monitoring plots."""
        # Create plots directory
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Convert timestamps to minutes for better readability
        minutes = [t / 60 for t in self.timestamps]
        
        # Plot system metrics
        plt.figure(figsize=(12, 8))
        
        # CPU and memory usage
        plt.subplot(2, 2, 1)
        plt.plot(minutes, self.cpu_usage, label='CPU Usage (%)')
        plt.plot(minutes, self.memory_usage, label='Memory Usage (%)')
        plt.title('System Resource Usage')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Usage (%)')
        plt.legend()
        plt.grid(True)
        
        # GPU metrics
        plt.subplot(2, 2, 2)
        plt.plot(minutes, self.gpu_usage, label='GPU Usage (%)')
        plt.plot(minutes, self.gpu_memory, label='GPU Memory (%)')
        plt.title('GPU Resource Usage')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Usage (%)')
        plt.legend()
        plt.grid(True)
        
        # Plot training metrics if available
        if self.training_loss and self.epochs:
            plt.subplot(2, 2, 3)
            plt.plot(self.epochs, self.training_loss, label='Training Loss')
            if self.validation_loss:
                plt.plot(self.epochs, self.validation_loss, label='Validation Loss')
            plt.title('Loss Progress')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            if self.training_accuracy:
                plt.plot(self.epochs, self.training_accuracy, label='Training Accuracy')
            if self.validation_accuracy:
                plt.plot(self.epochs, self.validation_accuracy, label='Validation Accuracy')
            plt.title('Accuracy Progress')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(plots_dir / f"training_monitor_{timestamp}.png", dpi=100)
        plt.close()
    
    def generate_summary_report(self):
        """Generate a summary report of the training session."""
        # Create report directory
        report_dir = self.results_dir / "reports"
        report_dir.mkdir(exist_ok=True, parents=True)
        
        # Calculate statistics
        report = {
            "duration_seconds": self.timestamps[-1] if self.timestamps else 0,
            "duration_formatted": self._format_time(self.timestamps[-1] if self.timestamps else 0),
            "cpu_usage": {
                "mean": np.mean(self.cpu_usage) if self.cpu_usage else 0,
                "max": np.max(self.cpu_usage) if self.cpu_usage else 0,
                "min": np.min(self.cpu_usage) if self.cpu_usage else 0
            },
            "memory_usage": {
                "mean": np.mean(self.memory_usage) if self.memory_usage else 0,
                "max": np.max(self.memory_usage) if self.memory_usage else 0,
                "min": np.min(self.memory_usage) if self.memory_usage else 0
            },
            "gpu_usage": {
                "mean": np.mean(self.gpu_usage) if self.gpu_usage else 0,
                "max": np.max(self.gpu_usage) if self.gpu_usage else 0,
                "min": np.min(self.gpu_usage) if self.gpu_usage else 0
            },
            "gpu_memory": {
                "mean": np.mean(self.gpu_memory) if self.gpu_memory else 0,
                "max": np.max(self.gpu_memory) if self.gpu_memory else 0,
                "min": np.min(self.gpu_memory) if self.gpu_memory else 0
            },
            "epochs_completed": self.current_epoch,
            "total_epochs": self.total_epochs,
            "final_metrics": {
                "training_loss": self.training_loss[-1] if self.training_loss else None,
                "validation_loss": self.validation_loss[-1] if self.validation_loss else None,
                "training_accuracy": self.training_accuracy[-1] if self.training_accuracy else None,
                "validation_accuracy": self.validation_accuracy[-1] if self.validation_accuracy else None,
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report as JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(report_dir / f"training_summary_{timestamp}.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate final plots
        self.generate_plots()
        
        # Print summary to console
        print("\n" + "="*50)
        print("TRAINING MONITORING SUMMARY")
        print("="*50)
        print(f"Duration: {report['duration_formatted']}")
        print(f"Epochs: {report['epochs_completed']}/{report['total_epochs']}")
        print(f"CPU Usage: {report['cpu_usage']['mean']:.1f}% (avg), {report['cpu_usage']['max']:.1f}% (peak)")
        print(f"Memory Usage: {report['memory_usage']['mean']:.1f}% (avg), {report['memory_usage']['max']:.1f}% (peak)")
        
        if HAS_GPUTIL:
            print(f"GPU Usage: {report['gpu_usage']['mean']:.1f}% (avg), {report['gpu_usage']['max']:.1f}% (peak)")
            print(f"GPU Memory: {report['gpu_memory']['mean']:.1f}% (avg), {report['gpu_memory']['max']:.1f}% (peak)")
        
        if self.training_loss:
            print(f"Final Training Loss: {report['final_metrics']['training_loss']:.4f}")
        if self.validation_loss:
            print(f"Final Validation Loss: {report['final_metrics']['validation_loss']:.4f}")
        if self.training_accuracy:
            print(f"Final Training Accuracy: {report['final_metrics']['training_accuracy']:.4f}")
        if self.validation_accuracy:
            print(f"Final Validation Accuracy: {report['final_metrics']['validation_accuracy']:.4f}")
        
        print(f"Report saved to: {report_dir}")
        print("="*50)
        
        logger.info(f"Summary report generated and saved to {report_dir}")
    
    def _format_time(self, seconds):
        """Format time in seconds to a readable string (HH:MM:SS)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main():
    """Main function to run the training monitor."""
    parser = argparse.ArgumentParser(description="Monitor ML training progress")
    parser.add_argument("--log_file", type=str, help="Path to the training log file")
    parser.add_argument("--results_dir", type=str, default="training_monitor", 
                       help="Directory to save monitoring results")
    parser.add_argument("--interval", type=int, default=5, 
                       help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    # Find the most recent log file if not specified
    if not args.log_file:
        log_files = list(Path('.').glob('training_*.log'))
        if log_files:
            args.log_file = str(sorted(log_files, key=os.path.getmtime)[-1])
            logger.info(f"Using most recent log file: {args.log_file}")
        else:
            logger.warning("No log file specified and none found. Metrics tracking will be limited.")
    
    # Create and start the monitor
    monitor = TrainingMonitor(
        log_file=args.log_file,
        results_dir=args.results_dir,
        refresh_interval=args.interval
    )
    
    try:
        monitor.start_monitoring()
        
        # Keep running until user interrupts
        print("Training monitor running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    main() 