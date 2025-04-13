#!/usr/bin/env python3
"""
Advanced AI Research System - Enhanced Mode
------------------------------------------
Runs the AI research system with all the advanced features enabled.
"""

import os
# Set tokenizers parallelism before importing any HuggingFace libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
import logging
import time
from datetime import datetime
import traceback

from advanced_ai_analyzer import CONFIG, logger
from advanced_ai_analyzer_paper_processor import PaperProcessor
from advanced_ai_analyzer_knowledge_base import KnowledgeBase
from advanced_ai_analyzer_learning import LearningSystem
from advanced_ai_analyzer_engine import RecursiveResearchEngine
from utils.progress_display import ProgressTracker

def configure_logging(log_level=logging.INFO):
    """Configure logging with file and console output
    
    Args:
        log_level: Logging level (default: INFO)
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"ai_research_{timestamp}.log")
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info(f"Logging initialized at level {logging.getLevelName(log_level)}")
    logger.info(f"Log file: {log_file}")

def setup_argparse():
    """Setup command line argument parsing"""
    parser = argparse.ArgumentParser(description='Advanced AI Research System - Enhanced Mode')
    
    parser.add_argument('--query', type=str, default="reinforcement learning deep neural networks",
                      help='Initial research query')
    parser.add_argument('--iterations', type=int, default=5,
                      help='Maximum number of research iterations')
    parser.add_argument('--papers', type=int, default=30,
                      help='Maximum papers to download per query')
    parser.add_argument('--gpu', action='store_true',
                      help='Force GPU usage (even if not automatically detected)')
    parser.add_argument('--cpu', action='store_true',
                      help='Force CPU usage (disable GPU)')
    parser.add_argument('--distributed', action='store_true',
                      help='Enable distributed training if multiple GPUs available')
    parser.add_argument('--incremental', action='store_true',
                      help='Enable incremental learning (default: True)')
    parser.add_argument('--knowledge-graph', action='store_true',
                      help='Enable knowledge graph generation (default: True)')
    parser.add_argument('--output-dir', type=str, default="research_output",
                      help='Directory to store output files')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    return parser.parse_args()

def update_config(args):
    """Update configuration based on command line arguments"""
    # Update GPU settings
    if args.gpu:
        CONFIG['use_gpu'] = True
    elif args.cpu:
        CONFIG['use_gpu'] = False
        
    # Update multi-GPU settings
    if args.distributed:
        CONFIG['use_distributed_training'] = True
    
    # Update learning settings
    CONFIG['use_incremental_learning'] = not args.gpu  # Enabled by default
    CONFIG['use_knowledge_graph'] = not args.knowledge_graph  # Enabled by default
    
    # Update paper limits
    CONFIG['max_papers_per_query'] = args.papers
    CONFIG['max_papers_total'] = args.papers * args.iterations * 2
    
    # Create output directories
    base_dir = args.output_dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    papers_dir = os.path.join(run_dir, 'papers')
    models_dir = os.path.join(run_dir, 'models')
    reports_dir = os.path.join(models_dir, 'reports')
    
    CONFIG['papers_dir'] = papers_dir
    CONFIG['models_dir'] = models_dir
    
    # Create directories
    for directory in [run_dir, papers_dir, models_dir, reports_dir]:
        os.makedirs(directory, exist_ok=True)
        
    # Update logging level if debug mode
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
            
    logger.info(f"Updated configuration:")
    logger.info(f"  GPU Mode: {'Enabled' if CONFIG['use_gpu'] else 'Disabled'}")
    logger.info(f"  Multi-GPU: {'Enabled' if CONFIG['use_distributed_training'] else 'Disabled'}")
    logger.info(f"  Incremental Learning: {'Enabled' if CONFIG['use_incremental_learning'] else 'Disabled'}")
    logger.info(f"  Knowledge Graph: {'Enabled' if CONFIG['use_knowledge_graph'] else 'Disabled'}")
    logger.info(f"  Papers per query: {CONFIG['max_papers_per_query']}")
    logger.info(f"  Output directory: {run_dir}")

def print_welcome_message(args):
    """Print welcome message with configuration details"""
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║       ADVANCED AI RESEARCH SYSTEM - ENHANCED MODE            ║
    ║                                                              ║
    ║  A recursively self-iterating AI researcher with advanced    ║
    ║  transformer-based embeddings, knowledge graph capabilities, ║
    ║  and distributed training.                                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Initial query: {args.query}
    Iterations: {args.iterations}
    Papers per query: {args.papers}
    Running on: {"GPU" if CONFIG['use_gpu'] else "CPU"}
    """)

def main():
    try:
        parser = argparse.ArgumentParser(description='Run AI research in advanced mode')
        parser.add_argument('--query', type=str, default="reinforcement learning deep neural networks", 
                            help='Initial research query')
        parser.add_argument('--iterations', type=int, default=5, help='Number of iterations to run')
        parser.add_argument('--papers', type=int, default=30, help='Papers to download per query')
        parser.add_argument('--models-dir', type=str, default='models', help='Directory for model storage')
        parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
        parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
        parser.add_argument('--distributed', action='store_true', help='Use distributed training if multiple GPUs')
        parser.add_argument('--output-dir', type=str, default='research_output', help='Directory for output')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        
        args = parser.parse_args()
        
        # Configure logging level based on debug flag
        log_level = logging.DEBUG if args.debug else logging.INFO
        configure_logging(log_level)
        
        # Update config with command-line arguments
        if args.gpu:
            CONFIG['use_gpu'] = True
        if args.cpu:
            CONFIG['use_gpu'] = False
        if args.distributed:
            CONFIG['use_distributed_training'] = True
        
        # Create timestamp-based output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"research_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models directory
        models_dir = os.path.join(output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        CONFIG['models_dir'] = models_dir
        
        # Create other directories
        papers_dir = os.path.join(output_dir, "papers")
        os.makedirs(papers_dir, exist_ok=True)
        CONFIG['papers_dir'] = papers_dir
        
        data_dir = os.path.join(output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        CONFIG['DATA_DIR'] = data_dir
        
        # Set up paper count
        CONFIG['max_papers_per_query'] = args.papers
        
        # Initialize the recursive research engine
        engine = RecursiveResearchEngine(papers_dir=papers_dir, models_dir=models_dir)
        
        # Run the research iterations
        initial_queries = args.query.split('|') if '|' in args.query else [args.query]
        
        logger.info(f"Starting recursive research with {len(initial_queries)} initial queries, running for {args.iterations} iterations")
        
        # Begin research
        final_report = engine.run(
            initial_queries=initial_queries,
            iterations=args.iterations,
            max_results=args.papers
        )
        
        # Print final summary
        if final_report:
            frontiers = final_report.get('research_frontiers', [])
            if frontiers:
                print(f"\nIdentified {len(frontiers)} research frontiers:")
                for i, frontier in enumerate(frontiers[:5], 1):
                    print(f"{i}. {frontier}")
                    
            print(f"\nResearch complete! Results saved to {output_dir}")
            print(f"Total papers processed: {final_report.get('statistics', {}).get('total_papers', 0)}")
            print(f"Total concepts learned: {final_report.get('statistics', {}).get('total_concepts', 0)}")
            
            # Save a readable summary
            summary_path = os.path.join(output_dir, "research_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Research Summary for: {args.query}\n")
                f.write(f"Date: {timestamp}\n")
                f.write(f"Iterations: {args.iterations}\n\n")
                
                f.write("Research Frontiers:\n")
                for i, frontier in enumerate(frontiers, 1):
                    f.write(f"{i}. {frontier}\n")
                
                f.write(f"\nTotal papers processed: {final_report.get('statistics', {}).get('total_papers', 0)}\n")
                f.write(f"Total concepts learned: {final_report.get('statistics', {}).get('total_concepts', 0)}\n")
                f.write(f"Execution time: {final_report.get('statistics', {}).get('execution_time_formatted', 'N/A')}\n")
    
    except KeyboardInterrupt:
        print("\nResearch interrupted by user.")
        logger.info("Research process interrupted by user")
        return 1
    except Exception as e:
        logger.critical(f"Uncaught exception in main process: {e}", exc_info=True)
        print(f"\nError: {e}")
        traceback_info = traceback.format_exc()
        logger.critical(f"Traceback: {traceback_info}")
        
        # Try to save a crash report
        try:
            crash_dir = os.path.join(args.output_dir, "crash_reports")
            os.makedirs(crash_dir, exist_ok=True)
            crash_file = os.path.join(crash_dir, f"crash_{timestamp}.txt")
            with open(crash_file, 'w') as f:
                f.write(f"Crash Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Query: {args.query}\n")
                f.write(f"Error: {e}\n\n")
                f.write("Traceback:\n")
                f.write(traceback_info)
            print(f"Crash report saved to {crash_file}")
        except:
            # If we can't save a crash report, just continue
            pass
        
        return 1
        
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1) 