"""
ARX2 Research API - Celery Tasks
--------------------------------
Celery tasks for the ARX2 Research API.
"""

import os
import sys
import time
import json
from datetime import datetime
import logging
from celery import shared_task, current_task

# Add the arx2 directory to sys.path to import arx2 modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Set tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import mock data for development purposes
from celery_tasks.mock_data import research_frontiers, paper_samples

# These imports would be the actual arx2 imports
# In a production environment, these would be properly imported
# For now, we'll rely on mock data
try:
    from arx2.advanced_ai_analyzer import CONFIG
    from arx2.advanced_ai_analyzer_paper_processor import PaperProcessor
    from arx2.advanced_ai_analyzer_knowledge_base import KnowledgeBase
    from arx2.advanced_ai_analyzer_learning import LearningSystem
    from arx2.advanced_ai_analyzer_engine import RecursiveResearchEngine
    REAL_IMPORTS_AVAILABLE = True
except ImportError:
    REAL_IMPORTS_AVAILABLE = False
    # Define a placeholder CONFIG for development
    CONFIG = {
        'use_gpu': True,
        'use_distributed_training': False,
        'use_knowledge_graph': True,
        'max_papers_per_query': 30,
        'max_papers_total': 300
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Task result storage
# In production, this would be replaced with a proper database
_task_results = {}
_task_statuses = {}
_task_progress = {}
_task_messages = {}

@shared_task(bind=True, name="research_tasks.run_research_analysis")
def run_research_analysis(self, task_id, query, iterations=5, papers_per_query=30, 
                         use_gpu=True, use_knowledge_graph=True, enable_distributed=False):
    """
    Run the ARX2 research analysis as a background task.
    
    Args:
        task_id (str): Unique ID for the task
        query (str): Research query string
        iterations (int): Number of iterations to run
        papers_per_query (int): Number of papers to retrieve per query
        use_gpu (bool): Whether to use GPU acceleration
        use_knowledge_graph (bool): Whether to use knowledge graph capabilities
        enable_distributed (bool): Enable distributed training if multiple GPUs
    
    Returns:
        dict: Analysis results
    """
    # If the real imports are available, use the actual implementation
    # Otherwise, use the mock implementation for development
    if REAL_IMPORTS_AVAILABLE:
        return run_real_research_analysis(
            task_id, query, iterations, papers_per_query, 
            use_gpu, use_knowledge_graph, enable_distributed
        )
    else:
        return mock_research_analysis(
            task_id, query, iterations, papers_per_query
        )

def run_real_research_analysis(task_id, query, iterations=5, papers_per_query=30, 
                         use_gpu=True, use_knowledge_graph=True, enable_distributed=False):
    """
    Run the actual ARX2 research analysis using the real implementation.
    """
    try:
        # Update task status
        _task_statuses[task_id] = "RUNNING"
        _task_messages[task_id] = f"Starting research analysis for query: {query}"
        _task_progress[task_id] = 0.0
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("research_output", f"research_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Update config settings
        CONFIG['use_gpu'] = use_gpu
        CONFIG['use_distributed_training'] = enable_distributed
        CONFIG['use_knowledge_graph'] = use_knowledge_graph
        CONFIG['max_papers_per_query'] = papers_per_query
        CONFIG['max_papers_total'] = papers_per_query * iterations * 2
        
        # Create directories
        models_dir = os.path.join(output_dir, "models")
        papers_dir = os.path.join(output_dir, "papers")
        data_dir = os.path.join(output_dir, "data")
        
        for directory in [models_dir, papers_dir, data_dir]:
            os.makedirs(directory, exist_ok=True)
            
        CONFIG['models_dir'] = models_dir
        CONFIG['papers_dir'] = papers_dir
        CONFIG['DATA_DIR'] = data_dir
        
        # Update progress
        _task_progress[task_id] = 0.05
        _task_messages[task_id] = "Initializing research components..."
        
        # Initialize components
        logger.info(f"Initializing research components for task {task_id}...")
        paper_processor = PaperProcessor(CONFIG)
        knowledge_base = KnowledgeBase(CONFIG)
        learning_system = LearningSystem(CONFIG, paper_processor, knowledge_base)
        
        # Initialize the recursive research engine
        research_engine = RecursiveResearchEngine(
            paper_processor=paper_processor,
            knowledge_base=knowledge_base,
            learning_system=learning_system,
            config=CONFIG
        )
        
        # Update progress
        _task_progress[task_id] = 0.1
        _task_messages[task_id] = "Starting research iterations..."
        
        # Run the research
        total_papers = 0
        results = []
        
        for i in range(iterations):
            iteration_query = query if i == 0 else research_engine.generate_next_query()
            
            # Update progress
            _task_progress[task_id] = 0.1 + (0.8 * (i / iterations))
            _task_messages[task_id] = f"Running iteration {i+1}/{iterations}: {iteration_query}"
            
            # Run iteration and get results
            iteration_result = research_engine.run_iteration(iteration_query)
            results.append(iteration_result)
            
            # Update total papers count
            total_papers += len(iteration_result.get('papers', []))
            
            # Update progress
            logger.info(f"Completed iteration {i+1}/{iterations}")
        
        # Consolidate and save results
        _task_progress[task_id] = 0.9
        _task_messages[task_id] = "Consolidating results and generating report..."
        
        # Get final research frontiers
        research_frontiers = research_engine.get_research_frontiers()
        
        # Save the results
        summary_path = os.path.join(output_dir, "research_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Research Summary for: {query}\n")
            f.write(f"Date: {timestamp}\n")
            f.write(f"Iterations: {iterations}\n\n")
            f.write("Research Frontiers:\n")
            
            for i, frontier in enumerate(research_frontiers[:10], 1):
                f.write(f"{i}. {frontier}\n")
            
            f.write(f"\nTotal papers processed: {total_papers}\n")
            f.write(f"Total concepts learned: {len(research_engine.knowledge_base.get_all_concepts())}\n")
            
        # Prepare result data
        result_data = {
            "task_id": task_id,
            "query": query,
            "iterations": iterations,
            "total_papers": total_papers,
            "output_directory": output_dir,
            "research_frontiers": research_frontiers[:10],
            "total_concepts": len(research_engine.knowledge_base.get_all_concepts()),
            "completion_time": datetime.now().isoformat()
        }
        
        # Update task status and store results
        _task_statuses[task_id] = "SUCCESS"
        _task_results[task_id] = result_data
        _task_progress[task_id] = 1.0
        _task_messages[task_id] = "Research analysis completed successfully!"
        
        logger.info(f"Task {task_id} completed successfully!")
        return result_data
        
    except Exception as e:
        # Log the error
        logger.error(f"Error in task {task_id}: {str(e)}")
        
        # Update task status
        _task_statuses[task_id] = "FAILURE"
        _task_messages[task_id] = f"Error: {str(e)}"
        
        # Re-raise the exception for Celery to handle
        raise e

# Utility functions for task management
def mock_research_analysis(task_id, query, iterations=5, papers_per_query=30):
    """
    Mock function for simulating research analysis for development purposes.
    This allows testing the API without actually running the full analysis.
    """
    try:
        # Update task status
        _task_statuses[task_id] = "RUNNING"
        _task_messages[task_id] = f"Starting mock research analysis for query: {query}"
        _task_progress[task_id] = 0.0
        
        # Simulate work with delays
        total_steps = iterations + 2  # Initialization + iterations + finalization
        
        # Initialization
        time.sleep(2)  # Simulate initialization
        _task_progress[task_id] = 1 / total_steps
        _task_messages[task_id] = "Initialized research components..."
        
        # Simulate iterations
        for i in range(iterations):
            time.sleep(3)  # Simulate iteration work
            _task_progress[task_id] = (i + 2) / total_steps
            _task_messages[task_id] = f"Completed iteration {i+1}/{iterations}"
        
        # Finalization
        time.sleep(1)  # Simulate finalization
        
        # Prepare mock results based on the research summary
        mock_results = {
            "task_id": task_id,
            "query": query,
            "iterations": iterations,
            "total_papers": 767,
            "output_directory": "research_output/mock_results",
            "research_frontiers": research_frontiers,  # Using the imported mock data
            "paper_details": paper_samples,  # Using the imported mock data
            "total_concepts": 13064,
            "completion_time": datetime.now().isoformat()
        }
        
        # Update task status and store results
        _task_statuses[task_id] = "SUCCESS"
        _task_results[task_id] = mock_results
        _task_progress[task_id] = 1.0
        _task_messages[task_id] = "Mock research analysis completed successfully!"
        
        return mock_results
        
    except Exception as e:
        # Log the error
        logger.error(f"Error in mock task {task_id}: {str(e)}")
        
        # Update task status
        _task_statuses[task_id] = "FAILURE"
        _task_messages[task_id] = f"Error: {str(e)}"
        
        # Re-raise the exception
        raise e 