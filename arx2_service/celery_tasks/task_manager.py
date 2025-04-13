"""
ARX2 Research API - Celery Task Manager
---------------------------------------
Manages Celery tasks and provides utility functions for task status and results.
"""

from celery import Celery
import os
from typing import Any, Dict, Optional

# Import task storage references from research_tasks
from celery_tasks.research_tasks import _task_results, _task_statuses, _task_progress, _task_messages

# Configure Celery
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

celery_app = Celery("arx2_service", 
                   broker=REDIS_URL,
                   backend=REDIS_URL)

# Configure Celery settings
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_time_limit=3600 * 8,  # 8 hours
    task_soft_time_limit=3600 * 7,  # 7 hours
)

# Include tasks
celery_app.autodiscover_tasks(["celery_tasks"], force=True)

def get_task_status(task_id: str) -> str:
    """
    Get the status of a task.
    
    Args:
        task_id (str): The ID of the task
        
    Returns:
        str: The status of the task (PENDING, RUNNING, SUCCESS, FAILURE)
    """
    # Check in-memory storage first (for development/testing)
    if task_id in _task_statuses:
        return _task_statuses[task_id]
    
    # If not found in memory, check Celery (for production)
    task = celery_app.AsyncResult(task_id)
    return task.status

def get_task_result(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the result of a completed task.
    
    Args:
        task_id (str): The ID of the task
        
    Returns:
        Optional[Dict[str, Any]]: The result of the task, or None if not available
    """
    # Check in-memory storage first (for development/testing)
    if task_id in _task_results:
        return _task_results[task_id]
    
    # If not found in memory, check Celery (for production)
    task = celery_app.AsyncResult(task_id)
    if task.ready() and task.successful():
        return task.result
    
    return None

def get_task_progress(task_id: str) -> Optional[float]:
    """
    Get the progress of a task.
    
    Args:
        task_id (str): The ID of the task
        
    Returns:
        Optional[float]: The progress of the task (0.0 to 1.0), or None if not available
    """
    if task_id in _task_progress:
        return _task_progress[task_id]
    
    return None

def get_task_message(task_id: str) -> Optional[str]:
    """
    Get the current message of a task.
    
    Args:
        task_id (str): The ID of the task
        
    Returns:
        Optional[str]: The current message of the task, or None if not available
    """
    if task_id in _task_messages:
        return _task_messages[task_id]
    
    return None 