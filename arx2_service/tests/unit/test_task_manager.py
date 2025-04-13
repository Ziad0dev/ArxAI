"""
Unit tests for the Celery task manager.

This module contains tests for the task_manager.py module, focusing on:
- Task status retrieval
- Task result retrieval
- Progress and message tracking
"""

import pytest
import uuid
from unittest.mock import patch, MagicMock

from celery_tasks.task_manager import (
    get_task_status,
    get_task_result,
    get_task_progress,
    get_task_message
)
from celery_tasks.research_tasks import (
    _task_results,
    _task_statuses,
    _task_progress,
    _task_messages
)

# Test get_task_status for task in memory storage
def test_get_task_status_from_memory():
    """
    Test retrieving task status from in-memory storage.
    
    This test verifies that:
    - The function correctly retrieves status from in-memory storage
    - It returns the expected status value
    """
    # Set up test data
    task_id = str(uuid.uuid4())
    _task_statuses[task_id] = "RUNNING"
    
    try:
        # Get the status
        status = get_task_status(task_id)
        
        # Verify the result
        assert status == "RUNNING"
        
    finally:
        # Clean up
        if task_id in _task_statuses:
            del _task_statuses[task_id]

# Test get_task_status for task in Celery backend
def test_get_task_status_from_celery():
    """
    Test retrieving task status from Celery backend.
    
    This test verifies that:
    - The function falls back to Celery AsyncResult when task not in memory
    - It correctly returns the Celery task status
    """
    # Generate a task ID that is not in memory storage
    task_id = str(uuid.uuid4())
    
    # Mock the Celery AsyncResult
    mock_async_result = MagicMock()
    mock_async_result.status = "PENDING"
    
    with patch('celery_tasks.task_manager.celery_app.AsyncResult', return_value=mock_async_result):
        # Get the status
        status = get_task_status(task_id)
        
        # Verify the result
        assert status == "PENDING"

# Test get_task_result for completed task in memory storage
def test_get_task_result_from_memory():
    """
    Test retrieving task result from in-memory storage.
    
    This test verifies that:
    - The function correctly retrieves result from in-memory storage
    - It returns the expected result data
    """
    # Set up test data
    task_id = str(uuid.uuid4())
    expected_result = {"key": "value", "test": 123}
    _task_results[task_id] = expected_result
    
    try:
        # Get the result
        result = get_task_result(task_id)
        
        # Verify the result
        assert result == expected_result
        
    finally:
        # Clean up
        if task_id in _task_results:
            del _task_results[task_id]

# Test get_task_result for task in Celery backend
def test_get_task_result_from_celery():
    """
    Test retrieving task result from Celery backend.
    
    This test verifies that:
    - The function falls back to Celery AsyncResult when task not in memory
    - It correctly returns the Celery task result for successful tasks
    - It returns None for tasks that are not ready or failed
    """
    # Generate a task ID that is not in memory storage
    task_id = str(uuid.uuid4())
    
    # Mock the Celery AsyncResult for a successful task
    mock_async_result = MagicMock()
    mock_async_result.ready.return_value = True
    mock_async_result.successful.return_value = True
    mock_async_result.result = {"key": "value", "test": 123}
    
    with patch('celery_tasks.task_manager.celery_app.AsyncResult', return_value=mock_async_result):
        # Get the result
        result = get_task_result(task_id)
        
        # Verify the result
        assert result == mock_async_result.result
    
    # Now test with a task that is not ready
    mock_async_result.ready.return_value = False
    
    with patch('celery_tasks.task_manager.celery_app.AsyncResult', return_value=mock_async_result):
        # Get the result
        result = get_task_result(task_id)
        
        # Verify it returns None
        assert result is None
    
    # Test with a task that failed
    mock_async_result.ready.return_value = True
    mock_async_result.successful.return_value = False
    
    with patch('celery_tasks.task_manager.celery_app.AsyncResult', return_value=mock_async_result):
        # Get the result
        result = get_task_result(task_id)
        
        # Verify it returns None
        assert result is None

# Test get_task_progress
def test_get_task_progress():
    """
    Test retrieving task progress.
    
    This test verifies that:
    - The function correctly retrieves progress from in-memory storage
    - It returns None for non-existent tasks
    """
    # Set up test data
    task_id = str(uuid.uuid4())
    _task_progress[task_id] = 0.75
    
    try:
        # Get the progress
        progress = get_task_progress(task_id)
        
        # Verify the result
        assert progress == 0.75
        
        # Test with non-existent task
        non_existent_task_id = str(uuid.uuid4())
        progress = get_task_progress(non_existent_task_id)
        assert progress is None
        
    finally:
        # Clean up
        if task_id in _task_progress:
            del _task_progress[task_id]

# Test get_task_message
def test_get_task_message():
    """
    Test retrieving task message.
    
    This test verifies that:
    - The function correctly retrieves message from in-memory storage
    - It returns None for non-existent tasks
    """
    # Set up test data
    task_id = str(uuid.uuid4())
    _task_messages[task_id] = "Processing papers..."
    
    try:
        # Get the message
        message = get_task_message(task_id)
        
        # Verify the result
        assert message == "Processing papers..."
        
        # Test with non-existent task
        non_existent_task_id = str(uuid.uuid4())
        message = get_task_message(non_existent_task_id)
        assert message is None
        
    finally:
        # Clean up
        if task_id in _task_messages:
            del _task_messages[task_id] 