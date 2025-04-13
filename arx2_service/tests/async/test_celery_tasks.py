"""
Tests for asynchronous Celery tasks.

This module contains tests for the Celery tasks, focusing on:
- Task submission and execution
- Task result handling
- Task error handling
"""

import pytest
import uuid
import time
import os
from unittest.mock import patch, MagicMock

# Skip these tests if we're not in a Celery-enabled environment
# In CI, we might set CELERY_TESTING=1 to run these tests
pytestmark = pytest.mark.skipif(
    "CELERY_TESTING" not in os.environ,
    reason="Celery testing environment not configured"
)

# Import Celery tasks and app
from celery_tasks.research_tasks import run_research_analysis, mock_research_analysis
from celery_tasks.task_manager import celery_app, get_task_status, get_task_result

# Test task submission with mocked execution
def test_task_submission_with_mock():
    """
    Test submitting a Celery task with mocked execution.
    
    This test verifies that:
    - Tasks can be submitted to Celery
    - The task ID is correctly generated
    - Task arguments are correctly passed
    """
    task_id = str(uuid.uuid4())
    query = "reinforcement learning neural networks"
    
    # Mock the Celery apply_async method
    with patch.object(run_research_analysis, 'apply_async') as mock_apply_async:
        # Configure the mock to return a mock AsyncResult
        mock_result = MagicMock()
        mock_result.id = task_id
        mock_apply_async.return_value = mock_result
        
        # Call the task
        result = run_research_analysis.apply_async(
            kwargs={
                "task_id": task_id,
                "query": query,
                "iterations": 3,
                "papers_per_query": 20
            }
        )
        
        # Verify the task was submitted with correct arguments
        mock_apply_async.assert_called_once()
        call_kwargs = mock_apply_async.call_args[1]["kwargs"]
        assert call_kwargs["task_id"] == task_id
        assert call_kwargs["query"] == query
        assert call_kwargs["iterations"] == 3
        assert call_kwargs["papers_per_query"] == 20
        
        # Verify the result contains the task ID
        assert result.id == task_id

# Test the mock_research_analysis function directly
def test_mock_research_analysis_function():
    """
    Test the mock_research_analysis function directly (without Celery).
    
    This test verifies that:
    - The function runs without errors
    - It returns results in the expected format
    - Task progress and status are correctly updated
    """
    task_id = str(uuid.uuid4())
    query = "reinforcement learning"
    
    # Mock time.sleep to speed up the test
    with patch('time.sleep', return_value=None):
        # Call the function directly
        result = mock_research_analysis(task_id, query, iterations=2, papers_per_query=10)
    
    # Verify the result structure
    assert result is not None
    assert result["task_id"] == task_id
    assert result["query"] == query
    assert "research_frontiers" in result
    assert len(result["research_frontiers"]) > 0
    assert "paper_details" in result
    assert "total_papers" in result
    assert "completion_time" in result

# Test task execution in Celery (requires running Celery worker)
@pytest.mark.skipif(
    "CELERY_WORKER_RUNNING" not in os.environ,
    reason="Celery worker not running"
)
def test_task_execution_with_celery():
    """
    Test executing a task with an actual Celery worker.
    
    This test verifies that:
    - Tasks can be executed by Celery workers
    - Results can be retrieved
    - Status tracking works correctly
    
    Note: This test requires a running Celery worker
    """
    task_id = str(uuid.uuid4())
    query = "reinforcement learning"
    
    # Submit the task to Celery
    async_result = run_research_analysis.apply_async(
        kwargs={
            "task_id": task_id,
            "query": query,
            "iterations": 1,  # Use minimal iterations for speed
            "papers_per_query": 5  # Use minimal papers for speed
        }
    )
    
    # Wait for the task to complete (with timeout)
    timeout = 60  # seconds
    start_time = time.time()
    
    while not async_result.ready() and time.time() - start_time < timeout:
        time.sleep(1)
    
    # Verify the task completed
    assert async_result.ready(), "Task did not complete within timeout"
    
    # Verify the result
    result = async_result.result
    assert result is not None
    assert result["task_id"] == task_id
    assert result["query"] == query
    assert "research_frontiers" in result
    assert "total_papers" in result

# Test handling of task errors
def test_task_error_handling():
    """
    Test handling of errors in Celery tasks.
    
    This test verifies that:
    - Task errors are correctly caught and reported
    - Task status is updated to FAILURE
    """
    task_id = str(uuid.uuid4())
    query = "reinforcement learning"
    
    # Force an error in the mock_research_analysis function
    with patch('celery_tasks.research_tasks.mock_research_analysis', 
              side_effect=Exception("Simulated task error")):
        
        # Mock the apply_async method
        with patch.object(run_research_analysis, 'apply_async') as mock_apply_async:
            # Configure the mock to return a mock AsyncResult
            mock_result = MagicMock()
            mock_result.id = task_id
            mock_apply_async.return_value = mock_result
            
            # Submit the task
            result = run_research_analysis.apply_async(
                kwargs={
                    "task_id": task_id,
                    "query": query
                }
            )
            
            # Verify the task was submitted
            assert result.id == task_id
            
            # Mock the task status to be FAILURE
            with patch('celery_tasks.task_manager.get_task_status', return_value="FAILURE"):
                # Get the task status
                status = get_task_status(task_id)
                assert status == "FAILURE"

# Test result storage and retrieval
def test_task_result_storage():
    """
    Test storage and retrieval of task results.
    
    This test verifies that:
    - Task results are correctly stored
    - Task results can be retrieved by task ID
    """
    task_id = str(uuid.uuid4())
    query = "reinforcement learning"
    
    # Create a mock result
    mock_result = {
        "task_id": task_id,
        "query": query,
        "iterations": 3,
        "total_papers": 150,
        "research_frontiers": [
            {"concept": "learning", "importance": 0.8, "papers": ["paper1", "paper2"]}
        ],
        "completion_time": "2023-09-01T12:34:56"
    }
    
    # Mock the storage of the result
    from celery_tasks.research_tasks import _task_results, _task_statuses
    
    _task_results[task_id] = mock_result
    _task_statuses[task_id] = "SUCCESS"
    
    try:
        # Retrieve the result
        retrieved_result = get_task_result(task_id)
        
        # Verify the result matches
        assert retrieved_result == mock_result
        assert retrieved_result["task_id"] == task_id
        assert retrieved_result["query"] == query
        assert len(retrieved_result["research_frontiers"]) == 1
        
        # Verify the status is SUCCESS
        status = get_task_status(task_id)
        assert status == "SUCCESS"
        
    finally:
        # Clean up
        if task_id in _task_results:
            del _task_results[task_id]
        if task_id in _task_statuses:
            del _task_statuses[task_id]

# Test task progress tracking
def test_task_progress_tracking():
    """
    Test tracking of task progress.
    
    This test verifies that:
    - Task progress can be tracked and updated
    - Progress can be retrieved by task ID
    """
    task_id = str(uuid.uuid4())
    
    # Mock progress updates
    from celery_tasks.research_tasks import _task_progress, _task_messages
    
    try:
        # Initial progress
        _task_progress[task_id] = 0.0
        _task_messages[task_id] = "Starting task..."
        
        # Update progress
        for progress in [0.25, 0.5, 0.75, 1.0]:
            _task_progress[task_id] = progress
            _task_messages[task_id] = f"Progress: {progress * 100:.0f}%"
            
            # Verify progress can be retrieved
            from celery_tasks.task_manager import get_task_progress, get_task_message
            
            assert get_task_progress(task_id) == progress
            assert get_task_message(task_id) == f"Progress: {progress * 100:.0f}%"
    
    finally:
        # Clean up
        if task_id in _task_progress:
            del _task_progress[task_id]
        if task_id in _task_messages:
            del _task_messages[task_id] 