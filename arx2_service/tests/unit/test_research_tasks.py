"""
Unit tests for the Celery research tasks.

This module contains tests for the research_tasks.py module, focusing on:
- The core run_research_analysis function
- Mock research_analysis function
- Task status and result tracking
"""

import pytest
import os
import uuid
from unittest.mock import patch, MagicMock
from datetime import datetime

from celery_tasks.research_tasks import (
    mock_research_analysis,
    run_research_analysis,
    _task_results,
    _task_statuses,
    _task_progress,
    _task_messages
)

# Test the mock_research_analysis function
def test_mock_research_analysis():
    """
    Test the mock_research_analysis function.
    
    This test verifies that:
    - The function completes without errors
    - It correctly updates task status and progress
    - It returns results in the expected format
    """
    # Generate a unique task ID for this test
    task_id = str(uuid.uuid4())
    query = "reinforcement learning neural networks"
    
    try:
        # Run the mock analysis
        with patch('time.sleep', return_value=None):  # Skip sleep calls for faster tests
            result = mock_research_analysis(task_id, query, iterations=2, papers_per_query=10)
        
        # Verify results
        assert result is not None
        assert result["task_id"] == task_id
        assert result["query"] == query
        assert "research_frontiers" in result
        assert len(result["research_frontiers"]) > 0
        assert "paper_details" in result
        assert "total_papers" in result
        assert "total_concepts" in result
        assert "completion_time" in result
        
        # Verify task tracking was updated
        assert task_id in _task_results
        assert _task_statuses[task_id] == "SUCCESS"
        assert _task_progress[task_id] == 1.0
        assert "completed" in _task_messages[task_id].lower()
        
    finally:
        # Clean up
        if task_id in _task_results:
            del _task_results[task_id]
        if task_id in _task_statuses:
            del _task_statuses[task_id]
        if task_id in _task_progress:
            del _task_progress[task_id]
        if task_id in _task_messages:
            del _task_messages[task_id]

# Test error handling in mock_research_analysis
def test_mock_research_analysis_error():
    """
    Test error handling in the mock_research_analysis function.
    
    This test verifies that:
    - Errors are correctly caught and reported
    - Task status is updated to FAILURE
    - An appropriate error message is stored
    """
    # Generate a unique task ID for this test
    task_id = str(uuid.uuid4())
    query = "reinforcement learning neural networks"
    
    try:
        # Force an error by patching time.sleep to raise an exception
        with patch('time.sleep', side_effect=Exception("Simulated error")):
            with pytest.raises(Exception):
                mock_research_analysis(task_id, query)
        
        # Verify task tracking was updated with failure
        assert _task_statuses[task_id] == "FAILURE"
        assert "error" in _task_messages[task_id].lower()
        
    finally:
        # Clean up
        if task_id in _task_statuses:
            del _task_statuses[task_id]
        if task_id in _task_messages:
            del _task_messages[task_id]

# Test the run_research_analysis function (which delegates to mock or real implementation)
@patch('celery_tasks.research_tasks.REAL_IMPORTS_AVAILABLE', False)
def test_run_research_analysis_delegates_to_mock():
    """
    Test that run_research_analysis delegates to mock_research_analysis when real imports aren't available.
    
    This test verifies that:
    - The main task function correctly chooses between mock and real implementations
    - It passes parameters correctly to the chosen implementation
    """
    task_id = str(uuid.uuid4())
    query = "reinforcement learning"
    
    # Mock both implementations to track which one is called
    with patch('celery_tasks.research_tasks.mock_research_analysis') as mock_impl:
        with patch('celery_tasks.research_tasks.run_real_research_analysis') as real_impl:
            # Set up the mock to return a result
            mock_impl.return_value = {"task_id": task_id, "status": "SUCCESS"}
            
            # Create a mock self object for the Celery task
            mock_self = MagicMock()
            
            # Call the function with explicit parameters to avoid duplication
            run_research_analysis.run(mock_self, task_id, query, iterations=3)
            
            # Verify mock implementation was called with correct parameters
            mock_impl.assert_called_once()
            args, kwargs = mock_impl.call_args
            assert args[0] == task_id
            assert args[1] == query
            assert kwargs.get('iterations') == 3
            
            # Verify real implementation was not called
            real_impl.assert_not_called()

# Test with real imports available
@patch('celery_tasks.research_tasks.REAL_IMPORTS_AVAILABLE', True)
def test_run_research_analysis_delegates_to_real():
    """
    Test that run_research_analysis delegates to run_real_research_analysis when real imports are available.
    
    This test verifies that:
    - The main task function correctly chooses between mock and real implementations
    - It passes parameters correctly to the chosen implementation
    """
    task_id = str(uuid.uuid4())
    query = "reinforcement learning"
    
    # Mock both implementations to track which one is called
    with patch('celery_tasks.research_tasks.mock_research_analysis') as mock_impl:
        with patch('celery_tasks.research_tasks.run_real_research_analysis') as real_impl:
            # Set up the real impl to return a result
            real_impl.return_value = {"task_id": task_id, "status": "SUCCESS"}
            
            # Create a mock self object for the Celery task
            mock_self = MagicMock()
            
            # Call the function with explicit parameters to avoid duplication
            run_research_analysis.run(mock_self, task_id, query, iterations=3, use_gpu=True)
            
            # Verify real implementation was called with correct parameters
            real_impl.assert_called_once()
            args, kwargs = real_impl.call_args
            assert args[0] == task_id
            assert args[1] == query
            assert kwargs.get('iterations') == 3
            assert kwargs.get('use_gpu') is True
            
            # Verify mock implementation was not called
            mock_impl.assert_not_called() 