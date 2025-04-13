"""
Tests for the research API endpoints.

This module contains tests for the /research/* API endpoints, focusing on:
- Starting a new research analysis
- Retrieving research results
- Error handling
"""

import pytest
import json
from unittest.mock import patch, MagicMock
import uuid
from datetime import datetime

# Test the /research/analyze endpoint
@pytest.mark.parametrize(
    "query_data,expected_status", [
        ({"query": "reinforcement learning", "iterations": 3, "papers_per_query": 20}, 200),
        ({"query": "t", "iterations": 3, "papers_per_query": 20}, 422),  # Too short query
        ({"query": "reinforcement learning", "iterations": 0, "papers_per_query": 20}, 422),  # Invalid iterations
        ({"query": "reinforcement learning", "iterations": 3, "papers_per_query": 0}, 422),  # Invalid papers count
    ]
)
def test_start_research_analysis(test_client, query_data, expected_status):
    """
    Test starting a new research analysis task.
    
    This test verifies that:
    - Valid research queries are accepted and return a task ID
    - Invalid queries (too short, invalid parameters) are rejected with appropriate errors
    - The response format matches expectations
    
    Args:
        test_client: FastAPI test client
        query_data: Input data for the research query
        expected_status: Expected HTTP status code
    """
    with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')):
        # Creating a deeper mock to avoid actual Celery connection
        mock_celery_result = MagicMock()
        mock_celery_result.id = "12345678-1234-5678-1234-567812345678"
        
        with patch('celery_tasks.research_tasks.run_research_analysis.delay', return_value=mock_celery_result) as mock_task:
            
            # For invalid requests (422), we don't need to patch routes.py code
            if expected_status == 422:
                response = test_client.post("/research/analyze", json=query_data)
                assert response.status_code == expected_status
                return
            
            response = test_client.post("/research/analyze", json=query_data)
            
            assert response.status_code == expected_status
            
            if expected_status == 200:
                data = response.json()
                assert data["task_id"] == "12345678-1234-5678-1234-567812345678"
                assert data["status"] == "queued"
                assert data["query"] == query_data["query"]
                assert "timestamp" in data
                assert "estimated_completion" in data
                
                # Verify that the Celery task was called with correct parameters
                mock_task.assert_called_once()
                call_args = mock_task.call_args[1]
                assert call_args["task_id"] == "12345678-1234-5678-1234-567812345678"
                assert call_args["query"] == query_data["query"]
                assert call_args["iterations"] == query_data["iterations"]
                assert call_args["papers_per_query"] == query_data["papers_per_query"]

# Test the /research/results/{task_id} endpoint for a successful task
def test_get_research_results_success(test_client, mock_task_data):
    """
    Test retrieving results for a successful research task.
    
    This test verifies that:
    - Completed tasks return the correct result data
    - The response format matches expectations
    
    Args:
        test_client: FastAPI test client
        mock_task_data: Mock task data (completed successfully)
    """
    task_id = mock_task_data["task_id"]
    
    response = test_client.get(f"/research/results/{task_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    assert data["status"] == "SUCCESS"
    assert data["progress"] == 0.33
    assert data["message"] == "Mock research analysis completed successfully!"
    assert data["result"] is not None
    assert "research_frontiers" in data["result"]
    assert len(data["result"]["research_frontiers"]) == 5

# Test the /research/results/{task_id} endpoint for a running task
def test_get_research_results_running(test_client, mock_running_task_data):
    """
    Test retrieving status for a running research task.
    
    This test verifies that:
    - Running tasks return the correct status information
    - The progress value is properly reported
    - No result data is included for incomplete tasks
    
    Args:
        test_client: FastAPI test client
        mock_running_task_data: Mock task data (in progress)
    """
    task_id = mock_running_task_data["task_id"]
    
    response = test_client.get(f"/research/results/{task_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    assert data["status"] == "RUNNING"
    assert data["progress"] == 0.41
    assert data["message"] == "Processing papers and analyzing research trends..."
    assert data["result"] is None

# Test the /research/results/{task_id} endpoint for a non-existent task
def test_get_research_results_not_found(test_client):
    """
    Test retrieving results for a non-existent task.
    
    This test verifies that:
    - Requests for non-existent tasks return appropriate status information
    - No result data is included
    
    Args:
        test_client: FastAPI test client
    """
    non_existent_task_id = str(uuid.uuid4())
    
    response = test_client.get(f"/research/results/{non_existent_task_id}")
    
    assert response.status_code == 200  # Note: Returns 200 with PENDING status instead of 404
    data = response.json()
    assert data["task_id"] == non_existent_task_id
    assert data["status"] == "PENDING"
    assert data["result"] is None

# Test the /research/results/{task_id} endpoint for a failed task
def test_get_research_results_failure(test_client, mock_task_id):
    """
    Test retrieving results for a failed task.
    
    This test verifies that:
    - Failed tasks return an error status
    - Appropriate error information is included
    
    Args:
        test_client: FastAPI test client
        mock_task_id: Mock task ID
    """
    # Set up a failed task
    from celery_tasks.research_tasks import _task_statuses, _task_messages
    
    _task_statuses[mock_task_id] = "FAILURE"
    _task_messages[mock_task_id] = "Error: Failed to process papers"
    
    try:
        response = test_client.get(f"/research/results/{mock_task_id}")
        
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Research task failed"
    finally:
        # Clean up
        if mock_task_id in _task_statuses:
            del _task_statuses[mock_task_id]
        if mock_task_id in _task_messages:
            del _task_messages[mock_task_id] 