"""
Test fixtures for the ARX2 Research API service.

This module contains reusable pytest fixtures that can be used across
different test modules.
"""

import os
import pytest
import asyncio
from fastapi.testclient import TestClient
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import redis
from unittest.mock import MagicMock, patch
import uuid
from datetime import datetime, timedelta

# Add the parent directory to the path to import the main app
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the FastAPI app and other components
from main import app
from celery_tasks.mock_data import research_frontiers, paper_samples
from celery_tasks.research_tasks import _task_results, _task_statuses, _task_progress, _task_messages
from config.app_config import settings

# Mock Redis fixtures
@pytest.fixture
def mock_redis():
    """
    Mock Redis connection for testing.
    
    Returns:
        MagicMock: A mock Redis client
    """
    mock_redis_client = MagicMock()
    mock_redis_client.get.return_value = None
    mock_redis_client.set.return_value = True
    mock_redis_client.delete.return_value = True
    return mock_redis_client

# Mock MongoDB fixtures
@pytest.fixture
def mock_mongodb():
    """
    Mock MongoDB client for testing.
    
    Returns:
        MagicMock: A mock MongoDB client
    """
    mock_mongo_client = MagicMock()
    mock_collection = MagicMock()
    mock_mongo_client.__getitem__.return_value.__getitem__.return_value = mock_collection
    
    # Set up find_one to return None (no results) by default
    mock_collection.find_one.return_value = None
    
    # Set up insert_one to return a mock result with inserted_id
    mock_insert_result = MagicMock()
    mock_insert_result.inserted_id = str(uuid.uuid4())
    mock_collection.insert_one.return_value = mock_insert_result
    
    return mock_mongo_client

# FastAPI TestClient fixture
@pytest.fixture
def test_client():
    """
    FastAPI TestClient for testing API endpoints.
    
    Returns:
        TestClient: A FastAPI test client
    """
    with TestClient(app) as client:
        yield client

# Mocked task ID and related data
@pytest.fixture
def mock_task_id():
    """
    Generate a mock task ID for testing.
    
    Returns:
        str: A UUID string
    """
    return str(uuid.uuid4())

@pytest.fixture
def mock_task_data(mock_task_id):
    """
    Create mock task data for testing task status and results.
    
    Args:
        mock_task_id: The mock task ID
        
    Returns:
        dict: A dictionary with task data
    """
    # Sample research task data
    task_data = {
        "task_id": mock_task_id,
        "query": "reinforcement learning deep neural networks",
        "iterations": 3,
        "total_papers": 150,
        "output_directory": f"research_output/mock_{mock_task_id}",
        "research_frontiers": research_frontiers[:5],
        "paper_details": paper_samples[:3],
        "total_concepts": 5000,
        "completion_time": datetime.now().isoformat()
    }
    
    # Update the global task data stores
    _task_results[mock_task_id] = task_data
    _task_statuses[mock_task_id] = "SUCCESS"
    _task_progress[mock_task_id] = 0.33
    _task_messages[mock_task_id] = "Mock research analysis completed successfully!"
    
    yield task_data
    
    # Clean up
    if mock_task_id in _task_results:
        del _task_results[mock_task_id]
    if mock_task_id in _task_statuses:
        del _task_statuses[mock_task_id]
    if mock_task_id in _task_progress:
        del _task_progress[mock_task_id]
    if mock_task_id in _task_messages:
        del _task_messages[mock_task_id]

@pytest.fixture
def mock_running_task_data(mock_task_id):
    """
    Create mock task data for a running task.
    
    Args:
        mock_task_id: The mock task ID
        
    Returns:
        dict: A dictionary with running task data
    """
    # Update the global task data stores for a running task
    _task_statuses[mock_task_id] = "RUNNING"
    _task_progress[mock_task_id] = 0.41
    _task_messages[mock_task_id] = "Processing papers and analyzing research trends..."
    
    yield {
        "task_id": mock_task_id,
        "status": "RUNNING",
        "progress": 0.41,
        "message": "Processing papers and analyzing research trends..."
    }
    
    # Clean up
    if mock_task_id in _task_statuses:
        del _task_statuses[mock_task_id]
    if mock_task_id in _task_progress:
        del _task_progress[mock_task_id]
    if mock_task_id in _task_messages:
        del _task_messages[mock_task_id]

# Authentication test fixtures
@pytest.fixture
def mock_user():
    """
    Mock user data for testing authentication.
    
    Returns:
        dict: A dictionary with user data
    """
    return {
        "id": str(uuid.uuid4()),
        "username": "testuser",
        "email": "test@example.com",
        "created_at": datetime.now().isoformat(),
        "subscription_status": "premium"
    }

@pytest.fixture
def mock_token():
    """
    Mock authentication token for testing.
    
    Returns:
        str: A mock JWT token
    """
    return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNjkzNTAwMDAwfQ.Gkx-Q4291LS9JM4cC1aVWq-wQ7r5thHUK1M9UxRe7Qo"

# Async event loop
@pytest.fixture(scope="session")
def event_loop():
    """
    Fixture for creating an asyncio event loop.
    
    Returns:
        asyncio.AbstractEventLoop: An event loop for async tests
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close() 