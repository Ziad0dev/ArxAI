"""
Integration tests for Redis interactions.

This module contains tests for Redis operations, focusing on:
- Connection handling
- Key-value operations
- List operations
- Expiration and TTL
- Pub/Sub operations
"""

import pytest
import os
import redis
import uuid
import json
import time
from unittest.mock import patch, MagicMock

# Skip these tests if the Redis URI environment variable is not set
# In CI environments, we would set this to a test Redis instance
pytestmark = pytest.mark.skipif(
    "REDIS_TEST_URI" not in os.environ,
    reason="Redis test URI not configured"
)

# Fixture for Redis connection
@pytest.fixture
def redis_client():
    """
    Create a Redis client for testing.
    
    This fixture connects to a test Redis instance using the REDIS_TEST_URI
    environment variable. Tests are isolated by using unique key prefixes.
    
    Returns:
        Redis: A Redis client connected to the test instance
    """
    # Use a test Redis URI from environment or fall back to a default
    uri = os.environ.get("REDIS_TEST_URI", "redis://localhost:6379/1")
    client = redis.from_url(uri)
    
    # Generate a unique prefix for test isolation
    prefix = f"test:{uuid.uuid4().hex}:"
    
    # Create a client data object to pass the prefix to tests
    client.test_prefix = prefix
    
    # Return the client
    yield client
    
    # Clean up - delete all keys with our test prefix
    for key in client.keys(f"{prefix}*"):
        client.delete(key)

# Test basic key-value operations
def test_redis_key_value_operations(redis_client):
    """
    Test basic key-value operations on Redis.
    
    This test verifies that:
    - Keys can be set
    - Keys can be retrieved
    - Keys can be checked for existence
    - Keys can be deleted
    
    Args:
        redis_client: Redis client fixture
    """
    # Skip if we're using a mock
    if isinstance(redis_client, MagicMock):
        pytest.skip("Using mock Redis, skipping actual Redis operations")
    
    prefix = redis_client.test_prefix
    
    # Set a key
    key = f"{prefix}user:1:name"
    redis_client.set(key, "John Doe")
    
    # Get the key
    value = redis_client.get(key)
    assert value.decode('utf-8') == "John Doe"
    
    # Check if key exists
    assert redis_client.exists(key) == 1
    
    # Delete the key
    redis_client.delete(key)
    assert redis_client.exists(key) == 0
    
    # Test with non-string values (using JSON serialization)
    user_data = {
        "id": "user1",
        "name": "John Doe",
        "email": "john@example.com",
        "subscription": "premium"
    }
    
    user_key = f"{prefix}user:1:data"
    redis_client.set(user_key, json.dumps(user_data))
    
    # Get and deserialize
    user_json = redis_client.get(user_key)
    retrieved_user = json.loads(user_json)
    
    assert retrieved_user["id"] == "user1"
    assert retrieved_user["name"] == "John Doe"
    assert retrieved_user["subscription"] == "premium"

# Test list operations
def test_redis_list_operations(redis_client):
    """
    Test list operations on Redis.
    
    This test verifies that:
    - Items can be pushed to lists
    - Lists can be retrieved
    - Items can be popped from lists
    
    Args:
        redis_client: Redis client fixture
    """
    # Skip if we're using a mock
    if isinstance(redis_client, MagicMock):
        pytest.skip("Using mock Redis, skipping actual Redis operations")
    
    prefix = redis_client.test_prefix
    list_key = f"{prefix}tasks:pending"
    
    # Push items to a list
    redis_client.rpush(list_key, "task1", "task2", "task3")
    
    # Get list length
    assert redis_client.llen(list_key) == 3
    
    # Get all items
    items = redis_client.lrange(list_key, 0, -1)
    assert [item.decode('utf-8') for item in items] == ["task1", "task2", "task3"]
    
    # Pop an item
    item = redis_client.lpop(list_key)
    assert item.decode('utf-8') == "task1"
    
    # Verify list was updated
    items = redis_client.lrange(list_key, 0, -1)
    assert [item.decode('utf-8') for item in items] == ["task2", "task3"]
    
    # Push more items
    redis_client.rpush(list_key, "task4", "task5")
    
    # Verify list was updated
    items = redis_client.lrange(list_key, 0, -1)
    assert [item.decode('utf-8') for item in items] == ["task2", "task3", "task4", "task5"]

# Test expiration (TTL)
def test_redis_expiration(redis_client):
    """
    Test key expiration in Redis.
    
    This test verifies that:
    - Keys can be set with expiration
    - TTL can be checked
    - Keys expire automatically
    
    Args:
        redis_client: Redis client fixture
    """
    # Skip if we're using a mock
    if isinstance(redis_client, MagicMock):
        pytest.skip("Using mock Redis, skipping actual Redis operations")
    
    prefix = redis_client.test_prefix
    key = f"{prefix}temp:session"
    
    # Set a key with expiration (1 second)
    redis_client.setex(key, 1, "session-data")
    
    # Check TTL
    ttl = redis_client.ttl(key)
    assert ttl <= 1 and ttl > 0
    
    # Verify key exists
    assert redis_client.exists(key) == 1
    
    # Wait for expiration
    time.sleep(2)
    
    # Verify key has expired
    assert redis_client.exists(key) == 0
    
    # Set a key without expiration
    permanent_key = f"{prefix}perm:config"
    redis_client.set(permanent_key, "config-data")
    
    # Check TTL (should be -1 for no expiration)
    assert redis_client.ttl(permanent_key) == -1
    
    # Set expiration on existing key
    redis_client.expire(permanent_key, 30)
    
    # Check TTL again
    ttl = redis_client.ttl(permanent_key)
    assert ttl <= 30 and ttl > 0

# Test hash operations
def test_redis_hash_operations(redis_client):
    """
    Test hash operations on Redis.
    
    This test verifies that:
    - Hash fields can be set
    - Hash fields can be retrieved
    - Hash operations work correctly
    
    Args:
        redis_client: Redis client fixture
    """
    # Skip if we're using a mock
    if isinstance(redis_client, MagicMock):
        pytest.skip("Using mock Redis, skipping actual Redis operations")
    
    prefix = redis_client.test_prefix
    hash_key = f"{prefix}user:profile:1"
    
    # Set hash fields
    redis_client.hset(hash_key, "name", "John Doe")
    redis_client.hset(hash_key, "email", "john@example.com")
    redis_client.hset(hash_key, "age", "30")
    
    # Get a single field
    name = redis_client.hget(hash_key, "name")
    assert name.decode('utf-8') == "John Doe"
    
    # Get all fields
    profile = redis_client.hgetall(hash_key)
    assert profile[b"name"].decode('utf-8') == "John Doe"
    assert profile[b"email"].decode('utf-8') == "john@example.com"
    assert profile[b"age"].decode('utf-8') == "30"
    
    # Check if field exists
    assert redis_client.hexists(hash_key, "email") == 1
    assert redis_client.hexists(hash_key, "address") == 0
    
    # Delete a field
    redis_client.hdel(hash_key, "age")
    assert redis_client.hexists(hash_key, "age") == 0
    
    # Set multiple fields at once
    redis_client.hmset(hash_key, {
        "age": "31",
        "address": "123 Main St",
        "phone": "555-1234"
    })
    
    # Get multiple fields
    values = redis_client.hmget(hash_key, "name", "address", "phone")
    assert values[0].decode('utf-8') == "John Doe"
    assert values[1].decode('utf-8') == "123 Main St"
    assert values[2].decode('utf-8') == "555-1234"
    
    # Increment a numeric field
    redis_client.hincrby(hash_key, "age", 1)
    age = redis_client.hget(hash_key, "age")
    assert age.decode('utf-8') == "32"

# Test Redis for use with Celery
def test_redis_celery_compatibility(redis_client):
    """
    Test Redis operations specific to Celery functionality.
    
    This test verifies that:
    - Redis can store and retrieve task metadata
    - List operations needed by Celery work correctly
    
    Args:
        redis_client: Redis client fixture
    """
    # Skip if we're using a mock
    if isinstance(redis_client, MagicMock):
        pytest.skip("Using mock Redis, skipping actual Redis operations")
    
    prefix = redis_client.test_prefix
    
    # Simulate Celery task metadata
    task_id = str(uuid.uuid4())
    task_key = f"{prefix}celery-task-meta-{task_id}"
    
    task_result = {
        "status": "SUCCESS",
        "result": {
            "task_id": task_id,
            "output": "Task completed successfully",
            "data": [1, 2, 3, 4, 5]
        },
        "traceback": None,
        "children": []
    }
    
    # Store task result
    redis_client.set(task_key, json.dumps(task_result))
    
    # Retrieve task result
    result_json = redis_client.get(task_key)
    retrieved_result = json.loads(result_json)
    
    assert retrieved_result["status"] == "SUCCESS"
    assert retrieved_result["result"]["task_id"] == task_id
    assert retrieved_result["result"]["data"] == [1, 2, 3, 4, 5]
    
    # Simulate Celery queues
    queue_key = f"{prefix}celery:queue:default"
    
    # Push tasks to the queue
    task1 = {"id": str(uuid.uuid4()), "task": "process_document", "args": ["doc1"]}
    task2 = {"id": str(uuid.uuid4()), "task": "send_email", "args": ["user@example.com"]}
    
    redis_client.lpush(queue_key, json.dumps(task1))
    redis_client.lpush(queue_key, json.dumps(task2))
    
    # Check queue length
    assert redis_client.llen(queue_key) == 2
    
    # Pop a task from the queue (simulating a worker)
    task_json = redis_client.rpop(queue_key)
    popped_task = json.loads(task_json)
    
    assert popped_task["task"] == "process_document"
    assert popped_task["args"] == ["doc1"]
    
    # Verify queue length decreased
    assert redis_client.llen(queue_key) == 1 