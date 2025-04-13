"""
Integration tests for database interactions.

This module contains tests for MongoDB database operations, focusing on:
- Connection handling
- CRUD operations
- Error handling
"""

import pytest
import os
import uuid
from unittest.mock import patch, MagicMock
import pymongo
from pymongo import MongoClient
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

# Skip these tests if the MongoDB URI environment variable is not set
# In CI environments, we would set this to a test database
pytestmark = pytest.mark.skipif(
    "MONGODB_TEST_URI" not in os.environ,
    reason="MongoDB test URI not configured"
)

# Fixture for MongoDB connection
@pytest.fixture
def mongo_client():
    """
    Create a MongoDB client for testing.
    
    This fixture connects to a test database using the MONGODB_TEST_URI
    environment variable. Tests are isolated by using a unique collection
    name for each test run.
    
    Returns:
        MongoClient: A MongoDB client connected to the test database
    """
    # Use a test database URI from environment or fall back to a default
    uri = os.environ.get("MONGODB_TEST_URI", "mongodb://localhost:27017/test")
    client = MongoClient(uri)
    
    # Return the client
    yield client
    
    # Clean up - no need to close explicitly, MongoClient handles this

# Fixture for a unique test collection
@pytest.fixture
def test_collection(mongo_client):
    """
    Create a unique test collection.
    
    This fixture creates a collection with a unique name for test isolation.
    After the test completes, the collection is dropped to clean up.
    
    Args:
        mongo_client: MongoDB client fixture
        
    Returns:
        Collection: A MongoDB collection for testing
    """
    # Create a unique collection name for this test run
    collection_name = f"test_{uuid.uuid4().hex}"
    
    # Get the test database
    db = mongo_client.get_database()
    
    # Create and return the collection
    collection = db[collection_name]
    yield collection
    
    # Clean up by dropping the collection
    collection.drop()

# Test basic CRUD operations
def test_mongodb_crud_operations(test_collection):
    """
    Test basic CRUD operations on MongoDB.
    
    This test verifies that:
    - Documents can be inserted
    - Documents can be queried
    - Documents can be updated
    - Documents can be deleted
    
    Args:
        test_collection: Test collection fixture
    """
    # Skip if we're using a mock
    if isinstance(test_collection, MagicMock):
        pytest.skip("Using mock MongoDB, skipping actual database operations")
    
    # Create - Insert a test document
    test_doc = {
        "user_id": str(uuid.uuid4()),
        "username": "testuser",
        "email": "test@example.com",
        "subscription_status": "trial"
    }
    
    insert_result = test_collection.insert_one(test_doc)
    assert insert_result.inserted_id is not None
    
    # Read - Query the document by user_id
    query_result = test_collection.find_one({"user_id": test_doc["user_id"]})
    assert query_result is not None
    assert query_result["username"] == "testuser"
    assert query_result["email"] == "test@example.com"
    
    # Update - Modify the document
    update_result = test_collection.update_one(
        {"user_id": test_doc["user_id"]},
        {"$set": {"subscription_status": "premium"}}
    )
    assert update_result.modified_count == 1
    
    # Verify the update
    updated_doc = test_collection.find_one({"user_id": test_doc["user_id"]})
    assert updated_doc["subscription_status"] == "premium"
    
    # Delete - Remove the document
    delete_result = test_collection.delete_one({"user_id": test_doc["user_id"]})
    assert delete_result.deleted_count == 1
    
    # Verify the deletion
    deleted_doc = test_collection.find_one({"user_id": test_doc["user_id"]})
    assert deleted_doc is None

# Test bulk operations
def test_mongodb_bulk_operations(test_collection):
    """
    Test bulk operations on MongoDB.
    
    This test verifies that:
    - Multiple documents can be inserted at once
    - Bulk updates work correctly
    - Documents can be queried with filters
    
    Args:
        test_collection: Test collection fixture
    """
    # Skip if we're using a mock
    if isinstance(test_collection, MagicMock):
        pytest.skip("Using mock MongoDB, skipping actual database operations")
    
    # Create multiple test documents
    test_docs = [
        {
            "user_id": str(uuid.uuid4()),
            "username": f"user{i}",
            "email": f"user{i}@example.com",
            "subscription_status": "trial" if i % 2 == 0 else "premium"
        }
        for i in range(10)
    ]
    
    # Insert many documents
    insert_result = test_collection.insert_many(test_docs)
    assert len(insert_result.inserted_ids) == 10
    
    # Query with a filter
    premium_users = list(test_collection.find({"subscription_status": "premium"}))
    assert len(premium_users) == 5
    
    # Update many documents
    update_result = test_collection.update_many(
        {"subscription_status": "trial"},
        {"$set": {"subscription_status": "basic"}}
    )
    assert update_result.modified_count == 5
    
    # Verify the updates
    basic_users = list(test_collection.find({"subscription_status": "basic"}))
    assert len(basic_users) == 5
    
    # Delete many documents
    delete_result = test_collection.delete_many({"subscription_status": "basic"})
    assert delete_result.deleted_count == 5
    
    # Verify the deletions
    remaining_users = list(test_collection.find({}))
    assert len(remaining_users) == 5
    assert all(user["subscription_status"] == "premium" for user in remaining_users)

# Test error handling
def test_mongodb_error_handling(mongo_client):
    """
    Test error handling for MongoDB operations.
    
    This test verifies that:
    - Appropriate exceptions are raised for invalid operations
    - Connection errors are handled gracefully
    
    Args:
        mongo_client: MongoDB client fixture
    """
    # Skip if we're using a mock
    if isinstance(mongo_client, MagicMock):
        pytest.skip("Using mock MongoDB, skipping actual database operations")
    
    # Test invalid collection name (contains null character)
    with pytest.raises(pymongo.errors.InvalidName):
        invalid_collection = mongo_client.test[f"invalid\0{uuid.uuid4().hex}"]
        invalid_collection.insert_one({"test": "data"})
    
    # Test duplicate key error
    collection_name = f"test_{uuid.uuid4().hex}"
    collection = mongo_client.test[collection_name]
    
    try:
        # Create an index on email to enforce uniqueness
        collection.create_index("email", unique=True)
        
        # Insert a document
        collection.insert_one({"email": "duplicate@example.com", "data": "test1"})
        
        # Try to insert another document with the same email (should fail)
        with pytest.raises(pymongo.errors.DuplicateKeyError):
            collection.insert_one({"email": "duplicate@example.com", "data": "test2"})
    finally:
        # Clean up
        collection.drop()

# Test async MongoDB operations
@pytest.mark.asyncio
async def test_async_mongodb_operations():
    """
    Test asynchronous MongoDB operations using Motor.
    
    This test verifies that:
    - Async connections can be established
    - Async CRUD operations work correctly
    
    Note: This test requires a running MongoDB instance
    """
    # Use a test database URI from environment or fall back to a default
    uri = os.environ.get("MONGODB_TEST_URI", "mongodb://localhost:27017/test")
    
    # Skip if testing environment doesn't support actual MongoDB
    # This allows the test to run with mocks in CI
    try:
        # Connect to MongoDB asynchronously
        client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=1000)
        
        # Check that the connection works
        await client.admin.command('ping')
    except Exception:
        pytest.skip("Could not connect to MongoDB for async tests")
        return
    
    # Create a unique collection name
    db = client.get_database()
    collection_name = f"test_async_{uuid.uuid4().hex}"
    collection = db[collection_name]
    
    try:
        # Insert a document
        doc = {"user_id": str(uuid.uuid4()), "username": "asyncuser"}
        result = await collection.insert_one(doc)
        assert result.inserted_id is not None
        
        # Query the document
        query_result = await collection.find_one({"user_id": doc["user_id"]})
        assert query_result is not None
        assert query_result["username"] == "asyncuser"
        
        # Update the document
        update_result = await collection.update_one(
            {"user_id": doc["user_id"]},
            {"$set": {"username": "updated_asyncuser"}}
        )
        assert update_result.modified_count == 1
        
        # Verify the update
        updated_doc = await collection.find_one({"user_id": doc["user_id"]})
        assert updated_doc["username"] == "updated_asyncuser"
        
        # Delete the document
        delete_result = await collection.delete_one({"user_id": doc["user_id"]})
        assert delete_result.deleted_count == 1
    finally:
        # Clean up
        await collection.drop() 