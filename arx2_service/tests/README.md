# ARX2 Research API Tests

This directory contains tests for the ARX2 Research API service.

## Test Structure

Tests are organized into the following categories:

- **Unit Tests**: Test individual components in isolation
  - Located in `tests/unit/`
  - Fast to run, no external dependencies

- **API Tests**: Test API endpoints
  - Located in `tests/api/`
  - Test request/response handling and integration with routes

- **Integration Tests**: Test integration with external services
  - Located in `tests/integration/`
  - Test database interactions, Redis operations, etc.

- **Async Tests**: Test asynchronous functionality
  - Located in `tests/async/`
  - Test Celery tasks and background processing

## Running Tests

### Running All Tests

```bash
cd arx2_service
pytest
```

### Running Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only API tests
pytest tests/api/

# Run tests with specific markers
pytest -m unit
pytest -m integration
pytest -m api
pytest -m async
```

### Running Specific Test Files

```bash
pytest tests/unit/test_research_tasks.py
```

### Running Specific Test Functions

```bash
pytest tests/unit/test_research_tasks.py::test_mock_research_analysis
```

## Test Configuration

The test configuration is defined in `pytest.ini` and includes:

- Test discovery patterns
- Markers for categorizing tests
- Timeout settings
- Logging configuration
- Warning filters

## Test Dependencies

Some tests require external services (MongoDB, Redis, Celery) to be running. These tests are marked with the appropriate markers (`integration`, `async`) and will be skipped if the required services are not available.

To run these tests, you can set up the required services using Docker:

```bash
docker-compose up -d redis mongodb
```

For Celery-related tests, you need to run a Celery worker:

```bash
# Set environment variable to enable Celery tests
export CELERY_TESTING=1

# Start a Celery worker (in a separate terminal)
celery -A celery_tasks.task_manager.celery_app worker --loglevel=info

# Set environment variable to indicate worker is running
export CELERY_WORKER_RUNNING=1

# Run the tests
pytest -m async
```

## Test Data

The tests use a combination of:

- **Mock Data**: For unit tests and isolated API tests
- **Test Fixtures**: For setting up and tearing down test environments
- **In-Memory Storage**: For tests that require data persistence without external services

## Coverage

To run tests with coverage reporting:

```bash
pytest --cov=arx2_service
```

Or to generate an HTML coverage report:

```bash
pytest --cov=arx2_service --cov-report=html
```

The HTML report will be generated in the `htmlcov/` directory. 