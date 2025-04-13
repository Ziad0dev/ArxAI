# ARX2 Research API

A FastAPI-based SaaS application for the ARX2 AI research analysis engine.

## Overview

The ARX2 Research API provides a web service for performing advanced AI research analysis on scientific papers. It leverages transformer-based embeddings, knowledge graphs, and deep learning to identify and rank research frontiers and trends in scientific literature.

## Features

- **Research Analysis**: Submit queries to analyze research trends and frontiers in scientific papers
- **Background Processing**: Long-running tasks are processed in the background with real-time progress tracking
- **User Management**: Create and manage user accounts with different subscription tiers
- **API Access**: All functionality is available via RESTful API endpoints
- **Scalable Architecture**: Built with FastAPI, Celery, Redis, and MongoDB for scalability

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd arx2_service
```

2. Start the services using Docker Compose:
```bash
docker-compose up -d
```

This will start the following services:
- Web API server on port 8000
- Celery worker for background tasks
- Redis for task queue and result backend
- MongoDB for data storage

### API Documentation

Once the service is running, you can access the OpenAPI documentation at:

- http://localhost:8000/docs
- http://localhost:8000/redoc

## API Endpoints

### Research Analysis

- `POST /research/analyze`: Start a new research analysis task
- `GET /research/results/{task_id}`: Get the results or status of a research task

### User Management

- `POST /users/`: Create a new user account
- `GET /users/{user_id}`: Get user details

### Authentication

- `POST /auth/login`: Log in and retrieve an access token

## Development

### Local Development Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
uvicorn main:app --reload
```

4. Start a Celery worker:
```bash
celery -A celery_tasks.task_manager.celery_app worker --loglevel=info
```

## Deployment

For production deployment, consider:

1. Using a reverse proxy like Nginx
2. Implementing proper SSL/TLS
3. Setting up monitoring and logging
4. Using a managed database service instead of the Docker MongoDB container
5. Scaling workers horizontally based on load

## Subscription Tiers

The API supports different subscription tiers with varying limits:

- **Free**: 2 queries/day, 20 papers/query, 3 iterations/query, 7 days storage
- **Basic**: 5 queries/day, 50 papers/query, 5 iterations/query, 30 days storage
- **Premium**: 20 queries/day, 100 papers/query, 10 iterations/query, 90 days storage
- **Enterprise**: Unlimited queries, 200 papers/query, 20 iterations/query, 365 days storage

## License

[License information] 