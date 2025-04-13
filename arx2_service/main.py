#!/usr/bin/env python3
"""
ARX2 Research API - FastAPI Application
---------------------------------------
Main entry point for the ARX2 Research API service.
"""

import os
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import time
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import json

from api.routes import research_router, user_router, auth_router
from config.app_config import settings

# Set tokenizers parallelism before importing any HuggingFace libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize FastAPI app
app = FastAPI(
    title="ARX2 Research API",
    description="API for advanced AI research analysis of scientific papers",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(user_router, prefix="/users", tags=["Users"])
app.include_router(research_router, prefix="/research", tags=["Research"])

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "name": "ARX2 Research API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI application using Uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 