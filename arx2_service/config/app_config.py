"""
ARX2 Research API - Configuration
--------------------------------
Configuration settings for the ARX2 Research API.
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings
import secrets

class Settings(BaseSettings):
    """Application settings"""
    
    # API settings
    API_VERSION: str = "1.0.0"
    APP_NAME: str = "ARX2 Research API"
    DEBUG: bool = Field(False, env="DEBUG")
    
    # Security settings
    SECRET_KEY: str = Field(secrets.token_urlsafe(32), env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(60 * 24, env="ACCESS_TOKEN_EXPIRE_MINUTES")  # 1 day
    ALGORITHM: str = "HS256"
    
    # Database settings
    MONGODB_URL: str = Field("mongodb://localhost:27017", env="MONGODB_URL")
    DATABASE_NAME: str = Field("arx2_service", env="DATABASE_NAME")
    
    # Redis settings
    REDIS_HOST: str = Field("localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(6379, env="REDIS_PORT")
    REDIS_URL: str = Field(None, env="REDIS_URL")
    
    # ARX2 settings
    ARX2_MAX_PAPERS_PER_QUERY: int = Field(30, env="ARX2_MAX_PAPERS_PER_QUERY")
    ARX2_MAX_ITERATIONS: int = Field(5, env="ARX2_MAX_ITERATIONS")
    ARX2_USE_GPU: bool = Field(True, env="ARX2_USE_GPU")
    ARX2_USE_DISTRIBUTED: bool = Field(False, env="ARX2_USE_DISTRIBUTED")
    ARX2_USE_KNOWLEDGE_GRAPH: bool = Field(True, env="ARX2_USE_KNOWLEDGE_GRAPH")
    
    # Storage settings
    STORAGE_DIR: str = Field("storage", env="STORAGE_DIR")
    MODELS_DIR: str = Field("models", env="MODELS_DIR")
    PAPERS_DIR: str = Field("papers", env="PAPERS_DIR")
    OUTPUT_DIR: str = Field("research_output", env="OUTPUT_DIR")
    
    # Subscription tiers and rate limits
    SUBSCRIPTION_TIERS: Dict[str, Dict[str, Any]] = {
        "free": {
            "max_queries_per_day": 2,
            "max_papers_per_query": 20,
            "max_iterations_per_query": 3,
            "storage_days": 7,
        },
        "basic": {
            "max_queries_per_day": 5,
            "max_papers_per_query": 50,
            "max_iterations_per_query": 5,
            "storage_days": 30,
        },
        "premium": {
            "max_queries_per_day": 20,
            "max_papers_per_query": 100,
            "max_iterations_per_query": 10,
            "storage_days": 90,
        },
        "enterprise": {
            "max_queries_per_day": -1,  # Unlimited
            "max_papers_per_query": 200,
            "max_iterations_per_query": 20,
            "storage_days": 365,
        }
    }
    
    # Payment settings
    STRIPE_API_KEY: Optional[str] = Field(None, env="STRIPE_API_KEY")
    STRIPE_WEBHOOK_SECRET: Optional[str] = Field(None, env="STRIPE_WEBHOOK_SECRET")
    
    # Frontend URL for CORS
    FRONTEND_URL: str = Field("http://localhost:3000", env="FRONTEND_URL")
    ALLOWED_ORIGINS: List[str] = Field(["http://localhost:3000"], env="ALLOWED_ORIGINS")
    
    @validator("REDIS_URL", pre=True)
    def assemble_redis_url(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        """Assemble Redis URL from host and port if not provided"""
        if v:
            return v
        redis_host = values.get("REDIS_HOST", "localhost")
        redis_port = values.get("REDIS_PORT", 6379)
        return f"redis://{redis_host}:{redis_port}/0"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Create directories
os.makedirs(settings.STORAGE_DIR, exist_ok=True)
os.makedirs(os.path.join(settings.STORAGE_DIR, settings.MODELS_DIR), exist_ok=True)
os.makedirs(os.path.join(settings.STORAGE_DIR, settings.PAPERS_DIR), exist_ok=True)
os.makedirs(os.path.join(settings.STORAGE_DIR, settings.OUTPUT_DIR), exist_ok=True) 