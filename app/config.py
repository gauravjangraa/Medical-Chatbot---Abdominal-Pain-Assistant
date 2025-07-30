import os
from typing import Dict, Any
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # Model Settings
    MODEL_NAME: str = "microsoft/DialoGPT-medium"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_RESPONSE_LENGTH: int = 512
    TEMPERATURE: float = 0.7
    
    # Vector Database Settings
    VECTOR_DB_PATH: str = "data/vector_db"
    EMBEDDING_DIMENSION: int = 384
    
    # Medical Knowledge Settings
    CONFIDENCE_THRESHOLD: float = 0.75
    MAX_SUGGESTIONS: int = 5
    
    # Training Settings
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 5e-5
    NUM_EPOCHS: int = 3
    
    class Config:
        env_file = ".env"

settings = Settings()