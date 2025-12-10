"""
Configuration settings for the backend application.

Environment variables can be set in a .env file in the backend directory.
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    PROJECT_NAME: str = "Cortex.AI - Autonomous Annotation Agent"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    UPLOAD_DIR: str = os.path.join(DATA_DIR, "uploads")
    OUTPUT_DIR: str = os.path.join(DATA_DIR, "output")
    MODELS_DIR: str = os.path.join(os.path.dirname(BASE_DIR), "models")
    
    # SAM 3 Configuration
    SAM3_PATH: str = os.path.join(BASE_DIR, "sam3")  # Path to SAM 3 installation
    SAM3_CHECKPOINT: Optional[str] = None  # Auto-download from HuggingFace if None
    SAM3_RESOLUTION: int = 1008  # SAM 3 default resolution
    SAM3_CONFIDENCE_THRESHOLD: float = 0.5  # Minimum confidence for detections
    
    # Device Configuration
    DEVICE: str = "cuda"  # "cuda" or "cpu"
    COMPILE_MODEL: bool = False  # Enable torch.compile for faster inference
    
    # Database
    DATABASE_URL: str = "sqlite:///database.db"
    
    # RAG Configuration
    CHROMADB_PATH: Optional[str] = None  # Defaults to DATA_DIR/chromadb
    RAG_SIMILARITY_THRESHOLD: float = 0.85  # Threshold for label correction
    
    # Video Processing
    VIDEO_SAMPLE_RATE: int = 1  # Process every Nth frame (1 = all frames)
    VIDEO_MAX_FRAMES: int = 10000  # Maximum frames to process per video
    
    # API Settings
    MAX_UPLOAD_SIZE_MB: int = 500  # Maximum upload size per file
    BATCH_SIZE: int = 4  # Batch size for image processing
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from environment


settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
os.makedirs(settings.DATA_DIR, exist_ok=True)

# Add SAM 3 to Python path if needed
import sys
if settings.SAM3_PATH not in sys.path:
    sys.path.insert(0, settings.SAM3_PATH)
