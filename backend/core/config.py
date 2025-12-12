"""
Configuration settings for the backend application.

Environment variables can be set in a .env file in the backend directory.
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    PROJECT_NAME: str = "ANNOTIX - Autonomous Annotation Agent"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Paths - Using storage/ at project root for cleaner organization
    BACKEND_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROJECT_ROOT: str = os.path.dirname(BACKEND_DIR)
    STORAGE_DIR: str = os.path.join(PROJECT_ROOT, "storage")
    UPLOAD_DIR: str = os.path.join(STORAGE_DIR, "projects")
    OUTPUT_DIR: str = os.path.join(STORAGE_DIR, "exports")
    MODELS_DIR: str = os.path.join(STORAGE_DIR, "models")
    DATA_DIR: str = STORAGE_DIR  # Legacy alias
    
    # SAM 3 Configuration
    SAM3_PATH: str = os.path.join(BACKEND_DIR, "sam3")  # Path to SAM 3 installation
    SAM3_CHECKPOINT: Optional[str] = None  # Auto-download from HuggingFace if None
    SAM3_RESOLUTION: int = 1008  # SAM 3 default resolution
    SAM3_CONFIDENCE_THRESHOLD: float = 0.5  # Minimum confidence for detections
    
    # Device Configuration
    DEVICE: str = "cuda"  # "cuda" or "cpu"
    COMPILE_MODEL: bool = False  # Enable torch.compile for faster inference
    
    # Database - Now in storage/
    @property
    def database_path(self) -> str:
        return os.path.join(self.STORAGE_DIR, "database.db")
    
    DATABASE_URL: str = ""  # Will be set after class init
    
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
