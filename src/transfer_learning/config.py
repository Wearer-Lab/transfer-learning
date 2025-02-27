"""Configuration settings for the video processing pipeline.

This module provides centralized configuration management for:
- API keys and model settings
- Directory paths and file handling
- Processing parameters
- Monitoring and logging settings
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables if .env exists
load_dotenv()

# Default directories
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = WORKSPACE_ROOT / "data"
CACHE_DIR = WORKSPACE_ROOT / ".cache"
LOGS_DIR = WORKSPACE_ROOT / "logs"

# Create default directories
for directory in [DATA_DIR, CACHE_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Directory Configuration
    data_dir: Path = Field(default=DATA_DIR)
    cache_dir: Path = Field(default=CACHE_DIR)
    logs_dir: Path = Field(default=LOGS_DIR)
    videos_dir: Path = Field(default=DATA_DIR / "videos")
    frames_dir: Path = Field(default=DATA_DIR / "frames")
    transcripts_dir: Path = Field(default=DATA_DIR / "transcripts")
    guides_dir: Path = Field(default=DATA_DIR / "guides")
    analysis_dir: Path = Field(default=DATA_DIR / "analysis")
    temp_dir: Path = Field(default=DATA_DIR / "temp")
    
    # Processing Configuration
    frame_extraction_interval: int = Field(default=30, env='FRAME_EXTRACTION_INTERVAL')
    max_frames_per_video: int = Field(default=100, env='MAX_FRAMES_PER_VIDEO')
    batch_size: int = Field(default=30, env='BATCH_SIZE')
    max_concurrent_batches: int = Field(default=100, env='MAX_CONCURRENT_BATCHES')
    
    # Video Processing
    supported_video_formats: List[str] = Field(
        default=[".mp4", ".avi", ".mov", ".mkv"],
        env='SUPPORTED_VIDEO_FORMATS'
    )
    max_video_size_mb: int = Field(default=500, env='MAX_VIDEO_SIZE_MB')
    
    # Model Configuration
    openai_api_key: str = Field(default="", env='OPENAI_API_KEY')
    openai_model: str = Field(default="gpt-4o-mini", env='OPENAI_MODEL')
    vision_model: str = Field(default="o3-mini", env='VISION_MODEL')
    fast_model: str = Field(default="gpt-4o-mini", env='FAST_MODEL')
    
    # Audio Processing
    whisper_model: str = Field(default='base', env='WHISPER_MODEL')
    whisper_device: str = Field(default='cpu', env='WHISPER_DEVICE')
    whisper_compute_type: str = Field(default='int8', env='WHISPER_COMPUTE_TYPE')
    
    # Cache Configuration
    enable_cache: bool = Field(default=True, env='ENABLE_CACHE')
    cache_ttl_hours: int = Field(default=24, env='CACHE_TTL_HOURS')
    
    # Monitoring Configuration
    enable_monitoring: bool = Field(default=True, env='ENABLE_MONITORING')
    log_level: str = Field(default="INFO", env='LOG_LEVEL')
    metrics_enabled: bool = Field(default=True, env='METRICS_ENABLED')
    
    # Optional API Keys
    anthropic_api_key: Optional[str] = Field(default=None, env='ANTHROPIC_API_KEY')
    huggingface_api_key: Optional[str] = Field(default=None, env='HUGGINGFACE_API_KEY')
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.cache_dir,
            self.logs_dir,
            self.videos_dir,
            self.frames_dir,
            self.transcripts_dir,
            self.guides_dir,
            self.analysis_dir,
            self.temp_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def current_date_dir(self) -> Path:
        """Get directory for current date's processing."""
        date_dir = self.data_dir / datetime.now().strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir

# Create global settings instance
settings = Settings()
