"""Metrics collection and monitoring for the video processing pipeline."""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import json
from pathlib import Path

@dataclass
class ProcessingMetrics:
    """Metrics for a single video processing run."""
    video_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    frame_count: int = 0
    processed_frames: int = 0
    transcription_duration: float = 0.0
    frame_analysis_duration: float = 0.0
    guide_generation_duration: float = 0.0
    error_count: int = 0
    warnings_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens_used: int = 0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary format."""
        return {
            "video_id": self.video_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "frame_count": self.frame_count,
            "processed_frames": self.processed_frames,
            "transcription_duration": self.transcription_duration,
            "frame_analysis_duration": self.frame_analysis_duration,
            "guide_generation_duration": self.guide_generation_duration,
            "error_count": self.error_count,
            "warnings_count": self.warnings_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_tokens_used": self.total_tokens_used,
            "total_duration": (self.end_time - self.start_time).total_seconds() if self.end_time else None
        }


class MetricsTracker:
    """Tracks and stores processing metrics."""
    
    def __init__(self, metrics_dir: str = "metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.current_metrics: Optional[ProcessingMetrics] = None
    
    def start_processing(self, video_id: str) -> None:
        """Start tracking metrics for a video."""
        self.current_metrics = ProcessingMetrics(
            video_id=video_id,
            start_time=datetime.now()
        )
    
    def end_processing(self) -> None:
        """End tracking metrics and save results."""
        if self.current_metrics:
            self.current_metrics.end_time = datetime.now()
            self._save_metrics()
    
    def update_metrics(self, **kwargs) -> None:
        """Update current metrics with new values."""
        if self.current_metrics:
            for key, value in kwargs.items():
                if hasattr(self.current_metrics, key):
                    setattr(self.current_metrics, key, value)
    
    def _save_metrics(self) -> None:
        """Save current metrics to file."""
        if self.current_metrics:
            # Sanitize the video_id to ensure it's a valid filename
            safe_id = "".join(c if c.isalnum() or c == '_' else '_' for c in self.current_metrics.video_id)
            # Limit the length to avoid excessively long filenames
            if len(safe_id) > 100:
                safe_id = safe_id[:100]
                
            metrics_file = self.metrics_dir / f"{safe_id}_{self.current_metrics.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            
            try:
                with open(metrics_file, 'w') as f:
                    json.dump(self.current_metrics.to_dict(), f, indent=2)
            except Exception as e:
                # Log the error but don't crash the application
                print(f"Error saving metrics: {str(e)}")
    
    def get_metrics(self) -> Optional[Dict]:
        """Get current metrics as dictionary."""
        return self.current_metrics.to_dict() if self.current_metrics else None


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, metrics_tracker: MetricsTracker, metric_name: str):
        self.metrics_tracker = metrics_tracker
        self.metric_name = metric_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics_tracker.update_metrics(**{self.metric_name: duration}) 