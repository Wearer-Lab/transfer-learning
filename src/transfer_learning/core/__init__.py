"""Core functionality for video processing and guide generation.

This package provides the main processing pipeline components:
- Frame extraction from videos
- Content analysis
- Audio transcription
- Video processing coordination
"""

from .frame_extractor import extract_frames, frames_extraction
from .video_processor import process_videos_async
from .content_analyzer import describe_frames
from .audio_transcriber import transcribe_audio

__all__ = [
    'extract_frames',
    'frames_extraction',
    'process_videos_async',
    'describe_frames',
    'transcribe_audio',
]
