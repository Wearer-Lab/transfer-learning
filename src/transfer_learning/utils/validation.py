"""Input validation utilities for the video processing pipeline.

This module provides functions for validating various types of input
such as file paths, URLs, and video formats.
"""

import os
import re
from pathlib import Path
from typing import Union, List

from ..config import settings

def is_valid_file_path(path: Union[str, Path]) -> bool:
    """
    Check if a path points to a valid file.
    
    Args:
        path: Path to check
        
    Returns:
        True if path exists and is a file, False otherwise
    """
    try:
        path = Path(path)
        return path.is_file()
    except Exception:
        return False

def is_valid_directory(path: Union[str, Path]) -> bool:
    """
    Check if a path points to a valid directory.
    
    Args:
        path: Path to check
        
    Returns:
        True if path exists and is a directory, False otherwise
    """
    try:
        path = Path(path)
        return path.is_dir()
    except Exception:
        return False

def is_valid_video_file(path: Union[str, Path]) -> bool:
    """
    Check if a file is a supported video format.
    
    Args:
        path: Path to the file
        
    Returns:
        True if file is a supported video format, False otherwise
    """
    try:
        path = Path(path)
        return (
            path.is_file() and
            path.suffix.lower() in settings.supported_video_formats
        )
    except Exception:
        return False

def is_valid_youtube_url(url: str) -> bool:
    """
    Check if a URL is a valid YouTube video URL.
    
    Args:
        url: URL to check
        
    Returns:
        True if URL is a valid YouTube video URL, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
        
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+(?:&\S*)?$",
        r"(?:https?://)?(?:www\.)?youtu\.be/[\w-]+(?:\?\S*)?$",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/[\w-]+(?:\?\S*)?$",
        r"(?:https?://)?(?:m\.)?youtube\.com/watch\?v=[\w-]+(?:&\S*)?$",
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/[\w-]+(?:\?\S*)?$"
    ]
    
    return any(re.match(pattern, url) for pattern in patterns)

def validate_file_size(path: Union[str, Path], max_size_mb: int = None) -> bool:
    """
    Check if a file is within the size limit.
    
    Args:
        path: Path to the file
        max_size_mb: Maximum allowed size in MB (defaults to settings.max_video_size_mb)
        
    Returns:
        True if file size is within limit, False otherwise
    """
    if max_size_mb is None:
        max_size_mb = settings.max_video_size_mb
        
    try:
        path = Path(path)
        size_mb = path.stat().st_size / (1024 * 1024)
        return size_mb <= max_size_mb
    except Exception:
        return False 