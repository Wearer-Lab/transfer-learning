"""Path handling utilities for the video processing pipeline.

This module provides utilities for handling file paths and directories
in a consistent way across the application.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Union, List, Optional

from ..config import settings

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_dated_directory(base_dir: Union[str, Path], date: Optional[datetime] = None) -> Path:
    """
    Get or create a dated subdirectory.
    
    Args:
        base_dir: Base directory path
        date: Date to use (defaults to current date)
        
    Returns:
        Path object for the dated directory
    """
    if date is None:
        date = datetime.now()
        
    dated_dir = Path(base_dir) / date.strftime("%Y-%m-%d")
    return ensure_directory(dated_dir)

def get_safe_filename(filename: str) -> str:
    """
    Convert a string into a safe filename.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename string
    """
    # Replace problematic characters
    safe_name = filename.replace(" ", "_")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-.")
    
    # Ensure the filename is not too long
    if len(safe_name) > 100:
        safe_name = safe_name[:100]
        
    return safe_name

def get_safe_path(path: Union[str, Path]) -> Path:
    """
    Convert a path to a safe path without problematic characters.
    
    Args:
        path: Original path
        
    Returns:
        Path object with safe components
    """
    path_str = str(path)
    # Replace spaces with underscores and remove other problematic characters
    safe_path = path_str.replace(' ', '_').replace(':', '_').replace('?', '_')
    return Path(safe_path)

def get_job_directory(base_dir: Union[str, Path]) -> Path:
    """
    Create a unique job directory with timestamp.
    
    Args:
        base_dir: Base directory path
        
    Returns:
        Path object for the unique job directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"job_{timestamp}"
    job_dir = Path(base_dir) / job_id
    return ensure_directory(job_dir)

def cleanup_temp_directory(temp_dir: Union[str, Path], max_age_hours: int = 24) -> None:
    """
    Remove temp files older than the specified age.
    If max_age_hours is 0, remove all files regardless of age.
    
    Args:
        temp_dir: Temporary directory path
        max_age_hours: Maximum age in hours for files to keep (0 means clean all)
    """
    temp_dir = Path(temp_dir)
    if not temp_dir.exists():
        return
        
    # If max_age_hours is 0, clean everything
    if max_age_hours == 0:
        for item in temp_dir.glob("*"):
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"Error cleaning up {item}: {str(e)}")
        return
        
    # Otherwise, clean files older than the specified age
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    for item in temp_dir.glob("*"):
        try:
            if item.is_file():
                if datetime.fromtimestamp(item.stat().st_mtime) < cutoff_time:
                    item.unlink()
            elif item.is_dir():
                if all(datetime.fromtimestamp(f.stat().st_mtime) < cutoff_time 
                       for f in item.glob("**/*") if f.is_file()):
                    shutil.rmtree(item)
        except Exception as e:
            print(f"Error cleaning up {item}: {str(e)}")

def get_unique_path(path: Union[str, Path]) -> Path:
    """
    Get a unique path by appending a number if necessary.
    
    Args:
        path: Original path
        
    Returns:
        Unique path that doesn't exist
    """
    path = Path(path)
    if not path.exists():
        return path
        
    counter = 1
    while True:
        new_path = path.parent / f"{path.stem}_{counter}{path.suffix}"
        if not new_path.exists():
            return new_path
        counter += 1

def get_relative_path(path: Union[str, Path], base: Optional[Union[str, Path]] = None) -> Path:
    """
    Get path relative to a base directory.
    
    Args:
        path: Path to make relative
        base: Base directory (defaults to workspace root)
        
    Returns:
        Relative path
    """
    path = Path(path)
    if base is None:
        base = settings.data_dir
    
    try:
        return path.relative_to(base)
    except ValueError:
        return path 

def clear_cache_directory(cache_dir: Union[str, Path]) -> None:
    """
    Clear all files from the cache directory.
    
    Args:
        cache_dir: Cache directory path
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return
        
    for item in cache_dir.glob("*"):
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            print(f"Error clearing cache item {item}: {str(e)}") 