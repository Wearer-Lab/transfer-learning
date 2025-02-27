"""Caching utilities for the video processing pipeline.

This module provides functionality for caching and retrieving processing results
to avoid redundant operations and improve performance.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import settings

def get_cache_key(identifier: str, process_type: str) -> str:
    """
    Generate a unique cache key for a given identifier and process type.
    
    Args:
        identifier: Unique identifier for the content
        process_type: Type of processing being performed
        
    Returns:
        Unique cache key string
    """
    return hashlib.md5(f"{process_type}:{identifier}".encode()).hexdigest()

def is_already_processed(identifier: str, process_type: str) -> bool:
    """
    Check if content has already been processed.
    
    Args:
        identifier: Unique identifier for the content
        process_type: Type of processing to check
        
    Returns:
        True if already processed, False otherwise
    """
    if not settings.enable_cache:
        return False
        
    cache_key = get_cache_key(identifier, process_type)
    cache_file = settings.cache_dir / f"{cache_key}.json"
    
    if not cache_file.exists():
        return False
    
    # Check cache TTL
    cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
    if cache_age > settings.cache_ttl_hours * 3600:
        return False
    
    return True

def get_cached_result(identifier: str, process_type: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached processing result.
    
    Args:
        identifier: Unique identifier for the content
        process_type: Type of processing to retrieve
        
    Returns:
        Cached result dictionary if found, None otherwise
    """
    if not settings.enable_cache:
        return None
        
    cache_key = get_cache_key(identifier, process_type)
    cache_file = settings.cache_dir / f"{cache_key}.json"
    
    if not cache_file.exists():
        return None
    
    try:
        return json.loads(cache_file.read_text())
    except Exception:
        return None

def mark_as_processed(identifier: str, process_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Mark content as processed and cache results.
    
    Args:
        identifier: Unique identifier for the content
        process_type: Type of processing performed
        metadata: Optional metadata about the processing
    """
    if not settings.enable_cache:
        return
        
    cache_key = get_cache_key(identifier, process_type)
    cache_file = settings.cache_dir / f"{cache_key}.json"
    
    cache_data = {
        "identifier": identifier,
        "process_type": process_type,
        "processed_at": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    
    cache_file.write_text(json.dumps(cache_data, indent=2)) 