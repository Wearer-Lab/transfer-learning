from rich.console import Console
from rich.json import JSON
from rich.syntax import Syntax
from typing import Any, Union, Dict, Optional
import json
import os
import hashlib
import re
from datetime import datetime

console = Console()

def print_json(data: Union[dict, str, Any], title: str = None) -> None:
    """
    Print JSON data in a nicely formatted way using rich.
    
    Args:
        data: The data to print. Can be a dictionary, JSON string, or any JSON-serializable object
        title: Optional title to display above the JSON
    """
    try:
        # Convert to string if it's not already
        if not isinstance(data, str):
            if hasattr(data, 'model_dump'):  # Handle Pydantic models
                json_str = json.dumps(data.model_dump(), indent=2)
            elif hasattr(data, 'dict'):  # Handle older Pydantic models
                json_str = json.dumps(data.dict(), indent=2)
            else:
                json_str = json.dumps(data, indent=2, default=lambda x: x.__dict__)
        else:
            json_str = data

        # Create a syntax-highlighted JSON display
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
        
        if title:
            console.print(f"\n[bold blue]{title}[/bold blue]")
        
        console.print(syntax)
    except Exception as e:
        console.print(f"[bold red]Error formatting JSON: {str(e)}[/bold red]")

# Cache directory for processed files
PROCESS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".process_cache")
os.makedirs(PROCESS_CACHE_DIR, exist_ok=True)

def get_file_hash(file_path: str) -> str:
    """
    Generate a hash for a file based on its path and modification time
    """
    if not os.path.exists(file_path):
        return hashlib.md5(file_path.encode()).hexdigest()
    
    # Use file path, size and modification time to create a unique identifier
    file_stat = os.stat(file_path)
    file_info = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
    return hashlib.md5(file_info.encode()).hexdigest()

def get_url_hash(url: str) -> str:
    """
    Generate a hash for a URL
    """
    return hashlib.md5(url.encode()).hexdigest()

def is_already_processed(identifier: str, process_type: str) -> bool:
    """
    Check if a file or URL has already been processed
    
    Args:
        identifier: File path or URL
        process_type: Type of processing (e.g., 'video', 'audio', 'frames', 'transcript')
    
    Returns:
        bool: True if already processed, False otherwise
    """
    # Check if it's a YouTube URL
    youtube_id = None
    if "youtube.com" in identifier or "youtu.be" in identifier:
        youtube_id = extract_youtube_id(identifier)
        
        # Check YouTube-specific cache first
        if youtube_id:
            youtube_cache_path = get_youtube_cache_path(identifier)
            if youtube_cache_path and os.path.exists(youtube_cache_path):
                try:
                    with open(youtube_cache_path, "r") as f:
                        cache_data = json.load(f)
                    if cache_data.get("process_type") == process_type:
                        return True
                except Exception:
                    pass  # Fall back to regular cache check
    
    # Determine if it's a file or URL
    if os.path.exists(identifier):
        # It's a file
        file_hash = get_file_hash(identifier)
    elif youtube_id:
        # It's a YouTube URL, use the video ID as part of the hash
        file_hash = f"youtube_{youtube_id}_{get_url_hash(identifier)}"
    else:
        # Assume it's a URL or other string identifier
        file_hash = get_url_hash(identifier)
    
    cache_file = os.path.join(PROCESS_CACHE_DIR, f"{process_type}_{file_hash}.json")
    return os.path.exists(cache_file)

def json_serializable(obj):
    """
    Convert an object to a JSON serializable format
    
    Args:
        obj: The object to convert
        
    Returns:
        A JSON serializable version of the object
    """
    if hasattr(obj, 'model_dump'):
        # Handle Pydantic v2 models
        return obj.model_dump()
    elif hasattr(obj, 'dict'):
        # Handle Pydantic v1 models
        return obj.dict()
    elif hasattr(obj, '__dict__'):
        # Handle custom objects
        return obj.__dict__
    elif isinstance(obj, (list, tuple)):
        # Handle lists and tuples
        return [json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        # Handle dictionaries
        return {k: json_serializable(v) for k, v in obj.items()}
    else:
        # Return the object as is if it's a primitive type
        return obj

def mark_as_processed(identifier: str, process_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Mark a file or URL as processed
    
    Args:
        identifier: File path or URL
        process_type: Type of processing (e.g., 'video', 'audio', 'frames', 'transcript')
        metadata: Optional metadata about the processing
    
    Returns:
        str: Path to the cache file
    """
    # Check if it's a YouTube URL
    youtube_id = None
    if "youtube.com" in identifier or "youtu.be" in identifier:
        youtube_id = extract_youtube_id(identifier)
    
    # Determine if it's a file or URL
    if os.path.exists(identifier):
        # It's a file
        file_hash = get_file_hash(identifier)
    elif youtube_id:
        # It's a YouTube URL, use the video ID as part of the hash
        file_hash = f"youtube_{youtube_id}_{get_url_hash(identifier)}"
    else:
        # Assume it's a URL or other string identifier
        file_hash = get_url_hash(identifier)
    
    cache_data = {
        "identifier": identifier,
        "process_type": process_type,
        "timestamp": os.path.getmtime(identifier) if os.path.exists(identifier) else None,
        "processed_at": datetime.now().isoformat(),
    }
    
    # Add YouTube-specific information if applicable
    if youtube_id:
        cache_data["youtube_id"] = youtube_id
    
    if metadata:
        # Convert metadata to JSON serializable format
        cache_data["metadata"] = json_serializable(metadata)
    
    cache_file = os.path.join(PROCESS_CACHE_DIR, f"{process_type}_{file_hash}.json")
    try:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=4)
    except TypeError as e:
        console.print(f"[bold yellow]Warning: Could not serialize cache data: {str(e)}[/bold yellow]")
        # Try again with a simpler version of the metadata
        if metadata:
            cache_data["metadata"] = {"error": "Could not serialize original metadata", "simplified": str(metadata)}
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=4)
    
    # If it's a YouTube URL, also save to the YouTube-specific cache
    if youtube_id:
        youtube_cache_path = get_youtube_cache_path(identifier)
        if youtube_cache_path:
            try:
                with open(youtube_cache_path, "w") as f:
                    json.dump(cache_data, f, indent=4)
                console.print(f"[bold green]Saved YouTube cache to {youtube_cache_path}[/bold green]")
            except Exception as e:
                console.print(f"[bold yellow]Warning: Could not save YouTube cache: {str(e)}[/bold yellow]")
    
    return cache_file

def get_cached_result(identifier: str, process_type: str) -> Optional[Dict[str, Any]]:
    """
    Get cached processing result if available
    
    Args:
        identifier: File path or URL
        process_type: Type of processing (e.g., 'video', 'audio', 'frames', 'transcript')
    
    Returns:
        Optional[Dict[str, Any]]: Cached result or None if not found
    """
    # Check if it's a YouTube URL
    youtube_id = None
    if "youtube.com" in identifier or "youtu.be" in identifier:
        youtube_id = extract_youtube_id(identifier)
        
        # Check YouTube-specific cache first
        if youtube_id:
            youtube_cache_path = get_youtube_cache_path(identifier)
            if youtube_cache_path and os.path.exists(youtube_cache_path):
                try:
                    with open(youtube_cache_path, "r") as f:
                        cache_data = json.load(f)
                    if cache_data.get("process_type") == process_type:
                        return cache_data
                except Exception as e:
                    console.print(f"[bold yellow]Error reading YouTube cache file {youtube_cache_path}: {str(e)}[/bold yellow]")
    
    # Determine if it's a file or URL
    if os.path.exists(identifier):
        # It's a file
        file_hash = get_file_hash(identifier)
    elif youtube_id:
        # It's a YouTube URL, use the video ID as part of the hash
        file_hash = f"youtube_{youtube_id}_{get_url_hash(identifier)}"
    else:
        # Assume it's a URL or other string identifier
        file_hash = get_url_hash(identifier)
    
    cache_file = os.path.join(PROCESS_CACHE_DIR, f"{process_type}_{file_hash}.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[bold yellow]Error reading cache file {cache_file}: {str(e)}[/bold yellow]")
    
    return None

def extract_youtube_id(url: str) -> Optional[str]:
    """
    Extract the YouTube video ID from a URL.
    
    Args:
        url: The YouTube URL
        
    Returns:
        The video ID if found, None otherwise
    """
    # Regular expressions for different YouTube URL formats
    youtube_regex = (
        r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    )
    
    match = re.search(youtube_regex, url)
    if match:
        return match.group(1)
    return None

def get_youtube_cache_path(url: str) -> Optional[str]:
    """
    Get the cache path for a YouTube video.
    
    Args:
        url: The YouTube URL
        
    Returns:
        The cache path if the URL is a valid YouTube URL, None otherwise
    """
    video_id = extract_youtube_id(url)
    if not video_id:
        return None
    
    # Create a cache directory for YouTube videos if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".youtube_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Return the cache path for this video ID
    return os.path.join(cache_dir, f"{video_id}.json") 