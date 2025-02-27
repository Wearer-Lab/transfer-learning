"""Formatting utilities for the video processing pipeline.

This module provides utilities for formatting and displaying data in a
consistent and readable way.
"""

import json
from typing import Any, Union, Optional
from rich.console import Console
from rich.syntax import Syntax

console = Console()

def format_json(data: Any) -> str:
    """
    Format data as a JSON string.
    
    Args:
        data: Data to format
        
    Returns:
        Formatted JSON string
    """
    if hasattr(data, 'model_dump'):
        return json.dumps(data.model_dump(), indent=2)
    elif hasattr(data, 'dict'):
        return json.dumps(data.dict(), indent=2)
    else:
        return json.dumps(data, indent=2, default=str)

def print_json(data: Union[dict, str, Any], title: Optional[str] = None) -> None:
    """
    Print JSON data with syntax highlighting.
    
    Args:
        data: Data to print
        title: Optional title to display above the data
    """
    try:
        # Convert to string if not already
        if not isinstance(data, str):
            json_str = format_json(data)
        else:
            json_str = data
            
        # Create syntax-highlighted display
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
        
        if title:
            console.print(f"\n[bold blue]{title}[/bold blue]")
            
        console.print(syntax)
    except Exception as e:
        console.print(f"[bold red]Error formatting data: {str(e)}[/bold red]")

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2h 30m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    
    return " ".join(parts)

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to human readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB" 