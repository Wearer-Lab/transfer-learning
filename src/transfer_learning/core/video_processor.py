"""Core video processing functionality.

This module provides the main video processing pipeline, handling both
local videos and YouTube content.
"""

import os
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from rich.console import Console
from rich.progress import Progress

from ..config import settings
from ..utils.validation import is_valid_video_file, validate_file_size
from ..utils.path import ensure_directory
from .frame_extractor import extract_frames, frames_extraction
from .content_analyzer import describe_frames
from .audio_transcriber import transcribe_audio

console = Console()

async def process_videos_async(
    path: Union[str, Path],
    output_dir: Union[str, Path],
    batch_size: int = 5,
    max_concurrent_batches: int = 3
) -> Dict[str, Any]:
    """
    Process video files asynchronously.
    
    Args:
        path: Path to video file or directory
        output_dir: Directory for output files
        batch_size: Number of frames to process in each batch
        max_concurrent_batches: Maximum number of concurrent batches
        
    Returns:
        Dictionary containing processing results
    """
    try:
        path = Path(path)
        output_dir = Path(output_dir)
        
        # Get list of video files to process
        video_files = []
        if path.is_file():
            if is_valid_video_file(path):
                video_files = [path]
        else:
            video_files = [
                f for f in path.glob("*")
                if is_valid_video_file(f)
            ]
        
        if not video_files:
            console.print("[bold red]No valid video files found to process.[/bold red]")
            return {}
            
        console.print(f"[bold blue]Found {len(video_files)} video(s) to process[/bold blue]")
        
        # Process each video
        results = {}
        for video_file in video_files:
            try:
                # Validate file size
                if not validate_file_size(video_file):
                    console.print(f"[bold red]Video file too large: {video_file}[/bold red]")
                    continue
                
                # Create output directories
                video_output_dir = output_dir / video_file.stem
                frames_dir = ensure_directory(video_output_dir / "frames")
                transcripts_dir = ensure_directory(video_output_dir / "transcripts")

                # Transcribe audio
                console.print(f"[bold blue]Transcribing audio from {video_file.name}[/bold blue]")
                transcript = await transcribe_audio(
                    video_file,
                    transcripts_dir,
                    model=settings.whisper_model,
                    device=settings.whisper_device
                )
                
                # Extract frames
                console.print(f"[bold blue]Extracting frames from {video_file.name}[/bold blue]")
                frame_paths = []
                async for frame_num, frame_path in extract_frames(
                    video_file,
                    frames_dir,
                    frame_interval=settings.frame_extraction_interval,
                    progress=Progress()
                ):
                    frame_paths.append(frame_path)
                    if len(frame_paths) >= settings.max_frames_per_video:
                        break
                
                if not frame_paths:
                    console.print(f"[bold red]No frames extracted from {video_file.name}[/bold red]")
                    continue
                
                # Analyze frames
                console.print(f"[bold blue]Analyzing frames from {video_file.name}[/bold blue]")
                frame_descriptions = await describe_frames(
                    frame_paths,
                    frames_dir,
                    batch_size=batch_size,
                    max_concurrent_batches=max_concurrent_batches
                )
                
                results[str(video_file)] = {
                    "frames": frame_paths,
                    "descriptions": frame_descriptions,
                    "transcript": transcript
                }
                
                console.print(f"[bold green]Successfully processed {video_file.name}[/bold green]")
                
            except Exception as e:
                console.print(f"[bold red]Error processing {video_file.name}: {str(e)}[/bold red]")
                continue
        
        return results
        
    except Exception as e:
        console.print(f"[bold red]Error in video processing: {str(e)}[/bold red]")
        return {}
