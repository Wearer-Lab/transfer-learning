"""Frame extraction utilities for video processing.

This module provides functionality for extracting frames from videos
at specified intervals or key points.
"""

import os
import cv2
import asyncio
import traceback
import logging
import glob
from multiprocessing import Pool, cpu_count
from rich.console import Console
from rich.traceback import install
from pathlib import Path
from typing import List, Generator, Tuple
from rich.progress import Progress

# Initialize rich console
console = Console()

# Import utility functions for caching
from ..utils.cache import is_already_processed, mark_as_processed, get_cached_result
from ..config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

async def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 30,
    progress: Progress = None
) -> Generator[Tuple[int, str], None, None]:
    """
    Extract frames from a video file at specified intervals.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Number of frames to skip between extractions
        progress: Optional progress bar
        
    Yields:
        Tuple of (frame number, frame path)
    """
    try:
        # Ensure output directory exists
        frames_dir = Path(output_dir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create progress task if progress bar provided
        task_id = None
        if progress:
            task_id = progress.add_task(
                f"[cyan]Extracting frames from {Path(video_path).name}...",
                total=total_frames
            )
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_path = frames_dir / f"frame_{frame_count:07d}.jpg"
                await asyncio.to_thread(cv2.imwrite, str(frame_path), frame)
                saved_count += 1
                yield frame_count, str(frame_path)
            
            frame_count += 1
            
            # Update progress if available
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
            
            # Allow other tasks to run
            await asyncio.sleep(0)
        
        # Clean up
        cap.release()
        
        if progress and task_id is not None:
            progress.update(task_id, completed=True)
            
        console.print(f"[green]Extracted {saved_count} frames from {Path(video_path).name}[/green]")
        
    except Exception as e:
        console.print(f"[bold red]Error extracting frames from {video_path}: {str(e)}[/bold red]")
        raise

async def frames_extraction(
    video_path: str,
    output_dir: str,
    frame_interval: int = 30,
    max_frames: int = None
) -> List[str]:
    """
    Extract frames from a video with optional maximum frame limit.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Number of frames to skip between extractions
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of paths to extracted frames
    """
    frame_paths = []
    
    with Progress() as progress:
        async for _, frame_path in extract_frames(
            video_path,
            output_dir,
            frame_interval,
            progress
        ):
            frame_paths.append(frame_path)
            if max_frames and len(frame_paths) >= max_frames:
                break
    
    return frame_paths

def process_video_segment(args):
    frame_number, video_path, frame_dir = args
    try:
        # Ensure video_path is a string
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(frame_dir, f"frame_{frame_number:07d}.jpg")
            cv2.imwrite(frame_path, frame)
            logging.info(f"Saved frame: {frame_path}")
            cap.release()
            return [frame_path]
        else:
            cap.release()
            return []
    except Exception as e:
        console.print(f"Error in process_video_segment: {str(e)}")
        return []
