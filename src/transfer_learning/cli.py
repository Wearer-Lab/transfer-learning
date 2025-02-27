"""Command-line interface for video processing and guide generation.

This module provides a CLI for processing videos and generating step-by-step guides.
It supports both local videos and YouTube content.
"""

import os
import sys
import asyncio
from typing import Optional, List
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Confirm, Prompt
import json
import datetime
import requests
from moviepy import VideoFileClip
from .config import settings
from .core.video_processor import process_videos_async
from .core.audio_transcriber import transcribe_audio
from .core.content_analyzer import describe_frames
from .core.frame_extractor import frames_extraction
from .guide.generator import GuideGenerator
from .utils.downloader import download_video_yt_dlp
from .utils.validation import (
    is_valid_youtube_url,
    is_valid_file_path,
    is_valid_directory,
    is_valid_video_file,
    validate_file_size
)
from .utils.path import (
    ensure_directory,
    get_dated_directory,
    get_safe_filename,
    get_job_directory,
    get_safe_path,
    cleanup_temp_directory,
    clear_cache_directory
)

from .monitoring.metrics import MetricsTracker
from .monitoring.logger import setup_logger

TEMP_DIR = settings.temp_dir
VIDEOS_DIR = settings.videos_dir
FRAMES_DIR = settings.frames_dir
TRANSCRIPTS_DIR = settings.transcripts_dir
GUIDES_DIR = settings.guides_dir
ANALYSIS_DIR = settings.analysis_dir
DATA_DIR = settings.data_dir

# Initialize components
app = typer.Typer(
    help="Video Processing Pipeline CLI for generating step-by-step guides from videos"
)
console = Console()
metrics = MetricsTracker()
logger = setup_logger(__name__)

def show_welcome():
    """Display welcome message and available commands."""
    welcome_panel = Panel.fit(
        "[bold blue]Video Processing Pipeline[/bold blue]\n"
        "Process videos and generate comprehensive step-by-step guides",
        title="Welcome",
        border_style="blue",
    )
    console.print(welcome_panel)

def show_commands():
    """Display available commands in a table."""
    table = Table(title="Available Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Usage", style="yellow")
    
    commands = [
        (
            "process-video",
            "Process a local video file and extract content",
            "process-video <path>"
        ),
        (
            "generate-guide",
            "Generate a step-by-step guide from a processed video",
            "generate-guide <path>"
        ),
        (
            "process-youtube",
            "Process a YouTube video and extract content",
            "process-youtube <url>"
        ),
        (
            "youtube-guide",
            "Generate a guide directly from a YouTube video",
            "youtube-guide <url>"
        ),
        (
            "transcribe",
            "Extract transcript from video",
            "transcribe <path_or_url>"
        ),
        (
            "analyze",
            "Analyze video content",
            "analyze <path_or_url>"
        ),
        (
            "download",
            "Download video from supported platforms",
            "download <url>"
        ),
        (
            "cleanup",
            "Clean up temporary files and directories",
            "cleanup [--max-age-hours=24] [--force]"
        ),
        (
            "stop",
            "Stop all running processes and clean temp/cache",
            "stop [--force] [--clean-temp] [--clean-cache] [--process-pattern=<pattern>]"
        ),
        (
            "config",
            "Configure environment variables and settings",
            "config [--show] [--reset]"
        ),
    ]
    
    for cmd, desc, usage in commands:
        table.add_row(cmd, desc, usage)
    
    console.print(table)

@app.command()
def process_video(
    path: str = typer.Argument(..., help="Path to video file or directory"),
    output_dir: str = typer.Option(DATA_DIR, help="Output directory for results"),
    batch_size: int = typer.Option(settings.batch_size, help="Number of frames to process in each batch"),
    max_concurrent: int = typer.Option(settings.max_concurrent_batches, help="Maximum number of concurrent batches"),
):
    """Process a local video file and extract content."""
    try:
        # Validate input path
        if not is_valid_file_path(path) and not is_valid_directory(path):
            logger.error(f"Invalid path: {path}")
            console.print(f"[bold red]Invalid path: {path}[/bold red]")
            return 1
            
        # Validate video file if it's a single file
        if is_valid_file_path(path) and not is_valid_video_file(path):
            logger.error(f"Invalid video format: {path}")
            console.print(f"[bold red]Invalid video format: {path}[/bold red]")
            return 1
            
        # Validate file size
        if is_valid_file_path(path) and not validate_file_size(path):
            logger.error(f"File too large: {path}")
            console.print(f"[bold red]File exceeds maximum size limit: {path}[/bold red]")
            return 1
            
        # Ensure output directory exists
        output_path = ensure_directory(output_dir)
        
        # Process video with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn()
        ) as progress:
            task_id = progress.add_task("[cyan]Processing video...", total=None)
            
            # Track metrics
            metrics.start_processing("video_processing")
            
            try:
                result = asyncio.run(process_videos_async(
                    path,
                    output_path,
                    batch_size=batch_size,
                    max_concurrent_batches=max_concurrent
                ))
                
                progress.update(task_id, completed=True)
                
                if result:
                    logger.info("Video processing completed successfully")
                    console.print("[bold green]Video processing completed successfully[/bold green]")
                    metrics.update_metrics(success_count=1)
                    return 0
                    
                logger.error("Video processing failed")
                console.print("[bold red]Video processing failed[/bold red]")
                metrics.update_metrics(error_count=1)
                return 1
                
            except Exception as e:
                logger.error(f"Error processing video: {str(e)}")
                console.print(f"[bold red]Error processing video: {str(e)}[/bold red]")
                metrics.update_metrics(error_count=1)
                return 1
            finally:
                metrics.end_processing()
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        return 1

@app.command()
def generate_guide(
    path: str = typer.Argument(..., help="Path to processed video data"),
    output_dir: str = typer.Option(GUIDES_DIR, help="Output directory for guides"),
    model: str = typer.Option(settings.openai_model, help="Model to use for guide generation"),
    temperature: float = typer.Option(0.2, help="Temperature for generation"),
    user_directive: str = typer.Option(None, help="User directive to customize guide generation"),
    skip_cache: bool = typer.Option(False, help="Skip cache and force regeneration"),
):
    """Generate a step-by-step guide from processed video content."""
    try:
        # Validate input path
        if not is_valid_file_path(path):
            logger.error(f"Invalid path: {path}")
            console.print(f"[bold red]Invalid path: {path}[/bold red]")
            return 1
            
        # Ensure output directory exists
        output_path = ensure_directory(output_dir)
        
        # Initialize guide generator
        generator = GuideGenerator(output_path, metrics, model)
        
        # Generate guide with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn()
        ) as progress:
            task_id = progress.add_task("[cyan]Generating guide...", total=None)
            
            # Track metrics
            metrics.start_processing("guide_generation")
            
            try:
                # Load the video processing data
                try:
                    with open(path, 'r') as f:
                        video_data = json.load(f)
                        
                    # Import the VideoProcessing model
                    from .models.video import VideoProcessing, ProcessOverview, ProcessStep, ProcessPrinciple
                    
                    # Create a VideoProcessing object from the data
                    if not video_data or not isinstance(video_data, dict):
                        logger.error("Invalid video processing data format")
                        console.print("[bold red]Invalid video processing data format[/bold red]")
                        return 1
                        
                    # Create a basic ProcessOverview
                    process_overview = ProcessOverview(
                        title=video_data.get("title", "Video Guide"),
                        overall_summary=video_data.get("summary", "Video processing results"),
                        process_steps=[
                            ProcessStep(
                                step_number=i+1,
                                title=step.get("title", f"Step {i+1}"),
                                description=step.get("description", "")
                            ) for i, step in enumerate(video_data.get("steps", []))
                        ],
                        principles=[
                            ProcessPrinciple(
                                name=principle.get("name", ""),
                                description=principle.get("description", ""),
                                application=principle.get("application", "")
                            ) for principle in video_data.get("principles", [])
                        ],
                        key_learnings=video_data.get("key_learnings", []),
                        difficulty_level=video_data.get("difficulty_level", "Medium")
                    )
                    
                    # Create the VideoProcessing object
                    video_processing = VideoProcessing(
                        dataset=process_overview,
                        frame_descriptions=video_data.get("frame_descriptions", []),
                        chunk_summaries=video_data.get("chunk_summaries", []),
                        video_url=video_data.get("video_url", ""),
                        transcript=video_data.get("transcript", "")
                    )
                    
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in file: {path}")
                    console.print(f"[bold red]Invalid JSON in file: {path}[/bold red]")
                    return 1
                except Exception as e:
                    logger.error(f"Error loading video processing data: {str(e)}")
                    console.print(f"[bold red]Error loading video processing data: {str(e)}[/bold red]")
                    return 1
                
                # Display user directive if provided
                if user_directive:
                    console.print(f"[bold blue]Using user directive:[/bold blue] {user_directive}")
                
                # Generate the guide
                guide = asyncio.run(generator.generate_guide(
                    video_processing,
                    user_directive=user_directive,
                    skip_cache=skip_cache
                ))
                
                progress.update(task_id, completed=True)
                
                if guide:
                    logger.info("Guide generated successfully")
                    console.print("[bold green]Guide generated successfully[/bold green]")
                    metrics.update_metrics(success_count=1)
                    return 0
                    
                logger.error("Guide generation failed")
                console.print("[bold red]Guide generation failed[/bold red]")
                metrics.update_metrics(error_count=1)
                return 1
                
            except Exception as e:
                logger.error(f"Error generating guide: {str(e)}")
                console.print(f"[bold red]Error generating guide: {str(e)}[/bold red]")
                metrics.update_metrics(error_count=1)
                return 1
            finally:
                metrics.end_processing()
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        return 1

@app.command()
def process_youtube(
    url: str = typer.Argument(..., help="YouTube video URL"),
    output_dir: str = typer.Option(DATA_DIR, help="Output directory for results"),
    batch_size: int = typer.Option(settings.batch_size, help="Number of frames to process in each batch"),
    max_concurrent: int = typer.Option(settings.max_concurrent_batches, help="Maximum number of concurrent batches"),
):
    """Process a YouTube video and extract content."""
    try:
        # Validate YouTube URL
        if not is_valid_youtube_url(url):
            logger.error(f"Invalid YouTube URL: {url}")
            console.print(f"[bold red]Invalid YouTube URL: {url}[/bold red]")
            return 1
            
        # Ensure output directory exists
        output_path = ensure_directory(output_dir)
        
        # Create a videos subdirectory in the output path
        videos_dir = ensure_directory(output_path / "videos")
        
        # Download and process video with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn()
        ) as progress:
            # Track metrics
            metrics.start_processing("youtube_processing")
            
            try:
                # First download the video
                download_task = progress.add_task("[cyan]Downloading video...", total=None)
                video_path = download_video_yt_dlp(url, str(videos_dir))
                progress.update(download_task, completed=True)
                
                if not video_path:
                    logger.error("Failed to download video")
                    console.print("[bold red]Failed to download video[/bold red]")
                    metrics.update_metrics(error_count=1)
                    return 1
                
                # Convert to safe path to avoid issues with spaces and special characters
                video_path = str(get_safe_path(video_path))
                    
                # Then process it
                process_task = progress.add_task("[cyan]Processing video...", total=None)
                result = asyncio.run(process_videos_async(
                    video_path,
                    output_path,
                    batch_size=batch_size,
                    max_concurrent_batches=max_concurrent
                ))
                progress.update(process_task, completed=True)
                
                if result:
                    logger.info("YouTube video processed successfully")
                    console.print("[bold green]YouTube video processed successfully[/bold green]")
                    metrics.update_metrics(success_count=1)
                    return 0
                    
                logger.error("YouTube video processing failed")
                console.print("[bold red]YouTube video processing failed[/bold red]")
                metrics.update_metrics(error_count=1)
                return 1
                
            except Exception as e:
                logger.error(f"Error processing YouTube video: {str(e)}")
                console.print(f"[bold red]Error processing YouTube video: {str(e)}[/bold red]")
                metrics.update_metrics(error_count=1)
                return 1
            finally:
                metrics.end_processing()
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        return 1

@app.command()
def youtube_guide(
    url: str = typer.Argument(..., help="YouTube video URL"),
    output_dir: str = typer.Option(GUIDES_DIR, help="Output directory for guides"),
    model: str = typer.Option(settings.openai_model, help="Model to use for guide generation"),
    temperature: float = typer.Option(0.2, help="Temperature for generation"),
    batch_size: int = typer.Option(settings.batch_size, help="Number of frames to process in each batch"),
    max_concurrent: int = typer.Option(settings.max_concurrent_batches, help="Maximum number of concurrent batches"),
    user_directive: str = typer.Option(None, help="User directive to customize guide generation"),
    skip_cache: bool = typer.Option(False, help="Skip cache and force regeneration"),
):
    """Generate a guide directly from a YouTube video."""
    try:        
        # First process the video
        console.print("[bold blue]Step 1: Processing YouTube video...[/bold blue]")
        process_result = process_youtube(
            url=url,
            output_dir=str(DATA_DIR),
            batch_size=batch_size,
            max_concurrent=max_concurrent
        )
        
        if process_result != 0:
            logger.error(f"Failed to process YouTube video: {url}")
            return process_result
            
        console.print("[bold blue]Step 2: Generating guide from processed video...[/bold blue]")
        
        # Find the processed video data file
        # Extract video ID or title from the URL to find the correct directory
        import glob
        
        # Get the video filename from the downloaded video
        video_path = None
        for video_file in Path(VIDEOS_DIR).glob("*"):
            if is_valid_video_file(video_file):
                # Check if this is the video we just processed
                video_path = video_file
                break
                
        if not video_path:
            logger.error("Could not find processed video file")
            console.print("[bold red]Could not find processed video file[/bold red]")
            return 1
            
        # Find the corresponding processed data directory
        video_stem = video_path.stem
        processed_dir = Path(DATA_DIR) / video_stem
        
        if not processed_dir.exists() or not processed_dir.is_dir():
            logger.error(f"Could not find processed data directory: {processed_dir}")
            console.print(f"[bold red]Could not find processed data directory: {processed_dir}[/bold red]")
            return 1
            
        # Find the frame analysis file
        frame_analysis_file = processed_dir / "frames" / "frame_analysis.json"
        
        if not frame_analysis_file.exists() or not frame_analysis_file.is_file():
            logger.error(f"Could not find frame analysis file: {frame_analysis_file}")
            console.print(f"[bold red]Could not find frame analysis file: {frame_analysis_file}[/bold red]")
            return 1
            
        # Load the frame analysis data
        try:
            with open(frame_analysis_file, 'r') as f:
                frame_analysis_data = json.load(f)
                
            # Import the VideoProcessing model
            from .models.video import VideoProcessing, ProcessOverview, ProcessStep, ProcessPrinciple
            
            # Create a VideoProcessing object from the frame analysis data
            # Extract necessary data from frame_analysis_data
            if not frame_analysis_data or not isinstance(frame_analysis_data, dict):
                logger.error("Invalid frame analysis data format")
                console.print("[bold red]Invalid frame analysis data format[/bold red]")
                return 1
                
            # Create a basic ProcessOverview
            process_overview = ProcessOverview(
                title=frame_analysis_data.get("title", f"Guide for {video_stem}"),
                overall_summary=frame_analysis_data.get("summary", "Video processing results"),
                process_steps=[
                    ProcessStep(
                        step_number=i+1,
                        title=step.get("title", f"Step {i+1}"),
                        description=step.get("description", "")
                    ) for i, step in enumerate(frame_analysis_data.get("steps", []))
                ],
                principles=[
                    ProcessPrinciple(
                        name=principle.get("name", ""),
                        description=principle.get("description", ""),
                        application=principle.get("application", "")
                    ) for principle in frame_analysis_data.get("principles", [])
                ],
                key_learnings=frame_analysis_data.get("key_learnings", []),
                difficulty_level=frame_analysis_data.get("difficulty_level", "Medium")
            )
            
            # Create the VideoProcessing object
            video_processing = VideoProcessing(
                dataset=process_overview,
                frame_descriptions=frame_analysis_data.get("frame_descriptions", []),
                chunk_summaries=frame_analysis_data.get("chunk_summaries", []),
                video_url=url,
                transcript=frame_analysis_data.get("transcript", "")
            )
            
            # Display user directive if provided
            if user_directive:
                console.print(f"[bold blue]Using user directive:[/bold blue] {user_directive}")
            
            # Initialize the guide generator
            generator = GuideGenerator(
                output_dir=output_dir,
                metrics_tracker=metrics,
                model=model,
                temperature=temperature
            )
            
            # Generate the guide
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn()
            ) as progress:
                task_id = progress.add_task("[cyan]Generating guide...", total=None)
                
                try:
                    # Generate the guide
                    guide_result = asyncio.run(generator.generate_guide(
                        video_processing,
                        user_directive=user_directive,
                        skip_cache=skip_cache
                    ))
                    
                    progress.update(task_id, completed=True)
                    
                    if guide_result:
                        logger.info("Guide generated successfully")
                        console.print("[bold green]Guide generated successfully[/bold green]")
                        return 0
                    else:
                        logger.error("Guide generation failed")
                        console.print("[bold red]Guide generation failed[/bold red]")
                        return 1
                except Exception as e:
                    progress.update(task_id, completed=True)
                    logger.error(f"Error during guide generation: {str(e)}")
                    console.print(f"[bold red]Error during guide generation: {str(e)}[/bold red]")
                    return 1
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in frame analysis file: {frame_analysis_file}")
            console.print(f"[bold red]Invalid JSON in frame analysis file: {frame_analysis_file}[/bold red]")
            return 1
        
    except Exception as e:
        logger.error(f"Error generating YouTube guide: {str(e)}")
        console.print(f"[bold red]Error generating YouTube guide: {str(e)}[/bold red]")
        return 1

@app.command()
def transcribe(
    path: str = typer.Argument(..., help="Path to video file or YouTube URL"),
    output_dir: str = typer.Option(TRANSCRIPTS_DIR, help="Output directory for transcripts"),
    model: str = typer.Option("base", help="Whisper model to use"),
    device: str = typer.Option("cpu", help="Device to use for transcription"),
):
    """Extract transcript from video."""
    try:
        # Determine if input is a file or URL
        if is_valid_youtube_url(path):
            # Download YouTube video first
            with Progress() as progress:
                task = progress.add_task("[cyan]Downloading video...", total=None)
                video_path = download_video_yt_dlp(path, str(VIDEOS_DIR))
                progress.update(task, completed=True)
                
                if not video_path:
                    logger.error("Failed to download video")
                    console.print("[bold red]Failed to download video[/bold red]")
                    return 1
        elif is_valid_video_file(path):
            video_path = path
        else:
            logger.error(f"Invalid input: {path}")
            console.print(f"[bold red]Invalid input: {path}[/bold red]")
            return 1
            
        # Ensure output directory exists
        output_path = ensure_directory(output_dir)
        
        # Transcribe with progress tracking
        with Progress() as progress:
            task = progress.add_task("[cyan]Transcribing audio...", total=None)
            
            # Track metrics
            metrics.start_processing("transcription")
            
            try:
                # Convert to safe path to avoid issues with spaces and special characters
                video_path = str(get_safe_path(video_path))
                
                result = asyncio.run(transcribe_audio(
                    video_path,
                    output_path,
                    model=model,
                    device=device
                ))
                
                progress.update(task, completed=True)
                
                if result:
                    logger.info("Transcription completed successfully")
                    console.print("[bold green]Transcription completed successfully[/bold green]")
                    metrics.update_metrics(success_count=1)
                        
                    return 0
                    
                logger.error("Transcription failed")
                console.print("[bold red]Transcription failed[/bold red]")
                metrics.update_metrics(error_count=1)
                return 1
                
            except Exception as e:
                logger.error(f"Error transcribing audio: {str(e)}")
                console.print(f"[bold red]Error transcribing audio: {str(e)}[/bold red]")
                metrics.update_metrics(error_count=1)
                return 1
            finally:
                metrics.end_processing()
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        return 1

@app.command()
def analyze(
    path: str = typer.Argument(..., help="Path to video file or YouTube URL"),
    output_dir: str = typer.Option(ANALYSIS_DIR, help="Output directory for analysis results"),
    batch_size: int = typer.Option(settings.batch_size, help="Number of frames to process in each batch"),
    max_concurrent: int = typer.Option(settings.max_concurrent_batches, help="Maximum number of concurrent batches"),
    model: str = typer.Option("base", help="Whisper model to use for transcription"),
    device: str = typer.Option("cpu", help="Device to use for transcription"),
):
    """Analyze video content."""
    try:        
        # Determine if input is a file or URL
        if is_valid_youtube_url(path):
            # Download YouTube video first
            with Progress() as progress:
                task = progress.add_task("[cyan]Downloading video...", total=None)
                video_path = download_video_yt_dlp(path, str(VIDEOS_DIR))
                progress.update(task, completed=True)
                
                if not video_path:
                    logger.error("Failed to download video")
                    console.print("[bold red]Failed to download video[/bold red]")
                    return 1
        elif is_valid_video_file(path):
            video_path = path
        else:
            logger.error(f"Invalid input: {path}")
            console.print(f"[bold red]Invalid input: {path}[/bold red]")
            return 1
            
        # Ensure output directory exists
        output_path = ensure_directory(output_dir)
        
        # Create subdirectories for different analysis components
        frames_output = ensure_directory(FRAMES_DIR)
        transcript_output = ensure_directory(TRANSCRIPTS_DIR)
        analysis_output = ensure_directory(ANALYSIS_DIR)
        
        # Track metrics
        metrics.start_processing("content_analysis")
        
        try:
            # Convert to safe path to avoid issues with spaces and special characters
            video_path = str(get_safe_path(video_path))
            
            # Step 1: Extract frames from the video
            with Progress() as progress:
                task = progress.add_task("[cyan]Extracting frames from video...", total=None)
                
                # Extract frames using the frames_extraction function
                kwargs = {
                    "video_path": video_path,
                    "output_dir": str(frames_output),
                    "frame_interval": settings.frame_extraction_interval,
                    "max_frames": settings.max_frames_per_video
                }
                frame_paths = asyncio.run(frames_extraction(**kwargs))
                
                progress.update(task, completed=True)
                
                if not frame_paths:
                    logger.error("Failed to extract frames from video")
                    console.print("[bold red]Failed to extract frames from video[/bold red]")
                    metrics.update_metrics(error_count=1)
                    return 1
                
                console.print(f"[bold green]Successfully extracted {len(frame_paths)} frames[/bold green]")
            
            # Step 2: Transcribe audio from the video
            with Progress() as progress:
                task = progress.add_task("[cyan]Transcribing audio...", total=None)
                
                transcript_result = asyncio.run(transcribe_audio(
                    video_path,
                    transcript_output,
                    model=model,
                    device=device
                ))
                
                progress.update(task, completed=True)
                
                if not transcript_result:
                    logger.warning("Audio transcription failed or no audio found")
                    console.print("[bold yellow]Audio transcription failed or no audio found[/bold yellow]")
                    transcript_result = {}
                else:
                    console.print("[bold green]Successfully transcribed audio[/bold green]")
            
            # Get video duration
            video_duration = None
            try:
                video = VideoFileClip(str(video_path))
                video_duration = video.duration
                video.close()
            except Exception as e:
                logger.warning(f"Could not determine video duration: {str(e)}")
            
            # Step 3: Analyze the extracted frames
            with Progress() as progress:
                task = progress.add_task("[cyan]Analyzing video content...", total=None)
                
                analysis_result = asyncio.run(describe_frames(
                    frame_paths,
                    analysis_output,
                    batch_size=batch_size,
                    max_concurrent_batches=max_concurrent,
                    transcript_data=transcript_result,
                    video_duration=video_duration
                ))
                
                progress.update(task, completed=True)
                
                if not analysis_result:
                    logger.error("Frame analysis failed")
                    console.print("[bold red]Frame analysis failed[/bold red]")
                    metrics.update_metrics(error_count=1)
                    return 1
            
            # Step 4: Combine results into a single analysis file
            combined_output = output_path / "combined_analysis.json"
            
            # Load transcript if available
            transcript_file = next(transcript_output.glob("*.json"), None)
            transcript_data = {}
            if transcript_file:
                try:
                    with open(transcript_file, 'r') as f:
                        transcript_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load transcript data: {str(e)}")
            
            # Combine frame analysis with transcript
            combined_data = {
                "metadata": {
                    "source": video_path,
                    "processed_at": datetime.datetime.now().isoformat(),
                    "frames_count": len(frame_paths),
                    "has_transcript": bool(transcript_data)
                },
                "frames_analysis": analysis_result,
                "transcript": transcript_data
            }
            
            # Save combined results
            with open(combined_output, 'w') as f:
                json.dump(combined_data, f, indent=2)
            
            logger.info("Content analysis completed successfully")
            console.print("[bold green]Content analysis completed successfully[/bold green]")
            console.print(f"[bold green]Combined analysis saved to {combined_output}[/bold green]")
            metrics.update_metrics(success_count=1)

            return 0
                
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            console.print(f"[bold red]Error analyzing content: {str(e)}[/bold red]")
            metrics.update_metrics(error_count=1)
            return 1
        finally:
            metrics.end_processing()
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        return 1

@app.command()
def download(
    url: str = typer.Argument(..., help="URL to download video from"),
    output_dir: str = typer.Option(VIDEOS_DIR, help="Output directory for downloaded videos"),
):
    """Download video from supported platforms."""
    try:
        # Validate URL
        if not is_valid_youtube_url(url):
            logger.error(f"Invalid URL: {url}")
            console.print(f"[bold red]Invalid URL: {url}[/bold red]")
            return 1
            
        # Ensure output directory exists
        output_path = ensure_directory(output_dir)
        
        # Download with progress tracking
        with Progress() as progress:
            task = progress.add_task("[cyan]Downloading video...", total=None)
            
            # Track metrics
            metrics.start_processing("video_download")
            
            try:
                # Create a safe output directory path
                safe_output_path = get_safe_path(output_path)
                
                result = download_video_yt_dlp(url, str(safe_output_path))
                
                progress.update(task, completed=True)
                
                if result:
                    # Convert result to safe path
                    safe_result = get_safe_path(result)
                    
                    logger.info(f"Video downloaded successfully to {safe_result}")
                    console.print(f"[bold green]Video downloaded successfully to {safe_result}[/bold green]")
                    metrics.update_metrics(success_count=1)
                    return 0
                    
                logger.error("Video download failed")
                console.print("[bold red]Video download failed[/bold red]")
                metrics.update_metrics(error_count=1)
                return 1
                
            except Exception as e:
                logger.error(f"Error downloading video: {str(e)}")
                console.print(f"[bold red]Error downloading video: {str(e)}[/bold red]")
                metrics.update_metrics(error_count=1)
                return 1
            finally:
                metrics.end_processing()
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        return 1

@app.command()
def cleanup(
    max_age_hours: int = typer.Option(24, help="Maximum age in hours for files to keep"),
    force: bool = typer.Option(False, help="Force cleanup without confirmation"),
):
    """Clean up temporary files and directories."""
    try:
        if not force:
            confirm = Confirm.ask(
                f"[bold yellow]This will delete all temporary files older than {max_age_hours} hours. Continue?[/bold yellow]"
            )
            if not confirm:
                console.print("[bold blue]Cleanup cancelled.[/bold blue]")
                return 0
        
        # Track metrics
        metrics.start_processing("temp_cleanup")
        
        try:
            # Clean up temp directory
            cleanup_temp_directory(TEMP_DIR, max_age_hours=max_age_hours)
            
            # Log success
            logger.info(f"Cleaned up temporary files older than {max_age_hours} hours")
            console.print(f"[bold green]Cleaned up temporary files older than {max_age_hours} hours[/bold green]")
            metrics.update_metrics(success_count=1)
            return 0
            
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")
            console.print(f"[bold red]Error cleaning up temporary files: {str(e)}[/bold red]")
            metrics.update_metrics(error_count=1)
            return 1
        finally:
            metrics.end_processing()
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        return 1

@app.command()
def stop(
    force: bool = typer.Option(False, help="Force stop without confirmation"),
    clean_temp: bool = typer.Option(True, help="Clean temporary files"),
    clean_cache: bool = typer.Option(True, help="Clean cache files"),
    process_pattern: str = typer.Option("transfer-learning", help="Pattern to match process names"),
):
    """Stop all running processes and optionally clean temp and cache directories."""
    try:
        if not force:
            confirm = Confirm.ask(
                "[bold yellow]This will stop all running processes and clean temporary/cache files. Continue?[/bold yellow]"
            )
            if not confirm:
                console.print("[bold blue]Stop operation cancelled.[/bold blue]")
                return 0
        
        # Track metrics
        metrics.start_processing("stop_operation")
        
        try:
            # Import process utilities
            from .utils.process import terminate_child_processes, terminate_related_processes
            
            # First terminate child processes
            console.print("[bold blue]Terminating child processes...[/bold blue]")
            terminate_child_processes()
            
            # Then terminate related processes
            console.print(f"[bold blue]Terminating processes matching '{process_pattern}'...[/bold blue]")
            terminated_count = terminate_related_processes(process_pattern)
            
            if terminated_count > 0:
                console.print(f"[bold green]Terminated {terminated_count} processes[/bold green]")
            else:
                console.print("[bold yellow]No matching processes found to terminate[/bold yellow]")
            
            # Clean temp directory if requested
            if clean_temp:
                console.print("[bold blue]Cleaning temporary files...[/bold blue]")
                cleanup_temp_directory(TEMP_DIR, max_age_hours=0)  # 0 means clean all files
                console.print("[bold green]Temporary files cleaned[/bold green]")
            
            # Clean cache directory if requested
            if clean_cache:
                console.print("[bold blue]Cleaning cache files...[/bold blue]")
                clear_cache_directory(settings.cache_dir)
                console.print("[bold green]Cache files cleaned[/bold green]")
            
            # Log success
            logger.info("Stop operation completed successfully")
            console.print("[bold green]Stop operation completed successfully[/bold green]")
            metrics.update_metrics(success_count=1)
            return 0
            
        except Exception as e:
            logger.error(f"Error during stop operation: {str(e)}")
            console.print(f"[bold red]Error during stop operation: {str(e)}[/bold red]")
            metrics.update_metrics(error_count=1)
            return 1
        finally:
            metrics.end_processing()
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        return 1

def reload_settings():
    """Reload settings from the config module."""
    # Force reload the settings module
    import importlib
    from . import config
    importlib.reload(config)
    # Return the reloaded settings
    return config.settings

def check_environment_variables():
    """Check for required environment variables and prompt user to set missing ones."""
    # Get current settings
    from .config import settings
    
    required_vars = {
        'OPENAI_API_KEY': {
            'description': 'OpenAI API key for model access',
            'is_secret': True,
            'required': True
        },
        'OPENAI_MODEL': {
            'description': 'OpenAI model to use (default: gpt-4o-mini)',
            'is_secret': False,
            'required': False,
            'default': 'gpt-4o-mini'
        },
        'VISION_MODEL': {
            'description': 'Vision model to use (default: o3-mini)',
            'is_secret': False,
            'required': False,
            'default': 'o3-mini'
        },
        'WHISPER_MODEL': {
            'description': 'Whisper model for transcription (default: base)',
            'is_secret': False,
            'required': False,
            'default': 'base'
        },
        'WHISPER_DEVICE': {
            'description': 'Device for Whisper model (cpu/cuda, default: cpu)',
            'is_secret': False,
            'required': False,
            'default': 'cpu'
        },
        'ANTHROPIC_API_KEY': {
            'description': 'Anthropic API key (optional)',
            'is_secret': True,
            'required': False
        },
        'HUGGINGFACE_API_KEY': {
            'description': 'HuggingFace API key (optional)',
            'is_secret': True,
            'required': False
        }
    }
    
    missing_required = []
    missing_optional = []
    invalid_keys = []
    
    # Check which variables are missing or invalid
    for var_name, var_info in required_vars.items():
        env_value = os.environ.get(var_name) or getattr(settings, var_name.lower(), None)
        
        if not env_value and var_info['required']:
            missing_required.append((var_name, var_info))
        elif not env_value and not var_info['required']:
            missing_optional.append((var_name, var_info))
        elif var_name == 'OPENAI_API_KEY' and not validate_openai_api_key(env_value):
            invalid_keys.append(var_name)
    
    # If no required variables are missing or invalid, just return
    if not missing_required and not missing_optional and not invalid_keys:
        return True
    
    # Show a message about missing or invalid environment variables
    console.print("\n[bold yellow]Environment Variable Check[/bold yellow]")
    
    # Handle invalid keys
    if invalid_keys:
        console.print("\n[bold red]Invalid API keys detected:[/bold red]")
        for var_name in invalid_keys:
            console.print(f"  • [bold]{var_name}[/bold]: The API key is invalid or has expired")
        
        console.print("\nYou need to update these API keys for the application to function properly.")
        update_keys = Confirm.ask("Would you like to update these keys now?")
        
        if update_keys:
            # Create or update .env file
            env_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / '.env'
            env_content = {}
            
            # Read existing .env file if it exists
            if env_path.exists():
                with open(env_path, 'r') as f:
                    for line in f:
                        if '=' in line and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            env_content[key] = value
            
            # Prompt for new API keys
            for var_name in invalid_keys:
                prompt_text = f"Enter new {var_name}"
                value = Prompt.ask(prompt_text, password=True)
                
                if value:
                    env_content[var_name] = value
                    # Also set in current environment
                    os.environ[var_name] = value
            
            # Write to .env file
            with open(env_path, 'w') as f:
                for key, value in env_content.items():
                    f.write(f"{key}={value}\n")
            
            console.print(f"[bold green]API keys updated and saved to {env_path}[/bold green]")
            
            # Reload settings
            reload_settings()
        else:
            console.print("[bold red]Invalid API keys not updated. Some features may not work correctly.[/bold red]")
    
    # Handle missing required variables
    if missing_required:
        console.print("\n[bold red]Missing required environment variables:[/bold red]")
        for var_name, var_info in missing_required:
            console.print(f"  • [bold]{var_name}[/bold]: {var_info['description']}")
        
        console.print("\nThese variables are required for the application to function properly.")
        setup_env = Confirm.ask("Would you like to set these variables now?")
        
        if not setup_env:
            console.print("[bold red]Required environment variables are missing. Some features may not work correctly.[/bold red]")
            return False
        
        # Create or update .env file
        env_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / '.env'
        env_content = {}
        
        # Read existing .env file if it exists
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env_content[key] = value
        
        # Prompt for missing required variables
        for var_name, var_info in missing_required:
            prompt_text = f"Enter {var_name} ({var_info['description']})"
            if var_info['is_secret']:
                value = Prompt.ask(prompt_text, password=True)
            else:
                value = Prompt.ask(prompt_text)
            
            if value:
                env_content[var_name] = value
                # Also set in current environment
                os.environ[var_name] = value
                
                # Validate OpenAI API key if that's what was just set
                if var_name == 'OPENAI_API_KEY' and not validate_openai_api_key(value):
                    console.print("[bold red]Warning: The OpenAI API key provided appears to be invalid.[/bold red]")
                    retry = Confirm.ask("Would you like to try a different API key?")
                    if retry:
                        new_value = Prompt.ask("Enter a valid OpenAI API key", password=True)
                        if new_value:
                            env_content[var_name] = new_value
                            os.environ[var_name] = new_value
        
        # Ask if user wants to set optional variables
        if missing_optional:
            setup_optional = Confirm.ask("Would you like to set optional environment variables as well?")
            
            if setup_optional:
                for var_name, var_info in missing_optional:
                    default_value = var_info.get('default', '')
                    prompt_text = f"Enter {var_name} ({var_info['description']})"
                    if default_value:
                        prompt_text += f" [default: {default_value}]"
                    
                    if var_info['is_secret']:
                        value = Prompt.ask(prompt_text, password=True, default=default_value)
                    else:
                        value = Prompt.ask(prompt_text, default=default_value)
                    
                    if value:
                        env_content[var_name] = value
                        # Also set in current environment
                        os.environ[var_name] = value
        
        # Write to .env file
        with open(env_path, 'w') as f:
            for key, value in env_content.items():
                f.write(f"{key}={value}\n")
        
        console.print(f"[bold green]Environment variables saved to {env_path}[/bold green]")
        
        # Reload settings
        reload_settings()
        
        return True
    
    # Handle only missing optional variables
    elif missing_optional:
        console.print("\n[bold yellow]Missing optional environment variables:[/bold yellow]")
        for var_name, var_info in missing_optional:
            default_value = var_info.get('default', 'None')
            console.print(f"  • [bold]{var_name}[/bold]: {var_info['description']} (default: {default_value})")
        
        setup_optional = Confirm.ask("Would you like to set these optional variables now?")
        
        if setup_optional:
            # Create or update .env file
            env_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / '.env'
            env_content = {}
            
            # Read existing .env file if it exists
            if env_path.exists():
                with open(env_path, 'r') as f:
                    for line in f:
                        if '=' in line and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            env_content[key] = value
            
            # Prompt for missing optional variables
            for var_name, var_info in missing_optional:
                default_value = var_info.get('default', '')
                prompt_text = f"Enter {var_name} ({var_info['description']})"
                if default_value:
                    prompt_text += f" [default: {default_value}]"
                
                if var_info['is_secret']:
                    value = Prompt.ask(prompt_text, password=True, default=default_value)
                else:
                    value = Prompt.ask(prompt_text, default=default_value)
                
                if value:
                    env_content[var_name] = value
                    # Also set in current environment
                    os.environ[var_name] = value
            
            # Write to .env file
            with open(env_path, 'w') as f:
                for key, value in env_content.items():
                    f.write(f"{key}={value}\n")
            
            console.print(f"[bold green]Environment variables saved to {env_path}[/bold green]")
            
            # Reload settings
            reload_settings()
    
    return True

@app.command()
def config(
    show: bool = typer.Option(False, help="Show current configuration"),
    reset: bool = typer.Option(False, help="Reset configuration to defaults"),
):
    """Configure environment variables and settings."""
    # Get current settings
    from .config import settings
    
    try:
        if show:
            # Display current configuration
            console.print("[bold blue]Current Configuration:[/bold blue]")
            
            # Create a table for configuration display
            config_table = Table(title="Environment Variables")
            config_table.add_column("Variable", style="cyan")
            config_table.add_column("Value", style="green")
            config_table.add_column("Status", style="yellow")
            
            # Define variables to display
            display_vars = {
                'OPENAI_API_KEY': {
                    'description': 'OpenAI API key',
                    'is_secret': True,
                    'required': True
                },
                'OPENAI_MODEL': {
                    'description': 'OpenAI model',
                    'is_secret': False,
                    'required': False
                },
                'VISION_MODEL': {
                    'description': 'Vision model',
                    'is_secret': False,
                    'required': False
                },
                'WHISPER_MODEL': {
                    'description': 'Whisper model',
                    'is_secret': False,
                    'required': False
                },
                'WHISPER_DEVICE': {
                    'description': 'Whisper device',
                    'is_secret': False,
                    'required': False
                },
                'ANTHROPIC_API_KEY': {
                    'description': 'Anthropic API key',
                    'is_secret': True,
                    'required': False
                },
                'HUGGINGFACE_API_KEY': {
                    'description': 'HuggingFace API key',
                    'is_secret': True,
                    'required': False
                }
            }
            
            # Add rows for each variable
            for var_name, var_info in display_vars.items():
                env_value = os.environ.get(var_name) or getattr(settings, var_name.lower(), None)
                
                # Mask secret values
                display_value = "********" if env_value and var_info['is_secret'] else str(env_value or "Not set")
                
                # Determine status
                if env_value:
                    status = "[green]Set[/green]"
                elif var_info['required']:
                    status = "[red]Missing (Required)[/red]"
                else:
                    status = "[yellow]Not set (Optional)[/yellow]"
                
                config_table.add_row(var_name, display_value, status)
            
            console.print(config_table)
            
            # Display directory configuration
            dir_table = Table(title="Directory Configuration")
            dir_table.add_column("Directory", style="cyan")
            dir_table.add_column("Path", style="green")
            dir_table.add_column("Status", style="yellow")
            
            # Add rows for each directory
            directories = {
                "Data Directory": settings.data_dir,
                "Cache Directory": settings.cache_dir,
                "Logs Directory": settings.logs_dir,
                "Videos Directory": settings.videos_dir,
                "Frames Directory": settings.frames_dir,
                "Transcripts Directory": settings.transcripts_dir,
                "Guides Directory": settings.guides_dir,
                "Analysis Directory": settings.analysis_dir,
                "Temp Directory": settings.temp_dir
            }
            
            for dir_name, dir_path in directories.items():
                status = "[green]Exists[/green]" if dir_path.exists() else "[red]Does not exist[/red]"
                dir_table.add_row(dir_name, str(dir_path), status)
            
            console.print(dir_table)
            
            return 0
            
        elif reset:
            # Confirm reset
            confirm = Confirm.ask(
                "[bold yellow]This will reset all configuration to defaults. Continue?[/bold yellow]"
            )
            if not confirm:
                console.print("[bold blue]Reset cancelled.[/bold blue]")
                return 0
            
            # Create a new .env file with defaults
            env_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / '.env'
            
            # Ask for required variables
            env_content = {}
            
            # Prompt for OpenAI API key (required)
            openai_key = Prompt.ask("Enter OPENAI_API_KEY (required)", password=True)
            if openai_key:
                env_content['OPENAI_API_KEY'] = openai_key
            
            # Write defaults for optional variables
            env_content['OPENAI_MODEL'] = 'gpt-4o-mini'
            env_content['VISION_MODEL'] = 'o3-mini'
            env_content['WHISPER_MODEL'] = 'base'
            env_content['WHISPER_DEVICE'] = 'cpu'
            env_content['WHISPER_COMPUTE_TYPE'] = 'int8'
            env_content['ENABLE_CACHE'] = 'true'
            env_content['CACHE_TTL_HOURS'] = '24'
            env_content['ENABLE_MONITORING'] = 'true'
            env_content['LOG_LEVEL'] = 'INFO'
            env_content['METRICS_ENABLED'] = 'true'
            
            # Write to .env file
            with open(env_path, 'w') as f:
                for key, value in env_content.items():
                    f.write(f"{key}={value}\n")
            
            console.print(f"[bold green]Configuration reset to defaults and saved to {env_path}[/bold green]")
            
            # Reload settings
            reload_settings()
            
            return 0
        
        # If no flags provided, run the environment variable check
        check_environment_variables()
        console.print("[bold green]Configuration complete.[/bold green]")
        return 0
        
    except Exception as e:
        logger.error(f"Error configuring environment variables: {str(e)}")
        console.print(f"[bold red]Error configuring environment variables: {str(e)}[/bold red]")
        return 1

def validate_openai_api_key(api_key):
    """Validate OpenAI API key by making a test request."""
    if not api_key:
        return False
        
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Make a simple request to the models endpoint
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            return True
        else:
            error_message = response.json().get("error", {}).get("message", "Unknown error")
            logger.error(f"OpenAI API key validation failed: {error_message}")
            return False
            
    except Exception as e:
        logger.error(f"Error validating OpenAI API key: {str(e)}")
        return False

def main():
    """Main entry point for the CLI."""
    # Install rich traceback handler
    install(show_locals=True)
    
    # Show welcome message
    show_welcome()
    
    # Check environment variables
    check_environment_variables()
    
    # Show available commands
    show_commands()
    
    # Run the CLI
    app() 