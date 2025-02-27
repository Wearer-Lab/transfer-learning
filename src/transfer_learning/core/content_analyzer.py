"""Visual content analysis for video frames.

This module provides functionality for analyzing visual content from video frames,
extracting meaningful information and descriptions that can be used to understand
and recreate what is being demonstrated.
"""

import base64
import json
import asyncio
import datetime
import hashlib
from typing import Dict, List, Any, Optional
from pathlib import Path
from rich.console import Console
from openai import OpenAI
from contextlib import nullcontext

from ..config import settings
from ..models.video import ImageProcessing
from ..monitoring.metrics import MetricsTracker, Timer
from ..utils.cache import (
    is_already_processed,
    mark_as_processed,
    get_cached_result
)

console = Console()

def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_image_hash(image_path: str) -> str:
    """
    Generate a hash for image content for caching.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Hash string representing the image content
    """
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

async def analyze_frame(
    image_path: str,
    metrics_tracker: Optional[MetricsTracker] = None,
    transcript_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze a single video frame to extract meaningful content.
    
    This function:
    1. Processes the visual content of a frame
    2. Identifies key elements and actions shown
    3. Extracts relevant details for understanding the content
    4. Provides context for how this frame fits into the overall process
    
    Args:
        image_path: Path to the frame image file
        metrics_tracker: Optional metrics tracking instance
        transcript_context: Optional transcript text related to this frame
        
    Returns:
        Dictionary containing the analysis results
    """
    try:
        # Track metrics if provided
        if metrics_tracker:
            metrics_tracker.start_processing(f"frame_{Path(image_path).name}")
        
        # Ensure image path is valid and exists
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            console.print(f"[bold red]Image file not found: {image_path}[/bold red]")
            return {
                "frame": image_path_obj.name,
                "analysis": "Error: Image file not found",
                "metadata": {
                    "processed_at": datetime.datetime.now().isoformat(),
                    "error": True,
                    "error_message": "Image file not found"
                }
            }
        
        # Check cache
        if is_already_processed(image_path, "frame_analysis"):
            console.print(f"[bold green]Found cached analysis for {Path(image_path).name}[/bold green]")
            cached = get_cached_result(image_path, "frame_analysis")
            if cached and "metadata" in cached:
                return cached["metadata"]
        
        # Initialize OpenAI client
        client = OpenAI(api_key=settings.openai_api_key)
        
        # Encode image
        try:
            base64_image = encode_image(image_path)
        except Exception as e:
            console.print(f"[bold red]Error encoding image {image_path}: {str(e)}[/bold red]")
            return {
                "frame": image_path_obj.name,
                "analysis": f"Error encoding image: {str(e)}",
                "metadata": {
                    "processed_at": datetime.datetime.now().isoformat(),
                    "error": True,
                    "error_message": f"Error encoding image: {str(e)}"
                }
            }
        
        # Prepare prompts
        system_prompt = """
        You are an expert video content analyst. Your task is to analyze video frames and extract detailed information about what is being demonstrated. For each frame:
        
        1. Describe the main action or process being shown
        2. Identify any tools, equipment, or resources visible
        3. Note specific techniques or methods being demonstrated
        4. Capture important details that would help someone replicate what's shown
        5. Recognize any safety measures or precautions being taken
        
        Provide clear, objective descriptions focusing on observable details and actions.
        Your analysis should help build a comprehensive understanding of the overall process.
        """
        
        # Add transcript context to the user prompt if available
        user_prompt = """
        Analyze this video frame and provide a detailed description of what's being demonstrated. Include:
        - What specific action or step is being shown
        - What tools or resources are being used
        - What techniques or methods are being demonstrated
        - Any important details needed to understand or replicate this step
        - Any visible safety measures or best practices
        
        Be specific and focus on observable details that contribute to understanding the process.
        """
        
        if transcript_context:
            user_prompt += f"""
            
            The following transcript text is associated with this frame, which may provide additional context:
            "{transcript_context}"
            
            Use this transcript information to enhance your analysis, but focus primarily on what is visually shown in the frame.
            """
        
        # Get frame analysis
        with Timer(metrics_tracker, "frame_analysis_duration") if metrics_tracker else nullcontext():
            try:
                model = settings.fast_model
                if "o1" not in model and "o1-mini" not in model and "o3" not in model and "o3-mini" not in model:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}",
                                            "detail": "high"
                                        }
                                    }
                                ]
                            }   
                        ],
                        max_tokens=1000,
                        temperature=0.2
                    )   
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt},
                                ]
                            }
                        ],
                        max_completion_tokens=1000
                    )
                    
                # Process response
                analysis = response.choices[0].message.content
                
            except Exception as api_error:
                console.print(f"[bold red]API error analyzing frame {image_path}: {str(api_error)}[/bold red]")
                if metrics_tracker:
                    metrics_tracker.update_metrics(error_count=1)
                return {
                    "frame": image_path_obj.name,
                    "analysis": f"Error during API call: {str(api_error)}",
                    "metadata": {
                        "processed_at": datetime.datetime.now().isoformat(),
                        "error": True,
                        "error_message": f"API error: {str(api_error)}"
                    }
                }
        
        # Structure the results
        result = {
            "frame": Path(image_path).name,
            "analysis": analysis,
            "metadata": {
                "processed_at": datetime.datetime.now().isoformat(),
                "model_used": model
            }
        }
        
        # Add transcript context to the result if available
        if transcript_context:
            result["transcript_context"] = transcript_context
        
        # Cache the result
        mark_as_processed(image_path, "frame_analysis", result)
        
        return result
        
    except Exception as e:
        console.print(f"[bold red]Error analyzing frame {image_path}: {str(e)}[/bold red]")
        if metrics_tracker:
            metrics_tracker.update_metrics(error_count=1)
        # Return a minimal valid result instead of just an error
        return {
            "frame": Path(image_path).name if isinstance(image_path, (str, Path)) else "unknown",
            "analysis": f"Error analyzing frame: {str(e)}",
            "metadata": {
                "processed_at": datetime.datetime.now().isoformat(),
                "error": True,
                "error_message": str(e)
            }
        }
    finally:
        if metrics_tracker:
            metrics_tracker.end_processing()

async def describe_frames(
    frame_paths: List[str],
    output_dir: str,
    batch_size: int = 5,
    max_concurrent_batches: int = 3,
    metrics_tracker: Optional[MetricsTracker] = None,
    transcript_data: Optional[Dict[str, Any]] = None,
    video_duration: Optional[float] = None
) -> Dict[str, Any]:
    """
    Analyze multiple video frames in parallel batches.
    
    Args:
        frame_paths: List of paths to frame image files
        output_dir: Directory to save analysis results
        batch_size: Number of frames to process in each batch
        max_concurrent_batches: Maximum number of batches to process in parallel
        metrics_tracker: Optional metrics tracking instance
        transcript_data: Optional transcript data to associate with frames
        video_duration: Optional total duration of the video in seconds
        
    Returns:
        Dictionary containing analysis results for all frames
    """
    try:
        # Ensure batch_size and max_concurrent_batches are integers
        try:
            batch_size = int(batch_size)
        except (TypeError, ValueError):
            console.print("[yellow]Warning: Invalid batch_size, using default of 5[/yellow]")
            batch_size = 5
            
        try:
            max_concurrent_batches = int(max_concurrent_batches)
        except (TypeError, ValueError):
            console.print("[yellow]Warning: Invalid max_concurrent_batches, using default of 3[/yellow]")
            max_concurrent_batches = 3
        
        if metrics_tracker:
            metrics_tracker.start_processing("batch_frame_analysis")
        
        # Process frames in batches
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        # Extract frame timestamps if possible
        frame_timestamps = {}
        for frame_path in frame_paths:
            # Try to extract timestamp from filename (assuming format like frame_00123.jpg where 00123 is the frame number)
            try:
                frame_name = Path(frame_path).stem
                if '_' in frame_name:
                    frame_num = int(frame_name.split('_')[1])
                    # Estimate timestamp based on frame number and video duration
                    if video_duration:
                        # Assuming frames are evenly distributed
                        timestamp = (frame_num / len(frame_paths)) * video_duration
                        frame_timestamps[frame_path] = timestamp
            except (ValueError, IndexError):
                # If we can't extract timestamp, we'll handle it later
                pass
        
        # Function to get relevant transcript segments for a frame
        def get_transcript_summary(frame_path: str) -> Optional[str]:
            if not transcript_data or "segments" not in transcript_data:
                return None
                
            timestamp = frame_timestamps.get(frame_path)
            if timestamp is None:
                return None
                
            # Find segments that are close to this timestamp
            relevant_segments = []
            for segment in transcript_data["segments"]:
                # Check if segment is within a window around the timestamp (e.g., ±10 seconds)
                if "start" in segment and "end" in segment:
                    if abs(segment["start"] - timestamp) <= 10 or abs(segment["end"] - timestamp) <= 10 or (segment["start"] <= timestamp <= segment["end"]):
                        relevant_segments.append(segment["text"])
            
            if relevant_segments:
                return " ".join(relevant_segments)
            
            # If no exact matches, return a general summary from nearby segments
            if "text" in transcript_data:
                return "Transcript summary not available for this specific frame."
                
            return None
        
        async def process_batch(batch: List[str]) -> None:
            async with semaphore:
                # Create tasks with transcript context
                tasks = []
                for frame in batch:
                    transcript_summary = get_transcript_summary(frame)
                    tasks.append(analyze_frame(frame, metrics_tracker, transcript_summary))
                
                batch_results = await asyncio.gather(*tasks)
                
                for frame, result in zip(batch, batch_results):
                    # Add transcript summary to the result if available
                    transcript_summary = get_transcript_summary(frame)
                    if transcript_summary and "transcript_context" not in result:
                        result["transcript_summary"] = transcript_summary
                    results[frame] = result
        
        # Create batches
        batches = [frame_paths[i:i + batch_size] for i in range(0, len(frame_paths), batch_size)]
        
        # Process all batches
        await asyncio.gather(*[process_batch(batch) for batch in batches])
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results_file = output_path / "frame_analysis.json"
        
        results_file.write_text(json.dumps(results, indent=2))
        
        console.print(f"[bold green]Frame analysis saved to {results_file}[/bold green]")
        return results
        
    except Exception as e:
        console.print(f"[bold red]Error in batch frame analysis: {str(e)}[/bold red]")
        if metrics_tracker:
            metrics_tracker.update_metrics(error_count=1)
        raise
    finally:
        if metrics_tracker:
            metrics_tracker.end_processing()
