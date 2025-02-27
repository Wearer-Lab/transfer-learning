"""Audio transcription functionality for video processing.

This module provides functionality for extracting and transcribing
audio from video files using Whisper models.
"""

import os
import json
import cv2
import whisper
import tempfile
from faster_whisper import WhisperModel
from moviepy import VideoFileClip
from rich.console import Console
from rich.traceback import install
import ctypes
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from ..utils.formatting import print_json
from ..utils.cache import is_already_processed, mark_as_processed, get_cached_result
from ..config import settings

# Manually set libc_name
if os.name == "nt":  # Check if system is Windows
    libc_name = "msvcrt.dll"
    ctypes.CDLL(libc_name)

# Initialize rich console
console = Console()


def ensure_serializable(obj: Any) -> Any:
    """
    Ensure an object is JSON serializable by converting Path objects to strings.
    
    Args:
        obj: The object to make serializable
        
    Returns:
        A JSON serializable version of the object
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_serializable(item) for item in obj]
    else:
        return obj


async def transcribe_audio(
    video_path: str,
    output_dir: str,
    model: str = "base",
    device: str = "cpu",
    compute_type: str = "int8"
) -> Dict[str, Any]:
    """
    Transcribe audio from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save transcription
        model: Whisper model to use
        device: Device to use for processing
        compute_type: Compute type for processing
        
    Returns:
        Dictionary containing transcription data
    """
    try:
        # Convert paths to strings to avoid serialization issues
        video_path_str = str(video_path)
        output_dir_str = str(output_dir)
        
        # Check cache
        if is_already_processed(video_path_str, "audio_transcription"):
            console.print(f"[bold green]Found cached transcription for {Path(video_path_str).name}[/bold green]")
            cached = get_cached_result(video_path_str, "audio_transcription")
            if cached and "metadata" in cached:
                transcript_file = cached["metadata"].get("transcript_file")
                if transcript_file and Path(transcript_file).exists():
                    console.print(f"[bold green]Loading cached transcript[/bold green]")
                    return json.loads(Path(transcript_file).read_text())
        
        console.print(f"[bold blue]Transcribing audio from {Path(video_path_str).name}[/bold blue]")
        
        # Extract audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            audio_path = temp_audio.name
            video = VideoFileClip(str(video_path_str))
            video.audio.write_audiofile(audio_path, logger=None)
        
        try:
            # Use faster-whisper if available
            if hasattr(whisper, "WhisperModel"):
                # Initialize model
                whisper_model = WhisperModel(
                    model,
                    device=device,
                    compute_type=compute_type
                )
                
                # Transcribe
                segments, info = whisper_model.transcribe(
                    audio_path,
                    beam_size=5
                )
                
                # Format results
                transcript_data = {
                    "text": " ".join(segment.text for segment in segments),
                    "segments": [
                        {
                            "text": segment.text,
                            "start": segment.start,
                            "end": segment.end
                        }
                        for segment in segments
                    ]
                }
            else:
                # Fall back to regular whisper
                whisper_model = whisper.load_model(model)
                result = whisper_model.transcribe(audio_path)
                transcript_data = {
                    "text": result["text"],
                    "segments": result["segments"]
                }
            
            # Save transcript
            output_path = Path(output_dir_str)
            output_path.mkdir(parents=True, exist_ok=True)
            transcript_file = output_path / "transcript.json"
            
            # Ensure transcript_data is serializable
            serializable_data = ensure_serializable(transcript_data)
            
            # Convert the transcript file path to string
            transcript_file_str = str(transcript_file)
            
            # Write to file
            with open(transcript_file_str, 'w') as json_file:
                json.dump(serializable_data, json_file, indent=2)
            
            # Cache result
            mark_as_processed(
                video_path_str,
                "audio_transcription",
                {"transcript_file": transcript_file_str}
            )
            
            console.print(f"[bold green]Transcription saved to {transcript_file_str}[/bold green]")
            return transcript_data
            
        finally:
            # Clean up temporary audio file
            os.unlink(audio_path)
            
    except Exception as e:
        console.print(f"[bold red]Error transcribing audio: {str(e)}[/bold red]")
        if isinstance(e, TypeError) and "not JSON serializable" in str(e):
            console.print("[bold yellow]This appears to be a JSON serialization error. Check for Path objects or other non-serializable types.[/bold yellow]")
        return {"error": str(e)}


def transcribe_audio_fast_whisper(video_path, transcript_dir):
    try:
        # Check if this audio has already been transcribed with fast whisper
        if is_already_processed(video_path, "audio_transcription_fast"):
            console.print(f"[bold green]Audio for {video_path} has already been transcribed with fast whisper. Loading cached transcript...[/bold green]")
            cached_result = get_cached_result(video_path, "audio_transcription_fast")
            if cached_result and "metadata" in cached_result:
                transcript_data = cached_result["metadata"].get("transcript_data")
                if transcript_data:
                    console.print(f"[bold green]Using cached fast whisper transcript[/bold green]")
                    return transcript_data
                else:
                    console.print(f"[bold yellow]Cached transcript data not found. Will retranscribe.[/bold yellow]")
            else:
                console.print(f"[bold yellow]Cache found but metadata missing. Will retranscribe.[/bold yellow]")
        
        console.print("[bold blue]Transcribing audio...[/bold blue]")
        # Load the faster-whisper model
        model_size = "small"
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        # Check if the input is a video file or a cv2.VideoCapture object
        if isinstance(video_path, str):
            # Extract audio from video
            video = VideoFileClip(video_path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                audio_path = temp_audio.name
                video.audio.write_audiofile(audio_path, logger=None)
        elif isinstance(video_path, cv2.VideoCapture):
            # For cv2.VideoCapture objects, we need to save the video first
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                temp_video_path = temp_video.name
                out = cv2.VideoWriter(
                    temp_video_path,
                    fourcc,
                    30.0,
                    (int(video_path.get(3)), int(video_path.get(4))),
                )
                while True:
                    ret, frame = video_path.read()
                    if not ret:
                        break
                    out.write(frame)
                out.release()

            # Now extract audio from the saved video
            video = VideoFileClip(temp_video_path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                audio_path = temp_audio.name
                video.audio.write_audiofile(audio_path, logger=None)

            # Clean up temporary video file
            os.unlink(temp_video_path)
        else:
            raise ValueError("Unsupported video_path type")

        # Transcribe the audio
        segments, info = model.transcribe(audio_path, beam_size=5)

        # Prepare the transcript data
        transcript_data = {
            "text": " ".join(segment.text for segment in segments),
            "segments": [
                {"text": segment.text, "start": segment.start, "end": segment.end}
                for segment in segments
            ],
        }

        # Clean up temporary audio file
        os.unlink(audio_path)

        # Ensure transcript_data is serializable
        serializable_data = ensure_serializable(transcript_data)
        
        # Save the transcript to a JSON file
        json_filename = os.path.join(transcript_dir, "transcript.json")
        with open(json_filename, "w") as json_file:
            json.dump(serializable_data, json_file, indent=2)

        print_json(serializable_data, "Fast Whisper Transcript Data")
        console.print(f"[bold green]Transcript saved to {json_filename}[/bold green]")

        # Mark as processed
        mark_as_processed(video_path, "audio_transcription_fast", {
            "transcript_data": serializable_data,
            "transcript_file": str(json_filename)
        })

        return transcript_data
    except Exception as e:
        console.print(f"[bold red]Error in transcribe_audio: {str(e)}[/bold red]")
        if isinstance(e, TypeError) and "not JSON serializable" in str(e):
            console.print("[bold yellow]This appears to be a JSON serialization error. Check for Path objects or other non-serializable types.[/bold yellow]")
        return {"error": "Unable to transcribe audio"}
