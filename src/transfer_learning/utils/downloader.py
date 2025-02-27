from pytube import YouTube
import yt_dlp
import os
import traceback
from datetime import datetime
from typing import Optional
from pathlib import Path

from rich.console import Console
from rich.traceback import install

from ..config import settings

# Initialize rich console
console = Console()

# Use settings for paths
VIDEOS_FOLDER = settings.videos_dir
DATASETS_FOLDER = settings.data_dir
FRAMES_FOLDER = settings.frames_dir
TRANSCRIPTS_FOLDER = settings.transcripts_dir

from .cache import is_already_processed, mark_as_processed, get_cached_result

def download_video_pythube(youtube_link):
    try:
        # Check if this video has already been downloaded
        if is_already_processed(youtube_link, "video_download_pythube"):
            console.print(f"[bold green]Video from {youtube_link} has already been downloaded. Loading cached path...[/bold green]")
            cached_result = get_cached_result(youtube_link, "video_download_pythube")
            if cached_result and "metadata" in cached_result:
                video_path = cached_result["metadata"].get("video_path")
                if video_path and os.path.exists(video_path):
                    console.print(f"[bold green]Using cached video at {video_path}[/bold green]")
                    return video_path
                else:
                    console.print(f"[bold yellow]Cached video file not found. Will redownload.[/bold yellow]")
            else:
                console.print(f"[bold yellow]Cache found but metadata missing. Will redownload.[/bold yellow]")
        
        console.print("[bold blue]Starting video download process.[/bold blue]")

        # Hardcode the download directory
        download_path = f"{VIDEOS_FOLDER}"
        if not os.path.exists(download_path):
            os.makedirs(download_path)
            console.print(f"[bold green]Created download directory at '{download_path}'.[/bold green]")

        yt = YouTube(youtube_link)
        console.print(f"[bold green]Successfully created YouTube object for link:[/bold green] {youtube_link}")

        # Log available streams
        streams = yt.streams.all()
        console.print(f"[bold yellow]Available streams:[/bold yellow] {streams}")

        video_stream = yt.streams.get_highest_resolution()
        console.print(f"[bold green]Retrieved highest resolution stream for video:[/bold green] {yt.title}")

        video_stream.download(output_path=download_path)
        video_path = os.path.join(download_path, f"{yt.title}.mp4")
        console.print(f"[bold green]Video '{yt.title}' has been downloaded successfully to '{download_path}'.[/bold green]")
        
        # Mark as processed
        mark_as_processed(youtube_link, "video_download_pythube", {"video_path": video_path})
        
        return video_path
    except Exception as e:
        console.print(
            "[bold red]An error occurred while downloading the video.[/bold red]"
        )
        console.print(f"[bold red]Error: {str(e)}.[/bold red]")
        console.print(f"[bold red]{traceback.format_exc()}[/bold red]")
        return None


def download_video_yt_dlp(url: str, output_dir: str = None) -> Optional[str]:
    try:
        # Check if this video has already been downloaded
        if is_already_processed(url, "video_download_yt_dlp"):
            console.print(f"[bold green]Video from {url} has already been downloaded. Loading cached path...[/bold green]")
            cached_result = get_cached_result(url, "video_download_yt_dlp")
            if cached_result and "metadata" in cached_result:
                video_path = cached_result["metadata"].get("video_path")
                if video_path and os.path.exists(video_path):
                    console.print(f"[bold green]Using cached video at {video_path}[/bold green]")
                    return video_path
                else:
                    console.print(f"[bold yellow]Cached video file not found. Will redownload.[/bold yellow]")
            else:
                console.print(f"[bold yellow]Cache found but metadata missing. Will redownload.[/bold yellow]")
        
        # Use provided output directory or default to VIDEOS_FOLDER
        download_dir = output_dir if output_dir else VIDEOS_FOLDER
        
        # Ensure download directory exists
        from ..utils.path import ensure_directory, get_safe_path
        
        download_dir_path = Path(download_dir)
        download_dir_path = ensure_directory(download_dir_path)
        
        # Create a safe output template that avoids problematic characters
        safe_output_template = str(download_dir_path / '%(title)s.%(ext)s')
        safe_output_template = safe_output_template.replace(' ', '_')
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': safe_output_template,
            'restrictfilenames': True,  # Restrict filenames to ASCII chars
            'ignoreerrors': True,       # Skip unavailable videos
            'no_warnings': False,       # Show warnings
            'quiet': False,             # Print messages to stdout
            'noplaylist': True,         # Download single video, not playlist
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                console.print(f"[bold red]Failed to extract info from URL: {url}[/bold red]")
                return None
                
            video_path = ydl.prepare_filename(info)
            
            # Ensure the video path is safe
            video_path = str(get_safe_path(video_path))
            
            # Verify the file exists
            if not os.path.exists(video_path):
                # Try to find the file with a different extension
                possible_extensions = ['.mp4', '.mkv', '.webm', '.avi']
                for ext in possible_extensions:
                    base_path = os.path.splitext(video_path)[0]
                    alt_path = f"{base_path}{ext}"
                    if os.path.exists(alt_path):
                        video_path = alt_path
                        break
                        
            if not os.path.exists(video_path):
                console.print(f"[bold red]Downloaded file not found at expected path: {video_path}[/bold red]")
                # Try to find any recently created video file in the directory
                download_dir_files = list(Path(download_dir_path).glob("*"))
                if download_dir_files:
                    # Sort by creation time, newest first
                    download_dir_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    video_path = str(download_dir_files[0])
                    console.print(f"[bold yellow]Using most recent file instead: {video_path}[/bold yellow]")
                else:
                    return None
            
            console.print(f"[bold green]Video Downloaded:[/bold green] {video_path}")

        console.print(f"[bold green]Download completed:[/bold green] {video_path}")
        
        # Mark as processed
        mark_as_processed(url, "video_download_yt_dlp", {"video_path": video_path})
        
        return video_path
    except Exception as e:
        console.print(
            f"[bold red]An error occurred while downloading the video: {str(e)}[/bold red]"
        )
        console.print_exception()
        return None
