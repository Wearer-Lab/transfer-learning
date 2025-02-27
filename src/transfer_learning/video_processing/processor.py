import cv2
import logging
from pathlib import Path
from typing import Optional, Generator, Tuple
import yt_dlp
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from ..models.base import VideoMetadata

class VideoProcessor:
    """Core video processing class that handles both local and YouTube videos."""
    
    def __init__(self, logging_level: int = logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)
    
    def get_video_metadata(self, file_path: str) -> VideoMetadata:
        """Extract metadata from a video file."""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {file_path}")
            
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            file_size = Path(file_path).stat().st_size
            
            return VideoMetadata(
                file_path=file_path,
                duration=duration,
                frame_count=frame_count,
                fps=fps,
                resolution=(width, height),
                file_size=file_size
            )
        finally:
            cap.release()
    
    def extract_frames(
        self, 
        video_path: str, 
        output_dir: str,
        frame_interval: int = 30,
        progress: Optional[Progress] = None
    ) -> Generator[Tuple[int, str], None, None]:
        """Extract frames from a video at specified intervals."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            task_id = None
            
            if progress:
                task_id = progress.add_task(
                    "[cyan]Extracting frames...",
                    total=frame_count
                )
            
            frame_number = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_number % frame_interval == 0:
                    frame_path = f"{output_dir}/frame_{frame_number:06d}.jpg"
                    cv2.imwrite(frame_path, frame)
                    yield frame_number, frame_path
                
                frame_number += 1
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)
                    
        finally:
            cap.release()
            if progress and task_id is not None:
                progress.remove_task(task_id)
    
    def download_youtube_video(
        self,
        url: str,
        output_dir: str,
        progress: Optional[Progress] = None
    ) -> str:
        """Download a YouTube video using yt-dlp."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        task_id = None
        if progress:
            task_id = progress.add_task(
                "[cyan]Downloading YouTube video...",
                total=100
            )
        
        def progress_hook(d):
            if d['status'] == 'downloading' and progress and task_id is not None:
                p = d.get('_percent_str', '0%').replace('%', '')
                try:
                    progress.update(task_id, completed=float(p))
                except ValueError:
                    pass
        
        ydl_opts = {
            'format': 'best',
            'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
            'progress_hooks': [progress_hook],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)
                return video_path
        finally:
            if progress and task_id is not None:
                progress.remove_task(task_id)
                
    def process_video(
        self,
        input_path: str,
        output_dir: str,
        frame_interval: int = 30,
        is_youtube: bool = False
    ) -> Tuple[VideoMetadata, Generator[Tuple[int, str], None, None]]:
        """Process a video file or YouTube URL and extract frames."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn()
        ) as progress:
            if is_youtube:
                self.logger.info(f"Downloading YouTube video: {input_path}")
                video_path = self.download_youtube_video(input_path, output_dir, progress)
            else:
                video_path = input_path
            
            self.logger.info(f"Processing video: {video_path}")
            metadata = self.get_video_metadata(video_path)
            frames = self.extract_frames(video_path, output_dir, frame_interval, progress)
            
            return metadata, frames 