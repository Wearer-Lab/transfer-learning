from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ProcessStep(BaseModel):
    """A generic step in any process captured from video."""
    step_number: int = Field(..., description="The order of this step in the process")
    title: str = Field(..., description="Short title describing the step")
    description: str = Field(..., description="Detailed description of what happens in this step")
    tools_used: Optional[List[str]] = Field(default=None, description="Software, tools, or equipment used")
    duration: Optional[float] = Field(default=None, description="Duration of this step in seconds")
    key_points: Optional[List[str]] = Field(default=None, description="Important points to note about this step")
    timestamp: Optional[float] = Field(default=None, description="Start time of this step in the video")

class ProcessPrinciple(BaseModel):
    """Key principles or concepts identified in the process."""
    name: str = Field(..., description="Name of the principle")
    description: str = Field(..., description="Detailed explanation of the principle")
    importance: str = Field(..., description="Why this principle matters in the process")
    examples: Optional[List[str]] = Field(default=None, description="Examples from the video demonstrating this principle")

class VideoProcess(BaseModel):
    """Overall representation of a process documented in a video."""
    title: str = Field(..., description="Title of the process")
    description: str = Field(..., description="Overview of the entire process")
    steps: List[ProcessStep] = Field(..., description="Sequential steps in the process")
    principles: Optional[List[ProcessPrinciple]] = Field(default=None, description="Key principles identified")
    total_duration: Optional[float] = Field(default=None, description="Total duration in seconds")
    created_at: datetime = Field(default_factory=datetime.now)
    source_type: str = Field(..., description="Type of video source (local/youtube)")
    source_path: str = Field(..., description="Path or URL to the source video")
    
class VideoMetadata(BaseModel):
    """Metadata about a processed video."""
    file_path: str = Field(..., description="Path to the video file")
    duration: float = Field(..., description="Duration in seconds")
    frame_count: int = Field(..., description="Total number of frames")
    fps: float = Field(..., description="Frames per second")
    resolution: tuple[int, int] = Field(..., description="Video resolution (width, height)")
    file_size: int = Field(..., description="File size in bytes")
    processed_at: datetime = Field(default_factory=datetime.now) 