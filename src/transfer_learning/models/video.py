"""Models for video processing and content analysis."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ProcessStep(BaseModel):
    """A single step in a process."""
    step_number: int = Field(..., description="Step number in sequence")
    title: str = Field(..., description="Title of the step")
    description: str = Field(..., description="Detailed description of the step")
    tools_used: Optional[List[str]] = Field(None, description="Tools used in this step")
    methods_used: Optional[List[str]] = Field(None, description="Methods or techniques used")

class ProcessPrinciple(BaseModel):
    """A principle applied in the process."""
    name: str = Field(..., description="Name of the principle")
    description: str = Field(..., description="Description of the principle")
    application: str = Field(..., description="How the principle is applied")

class ProcessOverview(BaseModel):
    """Overview of a process demonstrated in a video."""
    title: str = Field(..., description="Title of the process")
    overall_summary: str = Field(..., description="Overall summary of the process")
    process_steps: List[ProcessStep] = Field(..., description="Steps in the process")
    principles: List[ProcessPrinciple] = Field(..., description="Principles applied")
    key_learnings: List[str] = Field(..., description="Key learnings from the process")
    difficulty_level: str = Field(..., description="Difficulty level of the process")
    tools_required: Optional[List[str]] = Field(None, description="Tools required")
    estimated_duration: Optional[str] = Field(None, description="Estimated time to complete")

class ImageProcessing(BaseModel):
    """Analysis of a single video frame."""
    process_steps: List[ProcessStep] = Field(..., description="Steps identified in the frame")
    principles: List[ProcessPrinciple] = Field(..., description="Principles applied")
    overall_description: str = Field(..., description="Overall description of the frame")

class VideoProcessing(BaseModel):
    """Complete video processing results."""
    dataset: ProcessOverview
    frame_descriptions: List[str]
    chunk_summaries: List[str]
    video_url: Optional[str] = None
    transcript: Optional[str] = None 