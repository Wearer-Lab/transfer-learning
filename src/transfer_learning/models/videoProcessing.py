from pydantic import BaseModel, Field
from typing import List, Optional
from .process import ProcessOverview, ProcessStep, ProcessPrinciple


class VideoProcessing(BaseModel):
    dataset: ProcessOverview
    frame_descriptions: List[str]
    chunk_summaries: List[str]
    video_url: Optional[str] = None
    transcript: Optional[str] = None
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "dataset": {
                    "title": "Video Content Guide",
                    "overall_summary": "A comprehensive guide for the demonstrated process",
                    "process_steps": [
                        {
                            "step_number": 1,
                            "title": "Initial Setup",
                            "description": "Prepare the necessary tools and environment",
                            "tools_used": ["Required software", "Additional tools"],
                            "methods_used": ["Setup configuration", "Environment preparation"],
                        }
                    ],
                    "principles": [
                        {
                            "name": "Process Planning",
                            "description": "Identify the overall objectives and steps",
                            "application": "Process organization and execution",
                        }
                    ],
                    "key_learnings": [
                        "Process understanding",
                        "Efficient execution",
                    ],
                    "difficulty_level": "Intermediate",
                },
                "frame_descriptions": [
                    "The process begins with initial setup",
                    "Tools and environment are prepared",
                ],
                "chunk_summaries": [
                    "Setup and preparation phase",
                    "Process execution details",
                ],
                "video_url": "https://www.youtube.com/watch?v=example_video_id",
                "transcript": "This is a sample transcript of the video",
            }
        }


class ImageProcessing(BaseModel):
    process_steps: List[ProcessStep] = Field(
        description="List of process steps identified in the image"
    )
    principles: List[ProcessPrinciple] = Field(
        description="List of principles applied in the process"
    )
    overall_description: str = Field(
        description="Overall description of the process depicted in the image"
    )
