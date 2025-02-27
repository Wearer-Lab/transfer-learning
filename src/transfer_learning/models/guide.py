"""Models for step-by-step guides generated from processed videos."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class GuideStep(BaseModel):
    """A single step in a step-by-step guide."""
    step_number: int = Field(..., description="The step number")
    title: str = Field(..., description="The title of the step")
    description: str = Field(..., description="Detailed description of what to do in this step")
    expected_outcome: Optional[str] = Field(None, description="What should be achieved after completing this step")
    tools_required: Optional[List[str]] = Field(None, description="Tools needed for this step")
    tips: Optional[List[str]] = Field(None, description="Tips for successfully completing this step")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "step_number": 1,
                "title": "Set up the CAD environment",
                "description": "Open your CAD software and create a new project with the appropriate settings.",
                "expected_outcome": "A new project ready for design work",
                "tools_required": ["CAD software"],
                "tips": ["Make sure to set the correct units before starting"]
            }
        }


class StepByStepGuide(BaseModel):
    """A complete step-by-step guide for LLM agents."""
    title: str = Field(..., description="The title of the guide")
    introduction: str = Field(..., description="Introduction explaining the purpose and context of the guide")
    difficulty_level: str = Field(..., description="Difficulty level (Beginner, Intermediate, Advanced)")
    estimated_time: Optional[str] = Field(None, description="Estimated time to complete the process")
    prerequisites: Optional[List[str]] = Field(None, description="Prerequisites for following the guide")
    materials_required: Optional[List[str]] = Field(None, description="Materials required for the process")
    software_required: Optional[List[str]] = Field(None, description="Software required for the process")
    steps: List[GuideStep] = Field(..., description="The step-by-step instructions")
    troubleshooting: Optional[List[Dict[str, str]]] = Field(None, description="Common issues and their solutions")
    conclusion: str = Field(..., description="Conclusion summarizing the process and next steps")
    source_video_data: Optional[Dict[str, Any]] = Field(None, description="Metadata about the source video")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "title": "Creating a 3D Printed Screw",
                "introduction": "This guide will walk you through the process of designing a custom screw in CAD software for 3D printing.",
                "difficulty_level": "Intermediate",
                "estimated_time": "2 hours",
                "prerequisites": ["Basic knowledge of CAD software", "Understanding of 3D printing principles"],
                "materials_required": ["Computer with CAD software", "3D printer", "Filament"],
                "software_required": ["Fusion 360", "Cura"],
                "steps": [
                    {
                        "step_number": 1,
                        "title": "Set up the CAD environment",
                        "description": "Open your CAD software and create a new project with the appropriate settings.",
                        "expected_outcome": "A new project ready for design work",
                        "tools_required": ["CAD software"],
                        "tips": ["Make sure to set the correct units before starting"]
                    }
                ],
                "troubleshooting": [
                    {
                        "issue": "The screw threads are not printing correctly",
                        "solution": "Adjust the layer height in your slicer to a finer setting"
                    }
                ],
                "conclusion": "You have now successfully designed and 3D printed a custom screw. This process can be adapted for various types of fasteners.",
                "source_video_data": {
                    "title": "Screw Design Process",
                    "overall_summary": "A 3D printed part with a detailed design process",
                    "design_steps_count": 5,
                    "engineering_principles_count": 3,
                    "key_learnings_count": 4
                }
            }
        } 