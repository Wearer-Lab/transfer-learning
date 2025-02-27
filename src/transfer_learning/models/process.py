from pydantic import BaseModel, Field
from typing import List, Optional


class ProcessStep(BaseModel):
    step_number: int
    title: str
    description: str
    tools_used: List[str]
    methods_used: List[str]
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "step_number": 1,
                "title": "Initial Setup",
                "description": "Prepare the necessary tools and environment",
                "tools_used": ["Required software", "Additional tools"],
                "methods_used": ["Setup configuration", "Environment preparation"],
            }
        }


class ProcessPrinciple(BaseModel):
    name: str
    description: str
    application: str

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "name": "Process Planning",
                "description": "Identify the overall objectives and steps",
                "application": "Process organization and execution",
            }
        }


class ProcessOverview(BaseModel):
    title: str
    overall_summary: str
    process_steps: List[ProcessStep]
    principles: List[ProcessPrinciple]
    key_learnings: List[str]
    difficulty_level: str = Field(
        ..., description="Beginner, Intermediate, or Advanced"
    )
    tools_required: Optional[List[str]] = None
    estimated_duration: Optional[str] = None
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "title": "Video Content Process Guide",
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
            }
        }
