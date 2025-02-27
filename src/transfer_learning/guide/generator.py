"""Step-by-step guide generator for video content.

This module provides functionality to generate detailed step-by-step guides
from video content, making processes replicable for others.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from rich.console import Console
from openai import OpenAI

from ..config import Settings
from ..models.video import VideoProcessing
from ..models.guide import StepByStepGuide, GuideStep
from ..utils.cache import is_already_processed, mark_as_processed, get_cached_result
from ..monitoring.metrics import MetricsTracker, Timer
from .templates import GUIDE_GENERATION_PROMPT, YOUTUBE_SPECIFIC_PROMPT, USER_DIRECTED_PROMPT
from .parser import parse_guide_text

class GuideGenerator:
    """Generates comprehensive step-by-step guides from video content."""
    
    def __init__(
        self,
        output_dir: str,
        metrics_tracker: Optional[MetricsTracker] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2
    ):
        self.settings = Settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_tracker = metrics_tracker or MetricsTracker()
        self.model = model or self.settings.openai_model
        self.temperature = temperature
        self.console = Console()
    
    async def generate_guide(
        self,
        video_processing: VideoProcessing,
        user_directive: Optional[str] = None,
        skip_cache: bool = False
    ) -> Union[Dict[str, Any], StepByStepGuide]:
        """
        Generate a step-by-step guide from processed video data.
        
        Args:
            video_processing: The processed video data
            user_directive: Optional user-provided directive for guide generation
            skip_cache: Whether to skip cache lookup and force regeneration
            
        Returns:
            StepByStepGuide object or Dict containing the generated guide
        """
        # Generate a unique identifier for this video processing
        video_identifier = video_processing.video_url or f"video_{hash(str(video_processing))}"
        
        # Sanitize the identifier to ensure it's a valid filename
        if video_processing.video_url:
            # Replace invalid characters with underscores
            video_identifier = "video_" + "".join(c if c.isalnum() else "_" for c in video_identifier)
        
        # Start metrics tracking
        self.metrics_tracker.start_processing(video_identifier)
        
        try:
            with Timer(self.metrics_tracker, "guide_generation_duration"):
                # Check cache if not skipping
                if not skip_cache:
                    # Include user directive in cache key if provided
                    directive_hash = f"_{hash(user_directive)}" if user_directive else ""
                    identifier = f"guide_gen_{hash(str(video_processing))}{directive_hash}"
                    
                    if is_already_processed(identifier, "guide"):
                        self.console.print("[bold green]Found cached guide. Loading...[/bold green]")
                        cached = get_cached_result(identifier, "guide")
                        if cached and "guide" in cached:
                            return cached["guide"]
                
                # Prepare prompts
                if user_directive:
                    # Use user-directed prompt if directive is provided
                    system_prompt = USER_DIRECTED_PROMPT
                    self.console.print("[bold blue]Using user-directed guide generation...[/bold blue]")
                else:
                    system_prompt = GUIDE_GENERATION_PROMPT
                
                # Add YouTube-specific prompt if applicable
                if video_processing.video_url and "youtube" in video_processing.video_url.lower():
                    system_prompt += YOUTUBE_SPECIFIC_PROMPT
                
                # Create user prompt
                user_prompt = self._create_user_prompt(video_processing, user_directive)
                
                # Generate guide
                guide = await self._generate_guide_content(system_prompt, user_prompt)
                
                # Parse and structure the guide
                structured_guide = self._structure_guide(guide, video_processing)
                
                # Save guide
                self._save_guide(structured_guide, video_processing, user_directive)
                
                # Cache result if not skipping
                if not skip_cache:
                    directive_hash = f"_{hash(user_directive)}" if user_directive else ""
                    identifier = f"guide_gen_{hash(str(video_processing))}{directive_hash}"
                    mark_as_processed(identifier, "guide", {"guide": structured_guide})
                
                return structured_guide
                
        except Exception as e:
            self.metrics_tracker.update_metrics(error_count=1)
            self.console.print(f"[bold red]Error generating guide: {str(e)}[/bold red]")
            raise
        finally:
            self.metrics_tracker.end_processing()
    
    def _create_user_prompt(self, video_processing: VideoProcessing, user_directive: Optional[str] = None) -> str:
        """Create the user prompt for guide generation."""
        dataset = video_processing.dataset
        prompt = f"""
        Create a detailed step-by-step guide based on this video content.
        The guide should help someone replicate exactly what is demonstrated in the video.

        CONTENT INFORMATION:
        Title: {dataset.title}
        Overview: {dataset.overall_summary}

        PROCESS BREAKDOWN:
        {json.dumps([{
            "step": step.step_number,
            "action": step.title,
            "details": step.description,
            "resources": step.tools_used,
            "methods": step.methods_used
        } for step in dataset.process_steps], indent=2)}

        KEY CONCEPTS:
        {json.dumps([{
            "concept": principle.name,
            "explanation": principle.description,
            "practical_use": principle.application
        } for principle in dataset.principles], indent=2)}

        LEARNING OUTCOMES:
        {json.dumps(dataset.key_learnings, indent=2)}

        ADDITIONAL INFORMATION:
        Difficulty Level: {dataset.difficulty_level}
        Required Resources: {json.dumps(dataset.tools_required or [], indent=2)}
        Estimated Duration: {dataset.estimated_duration or "Unknown"}
        """
        
        if video_processing.video_url:
            prompt += f"\nSource: Video URL - {video_processing.video_url}"
        
        # Add user directive if provided
        if user_directive:
            prompt += f"""
            
            USER DIRECTIVE:
            {user_directive}
            
            Please tailor the guide according to this directive while ensuring it remains comprehensive and accurate.
            Focus on aspects that align with the user's goals and requirements.
            """
        
        # Add formatting instructions
        prompt += """
        
        Please structure the guide with:
        1. Clear, descriptive title
        2. Brief introduction explaining what will be learned/accomplished
        3. List of prerequisites and required resources
        4. Numbered steps with clear actions and expected outcomes
        5. Tips and best practices for each step where relevant
        6. Troubleshooting section for common issues
        7. Conclusion summarizing key points
        
        Focus on being clear, specific, and actionable. Include all necessary details
        that would help someone successfully replicate what is shown in the video.
        """
        
        return prompt
    
    async def _generate_guide_content(self, system_prompt: str, user_prompt: str) -> str:
        """Generate the guide content using the LLM."""
        api_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        }

        if "o1" not in self.model and "o1-mini" not in self.model and "o3" not in self.model and "o3-mini" not in self.model:
            api_params["max_tokens"] = 4000
            api_params["temperature"] = self.temperature
        else:
            api_params["max_completion_tokens"] = 4000
        
        # Use the correct async method for the OpenAI client
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            **api_params
        )
        return response.choices[0].message.content
    
    def _structure_guide(
        self,
        guide_text: str,
        video_processing: VideoProcessing
    ) -> Union[Dict[str, Any], StepByStepGuide]:
        """Structure the generated guide text."""
        try:
            components = parse_guide_text(guide_text)
            guide_steps = [GuideStep(**step) for step in components["steps"]]
            
            return StepByStepGuide(
                title=video_processing.dataset.title,
                introduction=components["introduction"],
                difficulty_level=video_processing.dataset.difficulty_level,
                estimated_time=video_processing.dataset.estimated_duration,
                prerequisites=components["prerequisites"],
                materials_required=components["materials_required"],
                tools_required=video_processing.dataset.tools_required,
                steps=guide_steps,
                troubleshooting=components["troubleshooting"],
                conclusion=components["conclusion"],
                source_video_data={
                    "title": video_processing.dataset.title,
                    "overall_summary": video_processing.dataset.overall_summary,
                    "steps_count": len(video_processing.dataset.process_steps),
                    "principles_count": len(video_processing.dataset.principles),
                    "key_learnings_count": len(video_processing.dataset.key_learnings),
                }
            ).model_dump()
        except Exception as e:
            self.console.print(f"[yellow]Warning: Using simplified guide format: {str(e)}[/yellow]")
            return {
                "title": video_processing.dataset.title,
                "guide_text": guide_text,
                "metadata": {
                    "difficulty_level": video_processing.dataset.difficulty_level,
                    "estimated_time": video_processing.dataset.estimated_duration,
                    "tools_required": video_processing.dataset.tools_required
                }
            }
    
    def _save_guide(
        self, 
        guide: Dict[str, Any], 
        video_processing: VideoProcessing, 
        user_directive: Optional[str] = None
    ) -> None:
        """Save the generated guide to a file."""
        safe_title = "".join(c if c.isalnum() or c in "- " else "_" for c in video_processing.dataset.title)
        
        # Add a suffix for user-directed guides
        directive_suffix = "_user_directed" if user_directive else ""
        guide_file = self.output_dir / f"{safe_title}{directive_suffix}_guide.json"
        
        with open(guide_file, "w") as f:
            json.dump(guide, f, indent=2)
        
        self.console.print(f"[bold green]Guide saved to {guide_file}[/bold green]") 