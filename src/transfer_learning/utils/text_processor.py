"""Text processing utilities for video content analysis.

This module provides functionality for processing and combining text content
from video transcripts and frame descriptions into structured, meaningful content.
"""

import json
import asyncio
from typing import List, Any, Dict
from rich.console import Console

from ..config import Settings
from ..models.video import VideoProcessing
from ..guide.templates import TEXT_PROCESSING_PROMPT
from ..monitoring.metrics import MetricsTracker, Timer
from .cache import is_already_processed, mark_as_processed, get_cached_result

# Initialize console and settings
console = Console()
settings = Settings()

def chunk_text(text: str, max_chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for processing.
    
    Args:
        text: Text to split into chunks
        max_chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    return splitter.split_text(text)

async def process_chunk(llm: Any, chunk: str) -> str:
    """
    Process a single text chunk using the provided LLM.
    
    Args:
        llm: Language model to use for processing
        chunk: Text chunk to process
        
    Returns:
        Processed text content
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    
    messages = [
        SystemMessage(content=TEXT_PROCESSING_PROMPT),
        HumanMessage(content=f"Analyze and extract key information from this content:\n\n{chunk}")
    ]
    
    try:
        response = await llm.ainvoke(messages)
        return response.content
    except Exception as e:
        console.print(f"[bold red]Error processing chunk: {str(e)}[/bold red]")
        return f"Error processing chunk: {str(e)}"

async def combine_content(
    frame_descriptions: Dict[str, str],
    transcript: Dict[str, Any],
    metrics_tracker: MetricsTracker
) -> VideoProcessing:
    """
    Combine frame descriptions and transcript into structured content.
    
    This function:
    1. Analyzes visual content from video frames
    2. Processes audio transcript content
    3. Combines both into a coherent, structured format
    4. Identifies key steps, concepts, and important details
    
    Args:
        frame_descriptions: Dictionary of frame descriptions
        transcript: Dictionary containing transcript data
        metrics_tracker: Instance for tracking processing metrics
        
    Returns:
        VideoProcessing object containing structured content analysis
    """
    try:
        # Create identifier for caching
        content_hash = hash(str(frame_descriptions) + str(transcript))
        cache_key = f"content_combination_{content_hash}"
        
        # Check cache
        if is_already_processed(cache_key, "content_combination"):
            console.print("[bold green]Found cached content analysis. Loading...[/bold green]")
            cached = get_cached_result(cache_key, "content_combination")
            if cached and "analysis" in cached:
                return VideoProcessing.model_validate(cached["analysis"])
        
        with Timer(metrics_tracker, "content_analysis_duration"):
            # Sample frames if there are too many
            if len(frame_descriptions) > 20:
                console.print(f"[bold yellow]Sampling {20} frames from {len(frame_descriptions)} total frames[/bold yellow]")
                import random
                keys = list(frame_descriptions.keys())
                step = max(1, len(keys) // 20)
                sampled_keys = keys[::step][:20]
                sampled_descriptions = {k: frame_descriptions[k] for k in sampled_keys}
            else:
                sampled_descriptions = frame_descriptions
            
            # Combine descriptions
            combined_descriptions = "\n".join(
                description if isinstance(description, str) else json.dumps(description)
                for description in sampled_descriptions.values()
            )
            
            # Process text in chunks
            max_chunk_size = 4000
            overlap = 100
            
            description_chunks = chunk_text(combined_descriptions, max_chunk_size, overlap)
            transcript_text = transcript.get("text", "No transcript available.")
            transcript_chunks = chunk_text(transcript_text, max_chunk_size, overlap)
            
            # Process chunks in batches
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            
            async def process_chunks_batch(chunks: List[str], batch_size: int = 2) -> List[str]:
                """Process multiple chunks in parallel batches."""
                results = []
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    tasks = [process_single_chunk(chunk) for chunk in batch]
                    batch_results = await asyncio.gather(*tasks)
                    results.extend(batch_results)
                return results
            
            async def process_single_chunk(chunk: str) -> str:
                """Process an individual content chunk."""
                try:
                    if "o1" not in settings.fast_openai_model and "o1-mini" not in settings.fast_openai_model and "o3" not in settings.fast_openai_model and "o3-mini" not in settings.fast_openai_model:
                        response = client.chat.completions.create(
                            model=settings.fast_openai_model,
                            messages=[
                                {
                                    "role": "system", 
                                    "content": "Analyze this content and identify key steps, concepts, and important details that would help someone understand and replicate what is being shown or described."
                                },
                                {"role": "user", "content": chunk}
                            ],
                            max_tokens=800,
                            temperature=0
                        )
                    else:
                        response = client.chat.completions.create(
                            model=settings.fast_openai_model,
                            messages=[
                                {
                                    "role": "system", 
                                    "content": "Analyze this content and identify key steps, concepts, and important details that would help someone understand and replicate what is being shown or described."
                                },
                                {"role": "user", "content": chunk}
                            ],
                            max_completion_tokens=800
                        )
                    return response.choices[0].message.content
                except Exception as e:
                    console.print(f"[bold red]Error processing chunk: {str(e)}[/bold red]")
                    return f"Error processing chunk: {str(e)}"
            
            # Process all chunks
            chunk_summaries = await process_chunks_batch(description_chunks + transcript_chunks)
            
            # Create VideoProcessing object
            result = VideoProcessing(
                frame_descriptions=list(sampled_descriptions.values()),
                chunk_summaries=chunk_summaries,
                transcript=transcript_text
            )
            
            # Cache the result
            mark_as_processed(cache_key, "content_combination", {
                "analysis": result.model_dump()
            })
            
            return result
            
    except Exception as e:
        console.print(f"[bold red]Error combining content: {str(e)}[/bold red]")
        metrics_tracker.update_metrics(error_count=1)
        raise
