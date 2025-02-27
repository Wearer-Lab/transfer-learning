"""Process management utilities for the video processing pipeline.

This module provides functionality for managing and terminating processes
related to the video processing pipeline.
"""

import os
import sys
import signal
import psutil
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def find_child_processes() -> List[psutil.Process]:
    """
    Find all child processes of the current process.
    
    Returns:
        List of child process objects
    """
    current_process = psutil.Process(os.getpid())
    return current_process.children(recursive=True)

def terminate_child_processes(timeout: int = 5) -> None:
    """
    Terminate all child processes of the current process.
    
    Args:
        timeout: Timeout in seconds before forcefully killing processes
    """
    children = find_child_processes()
    
    if not children:
        logger.info("No child processes found to terminate")
        return
        
    logger.info(f"Terminating {len(children)} child processes")
    
    # First try to terminate gracefully
    for process in children:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
            
    # Wait for processes to terminate
    gone, alive = psutil.wait_procs(children, timeout=timeout)
    
    # If any processes are still alive, kill them forcefully
    if alive:
        logger.warning(f"{len(alive)} processes did not terminate gracefully, killing forcefully")
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass

def find_related_processes(name_pattern: str) -> List[psutil.Process]:
    """
    Find all processes related to the application by name pattern.
    
    Args:
        name_pattern: Pattern to match in process names
        
    Returns:
        List of matching process objects
    """
    related_processes = []
    
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if the process name or command line contains the pattern
            if (name_pattern.lower() in process.info['name'].lower() or 
                any(name_pattern.lower() in cmd.lower() for cmd in process.info['cmdline'] if cmd)):
                related_processes.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
    return related_processes

def terminate_related_processes(name_pattern: str, timeout: int = 5) -> int:
    """
    Terminate all processes related to the application by name pattern.
    
    Args:
        name_pattern: Pattern to match in process names
        timeout: Timeout in seconds before forcefully killing processes
        
    Returns:
        Number of processes terminated
    """
    processes = find_related_processes(name_pattern)
    
    if not processes:
        logger.info(f"No processes matching '{name_pattern}' found")
        return 0
        
    current_pid = os.getpid()
    processes_to_terminate = [p for p in processes if p.pid != current_pid]
    
    if not processes_to_terminate:
        logger.info(f"No processes matching '{name_pattern}' found to terminate")
        return 0
        
    logger.info(f"Terminating {len(processes_to_terminate)} processes matching '{name_pattern}'")
    
    # First try to terminate gracefully
    for process in processes_to_terminate:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
            
    # Wait for processes to terminate
    gone, alive = psutil.wait_procs(processes_to_terminate, timeout=timeout)
    
    # If any processes are still alive, kill them forcefully
    if alive:
        logger.warning(f"{len(alive)} processes did not terminate gracefully, killing forcefully")
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
                
    return len(processes_to_terminate) 