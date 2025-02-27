"""Parser utilities for guide text processing."""

import re
from typing import Dict, Any, List

def parse_guide_text(guide_text: str) -> Dict[str, Any]:
    """Parse generated guide text into structured components."""
    components = {
        "introduction": "",
        "prerequisites": [],
        "materials_required": [],
        "steps": [],
        "troubleshooting": [],
        "conclusion": ""
    }
    
    # Extract introduction
    intro_match = re.search(r'^(.*?)(?=\n#|\n\d+\.)', guide_text, re.DOTALL)
    if intro_match:
        components["introduction"] = intro_match.group(1).strip()
    
    # Extract prerequisites
    prereq_match = re.search(r'(?:Prerequisites|Before You Begin):(.*?)(?=\n#|\n\d+\.|\n\*\*)', guide_text, re.DOTALL | re.IGNORECASE)
    if prereq_match:
        prereq_text = prereq_match.group(1).strip()
        components["prerequisites"] = [item.strip() for item in re.split(r'\n-|\n\*', prereq_text) if item.strip()]
    
    # Extract steps
    steps = []
    step_pattern = r'(?:Step )?(\d+)[\.:\)]\s+(?:\*\*)?([^\n]*?)(?:\*\*)?\n(.*?)(?=\n(?:Step )?(?:\d+)[\.:\)]|\n#|\Z)'
    for match in re.finditer(step_pattern, guide_text, re.DOTALL):
        step_num = int(match.group(1))
        title = match.group(2).strip()
        content = match.group(3).strip()
        
        step = {
            "step_number": step_num,
            "title": title,
            "description": content,
            "expected_outcome": None,
            "tools_required": None,
            "tips": None
        }
        
        # Extract expected outcome
        outcome_match = re.search(r'Expected Outcome:(.*?)(?=\n|$)', content, re.IGNORECASE)
        if outcome_match:
            step["expected_outcome"] = outcome_match.group(1).strip()
        
        # Extract tools
        tools_match = re.search(r'Tools Required:(.*?)(?=\n|$)', content, re.IGNORECASE)
        if tools_match:
            tools = tools_match.group(1).strip()
            step["tools_required"] = [t.strip() for t in tools.split(',')]
        
        # Extract tips
        tips_match = re.search(r'Tips:(.*?)(?=\n|$)', content, re.DOTALL | re.IGNORECASE)
        if tips_match:
            tips = tips_match.group(1).strip()
            step["tips"] = [t.strip() for t in re.split(r'\n-|\n\*', tips) if t.strip()]
        
        steps.append(step)
    
    components["steps"] = steps
    
    # Extract conclusion
    conclusion_match = re.search(r'(?:Conclusion|Summary):(.*?)(?=\Z)', guide_text, re.DOTALL | re.IGNORECASE)
    if conclusion_match:
        components["conclusion"] = conclusion_match.group(1).strip()
    
    return components 