"""
Prompt creation for chemistry task (vanilla LLM mode).
"""
from typing import Any


def create_prompt(problem_data: dict[str, Any]) -> str:
    """Create prompt for chemistry problems."""
    return problem_data['problem_statement']
