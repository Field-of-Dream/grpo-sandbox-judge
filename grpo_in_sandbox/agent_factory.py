"""Agent factory module - provides factory functions for creating Agent instances.

This module re-exports factory functions from grpo_in_sandbox.agent for
backward compatibility. New code should import directly from agent module.
"""

from .agent import (
    Agent,
    AgentArgs,
    AgentRegistry,
    create_agent,
    create_analyzer_agent,
    create_coder_agent,
    create_general_agent,
    create_research_agent,
    get_registry,
)

__all__ = [
    "Agent",
    "AgentArgs",
    "AgentRegistry",
    "create_agent",
    "create_coder_agent",
    "create_analyzer_agent",
    "create_research_agent",
    "create_general_agent",
    "get_registry",
]
