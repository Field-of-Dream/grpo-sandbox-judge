"""
Config Module - Configuration Management

Provides functions to load and manage prompt templates and agent configurations.
"""

import os
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class AgentConfig:
    """Agent configuration."""
    system_prompt: str = ""
    instance_prompt: str = ""
    working_dir: str = "/testbed"
    input_dir: str = "/testbed/input"
    output_dir: str = "/testbed/output"
    max_steps: int = 30
    max_token_limit: int = 65536
    extra_body: dict[str, Any] | None = None

    def format_system_prompt(self, **kwargs) -> str:
        return self.system_prompt.format(**kwargs)

    def format_instance_prompt(self, **kwargs) -> str:
        return self.instance_prompt.format(**kwargs)


def load_prompt_config(config_path: str) -> AgentConfig:
    """Load prompt configuration from YAML file."""
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return AgentConfig(
        system_prompt=config.get("system_prompt", ""),
        instance_prompt=config.get("instance_prompt", ""),
        working_dir=config.get("working_dir", "/testbed"),
        input_dir=config.get("input_dir", "/testbed/input"),
        output_dir=config.get("output_dir", "/testbed/output"),
    )


def get_default_config_path() -> str:
    """Get default prompt config path."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "config", "general.yaml")


def load_config(config_path: str | None = None) -> AgentConfig:
    """Load configuration from path or use default."""
    if config_path is None:
        config_path = get_default_config_path()
    return load_prompt_config(config_path)


__all__ = [
    "AgentConfig",
    "load_prompt_config",
    "load_config",
    "get_default_config_path",
]
