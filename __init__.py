"""
GRPO-in-Sandbox - Agentic Reinforcement Learning Library

A standard library for training language models using GRPO (Group Relative Policy Optimization)
within a code sandbox environment for general agentic intelligence.

Usage:
    from grpo_in_sandbox import train, RLHFTrainingConfig, create_runtime
    
    config = RLHFTrainingConfig(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        num_train_epochs=3,
    )
    results = train(config)
"""

__version__ = "0.2.0"

# 模块级常量
CMD_TIMEOUT = 120
DOCKER_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/local/go/bin:/opt/miniconda3/envs/testbed/bin"

# When installed as py_modules (root-level files), imports need to be explicit
from train import (
    train,
    RLHFTrainingConfig,
    ProductManager,
    RewardModel,
    CodeExecutor,
)

from runtime import (
    BaseRuntime,
    LocalRuntime,
    DockerRuntime,
    KaggleRuntime,
    create_runtime,
)

from agent import Agent, AgentArgs

from agent_configs import (
    AgentProfile,
    AgentTeam,
    load_team_from_yaml,
    save_team_to_yaml,
    create_team_from_template,
    DEFAULT_TEAM_TEMPLATES,
    profile_to_agent_args,
    run_agent_team,
)

from trajectory import Trajectory, TrajectoryStep

from config import AgentConfig, load_config

__all__ = [
    "train",
    "RLHFTrainingConfig",
    "ProductManager",
    "RewardModel",
    "CodeExecutor",
    "BaseRuntime",
    "LocalRuntime",
    "DockerRuntime",
    "KaggleRuntime",
    "create_runtime",
    "Agent",
    "AgentArgs",
    "AgentProfile",
    "AgentTeam",
    "load_team_from_yaml",
    "save_team_to_yaml",
    "create_team_from_template",
    "DEFAULT_TEAM_TEMPLATES",
    "profile_to_agent_args",
    "run_agent_team",
    "Trajectory",
    "TrajectoryStep",
    "AgentConfig",
    "load_config",
    "__version__",
    "CMD_TIMEOUT",
    "DOCKER_PATH",
]
