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

# 运行时常量（单一定义源: runtime.py）
from agent import Agent, AgentArgs
from agent_configs import (
    DEFAULT_TEAM_TEMPLATES,
    AgentProfile,
    AgentTeam,
    create_team_from_template,
    load_team_from_yaml,
    profile_to_agent_args,
    run_agent_team,
    save_team_to_yaml,
)
from config import AgentConfig, load_config
from runtime import (
    CMD_TIMEOUT,
    DOCKER_PATH,
    BaseRuntime,
    DockerRuntime,
    KaggleRuntime,
    LocalRuntime,
    create_runtime,
)

# When installed as py_modules (root-level files), imports need to be explicit
from train import (
    CodeExecutor,
    ProductManager,
    RewardModel,
    RLHFTrainingConfig,
    train,
)
from trajectory import Trajectory, TrajectoryStep

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
