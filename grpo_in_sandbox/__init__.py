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

__version__ = "0.2.1"

from .config import AgentConfig, load_config
from .runtime import (
    CMD_TIMEOUT,
    DOCKER_PATH,
    BaseRuntime,
    DockerRuntime,
    KaggleRuntime,
    LocalRuntime,
    create_runtime,
)
from .trajectory import Trajectory, TrajectoryStep


def __getattr__(name):
    """Lazy-load optional subsystems only when their public exports are accessed."""
    lazy_exports = {
        "agent": {
            "Agent",
            "AgentArgs",
        },
        "agent_configs": {
            "AgentProfile",
            "AgentTeam",
            "DEFAULT_TEAM_TEMPLATES",
            "create_team_from_template",
            "load_team_from_yaml",
            "profile_to_agent_args",
            "run_agent_team",
            "save_team_to_yaml",
        },
        "train": {
            "AITrainerMode",
            "AIJudge",
            "AIJudgeRewardModel",
            "SelfPlayGRPO",
            "CodeExecutor",
            "ProductManager",
            "RewardModel",
            "RLHFTrainingConfig",
            "TrainingBackend",
            "train",
        },
    }

    for module_name, exports in lazy_exports.items():
        if name in exports:
            import importlib

            mod = importlib.import_module(f".{module_name}", __package__)
            value = getattr(mod, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "train",
    "RLHFTrainingConfig",
    "TrainingBackend",
    "ProductManager",
    "RewardModel",
    "CodeExecutor",
    "AITrainerMode",
    "AIJudge",
    "AIJudgeRewardModel",
    "SelfPlayGRPO",
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
