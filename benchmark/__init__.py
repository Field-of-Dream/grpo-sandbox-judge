"""
基准测试模块 - 提供各种任务的基准测试功能

本模块包含在不同任务（如math、physics、biomed等）上
运行LLM基准测试的工具。
"""

from .runner import (
    run_benchmark,
    load_reward_function,
    load_task_config,
    BenchmarkResult,
)

__all__ = [
    "run_benchmark",
    "load_reward_function", 
    "load_task_config",
    "BenchmarkResult",
]
