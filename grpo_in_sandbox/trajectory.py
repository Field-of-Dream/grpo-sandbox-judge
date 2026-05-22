"""
轨迹模块 - 记录智能体执行过程中的完整历史

本模块定义TrajectoryStep和Trajectory类，用于记录智能体运行过程中的
每一步操作、思考、动作和观察结果。
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrajectoryStep(BaseModel):
    """
    智能体轨迹中的单一步骤。

    Attributes:
        thought: 智能体的思考内容
        reasoning_content: 推理过程内容（如chain-of-thought）
        action: 执行的动作（包含function和parameters）
        observation: 执行后的观察结果
        metadata: 附加元数据（如步数、执行时间等）
    """
    thought: str = ""
    reasoning_content: str = ""
    action: Dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        将步骤转换为字典格式。

        Returns:
            包含所有字段的字典
        """
        return {
            "thought": self.thought,
            "reasoning_content": self.reasoning_content,
            "action": self.action,
            "observation": self.observation,
            "metadata": self.metadata,
        }


class Trajectory(BaseModel):
    """
    完整的智能体运行轨迹。

    Attributes:
        problem_statement: 问题描述/任务陈述
        steps: 轨迹步骤列表
        metadata: 附加元数据
        reward_calc_time: 奖励计算耗时（可选）
        test_output: 测试输出（可选）
    """
    problem_statement: str
    steps: List[TrajectoryStep] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    reward_calc_time: Optional[float] = None
    test_output: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        将轨迹转换为字典格式。

        Returns:
            包含所有字段的字典
        """
        return {
            "problem_statement": self.problem_statement,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
            "reward_calc_time": self.reward_calc_time,
            "test_output": self.test_output,
        }
