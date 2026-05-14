"""
GRPO-in-Sandbox Training Module

This module provides a complete implementation for training language models using GRPO
(Group Relative Policy Optimization) within a code sandbox environment.

Key Components:
- RLHFTrainingConfig: Configuration for training
- ProductManager: Task prompt generation
- RewardModel: Sandbox-based reward scoring
- GRPOTrainer (via TRL): Unsloth-optimized GRPO training with vLLM acceleration

Usage:
    from grpo_in_sandbox import train, RLHFTrainingConfig

    config = RLHFTrainingConfig(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        num_train_epochs=3,
        max_steps=100,
    )
    results = train(config)
"""

import contextlib
import json
import logging
import os
import random
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

# Module-level logger
train_logger = logging.getLogger(__name__)


def _setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with consistent formatting."""
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        ))
        log.addHandler(handler)
        log.setLevel(level)
    return log


class AITrainerMode(str, Enum):
    """AI Judge training mode for self-play GRPO.

    Attributes:
        SANDBOX: Use sandbox execution for reward scoring (default)
        AI_JUDGE: Use LLM-as-judge for reward scoring
    """

    SANDBOX = "sandbox"
    AI_JUDGE = "ai_judge"


@dataclass
class RLHFTrainingConfig:
    """Configuration for GRPO (Group Relative Policy Optimization) training.

    This dataclass contains all hyperparameters and settings needed for training a language model
    using GRPO within a sandbox environment (Docker/Kaggle/Local).

    Args:
        model_name_or_path: HuggingFace model path or local directory.
            Examples: "Qwen/Qwen2.5-0.5B-Instruct", "./my_model"

    LoRA Parameters:
        lora_rank: LoRA attention rank (default: 16).
            Higher = more parameters, better quality, slower training.
        lora_alpha: LoRA scaling factor (default: 16).
            Typically set equal to lora_rank.
        lora_dropout: Dropout for LoRA layers (default: 0.0).
        target_modules: Modules to apply LoRA. Auto-detected if None.
            Common: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    Optimizer Parameters:
        learning_rate: Learning rate for optimizer (default: 5e-6).
            Smaller models can use higher lr (e.g., 1e-4 for 0.5B models).
        beta: KL penalty coefficient (default: 0.001).
            Controls policy vs reference model divergence.
            Higher = more regularization, less exploration.
        max_grad_norm: Gradient clipping threshold (default: 0.1).

    Generation Parameters:
        num_train_epochs: Number of training epochs (default: 3).
        max_seq_length: Max sequence length for model (default: 2048).
        temperature: Sampling temperature (default: 0.7).
            Higher = more creative, lower = more deterministic.
        top_p: Nucleus sampling threshold (default: 0.9).
        top_k: Top-k sampling parameter (default: 50).
        max_completion_length: Max tokens to generate (default: 1024).

    Batch Parameters:
        per_device_train_batch_size: Batch size per GPU (default: 4).
        gradient_accumulation_steps: Steps to accumulate gradients (default: 4).
            Effective batch = per_device_train_batch_size * gradient_accumulation_steps.

    GRPO Parameters:
        num_generations: Responses per prompt for advantage estimation (default: 4).
            More generations = better advantage estimation, slower training.

    Training Limits:
        max_steps: Max training steps (default: 100).
        max_token_limit: Context window size (default: 65536).
        max_tokens_per_call: Max tokens per LLM call (default: 65536).

    Runtime Parameters:
        docker_image: Docker image for sandbox (default: "cdx123/llm-in-sandbox:v0.1").
        output_dir: Directory for checkpoints and logs (default: "./output").
        seed: Random seed for reproducibility (default: 42).

    Early Stopping:
        patience: Epochs to wait before early stop (default: 3).
        min_improvement: Minimum improvement threshold (default: 0.01).

    vLLM Parameters:
        use_vllm: Use vLLM for fast inference (default: True).
        vllm_gpu_memory_utilization: GPU memory for vLLM (default: 0.7).
        fast_inference: Enable Unsloth fast inference (default: True).

    Example:
        >>> config = RLHFTrainingConfig(
        ...     model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        ...     num_train_epochs=3,
        ...     max_steps=100,
        ...     learning_rate=1e-4,
        ... )
        >>> results = train(config)
    """

    model_name_or_path: str = "./model"

    # LoRA settings
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] | None = None  # Auto-detect if None

    # Optimizer settings
    learning_rate: float = 5e-6
    beta: float = 0.001  # KL penalty coefficient (Unsloth default)
    max_grad_norm: float = 0.1

    # Generation settings
    num_train_epochs: int = 3
    max_seq_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_completion_length: int = 1024

    # Batch settings
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # GRPO-specific
    num_generations: int = 4  # Generations per prompt for GRPO group-relative advantage

    # Training limits
    max_steps: int = 100
    max_token_limit: int = 65536
    max_tokens_per_call: int = 65536

    # Runtime settings
    docker_image: str = "cdx123/llm-in-sandbox:v0.1"
    output_dir: str = "./output"
    seed: int = 42

    # Early stopping
    patience: int = 3
    min_improvement: float = 0.01

    # vLLM settings
    use_vllm: bool = True  # Use vLLM for fast generation during RL
    vllm_gpu_memory_utilization: float = 0.7
    fast_inference: bool = True  # Enable vLLM fast inference

    # AI Judge mode (for self-play GRPO)
    mode: AITrainerMode = AITrainerMode.SANDBOX  # 'sandbox' or 'ai_judge'
    judge_llm_name: str = "openai/gpt-4o-mini"  # LLM to use as judge
    judge_criteria: list[str] | None = None  # e.g. ["correctness", "clarity", "helpfulness"]
    judge_temperature: float = 0.1  # Low temp for consistent judging
    judge_base_url: str | None = None  # Custom API endpoint for judge

    def __repr__(self) -> str:
        """Show training configuration."""
        return (
            f"RLHFTrainingConfig("
            f"model={self.model_name_or_path}, "
            f"mode={self.mode.value}, "
            f"lr={self.learning_rate}, "
            f"beta={self.beta}, "
            f"lora_rank={self.lora_rank}, "
            f"batch={self.per_device_train_batch_size}, "
            f"num_generations={self.num_generations}, "
            f"steps={self.max_steps})"
        )


class ProductManager:
    """Manages task prompts for GRPO training.

    This class generates and manages task prompts that are used to generate
    completions for GRPO training. Each prompt represents a task the model
    should learn to complete (e.g., "Write a test for login functionality").

    The class supports:
    - Custom prompts via constructor or dynamic addition
    - Default Chinese QA prompts if none provided
    - Conversion to HuggingFace Dataset for GRPOTrainer

    Attributes:
        task_templates: List of user-provided prompt templates.
            If empty, uses _default_templates.
        num_generated: Counter for generate_task() calls.

    Example:
        >>> # Method 1: Pass prompts in constructor
        >>> pm = ProductManager(task_templates=[
        ...     "Write a function to sort a list",
        ...     "Implement binary search",
        ... ])
        >>>
        >>> # Method 2: Add prompts dynamically
        >>> pm = ProductManager()
        >>> pm.add_task_template("Your custom prompt here")
        >>>
        >>> # Generate a random task
        >>> task = pm.generate_task()
        >>> messages = pm.format_prompt(task)

    Usage - Provide custom prompts:
        from grpo_in_sandbox import ProductManager

        # Method 1: Pass prompts in constructor
        pm = ProductManager(task_templates=[
            "Write a function to sort a list",
            "Implement binary search",
        ])

        # Method 2: Add prompts dynamically
        pm = ProductManager()
        pm.add_task_template("Your custom prompt here")

    Usage with train():
        from grpo_in_sandbox import train, RLHFTrainingConfig, ProductManager

        pm = ProductManager(task_templates=["Your prompt 1", "Your prompt 2"])
        config = RLHFTrainingConfig(max_steps=10)
        results = train(config, product_manager=pm)
    """

    def __init__(self, task_templates: list[str] | None = None):
        """Initialize ProductManager.

        Args:
            task_templates: Optional list of custom prompt templates.
                          If None or empty, uses default Chinese QA templates.
        """
        self.task_templates: list[str] = task_templates if task_templates is not None else []
        self.num_generated = 0
        self._custom_system_prompt: str | None = None
        self._default_templates: list[str] = [
            "作为QA工程师，请测试以下功能：用户登录系统，包括正常登录、密码错误、账户锁定等场景。",
            "请进行回归测试：订单创建功能，验证库存扣减、支付流程、订单状态流转。",
            "执行API测试：用户管理接口，测试创建、查询、更新、删除用户的各项操作。",
            "请测试购物车功能：添加商品、修改数量、删除商品、结算流程。",
            "执行冒烟测试：检查系统核心功能的基本可用性。",
            "测试用户注册流程：验证表单验证、邮箱验证、验证码发送等功能。",
            "执行性能测试：模拟高并发场景，检查系统响应时间和稳定性。",
        ]

    def add_task_template(self, template: str) -> None:
        """Add a custom task prompt template."""
        self.task_templates.append(template)

    def add_task_templates(self, templates: list[str]) -> None:
        """Add multiple custom task prompt templates."""
        self.task_templates.extend(templates)

    @classmethod
    def from_file(cls, file_path: str) -> "ProductManager":
        """Create ProductManager from a file containing prompts."""
        with open(file_path, encoding="utf-8") as f:
            templates = [line.strip() for line in f if line.strip()]
        return cls(task_templates=templates)

    def generate_task(self) -> str:
        """Generate a random task prompt."""
        self.num_generated += 1
        templates = self.task_templates if self.task_templates else self._default_templates
        return random.choice(templates)

    def format_prompt(self, task: str, system_prompt: str | None = None) -> list[dict[str, str]]:
        """Format task as chat messages."""
        default_system = "你是一个专业的QA工程师，负责执行测试任务。"
        # Priority: parameter > _custom_system_prompt > default
        effective_system = system_prompt or self._custom_system_prompt or default_system
        return [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": f"任务：{task}"},
        ]

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set a custom system prompt."""
        self._custom_system_prompt = system_prompt

    @property
    def templates(self) -> list[str]:
        """Get all available templates."""
        return self.task_templates if self.task_templates else self._default_templates

    def __len__(self) -> int:
        """Return number of user-provided templates."""
        return len(self.task_templates)

    def to_hf_dataset(self):
        """Convert to HuggingFace dataset for GRPOTrainer.

        Returns:
            Dataset with 'prompt' and 'question' fields.
        """
        try:
            from datasets import Dataset
        except ImportError as e:
            raise ImportError("datasets library required. Install with: pip install datasets") from e

        templates = self.task_templates if self.task_templates else self._default_templates
        # Use from_list to create dataset from list of prompts (each prompt is a row)
        return Dataset.from_list([{"prompt": t} for t in templates])


class RewardModel:
    """Scores agent outputs for GRPO reward calculation.

    This reward model evaluates model completions and returns a score between 0 and 1.
    It uses two scoring methods:

    1. Keyword-based scoring (always active):
       - Checks for task-relevant keywords in response
       - rewards structure (numbered lists, steps)
       - rewards response length (indicates effort)

    2. Sandbox execution (if runtime provided):
       - Extracts test code from response
       - Executes tests in sandbox
       - Rewards passing tests with higher score

    Score weighting:
       - Base score (keyword): 60%
       - Sandbox score: 40%
       - Final score capped at 1.0

    Attributes:
        runtime: Optional sandbox runtime for executing test code.
            If None, only keyword-based scoring is used.

    Example:
        >>> from grpo_in_sandbox import RewardModel
        >>>
        >>> # Without sandbox (keyword only)
        >>> rm = RewardModel()
        >>> score = rm.score("Write tests for login", "1. Test normal login\\n2. Test wrong password")
        >>>
        >>> # With sandbox execution
        >>> rm = RewardModel(runtime=docker_runtime)
        >>> score = rm.score("Write tests", "def test_login(): ...")


    Args:
        runtime: Optional BaseRuntime instance for sandbox execution.
            Types: DockerRuntime, KaggleRuntime, LocalRuntime.
    """

    def __init__(self, runtime: Any = None):
        self.runtime = runtime

    def score(self, task: str, qa_output: str, trajectory: Any = None) -> float:
        """Score the agent output."""
        base_score = self._keyword_based_score(task, qa_output)
        if self.runtime is not None:
            sandbox_score = self._sandbox_score(qa_output)
            return min(0.6 * base_score + 0.4 * sandbox_score, 1.0)
        return base_score

    def _keyword_based_score(self, task: str, response: str) -> float:
        score = 0.0
        response_lower = response.lower()
        task_lower = task.lower()
        if "测试" in task_lower or "test" in task_lower:
            test_keywords = ["测试", "验证", "检查", "用例", "场景", "步骤", "test case"]
            score += min(sum(1 for kw in test_keywords if kw in response_lower) * 2, 20)
        if "登录" in task_lower and any(kw in response_lower for kw in ["正常登录", "密码错误", "username", "password"]):
            score += 15
        if "回归" in task_lower and any(kw in response_lower for kw in ["订单", "库存", "支付", "状态"]):
            score += 15
        if "API" in task_lower and any(kw in response_lower for kw in ["POST", "GET", "PUT", "DELETE", "接口", "endpoint"]):
            score += 15
        if len(response) > 100:
            score += 10
        if len(response) > 300:
            score += 10
        structure_keywords = ["1.", "2.", "3.", "第一", "第二", "-", "•", "步骤"]
        if any(kw in response for kw in structure_keywords):
            score += 10
        return min(score, 100.0) / 100.0

    def _sandbox_score(self, qa_output: str) -> float:
        try:
            if self.runtime is None:
                return 0.5
            if "def test_" not in qa_output and "import unittest" not in qa_output:
                return 0.5
            test_code = self._extract_test_code(qa_output)
            if not test_code:
                return 0.5

            # Use tempfile to avoid file path conflicts in parallel execution
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                test_file = f.name
                f.write(test_code)

            try:
                output, exit_code = self.runtime.run(f"python -m pytest {test_file} -v 2>&1 | head -50")
                if exit_code == 0:
                    return 1.0
                elif "error" in output.lower() or "fail" in output.lower():
                    return 0.3
                return 0.6
            finally:
                # Clean up temp file
                try:
                    import os
                    os.unlink(test_file)
                except OSError:
                    pass
        except Exception:
            return 0.5

    def _extract_test_code(self, qa_output: str) -> str:
        lines = qa_output.split("\n")
        import_lines = []
        test_lines = []
        in_test = False

        # First pass: collect import lines
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_lines.append(line)

        # Second pass: collect test functions/classes
        for line in lines:
            if "def test_" in line or "class Test" in line:
                in_test = True
            if in_test:
                test_lines.append(line)

        if test_lines:
            # Build code with imports first, then test code
            code_parts = []
            if import_lines:
                code_parts.extend(import_lines)
            code_parts.append("import unittest")
            code_parts.extend(test_lines)
            code = "\n".join(code_parts)
            if "unittest.main" not in code:
                code += "\n\nif __name__ == '__main__':\n    unittest.main()"
            return code
        return ""

    def close(self):
        if self.runtime:
            self.runtime.close()


class GRPOSandboxRewardFunc:
    """Reward function wrapper for GRPOTrainer that executes in sandbox.

    This wraps the RewardModel to match the GRPOTrainer reward_funcs interface:
    reward_funcs(completions: List[str], prompts: List[str]) -> List[float]
    """

    def __init__(self, reward_model: RewardModel, task_prefix: str = "任务："):
        self.reward_model = reward_model
        self.task_prefix = task_prefix

    def __call__(self, completions: list[str], prompts: list[str]) -> list[float]:
        """Compute rewards for completions.

        Args:
            completions: List of model-generated responses
            prompts: List of prompt strings (original tasks)

        Returns:
            List of reward scores (float between 0 and 1)
        """
        rewards = []
        for completion, prompt in zip(completions, prompts, strict=False):
            # Extract task from prompt (remove task_prefix if present)
            task = prompt
            if self.task_prefix in prompt:
                task = prompt.split(self.task_prefix)[-1].strip()
            reward = self.reward_model.score(task, completion)
            rewards.append(reward)
        return rewards


class AIJudge:
    """LLM-as-Judge: uses an AI model to score responses for GRPO training.

    This class calls a judge LLM (via litellm) to evaluate model-generated responses
    against configurable criteria. It returns structured scores (0.0-1.0) that can
    be used as reward signals for GRPO training.

    The judge prompt asks the LLM to rate the response on each criterion and provide
    a brief explanation. Scores are averaged across criteria and normalized.

    Attributes:
        llm_name: Model identifier for the judge LLM (e.g. "openai/gpt-4o-mini")
        criteria: List of scoring criteria (e.g. ["correctness", "clarity"])
        temperature: Sampling temperature for judge (low = consistent)
        base_url: Optional custom API endpoint
        system_prompt: System prompt template for the judge

    Example:
        >>> judge = AIJudge(llm_name="openai/gpt-4o-mini")
        >>> result = judge.score(
        ...     prompt="What is 2+2?",
        ...     response="The answer is 4.",
        ...     criteria=["correctness", "helpfulness"]
        ... )
        >>> result["score"]  # 0.92
        >>> result["scores"]  # {"correctness": 1.0, "helpfulness": 0.83}
        >>> result["explanation"]  # "The response correctly answers..."
    """

    DEFAULT_SYSTEM_PROMPT = """You are an expert evaluator. Your job is to score AI responses based on given criteria.
Score each criterion from 0.0 to 1.0, where:
- 1.0 = perfect
- 0.7 = good
- 0.5 = acceptable
- 0.3 = poor
- 0.0 = completely wrong or irrelevant

Be strict but fair. Provide a brief explanation for each score."""

    def __init__(
        self,
        llm_name: str = "openai/gpt-4o-mini",
        criteria: list[str] | None = None,
        temperature: float = 0.1,
        base_url: str | None = None,
    ):
        self.llm_name = llm_name
        self.criteria = criteria or ["correctness", "helpfulness", "clarity"]
        self.temperature = temperature
        self.base_url = base_url

    def _build_judge_prompt(
        self,
        prompt: str,
        response: str,
        criteria: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Build the judge prompt messages."""
        active_criteria = criteria or self.criteria
        criteria_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(active_criteria))

        user_content = f"""Please evaluate the following response.

## Original Prompt
{prompt}

## Response to Evaluate
{response}

## Scoring Criteria
{criteria_text}

## Instructions
Score each criterion from 0.0 to 1.0. Respond ONLY in JSON format:
{{
  "scores": {{"criterion_name": score, ...}},
  "explanation": "Brief explanation of scores",
  "overall_score": <average of all scores rounded to 2 decimals>
}}"""

        return [
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def score(
        self,
        prompt: str,
        response: str,
        criteria: list[str] | None = None,
    ) -> dict:
        """Score a response using the judge LLM.

        Args:
            prompt: The original prompt/question
            response: The model's response to evaluate
            criteria: Optional specific criteria (uses self.criteria if None)

        Returns:
            Dict with keys:
                - score: float 0.0-1.0 (average across criteria)
                - scores: dict of per-criterion scores
                - explanation: string explanation
        """
        import re

        import litellm

        messages = self._build_judge_prompt(prompt, response, criteria)

        kwargs = {
            "model": self.llm_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 512,
        }
        if self.base_url:
            kwargs["api_base"] = self.base_url

        try:
            result = litellm.completion(**kwargs)
            content = result.choices[0].message.content.strip()

            # Try to parse JSON response
            # First try direct JSON parse
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(1).strip())
                else:
                    # Try to find {...} in the content
                    brace_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if brace_match:
                        parsed = json.loads(brace_match.group(0))
                    else:
                        # Fallback: heuristic scoring
                        return self._heuristic_fallback(prompt, response, criteria)

            # Normalize and extract scores
            scores = parsed.get("scores", {})
            if not scores:
                # Try 'overall_score' directly
                overall = parsed.get("overall_score", 0.5)
                return {
                    "score": min(max(float(overall), 0.0), 1.0),
                    "scores": {"overall": min(max(float(overall), 0.0), 1.0)},
                    "explanation": parsed.get("explanation", ""),
                }

            # Average the per-criterion scores
            score_values = [min(max(float(v), 0.0), 1.0) for v in scores.values()]
            avg_score = sum(score_values) / len(score_values) if score_values else 0.5

            return {
                "score": round(avg_score, 4),
                "scores": {k: min(max(float(v), 0.0), 1.0) for k, v in scores.items()},
                "explanation": parsed.get("explanation", ""),
            }

        except Exception as e:
            # On any error, use fallback scoring
            logger = logging.getLogger(__name__)
            logger.warning(f"AIJudge error: {e}. Using fallback scoring.")
            return self._heuristic_fallback(prompt, response, criteria)

    def _heuristic_fallback(
        self,
        prompt: str,
        response: str,
        criteria: list[str] | None = None,
    ) -> dict:
        """Fallback scoring when judge LLM fails."""
        response_len = len(response)

        # Length-based score (longer = more effort)
        length_score = min(response_len / 500, 1.0) * 0.3

        # Structure score (has numbered lists, code blocks, etc.)
        structure_score = 0.0
        if any(c in response for c in ["1.", "2.", "3.", "- "]):
            structure_score += 0.2
        if "```" in response:
            structure_score += 0.2
        if len(response.split("\n")) > 3:
            structure_score += 0.1

        total = min(length_score + structure_score, 1.0)
        active_criteria = criteria or self.criteria

        return {
            "score": round(total, 4),
            "scores": {c: round(total, 4) for c in active_criteria},
            "explanation": "Fallback heuristic scoring (judge LLM unavailable).",
        }


class AIJudgeRewardModel:
    """Reward function wrapper for GRPOTrainer that uses an AI judge.

    Wraps AIJudge to match the GRPOTrainer reward_funcs interface:
    reward_funcs(completions: List[str], prompts: List[str]) -> List[float]

    Example:
        >>> judge = AIJudge(llm_name="openai/gpt-4o-mini")
        >>> reward_fn = AIJudgeRewardModel(judge)
        >>> scores = reward_fn(["response A", "response B"], ["prompt X", "prompt Y"])
    """

    def __init__(
        self,
        judge: AIJudge,
        criteria: list[str] | None = None,
    ):
        self.judge = judge
        self.criteria = criteria

    def __call__(self, completions: list[str], prompts: list[str]) -> list[float]:
        rewards = []
        for completion, prompt in zip(completions, prompts, strict=False):
            result = self.judge.score(prompt, completion, criteria=self.criteria)
            rewards.append(result["score"])
        return rewards


class SelfPlayGRPO:
    """Self-Play GRPO orchestrator: generate → judge → train iterative loop.

    This class orchestrates a complete self-play training loop where:
    1. The model generates responses to prompts (generate phase)
    2. An AI judge scores those responses (judge phase)
    3. GRPO training updates the model using judge scores as rewards (train phase)

    This enables RL training WITHOUT sandbox execution — the reward signal
    comes entirely from AI evaluation.

    Example:
        >>> config = RLHFTrainingConfig(
        ...     mode=AITrainerMode.AI_JUDGE,
        ...     model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        ...     judge_llm_name="openai/gpt-4o-mini",
        ...     judge_criteria=["correctness", "clarity", "helpfulness"],
        ...     max_steps=50,
        ... )
        >>> sp = SelfPlayGRPO(config)
        >>> results = sp.run()
    """

    def __init__(
        self,
        config: RLHFTrainingConfig,
        product_manager: ProductManager | None = None,
        judge: AIJudge | None = None,
    ):
        self.config = config
        self.pm = product_manager or ProductManager()

        # Create AI judge from config if not provided
        if judge is not None:
            self.judge = judge
        else:
            self.judge = AIJudge(
                llm_name=config.judge_llm_name,
                criteria=config.judge_criteria,
                temperature=config.judge_temperature,
                base_url=config.judge_base_url,
            )

        self.reward_fn = AIJudgeRewardModel(self.judge, criteria=config.judge_criteria)
        self.log = _setup_logger("SelfPlayGRPO")

    def run(self) -> dict[str, Any]:
        """Run the self-play GRPO training loop.

        Returns:
            Dict with training metrics including iteration history
        """
        self.log.info("=" * 60)
        self.log.info("Starting Self-Play GRPO Training")
        self.log.info(f"Mode: AI_JUDGE | Judge: {self.config.judge_llm_name}")
        self.log.info(f"Criteria: {self.config.judge_criteria or self.judge.criteria}")
        self.log.info(f"Model: {self.config.model_name_or_path}")
        self.log.info("=" * 60)

        # Delegate to the existing train() function with AI judge mode
        # The train() function handles model loading, dataset prep, etc.
        results = train(
            config=self.config,
            product_manager=self.pm,
            ai_judge=self.judge,  # Pass AI judge for reward computation
        )

        self.log.info("Self-Play GRPO Training Complete")
        results["mode"] = "ai_judge"
        results["judge_llm"] = self.config.judge_llm_name

        return results


class CodeExecutor:
    """Executes code in local environment."""

    def __init__(self, working_dir: str = "/tmp/testbed"):
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)

    def run(self, code: str, timeout: int = 30) -> tuple[str, int]:
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_path = f.name
        try:
            result = subprocess.run(['python', temp_path], capture_output=True, text=True, timeout=timeout)
            return result.stdout + result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return f"Execution timed out ({timeout}s)", -1
        except Exception as e:
            return f"Error: {repr(e)}", -1
        finally:
            with contextlib.suppress(OSError):
                os.unlink(temp_path)


def _create_reward_func_from_rm(
    reward_model: RewardModel,
) -> Callable[[list[str], list[str]], list[float]]:
    """Create a reward function compatible with GRPOTrainer from a RewardModel instance."""

    def reward_func(completions: list[str], prompts: list[str]) -> list[float]:
        rewards = []
        for completion, prompt in zip(completions, prompts, strict=False):
            task = prompt.strip()
            reward = reward_model.score(task, completion)
            rewards.append(reward)
        return rewards

    return reward_func


def _load_model_and_tokenizer(config: RLHFTrainingConfig):
    """Load model and tokenizer with Unsloth optimizations.

    Uses FastLanguageModel for efficient loading and optionally enables
    vLLM fast inference for accelerated generation during RL training.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = _setup_logger(__name__)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
    )

    # Ensure tokenizer has pad_token (required for training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure padding side is set for training
    tokenizer.padding_side = "left"

    # Load model with Unsloth
    model_kwargs = {
        "model": config.model_name_or_path,
        "max_seq_length": config.max_seq_length,
        "load_in_4bit": True,
        "fast_inference": config.fast_inference,
    }

    # Only add gpu_memory_utilization if using fast_inference (vLLM)
    if config.fast_inference:
        model_kwargs["gpu_memory_utilization"] = config.vllm_gpu_memory_utilization

    model, _ = FastLanguageModel.from_pretrained(**model_kwargs)

    # Apply LoRA PEFT adapter
    lora_kwargs = {
        "model": model,
        "r": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "use_gradient_checkpointing": "unsloth",  # Unsloth's long-context fine-tuning
        "random_state": config.seed,
    }

    if config.target_modules:
        lora_kwargs["target_modules"] = config.target_modules
    else:
        # Auto-detect target modules (common for most models)
        lora_kwargs["target_modules"] = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    model = FastLanguageModel.get_peft_model(**lora_kwargs)
    model = model.to(device)

    log.info(f"Model loaded with LoRA (rank={config.lora_rank}) on {device}")
    if config.fast_inference:
        log.info("vLLM fast inference enabled")

    return model, tokenizer, device


def _create_grpo_config(config: RLHFTrainingConfig):
    """Create TRL GRPOConfig from RLHFTrainingConfig."""
    try:
        from trl import GRPOConfig
    except ImportError as e:
        raise ImportError("trl library required. Install with: pip install trl") from e

    max_prompt_length = config.max_seq_length // 2
    max_completion_length = config.max_seq_length - max_prompt_length

    grpo_config = GRPOConfig(
        # Output
        output_dir=config.output_dir,
        seed=config.seed,

        # Optimization
        learning_rate=config.learning_rate,
        beta=config.beta,
        max_grad_norm=config.max_grad_norm,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        # Generation
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        max_completion_length=config.max_completion_length or max_completion_length,
        num_generations=config.num_generations,

        # Batch
        per_device_train_batch_size=config.per_device_train_batch_size,

        # Training limits
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,

        # vLLM
        use_vllm=config.use_vllm,
        vllm_mode="colocate" if config.use_vllm else None,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,

        # Logging
        logging_steps=1,
        log_completions=True,

        # Save
        save_steps=config.max_steps // 10 or 10,

        # Misc
        report_to="none",
    )

    return grpo_config


def train(
    config: RLHFTrainingConfig,
    product_manager: ProductManager | None = None,
    reward_model: RewardModel | None = None,
    ai_judge: AIJudge | None = None,
) -> dict[str, Any]:
    """Run GRPO training using Unsloth's optimized GRPOTrainer.

    This function uses TRL's GRPOTrainer with Unsloth's PatchFastRL optimizations
    for accelerated RL training with vLLM inference.

    Args:
        config: RLHFTrainingConfig with model and training parameters
        product_manager: Optional ProductManager for custom task prompts
        reward_model: Optional RewardModel for sandbox-based reward scoring
        ai_judge: Optional AIJudge for AI-judge-based reward scoring

    Returns:
        Dict with training metrics and results
    """
    log = _setup_logger(__name__)

    # Log training configuration


    random.seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    # Step 1: Patch TRL for Unsloth GRPO optimizations
    log.info("Applying Unsloth optimizations to TRL...")
    try:
        from unsloth import FastLanguageModel, PatchFastRL
        PatchFastRL(algorithm="grpo", FastLanguageModel=FastLanguageModel)
        log.info("Unsloth PatchFastRL applied successfully")
    except ImportError as e:
        log.warning(f"Could not apply PatchFastRL: {e}. Continuing with base Unsloth.")

    # Step 2: Load model and tokenizer with LoRA
    model, tokenizer, device = _load_model_and_tokenizer(config)
    log.info(f"Model loaded on device: {device}")

    # Step 3: Prepare dataset for GRPOTrainer
    pm = product_manager or ProductManager()
    rm = reward_model or RewardModel()

    try:
        from datasets import Dataset
    except ImportError as e:
        raise ImportError("datasets library required. Install with: pip install datasets") from e

    templates = pm.task_templates if pm.task_templates else pm._default_templates
    # GRPOTrainer needs 'prompt' field
    dataset = Dataset.from_dict({"prompt": templates})

    # For GRPOTrainer, we need to duplicate prompts for num_generations
    # The trainer handles generating multiple completions per prompt
    all_prompts = []
    for _ in range(config.num_generations):
        all_prompts.extend(templates)
    dataset = Dataset.from_dict({"prompt": all_prompts})

    log.info(f"Dataset prepared: {len(dataset)} samples ({config.num_generations} generations x {len(templates)} prompts)")

    # Step 4: Create reward function (sandbox or AI judge mode)
    if config.mode == AITrainerMode.AI_JUDGE or ai_judge is not None:
        # AI Judge mode
        effective_judge = ai_judge or AIJudge(
            llm_name=config.judge_llm_name,
            criteria=config.judge_criteria,
            temperature=config.judge_temperature,
            base_url=config.judge_base_url,
        )
        reward_func = AIJudgeRewardModel(effective_judge, criteria=config.judge_criteria)
        log.info(f"Using AI Judge reward: {config.judge_llm_name}")
        log.info(f"Judge criteria: {config.judge_criteria or effective_judge.criteria}")
    else:
        # Sandbox (default) mode
        rm = reward_model or RewardModel()
        reward_func = _create_reward_func_from_rm(rm)  # type: ignore[assignment]
        log.info("Using sandbox-based reward model")

    # Step 5: Create GRPOConfig
    grpo_config = _create_grpo_config(config)

    # Step 6: Initialize GRPOTrainer
    log.info("Initializing GRPOTrainer...")
    try:
        from trl import GRPOTrainer
    except ImportError as e:
        raise ImportError("trl library required. Install with: pip install trl") from e

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_func,
    )

    # Step 7: Prepare model for training
    model = model.for_training()
    log.info("Model prepared for training (for_training mode)")

    # Step 8: Train!
    log.info("Starting GRPO training...")
    try:
        trainer.train()
    except Exception as e:
        log.error(f"Training error: {e}")
        raise

    # Step 9: Save model
    final_checkpoint = os.path.join(config.output_dir, "final_model")
    log.info(f"Saving final model to {final_checkpoint}")
    trainer.save_model(final_checkpoint)
    tokenizer.save_pretrained(final_checkpoint)

    # Step 10: Return results
    log.info("=" * 60)
    log.info("Training Complete")
    log.info("=" * 60)

    results = {
        "config": {
            "model_name_or_path": config.model_name_or_path,
            "lora_rank": config.lora_rank,
            "learning_rate": config.learning_rate,
            "beta": config.beta,
            "num_train_epochs": config.num_train_epochs,
            "max_steps": config.max_steps,
        },
        "output_dir": config.output_dir,
        "final_checkpoint": final_checkpoint,
    }

    # Save results
    results_path = os.path.join(config.output_dir, "training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"Results saved to {results_path}")

    return results


# Keep legacy classes for backward compatibility
class ReferenceModel:
    """Reference model for KL divergence constraint (kept for compatibility)."""

    def __init__(self, model, tokenizer):
        import torch
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.device = next(model.parameters()).device
        self.reference_params = {}
        for name, param in model.named_parameters():
            self.reference_params[name] = param.data.clone()

    def compute_kl_divergence(self, input_ids, attention_mask):
        """Compute KL divergence between policy and reference model."""
        torch = self.torch
        with torch.no_grad():
            ref_logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        policy_logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        policy_probs = torch.nn.functional.softmax(policy_logits, dim=-1)
        kl = torch.nn.functional.kl_div(ref_log_probs, policy_probs, reduction='batchmean', log_target=True)
        self._last_kl_float = kl.item()
        return kl

    def reset(self):
        """Reset reference model to original parameters."""
        for name, param in self.model.named_parameters():
            if name in self.reference_params:
                param.data = self.reference_params[name].to(param.device)


class ReplayBuffer:
    """Experience replay buffer (kept for compatibility)."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: list[dict[str, Any]] = []
        self.advantages: list[float] = []

    def add(self, trajectory: dict[str, Any], reward: float):
        """Add a trajectory to the buffer."""
        self.buffer.append(trajectory)
        self.advantages.append(reward)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
            self.advantages.pop(0)

    def compute_advantages(self, baseline: float | None = None) -> list[float]:
        """Compute advantages using mean rewards as baseline."""
        if not self.advantages:
            return []
        if baseline is None:
            baseline = sum(self.advantages) / len(self.advantages)
        advantages = [r - baseline for r in self.advantages]
        mean_adv = sum(advantages) / len(advantages)
        std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5
        if std_adv > 1e-8:
            advantages = [(a - mean_adv) / std_adv for a in advantages]
        return advantages

    def sample(self, batch_size: int) -> list[dict[str, Any]]:
        """Sample a batch from the buffer."""
        if len(self.buffer) <= batch_size:
            return self.buffer.copy()
        return random.sample(self.buffer, batch_size)

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.advantages.clear()

    def __len__(self):
        return len(self.buffer)


class GRPOOptimizer:
    """GRPO Optimizer (legacy, kept for backward compatibility).

    Note: New code should use train() with GRPOTrainer for Unsloth-optimized
    GRPO training with vLLM acceleration and proper reference model support.
    """

    def __init__(self, model, tokenizer, beta: float = 0.01, lr: float = 1e-5, max_grad_norm: float = 1.0):
        import torch
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.device = next(model.parameters()).device
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        self.reference_model = ReferenceModel(model, tokenizer)

    def compute_grpo_loss(self, input_ids, attention_mask, response_ids, advantages: list[float]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute GRPO loss."""
        torch = self.torch
        f = torch.nn.functional
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=response_ids)
        policy_logits = outputs.logits
        response_logits = policy_logits[:, :-1, :]
        response_labels = response_ids[:, 1:]
        log_probs = f.log_softmax(response_logits, dim=-1)
        selected_log_probs = log_probs.gather(2, response_labels.unsqueeze(2)).squeeze(2)
        response_mask = attention_mask[:, 1:].float()
        policy_log_probs = (selected_log_probs * response_mask).sum(dim=-1) / (response_mask.sum(dim=-1) + 1e-8)
        advantages_tensor = torch.tensor(advantages, device=self.device)
        grpo_loss = -(policy_log_probs * advantages_tensor).mean()
        kl_div = self.reference_model.compute_kl_divergence(input_ids, attention_mask)
        kl_penalty = self.beta * kl_div
        total_loss = grpo_loss + kl_penalty
        kl_float = getattr(self.reference_model, '_last_kl_float', 0.0)
        metrics = {
            "grpo_loss": grpo_loss.item(),
            "kl_div": kl_float,
            "kl_penalty": kl_penalty.item() if hasattr(kl_penalty, 'item') else kl_penalty,
            "total_loss": total_loss.item()
        }
        return total_loss, metrics

    def step(self, input_ids, attention_mask, response_ids, advantages: list[float]) -> dict[str, float]:
        torch = self.torch
        loss, metrics = self.compute_grpo_loss(input_ids, attention_mask, response_ids, advantages)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.reference_model.reset()
        return metrics
