"""
grpo-in-sandbox training module

this module provides a complete implementation for training language models using grpo
(group relative policy optimization) within a code sandbox environment.

key components:
- rlhftrainingconfig: configuration for training
- productmanager: task prompt generation
- rewardmodel: sandbox-based reward scoring
- grpotrainer (via trl): unsloth-optimized grpo training with vllm acceleration

usage:
    from grpo_in_sandbox import train, rlhftrainingconfig

    config = rlhftrainingconfig(
        model_name_or_path="qwen/qwen2.5-0.5b-instruct",
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
from collections.abc import callable
from dataclasses import dataclass
from enum import enum
from typing import any

import torch
from transformers import autotokenizer
from unsloth import fastlanguagemodel

# module-level logger
train_logger = logging.getlogger(__name__)


def _setup_logger(name: str, level: int = logging.info) -> logging.logger:
    """configure and return a logger with consistent formatting."""
    log = logging.getlogger(name)
    if not log.handlers:
        handler = logging.streamhandler()
        handler.setformatter(logging.formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%h:%m:%s",
        ))
        log.addhandler(handler)
        log.setlevel(level)
    return log


class aitrainermode(str, enum):
    """ai judge training mode for self-play grpo.

    attributes:
        sandbox: use sandbox execution for reward scoring (default)
        ai_judge: use llm-as-judge for reward scoring
    """

    sandbox = "sandbox"
    ai_judge = "ai_judge"


"""
grpo-in-sandbox 训练模块

该模块提供了在代码沙箱环境中使用 GRPO（组相对策略优化）训练语言模型的完整实现。

关键组件：
- rlhftrainingconfig：训练配置
- productmanager：任务提示词生成
- rewardmodel：基于沙箱的奖励评分
- grpotrainer（通过 trl）：利用 unsloth 优化和 vLLM 加速的 GRPO 训练

用法：
    from grpo_in_sandbox import train, rlhftrainingconfig

    config = rlhftrainingconfig(
        model_name_or_path="qwen/qwen2.5-0.5b-instruct",
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
from collections.abc import callable
from dataclasses import dataclass
from enum import enum
from typing import any

import torch
from transformers import autotokenizer
from unsloth import fastlanguagemodel

# 模块级日志记录器
train_logger = logging.getlogger(__name__)


def _setup_logger(name: str, level: int = logging.info) -> logging.logger:
    """配置并返回一个格式一致的日志记录器。"""
    log = logging.getlogger(name)
    if not log.handlers:
        handler = logging.streamhandler()
        handler.setformatter(logging.formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%h:%m:%s",
        ))
        log.addhandler(handler)
        log.setlevel(level)
    return log


class aitrainermode(str, enum):
    """用于自我博弈 GRPO 的 AI 评判训练模式。

    属性：
        sandbox：使用沙箱执行进行奖励评分（默认）
        ai_judge：使用 LLM 作为评判进行奖励评分
    """

    sandbox = "sandbox"
    ai_judge = "ai_judge"


from typing import Optional, List
from enum import Enum

class AITrainerMode(Enum):
    SANDBOX = "sandbox"
    AI_JUDGE = "ai_judge"

class RLHFTrainingConfig:
    def __init__(
        self,
        model_name_or_path: str = "./model",
        # LoRA 设置
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        # 优化器设置
        learning_rate: float = 5e-6,
        beta: float = 0.001,
        max_grad_norm: float = 0.1,
        # 生成设置
        num_train_epochs: int = 3,
        max_seq_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_completion_length: int = 1024,
        # 批次设置
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        # GRPO 特定参数
        num_generations: int = 4,
        # 训练限制
        max_steps: int = 100,
        max_token_limit: int = 65536,
        max_tokens_per_call: int = 65536,
        # 运行时设置
        docker_image: str = "lcyisgay/llm-in-sandbox:v0.1",
        output_dir: str = "./output",
        seed: int = 42,
        # 早停设置
        patience: int = 3,
        min_improvement: float = 0.01,
        # vLLM 设置
        use_vllm: bool = True,
        vllm_gpu_memory_utilization: float = 0.7,
        fast_inference: bool = True,
        # AI 评判模式
        mode: AITrainerMode = AITrainerMode.SANDBOX,
        judge_llm_name: str = "openai/gpt-4o-mini",
        judge_criteria: Optional[List[str]] = None,
        judge_temperature: float = 0.1,
        judge_base_url: Optional[str] = None,
    ):
        '''
        此部分使用config.xxx更改参数
        例如更改model_name_or_path 使用 config.model_name_or_path
        '''
        self.model_name_or_path = model_name_or_path
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.learning_rate = learning_rate
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.num_train_epochs = num_train_epochs
        self.max_seq_length = max_seq_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_completion_length = max_completion_length
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_generations = num_generations
        self.max_steps = max_steps
        self.max_token_limit = max_token_limit
        self.max_tokens_per_call = max_tokens_per_call
        self.docker_image = docker_image
        self.output_dir = output_dir
        self.seed = seed
        self.patience = patience
        self.min_improvement = min_improvement
        self.use_vllm = use_vllm
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.fast_inference = fast_inference
        self.mode = mode
        self.judge_llm_name = judge_llm_name
        self.judge_criteria = judge_criteria
        self.judge_temperature = judge_temperature
        self.judge_base_url = judge_base_url
        
    def __repr__(self) -> str:
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

    def update(self, **kwargs) -> "RLHFTrainingConfig":
        """Update configuration parameters via kwargs.

        Usage:
            config.update(learning_rate=1e-5, max_steps=200)
            config.update(temperature=0.8, use_vllm=False)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    def set_model(self, model_name_or_path: str) -> "RLHFTrainingConfig":
        """Set the model name or path."""
        self.model_name_or_path = model_name_or_path
        return self

    def set_lora(self, rank: int = None, alpha: int = None, dropout: float = None) -> "RLHFTrainingConfig":
        """Update LoRA parameters."""
        if rank is not None:
            self.lora_rank = rank
        if alpha is not None:
            self.lora_alpha = alpha
        if dropout is not None:
            self.lora_dropout = dropout
        return self

    def set_optimizer(self, learning_rate: float = None, beta: float = None, max_grad_norm: float = None) -> "RLHFTrainingConfig":
        """Update optimizer parameters."""
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if beta is not None:
            self.beta = beta
        if max_grad_norm is not None:
            self.max_grad_norm = max_grad_norm
        return self

    def set_generation(self, temperature: float = None, top_p: float = None, top_k: int = None, max_completion_length: int = None) -> "RLHFTrainingConfig":
        """Update generation parameters."""
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if top_k is not None:
            self.top_k = top_k
        if max_completion_length is not None:
            self.max_completion_length = max_completion_length
        return self

    def set_batch(self, per_device_train_batch_size: int = None, gradient_accumulation_steps: int = None) -> "RLHFTrainingConfig":
        """Update batch settings."""
        if per_device_train_batch_size is not None:
            self.per_device_train_batch_size = per_device_train_batch_size
        if gradient_accumulation_steps is not None:
            self.gradient_accumulation_steps = gradient_accumulation_steps
        return self

    def set_training(self, max_steps: int = None, max_seq_length: int = None, num_train_epochs: int = None) -> "RLHFTrainingConfig":
        """Update training parameters."""
        if max_steps is not None:
            self.max_steps = max_steps
        if max_seq_length is not None:
            self.max_seq_length = max_seq_length
        if num_train_epochs is not None:
            self.num_train_epochs = num_train_epochs
        return self

    def set_runtime(self, docker_image: str = None, use_vllm: bool = None, vllm_gpu_memory_utilization: float = None) -> "RLHFTrainingConfig":
        """Update runtime settings."""
        if docker_image is not None:
            self.docker_image = docker_image
        if use_vllm is not None:
            self.use_vllm = use_vllm
        if vllm_gpu_memory_utilization is not None:
            self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        return self

    def set_judge(self, llm_name: str = None, temperature: float = None, base_url: str = None) -> "RLHFTrainingConfig":
        """Update AI judge settings."""
        if llm_name is not None:
            self.judge_llm_name = llm_name
        if temperature is not None:
            self.judge_temperature = temperature
        if base_url is not None:
            self.judge_base_url = base_url
        return self

    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        return {
            "model_name_or_path": self.model_name_or_path,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "learning_rate": self.learning_rate,
            "beta": self.beta,
            "max_grad_norm": self.max_grad_norm,
            "num_train_epochs": self.num_train_epochs,
            "max_seq_length": self.max_seq_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_completion_length": self.max_completion_length,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_generations": self.num_generations,
            "max_steps": self.max_steps,
            "max_token_limit": self.max_token_limit,
            "max_tokens_per_call": self.max_tokens_per_call,
            "docker_image": self.docker_image,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "patience": self.patience,
            "min_improvement": self.min_improvement,
            "use_vllm": self.use_vllm,
            "vllm_gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "fast_inference": self.fast_inference,
            "mode": self.mode.value,
            "judge_llm_name": self.judge_llm_name,
            "judge_criteria": self.judge_criteria,
            "judge_temperature": self.judge_temperature,
            "judge_base_url": self.judge_base_url,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RLHFTrainingConfig":
        """Create configuration from dictionary."""
        # Extract mode if present (handle both str and enum)
        mode = config_dict.pop("mode", None)
        if mode is not None and isinstance(mode, str):
            mode = AITrainerMode(mode)

        # Handle Optional fields that might be None
        target_modules = config_dict.pop("target_modules", None)
        judge_criteria = config_dict.pop("judge_criteria", None)
        judge_base_url = config_dict.pop("judge_base_url", None)

        # Filter to only known params
        known_params = {
            "model_name_or_path": "./model",
            "lora_rank": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": None,
            "learning_rate": 5e-6,
            "beta": 0.001,
            "max_grad_norm": 0.1,
            "num_train_epochs": 3,
            "max_seq_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_completion_length": 1024,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_generations": 4,
            "max_steps": 100,
            "max_token_limit": 65536,
            "max_tokens_per_call": 65536,
            "docker_image": "lcyisgay/llm-in-sandbox:v0.1",
            "output_dir": "./output",
            "seed": 42,
            "patience": 3,
            "min_improvement": 0.01,
            "use_vllm": True,
            "vllm_gpu_memory_utilization": 0.7,
            "fast_inference": True,
            "mode": AITrainerMode.SANDBOX,
            "judge_llm_name": "openai/gpt-4o-mini",
            "judge_criteria": None,
            "judge_temperature": 0.1,
            "judge_base_url": None,
        }

        # Override with provided values
        known_params.update(config_dict)

        # Ensure required fields are present
        if "model_name_or_path" not in known_params:
            known_params["model_name_or_path"] = "./model"
        if "mode" not in known_params or known_params["mode"] is None:
            known_params["mode"] = AITrainerMode.SANDBOX

        return cls(**known_params)

class productmanager:
    """manages task prompts for grpo training.

    this class generates and manages task prompts that are used to generate
    completions for grpo training. each prompt represents a task the model
    should learn to complete (e.g., "write a test for login functionality").

    the class supports:
    - custom prompts via constructor or dynamic addition
    - default chinese qa prompts if none provided
    - conversion to huggingface dataset for grpotrainer

    attributes:
        task_templates: list of user-provided prompt templates.
            if empty, uses _default_templates.
        num_generated: counter for generate_task() calls.

    example:
        >>> # method 1: pass prompts in constructor
        >>> pm = productmanager(task_templates=[
        ...     "write a function to sort a list",
        ...     "implement binary search",
        ... ])
        >>>
        >>> # method 2: add prompts dynamically
        >>> pm = productmanager()
        >>> pm.add_task_template("your custom prompt here")
        >>>
        >>> # generate a random task
        >>> task = pm.generate_task()
        >>> messages = pm.format_prompt(task)

    usage - provide custom prompts:
        from grpo_in_sandbox import productmanager

        # method 1: pass prompts in constructor
        pm = productmanager(task_templates=[
            "write a function to sort a list",
            "implement binary search",
        ])

        # method 2: add prompts dynamically
        pm = productmanager()
        pm.add_task_template("your custom prompt here")

    usage with train():
        from grpo_in_sandbox import train, rlhftrainingconfig, productmanager

        pm = productmanager(task_templates=["your prompt 1", "your prompt 2"])
        config = rlhftrainingconfig(max_steps=10)
        results = train(config, product_manager=pm)
    """

    def __init__(self, task_templates: list[str] | none = none):
        """initialize productmanager.

        args:
            task_templates: optional list of custom prompt templates.
                          if none or empty, uses default chinese qa templates.
        """
        self.task_templates: list[str] = task_templates if task_templates is not none else []
        self.num_generated = 0
        self._custom_system_prompt: str | none = none
        self._default_templates: list[str] = [
            "作为qa工程师，请测试以下功能：用户登录系统，包括正常登录、密码错误、账户锁定等场景。",
            "请进行回归测试：订单创建功能，验证库存扣减、支付流程、订单状态流转。",
            "执行api测试：用户管理接口，测试创建、查询、更新、删除用户的各项操作。",
            "请测试购物车功能：添加商品、修改数量、删除商品、结算流程。",
            "执行冒烟测试：检查系统核心功能的基本可用性。",
            "测试用户注册流程：验证表单验证、邮箱验证、验证码发送等功能。",
            "执行性能测试：模拟高并发场景，检查系统响应时间和稳定性。",
        ]

    def add_task_template(self, template: str) -> none:
        """add a custom task prompt template."""
        self.task_templates.append(template)

    def add_task_templates(self, templates: list[str]) -> none:
        """add multiple custom task prompt templates."""
        self.task_templates.extend(templates)

    @classmethod
    def from_file(cls, file_path: str) -> "productmanager":
        """create productmanager from a file containing prompts."""
        with open(file_path, encoding="utf-8") as f:
            templates = [line.strip() for line in f if line.strip()]
        return cls(task_templates=templates)

    def generate_task(self) -> str:
        """generate a random task prompt."""
        self.num_generated += 1
        templates = self.task_templates if self.task_templates else self._default_templates
        return random.choice(templates)

    def format_prompt(self, task: str, system_prompt: str | none = none) -> list[dict[str, str]]:
        """format task as chat messages."""
        default_system = "你是一个专业的qa工程师，负责执行测试任务。"
        # priority: parameter > _custom_system_prompt > default
        effective_system = system_prompt or self._custom_system_prompt or default_system
        return [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": f"任务：{task}"},
        ]

    def set_system_prompt(self, system_prompt: str) -> none:
        """set a custom system prompt."""
        self._custom_system_prompt = system_prompt

    @property
    def templates(self) -> list[str]:
        """get all available templates."""
        return self.task_templates if self.task_templates else self._default_templates

    def __len__(self) -> int:
        """return number of user-provided templates."""
        return len(self.task_templates)

    def to_hf_dataset(self):
        """convert to huggingface dataset for grpotrainer.

        returns:
            dataset with 'prompt' and 'question' fields.
        """
        try:
            from datasets import dataset
        except importerror as e:
            raise importerror("datasets library required. install with: pip install datasets") from e

        templates = self.task_templates if self.task_templates else self._default_templates
        # use from_list to create dataset from list of prompts (each prompt is a row)
        return dataset.from_list([{"prompt": t} for t in templates])


class rewardmodel:
    """scores agent outputs for grpo reward calculation.

    this reward model evaluates model completions and returns a score between 0 and 1.
    it uses two scoring methods:

    1. keyword-based scoring (always active):
       - checks for task-relevant keywords in response
       - rewards structure (numbered lists, steps)
       - rewards response length (indicates effort)

    2. sandbox execution (if runtime provided):
       - extracts test code from response
       - executes tests in sandbox
       - rewards passing tests with higher score

    score weighting:
       - base score (keyword): 60%
       - sandbox score: 40%
       - final score capped at 1.0

    attributes:
        runtime: optional sandbox runtime for executing test code.
            if none, only keyword-based scoring is used.

    example:
        >>> from grpo_in_sandbox import rewardmodel
        >>>
        >>> # without sandbox (keyword only)
        >>> rm = rewardmodel()
        >>> score = rm.score("write tests for login", "1. test normal login\\n2. test wrong password")
        >>>
        >>> # with sandbox execution
        >>> rm = rewardmodel(runtime=docker_runtime)
        >>> score = rm.score("write tests", "def test_login(): ...")


    args:
        runtime: optional baseruntime instance for sandbox execution.
            types: dockerruntime, kaggleruntime, localruntime.
    """

    def __init__(self, runtime: any = none):
        self.runtime = runtime

    def score(self, task: str, qa_output: str, trajectory: any = none) -> float:
        """score the agent output."""
        base_score = self._keyword_based_score(task, qa_output)
        if self.runtime is not none:
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
        if "api" in task_lower and any(kw in response_lower for kw in ["post", "get", "put", "delete", "接口", "endpoint"]):
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
            if self.runtime is none:
                return 0.5
            if "def test_" not in qa_output and "import unittest" not in qa_output:
                return 0.5
            test_code = self._extract_test_code(qa_output)
            if not test_code:
                return 0.5

            # use tempfile to avoid file path conflicts in parallel execution
            with tempfile.namedtemporaryfile(mode='w', suffix='.py', delete=false, encoding='utf-8') as f:
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
                # clean up temp file
                try:
                    import os
                    os.unlink(test_file)
                except oserror:
                    pass
        except exception:
            return 0.5

    def _extract_test_code(self, qa_output: str) -> str:
        lines = qa_output.split("\n")
        import_lines = []
        test_lines = []
        in_test = false

        # first pass: collect import lines
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_lines.append(line)

        # second pass: collect test functions/classes
        for line in lines:
            if "def test_" in line or "class test" in line:
                in_test = true
            if in_test:
                test_lines.append(line)

        if test_lines:
            # build code with imports first, then test code
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


class grposandboxrewardfunc:
    """reward function wrapper for grpotrainer that executes in sandbox.

    this wraps the rewardmodel to match the grpotrainer reward_funcs interface:
    reward_funcs(completions: list[str], prompts: list[str]) -> list[float]
    """

    def __init__(self, reward_model: rewardmodel, task_prefix: str = "任务："):
        self.reward_model = reward_model
        self.task_prefix = task_prefix

    def __call__(self, completions: list[str], prompts: list[str]) -> list[float]:
        """compute rewards for completions.

        args:
            completions: list of model-generated responses
            prompts: list of prompt strings (original tasks)

        returns:
            list of reward scores (float between 0 and 1)
        """
        rewards = []
        for completion, prompt in zip(completions, prompts, strict=false):
            # extract task from prompt (remove task_prefix if present)
            task = prompt
            if self.task_prefix in prompt:
                task = prompt.split(self.task_prefix)[-1].strip()
            reward = self.reward_model.score(task, completion)
            rewards.append(reward)
        return rewards


class aijudge:
    """llm-as-judge: uses an ai model to score responses for grpo training.

    this class calls a judge llm (via litellm) to evaluate model-generated responses
    against configurable criteria. it returns structured scores (0.0-1.0) that can
    be used as reward signals for grpo training.

    the judge prompt asks the llm to rate the response on each criterion and provide
    a brief explanation. scores are averaged across criteria and normalized.

    attributes:
        llm_name: model identifier for the judge llm (e.g. "openai/gpt-4o-mini")
        criteria: list of scoring criteria (e.g. ["correctness", "clarity"])
        temperature: sampling temperature for judge (low = consistent)
        base_url: optional custom api endpoint
        system_prompt: system prompt template for the judge

    example:
        >>> judge = aijudge(llm_name="openai/gpt-4o-mini")
        >>> result = judge.score(
        ...     prompt="what is 2+2?",
        ...     response="the answer is 4.",
        ...     criteria=["correctness", "helpfulness"]
        ... )
        >>> result["score"]  # 0.92
        >>> result["scores"]  # {"correctness": 1.0, "helpfulness": 0.83}
        >>> result["explanation"]  # "the response correctly answers..."
    """

    default_system_prompt = """you are an expert evaluator. your job is to score ai responses based on given criteria.
score each criterion from 0.0 to 1.0, where:
- 1.0 = perfect
- 0.7 = good
- 0.5 = acceptable
- 0.3 = poor
- 0.0 = completely wrong or irrelevant

be strict but fair. provide a brief explanation for each score."""

    def __init__(
        self,
        llm_name: str = "openai/gpt-4o-mini",
        criteria: list[str] | none = none,
        temperature: float = 0.1,
        base_url: str | none = none,
    ):
        self.llm_name = llm_name
        self.criteria = criteria or ["correctness", "helpfulness", "clarity"]
        self.temperature = temperature
        self.base_url = base_url

    def _build_judge_prompt(
        self,
        prompt: str,
        response: str,
        criteria: list[str] | none = none,
    ) -> list[dict[str, str]]:
        """build the judge prompt messages."""
        active_criteria = criteria or self.criteria
        criteria_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(active_criteria))

        user_content = f"""please evaluate the following response.

## original prompt
{prompt}

## response to evaluate
{response}

## scoring criteria
{criteria_text}

## instructions
score each criterion from 0.0 to 1.0. respond only in json format:
{{
  "scores": {{"criterion_name": score, ...}},
  "explanation": "brief explanation of scores",
  "overall_score": <average of all scores rounded to 2 decimals>
}}"""

        return [
            {"role": "system", "content": self.default_system_prompt},
            {"role": "user", "content": user_content},
        ]

    def score(
        self,
        prompt: str,
        response: str,
        criteria: list[str] | none = none,
    ) -> dict:
        """score a response using the judge llm.

        args:
            prompt: the original prompt/question
            response: the model's response to evaluate
            criteria: optional specific criteria (uses self.criteria if none)

        returns:
            dict with keys:
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
            content = result.choices[0].message.content.strip()  # type: ignore[attr-defined]

            # try to parse json response
            # first try direct json parse
            try:
                parsed = json.loads(content)
            except json.jsondecodeerror:
                # try to extract json from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.dotall)
                if json_match:
                    parsed = json.loads(json_match.group(1).strip())
                else:
                    # try to find {...} in the content
                    brace_match = re.search(r'\{.*\}', content, re.dotall)
                    if brace_match:
                        parsed = json.loads(brace_match.group(0))
                    else:
                        # fallback: heuristic scoring
                        return self._heuristic_fallback(prompt, response, criteria)

            # normalize and extract scores
            scores = parsed.get("scores", {})
            if not scores:
                # try 'overall_score' directly
                overall = parsed.get("overall_score", 0.5)
                return {
                    "score": min(max(float(overall), 0.0), 1.0),
                    "scores": {"overall": min(max(float(overall), 0.0), 1.0)},
                    "explanation": parsed.get("explanation", ""),
                }

            # average the per-criterion scores
            score_values = [min(max(float(v), 0.0), 1.0) for v in scores.values()]
            avg_score = sum(score_values) / len(score_values) if score_values else 0.5

            return {
                "score": round(avg_score, 4),
                "scores": {k: min(max(float(v), 0.0), 1.0) for k, v in scores.items()},
                "explanation": parsed.get("explanation", ""),
            }

        except exception as e:
            # on any error, use fallback scoring
            logger = logging.getlogger(__name__)
            logger.warning(f"aijudge error: {e}. using fallback scoring.")
            return self._heuristic_fallback(prompt, response, criteria)

    def _heuristic_fallback(
        self,
        prompt: str,
        response: str,
        criteria: list[str] | none = none,
    ) -> dict:
        """fallback scoring when judge llm fails."""
        response_len = len(response)

        # length-based score (longer = more effort)
        length_score = min(response_len / 500, 1.0) * 0.3

        # structure score (has numbered lists, code blocks, etc.)
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
            "explanation": "fallback heuristic scoring (judge llm unavailable).",
        }


class aijudgerewardmodel:
    """reward function wrapper for grpotrainer that uses an ai judge.

    wraps aijudge to match the grpotrainer reward_funcs interface:
    reward_funcs(completions: list[str], prompts: list[str]) -> list[float]

    example:
        >>> judge = aijudge(llm_name="openai/gpt-4o-mini")
        >>> reward_fn = aijudgerewardmodel(judge)
        >>> scores = reward_fn(["response a", "response b"], ["prompt x", "prompt y"])
    """

    def __init__(
        self,
        judge: aijudge,
        criteria: list[str] | none = none,
    ):
        self.judge = judge
        self.criteria = criteria

    def __call__(self, completions: list[str], prompts: list[str]) -> list[float]:
        rewards = []
        for completion, prompt in zip(completions, prompts, strict=false):
            result = self.judge.score(prompt, completion, criteria=self.criteria)
            rewards.append(result["score"])
        return rewards


class selfplaygrpo:
    """self-play grpo orchestrator: generate → judge → train iterative loop.

    this class orchestrates a complete self-play training loop where:
    1. the model generates responses to prompts (generate phase)
    2. an ai judge scores those responses (judge phase)
    3. grpo training updates the model using judge scores as rewards (train phase)

    this enables rl training without sandbox execution — the reward signal
    comes entirely from ai evaluation.

    example:
        >>> config = rlhftrainingconfig(
        ...     mode=aitrainermode.ai_judge,
        ...     model_name_or_path="qwen/qwen2.5-0.5b-instruct",
        ...     judge_llm_name="openai/gpt-4o-mini",
        ...     judge_criteria=["correctness", "clarity", "helpfulness"],
        ...     max_steps=50,
        ... )
        >>> sp = selfplaygrpo(config)
        >>> results = sp.run()
    """

    def __init__(
        self,
        config: rlhftrainingconfig,
        product_manager: productmanager | none = none,
        judge: aijudge | none = none,
    ):
        self.config = config
        self.pm = product_manager or productmanager()

        # create ai judge from config if not provided
        if judge is not none:
            self.judge = judge
        else:
            self.judge = aijudge(
                llm_name=config.judge_llm_name,
                criteria=config.judge_criteria,
                temperature=config.judge_temperature,
                base_url=config.judge_base_url,
            )

        self.reward_fn = aijudgerewardmodel(self.judge, criteria=config.judge_criteria)
        self.log = _setup_logger("selfplaygrpo")

    def run(self) -> dict[str, any]:
        """run the self-play grpo training loop.

        returns:
            dict with training metrics including iteration history
        """
        self.log.info("=" * 60)
        self.log.info("starting self-play grpo training")
        self.log.info(f"mode: ai_judge | judge: {self.config.judge_llm_name}")
        self.log.info(f"criteria: {self.config.judge_criteria or self.judge.criteria}")
        self.log.info(f"model: {self.config.model_name_or_path}")
        self.log.info("=" * 60)

        # delegate to the existing train() function with ai judge mode
        # the train() function handles model loading, dataset prep, etc.
        results = train(
            config=self.config,
            product_manager=self.pm,
            ai_judge=self.judge,  # pass ai judge for reward computation
        )

        self.log.info("self-play grpo training complete")
        results["mode"] = "ai_judge"
        results["judge_llm"] = self.config.judge_llm_name

        return results


class codeexecutor:
    """executes code in local environment."""

    def __init__(self, working_dir: str = "/tmp/testbed"):
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=true)

    def run(self, code: str, timeout: int = 30) -> tuple[str, int]:
        import subprocess
        import tempfile
        with tempfile.namedtemporaryfile(mode='w', suffix='.py', delete=false, encoding='utf-8') as f:
            f.write(code)
            temp_path = f.name
        try:
            result = subprocess.run(['python', temp_path], capture_output=true, text=true, timeout=timeout)
            return result.stdout + result.stderr, result.returncode
        except subprocess.timeoutexpired:
            return f"execution timed out ({timeout}s)", -1
        except exception as e:
            return f"error: {repr(e)}", -1
        finally:
            with contextlib.suppress(oserror):
                os.unlink(temp_path)


def _create_reward_func_from_rm(
    reward_model: rewardmodel,
) -> callable[[list[str], list[str]], list[float]]:
    """create a reward function compatible with grpotrainer from a rewardmodel instance."""

    def reward_func(completions: list[str], prompts: list[str]) -> list[float]:
        rewards = []
        for completion, prompt in zip(completions, prompts, strict=false):
            task = prompt.strip()
            reward = reward_model.score(task, completion)
            rewards.append(reward)
        return rewards

    return reward_func


def _load_model_and_tokenizer(config: rlhftrainingconfig):
    """load model and tokenizer with unsloth optimizations.

    uses fastlanguagemodel for efficient loading and optionally enables
    vllm fast inference for accelerated generation during rl training.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = _setup_logger(__name__)

    # load tokenizer
    tokenizer = autotokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=true,
    )

    # ensure tokenizer has pad_token (required for training)
    if tokenizer.pad_token is none:
        tokenizer.pad_token = tokenizer.eos_token

    # ensure padding side is set for training
    tokenizer.padding_side = "left"

    # load model with unsloth
    model_kwargs = {
        "model": config.model_name_or_path,
        "max_seq_length": config.max_seq_length,
        "load_in_4bit": true,
        "fast_inference": config.fast_inference,
    }

    # only add gpu_memory_utilization if using fast_inference (vllm)
    if config.fast_inference:
        model_kwargs["gpu_memory_utilization"] = config.vllm_gpu_memory_utilization

    model, _ = fastlanguagemodel.from_pretrained(**model_kwargs)

    # apply lora peft adapter
    lora_kwargs = {
        "model": model,
        "r": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "use_gradient_checkpointing": "unsloth",  # unsloth's long-context fine-tuning
        "random_state": config.seed,
    }

    if config.target_modules:
        lora_kwargs["target_modules"] = config.target_modules
    else:
        # auto-detect target modules (common for most models)
        lora_kwargs["target_modules"] = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    model = fastlanguagemodel.get_peft_model(**lora_kwargs)
    model = model.to(device)

    log.info(f"model loaded with lora (rank={config.lora_rank}) on {device}")
    if config.fast_inference:
        log.info("vllm fast inference enabled")

    return model, tokenizer, device


def _create_grpo_config(config: rlhftrainingconfig):
    """create trl grpoconfig from rlhftrainingconfig."""
    try:
        from trl import grpoconfig
    except importerror as e:
        raise importerror("trl library required. install with: pip install trl") from e

    max_prompt_length = config.max_seq_length // 2
    max_completion_length = config.max_seq_length - max_prompt_length

    grpo_config = grpoconfig(
        # output
        output_dir=config.output_dir,
        seed=config.seed,

        # optimization
        learning_rate=config.learning_rate,
        beta=config.beta,
        max_grad_norm=config.max_grad_norm,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        # generation
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        max_completion_length=config.max_completion_length or max_completion_length,
        num_generations=config.num_generations,

        # batch
        per_device_train_batch_size=config.per_device_train_batch_size,

        # training limits
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,

        # vllm
        use_vllm=config.use_vllm,
        vllm_mode="colocate" if config.use_vllm else none,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,

        # logging
        logging_steps=1,
        log_completions=true,

        # save
        save_steps=config.max_steps // 10 or 10,

        # misc
        report_to="none",
    )

    return grpo_config


def train(
    config: rlhftrainingconfig,
    product_manager: productmanager | none = none,
    reward_model: rewardmodel | none = none,
    ai_judge: aijudge | none = none,
) -> dict[str, any]:
    """run grpo training using unsloth's optimized grpotrainer.

    this function uses trl's grpotrainer with unsloth's patchfastrl optimizations
    for accelerated rl training with vllm inference.

    args:
        config: rlhftrainingconfig with model and training parameters
        product_manager: optional productmanager for custom task prompts
        reward_model: optional rewardmodel for sandbox-based reward scoring
        ai_judge: optional aijudge for ai-judge-based reward scoring

    returns:
        dict with training metrics and results
    """
    log = _setup_logger(__name__)

    # log training configuration


    random.seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=true)

    # step 1: patch trl for unsloth grpo optimizations
    log.info("applying unsloth optimizations to trl...")
    try:
        from unsloth import fastlanguagemodel, patchfastrl
        patchfastrl(algorithm="grpo", fastlanguagemodel=fastlanguagemodel)
        log.info("unsloth patchfastrl applied successfully")
    except importerror as e:
        log.warning(f"could not apply patchfastrl: {e}. continuing with base unsloth.")

    # step 2: load model and tokenizer with lora
    model, tokenizer, device = _load_model_and_tokenizer(config)
    log.info(f"model loaded on device: {device}")

    # step 3: prepare dataset for grpotrainer
    pm = product_manager or productmanager()
    rm = reward_model or rewardmodel()

    try:
        from datasets import dataset
    except importerror as e:
        raise importerror("datasets library required. install with: pip install datasets") from e

    templates = pm.task_templates if pm.task_templates else pm._default_templates
    # grpotrainer needs 'prompt' field
    dataset = dataset.from_dict({"prompt": templates})

    # for grpotrainer, we need to duplicate prompts for num_generations
    # the trainer handles generating multiple completions per prompt
    all_prompts = []
    for _ in range(config.num_generations):
        all_prompts.extend(templates)
    dataset = dataset.from_dict({"prompt": all_prompts})

    log.info(f"dataset prepared: {len(dataset)} samples ({config.num_generations} generations x {len(templates)} prompts)")

    # step 4: create reward function (sandbox or ai judge mode)
    if config.mode == aitrainermode.ai_judge or ai_judge is not none:
        # ai judge mode
        effective_judge = ai_judge or aijudge(
            llm_name=config.judge_llm_name,
            criteria=config.judge_criteria,
            temperature=config.judge_temperature,
            base_url=config.judge_base_url,
        )
        reward_func = aijudgerewardmodel(effective_judge, criteria=config.judge_criteria)
        log.info(f"using ai judge reward: {config.judge_llm_name}")
        log.info(f"judge criteria: {config.judge_criteria or effective_judge.criteria}")
    else:
        # sandbox (default) mode
        rm = reward_model or rewardmodel()
        reward_func = _create_reward_func_from_rm(rm)  # type: ignore[assignment]
        log.info("using sandbox-based reward model")

    # step 5: create grpoconfig
    grpo_config = _create_grpo_config(config)

    # step 6: initialize grpotrainer
    log.info("initializing grpotrainer...")
    try:
        from trl import grpotrainer
    except importerror as e:
        raise importerror("trl library required. install with: pip install trl") from e

    trainer = grpotrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_func,
    )

    # step 7: prepare model for training
    model = model.for_training()
    log.info("model prepared for training (for_training mode)")

    # step 8: train!
    log.info("starting grpo training...")
    try:
        trainer.train()
    except exception as e:
        log.error(f"training error: {e}")
        raise

    # step 9: save model
    final_checkpoint = os.path.join(config.output_dir, "final_model")
    log.info(f"saving final model to {final_checkpoint}")
    trainer.save_model(final_checkpoint)
    tokenizer.save_pretrained(final_checkpoint)

    # step 10: return results
    log.info("=" * 60)
    log.info("training complete")
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

    # save results
    results_path = os.path.join(config.output_dir, "training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=false)
    log.info(f"results saved to {results_path}")

    return results


# keep legacy classes for backward compatibility
class referencemodel:
    """reference model for kl divergence constraint (kept for compatibility)."""

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
        """compute kl divergence between policy and reference model."""
        torch = self.torch
        with torch.no_grad():
            ref_logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        policy_logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        policy_probs = torch.nn.functional.softmax(policy_logits, dim=-1)
        kl = torch.nn.functional.kl_div(ref_log_probs, policy_probs, reduction='batchmean', log_target=true)
        self._last_kl_float = kl.item()
        return kl

    def reset(self):
        """reset reference model to original parameters."""
        for name, param in self.model.named_parameters():
            if name in self.reference_params:
                param.data = self.reference_params[name].to(param.device)


class replaybuffer:
    """experience replay buffer (kept for compatibility)."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: list[dict[str, any]] = []
        self.advantages: list[float] = []

    def add(self, trajectory: dict[str, any], reward: float):
        """add a trajectory to the buffer."""
        self.buffer.append(trajectory)
        self.advantages.append(reward)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
            self.advantages.pop(0)

    def compute_advantages(self, baseline: float | none = none) -> list[float]:
        """compute advantages using mean rewards as baseline."""
        if not self.advantages:
            return []
        if baseline is none:
            baseline = sum(self.advantages) / len(self.advantages)
        advantages = [r - baseline for r in self.advantages]
        mean_adv = sum(advantages) / len(advantages)
        std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5
        if std_adv > 1e-8:
            advantages = [(a - mean_adv) / std_adv for a in advantages]
        return advantages

    def sample(self, batch_size: int) -> list[dict[str, any]]:
        """sample a batch from the buffer."""
        if len(self.buffer) <= batch_size:
            return self.buffer.copy()
        return random.sample(self.buffer, batch_size)

    def clear(self):
        """clear the buffer."""
        self.buffer.clear()
        self.advantages.clear()

    def __len__(self):
        return len(self.buffer)


class grpooptimizer:
    """grpo optimizer (legacy, kept for backward compatibility).

    note: new code should use train() with grpotrainer for unsloth-optimized
    grpo training with vllm acceleration and proper reference model support.
    """

    def __init__(self, model, tokenizer, beta: float = 0.01, lr: float = 1e-5, max_grad_norm: float = 1.0):
        import torch
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.device = next(model.parameters()).device
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.adamw(model.parameters(), lr=lr, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.linearlr(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        self.reference_model = referencemodel(model, tokenizer)

    def compute_grpo_loss(self, input_ids, attention_mask, response_ids, advantages: list[float]) -> tuple[torch.tensor, dict[str, float]]:
        """compute grpo loss."""
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

class data:
    def __init__(self, data_file_location):
        self.data_file_location = data_file_location
        self.data = []
        self.invalid_entries = []
        # 获取该类专属的 logger
        self.logger = logging.getlogger(self.__class__.__name__)
        self._load_and_check()

    def _load_and_check(self):
        try:
            with open(self.data_file_location, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    self.logger.error("文件为空: %s", self.data_file_location)
                    raise valueerror("文件为空")

                if content.startswith('['):
                    items = json.loads(content)
                    if not isinstance(items, list):
                        raise valueerror("json 根节点不是数组")
                    self._validate_items(items, from_array=true)
                else:
                    f.seek(0)
                    for line_num, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            self._validate_single(obj, line_num)
                            self.data.append(obj)
                        except json.jsondecodeerror as e:
                            msg = f"第 {line_num} 行 json 解析错误: {e}"
                            self.logger.error(msg)
                            self.invalid_entries.append((line_num, msg))
        except filenotfounderror as e:
            self.logger.exception("文件不存在: %s", self.data_file_location)
            raise
        except json.jsondecodeerror as e:
            self.logger.exception("文件 json 解析失败")
            raise valueerror(f"json 解析失败: {e}")

    def _validate_single(self, item, line_info):
        if not isinstance(item, dict):
            msg = f"位置 {line_info}: 元素不是 json 对象"
            self.logger.error(msg)
            self.invalid_entries.append((line_info, msg))
            return
        if "question" not in item:
            msg = f"位置 {line_info}: 缺少 'question' 字段"
            self.logger.error(msg)
            self.invalid_entries.append((line_info, msg))
            return
        if not isinstance(item["question"], str) or not item["question"].strip():
            msg = f"位置 {line_info}: 'question' 字段不是非空字符串"
            self.logger.error(msg)
            self.invalid_entries.append((line_info, msg))
            return
