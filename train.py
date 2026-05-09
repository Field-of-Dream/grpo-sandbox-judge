"""
GRPO-in-Sandbox Training Module

This module provides a complete implementation for training language models using GRPO
(Group Relative Policy Optimization) within a code sandbox environment.

Key Components:
- RLHFTrainingConfig: Configuration for training
- ReferenceModel: Reference model for KL divergence
- ReplayBuffer: Experience replay buffer
- GRPOOptimizer: GRPO policy optimization
- RewardModel: Reward function
- train(): Main training function with full GRPO loop

Usage:
    from grpo_in_sandbox import train, RLHFTrainingConfig

    config = RLHFTrainingConfig(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        num_train_epochs=3,
        max_steps=100,
    )
    results = train(config)
"""

import os
import json
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union


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

@dataclass
class RLHFTrainingConfig:
    """Configuration for GRPO training."""

    model_name_or_path: str = "./model"

    # Optimizer settings
    learning_rate: float = 1e-5
    beta: float = 0.01  # KL penalty coefficient
    max_grad_norm: float = 1.0

    # Generation settings
    num_train_epochs: int = 3
    max_seq_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

    # Batch settings
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4

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

    def __repr__(self) -> str:
        """Show training configuration."""
        return (
            f"RLHFTrainingConfig("
            f"model={self.model_name_or_path}, "
            f"lr={self.learning_rate}, "
            f"beta={self.beta}, "
            f"batch={self.per_device_train_batch_size}, "
            f"steps={self.max_steps})"
        )


class ProductManager:
    """Generates tasks/problems for training.

    This class manages task prompts for GRPO training. Users can provide
    their own prompts via the constructor or add them dynamically.

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

    def __init__(self, task_templates: Optional[List[str]] = None):
        """Initialize ProductManager.

        Args:
            task_templates: Optional list of custom prompt templates.
                          If None or empty, uses default Chinese QA templates.
        """
        self.task_templates: List[str] = task_templates if task_templates is not None else []
        self.num_generated = 0
        self._default_templates: List[str] = [
            "作为QA工程师，请测试以下功能：用户登录系统，包括正常登录、密码错误、账户锁定等场景。",
            "请进行回归测试：订单创建功能，验证库存扣减、支付流程、订单状态流转。",
            "执行API测试：用户管理接口，测试创建、查询、更新、删除用户的各项操作。",
            "请测试购物车功能：添加商品、修改数量、删除商品、结算流程。",
            "执行冒烟测试：检查系统核心功能的基本可用性。",
            "测试用户注册流程：验证表单验证、邮箱验证、验证码发送等功能。",
            "执行性能测试：模拟高并发场景，检查系统响应时间和稳定性。",
        ]

    def add_task_template(self, template: str) -> None:
        """Add a custom task prompt template.

        Args:
            template: Prompt text to add to the template list.
        """
        self.task_templates.append(template)

    def add_task_templates(self, templates: List[str]) -> None:
        """Add multiple custom task prompt templates.

        Args:
            templates: List of prompt texts to add.
        """
        self.task_templates.extend(templates)

    @classmethod
    def from_file(cls, file_path: str) -> "ProductManager":
        """Create ProductManager from a file containing prompts.

        Args:
            file_path: Path to a text file with one prompt per line.

        Returns:
            ProductManager instance with prompts loaded.

        Usage:
            pm = ProductManager.from_file("/path/to/prompts.txt")
        """
        with open(file_path, "r", encoding="utf-8") as f:
            templates = [line.strip() for line in f if line.strip()]
        return cls(task_templates=templates)

    def generate_task(self) -> str:
        """Generate a random task prompt.

        Returns:
            A task prompt string. If user provided templates exist, use those.
            Otherwise, uses default Chinese QA templates.
        """
        self.num_generated += 1
        templates = self.task_templates if self.task_templates else self._default_templates
        return random.choice(templates)

    def format_prompt(self, task: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Format task as chat messages.

        Args:
            task: The task/prompt text.
            system_prompt: Optional custom system prompt. If not provided, uses default.

        Returns:
            List of message dicts in chat template format.
        """
        default_system = "你是一个专业的QA工程师，负责执行测试任务。"
        if(system_prompt):
            logging.basicConfig(level=logging.WARING)
            logging.WARN("Not Using custom system prompt. Make sure it is appropriate for the task.")            
        return [
            {"role": "system", "content": system_prompt or default_system},
            {"role": "user", "content": f"任务：{task}"},
        ]

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set a custom system prompt.

        Args:
            system_prompt: Custom system prompt text.
        """
        self._custom_system_prompt = system_prompt

    @property
    def templates(self) -> List[str]:
        """Get all available templates."""
        return self.task_templates if self.task_templates else self._default_templates

    def __len__(self) -> int:
        """Return number of user-provided templates."""
        return len(self.task_templates)


class ReferenceModel:
    """Reference model for KL divergence constraint."""

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
        """Compute KL divergence between policy and reference model.

        Returns:
            KL divergence as a tensor (for loss computation) and also logs the float value.
        """
        torch = self.torch
        with torch.no_grad():
            ref_logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        policy_logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        policy_probs = torch.nn.functional.softmax(policy_logits, dim=-1)
        kl = torch.nn.functional.kl_div(ref_log_probs, policy_probs, reduction='batchmean', log_target=True)
        # Store float value for logging before returning tensor
        self._last_kl_float = kl.item()
        return kl

    def reset(self):
        """Reset reference model to original parameters."""
        for name, param in self.model.named_parameters():
            if name in self.reference_params:
                param.data = self.reference_params[name].to(param.device)


class ReplayBuffer:
    """Experience replay buffer for storing trajectories."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: List[Dict[str, Any]] = []
        self.advantages: List[float] = []

    def add(self, trajectory: Dict[str, Any], reward: float):
        """Add a trajectory to the buffer."""
        self.buffer.append(trajectory)
        self.advantages.append(reward)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
            self.advantages.pop(0)

    def compute_advantages(self, baseline: Optional[float] = None) -> List[float]:
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

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
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


class CodeExecutor:
    """Executes code in local environment."""

    def __init__(self, working_dir: str = "/tmp/testbed"):
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)

    def run(self, code: str, timeout: int = 30) -> Tuple[str, int]:
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
            try:
                os.unlink(temp_path)
            except:
                pass


class RewardModel:
    """Reward model for evaluating agent outputs."""

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
        if "登录" in task_lower:
            if any(kw in response_lower for kw in ["正常登录", "密码错误", "username", "password"]):
                score += 15
        if "回归" in task_lower:
            if any(kw in response_lower for kw in ["订单", "库存", "支付", "状态"]):
                score += 15
        if "API" in task_lower:
            if any(kw in response_lower for kw in ["POST", "GET", "PUT", "DELETE", "接口", "endpoint"]):
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
            if "def test_" not in qa_output and "import unittest" not in qa_output:
                return 0.5
            test_code = self._extract_test_code(qa_output)
            if not test_code:
                return 0.5
            test_file = "/tmp/verify_test.py"
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(test_code)
            output, exit_code = self.runtime.run(f"python -m pytest {test_file} -v 2>&1 | head -50")
            if exit_code == 0:
                return 1.0
            elif "error" in output.lower() or "fail" in output.lower():
                return 0.3
            return 0.6
        except:
            return 0.5

    def _extract_test_code(self, qa_output: str) -> str:
        lines = qa_output.split("\n")
        in_test = False
        test_lines = []
        for line in lines:
            if "def test_" in line or "class Test" in line:
                in_test = True
            if in_test:
                test_lines.append(line)
        if test_lines:
            code = "import unittest\n" + "\n".join(test_lines)
            if "unittest.main" not in code:
                code += "\n\nif __name__ == '__main__':\n    unittest.main()"
            return code
        return ""

    def close(self):
        if self.runtime:
            self.runtime.close()


class GRPOOptimizer:
    """GRPO (Group Relative Policy Optimization) Optimizer."""

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

    def compute_grpo_loss(self, input_ids, attention_mask, response_ids, advantages: List[float]):
        """Compute GRPO loss.

        Returns:
            Tuple of (total_loss tensor, metrics dict)
        """
        torch = self.torch
        F = torch.nn.functional
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=response_ids)
        policy_logits = outputs.logits
        response_logits = policy_logits[:, :-1, :]
        response_labels = response_ids[:, 1:]
        log_probs = F.log_softmax(response_logits, dim=-1)
        selected_log_probs = log_probs.gather(2, response_labels.unsqueeze(2)).squeeze(2)
        response_mask = attention_mask[:, 1:].float()
        policy_log_probs = (selected_log_probs * response_mask).sum(dim=-1) / (response_mask.sum(dim=-1) + 1e-8)
        advantages_tensor = torch.tensor(advantages, device=self.device)
        grpo_loss = -(policy_log_probs * advantages_tensor).mean()
        # KL as tensor for gradient (beta is a float, multiplication works via broadcasting)
        kl_div = self.reference_model.compute_kl_divergence(input_ids, attention_mask)
        kl_penalty = self.beta * kl_div
        total_loss = grpo_loss + kl_penalty
        # Get float values for logging
        kl_float = getattr(self.reference_model, '_last_kl_float', 0.0)
        metrics = {"grpo_loss": grpo_loss.item(), "kl_div": kl_float, "kl_penalty": kl_penalty.item() if hasattr(kl_penalty, 'item') else kl_penalty, "total_loss": total_loss.item()}
        return total_loss, metrics

    def step(self, input_ids, attention_mask, response_ids, advantages: List[float]) -> Dict[str, float]:
        torch = self.torch
        loss, metrics = self.compute_grpo_loss(input_ids, attention_mask, response_ids, advantages)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.reference_model.reset()
        return metrics


def _load_model_and_tokenizer(config: RLHFTrainingConfig):
    """Load model and tokenizer."""
    try:
        import torch
        from transformers import AutoTokenizer
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model, _ = FastLanguageModel.from_pretrained(model_name=config.model_name_or_path, max_seq_length=config.max_seq_length, load_in_4bit=False)
    model = model.to(device)
    return model, tokenizer, device


def train(config: RLHFTrainingConfig, product_manager: Optional[ProductManager] = None, reward_model: Optional[RewardModel] = None) -> Dict[str, Any]:
    """Run full GRPO training loop in sandbox environment."""
    log = _setup_logger(__name__)
    
    # Log training configuration
    log.info("=" * 50)
    log.info("GRPO Training Configuration:")
    log.info(f"  Model: {config.model_name_or_path}")
    log.info(f"  Learning rate: {config.learning_rate}")
    log.info(f"  KL beta: {config.beta}")
    log.info(f"  Max grad norm: {config.max_grad_norm}")
    log.info(f"  Batch size: {config.per_device_train_batch_size}")
    log.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    log.info(f"  Max steps: {config.max_steps}")
    log.info(f"  Max seq length: {config.max_seq_length}")
    log.info(f"  Temperature: {config.temperature}")
    log.info(f"  Top p: {config.top_p}")
    log.info(f"  Output dir: {config.output_dir}")
    log.info(f"  Seed: {config.seed}")
    log.info("=" * 50)
    
    random.seed(config.seed)
    model, tokenizer, device = _load_model_and_tokenizer(config)
    log.info(f"Model loaded on device: {device}")
    grpo_optimizer = GRPOOptimizer(model=model, tokenizer=tokenizer, beta=config.beta, lr=config.learning_rate, max_grad_norm=config.max_grad_norm)
    pm = product_manager or ProductManager()
    rm = reward_model or RewardModel()
    replay_buffer = ReplayBuffer(max_size=config.max_steps)
    training_metrics = {"rewards": [], "kl_divs": [], "losses": []}
    best_reward = 0.0
    patience_counter = 0
    os.makedirs(config.output_dir, exist_ok=True)
    log.info(f"Output directory created: {config.output_dir}")
    log.info("Starting training loop...")
    try:
        import torch
        for step in range(config.max_steps):
            log.info(f"--- Step {step + 1}/{config.max_steps} ---")
            task = pm.generate_task()
            log.debug(f"Generated task: {task[:50]}...")
            messages = pm.format_prompt(task)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=config.max_seq_length).to(device)
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids))
            num_samples = config.per_device_train_batch_size
            all_responses = []
            all_rewards = []
            for sample_idx in range(num_samples):
                with torch.no_grad():
                    outputs = model.generate(input_ids=input_ids, max_new_tokens=512, temperature=config.temperature, do_sample=True, top_p=config.top_p, pad_token_id=tokenizer.pad_token_id)
                response_ids = outputs[0][input_ids.shape[1]:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)
                all_responses.append(response_ids)
                reward = rm.score(task, response)
                all_rewards.append(reward)
            mean_reward = sum(all_rewards) / len(all_rewards)
            advantages = [r - mean_reward for r in all_rewards]
            mean_adv = sum(advantages) / len(advantages)
            std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5
            if std_adv > 1e-8:
                advantages = [(a - mean_adv) / std_adv for a in advantages]
            log.info(f"  Reward: {mean_reward:.4f}, Adv mean: {mean_adv:.4f}")
            trajectory = {"task": task, "input_ids": input_ids, "attention_mask": attention_mask, "responses": all_responses}
            replay_buffer.add(trajectory, mean_reward)
            training_metrics["rewards"].append(mean_reward)
            if (step + 1) % config.gradient_accumulation_steps == 0 and len(replay_buffer) >= config.per_device_train_batch_size:
                batch_size = min(config.per_device_train_batch_size, len(replay_buffer))
                batch = replay_buffer.sample(batch_size)
                batch_input_ids = torch.cat([t['input_ids'] for t in batch])
                batch_attention_mask = torch.cat([t['attention_mask'] for t in batch])
                batch_responses = torch.cat([t['responses'][0] if t['responses'] else t['input_ids'] for t in batch]).to(device)
                batch_rewards = list(replay_buffer.advantages[-batch_size:])
                batch_baseline = sum(batch_rewards) / len(batch_rewards)
                batch_advantages = [r - batch_baseline for r in batch_rewards]
                if batch_advantages:
                    mean_a = sum(batch_advantages) / len(batch_advantages)
                    std_a = (sum((a - mean_a) ** 2 for a in batch_advantages) / len(batch_advantages)) ** 0.5
                    if std_a > 1e-8:
                        batch_advantages = [(a - mean_a) / std_a for a in batch_advantages]
                try:
                    metrics = grpo_optimizer.step(batch_input_ids, batch_attention_mask, batch_responses, batch_advantages)
                    training_metrics["kl_divs"].append(metrics['kl_div'])
                    training_metrics["losses"].append(metrics['total_loss'])
                    print(f"  Loss: {metrics['total_loss']:.4f}, KL: {metrics['kl_div']:.4f}")
                except Exception as e:
                    print(f"  Optimization error: {e}")
            if mean_reward > best_reward + config.min_improvement:
                best_reward = mean_reward
                patience_counter = 0
                checkpoint_path = os.path.join(config.output_dir, "best_model")
                print(f"  Saving checkpoint to {checkpoint_path}")
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
            else:
                patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping: no improvement for {config.patience} steps")
                break
    finally:
        avg_reward = sum(training_metrics["rewards"]) / len(training_metrics["rewards"]) if training_metrics["rewards"] else 0.0
        avg_kl = sum(training_metrics["kl_divs"]) / len(training_metrics["kl_divs"]) if training_metrics["kl_divs"] else 0.0
        avg_loss = sum(training_metrics["losses"]) / len(training_metrics["losses"]) if training_metrics["losses"] else 0.0
        print(f"\n{'=' * 50}")
        print("Training Complete")
        print(f"{'=' * 50}")
        print(f"Average reward: {avg_reward:.4f}")
        print(f"Average KL div: {avg_kl:.4f}")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Best reward: {best_reward:.4f}")
        results = {"config": {"model_name_or_path": config.model_name_or_path, "learning_rate": config.learning_rate, "beta": config.beta, "num_train_epochs": config.num_train_epochs, "max_steps": config.max_steps}, "metrics": {"avg_reward": avg_reward, "avg_kl_div": avg_kl, "avg_loss": avg_loss, "best_reward": best_reward}, "training_history": training_metrics}
        results_path = os.path.join(config.output_dir, "training_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved: {results_path}")
        final_path = os.path.join(config.output_dir, "final_model")
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"Final model saved: {final_path}")
        rm.close()
    return results


__all__ = ["RLHFTrainingConfig", "ProductManager", "RewardModel", "ReferenceModel", "ReplayBuffer", "GRPOOptimizer", "train"]