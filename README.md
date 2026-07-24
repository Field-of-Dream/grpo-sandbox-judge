# GRPO-in-Sandbox Package

Agentic Reinforcement Learning Library.
It's from llm_in_sandbox（）

## Core API

```python
from grpo_in_sandbox import (
    train,
    RLHFTrainingConfig,
    create_runtime,
    Agent,
    AgentArgs,
    RewardModel,
    ProductManager,
    BaseRuntime,
    LocalRuntime,
    DockerRuntime,
    KaggleRuntime,
    Trajectory,
    TrajectoryStep,
    AgentConfig,
    load_config,
)
```

## Module Structure

| Module | Description |
|--------|-------------|
| `train.py` | GRPO training interface |
| `runtime.py` | Unified runtime abstraction |
| `config.py` | Configuration management |
| `agent.py` | Agent implementation |
| `trajectory.py` | Trajectory recording |
| `observation.py` | Action results |
| `action.py` | Action representation |
| `tools.py` | Tool definitions |
| `docker_runtime.py` | Docker backend |

## Training Example

```python
from grpo_in_sandbox import train, RLHFTrainingConfig

config = RLHFTrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    num_train_epochs=3,
)
results = train(config)
```

### Training Backends

`train()` drives TRL's `GRPOTrainer` with a selectable model-loading backend:

| `backend` | Install | Notes |
|-----------|---------|-------|
| `"auto"` (default) | — | Uses Unsloth if installed, otherwise pure TRL |
| `"trl"` | `pip install "grpo-in-sandbox[training]"` | Pure TRL + Transformers + PEFT, no Unsloth needed |
| `"unsloth"` | `pip install "grpo-in-sandbox[unsloth]"` | Unsloth-optimized loading + vLLM fast inference |

```python
from grpo_in_sandbox import train, RLHFTrainingConfig

config = RLHFTrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    backend="trl",    # pure TRL — no Unsloth required
    use_vllm=False,   # optional; auto-disabled when vllm is not installed
)
results = train(config)
```

## AI Judge Self-Play Mode

Train models WITHOUT sandbox execution using AI-as-Judge for reward signals.

```python
from grpo_in_sandbox import (
    SelfPlayGRPO,
    RLHFTrainingConfig,
    AITrainerMode,
    AIJudge,
    ProductManager,
)

# Method 1: Using SelfPlayGRPO orchestrator
config = RLHFTrainingConfig(
    mode=AITrainerMode.AI_JUDGE,
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    judge_llm_name="openai/gpt-4o-mini",
    judge_criteria=["correctness", "clarity", "helpfulness"],
    max_steps=50,
)
sp = SelfPlayGRPO(config)
sp.run()

# Method 2: Using train() with ai_judge parameter
judge = AIJudge(llm_name="openai/gpt-4o-mini")
results = train(config, ai_judge=judge)
```

For full API tutorial, see [docs/AI_JUDGE_GRPO_TUTORIAL.md](docs/AI_JUDGE_GRPO_TUTORIAL.md)

## Runtime Example

```python
from grpo_in_sandbox import create_runtime

runtime = create_runtime(backend="docker")
output, exit_code = runtime.run("echo hello")
stdout, stderr, exit_code = runtime.demux_run("echo hello")
runtime.close()
```

## Agent Example

```python
from grpo_in_sandbox import Agent, AgentArgs, create_runtime

runtime = create_runtime(backend="docker")

args = AgentArgs(
    system_prompt="You are a helpful assistant.",
    instance_prompt="{problem_statement}",
    llm_name="openai/gpt-4",
)

agent = Agent(args)
trajectory = agent.run(
    runtime=runtime,
    problem_statement="Solve 2+2",
)
```

# 需要安装的依赖项

```bash
pip install grpo-in-sandbox               # 核心（沙箱 / Agent / CLI）
pip install "grpo-in-sandbox[training]"   # + GRPO 训练（纯 TRL 后端：transformers/trl/peft/accelerate/torch）
pip install "grpo-in-sandbox[unsloth]"    # + Unsloth 优化后端（含 training 依赖）
pip install "grpo-in-sandbox[quant]"      # + bitsandbytes（load_in_4bit=True 时需要）
```
