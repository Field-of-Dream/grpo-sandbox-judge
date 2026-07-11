# GRPO-in-Sandbox

Agentic Reinforcement Learning Library using GRPO within a code sandbox.

## Installation

```bash
pip install grpo-in-sandbox
```

Training requires extra dependencies:

```bash
pip install "grpo-in-sandbox[training]"   # pure TRL backend (transformers + trl + peft)
pip install "grpo-in-sandbox[unsloth]"    # Unsloth-optimized backend (includes training deps)
```

Or from source:

```bash
git clone https://github.com/llm-in-sandbox/llm-in-sandbox.git
cd llm-in-sandbox
pip install -e .
```

## Quick Start

### Basic Training

```python
from grpo_in_sandbox import train, RLHFTrainingConfig

config = RLHFTrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    backend="trl",          # "auto" (default) | "trl" | "unsloth"
    num_train_epochs=3,
    learning_rate=1e-5,
)
results = train(config)
```

The `backend` field selects how the policy model is loaded (both paths use TRL's
`GRPOTrainer`): `"trl"` uses plain Transformers + PEFT, `"unsloth"` uses Unsloth's
`FastLanguageModel` with vLLM fast inference, and `"auto"` prefers Unsloth when it
is installed and otherwise falls back to `"trl"`.

### Using Custom Runtime

```python
from grpo_in_sandbox import (
    create_runtime,
    RLHFTrainingConfig,
    ProductManager,
    RewardModel,
)

runtime = create_runtime(backend="docker")

reward_model = RewardModel(runtime=runtime)
pm = ProductManager()

config = RLHFTrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    num_train_epochs=3,
)

results = train(config, product_manager=pm, reward_model=reward_model)
```

### Running Agent

```python
from grpo_in_sandbox import Agent, AgentArgs, create_runtime, load_config

runtime = create_runtime(backend="docker")
config = load_config()

args = AgentArgs(
    system_prompt=config.system_prompt,
    instance_prompt=config.instance_prompt,
    llm_name="openai/gpt-4",
)

agent = Agent(args)
trajectory = agent.run(
    runtime=runtime,
    problem_statement="Write a hello world program in Python",
)

print(trajectory.to_dict())
```

## API

### Core Classes

| Class | Description |
|-------|-------------|
| `train()` | Main GRPO training function |
| `RLHFTrainingConfig` | Training configuration |
| `create_runtime()` | Create sandbox runtime |
| `Agent` | Agent for executing tasks |
| `RewardModel` | Reward function |
| `ProductManager` | Task generator |

### Runtime Backends

```python
from grpo_in_sandbox import create_runtime

# Auto-detect environment
runtime = create_runtime(backend="auto")

# Explicit backends
docker_runtime = create_runtime(backend="docker", docker_image="cdx123/llm-in-sandbox:v0.1")
kaggle_runtime = create_runtime(backend="kaggle")
local_runtime = create_runtime(backend="local")
```

### Configuration

```python
from grpo_in_sandbox import load_config, AgentConfig

config = load_config()  # Uses default config/general.yaml
config = load_config("config/my_config.yaml")

system_prompt = config.format_system_prompt(
    working_dir="/testbed",
    input_dir="/testbed/input", 
    output_dir="/testbed/output",
)
```

## CLI Usage

```bash
# Run agent
grpo-in-sandbox run --query "write hello world" --llm_name openai/gpt-4

# Build Docker image
grpo-in-sandbox build
```

## Examples

See `grpo_in_sandbox/example.ipynb` for more examples.

## License

Apache 2.0