# AI Judge Self-Play GRPO — API Tutorial

This tutorial covers the AI Judge mode for training language models using GRPO without sandbox execution. The reward signal comes from an LLM-as-judge instead of code execution, enabling self-play training for open-ended generation tasks.

## Overview

AI Judge Self-Play mode replaces sandbox-based reward scoring with LLM evaluation. Instead of running code to verify correctness, an AI judge model scores responses based on customizable criteria like correctness, clarity, and helpfulness. This enables training models on tasks that don't have deterministic answers, such as writing, summarization, or instruction following.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Generator      │     │   AI Judge       │     │    GRPO         │
│  Model           │────▶│   LLM            │────▶│    Trainer      │
│  (to train)      │     │   (judge)        │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ▲                        │                        │
        │                        ▼                        │
        │                Score (0.0-1.0)               │
        │                Reward Signal                  │
        └────────────────────────────────────────────┘
```

The training loop works as follows:
1. **Generate**: The generator model produces responses to prompts
2. **Judge**: The AI judge evaluates each response against criteria
3. **Train**: GRPO uses judge scores as reward signals to update the generator

## Quick Start

```python
from grpo_in_sandbox import SelfPlayGRPO, RLHFTrainingConfig, AITrainerMode

config = RLHFTrainingConfig(
    mode=AITrainerMode.AI_JUDGE,
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    judge_llm_name="openai/gpt-4o-mini",
    judge_criteria=["correctness", "clarity", "helpfulness"],
    max_steps=10,
)

sp = SelfPlayGRPO(config)
results = sp.run()
print(f"Final reward: {results.get('final_reward', 'N/A')}")
```

## API Reference

### 1. AITrainerMode

Enum controlling which reward mechanism to use for training.

```python
from grpo_in_sandbox import AITrainerMode

AITrainerMode.SANDBOX   # Use sandbox execution (default)
AITrainerMode.AI_JUDGE   # Use LLM-as-judge for rewards
```

### 2. RLHFTrainingConfig — New Fields

Extended configuration for AI Judge mode. New fields (lines 191-196 in train.py):

```python
@dataclass
class RLHFTrainingConfig:
    # ... existing fields ...

    # AI Judge mode (for self-play GRPO)
    mode: AITrainerMode = AITrainerMode.SANDBOX  # 'sandbox' or 'ai_judge'
    judge_llm_name: str = "openai/gpt-4o-mini"  # LLM to use as judge
    judge_criteria: list[str] | None = None  # e.g. ["correctness", "clarity", "helpfulness"]
    judge_temperature: float = 0.1  # Low temp for consistent judging
    judge_base_url: str | None = None  # Custom API endpoint for judge
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | AITrainerMode | SANDBOX | Switches between sandbox and AI judge |
| `judge_llm_name` | str | "openai/gpt-4o-mini" | Model ID for judge LLM |
| `judge_criteria` | list[str] \| None | None | Scoring dimensions |
| `judge_temperature` | float | 0.1 | Low temperature for consistent judging |
| `judge_base_url` | str \| None | None | Custom API endpoint for judge |

### 3. AIJudge

LLM-as-Judge class that scores responses using an AI model.

**Constructor:**

```python
class AIJudge:
    def __init__(
        self,
        llm_name: str = "openai/gpt-4o-mini",
        criteria: list[str] | None = None,
        temperature: float = 0.1,
        base_url: str | None = None,
    ):
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_name` | str | "openai/gpt-4o-mini" | Judge model identifier |
| `criteria` | list[str] \| None | ["correctness", "helpfulness", "clarity"] | Scoring criteria |
| `temperature` | float | 0.1 | Low temp for consistent judging |
| `base_url` | str \| None | None | Custom API endpoint |

**Methods:**

`score(prompt, response, criteria=None)` -> dict

Scores a response and returns a dict with keys:
- `score`: float (0.0-1.0) — average across criteria
- `scores`: dict — per-criterion scores
- `explanation`: str — brief explanation

**Example:**

```python
from grpo_in_sandbox import AIJudge

judge = AIJudge(llm_name="openai/gpt-4o-mini")

result = judge.score(
    prompt="What is 2+2?",
    response="The answer is 4. Addition of 2 and 2 gives 4."
)

print(result["score"])  # e.g., 0.92
print(result["scores"])  # {"correctness": 1.0, "helpfulness": 0.83, "clarity": 0.92}
print(result["explanation"])
```

`_heuristic_fallback(prompt, response, criteria)` -> dict

Fallback scoring when the judge LLM fails. Considers:
- Response length (longer = more effort, up to 500 chars)
- Structure (numbered lists, code blocks, newlines)
- Maximum score: 1.0

```python
result = judge._heuristic_fallback(
    prompt="What is 2+2?",
    response="The answer is 4.",
    criteria=["correctness", "helpfulness"]
)
# Returns fallback scores based on length and structure
```

### 4. AIJudgeRewardModel

Wraps AIJudge for GRPOTrainer compatibility. Matches the reward function interface.

```python
class AIJudgeRewardModel:
    def __init__(
        self,
        judge: AIJudge,
        criteria: list[str] | None = None,
    ):
        self.judge = judge
        self.criteria = criteria

    def __call__(self, completions: list[str], prompts: list[str]) -> list[float]:
        # Returns list of scores, one per completion
```

**Example:**

```python
from grpo_in_sandbox import AIJudge, AIJudgeRewardModel

judge = AIJudge(llm_name="openai/gpt-4o-mini")
reward_fn = AIJudgeRewardModel(judge, criteria=["correctness", "clarity"])

scores = reward_fn(
    completions=["Response A", "Response B"],
    prompts=["Prompt X", "Prompt Y"]
)
# Returns: [0.85, 0.72]
```

### 5. SelfPlayGRPO

Orchestrator for the generate → judge → train loop.

```python
class SelfPlayGRPO:
    def __init__(
        self,
        config: RLHFTrainingConfig,
        product_manager: ProductManager | None = None,
        judge: AIJudge | None = None,
    ):
```

**Methods:**

`run()` -> dict

Runs the full self-play training loop and returns results.

**Example:**

```python
from grpo_in_sandbox import SelfPlayGRPO, RLHFTrainingConfig, AITrainerMode

config = RLHFTrainingConfig(
    mode=AITrainerMode.AI_JUDGE,
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    judge_llm_name="openai/gpt-4o-mini",
    judge_criteria=["correctness", "clarity", "helpfulness"],
    max_steps=50,
)

sp = SelfPlayGRPO(config)
results = sp.run()
```

### 6. train() Function

The `train()` function now accepts an optional `ai_judge` parameter:

```python
def train(
    config: RLHFTrainingConfig,
    product_manager: ProductManager | None = None,
    reward_model: RewardModel | None = None,
    ai_judge: AIJudge | None = None,
) -> dict[str, Any]:
```

When `config.mode == AITrainerMode.AI_JUDGE` or `ai_judge` is provided, the function uses AI judge for rewards instead of sandbox execution.

**Example:**

```python
from grpo_in_sandbox import train, RLHFTrainingConfig, AIJudge

# Method 1: Via config
config = RLHFTrainingConfig(
    mode=AITrainerMode.AI_JUDGE,
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    judge_llm_name="openai/gpt-4o-mini",
    max_steps=50,
)
results = train(config)

# Method 2: Via ai_judge parameter
judge = AIJudge(llm_name="openai/gpt-4o-mini")
results = train(config, ai_judge=judge)
```

## Customizing the Judge

### Custom Criteria

Define your own scoring dimensions:

```python
judge = AIJudge(
    llm_name="openai/gpt-4o-mini",
    criteria=["correctness", "completeness", "safety", "format"]
)

result = judge.score(
    prompt="Explain photosynthesis",
    response="Photosynthesis is the process by which plants..."
)
# Scores on: correctness, completeness, safety, format
```

### Custom Judge Model

Use any litellm-compatible model:

```python
# Open-source model via vLLM
judge = AIJudge(
    llm_name="local/llama-3-8b",
    base_url="http://localhost:8000/v1"
)

# Anthropic model
judge = AIJudge(llm_name="anthropic/claude-3-sonnet")

# Azure OpenAI
judge = AIJudge(
    llm_name="azure/gpt-4",
    base_url="https://your-resource.openai.azure.com/"
)
```

### Custom API Endpoint

Point to your own API server:

```python
judge = AIJudge(
    llm_name="openai/gpt-4o-mini",
    base_url="https://api.example.com/v1"
)
```

## Judge Prompt Structure

The AI Judge uses a structured prompt to evaluate responses:

**System Prompt:**
```
You are an expert evaluator. Your job is to score AI responses based on given criteria.
Score each criterion from 0.0 to 1.0, where:
- 1.0 = perfect
- 0.7 = good
- 0.5 = acceptable
- 0.3 = poor
- 0.0 = completely wrong or irrelevant

Be strict but fair. Provide a brief explanation for each score.
```

**User Prompt Template:**
```
Please evaluate the following response.

## Original Prompt
{prompt}

## Response to Evaluate
{response}

## Scoring Criteria
1. correctness
2. clarity

## Instructions
Score each criterion 0.0 to 1.0. Respond ONLY in JSON format:
{
  "scores": {"criterion_name": score, ...},
  "explanation": "Brief explanation of scores",
  "overall_score": <average of all scores rounded to 2 decimals>
}
```

## Error Handling & Fallback

The AIJudge handles failures gracefully:

1. **JSON parse errors**: Attempts to extract JSON from markdown code blocks
2. **Malformed responses**: Uses heuristic fallback scoring
3. **API errors**: Falls back to heuristic scoring

**Heuristic Fallback Scoring:**

When the judge LLM fails, responses are scored based on:
- Length: up to 0.3 for responses up to 500 characters
- Structure: 0.2 for numbered lists, 0.2 for code blocks, 0.1 for multiple lines
- Maximum: 1.0

```python
result = judge.score(prompt, response)
# If judge fails, returns:
# {
#     "score": 0.4,
#     "scores": {"correctness": 0.4, "helpfulness": 0.4, "clarity": 0.4},
#     "explanation": "Fallback heuristic scoring (judge LLM unavailable)."
# }
```

## Comparison: Sandbox vs AI Judge Mode

| Aspect | Sandbox Mode | AI Judge Mode |
|--------|-------------|---------------|
| **Reward Source** | Code execution results | LLM evaluation |
| **Requires Docker** | Yes | No |
| **Requires GPU** | For training only | For training + judge |
| **Best For** | Code generation tasks | Open-ended generation |
| **Determinism** | Deterministic | Probabilistic (low temp helps) |
| **Setup Complexity** | Higher (Docker) | Lower |

## Complete Example: Full Training Pipeline

```python
from grpo_in_sandbox import (
    SelfPlayGRPO,
    RLHFTrainingConfig,
    AITrainerMode,
    AIJudge,
    ProductManager,
    train,
)

# Example 1: Using SelfPlayGRPO orchestrator

config = RLHFTrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    mode=AITrainerMode.AI_JUDGE,
    judge_llm_name="openai/gpt-4o-mini",
    judge_criteria=["correctness", "clarity", "helpfulness"],
    judge_temperature=0.1,
    num_train_epochs=3,
    max_steps=50,
    output_dir="./output_ai_judge",
)

sp = SelfPlayGRPO(config)
results = sp.run()

print(f"Training complete!")
print(f"Model saved to: {results.get('model_path', './output_ai_judge/final_model')}")
print(f"Final reward: {results.get('final_reward', 'N/A')}")

# Example 2: Using train() directly with custom judge

config = RLHFTrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    num_train_epochs=3,
    max_steps=50,
)

# Create custom judge with specific criteria
judge = AIJudge(
    llm_name="openai/gpt-4o-mini",
    criteria=["correctness", "completeness", "format"],
    temperature=0.1,
)

results = train(config, ai_judge=judge)

# Example 3: Using open-source judge via vLLM

config = RLHFTrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    mode=AITrainerMode.AI_JUDGE,
    judge_llm_name="qwen/qwen2.5-7b-instruct",
    judge_base_url="http://localhost:8000/v1",
    judge_criteria=["correctness", "helpfulness"],
    max_steps=50,
)

sp = SelfPlayGRPO(config)
results = sp.run()
```

## Tips & Best Practices

1. **Use low temperature (0.1)** for consistent judging — higher temperatures introduce variance in scores

2. **Start with broad criteria** like ["correctness", "helpfulness", "clarity"], then refine based on your task

3. **Try open-source judges** for cost savings:
   - "Qwen/Qwen2.5-7B-Instruct" via vLLM
   - "meta-llama/Llama-3-8B-Instruct" via vLLM

4. **Monitor judge output** for quality — check the explanation field to understand scoring

5. **Combine with sandbox** for hybrid rewards — use sandbox for tasks with verifiable answers, AI judge for open-ended aspects

6. **Configure judge_criteria in config** or pass to AIJudge constructor — the config values are used as defaults

7. **Set judge_base_url** for local models or custom API endpoints

8. **Handle judge failures gracefully** — the heuristic fallback ensures training continues even when the judge LLM is unavailable