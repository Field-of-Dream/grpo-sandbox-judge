# AI Judge 自我对弈 GRPO — API 教程

本教程涵盖了不使用沙箱执行的 GRPO 语言模型训练的 AI Judge 模式。奖励信号来自 LLM-as-judge，而不是代码执行，这使得自我对弈训练可用于开放式生成任务。

## 概述

AI Judge 自我对弈模式用 LLM 评估替代了基于沙箱的奖励评分。不是运行代码来验证正确性，而是 AI judge 模型根据可定制的标准（如正确性、清晰度和有用性）对响应进行评分。这使得模型能够在没有确定性答案的任务上进行训练，如写作、总结或指令遵循。

## 架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  生成器模型      │     │   AI Judge       │     │    GRPO         │
│  (待训练)        │────▶│   LLM 评判模型   │────▶│    训练器        │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ▲                        │                        │
        │                        ▼                        │
        │                分数 (0.0-1.0)                   │
        │                奖励信号                          │
        └────────────────────────────────────────────┘
```

训练循环的工作原理如下：
1. **生成**：生成器模型对提示生成响应
2. **评判**：AI judge 根据标准评估每个响应
3. **训练**：GRPO 使用 judge 分数作为奖励信号来更新生成器

## 快速开始

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
print(f"最终奖励: {results.get('final_reward', 'N/A')}")
```

## API 参考

### 1. AITrainerMode

控制使用哪种奖励机制进行训练的枚举。

```python
from grpo_in_sandbox import AITrainerMode

AITrainerMode.SANDBOX   # 使用沙箱执行（默认）
AITrainerMode.AI_JUDGE   # 使用 LLM-as-judge 进行奖励
```

### 2. RLHFTrainingConfig — 新字段

AI Judge 模式的扩展配置。新字段（train.py 第 191-196 行）：

```python
@dataclass
class RLHFTrainingConfig:
    # ... 现有字段 ...

    # AI Judge 模式（用于自我对弈 GRPO）
    mode: AITrainerMode = AITrainerMode.SANDBOX  # 'sandbox' 或 'ai_judge'
    judge_llm_name: str = "openai/gpt-4o-mini"  # 用作评判的 LLM
    judge_criteria: list[str] | None = None  # 例如 ["correctness", "clarity", "helpfulness"]
    judge_temperature: float = 0.1  # 低温度以保证评判的一致性
    judge_base_url: str | None = None  # 评判模型的自定义 API 端点
```

| 字段 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `mode` | AITrainerMode | SANDBOX | 在沙箱和 AI judge 之间切换 |
| `judge_llm_name` | str | "openai/gpt-4o-mini" | 评判 LLM 的模型 ID |
| `judge_criteria` | list[str] \| None | None | 评分维度 |
| `judge_temperature` | float | 0.1 | 低温度以保证评判的一致性 |
| `judge_base_url` | str \| None | None | 评判模型的自定义 API 端点 |

### 3. AIJudge

LLM-as-Judge 类，使用 AI 模型对响应进行评分。

**构造函数：**

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

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `llm_name` | str | "openai/gpt-4o-mini" | 评判模型标识符 |
| `criteria` | list[str] \| None | ["correctness", "helpfulness", "clarity"] | 评分标准 |
| `temperature` | float | 0.1 | 低温度以保证评判的一致性 |
| `base_url` | str \| None | None | 自定义 API 端点 |

**方法：**

`score(prompt, response, criteria=None)` -> dict

对响应进行评分，返回包含以下键的字典：
- `score`: float (0.0-1.0) — 标准的平均值
- `scores`: dict — 每个标准的分数
- `explanation`: str — 简短解释

**示例：**

```python
from grpo_in_sandbox import AIJudge

judge = AIJudge(llm_name="openai/gpt-4o-mini")

result = judge.score(
    prompt="什么是 2+2?",
    response="答案是 4。2 加 2 的和是 4。"
)

print(result["score"])  # 例如 0.92
print(result["scores"])  # {"correctness": 1.0, "helpfulness": 0.83, "clarity": 0.92}
print(result["explanation"])
```

`_heuristic_fallback(prompt, response, criteria)` -> dict

当 judge LLM 失败时的备用评分。考虑：
- 响应长度（较长 = 更多努力，最多 500 个字符）
- 结构（编号列表、代码块、换行符）
- 最大分数：1.0

```python
result = judge._heuristic_fallback(
    prompt="什么是 2+2?",
    response="答案是 4。",
    criteria=["correctness", "helpfulness"]
)
# 根据长度和结构返回备用分数
```

### 4. AIJudgeRewardModel

为 GRPOTrainer 兼容性封装 AIJudge。匹配奖励函数接口。

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
        # 返回分数列表，每个完成对应一个分数
```

**示例：**

```python
from grpo_in_sandbox import AIJudge, AIJudgeRewardModel

judge = AIJudge(llm_name="openai/gpt-4o-mini")
reward_fn = AIJudgeRewardModel(judge, criteria=["correctness", "clarity"])

scores = reward_fn(
    completions=["响应 A", "响应 B"],
    prompts=["提示 X", "提示 Y"]
)
# 返回: [0.85, 0.72]
```

### 5. SelfPlayGRPO

生成 → 评判 → 训练循环的编排器。

```python
class SelfPlayGRPO:
    def __init__(
        self,
        config: RLHFTrainingConfig,
        product_manager: ProductManager | None = None,
        judge: AIJudge | None = None,
    ):
```

**方法：**

`run()` -> dict

运行完整的自我对弈训练循环并返回结果。

**示例：**

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

### 6. train() 函数

`train()` 函数现在接受可选的 `ai_judge` 参数：

```python
def train(
    config: RLHFTrainingConfig,
    product_manager: ProductManager | None = None,
    reward_model: RewardModel | None = None,
    ai_judge: AIJudge | None = None,
) -> dict[str, Any]:
```

当 `config.mode == AITrainerMode.AI_JUDGE` 或 `ai_judge` 被提供时，该函数使用 AI judge 进行奖励而不是沙箱执行。

**示例：**

```python
from grpo_in_sandbox import train, RLHFTrainingConfig, AIJudge

# 方法 1：通过配置
config = RLHFTrainingConfig(
    mode=AITrainerMode.AI_JUDGE,
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    judge_llm_name="openai/gpt-4o-mini",
    max_steps=50,
)
results = train(config)

# 方法 2：通过 ai_judge 参数
judge = AIJudge(llm_name="openai/gpt-4o-mini")
results = train(config, ai_judge=judge)
```

## 自定义评判模型

### 自定义标准

定义您自己的评分维度：

```python
judge = AIJudge(
    llm_name="openai/gpt-4o-mini",
    criteria=["correctness", "completeness", "safety", "format"]
)

result = judge.score(
    prompt="解释光合作用",
    response="光合作用是植物通过...的过程"
)
# 按照以下标准评分：正确性、完整性、安全性、格式
```

### 自定义评判模型

使用任何 litellm 兼容的模型：

```python
# 通过 vLLM 的开源模型
judge = AIJudge(
    llm_name="local/llama-3-8b",
    base_url="http://localhost:8000/v1"
)

# Anthropic 模型
judge = AIJudge(llm_name="anthropic/claude-3-sonnet")

# Azure OpenAI
judge = AIJudge(
    llm_name="azure/gpt-4",
    base_url="https://your-resource.openai.azure.com/"
)
```

### 自定义 API 端点

指向您自己的 API 服务器：

```python
judge = AIJudge(
    llm_name="openai/gpt-4o-mini",
    base_url="https://api.example.com/v1"
)
```

## 评判提示结构

AI Judge 使用结构化提示来评估响应：

**系统提示：**
```
你是一位专业的评估者。你的工作是根据给定的标准对 AI 响应进行评分。
请根据以下标准对每个标准进行 0.0 到 1.0 的评分：
- 1.0 = 完美
- 0.7 = 很好
- 0.5 = 可以接受
- 0.3 = 较差
- 0.0 = 完全错误或不相关

要严格但公正。为每个分数提供简短说明。
```

**用户提示模板：**
```
请评估以下响应。

## 原始提示
{prompt}

## 要评估的响应
{response}

## 评分标准
1. 正确性
2. 清晰度

## 说明
对每个标准进行 0.0 到 1.0 的评分。仅用 JSON 格式响应：
{
  "scores": {"criterion_name": score, ...},
  "explanation": "分数的简短解释",
  "overall_score": <所有分数的平均值，四舍五入到 2 位小数>
}
```

## 错误处理与备用方案

AIJudge 可以优雅地处理失败：

1. **JSON 解析错误**：尝试从 markdown 代码块中提取 JSON
2. **格式错误的响应**：使用启发式备用评分
3. **API 错误**：回退到启发式评分

**启发式备用评分：**

当 judge LLM 失败时，响应基于以下因素进行评分：
- 长度：最多 500 个字符的响应得分最多 0.3
- 结构：编号列表得 0.2，代码块得 0.2，多行得 0.1
- 最大值：1.0

```python
result = judge.score(prompt, response)
# 如果 judge 失败，返回：
# {
#     "score": 0.4,
#     "scores": {"correctness": 0.4, "helpfulness": 0.4, "clarity": 0.4},
#     "explanation": "启发式备用评分（judge LLM 不可用）。"
# }
```

## 对比：沙箱模式 vs AI Judge 模式

| 方面 | 沙箱模式 | AI Judge 模式 |
|------|----------|---------------|
| **奖励来源** | 代码执行结果 | LLM 评估 |
| **需要 Docker** | 是 | 否 |
| **需要 GPU** | 仅训练 | 训练 + 评判 |
| **最适合** | 代码生成任务 | 开放式生成 |
| **确定性** | 确定性 | 概率性（低温度有帮助） |
| **设置复杂度** | 更高（Docker） | 更低 |

## 完整示例：完整训练管道

```python
from grpo_in_sandbox import (
    SelfPlayGRPO,
    RLHFTrainingConfig,
    AITrainerMode,
    AIJudge,
    ProductManager,
    train,
)

# 示例 1：使用 SelfPlayGRPO 编排器

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

print(f"训练完成！")
print(f"模型保存到: {results.get('model_path', './output_ai_judge/final_model')}")
print(f"最终奖励: {results.get('final_reward', 'N/A')}")

# 示例 2：直接使用 train() 和自定义 judge

config = RLHFTrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    num_train_epochs=3,
    max_steps=50,
)

# 创建具有特定标准的自定义 judge
judge = AIJudge(
    llm_name="openai/gpt-4o-mini",
    criteria=["correctness", "completeness", "format"],
    temperature=0.1,
)

results = train(config, ai_judge=judge)

# 示例 3：使用通过 vLLM 的开源 judge

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

## 提示与最佳实践

1. **使用低温度（0.1）**以保证一致的评判 — 较高的温度会在分数中引入方差

2. **从广泛的标准开始**，如["correctness", "helpfulness", "clarity"]，然后根据您的任务进行优化

3. **尝试开源评判器**以节省成本：
   - "Qwen/Qwen2.5-7B-Instruct" 通过 vLLM
   - "meta-llama/Llama-3-8B-Instruct" 通过 vLLM

4. **监控评判输出**质量 — 检查说明字段以了解评分

5. **与沙箱结合**进行混合奖励 — 对具有可验证答案的任务使用沙箱，对开放式方面使用 AI judge

6. **在配置中配置 judge_criteria** 或传递给 AIJudge 构造函数 — 配置值用作默认值

7. **为本地模型或自定义 API 端点设置 judge_base_url**

8. **优雅地处理 judge 失败** — 启发式备用方案确保即使在 judge LLM 不可用时训练也能继续
