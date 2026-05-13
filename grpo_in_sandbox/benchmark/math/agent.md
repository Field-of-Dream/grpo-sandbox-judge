# math

数学问题基准测试，使用 AIME 2025 竞赛题目。

## 评估指标

准确率 via `math-verify`。答案格式：`\boxed{answer}`

## 数据集

`daixuancheng/llm-in-sandbox-bench` (config: `math`)

## 文件

- [config.yaml](config.yaml) - 基准配置
- [reward.py](reward.py) - 奖励函数（使用 math-verify）
- [vanilla_llm_prompt.py](vanilla_llm_prompt.py) - 基础提示