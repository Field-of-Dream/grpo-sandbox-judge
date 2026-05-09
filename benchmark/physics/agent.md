# physics

物理推理基准测试，使用 UGPhysics 本科物理题目。

## 评估指标

LLM-as-a-Judge 准确率（需要第二阶段 judge.py 评分）

## 数据集

`daixuancheng/llm-in-sandbox-bench` (config: `physics`)

## 两阶段评估

1. 运行 `llm-in-sandbox benchmark --task physics` 生成轨迹
2. 使用 `python judge.py --input trajectory.json` 进行最终评分

## 文件

- [config.yaml](config.yaml) - 基准配置
- [reward.py](reward.py) - 奖励函数
- [judge.py](judge.py) - 判断函数（需要 Qwen3-235B-A22B-Instruct-2507）
- [vanilla_llm_prompt.py](vanilla_llm_prompt.py) - 基础提示