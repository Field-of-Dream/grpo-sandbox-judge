# chem

化学分析基准测试，使用 ChemBench4K 多选题。

## 评估指标

准确率（支持 `\boxed{A}`, `Answer: A` 等格式）

## 数据集

`daixuancheng/llm-in-sandbox-bench` (config: `chem`)

## 文件

- [config.yaml](config.yaml) - 基准配置
- [reward.py](reward.py) - 奖励函数
- [vanilla_llm_prompt.py](vanilla_llm_prompt.py) - 基础提示