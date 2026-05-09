# instruct_pretrain

指令预训练数据基准，用于 RL 训练。

> 注意：此基准用于 LLM-in-Sandbox-RL 训练，不用于评测。

## 评估指标

根据问题类型路由：
- 单选题：准确率
- 多选题：F1 分数
- 开放式：数学验证 + ROUGE-L

## 数据集

`daixuancheng/llm-in-sandbox-rl` (config: `instruct_pretrain`)

## 文件

- [config.yaml](config.yaml) - 基准配置
- [reward.py](reward.py) - 奖励函数（路由到其他 scorer）
- [vanilla_llm_prompt.py](vanilla_llm_prompt.py) - 基础提示