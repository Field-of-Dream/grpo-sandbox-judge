# instruct_follow

指令跟随基准测试，使用 IFBench 评估指令执行能力。

## 评估指标

IFBench loose pass rate

## 数据集

`daixuancheng/llm-in-sandbox-bench` (config: `instruct_follow`)

## 文件

- [config.yaml](config.yaml) - 基准配置
- [reward.py](reward.py) - 奖励函数
- [vanilla_llm_prompt.py](vanilla_llm_prompt.py) - 基础提示