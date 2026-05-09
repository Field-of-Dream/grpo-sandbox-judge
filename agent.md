# grpo_in_sandbox

主源代码目录，包含 Agent、Docker 运行时、CLI 等核心模块。

## 目录结构

| 文件 | 描述 |
|------|------|
| [agent.py](agent.py) | Agent 核心逻辑 |
| [cli.py](cli.py) | 命令行入口 |
| [docker_runtime.py](docker_runtime.py) | Docker 沙箱运行时 |
| [tools.py](tools.py) | 工具定义 |
| [observation.py](observation.py) | 观察/状态处理 |
| [trajectory.py](trajectory.py) | 轨迹记录 |
| [action.py](action.py) | 动作定义 |

## 子目录

- [benchmark/](benchmark/) - 基准测试
- [config/](config/) - 配置