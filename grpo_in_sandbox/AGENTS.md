# MAIN PACKAGE

**Location:** grpo_in_sandbox/

## OVERVIEW
Core package implementing LLM agent orchestration (694 lines), runtime abstraction (Docker/Kaggle/Local), and GRPO training (1213 lines). Exports public API via __init__.py. 15 .py files, 5602 total lines across package.

## STRUCTURE
```
grpo_in_sandbox/
├── __init__.py           # Public API exports, lazy training imports
├── agent.py              # Core Agent class (694 lines)
├── agent_configs.py      # Multi-agent team system (256 lines)
├── action.py             # Tool call representation
├── observation.py        # Action result wrapper
├── trajectory.py         # Execution history (pydantic)
├── runtime.py            # Backend factory (BaseRuntime, create_runtime)
├── docker_runtime.py     # Docker container lifecycle (244 lines)
├── kaggle_runtime.py     # Kaggle notebook exec
├── tools.py              # JSON schema tool definitions (3 tools)
├── cli.py                # Fire-based CLI (725 lines)
├── train.py              # GRPO training interface (1213 lines)
├── config.py             # YAML config loader
├── benchmark/            # Evaluation tasks (has own AGENTS.md)
└── config/               # YAML configs (6 files)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|---------|-------|
| Agent logic | agent.py | Agent class, tool execution loop, model_query |
| Agent team | agent_configs.py | AgentTeam, AgentProfile, YAML import/export |
| Runtime factory | runtime.py | create_runtime() auto-detects docker/kaggle/local |
| Docker runtime | docker_runtime.py | Container lifecycle, exec_run, file copy via tar |
| Tool schema | tools.py | 3 JSON tools: execute_bash, str_replace_editor, submit |
| GRPO training | train.py | RLHFTrainingConfig, train(), RewardModel, AIJudge |
| CLI | cli.py | run_agent_query, run_benchmark, build_docker_image |
| Execution history | trajectory.py, action.py, observation.py | Step recording |

## CONVENTIONS
- **Lazy training imports**: torch/transformers/unsloth loaded via __init__.py __getattr__
- **Runtime abstraction**: BaseRuntime ABC → DockerRuntime/KaggleRuntime/LocalRuntime
- **Trajectory**: Pydantic v2 BaseModel for TrajectoryStep/Trajectory
- **Rich console**: Colorful output via rich library, Panel/Table/rule formatting
- **max_steps**: Default 30 turns; max_token_limit: 65536

## ANTI-PATTERNS
- **observation.py**: NEVER ASK FOR HUMAN HELP (agent must use tool calls, not ask user)
- **No __init__.py in benchmark subdirs** (intentional, avoids import mismatch)
- **No facade module**: grpo_in_sandbox.py was removed — use __init__.py exports
- **Non-Python view blocked**: str_replace_editor view only allows .py/.rst files
