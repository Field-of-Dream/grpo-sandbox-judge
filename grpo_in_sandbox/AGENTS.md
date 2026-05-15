# MAIN PACKAGE

**Location:** grpo_in_sandbox/

## OVERVIEW
Core package implementing LLM agent orchestration, runtime abstraction (Docker/Kaggle), and GRPO training. Exports public API via __init__.py.

## STRUCTURE
```
grpo_in_sandbox/
├── __init__.py           # Public API exports
├── agent.py              # Core Agent class
├── agent_configs.py      # Multi-agent team system
├── agent_factory.py      # Agent factory helpers
├── action.py             # Tool call representation
├── observation.py         # Action result wrapper
├── trajectory.py          # Execution history
├── runtime.py             # Backend factory (BaseRuntime, create_runtime)
├── docker_runtime.py      # Docker container lifecycle
├── kaggle_runtime.py      # Kaggle notebook exec
├── tools.py              # JSON schema tool definitions
├── cli.py                 # Fire-based CLI (run, build, benchmark)
├── train.py               # GRPO training interface
├── config.py              # Config loader
├── benchmark/             # Evaluation tasks (has own AGENTS.md)
└── config/                # YAML configs
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|---------|-------|
| Agent logic | agent.py | Agent class, tool execution loop |
| Agent team | agent_configs.py | AgentTeam, AgentProfile |
| Runtime backends | runtime.py, docker_runtime.py, kaggle_runtime.py | create_runtime factory |
| Tool schema | tools.py | JSON schema for function calling |
| GRPO training | train.py | RLHFTrainingConfig, train() |
| CLI | cli.py | run_agent_query, run_benchmark |
| Execution history | trajectory.py, action.py, observation.py | Step recording |

## CONVENTIONS
- **Facade pattern**: grpo_in_sandbox.py at root re-exports everything
- **Runtime abstraction**: BaseRuntime → DockerRuntime/KaggleRuntime/LocalRuntime
- **Trajectory recording**: Captures thought/action/observation per step

## ANTI-PATTERNS
- **observation.py**: NEVER ASK FOR HUMAN HELP directive
- **No __init__.py in benchmark subdirs** (intentional)
