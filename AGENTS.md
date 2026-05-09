# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-09
**Commit:** 31575de
**Branch:** agent_gpro
**Root:** grpo_in_sandbox/

## OVERVIEW
LLM-in-Sandbox enables LLMs to execute code in isolated Docker sandboxes for agentic capabilities. Provides tool-use for science, math, physics tasks via litellm (OpenAI, Anthropic, vLLM, SGLang).

## STRUCTURE
```
grpo_in_sandbox/
├── agent.py           # Core Agent - LLM orchestration
├── agent_configs.py  # Multi-agent team configuration system
├── runtime.py        # Runtime abstraction (Docker/Kaggle/Local)
├── docker_runtime.py # Docker container lifecycle
├── tools.py          # Tool definitions (JSON schema)
├── train.py         # GRPO training interface
├── benchmark/      # Evaluation tasks (math, physics, chem, biomed...)
└── config/        # Configuration (YAML configs)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|---------|-------|
| Agent logic | agent.py | Core class, tool execution |
| Agent configs | agent_configs.py | Multi-agent team system |
| Runtime backend | runtime.py, docker_runtime.py, kaggle_runtime.py | Backend abstraction |
| Tool definitions | tools.py | JSON schema for function calling |
| GRPO training | train.py, train_kaggle.py | RL training interface |
| Benchmarks | benchmark/{task}/ | Task-specific reward, prompts |

## CODE MAP
| Symbol | Type | Location | Role |
|---------|------|---------|-------|
| Agent | class | agent.py | Core orchestration |
| AgentProfile | class | agent_configs.py | Agent profile definition |
| AgentTeam | class | agent_configs.py | Multi-agent team management |
| create_runtime | function | runtime.py | Backend factory |
| DockerRuntime | class | docker_runtime.py | Container mgmt |
| Trajectory | class | trajectory.py | Execution history |

## CONVENTIONS
- **max_steps**: Default 30 conversation turns
- **Token limit**: Default 65536, auto-reduces on overflow
- **Trajectory**: Records thought/action/observation for RL training
- **Ruff**: Linter (line-length=100, target=py310)
- **mypy**: Strict type checking required

## ANTI-PATTERNS (THIS PROJECT)
- **File tools**: Only use `.py` and `.rst` files (context saving)
- **Type safety**: NEVER use `as any`, `@ts-ignore`

## UNIQUE STYLES
- Uses `rich` for colorful console output
- Docker containers cleaned via atexit handlers

## COMMANDS
```bash
llm-sandbox run --query "..." --llm_name openai/gpt-4
llm-sandbox build
python -m grpo_in_sandbox.benchmark.runner run_benchmark(...)
```

## NOTES
- Workspace has CLAUDE.md at parent level - check for project-wide guidance