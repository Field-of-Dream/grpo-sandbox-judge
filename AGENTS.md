# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-14
**Commit:** 1134fc01
**Branch:** main
**Root:** grpo_in_sandbox/

## OVERVIEW
LLM-in-Sandbox enables LLMs to execute code in isolated Docker sandboxes for agentic capabilities. Provides tool-use for science, math, physics tasks via litellm (OpenAI, Anthropic, vLLM, SGLang). Uses flat layout (no src/ dir) with grpo_in_sandbox/ as main package.

## STRUCTURE
```
grpo_in_sandbox/                   # Project root
├── grpo_in_sandbox.py             # Facade module (re-exports package)
├── grpo_in_sandbox/               # Main package (67 Python files)
│   ├── __init__.py                # Package init + exports
│   ├── agent.py                   # Core Agent - LLM orchestration
│   ├── agent_configs.py           # Multi-agent team configuration
│   ├── agent_factory.py           # Agent factory helpers
│   ├── action.py                  # Action representation
│   ├── observation.py             # Action results wrapper
│   ├── trajectory.py              # Execution history recording
│   ├── runtime.py                 # Runtime abstraction (Docker/Kaggle/Local)
│   ├── docker_runtime.py          # Docker container lifecycle
│   ├── kaggle_runtime.py          # Kaggle notebook runtime
│   ├── tools.py                   # Tool definitions (JSON schema)
│   ├── cli.py                     # CLI entry point (fire-based)
│   ├── train.py                   # GRPO training interface
│   ├── config.py                  # Configuration loader
│   ├── benchmark/                 # Evaluation tasks (has own AGENTS.md)
│   │   ├── runner.py
│   │   ├── math/
│   │   ├── physics/
│   │   ├── chem/
│   │   ├── biomed/
│   │   ├── long_context/
│   │   ├── instruct_follow/
│   │   ├── instruct_pretrain/
│   │   └── demo/
│   └── config/                    # YAML configs (6 files)
├── pyproject.toml
├── ruff.toml
└── .github/workflows/python-app.yml
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|---------|-------|
| Agent logic | agent.py | Core class, tool execution |
| Agent configs | agent_configs.py | Multi-agent team system |
| Runtime backend | runtime.py, docker_runtime.py, kaggle_runtime.py | Backend abstraction |
| Tool definitions | tools.py | JSON schema for function calling |
| GRPO training | train.py | RL training interface |
| CLI entry | cli.py | fire-based commands |
| Benchmarks | benchmark/runner.py | Task runner |
| Execution history | trajectory.py, action.py, observation.py | Step recording |

## CODE MAP
| Symbol | Type | Location | Role |
|---------|------|---------|-------|
| Agent | class | agent.py | Core orchestration |
| AgentArgs | class | agent.py | Agent configuration |
| AgentProfile | class | agent_configs.py | Agent profile definition |
| AgentTeam | class | agent_configs.py | Multi-agent team management |
| create_runtime | function | runtime.py | Backend factory |
| BaseRuntime | class | runtime.py | Runtime base class |
| DockerRuntime | class | docker_runtime.py | Container mgmt |
| KaggleRuntime | class | kaggle_runtime.py | Kaggle notebook exec |
| Trajectory | class | trajectory.py | Execution history |
| TrajectoryStep | class | trajectory.py | Single step record |
| Action | class | action.py | Tool call representation |
| Observation | class | observation.py | Tool result wrapper |
| run_agent_query | function | cli.py | CLI query runner |
| run_benchmark | function | cli.py | Benchmark runner |
| RLHFTrainingConfig | class | train.py | RL training config |

## CONVENTIONS
- **max_steps**: Default 30 conversation turns
- **Token limit**: Default 65536, auto-reduces on overflow
- **Trajectory**: Records thought/action/observation for RL training
- **Ruff**: Linter (line-length=100, target-version=py310)
- **mypy**: python_version=3.10, excludes benchmark/
- **EditorConfig**: charset=utf-8, end_of_line=lf, indent_style=space, indent_size=4

## ANTI-PATTERNS (THIS PROJECT)
- **File tools**: Only use `.py` and `.rst` files (context saving)
- **Benchmark subdirs**: No `__init__.py` in benchmark/{task}/ directories (intentional)
- **Flat layout**: No src/ directory, root-level modules
- **NEVER comments**: observation.py contains a "NEVER ASK FOR HUMAN HELP" directive

## UNIQUE STYLES
- Uses `rich` for colorful console output
- Docker containers cleaned via atexit handlers
- Flat layout (no src/ dir), no __init__.py in benchmark/tasks

## COMMANDS
```bash
llm-sandbox run --query "..." --llm_name openai/gpt-4
llm-sandbox build
llm-sandbox --help
python -m grpo_in_sandbox.cli run_agent_query(...)
python -m grpo_in_sandbox.benchmark.runner run_benchmark(...)
```

## NOTES
- Workspace has CLAUDE.md at parent level - check for project-wide guidance
- CLI entry point: cli.py provides fire-based commands (run, build, benchmark)
- Test config exists in pyproject.toml but no tests/ directory yet