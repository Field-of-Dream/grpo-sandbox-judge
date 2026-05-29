# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-27
**Commit:** 2bc405e
**Branch:** main
**Root:** grpo_in_sandbox/

## OVERVIEW
LLM-in-Sandbox enables LLMs to execute code in isolated Docker sandboxes for agentic capabilities. Provides tool-use for science, math, physics tasks via litellm (OpenAI, Anthropic, vLLM, SGLang). Uses flat layout (no src/ dir) with grpo_in_sandbox/ as main package (32 .py files, 5602 lines).

## STRUCTURE
```
grpo_in_sandbox/                   # Project root
├── grpo_in_sandbox/               # Main package (15 .py files)
│   ├── __init__.py                # Package init + lazy training imports
│   ├── agent.py                   # Core Agent (694 lines)
│   ├── agent_configs.py           # Multi-agent team system
│   ├── action.py                  # Tool call representation
│   ├── observation.py             # Action results wrapper
│   ├── trajectory.py              # Execution history (pydantic)
│   ├── runtime.py                 # Backend factory (BaseRuntime, create_runtime)
│   ├── docker_runtime.py          # Docker container lifecycle
│   ├── kaggle_runtime.py          # Kaggle notebook runtime
│   ├── tools.py                   # Tool JSON schemas (3 tools)
│   ├── cli.py                     # Fire-based CLI (725 lines)
│   ├── train.py                   # GRPO training interface (1213 lines)
│   ├── config.py                  # YAML config loader
│   ├── benchmark/                 # 8 evaluation tasks, runner.py, judge.py
│   └── config/                    # 6 YAML config files
├── docs/                          # Tutorials
├── .sisyphus/                     # Work plans, run-continuation state
├── pyproject.toml
├── ruff.toml                      # line-length=100, target=py310
├── .editorconfig
├── pyrightconfig.json          # basic mode, py310 target
├── .github/workflows/          # CI (python-app.yml) + release pipeline
├── DOCS.md
└── README.md
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|---------|-------|
| Agent logic | agent.py | Core class, tool execution loop, 694 lines |
| Agent teams | agent_configs.py | AgentTeam, AgentProfile, YAML import/export |
| Runtime factory | runtime.py | create_runtime() - docker/kaggle/local auto-detect |
| Docker runtime | docker_runtime.py | Container lifecycle, exec commands, file copy |
| Tool schemas | tools.py | 3 tools: execute_bash, str_replace_editor, submit |
| GRPO training | train.py | RLHFTrainingConfig, train() via unsloth GRPOTrainer |
| CLI | cli.py | run, build, benchmark commands (fire) |
| Benchmarks | benchmark/runner.py | Parallel evaluator + 8 task domains |
| LLM-as-Judge | benchmark/{physics,long_context}/judge.py | Scoring via judge LLM |
| Execution history | trajectory.py, action.py, observation.py | Thought/action/observation per step |

## CONVENTIONS
- **Flat layout**: No src/ dir, grpo_in_sandbox/ is the package root
- **Ruff**: line-length=100, target-version=py310, select=E/F/W/I/N/UP/B/C4/SIM/PYI, ignore=E501/B008/C901/SIM115
- **mypy**: python_version=3.10, ignores missing imports, excludes benchmark/
- **EditorConfig**: charset=utf-8, end_of_line=lf, indent_style=space, indent_size=4 (.py), indent_size=2 (.yaml/.json/.toml)
- **Pyright**: basic mode, scoped to grpo_in_sandbox/, excludes benchmark/
- **Pydantic v2**: Trajectory/TrajectoryStep use pydantic BaseModel
- **Rich console**: Colorful CLI output via rich library
- **Lazy imports**: torch/transformers/unsloth loaded on-demand via __getattr__
- **max_steps**: Default 30 turns; max_token_limit: 65536 default
- **No tests**: pytest configured but no tests/ directory exists (CI pytest step disabled)

## ANTI-PATTERNS (THIS PROJECT)
- **Benchmark subdirs**: No `__init__.py` in benchmark/{task}/ (intentional, avoids import mismatch)
- **NEVER ASK FOR HUMAN HELP**: observation.py directive (agent must always use tool calls)
- **No .py files outside .py/.rst**: View command blocks non-Python file viewing
- **Facade module removed**: grpo_in_sandbox.py no longer exists (was removed in refactor)

## KEY SYMBOLS
| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| Agent | class | agent.py | 30+ | Core LLM orchestration loop |
| AgentArgs | dataclass | agent.py | 15+ | Agent configuration (model, prompts) |
| AgentProfile | dataclass | agent_configs.py | 5+ | Single agent definition |
| AgentTeam | class | agent_configs.py | 3+ | Multi-agent team container |
| BaseRuntime | ABC | runtime.py | 4 | Runtime interface (run/demux_run/close) |
| DockerRuntime | class | docker_runtime.py | 6 | Docker sandbox backend |
| KaggleRuntime | class | kaggle_runtime.py | 3 | Kaggle notebook backend |
| LocalRuntime | class | runtime.py | 2 | Local subprocess backend |
| Trajectory | pydantic | trajectory.py | 5 | Full execution history |
| RLHFTrainingConfig | class | train.py | 3 | GRPO training parameters |
| AIJudge | class | train.py | 2 | LLM-as-judge scorer |

## COMMANDS
```bash
# CLI
llm-sandbox run --query "..." --llm_name openai/gpt-4
llm-sandbox build
llm-sandbox benchmark --task math --llm_name gpt-4

# Python API
python -m grpo_in_sandbox.cli run_agent_query(...)
python -m grpo_in_sandbox.benchmark.runner run_benchmark(...)

# Dev
ruff check grpo_in_sandbox
mypy grpo_in_sandbox
```

## NOTES
- **No tests exist**: pytest is configured in pyproject.toml (addopts="-v --strict-markers") but no test files have been created. CI pytest step is disabled.
- **Known bugs**: See earlier session analysis - 17 bugs found (3 critical: self-KL-comparison in train.py, missing SelfPlayGRPO/CodeExecutor exports, wrong torch API names).
- **Sisyphus work plans** stored in `.sisyphus/plans/` for automatic execution.
- **Python version**: Requires Python >= 3.10. Targets py310 for mypy/ruff.
- **Docker image**: Default `cdx123/llm-in-sandbox:v0.1`, configured in cli.py.