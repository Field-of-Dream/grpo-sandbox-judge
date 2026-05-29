# BENCHMARK TASKS

**Location:** benchmark/

## OVERVIEW
Evaluation tasks for LLM agents across 8 scientific domains. Each subdomain runs independently via `ProcessPoolExecutor` in runner.py (700 lines). Benchmarks use a non-package layout (no `__init__.py` in benchmark/ or subdirectories — intentional). Two modes: `llm-in-sandbox` (Docker sandbox) and `llm` (vanilla API). Special domains (physics, long_context) use judge.py for LLM-as-judge scoring.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Runner | benchmark/runner.py | BenchmarkRunner, parallel executor, 700 lines |
| math | benchmark/math/ | Arithmetic, algebra, geometry via reward.py |
| physics | benchmark/physics/ | Physics problems + judge.py LLM-as-judge |
| chem | benchmark/chem/ | Chemical reactions, stoichiometry |
| biomed | benchmark/biomed/ | Biochemical reasoning |
| long_context | benchmark/long_context/ | Extended context + judge.py |
| instruct_follow | benchmark/instruct_follow/ | Instruction following tasks |
| instruct_pretrain | benchmark/instruct_pretrain/ | Model training tasks |
| demo | benchmark/demo/ | Testing harness with data.json |
| judge.py | physics/judge.py, long_context/judge.py | LLM-as-judge scoring (Qwen3) |

## CONVENTIONS
- Each task subdir: config.yaml + reward.py + vanilla_llm_prompt.py + agent.md + README.md
- No `__init__.py` in any benchmark/ subdir (import uses spec_from_file_location)
- Reward functions: each domain has `compute_score(answer, ground_truth) -> float`
- Config: HuggingFace `dataset` name, `split`, optional `config` in config.yaml
- Two run modes: `--mode llm-in-sandbox` (default, Docker agent) or `--mode llm` (direct API)

## COMMANDS
```bash
llm-sandbox benchmark --task math --llm_name gpt-4
python -m grpo_in_sandbox.benchmark.runner run_benchmark(task="math", ...)
# LLM-as-Judge (post-processing):
python benchmark/physics/judge.py --input <trajectory.json> --judge_model qwen3-235B
```

## ANTI-PATTERNS
- Avoid mutating shared state across benchmark tasks (ProcessPoolExecutor isolation)