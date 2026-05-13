# BENCHMARK TASKS

**Location:** benchmark/

## OVERVIEW
Evaluation tasks for LLM agents across scientific domains. Each subdomain runs independently via `BenchmarkRunner` in runner.py. Benchmarks use a non-package layout (no `__init__.py` in benchmark/ or subdirectories). Special domains (physics, long_context) use judge.py for LLM-as-judge evaluation.

## WHERE TO LOOK
| Task | Domain | Notes |
|------|--------|-------|
| runner.py | Execution | BenchmarkRunner, parallel execution |
| math | Math problems | Arithmetic, algebra, geometry |
| physics | Physics problems | Mechanics, electromagnetism |
| chem | Chemistry | Reactions, stoichiometry |
| biomed | Biomedical | Biochemical reasoning |
| long_context | Context handling | Extended prompts |
| instruct_follow | Instruction following | Task-specific instructions |
| instruct_pretrain | Pretraining | Model training tasks |
| demo | Demo/Testing | Testing harness |
| judge.py | LLM evaluation | Used in physics, long_context |

## CONVENTIONS
- Each task subdir: config.yaml, reward.py, vanilla_llm_prompt.py (3 files minimum)
- Sub-tasks per domain:
  - math: arithmetic, algebra, geometry
  - physics: mechanics, electromagnetism
  - chem: reactions, stoichiometry
  - biomed: biochemical reasoning
- Rewards: Task-specific scoring functions in reward.py
- Runner uses reward.py scoring functions per domain

## COMMANDS
```bash
python -m grpo_in_sandbox.benchmark.runner run_benchmark(...)
```

## ANTI-PATTERNS
- Avoid mutating shared state across benchmark tasks