# BENCHMARK TASKS

**Location:** benchmark/

## OVERVIEW
Evaluation tasks for LLM agents across scientific domains.

## WHERE TO LOOK
| Task | Domain | Notes |
|------|--------|-------|
| math | Math problems | Arithmetic, algebra, geometry |
| physics | Physics problems | Mechanics, electromagnetism |
| chem | Chemistry | Reactions, stoichiometry |
| biomed | Biomedical | Biochemical reasoning |
| long_context | Context handling | Extended prompts |
| instruct_follow | Instruction following | Task-specific instructions |
| instruct_pretrain | Pretraining | Model training tasks |
| demo | Demo/Testing | Testing harness |

## CONVENTIONS
- Each task: config.yaml, reward.py, vanilla_llm_prompt.py
- Rewards: Task-specific scoring functions