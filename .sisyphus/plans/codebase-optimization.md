# Codebase Optimization - Delete Redundancies & Restructure

## TL;DR

> **Quick Summary**: Remove 3 completely redundant files (grpo_in_sandbox.py, agent_factory.py, agent.md), consolidate ruff config into one file, deduplicate `get_logger()` across 4 files into a shared utility, and fix DockerRuntime/KaggleRuntime to properly inherit BaseRuntime.
> 
> **Deliverables**: Delete 3 files, merge/refactor 5 files
> **Estimated Effort**: Medium
> **Parallel Execution**: Some tasks are parallel (deletions), some sequential (refactoring)
> **Commit Strategy**: 2 commits (deletions first, then refactoring)

---

## Context

### Original Request
"优化整体代码，结构删除多余部分" — optimize overall code, delete redundant parts.

### Diagnosis
Full codebase audit completed. Key findings:

#### 1. FULLY REDUNDANT FILES (safe to delete)
- `grpo_in_sandbox.py` — Pure re-export of `__init__.py`, nothing imports from it
- `agent_factory.py` — Pure re-export from `agent.py`, nothing imports from it  
- `agent.md` — 20-line table, content covered by `AGENTS.md`

#### 2. CONFIG SPLIT
- `ruff.toml` and `pyproject.toml` both have ruff config with different settings. `ruff.toml` takes precedence, making pyproject's `[tool.ruff]` dead code. Need to consolidate into pyproject.toml.

#### 3. DUPLICATE CODE
- `get_logger()` defined identically in: `runtime.py`, `docker_runtime.py`, `kaggle_runtime.py`, `agent.py` — 4 copies
- `DockerRuntime` and `KaggleRuntime` don't extend `BaseRuntime` from `runtime.py`, causing interface drift and code duplication

---

## Work Objectives

### Core Objective
Clean up codebase by removing dead files, consolidating configs, and deduplicating code.

### Must Have
- Delete `grpo_in_sandbox.py` (redundant facade)
- Delete `agent_factory.py` (redundant re-export)
- Delete `agent.md` (redundant doc)
- Consolidate ruff config into `pyproject.toml`, delete `ruff.toml`
- Extract shared `get_logger()` into `runtime.py`, remove from other files
- Refactor `DockerRuntime` and `KaggleRuntime` to inherit from `BaseRuntime`

### Must NOT Have
- Do NOT change any business logic
- Do NOT change any public API
- Do NOT break existing imports
- Do NOT restructure to src-layout (too invasive)
- Do NOT touch `train.py` (core training module, high risk)

---

## Tasks

### Wave 1: Safe Deletions & Config Fix (Parallel)

#### Task 1: Delete `grpo_in_sandbox.py`
- **File**: `grpo_in_sandbox.py`
- **Action**: Delete the file
- **Reason**: 100% redundant facade. Everything re-exported is already in `__init__.py`.
- **Evidence**: grep for `from grpo_in_sandbox import` and `import grpo_in_sandbox` returns only docstrings/examples, no real imports.
- **Verification**: `python -c "from __init__ import __version__; print(__version__)"` still works

#### Task 2: Delete `agent_factory.py`
- **File**: `agent_factory.py`
- **Action**: Delete the file
- **Reason**: 100% redundant re-export from `agent.py`. `from agent_factory import` appears nowhere in the codebase.
- **Verification**: `python -c "from agent import Agent, create_agent"` still works

#### Task 3: Delete `agent.md`
- **File**: `agent.md`
- **Action**: Delete the file
- **Reason**: 20-line doc, content already covered by `AGENTS.md` and `README.md`

#### Task 4: Consolidate ruff config
- **File**: `ruff.toml` and `pyproject.toml`
- **Action**: 
  1. Copy all settings from `ruff.toml` (which has more complete config including `[format]` and `[lint.isort]`) into `pyproject.toml` under `[tool.ruff]`
  2. Ensure `pyproject.toml` has: all ignore rules, all per-file-ignores, isort config, format config
  3. Delete `ruff.toml`
- **Verification**: `ruff check .` passes, `ruff format --check .` passes

### Wave 2: Code Deduplication (Sequential after Wave 1)

#### Task 5: Extract shared `get_logger()` to runtime.py
- **Files**: `runtime.py`, `docker_runtime.py`, `kaggle_runtime.py`, `agent.py`
- **Action**:
  1. Keep the `_setup_logger()` function in `runtime.py` (already there)
  2. In `docker_runtime.py`: Remove the local `get_logger()` and import `_setup_logger` from runtime
  3. In `kaggle_runtime.py`: Remove the local `get_logger()` and import `_setup_logger` from runtime
  4. In `agent.py`: Keep the RichHandler-based `get_logger()` since it's richer (RichHandler with colors)
- **Reason**: `get_logger()` duplicated identically in 3 runtime files
- **Verification**: `python -c "from runtime import _setup_logger"` works

#### Task 6: Refactor DockerRuntime to inherit BaseRuntime
- **Files**: `docker_runtime.py`, `runtime.py`
- **Action**:
  1. Change `class DockerRuntime:` to `class DockerRuntime(BaseRuntime):`
  2. Ensure all abstract methods from `BaseRuntime` are implemented: `run()`, `demux_run()`, `close()`, `copy_to_container()`, `copy_from_container()`
  3. `run()` method already exists — ensure signature matches
  4. Currently missing `demux_run()` — need to add (or add to BaseRuntime as optional)
- **Must NOT do**: Don't break the `DockerRuntime` construction pattern (different signature from BaseRuntime.init)
- **Verification**: `python -c "from docker_runtime import DockerRuntime; from runtime import BaseRuntime; assert issubclass(DockerRuntime, BaseRuntime)"`

#### Task 7: Refactor KaggleRuntime to inherit BaseRuntime
- **Files**: `kaggle_runtime.py`, `runtime.py`
- **Action**: Same pattern as Task 6
  1. Change `class KaggleRuntime:` to `class KaggleRuntime(BaseRuntime):`
  2. Add missing method stubs for `copy_to_container`, `copy_from_container` if needed
  3. Ensure `run()` and `demux_run()` signatures match
- **Verification**: `python -c "from kaggle_runtime import KaggleRuntime; from runtime import BaseRuntime; assert issubclass(KaggleRuntime, BaseRuntime)"`

---

## Verification Strategy

### After Wave 1
```bash
ruff check .            # Must pass (ruff config consolidated)
python --version         # Must still work
python -c "from __init__ import Agent, train; print('OK')"  # Imports still work
```

### After Wave 2
```bash
python -c "from runtime import _setup_logger; print('OK')"
python -c "from docker_runtime import DockerRuntime; from runtime import BaseRuntime; assert issubclass(DockerRuntime, BaseRuntime); print('OK')"
python -c "from kaggle_runtime import KaggleRuntime; assert issubclass(KaggleRuntime, BaseRuntime); print('OK')"
mypy -p grpo_in_sandbox  # Must pass (if mypy available)
```

---

## TODOs

  - [x] 1. Delete `grpo_in_sandbox.py`
  - [ ] 2. Delete `agent_factory.py`
  - [ ] 3. Delete `agent.md`
  - [ ] 4. Consolidate ruff config into `pyproject.toml` and delete `ruff.toml`
  - [ ] 5. Extract shared `get_logger()` into `runtime.py`, remove duplicates from `docker_runtime.py` and `kaggle_runtime.py`
  - [ ] 6. Refactor `DockerRuntime` to inherit `BaseRuntime`
  - [ ] 7. Refactor `KaggleRuntime` to inherit `BaseRuntime`

---

## Final Verification Wave

- [ ] F1. All imports work: `python -c "from __init__ import Agent, train, BaseRuntime, DockerRuntime, KaggleRuntime, create_runtime; print('ALL IMPORTS OK')"`
- [ ] F2. Ruff passes: `ruff check .` exit 0
- [ ] F3. Runtime inheritance verified: DockerRuntime and KaggleRuntime are proper BaseRuntime subclasses

---

## Commit Strategy

**Commit 1 — Cleanup** (Tasks 1-4):
```
chore: remove redundant files and consolidate ruff config

- Delete grpo_in_sandbox.py (redundant facade)
- Delete agent_factory.py (redundant re-export)
- Delete agent.md (redundant doc)
- Consolidate ruff config into pyproject.toml, remove ruff.toml
```

**Commit 2 — Refactor** (Tasks 5-7):
```
refactor: deduplicate get_logger and align runtime class hierarchy

- Extract shared _setup_logger utility from runtime.py
- Remove duplicated get_logger from DockerRuntime and KaggleRuntime
- Make DockerRuntime and KaggleRuntime inherit from BaseRuntime
```
