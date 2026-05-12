# Fix CI Type Check (mypy) and Lint (ruff) for Flat Layout

## TL;DR

> **Quick Summary**: GitHub Actions CI fails at `type check` step (exit code 2) because `mypy grpo_in_sandbox` looks for a `grpo_in_sandbox/` subdirectory that doesn't exist. The project uses a flat layout (`.py` files at root level), not a subdirectory. Fix by changing `mypy` to use `-p` (Python module import) and `ruff` to use proper path.
> 
> **Deliverables**:
> - 1 line change in `.github/workflows/python-app.yml` (mypy command)
> - 1 line change in `.github/workflows/python-app.yml` (ruff command)
> 
> **Estimated Effort**: Quick
> **Parallel Execution**: NO (single file, 2 edits)
> **Critical Path**: Edit → Commit → Push

---

## Context

### Original Request
User reports GitHub Actions CI fails with "process completed with exit code 2" at the type check step.

### Diagnosis
The project uses a **flat layout** — all Python source files are at the repository root level:

```
repo/
├── __init__.py
├── agent.py
├── train.py
├── runtime.py
├── ...
└── .github/workflows/python-app.yml
```

There is **no** `grpo_in_sandbox/` subdirectory.

The CI workflow runs:
```yaml
mypy grpo_in_sandbox    # ❌ mypy looks for ./grpo_in_sandbox/ → NOT FOUND → exit 2
ruff check grpo_in_sandbox  # ❌ same issue
```

Both commands expect a filesystem path, but the flat layout has no such subdirectory.

---

## Work Objectives

### Core Objective
Fix `type check` and `lint` steps in CI so they correctly analyze the flat-layout source files.

### Concrete Deliverables
- `mypy -p grpo_in_sandbox` (uses Python module import instead of filesystem path)
- `ruff check . --exclude build,dist,.github,__pycache__` or just `ruff check *.py`

### Must Have
- Type check step completes with exit code 0 (no type errors)
- Lint step completes successfully
- No changes to source code or project structure

### Must NOT Have
- No restructuring to src-layout (too invasive)
- No changes to `pyproject.toml`

---

## Verification Strategy

### QA Policy
CI pipeline will verify: after fix, the `type check` and `lint` steps should pass.

---

## Execution Strategy

### Single Wave

```
Wave 1 (Edit workflow file):
├── Task 1: Fix mypy command (mypy grpo_in_sandbox → mypy -p grpo_in_sandbox)
└── Task 2: Fix ruff command (ruff check grpo_in_sandbox → ruff check .)
```

---

## TODOs

- [ ] 1. Fix `mypy` command in CI workflow

  **What to do**:
  - Edit `.github/workflows/python-app.yml` line 26
  - Change `mypy grpo_in_sandbox` to `mypy -p grpo_in_sandbox`
  - The `-p` flag tells mypy to resolve via Python module import (uses installed package) instead of filesystem path

  **Why this fix works**:
  After `pip install -e .`, the package is importable as `grpo_in_sandbox` via Python. `mypy -p` uses Python's import mechanism, so it finds the installed package correctly even though there's no `grpo_in_sandbox/` directory on disk.

  **Must NOT do**:
  - Do not change `mypy .` — this would check `.github/`, `build/`, etc.
  - Do not restructure to src-layout

  **Parallelization**:
  - **Can Run In Parallel**: NO (single file)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 2
  - **Blocked By**: None

  **References**:
  - `.github/workflows/python-app.yml:26` — The line to change
  - `https://mypy.readthedocs.io/en/stable/running_mypy.html#following-imports` — mypy `-p` flag documentation

  **Acceptance Criteria**:
  - `mypy -p grpo_in_sandbox` runs without exit code 2
  - CI type check step passes

  **Commit**: YES
  - Message: `fix(ci): use mypy -p for flat-layout package resolution`
  - Files: `.github/workflows/python-app.yml`

- [ ] 2. Fix `ruff` command in CI workflow

  **What to do**:
  - Edit `.github/workflows/python-app.yml` line 29
  - Change `ruff check grpo_in_sandbox` to `ruff check .`
  - `ruff check .` checks all `.py` files in the current directory (flat layout)

  **Must NOT do**:
  - Do not use `ruff check *.py` (won't recurse into subdirectories like `benchmark/`)
  - Do not restructure to src-layout

  **Parallelization**:
  - **Can Run In Parallel**: NO (same file)
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: Task 1

  **References**:
  - `.github/workflows/python-app.yml:29` — The line to change

  **Acceptance Criteria**:
  - `ruff check .` runs without errors
  - CI lint step passes

  **Commit**: YES (with Task 1)
  - Message: `fix(ci): use mypy -p for flat-layout package resolution`
  - Files: `.github/workflows/python-app.yml`

---

## Final Verification Wave

- [ ] F1. **Verify workflow syntax** — `python -c "import yaml; yaml.safe_load(open('.github/workflows/python-app.yml'))"` 
- [ ] F2. **Push and verify CI passes** — Push to GitHub, check Actions tab
- [ ] F3. **Confirm exit code** — Both `type check` and `lint` steps should show green checkmark

---

## Commit Strategy

- **1-2**: `fix(ci): use mypy -p for flat-layout package resolution`

---

## Success Criteria

### Verification
```bash
mypy -p grpo_in_sandbox  # Expected: exit 0, no errors
ruff check .              # Expected: exit 0, no errors
```

### Final Checklist
- [ ] `type check` step passes in CI
- [ ] `lint` step passes in CI
- [ ] No source code modified
