"""Regression tests for the repaired / previously-unfinished features.

Covers:
- Trajectory terminal-status fields (status/stop_reason/error/submitted).
- Tolerant prompt formatting + the team-YAML {task_description} alias.
- Bundled team YAML files load and format without KeyError.
- Console-script entry points declared in pyproject.
- RewardModel sandbox scoring copies the test into the container and reads the
  exit code correctly (string "0", not int 0).
"""

import sys
from pathlib import Path

import pytest

from grpo_in_sandbox.trajectory import Trajectory

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    tomllib = None

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "grpo_in_sandbox" / "config"


# --- Trajectory status -------------------------------------------------------

def test_trajectory_exposes_terminal_status_fields():
    traj = Trajectory(
        problem_statement="p",
        status=Trajectory.STATUS_COMPLETED,
        stop_reason="submit",
        submitted=True,
    )
    d = traj.to_dict()
    assert d["status"] == "completed"
    assert d["stop_reason"] == "submit"
    assert d["submitted"] is True
    assert d["error"] is None


def test_trajectory_defaults_to_max_steps_status():
    traj = Trajectory(problem_statement="p")
    assert traj.status == Trajectory.STATUS_MAX_STEPS
    assert traj.submitted is False


# --- tolerant prompt formatting ---------------------------------------------

def test_safe_format_fills_alias_and_keeps_unknown_placeholders():
    pytest.importorskip("litellm", reason="grpo_in_sandbox.agent imports litellm")
    from grpo_in_sandbox.agent import _safe_format

    tmpl = "<task>{task_description}</task> keep {unknown} literal"
    out = _safe_format(tmpl, problem_statement="DO X", task_description="DO X")
    assert "DO X" in out
    assert "{unknown}" in out  # unknown placeholder left intact, no crash


def test_safe_format_survives_literal_braces():
    pytest.importorskip("litellm", reason="grpo_in_sandbox.agent imports litellm")
    from grpo_in_sandbox.agent import _safe_format

    tmpl = "code: def f(): return {'a': 1}"
    # Must not raise even though the braces are not valid format fields.
    assert _safe_format(tmpl, problem_statement="x") == tmpl


# --- bundled team YAML -------------------------------------------------------

@pytest.mark.parametrize("yaml_name", ["product_team.yaml", "company_team.yaml"])
def test_bundled_team_yaml_uses_supported_placeholder(yaml_name):
    path = CONFIG_DIR / yaml_name
    text = path.read_text(encoding="utf-8")
    assert "{task_description}" not in text
    assert "{problem_statement}" in text


@pytest.mark.parametrize("yaml_name", ["product_team.yaml", "company_team.yaml"])
def test_bundled_team_yaml_instance_prompt_formats(yaml_name):
    pytest.importorskip("litellm", reason="agent_configs imports agent -> litellm")
    from grpo_in_sandbox.agent_configs import load_team_from_yaml

    team = load_team_from_yaml(str(CONFIG_DIR / yaml_name))
    assert len(team) > 0
    for profile in team:
        rendered = profile.format_instance_prompt(problem_statement="BUILD A THING")
        assert "BUILD A THING" in rendered


# --- console script ----------------------------------------------------------

@pytest.mark.skipif(tomllib is None, reason="tomllib requires Python 3.11+")
def test_console_scripts_declared_in_pyproject():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = data.get("project", {}).get("scripts", {})
    assert scripts.get("llm-in-sandbox") == "grpo_in_sandbox.cli:main"
    assert "grpo_in_sandbox.cli:main" in scripts.values()


# --- RewardModel sandbox scoring --------------------------------------------

class _FakeRuntime:
    """Records interactions so we can assert the test file reaches the sandbox."""

    def __init__(self, exit_code="0", output="1 passed"):
        self.copied = []
        self.commands = []
        self._exit = exit_code
        self._output = output

    def copy_to_container(self, src, dest):
        self.copied.append((src, dest))

    def run(self, cmd, *args, **kwargs):
        self.commands.append(cmd)
        if cmd.startswith("rm "):
            return "", "0"
        return self._output, self._exit


def _reward_model():
    pytest.importorskip("torch", reason="grpo_in_sandbox.train imports torch")
    from grpo_in_sandbox.train import RewardModel

    return RewardModel


TEST_SNIPPET = "import unittest\n\ndef test_ok():\n    assert True\n"


def test_sandbox_score_copies_test_into_container_and_passes():
    reward_cls = _reward_model()
    fake = _FakeRuntime(exit_code="0", output="collected 1 item\n1 passed")
    rm = reward_cls(runtime=fake)

    score = rm._sandbox_score(TEST_SNIPPET)

    assert score == 1.0
    # The test file must be copied INTO the container...
    assert fake.copied, "test file was never copied into the container"
    container_dest = fake.copied[0][1]
    assert container_dest.startswith("/tmp/grpo_reward_")
    # ...and pytest must run against that container path, not the host temp path.
    pytest_cmds = [c for c in fake.commands if "pytest" in c]
    assert pytest_cmds and container_dest in pytest_cmds[0]
    # No `| head` pipe that would mask the real pytest exit code.
    assert "| head" not in pytest_cmds[0]


def test_sandbox_score_reports_failure_on_nonzero_exit():
    reward_cls = _reward_model()
    fake = _FakeRuntime(exit_code="Error: Exit code 1", output="1 failed")
    rm = reward_cls(runtime=fake)

    score = rm._sandbox_score(TEST_SNIPPET)

    assert score == 0.3  # failure path, not the 1.0 "passed" path
