"""Security regression tests for the sandbox boundary and untrusted inputs.

Covers the Critical/High findings from the audit:
- create_runtime('auto') fails closed instead of falling back to host execution.
- Explicit local/kaggle backends warn about the lack of isolation.
- docker_kwargs that would break isolation are rejected.
- Container archive extraction rejects path traversal / symlinks / escapes.
- Dataset-supplied names cannot escape their intended directory.
"""

import io
import logging
import sys
import tarfile
import types

import pytest

import grpo_in_sandbox.docker_runtime as dr
from grpo_in_sandbox import runtime as rt
from grpo_in_sandbox.benchmark.runner import _safe_path_component

# --- create_runtime fail-closed ---------------------------------------------

def test_auto_backend_fails_closed_when_docker_unavailable(monkeypatch):
    """auto must NOT silently fall back to non-isolated host execution."""
    monkeypatch.delenv("KAGGLE_KERNEL_TYPE", raising=False)

    fake_docker = types.ModuleType("docker")

    def _boom(*args, **kwargs):
        raise RuntimeError("docker daemon not reachable")

    fake_docker.from_env = _boom
    monkeypatch.setitem(sys.modules, "docker", fake_docker)

    with pytest.raises(RuntimeError, match="Refusing to fall back to host"):
        rt.create_runtime(backend="auto")


def test_explicit_local_backend_warns_about_no_isolation(caplog):
    with caplog.at_level(logging.WARNING):
        runtime = rt.create_runtime(backend="local")
    assert runtime is not None
    assert any("HOST machine" in rec.getMessage() for rec in caplog.records)


def test_explicit_kaggle_backend_warns_outside_kaggle(caplog, monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_TYPE", raising=False)
    with caplog.at_level(logging.WARNING):
        rt.create_runtime(backend="kaggle")
    assert any("HOST machine" in rec.getMessage() for rec in caplog.records)


# --- docker_kwargs validation ------------------------------------------------

@pytest.mark.parametrize("bad_kwargs", [
    {"privileged": True},
    {"devices": ["/dev/sda:/dev/sda"]},
    {"cap_add": ["SYS_ADMIN"]},
    {"volumes": {"/": {"bind": "/host", "mode": "rw"}}},
    {"mounts": ["/var/run/docker.sock:/var/run/docker.sock"]},
    {"network_mode": "host"},
    {"pid_mode": "host"},
    {"ipc_mode": "host"},
    {"security_opt": ["seccomp=unconfined"]},
])
def test_dangerous_docker_kwargs_are_rejected(bad_kwargs):
    with pytest.raises(ValueError):
        dr._validate_docker_kwargs(bad_kwargs)


def test_safe_docker_kwargs_are_accepted():
    # Should not raise.
    dr._validate_docker_kwargs(
        {"mem_limit": "2g", "user": "1000:1000", "network_mode": "none", "pids_limit": 128}
    )


def test_default_security_kwargs_harden_the_container():
    assert dr.DEFAULT_SECURITY_KWARGS["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in dr.DEFAULT_SECURITY_KWARGS["security_opt"]
    assert dr.DEFAULT_SECURITY_KWARGS["pids_limit"] > 0
    assert dr.DEFAULT_SECURITY_KWARGS["mem_limit"]


# --- safe archive extraction -------------------------------------------------

def _new_runtime_shell():
    """A DockerRuntime with only the fields _safe_extract needs (no daemon)."""
    inst = dr.DockerRuntime.__new__(dr.DockerRuntime)
    inst.logger = logging.getLogger("test-docker-runtime")
    return inst


def _build_malicious_archive() -> io.BytesIO:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        # legit file under base dir
        data = b"safe-content"
        info = tarfile.TarInfo("output/good.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

        # parent-directory traversal
        evil = b"pwned"
        info = tarfile.TarInfo("output/../escape.txt")
        info.size = len(evil)
        tar.addfile(info, io.BytesIO(evil))

        # absolute path
        info = tarfile.TarInfo("/tmp/abs_escape.txt")
        info.size = len(evil)
        tar.addfile(info, io.BytesIO(evil))

        # symlink pointing outside the tree
        info = tarfile.TarInfo("output/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    buf.seek(0)
    return buf


def test_safe_extract_only_writes_safe_regular_files(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    runtime = _new_runtime_shell()

    with tarfile.open(fileobj=_build_malicious_archive(), mode="r") as tar:
        runtime._safe_extract(tar, str(dest), "output")

    # Only the legitimate file made it through.
    assert (dest / "good.txt").read_text() == "safe-content"
    assert not (dest / "link").exists()
    assert not (tmp_path / "escape.txt").exists()
    assert not (dest / "escape.txt").exists()


def test_safe_extract_enforces_member_limit(tmp_path, monkeypatch):
    dest = tmp_path / "dest2"
    dest.mkdir()
    monkeypatch.setattr(dr, "MAX_ARCHIVE_MEMBERS", 1)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for i in range(3):
            data = b"x"
            info = tarfile.TarInfo(f"output/f{i}.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)

    runtime = _new_runtime_shell()
    with tarfile.open(fileobj=buf, mode="r") as tar, pytest.raises(ValueError, match="members"):
        runtime._safe_extract(tar, str(dest), "output")


# --- dataset path-component validation --------------------------------------

@pytest.mark.parametrize("bad", ["../etc", "/etc/passwd", "a/b", "..", ".", "", "a\x00b"])
def test_unsafe_path_components_are_rejected(bad):
    with pytest.raises(ValueError):
        _safe_path_component(bad, "problem id")


def test_safe_path_component_returns_valid_name():
    assert _safe_path_component("problem_42", "problem id") == "problem_42"
