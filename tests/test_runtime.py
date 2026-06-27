from pathlib import Path

import pytest

from grpo_in_sandbox.runtime import KaggleRuntime, LocalRuntime, create_runtime


def test_create_local_runtime_runs_shell_command():
    runtime = create_runtime(backend="local")

    output, exit_code = runtime.run("echo hello")

    assert output == "hello\n"
    assert exit_code == "0"


def test_create_kaggle_runtime_runs_shell_command_locally():
    runtime = create_runtime(backend="kaggle")

    output, exit_code = runtime.run("echo hello")

    assert output == "hello\n"
    assert exit_code == "0"
    assert runtime.working_dir == "/tmp/testbed"


def test_kaggle_runtime_run_python_is_explicit_python_entrypoint(tmp_path):
    runtime = KaggleRuntime(working_dir=str(tmp_path))

    output, exit_code = runtime.run_python('print("hello")')

    assert output == "hello\n"
    assert exit_code == 0


def test_kaggle_runtime_demux_run_separates_stdout_and_stderr(tmp_path):
    runtime = KaggleRuntime(working_dir=str(tmp_path))

    stdout, stderr, exit_code = runtime.demux_run(
        "printf out; printf err >&2"
    )

    assert stdout == "out"
    assert stderr == "err"
    assert exit_code == "0"


def test_create_runtime_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown runtime backend"):
        create_runtime(backend="bad")


def test_local_runtime_copy_from_container_creates_parent_directory(tmp_path):
    src = tmp_path / "source.txt"
    src.write_text("content")
    dest = tmp_path / "nested" / "copy.txt"
    runtime = LocalRuntime(working_dir=str(tmp_path))

    runtime.copy_from_container(str(src), str(dest))

    assert dest.read_text() == "content"


def test_kaggle_runtime_copy_from_container_supports_plain_filename(tmp_path, monkeypatch):
    src = tmp_path / "source.txt"
    src.write_text("content")
    runtime = KaggleRuntime(working_dir=str(tmp_path))

    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        runtime.copy_from_container(str(src), "plain.txt")

    assert (tmp_path / "plain.txt").read_text() == "content"


def test_kaggle_runtime_copy_from_container_creates_parent_directory(tmp_path):
    src = tmp_path / "source.txt"
    src.write_text("content")
    dest = tmp_path / "nested" / "copy.txt"
    runtime = KaggleRuntime(working_dir=str(tmp_path))

    runtime.copy_from_container(str(src), str(dest))

    assert Path(dest).read_text() == "content"
