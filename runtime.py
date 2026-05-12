"""Runtime Module - Unified sandbox runtime abstraction (Docker/Kaggle/Local)."""

import contextlib
import logging
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def _setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        ))
        log.addHandler(handler)
        log.setLevel(level)
    return log


CMD_TIMEOUT = 120
DOCKER_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/local/go/bin:/opt/miniconda3/envs/testbed/bin"


class BaseRuntime(ABC):
    """Abstract base class for sandbox runtimes."""

    @abstractmethod
    def run(self, code: str, timeout: int = 30, workdir: str = None) -> tuple[str, str]:
        pass

    @abstractmethod
    def demux_run(self, code: str, timeout: int = 30, workdir: str = None) -> tuple[str, str, str]:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def copy_to_container(self, src_path: str, dest_path: str):
        pass

    @abstractmethod
    def copy_from_container(self, container_path: str, local_path: str):
        pass


class LocalRuntime(BaseRuntime):
    """Local runtime - runs commands directly on the host machine."""

    def __init__(self, working_dir: str = "/tmp/testbed"):
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)

    def _run(self, code: str, timeout: int = 30, workdir: str = None):
        exec_workdir = workdir or self.working_dir
        return subprocess.run(
            code, shell=True, capture_output=True,
            text=True, timeout=timeout, cwd=exec_workdir,
        )

    def run(self, code: str, timeout: int = 30, workdir: str = None) -> tuple[str, str]:
        result = self._run(code, timeout, workdir)
        output = result.stdout + result.stderr
        exit_code = str(result.returncode)
        if result.returncode != 0:
            return output, f"Error: Exit code {exit_code}"
        return output, exit_code

    def demux_run(self, code: str, timeout: int = 30, workdir: str = None) -> tuple[str, str, str]:
        result = self._run(code, timeout, workdir)
        exit_code = str(result.returncode)
        if result.returncode != 0:
            return result.stdout, result.stderr, f"Error: Exit code {exit_code}"
        return result.stdout, result.stderr, exit_code

    def close(self):
        pass

    def copy_to_container(self, src_path: str, dest_path: str):
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src_path, dest_path)

    def copy_from_container(self, container_path: str, local_path: str):
        shutil.copy2(container_path, local_path)


class DockerRuntime(BaseRuntime):
    """Docker runtime - delegates to docker_runtime.DockerRuntime."""

    def __init__(self, docker_image="cdx123/llm-in-sandbox:v0.1", repo_path="/testbed",
                 command="sleep infinity", logger=None, **docker_kwargs):
        from grpo_in_sandbox.docker_runtime import DockerRuntime as _Docker
        self._runtime = _Docker(docker_image=docker_image, repo_path=repo_path,
                                command=command, logger=logger, **docker_kwargs)

    def run(self, code, timeout=30, workdir=None):
        return self._runtime.run(code, timeout, workdir)

    def demux_run(self, code, timeout=30, workdir=None):
        return self._runtime.demux_run(code, timeout, workdir)

    def close(self):
        self._runtime.close()

    def copy_to_container(self, src_path, dest_path):
        self._runtime.copy_to_container(src_path, dest_path)

    def copy_from_container(self, container_path, local_path):
        self._runtime.copy_from_container(container_path, local_path)


class KaggleRuntime(BaseRuntime):
    """Kaggle runtime - runs commands in Kaggle notebooks."""

    def __init__(self, working_dir: str = "/kaggle/working"):
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)
        self._exec_history = []

    def _run_code(self, code: str, timeout: int = 30) -> str:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
            )
            output = result.stdout + result.stderr
            self._exec_history.append({"code": code[:100], "output": output[:500]})
            return output
        finally:
            with contextlib.suppress(OSError):
                os.unlink(temp_path)

    def run(self, code: str, timeout: int = 30, workdir: str = None) -> tuple[str, str]:
        try:
            output = self._run_code(code, timeout)
            return output, "0"
        except subprocess.TimeoutExpired:
            return f"Execution timed out ({timeout}s)", "-1"
        except Exception as e:
            return f"Error: {repr(e)}", "-1"

    def demux_run(self, code: str, timeout: int = 30, workdir: str = None) -> tuple[str, str, str]:
        try:
            output = self._run_code(code, timeout)
            return output, "", "0"
        except subprocess.TimeoutExpired:
            return f"Execution timed out ({timeout}s)", "", "-1"
        except Exception as e:
            return f"Error: {repr(e)}", f"Error: {repr(e)}", "-1"

    def close(self):
        self._exec_history.clear()

    def copy_to_container(self, src_path: str, dest_path: str):
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src_path, dest_path)

    def copy_from_container(self, container_path: str, local_path: str):
        shutil.copy2(container_path, local_path)


def create_runtime(
    backend: str = "auto",
    **kwargs,
) -> BaseRuntime:
    """Create a runtime instance based on the environment.

    Args:
        backend: Runtime backend ("docker", "kaggle", "local", "auto")
        **kwargs: Additional arguments for the runtime

    Returns:
        BaseRuntime: A runtime instance
    """
    log = _setup_logger(__name__)
    log.info(f"Creating runtime with backend: {backend}")

    if backend == "auto":
        if os.environ.get("KAGGLE_KERNEL_TYPE"):
            log.info("Auto-detected: Kaggle environment")
            return KaggleRuntime(**kwargs)
        try:
            import docker
            docker.from_env()
            log.info("Auto-detected: Docker available")
            return DockerRuntime(**kwargs)
        except Exception as e:
            log.info(f"Auto-detected: Using local (Docker not available: {e})")
            return LocalRuntime(**kwargs)

    backends = {
        "docker": DockerRuntime,
        "kaggle": KaggleRuntime,
        "local": LocalRuntime,
    }

    runtime_class = backends.get(backend, LocalRuntime)
    log.info(f"Created runtime: {runtime_class.__name__}")

    return runtime_class(**kwargs)


__all__ = [
    "BaseRuntime",
    "LocalRuntime",
    "DockerRuntime",
    "KaggleRuntime",
    "create_runtime",
]
