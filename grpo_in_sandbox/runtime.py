"""Runtime Module - Unified sandbox runtime abstraction (Docker/Kaggle/Local)."""

import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

#输出日志的设置，日志级别为INFO，输出格式为时间、模块名、日志级别和消息内容
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
    def run(self, code: str, timeout: int = 30, workdir: str | None = None) -> tuple[str, str]:
        pass

    @abstractmethod
    def demux_run(self, code: str, timeout: int = 30, workdir: str | None = None) -> tuple[str, str, str]:
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

    def _run(self, code: str, timeout: int = 30, workdir: str | None = None):
        exec_workdir = workdir or self.working_dir
        return subprocess.run(
            code, shell=True, capture_output=True,
            text=True, timeout=timeout, cwd=exec_workdir,
        )

    def run(self, code: str, timeout: int = 30, workdir: str | None = None) -> tuple[str, str]:
        result = self._run(code, timeout, workdir)
        output = result.stdout + result.stderr
        exit_code = str(result.returncode)
        if result.returncode != 0:
            return output, f"Error: Exit code {exit_code}"
        return output, exit_code

    def demux_run(self, code: str, timeout: int = 30, workdir: str | None = None) -> tuple[str, str, str]:
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
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
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
        #定义在ducker运行的时候的参数

    def demux_run(self, code, timeout=30, workdir=None):
        return self._runtime.demux_run(code, timeout, workdir)

    def close(self):
        self._runtime.close()

    def copy_to_container(self, src_path, dest_path):
        self._runtime.copy_to_container(src_path, dest_path)

    def copy_from_container(self, container_path, local_path):
        self._runtime.copy_from_container(container_path, local_path)


class KaggleRuntime(BaseRuntime):
    """Kaggle runtime - delegates to kaggle_runtime.KaggleRuntime."""

    def __init__(self, working_dir: str | None = None, logger=None):
        from grpo_in_sandbox.kaggle_runtime import KaggleRuntime as _Kaggle

        self._runtime = _Kaggle(working_dir=working_dir, logger=logger)
        self.working_dir = self._runtime.working_dir

    def run(self, code: str, timeout: int = 30, workdir: str | None = None):
        return self._runtime.run(code, timeout, workdir)

    def demux_run(self, code: str, timeout: int = 30, workdir: str | None = None):
        return self._runtime.demux_run(code, timeout, workdir)

    def run_python(self, code: str, timeout: int = 30):
        return self._runtime.run_python(code, timeout)

    def close(self):
        self._runtime.close()

    def copy_to_container(self, src_path: str, dest_path: str):
        self._runtime.copy_to_container(src_path, dest_path)

    def copy_dir_to_container(self, src_dir: str, dest_dir: str):
        self._runtime.copy_dir_to_container(src_dir, dest_dir)

    def copy_from_container(self, container_path: str, local_path: str):
        self._runtime.copy_from_container(container_path, local_path)


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

    if backend not in backends:
        available = ", ".join(["auto", *backends])
        raise ValueError(f"Unknown runtime backend: {backend!r}. Available: {available}")

    runtime_class = backends[backend]
    log.info(f"Created runtime: {runtime_class.__name__}")

    return runtime_class(**kwargs)


__all__ = [
    "BaseRuntime",
    "LocalRuntime",
    "DockerRuntime",
    "KaggleRuntime",
    "create_runtime",
]
