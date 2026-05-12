"""
Runtime Module - Unified Sandbox Runtime Abstraction.

This module provides a unified interface for executing code in isolated environments.
Supports multiple backends:

    DockerRuntime    - Docker containers (production, multi-language)
    KaggleRuntime  - Kaggle notebooks
    LocalRuntime - Local subprocess (development/testing)

Architecture:
    +------------------+
    |   BaseRuntime  |  (abstract interface)
    +------------------+
            |
    +----+----+----+
    |         |       |
    v         v       v
  Docker  Kaggle  Local

API:
    run(code, timeout)      -> (output, exit_code)
    demux_run(code)       -> (stdout, stderr, exit_code)
    copy_to_container()   -> copies files to sandbox
    copy_from_container() -> copies files from sandbox
    close()           -> cleanup resources

Usage:
    >>> from runtime import create_runtime
    >>>
    >>> # Create runtime (auto-detects based on environment)
    >>> runtime = create_runtime("docker")
    >>> # or
    >>> runtime = create_runtime("local")
    >>>
    >>> # Execute code
    >>> output, exit_code = runtime.run("echo Hello", timeout=30)
    >>> print(output)  # "Hello\n"
    >>>
    >>> # Cleanup
    >>> runtime.close()

Factory Function:
    create_runtime(backend: str) -> BaseRuntime
        backend: "docker", "kaggle", or "local"
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import os
import subprocess
import tempfile
import shutil
import logging


# Module-level logger
logger = logging.getLogger(__name__)


def _setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with consistent formatting."""
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
    """Abstract base class for sandbox runtimes.

    All runtime implementations must inherit from this class and implement:
    - run(): Execute code and return output + exit code
    - demux_run(): Execute with separate stdout/stderr
    - copy_to_container(): Copy files to sandbox
    - copy_from_container(): Copy files from sandbox
    - close(): Cleanup resources

    Attributes:
        All attributes are implementation-defined.

    Example:
        >>> class MyRuntime(BaseRuntime):
        ...     def run(self, code, timeout=30, workdir=None):
        ...         return f"Ran: {code}", "0"
    """
    
    @abstractmethod
    def run(self, code: str, timeout: int = 30, workdir: str = None) -> Tuple[str, str]:
        """Execute bash command. Returns (output, exit_code)."""
        pass
    
    @abstractmethod
    def demux_run(self, code: str, timeout: int = 30, workdir: str = None) -> Tuple[str, str, str]:
        """Execute with separate stdout/stderr. Returns (stdout, stderr, exit_code)."""
        pass
    
    @abstractmethod
    def close(self):
        """Clean up runtime resources."""
        pass
    
    @abstractmethod
    def copy_to_container(self, src_path: str, dest_path: str):
        """Copy file from host to sandbox."""
        pass
    
    @abstractmethod
    def copy_from_container(self, container_path: str, local_path: str):
        """Copy file from sandbox to host."""
        pass


class LocalRuntime(BaseRuntime):
    """Local runtime - runs commands directly on the host machine."""
    
    def __init__(self, working_dir: str = "/tmp/testbed"):
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)
    
    def run(self, code: str, timeout: int = 30, workdir: str = None) -> Tuple[str, str]:
        exec_workdir = workdir or self.working_dir
        
        result = subprocess.run(
            code,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=exec_workdir,
        )
        
        output = result.stdout + result.stderr
        exit_code = str(result.returncode)
        
        if result.returncode != 0:
            return output, f"Error: Exit code {exit_code}"
        
        return output, exit_code
    
    def demux_run(self, code: str, timeout: int = 30, workdir: str = None) -> Tuple[str, str, str]:
        exec_workdir = workdir or self.working_dir
        
        result = subprocess.run(
            code,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=exec_workdir,
        )
        
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
    """Docker runtime - runs commands in Docker containers."""
    
    def __init__(
        self,
        docker_image: str = "cdx123/llm-in-sandbox:v0.1",
        repo_path: str = "/testbed",
        command: str = "sleep infinity",
        logger=None,
        **docker_kwargs,
    ):
        from grpo_in_sandbox.docker_runtime import DockerRuntime as _Docker
        
        self._runtime = _Docker(
            docker_image=docker_image,
            repo_path=repo_path,
            command=command,
            logger=logger,
            **docker_kwargs,
        )
    
    def run(self, code: str, timeout: int = 30, workdir: str = None) -> Tuple[str, str]:
        return self._runtime.run(code, timeout, workdir)
    
    def demux_run(self, code: str, timeout: int = 30, workdir: str = None) -> Tuple[str, str, str]:
        return self._runtime.demux_run(code, timeout, workdir)
    
    def close(self):
        self._runtime.close()
    
    def copy_to_container(self, src_path: str, dest_path: str):
        self._runtime.copy_to_container(src_path, dest_path)
    
    def copy_from_container(self, container_path: str, local_path: str):
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
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    
    def run(self, code: str, timeout: int = 30, workdir: str = None) -> Tuple[str, str]:
        try:
            output = self._run_code(code, timeout)
            return output, "0"
        except subprocess.TimeoutExpired:
            return f"Execution timed out ({timeout}s)", "-1"
        except Exception as e:
            return f"Error: {repr(e)}", "-1"
    
    def demux_run(self, code: str, timeout: int = 30, workdir: str = None) -> Tuple[str, str, str]:
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
    """
    Create a runtime based on the environment.
    
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