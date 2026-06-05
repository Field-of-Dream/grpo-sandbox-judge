"""
Kaggle 运行时模块 - 在 Kaggle 环境中执行命令

本模块提供 KaggleRuntime 类，替代 DockerRuntime 用于 Kaggle 环境。
"""

import contextlib
import os
import shutil
import subprocess
import tempfile

from .agent import get_logger
from .runtime import BaseRuntime


class KaggleRuntime(BaseRuntime):
    """
    Kaggle 运行时 - 在 Kaggle Notebook 环境中执行命令。

    替代 DockerRuntime，核心功能：
    1. 执行 Python/bash 命令
    2. 文件操作
    3. 工作目录管理
    """

    def __init__(
        self,
        working_dir: str = "/tmp/testbed",
        logger=None,
    ):
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(os.path.join(working_dir, "input"), exist_ok=True)
        os.makedirs(os.path.join(working_dir, "output"), exist_ok=True)

        if logger is None:
            self.logger = get_logger("KaggleRuntime")
        else:
            self.logger = logger

        self.logger.info("Kaggle environment initialized")
        self.logger.info(f"Working directory: {self.working_dir}")

    def run(
        self,
        code: str,
        timeout: int = 60,
        workdir: str | None = None,
    ) -> tuple[str, str]:
        """
        执行命令。

        Args:
            code: 要执行的 bash 命令
            timeout: 超时时间（秒）
            workdir: 工作目录

        Returns:
            (output, exit_code) - 命令输出和退出码
        """
        exec_workdir = self.working_dir if workdir is None else workdir

        try:
            result = subprocess.run(
                ["bash", "-c", code],
                cwd=exec_workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr

            if result.returncode != 0:
                return output, f"Error: Exit code {result.returncode}"

            return output, str(result.returncode)

        except subprocess.TimeoutExpired:
            return f"The command took too long to execute (>{timeout}s)", "-1"
        except Exception as e:
            return f"Error: {repr(e)}", "-1"

    def run_python(self, code: str, timeout: int = 60) -> tuple[str, int]:
        """
        执行 Python 代码。

        Args:
            code: Python 代码
            timeout: 超时时间（秒）

        Returns:
            (output, exit_code)
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr
            return output, result.returncode
        except subprocess.TimeoutExpired:
            return f"Python execution timed out ({timeout}s)", -1
        except Exception as e:
            return f"Error: {repr(e)}", -1
        finally:
            with contextlib.suppress(OSError):
                os.unlink(temp_path)

    def copy_to_container(self, src_path: str, dest_path: str):
        """复制文件到工作目录"""
        try:
            dest_dir = os.path.dirname(dest_path)
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            self.logger.info(f"Copied {src_path} to {dest_path}")
        except Exception as e:
            self.logger.error(f"Error copying file: {repr(e)}")
            raise

    def copy_dir_to_container(self, src_dir: str, dest_dir: str):
        """复制目录到工作目录"""
        try:
            os.makedirs(dest_dir, exist_ok=True)
            for item in os.listdir(src_dir):
                src_item = os.path.join(src_dir, item)
                dest_item = os.path.join(dest_dir, item)
                if os.path.isdir(src_item):
                    shutil.copytree(src_item, dest_item)
                else:
                    shutil.copy2(src_item, dest_item)
            self.logger.info(f"Copied directory {src_dir} to {dest_dir}")
        except Exception as e:
            self.logger.error(f"Error copying directory: {repr(e)}")
            raise

    def copy_from_container(self, container_path: str, local_path: str):
        """从工作目录复制文件"""
        try:
            if os.path.isdir(container_path):
                shutil.copytree(container_path, local_path, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                shutil.copy2(container_path, local_path)
            self.logger.info(f"Copied {container_path} to {local_path}")
        except Exception as e:
            self.logger.error(f"Error copying from container: {repr(e)}")
            raise

    def get_task_instruction(self, problem_statement: str) -> str:
        """返回任务指令"""
        return problem_statement

    def close(self):
        """清理临时文件"""
        if os.path.isdir(self.working_dir):
            shutil.rmtree(self.working_dir, ignore_errors=True)

    def __del__(self):
        self.close()
