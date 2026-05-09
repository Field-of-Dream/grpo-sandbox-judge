"""
Docker运行时模块 - 在Docker容器中执行命令

本模块定义DockerRuntime类，负责在Docker容器中执行bash命令、
文件传输等操作，是智能体与沙箱环境交互的基础。
"""
import os
import re
import time
import uuid
import tarfile
import io
import datetime
import hashlib
import logging
import shlex
import docker
from typing import Dict, Tuple, Any, Optional

from . import CMD_TIMEOUT, DOCKER_PATH


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器。

    Args:
        name: 日志记录器名称

    Returns:
        配置好的Logger实例
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class DockerRuntime:
    """
    Docker运行时 - 在Docker容器中执行命令。

    该类负责：
    1. 启动和管理Docker容器
    2. 在容器中执行bash命令
    3. 在宿主机与容器之间传输文件

    Attributes:
        docker_image: 使用的Docker镜像名称
        repo_path: 容器中的工作目录
        container: Docker容器对象
    """

    def __init__(
        self,
        docker_image: str,
        repo_path: str = "/testbed",
        command: str = "sleep infinity",
        logger=None,
        **docker_kwargs,
    ):
        """
        初始化Docker运行时。

        Args:
            docker_image: Docker镜像名称
            repo_path: 容器中的工作目录路径
            command: 容器启动命令
            logger: 日志记录器（可选）
            **docker_kwargs: 传递给docker.run的额外参数
        """
        self.docker_image = docker_image
        self.repo_path = repo_path
        self.command = command
        self.docker_kwargs = docker_kwargs
        
        # 设置日志记录器
        if logger is None:
            self.logger = get_logger("DockerRuntime")
        elif logger is False:
            # 静默模式 - 只显示警告和错误
            self.logger = logging.getLogger("DockerRuntime.quiet")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                self.logger.addHandler(handler)
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger = logger

        self.client = docker.from_env(timeout=120)
        
        # 启动容器
        self.container = None
        self.container_name = self._get_container_name(docker_image)
        self.start_container(docker_image, command, self.container_name, **docker_kwargs)
        
        # 初始化环境
        self.setup_env()
        self.logger.info(f"Docker environment initialized")
        self.logger.info(f"Docker image: {self.docker_image}")
        self.logger.info(f"Container ID: {self.container.id}")

    @staticmethod
    def _get_container_name(image_name: str) -> str:
        """
        根据镜像名称生成唯一的容器名称。

        使用镜像名称、当前时间和进程ID的哈希值来生成唯一名称，
        确保多次运行不会产生冲突。

        Args:
            image_name: Docker镜像名称

        Returns:
            唯一的容器名称字符串
        """
        process_id = str(os.getpid())
        current_time = str(datetime.datetime.now())
        unique_string = current_time + process_id
        hash_object = hashlib.sha256(unique_string.encode())
        image_name_sanitized = image_name.replace("/", "-").replace(":", "-")
        return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"

    def start_container(self, docker_image: str, command: str, container_name: str, **docker_kwargs):
        """
        启动Docker容器。

        如果容器已存在则复用，否则创建新容器。

        Args:
            docker_image: Docker镜像名称
            command: 启动命令
            container_name: 容器名称
            **docker_kwargs: 传递给docker.run的额外参数
        """
        try:
            # 检查容器是否已存在
            self.container = self.client.containers.get(container_name)
            self.logger.info(f"Found existing container: {container_name}")
            if self.container.status != "running":
                self.container.start()
            return
        except docker.errors.NotFound:
            pass

        # 拉取镜像（如果不存在）
        try:
            self.client.images.get(docker_image)
        except docker.errors.ImageNotFound:
            self.logger.info(f"Pulling Docker image: {docker_image}")
            self.client.images.pull(docker_image)

        # 创建并启动容器
        env_vars = {
            "PATH": DOCKER_PATH,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",  # 禁用pip版本检查提示
            "PIP_ROOT_USER_ACTION": "ignore",  # 抑制root用户警告
            "PIP_NO_WARN_SCRIPT_LOCATION": "1",  # 抑制脚本位置警告
            **docker_kwargs.get("environment", {})
        }
        
        self.container = self.client.containers.run(
            docker_image,
            command=command,
            name=container_name,
            detach=True,
            stdin_open=True,
            tty=True,
            environment=env_vars,
            working_dir=self.repo_path,
            **{k: v for k, v in docker_kwargs.items() if k != "environment"},
        )
        self.logger.info(f"Started container: {container_name}")

    def setup_env(self):
        """
        设置容器环境。

        执行以下初始化操作：
        1. 创建工作目录
        2. 创建输入/输出目录
        3. 初始化git仓库（可选）
        4. 配置pip使用清华镜像源（对中国大陆用户更快）
        """
        # 确保工作目录存在
        self.run(f"mkdir -p {self.repo_path}")
        # 创建输入/输出目录
        self.run(f"mkdir -p {self.repo_path}/input {self.repo_path}/output")
        # 初始化git仓库用于跟踪更改（可选）
        self.run(f"cd {self.repo_path} && git init 2>/dev/null || true")
        
        # 配置pip使用清华镜像（对中国大陆用户更快）
        self.run("mkdir -p ~/.pip && cat > ~/.pip/pip.conf << 'EOF'\n"
                 "[global]\n"
                 "index-url = https://pypi.tuna.tsinghua.edu.cn/simple\n"
                 "trusted-host = pypi.tuna.tsinghua.edu.cn\n"
                 "EOF")

    def run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        workdir: str = None,
    ) -> Tuple[str, str]:
        """
        在容器中执行命令（组合输出模式）。

        Args:
            code: 要执行的bash命令
            timeout: 超时时间（秒）
            workdir: 工作目录（默认为repo_path）

        Returns:
            (output, exit_code_or_error) - 命令输出和退出码
        """
        exec_workdir = self.repo_path if workdir is None else workdir
        
        # 将命令包装在bash -c中以支持shell内置命令如'cd'
        # 使用shlex.quote正确转义命令
        command = ["bash", "-c", f"timeout {timeout} bash -c {shlex.quote(code)}"]

        try:
            exec_result = self.container.exec_run(
                command,
                workdir=exec_workdir,
                environment={
                    "PATH": DOCKER_PATH,
                    "PIP_DISABLE_PIP_VERSION_CHECK": "1",  # 禁用pip版本检查提示
                    "PIP_ROOT_USER_ACTION": "ignore",  # 抑制root用户警告
                    "PIP_NO_WARN_SCRIPT_LOCATION": "1",  # 抑制脚本位置警告
                },
            )
            output = exec_result.output.decode("utf-8", errors="replace")
            exit_code = exec_result.exit_code

            # 超时处理（exit code 124表示timeout命令超时）
            if exit_code == 124:
                return f"The command took too long to execute (>{timeout}s)", "-1"

            # 移除ANSI转义码
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            
            if exit_code != 0:
                return output, f"Error: Exit code {exit_code}"

            return output, str(exit_code)

        except Exception as e:
            return f"Error: {repr(e)}", "-1"

    def demux_run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        workdir: str = None,
    ) -> Tuple[str, str, str]:
        """
        在容器中执行命令（分离输出模式）。

        使用demux=True来分别获取stdout和stderr流。

        Args:
            code: 要执行的bash命令
            timeout: 超时时间（秒）
            workdir: 工作目录（默认为repo_path）

        Returns:
            (stdout, stderr, exit_code_or_error) - 标准输出、标准错误和退出码
        """
        exec_workdir = self.repo_path if workdir is None else workdir
        
        # 将命令包装在bash -c中以支持shell内置命令如'cd'
        command = ["bash", "-c", f"timeout {timeout} bash -c {shlex.quote(code)}"]

        try:
            exec_result = self.container.exec_run(
                command,
                workdir=exec_workdir,
                demux=True,  # 关键变化：分离stdout和stderr
                environment={
                    "PATH": DOCKER_PATH,
                    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                    "PIP_ROOT_USER_ACTION": "ignore",
                    "PIP_NO_WARN_SCRIPT_LOCATION": "1",
                },
            )
            
            # 当demux=True时，输出是一个(stdout_data, stderr_data)元组
            stdout_data, stderr_data = exec_result.output
            exit_code = exec_result.exit_code

            # 处理None情况并解码输出
            stdout = stdout_data.decode("utf-8", errors="replace") if stdout_data else ""
            stderr = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""

            # 超时处理
            if exit_code == 124:
                return f"The command took too long to execute (>{timeout}s)", "", "-1"

            # 移除ANSI转义码
            stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout)
            stderr = re.sub(r"\x1b\[[0-9;]*m|\r", "", stderr)
            
            if exit_code != 0:
                return stdout, stderr, f"Error: Exit code {exit_code}"

            return stdout, stderr, str(exit_code)

        except Exception as e:
            error_msg = f"Error: {repr(e)}"
            return error_msg, error_msg, "-1"

    def copy_to_container(self, src_path: str, dest_path: str):
        """
        将文件从宿主机复制到容器中。

        Args:
            src_path: 源文件路径（宿主机）
            dest_path: 目标路径（容器内）
        """
        try:
            # 确保目标目录存在
            dest_dir = os.path.dirname(dest_path)
            if dest_dir:
                self.run(f"mkdir -p {dest_dir}")

            # 在内存中创建tar存档
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                tar.add(src_path, arcname=os.path.basename(dest_path))
            tar_stream.seek(0)

            # 将存档放入容器
            self.container.put_archive(dest_dir or "/", tar_stream)
            self.logger.info(f"Copied {src_path} to {dest_path}")
        except Exception as e:
            self.logger.error(f"Error copying file to container: {repr(e)}")
            raise

    def copy_dir_to_container(self, src_dir: str, dest_dir: str):
        """
        将目录从宿主机复制到容器中。

        Args:
            src_dir: 源目录路径（宿主机）
            dest_dir: 目标目录路径（容器内）
        """
        try:
            # 确保目标目录存在
            self.run(f"mkdir -p {dest_dir}")

            # 在内存中创建tar存档
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                for item in os.listdir(src_dir):
                    item_path = os.path.join(src_dir, item)
                    tar.add(item_path, arcname=item)
            tar_stream.seek(0)

            # 将存档放入容器
            self.container.put_archive(dest_dir, tar_stream)
            self.logger.info(f"Copied directory {src_dir} to {dest_dir}")
        except Exception as e:
            self.logger.error(f"Error copying directory to container: {repr(e)}")
            raise

    def copy_from_container(self, container_path: str, local_path: str):
        """
        将文件或目录从容器复制到宿主机。

        Args:
            container_path: 容器内的路径
            local_path: 宿主机上的目标路径
        """
        try:
            # 从容器获取存档
            bits, stat = self.container.get_archive(container_path)
            
            # 创建本地目录
            os.makedirs(local_path, exist_ok=True)
            
            # 在内存中提取存档
            tar_stream = io.BytesIO()
            for chunk in bits:
                tar_stream.write(chunk)
            tar_stream.seek(0)
            
            # 从container_path获取基础目录名（例如"/testbed/output" -> "output"）
            base_dir = os.path.basename(container_path.rstrip('/'))
            
            with tarfile.open(fileobj=tar_stream, mode='r') as tar:
                # 处理所有成员：去除base_dir前缀并提取
                members_to_extract = []
                for member in tar.getmembers():
                    # 跳过根目录本身
                    if member.name == base_dir:
                        continue
                    
                    # 去除路径中的基础目录
                    if member.name.startswith(base_dir + '/'):
                        member.name = member.name[len(base_dir) + 1:]
                    
                    if not member.name:
                        continue
                    
                    members_to_extract.append(member)
                
                # 提取所有成员（自动处理嵌套目录）
                for member in members_to_extract:
                    target_path = os.path.join(local_path, member.name)
                    # 为所有类型（文件、目录、符号链接）创建父目录
                    parent_dir = os.path.dirname(target_path)
                    if parent_dir:
                        os.makedirs(parent_dir, exist_ok=True)
                    
                    tar.extract(member, path=local_path)
            
            self.logger.info(f"Copied {container_path} from container to {local_path}")
        except Exception as e:
            self.logger.error(f"Error copying from container: {repr(e)}")
            raise

    def get_task_instruction(self, problem_statement: str) -> str:
        """
        返回任务指令。

        Args:
            problem_statement: 问题描述

        Returns:
            任务指令字符串
        """
        return problem_statement

    def close(self):
        """
        停止并移除容器。

        这是清理资源的标准方法，应该在任务完成后调用。
        """
        if self.container:
            try:
                self.container.stop(timeout=5)
                self.container.remove(force=True)
                self.logger.info(f"Container {self.container_name} stopped and removed")
            except Exception as e:
                self.logger.warning(f"Error stopping container: {e}")
            finally:
                self.container = None  # 标记为已关闭
    
    def __del__(self):
        """
        析构函数 - 确保即使close()未被调用也能清理容器。

        这是最后的清理手段。
        """
        if hasattr(self, 'container') and self.container is not None:
            try:
                self.container.stop(timeout=2)
                self.container.remove(force=True)
            except Exception:
                pass  # 尽力而为的清理
