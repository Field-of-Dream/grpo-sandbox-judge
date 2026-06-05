"""Docker运行时 - 在Docker容器中执行bash命令和文件传输。"""
import datetime
import hashlib
import io
import logging
import os
import re
import shlex
import tarfile

import docker
import docker.errors

from .agent import get_logger
from .runtime import CMD_TIMEOUT, DOCKER_PATH, BaseRuntime


class DockerRuntime(BaseRuntime):
    """Docker运行时 - 管理容器生命周期和执行命令。"""

    def __init__(
        self,
        docker_image: str,
        repo_path: str = "/testbed",
        command: str = "sleep infinity",
        logger: logging.Logger | None = None,
        **docker_kwargs,
    ):
        self.docker_image = docker_image
        self.repo_path = repo_path
        self.command = command
        self.docker_kwargs = docker_kwargs

        if logger is None:
            self.logger = get_logger("DockerRuntime")
        else:
            self.logger = logger

        self.client = docker.from_env(timeout=120)

        self.container = None
        self.container_name = self._get_container_name(docker_image)
        self.start_container(docker_image, command, self.container_name, **docker_kwargs)

        self.setup_env()
        self.logger.info("Docker environment initialized")
        self.logger.info(f"Docker image: {self.docker_image}")
        assert self.container is not None, "Container failed to start"
        self.logger.info(f"Container ID: {self.container.id}")

    @staticmethod
    def _get_container_name(image_name: str) -> str:
        process_id = str(os.getpid())
        current_time = str(datetime.datetime.now())
        unique_string = current_time + process_id
        hash_object = hashlib.sha256(unique_string.encode())
        image_name_sanitized = image_name.replace("/", "-").replace(":", "-")
        return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"

    def start_container(self, docker_image: str, command: str, container_name: str, **docker_kwargs):
        try:
            self.container = self.client.containers.get(container_name)
            assert self.container is not None
            self.logger.info(f"Found existing container: {container_name}")
            if self.container.status != "running":
                self.container.start()
            return
        except docker.errors.NotFound:
            pass

        try:
            self.client.images.get(docker_image)
        except docker.errors.ImageNotFound:
            self.logger.info(f"Pulling Docker image: {docker_image}")
            self.client.images.pull(docker_image)

        env_vars = {
            "PATH": DOCKER_PATH,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_ROOT_USER_ACTION": "ignore",
            "PIP_NO_WARN_SCRIPT_LOCATION": "1",
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
        self.run(f"mkdir -p {self.repo_path}")
        self.run(f"mkdir -p {self.repo_path}/input {self.repo_path}/output")
        self.run(f"cd {self.repo_path} && git init 2>/dev/null || true")
        self.run("mkdir -p ~/.pip && cat > ~/.pip/pip.conf << 'EOF'\n"
                 "[global]\n"
                 "index-url = https://pypi.tuna.tsinghua.edu.cn/simple\n"
                 "trusted-host = pypi.tuna.tsinghua.edu.cn\n"
                 "EOF")

    def run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        workdir: str | None = None,
    ) -> tuple[str, str]:
        """在容器中执行命令（组合输出模式）。"""
        exec_workdir = self.repo_path if workdir is None else workdir
        command = ["bash", "-c", f"timeout {timeout} bash -c {shlex.quote(code)}"]

        try:
            assert self.container is not None
            exec_result = self.container.exec_run(
                command,
                workdir=exec_workdir,
                environment={
                    "PATH": DOCKER_PATH,
                    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                    "PIP_ROOT_USER_ACTION": "ignore",
                    "PIP_NO_WARN_SCRIPT_LOCATION": "1",
                },
            )
            output = exec_result.output.decode("utf-8", errors="replace")
            exit_code = exec_result.exit_code

            if exit_code == 124:
                return f"The command took too long to execute (>{timeout}s)", "-1"

            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)

            if exit_code != 0:
                return output, str(exit_code)

            return output, str(exit_code)

        except Exception as e:
            return f"Error: {repr(e)}", "-1"

    def demux_run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        workdir: str | None = None,
    ) -> tuple[str, str, str]:
        """在容器中执行命令（分离输出模式）。"""
        exec_workdir = self.repo_path if workdir is None else workdir
        command = ["bash", "-c", f"timeout {timeout} bash -c {shlex.quote(code)}"]

        try:
            assert self.container is not None
            exec_result = self.container.exec_run(
                command,
                workdir=exec_workdir,
                demux=True,
                environment={
                    "PATH": DOCKER_PATH,
                    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                    "PIP_ROOT_USER_ACTION": "ignore",
                    "PIP_NO_WARN_SCRIPT_LOCATION": "1",
                },
            )

            stdout_data, stderr_data = exec_result.output
            exit_code = exec_result.exit_code

            stdout = stdout_data.decode("utf-8", errors="replace") if stdout_data else ""
            stderr = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""

            if exit_code == 124:
                return f"The command took too long to execute (>{timeout}s)", "", "-1"

            stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout)
            stderr = re.sub(r"\x1b\[[0-9;]*m|\r", "", stderr)

            if exit_code != 0:
                return stdout, stderr, str(exit_code)

            return stdout, stderr, str(exit_code)

        except Exception as e:
            error_msg = f"Error: {repr(e)}"
            return error_msg, error_msg, "-1"

    def copy_to_container(self, src_path: str, dest_path: str):
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            self.run(f"mkdir -p {dest_dir}")

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            tar.add(src_path, arcname=os.path.basename(dest_path))
        tar_stream.seek(0)

        assert self.container is not None
        self.container.put_archive(dest_dir or "/", tar_stream)
        self.logger.info(f"Copied {src_path} to {dest_path}")

    def copy_dir_to_container(self, src_dir: str, dest_dir: str):
        self.run(f"mkdir -p {dest_dir}")

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            for item in os.listdir(src_dir):
                item_path = os.path.join(src_dir, item)
                tar.add(item_path, arcname=item)
        tar_stream.seek(0)

        assert self.container is not None
        self.container.put_archive(dest_dir, tar_stream)
        self.logger.info(f"Copied directory {src_dir} to {dest_dir}")

    def copy_from_container(self, container_path: str, local_path: str):
        """
        将文件或目录从容器复制到宿主机。

        Args:
            container_path: 容器内的路径
            local_path: 宿主机上的目标路径
        """
        try:
            # 从容器获取存档
            assert self.container is not None
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
        return problem_statement

    def close(self):
        if self.container:
            try:
                self.container.stop(timeout=5)
                self.container.remove(force=True)
                self.logger.info(f"Container {self.container_name} stopped and removed")
            except Exception as e:
                self.logger.warning(f"Error stopping container: {e}")
            finally:
                self.container = None

    def __del__(self):
        if hasattr(self, 'container') and self.container is not None:
            try:
                self.container.stop(timeout=2)
                self.container.remove(force=True)
            except Exception:
                pass
