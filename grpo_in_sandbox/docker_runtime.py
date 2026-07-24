"""Docker运行时 - 在Docker容器中执行bash命令和文件传输。"""
import datetime
import hashlib
import io
import logging
import os
import re
import shlex
import shutil
import tarfile

from . import CMD_TIMEOUT, DOCKER_PATH

# --- Archive-extraction safety limits (copy_from_container) -------------------
# Untrusted container output is unpacked onto the host. Bound the work and reject
# anything that could escape the destination directory.
MAX_ARCHIVE_MEMBERS = 10000          # total entries permitted in one archive
MAX_ARCHIVE_MEMBER_BYTES = 512 * 1024 * 1024   # per-file cap: 512 MiB
MAX_ARCHIVE_TOTAL_BYTES = 2 * 1024 * 1024 * 1024  # whole-archive cap: 2 GiB

# --- Docker hardening ---------------------------------------------------------
# Applied to every container unless the caller explicitly overrides the key.
# root-with-no-capabilities keeps the documented pip/mkdir workflow intact while
# removing the kernel privileges an escape would need.
DEFAULT_SECURITY_KWARGS: dict = {
    "cap_drop": ["ALL"],
    "security_opt": ["no-new-privileges:true"],
    "pids_limit": 512,
    "mem_limit": "4g",
}

# docker_kwargs that hand the container a path to the host and are always refused.
_FORBIDDEN_DOCKER_KWARGS = {
    "privileged",       # full host access
    "devices",          # raw device access
    "cap_add",          # re-granting dropped capabilities
    "volumes",          # host bind mounts
    "mounts",           # host bind mounts
    "binds",            # host bind mounts
    "volumes_from",     # inherit another container's mounts
    "device_requests",  # GPUs/other host devices
}


def _validate_docker_kwargs(docker_kwargs: dict) -> None:
    """Reject docker_kwargs that would break sandbox isolation.

    Raises:
        ValueError: if a forbidden key or a host-sharing value is present.
    """
    for key in _FORBIDDEN_DOCKER_KWARGS:
        if docker_kwargs.get(key):
            raise ValueError(
                f"docker_kwargs[{key!r}] is not allowed: it can grant the sandbox "
                "access to the host. Remove it to keep the container isolated."
            )

    # Host namespace sharing (network/pid/ipc/userns=host) defeats isolation.
    for ns_key in ("network_mode", "pid_mode", "ipc_mode", "userns_mode", "network"):
        value = docker_kwargs.get(ns_key)
        if isinstance(value, str) and value.split(":", 1)[0] == "host":
            raise ValueError(
                f"docker_kwargs[{ns_key!r}]={value!r} shares a host namespace with "
                "the sandbox and is not allowed."
            )

    # security_opt that turns off the default confinement profiles.
    sec_opts = docker_kwargs.get("security_opt") or []
    for opt in sec_opts:
        normalized = str(opt).replace(" ", "").lower()
        if normalized in ("seccomp=unconfined", "apparmor=unconfined", "systempaths=unconfined"):
            raise ValueError(
                f"docker_kwargs['security_opt'] entry {opt!r} disables kernel "
                "confinement and is not allowed."
            )


def _is_within_directory(directory: str, target: str) -> bool:
    """True if ``target`` resolves to a path inside ``directory``."""
    directory = os.path.realpath(directory)
    target = os.path.realpath(target)
    prefix = directory if directory.endswith(os.sep) else directory + os.sep
    return target == directory or target.startswith(prefix)


def get_logger(name: str) -> logging.Logger:
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
        _validate_docker_kwargs(docker_kwargs)
        self.docker_kwargs = docker_kwargs

        if logger is None:
            self.logger = get_logger("DockerRuntime")
        elif logger is False:
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

        import docker  # lazy: keeps pure-Python helpers importable without the SDK

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
        import docker
        import docker.errors

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

        # Hardened defaults; explicit docker_kwargs win (forbidden ones already
        # rejected in __init__ via _validate_docker_kwargs).
        run_kwargs = {
            **DEFAULT_SECURITY_KWARGS,
            **{k: v for k, v in docker_kwargs.items() if k != "environment"},
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
            **run_kwargs,
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
                return output, f"Error: Exit code {exit_code}"

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
                return stdout, stderr, f"Error: Exit code {exit_code}"

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
                self._safe_extract(tar, local_path, base_dir)

            self.logger.info(f"Copied {container_path} from container to {local_path}")
        except Exception as e:
            self.logger.error(f"Error copying from container: {repr(e)}")
            raise

    def _safe_extract(self, tar: tarfile.TarFile, local_path: str, base_dir: str) -> None:
        """Extract a container archive onto the host, rejecting escapes.

        Container output is untrusted. Only regular files and directories are
        written, and every destination is verified to stay inside ``local_path``.
        Symlinks, hardlinks, device nodes and FIFOs are skipped, as are absolute
        paths and ``..`` traversal. Extraction is bounded by member count and
        size so a hostile archive cannot exhaust host disk.
        """
        dest_root = os.path.realpath(local_path)
        total_bytes = 0
        member_count = 0

        # Iterate lazily with ``tar.next()`` instead of ``getmembers()``: the
        # latter parses every header into an in-memory list up front, so a
        # hostile archive with a huge number of entries could exhaust host
        # memory before the member-count bound below is ever enforced. Reading
        # one header at a time keeps memory flat and lets us abort early.
        while True:
            member = tar.next()
            if member is None:
                break

            # Count every header read so a flood of entries (including ones
            # skipped below) cannot bypass the bound or spin indefinitely.
            member_count += 1
            if member_count > MAX_ARCHIVE_MEMBERS:
                raise ValueError(
                    f"Archive exceeds {MAX_ARCHIVE_MEMBERS} members; refusing to extract."
                )

            # Strip the leading base_dir component (e.g. "output/foo" -> "foo").
            name = member.name
            if name == base_dir:
                continue
            if name.startswith(base_dir + "/"):
                name = name[len(base_dir) + 1:]
            if not name:
                continue

            # Reject absolute paths and parent-directory traversal outright.
            if os.path.isabs(name) or ".." in name.split("/"):
                self.logger.warning(f"Skipping unsafe archive member path: {member.name!r}")
                continue

            # Only regular files and directories are allowed onto the host.
            if not (member.isfile() or member.isdir()):
                self.logger.warning(
                    f"Skipping non-regular archive member {member.name!r} "
                    f"(type={member.type!r})"
                )
                continue

            target_path = os.path.join(dest_root, name)
            if not _is_within_directory(dest_root, target_path):
                self.logger.warning(
                    f"Skipping archive member escaping destination: {member.name!r}"
                )
                continue

            if member.isdir():
                os.makedirs(target_path, exist_ok=True)
                continue

            if member.size > MAX_ARCHIVE_MEMBER_BYTES:
                raise ValueError(
                    f"Archive member {name!r} is {member.size} bytes, exceeding the "
                    f"{MAX_ARCHIVE_MEMBER_BYTES}-byte per-file limit."
                )
            total_bytes += member.size
            if total_bytes > MAX_ARCHIVE_TOTAL_BYTES:
                raise ValueError(
                    f"Archive exceeds the {MAX_ARCHIVE_TOTAL_BYTES}-byte total limit."
                )

            os.makedirs(os.path.dirname(target_path) or dest_root, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with source, open(target_path, "wb") as dst:
                shutil.copyfileobj(source, dst, length=1024 * 1024)

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
