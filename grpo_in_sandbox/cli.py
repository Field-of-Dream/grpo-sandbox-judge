#!/usr/bin/env python
"""
命令行界面模块 - 在本地Docker容器中运行LLM智能体

提供以下命令：
- build: 构建Docker镜像
- run: 运行智能体执行任务
- benchmark: 运行基准测试
- train: 运行GRPO训练

使用方法：
    llm-in-sandbox build
    llm-in-sandbox run --query "Your task" --llm_name gpt-4
    llm-in-sandbox benchmark --task math --llm_name gpt-4
"""
import datetime
import json
import os
import subprocess
import sys
import warnings
from importlib import resources
from pathlib import Path
from typing import Any

import docker
import docker.errors
import fire
import yaml
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .agent import Agent, AgentArgs, get_logger
from .docker_runtime import DockerRuntime

# 抑制来自litellm的pydantic序列化警告
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# Rich控制台
console = Console()

# 默认Docker镜像名称和版本
DEFAULT_DOCKER_IMAGE = "cdx123/llm-in-sandbox:v0.1"
# 环境变量名称，用于指定配置文件
SETTINGS_ENV_VAR = "LLM_IN_SANDBOX_CONFIG"
# 默认配置文件查找位置（按优先级顺序）
DEFAULT_SETTINGS_LOCATIONS = [
    Path.cwd() / "grpo-in-sandbox.yaml",
    Path.cwd() / "grpo_in_sandbox.yaml",
    Path.home() / ".grpo-in-sandbox" / "config.yaml",
    Path.home() / ".grpo-in-sandbox.yaml",
]


def _fix_string_bools(obj: Any) -> Any:
    """
    递归地将字符串'true'/'false'转换为布尔值True/False。

    用于处理从命令行JSON解析的布尔值问题。
    """
    if isinstance(obj, dict):
        return {k: _fix_string_bools(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_fix_string_bools(item) for item in obj]
    elif isinstance(obj, str):
        if obj.lower() == 'true':
            return True
        elif obj.lower() == 'false':
            return False
    return obj


def get_default_config_path() -> Path:
    """
    获取默认的提示词配置文件路径。

    查找顺序：
    1. 包资源目录中的config/general.yaml
    2. 回退到相对于__file__的路径
    """
    # 尝试从包资源获取
    try:
        with resources.files("grpo_in_sandbox.config") as config_dir:  # type: ignore[attr-defined]
            return Path(config_dir) / "general.yaml"
    except (TypeError, FileNotFoundError):
        # 回退到相对路径
        return Path(__file__).parent / "config" / "general.yaml"


def load_runtime_settings(explicit_path: str | None = None):
    """
    从YAML配置文件加载CLI默认值（llm_name/llm_base_url等）。

    配置文件查找顺序（第一个找到的生效）：
    1. 显式指定的路径（explicit_path）
    2. 环境变量 LLM_IN_SANDBOX_CONFIG 指定的路径
    3. 默认位置列表中的文件
    """
    candidates = []
    seen = set()

    def _add_candidate(candidate):
        """添加候选配置文件路径。"""
        if not candidate:
            return
        path = Path(candidate).expanduser()
        if path in seen:
            return
        seen.add(path)
        candidates.append(path)

    _add_candidate(explicit_path)
    _add_candidate(os.environ.get(SETTINGS_ENV_VAR))
    for default_path in DEFAULT_SETTINGS_LOCATIONS:
        _add_candidate(default_path)

    for candidate in candidates:
        if candidate.is_file():
            with open(candidate) as f:
                data = yaml.safe_load(f) or {}
            return data, candidate

    return {}, None


def find_dockerfile() -> Path | None:
    """
    查找用于构建默认镜像的Dockerfile。

    查找顺序：
    1. 开发模式：docker/ 是 grpo_in_sandbox/ 的同级目录
    2. 安装模式：检查 sys.prefix/share/grpo-in-sandbox/docker
    """
    # 尝试1：开发模式 - docker/ 是 grpo_in_sandbox/ 的同级目录
    script_dir = Path(__file__).parent
    dev_docker_dir = script_dir.parent / "docker"
    if (dev_docker_dir / "Dockerfile").exists():
        return dev_docker_dir / "Dockerfile"

    # 尝试2：安装模式 - 检查 sys.prefix 中的共享数据
    installed_docker_dir = Path(sys.prefix) / "share" / "grpo-in-sandbox" / "docker"
    if (installed_docker_dir / "Dockerfile").exists():
        return installed_docker_dir / "Dockerfile"

    return None


def ensure_docker_image(image_name: str, logger) -> bool:
    """
    确保Docker镜像存在。如果本地不存在，则尝试从Docker Hub拉取。

    Args:
        image_name: Docker镜像名称
        logger: 日志记录器

    Returns:
        镜像是否可用
    """
    client = docker.from_env()

    try:
        client.images.get(image_name)
        return True  # 镜像已存在
    except docker.errors.ImageNotFound:
        # 尝试从Docker Hub拉取
        console.print(Panel.fit(
            f"[yellow]🐳 Docker镜像 '{image_name}' 本地未找到。[/yellow]\n"
            f"[dim]正在从Docker Hub拉取...[/dim]",
            border_style="yellow",
        ))
        try:
            logger.info(f"正在从Docker Hub拉取镜像 '{image_name}'...")
            client.images.pull(image_name)
            console.print(Panel.fit(
                f"[green]✅ 成功拉取Docker镜像 '{image_name}'！[/green]",
                border_style="green",
            ))
            return True
        except docker.errors.APIError as e:
            logger.warning(f"拉取镜像失败: {e}")
            return False


def build_docker_image(
    image_name: str = DEFAULT_DOCKER_IMAGE,
    force: bool = False,
):
    """
    构建LLM-in-Sandbox的Docker镜像。

    此命令构建智能体使用的默认Docker镜像。
    在使用'run'命令之前只需运行一次。

    Args:
        image_name: 要构建的Docker镜像名称（默认：llm-in-sandbox:v0.1）
        force: 即使镜像已存在也强制重建

    Example:
        llm-in-sandbox build
        llm-in-sandbox build --force  # 强制重建
        llm-in-sandbox build --image_name my-custom-image:v1
    """
    get_logger("llm-in-sandbox")
    client = docker.from_env()

    # 检查镜像是否已存在
    if not force:
        try:
            client.images.get(image_name)
            console.print(Panel.fit(
                f"[green]✅ Docker镜像 '{image_name}' 已存在！[/green]\n"
                f"[dim]使用 --force 强制重建[/dim]",
                border_style="green",
            ))
            return
        except docker.errors.ImageNotFound:
            pass

    # 查找Dockerfile
    dockerfile = find_dockerfile()
    if dockerfile is None:
        console.print(Panel.fit(
            f"[red]❌ 无法找到用于构建 '{image_name}' 的Dockerfile[/red]\n"
            f"[dim]请手动构建：docker build -t {image_name} <path-to-dockerfile>[/dim]",
            border_style="red",
        ))
        sys.exit(1)

    # 构建镜像
    console.print()
    console.print(Panel.fit(
        f"[yellow]🐳 正在构建Docker镜像 '{image_name}'...[/yellow]\n"
        f"[dim]Dockerfile: {dockerfile}[/dim]",
        border_style="yellow",
    ))
    console.print()

    docker_dir = dockerfile.parent
    try:
        subprocess.run(
            ["docker", "build", "-t", image_name, "-f", str(dockerfile), str(docker_dir)],
            check=True,
        )
        console.print()
        console.print(Panel.fit(
            f"[green]✅ Docker镜像 '{image_name}' 构建成功！[/green]\n"
            f"[dim]现在可以运行：llm-in-sandbox run --query \"Your task\"[/dim]",
            border_style="green",
        ))
    except subprocess.CalledProcessError as e:
        console.print(Panel.fit(
            f"[red]❌ 构建Docker镜像失败（退出码 {e.returncode}）[/red]",
            border_style="red",
        ))
        sys.exit(1)
    except FileNotFoundError:
        console.print(Panel.fit(
            "[red]❌ 未找到Docker。请先安装Docker。[/red]",
            border_style="red",
        ))
        sys.exit(1)


def load_prompt_config(config_path: str) -> dict:
    """
    从yaml文件加载提示词配置。

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def run_agent_query(
    query: str,
    llm_name: str | None = None,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    max_steps: int = 100,
    temperature: float = 1.0,
    max_token_limit: int = 65536,
    max_tokens_per_call: int = 65536,
    input_dir: str | None = None,
    output_dir: str | None = None,
    llm_base_url: str | None = None,
    api_key: str | None = None,
    prompt_config: str | None = None,
    save_litellm_response: bool = False,
    extra_body: str | None = None,
    settings: str | None = None,
):
    """
    在Docker容器中运行LLM智能体完成任务。

    参数:
        query: 任务描述/问题陈述
        llm_name: LLM模型名称
        docker_image: 使用的Docker镜像（默认：cdx123/llm-in-sandbox:v0.1）
        max_steps: 最大步数（默认：100）
        temperature: LLM采样温度（默认：1.0）
        max_token_limit: 整个轨迹的最大token限制
        max_tokens_per_call: 每次LLM API调用的最大token数
        input_dir: 本地目录，复制到容器的{input_dir}
        output_dir: 本地目录，保存容器的{output_dir}内容
        llm_base_url: LLM API基础URL（默认：从LLM_BASE_URL环境变量获取）
        api_key: LLM服务的API密钥（默认：从OPENAI_API_KEY环境变量获取）
        prompt_config: 包含system_prompt和instance_prompt的yaml文件路径
        save_litellm_response: 是否保存完整的litellm响应
        extra_body: 包含在LLM API调用中的额外JSON体
        settings: 提供默认值的YAML文件路径（如llm_name和llm_base_url）

    返回:
        包含所有步骤和结果的Trajectory对象
    """
    logger = get_logger("llm-in-sandbox")

    # 加载运行时设置
    runtime_settings, runtime_settings_path = load_runtime_settings(settings)
    if runtime_settings_path:
        logger.info(f"已从以下位置加载运行时设置：{runtime_settings_path}")

    def _with_setting(value, key):
        """获取设置值，优先使用显式值，否则使用配置文件中的值。"""
        if value in (None, ""):
            return runtime_settings.get(key)
        return value

    # 解析参数：CLI参数 > 环境变量 > 配置文件 > 默认值
    llm_name = _with_setting(llm_name, "llm_name")
    llm_base_url = _with_setting(llm_base_url, "llm_base_url")
    api_key = _with_setting(api_key, "api_key")
    prompt_config = _with_setting(prompt_config, "prompt_config")

    if not llm_name:
        raise ValueError(
            "llm_name是必需的。请提供 --llm_name 或在设置YAML文件中设置。"
        )

    # 保存原始环境变量以便恢复
    _orig_openai_key = os.environ.get("OPENAI_API_KEY")
    _orig_anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    _orig_llm_base_url = os.environ.get("LLM_BASE_URL")

    # 根据模型类型设置API密钥
    if api_key:
        os.environ["OPENAI_API_KEY"] = str(api_key)
        os.environ["ANTHROPIC_API_KEY"] = str(api_key)
    else:
        # 如果未提供，设置虚拟密钥（某些服务器不需要认证）
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = "dummy"
        if not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = "dummy"

    # 从yaml加载提示词配置（如果未提供则使用默认）
    config_path = prompt_config if prompt_config else get_default_config_path()
    if Path(config_path).exists():
        logger.info(f"正在加载提示词配置：{config_path}")
        config = load_prompt_config(str(config_path))
        system_prompt = config.get("system_prompt", "")
        instance_prompt = config.get("instance_prompt", "")
        # 从配置获取容器路径（默认：/testbed, /testbed/input, /testbed/output）
        working_dir = config.get("working_dir", "/testbed")
        container_input_dir = config.get("input_dir", "/testbed/input")
        container_output_dir = config.get("output_dir", "/testbed/output")
        # 替换提示词中的占位符
        system_prompt = system_prompt.replace("{input_dir}", container_input_dir).replace("{output_dir}", container_output_dir).replace("{working_dir}", working_dir)
        instance_prompt = instance_prompt.replace("{input_dir}", container_input_dir).replace("{output_dir}", container_output_dir).replace("{working_dir}", working_dir)
    else:
        raise FileNotFoundError(f"未找到提示词配置：{config_path}")

    # 为自定义LLM端点自动添加openai/前缀
    if not llm_name.startswith(("openai/", "anthropic/", "azure/", "hosted_vllm/")):
        llm_name = f"openai/{llm_name}"
        logger.info(f"自动为模型添加 'openai/' 前缀：{llm_name}")

    # 设置带时间戳的输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path.cwd() / "output" / timestamp  # type: ignore[assignment]
    else:
        output_dir = Path(str(output_dir)) / timestamp  # type: ignore[assignment]
    output_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]

    # 设置LLM基础URL
    if llm_base_url:
        os.environ["LLM_BASE_URL"] = llm_base_url

    # 确保Docker镜像存在
    if not ensure_docker_image(docker_image, logger):
        console.print(Panel.fit(
            f"[red]❌ 未找到Docker镜像 '{docker_image}'！[/red]\n"
            f"[dim]请先构建：llm-in-sandbox build[/dim]",
            border_style="red",
        ))
        sys.exit(1)

    # 初始化Docker运行时
    logger.info("正在启动Docker容器...")
    runtime = DockerRuntime(
        docker_image=docker_image,
        repo_path=working_dir,
        logger=logger,
    )

    # 如果提供了input_dir，则复制输入文件到容器
    if input_dir and os.path.isdir(input_dir):
        logger.info(f"正在复制输入文件从 {input_dir} 到容器的 {container_input_dir}")
        runtime.copy_dir_to_container(input_dir, container_input_dir)

    try:
        # 处理extra_body：可以是dict（来自fire）或JSON字符串
        extra_body_dict = None
        if extra_body:
            if isinstance(extra_body, dict):
                extra_body_dict = extra_body
            elif isinstance(extra_body, str):
                try:
                    extra_body_dict = json.loads(extra_body)
                except json.JSONDecodeError as e:
                    logger.error(f"解析extra_body JSON失败: {e}")
                    raise ValueError(f"无效的extra_body JSON: {extra_body}") from e
            # 修复字符串布尔值如'true' -> True
            extra_body_dict = _fix_string_bools(extra_body_dict)
            logger.info(f"使用extra_body: {extra_body_dict}")

        # 初始化智能体
        agent_args = AgentArgs(
            system_prompt=system_prompt,
            instance_prompt=instance_prompt,
            llm_name=llm_name,
            llm_base_url=llm_base_url or os.environ.get("LLM_BASE_URL"),
            save_litellm_response=save_litellm_response,
            output_dir=str(output_dir),
            extra_body=extra_body_dict,
        )
        agent = Agent(args=agent_args, logger=logger)

        # 运行智能体
        logger.info("正在启动智能体...")
        trajectory = agent.run(
            runtime=runtime,
            problem_statement=query,
            max_steps=max_steps,
            temperature=temperature,
            max_token_limit=max_token_limit,
            max_tokens_per_call=max_tokens_per_call,
        )

        # 将输出文件从容器复制到files/子目录
        files_dir = Path(str(output_dir)) / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"正在复制输出文件从容器的 {container_output_dir} 到 {files_dir}")
        try:
            runtime.copy_from_container(container_output_dir, str(files_dir))
        except Exception as e:
            logger.warning(f"无法复制容器输出：{e}")

        # 保存轨迹
        trajectory_file = Path(str(output_dir)) / "trajectory.json"

        with open(trajectory_file, "w") as f:
            json.dump(trajectory.to_dict(), f, indent=2, ensure_ascii=False)

        # 打印完成横幅
        console.print()
        console.print(Panel.fit(
            f"[bold green]✅ 智能体在 {len(trajectory.steps)} 步内完成[/bold green]",
            border_style="green",
        ))

        # 打印输出路径
        console.print()
        console.print("[bold]📦 输出保存到：[/bold]")
        paths_table = Table(show_header=False, box=None, padding=(0, 2))
        paths_table.add_column("标签", style="bold blue")
        paths_table.add_column("路径", style="white")
        paths_table.add_row("智能体输出文件", str(files_dir))
        paths_table.add_row("执行轨迹", str(trajectory_file))
        console.print(paths_table)

        # 如果存在answer.txt则打印
        answer_file = files_dir / "answer.txt"
        if answer_file.exists():
            answer_content = answer_file.read_text().strip()
            if answer_content:
                console.print()
                console.print(Panel(
                    f"{escape(answer_content)}\n\n[dim]📁 {answer_file}[/dim]",
                    title="[bold cyan]📄 答案[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                ))

    finally:
        # 清理并恢复原始环境变量
        logger.info("正在清理Docker容器...")
        runtime.close()

        # 恢复原始环境变量
        if _orig_openai_key is not None:
            os.environ["OPENAI_API_KEY"] = _orig_openai_key
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if _orig_anthropic_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = _orig_anthropic_key
        elif "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]
        if _orig_llm_base_url is not None:
            os.environ["LLM_BASE_URL"] = _orig_llm_base_url
        elif "LLM_BASE_URL" in os.environ:
            del os.environ["LLM_BASE_URL"]


def run_benchmark(
     task: str,
     llm_name: str | None = None,
     docker_image: str = DEFAULT_DOCKER_IMAGE,
     max_steps: int = 100,
     temperature: float | None = None,
     max_token_limit: int = 65536,
     max_tokens_per_call: int = 65536,
     max_response_len: int = 65536,
     output_dir: str | None = None,
     llm_base_url: str | None = None,
     api_key: str | None = None,
     extra_body: str | None = None,
     settings: str | None = None,
     num_workers: int | None = None,
     start_id: int | None = None,
     end_id: int | None = None,
     mode: str = "llm-in-sandbox",
     save_litellm_response: bool = False,
 ):
    """
    在特定任务上运行基准测试。

    支持两种模式：
    - "llm-in-sandbox"（默认）：使用沙箱环境运行智能体
    - "llm"：不使用沙箱，直接调用LLM API

    Args:
        task: 任务名称（如 math, physics, biomed 等）
        llm_name: LLM模型名称
        docker_image: Docker镜像（仅llm-in-sandbox模式需要）
        max_steps: 最大步数
        temperature: 采样温度
        max_token_limit: 最大token限制
        max_tokens_per_call: 每次调用最大token数
        max_response_len: 最大响应长度（仅llm模式）
        output_dir: 输出目录
        llm_base_url: LLM API基础URL
        api_key: API密钥
        extra_body: 额外的请求体
        settings: 设置文件路径
        num_workers: 并行工作进程数
        start_id: 起始索引（包含）
        end_id: 结束索引（不包含）
        mode: 运行模式 - "llm-in-sandbox" 或 "llm"
        save_litellm_response: 是否保存litellm响应

    Example:
        llm-in-sandbox benchmark --task math --llm_name openai/gpt-5 --num_workers 4
        llm-in-sandbox benchmark --task math --llm_name openai/gpt-5 --mode llm
    """
    from grpo_in_sandbox.benchmark.runner import run_benchmark as _run_benchmark

    logger = get_logger("grpo-in-sandbox")
    runtime_settings, _ = load_runtime_settings(settings)

    # 解析参数：CLI参数 > 环境变量 > 配置文件 > 默认值
    llm_name = llm_name or os.environ.get("LLM_NAME") or runtime_settings.get("llm_name")
    llm_base_url = llm_base_url or os.environ.get("LLM_BASE_URL") or runtime_settings.get("llm_base_url")
    api_key = api_key or os.environ.get("LLM_API_KEY") or runtime_settings.get("api_key")
    if temperature is None:
        temperature = float(os.environ.get("LLM_TEMPERATURE", "1.0"))
    if num_workers is None:
        num_workers = int(os.environ.get("LLM_NUM_WORKERS", "1"))

    if not llm_name:
        raise ValueError("llm_name是必需的")

    # 设置API密钥（为本地vLLM使用占位符）
    api_key = api_key or "sk-placeholder"
    os.environ["OPENAI_API_KEY"] = os.environ["ANTHROPIC_API_KEY"] = str(api_key)
    if llm_base_url:
        os.environ["LLM_BASE_URL"] = llm_base_url
    if not llm_name.startswith(("openai/", "anthropic/", "azure/", "hosted_vllm/")):
        llm_name = f"openai/{llm_name}"

    # 设置输出目录：output/{timestamp}_{task}_{llm_name}_{mode}/
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "vanillaLLM" if mode == "llm" else "LLMinSandbox"
    llm_name_safe = llm_name.replace("/", "_")  # openai/qwen3_coder -> openai_qwen3_coder
    output_dir = Path(output_dir or Path.cwd() / "output") / f"{timestamp}_{task}_{llm_name_safe}_{mode_suffix}"  # type: ignore[assignment]
    output_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]

    # 验证模式
    if mode not in ("llm", "llm-in-sandbox"):
        raise ValueError(f"无效的模式：{mode}。必须是'llm'或'llm-in-sandbox'")

    # 仅在llm-in-sandbox模式下检查docker镜像
    if mode == "llm-in-sandbox" and not ensure_docker_image(docker_image, logger):
        console.print(f"[red]未找到Docker镜像 '{docker_image}'！[/red]")
        sys.exit(1)

    # 处理extra_body
    extra_body_dict = None
    if extra_body:
        if isinstance(extra_body, dict):
            extra_body_dict = extra_body
        elif isinstance(extra_body, str):
            try:
                extra_body_dict = json.loads(extra_body)
            except json.JSONDecodeError as e:
                logger.error(f"解析extra_body JSON失败: {e}")
                raise ValueError(f"无效的extra_body JSON: {extra_body}") from e
        # 修复字符串布尔值
        extra_body_dict = _fix_string_bools(extra_body_dict)
        logger.info(f"使用extra_body: {extra_body_dict}")

    # 智能体配置（传递给子进程）
    agent_config = {
        "docker_image": docker_image,
        "llm_name": llm_name,
        "llm_base_url": llm_base_url or os.environ.get("LLM_BASE_URL"),
        "max_steps": max_steps,
        "temperature": temperature,
        "max_token_limit": max_token_limit,
        "max_tokens_per_call": max_tokens_per_call,
        "max_response_len": max_response_len,
        "extra_body": extra_body_dict,
        "save_litellm_response": save_litellm_response,
    }

    # 运行基准测试
    results = _run_benchmark(
        task_name=task,
        agent_config=agent_config,
        output_dir=str(output_dir),
        num_workers=num_workers,
        start_id=start_id,
        end_id=end_id,
        mode=mode,
    )

    return results


def config(
     show: bool = False,
     set_llm_name: str | None = None,
     set_base_url: str | None = None,
     set_api_key: str | None = None,
     set_prompt_config: str | None = None,
     init: bool = False,
 ):
    """
    管理LLM API配置设置。

    显示或修改存储在 ~/.grpo-in-sandbox/config.yaml 的配置。

    Args:
        show: 显示当前配置及来源
        set_llm_name: 设置默认模型名称 (如 openai/gpt-4o-mini)
        set_base_url: 设置API基础URL (如 https://api.openai.com/v1)
        set_api_key: 设置API密钥
        set_prompt_config: 设置提示词配置文件路径
        init: 初始化配置文件（使用默认值或交互式提示）

    Example:
        llm-in-sandbox config --show
        llm-in-sandbox config --set-llm-name openai/gpt-4o
        llm-in-sandbox config --init
    """
    config_path = Path.home() / ".grpo-in-sandbox" / "config.yaml"

    # 初始化模式：创建默认配置
    if init:
        # 确保目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)

        default_config = {
            "llm_name": "openai/gpt-4o-mini",
            "llm_base_url": "https://api.openai.com/v1",
            # api_key - 不设置默认值，需要用户显式提供
            # prompt_config - 不设置默认值
        }

        # 交互式提示用户输入
        console.print(Panel.fit(
            "[bold]初始化LLM配置[/bold]\n"
            f"配置文件将保存到: [cyan]{config_path}[/cyan]",
            border_style="blue",
        ))

        # 读取用户输入
        llm_name = input("模型名称 [openai/gpt-4o-mini]: ").strip()
        if not llm_name:
            llm_name = default_config["llm_name"]

        base_url = input("API Base URL [https://api.openai.com/v1]: ").strip()
        if not base_url:
            base_url = default_config["llm_base_url"]

        api_key = input("API密钥 (直接输入或留空跳过): ").strip()
        prompt_config = input("提示词配置文件路径 (留空跳过): ").strip()

        # 构建配置
        new_config = {
            "llm_name": llm_name,
            "llm_base_url": base_url,
        }
        if api_key:
            new_config["api_key"] = api_key
        if prompt_config:
            new_config["prompt_config"] = prompt_config

        # 写入文件
        with open(config_path, "w") as f:
            yaml.safe_dump(new_config, f, default_flow_style=False, sort_keys=False)

        console.print(Panel.fit(
            f"[green]✅ 配置已保存到 {config_path}[/green]",
            border_style="green",
        ))
        console.print()

    # 处理设置请求
    set_values = {}
    if set_llm_name is not None:
        set_values["llm_name"] = set_llm_name
    if set_base_url is not None:
        set_values["llm_base_url"] = set_base_url
    if set_api_key is not None:
        set_values["api_key"] = set_api_key
    if set_prompt_config is not None:
        set_values["prompt_config"] = set_prompt_config

    if set_values:
        # 确保目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # 读取现有配置并合并
        config_data: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path) as f:
                config_data = yaml.safe_load(f) or {}

        # 合并新值
        config_data.update(set_values)

        # 写入文件
        with open(config_path, "w") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)

        console.print(Panel.fit(
            f"[green]✅ 已更新配置: {', '.join(set_values.keys())}[/green]",
            border_style="green",
        ))
        console.print()

    # 显示模式（默认或带--show 或设置值后确认）
    if show or (not init and not set_values) or set_values:
        console.print(Panel.fit(
            "[bold]当前LLM配置[/bold]",
            border_style="blue",
        ))

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("配置项", style="cyan", width=20)
        table.add_column("值", style="white")
        table.add_column("来源", style="dim", width=30)

        # 确定每个配置值的来源
        config_keys = ["llm_name", "llm_base_url", "api_key", "prompt_config"]
        env_var_map = {
            "llm_name": "LLM_NAME",
            "llm_base_url": "LLM_BASE_URL",
            "api_key": "OPENAI_API_KEY",
            "prompt_config": "LLM_PROMPT_CONFIG",
        }

        for key in config_keys:
            value = None
            source = None

            # 检查环境变量
            if key in env_var_map:
                env_val = os.environ.get(env_var_map[key])
                if env_val:
                    value = env_val
                    source = f"环境变量 ${env_var_map[key]}"

            # 检查配置文件（如果环境变量未设置）
            if value is None:
                # 检查显式配置文件路径
                for candidate in DEFAULT_SETTINGS_LOCATIONS:
                    if candidate.is_file():
                        with open(candidate) as f:
                            file_data = yaml.safe_load(f) or {}
                        if key in file_data:
                            value = file_data[key]
                            source = f"配置文件 {candidate}"
                            break

            # 如果配置文件中也没有，检查运行时设置（包含默认值）
            if value is None:
                runtime_val, location = load_runtime_settings()
                if key in runtime_val and runtime_val[key]:
                    value = runtime_val[key]
                    source = f"配置文件 {location}" if location else "默认值"

            # 处理敏感值（如api_key）
            display_value = value if value else "(未设置)"
            if key == "api_key" and value and len(value) > 4:
                display_value = value[:4] + "****"

            table.add_row(key, display_value, source or "默认值")

        console.print(table)

        # 显示配置文件路径（帮助用户定位）
        console.print(f"\n[dim]配置文件路径: {config_path}[/dim]")
        console.print(f"[dim]查找顺序: {DEFAULT_SETTINGS_LOCATIONS}[/dim]")


def run_training(
    model_name: str = "./model",
    prompt: str | None = None,
    prompt_file: str | None = None,
    dataset_file: str | None = None,
    dataset_format: str = "json",
    output_dir: str = "./output",
    max_steps: int = 100,
    num_train_epochs: int = 3,
    learning_rate: float = 5e-6,
    lora_rank: int = 16,
    per_device_train_batch_size: int = 4,
    num_generations: int = 4,
    max_seq_length: int = 2048,
    temperature: float = 0.7,
    use_vllm: bool = True,
):
    """Run GRPO training with the given configuration.

    Trains a language model using GRPO (Group Relative Policy Optimization)
    within a code sandbox environment.

    Args:
        model_name: HuggingFace model name or path (default: "./model")
        prompt: Single training prompt. If not provided, uses default QA prompts.
        prompt_file: Path to file with prompts (one per line).
        dataset_file: Path to dataset file (JSON or CSV) for training prompts.
        dataset_format: Format of dataset file: "json" or "csv" (default: "json").
        output_dir: Output directory for trained model.
        max_steps: Maximum training steps (default: 100).
        num_train_epochs: Number of training epochs (default: 3).
        learning_rate: Learning rate (default: 5e-6).
        lora_rank: LoRA rank (default: 16).
        per_device_train_batch_size: Per-device batch size (default: 4).
        num_generations: Generations per prompt for GRPO (default: 4).
        max_seq_length: Max sequence length (default: 2048).
        temperature: Sampling temperature (default: 0.7).
        use_vllm: Use vLLM fast inference (default: True).

    Example:
        llm-sandbox train --model_name Qwen/Qwen2.5-0.5B-Instruct --prompt "写一个排序函数" --max_steps 10
        llm-sandbox train --model_name ./model --prompt_file prompts.txt --epochs 3
        llm-sandbox train --model_name ./model --dataset_file data.json --dataset_format json
    """
    from grpo_in_sandbox import RLHFTrainingConfig, ProductManager, train

    config = RLHFTrainingConfig(
        model_name_or_path=model_name,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        max_seq_length=max_seq_length,
        temperature=temperature,
        use_vllm=use_vllm,
        output_dir=output_dir,
    )

    if dataset_file:
        if dataset_format == "csv":
            pm = ProductManager.from_csv(dataset_file)
        else:
            pm = ProductManager.from_json(dataset_file)
    elif prompt:
        pm = ProductManager(task_templates=[prompt])
    elif prompt_file:
        pm = ProductManager.from_file(prompt_file)
    else:
        pm = ProductManager()  # uses default Chinese QA templates

    results = train(config, product_manager=pm)

    # Print summary
    console.print()
    console.print(Panel.fit(
        f"[bold green]✅ GRPO训练完成！[/bold green]",
        border_style="green",
    ))
    console.print(f"模型: {results['config']['model_name_or_path']}")
    console.print(f"输出目录: {results['output_dir']}")
    console.print(f"最终检查点: {results['final_checkpoint']}")
    return results


def main():
    """
    CLI主入口点。

    使用fire库将函数转换为命令行接口。
    """
    fire.Fire({
        "run": run_agent_query,
        "build": build_docker_image,
        "benchmark": run_benchmark,
        "config": config,
        "train": run_training,
    })


if __name__ == "__main__":
    main()
