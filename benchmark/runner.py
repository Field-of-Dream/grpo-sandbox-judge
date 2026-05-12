"""
基准测试运行器模块 - 支持并行执行的基准测试框架

本模块提供在各种任务（如math、physics、biomed等）上运行LLM基准测试的功能。
支持两种模式：
1. llm-in-sandbox模式：在Docker沙箱中运行完整智能体
2. llm模式：直接调用LLM API（无沙箱）

主要功能：
- 加载任务配置、数据集和评分函数
- 支持单进程和多进程并行执行
- 自动处理token限制等错误
- 生成详细的轨迹和结果日志
"""

import atexit
import importlib.util
import json
import os
import signal
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BenchmarkResult:
    """
    单个基准测试问题的结果。

    Attributes:
        problem_id: 问题ID
        score: 评分（0.0到1.0）
        agent_answer: 智能体的回答
        ground_truth: 标准答案
        problem_statement: 问题描述
        trajectory: 执行轨迹列表
        error: 错误信息（如果有）
    """
    problem_id: str
    score: float
    agent_answer: str
    ground_truth: str
    problem_statement: str = ""
    trajectory: list = field(default_factory=list)
    error: str | None = None


def _cleanup_docker_containers(docker_image: str):
    """
    清理指定镜像的所有Docker容器。

    在进程被终止时调用，确保没有遗留的容器。
    """
    try:
        import subprocess
        # 首先停止，然后删除（强制删除可能对运行中的容器无效）
        subprocess.run(
            f"docker ps -aq --filter 'ancestor={docker_image}' | xargs -r docker stop 2>/dev/null",
            shell=True,
        )
        subprocess.run(
            f"docker ps -aq --filter 'ancestor={docker_image}' | xargs -r docker rm -f 2>/dev/null",
            shell=True,
        )
        print(f"已清理 {docker_image} 的Docker容器")
    except Exception as e:
        print(f"清理容器时出错：{e}")


def load_reward_function(task_name: str) -> Callable:
    """
    从 benchmark/{task_name}/reward.py 加载评分函数。

    Args:
        task_name: 任务名称（如 "math", "physics"）

    Returns:
        compute_score 函数，用于计算答案的评分
    """
    benchmark_dir = Path(__file__).parent
    reward_path = benchmark_dir / task_name / "reward.py"

    if not reward_path.exists():
        raise FileNotFoundError(f"未找到评分函数：{reward_path}")

    spec = importlib.util.spec_from_file_location("reward", reward_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "compute_score"):
        raise AttributeError("reward.py 必须定义 compute_score 函数")

    return module.compute_score


def load_prompt_function(task_name: str) -> Callable:
    """
    从 benchmark/{task_name}/vanilla_llm_prompt.py 加载提示词创建函数。

    Args:
        task_name: 任务名称

    Returns:
        create_prompt 函数，用于创建LLM提示词
    """
    benchmark_dir = Path(__file__).parent
    prompt_path = benchmark_dir / task_name / "vanilla_llm_prompt.py"

    if not prompt_path.exists():
        # 默认：直接返回 problem_statement
        return lambda problem_data: problem_data['problem_statement']

    spec = importlib.util.spec_from_file_location("vanilla_llm_prompt", prompt_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "create_prompt"):
        # 默认：直接返回 problem_statement
        return lambda problem_data: problem_data['problem_statement']

    return module.create_prompt


def load_task_config(task_name: str) -> dict:
    """
    从 benchmark/{task_name}/config.yaml 加载任务配置。

    Args:
        task_name: 任务名称

    Returns:
        任务配置字典
    """
    benchmark_dir = Path(__file__).parent
    config_path = benchmark_dir / task_name / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"未找到任务配置：{config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset_from_config(task_config: dict):
    """
    从HuggingFace或本地JSON文件加载数据集。

    Args:
        task_config: 任务配置字典，包含dataset和可选的split、data_files等字段

    Returns:
        HuggingFace数据集对象
    """
    from datasets import load_dataset

    dataset_name = task_config["dataset"]
    split = task_config.get("split", "test")

    if dataset_name == "json":
        # 从本地JSON文件加载，相对路径相对于benchmark目录
        data_files = task_config.get("data_files")
        if data_files and not Path(data_files).is_absolute():
            data_files = str(Path(__file__).parent / data_files)
        ds = load_dataset("json", data_files=data_files, split="train")
    else:
        # 从HuggingFace加载
        config = task_config.get("config")
        ds = load_dataset(dataset_name, config, split=split)

    return ds


def create_agent_runner(
    docker_image: str,
    llm_name: str,
    llm_base_url: str,
    max_steps: int,
    temperature: float,
    max_token_limit: int,
    max_tokens_per_call: int,
    extra_body: dict = None,
    task_system_prompt: str = None,
    task_instance_prompt: str = None,
    save_litellm_response: bool = False,
    working_dir: str = None,
    input_dir: str = None,
    output_dir: str = None,
    **kwargs,  # 忽略额外参数（如仅用于vanilla LLM的max_response_len）
) -> Callable:
    """
    工厂函数，创建智能体运行器。

    返回的函数签名：
        (query, input_files, local_output_dir) -> (answer, trajectory, console_output)

    Args:
        docker_image: Docker镜像名称
        llm_name: LLM模型名称
        llm_base_url: LLM API基础URL
        max_steps: 最大步数
        temperature: 采样温度
        max_token_limit: 最大token限制
        max_tokens_per_call: 每次调用最大token数
        extra_body: 额外的请求体参数
        task_system_prompt: 任务特定的系统提示词
        task_instance_prompt: 任务特定的实例提示词
        save_litellm_response: 是否保存litellm响应
        working_dir: 容器内工作目录
        input_dir: 容器内输入目录
        output_dir: 容器内输出目录
    """
    # 在此处导入以避免循环导入
    import logging
    import shutil
    import tempfile

    from grpo_in_sandbox.agent import Agent, AgentArgs
    from grpo_in_sandbox.cli import get_default_config_path, load_prompt_config
    from grpo_in_sandbox.docker_runtime import DockerRuntime

    # 从output_dir构建answer_path
    answer_path = f"{output_dir}/answer.txt"

    # 静默日志记录器 - 错误将被捕获为异常
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.WARNING)

    def agent_runner(query: str, input_files: dict, local_output_dir: str) -> str:
        """
        在单个问题上运行智能体。

        Args:
            query: 问题描述
            input_files: 输入文件字典 {filename: content}
            local_output_dir: 本地输出目录

        Returns:
            (answer, trajectory, console_output) 元组
        """
        runtime = DockerRuntime(
            docker_image=docker_image,
            repo_path=working_dir,
            logger=logger,
        )

        # 注册atexit清理，以防进程被杀死
        atexit.register(runtime.close)

        # 使用临时目录复制输入文件
        temp_dir = None
        if input_files:
            temp_dir = tempfile.mkdtemp()
            for filename, content in input_files.items():
                if content is None:
                    continue
                temp_path = Path(temp_dir) / filename
                temp_path.write_text(content)
            runtime.copy_dir_to_container(temp_dir, input_dir)

        try:
            # 如果提供了任务特定的提示词则使用，否则使用general.yaml默认值
            if task_system_prompt:
                system_prompt = task_system_prompt
                instance_prompt = task_instance_prompt or ""
            else:
                config_path = get_default_config_path()
                config = load_prompt_config(config_path)
                system_prompt = config.get("system_prompt", "")
                instance_prompt = config.get("instance_prompt", "")

            agent_args = AgentArgs(
                system_prompt=system_prompt,
                instance_prompt=instance_prompt,
                llm_name=llm_name,
                llm_base_url=llm_base_url,
                output_dir=local_output_dir,
                extra_body=extra_body,
                quiet=True,  # 在基准测试模式下捕获控制台输出
                save_litellm_response=save_litellm_response,
            )
            agent = Agent(args=agent_args, logger=logger)

            trajectory = agent.run(
                runtime=runtime,
                problem_statement=query,
                max_steps=max_steps,
                temperature=temperature,
                max_token_limit=max_token_limit,
                max_tokens_per_call=max_tokens_per_call,
            )

            # 获取捕获的控制台输出
            console_output = agent.get_console_output()

            # 直接从容器读取答案
            answer = ""
            try:
                output, _ = runtime.run(f"cat {answer_path} 2>/dev/null || echo ''")
                answer = output.strip()
            except Exception:
                pass

            # 返回答案、轨迹和控制台输出
            return answer, trajectory, console_output

        finally:
            # 注销atexit，因为我们正在正常清理
            atexit.unregister(runtime.close)
            runtime.close()
            # 清理临时目录
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    return agent_runner


def run_vanilla_llm(query: str, agent_config: dict) -> tuple:
    """
    不使用沙箱直接运行LLM（vanilla模式）。

    Args:
        query: 查询/问题描述
        agent_config: 智能体配置字典

    Returns:
        (answer, console_output) 元组
    """
    import re
    import time

    import litellm

    llm_name = agent_config["llm_name"]
    llm_base_url = agent_config.get("llm_base_url")
    temperature = agent_config.get("temperature", 1.0)
    # vanilla LLM使用max_response_len
    max_tokens = agent_config.get("max_response_len", 65536)
    extra_body = agent_config.get("extra_body")

    # Token限制处理配置
    max_retries = 5
    min_completion_tokens = 8192
    current_max_tokens = max_tokens

    for attempt in range(max_retries):
        kwargs = {
            "model": llm_name,
            "messages": [{"role": "user", "content": query}],
            "temperature": temperature,
            "max_tokens": current_max_tokens,
            "timeout": 1800,  # 30分钟HTTP超时（包含队列+生成时间）
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        if llm_base_url:
            kwargs["base_url"] = llm_base_url

        try:
            response = litellm.completion(**kwargs)
            answer = response.choices[0].message.content or ""
            console_output = f"Query:\n{query}\n\nResponse:\n{answer}\n"
            return answer, console_output

        except Exception as e:
            error_msg = str(e)
            print(f"❌ [DEBUG] 尝试：{attempt+1}/{max_retries} LiteLLM调用失败：")
            print(f"   - 错误类型：{type(e).__name__}")
            print(f"   - 错误消息：{error_msg}")

            # 检查是否是token限制错误
            if "token" in error_msg.lower() and ("exceed" in error_msg.lower() or "limit" in error_msg.lower() or "maximum" in error_msg.lower()):
                # 解析错误消息
                context_match = re.search(r'(?:maximum|context)[^\d]*(\d+)\s*tokens', error_msg, re.IGNORECASE)
                input_match = re.search(r'(\d+)\s*tokens?\s*(?:from\s*(?:the\s*)?(?:input|messages?)|in\s*(?:the\s*)?(?:input|prompt|messages?))', error_msg, re.IGNORECASE)

                max_context = int(context_match.group(1)) if context_match else None
                current_input_tokens = int(input_match.group(1)) if input_match else None

                print(f"📊 解析错误：max_context={max_context}, input={current_input_tokens}, current_max_tokens={current_max_tokens}")

                if max_context and current_input_tokens:
                    # 策略1：减少max_tokens（completion tokens）
                    available_for_completion = max_context - current_input_tokens - 100  # 100缓冲

                    if available_for_completion >= min_completion_tokens:
                        new_max_tokens = min(available_for_completion, current_max_tokens)
                        if new_max_tokens < current_max_tokens:
                            print(f"📉 Token限制超出，减少max_tokens：{current_max_tokens} -> {new_max_tokens}（输入{current_input_tokens}，上下文{max_context}）")
                            current_max_tokens = new_max_tokens
                            continue

                    # 没有足够空间
                    print(f"⚠️ 输入tokens（{current_input_tokens}）太大，无法放入上下文（{max_context}）且保留最小completion空间")
                    raise
                else:
                    # 无法完整解析，尝试减少max_tokens
                    if current_max_tokens > min_completion_tokens:
                        new_max_tokens = max(int(current_max_tokens * 0.5), min_completion_tokens)
                        print(f"📉 Token限制超出（不完整解析），减少max_tokens：{current_max_tokens} -> {new_max_tokens}")
                        current_max_tokens = new_max_tokens
                        continue
                    else:
                        print(f"⚠️ max_tokens已达到最小值（{min_completion_tokens}），无法进一步减少")
                        raise
            else:
                # 非token限制错误
                if "RateLimitError" in str(e):
                    print("遇到速率限制，休眠60秒...")
                    time.sleep(60)
                    continue
                raise

    # 重试次数用尽
    raise RuntimeError(f"{max_retries}次重试后失败")


def run_single_problem(args: dict) -> BenchmarkResult:
    """
    在单个问题上运行智能体并计算评分。

    此函数设计为与ProcessPoolExecutor一起使用。

    Args:
        args: 包含以下键的字典：
            - problem: 问题数据字典
            - agent_config: 智能体配置
            - task_name: 任务名称
            - prompt_config: 提示词配置
            - output_dir: 输出目录
            - logs_dir: 日志目录
            - mode: 运行模式

    Returns:
        BenchmarkResult对象
    """
    problem = args["problem"]
    agent_config = args["agent_config"]
    task_name = args["task_name"]
    prompt_config = args["prompt_config"]
    output_dir = args["output_dir"]
    logs_dir = args["logs_dir"]
    mode = args.get("mode", "llm-in-sandbox")

    problem_id = problem["id"]
    ground_truth = problem["ground_truth"]
    problem_statement = problem["problem_statement"]

    # 为每个问题创建输出目录用于litellm日志（仅在启用时）
    save_litellm_response = agent_config.get("save_litellm_response", False)
    problem_output_dir = os.path.join(output_dir, "litellm_logs", problem_id) if save_litellm_response else None
    if problem_output_dir:
        os.makedirs(problem_output_dir, exist_ok=True)

    compute_score_func = load_reward_function(task_name)

    try:
        if mode == "llm":
            # Vanilla LLM模式 - 直接API调用，不使用沙箱
            # 使用任务特定的提示词函数
            create_prompt_func = load_prompt_function(task_name)
            query = create_prompt_func(problem)
            agent_answer, console_output = run_vanilla_llm(query, agent_config)
            trajectory = []
        else:
            # LLM-in-Sandbox模式 - 使用任务特定的系统提示词
            # 从配置获取容器路径
            working_dir = prompt_config.get("working_dir", "/testbed")
            input_dir = prompt_config.get("input_dir", "/testbed/documents")
            output_dir_config = prompt_config.get("output_dir", "/testbed")

            # 替换提示词中的目录占位符
            system_prompt = prompt_config["system_prompt"]
            instance_prompt = prompt_config["instance_prompt"]
            system_prompt = system_prompt.replace("{working_dir}", working_dir).replace("{input_dir}", input_dir).replace("{output_dir}", output_dir_config)
            instance_prompt = instance_prompt.replace("{working_dir}", working_dir).replace("{input_dir}", input_dir).replace("{output_dir}", output_dir_config)

            run_agent_func = create_agent_runner(
                **agent_config,
                task_system_prompt=system_prompt,
                task_instance_prompt=instance_prompt,
                working_dir=working_dir,
                input_dir=input_dir,
                output_dir=output_dir_config,
            )
            input_files_raw = problem.get("input_files") or {}
            # 如需要解析JSON字符串（HuggingFace存储为字符串）
            if isinstance(input_files_raw, str):
                import json
                input_files = json.loads(input_files_raw) if input_files_raw else {}
            else:
                input_files = input_files_raw
            # 只传递problem_statement作为查询（系统提示词现在在agent配置中）
            agent_answer, trajectory, console_output = run_agent_func(
                query=problem_statement,
                input_files=input_files,
                local_output_dir=problem_output_dir,
            )

        # 转换轨迹为字典列表（如需要）
        traj_list = []
        if hasattr(trajectory, 'steps'):
            traj_list = [s.to_dict() if hasattr(s, 'to_dict') else s for s in trajectory.steps]
        elif isinstance(trajectory, list):
            traj_list = trajectory

        # 计算评分（传递问题字段作为kwargs用于任务特定评分）
        # 从kwargs中移除'ground_truth'以避免重复参数错误
        problem_kwargs = {k: v for k, v in problem.items() if k != 'ground_truth'}
        score = compute_score_func(agent_answer, ground_truth, **problem_kwargs)

        # 保存日志为.txt（捕获的控制台输出 + 结果摘要）
        log_text = console_output
        log_text += f"\n{'=' * 80}\n"
        log_text += "### 结果 ###\n"
        log_text += f"问题ID: {problem_id}\n"
        log_text += f"智能体答案: {agent_answer}\n"
        log_text += f"标准答案: {ground_truth}\n"
        log_text += f"评分: {score:.4f}\n"
        log_text += f"{'=' * 80}\n"

        log_path = os.path.join(logs_dir, f"{problem_id}.txt")
        with open(log_path, "w") as f:
            f.write(log_text)

        return BenchmarkResult(
            problem_id=problem_id,
            score=score,
            agent_answer=agent_answer,
            ground_truth=ground_truth,
            problem_statement=problem_statement,
            trajectory=traj_list,
        )

    except (ImportError, ModuleNotFoundError):
        # 配置错误应停止整个基准测试
        raise
    except Exception as e:
        import traceback
        error_str = str(e)
        # 认证错误应停止整个基准测试
        if "AuthenticationError" in error_str or "api_key" in error_str.lower():
            raise

        # Docker/containerd错误应停止整个基准测试
        if "containerd.sock" in error_str or "connection refused" in error_str.lower():
            raise RuntimeError(
                f"Docker守护进程未运行或崩溃。原始错误：{error_str}\n"
                f"请通过运行以下命令重启Docker：\n"
                f"  pkill -9 dockerd 2>/dev/null; pkill -9 containerd 2>/dev/null\n"
                f"  sleep 1 && containerd &\n"
                f"  sleep 3 && rm -f /var/run/docker.pid && dockerd &\n"
            ) from e

        # 打印错误到控制台，包含完整回溯
        print(f"[{problem_id}] 错误：{e}")
        traceback.print_exc()

        # 保存错误日志为.txt
        log_text = f"问题ID: {problem_id}\n"
        log_text += f"错误：{error_str}\n"
        log_text += f"标准答案: {ground_truth}\n"

        log_path = os.path.join(logs_dir, f"{problem_id}.txt")
        with open(log_path, "w") as f:
            f.write(log_text)

        return BenchmarkResult(
            problem_id=problem_id,
            score=0.0,
            agent_answer="",
            ground_truth=ground_truth,
            problem_statement=problem_statement,
            trajectory=[],
            error=error_str,
        )


def run_benchmark(
    task_name: str,
    agent_config: dict,
    output_dir: str,
    num_workers: int = 1,
    start_id: int = None,
    end_id: int = None,
    mode: str = "llm-in-sandbox",
) -> dict:
    """
    使用并行执行在任务上运行基准测试。

    Args:
        task_name: 基准测试任务名称（如 math, physics, biomed 等）
        agent_config: 包含智能体配置的字典（docker_image, llm_name 等）
        output_dir: 保存输出的目录（已命名为 {timestamp}_{task}）
        num_workers: 并行工作进程数（ProcessPoolExecutor）
        start_id: 起始索引（0-based，包含）
        end_id: 结束索引（0-based，不包含）
        mode: "llm-in-sandbox"（默认）或 "llm"（vanilla LLM）

    Returns:
        包含结果和统计信息的字典

    输出结构：
        {output_dir}/
            logs/           # 每个问题的日志（人类可读的.txt）
            trajectory.json # 所有轨迹
            results.json    # 最终结果
    """
    from rich.console import Console

    console = Console()
    docker_image = agent_config.get("docker_image")

    # 加载任务配置
    task_config = load_task_config(task_name)
    compute_score = load_reward_function(task_name)

    # 提前测试评分函数以捕获缺失的依赖
    try:
        compute_score("test", "test")
    except ImportError as e:
        console.print(f"[red]缺失依赖：{e}[/red]")
        raise
    except Exception:
        pass  # 其他错误是可以的，我们只想检查导入

    # 加载提示词配置（支持合并和分离两种格式）
    if "system_prompt" in task_config:
        # 新格式：提示词配置合并到config.yaml
        prompt_config = {
            "system_prompt": task_config["system_prompt"],
            "instance_prompt": task_config.get("instance_prompt", ""),
        }
    elif "prompt_config" in task_config:
        # 旧格式：分离的prompt_config.yaml文件
        prompt_config_path = task_config["prompt_config"]
        with open(prompt_config_path) as f:
            prompt_config = yaml.safe_load(f)
    else:
        raise ValueError("任务配置必须包含'system_prompt'或'prompt_config'")

    # 加载数据集
    dataset = load_dataset_from_config(task_config)
    problems = list(dataset)

    # 按start_id和end_id过滤
    if start_id is not None or end_id is not None:
        start_id = start_id or 0
        end_id = end_id or len(problems)
        problems = problems[start_id:end_id]

    total_problems = len(problems)

    mode_display = "[green]LLM-in-Sandbox[/green]" if mode == "llm-in-sandbox" else "[yellow]Vanilla LLM[/yellow]"
    console.print(f"[bold cyan]🚀 基准测试：{task_name}[/bold cyan]（{mode_display}）")
    if start_id is not None or end_id is not None:
        console.print(f"   范围：[{start_id}, {end_id})")
    console.print(f"   问题数：{total_problems}，工作进程：{num_workers}")
    console.print(f"   输出：{output_dir}")
    console.print(f"   模型：{agent_config.get('llm_name', 'N/A')}")
    console.print(f"   温度：{agent_config.get('temperature', 'N/A')}")
    if mode == "llm-in-sandbox":
        console.print(f"   最大步数：{agent_config.get('max_steps', 'N/A')}")
        console.print(f"   最大Token限制：{agent_config.get('max_token_limit', 'N/A')}")
        console.print(f"   每次调用最大Token：{agent_config.get('max_tokens_per_call', 'N/A')}")
    elif mode == "llm":
        console.print(f"   最大响应长度：{agent_config.get('max_response_len', 'N/A')}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    results = []
    completed_count = 0
    running_sum = 0.0
    error_count = 0

    from tqdm import tqdm

    console.print(f"[cyan]使用 {num_workers} 个工作进程开始 {total_problems} 个问题...[/cyan]")

    import time
    start_time = time.time()

    pbar = tqdm(total=total_problems, desc="运行中", dynamic_ncols=True)

    def update_pbar(result):
        """更新进度条和统计信息。"""
        nonlocal completed_count, running_sum, error_count
        completed_count += 1
        running_sum += result.score
        if result.error:
            error_count += 1
        mean_score = running_sum / completed_count
        pbar.set_postfix({"评分": f"{mean_score:.3f}", "错误": error_count})
        pbar.update(1)

    if num_workers == 1:
        # 顺序执行
        for problem in problems:
            args = {
                "problem": problem,
                "agent_config": agent_config,
                "task_name": task_name,
                "prompt_config": prompt_config,
                "output_dir": output_dir,
                "logs_dir": logs_dir,
                "mode": mode,
            }
            result = run_single_problem(args)
            results.append(result)
            update_pbar(result)
    else:
        # 使用ProcessPoolExecutor并行执行
        worker_args_list = [
            {
                "problem": problem,
                "agent_config": agent_config,
                "task_name": task_name,
                "prompt_config": prompt_config,
                "output_dir": output_dir,
                "logs_dir": logs_dir,
                "mode": mode,
            }
            for problem in problems
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 在SIGINT/SIGTERM时注册清理
            def cleanup_handler(signum, frame):
                pbar.close()
                console.print("\n[yellow]中断！正在清理...[/yellow]")
                executor.shutdown(wait=False, cancel_futures=True)
                if docker_image:
                    _cleanup_docker_containers(docker_image)
                sys.exit(1)

            old_sigint = signal.signal(signal.SIGINT, cleanup_handler)
            old_sigterm = signal.signal(signal.SIGTERM, cleanup_handler)

            try:
                futures = {
                    executor.submit(run_single_problem, args): args["problem"]["id"]
                    for args in worker_args_list
                }

                # 每个问题超时：最多30分钟
                problem_timeout = 1800
                for future in as_completed(futures, timeout=problem_timeout * len(futures)):
                    try:
                        result = future.result(timeout=problem_timeout)
                    except TimeoutError:
                        problem_id = futures[future]
                        console.print(f"[red]超时：{problem_id}[/red]")
                        result = BenchmarkResult(
                            problem_id=problem_id,
                            score=0.0,
                            agent_answer="",
                            ground_truth="",
                            problem_statement="",
                            trajectory=[],
                            error="Timeout",
                        )
                    results.append(result)
                    update_pbar(result)
            finally:
                # 恢复原始信号处理器
                signal.signal(signal.SIGINT, old_sigint)
                signal.signal(signal.SIGTERM, old_sigterm)

    pbar.close()

    # 计算最终统计信息
    elapsed_time = time.time() - start_time
    scores = [r.score for r in results]
    errors = [r for r in results if r.error]

    stats = {
        "task": task_name,
        "total": len(results),
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "num_errors": len(errors),
        "elapsed_time_seconds": elapsed_time,
    }

    # 保存trajectory.json（每个实例包含轨迹+奖励+答案+标准答案+问题描述）
    trajectory_data = {
        r.problem_id: {
            "problem_statement": r.problem_statement,
            "trajectory": r.trajectory,
            "reward": r.score,
            "agent_answer": r.agent_answer,
            "ground_truth": r.ground_truth,
        } for r in results
    }
    trajectory_path = os.path.join(output_dir, "trajectory.json")
    with open(trajectory_path, "w") as f:
        json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

    # 保存results.json（配置 + 统计摘要）
    results_data = {
        "config": {
            "mode": mode,
            **agent_config,
        },
        **stats,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    # 打印最终摘要
    elapsed_str = f"{int(elapsed_time // 3600)}小时 {int((elapsed_time % 3600) // 60)}分钟 {int(elapsed_time % 60)}秒"

    console.print()
    console.print(f"[bold]📊 结果：{task_name}[/bold]")
    console.print(f"   平均评分：[green]{stats['mean_score']:.4f}[/green]")
    console.print(f"   错误数：[red]{stats['num_errors']}[/red]" if stats['num_errors'] > 0 else "   错误数：0")
    console.print(f"   总时间：{elapsed_str}")
    console.print(f"   输出：{output_dir}")

    return results_data
