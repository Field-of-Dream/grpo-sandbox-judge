"""
Agent Module - LLM-powered autonomous agent for code execution in sandboxed environments.

Usage:
    >>> from agent import Agent, AgentArgs
    >>> args = AgentArgs(llm_name="openai/gpt-4", system_prompt="You are a coder.", instance_prompt="Fix file.py")
    >>> agent = Agent(args)
    >>> trajectory = agent.run(runtime, "Create hello.py")
"""

import copy
import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import litellm
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.panel import Panel

from .action import Action
from .tools import execute_bash_tool, str_replace_editor_tool, submit_tool
from .trajectory import Trajectory, TrajectoryStep

litellm.drop_params = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger with RichHandler for colorful output."""
    logger = logging.getLogger(name)

    # Remove existing handlers
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    # Remove root logger handlers
    root_logger = logging.getLogger()
    while root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])

    logger.setLevel(logging.INFO)

    # Create RichHandler
    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        show_time=True,
        show_level=True,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_handler)

    return logger


logger = get_logger(__name__)


# Configuration dataclass
@dataclass
class AgentArgs:
    """Configuration for Agent initialization.

    Args:
        system_prompt: System prompt template for the LLM.
            Defines the agent's role and behavior.
            Example: "You are a Python expert who writes clean code."
        instance_prompt: Task prompt for this session.
            The specific task for the agent to complete.
            Example: "Fix the bug in /testbed/app.py"
        llm_name: Model identifier for litellm.
            Format: "provider/model_name"
            Examples:
                - "openai/gpt-4"
                - "anthropic/claude-3-opus-20240229"
                - "azure/gpt-35-turbo"
                - "ollama/llama2"
        llm_base_url: Optional API endpoint for local models.
            Example: "http://localhost:8000/v1"
        max_retries: Maximum retry attempts for API calls (default: 5).
        save_litellm_response: Save request/response to file (default: False).
            Useful for debugging.
        output_dir: Directory for saved responses (default: None).
            Required if save_litellm_response is True.
        extra_body: Additional parameters for LLM API.
            Example: {"temperature": 0.7}
        quiet: Suppress console output (default: False).
            Set True for benchmark mode.
    """
    system_prompt: str  # System prompt for LLM
    instance_prompt: str  # Task prompt for this session
    llm_name: str  # Model name for litellm
    llm_base_url: str | None = None  # Custom API base URL
    max_retries: int = 5  # Max API retry attempts
    save_litellm_response: bool = False  # Save LLM responses
    output_dir: str | None = None  # Output directory
    extra_body: dict[str, Any] | None = None  # Extra API params
    quiet: bool = False  # Suppress output for benchmarks


class Agent:
    """Autonomous agent for code execution using LLM tool calling.

    Orchestrates LLM queries, tool execution, and trajectory recording
    across Docker/Kaggle/Local sandbox runtimes.
    """

    def __init__(self, args: AgentArgs, logger=None):
        self.args = args
        self.llm_name = args.llm_name
        self.quiet = args.quiet

        if self.quiet:
            import io
            self._console_buffer = io.StringIO()
            self.console = Console(file=self._console_buffer, record=True, force_terminal=True, width=100)
        else:
            self._console_buffer = None  # type: ignore[assignment]
            self.console = Console()

        self.logger = logger or get_logger("Agent")
        if self.quiet:
            self.logger.setLevel(logging.WARNING)

        self.llm_base_url = args.llm_base_url
        if self.llm_base_url is None and ("openai/" in self.llm_name or "hosted_vllm" in self.llm_name):
            self.llm_base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")

        self.system_prompt_template = args.system_prompt
        self.instance_prompt_template = args.instance_prompt
        self.max_retries = args.max_retries
        self.extra_body = args.extra_body

        self.logger.info(f"Initialized Agent with LLM: {self.llm_name}")
        self.logger.info(f"🔗 LLM Base URL: {self.llm_base_url}")
        self.logger.info(f"📦 Extra body: {self.extra_body}")

        self.save_litellm_response = args.save_litellm_response
        self.output_dir = args.output_dir
        self.llm_call_count = 0
        if self.save_litellm_response:
            self.logger.info(f"📝 Save LiteLLM response enabled, output_dir: {self.output_dir}")

        self.trajectory_steps: list[TrajectoryStep] = []
        self.history: list[dict[str, str]] = []

    def get_console_output(self) -> str:
        """Get captured console output (only available in quiet mode)."""
        if self.quiet and hasattr(self.console, 'export_text'):
            return self.console.export_text()
        return ""

    def reset(self):
        """Reset the agent's trajectory."""
        self.trajectory_steps = []
        self.history = []

    def _count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count tokens in messages."""
        try:
            return litellm.token_counter(model=self.llm_name, messages=messages)
        except Exception:
            # Rough estimate: 4 chars per token
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            return total_chars // 4

    def model_query(
        self,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens_per_call: int = 65536,
        max_token_limit: int = 65536,
    ) -> tuple[Any, float]:
        """向 LLM 发送查询请求，支持自动重试和 token 限制处理。"""
        tools = [str_replace_editor_tool, execute_bash_tool, submit_tool]

        retries = 0
        messages_ = copy.deepcopy(messages)

        total_tokens = self._count_tokens(messages_)
        self.logger.info(f"Total tokens in conversation: {total_tokens}")

        if total_tokens > max_token_limit:
            self.logger.warning(f"Total tokens: {total_tokens} > {max_token_limit}")
            raise ValueError(f"Total tokens: {total_tokens} > {max_token_limit}")

        start_time = time.time()
        response = None

        while retries < self.max_retries:
            try:
                kwargs = {}
                if "o3" not in self.llm_name and "o4" not in self.llm_name:
                    kwargs["temperature"] = temperature

                extra_params = {}
                if self.extra_body:
                    extra_params["extra_body"] = self.extra_body

                response = litellm.completion(
                    model=self.llm_name,
                    tools=tools,
                    messages=messages_,
                    timeout=1200,
                    api_base=self.llm_base_url,
                    max_tokens=max_tokens_per_call,
                    **extra_params,
                    **kwargs,
                )
                self.logger.info("LLM query complete")

                if self.save_litellm_response and self.output_dir:
                    self._save_litellm_response(messages_, response, extra_params, kwargs)

                break

            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"LLM query failed @ {retries}: {e}")

                # Check if it's a token limit error
                if "token" in error_msg.lower() and ("exceed" in error_msg.lower() or "limit" in error_msg.lower() or "maximum" in error_msg.lower()):
                    self.logger.warning("⚠️ Token limit error detected, attempting to handle...")

                    # Parse error message:
                    # "maximum context length of 163840 tokens. You requested a total of 179069 tokens: 113533 tokens from the input messages and 65536 tokens for the completion"
                    context_match = re.search(r'(?:maximum|context)[^\d]*(\d+)\s*tokens', error_msg, re.IGNORECASE)
                    input_match = re.search(r'(\d+)\s*tokens?\s*(?:from\s*(?:the\s*)?(?:input|messages?)|in\s*(?:the\s*)?(?:input|prompt|messages?))', error_msg, re.IGNORECASE)

                    max_context = int(context_match.group(1)) if context_match else None
                    current_input_tokens = int(input_match.group(1)) if input_match else None

                    # Minimum completion tokens to preserve
                    min_completion_tokens = 8192

                    self.logger.warning(f"📊 Parsed error: max_context={max_context}, input={current_input_tokens}, max_tokens_per_call={max_tokens_per_call}")

                    if max_context and current_input_tokens:
                        # Strategy 1: Reduce max_tokens (completion tokens)
                        available_for_completion = max_context - current_input_tokens - 100  # 100 buffer

                        if available_for_completion >= min_completion_tokens:
                            new_max_tokens = min(available_for_completion, max_tokens_per_call)
                            if new_max_tokens < max_tokens_per_call:
                                self.logger.warning(f"📉 Token limit exceeded, reducing max_tokens: {max_tokens_per_call} -> {new_max_tokens} (input {current_input_tokens}, context {max_context})")
                                max_tokens_per_call = new_max_tokens
                                continue

                        # Not enough space even with min_completion_tokens
                        self.logger.warning(f"⚠️ Input tokens ({current_input_tokens}) too large, cannot fit in context ({max_context}) with min completion space")
                        raise
                    else:
                        # Cannot parse complete info, try reducing max_tokens
                        if max_tokens_per_call > min_completion_tokens:
                            new_max_tokens = max(int(max_tokens_per_call * 0.5), min_completion_tokens)
                            self.logger.warning(f"📉 Token limit exceeded (incomplete parse), reducing max_tokens: {max_tokens_per_call} -> {new_max_tokens}")
                            max_tokens_per_call = new_max_tokens
                            continue
                        else:
                            self.logger.warning(f"⚠️ max_tokens already at minimum ({min_completion_tokens}), cannot reduce further")
                            raise

                retries += 1

                if "RateLimitError" in str(e):
                    time.sleep(60)

                if retries >= self.max_retries:
                    raise e

        exec_time = time.time() - start_time
        return response, exec_time

    def _save_litellm_response(self, messages: list[dict], response, extra_params: dict, kwargs: dict):
        """Save litellm request and response to output_dir for debugging."""
        try:
            self.llm_call_count += 1
            save_path = self.output_dir
            if save_path is None:
                return
            os.makedirs(save_path, exist_ok=True)

            # Save request
            request_data = {
                "call_id": self.llm_call_count,
                "model": self.llm_name,
                "api_base": self.llm_base_url,
                "messages": messages,
                "extra_params": extra_params,
                "kwargs": kwargs,
            }
            request_file = os.path.join(save_path, f"request_{self.llm_call_count:03d}.json")
            with open(request_file, "w", encoding="utf-8") as f:
                json.dump(request_data, f, indent=2, ensure_ascii=False)

            # Save response (raw dict from litellm)
            response_data = {
                "call_id": self.llm_call_count,
                "response": response.model_dump() if hasattr(response, "model_dump") else response.to_dict() if hasattr(response, "to_dict") else str(response),
            }
            response_file = os.path.join(save_path, f"response_{self.llm_call_count:03d}.json")
            with open(response_file, "w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Saved LiteLLM request/response #{self.llm_call_count} to {save_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save litellm response: {e}")

    def parse_response(self, response) -> tuple[str, Action, str | None]:
        """从 LLM 响应中提取 thought、action 和 tool_call_id。"""
        thought = response.choices[0].message.content or ""

        tool_call_id = None
        try:
            tool_call = response.choices[0].message.tool_calls[0]
            action = Action(
                function_name=tool_call.function.name,
                parameters=json.loads(tool_call.function.arguments),
            )
            tool_call_id = tool_call.id
        except Exception:
            action = Action(function_name="", parameters={})

        return thought, action, tool_call_id

    def run(
        self,
        runtime,  # DockerRuntime
        problem_statement: str,
        max_steps: int = 30,
        max_token_limit: int = 65536,
        max_tokens_per_call: int = 65536,
        temperature: float = 1.0,
) -> Trajectory:
        """Execute the agent to complete a task via LLM-driven tool calling."""
        self.reset()
        self.logger.info("Starting agent run:")
        self.logger.info(f"max_steps={max_steps}")
        self.logger.info(f"max_token_limit={max_token_limit}")
        self.logger.info(f"max_tokens_per_call={max_tokens_per_call}")
        self.logger.info(f"temperature={temperature}")

        system_prompt = self.system_prompt_template.format(
            command_docs="",
            demo="",
        )

        if self.instance_prompt_template:
            user_prompt = self.instance_prompt_template.format(
                problem_statement=problem_statement,
            )
        else:
            user_prompt = problem_statement

        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Print prompts
        for title, content in [("SYSTEM PROMPT", system_prompt), ("USER PROMPT", user_prompt)]:
            self.console.print(Panel(
                escape(content[:2000] + "..." if len(content) > 2000 else content),
                title=f"[bold cyan]{title}[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            ))

        done = False
        step_count = 0

        while not done and step_count < max_steps:
            self.console.print()
            self.console.rule(f"[bold blue]Step {step_count + 1}/{max_steps}[/bold blue]", style="blue")

            # Add step count message to history
            steps_remaining = max_steps - step_count
            if steps_remaining > 0:
                step_msg = f"Steps Remaining: {steps_remaining}"
            else:
                step_msg = "You have reached the maximum number of steps. Please submit your answer NOW."
            self.history[-1]["content"] += f"\n{step_msg}"
            self.logger.info(step_msg)

            messages = self.history.copy()

            # Query LLM
            try:
                response, exec_time = self.model_query(
                    messages=messages,
                    temperature=temperature,
                    max_token_limit=max_token_limit,
                    max_tokens_per_call=max_tokens_per_call,
                )
            except Exception as e:
                self.logger.error(f"LLM query failed: {e}")
                break

            # Extract reasoning_content from response
            reasoning_content = ""
            try:
                message = response.choices[0].message
                if hasattr(message, 'provider_specific_fields') and message.provider_specific_fields:
                    reasoning_content = message.provider_specific_fields.get('thinking') or message.provider_specific_fields.get('reasoning_content') or ""
            except Exception as e:
                self.logger.warning(f"fail to extract reasoning_content: {e}")

            # Pretty print reasoning_content if present
            if reasoning_content:
                thought_display = reasoning_content[:2000] + "..." if len(reasoning_content) > 2000 else reasoning_content
                self.console.print(Panel(
                    escape(thought_display),
                    title="[bold magenta]🧠 REASONING CONTENT[/bold magenta]",
                    border_style="magenta",
                    padding=(0, 1),
                ))

            # Parse response
            thought, action, tool_call_id = self.parse_response(response)

            # Print thought
            if thought:
                thought_display = thought[:1000] + "..." if len(thought) > 1000 else thought
                self.console.print(Panel(
                    escape(thought_display),
                    title="[bold magenta]💭 THOUGHT[/bold magenta]",
                    border_style="magenta",
                    padding=(0, 1),
                ))

            # Print action
            action_text = f"[bold]{action.function_name}[/bold]"
            if action.parameters:
                params_str = json.dumps(action.parameters, indent=2, ensure_ascii=False)
                if len(params_str) > 300:
                    params_str = params_str[:300] + "..."
                action_text += f"\n{escape(params_str)}"
            self.console.print(Panel(
                action_text,
                title="[bold yellow]⚡ ACTION[/bold yellow]",
                border_style="yellow",
                padding=(0, 1),
            ))

            # Execute and observe
            observation = self._execute_action(action, runtime)

            # Print observation
            obs_display = observation[:800] + "..." if len(observation) > 800 else observation
            self.console.print(Panel(
                escape(obs_display),
                title="[bold green]👁 OBSERVATION[/bold green]",
                border_style="green",
                padding=(0, 1),
            ))

            # Record step
            step_record = TrajectoryStep(
                thought=thought,
                reasoning_content=reasoning_content,
                action=action.to_dict(),
                observation=observation,
                metadata={"step": step_count + 1, "exec_time": exec_time},
            )
            self.trajectory_steps.append(step_record)

            # Update history
            assistant_msg = response.choices[0].message
            if hasattr(assistant_msg, 'model_dump'):
                assistant_msg_dict = assistant_msg.model_dump(exclude_none=True)
            elif hasattr(assistant_msg, 'to_dict'):
                assistant_msg_dict = assistant_msg.to_dict()
            else:
                assistant_msg_dict = dict(assistant_msg)
            self.history.append(assistant_msg_dict)

            # Add tool result or CONTINUE_MSG
            if action.function_name and tool_call_id:
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": observation,
                })
            elif not action.function_name:
                # Model didn't use tool_calls, send CONTINUE_MSG as user message
                self.history.append({
                    "role": "user",
                    "content": observation,  # This is CONTINUE_MSG
                })

            # Check if done
            if action.function_name == "submit":
                done = True
                self.logger.info("Agent submitted answer")

            step_count += 1

        # Create trajectory
        trajectory = Trajectory(
            problem_statement=problem_statement,
            steps=self.trajectory_steps,
            metadata={"total_steps": step_count},
        )

        return trajectory

    def _execute_action(self, action: Action, runtime) -> str:
        """根据 action 类型执行对应操作：execute_bash / str_replace_editor / submit。"""
        from .observation import Observation

        if not action.function_name or not isinstance(action.parameters, dict):
            obs = Observation(bash_output="", error_code=0, action=action)
            return str(obs)

        if action.function_name == "execute_bash":
            command = action.parameters.get("command", "")
            if not command:
                return "Error: No command provided."
            stdout, stderr, exit_code = runtime.demux_run(command)

            obs = Observation(
                bash_output=stdout + stderr,
                error_code=exit_code,
                action=action,
                stdout=stdout,
                stderr=stderr,
            )
            return str(obs)

        elif action.function_name == "str_replace_editor":
            result = self._execute_str_replace_editor(action, runtime)
            obs = Observation(bash_output=result, error_code=0, action=action)
            return str(obs)

        elif action.function_name == "submit":
            obs = Observation(bash_output="", error_code=0, action=action)
            return str(obs)

        else:
            return f"Unknown action: {action.function_name}"

    def _execute_str_replace_editor(self, action: Action, runtime) -> str:
        """执行 str_replace_editor 文件操作，支持 view/create/str_replace/insert 命令。"""
        snippet_lines = 4
        max_response_len = 10000
        truncated_message = (
            "<response clipped><NOTE>To save on context only part of this file has been "
            "shown to you. You should retry this tool after you have searched inside the file "
            "with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
        )

        def maybe_truncate(content: str, truncate_after: int = max_response_len) -> str:
            if not truncate_after or len(content) <= truncate_after:
                return content
            return content[:truncate_after] + truncated_message

        def make_output(file_content: str, file_descriptor: str, init_line: int = 1) -> str:
            """Format file content with line numbers like cat -n."""
            file_content = maybe_truncate(file_content)
            file_content = file_content.expandtabs()
            lines = file_content.split("\n")
            numbered = "\n".join(f"{i + init_line:6}\t{line}" for i, line in enumerate(lines))
            return f"Here's the result of running `cat -n` on {file_descriptor}:\n{numbered}\n"

        params = action.parameters
        command = params.get("command", "")
        path = params.get("path", "")

        if not command or not path:
            return "Error: command and path are required."

        if command == "view":
            check_cmd = f"test -d {path} && echo 'dir' || (test -f {path} && echo 'file' || echo 'notfound')"
            path_type, _ = runtime.run(check_cmd)
            path_type = path_type.strip()

            if path_type == "dir":
                cmd = f"find {path} -maxdepth 2 -not -path '*/.*' \\( -type d -o -name '*.py' -o -name '*.rst' \\) | head -100"
                output, _ = runtime.run(cmd)
                if output:
                    return f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden:\n{output}"
                return f"(empty directory: {path})"
            elif path_type == "file":
                if not (path.endswith('.py') or path.endswith('.rst')):
                    return f"ERROR: Viewing non-Python files is disallowed for saving context. File '{path}' is not a .py or .rst file."

                view_range = params.get("view_range")
                file_content, _ = runtime.run(f"cat {path}")
                if not file_content:
                    return f"(empty file: {path})"

                file_content = file_content.expandtabs()
                lines = file_content.split('\n')
                total_lines = len(lines)

                if view_range and len(view_range) == 2:
                    start, end = view_range
                    if not (1 <= start <= total_lines):
                        return f"Error: Invalid view_range {view_range}: start line must be in [1, {total_lines}]"
                    if end != -1 and (end < start or end > total_lines):
                        return f"Error: Invalid view_range {view_range}: end must be >= start and <= {total_lines}, or -1 to view until end."

                    sliced_lines = lines[start - 1:] if end == -1 else lines[start - 1:end]
                    numbered = "\n".join(f"{i + start:6}\t{line}" for i, line in enumerate(sliced_lines))
                else:
                    numbered = "\n".join(f"{i + 1:6}\t{line}" for i, line in enumerate(lines))

                result = f"Here's the result of running `cat -n` on {path}:\n{numbered}"
                return maybe_truncate(result)
            else:
                return f"Error: Path not found: {path}"

        elif command == "create":
            file_text = str(params.get("file_text", ""))
            escaped_text = file_text.replace("'", "'\"'\"'")
            cmd = f"mkdir -p $(dirname {path}) && echo '{escaped_text}' > {path}"
            output, exit_code = runtime.run(cmd)
            if "Error" in str(exit_code):
                return f"Error creating file: {output}"

            success_msg = f"File created at {path}. "
            success_msg += make_output(file_text, str(path))
            success_msg += "Review the file and make sure that it is as expected. Edit the file if necessary."
            return success_msg

        elif command == "str_replace":
            old_str = params.get("old_str", "")
            new_str = params.get("new_str", "")

            if not old_str:
                return "Error: old_str is required for str_replace."

            file_content, _ = runtime.run(f"cat {path}")
            if not file_content:
                return f"Error: Could not read file {path}"

            file_content = file_content.expandtabs()
            old_str = old_str.expandtabs()
            new_str = new_str.expandtabs() if new_str else ""

            occurrences = file_content.count(old_str)
            if occurrences == 0:
                return f"Error: No occurrences of old_str found in {path} for replacement."
            if occurrences > 1:
                return f"Error: Multiple occurrences ({occurrences}) of old_str found in {path}. Please ensure it is unique before using str_replace."

            replacement_line = file_content.split(old_str)[0].count("\n")
            new_content = file_content.replace(old_str, new_str if new_str else "", 1)

            escaped_content = new_content.replace("'", "'\"'\"'")
            cmd = f"echo '{escaped_content}' > {path}"
            _, exit_code = runtime.run(cmd)

            if "Error" in str(exit_code):
                return f"Error writing file: {path}"

            new_lines = new_content.split("\n")
            start_line = max(0, replacement_line - snippet_lines)
            end_line = replacement_line + snippet_lines + (new_str or "").count("\n")
            snippet = "\n".join(new_lines[start_line:end_line + 1])

            success_msg = f"The file {path} has been edited. "
            success_msg += make_output(snippet, f"a snippet of {path}", start_line + 1)
            success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."
            return success_msg

        elif command == "insert":
            insert_line = params.get("insert_line", 0)
            new_str = params.get("new_str", "")

            if not new_str:
                return "Error: new_str is required for insert."

            file_content, _ = runtime.run(f"cat {path}")
            # Expand tabs (like R2E-Gym)
            file_content = file_content.expandtabs() if file_content else ""
            new_str = new_str.expandtabs()
            lines = file_content.split('\n') if file_content else []

            if insert_line < 0 or insert_line > len(lines):
                return f"Error: Invalid insert_line {insert_line}. Must be in [0, {len(lines)}]."

            new_str_lines = new_str.split("\n")
            new_lines = lines[:insert_line] + new_str_lines + lines[insert_line:]
            new_content = '\n'.join(new_lines)

            escaped_content = new_content.replace("'", "'\"'\"'")
            cmd = f"echo '{escaped_content}' > {path}"
            _, exit_code = runtime.run(cmd)

            if "Error" in str(exit_code):
                return f"Error writing file: {path}"

            snippet_parts = (
                lines[max(0, insert_line - snippet_lines):insert_line]
                + new_str_lines
                + lines[insert_line:insert_line + snippet_lines]
            )
            snippet = "\n".join(snippet_parts)

            success_msg = f"The file {path} has been edited. "
            success_msg += make_output(snippet, "a snippet of the edited file", max(1, insert_line - snippet_lines + 1))
            success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."
            return success_msg

        else:
            return f"Unknown command: {command}"


# =============================================================================
# Agent 工厂和注册表
# =============================================================================

class AgentRegistry:
    """Agent注册表 - 管理多个Agent实例"""

    def __init__(self):
        self._agents: dict[str, Agent] = {}
        self._factories: dict[str, Callable] = {}

    def register(self, name: str, agent: 'Agent'):
        """注册一个Agent实例"""
        self._agents[name] = agent

    def register_factory(self, name: str, factory: Callable):
        """注册一个Agent工厂函数"""
        self._factories[name] = factory

    def get(self, name: str) -> Optional['Agent']:
        """获取已注册的Agent实例"""
        return self._agents.get(name)

    def create(self, name: str, **kwargs) -> Optional['Agent']:
        """通过工厂函数创建Agent"""
        factory = self._factories.get(name)
        if factory:
            return factory(**kwargs)
        return None

    def list_agents(self) -> list[str]:
        """列出所有已注册的Agent"""
        return list(self._agents.keys())

    def list_factories(self) -> list[str]:
        """列出所有已注册的工厂"""
        return list(self._factories.keys())

    def remove(self, name: str):
        """移除Agent实例"""
        if name in self._agents:
            del self._agents[name]

    def clear(self):
        """清空所有Agent"""
        self._agents.clear()


_global_registry = AgentRegistry()


def _create_agent_factory(agent_type: str):
    """创建指定类型Agent的工厂函数"""
    def factory(llm_name: str, llm_base_url: str | None = None,
        system_prompt: str | None = None, instance_prompt: str | None = None,
        quiet: bool = False, **kwargs) -> 'Agent':
        prompts = {
            "coder": ("""你是一个专业的Coder Agent，专注于编写高质量的代码。\n\n<CAPABILITIES>\n- 编写Python、JavaScript、Java、C++、Go、Rust等多种编程语言\n- 代码重构和优化\n- 调试和bug修复\n- 单元测试编写\n- 代码审查\n</CAPABILITIES>\n\n<TOOLS>\n- execute_bash: 执行shell命令和脚本\n- str_replace_editor: 查看、创建和编辑文件\n- submit: 提交完成的任务\n</TOOLS>\n\n<WORKFLOW>\n1. 理解需求并分析问题\n2. 设计代码结构和算法\n3. 编写代码并测试\n4. 调试和优化\n5. 提交完成的任务\n</WORKFLOW>""",
             """<problem>{problem_statement}</problem>\n\n请编写代码解决这个问题，并确保代码正确运行。"""),
            "analyzer": ("""你是一个专业的Analyzer Agent，专注于分析数据和调试问题。\n\n<CAPABILITIES>\n- 日志分析和问题定位\n- 性能分析和优化建议\n- 错误追踪和根因分析\n- 数据分析和可视化\n- 代码审查和静态分析\n</CAPABILITIES>\n\n<TOOLS>\n- execute_bash: 执行shell命令和分析脚本\n- str_replace_editor: 查看日志和代码文件\n- submit: 提交分析结果\n</TOOLS>\n\n<WORKFLOW>\n1. 收集相关日志和信息\n2. 分析问题模式和趋势\n3. 定位根本原因\n4. 提供解决方案建议\n5. 提交分析报告\n</WORKFLOW>""",
             """<problem>{problem_statement}</problem>\n\n请分析这个问题，提供详细的分析报告和解决方案。"""),
            "research": ("""你是一个专业的Research Agent，专注于信息收集和研究。\n\n<CAPABILITIES>\n- 网络搜索和文献检索\n- 信息整理和总结\n- 比较分析和趋势研究\n- 数据收集和验证\n- 报告撰写\n</CAPABILITIES>\n\n<TOOLS>\n- execute_bash: 执行搜索命令和爬虫脚本\n- str_replace_editor: 查看和编辑文件\n- submit: 提交研究报告\n</TOOLS>\n\n<WORKFLOW>\n1. 理解研究目标\n2. 收集相关信息\n3. 整理和分析信息\n4. 撰写研究报告\n5. 提交完成的研究\n</WORKFLOW>""",
             """<topic>{problem_statement}</topic>\n\n请进行深入研究，并提供详细的研究报告。"""),
            "general": ("""你是一个智能助手，能够帮助用户完成各种任务。\n\n<CAPABILITIES>\n- 问题解答和信息提供\n- 代码编写和调试\n- 数据分析和处理\n- 文档撰写和编辑\n- 各种专业任务\n</CAPABILITIES>\n\n<TOOLS>\n- execute_bash: 执行shell命令\n- str_replace_editor: 查看和编辑文件\n- submit: 提交任务结果\n</TOOLS>\n\n请按要求完成任务。""",
             """<task>{problem_statement}</task>\n\n请完成这个任务。"""),
        }
        sp, ip = prompts.get(agent_type, prompts["general"])
        args = AgentArgs(
            system_prompt=system_prompt or sp,
            instance_prompt=instance_prompt or ip,
            llm_name=llm_name,
            llm_base_url=llm_base_url,
            quiet=quiet,
            **kwargs
        )
        return Agent(args=args)
    return factory


def create_coder_agent(llm_name: str, llm_base_url: str | None = None,
    system_prompt: str | None = None, instance_prompt: str | None = None, **kwargs) -> 'Agent':
    return _create_agent_factory("coder")(llm_name, llm_base_url, system_prompt, instance_prompt, **kwargs)


def create_analyzer_agent(llm_name: str, llm_base_url: str | None = None,
    system_prompt: str | None = None, instance_prompt: str | None = None, **kwargs) -> 'Agent':
    return _create_agent_factory("analyzer")(llm_name, llm_base_url, system_prompt, instance_prompt, **kwargs)


def create_research_agent(llm_name: str, llm_base_url: str | None = None,
    system_prompt: str | None = None, instance_prompt: str | None = None, **kwargs) -> 'Agent':
    return _create_agent_factory("research")(llm_name, llm_base_url, system_prompt, instance_prompt, **kwargs)


def create_general_agent(llm_name: str, llm_base_url: str | None = None,
    system_prompt: str | None = None, instance_prompt: str | None = None,
    quiet: bool = False, **kwargs) -> 'Agent':
    return _create_agent_factory("general")(llm_name, llm_base_url, system_prompt, instance_prompt, quiet, **kwargs)


def create_agent(agent_type: str = "general", llm_name: str = "openai/gpt-4",
    llm_base_url: str | None = None, system_prompt: str | None = None,
    instance_prompt: str | None = None, **kwargs) -> 'Agent':
    """创建Agent的工厂函数，支持coder/analyzer/research/general类型"""
    factories = {
        "coder": create_coder_agent,
        "analyzer": create_analyzer_agent,
        "research": create_research_agent,
        "general": create_general_agent,
    }
    factory = factories.get(agent_type, create_general_agent)
    assert factory is not None
    return factory(llm_name=llm_name, llm_base_url=llm_base_url,  # type: ignore[operator]
        system_prompt=system_prompt, instance_prompt=instance_prompt, **kwargs)


def get_registry() -> AgentRegistry:
    """获取全局Agent注册表"""
    return _global_registry


_global_registry.register_factory("coder", create_coder_agent)
_global_registry.register_factory("analyzer", create_analyzer_agent)
_global_registry.register_factory("research", create_research_agent)
_global_registry.register_factory("general", create_general_agent)


