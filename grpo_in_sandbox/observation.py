
import json

from .action import Action

# 当智能体未调用任何工具时返回的提示消息
CONTINUE_MSG = """
You forgot to use a function call in your response.
YOU MUST USE A FUNCTION CALL IN EACH RESPONSE.

IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.
"""

class Observation:
    """
    表示动作执行后的观察结果。

    Attributes:
        bash_output: 命令的组合输出（兼容旧版）
        error_code: 命令的退出码
        action: 对应的动作对象
        num_lines: 截断输出时保留的行数
        stdout: 标准输出（分离后）
        stderr: 标准错误（分离后）
    """

    def __init__(self, bash_output: str, error_code: int, action: Action, num_lines: int = 40,
                 stdout: str | None = None, stderr: str | None = None):
        """
        初始化观察结果。

        Args:
            bash_output: 组合输出（兼容旧版）
            error_code: 命令退出码
            action: 对应的动作对象
            num_lines: 截断时保留的首尾行数
            stdout: 标准输出（分离后）
            stderr: 标准错误（分离后）
        """
        self.bash_output = bash_output
        self.error_code = error_code
        self.action = action
        self.num_lines = num_lines
        self.stdout = stdout
        self.stderr = stderr

    def _truncate_output(self, output: str) -> str:
        """
        截断过长的输出，保留首尾部分以节省上下文。

        Args:
            output: 要截断的输出字符串

        Returns:
            截断后的输出，保留首尾各num_lines行
        """
        if not output:
            return ""
        lines = output.splitlines()
        if len(lines) > 2 * self.num_lines:
            top_lines = "\n".join(lines[:self.num_lines])
            bottom_lines = "\n".join(lines[-self.num_lines:])
            divider = "-" * 50
            return (
                f"{top_lines}\n"
                f"{divider}\n"
                f"<Observation truncated in middle for saving context>\n"
                f"{divider}\n"
                f"{bottom_lines}"
            )
        return output

    def __str__(self):
        """
        将观察结果转换为字符串格式。

        Returns:
            格式化的观察结果字符串
        """
        # 安全获取函数名
        func_name = getattr(self.action, 'function_name', '') if self.action else ''

        if not func_name:
            return CONTINUE_MSG
        elif func_name == "submit":
            return "<<< Finished >>>"
        else:
            if func_name == "execute_bash" or func_name == "bash":
                # 检查是否有分离的stdout/stderr
                if self.stdout is not None or self.stderr is not None:
                    # 使用分离的stdout/stderr格式
                    stdout_str = self._truncate_output(self.stdout or "")
                    stderr_str = self._truncate_output(self.stderr or "")

                    output_parts = [f"Exit code: {self.error_code}"]
                    # 使用转义括号防止Rich解释为标记
                    output_parts.append(f"Execution output of \\[{func_name}]:")

                    if stdout_str.strip():
                        output_parts.append(f"\\[STDOUT]\n{stdout_str}")
                    else:
                        output_parts.append("\\[STDOUT]\n")

                    if stderr_str.strip():
                        output_parts.append(f"\\[STDERR]\n{stderr_str}")
                    # 仅在实际有stderr内容时显示STDERR部分

                    output = "\n".join(output_parts)
                else:
                    # 回退到旧版组合输出
                    truncated_output = self._truncate_output(self.bash_output or "")
                    output = (
                        f"Exit code: {self.error_code}\n"
                        f"Execution output of \\[{func_name}]:\n"
                        f"{truncated_output}"
                    )
            else:
                # For non-bash tools, show the action that was performed
                params = getattr(self.action, 'parameters', {})
                params_str = json.dumps(params, ensure_ascii=False) if params else 'none'
                output = f"Execution output of \\[{func_name}]:\nAction parameters: {params_str}"
            return output
