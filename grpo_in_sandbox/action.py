"""
动作模块 - 表示智能体的函数调用操作

本模块定义了Action类，用于封装智能体可以执行的函数调用，
包括函数名称和参数。
"""



class Action:
    """
    表示一个动作，包含：
      - function_name: 函数名称（如 'execute_bash', 'str_replace_editor'）
      - parameters: 参数字典，参数名到值的映射
    """

    def __init__(self, function_name: str, parameters: dict[str, str], function_id: str = None):
        """
        初始化动作。

        Args:
            function_name: 要执行的函数名称
            parameters: 参数名到值的字典
            function_id: 可选的函数调用标识符
        """
        self.function_name = function_name
        self.parameters = parameters
        self.function_id = function_id

    def __str__(self) -> str:
        """返回动作的字符串表示。"""
        return str(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """
        将动作转换为字典格式，用于序列化。

        Returns:
            包含 'function' 和 'parameters' 键的字典
        """
        return {"function": self.function_name, "parameters": self.parameters}
