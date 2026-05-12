# 简化代码文档计划

## 目标
简化各个模块的 docstring，使用简洁的中文，避免过长的描述。

## 需要修改的文件

### 1. docker_runtime.py
- 当前：过长的中文描述
- 修改为：
```python
"""
Docker容器运行环境 - 在容器里执行命令

用法:
    runtime = DockerRuntime(docker_image="xxx", repo_path="/testbed")
    output, code = runtime.run("echo hello")
    runtime.close()
"""
```

### 2. tools.py
- 当前：过长的中文描述
- 修改为：
```python
"""
工具定义 - Agent可用的三个工具

1. str_replace_editor: 查看、创建、编辑文件
2. execute_bash: 执行bash命令
3. submit: 完成任务
"""
```

### 3. action.py
- 检查并简化 docstring

### 4. observation.py  
- 检查并简化 docstring

## 执行策略

使用 Edit 工具直接修改，每个文件只需要 1-2 行修改。

---

## TODOs

- [ ] 1. 简化 docker_runtime.py 顶部 docstring
- [ ] 2. 简化 tools.py 顶部 docstring
- [ ] 3. 检查并简化 action.py
- [ ] 4. 检查并简化 observation.py

## 验证

修改后运行 lsp_diagnostics 确认无语法错误。