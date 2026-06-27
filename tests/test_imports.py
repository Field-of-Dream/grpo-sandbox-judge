import importlib


def test_benchmark_runner_import_does_not_require_agent_dependencies():
    module = importlib.import_module("grpo_in_sandbox.benchmark.runner")

    assert hasattr(module, "load_dataset_from_config")


def test_lightweight_public_api_import_does_not_require_agent_dependencies():
    package = importlib.import_module("grpo_in_sandbox")

    assert hasattr(package, "create_runtime")
