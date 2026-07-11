"""Tests for the GRPO training backend selection (pure TRL vs Unsloth)."""

import importlib
import importlib.util

import pytest


@pytest.fixture
def train_module():
    pytest.importorskip("torch", reason="grpo_in_sandbox.train imports torch")
    return importlib.import_module("grpo_in_sandbox.train")


def test_train_module_imports_without_training_dependencies(train_module):
    # Importing must not require unsloth/transformers/trl (they load lazily).
    assert hasattr(train_module, "TrainingBackend")
    assert hasattr(train_module, "train")


def test_training_backend_is_exported_from_package():
    pytest.importorskip("torch", reason="grpo_in_sandbox.train imports torch")
    package = importlib.import_module("grpo_in_sandbox")

    backend_enum = package.TrainingBackend

    assert backend_enum("trl").value == "trl"


def test_resolve_backend_accepts_strings_and_enum_members(train_module):
    resolve = train_module._resolve_backend
    backend_enum = train_module.TrainingBackend

    assert resolve("trl") is backend_enum.TRL
    assert resolve("unsloth") is backend_enum.UNSLOTH
    assert resolve("TRL") is backend_enum.TRL  # case-insensitive
    assert resolve(backend_enum.TRL) is backend_enum.TRL


def test_resolve_backend_rejects_unknown_backend(train_module):
    with pytest.raises(ValueError, match="Unknown training backend"):
        train_module._resolve_backend("bogus")


def test_resolve_backend_auto_prefers_unsloth_when_installed(train_module, monkeypatch):
    monkeypatch.setattr(train_module, "find_spec", lambda name: object())

    resolved = train_module._resolve_backend("auto")

    assert resolved is train_module.TrainingBackend.UNSLOTH


def test_resolve_backend_auto_falls_back_to_trl_without_unsloth(train_module, monkeypatch):
    monkeypatch.setattr(train_module, "find_spec", lambda name: None)

    resolved = train_module._resolve_backend("auto")

    assert resolved is train_module.TrainingBackend.TRL


def test_config_defaults_to_auto_backend(train_module):
    config = train_module.RLHFTrainingConfig()

    assert train_module.TrainingBackend(config.backend) is train_module.TrainingBackend.AUTO
    assert config.load_in_4bit is False
    assert "backend=auto" in repr(config)


def test_trl_loader_raises_helpful_error_without_transformers(train_module):
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers installed; missing-dependency path not reachable")

    with pytest.raises(ImportError, match=r"grpo-in-sandbox\[training\]"):
        train_module._load_model_and_tokenizer_trl(train_module.RLHFTrainingConfig())


def test_unsloth_loader_raises_helpful_error_without_unsloth(train_module):
    if importlib.util.find_spec("unsloth") is not None:
        pytest.skip("unsloth installed; missing-dependency path not reachable")

    with pytest.raises(ImportError, match=r"grpo-in-sandbox\[unsloth\]"):
        train_module._load_model_and_tokenizer_unsloth(train_module.RLHFTrainingConfig())
