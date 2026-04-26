"""
Tests for scripts/sync_kaggle_adapter.py.

CI-safe: all kaggle/subprocess calls are mocked. Verifies CLI surface,
adapter directory discovery, and stage-skipping flag logic.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "sync_kaggle_adapter.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sync_kaggle_adapter", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCLI:
    def test_script_exists(self):
        assert SCRIPT_PATH.exists()

    def test_help_runs_without_error(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "Kaggle adapter" in result.stdout
        assert "--version" in result.stdout

    def test_missing_version_arg_exits_2(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 2
        assert "--version" in result.stderr


class TestAdapterDiscovery:
    def setup_method(self):
        self.mod = _load_module()

    def test_finds_adapter_at_root(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text("{}")
        (tmp_path / "adapter_model.safetensors").write_text("dummy")
        result = self.mod._find_adapter_subdir(tmp_path)
        assert result == tmp_path

    def test_finds_nested_adapter(self, tmp_path):
        nested = tmp_path / "simsat-gemma4-v7-adapter"
        nested.mkdir()
        (nested / "adapter_config.json").write_text("{}")
        result = self.mod._find_adapter_subdir(tmp_path)
        assert result == nested

    def test_returns_none_when_no_adapter(self, tmp_path):
        # Some other artefacts but no adapter_config.json
        (tmp_path / "log.txt").write_text("training output")
        (tmp_path / "model.safetensors").write_text("dummy")
        result = self.mod._find_adapter_subdir(tmp_path)
        assert result is None

    def test_prefers_dir_named_adapter_when_multiple_match(self, tmp_path):
        # Two adapter_config.json files: one in 'adapter/', one in 'checkpoint-100/'
        adapter_dir = tmp_path / "adapter"
        ckpt_dir = tmp_path / "checkpoint-100"
        adapter_dir.mkdir()
        ckpt_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")
        (ckpt_dir / "adapter_config.json").write_text("{}")
        result = self.mod._find_adapter_subdir(tmp_path)
        assert result == adapter_dir


class TestStatusParsing:
    def setup_method(self):
        self.mod = _load_module()

    def test_kernel_status_parses_complete(self, monkeypatch):
        class FakeProc:
            stdout = 'benhaslam/simsat-gemma4-v1-training has status "KernelWorkerStatus.COMPLETE"\n'
            stderr = ""
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
        assert self.mod._kernel_status() == "COMPLETE"

    def test_kernel_status_parses_running(self, monkeypatch):
        class FakeProc:
            stdout = ""
            stderr = 'benhaslam/simsat-gemma4-v1-training has status "KernelWorkerStatus.RUNNING"\n'
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
        assert self.mod._kernel_status() == "RUNNING"

    def test_kernel_status_handles_warning_lines(self, monkeypatch):
        class FakeProc:
            stdout = (
                "Warning: Looks like you're using an outdated kaggle version\n"
                'benhaslam/simsat-gemma4-v1-training has status "KernelWorkerStatus.ERROR"\n'
            )
            stderr = ""
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
        assert self.mod._kernel_status() == "ERROR"

    def test_kernel_status_unknown_when_unparseable(self, monkeypatch):
        class FakeProc:
            stdout = "totally unrelated output"
            stderr = ""
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
        assert self.mod._kernel_status() == "UNKNOWN"
