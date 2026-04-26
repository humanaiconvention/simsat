"""
Tests for scripts/diagnose_gemma4_checkpoint.py.

These verify the CLI surface and verdict-classification logic without
requiring transformers / peft / torch / Kaggle adapter weights. The
heavy paths (_check_masking, _check_loss) are integration-tested
manually after a real Kaggle download — they are not exercised here.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "diagnose_gemma4_checkpoint.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("diagnose_gemma4", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCLI:
    def test_script_exists_and_is_executable(self):
        assert SCRIPT_PATH.exists()

    def test_help_runs_without_error(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "Triage a trained Gemma-4 checkpoint" in result.stdout

    def test_loss_check_without_adapter_exits_2(self):
        # Should fail-fast with usage error, not crash mid-run.
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--check", "loss"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 2
        assert "requires --adapter-path" in result.stderr


class TestVerdictLogic:
    def setup_method(self):
        self.mod = _load_module()

    def test_masking_broken_below_threshold(self):
        v = self.mod._emit_verdict(mask_ratio=0.20, loss=None)
        assert v == "MASKING_BROKEN"

    def test_masking_too_aggressive_above_threshold(self):
        v = self.mod._emit_verdict(mask_ratio=0.99, loss=None)
        assert v == "MASKING_TOO_AGGRESSIVE"

    def test_masking_ok_loss_descended(self):
        v = self.mod._emit_verdict(mask_ratio=0.75, loss=2.4)
        assert v == "MASKING_OK_LOSS_DESCENDED"

    def test_masking_ok_loss_flat_at_v3_baseline(self):
        v = self.mod._emit_verdict(mask_ratio=0.75, loss=3.9454)
        assert v == "MASKING_OK_LOSS_FLAT"

    def test_masking_ok_no_loss_check(self):
        v = self.mod._emit_verdict(mask_ratio=0.75, loss=None)
        assert v == "MASKING_OK_LOSS_NOT_CHECKED"

    def test_inconclusive_when_nothing_provided(self):
        v = self.mod._emit_verdict(mask_ratio=None, loss=None)
        assert v == "INCONCLUSIVE"

    def test_boundary_low_healthy(self):
        # Exactly at MASK_RATIO_MIN_HEALTHY (0.40) → still healthy
        v = self.mod._emit_verdict(mask_ratio=0.40, loss=2.0)
        assert v == "MASKING_OK_LOSS_DESCENDED"

    def test_boundary_high_healthy(self):
        # Exactly at MASK_RATIO_MAX_HEALTHY (0.95) → still healthy
        v = self.mod._emit_verdict(mask_ratio=0.95, loss=2.0)
        assert v == "MASKING_OK_LOSS_DESCENDED"


class TestNextStepRouting:
    def setup_method(self):
        self.mod = _load_module()

    def test_descended_verdict_includes_eval_command(self, tmp_path):
        adapter_dir = tmp_path / "adapter_v4"
        msg = self.mod._next_step_for("MASKING_OK_LOSS_DESCENDED", adapter_dir)
        # Routes to the SimSat fine-tune backend (NOT the legacy gemma4_haic_local).
        assert "OBSERVATION_VLA_BACKEND=gemma4" in msg
        assert "OBSERVATION_VLM_LORA_PATH" in msg
        assert "observation_vla_eval.py" in msg
        assert str(adapter_dir) in msg
        # Sanity: the warning about the legacy backend is included.
        assert "gemma4_haic_local" in msg.lower() or "v35-gov" in msg.lower()

    def test_broken_verdict_recommends_fix_16(self):
        msg = self.mod._next_step_for("MASKING_BROKEN", None)
        assert "Fix #16" in msg
        assert "_patch_notebook_v5" in msg

    def test_flat_verdict_recommends_hyperparameter_check(self):
        msg = self.mod._next_step_for("MASKING_OK_LOSS_FLAT", Path("/tmp/x"))
        assert "learning rate" in msg.lower() or "hyperparameter" in msg.lower()
