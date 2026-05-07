"""Unit tests for `observation_vla.lfm_ttt.OnlineLoRAStepper`.

These tests use an injected `forward_loss_fn` so they verify the gate
logic, history bookkeeping, and step-vs-skip decision *without* needing
a real LFM2.5-VL model loaded — the tests must run on CI machines that
don't have GPU or transformers + LiquidAI weights cached.
"""
from __future__ import annotations

import pytest

from observation_vla.lfm_ttt import (
    OnlineLoRAStepper,
    OnlineStepResult,
    _action_error_sign,
    build_lfm_snapshot,
)
from haic.viability import (
    TTT_BIAS_WINDOW,
    evaluate_ttt_viability,
)


# ---------------------------------------------------------------------------
# Pure-function tests: action error sign + snapshot shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "predicted, target, expected_sign",
    [
        ("accept", "accept", 0),
        ("accept", "skip", 1),
        ("skip", "accept", -1),
        ("refine", "defer", 1),
        ("defer", "refine", -1),
        ("refine", "refine", 0),
        (None, "accept", 0),       # unparseable prediction → 0
        ("garbled", "accept", 0),   # unknown action → 0
    ],
)
def test_action_error_sign_ladder(predicted, target, expected_sign):
    assert _action_error_sign(predicted, target) == expected_sign


def test_snapshot_has_shape_compatible_with_evaluate_ttt_viability():
    snap = build_lfm_snapshot(
        update_count=5,
        lora_delta_l2=0.1,
        recent_errors=[{"error": 0} for _ in range(3)],
    )
    # Should not raise; the snapshot must be the same shape WCLITrustModel produces.
    gates = evaluate_ttt_viability(snap)
    assert set(gates) == {"weight_drift", "update_rate", "error_bias"}
    assert all(isinstance(v, bool) for v in gates.values())


# ---------------------------------------------------------------------------
# OnlineLoRAStepper behavior
# ---------------------------------------------------------------------------

class _FakeOptimizer:
    """Optimizer stub that records calls without mutating anything."""

    def __init__(self):
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self):
        self.step_calls += 1

    def zero_grad(self):
        self.zero_grad_calls += 1


class _FakeModel:
    """Model stub for tests; doesn't need to be a real PEFT model."""

    def __init__(self):
        self.train_called = 0

    def train(self):
        self.train_called += 1
        return self

    def eval(self):
        return self

    def named_parameters(self):
        return iter([])  # no LoRA params; lora_delta_l2 stays at 0


def _make_forward_loss_fn(predictions: list[str]):
    """Return a forward_loss_fn that yields the next prediction in `predictions`."""
    iterator = iter(predictions)

    def fn(model, processor, messages, target):
        try:
            pred = next(iterator)
        except StopIteration:
            pred = None
        # Loss as a python float — stepper handles this without backward()
        return 1.0, pred

    return fn


def _msgs(action_target: str) -> tuple[list[dict], str]:
    """Build (messages_no_assistant, target_json_str) pair."""
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": "You are a tester."}]},
        {"role": "user",   "content": [{"type": "text", "text": "Assess."}]},
    ]
    target = '{"recommended_action": "' + action_target + '", "confidence": 0.5}'
    return msgs, target


def test_online_step_executes_when_predictions_match_target():
    """Matching predictions ⇒ action_error=0 ⇒ gate passes ⇒ step fires."""
    model = _FakeModel()
    opt = _FakeOptimizer()
    stepper = OnlineLoRAStepper(
        model=model, processor=None, optimizer=opt,
        forward_loss_fn=_make_forward_loss_fn(["accept"] * 5),
    )

    msgs, target = _msgs("accept")
    for _ in range(5):
        result = stepper.online_step(msgs, target)
        assert isinstance(result, OnlineStepResult)
        assert result.did_step is True
        assert result.action_error == 0
        assert result.blocked_by is None

    assert stepper.update_count == 5
    assert stepper.skipped_count == 0


def test_error_bias_blocks_after_full_window_same_sign():
    """10 same-sign errors ⇒ error_bias gate blocks the 11th step."""
    model = _FakeModel()
    opt = _FakeOptimizer()
    # All predictions over-call (predict accept when target is skip → +1 sign).
    stepper = OnlineLoRAStepper(
        model=model, processor=None, optimizer=opt,
        forward_loss_fn=_make_forward_loss_fn(["accept"] * 12),
    )

    msgs, target = _msgs("skip")
    # First TTT_BIAS_WINDOW (10) steps proceed; the gate is vacuous before
    # the window is full.
    for i in range(TTT_BIAS_WINDOW):
        result = stepper.online_step(msgs, target)
        assert result.did_step is True, f"step {i+1} should have fired before window full"
        assert result.action_error == 1

    # 11th: gate fires (window of 10 same-sign errors)
    result = stepper.online_step(msgs, target)
    assert result.did_step is False
    assert result.blocked_by == "error_bias"
    assert result.action_error == 1
    assert stepper.skipped_count == 1
    assert stepper.update_count == TTT_BIAS_WINDOW  # not incremented on skip


def test_skipped_step_advances_window_so_gate_can_clear():
    """After firing, opposite-sign errors must let the gate clear once the
    last-10 window diversifies below 70% same-sign.

    Trace:
      Phase 1: 10 over-calls (predict 'accept' on target 'skip', error=+1).
               History = [+1]*10. The 11th step would fire the gate.
      Phase 2: predict 'skip' on target 'accept', error=-1. Each step:
               P2.1 gate sees last 10 = [+1]*10  ⇒ 100%, blocks
               P2.2 gate sees last 10 = [+1]*9 + [-1]*1  ⇒  90%, blocks
               P2.3 gate sees last 10 = [+1]*8 + [-1]*2  ⇒  80%, blocks
               P2.4 gate sees last 10 = [+1]*7 + [-1]*3  ⇒  70%, blocks (>=)
               P2.5 gate sees last 10 = [+1]*6 + [-1]*4  ⇒  60%, CLEARS
    """
    model = _FakeModel()
    opt = _FakeOptimizer()
    preds = ["accept"] * 10 + ["skip"] * 6
    stepper = OnlineLoRAStepper(
        model=model, processor=None, optimizer=opt,
        forward_loss_fn=_make_forward_loss_fn(preds),
    )

    msgs, target_skip = _msgs("skip")
    for _ in range(10):
        stepper.online_step(msgs, target_skip)

    msgs, target_acc = _msgs("accept")
    p2_results = []
    for _ in range(6):
        result = stepper.online_step(msgs, target_acc)
        p2_results.append(result.did_step)

    # P2 steps 1-4 should be blocked (gate ≥ 70% same-sign),
    # step 5 onward should clear once the window drops to 60%.
    assert p2_results[:4] == [False, False, False, False], (
        f"Expected first 4 P2 steps to be blocked, got: {p2_results[:4]}"
    )
    assert p2_results[4] is True, (
        f"Expected P2 step 5 to clear (window: 6 pos / 4 neg = 60%), got blocked"
    )


def test_history_includes_blocked_entries_with_blocked_flag():
    """Blocked steps still appear in the history with `blocked: True`."""
    model = _FakeModel()
    opt = _FakeOptimizer()
    stepper = OnlineLoRAStepper(
        model=model, processor=None, optimizer=opt,
        forward_loss_fn=_make_forward_loss_fn(["accept"] * 12),
    )
    msgs, target = _msgs("skip")
    for _ in range(12):
        stepper.online_step(msgs, target)

    history = stepper.history()
    blocked = [h for h in history if h.get("blocked")]
    applied = [h for h in history if not h.get("blocked")]
    assert len(blocked) == 2  # steps 11 and 12 should both be blocked
    assert len(applied) == TTT_BIAS_WINDOW
    # The blocked entries record the same +1 action_error
    assert all(b["error"] == 1 for b in blocked)


def test_optimizer_step_count_matches_executed_steps():
    """Tracks: optimizer.step is called only on did_step=True (not on skip)."""
    model = _FakeModel()
    opt = _FakeOptimizer()
    # Loss as float means stepper short-circuits — optimizer is never called.
    # Real test: provide a torch tensor stub with backward().
    import types

    class _LossLikeTensor:
        def __init__(self, v):
            self.v = float(v)

        def backward(self):
            return None

        def detach(self):
            class _D:
                def __init__(self, v):
                    self._v = v

                def item(self):
                    return self._v
            return _D(self.v)

    def fwd(model, proc, msgs, target):
        return _LossLikeTensor(1.5), "accept"

    stepper = OnlineLoRAStepper(
        model=model, processor=None, optimizer=opt,
        forward_loss_fn=fwd,
    )
    msgs, target = _msgs("accept")
    for _ in range(5):
        stepper.online_step(msgs, target)

    assert opt.step_calls == 5
    assert opt.zero_grad_calls == 5


def test_unparseable_prediction_yields_zero_error_sign():
    """Garbled predictions don't bias the window in either direction."""
    model = _FakeModel()
    opt = _FakeOptimizer()
    stepper = OnlineLoRAStepper(
        model=model, processor=None, optimizer=opt,
        forward_loss_fn=_make_forward_loss_fn([None, "", "garbage"]),
    )
    msgs, target = _msgs("accept")
    for _ in range(3):
        result = stepper.online_step(msgs, target)
        assert result.action_error == 0
        assert result.did_step is True  # unbiased, gate doesn't fire


def test_snapshot_after_returns_correct_shape():
    """The result.snapshot_after must be `evaluate_ttt_viability`-compatible."""
    model = _FakeModel()
    opt = _FakeOptimizer()
    stepper = OnlineLoRAStepper(
        model=model, processor=None, optimizer=opt,
        forward_loss_fn=_make_forward_loss_fn(["accept"] * 3),
    )
    msgs, target = _msgs("accept")
    result = stepper.online_step(msgs, target)
    snap = result.snapshot_after
    gates = evaluate_ttt_viability(snap)
    assert set(gates) == {"weight_drift", "update_rate", "error_bias"}
