"""LFM2.5-VL TTT — online LoRA gradient updates per encounter, under
viability gates.

This module implements the **VLA-layer TTT** architectural lane that the
submission docs previously described as "wired, requires live encounter
stream." With this file, the lane is *implemented* — `OnlineLoRAStepper`
performs a single forward + backward + optimizer step on the LoRA adapter
weights of a `PeftModel`-wrapped vision-language model, gated by the same
six-gate viability filter that already governs the `WCLITrustModel`
trust-layer TTT.

Architectural symmetry with the trust-layer:

    trust-layer TTT                  VLA-layer (LFM) TTT
    ---------------                  -------------------
    WCLITrustModel.online_update     OnlineLoRAStepper.online_step
    realized_utility - learned_score cross-entropy on assistant tokens
    weight_drift on 5 WCLI weights   lora_delta_l2 on adapter tensors
    error_bias on +/- 1 errors       error_bias on +/- 1 action mismatches
    update_count ceiling             update_count ceiling
    record_skipped_observation       record_skipped_observation (mirrored)

The `error_bias` gate is **BLOCKING** in both layers: a same-sign run of
operator-vs-prediction errors over the last 10 updates suppresses the
step rather than reinforcing the bias direction.

Usage (synchronous, matches the trust-layer pattern):

    stepper = OnlineLoRAStepper(peft_model, processor, optimizer)

    for encounter in stream:
        result = stepper.online_step(
            messages=encounter.messages_no_assistant(),
            target_text=encounter.operator_target_json_str(),
        )
        if result["did_step"]:
            log("ttt step %d: loss=%.3f action_error=%+d",
                stepper.update_count, result["loss"], result["action_error"])
        else:
            log("ttt step %d SKIPPED by gate: %s",
                stepper.update_count, result["blocked_by"])

This module is designed so it can be unit-tested without a real LFM model
(see `tests/test_lfm_ttt.py`) — the `_action_error_sign` and gate logic
are pure-Python, and `online_step` accepts an injected `forward_loss_fn`
in tests.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

# These names are reused from the trust-layer TTT viability module so the
# blocking semantics stay aligned across both layers.
from haic.viability import (
    MAX_TTT_UPDATE_COUNT,
    MAX_TTT_WEIGHT_DRIFT,
    TTT_BIAS_THRESHOLD,
    TTT_BIAS_WINDOW,
)

ActionLabel = str  # one of "accept" | "refine" | "defer" | "skip"


# ---------------------------------------------------------------------------
# Action ordering (for signed error semantics)
# ---------------------------------------------------------------------------

# We map the four operator actions onto a usefulness ladder so that
# "predicted higher than operator" produces a +1 sign and "predicted lower"
# produces -1. The ladder order matches the band map used elsewhere in the
# packet: skip=0.20, defer=0.40, refine=0.55, accept=0.85.
_ACTION_RANK: dict[str, int] = {
    "skip": 0,
    "defer": 1,
    "refine": 2,
    "accept": 3,
}


def _action_error_sign(predicted: ActionLabel | None, target: ActionLabel) -> int:
    """Return +1 if predicted is more aggressive than target, -1 if less, 0 if match.

    Returns 0 when the prediction is missing or unparseable — that's a
    structural failure, not a directional one, and shouldn't bias the
    error_bias window in either direction.
    """
    if predicted not in _ACTION_RANK or target not in _ACTION_RANK:
        return 0
    delta = _ACTION_RANK[predicted] - _ACTION_RANK[target]
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Viability snapshot — same shape as WCLITrustModel.get_weight_snapshot()
# so we can reuse `evaluate_ttt_viability` directly.
# ---------------------------------------------------------------------------


def build_lfm_snapshot(
    update_count: int,
    lora_delta_l2: float,
    recent_errors: list[dict],
    policy_id: str = "lfm-ttt-poc",
) -> dict:
    """Build a snapshot dict in the format `evaluate_ttt_viability` expects.

    `lora_delta_l2` is the L2 norm of the LoRA-adapter delta vs the
    pre-TTT baseline. We expose it as a single `lfm_lora` key in
    `drift_from_policy_defaults` so the existing `weight_drift` gate
    (threshold `MAX_TTT_WEIGHT_DRIFT = 0.30`) fires when adapter drift
    exceeds 30% of the pre-TTT magnitude.

    `recent_errors` is a list of `{"error": +1/-1/0}` dicts in
    chronological order.
    """
    return {
        "learned_score_weights": {},  # not used by the TTT gates
        "trust_score_weights": {},
        "drift_from_policy_defaults": {"lfm_lora": float(lora_delta_l2)},
        "update_count": int(update_count),
        "policy_id": policy_id,
        "recent_updates": list(recent_errors),
    }


# ---------------------------------------------------------------------------
# OnlineLoRAStepper
# ---------------------------------------------------------------------------


@dataclass
class OnlineStepResult:
    did_step: bool
    update_idx: int
    blocked_by: str | None
    action_error: int
    predicted_action: str | None
    target_action: str | None
    loss: float | None
    lora_delta_l2: float
    snapshot_after: dict = field(default_factory=dict)


class OnlineLoRAStepper:
    """Wraps a PEFT-wrapped LFM model + processor + optimizer and exposes
    `online_step(messages, target_text)` that performs one forward +
    backward + step, gated by the six-gate viability filter.

    Constructor parameters
    ----------------------
    model
        A `peft.PeftModel`-wrapped vision-language model. Required to be
        in train mode for backward to work; the stepper sets it on entry
        and restores on exit.
    processor
        The HuggingFace processor for `model` (used to apply chat
        template + tokenize).
    optimizer
        Any `torch.optim.Optimizer` over the LoRA parameters of `model`.
        Caller is responsible for instantiating it once at the start of
        the stream.
    parse_action_fn
        Callable `(text: str) -> str | None` that extracts the
        `recommended_action` field from a generated assistant string.
        Defaults to a JSON-aware extractor that handles code fences and
        partial outputs.
    forward_loss_fn
        Optional injection point for tests. If None, the stepper uses
        `model(**inputs)` and reads `output.loss`. If provided, it is
        called as `forward_loss_fn(model, processor, messages, target)`
        and must return a `(loss_tensor, predicted_action_string)`
        pair.

    Behavior
    --------
    Each call to `online_step` does the following in order:

        1. Compute initial LoRA-delta L2 (lazy; cached after first call).
        2. Forward pass to get loss + predicted action string.
        3. Compute action_error (signed +1/-1/0).
        4. Build the viability snapshot from the BEFORE state.
        5. If error_bias gate fires: log to history, return without
           stepping (mirrors `record_skipped_observation` in the trust
           layer — keeps the window advancing so the gate can clear).
        6. Otherwise: backward + step + zero_grad.
        7. Append to history.
        8. Return an `OnlineStepResult`.

    The history is bounded at TTT_BIAS_WINDOW * 4 (the BLOCKING gate
    only ever inspects the last TTT_BIAS_WINDOW entries).
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        optimizer: Any,
        parse_action_fn: Callable[[str], str | None] | None = None,
        forward_loss_fn: Callable[..., tuple[Any, str | None]] | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.optimizer = optimizer
        self._parse_action = parse_action_fn or _default_parse_action
        self._forward_loss = forward_loss_fn  # None ⇒ use real forward path

        self.update_count: int = 0
        self.skipped_count: int = 0
        # History stores enough entries for the gate AND a small audit tail.
        self._history: deque[dict] = deque(maxlen=TTT_BIAS_WINDOW * 4)
        self._initial_lora_state: dict[str, Any] | None = None
        self._last_lora_delta_l2: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def online_step(
        self,
        messages: list[dict],
        target_text: str,
    ) -> OnlineStepResult:
        """Run one TTT step. See class docstring."""
        # Lazy initialisation of the LoRA-delta baseline.
        if self._initial_lora_state is None:
            self._initial_lora_state = self._snapshot_lora_state()

        target_action = self._parse_action(target_text)

        # Forward pass: gets loss + predicted action.
        loss_tensor, predicted_action = self._forward(messages, target_text)
        loss_value = float(loss_tensor.detach().item()) if hasattr(loss_tensor, "detach") else float(loss_tensor)
        action_error = _action_error_sign(predicted_action, target_action or "")

        # Pre-update viability check (BLOCKING on error_bias).
        from haic.viability import evaluate_ttt_viability  # avoid circular import at top

        snap_before = self._snapshot()
        gates_before = evaluate_ttt_viability(snap_before)

        if not gates_before.get("error_bias", True):
            # Blocked — but advance the window so the gate can clear.
            self.skipped_count += 1
            self._history.append({
                "update_idx": self.update_count,
                "error": action_error,
                "loss": loss_value,
                "blocked": True,
                "predicted_action": predicted_action,
                "target_action": target_action,
            })
            return OnlineStepResult(
                did_step=False,
                update_idx=self.update_count,
                blocked_by="error_bias",
                action_error=action_error,
                predicted_action=predicted_action,
                target_action=target_action,
                loss=loss_value,
                lora_delta_l2=self._last_lora_delta_l2,
                snapshot_after=snap_before,
            )

        # Backward + step.
        if hasattr(loss_tensor, "backward"):
            loss_tensor.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        # else: loss is a python float (test path) — skip the optimizer call.

        self.update_count += 1
        self._last_lora_delta_l2 = self._compute_lora_delta_l2()

        self._history.append({
            "update_idx": self.update_count,
            "error": action_error,
            "loss": loss_value,
            "blocked": False,
            "predicted_action": predicted_action,
            "target_action": target_action,
        })

        snap_after = self._snapshot()
        return OnlineStepResult(
            did_step=True,
            update_idx=self.update_count,
            blocked_by=None,
            action_error=action_error,
            predicted_action=predicted_action,
            target_action=target_action,
            loss=loss_value,
            lora_delta_l2=self._last_lora_delta_l2,
            snapshot_after=snap_after,
        )

    def get_snapshot(self) -> dict:
        return self._snapshot()

    def history(self) -> list[dict]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict:
        return build_lfm_snapshot(
            update_count=self.update_count,
            lora_delta_l2=self._last_lora_delta_l2,
            recent_errors=list(self._history),
        )

    def _forward(self, messages: list[dict], target_text: str) -> tuple[Any, str | None]:
        """Forward pass: returns (loss_tensor, predicted_action_string)."""
        if self._forward_loss is not None:
            return self._forward_loss(self.model, self.processor, messages, target_text)

        # Real path — apply chat template (with assistant target appended)
        # and run model forward. The predicted action is read by sampling
        # from the assistant continuation; we use teacher-forced max-likelihood
        # for loss and a greedy short-generate for the action.
        full_messages = list(messages) + [
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
        ]
        inputs = self.processor.apply_chat_template(
            [full_messages],
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        # Move to model device
        try:
            inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        except Exception:
            pass
        # Build labels with -100 mask on pad and image tokens, same as training collator.
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        image_token_id = getattr(self.model.config, "image_token_id", 396)
        labels[labels == image_token_id] = -100
        inputs["labels"] = labels

        out = self.model(**inputs)
        loss = out.loss

        # Greedy generate to get a predicted action string for the gate signal.
        # We do NOT update on this generation — it's diagnostic only.
        try:
            import torch  # type: ignore

            with torch.no_grad():
                gen_inputs = self.processor.apply_chat_template(
                    [list(messages)],
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                ).to(self.model.device)
                gen = self.model.generate(**gen_inputs, max_new_tokens=64, do_sample=False)
                prompt_len = gen_inputs["input_ids"].shape[1]
                new_tokens = gen[0, prompt_len:]
                predicted_text = self.processor.decode(new_tokens, skip_special_tokens=True)
        except Exception:
            predicted_text = ""

        return loss, self._parse_action(predicted_text)

    def _snapshot_lora_state(self) -> dict[str, Any]:
        """Capture the LoRA-adapter parameter values for delta tracking.

        Only stores the LoRA adapter tensors (those with `lora_` in their
        name), not the full base model.
        """
        state: dict[str, Any] = {}
        try:
            for name, param in self.model.named_parameters():
                if "lora_" in name and param.requires_grad:
                    state[name] = param.detach().clone()
        except Exception:
            pass
        return state

    def _compute_lora_delta_l2(self) -> float:
        """L2 norm of the LoRA delta vs the pre-TTT baseline, normalized
        by the baseline's L2 norm so the result is comparable to
        `MAX_TTT_WEIGHT_DRIFT = 0.30` (drift fraction)."""
        if not self._initial_lora_state:
            return 0.0
        try:
            import torch  # type: ignore

            delta_sq = torch.tensor(0.0)
            base_sq = torch.tensor(0.0)
            for name, param in self.model.named_parameters():
                if name in self._initial_lora_state:
                    base = self._initial_lora_state[name]
                    delta = (param.detach() - base).float()
                    delta_sq = delta_sq + (delta ** 2).sum()
                    base_sq = base_sq + (base.float() ** 2).sum()
            if float(base_sq) <= 0.0:
                return float(delta_sq.sqrt().item())
            return float((delta_sq / (base_sq + 1e-12)).sqrt().item())
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Default action parser
# ---------------------------------------------------------------------------


def _default_parse_action(text: str) -> str | None:
    """Best-effort extractor for `recommended_action` from a model output."""
    import json
    import re

    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)

    start = s.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    snippet = s[start : i + 1]
                    try:
                        payload = json.loads(snippet)
                        a = payload.get("recommended_action")
                        if isinstance(a, str) and a.lower() in _ACTION_RANK:
                            return a.lower()
                    except json.JSONDecodeError:
                        break

    # Fall-back: scan for the literal action keywords.
    lower = s.lower()
    for action in _ACTION_RANK:
        if action in lower:
            return action
    return None
