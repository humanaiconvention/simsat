"""Genesis VLA backend — General AI Track collaborator seat (Guilherme Ferrari Brescia).

Native loader for Genesis-152M-Instruct (custom architecture; HF model card:
https://huggingface.co/guiferrarib/genesis-152m-instruct). Uses the genesis
package directly — does NOT delegate to TransformersVLMAdapter.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLLABORATOR SETUP — Guilherme Ferrari Brescia (Genesis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — Clone your genesis package repo and set the path:

    # HF model card: https://huggingface.co/guiferrarib/genesis-152m-instruct
    # (the loader needs the source package, not just the safetensors)
    export GENESIS_REPO_PATH=/path/to/genesis-package

Step 2 — Point to your weights file:

    export GENESIS_WEIGHTS_PATH=/path/to/genesis_152m_instruct.safetensors

Step 3 — Set the backend and run eval:

    export OBSERVATION_VLA_BACKEND=genesis
    export OBSERVATION_VLA_DEVICE=cuda   # or cpu
    python scripts/observation_vla_eval.py --inprocess

You should see: runtime_mode=genesis_local in the output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your model's response must contain a single JSON object:

{
  "usable_observation": true,
  "scene_match_score": 0.82,        // float 0-1: does the tile match target?
  "salience_score": 0.75,           // float 0-1: signal / detail quality
  "change_or_event_score": 0.60,    // float 0-1: notable change detected?
  "occlusion_or_cloud_risk": 0.15,  // float 0-1: cloud / occlusion fraction
  "confidence": 0.78,               // float 0-1: overall assessment confidence
  "recommended_action": "accept",   // "accept" | "defer" | "refine" | "skip"
  "rationale_tags": ["genesis_local", "sentinel_support"]
}

Full prompt format and worked examples: see COLLABORATOR_GUIDE.md at repo root.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your model sits in the VLA lane of a governed on-orbit continual-learning loop:

  Sentinel tile → Genesis (you) → ObservationVLA JSON
    → WCLI trust layer → viability gates (6 non-compensatory checks)
    → accept / defer / refine / skip decision
    → mission-response layer (logged utility)
    → TTT feedback loop (trust-layer weights updated from realized utility)

See CHALLENGE_ENTRY.md and COLLABORATOR_GUIDE.md for the full framing.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {
    "usable_observation",
    "scene_match_score",
    "salience_score",
    "change_or_event_score",
    "occlusion_or_cloud_risk",
    "confidence",
    "recommended_action",
    "rationale_tags",
}
_SUPPORTED_ACTIONS = {"accept", "defer", "refine", "skip"}

_SYSTEM_PROMPT = (
    "You are a satellite encounter-assessment AI. "
    "Given metadata about a planned Earth observation, return a compact JSON assessment. "
    "Fields: usable_observation (bool), scene_match_score (float 0-1), "
    "salience_score (float 0-1), change_or_event_score (float 0-1), "
    "occlusion_or_cloud_risk (float 0-1), confidence (float 0-1), "
    "recommended_action (accept|defer|refine|skip), rationale_tags (list[str]). "
    "Return only valid JSON, no markdown, no explanation."
)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _resolve_weights_path() -> Path:
    env = os.environ.get("GENESIS_WEIGHTS_PATH", "")
    if env:
        return Path(env).expanduser()
    # Fallback: look relative to GENESIS_WEIGHTS_DIR (legacy) or ./weights/genesis/
    root = Path(os.environ.get("GENESIS_WEIGHTS_DIR", "./weights/genesis")).expanduser()
    return root / "genesis_152m_instruct.safetensors"


def _resolve_repo_path() -> Path | None:
    env = os.environ.get("GENESIS_REPO_PATH", "")
    return Path(env).expanduser() if env else None


class GenesisAdapter:
    """Native Genesis-152M adapter for ObservationVLA.

    Loads genesis_152m_instruct.safetensors using the genesis package
    (HF model: guiferrarib/genesis-152m-instruct). Falls back to stub if
    the package or weights are unavailable.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        repo_path: str | None = None,
        device: str | None = None,
        allow_fallback: bool = True,
        max_new_tokens: int = 256,
    ) -> None:
        self.allow_fallback = allow_fallback
        self.max_new_tokens = max_new_tokens
        self._load_error: str | None = None
        self._model = None
        self._tokenizer = None
        self._device = (
            device
            or os.environ.get("OBSERVATION_VLA_DEVICE")
            or ("cuda" if self._cuda_available() else "cpu")
        )

        resolved_weights = Path(weights_path).expanduser() if weights_path else _resolve_weights_path()
        resolved_repo = Path(repo_path).expanduser() if repo_path else _resolve_repo_path()

        try:
            self._load(resolved_weights, resolved_repo)
        except Exception as exc:
            self._load_error = str(exc)
            if not allow_fallback:
                raise
            logger.warning(
                "GenesisAdapter load failed — falling back to stub. Reason: %s",
                exc,
            )

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load(self, weights_path: Path, repo_path: Path | None) -> None:
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Genesis weights not found at {weights_path}. "
                "Set GENESIS_WEIGHTS_PATH to the .safetensors file."
            )

        # Add genesis repo to sys.path if provided and not already importable
        _added_path: str | None = None
        try:
            import genesis  # noqa: F401 — test import
        except ImportError:
            if repo_path is None:
                raise ImportError(
                    "genesis package not importable. Set GENESIS_REPO_PATH to "
                    "the local genesis package repo root, or pip install it. "
                    "HF model: https://huggingface.co/guiferrarib/genesis-152m-instruct"
                )
            if not repo_path.exists():
                raise FileNotFoundError(f"GENESIS_REPO_PATH does not exist: {repo_path}")
            _added_path = str(repo_path)
            if _added_path not in sys.path:
                sys.path.insert(0, _added_path)

        try:
            import torch
            from safetensors import safe_open
            from safetensors.torch import load_file
            from genesis.model import Genesis, GenesisConfig
            from genesis.tokenizer import get_tokenizer
        except ImportError as exc:
            raise ImportError(f"Failed to import genesis dependencies: {exc}") from exc

        # Read config from safetensors metadata
        with safe_open(str(weights_path), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        if "genesis_config_json" in meta:
            config = GenesisConfig(**json.loads(meta["genesis_config_json"]))
        else:
            config = GenesisConfig.genesis_best()

        state_dict = load_file(str(weights_path), device=self._device)

        model = Genesis(config).to(self._device)
        # Tie word embeddings if lm_head was not saved separately
        if "lm_head.weight" not in state_dict and getattr(config, "tie_word_embeddings", True):
            if "tok_emb.weight" in state_dict:
                state_dict["lm_head.weight"] = state_dict["tok_emb.weight"]
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        tokenizer = get_tokenizer("neox")
        if hasattr(tokenizer, "add_chat_tokens"):
            tokenizer.add_chat_tokens()

        self._model = model
        self._tokenizer = tokenizer

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def runtime_mode(self) -> str:
        if self._load_error:
            return "stub_fallback"
        return "genesis_local"

    @property
    def model_id(self) -> str:
        if self._load_error:
            return "genesis:stub:fallback"
        return "genesis:native:genesis-152m-instruct"

    def assess(
        self,
        prompt: str,
        images: list[Any],
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        if self._load_error or self._model is None:
            return self._fallback_payload(prompt, response_schema, error=self._load_error or "not loaded")

        try:
            raw_text = self._generate(prompt)
            payload = self._parse(raw_text)
            self._enrich(payload, raw_text=raw_text)
            return payload
        except Exception as exc:
            if not self.allow_fallback:
                raise
            return self._fallback_payload(prompt, response_schema, error=str(exc))

    # ── Generation ───────────────────────────────────────────────────────────

    def _build_prompt(self, user_text: str) -> str:
        return (
            f"<|im_start|>system\n{_SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_text}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _generate(self, user_text: str) -> str:
        import torch

        full_prompt = self._build_prompt(user_text)
        input_ids = self._tokenizer.encode(full_prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)

        stop_tokens = None
        if hasattr(self._tokenizer, "im_end_id") and self._tokenizer.im_end_id is not None:
            stop_tokens = [self._tokenizer.im_end_id]

        with torch.no_grad():
            output_ids = self._model.generate(
                input_tensor,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,   # low temp for deterministic JSON
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.05,
                stop_tokens=stop_tokens,
            )

        new_tokens = output_ids[0][input_tensor.shape[1]:]
        response = self._tokenizer.decode(new_tokens.tolist())
        return response.replace("<|im_end|>", "").strip()

    # ── Parsing ──────────────────────────────────────────────────────────────

    def _parse(self, raw_text: str) -> dict[str, Any]:
        stripped = raw_text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object in genesis output: {raw_text[:200]!r}")
        payload = json.loads(match.group(0))

        missing = [k for k in _REQUIRED_KEYS if k not in payload]
        if missing:
            raise ValueError(f"genesis output missing required keys: {missing}")

        payload["usable_observation"] = bool(payload["usable_observation"])
        for k in ("scene_match_score", "salience_score", "change_or_event_score",
                  "occlusion_or_cloud_risk", "confidence"):
            payload[k] = _clamp(float(payload[k]))
        payload["recommended_action"] = str(payload["recommended_action"]).strip().lower()
        if payload["recommended_action"] not in _SUPPORTED_ACTIONS:
            payload["recommended_action"] = "refine"
        payload["rationale_tags"] = [str(t) for t in payload.get("rationale_tags", [])]
        return payload

    def _enrich(self, payload: dict[str, Any], *, raw_text: str) -> None:
        tags = list(payload.get("rationale_tags", []))
        for tag in ("genesis_local", "genesis_native"):
            if tag not in tags:
                tags.append(tag)
        payload["rationale_tags"] = list(dict.fromkeys(tags))
        payload["raw_response_text"] = (
            f"genesis_local:{payload['recommended_action']}:"
            f"conf={payload['confidence']:.2f}|"
            + raw_text.replace("\n", " ")[:600]
        )
        payload["_model_id"] = self.model_id

    # ── Fallback ─────────────────────────────────────────────────────────────

    def _fallback_payload(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        *,
        error: str = "",
    ) -> dict[str, Any]:
        return {
            "usable_observation": False,
            "scene_match_score": 0.5,
            "salience_score": 0.5,
            "change_or_event_score": 0.5,
            "occlusion_or_cloud_risk": 0.5,
            "confidence": 0.1,
            "recommended_action": "defer",
            "rationale_tags": ["genesis_local", "stub_fallback"],
            "raw_response_text": f"genesis_fallback:defer:0.10|error={error[:200]}",
            "_model_id": self.model_id,
        }
