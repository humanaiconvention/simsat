#!/usr/bin/env python3
"""Liquid Track — Stage 1 pretrain (behavior cloning on Sentinel tile corpus).

This is the first stage of the three-stage Liquid Track training chain
(per CHALLENGE_ENTRY.md):

  Stage 1: Pretrain on the 86-trace Sentinel tile corpus. ← THIS SCRIPT
  Stage 2: Fine-tune with SimSat scenario-pack augmentation.
  Stage 3: Two-scope TTT per live pass.

Pre-2026-04-27 the MuZero policy was just playback of the stored VLA
`recommended_action`, which made the encoder embedding decorative —
it had no influence on reward. Stage 1 closes that gap by training a
small policy head on top of LFM2.5-VL-450M's pooled output:

  PNG tile → LFM2.5-VL-450M vision tower → (768-dim embed)
            → MLP (768 → 256 → 4) → action distribution
                                    over {accept, defer, refine, skip}

Labels:
  - operator_action when label_source == "operator_review"  (37 cases)
  - assessment.recommended_action otherwise                  (49 cases)

This produces a real learned policy that takes the encoder's
representation as input. It's the minimum that makes the Liquid Track
end-to-end training claim honest.

Usage:
    python scripts/muzero_stage1_pretrain.py
    python scripts/muzero_stage1_pretrain.py --device cpu
    python scripts/muzero_stage1_pretrain.py --encoder LiquidAI/LFM2.5-VL-1.6B
    python scripts/muzero_stage1_pretrain.py --epochs 100 --val-frac 0.2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = REPO_ROOT / "src" / "sim"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

ACTIONS = ("accept", "defer", "refine", "skip")
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


def _load_corpus(traces_path: Path, outcomes_path: Path, assets_dir: Path) -> list[dict]:
    """Build (trace_id, label, asset_path) triples for all traces with a PNG.

    Operator-reviewed labels take precedence; fall back to model assessment.
    """
    traces = json.loads(traces_path.read_text())["traces"]
    outcomes = json.loads(outcomes_path.read_text())["outcomes"]
    op_labels = {
        o["trace_id"]: o["operator_action"]
        for o in outcomes
        if o.get("label_source") == "operator_review"
    }

    rows: list[dict] = []
    for trace in traces:
        tid = trace.get("trace_id")
        if not tid:
            continue
        # Find asset
        matches = list(assets_dir.glob(f"*{tid}*.png"))
        if not matches:
            continue
        # Choose label
        if tid in op_labels:
            label = op_labels[tid]
            quality = "reviewed"
        else:
            label = (trace.get("assessment") or {}).get("recommended_action")
            quality = "simulated"
        if label not in ACTION_TO_IDX:
            continue
        rows.append({
            "trace_id": tid,
            "label": label,
            "label_idx": ACTION_TO_IDX[label],
            "quality": quality,
            "asset": matches[0],
            "scenario_pack": trace.get("scenario_pack", "unknown"),
        })
    return rows


def _encode_corpus(rows: list[dict], encoder, device: str) -> np.ndarray:
    """Encode each row's PNG asset; return (N, embed_dim) float32 array."""
    from PIL import Image

    embeds = np.zeros((len(rows), encoder.embed_dim), dtype=np.float32)
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        img = Image.open(row["asset"]).convert("RGB")
        # The encoder expects (3, H, W) float32 [0,1]
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        embeds[i] = encoder.encode(arr)
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            elapsed = time.perf_counter() - t0
            print(f"  encoded {i+1}/{len(rows)}  elapsed={elapsed:.1f}s", flush=True)
    return embeds


def _stratified_split(rows: list[dict], val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Per-action stratified train/val split (so val has every action)."""
    rng = np.random.default_rng(seed)
    by_action: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        by_action.setdefault(row["label_idx"], []).append(i)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for _, indices in by_action.items():
        idx_arr = np.asarray(indices)
        rng.shuffle(idx_arr)
        n_val = max(1, int(round(len(idx_arr) * val_frac))) if len(idx_arr) >= 2 else 0
        val_idx.extend(idx_arr[:n_val].tolist())
        train_idx.extend(idx_arr[n_val:].tolist())
    return train_idx, val_idx


def _train_head(
    embeds: np.ndarray,
    labels: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    *,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    device: str,
    seed: int,
) -> tuple[Any, dict]:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    embed_dim = embeds.shape[1]
    n_actions = len(ACTIONS)

    head = nn.Sequential(
        nn.Linear(embed_dim, hidden),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, n_actions),
    ).to(device)

    X = torch.from_numpy(embeds).to(device)
    y = torch.from_numpy(labels).long().to(device)

    Xt, yt = X[train_idx], y[train_idx]
    Xv, yv = X[val_idx], y[val_idx]

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history: list[dict] = []
    best_val = (0.0, -1)  # (acc, epoch)
    best_state = None

    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(len(Xt), device=device)
        epoch_loss = 0.0
        for start in range(0, len(Xt), batch_size):
            idx = perm[start:start + batch_size]
            logits = head(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += float(loss) * len(idx)
        epoch_loss /= max(1, len(Xt))

        head.eval()
        with torch.no_grad():
            train_pred = head(Xt).argmax(dim=1)
            val_pred = head(Xv).argmax(dim=1)
            train_acc = float((train_pred == yt).float().mean())
            val_acc = float((val_pred == yv).float().mean())
        history.append({"epoch": epoch, "loss": epoch_loss, "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_val[0]:
            best_val = (val_acc, epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={epoch_loss:.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()

    # Final per-action breakdown on val
    with torch.no_grad():
        val_pred = head(Xv).argmax(dim=1).cpu().numpy()
        val_true = yv.cpu().numpy()
    return head, {
        "history": history,
        "best_val_acc": best_val[0],
        "best_val_epoch": best_val[1],
        "final_val_acc": history[-1]["val_acc"] if history else 0.0,
        "val_pred": val_pred.tolist(),
        "val_true": val_true.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default="LiquidAI/LFM2.5-VL-450M",
                        help="HF model_id for the tile encoder.")
    parser.add_argument("--device", default=None,
                        help="cuda | cpu (auto-detect if omitted).")
    parser.add_argument("--traces", type=Path,
                        default=REPO_ROOT / "src/sim/data/observation_vla/traces.json")
    parser.add_argument("--outcomes", type=Path,
                        default=REPO_ROOT / "src/sim/data/observation_vla/outcomes.json")
    parser.add_argument("--assets-dir", type=Path,
                        default=REPO_ROOT / "review_queue_assets")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "weights/muzero/stage1_bc")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder}")
    print()

    print("Loading corpus...")
    rows = _load_corpus(args.traces, args.outcomes, args.assets_dir)
    by_quality = Counter(r["quality"] for r in rows)
    by_action = Counter(r["label"] for r in rows)
    by_pack = Counter(r["scenario_pack"] for r in rows)
    print(f"  loaded {len(rows)} (trace, label, asset) rows")
    print(f"  quality: {dict(by_quality)}")
    print(f"  actions: {dict(by_action)}")
    print(f"  scenario packs: {dict(by_pack)}")
    if not rows:
        print("  no rows — check assets_dir and trace data")
        return 1

    print("\nLoading encoder...")
    from muzero.tile_encoder import HFVisionTowerEncoder
    t0 = time.perf_counter()
    encoder = HFVisionTowerEncoder(model_id=args.encoder, device=device)
    print(f"  ready in {time.perf_counter()-t0:.1f}s; embed_dim={encoder.embed_dim}")

    print("\nEncoding tiles...")
    embeds = _encode_corpus(rows, encoder, device)
    labels = np.asarray([r["label_idx"] for r in rows], dtype=np.int64)

    print("\nStratified split...")
    train_idx, val_idx = _stratified_split(rows, args.val_frac, args.seed)
    print(f"  train={len(train_idx)}  val={len(val_idx)}")

    print("\nTraining policy head...")
    head, train_stats = _train_head(
        embeds, labels, train_idx, val_idx,
        hidden=args.hidden, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, weight_decay=args.weight_decay,
        device=device, seed=args.seed,
    )

    val_pred = np.asarray(train_stats["val_pred"])
    val_true = np.asarray(train_stats["val_true"])
    per_action = {}
    for i, name in enumerate(ACTIONS):
        mask = val_true == i
        if mask.sum() == 0:
            continue
        per_action[name] = {
            "n": int(mask.sum()),
            "correct": int((val_pred[mask] == val_true[mask]).sum()),
            "acc": float((val_pred[mask] == val_true[mask]).mean()),
        }
    print(f"\nBest val acc: {train_stats['best_val_acc']:.3f} (epoch {train_stats['best_val_epoch']+1})")
    print("Per-action val breakdown:")
    for name, stats in per_action.items():
        print(f"  {name:8s}  {stats['correct']:>2d}/{stats['n']:<2d}  acc={stats['acc']:.3f}")

    # Save checkpoint
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "policy_head.pt"
    torch.save({
        "state_dict": head.state_dict(),
        "embed_dim": int(encoder.embed_dim),
        "hidden": args.hidden,
        "n_actions": len(ACTIONS),
        "actions": list(ACTIONS),
        "encoder_model_id": args.encoder,
    }, ckpt_path)
    print(f"\nSaved policy head: {ckpt_path}")

    summary = {
        "encoder_model_id": args.encoder,
        "embed_dim": int(encoder.embed_dim),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "best_val_acc": train_stats["best_val_acc"],
        "best_val_epoch": train_stats["best_val_epoch"],
        "final_val_acc": train_stats["final_val_acc"],
        "per_action_val": per_action,
        "actions": list(ACTIONS),
        "label_quality_breakdown": dict(by_quality),
        "scenario_pack_breakdown": dict(by_pack),
        "hyperparams": {
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden": args.hidden,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "val_frac": args.val_frac,
            "seed": args.seed,
        },
    }
    summary_path = args.out_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
