#!/usr/bin/env python3
"""Liquid Track — Stage 2 pretrain (scenario-pack augmentation + class balancing).

Stage 1 (`scripts/muzero_stage1_pretrain.py`) trained a BC policy head on
75 (tile, action) rows but the corpus is 64% refine, 23% accept, 13%
defer+skip. The trained head reaches val_acc 0.80 in supervised terms but
argmax-collapses to refine on the full eval (see MUZERO_LFM_EVAL.md
"class collapse on borderline cases").

Stage 2 widens the BC head's decision margins by:
  1. Visual augmentation of every (tile, action) pair — horizontal flip,
     small rotations, brightness/contrast jitter, color jitter. Minority
     classes get more augmentation passes than majority so the dataset
     ends class-balanced.
  2. Class-balanced cross-entropy loss as a backstop in case augmentation
     under-balances any class.

Pipeline:
  PNG tile (+ K augmentations) → LFM2.5-VL-450M vision tower → 768-dim
        → MLP (768 → 256 → 4) → action softmax over
            {accept, defer, refine, skip}

Architecture is identical to Stage 1 so the trained checkpoint is a
drop-in replacement: pass `--policy-head weights/muzero/stage2_bc/policy_head.pt`
to `muzero_lfm_eval.py --policy bc`.

Usage:
    python scripts/muzero_stage2_pretrain.py
    python scripts/muzero_stage2_pretrain.py --aug-per-minority 8 --epochs 60
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
    """Same loader as Stage 1: (trace_id, label, asset_path) per labelled row."""
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
        matches = list(assets_dir.glob(f"*{tid}*.png"))
        if not matches:
            continue
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


def _augment_image(img, mode_seed: int):
    """Return a deterministic-by-seed augmentation of a PIL RGB image.

    Combines safe transforms that preserve action-label semantics:
    horizontal flip, small rotation, brightness/contrast/saturation jitter.
    Crops are intentionally avoided — they can move the target out of frame.
    """
    from PIL import Image, ImageEnhance

    rng = np.random.default_rng(mode_seed)
    out = img
    if rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    angle = float(rng.uniform(-12, 12))
    if abs(angle) > 0.1:
        out = out.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    # Multiplicative jitter: 0.85x..1.15x for brightness/contrast/saturation.
    for enh_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        out = enh_cls(out).enhance(float(rng.uniform(0.85, 1.15)))
    return out


def _build_augmented_dataset(
    rows: list[dict],
    encoder,
    *,
    target_per_class: int,
    max_aug_ratio: float,
    seed: int,
    defer_cap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Encode original tile + K augmentations per row.

    Each class is amplified up to `target_per_class` examples, but
    `max_aug_ratio` caps how many augmentations a class can produce
    relative to its original count. This prevents over-amplifying classes
    with very few originals (which produces synthetic features that don't
    generalize — the Stage 2-v1 lesson, where 7 skip originals were
    augmented ~7x and skip got over-predicted in eval).

    Effective target per class = min(target_per_class, ceil(originals * (1 + max_aug_ratio))).

    Returns (embeddings, labels, per_row_meta).
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    by_class: dict[int, list[dict]] = {}
    for row in rows:
        by_class.setdefault(row["label_idx"], []).append(row)
    print(f"  pre-augmentation per-class: {{ {', '.join(f'{ACTIONS[k]}: {len(v)}' for k, v in sorted(by_class.items()))} }}")
    aug_plan: list[dict] = []  # one entry per encoded sample
    effective_targets: dict[int, int] = {}
    for label_idx, class_rows in by_class.items():
        if not class_rows:
            continue
        # Apply per-class cap on augmentation
        cap = int(np.ceil(len(class_rows) * (1.0 + max_aug_ratio)))
        effective = min(target_per_class, cap)
        if defer_cap is not None and label_idx == ACTION_TO_IDX["defer"]:
            effective = min(effective, defer_cap)
        effective_targets[label_idx] = effective
        # First include each original row exactly once.
        for r in class_rows:
            aug_plan.append({"row": r, "aug_seed": -1, "is_aug": False})
        needed = max(0, effective - len(class_rows))
        if needed == 0:
            continue
        seeds = rng.integers(0, 2**31 - 1, size=needed)
        for i, s in enumerate(seeds):
            r = class_rows[i % len(class_rows)]
            aug_plan.append({"row": r, "aug_seed": int(s), "is_aug": True})

    print(f"  effective per-class targets (cap-applied): {{ {', '.join(f'{ACTIONS[k]}: {effective_targets[k]}' for k in sorted(effective_targets))} }}")
    print(f"  augmented dataset size: {len(aug_plan)} (max_aug_ratio={max_aug_ratio})")

    embeds = np.zeros((len(aug_plan), encoder.embed_dim), dtype=np.float32)
    labels = np.zeros(len(aug_plan), dtype=np.int64)
    t0 = time.perf_counter()
    for i, item in enumerate(aug_plan):
        row = item["row"]
        img = Image.open(row["asset"]).convert("RGB")
        if item["is_aug"]:
            img = _augment_image(img, item["aug_seed"])
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        embeds[i] = encoder.encode(arr)
        labels[i] = row["label_idx"]
        if (i + 1) % 25 == 0 or i == len(aug_plan) - 1:
            print(f"  encoded {i+1}/{len(aug_plan)}  elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return embeds, labels, aug_plan


def _stratified_split(labels: np.ndarray, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for i, lab in enumerate(labels.tolist()):
        by_class.setdefault(lab, []).append(i)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for indices in by_class.values():
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
    use_class_weights: bool,
    defer_weight: float = 1.0,
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

    # Inverse-frequency class weights (acts as a backstop in case
    # augmentation under-balances any class).
    if use_class_weights:
        train_cls_count = torch.bincount(yt, minlength=n_actions).clamp(min=1)
        cls_weights = (train_cls_count.float().mean() / train_cls_count.float()).to(device)
        if defer_weight != 1.0:
            defer_idx = ACTIONS.index("defer")
            cls_weights[defer_idx] *= defer_weight
    else:
        cls_weights = None
    print(f"  class weights: {cls_weights.tolist() if cls_weights is not None else 'off'} (defer_weight={defer_weight})")

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=cls_weights)

    history: list[dict] = []
    best_val = (0.0, -1)
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
    parser.add_argument("--encoder", default="LiquidAI/LFM2.5-VL-450M")
    parser.add_argument("--device", default=None)
    parser.add_argument("--traces", type=Path,
                        default=REPO_ROOT / "src/sim/data/observation_vla/traces.json")
    parser.add_argument("--outcomes", type=Path,
                        default=REPO_ROOT / "src/sim/data/observation_vla/outcomes.json")
    parser.add_argument("--assets-dir", type=Path,
                        default=REPO_ROOT / "review_queue_assets")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "weights/muzero/stage2_bc")
    parser.add_argument("--target-per-class", type=int, default=48,
                        help="Upper bound on per-class samples after augmentation.")
    parser.add_argument("--max-aug-ratio", type=float, default=4.0,
                        help=("Per-class cap: a class with N originals "
                              "produces at most N*(1+max_aug_ratio) total "
                              "examples. Default 4.0 means a class with 7 "
                              "originals tops out at 35, not 48 — a modest "
                              "cap on over-amplification of minority classes. "
                              "Set higher (e.g. 999) to disable the cap. The "
                              "right fix for too-few minority originals is "
                              "more real reviews via build_defer_queue.py + "
                              "batch_review.py, not larger ratios here."))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable inverse-frequency class weighting in CE loss.")
    parser.add_argument("--defer-weight", type=float, default=1.0,
                        help="Multiplier applied to the defer class loss weight after "
                             "inverse-frequency calculation. <1.0 penalises over-prediction "
                             "of defer (e.g. 0.5 halves defer's contribution to the loss). "
                             "Default 1.0 = no adjustment.")
    parser.add_argument("--defer-cap", type=int, default=None,
                        help="Hard ceiling on augmented defer examples, applied after "
                             "max_aug_ratio. E.g. --defer-cap 24 keeps defer at most 24 "
                             "examples even if target_per_class=48. Reduces defer's training "
                             "prior without changing other class targets. Default: no cap.")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder}")
    print()

    print("Loading corpus...")
    rows = _load_corpus(args.traces, args.outcomes, args.assets_dir)
    by_action = Counter(r["label"] for r in rows)
    print(f"  {len(rows)} (trace, label, asset) rows")
    print(f"  per-action: {dict(by_action)}")
    if not rows:
        return 1

    print("\nLoading encoder...")
    from muzero.tile_encoder import HFVisionTowerEncoder
    t0 = time.perf_counter()
    encoder = HFVisionTowerEncoder(model_id=args.encoder, device=device)
    print(f"  ready in {time.perf_counter()-t0:.1f}s; embed_dim={encoder.embed_dim}")

    print("\nBuilding augmented dataset...")
    embeds, labels, aug_plan = _build_augmented_dataset(
        rows, encoder,
        target_per_class=args.target_per_class,
        max_aug_ratio=args.max_aug_ratio,
        seed=args.seed,
        defer_cap=args.defer_cap,
    )
    final_per_class = Counter(int(l) for l in labels)
    print(f"  post-aug per-class: {{ {', '.join(f'{ACTIONS[k]}: {final_per_class[k]}' for k in sorted(final_per_class))} }}")
    n_aug = sum(1 for item in aug_plan if item["is_aug"])
    print(f"  originals: {len(aug_plan) - n_aug}  augmentations: {n_aug}")

    print("\nStratified split...")
    train_idx, val_idx = _stratified_split(labels, args.val_frac, args.seed)
    print(f"  train={len(train_idx)}  val={len(val_idx)}")

    print("\nTraining policy head...")
    head, train_stats = _train_head(
        embeds, labels, train_idx, val_idx,
        hidden=args.hidden, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, weight_decay=args.weight_decay,
        device=device, seed=args.seed,
        use_class_weights=not args.no_class_weights,
        defer_weight=args.defer_weight,
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
        print(f"  {name:8s}  {stats['correct']:>3d}/{stats['n']:<3d}  acc={stats['acc']:.3f}")

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
        "stage": 2,
        "encoder_model_id": args.encoder,
        "embed_dim": int(encoder.embed_dim),
        "n_originals": len(aug_plan) - n_aug,
        "n_augmentations": n_aug,
        "target_per_class": args.target_per_class,
        "max_aug_ratio": args.max_aug_ratio,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "best_val_acc": train_stats["best_val_acc"],
        "best_val_epoch": train_stats["best_val_epoch"],
        "final_val_acc": train_stats["final_val_acc"],
        "per_action_val": per_action,
        "actions": list(ACTIONS),
        "use_class_weights": not args.no_class_weights,
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
