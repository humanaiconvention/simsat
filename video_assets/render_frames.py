#!/usr/bin/env python3
"""Render the 9 demo-video shot frames as 1920x1080 PNG images.

Produces video_assets/frames/shot{1..9}.png. Then build_video.py can stitch
them into video_assets/silent_demo.mp4 with the timing from VIDEO_SCRIPT.md.

Style: dark navy background (#0a0e27), pale cyan text (#a0e5ff), accent
yellow (#fef3c7) for emphasis, soft red (#fee2e2) for warnings.
Monospace font via Cascadia Mono (bundled with Windows Terminal).
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "video_assets"
FRAMES = ASSETS / "frames"
FRAMES.mkdir(exist_ok=True)

W, H = 1920, 1080
BG = (10, 14, 39)              # dark navy
TEXT = (220, 230, 245)         # off-white
ACCENT = (254, 243, 199)       # warm yellow (emphasis)
GREEN = (104, 211, 145)        # success
RED = (252, 129, 129)          # negative result
DIM = (160, 174, 192)          # secondary

MONO = "C:/Windows/Fonts/CascadiaMono.ttf"
MONO_BOLD = "C:/Windows/Fonts/CascadiaCode.ttf"  # Cascadia Code has bold variant


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(MONO_BOLD if bold else MONO, size)


def new_frame(label: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    if label:
        d.text((W - 200, H - 50), label, fill=DIM, font=font(18))
    return img, d


def text_block(d, x, y, lines, sz=28, color=TEXT, bold=False, line_h=None):
    f = font(sz, bold)
    line_h = line_h or int(sz * 1.4)
    for i, line in enumerate(lines):
        d.text((x, y + i * line_h), line, fill=color, font=f)
    return y + len(lines) * line_h


# ---------------- SHOT 1: HumanAI Convention logo on black ----------------
def shot1():
    """Two frames for the cold open:

    - shot1.png       — phi-with-dot mark only, centered, white on pure black.
                        Build_video.py fades this in from black over 2s.
    - shot1_full.png  — same composition, with the website-style "Human AI /
                        Convention" wordmark rendered below the mark in
                        Segoe UI (Inter is the website's preferred face but
                        isn't installed; Segoe UI Variable is the closest
                        Windows-bundled match). Build_video.py crossfades
                        from shot1 to shot1_full at ~4-5s, timed for the
                        moment the founder says 'HumanAI Convention'.
    """
    # Re-render the logo at a smaller scale (was 820) so the full mark
    # (including the bar's bottom terminal) fits with proper breathing
    # room above + below the lockup, and so the mark doesn't get cropped.
    BRAND = ROOT / "video_assets" / "brand" / "logo_video.png"
    LOGO_HEIGHT = 720
    import cairosvg
    cairosvg.svg2png(
        url=str(ROOT / "video_assets" / "brand" / "logo.svg"),
        output_height=LOGO_HEIGHT,
        write_to=str(BRAND),
    )

    # Crop the rendered PNG to just the mark glyph (no wordmark space —
    # cairosvg can't render the Inter wordmark anyway, so the bottom of
    # the SVG is empty pixels). The bar bottom sits at viewBox y≈257,
    # so we crop just past that to viewBox y=263 to keep the full bar
    # plus a hair of margin without including stray empty rows.
    full = Image.open(BRAND).convert("RGBA")
    mark_h = int(full.height * (263.0 / 272.71322893124))
    mark = full.crop((0, 0, full.width, mark_h))

    # Mark + wordmark layout — compute mark position so the FULL lockup
    # (mark + 2 wordmark lines) is vertically centered with a slight
    # upward bias so the mark feels prominent during the cold-open phase
    # before the wordmark crossfades in. The same mark position is used
    # in both frames so the crossfade is purely wordmark-on-vs-off.
    scale = full.height / 272.71322893124
    line1_size = int(39.91544817563599 * scale)   # "Human AI" ~105 px at 720 height
    line2_size = int(25.400739748131986 * scale)  # "Convention" ~67 px
    gap_mark_to_line1 = int(34 * scale)   # generous breathing room (was 20 viewBox units)
    gap_line1_to_line2 = int(10 * scale)
    total_h = mark.height + gap_mark_to_line1 + line1_size + gap_line1_to_line2 + line2_size
    # Center vertically, then bias up by ~30px so the mark sits a touch
    # above center — addresses the "mark feels low / cuts off from text"
    # perception during the mark-only opening seconds.
    mark_top = max(20, (H - total_h) // 2 - 30)
    mark_x = (W - mark.width) // 2

    # Frame 1: mark only at the locked position (so the crossfade is purely
    # wordmark-on-vs-off and doesn't visibly shift the mark).
    canvas1 = Image.new("RGB", (W, H), (0, 0, 0))
    canvas1.paste(mark, (mark_x, mark_top), mark)
    canvas1.save(FRAMES / "shot1.png")

    # Frame 2: mark + wordmark, mark at the same locked position
    canvas2 = Image.new("RGB", (W, H), (0, 0, 0))
    canvas2.paste(mark, (mark_x, mark_top), mark)
    top = mark_top  # alias for the existing wordmark math below

    d2 = ImageDraw.Draw(canvas2)
    SEGOE_LIGHT = "C:/Windows/Fonts/segoeuil.ttf"
    SEGOE_REG = "C:/Windows/Fonts/segoeui.ttf"
    f_human = ImageFont.truetype(SEGOE_LIGHT, line1_size)
    f_ai = ImageFont.truetype(SEGOE_REG, line1_size)
    f_conv = ImageFont.truetype(SEGOE_LIGHT, line2_size)

    # Line 1: "Human" (light) + " AI" (regular), tracked slightly
    line1_y = top + mark.height + gap_mark_to_line1
    human_str = "Human"
    ai_str = " AI"
    human_w = d2.textlength(human_str, font=f_human)
    ai_w = d2.textlength(ai_str, font=f_ai)
    total_w1 = human_w + ai_w
    x1 = (W - total_w1) // 2
    d2.text((x1, line1_y), human_str, fill=(255, 255, 255), font=f_human)
    d2.text((x1 + human_w, line1_y), ai_str, fill=(255, 255, 255), font=f_ai)

    # Line 2: "Convention" — slightly dimmed (matches SVG opacity 0.9)
    line2_y = line1_y + line1_size + gap_line1_to_line2
    conv_str = "Convention"
    conv_w = d2.textlength(conv_str, font=f_conv)
    x2 = (W - conv_w) // 2
    d2.text((x2, line2_y), conv_str, fill=(230, 230, 230), font=f_conv)

    canvas2.save(FRAMES / "shot1_full.png")


# ---------------- SHOT 2: same title, fade Rotterdam in ----------------
def shot2():
    """Title card with Rotterdam Sentinel image as a band across the bottom."""
    img, d = new_frame("SimSat — 2/9")

    # Top half: title
    title = "SimSat"
    f1 = font(120, bold=True)
    f2 = font(36)
    bb = d.textbbox((0, 0), title, font=f1)
    d.text(((W - (bb[2] - bb[0])) / 2, 80), title, fill=ACCENT, font=f1)
    sub = "On-orbit inference, refined by operator-labelled JSON within uplink bandwidth"
    bb = d.textbbox((0, 0), sub, font=f2)
    d.text(((W - (bb[2] - bb[0])) / 2, 240), sub, fill=TEXT, font=f2)

    # Bottom half: Rotterdam image (resized + cropped to a wide band)
    rot_path = ROOT / "submission_assets" / "urban_coastal_ambiguity_port_of_rotterdam.png"
    rot = Image.open(rot_path).convert("RGB")
    # Fit width 1920, crop center to 600 tall
    rw, rh = rot.size
    new_w = W
    new_h = int(rh * (W / rw))
    rot = rot.resize((new_w, new_h), Image.LANCZOS)
    # Crop to 600 from middle
    band_h = 540
    crop_top = (new_h - band_h) // 2
    rot = rot.crop((0, crop_top, W, crop_top + band_h))
    img.paste(rot, (0, 540))

    # Caption over the image (low-opacity dark box + white text)
    cap_band = Image.new("RGBA", (W, 70), (10, 14, 39, 200))
    cap_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    cap_layer.paste(cap_band, (0, H - 70))
    img = Image.alpha_composite(img.convert("RGBA"), cap_layer).convert("RGB")
    d = ImageDraw.Draw(img)
    cap = "Sentinel-2 · Port of Rotterdam · 48.78% cloud · canonical case"
    f3 = font(26)
    bb = d.textbbox((0, 0), cap, font=f3)
    d.text(((W - (bb[2] - bb[0])) / 2, H - 50), cap, fill=TEXT, font=f3)
    img.save(FRAMES / "shot2.png")


# ---------------- SHOT 3: architecture diagram ----------------
def shot3():
    """Center the architecture PNG with a header."""
    img, d = new_frame("SimSat — 3/9 · Architecture")
    arch_path = ROOT / "fig" / "architecture_diagram.png"
    arch = Image.open(arch_path).convert("RGB")
    # Architecture is 1449x1684 (portrait). Scale to fit 1000 height.
    aw, ah = arch.size
    target_h = 1000
    target_w = int(aw * (target_h / ah))
    arch = arch.resize((target_w, target_h), Image.LANCZOS)
    img.paste(arch, ((W - target_w) // 2, 80))
    img.save(FRAMES / "shot3.png")


# ---------------- SHOT 4: v3 holdout table ----------------
def shot4():
    img, d = new_frame("SimSat — 4/9 · v3 Holdout (32 samples, 8 per class)")
    y = 80
    f_h1 = font(48, bold=True)
    f_h2 = font(36)
    d.text((100, y), "LFM2.5-VL-450M v3 — Matched-Pair Holdout", fill=ACCENT, font=f_h1); y += 80

    rows = [
        ("Metric",                  "Base",    "Tuned (v3)", "Delta",   None),
        ("─" * 26,                  "─" * 8,   "─" * 11,     "─" * 8,   None),
        ("parse_rate",              "1.000",   "1.000",      "+0.000",  TEXT),
        ("exact_action_agreement",  "0.156",   "0.844",      "+0.688",  ACCENT),
        ("useful_agreement",        "0.312",   "0.688",      "+0.375",  TEXT),
        ("score_mae (lower better)","0.365",   "0.055",      "−0.310",  ACCENT),
    ]
    f = font(32)
    for col, (m, b, t, dlt, color) in enumerate(rows):
        c = color or TEXT
        d.text((100, y),  m,   fill=c, font=f)
        d.text((780, y),  b,   fill=c, font=f)
        d.text((990, y),  t,   fill=c if c != ACCENT else GREEN, font=f if c != ACCENT else font(32, bold=True))
        d.text((1280, y), dlt, fill=c if c != ACCENT else GREEN, font=f if c != ACCENT else font(32, bold=True))
        y += 50

    y += 30
    d.text((100, y), "Per-class action accuracy (tuned):", fill=DIM, font=f_h2); y += 60

    pcs = [
        ("accept", "1.000",  GREEN),
        ("refine", "1.000",  GREEN),
        ("defer",  "0.625",  TEXT),
        ("skip",   "0.750",  TEXT),
    ]
    for cls, val, color in pcs:
        d.text((140, y), f"{cls:<10}", fill=DIM, font=f)
        d.text((320, y), val,            fill=color, font=font(32, bold=True))
        y += 50

    y += 30
    d.text((100, y), "Adapter: HumanAIConvention/simsat-lfm25vl-450m-v3 · Apache-2.0",
           fill=DIM, font=font(24)); y += 36
    d.text((100, y), "Kernel:  benhaslam/simsat-lfm2-5-vl-v3-training (public)",
           fill=DIM, font=font(24))

    img.save(FRAMES / "shot4.png")


# ---------------- SHOT 5: same as 4 but emphasizing MAE ----------------
def shot5():
    """Same content, with extra callout on score_mae."""
    img, d = new_frame("SimSat — 5/9 · MAE drop")
    y = 80
    f_h1 = font(48, bold=True)
    d.text((100, y), "score_mae: 0.365 → 0.055", fill=ACCENT, font=f_h1); y += 90
    d.text((100, y), "−31.0 percentage points", fill=GREEN, font=f_h1); y += 130

    f = font(32)
    d.text((100, y), "Band-mapped prediction error against operator usefulness scores.",
           fill=TEXT, font=f); y += 60
    d.text((100, y), "Lower is better.",
           fill=DIM, font=f); y += 100

    d.text((100, y), "Per-class breakdown:",
           fill=DIM, font=font(36, bold=True)); y += 70
    pcs = [
        ("accept",  "1.000  perfect"),
        ("refine",  "1.000  perfect"),
        ("defer",   "0.625"),
        ("skip",    "0.750"),
    ]
    for cls, val in pcs:
        col = GREEN if "perfect" in val else TEXT
        d.text((140, y), f"{cls:<10}{val}", fill=col, font=font(34, bold="perfect" in val)); y += 56

    y += 80
    d.text((100, y), "Adapter live: huggingface.co/HumanAIConvention/simsat-lfm25vl-450m-v3",
           fill=DIM, font=font(26))

    img.save(FRAMES / "shot5.png")


# ---------------- SHOT 6: Rotterdam case ----------------
def shot6():
    img, d = new_frame("SimSat — 6/9 · Rotterdam · trace_4f65355f...")

    # Left half: Sentinel image
    rot_path = ROOT / "submission_assets" / "urban_coastal_ambiguity_port_of_rotterdam.png"
    rot = Image.open(rot_path).convert("RGB")
    target_w, target_h = 900, 900
    rw, rh = rot.size
    scale = min(target_w / rw, target_h / rh)
    rot = rot.resize((int(rw * scale), int(rh * scale)), Image.LANCZOS)
    img.paste(rot, (60, (H - rot.height) // 2))

    # Right half: case panel
    x = 1020
    y = 80
    f_h1 = font(48, bold=True)
    d.text((x, y), "Rotterdam Case", fill=ACCENT, font=f_h1); y += 80

    f = font(28)
    f_b = font(28, bold=True)
    rows = [
        ("Target:",        "Port of Rotterdam"),
        ("Sentinel:",      "sentinel-2c"),
        ("Cloud cover:",   "48.78%"),
        ("Reviewer:",      "ben"),
    ]
    for label, val in rows:
        d.text((x, y), label, fill=DIM, font=f)
        d.text((x + 220, y), val, fill=TEXT, font=f); y += 42

    y += 60
    d.text((x, y), "calibration loop:", fill=DIM, font=font(28, bold=True)); y += 60

    # Three boxes: scaffold, trust, operator
    box_w, box_h = 200, 100
    bx = x
    labels = [("scaffold", "ACCEPT", GREEN), ("trust", "REFINE", ACCENT), ("operator", "ACCEPT", GREEN)]
    for i, (lbl, val, color) in enumerate(labels):
        bx2 = x + i * (box_w + 60)
        d.rectangle([bx2, y, bx2 + box_w, y + box_h], outline=color, width=3)
        d.text((bx2 + 12, y + 12), lbl, fill=DIM, font=font(20))
        bb = d.textbbox((0, 0), val, font=font(36, bold=True))
        d.text((bx2 + (box_w - (bb[2] - bb[0])) / 2, y + 45), val, fill=color, font=font(36, bold=True))
        if i < 2:
            d.text((bx2 + box_w + 14, y + 32), "→", fill=DIM, font=font(48, bold=True))
    y += box_h + 60

    f2 = font(24)
    d.text((x, y), "Visible 51% of the basin was operationally enough.", fill=TEXT, font=f2); y += 40
    d.text((x, y), "Operator override = signal for trust-layer TTT to retune.", fill=DIM, font=f2); y += 40
    y += 30
    d.text((x, y), "This is the on-orbit calibration loop, end to end.", fill=ACCENT, font=font(26, bold=True))

    img.save(FRAMES / "shot6.png")


# ---------------- SHOT 7: TTT — two-class lift (HEADLINE) ----------------
def shot7():
    img, d = new_frame("SimSat — 7/9 · TTT empirically lifts two target classes")
    y = 50
    d.text((100, y), "VLA-layer TTT — two-class lift on v3", fill=ACCENT, font=font(40, bold=True)); y += 65

    f = font(24)
    d.text((100, y), "16 class-only train rows streamed → 8 class-only holdout probe.",
           fill=TEXT, font=f); y += 36
    d.text((100, y), "Action-token-weighted CE loss (mask all but recommended_action).",
           fill=DIM, font=font(20)); y += 50

    rows = [
        ("Class",          "PRE",     "POST",    "Δ",          "predictions (post)"),
        ("─" * 8,          "─" * 6,   "─" * 6,   "─" * 8,      "─" * 26),
        ("skip",           "0.375",   "0.750",   "+0.375",     "6 skip + 2 accept"),
        ("defer",          "0.125",   "0.875",   "+0.750",     "7 defer + 1 accept"),
        ("",               "",        "",        "",           ""),
        ("AVERAGE LIFT",   "",        "",        "+0.5625",    "across 2 classes"),
    ]
    for col_i, (m, pre, post, dlt, preds) in enumerate(rows):
        if not m and not pre and not post and not dlt:
            y += 14
            continue
        is_data = col_i >= 2
        c = TEXT
        if dlt and dlt.startswith("+"):
            c = GREEN
        if "AVERAGE" in m:
            c = ACCENT
        d.text((100, y),  m,    fill=c if is_data else DIM, font=font(24, bold=("AVERAGE" in m)))
        d.text((420, y),  pre,  fill=c if is_data else TEXT, font=f)
        d.text((620, y),  post, fill=c if is_data else TEXT, font=font(24, bold=(c == GREEN)))
        d.text((820, y),  dlt,  fill=c if is_data else TEXT, font=font(24, bold=(c == GREEN or "AVERAGE" in m)))
        d.text((1100, y), preds, fill=c if is_data else DIM, font=font(22))
        y += 36

    y += 20
    d.text((100, y), "Loss in both runs: ~1.0 → ~0 over 16 steps (action-token CE drove to zero)",
           fill=TEXT, font=font(22)); y += 32
    d.text((100, y), "lora_delta_l2: 0.008 → 0.04 (monotonic, real weight movement, both runs)",
           fill=TEXT, font=font(22)); y += 50

    d.text((100, y), "TTT under operator-curated stream + action-weighted loss",
           fill=ACCENT, font=font(28, bold=True)); y += 38
    d.text((100, y), "empirically LIFTS target-class accuracy per pass.",
           fill=ACCENT, font=font(28, bold=True)); y += 38
    d.text((100, y), "Validated on TWO independent classes on a real LFM2.5-VL checkpoint.",
           fill=ACCENT, font=font(24, bold=True)); y += 60

    d.text((100, y), "Backed by 5/30/50-step stability receipts (no divergence, parse 1.0 throughout).",
           fill=DIM, font=font(20))

    img.save(FRAMES / "shot7.png")


# ---------------- SHOT 8: v4 + v5 + v3+ negative results ----------------
def shot8():
    img, d = new_frame("SimSat — 8/9 · three honest negatives")
    y = 60
    d.text((100, y), "v4 + v5 + v3+ — three published negatives", fill=ACCENT, font=font(40, bold=True)); y += 65

    f = font(24)
    fb = font(24, bold=True)
    d.text((100, y), "Three independent lines (data-imbalanced, data-balanced, recipe) — same 32-row holdout.",
           fill=TEXT, font=f); y += 50

    rows = [
        ("Metric",                  "v3",     "v4",     "v5",     "v3+",    "vs v3"),
        ("─" * 26,                  "─" * 6,  "─" * 6,  "─" * 6,  "─" * 6,  "─" * 6),
        ("exact_action_agreement",  "0.844",  "0.781",  "0.688",  "0.812",  "all <"),
        ("score_mae (lower better)","0.055",  "0.066",  "0.084",  "0.067",  "all >"),
        ("useful_agreement",        "0.688",  "0.656",  "0.812",  "0.844",  "v5/v3+ >"),
        ("",                        "",       "",       "",       "",       ""),
        ("accept per-class",        "1.000",  "1.000",  "1.000",  "1.000",  "saturated"),
        ("refine per-class",        "1.000",  "1.000",  "0.875",  "1.000",  "stable"),
        ("defer per-class",         "0.625",  "0.500",  "0.625",  "0.750",  "v3+ best"),
        ("skip per-class",          "0.750",  "0.625",  "0.250",  "0.500",  "v3 best"),
    ]
    for col_i, (m, v3, v4, v5, v3p, dlt) in enumerate(rows):
        if not m and not v3:
            y += 14
            continue
        is_data = col_i >= 2
        c = TEXT
        bad = is_data and ("all <" in dlt or "all >" in dlt or "v3 best" in dlt)
        good = is_data and ("v5/v3+ >" in dlt or "v3+ best" in dlt)
        if bad: c = RED
        if good: c = GREEN
        d.text((100, y),  m,   fill=c if is_data else TEXT, font=f)
        d.text((720, y),  v3,  fill=c, font=fb if is_data else f)
        d.text((860, y),  v4,  fill=c, font=f)
        d.text((1000, y), v5,  fill=c, font=f)
        d.text((1140, y), v3p, fill=c, font=f)
        d.text((1280, y), dlt, fill=c, font=fb if (bad or good) else f)
        y += 38

    y += 20
    d.text((100, y), "Three independent angles, same conclusion: v3 is at a local optimum.",
           fill=ACCENT, font=font(26, bold=True)); y += 40
    d.text((100, y), "v4: more imbalanced data → −6.3 pp action.",
           fill=DIM, font=font(22)); y += 32
    d.text((100, y), "v5: more class-balanced data → −15.6 pp action; skip cratered.",
           fill=DIM, font=font(22)); y += 32
    d.text((100, y), "v3+: recipe variant (lr↓, epochs↑, dropout↑) → −3.1 pp action.",
           fill=DIM, font=font(22)); y += 50
    d.text((100, y), "Empirical evidence — not just claim — for the runtime-TTT lane.",
           fill=ACCENT, font=font(28, bold=True)); y += 50
    d.text((100, y), "Decision: v3 retained canonical. None of v4/v5/v3+ promoted.",
           fill=ACCENT, font=font(24, bold=True))

    img.save(FRAMES / "shot8.png")


# ---------------- SHOT 9: close ----------------
def shot9():
    img, d = new_frame("SimSat — 9/9")
    y = 60
    d.text((100, y), "humanaiconvention.com",
           fill=ACCENT, font=font(72, bold=True)); y += 100
    d.text((100, y), "SimSat — On-orbit inference, refined by operator-labelled JSON",
           fill=TEXT, font=font(28)); y += 70

    f = font(26)
    fb = font(26, bold=True)
    rows = [
        ("Code",     "github.com/HumanAIConvention/SimSat",            "AGPL-3"),
        ("",         "",                                                ""),
        ("Weights",  "huggingface.co/HumanAIConvention/",              "Apache-2.0"),
        ("",         "  simsat-lfm25vl-450m-v3   ← Liquid Track",      ""),
        ("",         "  simsat-lfm25vl-450m-v1   reference",           ""),
        ("",         "  simsat-gemma4-v11        ← General AI Track",  ""),
        ("",         "",                                                ""),
        ("Kernels",  "kaggle.com/code/benhaslam (public)",             ""),
        ("Datasets", "kaggle.com/datasets/benhaslam (public)",         ""),
    ]
    for label, val, license in rows:
        if label:
            d.text((100, y), label, fill=DIM, font=fb)
        d.text((280, y), val, fill=TEXT, font=f)
        if license:
            d.text((1500, y), license, fill=ACCENT, font=fb)
        y += 40

    y += 50
    d.text((100, y), "Built for the prize hardware — Orin 16 GB.",
           fill=ACCENT, font=font(32, bold=True)); y += 56
    d.text((100, y), "Thanks.", fill=TEXT, font=font(44, bold=True))

    img.save(FRAMES / "shot9.png")


def main():
    funcs = [shot1, shot2, shot3, shot4, shot5, shot6, shot7, shot8, shot9]
    for i, fn in enumerate(funcs, 1):
        fn()
        print(f"  rendered shot{i}.png")
    print(f"\nAll 9 frames -> {FRAMES.absolute()}")


if __name__ == "__main__":
    main()
