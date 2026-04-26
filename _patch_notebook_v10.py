"""Patch notebook to v10: address accept-bias on borderline refine cases.

v9 eval (7 operator-reviewed cases) showed the model picks `accept` too often
on borderline `refine` cases (Rotterdam x2). Root cause: only 2 training epochs
means the model doesn't fully separate the accept/refine boundary.

Two complementary fixes:
  1. Bump num_train_epochs 2 -> 4 (more gradient steps over the same data)
  2. Dataset-level refine upweighting is handled separately in
     datasets/simsat-gemma4-v1/prepare_dataset.py (REFINE_BOOST flag)

Also bumps all version strings v9 -> v10.
"""
import json, pathlib, sys

nb_path = pathlib.Path("notebooks/kaggle-simsat-gemma4-v1/notebook.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

# ── Patch 1: num_train_epochs 2 -> 4 ────────────────────────────────────────
epochs_patched = False
for c in nb["cells"]:
    if c.get("cell_type") != "code" or not isinstance(c["source"], list):
        continue
    for i, line in enumerate(c["source"]):
        if "num_train_epochs=2," in line:
            c["source"][i] = line.replace(
                "num_train_epochs=2,",
                "num_train_epochs=4,   # v10: 4 epochs to sharpen accept/refine boundary",
            )
            epochs_patched = True
            break
    if epochs_patched:
        break

if not epochs_patched:
    sys.exit("ERROR: num_train_epochs=2 line not found — already patched or notebook changed")

# ── Patch 2: version-string sweep v9 -> v10 ─────────────────────────────────
for c in nb["cells"]:
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    new_src = src
    new_src = new_src.replace("SIMSAT GEMMA-4-E2B v9 COMPLETE", "SIMSAT GEMMA-4-E2B v10 COMPLETE")
    new_src = new_src.replace('"version": "simsat-gemma4-v9"', '"version": "simsat-gemma4-v10"')
    new_src = new_src.replace("simsat-gemma4-v9-adapter", "simsat-gemma4-v10-adapter")
    new_src = new_src.replace("simsat_gemma4_v9_summary.json", "simsat_gemma4_v10_summary.json")
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook written.")

# ── Verify ───────────────────────────────────────────────────────────────────
nb2 = json.loads(nb_path.read_text(encoding="utf-8"))
all_src = "\n".join(
    ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
    for c in nb2["cells"]
)

# Compile the cell that contains num_train_epochs to catch syntax errors
for i, c in enumerate(nb2["cells"]):
    if c.get("cell_type") != "code" or not isinstance(c["source"], list):
        continue
    src = "".join(c["source"])
    if "num_train_epochs=4" in src:
        try:
            compile(src, f"<cell {i}>", "exec")
            print(f"  OK cell {i} compiles")
        except SyntaxError as e:
            print(f"  X  cell {i} SYNTAX ERROR at line {e.lineno}: {e.msg}")
            sys.exit(1)
        break

checks = [
    ("epochs bumped to 4", "num_train_epochs=4" in all_src),
    ("epochs=2 gone", "num_train_epochs=2," not in all_src),
    ("v10 adapter dir", "simsat-gemma4-v10-adapter" in all_src),
    ("v10 summary file", "simsat_gemma4_v10_summary.json" in all_src),
    ("v10 version string", '"version": "simsat-gemma4-v10"' in all_src),
    ("v9 adapter dir gone", "simsat-gemma4-v9-adapter" not in all_src),
    ("response_template intact", '"<|turn>model\\n"' in all_src),
]
passed = sum(1 for _, ok in checks if ok)
for label, ok in checks:
    print(f"  {'OK' if ok else 'X '} {label}")
print(f"{passed}/{len(checks)} checks passed")
sys.exit(0 if passed == len(checks) else 1)
