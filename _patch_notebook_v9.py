"""Patch notebook to v9: Fix #20 — comma escaped inside Fix #19 comment.

v8 (which contained Fix #19's correct response_template) crashed at
`SyntaxError: invalid syntax. Perhaps you forgot a comma?` on cell 7.

The Fix #19 patch script wrote:

    response_template="<|turn>model\\n"  # Fix #19: ... not <start_of_turn>,
    tokenizer=tokenizer,

The comma that USED to separate response_template= from tokenizer= ended
up trapped INSIDE the inline comment (after `<start_of_turn>`) instead of
between the two kwargs. Python saw two adjacent kwarg assignments with no
comma, hence the SyntaxError.

Fix #20 (v9): place the comma BEFORE the inline comment.

    response_template="<|turn>model\\n",  # Fix #19: ... not <start_of_turn>

Also bump version strings v8 -> v9 throughout. The actual Fix #19 (correct
response_template content) stays exactly as v8 had it — Fix #20 is purely
the syntactic correction.
"""
import json, pathlib, sys

nb_path = pathlib.Path("notebooks/kaggle-simsat-gemma4-v1/notebook.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

# Force the canonical line via raw string (avoids escape gymnastics).
correct_line = r'    response_template="<|turn>model\n",  # Fix #19: Gemma-4 uses <|turn> special token (id 105), not <start_of_turn>'
target = correct_line + "\n"

fixed = False
for c in nb["cells"]:
    if c.get("cell_type") != "code" or not isinstance(c["source"], list):
        continue
    for i, line in enumerate(c["source"]):
        if "response_template=" in line and ("turn>model" in line or "start_of_turn>model" in line):
            c["source"][i] = target
            fixed = True
            break
    if fixed:
        break

if not fixed:
    sys.exit("ERROR: response_template line not found")

# Bump v8 -> v9 / Fix #19 -> Fix #20 in description, version-string sweep.
for c in nb["cells"]:
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    new_src = src
    new_src = new_src.replace("SimSat Gemma-4-E2B v8", "SimSat Gemma-4-E2B v9 (Fix #20)")
    new_src = new_src.replace("SIMSAT GEMMA-4-E2B v8 COMPLETE", "SIMSAT GEMMA-4-E2B v9 COMPLETE")
    new_src = new_src.replace('"version": "simsat-gemma4-v8"', '"version": "simsat-gemma4-v9"')
    new_src = new_src.replace("simsat-gemma4-v8-adapter", "simsat-gemma4-v9-adapter")
    new_src = new_src.replace("simsat_gemma4_v8_summary.json", "simsat_gemma4_v9_summary.json")
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook written.")

# Verify: cell containing response_template must compile cleanly.
nb2 = json.loads(nb_path.read_text(encoding="utf-8"))
for i, c in enumerate(nb2["cells"]):
    if c.get("cell_type") != "code" or not isinstance(c["source"], list):
        continue
    src = "".join(c["source"])
    if "response_template=" in src and "turn>model" in src:
        try:
            compile(src, f"<cell {i}>", "exec")
            print(f"  OK cell {i} compiles")
        except SyntaxError as e:
            print(f"  X  cell {i} SYNTAX ERROR at line {e.lineno}: {e.msg}")
            sys.exit(1)
        break

# Verify version-string sweep
all_src = "\n".join(
    ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
    for c in nb2["cells"]
)
checks = [
    ("v9 in title", "v9" in all_src),
    ("Fix #20 marker", "Fix #20" in all_src),
    ("v9 adapter dir", "simsat-gemma4-v9-adapter" in all_src),
    ("v9 summary file", "simsat_gemma4_v9_summary.json" in all_src),
    ("v8 adapter dir gone", "simsat-gemma4-v8-adapter" not in all_src),
    ("response_template still has Fix #19 fix",
     '"<|turn>model\\n"' in all_src),
]
passed = sum(1 for _, ok in checks if ok)
for label, ok in checks:
    print(f"  {'OK' if ok else 'X '} {label}")
print(f"{passed}/{len(checks)} checks passed")
sys.exit(0 if passed == len(checks) else 1)
