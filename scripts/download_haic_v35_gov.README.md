# Download HAIC v35-gov weights

Populates `./weights/haic-v35-gov/` from the `benhaslam/haic-gemma4-v35-gov-unsloth` Kaggle kernel output.

## Prerequisites

- Python + `kaggle` CLI: `pip install kaggle`
- Kaggle API token at `~/.kaggle/kaggle.json` (chmod 600 on POSIX)
  - Get one at https://www.kaggle.com/settings → Account → Create New Token
- Access to the kernel: either it's public OR Ben has added you as a collaborator.
  Current state (2026-04-21): Ben has access. Guilherme will get access after local testing passes.

## Run

**POSIX (Linux / macOS / Git Bash):**
```bash
cd <repo root>
bash scripts/download_haic_v35_gov.sh
```

**Windows (PowerShell):**
```powershell
cd <repo root>
pwsh .\scripts\download_haic_v35_gov.ps1
```

## What you get

After running, `./weights/haic-v35-gov/` will contain:

```
weights/haic-v35-gov/
├── _raw_kernel_output/          # Unorganized pull from Kaggle (kept for reference)
├── base/                        # 5-shard merged safetensors Gemma-4 + tokenizer
├── adapter/                     # LoRA adapter for peft-based loading (≤ 5 MB — the satellite-upload claim)
├── gguf/                        # F16 GGUF quantized (for llama.cpp / Orin on-orbit)
└── eval/                        # haic_v35_gov_full_results.json, prism_gemma4_v35_gov.json, trainer_state.json
```

## Expected sizes (rough)

| Subfolder | Size | Purpose |
|---|---|---|
| `base/` | ~5–10 GB | Full fine-tuned model for local dev |
| `adapter/` | 2–50 MB | LoRA adapter only — fits the hackathon's 5 MB satellite upload budget |
| `gguf/` | ~2–6 GB | Quantized F16, CPU-friendly inference |
| `eval/` | <1 MB | Training metrics + PRISM entropy + evaluation results |

**If `adapter/adapter_model.safetensors` is > 5 MB**, the "fits the 5 MB satellite upload budget" pitch for the Liquid Track / General Track space-credits prize doesn't hold. Re-check the LoRA rank in training args; a rank-8 LoRA on Gemma-4 2B should fit comfortably.

## Next step

Once weights are on disk, enable the backend in `docker-compose.yaml`:

```yaml
services:
  sim:
    environment:
      - OBSERVATION_VLA_BACKEND=gemma4_haic_local
      - HAIC_GEMMA4_MODE=lora
      - HAIC_GEMMA4_WEIGHTS_DIR=/app/weights/haic-v35-gov
```

Then follow the verification checklist in `src/sim/observation_vla/README_ENTRY_B.md`.
