# Morning Git Push Guide

Updated: 2026-05-05 (autonomous session)

## IMMEDIATE: Push v12 (commit already made locally)

```bash
cd D:\SimSat
git push fork collab/heatmap-backend
```

Then run v12 on Kaggle (same attachments as v11):
```bash
cd notebooks/kaggle-simsat-gemma4-v1
python push.py --kernel-only
```

Watch for `LoRA tensor coverage: 490/490 (PASS)` in Cell 7 output.

### Local commits not yet pushed:
- `ea56234` — feat: v12 Gemma-4 fine-tune — fix GQA k/v LoRA partial save
- `afbbece` — docs: LFM track benchmarks  
- `de9b7ee` — chore: overnight benchmark results

---

## Earlier notes (from overnight runs)

**Pre-push checklist**

1. **Close VSCode and GitHub Desktop** before running any git commands (they held the index.lock earlier — it's gone now but be safe).

2. **Check the Kaggle notebook** — if the float16 N=100 benchmark or scenario study is still running, wait for it, then copy the result JSON from `/kaggle/working/` and save it to `benchmark_results/`.

## Files to commit tonight

### New directory (untracked — add explicitly):
```
benchmark_results/
```
Contains:
- `gemma4_e2b_t4_benchmark.json` — float16 N=20 baseline (verified on 2 T4s)
- `gemma4_e2b_t4_float16_n100_benchmark.json` — **copy from Kaggle when run completes**
- `eval_analysis_current.json` — honest backend distribution analysis (CLIP vs stub vs Gemma-4)
- `BENCHMARK_RESULTS.md` — formatted benchmark table + scenario study results (update pending lines when runs complete)
- `kaggle_scenario_assessment_study.py` — 90-scenario Gemma-4 study script
- `kaggle_float16_n100_benchmark.py` — float16 N=100 benchmark script

### Modified (already tracked):
- `SUBMISSION_BRIEF.md` — added Inference Benchmarks section, backend honesty note, scenario study pointer

## Benchmark results to fill in before pushing

### Float16 N=100 — from Kaggle `/kaggle/working/simsat_vla_benchmark_float16_n100.json`:
Update `BENCHMARK_RESULTS.md` line:
```
| float16 | 100 | pending | pending | pending | pending |
```
Replace with real numbers.

Also update `SUBMISSION_BRIEF.md` table line:
```
| float16 | 100 | pending (running) | — | — | — |
```

### Scenario assessment study — from Kaggle `/kaggle/working/scenario_assessment_summary.json`:
Update `BENCHMARK_RESULTS.md` scenario assessment section with real numbers.

## Suggested commit message

```
chore: overnight benchmark runs — float16 N=100, scenario study, honest TTT analysis

- benchmark_results/: new directory with N=100 float16 benchmark (p50/p90/p95/p99/std)
- benchmark_results/eval_analysis_current.json: honest backend breakdown
  (clip_local=55, stub=31, gemma4=0 in runtime DB; CLIP mean usefulness 0.647 vs stub 0.867)
- benchmark_results/BENCHMARK_RESULTS.md: full benchmark + scenario study table
- scenario study: 90 Gemma-4 assessments, 27 geographic locations, 4 cloud conditions
- SUBMISSION_BRIEF.md: Inference Benchmarks section + TTT/backend honesty note
- int8 not available: bitsandbytes incompatible with transformers HEAD on Kaggle T4
```

## Git commands

```bash
cd D:\SimSat
git add benchmark_results/
git add SUBMISSION_BRIEF.md
