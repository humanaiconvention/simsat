# CONTRIBUTING.md

## Branches

- `main` — stable, always CI-green. Do not push directly.
- `collab/<name>` — collaborator feature branches (e.g. `collab/genesis-backend`, `collab/heatmap-backend`). Open PRs from here to `main`.
- Hotfixes: branch off `main`, name `fix/<short-description>`, PR back to `main`.

## Before opening a PR

1. Run the smoke path:
   ```bash
   pip install -r requirements.txt
   python scripts/quickstart.py
   ```
   Exit 0 is required. This runs 33 tests and the eval against 5 pinned reviewed cases.

2. Run the full test suite:
   ```bash
   pytest tests/ -x
   ```

3. If your change touches `observation_vla/` or any eval script, run:
   ```bash
   python scripts/observation_vla_eval.py --inprocess
   ```
   Confirm action-agreement does not regress from the numbers in `OBSERVATION_VLA_EVAL.md`.

## CI

GitHub Actions runs `.github/workflows/tests.yml` on every push and PR. It tests against Python 3.11 and 3.13. Heavy GPU packages (torch, torchvision, cartopy) are excluded from CI; the test suite is designed to run without them.

A green CI badge is required before merge.

## PR checklist

- [ ] `quickstart.py` exits 0
- [ ] `pytest tests/ -x` passes
- [ ] No hardcoded paths or credentials introduced
- [ ] `.env.example` updated if new env vars added
- [ ] `KNOWN_ISSUES.md` updated if a known issue is resolved or introduced

## File hygiene

- Do not commit `.env` (gitignored).
- Do not commit model weights or large data files — point to Kaggle datasets or HuggingFace repos instead.
- Generated files (`SUBMISSION_PACKET.md`, `OBSERVATION_VLA_EVAL.md`) are tracked in git so judges have a snapshot; regenerate them with the scripts before committing if your change affects their content.

## Questions

Open an issue or ping in the hackathon channel. For model integration questions see `COLLABORATOR_GUIDE.md`.
