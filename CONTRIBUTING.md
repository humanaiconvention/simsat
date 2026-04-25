# Contributing to SimSat

Thanks for plugging in. This file is the short version. For the architectural
context, read [`COLLABORATOR_GUIDE.md`](./COLLABORATOR_GUIDE.md) (model integration)
and [`README.md`](./README.md) (overall project shape).

---

## First clone

```bash
git clone <repo-url>
cd SimSat
cp .env.example .env                       # fill in keys you actually have
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/quickstart.py               # 33 tests + eval against pinned cases
```

If `quickstart.py` exits 0, your clone is wired correctly.

---

## Branches

- `main` / `master` is the integration branch. Never push directly.
- Work on a topic branch: `feat/<short-slug>`, `fix/<short-slug>`, or
  `docs/<short-slug>`. Collaborator integration branches: `collab/<your-name>`.
- Open a PR back to the integration branch; CI must be green before merge.

---

## Commits

Single-purpose commits, conventional style. Match the existing log:

```
feat(observation_vla): add Genesis backend wiring
fix(encounter): null guard in materialize_top_k
docs(collaborator-guide): clarify JSON schema field bounds
test(viability): cover error_bias gate boundary
```

Keep regenerated markdown reports (`SUBMISSION_*.md`, `OBSERVATION_VLA_*.md`,
`REVIEW_*.md`, `CHALLENGE_*.md`) out of unrelated PRs — they cause large diffs.
Use `git checkout -- <file>` to discard regen noise before staging.

---

## Tests

```bash
python -m pytest src/sim/tests/ -v          # unit + smoke (33 tests, ~7s)
python scripts/quickstart.py                 # smoke + eval roundtrip
python scripts/observation_vla_eval.py --inprocess   # eval-only against pins
```

CI runs `pytest` on every push and PR (Python 3.11 + 3.13, ubuntu-latest).
A second CI job greps for hardcoded `D:/SimSat` or `C:/Users/benja` paths and
fails the build if any appear in source. Use:

```python
REPO_ROOT = Path(__file__).resolve().parents[N]   # N depends on file depth
```

---

## Adding a new ObservationVLA backend

The integration surface is one JSON contract — see `COLLABORATOR_GUIDE.md`.
Minimal steps:

1. Drop your adapter at `src/sim/observation_vla/<your_backend>_local.py`,
   subclassing `TransformersVLMAdapter` if you can.
2. Register it in `src/sim/observation_vla/backend_factory.py`.
3. Add an `OBSERVATION_VLA_BACKEND=<your_backend>` block in `.env.example`.
4. Verify with: `OBSERVATION_VLA_BACKEND=<your_backend> python scripts/observation_vla_eval.py --inprocess`
5. Compare action-agreement and MAE numbers against the `clip_local` baseline.

---

## What goes where

| Directory | Purpose |
|---|---|
| `src/sim/` | Runtime: API, encounter planner, ObservationVLA, mission response, viability gates |
| `scripts/` | Operational tooling — eval, casebook builders, review workflows, quickstart |
| `tests/` | Integration tests (live API path, manifest, regression fixtures) |
| `src/sim/tests/` | Unit + smoke tests (no network, no GPU) |
| `notebooks/` | Kaggle T4 fine-tuning notebooks |
| `data/` | Tracked fixture JSON (traces, outcomes, pinned cases). Do not edit by hand. |
| `weights/` | Local model weights (gitignored). Set paths via `.env`. |
| `exports/` | Generated training data (gitignored). Regenerate via `tools/export_training_signal.py`. |

---

## Safety / scope

- **No deletes.** Move to `archive/` or mark archived. Exception: build artifacts.
- **Never commit `.env`, model weights, or anything in `weights/`.**
- **Submission markdown files are tracked** — they're how a collaborator sees
  the latest pinned-case state without running the full eval. Don't gitignore.
- **The 4 pinned reviewed cases are the integration target.** If your change
  causes their action recommendations or usefulness scores to shift, call it
  out in the PR description.

---

## Questions

Ben Haslam — benjamin.haslam@gmail.com
