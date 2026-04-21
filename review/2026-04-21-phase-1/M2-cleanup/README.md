# M2 — Submission hygiene cleanup

## What's being removed

### 6 stray temp files at `src/sim/` root

```
src/sim/tmp62af49zc_simsat_api.py
src/sim/tmp74doudc6_simsat_api.py
src/sim/tmpadb6otxm_simsat_api.py
src/sim/tmpf2xgzt02_simsat_api.py
src/sim/tmpp6sdrpv__simsat_api.py
src/sim/tmpq5ulz5bd_simsat_api.py
```

All six are ~500 bytes, dated April 1–2 2026 — leftover from earlier API iteration. They are not imported anywhere. Judges will grep.

### Suspect duplicate `gui.py`

During the review, a standalone `gui.py` was attached that differs from `src/sim/gui.py`:
- `src/sim/gui.py` (4607 bytes, used by `main.py`) — correct, keep.
- The standalone version (3159 bytes) has a hardcoded LAN IP `192.168.115.95:8080` and missing `TOPIC_SATELLITE_GROUND_POSITION` / `TOPIC_SIMULATION_TICK` imports, which would cause a NameError at import time.

If a duplicate exists anywhere in the tree (e.g., `scripts/gui.py`, `old/gui.py`, `legacy/gui.py`), remove it. The PowerShell script below searches for copies with that LAN IP and reports them for manual review.

## How to apply

From `D:\SimSat`:

```powershell
# PowerShell (pwsh or Windows PowerShell)
cd D:\SimSat
pwsh .\review\2026-04-21-phase-1\M2-cleanup\cleanup.ps1
```

Then verify with `git status` and commit.

The script:
1. Removes all 6 `tmp*_simsat_api.py` files under `src\sim\`.
2. Scans the repo for `gui.py` files **outside** `src\sim\gui.py` that contain the string `192.168.115.95`, and prints them for manual review (it does NOT auto-delete those — you confirm first).
