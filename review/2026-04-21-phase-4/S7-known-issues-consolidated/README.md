# S7 — Consolidated Known Issues

## What this is

`KNOWN_ISSUES.md` in this directory is the single-file audit trail of every correctness, rigor, and narrative issue surfaced during the 2026-04-21 full-repo review, grouped by tier, with status and patch pointers.

## Why ship this

Judges who do a thorough review will grep the repo for anything that looks like "TODO", "FIXME", or "KNOWN_ISSUES". Giving them a single consolidated file tells a cleaner story than scattering items across per-phase READMEs: *"we found these, here's what's shipped vs. open, and here's where to look."*

It also keeps honesty ahead of PR — disclosing Tier 3 items ourselves (e.g., `DJANGO_DEBUG=1` default, missing propagator unit test) costs less than having a judge surface them in their review notes.

## Apply

Drop `KNOWN_ISSUES.md` at the repo root (`D:\SimSat\KNOWN_ISSUES.md`). The top-level `README.md` patch from S4 already references it.

## Maintain

Before submission, sweep the doc:
- Flip "Awaiting local apply" → "Applied" for anything Ben has landed in `D:\SimSat\`
- Flip "Open" → "Applied" or "Deferred, documented" for any Tier 3 items that get worked
- Leave Tier 4 "Entry B specific" items as they are until the team-discussion thesis call lands

If the doc gets stale, it becomes a liability rather than an asset — judges comparing it against the actual repo state will notice drift.
