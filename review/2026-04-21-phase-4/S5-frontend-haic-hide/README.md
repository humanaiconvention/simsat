# S5 — Hide HAIC from the dashboard frontend

## Why

Earlier today the team flipped from "feature HAIC" → "hide HAIC" in the submission narrative. The HAIC content stays in the repo for anyone who reads the source, but it should not be visible in the product surface a judge interacts with. Today's submission-facing docs (`SUBMISSION_BRIEF.md`, `CHALLENGE_ENTRY.md`, top-level `README.md`) are already HAIC-silent. The only remaining visible HAIC surface is the dashboard frontend.

Per today's review:

> `App.tsx:48` renders a "HAIC Convention Layer" badge in the dashboard header. HAIC is visually THE product.

> `dist/arch/*.html` reveals an Architecture Explorer with model visualizations ... Yet `SUBMISSION_BRIEF.md` and `CHALLENGE_ENTRY.md` do not mention HAIC once.

The submission docs are staying HAIC-silent; the dashboard needs to match.

## Scope — what this does

1. **Remove the "HAIC Convention Layer" badge** from the dashboard header (`App.tsx`, around line 48).
2. **Remove the Architecture Explorer button / link** from the dashboard's visible navigation if one is present (the `dist/arch/*.html` static pages can stay on disk — they just should not be linked from the UI).
3. Leave the underlying HAIC code in place (`src/sim/haic/*`) so reviewers who grep the repo can still find it. Hiding ≠ deleting.

## How to apply

This bundle does not include a literal `App.tsx` diff because the React dashboard is not in the `humanaiconvention/simsat` GitHub main branch — it lives only in `D:\SimSat\dashboard\frontend\src\App.tsx` locally. Ben should apply the edits directly to the local file.

### Edit 1 — Remove the HAIC badge

Open `dashboard/frontend/src/App.tsx` and find the element around **line 48** that renders `"HAIC Convention Layer"` (likely a `<Badge>`, `<span className="...">`, or `<div>` inside the header JSX).

Remove the entire element, including:
- the JSX node itself
- any associated CSS class usage that becomes dead after removal
- any import for a `Badge` / `HaicBadge` / similar component used only for this element

Example of what to look for (names may differ):

```tsx
// BEFORE — inside the dashboard header JSX
<header className="dashboard-header">
  <h1>SimSat</h1>
  <div className="haic-badge">HAIC Convention Layer</div>  {/* ← remove this line */}
  <NavControls />
</header>
```

### Edit 2 — Remove the Architecture Explorer link

Search the frontend source tree for references to the Architecture Explorer — likely an `<a>` or `<Link>` that targets `dist/arch/`, `/arch/`, `/architecture/`, or similar:

```bash
grep -rn "arch" dashboard/frontend/src/
grep -rn "Architecture" dashboard/frontend/src/
```

Remove the link element. Leave the underlying `dist/arch/*.html` files in place — they are static assets, not runtime dependencies, and deleting them is out of scope for a pitch-surface change.

### Edit 3 — Verify

Rebuild the frontend:

```bash
cd dashboard/frontend && npm run build
```

Open the dashboard:

```bash
docker compose up
# visit http://localhost:8000
```

Visual check:
- Header should show `SimSat` with no `HAIC Convention Layer` badge
- Navigation should have no `Architecture Explorer` button or link
- Encounter Planner panel, Satellite / Telemetry panels, and simulation controls should all work unchanged

Grep check:

```bash
grep -rn "HAIC" dashboard/frontend/src/         # expect 0 or only code comments
grep -rn "Convention Layer" dashboard/frontend/src/   # expect 0
grep -rn "Architecture Explorer" dashboard/frontend/src/  # expect 0 in JSX; OK in unused imports until Ben removes them
```

## What this does **not** do

- **Does not** delete the `src/sim/haic/*` backend code. Hiding from the pitch surface, not removing from the repo.
- **Does not** delete `dist/arch/*.html`. Static files stay; they just aren't linked from the UI.
- **Does not** modify the backend `capabilities` endpoint or anything Django-side. No API surface changes.
- **Does not** hide HAIC from the source-level repo (e.g., doesn't rename `src/sim/haic/` or scrub HAIC from internal docs). The Liquid AI team's technical reviewers will still see HAIC if they clone and grep — that's fine, and expected for a rigorous review.

## Reversibility

Pure UI revert: restore the removed JSX elements from git history. No data migration, no backend state to rebuild. If the team flips back to "feature HAIC", the reverse patch is one `git revert` away.
