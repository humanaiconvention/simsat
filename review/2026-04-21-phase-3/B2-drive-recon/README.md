# Phase 3 B2 — Drive recon (BLOCKED on OAuth scope)

**Status:** blocked
**Goal:** find existing Colab notebooks in Ben's two Google Drives that relate to HAIC, LFM2-VL, SimSat, or Gemma fine-tuning — to map existing scaffolding for Entry A (Liquid Track).

## What's blocked

Both connected Drive accounts returned `403 PERMISSION_DENIED / ACCESS_TOKEN_SCOPE_INSUFFICIENT` on `GOOGLEDRIVE_FIND_FILE`:

| Account | Email | Status |
|---|---|---|
| Google Drive (`geo9ncv3`) | ben@humanaiconvention.com | insufficient scope |
| Google Drive 2 (`darkp7o1`) | benjamin.haslam@gmail.com | insufficient scope |

The connection was created 2026-04-21 16:32 UTC but apparently with a narrow set of scopes that don't include `drive.metadata.readonly` or `drive.readonly`. Listing / searching files requires at minimum `drive.metadata.readonly`.

Auth config id (for platform-support escalation): `ac_GcVL6QjxCcpE`

## How to unblock

Three options:

1. **Reconnect Google Drive with broader scopes.** In the platform's Settings → Integrations page, disconnect and reconnect each Google Drive account, granting `Drive > See and download all your Google Drive files` (or broader) on the OAuth consent screen. This is the cleanest path.
2. **Escalate to platform support** with auth config id `ac_GcVL6QjxCcpE` and request that the Google Drive integration include `drive.readonly` or `drive.metadata.readonly` in requested scopes.
3. **Ben pastes notebook URLs directly.** Fallback — Ben lists the Colab notebooks he wants inspected; the agent fetches each via a browser session.

## What I'd do once unblocked

Run two parallel searches (one per Drive account):

```
q: (name contains 'haic' or name contains 'lfm2' or name contains 'simsat' or name contains 'gemma' or name contains 'grounding')
   and mimeType = 'application/vnd.google.colaboratory'
   and trashed = false
fields: id, name, modifiedTime, webViewLink, parents
orderBy: modifiedTime desc
```

Then dedupe by notebook content-hash (notebooks sometimes live in both drives due to shared-drive mirrors).

For each match:
- Fetch the notebook JSON via `GOOGLEDRIVE_DOWNLOAD_FILE` (exports as `.ipynb`).
- Read the first/last few cells to classify: is this a training script? an eval? a data generator?
- Report to Ben as a table (notebook name → purpose → last modified → whether it contains LFM2 imports).

This unblocks Entry A because LFM2-VL work may already have a notebook template Ben has used before.
