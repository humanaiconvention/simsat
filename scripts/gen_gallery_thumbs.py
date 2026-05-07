#!/usr/bin/env python3
"""Generate JPEG thumbnails for all unreviewed traces in the pool.

Drops thumbnails at /tmp/gallery_thumbs/{trace_id}.jpg (resolves to
D:\\tmp\\gallery_thumbs on Windows) so that gen_gallery_review.py can
embed them as base64.
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from operator_review import InProcessClient


def main() -> int:
    thumb_dir = Path("/tmp/gallery_thumbs")
    thumb_dir.mkdir(parents=True, exist_ok=True)

    c = InProcessClient()
    pool = c.get_json("/observation-vla/traces", limit=1000).get("traces", [])
    print(f"Pool: {len(pool)} traces")

    from PIL import Image

    n_skipped, n_made, n_fail = 0, 0, 0
    for i, t in enumerate(pool):
        tid = t.get("trace_id")
        if not tid:
            continue

        out = thumb_dir / f"{tid}.jpg"
        if out.exists() and out.stat().st_size > 0:
            n_skipped += 1
            continue

        # Skip already-reviewed
        detail = c.get_json(f"/observation-vla/trace/{tid}")
        cur = detail.get("current_outcome") or {}
        if cur.get("label_source") == "operator_review":
            n_skipped += 1
            continue

        sample = detail["trace"]["sample"]
        images = sample.get("images") or []
        if not images:
            print(f"[{i+1}/{len(pool)}] {tid[:24]}  no images")
            n_fail += 1
            continue

        md = images[0].get("metadata", {})
        target_meta = sample.get("target_metadata") or {}
        # Resolve a center+size_km to query the sentinel endpoint
        lon = sample.get("target_lon") or target_meta.get("lon")
        lat = sample.get("target_lat") or target_meta.get("lat")
        if lon is None or lat is None:
            fp = md.get("footprint")
            if fp and len(fp) == 4:
                lon = (fp[0] + fp[2]) / 2
                lat = (fp[1] + fp[3]) / 2
        size_km = md.get("size_km") or 15.0
        ts = md.get("timestamp") or images[0].get("timestamp")

        if lon is None or lat is None or ts is None:
            print(f"[{i+1}/{len(pool)}] {tid[:24]}  missing lon/lat/ts")
            n_fail += 1
            continue

        try:
            res = c._context.get(
                "/data/image/sentinel",
                params={"lon": lon, "lat": lat, "size_km": size_km, "timestamp": ts},
            )
            if res.status_code != 200:
                print(f"[{i+1}/{len(pool)}] {tid[:24]}  http {res.status_code}: {res.text[:60]}")
                n_fail += 1
                continue
            img = Image.open(io.BytesIO(res.content)).convert("RGB")
            img.thumbnail((512, 512), Image.LANCZOS)
            img.save(out, "JPEG", quality=82)
            n_made += 1
            if (n_made % 8) == 0:
                print(f"[{i+1}/{len(pool)}] thumbs made={n_made}, skip={n_skipped}, fail={n_fail}")
        except Exception as e:
            print(f"[{i+1}/{len(pool)}] {tid[:24]}  error: {type(e).__name__}: {e}")
            n_fail += 1

    print()
    print(f"Total: made={n_made}, skipped={n_skipped}, failed={n_fail}, dir={thumb_dir.absolute()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
