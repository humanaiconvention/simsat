from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

import requests


class HttpClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def get_json(self, path: str, **params) -> dict:
        response = requests.get(f"{self.base_url}{path}", params=params or None, timeout=90)
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()


class InProcessClient:
    def __init__(self) -> None:
        from fastapi.testclient import TestClient

        root = Path(__file__).resolve().parents[1]
        sim_root = root / "src" / "sim"
        if str(sim_root) not in sys.path:
            sys.path.insert(0, str(sim_root))
        from api import api  # type: ignore

        self._client = TestClient(api)
        self._context = self._client.__enter__()

    def close(self) -> None:
        self._client.__exit__(None, None, None)

    def get_json(self, path: str, **params) -> dict:
        response = self._context.get(path, params=params or None)
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "case"


def export_stimulus_image(stimulus_id: str, output_path: Path) -> bool:
    root = Path(__file__).resolve().parents[1]
    sim_root = root / "src" / "sim"
    if str(sim_root) not in sys.path:
        sys.path.insert(0, str(sim_root))
    from haic.stimulus_store import get_stimulus_store  # type: ignore

    stimulus = get_stimulus_store().get(stimulus_id)
    if stimulus is None:
        return False
    for image in stimulus.images:
        if image.image_b64:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(base64.b64decode(image.image_b64))
            return True
    return False


def mission_response_line(bundle: dict) -> str | None:
    mission_response = bundle.get("mission_response")
    if not isinstance(mission_response, dict):
        return None
    action = mission_response.get("action") or {}
    outcome = mission_response.get("outcome") or {}
    recommended_action = action.get("recommended_action")
    utility_realized = outcome.get("utility_realized")
    if not recommended_action:
        return None
    return f"- Mission response: action=`{recommended_action}`, utility_realized=`{utility_realized}`"


def build_casebook(client, output_path: Path, assets_dir: Path) -> Path:
    submission_cases = client.get_json("/observation-vla/submission-cases").get("cases", [])
    if not submission_cases:
        raise RuntimeError("No pinned submission cases found")

    ordered_cases = sorted(submission_cases, key=lambda item: item.get("scenario_pack", ""))
    lines: list[str] = []
    lines.append("# SimSat Submission Casebook")
    lines.append("")
    lines.append(
        f"Generated on `{datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}` from pinned operator-reviewed submission cases."
    )
    lines.append("")

    for case in ordered_cases:
        scenario_pack = case.get("scenario_pack", "unknown")
        trace_id = case.get("trace_id", "")
        bundle = client.get_json(f"/observation-vla/review-bundle/{trace_id}")
        target = bundle.get("target", {})
        current_outcome = bundle.get("current_outcome") or {}
        trace = bundle.get("trace", {})
        assessment = trace.get("assessment", {})
        sample = trace.get("sample", {})
        probe = bundle.get("probe", {}) or sample.get("probe", {})
        stimulus_id = trace.get("stimulus_id")
        asset_name = f"{_slug(scenario_pack)}_{_slug(target.get('label', target.get('target_id', 'case')))}.png"
        asset_path = assets_dir / asset_name
        image_written = bool(stimulus_id) and export_stimulus_image(str(stimulus_id), asset_path)

        lines.append(f"## {scenario_pack}")
        lines.append("")
        lines.append(f"- Target: `{target.get('label', target.get('target_id', 'unknown'))}`")
        lines.append(f"- Trace: `{trace_id}`")
        lines.append(f"- Reviewer: `{case.get('reviewer')}`")
        lines.append(f"- Operator action: `{current_outcome.get('operator_action')}`")
        lines.append(f"- Useful: `{current_outcome.get('useful')}`")
        lines.append(f"- Usefulness score: `{current_outcome.get('usefulness_score')}`")
        lines.append(f"- Observation runtime: `{assessment.get('runtime_mode')}`")
        lines.append(f"- Sentinel source: `{probe.get('sentinel_source')}`")
        lines.append(f"- Cloud cover: `{probe.get('sentinel_cloud_cover')}`")
        if current_outcome.get("notes"):
            lines.append(f"- Review notes: {current_outcome.get('notes')}")
        mission_line = mission_response_line(bundle)
        if mission_line:
            lines.append(mission_line)
        lines.append("")
        if image_written:
            lines.append(f"![{target.get('label', 'Submission case')}]({asset_path.resolve()})")
            lines.append("")
            lines.append(f"- Image asset: [{asset_path.resolve()}]({asset_path.resolve()})")
        else:
            lines.append("- Image asset: not available in local stimulus store")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a visual casebook from pinned SimSat submission cases.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/sim")
    parser.add_argument("--inprocess", action="store_true", help="Run against an in-process FastAPI app instead of a live server.")
    parser.add_argument("--output", default=str(Path("D:/SimSat/SUBMISSION_CASEBOOK.md")))
    parser.add_argument("--assets-dir", default=str(Path("D:/SimSat/submission_assets")))
    args = parser.parse_args()

    client = None
    try:
        client = InProcessClient() if args.inprocess else HttpClient(args.base_url)
        output_path = Path(args.output)
        assets_dir = Path(args.assets_dir)
        result = build_casebook(client, output_path=output_path, assets_dir=assets_dir)
        print(f"Submission casebook written to {result}")
    except Exception as exc:
        print(f"Submission casebook generation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        if isinstance(client, InProcessClient):
            client.close()


if __name__ == "__main__":
    main()
