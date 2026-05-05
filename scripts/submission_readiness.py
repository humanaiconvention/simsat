from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import requests


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "case"


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


def build_readiness_report(
    client,
    packet_path: Path,
    casebook_path: Path,
    assets_dir: Path,
    observation_eval_path: Path,
) -> str:
    capabilities = client.get_json("/capabilities")
    targets_payload = client.get_json("/encounter/targets")
    scenario_packs = targets_payload.get("scenario_packs", [])
    submission_cases_payload = client.get_json("/observation-vla/submission-cases")
    submission_cases = submission_cases_payload.get("cases", [])
    submission_cases_by_scenario = {
        case.get("scenario_pack"): case for case in submission_cases
    }

    lines: list[str] = []
    lines.append("# SimSat Submission Readiness")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append(f"- Sentinel enabled: `{capabilities.get('sentinel_enabled')}`")
    lines.append(f"- Mapbox enabled: `{capabilities.get('mapbox_enabled')}`")
    lines.append(f"- Sentinel-first challenge scoring: `{capabilities.get('sentinel_first_challenge_scoring')}`")
    lines.append(f"- ObservationVLA runtime: `{capabilities.get('observation_vla_runtime_mode')}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append(f"- Submission packet exists: `{packet_path.exists()}`")
    lines.append(f"- Submission casebook exists: `{casebook_path.exists()}`")
    lines.append(f"- Submission assets dir exists: `{assets_dir.exists()}`")
    lines.append(f"- ObservationVLA eval exists: `{observation_eval_path.exists()}`")
    lines.append(f"- Pinned submission cases: `{len(submission_cases)}`")
    lines.append("")
    lines.append("## Scenario Status")
    lines.append("")
    lines.append("| scenario | reviewed_ready | pinned_trace | reviewer | packet_case | image_asset |")
    lines.append("| --- | --- | --- | --- | --- | --- |")

    packet_text = packet_path.read_text(encoding="utf-8") if packet_path.exists() else ""
    casebook_text = casebook_path.read_text(encoding="utf-8") if casebook_path.exists() else ""
    packet_text_lower = packet_text.lower()
    casebook_text_lower = casebook_text.lower()

    for scenario_pack in scenario_packs:
        evidence = client.get_json(
            "/encounter/evidence",
            scenario_pack=scenario_pack,
            case_limit=3,
            trace_limit=100,
        )
        pinned_case = submission_cases_by_scenario.get(scenario_pack, {})
        target_label = pinned_case.get("target_label", "")
        trace_id = str(pinned_case.get("trace_id", ""))
        packet_case = False
        if target_label:
            packet_case = target_label.lower() in packet_text_lower
        if trace_id and not packet_case:
            packet_case = trace_id.lower() in packet_text_lower

        image_slug = "_".join(
            [
                _slug(scenario_pack),
                _slug(target_label),
            ]
        ).strip("_")
        matching_assets = list(assets_dir.glob(f"{image_slug}*.png")) if assets_dir.exists() and image_slug else []
        if not matching_assets and target_label:
            matching_assets = list(assets_dir.glob(f"*{_slug(target_label)}*.png"))
        image_asset = bool(matching_assets)
        if image_asset and casebook_text:
            has_target_ref = bool(target_label) and target_label.lower() in casebook_text_lower
            has_trace_ref = bool(trace_id) and trace_id.lower() in casebook_text_lower
            image_asset = has_target_ref or has_trace_ref
        lines.append(
            "| "
            + " | ".join(
                [
                    scenario_pack,
                    str(bool(evidence.get("reviewed_submission_ready", False))),
                    str(pinned_case.get("trace_id", "")),
                    str(pinned_case.get("reviewer", "")),
                    str(packet_case),
                    str(bool(image_asset)),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Honesty Boundary")
    lines.append("")
    lines.append(
        "- ObservationVLA is now image-model-backed via the configured runtime. "
        "The reviewed evaluation pool grew from N=8 to N=37 on 2026-04-27 via "
        "`scripts/batch_review.py`; v11 (the first SimSat Gemma-4 fine-tune to "
        "actually train language-model parameters — see "
        "`notebooks/GEMMA4_LORA_NULL_TRAINING_AUDIT.md`) reaches exact action "
        "agreement 0.86 and useful agreement 0.97 over those 37 cases. "
        "The local backend should still be treated as an evidence scorer rather "
        "than a trusted autonomous action policy."
    )
    lines.append("- Mission-response actions are policy outputs with logged utility, not live spacecraft actuation.")
    lines.append("- The reviewed packet is Sentinel-first and does not depend on Mapbox.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a low-compute submission-readiness checklist for SimSat.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/sim")
    parser.add_argument("--inprocess", action="store_true", help="Run against an in-process FastAPI app instead of a live server.")
    _repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--packet", default=str(_repo_root / "SUBMISSION_PACKET.md"))
    parser.add_argument("--casebook", default=str(_repo_root / "SUBMISSION_CASEBOOK.md"))
    parser.add_argument("--assets-dir", default=str(_repo_root / "submission_assets"))
    parser.add_argument("--observation-eval", default=str(_repo_root / "OBSERVATION_VLA_EVAL.md"))
    parser.add_argument("--output", default=str(_repo_root / "SUBMISSION_READINESS.md"))
    args = parser.parse_args()

    client = None
    try:
        client = InProcessClient() if args.inprocess else HttpClient(args.base_url)
        report = build_readiness_report(
            client,
            packet_path=Path(args.packet),
            casebook_path=Path(args.casebook),
            assets_dir=Path(args.assets_dir),
            observation_eval_path=Path(args.observation_eval),
        )
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
        print(f"Submission readiness written to {output_path}")
    except Exception as exc:
        print(f"Submission readiness generation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        if isinstance(client, InProcessClient):
            client.close()


if __name__ == "__main__":
    main()
