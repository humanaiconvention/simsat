from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from challenge_run_config import (  # noqa: E402
    KNOWN_SCENARIO_PACKS,
    default_scenario_hours,
    parse_scenario_hours,
    resolve_scenario_hours,
    scenario_hours_args,
)


def test_default_scenario_hours_match_current_challenge_policy():
    smoke = default_scenario_hours("smoke")
    competition = default_scenario_hours("competition")

    assert smoke["maritime_chokepoints"] == 8.0
    assert competition["maritime_chokepoints"] == 48.0
    assert smoke["disaster_response_weather"] == 16.0
    assert competition["disaster_response_weather"] == 16.0
    assert smoke["urban_coastal_ambiguity"] == 8.0
    assert competition["urban_coastal_ambiguity"] == 8.0


def test_resolve_scenario_hours_applies_policy_then_overrides():
    resolved = resolve_scenario_hours(
        scenarios=KNOWN_SCENARIO_PACKS,
        default_hours=6.0,
        policy="competition",
        overrides={"urban_coastal_ambiguity": 12.0},
    )

    assert resolved["maritime_chokepoints"] == 48.0
    assert resolved["disaster_response_weather"] == 16.0
    assert resolved["urban_coastal_ambiguity"] == 12.0


def test_parse_and_render_scenario_hour_args_round_trip():
    parsed = parse_scenario_hours(
        [
            "maritime_chokepoints=48",
            "disaster_response_weather=16",
        ]
    )

    assert parsed == {
        "maritime_chokepoints": 48.0,
        "disaster_response_weather": 16.0,
    }
    assert scenario_hours_args(parsed) == [
        "--scenario-hours",
        "maritime_chokepoints=48",
        "--scenario-hours",
        "disaster_response_weather=16",
    ]
