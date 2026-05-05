from __future__ import annotations

from dataclasses import dataclass


KNOWN_SCENARIO_PACKS = [
    "maritime_chokepoints",
    "disaster_response_weather",
    "urban_coastal_ambiguity",
    "pedospheric_integrity",
]


@dataclass(frozen=True)
class ScenarioRunPolicy:
    smoke_hours: float
    competition_hours: float


DEFAULT_SCENARIO_RUN_POLICIES: dict[str, ScenarioRunPolicy] = {
    "maritime_chokepoints": ScenarioRunPolicy(smoke_hours=8.0, competition_hours=48.0),
    "disaster_response_weather": ScenarioRunPolicy(smoke_hours=16.0, competition_hours=16.0),
    "urban_coastal_ambiguity": ScenarioRunPolicy(smoke_hours=8.0, competition_hours=8.0),
    "pedospheric_integrity": ScenarioRunPolicy(smoke_hours=12.0, competition_hours=24.0),
}


def parse_scenario_hours(items: list[str]) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --scenario-hours entry {item!r}; expected scenario=hours")
        scenario, raw_hours = item.split("=", 1)
        scenario = scenario.strip()
        if not scenario:
            raise ValueError(f"Invalid --scenario-hours entry {item!r}; scenario name is empty")
        try:
            hours = float(raw_hours)
        except ValueError as exc:
            raise ValueError(f"Invalid hour value in --scenario-hours entry {item!r}") from exc
        if hours <= 0 or hours > 48:
            raise ValueError(f"Scenario hours must be in (0, 48], got {hours} for {scenario}")
        mapping[scenario] = hours
    return mapping


def default_scenario_hours(policy: str) -> dict[str, float]:
    if policy not in {"smoke", "competition"}:
        raise ValueError(f"Unknown scenario policy {policy!r}")
    return {
        scenario: (
            config.smoke_hours
            if policy == "smoke"
            else config.competition_hours
        )
        for scenario, config in DEFAULT_SCENARIO_RUN_POLICIES.items()
    }


def resolve_scenario_hours(
    *,
    scenarios: list[str],
    default_hours: float,
    policy: str | None = None,
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    hours_by_scenario = {scenario: default_hours for scenario in scenarios}
    if policy in {"smoke", "competition"}:
        for scenario, hours in default_scenario_hours(policy).items():
            if scenario in hours_by_scenario:
                hours_by_scenario[scenario] = hours
    for scenario, hours in (overrides or {}).items():
        if scenario in hours_by_scenario:
            hours_by_scenario[scenario] = hours
    return hours_by_scenario


def scenario_hours_args(hours_by_scenario: dict[str, float]) -> list[str]:
    args: list[str] = []
    for scenario in KNOWN_SCENARIO_PACKS:
        hours = hours_by_scenario.get(scenario)
        if hours is None:
            continue
        args.extend(["--scenario-hours", f"{scenario}={hours:g}"])
    return args
