"""Regression test for simulator.py timezone bug (Phase 1 - M1).

Before the fix at simulator.py:111, `datetime.fromtimestamp(utcg_time).isoformat()`
returned a naive, local-time ISO string. Downstream consumers in the encounter
service treated that string as UTC via `_coerce_datetime`, which produced
phase-shifted encounter windows whenever the host's timezone was not UTC.

After the fix, the emitted `time` field is an explicit UTC ISO-8601 string
ending in "Z", regardless of the host's local timezone.

Drop this file into `src/sim/tests/` (create the folder if needed) and run:

    cd src/sim
    python -m pytest tests/test_simulator_timezone.py -v
"""

from __future__ import annotations

import datetime
import os
import sys

import pytest
from pydispatch import dispatcher

# Resolve sim source path from the tests/ subfolder
_SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

from simulator import Simulator, TOPIC_SATELLITE_GROUND_POSITION  # noqa: E402
from orbit_config import DEFAULT_TLE, SATELLITE_NAME  # noqa: E402


# Epoch 1773144000 = 2026-03-10T12:00:00Z
# Chosen to be within a few weeks of the default TLE epoch (2026-03-16) so
# SGP4 propagation stays inside its validity window.
_FIXED_EPOCH = 1773144000
_EXPECTED_UTC = datetime.datetime(2026, 3, 10, 12, 0, 0, tzinfo=datetime.timezone.utc)


@pytest.fixture
def capture_positions():
    """Capture TOPIC_SATELLITE_GROUND_POSITION signals during a test."""
    captured: list[dict] = []

    def handler(sender, data, time, time_epsec):
        captured.append({"time": time, "time_epsec": time_epsec, "data": data})

    dispatcher.connect(handler, signal=TOPIC_SATELLITE_GROUND_POSITION)
    try:
        yield captured
    finally:
        dispatcher.disconnect(handler, signal=TOPIC_SATELLITE_GROUND_POSITION)


def _step_once(capture_list: list[dict]) -> None:
    sim = Simulator(
        SATELLITE_NAME,
        TLE=DEFAULT_TLE,
        t0=_FIXED_EPOCH,
        timing_mode=0,  # run as fast as possible, no wall-clock throttle
        time_step=10,
    )
    sim.sim_is_running = True
    sim.start_time = None
    sim.sim_step()
    assert capture_list, "sim_step() should have emitted at least one ground position"


def test_emitted_time_has_z_suffix(capture_positions):
    """The emitted time field must be a UTC-marked ISO string."""
    _step_once(capture_positions)
    time_str = capture_positions[-1]["time"]
    assert time_str.endswith("Z"), (
        f"Expected UTC ISO with 'Z' suffix, got: {time_str!r}"
    )


def test_emitted_time_parses_as_utc(capture_positions):
    """The emitted time field must parse to a timezone-aware UTC datetime."""
    _step_once(capture_positions)
    time_str = capture_positions[-1]["time"]
    parsed = datetime.datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None, (
        f"Expected timezone-aware datetime, got naive: {time_str!r}"
    )
    assert parsed.utcoffset() == datetime.timedelta(0), (
        f"Expected UTC offset 0, got: {parsed.utcoffset()} from {time_str!r}"
    )


def test_emitted_time_matches_epoch_regardless_of_local_tz(monkeypatch, capture_positions):
    """Regression: simulate a non-UTC host timezone and verify the emitted time still represents UTC.

    The pre-fix code leaked local time through, so on a UTC-5 host this would
    emit '2026-03-10T07:00:00' instead of '2026-03-10T12:00:00Z'.

    We check date + hour rather than the exact second because the sim advances by
    time_step=10 before emitting, so the timestamp is _FIXED_EPOCH + step.
    """
    monkeypatch.setenv("TZ", "America/New_York")
    try:
        import time as _time
        if hasattr(_time, "tzset"):
            _time.tzset()
    except Exception:
        # Windows has no tzset; the fix uses an explicit tz= kwarg so the
        # assertion below is still valid.
        pass

    _step_once(capture_positions)
    time_str = capture_positions[-1]["time"]
    parsed = datetime.datetime.fromisoformat(time_str.replace("Z", "+00:00"))

    # Must be within 5 minutes of the expected epoch (step-independent)
    delta_secs = abs((parsed - _EXPECTED_UTC).total_seconds())
    assert delta_secs < 300, (
        f"Parsed time {parsed.isoformat()} deviates >5 min from expected UTC epoch "
        f"{_EXPECTED_UTC.isoformat()} — likely a timezone shift bug. raw={time_str!r}"
    )
    # And the UTC date/hour must match exactly (catches a UTC-5 shift that would produce 07:xx)
    assert (parsed.year, parsed.month, parsed.day, parsed.hour) == (
        _EXPECTED_UTC.year, _EXPECTED_UTC.month, _EXPECTED_UTC.day, _EXPECTED_UTC.hour
    ), (
        f"Date/hour mismatch — expected 2026-03-10T12:xx:xxZ, "
        f"got: {parsed.isoformat()} (raw: {time_str!r})"
    )
