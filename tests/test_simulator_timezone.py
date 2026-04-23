from __future__ import annotations

import datetime
import time as time_module

import pytest
from pydispatch import dispatcher

from orbit_config import DEFAULT_TLE, SATELLITE_NAME
from simulator import Simulator, TOPIC_SATELLITE_GROUND_POSITION


FIXED_EPOCH = 1773144000  # 2026-03-10T12:00:00Z
EXPECTED_UTC = datetime.datetime(2026, 3, 10, 12, 0, 10, tzinfo=datetime.timezone.utc)


@pytest.fixture
def capture_positions():
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
        t0=FIXED_EPOCH,
        timing_mode=0,
        time_step=10,
    )
    sim.sim_is_running = True
    sim.start_time = None
    sim.sim_step()
    assert capture_list, "sim_step() should emit a ground-position event"


def test_emitted_time_has_z_suffix(capture_positions):
    _step_once(capture_positions)
    assert capture_positions[-1]["time"].endswith("Z")


def test_emitted_time_parses_as_utc(capture_positions):
    _step_once(capture_positions)
    parsed = datetime.datetime.fromisoformat(capture_positions[-1]["time"].replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == datetime.timedelta(0)


def test_emitted_time_matches_epoch_regardless_of_local_tz(monkeypatch, capture_positions):
    monkeypatch.setenv("TZ", "America/New_York")
    if hasattr(time_module, "tzset"):
        time_module.tzset()

    _step_once(capture_positions)
    parsed = datetime.datetime.fromisoformat(capture_positions[-1]["time"].replace("Z", "+00:00"))
    assert parsed == EXPECTED_UTC
