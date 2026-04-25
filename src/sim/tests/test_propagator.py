"""Propagator unit tests — validates get_orbital_location against the default TLE.

Covers:
  1. Position at TLE epoch is physically plausible (altitude, lat/lon bounds).
  2. Propagation is deterministic (same timestamp → same result).
  3. Propagation advances (t+100 s gives a different position from t).
  4. The _coerce_orbit_time helper accepts float, datetime, ISO string, and
     np.datetime64 without raising.

Reference values (approximate):
  TLE epoch 26075.16558042 → 2026-03-16T03:58:26Z
  Inclination 98.5677°, mean motion 14.308 rev/day → altitude ≈ 793 km
  Expected altitude range at epoch: 788–798 km (eccentricity 0.0000884)
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np
import pytest

_SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

from simulator import Simulator          # noqa: E402
from orbit_config import DEFAULT_TLE, SATELLITE_NAME  # noqa: E402

# TLE epoch: year=2026, day=75.16558042
# day 75 of 2026 = 2026-03-16; fraction 0.16558042 * 86400 ≈ 14306 s ≈ 03:58:26 UTC
_TLE_EPOCH_UTC = datetime.datetime(2026, 3, 16, 3, 58, 26, tzinfo=datetime.timezone.utc)
_TLE_EPOCH_TS = _TLE_EPOCH_UTC.timestamp()


@pytest.fixture(scope="module")
def sim():
    return Simulator(
        SATELLITE_NAME,
        TLE=DEFAULT_TLE,
        t0=_TLE_EPOCH_TS,
        timing_mode=0,
        time_step=10,
    )


# ---------------------------------------------------------------------------
# 1. Plausibility at TLE epoch
# ---------------------------------------------------------------------------

def test_altitude_at_epoch_is_plausible(sim):
    """Altitude at TLE epoch should be ~793 km ± 10 km for this near-circular SSO."""
    _lon, _lat, alt = sim.get_orbital_location(_TLE_EPOCH_TS)
    assert 780 < alt < 810, (
        f"Altitude {alt:.1f} km out of expected [780, 810] km range for this orbit"
    )


def test_lat_lon_at_epoch_in_range(sim):
    """lon ∈ [-180, 180] and lat ∈ [-99, 99] at TLE epoch."""
    lon, lat, _alt = sim.get_orbital_location(_TLE_EPOCH_TS)
    assert -180.0 <= lon <= 180.0, f"lon out of range: {lon}"
    assert -99.0 <= lat <= 99.0, f"lat out of range: {lat}"


# ---------------------------------------------------------------------------
# 2. Determinism
# ---------------------------------------------------------------------------

def test_propagation_is_deterministic(sim):
    """Same timestamp must produce the same lon/lat/alt."""
    r1 = sim.get_orbital_location(_TLE_EPOCH_TS)
    r2 = sim.get_orbital_location(_TLE_EPOCH_TS)
    assert r1 == r2, f"Non-deterministic propagation: {r1} != {r2}"


# ---------------------------------------------------------------------------
# 3. Time advancement
# ---------------------------------------------------------------------------

def test_position_changes_after_100s(sim):
    """100 s of propagation must move the sub-satellite point measurably."""
    lon0, lat0, alt0 = sim.get_orbital_location(_TLE_EPOCH_TS)
    lon1, lat1, alt1 = sim.get_orbital_location(_TLE_EPOCH_TS + 100)
    # At ~7.5 km/s ground speed the satellite moves ~750 km in 100 s — expect >0.1° change
    delta = abs(lon1 - lon0) + abs(lat1 - lat0)
    assert delta > 0.1, (
        f"Position barely changed over 100 s (Δ={delta:.4f}°) — propagation may be stuck"
    )


# ---------------------------------------------------------------------------
# 4. _coerce_orbit_time accepts multiple input types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ts_input", [
    _TLE_EPOCH_TS,                                          # float (Unix seconds)
    _TLE_EPOCH_UTC,                                         # datetime (UTC-aware)
    "2026-03-16T03:58:26Z",                                 # ISO-8601 Z string
    "2026-03-16T03:58:26+00:00",                            # ISO-8601 offset string
    np.datetime64(_TLE_EPOCH_UTC.replace(tzinfo=None)),     # np.datetime64 (naive UTC)
])
def test_coerce_accepts_input_type(sim, ts_input):
    """_coerce_orbit_time must accept float, datetime, ISO string, and np.datetime64."""
    lon, lat, alt = sim.get_orbital_location(ts_input)
    assert isinstance(lon, float)
    assert isinstance(lat, float)
    assert isinstance(alt, float)
    assert 780 < alt < 810, f"Unexpected altitude {alt:.1f} km for input {ts_input!r}"
