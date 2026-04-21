# M1 — Fix `simulator.py` timezone bug

## The bug

`src/sim/simulator.py` line 111:

```python
time=datetime.datetime.fromtimestamp(utcg_time).isoformat(),
```

`datetime.fromtimestamp()` with no `tz=` argument returns a **naive, local-time** datetime. Its `.isoformat()` produces a string like `2026-03-10T07:00:00` (with no timezone marker) when the host is on UTC−5, instead of `2026-03-10T12:00:00Z`.

## Why it matters

This string is emitted on the `TOPIC_SATELLITE_GROUND_POSITION` signal. Downstream:

1. `src/sim/camera.py::on_satellite_ground_position` stores it as `shared_data_dict["last_updated"]`.
2. `src/sim/encounter/service.py::_resolve_start_time` (lines 65–73) accepts that string and treats it as UTC via `ephemeris._coerce_datetime`, which assumes `tzinfo=UTC` for naive inputs.
3. Result: **every encounter window is phase-shifted by the host's UTC offset.** A judge running the demo on a US-East laptop gets windows 5 hours off; on CET, 1 hour off; on IST, 5.5 hours off.

The rest of the codebase is UTC-correct:
- `set_start_time` (line 178) explicitly rejects non-UTC input ✓
- All ISO-8601 formatters in `ephemeris.py`, `planner.py`, `service.py` use `"+00:00".replace("+00:00","Z")` ✓

Only this one call site leaks local time.

## The fix

One-line change. Apply `simulator.py.patch` (unified diff against `src/sim/simulator.py`):

```bash
cd D:\SimSat
git apply review/2026-04-21-phase-1/M1-timezone-fix/simulator.py.patch
```

Or edit manually:

```diff
-            time=datetime.datetime.fromtimestamp(utcg_time).isoformat(),
+            time=datetime.datetime.fromtimestamp(utcg_time, tz=datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
```

## Regression test

`test_simulator_timezone.py` contains a pytest-style regression test. Drop it into `src/sim/tests/` (or wherever your pytest collection runs) and run:

```bash
cd D:\SimSat\src\sim
python -m pytest tests/test_simulator_timezone.py -v
```

The test asserts the emitted `time` field on `TOPIC_SATELLITE_GROUND_POSITION`:
1. Ends with `Z` (UTC marker)
2. Parses to a timezone-aware datetime with UTC offset 0
3. Matches the input epoch regardless of the host's local timezone

**Note on dependencies:** the test imports `pyorbital`, `pydispatch`, `pytest` — all already in `requirements.txt`.
