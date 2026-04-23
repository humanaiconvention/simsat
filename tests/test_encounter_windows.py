from __future__ import annotations

from encounter.schemas import EphemerisPoint, TargetSpec
from encounter.windows import WindowDetector


def test_enumerate_windows_builds_expected_window():
    detector = WindowDetector(min_elevation_degrees=30.0, min_window_seconds=20.0)
    target = TargetSpec(target_id="target-1", label="Target One", lon=0.0, lat=0.0, priority=0.9)
    ephemeris = [
        EphemerisPoint(timestamp="2026-03-10T12:00:00Z", satellite_position=[0.0, 0.0, 500.0]),
        EphemerisPoint(timestamp="2026-03-10T12:00:10Z", satellite_position=[0.0, 0.0, 500.0]),
        EphemerisPoint(timestamp="2026-03-10T12:00:20Z", satellite_position=[0.0, 0.0, 500.0]),
        EphemerisPoint(timestamp="2026-03-10T12:00:30Z", satellite_position=[90.0, 0.0, 500.0]),
    ]

    windows = detector.enumerate_windows(ephemeris_points=ephemeris, targets=[target])

    assert len(windows) == 1
    window = windows[0]
    assert window.window_id.startswith("enc_")
    assert window.target_id == target.target_id
    assert window.start_time == "2026-03-10T12:00:00Z"
    assert window.end_time == "2026-03-10T12:00:20Z"
    assert window.peak_time == "2026-03-10T12:00:00Z"
    assert window.duration_seconds == 30.0
    assert window.geometry.target_visible is True
    assert window.geometry.elevation_degrees >= detector.min_elevation_degrees
