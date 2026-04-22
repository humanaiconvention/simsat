"""Shared orbit configuration for the SimSat default satellite."""

SATELLITE_NAME = "SatelliteName"
TLE_LINE1 = "1 60989U 24157A   26075.16558042  .00000129  00000-0  65710-4 0  9997"
TLE_LINE2 = "2 60989  98.5677 151.2852 0000884 109.8893 250.2385 14.30816791 79683"
DEFAULT_TLE = (TLE_LINE1, TLE_LINE2)


def get_default_orbit() -> tuple[str, tuple[str, str]]:
    """Return the configured satellite name and TLE tuple."""
    return SATELLITE_NAME, DEFAULT_TLE
