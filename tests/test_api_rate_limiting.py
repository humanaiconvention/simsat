"""Tests for SENTINEL_MAPBOX_RATE_LIMIT-controlled rate limiting on imagery endpoints.

The endpoints don't need to succeed for these tests — they need to demonstrate
that the rate limiter engages and produces HTTP 429 once the per-IP budget is
spent, regardless of what the underlying handler returns. We use a fresh
import of `api` inside each test so the env var is read at module init.
"""
from __future__ import annotations

import importlib
import sys
from collections import Counter

import pytest


def _fresh_api(monkeypatch, limit_spec: str):
    """Re-import the api module with a specific rate-limit env var."""
    monkeypatch.setenv("SENTINEL_MAPBOX_RATE_LIMIT", limit_spec)
    for mod in ("api",):
        if mod in sys.modules:
            del sys.modules[mod]
    return importlib.import_module("api")


def _hammer(client, path: str, n: int) -> Counter:
    return Counter(client.get(path).status_code for _ in range(n))


@pytest.mark.parametrize(
    "endpoint",
    [
        "/data/current/image/sentinel",
        "/data/current/image/mapbox",
    ],
)
def test_env_var_caps_rate_limit(monkeypatch, endpoint):
    """A low SENTINEL_MAPBOX_RATE_LIMIT must produce 429s within the test burst."""
    api_module = _fresh_api(monkeypatch, "2/minute")
    from fastapi.testclient import TestClient

    with TestClient(api_module.api) as client:
        statuses = _hammer(client, endpoint, n=6)

    assert statuses.get(429, 0) >= 3, (
        f"Expected ≥3 rate-limited responses with limit=2/minute, got {dict(statuses)}"
    )


def test_module_constant_reflects_env_var(monkeypatch):
    """The exported SENTINEL_MAPBOX_RATE_LIMIT constant must come from the env var."""
    api_module = _fresh_api(monkeypatch, "7/hour")
    assert api_module.SENTINEL_MAPBOX_RATE_LIMIT == "7/hour"


def test_rate_limit_exceeded_handler_registered(monkeypatch):
    """slowapi's RateLimitExceeded must be wired to the FastAPI exception handlers."""
    api_module = _fresh_api(monkeypatch, "30/minute")
    from slowapi.errors import RateLimitExceeded

    assert RateLimitExceeded in api_module.api.exception_handlers
