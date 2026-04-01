"""
Simulator proxy — forwards /sim/* requests from the Django dashboard
to the SimSat simulator process (port 9005 in Docker, configurable via SIM_URL).

Allows the React frontend to call /sim/haic/... without CORS issues,
since everything goes through the same Django origin.
"""

from __future__ import annotations

import os
import urllib.parse

import requests as http
from django.http import HttpRequest, HttpResponse, StreamingHttpResponse

SIM_URL = os.environ.get("SIM_URL", "http://sim:8000")


def proxy_to_sim(request: HttpRequest, path: str) -> HttpResponse:
    """
    Transparent proxy: forward request to simulator and return response.
    Preserves method, query string, body, and most headers.
    """
    # Build target URL
    query = request.META.get("QUERY_STRING", "")
    target = f"{SIM_URL}/{path}"
    if query:
        target = f"{target}?{query}"

    # Forward only safe headers (skip Django-specific ones)
    forward_headers = {}
    for key, value in request.META.items():
        if key.startswith("HTTP_") and key not in (
            "HTTP_HOST", "HTTP_X_FORWARDED_FOR", "HTTP_X_FORWARDED_HOST"
        ):
            header = key[5:].replace("_", "-").title()
            forward_headers[header] = value
    if request.content_type:
        forward_headers["Content-Type"] = request.content_type

    try:
        resp = http.request(
            method=request.method,
            url=target,
            headers=forward_headers,
            data=request.body if request.method not in ("GET", "HEAD") else None,
            timeout=30,
            stream=True,
            allow_redirects=False,
        )
    except http.exceptions.ConnectionError:
        return HttpResponse(
            b'{"error": "Simulator not available"}',
            status=503,
            content_type="application/json",
        )

    # Build Django response
    response = HttpResponse(
        content=resp.content,
        status=resp.status_code,
        content_type=resp.headers.get("Content-Type", "application/json"),
    )

    # Forward select headers
    for header in ("Cache-Control", "Access-Control-Expose-Headers",
                   "Sentinel-Metadata", "Mapbox-Metadata",
                   "X-Stimulus-Type"):
        if header in resp.headers:
            response[header] = resp.headers[header]

    return response
