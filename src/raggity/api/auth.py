"""Service-to-service auth: one shared key between Raggity and the agent gateway.

Raggity is meant to sit on a private network behind your gateway; this key stops anything
else on that network (or anything that finds the port) from reading or writing collections.
It is not end-user auth. The gateway authenticates visitors and decides which `filter`
each one gets.
"""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status


def _presented_key(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-api-key")


def require_api_key(request: Request) -> None:
    expected = request.app.state.service.settings.api_key
    if expected is None:
        return  # auth disabled for local development; main.py logs a warning at startup
    presented = _presented_key(request)
    # compare_digest takes the same time whether the first or last byte differs, so the
    # response time can't be used to guess the key one character at a time.
    if presented is None or not hmac.compare_digest(
        presented.encode(), expected.get_secret_value().encode()
    ):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
