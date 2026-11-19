"""Octen SDK client construction and error routing shared by the tools."""

from __future__ import annotations

import os
from typing import Optional

import octen
from pydantic import SecretStr


def resolve_api_key(explicit: Optional[SecretStr]) -> SecretStr:
    """Return the explicit key, else ``OCTEN_API_KEY`` from the environment."""
    if explicit is not None and explicit.get_secret_value():
        return explicit
    env_key = os.environ.get("OCTEN_API_KEY", "")
    if not env_key:
        raise ValueError(
            "OCTEN_API_KEY must be provided as a parameter or set as an environment variable."
        )
    return SecretStr(env_key)


def sync_client(api_key: SecretStr) -> octen.Octen:
    """Create a sync client that does not retry a request once it may have been sent.

    A retried request that already reached the server can be billed twice.
    """
    return octen.Octen(api_key=api_key.get_secret_value(), retry_after_send=False)


def async_client(api_key: SecretStr) -> octen.AsyncOcten:
    """Async twin of :func:`sync_client`."""
    return octen.AsyncOcten(api_key=api_key.get_secret_value(), retry_after_send=False)


def classify(exc: Exception) -> Optional[str]:
    """Return the message to hand to the model, or ``None`` if ``exc`` must be raised.

    Transient failures, server-side rejections of model-supplied values and 2xx
    responses that carry an error envelope or a malformed body go to the model;
    configuration problems (bad key, no access, invalid developer options) and anything
    unrecognised are raised so the developer sees them.
    """
    if isinstance(exc, (octen.OctenTimeoutError, octen.OctenConnectionError)):
        return str(exc)
    if isinstance(exc, octen.OctenRateLimitError):
        return str(exc)
    if isinstance(exc, octen.OctenAPIError):
        status = exc.status_code
        if status is None:
            return None
        if 200 <= status <= 299:
            # An error envelope that reports an auth or access failure is still a config error.
            data = exc.response_data
            code = data.get("code") if isinstance(data, dict) else None
            return None if code in (401, 403) else str(exc)
        if 500 <= status <= 599 or status in (400, 408, 422):
            return str(exc)
    return None
