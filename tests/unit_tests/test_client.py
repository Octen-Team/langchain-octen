"""Unit tests for the shared client factories and error routing."""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from octen import (
    OctenAPIError,
    OctenAuthenticationError,
    OctenConnectionError,
    OctenRateLimitError,
    OctenTimeoutError,
    OctenValidationError,
)
from pydantic import SecretStr

from langchain_octen._client import async_client, classify, resolve_api_key, sync_client

FAKE_API_KEY = "test-api-key-12345"


@patch("octen.Octen")
def test_sync_client_disables_retry_after_send(mock_octen_cls: MagicMock) -> None:
    client = sync_client(SecretStr(FAKE_API_KEY))
    mock_octen_cls.assert_called_once_with(api_key=FAKE_API_KEY, retry_after_send=False)
    assert client is mock_octen_cls.return_value


@patch("octen.AsyncOcten")
def test_async_client_disables_retry_after_send(mock_async_cls: MagicMock) -> None:
    client = async_client(SecretStr(FAKE_API_KEY))
    mock_async_cls.assert_called_once_with(api_key=FAKE_API_KEY, retry_after_send=False)
    assert client is mock_async_cls.return_value


def test_resolve_api_key_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCTEN_API_KEY", "from-env")
    assert resolve_api_key(SecretStr(FAKE_API_KEY)).get_secret_value() == FAKE_API_KEY


def test_resolve_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCTEN_API_KEY", "from-env")
    assert resolve_api_key(None).get_secret_value() == "from-env"


def test_resolve_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCTEN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OCTEN_API_KEY"):
        resolve_api_key(None)


@pytest.mark.parametrize(
    ("exc", "to_model"),
    [
        (OctenAuthenticationError(), False),
        (OctenAPIError("forbidden", status_code=403), False),
        (OctenValidationError("bad count"), False),
        (OctenAPIError("not found", status_code=404), False),
        (OctenRateLimitError(), True),
        (OctenAPIError("unavailable", status_code=503), True),
        (OctenAPIError("bad request", status_code=400), True),
        (OctenAPIError("unprocessable", status_code=422), True),
        (OctenAPIError("request timeout", status_code=408), True),
        (OctenAPIError("API returned code 500", status_code=200), True),
        (OctenAPIError("malformed successful response", status_code=200), True),
        (OctenAPIError("accepted with an error envelope", status_code=202), True),
        (OctenAPIError("conflict", status_code=409), False),
        (OctenAPIError("no status"), False),
        (OctenTimeoutError(), True),
        (OctenConnectionError(), True),
    ],
)
def test_classify(exc: Exception, to_model: bool) -> None:
    message: Optional[str] = classify(exc)
    if to_model:
        assert message is not None
        assert str(exc) in message
    else:
        assert message is None
