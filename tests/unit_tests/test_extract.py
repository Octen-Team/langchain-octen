"""Unit tests for the Extract tool (mocked SDK or HTTP layer, no network)."""

from __future__ import annotations

import inspect
import json
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import octen
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

from langchain_octen import OctenExtract

FAKE_API_KEY = "test-api-key-12345"
URL = "https://example.com/"
URLS = ["https://example.com/", "https://docs.octen.ai/"]
# All developer options, with the SDK names they are sent under.
OPTIONS: Dict[str, Any] = {
    "mode": "advanced",
    "format": "text",
    "max_age_seconds": 600,
    "url_timeout": 45,
    "include_images": True,
    "include_videos": True,
    "include_audio": True,
    "include_links": {"scope": "prefer_external", "max_links": 10},
}
SDK_OPTIONS: Dict[str, Any] = {
    **{k: v for k, v in OPTIONS.items() if k != "url_timeout"},
    "timeout": 45,
}
# Server default per-URL timeout (30 s) plus 90 s of headroom.
DEFAULT_REQUEST_TIMEOUT = 120

# Response shape from https://docs.octen.ai/api-reference/extract.md (example and schema).
MIXED_DATA: Dict[str, Any] = {
    "results": [
        {
            "url": "https://example.com/",
            "status": "success",
            "resolved_mode": "standard",
            "title": "Example Domain",
            "full_content": "# Example Domain\n\nThis domain is for use in examples.",
            "highlights": None,
            "time_published": None,
            "time_last_crawled": "2026-07-03T09:35:43Z",
            "page_structure": {"primary": "Content Page", "secondary": None},
            "category": {"primary": "Computers, Electronics & Technology", "secondary": None},
            "favicon": None,
        },
        {
            "url": "https://nonexistent.invalid/",
            "status": "failed",
            "error_message": "fetch failed",
        },
    ]
}
MIXED_META: Dict[str, Any] = {
    "usage": {
        "total_urls": 2,
        "successful_urls": 1,
        "successful_by_mode": {"standard_urls": 1, "advanced_urls": 0},
    },
    "latency": 1832,
    "warning": "",
}


def _response(data: Any = None, meta: Any = None, request_id: Any = "rid") -> octen.ExtractResponse:
    extra = {} if request_id is None else {"request_id": request_id}
    return octen.ExtractResponse(
        code=0,
        msg="success",
        data=MIXED_DATA if data is None else data,
        meta=MIXED_META if meta is None else meta,
        **extra,
    )


def _tool(**kwargs: Any) -> OctenExtract:
    return OctenExtract(octen_api_key=SecretStr(FAKE_API_KEY), **kwargs)


@pytest.fixture
def sync_client() -> Any:
    with patch("octen.Octen") as mock_cls:
        mock_cls.return_value.extract.extract.return_value = _response()
        yield mock_cls


@pytest.fixture
def async_client() -> Any:
    with patch("octen.AsyncOcten") as mock_cls:
        mock_cls.return_value = AsyncMock()
        mock_cls.return_value.extract.extract.return_value = _response()
        yield mock_cls


def _extract(mock_cls: MagicMock) -> MagicMock:
    return mock_cls.return_value.extract.extract  # type: ignore[no-any-return]


# --- Schema and description ------------------------------------------------------------


def test_name_and_model_args() -> None:
    tool = _tool()
    assert tool.name == "octen_extract"
    assert set(tool.args) == {"urls", "query"}


def test_default_cap_is_five() -> None:
    assert _tool().max_urls == 5


def test_description_states_cap_and_per_url_failures() -> None:
    description = _tool(max_urls=3).description
    assert "at most 3 URLs" in description
    assert "failed" in description and "error_message" in description


def test_description_names_no_developer_option() -> None:
    description = _tool().description
    for name in (*OPTIONS, "max_urls", "timeout", "advanced"):
        assert name not in description, name


def test_custom_description_kept() -> None:
    assert _tool(description="Read pages.").description == "Read pages."


# --- Requests --------------------------------------------------------------------------


def test_model_args_and_developer_options_forwarded(sync_client: MagicMock) -> None:
    _tool(timeout=100.0, **OPTIONS).invoke({"urls": URLS, "query": "pricing"})

    sync_client.assert_called_once_with(api_key=FAKE_API_KEY, retry_after_send=False)
    _extract(sync_client).assert_called_once_with(
        urls=URLS, query="pricing", request_timeout=100.0, **SDK_OPTIONS
    )
    sync_client.return_value.close.assert_called_once()


def test_unset_options_not_sent(sync_client: MagicMock) -> None:
    _tool().invoke({"urls": [URL]})

    _extract(sync_client).assert_called_once_with(
        urls=[URL], request_timeout=DEFAULT_REQUEST_TIMEOUT
    )


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, 120),
        ({"url_timeout": 1}, 91),
        ({"url_timeout": 60}, 150),
        ({"url_timeout": 60, "timeout": 20.0}, 20.0),
        ({"timeout": 7}, 7),
    ],
    ids=["default", "url_timeout-1", "url_timeout-60", "explicit-short", "explicit"],
)
def test_request_timeout(options: Dict[str, Any], expected: float, sync_client: MagicMock) -> None:
    _tool(**options).invoke({"urls": [URL]})

    _, kwargs = _extract(sync_client).call_args
    assert kwargs["request_timeout"] == expected


def _wire(tool: OctenExtract, args: Dict[str, Any]) -> List[Tuple[str, str, Any, Any]]:
    seen: List[Tuple[str, str, Any, Any]] = []

    def fake(
        self: Any, method: str, endpoint: str, timeout: Any = None, json: Any = None, **kw: Any
    ) -> Dict[str, Any]:
        seen.append((method, endpoint, json, timeout))
        return {
            "code": 0,
            "msg": "success",
            "request_id": "rid",
            "data": MIXED_DATA,
            "meta": MIXED_META,
        }

    with patch("octen.core.http_client.HTTPClient.request", fake):
        tool.invoke(args)
    return seen


def test_wire_default_body_is_only_the_urls() -> None:
    seen = _wire(_tool(), {"urls": [URL]})
    assert seen == [("POST", "/extract", {"urls": [URL]}, 120)]


def test_wire_all_options() -> None:
    seen = _wire(_tool(timeout=100.0, **OPTIONS), {"urls": URLS, "query": "pricing"})
    expected_body = {
        "urls": URLS,
        "format": "text",
        "max_age_seconds": 600,
        "query": "pricing",
        "timeout": 45,
        "include_images": True,
        "include_videos": True,
        "include_audio": True,
        "mode": "advanced",
        "include_links": {"scope": "prefer_external", "max_links": 10},
    }
    assert json.dumps(seen) == json.dumps([("POST", "/extract", expected_body, 100.0)])


def test_empty_include_links_turns_links_on(sync_client: MagicMock) -> None:
    _tool(include_links={}).invoke({"urls": [URL]})

    _, kwargs = _extract(sync_client).call_args
    assert kwargs["include_links"] == {}


def test_forwarded_names_are_sdk_parameters() -> None:
    from octen.async_client import AsyncExtractResource
    from octen.resources.extract import ExtractResource

    tool = _tool()
    sent = {"urls", "query", "timeout", "request_timeout"} | (
        set(tool._developer_options) - {"url_timeout"}
    )
    for cls in (ExtractResource, AsyncExtractResource):
        params = set(inspect.signature(cls.extract).parameters)
        assert sent <= params, (cls.__name__, sent - params)


def test_include_links_is_copied(sync_client: MagicMock) -> None:
    links = {"scope": "prefer_internal", "max_links": 50}
    tool = _tool(include_links=links)
    links["max_links"] = 999

    tool.invoke({"urls": [URL]})

    _, kwargs = _extract(sync_client).call_args
    assert kwargs["include_links"] == {"scope": "prefer_internal", "max_links": 50}


# --- Model input that never sends a request --------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [{**{k: v}} for k, v in OPTIONS.items()]
    + [{"max_urls": 20}, {"timeout": 5}, {"request_timeout": 5}],
    ids=lambda e: next(iter(e)),
)
def test_model_cannot_set_developer_options(extra: Dict[str, Any], sync_client: MagicMock) -> None:
    result = _tool().invoke({"urls": [URL], **extra})

    assert isinstance(result, str)
    assert next(iter(extra)) in result
    sync_client.assert_not_called()


def test_more_urls_than_cap_never_sends(sync_client: MagicMock) -> None:
    urls = [f"https://example.com/{i}" for i in range(3)]

    result = _tool(max_urls=2).invoke({"urls": urls})

    assert isinstance(result, str)
    assert "3 URLs" in result and "at most 2" in result and "split" in result
    sync_client.assert_not_called()


def test_default_cap_rejects_six(sync_client: MagicMock) -> None:
    result = _tool().invoke({"urls": [f"https://example.com/{i}" for i in range(6)]})

    assert isinstance(result, str) and "at most 5" in result
    sync_client.assert_not_called()


@pytest.mark.parametrize("cap", [1, 5, 20])
def test_exactly_cap_is_sent(cap: int, sync_client: MagicMock) -> None:
    urls = [f"https://example.com/{i}" for i in range(cap)]

    _tool(max_urls=cap).invoke({"urls": urls})

    _extract(sync_client).assert_called_once_with(
        urls=urls, request_timeout=DEFAULT_REQUEST_TIMEOUT
    )


@pytest.mark.asyncio
async def test_more_urls_than_cap_never_sends_async(async_client: MagicMock) -> None:
    result = await _tool(max_urls=1).ainvoke({"urls": URLS})

    assert isinstance(result, str) and "2 URLs" in result and "at most 1" in result
    async_client.assert_not_called()


@pytest.mark.parametrize(
    ("args", "named"),
    [
        ({"urls": []}, "urls"),
        ({"urls": [""]}, "urls"),
        ({"urls": ["https://example.com/" + "a" * 2029]}, "urls"),
        ({"urls": URL}, "urls"),
        ({"urls": [URL], "query": "x" * 501}, "query"),
        ({"urls": [URL], "query": ""}, "query"),
        ({"url": [URL]}, "url"),
    ],
    ids=[
        "empty-list",
        "empty-url",
        "url-2049",
        "url-not-list",
        "query-501",
        "query-empty",
        "misspelt-urls",
    ],
)
def test_invalid_model_input_goes_to_model(
    args: Dict[str, Any], named: str, sync_client: MagicMock
) -> None:
    result = _tool().invoke(args)

    assert isinstance(result, str)
    assert result.startswith("Invalid tool input") and named in result
    sync_client.assert_not_called()


def test_url_of_2048_chars_is_sent(sync_client: MagicMock) -> None:
    url = "https://example.com/" + "a" * 2028
    assert len(url) == 2048

    _tool().invoke({"urls": [url]})

    _extract(sync_client).assert_called_once()


def test_plain_string_input_goes_to_model(sync_client: MagicMock) -> None:
    result = _tool().invoke(URL)

    assert isinstance(result, str) and "urls" in result
    sync_client.assert_not_called()


# --- Construction ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "bad_name"),
    [
        ({"max_urls": 0}, "max_urls"),
        ({"max_urls": 21}, "max_urls"),
        ({"max_urls": True}, "max_urls"),
        ({"max_urls": "5"}, "max_urls"),
        ({"max_urls": 5.0}, "max_urls"),
        ({"max_urls": None}, "max_urls"),
        ({"mode": "turbo"}, "mode"),
        ({"format": "html"}, "format"),
        ({"max_age_seconds": 299}, "max_age_seconds"),
        ({"max_age_seconds": 31_536_001}, "max_age_seconds"),
        ({"max_age_seconds": True}, "max_age_seconds"),
        ({"url_timeout": 0}, "url_timeout"),
        ({"url_timeout": 61}, "url_timeout"),
        ({"url_timeout": 30.5}, "url_timeout"),
        ({"include_images": "yes"}, "include_images"),
        ({"include_videos": 1}, "include_videos"),
        ({"include_audio": "true"}, "include_audio"),
        ({"include_links": {"scop": "prefer_internal"}}, "include_links"),
        ({"include_links": {"scope": "all"}}, "include_links"),
        ({"include_links": {"max_links": 0}}, "include_links"),
        ({"include_links": {"max_links": 1001}}, "include_links"),
        ({"include_links": True}, "include_links"),
        ({"timeout": 0}, "timeout"),
        ({"timeout": float("inf")}, "timeout"),
        ({"max_url": 3}, "max_url"),
    ],
    ids=repr,
)
def test_invalid_developer_option_raises_at_construction(
    options: Dict[str, Any], bad_name: str, sync_client: MagicMock
) -> None:
    with pytest.raises(ValueError, match=bad_name):
        _tool(**options)
    sync_client.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"max_urls": 1},
        {"max_urls": 20},
        {"max_age_seconds": 300},
        {"max_age_seconds": 31_536_000},
        {"url_timeout": 1},
        {"url_timeout": 60},
        {"mode": "standard"},
        {"mode": "auto"},
        {"format": "markdown"},
        {"include_links": {}},
        {"include_links": {"scope": "prefer_internal", "max_links": 1000}},
        OPTIONS,
    ],
    ids=repr,
)
def test_valid_developer_options_accepted(options: Dict[str, Any]) -> None:
    _tool(**options)


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCTEN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OCTEN_API_KEY"):
        OctenExtract()


def test_key_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCTEN_API_KEY", "from-env")
    assert OctenExtract().octen_api_key.get_secret_value() == "from-env"  # type: ignore[union-attr]


def test_option_error_does_not_echo_key() -> None:
    secret = "sk-live-SECRET-0123456789"
    with pytest.raises(ValueError) as exc_info:
        OctenExtract(octen_api_key=secret, max_urls=99)
    assert secret not in str(exc_info.value) and secret not in repr(exc_info.value)


# --- Results ---------------------------------------------------------------------------


def test_mixed_batch_keeps_every_item(sync_client: MagicMock) -> None:
    result = _tool().invoke({"urls": ["https://example.com/", "https://nonexistent.invalid/"]})

    assert result == {"data": MIXED_DATA, "request_id": "rid"}
    failed = result["data"]["results"][1]
    assert failed == {
        "url": "https://nonexistent.invalid/",
        "status": "failed",
        "error_message": "fetch failed",
    }
    assert result["data"]["results"][0]["resolved_mode"] == "standard"


def test_content_is_not_truncated(sync_client: MagicMock) -> None:
    long_item = {
        "url": URL,
        "status": "success",
        "resolved_mode": "standard",
        "full_content": "x" * 500_000,
    }
    _extract(sync_client).return_value = _response(data={"results": [long_item]})

    result = _tool().invoke({"urls": [URL]})

    assert result["data"]["results"][0]["full_content"] == "x" * 500_000


def test_warning_kept(sync_client: MagicMock) -> None:
    _extract(sync_client).return_value = _response(meta={**MIXED_META, "warning": "w1; w2"})

    result = _tool().invoke({"urls": [URL]})

    assert result == {"data": MIXED_DATA, "request_id": "rid", "warning": "w1; w2"}


@pytest.mark.asyncio
async def test_async_uses_async_client_and_closes(async_client: MagicMock) -> None:
    result = await _tool(url_timeout=10, timeout=40.0).ainvoke({"urls": URLS, "query": "q"})

    async_client.assert_called_once_with(api_key=FAKE_API_KEY, retry_after_send=False)
    _extract(async_client).assert_awaited_once_with(
        urls=URLS, query="q", timeout=10, request_timeout=40.0
    )
    async_client.return_value.close.assert_awaited_once()
    assert result == {"data": MIXED_DATA, "request_id": "rid"}


# --- Errors ----------------------------------------------------------------------------

MAY_HAVE_BEEN_SENT = "may already have reached the server and been billed"
TRANSIENT = [
    OctenRateLimitError(request_id="rid-429"),
    OctenAPIError("upstream unavailable", status_code=503),
    OctenAPIError("bad gateway", status_code=502),
    OctenAPIError("Invalid params", status_code=400),
    OctenAPIError("API returned code 500", status_code=200),
    OctenTimeoutError(timeout=120.0, request_sent=True),
    OctenConnectionError("connection reset", request_sent=None),
    OctenConnectionError("connection refused", request_sent=False),
]
CONFIG = [
    OctenAuthenticationError(),
    OctenAPIError("insufficient balance", status_code=403),
    OctenValidationError("extract request failed validation"),
    RuntimeError("unexpected"),
]


@pytest.mark.parametrize("exc", TRANSIENT, ids=repr)
def test_transient_errors_go_to_model(exc: Exception, sync_client: MagicMock) -> None:
    _extract(sync_client).side_effect = exc

    result = _tool().invoke({"urls": [URL]})

    assert isinstance(result, str) and str(exc) in result
    sync_client.return_value.close.assert_called_once()


@pytest.mark.parametrize(
    "exc",
    [
        OctenTimeoutError(timeout=120.0, request_sent=True),
        OctenAPIError("bad gateway", status_code=502),
        OctenAPIError("API returned code 500", status_code=200),
    ],
    ids=repr,
)
def test_possibly_billed_failures_warn_against_retry(
    exc: Exception, sync_client: MagicMock
) -> None:
    _extract(sync_client).side_effect = exc

    result = _tool().invoke({"urls": [URL]})

    assert isinstance(result, str) and MAY_HAVE_BEEN_SENT in result


@pytest.mark.parametrize("exc", CONFIG, ids=repr)
def test_config_errors_raise(exc: Exception, sync_client: MagicMock) -> None:
    _extract(sync_client).side_effect = exc

    with pytest.raises(type(exc)):
        _tool().invoke({"urls": [URL]})
    sync_client.return_value.close.assert_called_once()


@pytest.mark.asyncio
async def test_async_transient_error_goes_to_model(async_client: MagicMock) -> None:
    exc = OctenTimeoutError(timeout=120.0, request_sent=True)
    _extract(async_client).side_effect = exc

    result = await _tool().ainvoke({"urls": [URL]})

    assert isinstance(result, str) and MAY_HAVE_BEEN_SENT in result
    async_client.return_value.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_config_error_raises(async_client: MagicMock) -> None:
    _extract(async_client).side_effect = OctenAuthenticationError()

    with pytest.raises(OctenAuthenticationError):
        await _tool().ainvoke({"urls": [URL]})
    async_client.return_value.close.assert_awaited_once()
