"""Unit tests for the Image Search and Design Search tools (mocked SDK or HTTP layer)."""

from __future__ import annotations

import builtins
import copy
import inspect
import json
import logging
import sys
from typing import Any, Dict, List, Tuple, Type, Union
from unittest.mock import AsyncMock, MagicMock, patch

import octen
import pytest
from langchain_core.tools import ToolException
from octen import (
    OctenAPIError,
    OctenAuthenticationError,
    OctenConnectionError,
    OctenRateLimitError,
    OctenTimeoutError,
    OctenValidationError,
)
from pydantic import SecretStr

from langchain_octen import OctenDesignSearch, OctenImageSearch

FAKE_API_KEY = "test-api-key-12345"
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Example.jpg"
GROUP_ID = "deck_3b7f0e92c4a1d85f6e2b9c07a3d4f1e8b5c26a9d0f7e3b1c4a8d2e6f9b0c5a7d"
MAY_HAVE_BEEN_SENT = "may already have reached the server and been billed"

Tool = Union[OctenImageSearch, OctenDesignSearch]

# Result fields from https://docs.octen.ai/api-reference/image-search (ImageSearchResult).
IMAGE_ITEM: Dict[str, Any] = {
    "title": "Golden Gate Bridge",
    "url": "https://cdn.example.com/ggb.jpg",
    "source_page": "https://www.example.com/topic/Golden-Gate-Bridge",
    "description": "The bridge at sunset.",
    "width": 1600,
    "height": 1109,
    "thumbnail": "https://cdn.octen.ai/thumb/ggb.jpg",
}
IMAGE_DATA: Dict[str, Any] = {
    "results": [
        # Undocumented and empty fields reach the model unchanged, as in the other tools.
        {**IMAGE_ITEM, "summary": "old design corpus field", "html_snippet": "<div/>"},
        {**IMAGE_ITEM, "url": "https://cdn.example.com/2.jpg", "description": ""},
    ]
}

# Shapes from the example in https://docs.octen.ai/api-reference/design-search.
GROUP: Dict[str, Any] = {
    "group_id": GROUP_ID,
    "type": "deck",
    "name": "Example Startup Seed Deck",
    "position": 5,
    "total": 14,
    "master": {"majorFont": "Calibri Light", "palette": {"accent1": "#156082"}},
}
DESIGN_IMAGE: Dict[str, Any] = {
    "type": "slide",
    "title": "Market opportunity",
    "description": "A slide with three nested circles.",
    "url": "https://cdn.octen.ai/design/deck/p-005.png",
    "thumbnail": "https://cdn.octen.ai/design/thumb/9e5b.jpg",
    "source_page": "https://www.example-startup.com/files/seed-deck.pptx",
    "width": 1920,
    "height": 1080,
    "group": GROUP,
    "design_facts": {"styleFacts": {"fontFamilies": ["Calibri"]}},
    "snippet": {"format": "html", "code": "<section>...</section>"},
}
APP_SCREEN: Dict[str, Any] = {
    "type": "app_screen",
    "title": "Onboarding",
    "url": "https://cdn.octen.ai/design/app/1.png",
    "thumbnail": "https://cdn.octen.ai/design/thumb/1.jpg",
    "width": 1170,
    "height": 2532,
}
WEB_RESULT: Dict[str, Any] = {
    "title": "Pricing Page Design: 12 Patterns That Convert",
    "url": "https://www.example-design-blog.com/pricing-page-patterns",
    "highlight": "Three plans side by side.",
    "full_content": "# Pricing pages",
    "authors": "Example Design Blog",
    "time_published": "2026-09-08T00:00:00Z",
    "time_last_crawled": "2026-11-02T06:15:00Z",
    "cover_image": {"url": "https://img.example.com/cover.png", "description": "Cards"},
}
DESIGN_DATA: Dict[str, Any] = {
    "query": "pricing section",
    "images": [
        {**DESIGN_IMAGE, "group": {**GROUP, "extra": 1}, "internal_score": 0.9},
        {**APP_SCREEN, "source_page": None, "group": None, "snippet": None},
    ],
    "results": [
        {
            **WEB_RESULT,
            "favicon": "https://www.example-design-blog.com/favicon.ico",
            "images": [{"url": "https://img.example.com/1.png"}],
        }
    ],
    "specs": {"undocumented": True},
}


def _image_response(data: Any = None, meta: Any = None, request_id: Any = "rid") -> Any:
    extra = {} if request_id is None else {"request_id": request_id}
    return octen.ImageSearchResponse(
        code=0,
        msg="success",
        data=IMAGE_DATA if data is None else data,
        meta={"usage": {"num_search_queries": 1}} if meta is None else meta,
        **extra,
    )


def _design_response(data: Any = None, meta: Any = None, request_id: Any = "rid") -> Any:
    return octen.DesignSearchResponse(
        code=0,
        msg="success",
        request_id=request_id,
        data=DESIGN_DATA if data is None else data,
        meta={"usage": {"num_search_queries": 1}} if meta is None else meta,
    )


def _image(**kwargs: Any) -> OctenImageSearch:
    return OctenImageSearch(octen_api_key=SecretStr(FAKE_API_KEY), **kwargs)


def _design(**kwargs: Any) -> OctenDesignSearch:
    return OctenDesignSearch(octen_api_key=SecretStr(FAKE_API_KEY), **kwargs)


@pytest.fixture
def sync_client() -> Any:
    with patch("octen.Octen") as mock_cls:
        mock_cls.return_value.image_search.search.return_value = _image_response()
        mock_cls.return_value.design_search.search.return_value = _design_response()
        yield mock_cls


@pytest.fixture
def async_client() -> Any:
    with patch("octen.AsyncOcten") as mock_cls:
        mock_cls.return_value = AsyncMock()
        mock_cls.return_value.image_search.search.return_value = _image_response()
        mock_cls.return_value.design_search.search.return_value = _design_response()
        yield mock_cls


def _img_search(mock_cls: MagicMock) -> MagicMock:
    return mock_cls.return_value.image_search.search  # type: ignore[no-any-return]


def _des_search(mock_cls: MagicMock) -> MagicMock:
    return mock_cls.return_value.design_search.search  # type: ignore[no-any-return]


def _wire(tool: Tool, args: Any, data: Dict[str, Any]) -> List[Tuple[str, str, Any, Any]]:
    seen: List[Tuple[str, str, Any, Any]] = []

    def fake(
        self: Any, method: str, endpoint: str, timeout: Any = None, json: Any = None, **kw: Any
    ) -> Dict[str, Any]:
        seen.append((method, endpoint, json, timeout))
        return {"code": 0, "msg": "success", "request_id": "rid", "data": data, "meta": {}}

    with patch("octen.core.http_client.HTTPClient.request", fake):
        tool.invoke(args)
    return seen


# --- Image Search: schema and description -----------------------------------------------

IMAGE_OPTIONS: Dict[str, Any] = {"count": 3, "safesearch": "off"}


def test_image_name_and_model_args() -> None:
    tool = _image()
    assert tool.name == "octen_image_search"
    assert set(tool.args) == {"query", "image_url", "include_domains", "exclude_domains"}


def test_image_description_names_inputs_and_no_developer_option() -> None:
    description = _image().description
    assert "`query`" in description and "`image_url`" in description
    assert "exactly one" in description.lower()
    for name in ("count", "safesearch", "timeout", "base64"):
        assert name not in description, name


# --- Image Search: requests -------------------------------------------------------------


def test_image_text_query_and_options_forwarded(sync_client: MagicMock) -> None:
    _image(timeout=9.0, **IMAGE_OPTIONS).invoke(
        {"query": "golden gate", "include_domains": ["a.com"], "exclude_domains": ["b.com"]}
    )

    sync_client.assert_called_once_with(api_key=FAKE_API_KEY, retry_after_send=False)
    _img_search(sync_client).assert_called_once_with(
        query="golden gate",
        include_domains=["a.com"],
        exclude_domains=["b.com"],
        count=3,
        safesearch="off",
        timeout=9.0,
    )
    sync_client.return_value.close.assert_called_once()


def test_image_url_forwarded(sync_client: MagicMock) -> None:
    _image().invoke({"image_url": IMAGE_URL})

    _img_search(sync_client).assert_called_once_with(image_url=IMAGE_URL)


def test_image_unset_options_not_sent(sync_client: MagicMock) -> None:
    _image().invoke({"query": "q"})

    _img_search(sync_client).assert_called_once_with(query="q")


def test_image_plain_string_is_the_query(sync_client: MagicMock) -> None:
    _image().invoke("golden gate")

    _img_search(sync_client).assert_called_once_with(query="golden gate")


def test_image_wire_text_body() -> None:
    seen = _wire(_image(), {"query": "q"}, IMAGE_DATA)
    assert seen == [("POST", "/image-search", {"inputs": [{"type": "text", "data": "q"}]}, None)]


def test_image_wire_url_body() -> None:
    seen = _wire(_image(), {"image_url": IMAGE_URL}, IMAGE_DATA)
    body = {"inputs": [{"type": "image", "url": IMAGE_URL}]}
    assert seen == [("POST", "/image-search", body, None)]


def test_image_wire_all_options() -> None:
    args = {"query": "q", "include_domains": ["a.com"], "exclude_domains": ["b.com"]}
    seen = _wire(_image(timeout=9.0, **IMAGE_OPTIONS), args, IMAGE_DATA)
    body = {
        "inputs": [{"type": "text", "data": "q"}],
        "count": 3,
        "include_domains": ["a.com"],
        "exclude_domains": ["b.com"],
        "safesearch": "off",
    }
    assert json.dumps(seen) == json.dumps([("POST", "/image-search", body, 9.0)])


def test_image_ten_domains_sent(sync_client: MagicMock) -> None:
    domains = [f"d{i}.com" for i in range(10)]
    _image().invoke({"query": "q", "include_domains": domains, "exclude_domains": domains})

    _img_search(sync_client).assert_called_once_with(
        query="q", include_domains=domains, exclude_domains=domains
    )


def test_image_forwarded_names_are_sdk_parameters() -> None:
    from octen.async_client import AsyncImageSearchResource
    from octen.resources.image_search import ImageSearchResource

    sent = {"query", "image_url", "include_domains", "exclude_domains", "timeout"} | set(
        _image()._developer_options
    )
    for cls in (ImageSearchResource, AsyncImageSearchResource):
        params = set(inspect.signature(cls.search).parameters)
        assert sent <= params, (cls.__name__, sent - params)


# --- Image Search: model input that never sends a request -------------------------------


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"query": "q", "image_url": IMAGE_URL}, "not both"),
        ({}, "Pass `query`"),
        ({"include_domains": ["a.com"]}, "Pass `query`"),
    ],
    ids=["both", "neither", "filters-only"],
)
def test_image_exactly_one_input(
    args: Dict[str, Any], expected: str, sync_client: MagicMock
) -> None:
    result = _image().invoke(args)

    assert isinstance(result, str)
    assert result.startswith("Invalid tool input") and expected in result, result
    assert "`image_url`" in result
    sync_client.assert_not_called()


@pytest.mark.asyncio
async def test_image_exactly_one_input_async(async_client: MagicMock) -> None:
    result = await _image().ainvoke({"query": "q", "image_url": IMAGE_URL})

    assert isinstance(result, str) and "not both" in result
    async_client.assert_not_called()


@pytest.mark.parametrize(
    ("args", "named"),
    [
        ({"query": ""}, "query"),
        ({"query": "x" * 501}, "query"),
        ({"image_url": ""}, "image_url"),
        ({"image_url": "ftp://example.com/a.jpg"}, "image_url"),
        ({"image_url": "http://example.com/a.jpg"}, "image_url"),
        ({"image_url": "https://1.2.3.4/a.png"}, "image_url"),
        ({"image_url": "https://[2001:db8::1]/a.png"}, "image_url"),
        ({"image_url": "https://[2001:db8::1]:8443/a.png"}, "image_url"),
        ({"image_url": "data:image/png;base64,iVBORw0KGgo="}, "image_url"),
        ({"image_url": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"}, "image_url"),
        ({"image_url": "https://"}, "image_url"),
        ({"image_url": "https://:443/photo.png"}, "image_url"),
        ({"image_url": "https://[example.com/x.jpg"}, "image_url"),
        ({"image_url": "https://example.com:abc/x.png"}, "image_url"),
        ({"image_url": "https://example.com:70000/x.png"}, "image_url"),
        ({"image_url": "https://exa mple.com/photo.png"}, "image_url"),
        ({"image_url": "https://example.com/a b.png"}, "image_url"),
        ({"image_url": " https://example.com/a.png"}, "image_url"),
        ({"image_url": "https://example.com/a.png\n"}, "image_url"),
        ({"image_url": "\x00https://example.com/a.png"}, "image_url"),
        ({"image_url": "https://example.com/a\x7f.png"}, "image_url"),
        ({"image_url": "https:///example.com/a.png"}, "image_url"),
        ({"image_url": "https://\\example.com/a.png"}, "image_url"),
        ({"image_url": "https://exa<mple.com/a.png"}, "image_url"),
        ({"image_url": "https://exa%zzmple.com/a.png"}, "image_url"),
        ({"image_url": "https://example.com/a%zz.png"}, "image_url"),
        ({"image_url": "https://example.com/a%4.png"}, "image_url"),
        ({"image_url": "https://example.com/photo.png#50%"}, "image_url"),
        ({"image_url": "https://example.com/photo.png?q=1#50%"}, "image_url"),
        ({"image_url": "https://example.com/photo%;ab"}, "image_url"),
        ({"image_url": "https://ex%zzample.com/a.png"}, "image_url"),
        # A host escape must encode a non-ASCII byte or be %25.
        ({"image_url": "https://%65xample.com/photo.jpg"}, "image_url"),
        ({"image_url": "https://example%2ecom/photo.png"}, "image_url"),
        ({"image_url": "https://ex%2Dample.com/a.png"}, "image_url"),
        # A "?" inside the fragment does not start a query.
        ({"image_url": "https://example.com/photo.png#a?b=50%"}, "image_url"),
        ({"image_url": "https://[2001:zz::1]/a.png"}, "image_url"),
        ({"image_url": "https://[2001:4860::1]junk:443/a.png"}, "image_url"),
        ({"image_url": "https://[2001:4860::1]:44x/a.png"}, "image_url"),
        ({"image_url": "https://ex[am]ple.com/a.png"}, "image_url"),
        ({"image_url": "https://a@b@example.com/a.png"}, "image_url"),
        ({"image_url": 'https://exa"mple.com/a.png'}, "image_url"),
        ({"image_url": "https://exa`mple.com/a.png"}, "image_url"),
        ({"query": "   "}, "query"),
        ({"query": "\t\n"}, "query"),
        ({"query": "q", "include_domains": [f"d{i}.com" for i in range(11)]}, "include_domains"),
        ({"query": "q", "exclude_domains": [f"d{i}.com" for i in range(11)]}, "exclude_domains"),
        ({"query": "q", "include_domains": "a.com"}, "include_domains"),
        ({"imageurl": IMAGE_URL}, "imageurl"),
    ],
    ids=[
        "query-empty",
        "query-501",
        "url-empty",
        "url-ftp",
        "url-http",
        "url-ipv4",
        "url-ipv6",
        "url-ipv6-port",
        "url-data-uri",
        "url-base64",
        "url-no-host",
        "url-port-only",
        "url-unparseable",
        "url-port-not-numeric",
        "url-port-out-of-range",
        "url-space-in-host",
        "url-space-in-path",
        "url-leading-space",
        "url-trailing-newline",
        "url-leading-nul",
        "url-del-char",
        "url-empty-authority",
        "url-backslash-authority",
        "url-bad-host-char",
        "url-bad-escape-in-host",
        "url-bad-escape-in-path",
        "url-short-escape",
        "url-bad-escape-in-fragment",
        "url-query-then-bad-fragment",
        "url-bad-escape-before-params",
        "url-bad-escape-in-host-2",
        "url-host-escape-ascii-letter",
        "url-host-escape-dot",
        "url-host-escape-dash",
        "url-question-mark-in-fragment",
        "url-bad-ipv6",
        "url-ipv6-trailing-junk",
        "url-ipv6-bad-port",
        "url-brackets-in-host",
        "url-two-at-signs",
        "url-double-quote",
        "url-backtick",
        "query-blank",
        "query-whitespace",
        "include-11",
        "exclude-11",
        "domains-not-list",
        "misspelt-image_url",
    ],
)
def test_image_invalid_model_input_goes_to_model(
    args: Dict[str, Any], named: str, sync_client: MagicMock
) -> None:
    result = _image().invoke(args)

    assert isinstance(result, str)
    assert result.startswith("Invalid tool input") and named in result, result
    sync_client.assert_not_called()


@pytest.mark.parametrize(
    "extra",
    [
        {"count": 10},
        {"safesearch": "off"},
        {"timeout": 5},
        {"topic": "design"},
        {"inputs": [{"type": "text", "data": "q"}]},
        {"image_data": "iVBORw0KGgo="},
    ],
    ids=lambda e: next(iter(e)),
)
def test_image_model_cannot_set_developer_options(
    extra: Dict[str, Any], sync_client: MagicMock
) -> None:
    result = _image().invoke({"query": "q", **extra})

    assert isinstance(result, str) and next(iter(extra)) in result
    sync_client.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "https://:443/photo.png",
        "https://[example.com/x.jpg",
        "https://",
        "https://example.com:abc/x.png",
        "https://example.com:70000/x.png",
        "http://example.com/a.png",
        "https://1.2.3.4/a.png",
        "https://[2001:db8::1]/a.png",
    ],
    ids=repr,
)
def test_image_url_without_host_gets_guidance(url: str, sync_client: MagicMock) -> None:
    result = _image().invoke({"image_url": url})

    assert isinstance(result, str) and "public https URL of an image" in result, result
    sync_client.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "HTTPS://EXAMPLE.COM/A.PNG",
        "https://example.com:8443/a.png",
        "https://example.com/a%20b.png",
        "https://cdn_1.example.com/a.png",
        # An internationalized host with combining marks (Devanagari vowel signs).
        "https://\u0930\u093e\u091c\u092d\u093e\u0937\u093e.\u092d\u093e\u0930\u0924/a.png",
        # The same kind of host in its punycode (IDNA ASCII) form.
        "https://xn--81b8ah4a6acb.xn--h2brj9c/a.png",
        # An Arabic label ending in a digit: valid IDNA2008, refused by Python's IDNA2003.
        "https://\u0645\u062b\u0627\u06441.\u0645\u0635\u0631/a.png",
        "https://xn--1-ymcl5hc.xn--wgbh1c/a.png",
        "https://user:pass@example.com/a.png",
        # A non-ASCII space (U+00A0) in the path is left to the URL parser and sent as is.
        "https://example.com/caf\u00e9\u00a0menu.png",
        # The server keeps only the query raw, so a bare % is fine there.
        "https://example.com/photo.png?label=50%",
        "https://example.com/photo.png?q=%zz#ok",
        "https://example.com/photo.png#50%25",
        "https://caf%C3%A9.com/a.png",
        "https://ex%25ample.com/a.png",
        "https://us%65r:p%2e@example.com/a.png",
    ],
    ids=[
        "upper",
        "port",
        "escape",
        "underscore",
        "idn-marks",
        "punycode",
        "arabic-digit",
        "arabic-punycode",
        "userinfo",
        "nbsp-in-path",
        "bare-percent-in-query",
        "bad-escape-in-query-then-fragment",
        "escape-in-fragment",
        "host-escape-non-ascii",
        "host-escape-percent",
        "userinfo-ascii-escapes",
    ],
)
def test_image_https_urls_accepted(url: str, sync_client: MagicMock) -> None:
    _image().invoke({"image_url": url})

    _img_search(sync_client).assert_called_once_with(image_url=url)


# --- Image Search: construction ---------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "bad_name"),
    [
        ({"count": 0}, "count"),
        ({"count": 11}, "count"),
        ({"count": True}, "count"),
        ({"count": "5"}, "count"),
        ({"count": 5.0}, "count"),
        ({"safesearch": "moderate"}, "safesearch"),
        ({"safesearch": True}, "safesearch"),
        ({"timeout": 0}, "timeout"),
        ({"timeout": float("nan")}, "timeout"),
        ({"cuont": 3}, "cuont"),
        ({"topic": "design"}, "topic"),
        ({"include_domains": ["a.com"]}, "include_domains"),
        ({"response_format": "content_and_artifact"}, "response_format"),
    ],
    ids=repr,
)
def test_image_invalid_developer_option_raises(
    options: Dict[str, Any], bad_name: str, sync_client: MagicMock
) -> None:
    with pytest.raises(ValueError, match=bad_name):
        _image(**options)
    sync_client.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [{"count": 1}, {"count": 10}, {"safesearch": "strict"}, IMAGE_OPTIONS, {"timeout": 0.5}],
    ids=repr,
)
def test_image_valid_developer_options_accepted(options: Dict[str, Any]) -> None:
    _image(**options)


# --- Design Search: schema and description ----------------------------------------------

IMAGES: Dict[str, Any] = {
    "count": 3,
    "design_facts": False,
    "snippet": {"enable": True, "max_tokens": 1000},
}
WEB: Dict[str, Any] = {
    "count": 2,
    "highlight": {"enable": True, "max_tokens": 300},
    "full_content": {"enable": True, "max_tokens": 2048},
    "include_domains": ["a.com"],
    "exclude_domains": ["b.com"],
    "language": ["en"],
    "time_basis": "published",
    "time_range": "year",
    "start_time": "2025-01-01T00:00:00Z",
    "end_time": "2025-06-30T23:59:59+08:00",
}


def test_design_name_and_model_args() -> None:
    tool = _design()
    assert tool.name == "octen_design_search"
    assert set(tool.args) == {"query", "group_id"}


@pytest.mark.parametrize(
    ("images", "expected"),
    [(None, "up to 5 images (the default)"), ({"count": 30}, "up to 30 images"),
     ({"design_facts": False}, "up to 5 images (the default)")],
    ids=["unset", "count-30", "count-unset"],
)  # fmt: skip
def test_design_description_states_group_image_limit(images: Any, expected: str) -> None:
    description = _design(images=images).description
    assert f"returns the first images of that group in group order, {expected}" in description
    assert "every image" not in description and "all images" not in description


def test_design_without_web_promises_no_web_results() -> None:
    description = _design(web_search={"enable": False}).description
    assert "`results`" not in description and "Web results" not in description
    assert "group_id" in description and "screenshot" in description


def test_design_with_both_parts_promises_both() -> None:
    description = _design().description
    assert "`results`" in description and "screenshot" in description


def test_design_with_both_parts_off_raises() -> None:
    with pytest.raises(ValueError, match="both"):
        _design(images={"enable": False}, web_search={"enable": False})


def test_design_without_images_offers_no_group_lookup() -> None:
    description = _design(images={"enable": False}).description
    assert "group_id" not in description and "group" not in description.lower()
    assert "`query`" in description


@pytest.mark.parametrize("images", [{"enable": False}, {"enable": False, "count": 3}], ids=repr)
def test_design_without_images_rejects_group_only_call(
    images: Dict[str, Any], sync_client: MagicMock
) -> None:
    tool = _design(images=images)
    result = tool.invoke({"group_id": GROUP_ID})

    # The schema has no group_id and requires query, so the model is told both.
    assert isinstance(result, str) and result.startswith("Invalid tool input"), result
    assert "query" in result and "group_id" in result, result
    # A direct call that bypasses the schema is refused too.
    with pytest.raises(ToolException, match="pass `query`"):
        tool._run(group_id=GROUP_ID)
    sync_client.assert_not_called()


@pytest.mark.asyncio
async def test_design_without_images_rejects_group_only_call_async(
    async_client: MagicMock,
) -> None:
    tool = _design(images={"enable": False})
    result = await tool.ainvoke({"group_id": GROUP_ID})

    assert isinstance(result, str) and "query" in result and "group_id" in result, result
    with pytest.raises(ToolException, match="pass `query`"):
        await tool._arun(group_id=GROUP_ID)
    async_client.assert_not_called()


def test_design_without_images_still_searches(sync_client: MagicMock) -> None:
    _design(images={"enable": False}).invoke({"query": "q"})

    _des_search(sync_client).assert_called_once_with(query="q", images={"enable": False})


def test_design_custom_description_kept() -> None:
    assert _design(images={"count": 3}, description="Find designs.").description == (
        "Find designs."
    )


def test_design_description_explains_group_follow_up() -> None:
    description = _design().description
    assert "`group_id`" in description and "group.group_id" in description
    for name in ("count", "web_search", "max_tokens", "timeout"):
        assert name not in description, name


# --- Design Search: requests ------------------------------------------------------------


def test_design_query_and_options_forwarded(sync_client: MagicMock) -> None:
    _design(images=IMAGES, web_search=WEB, timeout=9.0).invoke({"query": "pricing section"})

    sync_client.assert_called_once_with(api_key=FAKE_API_KEY, retry_after_send=False)
    _des_search(sync_client).assert_called_once_with(
        query="pricing section", images=IMAGES, web_search=WEB, timeout=9.0
    )
    sync_client.return_value.close.assert_called_once()


def test_design_unset_options_not_sent(sync_client: MagicMock) -> None:
    _design().invoke({"query": "q"})

    _des_search(sync_client).assert_called_once_with(query="q")


def test_design_group_id_alone(sync_client: MagicMock) -> None:
    _design().invoke({"group_id": GROUP_ID})

    _des_search(sync_client).assert_called_once_with(images={"group_id": GROUP_ID})


def test_design_group_id_merged_into_developer_images(sync_client: MagicMock) -> None:
    tool = _design(images={"count": 30})
    tool.invoke({"group_id": GROUP_ID, "query": "seed deck"})
    tool.invoke({"query": "seed deck"})

    first, second = _des_search(sync_client).call_args_list
    assert first.kwargs == {"query": "seed deck", "images": {"count": 30, "group_id": GROUP_ID}}
    # The merge must not leak into the tool's own option or the next call.
    assert second.kwargs == {"query": "seed deck", "images": {"count": 30}}
    assert tool.images == {"count": 30}


def test_design_empty_option_dicts_are_sent(sync_client: MagicMock) -> None:
    _design(images={}, web_search={}).invoke({"query": "q"})

    _des_search(sync_client).assert_called_once_with(query="q", images={}, web_search={})


def test_design_plain_string_is_the_query(sync_client: MagicMock) -> None:
    _design().invoke("pricing section")

    _des_search(sync_client).assert_called_once_with(query="pricing section")


def test_design_wire_default_body() -> None:
    seen = _wire(_design(), {"query": "q"}, DESIGN_DATA)
    assert seen == [("POST", "/design-search", {"query": "q"}, None)]


def test_design_wire_group_body() -> None:
    seen = _wire(_design(images={"count": 30}), {"group_id": GROUP_ID}, DESIGN_DATA)
    body = {"images": {"count": 30, "group_id": GROUP_ID}}
    assert seen == [("POST", "/design-search", body, None)]


def test_design_wire_all_options() -> None:
    seen = _wire(_design(images=IMAGES, web_search=WEB, timeout=9.0), {"query": "q"}, DESIGN_DATA)
    body = {
        "query": "q",
        "images": {
            "count": 3,
            "design_facts": False,
            "snippet": {"enable": True, "max_tokens": 1000},
        },
        "web_search": {
            "count": 2,
            "highlight": {"enable": True, "max_tokens": 300},
            "full_content": {"enable": True, "max_tokens": 2048},
            "include_domains": ["a.com"],
            "exclude_domains": ["b.com"],
            "language": ["en"],
            "time_basis": "published",
            "time_range": "year",
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-06-30T23:59:59+08:00",
        },
    }
    assert json.dumps(seen) == json.dumps([("POST", "/design-search", body, 9.0)])


def test_design_list_options_are_sent_in_full(sync_client: MagicMock) -> None:
    web = {"language": ["en", "de"], "include_domains": ("a.com",), "exclude_domains": ["b.com"]}
    tool = _design(web_search=web)

    tool.invoke({"query": "q"})
    tool.invoke({"query": "q"})

    for call in _des_search(sync_client).call_args_list:
        sent = call.kwargs["web_search"]
        assert list(sent["language"]) == ["en", "de"], sent
        assert list(sent["include_domains"]) == ["a.com"], sent
        assert list(sent["exclude_domains"]) == ["b.com"], sent


def test_design_option_dicts_are_copied(sync_client: MagicMock) -> None:
    images = copy.deepcopy(IMAGES)
    web = copy.deepcopy(WEB)
    tool = _design(images=images, web_search=web)
    images["count"] = 99
    images["snippet"]["max_tokens"] = 99_999
    web["include_domains"].append("c.com")

    tool.invoke({"query": "q"})

    _des_search(sync_client).assert_called_once_with(query="q", images=IMAGES, web_search=WEB)


def test_design_forwarded_names_are_sdk_parameters() -> None:
    from octen.async_client import AsyncDesignSearchResource
    from octen.resources.design_search import DesignSearchResource

    sent = {"query", "timeout"} | set(_design()._developer_options)
    for cls in (DesignSearchResource, AsyncDesignSearchResource):
        params = set(inspect.signature(cls.search).parameters)
        assert sent <= params, (cls.__name__, sent - params)


# --- Design Search: model input that never sends a request ------------------------------


@pytest.mark.parametrize("args", [{"query": "  "}, {"query": " ", "group_id": GROUP_ID}], ids=repr)
def test_design_blank_query_goes_to_model(args: Dict[str, Any], sync_client: MagicMock) -> None:
    result = _design().invoke(args)

    assert isinstance(result, str) and result.startswith("Invalid tool input"), result
    assert "query" in result and "blank" in result
    sync_client.assert_not_called()


SIGNED_URL = "https://cdn.example.com/photo.jpg?sig=SECRET-TOKEN-123&se=2026"


@pytest.mark.parametrize(
    ("args", "kind", "hidden"),
    [({"image_url": SIGNED_URL}, "image URL", "SECRET-TOKEN-123"),
     ({"query": "private words"}, "text query", "private words")],
    ids=["image-url", "text-query"],
)  # fmt: skip
@pytest.mark.parametrize(
    "exc",
    [
        OctenAuthenticationError(request_id="rid-401"),
        OctenValidationError(f"image search request failed validation: input {SIGNED_URL}"),
        OctenAPIError(
            f"rejected {SIGNED_URL} private words", status_code=403, request_id="rid-403"
        ),
    ],
    ids=["401", "sdk-validation", "403-echoing-input"],
)
def test_image_error_log_names_input_kind_not_input(
    exc: Exception, args: Dict[str, Any], kind: str, hidden: str, sync_client: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:  # fmt: skip
    _img_search(sync_client).side_effect = exc

    with caplog.at_level(logging.DEBUG, logger="langchain_octen"):
        with pytest.raises(type(exc)):
            _image().invoke(args)

    assert hidden not in caplog.text and "sig=" not in caplog.text, caplog.text
    assert kind in caplog.text and type(exc).__name__ in caplog.text, caplog.text
    request_id = getattr(exc, "request_id", None)
    if request_id:
        assert request_id in caplog.text


@pytest.mark.asyncio
async def test_image_error_log_hides_url_async(
    async_client: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    _img_search(async_client).side_effect = OctenAuthenticationError(request_id="rid-401")

    with caplog.at_level(logging.DEBUG, logger="langchain_octen"):
        with pytest.raises(OctenAuthenticationError):
            await _image().ainvoke({"image_url": SIGNED_URL})

    assert "SECRET-TOKEN-123" not in caplog.text and "image URL" in caplog.text


@pytest.mark.parametrize(
    "exc",
    [OctenAPIError(f"too large: {SIGNED_URL}", status_code=413),
     # Upstream text can echo a truncated or normalized copy that a replace would miss.
     OctenAPIError("bad image cdn.example.com/photo.jpg?sig=SECRET-TOKEN-1...", status_code=415),
     OctenAPIError("unreadable https://CDN.EXAMPLE.COM/photo.jpg?sig=SECRET-TOKEN-123",
                   status_code=422, request_id="rid-422"),
     OctenAPIError("upstream unavailable sig=SECRET-TOKEN-123", status_code=503),
     OctenRateLimitError(request_id="rid-429"),
     OctenTimeoutError(timeout=30.0, request_sent=True)],
    ids=["413", "415-truncated", "422-normalized", "503", "429", "timeout"],
)  # fmt: skip
def test_image_model_message_hides_url_query(exc: Exception, sync_client: MagicMock) -> None:
    _img_search(sync_client).side_effect = exc

    result = _image().invoke({"image_url": SIGNED_URL})

    assert isinstance(result, str) and "SECRET-TOKEN" not in result, result
    assert str(exc) not in result and type(exc).__name__ in result, result
    request_id = getattr(exc, "request_id", None)
    if request_id:
        assert f"request_id {request_id}" in result, result


@pytest.mark.parametrize("group_id", ["g" * 127, "\U0001f600" * 127], ids=["ascii", "astral"])
def test_design_group_id_of_127_code_points_is_sent(group_id: str, sync_client: MagicMock) -> None:
    _design().invoke({"group_id": group_id})

    _des_search(sync_client).assert_called_once_with(images={"group_id": group_id})


UNAVAILABLE = "temporarily unavailable"


@pytest.mark.parametrize(
    "exc",
    [OctenAPIError("upstream unavailable", status_code=503),
     OctenAPIError("internal", status_code=500),
     OctenAPIError("request timeout", status_code=408),
     OctenConnectionError("refused", request_sent=False),
     OctenTimeoutError(timeout=5.0, request_sent=False)],
    ids=["503", "500", "408", "connection-unsent", "timeout-unsent"],
)  # fmt: skip
def test_image_url_transient_failure_is_neutral(exc: Exception, sync_client: MagicMock) -> None:
    _img_search(sync_client).side_effect = exc

    result = _image().invoke({"image_url": SIGNED_URL})

    assert isinstance(result, str) and UNAVAILABLE in result, result
    assert "another image" not in result and MAY_HAVE_BEEN_SENT not in result, result


@pytest.mark.parametrize(
    "exc",
    [OctenAPIError("bad gateway", status_code=502),
     OctenAPIError("gateway timeout", status_code=504),
     OctenTimeoutError(timeout=5.0, request_sent=True),
     OctenConnectionError("reset", request_sent=None)],
    ids=["502", "504", "timeout-sent", "connection-unknown"],
)  # fmt: skip
def test_image_url_possibly_billed_failure_has_no_image_advice(
    exc: Exception, sync_client: MagicMock
) -> None:
    _img_search(sync_client).side_effect = exc

    result = _image().invoke({"image_url": SIGNED_URL})

    assert isinstance(result, str) and MAY_HAVE_BEEN_SENT in result, result
    assert "another image" not in result, result


@pytest.mark.parametrize("status", [400, 413, 415, 422])
def test_image_url_input_rejection_suggests_another_image(
    status: int, sync_client: MagicMock
) -> None:
    _img_search(sync_client).side_effect = OctenAPIError("rejected", status_code=status)

    result = _image().invoke({"image_url": SIGNED_URL})

    assert isinstance(result, str) and "another image URL" in result, result
    assert UNAVAILABLE not in result, result


def test_image_url_timeout_keeps_billing_warning(sync_client: MagicMock) -> None:
    _img_search(sync_client).side_effect = OctenTimeoutError(timeout=30.0, request_sent=True)

    result = _image().invoke({"image_url": SIGNED_URL})

    assert isinstance(result, str) and MAY_HAVE_BEEN_SENT in result, result


def test_image_text_query_keeps_upstream_text(sync_client: MagicMock) -> None:
    exc = OctenAPIError("Input unreadable or invalid", status_code=422)
    _img_search(sync_client).side_effect = exc

    result = _image().invoke({"query": "q"})

    assert isinstance(result, str) and str(exc) in result, result


@pytest.mark.parametrize(
    "exc",
    [OctenTimeoutError(timeout=30.0, request_sent=True),
     OctenAPIError("bad gateway", status_code=502),
     OctenAPIError("API returned code 500", status_code=200)],
    ids=["timeout", "502", "2xx-envelope"],
)  # fmt: skip
@pytest.mark.parametrize("args", [{"image_url": SIGNED_URL}, {"query": "q"}], ids=["url", "text"])
def test_billing_warning_never_suggests_retrying(
    exc: Exception, args: Dict[str, Any], sync_client: MagicMock
) -> None:
    _img_search(sync_client).side_effect = exc

    result = _image().invoke(args)

    assert isinstance(result, str) and MAY_HAVE_BEEN_SENT in result, result
    assert "Try " not in result and "try again" not in result.lower(), result


SIGNED_403 = OctenAPIError(
    f"forbidden for {SIGNED_URL}", status_code=403, response_data={"msg": SIGNED_URL},
    request_id="rid-403",
)  # fmt: skip


@pytest.mark.parametrize(
    "exc",
    [SIGNED_403,
     OctenAuthenticationError(f"bad key {SIGNED_URL}", response_data=SIGNED_URL,
                              request_id="rid-401"),
     OctenValidationError(f"failed {SIGNED_URL}", errors=[{"input": SIGNED_URL}]),
     RuntimeError(f"unexpected {SIGNED_URL}")],
    ids=["403", "401", "validation", "runtime"],
)  # fmt: skip
def test_raised_error_for_image_url_is_sanitized(
    exc: Exception, sync_client: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    _img_search(sync_client).side_effect = exc
    # Octen classes are rebuilt as themselves; anything else becomes a generic OctenError.
    expected = type(exc) if isinstance(exc, octen.OctenError) else octen.OctenError

    with caplog.at_level(logging.DEBUG, logger="langchain_octen"):
        with pytest.raises(Exception) as info:
            _image().invoke({"image_url": SIGNED_URL})

    raised = info.value
    assert type(raised) is expected, type(raised)
    rendered = "".join(
        __import__("traceback").format_exception(type(raised), raised, raised.__traceback__)
    )
    for text in (str(raised), repr(raised), rendered, caplog.text):
        assert "SECRET-TOKEN" not in text, text
    assert raised.__cause__ is None and raised.__context__ is None
    assert "SECRET-TOKEN" not in repr(getattr(raised, "response_data", None))
    assert "SECRET-TOKEN" not in repr(getattr(raised, "errors", None))
    for attr in ("status_code", "request_id"):
        assert getattr(raised, attr, None) == getattr(exc, attr, None)
    sync_client.return_value.close.assert_called_once()


@pytest.mark.asyncio
async def test_raised_error_for_image_url_is_sanitized_async(async_client: MagicMock) -> None:
    _img_search(async_client).side_effect = SIGNED_403

    with pytest.raises(OctenAPIError) as info:
        await _image().ainvoke({"image_url": SIGNED_URL})

    assert "SECRET-TOKEN" not in str(info.value) and "SECRET-TOKEN" not in repr(info.value)
    assert info.value.status_code == 403 and info.value.request_id == "rid-403"
    assert info.value.__cause__ is None and info.value.__context__ is None


@pytest.mark.parametrize(
    ("exc", "kept"),
    [
        (OctenAPIError(SIGNED_URL, status_code=403, request_id="r"),
         {"status_code": 403, "request_id": "r"}),
        (OctenAuthenticationError(SIGNED_URL, request_id="r"),
         {"status_code": 401, "request_id": "r"}),
        (OctenRateLimitError(SIGNED_URL, retry_after=3.0, request_id="r"),
         {"status_code": 429, "request_id": "r", "retry_after": 3.0}),
        (OctenValidationError(SIGNED_URL, errors=[SIGNED_URL]), {"errors": []}),
        (OctenTimeoutError(SIGNED_URL, timeout=5.0, request_sent=True),
         {"timeout": 5.0, "request_sent": True}),
        (OctenConnectionError(SIGNED_URL, request_sent=False), {"request_sent": False}),
    ],
    ids=["api", "auth", "rate-limit", "validation", "timeout", "connection"],
)  # fmt: skip
def test_sanitized_octen_errors_are_rebuilt_as_themselves(
    exc: Exception, kept: Dict[str, Any]
) -> None:
    from langchain_octen.image_tools import _without_upstream_text

    clean = _without_upstream_text(exc)

    assert type(clean) is type(exc)
    assert "SECRET-TOKEN" not in str(clean) and "SECRET-TOKEN" not in repr(clean)
    assert "SECRET-TOKEN" not in repr(vars(clean))
    for attr, value in kept.items():
        assert getattr(clean, attr) == value, attr


@pytest.mark.skipif(sys.version_info < (3, 11), reason="ExceptionGroup is new in 3.11")
def test_raised_exception_group_for_image_url_is_generic(sync_client: MagicMock) -> None:
    group_cls = getattr(builtins, "ExceptionGroup")
    exc = group_cls(f"several failures for {SIGNED_URL}", [ValueError(SIGNED_URL)])
    exc.request_id = "rid-eg"
    _img_search(sync_client).side_effect = exc

    with pytest.raises(octen.OctenError) as info:
        _image().invoke({"image_url": SIGNED_URL})

    raised = info.value
    rendered = "".join(
        __import__("traceback").format_exception(type(raised), raised, raised.__traceback__)
    )
    for text in (str(raised), repr(raised), rendered):
        assert "SECRET-TOKEN" not in text, text
    assert type(raised) is octen.OctenError and raised.request_id == "rid-eg"
    assert raised.__cause__ is None and raised.__context__ is None


def test_raised_error_for_image_url_drops_notes_and_extras(sync_client: MagicMock) -> None:
    exc = OctenAPIError("forbidden", status_code=403, request_id="rid-403")
    exc.__notes__ = [f"while fetching {SIGNED_URL}"]
    exc.upstream_url = SIGNED_URL  # type: ignore[attr-defined]
    _img_search(sync_client).side_effect = exc

    with pytest.raises(OctenAPIError) as info:
        _image().invoke({"image_url": SIGNED_URL})

    raised = info.value
    rendered = "".join(
        __import__("traceback").format_exception(type(raised), raised, raised.__traceback__)
    )
    for text in (str(raised), repr(raised), rendered, repr(vars(raised))):
        assert "SECRET-TOKEN" not in text, text
    assert not hasattr(raised, "__notes__") and not hasattr(raised, "upstream_url")
    assert (raised.status_code, raised.request_id) == (403, "rid-403")


def test_raised_error_for_text_query_is_unchanged(sync_client: MagicMock) -> None:
    _img_search(sync_client).side_effect = SIGNED_403

    with pytest.raises(OctenAPIError) as info:
        _image().invoke({"query": "q"})

    assert info.value is SIGNED_403


def test_model_error_for_image_url_has_no_upstream_chain(sync_client: MagicMock) -> None:
    _img_search(sync_client).side_effect = OctenAPIError(f"slow {SIGNED_URL}", status_code=503)
    tool = _image(handle_tool_error=False)

    with pytest.raises(ToolException) as info:
        tool.invoke({"image_url": SIGNED_URL})

    assert info.value.__cause__ is None and info.value.__context__ is None
    assert "SECRET-TOKEN" not in repr(info.value)


def test_design_without_images_custom_description_keeps_query_schema(
    sync_client: MagicMock,
) -> None:
    tool = _design(images={"enable": False}, description="Find design pages.")
    assert tool.description == "Find design pages."
    assert set(tool.args) == {"query"}
    result = tool.invoke({"query": "q", "group_id": GROUP_ID})
    assert isinstance(result, str) and "group_id" in result
    sync_client.assert_not_called()


def test_design_without_images_schema_requires_query() -> None:
    tool = _design(images={"enable": False})
    assert set(tool.args) == {"query"}
    schema = tool.tool_call_schema.model_json_schema()
    assert schema.get("required") == ["query"] and "group_id" not in schema["properties"]


def test_design_without_images_drops_group_id(sync_client: MagicMock) -> None:
    tool = _design(images={"enable": False})
    result = tool.invoke({"query": "q", "group_id": GROUP_ID})
    assert isinstance(result, str) and "group_id" in result, result
    sync_client.assert_not_called()

    # A direct call that bypasses the schema still does not forward group_id.
    tool._run(query="q", group_id=GROUP_ID)
    _des_search(sync_client).assert_called_once_with(query="q", images={"enable": False})


def test_design_with_images_schema_keeps_group_id() -> None:
    assert set(_design().args) == {"query", "group_id"}
    assert set(_design(images={"count": 2}).args) == {"query", "group_id"}


@pytest.mark.parametrize("args", [{}, {"query": None}, {"group_id": None}], ids=repr)
def test_design_query_or_group_required(args: Dict[str, Any], sync_client: MagicMock) -> None:
    result = _design().invoke(args)

    assert isinstance(result, str)
    assert result.startswith("Invalid tool input") and "`query`" in result, result
    assert "`group_id`" in result
    sync_client.assert_not_called()


@pytest.mark.asyncio
async def test_design_query_or_group_required_async(async_client: MagicMock) -> None:
    result = await _design().ainvoke({})

    assert isinstance(result, str) and "`group_id`" in result
    async_client.assert_not_called()


@pytest.mark.parametrize(
    ("args", "named"),
    [
        ({"query": ""}, "query"),
        ({"query": "x" * 501}, "query"),
        ({"group_id": ""}, "group_id"),
        ({"group_id": "g" * 128}, "group_id"),
        ({"group_id": "   "}, "group_id"),
        ({"group_id": "\U0001f600" * 128}, "group_id"),
        ({"group_id": 123}, "group_id"),
        ({"qeury": "q"}, "qeury"),
    ],
    ids=[
        "query-empty",
        "query-501",
        "group-empty",
        "group-128",
        "group-blank",
        "group-128-astral",
        "group-int",
        "misspelt-query",
    ],
)
def test_design_invalid_model_input_goes_to_model(
    args: Dict[str, Any], named: str, sync_client: MagicMock
) -> None:
    result = _design().invoke(args)

    assert isinstance(result, str)
    assert result.startswith("Invalid tool input") and named in result, result
    sync_client.assert_not_called()


@pytest.mark.parametrize(
    "extra",
    [
        {"images": {"count": 100}},
        {"web_search": {"enable": False}},
        {"count": 100},
        {"design_facts": True},
        {"snippet": {"enable": True}},
        {"timeout": 5},
        {"include_domains": ["a.com"]},
    ],
    ids=lambda e: next(iter(e)),
)
def test_design_model_cannot_set_developer_options(
    extra: Dict[str, Any], sync_client: MagicMock
) -> None:
    result = _design().invoke({"query": "q", **extra})

    assert isinstance(result, str) and next(iter(extra)) in result
    sync_client.assert_not_called()


# --- Design Search: construction --------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "bad_name"),
    [
        ({"images": {"count": 0}}, "count"),
        ({"images": {"count": 101}}, "count"),
        # Switches must be real booleans: "false" would be coerced by octen but not here.
        ({"images": {"enable": "false"}}, "images.enable"),
        ({"web_search": {"enable": 0}}, "web_search.enable"),
        # One-shot iterators would be consumed by the construction check and then send
        # nothing; lists are required.
        ({"web_search": {"language": iter(["en"])}}, "web_search.language"),
        ({"web_search": {"include_domains": (d for d in ["a.com"])}}, "web_search.include_domains"),
        ({"web_search": {"exclude_domains": map(str, ["b.com"])}}, "web_search.exclude_domains"),
        ({"images": {"snippet": {"max_tokens": iter([1000])}}}, "images.snippet.max_tokens"),
        ({"images": {"cuont": 3}}, "cuont"),
        ({"images": {"group_id": GROUP_ID}}, "group_id"),
        ({"images": {"snippet": {"max_tokens": 99}}}, "max_tokens"),
        ({"images": {"snippet": {"max_tokens": 100_001}}}, "max_tokens"),
        ({"images": {"snippet": {"enabled": True}}}, "enabled"),
        ({"images": True}, "images"),
        ({"web_search": {"count": 0}}, "count"),
        ({"web_search": {"count": 101}}, "count"),
        ({"web_search": {"highlight": {"enabled": True}}}, "enabled"),
        ({"web_search": {"highlight": {"max_tokens": 99}}}, "max_tokens"),
        ({"web_search": {"full_content": {"max_token": 500}}}, "max_token"),
        ({"web_search": {"full_content": {"max_tokens": 100_001}}}, "max_tokens"),
        ({"web_search": {"language": ["xx"]}}, "language"),
        ({"web_search": {"language": "en"}}, "language"),
        ({"web_search": {"time_basis": "indexed"}}, "time_basis"),
        ({"web_search": {"time_range": "decade"}}, "time_range"),
        ({"web_search": {"start_time": "2025-01-01"}}, "start_time"),
        ({"web_search": {"end_time": "2025-13-01T00:00:00Z"}}, "end_time"),
        ({"web_search": {"start_time": "2025-02-30T00:00:00Z"}}, "start_time"),
        ({"web_search": {"start_time": "2025-01-01T24:00:00Z"}}, "start_time"),
        ({"web_search": {"end_time": "2025-01-01T00:00:00+24:00"}}, "end_time"),
        ({"web_search": {"end_time": "2025-01-01t00:00:00z"}}, "end_time"),
        (
            {
                "web_search": {
                    "start_time": "2025-06-01T00:00:00Z",
                    "end_time": "2025-01-01T00:00:00Z",
                }
            },
            "end_time",
        ),
        # Near the ends of the calendar: the offset pushes the UTC instant out of range.
        (
            {
                "web_search": {
                    "start_time": "9999-12-31T23:00:00-01:00",
                    "end_time": "0001-01-01T00:00:00+01:00",
                }
            },
            "end_time",
        ),
        ({"web_search": {"highlight": {1: True, "enable": True}}}, "1"),
        ({"web_search": {"full_content": {1: True, "x": 2, None: 3}}}, "None"),
        # Reversed by one nanosecond, below microsecond precision.
        (
            {
                "web_search": {
                    "start_time": "2025-01-01T00:00:00.0000000002Z",
                    "end_time": "2025-01-01T00:00:00.0000000001Z",
                }
            },
            "end_time",
        ),
        # 00:00+01:00 is 23:00Z the day before: earlier, though the string sorts later.
        (
            {
                "web_search": {
                    "start_time": "2025-01-01T00:00:00Z",
                    "end_time": "2025-01-01T00:00:00.5+01:00",
                }
            },
            "end_time",
        ),
        ({"web_search": {"include_domains": ["a" * 61]}}, "include_domains"),
        ({"web_search": {"exclude_domains": ["a.com"] * 1201}}, "exclude_domains"),
        ({"web_search": {"safesearch": "off"}}, "safesearch"),
        ({"web_search": "on"}, "web_search"),
        ({"timeout": -1}, "timeout"),
        ({"imgaes": {"count": 3}}, "imgaes"),
        ({"query": "fixed"}, "query"),
    ],
    ids=repr,
)
def test_design_invalid_developer_option_raises(
    options: Dict[str, Any], bad_name: str, sync_client: MagicMock
) -> None:
    with pytest.raises(ValueError, match=bad_name):
        _design(**options)
    sync_client.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"images": {}},
        {"images": {"count": 1}},
        {"images": {"count": 100, "enable": True}},
        {"images": {"snippet": {"enable": False}}},
        {"images": {"snippet": {"max_tokens": 100}}},
        {"images": {"snippet": {"max_tokens": 100_000}}},
        {"web_search": {"enable": False}},
        {"web_search": {"count": 1}},
        {"web_search": {"count": 100}},
        {"web_search": {"language": ["ar", "zh"]}},
        {"images": IMAGES, "web_search": WEB},
        {
            "web_search": {
                "start_time": "0001-01-01T00:00:00+01:00",
                "end_time": "9999-12-31T23:00:00-01:00",
            }
        },
        {
            "web_search": {
                "start_time": "0001-01-01T00:00:00+01:00",
                "end_time": "0001-01-01T00:00:00Z",
            }
        },
        {
            "web_search": {
                "start_time": "2025-01-01T08:00:00+08:00",
                "end_time": "2025-01-01T00:00:00Z",
            }
        },
        {
            "web_search": {
                "start_time": "2025-01-01T00:00:00Z",
                "end_time": "2025-01-01T00:00:00.000001Z",
            }
        },
    ],
    ids=repr,
)
def test_design_valid_developer_options_accepted(options: Dict[str, Any]) -> None:
    _design(**options)


# --- Results ----------------------------------------------------------------------------


def test_image_data_returned_unchanged(sync_client: MagicMock) -> None:
    result = _image().invoke({"query": "q"})

    assert result == {"data": IMAGE_DATA, "request_id": "rid"}
    assert result["data"]["results"][0]["summary"] == "old design corpus field"


def test_design_data_returned_unchanged(sync_client: MagicMock) -> None:
    result = _design().invoke({"query": "pricing section"})

    assert result == {"data": DESIGN_DATA, "request_id": "rid"}
    assert result["data"]["specs"] == {"undocumented": True}
    assert result["data"]["results"][0]["favicon"].endswith("favicon.ico")
    assert result["data"]["images"][0]["group"]["group_id"] == GROUP_ID


def test_design_missing_parts_stay_missing(sync_client: MagicMock) -> None:
    # images.enable / web_search.enable false: the API omits that list.
    _des_search(sync_client).return_value = _design_response(data={"results": [WEB_RESULT]})

    result = _design().invoke({"group_id": GROUP_ID})

    assert result == {"data": {"results": [WEB_RESULT]}, "request_id": "rid"}


@pytest.mark.parametrize(
    ("make_response", "method", "args"),
    [
        (_image_response, _img_search, {"query": "q"}),
        (_design_response, _des_search, {"query": "q"}),
    ],
    ids=["image", "design"],
)
@pytest.mark.parametrize(
    ("meta", "expected"),
    [({"warning": "partial"}, "partial"), ({"warning": ""}, None), ({}, None)],
    ids=["warning", "empty", "absent"],
)
def test_warning_kept_only_when_present(
    make_response: Any,
    method: Any,
    args: Dict[str, Any],
    meta: Dict[str, Any],
    expected: Any,
    sync_client: MagicMock,
) -> None:
    method(sync_client).return_value = make_response(data={"results": []}, meta=meta)
    tool: Tool = _image() if method is _img_search else _design()

    result = tool.invoke(args)

    assert result.get("warning") == expected
    assert ("warning" in result) == (expected is not None)


@pytest.mark.parametrize(
    ("make_response", "method", "data"),
    [
        (_image_response, _img_search, {"results": "nope"}),
        (_image_response, _img_search, {"results": ["nope"]}),
        (_design_response, _des_search, {"images": {"a": 1}, "results": [1]}),
    ],
    ids=["image-results-not-list", "image-item-not-dict", "design-odd-shapes"],
)
def test_unexpected_shapes_pass_through(
    make_response: Any, method: Any, data: Dict[str, Any], sync_client: MagicMock
) -> None:
    # Like the other tools, the tools do not reshape or police the response data.
    method(sync_client).return_value = make_response(data=data)
    tool: Tool = _image() if method is _img_search else _design()

    assert tool.invoke({"query": "q"}) == {"data": data, "request_id": "rid"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_cls", "method", "args", "expected"),
    [
        (OctenImageSearch, _img_search, {"image_url": IMAGE_URL}, {"image_url": IMAGE_URL}),
        (
            OctenDesignSearch,
            _des_search,
            {"group_id": GROUP_ID},
            {"images": {"group_id": GROUP_ID}},
        ),
    ],
    ids=["image", "design"],
)
async def test_async_uses_async_client_and_closes(
    tool_cls: Type[Tool],
    method: Any,
    args: Dict[str, Any],
    expected: Dict[str, Any],
    async_client: MagicMock,
) -> None:
    result = await tool_cls(octen_api_key=SecretStr(FAKE_API_KEY)).ainvoke(args)

    async_client.assert_called_once_with(api_key=FAKE_API_KEY, retry_after_send=False)
    method(async_client).assert_awaited_once_with(**expected)
    async_client.return_value.close.assert_awaited_once()
    assert result["request_id"] == "rid"


# --- Errors ----------------------------------------------------------------------------

TRANSIENT = [
    OctenRateLimitError(request_id="rid-429"),
    OctenAPIError("upstream unavailable", status_code=503),
    OctenAPIError("bad gateway", status_code=502),
    OctenAPIError("Input unreadable or invalid", status_code=422),
    OctenAPIError("Invalid params", status_code=400),
    OctenAPIError("API returned code 500", status_code=200),
    OctenTimeoutError(timeout=30.0, request_sent=True),
    OctenConnectionError("connection reset", request_sent=None),
    OctenConnectionError("connection refused", request_sent=False),
]
POSSIBLY_BILLED = [
    OctenTimeoutError(timeout=30.0, request_sent=True),
    OctenAPIError("bad gateway", status_code=502),
    OctenAPIError("API returned code 500", status_code=200),
]
CONFIG = [
    OctenAuthenticationError(),
    OctenAPIError("insufficient balance", status_code=403),
    OctenValidationError("image search request failed validation"),
    RuntimeError("unexpected"),
]
IMAGE_INPUT_ERRORS = [
    (OctenAPIError("Payload too large", status_code=413), "smaller image"),
    (OctenAPIError("Unsupported media type", status_code=415), "JPEG, PNG, WEBP"),
]


@pytest.mark.parametrize(("exc", "hint"), IMAGE_INPUT_ERRORS, ids=["413", "415"])
def test_image_input_errors_go_to_model(exc: Exception, hint: str, sync_client: MagicMock) -> None:
    _img_search(sync_client).side_effect = exc

    result = _image().invoke({"image_url": IMAGE_URL})

    assert isinstance(result, str) and hint in result and "another image" in result, result
    assert f"status {exc.status_code}" in result and str(exc) not in result, result  # type: ignore[attr-defined]
    sync_client.return_value.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(("exc", "hint"), IMAGE_INPUT_ERRORS, ids=["413", "415"])
async def test_image_input_errors_go_to_model_async(
    exc: Exception, hint: str, async_client: MagicMock
) -> None:
    _img_search(async_client).side_effect = exc

    result = await _image().ainvoke({"image_url": IMAGE_URL})

    assert isinstance(result, str) and hint in result, result


@pytest.mark.parametrize(
    "exc",
    [
        OctenAuthenticationError(),
        OctenAPIError("insufficient balance", status_code=403),
        OctenAPIError("Payload too large", status_code=200, response_data={"code": 403}),
    ],
    ids=repr,
)
def test_image_auth_and_access_errors_still_raise(exc: Exception, sync_client: MagicMock) -> None:
    _img_search(sync_client).side_effect = exc

    with pytest.raises(type(exc)):
        _image().invoke({"image_url": IMAGE_URL})


@pytest.mark.parametrize("status", [413, 415])
def test_design_413_and_415_still_raise(status: int, sync_client: MagicMock) -> None:
    # Only Image Search takes an image input; Design Search documents neither status.
    _des_search(sync_client).side_effect = OctenAPIError("rejected", status_code=status)

    with pytest.raises(OctenAPIError):
        _design().invoke({"query": "q"})


TOOLS = [
    (OctenImageSearch, _img_search, {"query": "q"}),
    (OctenDesignSearch, _des_search, {"query": "q"}),
]
TOOL_IDS = ["image", "design"]


@pytest.mark.parametrize(("tool_cls", "method", "args"), TOOLS, ids=TOOL_IDS)
@pytest.mark.parametrize("exc", TRANSIENT, ids=repr)
def test_transient_errors_go_to_model(
    exc: Exception,
    tool_cls: Type[Tool],
    method: Any,
    args: Dict[str, Any],
    sync_client: MagicMock,
) -> None:
    method(sync_client).side_effect = exc

    result = tool_cls(octen_api_key=SecretStr(FAKE_API_KEY)).invoke(args)

    assert isinstance(result, str) and str(exc) in result
    sync_client.return_value.close.assert_called_once()


@pytest.mark.parametrize(("tool_cls", "method", "args"), TOOLS, ids=TOOL_IDS)
@pytest.mark.parametrize("exc", POSSIBLY_BILLED, ids=repr)
def test_possibly_billed_failures_warn_against_retry(
    exc: Exception,
    tool_cls: Type[Tool],
    method: Any,
    args: Dict[str, Any],
    sync_client: MagicMock,
) -> None:
    method(sync_client).side_effect = exc

    result = tool_cls(octen_api_key=SecretStr(FAKE_API_KEY)).invoke(args)

    assert isinstance(result, str) and MAY_HAVE_BEEN_SENT in result


@pytest.mark.parametrize(("tool_cls", "method", "args"), TOOLS, ids=TOOL_IDS)
@pytest.mark.parametrize("exc", CONFIG, ids=repr)
def test_config_errors_raise(
    exc: Exception,
    tool_cls: Type[Tool],
    method: Any,
    args: Dict[str, Any],
    sync_client: MagicMock,
) -> None:
    method(sync_client).side_effect = exc

    with pytest.raises(type(exc)):
        tool_cls(octen_api_key=SecretStr(FAKE_API_KEY)).invoke(args)
    sync_client.return_value.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool_cls", "method", "args"), TOOLS, ids=TOOL_IDS)
async def test_async_transient_error_goes_to_model(
    tool_cls: Type[Tool], method: Any, args: Dict[str, Any], async_client: MagicMock
) -> None:
    method(async_client).side_effect = OctenTimeoutError(timeout=30.0, request_sent=True)

    result = await tool_cls(octen_api_key=SecretStr(FAKE_API_KEY)).ainvoke(args)

    assert isinstance(result, str) and MAY_HAVE_BEEN_SENT in result
    async_client.return_value.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool_cls", "method", "args"), TOOLS, ids=TOOL_IDS)
async def test_async_config_error_raises(
    tool_cls: Type[Tool], method: Any, args: Dict[str, Any], async_client: MagicMock
) -> None:
    method(async_client).side_effect = OctenAuthenticationError()

    with pytest.raises(OctenAuthenticationError):
        await tool_cls(octen_api_key=SecretStr(FAKE_API_KEY)).ainvoke(args)
    async_client.return_value.close.assert_awaited_once()


@pytest.mark.parametrize("tool_cls", [OctenImageSearch, OctenDesignSearch])
def test_missing_key_raises(tool_cls: Type[Tool], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCTEN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OCTEN_API_KEY"):
        tool_cls()


@pytest.mark.parametrize("tool_cls", [OctenImageSearch, OctenDesignSearch])
def test_key_read_from_env(tool_cls: Type[Tool], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCTEN_API_KEY", "from-env")
    assert tool_cls().octen_api_key.get_secret_value() == "from-env"  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("tool_cls", "bad"),
    [(OctenImageSearch, {"count": 99}), (OctenDesignSearch, {"images": {"count": 999}})],
)
def test_option_error_does_not_echo_key(tool_cls: Type[Tool], bad: Dict[str, Any]) -> None:
    secret = "sk-live-SECRET-0123456789"
    with pytest.raises(ValueError) as exc_info:
        tool_cls(octen_api_key=secret, **bad)
    assert secret not in str(exc_info.value) and secret not in repr(exc_info.value)
