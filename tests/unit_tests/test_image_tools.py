"""Unit tests for the Image Search and Design Search tools (mocked SDK or HTTP layer)."""

from __future__ import annotations

import copy
import inspect
import json
from typing import Any, Dict, List, Tuple, Type, Union
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
        # Fields the reference does not document, and an empty description, are dropped.
        {**IMAGE_ITEM, "summary": "old design corpus field", "html_snippet": "<div/>"},
        {**IMAGE_ITEM, "url": "https://cdn.example.com/2.jpg", "description": ""},
    ]
}
IMAGE_TRIMMED: Dict[str, Any] = {
    "results": [
        IMAGE_ITEM,
        {k: v for k, v in IMAGE_ITEM.items() if k != "description"}
        | {"url": "https://cdn.example.com/2.jpg"},
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
DESIGN_TRIMMED: Dict[str, Any] = {
    "query": "pricing section",
    "images": [DESIGN_IMAGE, APP_SCREEN],
    "results": [WEB_RESULT],
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
        ({"image_url": "data:image/png;base64,iVBORw0KGgo="}, "image_url"),
        ({"image_url": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"}, "image_url"),
        ({"image_url": "https://"}, "image_url"),
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
        "url-data-uri",
        "url-base64",
        "url-no-host",
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


@pytest.mark.parametrize("url", ["http://example.com/a.png", "HTTPS://EXAMPLE.COM/A.PNG"])
def test_image_http_urls_accepted(url: str, sync_client: MagicMock) -> None:
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
        ({"group_id": 123}, "group_id"),
        ({"qeury": "q"}, "qeury"),
    ],
    ids=["query-empty", "query-501", "group-empty", "group-int", "misspelt-query"],
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
    ],
    ids=repr,
)
def test_design_valid_developer_options_accepted(options: Dict[str, Any]) -> None:
    _design(**options)


# --- Results ----------------------------------------------------------------------------


def test_image_results_trimmed_to_documented_fields(sync_client: MagicMock) -> None:
    result = _image().invoke({"query": "q"})

    assert result == {"data": IMAGE_TRIMMED, "request_id": "rid"}


def test_design_results_trimmed_to_documented_fields(sync_client: MagicMock) -> None:
    result = _design().invoke({"query": "pricing section"})

    assert result == {"data": DESIGN_TRIMMED, "request_id": "rid"}
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
    "data",
    [{"results": "nope"}, {"results": ["nope"]}],
    ids=["results-not-list", "item-not-dict"],
)
def test_image_malformed_result_goes_to_model(data: Dict[str, Any], sync_client: MagicMock) -> None:
    _img_search(sync_client).return_value = _image_response(data=data)

    result = _image().invoke({"query": "q"})

    assert isinstance(result, str) and "malformed" in result and MAY_HAVE_BEEN_SENT in result
    assert "rid" in result


@pytest.mark.parametrize(
    "data",
    [{"images": {"a": 1}}, {"results": [1]}, {"images": [{**DESIGN_IMAGE, "group": "g"}]}],
    ids=["images-not-list", "result-not-dict", "group-not-dict"],
)
def test_design_malformed_result_goes_to_model(
    data: Dict[str, Any], sync_client: MagicMock
) -> None:
    _des_search(sync_client).return_value = _design_response(data=data)

    result = _design().invoke({"query": "q"})

    assert isinstance(result, str) and "malformed" in result and MAY_HAVE_BEEN_SENT in result


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
    OctenAPIError("Payload too large", status_code=413),
    OctenAPIError("Unsupported media type", status_code=415),
    OctenValidationError("image search request failed validation"),
    RuntimeError("unexpected"),
]
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
