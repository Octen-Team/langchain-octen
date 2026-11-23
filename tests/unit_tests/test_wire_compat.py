"""The requests the existing tools send stay unchanged across releases.

EXPECTED was captured by running _capture_sync() on langchain-octen 0.2.0
(origin/main 20e047e) with octen 0.12.0, the version its poetry.lock pins; this branch
runs the same inputs with octen 0.13.0. A difference means a 0.2.0 tool now sends a
different request, whether from this package or from the octen upgrade.

The octen_search_premier entry was captured the same way on this branch at b3cc8ad,
with octen 0.13.0, before the Extract tool was added. The octen_extract entry was captured
on langchain-octen 0.3.0 (origin/main e63a5f6) with octen 0.13.0.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from langchain_octen import (
    OctenBroadSearch,
    OctenBusinessSearch,
    OctenExtract,
    OctenNewsSearch,
    OctenPlaceSearch,
    OctenPulseSearch,
    OctenSearchPremier,
    OctenSearchResults,
)

KEY = SecretStr("test-api-key-12345")
CANNED = {
    "code": 0,
    "msg": "success",
    "request_id": "rid",
    "data": {"results": [{"title": "T", "url": "https://example.com"}]},
    "meta": {},
}

Call = Tuple[str, str, Dict[str, Any], Any]


def _tools() -> List[Tuple[Any, Dict[str, Any]]]:
    return [
        (
            OctenSearchResults(
                octen_api_key=KEY,
                max_results=3,
                include_domains=["dev.example"],
                include_text=["python"],
                exclude_text=["java"],
                time_basis="published",
                start_time="2025-01-01T00:00:00Z",
                end_time="2025-06-01T00:00:00Z",
                highlight_enabled=True,
                highlight_max_tokens=300,
                format="markdown",
                safesearch="off",
                full_content_enabled=True,
                full_content_max_tokens=1000,
                timeout=9.0,
            ),
            {"query": "q", "exclude_domains": ["b.com"]},
        ),
        (
            OctenBroadSearch(
                octen_api_key=KEY,
                count=2,
                highlight={"enable": True, "max_tokens": 300},
                full_content={"enable": False},
                max_queries=3,
                timeout=9.0,
            ),
            {"query": "q", "include_domains": ["a.com"], "time_range": "week", "language": ["en"]},
        ),
        (
            OctenNewsSearch(octen_api_key=KEY, count=2, subjects={"enable": True, "count": 1}),
            {"query": "q", "exclude_domains": ["b.com"], "time_range": "month"},
        ),
        (
            OctenBusinessSearch(octen_api_key=KEY, entities={"count": 1}),
            {"query": "q", "language": ["en"]},
        ),
        (
            OctenPulseSearch(octen_api_key=KEY, count=1, mode="standard"),
            {"query": "q"},
        ),
        (
            OctenPlaceSearch(octen_api_key=KEY, places={"count": 1}),
            {
                "query": "q",
                "latitude": 35.0,
                "longitude": 139.0,
                "radius": 500,
                "user_location": {"city": "Tokyo"},
            },
        ),
        (
            OctenSearchPremier(
                octen_api_key=KEY,
                count=3,
                result_type="dynamic",
                highlight={"enable": True, "max_tokens": 600},
                full_content={"enable": True, "max_tokens": 1000},
                include_text=["kafka"],
                exclude_text=["ads"],
                language=["en", "de"],
                time_basis="published",
                time_range="year",
                start_time="2025-01-01T00:00:00Z",
                end_time="2025-06-30T23:59:59+08:00",
                format="markdown",
                safesearch="off",
                include_images=True,
                timeout=9.0,
            ),
            {"query": "q", "include_domains": ["a.com"], "exclude_domains": ["b.com"]},
        ),
        (
            OctenExtract(
                octen_api_key=KEY,
                max_urls=3,
                mode="advanced",
                format="text",
                max_age_seconds=600,
                url_timeout=45,
                include_images=True,
                include_videos=True,
                include_audio=True,
                include_links={"scope": "prefer_external", "max_links": 10},
                timeout=100.0,
            ),
            {"urls": ["https://example.com/", "https://docs.octen.ai/"], "query": "q"},
        ),
    ]


# Captured from langchain-octen 0.2.0 (20e047e) with octen 0.12.0; see the module docstring.
EXPECTED: Dict[str, Call] = {
    "octen_search": (
        "POST",
        "/search",
        {
            "count": 3,
            "include_domains": ["dev.example"],
            "exclude_domains": ["b.com"],
            "time_basis": "published",
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-06-01T00:00:00Z",
            "highlight": {"enable": True, "max_tokens": 300},
            "full_content": {"enable": True, "max_tokens": 1000},
            "include_text": ["python"],
            "exclude_text": ["java"],
            "format": "markdown",
            "safesearch": "off",
            "query": "q",
        },
        9.0,
    ),
    "octen_broad_search": (
        "POST",
        "/broad-search",
        {
            "query": "q",
            "max_queries": 3,
            "search_options": {
                "count": 2,
                "include_domains": ["a.com"],
                "language": ["en"],
                "time_range": "week",
                "highlight": {"enable": True, "max_tokens": 300},
                "full_content": {"enable": False, "max_tokens": 2048},
            },
        },
        9.0,
    ),
    "octen_news_search": (
        "POST",
        "/news-search",
        {
            "count": 2,
            "exclude_domains": ["b.com"],
            "time_range": "month",
            "query": "q",
            "subjects": {"enable": True, "count": 1},
        },
        None,
    ),
    "octen_business_search": (
        "POST",
        "/business-search",
        {"language": ["en"], "query": "q", "entities": {"count": 1}},
        None,
    ),
    "octen_pulse_search": (
        "POST",
        "/pulse-search",
        {"query": "q", "mode": "standard", "count": 1},
        None,
    ),
    "octen_place_search": (
        "POST",
        "/place-search",
        {
            "query": "q",
            "latitude": 35.0,
            "longitude": 139.0,
            "radius": 500,
            "user_location": {"type": "approximate", "city": "Tokyo"},
            "places": {"count": 1},
        },
        None,
    ),
    # Captured at b3cc8ad with octen 0.13.0; see the module docstring.
    "octen_search_premier": (
        "POST",
        "/search-premier",
        {
            "count": 3,
            "include_domains": ["a.com"],
            "exclude_domains": ["b.com"],
            "language": ["en", "de"],
            "time_basis": "published",
            "time_range": "year",
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-06-30T23:59:59+08:00",
            "highlight": {"enable": True, "max_tokens": 600},
            "full_content": {"enable": True, "max_tokens": 1000},
            "query": "q",
            "result_type": "dynamic",
            "include_text": ["kafka"],
            "exclude_text": ["ads"],
            "format": "markdown",
            "safesearch": "off",
            "include_images": True,
        },
        9.0,
    ),
    # Captured on 0.3.0 (e63a5f6) with octen 0.13.0; see the module docstring.
    "octen_extract": (
        "POST",
        "/extract",
        {
            "urls": ["https://example.com/", "https://docs.octen.ai/"],
            "format": "text",
            "max_age_seconds": 600,
            "query": "q",
            "timeout": 45,
            "include_images": True,
            "include_videos": True,
            "include_audio": True,
            "mode": "advanced",
            "include_links": {"scope": "prefer_external", "max_links": 10},
        },
        100.0,
    ),
}


def _capture_sync() -> Dict[str, Call]:
    calls: Dict[str, Call] = {}
    for tool, args in _tools():
        seen: List[Call] = []

        def fake(
            self: Any,
            method: str,
            endpoint: str,
            timeout: Any = None,
            json: Any = None,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            seen.append((method, endpoint, json, timeout))
            return dict(CANNED)

        with patch("octen.core.http_client.HTTPClient.request", fake):
            tool.invoke(args)
        assert len(seen) == 1, (tool.name, seen)
        calls[tool.name] = seen[0]
    return calls


async def _capture_async() -> Dict[str, Call]:
    calls: Dict[str, Call] = {}
    for tool, args in _tools():
        seen: List[Call] = []

        async def fake(
            self: Any,
            method: str,
            endpoint: str,
            timeout: Any = None,
            json: Any = None,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            seen.append((method, endpoint, json, timeout))
            return dict(CANNED)

        with patch("octen.core.async_http_client.AsyncHTTPClient.request", fake):
            await tool.ainvoke(args)
        assert len(seen) == 1, (tool.name, seen)
        calls[tool.name] = seen[0]
    return calls


def _serialize(calls: Dict[str, Call]) -> str:
    # json.dumps keeps key order, so a reordered body fails too.
    return json.dumps(calls)


def test_existing_tools_send_the_same_requests() -> None:
    assert _serialize(_capture_sync()) == _serialize(EXPECTED)


@pytest.mark.asyncio
async def test_existing_tools_send_the_same_requests_async() -> None:
    assert _serialize(await _capture_async()) == _serialize(EXPECTED)


def _post_body(tool: Any, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    seen: List[Call] = []

    def fake(
        self: Any,
        method: str,
        endpoint: str,
        timeout: Any = None,
        json: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        seen.append((method, endpoint, json, timeout))
        return dict(CANNED)

    with patch("octen.core.http_client.HTTPClient.request", fake):
        tool.invoke(args)
    assert len(seen) == 1, seen
    return seen[0][1], seen[0][2]


# These two have no earlier release to capture from. The expected bodies come from the
# OpenAPI field name and position: `result_type` at the top level of /search, and inside
# `search_options` (not at the top level) of /broad-search.
def test_web_search_sends_result_type_at_the_top_level() -> None:
    tool = OctenSearchResults(octen_api_key=KEY, max_results=3, result_type="dynamic")

    endpoint, body = _post_body(tool, {"query": "q"})

    assert endpoint == "/search"
    assert body == {"query": "q", "count": 3, "result_type": "dynamic"}


def test_broad_search_sends_result_type_inside_search_options_only() -> None:
    tool = OctenBroadSearch(octen_api_key=KEY, count=2, max_queries=3, result_type="dynamic")

    endpoint, body = _post_body(tool, {"query": "q"})

    assert endpoint == "/broad-search"
    assert body == {
        "query": "q",
        "max_queries": 3,
        "search_options": {"count": 2, "result_type": "dynamic"},
    }
