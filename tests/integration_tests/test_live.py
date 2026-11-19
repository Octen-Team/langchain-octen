"""Live tests against the Octen API.

Skipped unless OCTEN_LIVE=1 and OCTEN_API_KEY are set. Each call is billed, so every
tool is invoked once with the smallest result count.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from octen import OctenAuthenticationError
from octen.resources.search import SearchResource
from pydantic import SecretStr

from langchain_octen import (
    OctenBroadSearch,
    OctenBusinessSearch,
    OctenNewsSearch,
    OctenPlaceSearch,
    OctenPulseSearch,
    OctenSearchResults,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("OCTEN_LIVE") != "1" or not os.environ.get("OCTEN_API_KEY"),
    reason="set OCTEN_LIVE=1 and OCTEN_API_KEY to run live tests",
)


def _check(name: str, result: Any) -> None:
    assert isinstance(result, dict), result
    assert result["data"], result
    assert result["request_id"], result
    print(f"\n{name} request_id={result['request_id']} warning={result.get('warning')!r}")


def test_octen_search_results(monkeypatch: pytest.MonkeyPatch) -> None:
    # The old tool returns a plain list; spy on the SDK to record the request id.
    responses: List[Any] = []
    original = SearchResource.search

    def spy(self: SearchResource, *args: Any, **kwargs: Any) -> Any:
        response = original(self, *args, **kwargs)
        responses.append(response)
        return response

    monkeypatch.setattr(SearchResource, "search", spy)

    results = OctenSearchResults(max_results=1).invoke({"query": "Python programming language"})

    assert isinstance(results, list) and results, results
    assert "error" not in results[0], results
    request_id = (responses[0].model_extra or {}).get("request_id")
    assert request_id
    print(f"\nocten_search request_id={request_id}")


def test_broad_search() -> None:
    tool = OctenBroadSearch(count=1, max_queries=1)
    result = tool.invoke({"query": "compare open source vector databases"})
    _check(tool.name, result)
    assert result["data"].get("search_results")


def test_news_search() -> None:
    tool = OctenNewsSearch(count=1)
    result = tool.invoke({"query": "artificial intelligence regulation", "time_range": "month"})
    _check(tool.name, result)
    assert result["data"].get("results")


@pytest.mark.asyncio
async def test_business_search_async() -> None:
    tool = OctenBusinessSearch(count=1)
    result = await tool.ainvoke({"query": "NVIDIA company overview"})
    _check(tool.name, result)
    assert result["data"].get("results") or result["data"].get("entities")


def test_pulse_search() -> None:
    tool = OctenPulseSearch(count=1)
    result = tool.invoke({"query": "github docs"})
    _check(tool.name, result)
    assert result["data"].get("results")


def test_place_search() -> None:
    tool = OctenPlaceSearch(places={"count": 1})
    args: Dict[str, Any] = {
        "query": "coffee shops",
        "latitude": 35.6812,
        "longitude": 139.7671,
        "radius": 1000,
    }
    result = tool.invoke(args)
    _check(tool.name, result)
    assert result["data"].get("places") or result.get("warning") == "places_unavailable"


def test_bad_key_raises_authentication_error() -> None:
    tool = OctenPulseSearch(octen_api_key=SecretStr("invalid-key-12345"), count=1)
    with pytest.raises(OctenAuthenticationError) as exc_info:
        tool.invoke({"query": "github docs"})
    print(f"\nbad key request_id={exc_info.value.request_id}")


def test_out_of_range_developer_count_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="count"):
        OctenNewsSearch(count=1000)
