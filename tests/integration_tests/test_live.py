"""Live tests against the Octen API.

Skipped unless OCTEN_LIVE=1 and OCTEN_API_KEY are set. Each call is billed, so every
tool is invoked once with the smallest result count.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import octen
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
    _utilities,
    vertical_tools,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("OCTEN_LIVE") != "1" or not os.environ.get("OCTEN_API_KEY"),
    reason="set OCTEN_LIVE=1 and OCTEN_API_KEY to run live tests",
)


@pytest.fixture(autouse=True)
def _single_attempt(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Any:
    """Bill each live test at most once: the tools' clients get max_retries=0 here only."""
    created: List[str] = []

    def sync_client(api_key: SecretStr) -> octen.Octen:
        created.append("sync")
        return octen.Octen(
            api_key=api_key.get_secret_value(), retry_after_send=False, max_retries=0
        )

    def async_client(api_key: SecretStr) -> octen.AsyncOcten:
        created.append("async")
        return octen.AsyncOcten(
            api_key=api_key.get_secret_value(), retry_after_send=False, max_retries=0
        )

    for module in (vertical_tools, _utilities):
        monkeypatch.setattr(module, "sync_client", sync_client)
        monkeypatch.setattr(module, "async_client", async_client)
    yield
    if request.node.name != "test_out_of_range_developer_count_raises_at_construction":
        assert created, "the tool did not use the patched client factory"


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
    # data.search_results is one group per sub-query: {"query", "results", "latency"}.
    groups = result["data"]["search_results"]
    assert groups, result
    assert any(group.get("results") for group in groups), groups


def test_news_search() -> None:
    tool = OctenNewsSearch(count=1, subjects={"enable": True, "count": 1})
    result = tool.invoke({"query": "artificial intelligence regulation", "time_range": "month"})
    _check(tool.name, result)
    assert result["data"]["results"], result
    storylines = result["data"].get("subjects") or []
    assert any(s.get("name") for s in storylines), storylines
    print(f"octen_news_search storylines={[s.get('name') for s in storylines]}")


@pytest.mark.asyncio
async def test_business_search_async() -> None:
    tool = OctenBusinessSearch(count=1, entities={"enable": True, "count": 1})
    result = await tool.ainvoke({"query": "NVIDIA company overview"})
    _check(tool.name, result)
    entities = result["data"].get("entities") or []
    assert any(entity.get("name") for entity in entities), entities
    print(f"octen_business_search entity names={[e.get('name') for e in entities]}")


def test_pulse_search() -> None:
    tool = OctenPulseSearch(count=1)
    # Plain-string input, as some agents and chains pass it; the other tests use dicts.
    result = tool.invoke("github docs")
    _check(tool.name, result)
    assert result["data"]["results"], result


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
    if result.get("warning") == "places_unavailable":
        pytest.skip("place engine did not answer in time (warning places_unavailable)")
    places = result["data"]["places"]
    assert places, result
    assert all(place.get("name") for place in places), places


def test_bad_key_raises_authentication_error() -> None:
    tool = OctenPulseSearch(octen_api_key=SecretStr("invalid-key-12345"), count=1)
    with pytest.raises(OctenAuthenticationError) as exc_info:
        tool.invoke({"query": "github docs"})
    print(f"\nbad key request_id={exc_info.value.request_id}")


def test_out_of_range_developer_count_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="count"):
        OctenNewsSearch(count=1000)
