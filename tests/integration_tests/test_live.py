"""Live tests against the Octen API.

Skipped unless OCTEN_LIVE=1 and OCTEN_API_KEY are set. Each call is billed, so every
tool is invoked with a small result count: once, or for the Web Search Premier tests
two calls (a filtered call and a same-query control).
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

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
    OctenSearchPremier,
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
    """Bill each request at most once: the tools' clients get max_retries=0 here only.

    Tests in _NO_REQUEST get factories that fail, so they can never send a request.
    """
    created: List[str] = []
    no_request = request.node.name in _NO_REQUEST

    def sync_client(api_key: SecretStr) -> octen.Octen:
        assert not no_request, "this test must not send a request"
        created.append("sync")
        return octen.Octen(
            api_key=api_key.get_secret_value(), retry_after_send=False, max_retries=0
        )

    def async_client(api_key: SecretStr) -> octen.AsyncOcten:
        assert not no_request, "this test must not send a request"
        created.append("async")
        return octen.AsyncOcten(
            api_key=api_key.get_secret_value(), retry_after_send=False, max_retries=0
        )

    for module in (vertical_tools, _utilities):
        monkeypatch.setattr(module, "sync_client", sync_client)
        monkeypatch.setattr(module, "async_client", async_client)
    yield
    if not no_request:
        assert created, "the tool did not use the patched client factory"


# Tests that must not reach the API at all.
_NO_REQUEST = {
    "test_out_of_range_developer_count_raises_at_construction",
    "test_search_premier_model_cannot_set_count",
    "test_search_premier_bad_option_raises_at_construction",
}


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


# The SDK's own Premier live tests use this query; its results span several years.
TIME_QUERY = "electric vehicle battery"
YEAR_2025 = ("2025-01-01T00:00:00Z", "2025-12-31T23:59:59Z")
# Single-bound controls use a window with content on both sides. With only end_time=2025-12-31
# the server returned mostly undated results and nothing before 2025 (run of 2026-11-21).
H1_2026 = ("2026-01-01T00:00:00Z", "2026-06-30T23:59:59Z")
_STAMP = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)?")


def _ts(value: Any) -> Optional[datetime]:
    """None when absent; raises on a value that does not parse, so it cannot pass unchecked."""
    if value is None or value == "":
        return None
    match = _STAMP.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"unparseable timestamp {value!r}")
    base, frac, tz = match.groups()
    # fromisoformat before 3.11 takes neither Z nor fractions other than 3 or 6 digits.
    frac = "." + (frac + "000000")[:6] if frac else ""
    return datetime.fromisoformat(base + frac + ("+00:00" if tz in (None, "Z") else tz))


def _published(result: Dict[str, Any]) -> List[Optional[datetime]]:
    return [_ts(r.get("time_published")) for r in result["data"]["results"]]


def _hosts(result: Dict[str, Any]) -> List[str]:
    return [urlparse(r["url"]).hostname or "" for r in result["data"]["results"]]


def _on(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


@pytest.fixture
def premier_responses(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """Record octen's responses: meta (e.g. result_type) is not part of the tool result."""
    responses: List[Any] = []
    original = SearchResource.search_premier

    def spy(self: SearchResource, *args: Any, **kwargs: Any) -> Any:
        response = original(self, *args, **kwargs)
        responses.append(response)
        return response

    monkeypatch.setattr(SearchResource, "search_premier", spy)
    return responses


def test_search_premier_include_domains() -> None:
    # Same query with and without the model's domain filter.
    tool = OctenSearchPremier(count=5)
    query = "Apache Kafka"
    control = tool.invoke({"query": query})
    filtered = tool.invoke({"query": query, "include_domains": ["wikipedia.org"]})
    _check(f"{tool.name} control", control)
    _check(f"{tool.name} include_domains", filtered)
    print(f"control hosts={_hosts(control)} filtered hosts={_hosts(filtered)}")
    assert any(not _on(h, "wikipedia.org") for h in _hosts(control)), control
    assert _hosts(filtered), filtered
    assert all(_on(h, "wikipedia.org") for h in _hosts(filtered)), filtered


def test_search_premier_developer_options(premier_responses: List[Any]) -> None:
    # Same query with server defaults and with full content plus dynamic results.
    query = "Kafka default retention period for topics"
    control = OctenSearchPremier(count=2).invoke({"query": query})
    tool = OctenSearchPremier(
        count=2, result_type="dynamic", full_content={"enable": True, "max_tokens": 200}
    )
    result = tool.invoke({"query": query})
    _check(f"{tool.name} control", control)
    _check(f"{tool.name} options", result)
    metas = [r.meta for r in premier_responses]
    print(f"meta result_type={[m.get('result_type') for m in metas]}")
    assert [m.get("result_type") for m in metas] == ["standard", "dynamic"], metas
    assert 0 < len(control["data"]["results"]) <= 2, control
    assert 0 < len(result["data"]["results"]) <= 2, result
    assert not any(r.get("full_content") for r in control["data"]["results"]), control
    assert all(r.get("full_content") for r in result["data"]["results"]), result


@pytest.mark.asyncio
async def test_search_premier_time_window_async() -> None:
    # The filtered call first, then the same query without bounds as the control.
    start, end = _ts(YEAR_2025[0]), _ts(YEAR_2025[1])
    assert start and end
    tool = OctenSearchPremier(
        count=5, time_basis="published", start_time=YEAR_2025[0], end_time=YEAR_2025[1]
    )
    result = await tool.ainvoke({"query": TIME_QUERY})
    _check(f"{tool.name} 2025", result)
    published = _published(result)
    print(f"2025 time_published={published}")
    assert published, result
    assert all(t is not None and start <= t <= end for t in published), published

    control = await OctenSearchPremier(count=5, time_basis="published").ainvoke(
        {"query": TIME_QUERY}
    )
    _check(f"{tool.name} control", control)
    print(f"control time_published={_published(control)}")
    assert any(t and not start <= t <= end for t in _published(control)), control


def test_search_premier_each_time_bound() -> None:
    # Without start_time a result before the window comes back, and without end_time one after.
    start, end = _ts(H1_2026[0]), _ts(H1_2026[1])
    assert start and end
    end_only = OctenSearchPremier(count=10, time_basis="published", end_time=H1_2026[1])
    start_only = OctenSearchPremier(count=10, time_basis="published", start_time=H1_2026[0])
    before = end_only.invoke({"query": TIME_QUERY})
    after = start_only.invoke({"query": TIME_QUERY})
    _check("end-only", before)
    _check("start-only", after)
    print(f"end-only time_published={_published(before)}")
    print(f"start-only time_published={_published(after)}")
    assert any(t and t < start for t in _published(before)), before
    assert any(t and t > end for t in _published(after)), after


def test_search_premier_model_cannot_set_count() -> None:
    result = OctenSearchPremier(count=1).invoke({"query": "Apache Kafka", "count": 50})
    assert isinstance(result, str) and "count" in result, result


def test_search_premier_bad_option_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="result_type"):
        OctenSearchPremier(result_type="smart")
