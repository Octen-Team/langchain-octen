"""Live tests against the Octen API.

Skipped unless OCTEN_LIVE=1 and OCTEN_API_KEY are set. Each call is billed, so every
tool is invoked with a small result count: once, or for the Web Search Premier, Extract,
Image Search and Design Search tests two or three calls (calls with the option and
same-input controls). Extract bills each successfully read URL; the Extract tests read 15
at most. The Image Search and Design Search tests make 10 calls.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import octen
import pytest
from octen import OctenAuthenticationError
from octen.async_client import AsyncExtractResource
from octen.core.http_client import HTTPClient
from octen.resources.search import SearchResource
from pydantic import SecretStr

from langchain_octen import (
    OctenBroadSearch,
    OctenBusinessSearch,
    OctenDesignSearch,
    OctenExtract,
    OctenImageSearch,
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
    "test_extract_cap_rejects_without_request",
    "test_extract_model_cannot_set_mode",
    "test_image_search_violations_send_nothing",
    "test_design_search_violations_send_nothing",
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
# Single-bound checks need unfiltered results dated on both sides of the window; on
# 2026-11-21 they ran from 2022 to 2026-01-30, so the window ends before the latest.
BOUND_WINDOW = ("2024-01-01T00:00:00Z", "2025-06-30T23:59:59Z")
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
    start, end = _ts(BOUND_WINDOW[0]), _ts(BOUND_WINDOW[1])
    assert start and end
    # Unfiltered control first: dated results on both sides make each bound check discriminate.
    unfiltered = OctenSearchPremier(count=10, time_basis="published").invoke({"query": TIME_QUERY})
    _check("unfiltered", unfiltered)
    print(f"unfiltered time_published={_published(unfiltered)}")
    unfiltered_dated = [t for t in _published(unfiltered) if t]
    stale = "the live index changed; pick a new TIME_QUERY / BOUND_WINDOW"
    assert any(t < start for t in unfiltered_dated), f"no result before start: {stale}"
    assert any(t > end for t in unfiltered_dated), f"no result after end: {stale}"
    end_only = OctenSearchPremier(count=10, time_basis="published", end_time=BOUND_WINDOW[1])
    start_only = OctenSearchPremier(count=10, time_basis="published", start_time=BOUND_WINDOW[0])
    before = end_only.invoke({"query": TIME_QUERY})
    after = start_only.invoke({"query": TIME_QUERY})
    _check("end-only", before)
    _check("start-only", after)
    print(f"end-only time_published={_published(before)}")
    print(f"start-only time_published={_published(after)}")
    # Each call keeps its own bound, and its control shows the result on the other side.
    before_dated = [t for t in _published(before) if t]
    after_dated = [t for t in _published(after) if t]
    assert before_dated and all(t <= end for t in before_dated), before
    assert after_dated and all(t >= start for t in after_dated), after
    assert any(t < start for t in before_dated), before
    assert any(t > end for t in after_dated), after


def test_search_premier_model_cannot_set_count() -> None:
    result = OctenSearchPremier(count=1).invoke({"query": "Apache Kafka", "count": 50})
    assert isinstance(result, str) and "count" in result, result


def test_search_premier_bad_option_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="result_type"):
        OctenSearchPremier(result_type="smart")


# Stable public pages; .invalid is reserved (RFC 2606), so that host never resolves.
EXAMPLE = "https://example.com/"
DOCS = "https://docs.octen.ai/api-reference/extract"
# The page the API reference uses for its media example.
WITH_IMAGES = "https://octen.ai/"
UNREACHABLE = "https://nonexistent.invalid/"


def _items(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return result["data"]["results"]  # type: ignore[no-any-return]


def _by_url(result: Dict[str, Any], urls: List[str]) -> Dict[str, Dict[str, Any]]:
    """Items keyed by url; each input URL must come back exactly once, with no extras."""
    items = _items(result)
    assert sorted(i["url"] for i in items) == sorted(urls), items
    return {i["url"]: i for i in items}


def test_extract_reads_pages() -> None:
    tool = OctenExtract(max_urls=2)
    result = tool.invoke({"urls": [EXAMPLE, DOCS]})
    _check(tool.name, result)
    items = list(_by_url(result, [EXAMPLE, DOCS]).values())
    print(f"titles={[i.get('title') for i in items]}")
    assert all(i["status"] == "success" for i in items), items
    assert all(i.get("full_content") and i.get("title") for i in items), items
    assert not any(i.get("highlights") for i in items), items
    # Links are opt-in: without include_links none come back.
    assert not any(i.get("links") for i in items), items


@pytest.mark.asyncio
async def test_extract_failed_url_reported_per_url_async(monkeypatch: pytest.MonkeyPatch) -> None:
    responses: List[Any] = []
    original = AsyncExtractResource.extract

    async def spy(self: AsyncExtractResource, *args: Any, **kwargs: Any) -> Any:
        response = await original(self, *args, **kwargs)
        responses.append(response)
        return response

    monkeypatch.setattr(AsyncExtractResource, "extract", spy)
    tool = OctenExtract(max_urls=2)
    result = await tool.ainvoke({"urls": [EXAMPLE, UNREACHABLE]})
    _check(tool.name, result)
    by_url = _by_url(result, [EXAMPLE, UNREACHABLE])
    ok, failed = by_url[EXAMPLE], by_url[UNREACHABLE]
    print(f"failed item={failed} usage={responses[0].usage}")
    assert ok["url"] == EXAMPLE and ok["status"] == "success" and ok.get("full_content"), ok
    assert failed["url"] == UNREACHABLE and failed["status"] == "failed", failed
    assert failed.get("error_message"), failed
    assert not failed.get("full_content"), failed
    # Only the page that was read is billed.
    assert responses[0].successful_urls == 1, responses[0].meta


def test_extract_query_returns_highlights() -> None:
    # The same page with the model's query and without it.
    tool = OctenExtract(max_urls=1)
    focused = tool.invoke({"urls": [DOCS], "query": "how long cached content is kept"})
    control = tool.invoke({"urls": [DOCS]})
    _check(f"{tool.name} query", focused)
    _check(f"{tool.name} control", control)
    (item,), (plain,) = _items(focused), _items(control)
    print(f"highlights={item.get('highlights')!r:.300}")
    assert item["status"] == "success" and item.get("highlights"), item
    assert not item.get("full_content"), item
    assert plain.get("full_content") and not plain.get("highlights"), plain


def test_extract_advanced_mode_reaches_server() -> None:
    advanced = OctenExtract(max_urls=1, mode="advanced").invoke({"urls": [EXAMPLE]})
    control = OctenExtract(max_urls=1).invoke({"urls": [EXAMPLE]})
    _check("advanced", advanced)
    _check("control", control)
    (item,), (plain,) = _items(advanced), _items(control)
    print(f"resolved_mode={item.get('resolved_mode')} control={plain.get('resolved_mode')}")
    # octen documents that an advanced request may resolve to standard: re-check before
    # calling a failure here a forwarding bug.
    assert item["status"] == "success" and item["resolved_mode"] == "advanced", item
    assert plain["resolved_mode"] == "standard", plain


def test_extract_text_format_reaches_server() -> None:
    text = OctenExtract(max_urls=1, format="text").invoke({"urls": [DOCS]})
    control = OctenExtract(max_urls=1).invoke({"urls": [DOCS]})
    _check("text", text)
    _check("markdown control", control)
    (item,), (plain,) = _items(text), _items(control)
    markdown_heading = re.compile(r"^#{1,6} ", re.MULTILINE)
    print(f"text={item.get('full_content')!r:.200}\nmarkdown={plain.get('full_content')!r:.200}")
    assert markdown_heading.search(plain["full_content"]), plain
    assert item.get("full_content") and not markdown_heading.search(item["full_content"]), item


def test_extract_links_reach_server() -> None:
    # The control takes the server's default link cap (200) and must return more than 3.
    control = OctenExtract(max_urls=1, include_links={}).invoke({"urls": [DOCS]})
    _check("links control", control)
    (plain,) = _items(control)
    print(f"control links={len(plain.get('links') or [])}")
    assert len(plain.get("links") or []) > 3, plain
    linked = OctenExtract(max_urls=1, include_links={"max_links": 3}).invoke({"urls": [DOCS]})
    _check("links", linked)
    (item,) = _items(linked)
    print(f"links={item.get('links')}")
    assert len(item.get("links") or []) == 3, item
    assert all(link.get("url") for link in item["links"]), item


def test_extract_images_reach_server() -> None:
    # The flagged call first, so a missing flag fails before the control is billed.
    flagged = OctenExtract(max_urls=1, include_images=True).invoke({"urls": [WITH_IMAGES]})
    _check("images", flagged)
    (item,) = _items(flagged)
    print(f"images={item.get('images')!r:.300} cover_image={item.get('cover_image')}")
    assert item["status"] == "success" and item.get("images"), item
    assert all(image.get("url") for image in item["images"]), item
    assert (item.get("cover_image") or {}).get("url"), item
    control = OctenExtract(max_urls=1).invoke({"urls": [WITH_IMAGES]})
    _check("images control", control)
    (plain,) = _items(control)
    assert plain["status"] == "success" and not plain.get("images"), plain
    assert not plain.get("cover_image"), plain


def test_extract_max_age_reaches_server() -> None:
    # Control first: the check only discriminates when the default-cache copy is stale.
    # time_last_crawled is when Octen last fetched the page; allow 60 s of clock skew.
    window, margin = timedelta(seconds=300), timedelta(seconds=60)
    control = OctenExtract(max_urls=1).invoke({"urls": [DOCS]})
    _check("max_age control", control)
    cached = _ts(_items(control)[0].get("time_last_crawled"))
    print(f"control time_last_crawled={cached}")
    assert cached is not None, control
    if cached >= datetime.now(timezone.utc) - window - margin:
        pytest.skip(f"inconclusive: the cached copy of {DOCS} is fresh ({cached})")
    sent = datetime.now(timezone.utc)
    fresh = OctenExtract(max_urls=1, max_age_seconds=300).invoke({"urls": [DOCS]})
    received = datetime.now(timezone.utc)
    _check("max_age_seconds=300", fresh)
    (item,) = _items(fresh)
    crawled = _ts(item.get("time_last_crawled"))
    print(f"sent={sent.isoformat()} time_last_crawled={crawled}")
    assert crawled is not None, item
    assert sent - window - margin <= crawled <= received + margin, item
    assert crawled > cached, item


def test_extract_cap_rejects_without_request() -> None:
    result = OctenExtract(max_urls=1).invoke({"urls": [EXAMPLE, DOCS]})
    assert isinstance(result, str) and "at most 1" in result, result


def test_extract_model_cannot_set_mode() -> None:
    result = OctenExtract().invoke({"urls": [EXAMPLE], "mode": "advanced"})
    assert isinstance(result, str) and "mode" in result, result


IMAGE_QUERY = "golden gate bridge"
# Wikimedia Commons "File:Siluete.jpg" (800 KB JPEG): description "Tour Eiffel", category
# "Eiffel Tower in 2019". The URL names no subject, so a matching result shows the image was used.
REFERENCE_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/2/23/Siluete.jpg"


def _source_hosts(result: Dict[str, Any]) -> List[str]:
    return [urlparse(r["source_page"]).hostname or "" for r in result["data"]["results"]]


def _registered(host: str) -> str:
    return ".".join(host.split(".")[-2:])


@pytest.fixture
def request_bodies(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """Record the JSON body of each request octen sends."""
    bodies: List[Any] = []
    original = HTTPClient.request

    def spy(self: HTTPClient, *args: Any, **kwargs: Any) -> Any:
        bodies.append(kwargs.get("json"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HTTPClient, "request", spy)
    return bodies


def test_image_search_by_text_with_filters(request_bodies: List[Any]) -> None:
    # The filtered call first, so a dropped filter or count fails before the control is billed.
    included = OctenImageSearch(count=2).invoke(
        {"query": IMAGE_QUERY, "include_domains": ["wikipedia.org"]}
    )
    _check("include_domains + count=2", included)
    print(f"included hosts={_source_hosts(included)}")
    assert len(included["data"]["results"]) == 2, included
    assert all(_on(h, "wikipedia.org") for h in _source_hosts(included)), included

    control = OctenImageSearch().invoke({"query": IMAGE_QUERY})
    _check("image control", control)
    hosts = _source_hosts(control)
    print(f"control hosts={hosts}")
    # Server default count is 5; the control must show off-filter hosts to discriminate.
    assert len(hosts) > 2, control
    assert any(not _on(h, "wikipedia.org") for h in hosts), control
    assert all(r.get("url") and r.get("source_page") for r in control["data"]["results"])

    dropped = _registered(hosts[0])
    tool = OctenImageSearch(safesearch="off")
    excluded = tool.invoke({"query": IMAGE_QUERY, "exclude_domains": [dropped]})
    _check(f"exclude_domains {dropped}", excluded)
    print(f"excluded hosts={_source_hosts(excluded)}")
    assert excluded["data"]["results"], excluded
    assert not any(_on(h, dropped) for h in _source_hosts(excluded)), excluded
    # safesearch is not echoed in the response; the server accepted the request carrying it.
    assert request_bodies[-1]["safesearch"] == "off", request_bodies[-1]
    assert "safesearch" not in request_bodies[1], request_bodies[1]


def test_image_search_by_url(request_bodies: List[Any]) -> None:
    result = OctenImageSearch(count=3).invoke({"image_url": REFERENCE_IMAGE})
    _check("image_url", result)
    print(f"image_url hosts={_source_hosts(result)}")
    assert request_bodies[0]["inputs"] == [{"type": "image", "url": REFERENCE_IMAGE}]
    assert 0 < len(result["data"]["results"]) <= 3, result
    assert all(r.get("url") for r in result["data"]["results"]), result
    about = [
        f"{r.get('title') or ''} {r.get('description') or ''} {r.get('source_page') or ''}"
        for r in result["data"]["results"]
    ]
    print(f"image_url about={about}")
    assert any(re.search(r"eiffel", text, re.IGNORECASE) for text in about), about


def test_image_search_violations_send_nothing() -> None:
    tool = OctenImageSearch()
    both = tool.invoke({"query": IMAGE_QUERY, "image_url": REFERENCE_IMAGE})
    neither = tool.invoke({"include_domains": ["wikipedia.org"]})
    count = tool.invoke({"query": IMAGE_QUERY, "count": 10})
    assert isinstance(both, str) and "not both" in both, both
    assert isinstance(neither, str) and "image_url" in neither, neither
    assert isinstance(count, str) and "count" in count, count


DESIGN_QUERY = "pricing section for a SaaS landing page"


@pytest.fixture(scope="module")
def design_control() -> Dict[str, Any]:
    """One default-options call, shared as the control of the Design Search checks."""
    # Module-scoped, so it builds its own single-attempt client.
    client = octen.Octen(api_key=os.environ["OCTEN_API_KEY"], retry_after_send=False, max_retries=0)
    tool = OctenDesignSearch()
    try:
        response = client.design_search.search(query=DESIGN_QUERY)
    finally:
        client.close()
    result = tool._result(response)
    _check("design control", result)
    return result


def _paths(images: List[Dict[str, Any]]) -> List[str]:
    # Image URLs carry a per-response signature in the query string; the path is stable.
    return [urlparse(i["url"]).path for i in images]


def test_design_search_developer_image_options() -> None:
    # Facts and snippets off first, so a dropped option fails before the control is billed.
    web_off = {"enable": False}
    off = OctenDesignSearch(
        images={"count": 2, "design_facts": False, "snippet": {"enable": False}},
        web_search=web_off,
    ).invoke({"query": DESIGN_QUERY})
    _check("design facts and snippets off", off)
    images = off["data"]["images"]
    print(f"types={[i.get('type') for i in images]} keys={sorted(off['data'])}")
    assert len(images) == 2, off
    assert all(i.get("url") and i.get("type") for i in images), images
    # data is unchanged, so a turned-off part may be absent or empty.
    assert not any(i.get("design_facts") or i.get("snippet") for i in images), images
    assert not off["data"].get("results"), off
    # Same query and count with the server defaults for facts and snippets.
    control = OctenDesignSearch(images={"count": 2}, web_search=web_off).invoke(
        {"query": DESIGN_QUERY}
    )
    _check("design facts and snippets control", control)
    assert len(control["data"]["images"]) == 2, control
    by_path = dict(zip(_paths(control["data"]["images"]), control["data"]["images"]))
    matched = [by_path[p] for p in _paths(images) if p in by_path]
    print(f"matched={len(matched)} snippets={[bool(i.get('snippet')) for i in matched]}")
    assert matched, (control, off)
    assert any(i.get("design_facts") for i in matched), matched
    assert any(i.get("snippet") for i in matched), matched


# A domain with many pages for DESIGN_QUERY, so the uncapped control returns more than 2.
WEB_DOMAIN = "dribbble.com"


def test_design_search_developer_web_options(request_bodies: List[Any]) -> None:
    # The capped call first, so a dropped option fails before the control is billed.
    images_off = {"enable": False}
    web = {"include_domains": [WEB_DOMAIN], "highlight": {"enable": False}}
    capped = OctenDesignSearch(images=images_off, web_search={**web, "count": 2}).invoke(
        {"query": DESIGN_QUERY}
    )
    _check("design web capped", capped)
    got = capped["data"]["results"]
    print(f"capped hosts={[urlparse(r['url']).hostname for r in got]}")
    assert not capped["data"].get("images"), capped
    assert 0 < len(got) <= 2, capped
    assert all(_on(urlparse(r["url"]).hostname or "", WEB_DOMAIN) for r in got), got
    assert not any(r.get("highlight") for r in got), got
    # Differs only in count: same domain filter, highlights off, no cap.
    control = OctenDesignSearch(images=images_off, web_search=web).invoke({"query": DESIGN_QUERY})
    _check("design web control", control)
    uncapped = control["data"]["results"]
    print(f"uncapped hosts={[urlparse(r['url']).hostname for r in uncapped]}")
    assert len(uncapped) > 2, control
    assert all(_on(urlparse(r["url"]).hostname or "", WEB_DOMAIN) for r in uncapped), uncapped
    # The highlight opt-out reached the API in both requests; the server-default call in
    # the group test shows highlights are on otherwise.
    sent = [body["web_search"] for body in request_bodies]
    print(f"sent web_search={sent}")
    assert [w.get("highlight", {}).get("enable") for w in sent] == [False, False], sent
    assert [w.get("count") for w in sent] == [2, None], sent


def test_design_search_group_follow_up(design_control: Dict[str, Any]) -> None:
    # Server defaults, the controls for the options above: more than 2 images, and web
    # results with highlights on several domains.
    data = design_control["data"]
    hosts = [urlparse(r["url"]).hostname or "" for r in data.get("results") or []]
    print(f"default images={len(data['images'])} web hosts={hosts}")
    assert len(data["images"]) > 2, data
    assert hosts and any(not _on(h, WEB_DOMAIN) for h in hosts), hosts
    assert all(r.get("highlight") for r in data["results"]), data["results"]
    images = data["images"]
    # The requested count is below the group's size and the API default (5), so an
    # ignored count would return more images than asked for.
    requested = 3
    seeds = [
        i
        for i in images
        if isinstance(i.get("group"), dict) and (i["group"].get("total") or 0) > requested
    ]
    assert seeds, images
    seed = seeds[0]
    group_id, seat = seed["group"]["group_id"], seed["group"]["position"]
    print(f"group={seed['group']}")
    # The search mixed in images from other groups, so an all-in-group answer discriminates.
    assert any((i.get("group") or {}).get("group_id") != group_id for i in images), images
    result = OctenDesignSearch(images={"count": requested}).invoke({"group_id": group_id})
    _check("design group_id", result)
    got = result["data"]["images"]
    print(f"group ids={[i['group']['group_id'] for i in got]}")
    print(f"group positions={[i['group']['position'] for i in got]}")
    # Exactly the group's first `requested` images, in order.
    assert [i["group"]["group_id"] for i in got] == [group_id] * requested, got
    assert [i["group"]["position"] for i in got] == list(range(1, requested + 1)), got
    if seat <= requested:
        assert _paths([got[seat - 1]]) == _paths([seed]), (seed, got)


def test_design_search_violations_send_nothing() -> None:
    tool = OctenDesignSearch()
    missing = tool.invoke({})
    count = tool.invoke({"query": DESIGN_QUERY, "images": {"count": 100}})
    assert isinstance(missing, str) and "group_id" in missing, missing
    assert isinstance(count, str) and "images" in count, count
