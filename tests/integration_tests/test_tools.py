"""Integration tests for OctenSearchResults (requires OCTEN_API_KEY)."""

from __future__ import annotations

import os

import pytest

from langchain_octen import OctenSearchResults

pytestmark = pytest.mark.skipif(
    not os.environ.get("OCTEN_API_KEY"),
    reason="OCTEN_API_KEY not set",
)


@pytest.fixture
def tool() -> OctenSearchResults:
    return OctenSearchResults(max_results=3)


class TestOctenSearchResultsIntegration:
    def test_basic_search(self, tool: OctenSearchResults) -> None:
        results = tool.invoke({"query": "Python programming language"})
        assert isinstance(results, list)
        assert len(results) > 0
        assert "title" in results[0]
        assert "url" in results[0]

    def test_search_with_domain_filter(self, tool: OctenSearchResults) -> None:
        results = tool.invoke(
            {
                "query": "machine learning",
                "include_domains": ["arxiv.org"],
            }
        )
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "arxiv.org" in r["url"]

    def test_search_with_exclude_domain(self, tool: OctenSearchResults) -> None:
        results = tool.invoke(
            {
                "query": "Python tutorial",
                "exclude_domains": ["youtube.com"],
            }
        )
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "youtube.com" not in r["url"]

    @pytest.mark.asyncio
    async def test_async_search(self, tool: OctenSearchResults) -> None:
        results = await tool.ainvoke({"query": "LangChain framework"})
        assert isinstance(results, list)
        assert len(results) > 0
        assert "title" in results[0]

    def test_search_with_custom_config(self) -> None:
        tool = OctenSearchResults(
            max_results=2,
            search_type="keyword",
            safesearch="strict",
        )
        results = tool.invoke({"query": "artificial intelligence news"})
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_search_with_time_filter(self) -> None:
        tool = OctenSearchResults(
            max_results=3,
            time_basis="published",
            start_time="2025-01-01",
            end_time="2026-03-30",
        )
        results = tool.invoke({"query": "Python release"})
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_with_highlight(self) -> None:
        tool = OctenSearchResults(
            max_results=3,
            highlight_enabled=True,
            highlight_max_tokens=100,
        )
        results = tool.invoke({"query": "machine learning tutorial"})
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_with_full_content(self) -> None:
        tool = OctenSearchResults(
            max_results=2,
            full_content_enabled=True,
            full_content_max_tokens=500,
        )
        results = tool.invoke({"query": "Python documentation"})
        assert isinstance(results, list)
        assert len(results) > 0

    def test_invalid_api_key_returns_error(self) -> None:
        from pydantic import SecretStr

        tool = OctenSearchResults(octen_api_key=SecretStr("invalid-key-12345"))
        results = tool.invoke({"query": "test"})
        assert isinstance(results, list)
        assert len(results) == 1
        assert "error" in results[0]
