"""Unit tests for OctenSearchResults (mocked SDK, no network)."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import ToolException
from pydantic import SecretStr

from langchain_octen import OctenSearchResults
from langchain_octen._utilities import OctenSearchAPIWrapper

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FAKE_API_KEY = "test-api-key-12345"


def _make_search_response(results: List[Dict[str, Any]]) -> MagicMock:
    """Create a mock SearchResponse with the given results list."""
    resp = MagicMock()
    resp.results = results
    return resp


SAMPLE_RESULTS = [
    {
        "title": "Python.org",
        "url": "https://python.org",
        "highlight": "Python is a programming language.",
        "full_content": None,
        "authors": None,
        "time_published": "2024-01-01T00:00:00Z",
        "time_last_crawled": None,
    },
    {
        "title": "Real Python",
        "url": "https://realpython.com",
        "highlight": "Learn Python online.",
        "full_content": None,
        "authors": "Real Python Team",
        "time_published": None,
        "time_last_crawled": None,
    },
]


# ---------------------------------------------------------------------------
# OctenSearchAPIWrapper tests
# ---------------------------------------------------------------------------


class TestOctenSearchAPIWrapper:
    def test_init_with_explicit_key(self) -> None:
        wrapper = OctenSearchAPIWrapper(octen_api_key=SecretStr(FAKE_API_KEY))
        assert wrapper.octen_api_key.get_secret_value() == FAKE_API_KEY

    def test_init_reads_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCTEN_API_KEY", FAKE_API_KEY)
        wrapper = OctenSearchAPIWrapper()
        assert wrapper.octen_api_key.get_secret_value() == FAKE_API_KEY

    def test_init_raises_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OCTEN_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OCTEN_API_KEY"):
            OctenSearchAPIWrapper()

    def test_format_results_omits_none(self) -> None:
        raw = [{"title": "A", "url": "https://a.com", "highlight": None}]
        cleaned = OctenSearchAPIWrapper._format_results(raw)
        assert cleaned == [{"title": "A", "url": "https://a.com"}]

    @patch("octen.Octen")
    def test_results_sync(self, mock_octen_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_octen_cls.return_value = mock_client
        mock_client.search.search.return_value = _make_search_response(SAMPLE_RESULTS)

        wrapper = OctenSearchAPIWrapper(octen_api_key=SecretStr(FAKE_API_KEY))
        results = wrapper.results("Python programming")

        mock_octen_cls.assert_called_once_with(api_key=FAKE_API_KEY)
        mock_client.search.search.assert_called_once()
        mock_client.close.assert_called_once()
        assert len(results) == 2
        # None fields should be omitted
        assert "full_content" not in results[0]

    @pytest.mark.asyncio
    @patch("octen.AsyncOcten")
    async def test_results_async(self, mock_async_cls: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_async_cls.return_value = mock_client
        mock_client.search.search.return_value = _make_search_response(SAMPLE_RESULTS)

        wrapper = OctenSearchAPIWrapper(octen_api_key=SecretStr(FAKE_API_KEY))
        results = await wrapper.results_async("Python programming")

        mock_async_cls.assert_called_once_with(api_key=FAKE_API_KEY)
        mock_client.search.search.assert_called_once()
        mock_client.close.assert_called_once()
        assert len(results) == 2

    def test_build_search_kwargs_defaults(self) -> None:
        wrapper = OctenSearchAPIWrapper(octen_api_key=SecretStr(FAKE_API_KEY), max_results=10)
        kwargs = wrapper._build_search_kwargs("test query")
        assert kwargs == {"query": "test query", "count": 10}

    def test_build_search_kwargs_with_options(self) -> None:
        wrapper = OctenSearchAPIWrapper(
            octen_api_key=SecretStr(FAKE_API_KEY),
            max_results=3,
            search_type="semantic",
            include_domains=["example.com"],
            safesearch="off",
            format="markdown",
        )
        kwargs = wrapper._build_search_kwargs("test", exclude_domains=["bad.com"])
        assert kwargs["query"] == "test"
        assert kwargs["count"] == 3
        assert kwargs["search_type"] == "semantic"
        assert kwargs["include_domains"] == ["example.com"]
        assert kwargs["exclude_domains"] == ["bad.com"]
        assert kwargs["safesearch"] == "off"
        assert kwargs["format"] == "markdown"

    def test_build_search_kwargs_per_call_domain_override(self) -> None:
        wrapper = OctenSearchAPIWrapper(
            octen_api_key=SecretStr(FAKE_API_KEY),
            include_domains=["default.com"],
        )
        kwargs = wrapper._build_search_kwargs("test", include_domains=["override.com"])
        assert kwargs["include_domains"] == ["override.com"]


# ---------------------------------------------------------------------------
# OctenSearchResults (BaseTool) tests
# ---------------------------------------------------------------------------


class TestOctenSearchResults:
    def test_init_with_key(self) -> None:
        tool = OctenSearchResults(octen_api_key=SecretStr(FAKE_API_KEY))
        assert tool.name == "octen_search"
        assert tool.api_wrapper is not None

    def test_init_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCTEN_API_KEY", FAKE_API_KEY)
        tool = OctenSearchResults()
        assert tool.api_wrapper.octen_api_key.get_secret_value() == FAKE_API_KEY

    @patch("octen.Octen")
    def test_run_returns_results(self, mock_octen_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_octen_cls.return_value = mock_client
        mock_client.search.search.return_value = _make_search_response(SAMPLE_RESULTS)

        tool = OctenSearchResults(octen_api_key=SecretStr(FAKE_API_KEY))
        results = tool._run("Python")
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["title"] == "Python.org"

    @patch("octen.Octen")
    def test_run_empty_results_raises(self, mock_octen_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_octen_cls.return_value = mock_client
        mock_client.search.search.return_value = _make_search_response([])

        tool = OctenSearchResults(octen_api_key=SecretStr(FAKE_API_KEY))
        with pytest.raises(ToolException, match="No results found"):
            tool._run("xyznonexistent12345")

    @patch("octen.Octen")
    def test_run_sdk_error_returns_error_dict(self, mock_octen_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_octen_cls.return_value = mock_client
        mock_client.search.search.side_effect = RuntimeError("API down")

        tool = OctenSearchResults(octen_api_key=SecretStr(FAKE_API_KEY))
        results = tool._run("test")
        assert len(results) == 1
        assert "error" in results[0]
        assert "API down" in results[0]["error"]

    @pytest.mark.asyncio
    @patch("octen.AsyncOcten")
    async def test_arun_returns_results(self, mock_async_cls: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_async_cls.return_value = mock_client
        mock_client.search.search.return_value = _make_search_response(SAMPLE_RESULTS)

        tool = OctenSearchResults(octen_api_key=SecretStr(FAKE_API_KEY))
        results = await tool._arun("Python")
        assert isinstance(results, list)
        assert len(results) == 2

    def test_tool_config_forwarded_to_wrapper(self) -> None:
        tool = OctenSearchResults(
            octen_api_key=SecretStr(FAKE_API_KEY),
            max_results=10,
            search_type="keyword",
            safesearch="off",
        )
        assert tool.api_wrapper.max_results == 10
        assert tool.api_wrapper.search_type == "keyword"
        assert tool.api_wrapper.safesearch == "off"
