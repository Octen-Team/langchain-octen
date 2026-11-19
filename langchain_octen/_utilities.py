"""Octen Search API wrapper for LangChain."""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, SecretStr, model_validator

from langchain_octen._client import async_client, sync_client

SEARCH_TYPE_DEPRECATION = (
    "search_type is ignored: the Octen API no longer accepts it. "
    "It will be removed in a future release."
)


class OctenSearchAPIWrapper(BaseModel):
    """Wrapper around the Octen Search API.

    Uses the ``octen`` Python SDK to execute web searches.
    A new client is created and closed per call to stay Pydantic-serialization safe.
    """

    octen_api_key: SecretStr
    max_results: int = 5
    search_type: Optional[str] = None
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    include_text: Optional[List[str]] = None
    exclude_text: Optional[List[str]] = None
    time_basis: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    highlight_enabled: Optional[bool] = None
    highlight_max_tokens: Optional[int] = None
    format: Optional[str] = None
    safesearch: Optional[str] = None
    full_content_enabled: Optional[bool] = None
    full_content_max_tokens: Optional[int] = None
    timeout: Optional[float] = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.search_type is not None:
            warnings.warn(SEARCH_TYPE_DEPRECATION, FutureWarning, stacklevel=2)

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Read OCTEN_API_KEY from environment if not provided."""
        if not values.get("octen_api_key"):
            api_key = os.environ.get("OCTEN_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "OCTEN_API_KEY must be provided as a parameter or set as an "
                    "environment variable."
                )
            values["octen_api_key"] = SecretStr(api_key)
        return values

    def _build_search_kwargs(
        self,
        query: str,
        *,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build kwargs dict for the SDK search call."""
        # search_type is deliberately not forwarded: octen>=0.5 rejects it.
        kwargs: Dict[str, Any] = {"query": query, "count": self.max_results}

        # Merge per-call domain filters with instance defaults
        inc = include_domains or self.include_domains
        exc = exclude_domains or self.exclude_domains
        if inc:
            kwargs["include_domains"] = inc
        if exc:
            kwargs["exclude_domains"] = exc

        if self.include_text:
            kwargs["include_text"] = self.include_text
        if self.exclude_text:
            kwargs["exclude_text"] = self.exclude_text
        if self.time_basis is not None:
            kwargs["time_basis"] = self.time_basis
        if self.start_time is not None:
            kwargs["start_time"] = self.start_time
        if self.end_time is not None:
            kwargs["end_time"] = self.end_time
        if self.format is not None:
            kwargs["format"] = self.format
        if self.safesearch is not None:
            kwargs["safesearch"] = self.safesearch
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout

        # Highlight options
        if self.highlight_enabled is not None:
            from octen.models.search import HighlightOptions

            highlight_kwargs: Dict[str, Any] = {"enable": self.highlight_enabled}
            if self.highlight_max_tokens is not None:
                highlight_kwargs["max_tokens"] = self.highlight_max_tokens
            kwargs["highlight"] = HighlightOptions(**highlight_kwargs)

        # Full content options
        if self.full_content_enabled is not None:
            from octen.models.search import FullContentOptions

            fc_kwargs: Dict[str, Any] = {"enable": self.full_content_enabled}
            if self.full_content_max_tokens is not None:
                fc_kwargs["max_tokens"] = self.full_content_max_tokens
            kwargs["full_content"] = FullContentOptions(**fc_kwargs)

        return kwargs

    @staticmethod
    def _format_results(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize raw result dicts, omitting None-valued fields."""
        cleaned: List[Dict[str, Any]] = []
        for r in raw_results:
            entry = {k: v for k, v in r.items() if v is not None}
            cleaned.append(entry)
        return cleaned

    def results(
        self,
        query: str,
        *,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a synchronous search and return cleaned results."""
        kwargs = self._build_search_kwargs(
            query, include_domains=include_domains, exclude_domains=exclude_domains
        )
        client = sync_client(self.octen_api_key)
        try:
            response = client.search.search(**kwargs)
            return self._format_results(response.results)
        finally:
            client.close()

    async def results_async(
        self,
        query: str,
        *,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute an asynchronous search and return cleaned results."""
        kwargs = self._build_search_kwargs(
            query, include_domains=include_domains, exclude_domains=exclude_domains
        )
        client = async_client(self.octen_api_key)
        try:
            response = await client.search.search(**kwargs)
            return self._format_results(response.results)
        finally:
            await client.close()
