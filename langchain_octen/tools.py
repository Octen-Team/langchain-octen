"""Octen Search tool for LangChain."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, SecretStr

from langchain_octen._utilities import SEARCH_TYPE_DEPRECATION, OctenSearchAPIWrapper


class OctenSearchInput(BaseModel):
    """Input schema for the Octen search tool.

    Only exposes fields that are safe for an LLM to control per-query.
    """

    query: str = Field(description="The search query string.")
    include_domains: Optional[List[str]] = Field(
        default=None,
        description="Optional list of domains to restrict results to.",
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description="Optional list of domains to exclude from results.",
    )


class OctenSearchResults(BaseTool):
    """Tool that queries the Octen Web Search API and returns structured results.

    Setup:
        Install ``langchain-octen`` and set the ``OCTEN_API_KEY`` environment variable.

        .. code-block:: bash

            pip install langchain-octen
            export OCTEN_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_octen import OctenSearchResults

            tool = OctenSearchResults(max_results=5)

    Invoke directly:
        .. code-block:: python

            tool.invoke({"query": "LangChain Python framework"})

    Use within an agent:
        .. code-block:: python

            from langgraph.prebuilt import create_react_agent

            agent = create_react_agent(llm, [tool])
    """

    name: str = "octen_search"
    description: str = (
        "A web search engine powered by Octen. "
        "Useful for finding up-to-date information on the internet. "
        "Input should be a search query string."
    )
    args_schema: Type[BaseModel] = OctenSearchInput
    handle_tool_error: bool = True

    api_wrapper: OctenSearchAPIWrapper = Field(default=None, exclude=True)  # type: ignore[assignment]

    # --- Config fields (forwarded to OctenSearchAPIWrapper) ---
    octen_api_key: Optional[SecretStr] = None
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

    def __init__(self, **kwargs: Any) -> None:
        """Initialize OctenSearchResults, creating the underlying API wrapper."""
        super().__init__(**kwargs)
        if self.search_type is not None:
            warnings.warn(SEARCH_TYPE_DEPRECATION, DeprecationWarning, stacklevel=2)
        wrapper_kwargs: Dict[str, Any] = {}
        if self.octen_api_key is not None:
            wrapper_kwargs["octen_api_key"] = self.octen_api_key
        wrapper_kwargs["max_results"] = self.max_results
        for field_name in (
            "include_domains",
            "exclude_domains",
            "include_text",
            "exclude_text",
            "time_basis",
            "start_time",
            "end_time",
            "highlight_enabled",
            "highlight_max_tokens",
            "format",
            "safesearch",
            "full_content_enabled",
            "full_content_max_tokens",
            "timeout",
        ):
            val = getattr(self, field_name)
            if val is not None:
                wrapper_kwargs[field_name] = val
        self.api_wrapper = OctenSearchAPIWrapper(**wrapper_kwargs)

    def _run(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronously execute search."""
        try:
            results = self.api_wrapper.results(
                query, include_domains=include_domains, exclude_domains=exclude_domains
            )
        except Exception as e:
            return [{"error": str(e)}]
        if not results:
            raise ToolException(f"No results found for query: {query!r}")
        return results

    async def _arun(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronously execute search."""
        try:
            results = await self.api_wrapper.results_async(
                query, include_domains=include_domains, exclude_domains=exclude_domains
            )
        except Exception as e:
            return [{"error": str(e)}]
        if not results:
            raise ToolException(f"No results found for query: {query!r}")
        return results
