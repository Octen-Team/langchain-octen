"""Octen Extract tool: read web pages by URL.

The model only chooses the URLs and an optional focus query. The URL cap and every
option that changes what a call costs or returns are set by the developer.
"""

from __future__ import annotations

import copy
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Type

import octen
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import ToolException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated

from langchain_octen.vertical_tools import _OctenVerticalTool

# Limits from https://docs.octen.ai/api-reference/extract.md
_API_MAX_URLS = 20
_Url = Annotated[str, StringConstraints(min_length=1, max_length=2048)]
# The server's default per-URL timeout, plus the headroom the hosted Octen MCP server
# allows for a batch: url_timeout bounds each page on the server, not the whole call.
_SERVER_URL_TIMEOUT = 30
_REQUEST_HEADROOM = 90
_PLACEHOLDER_URL = "https://example.com/"

_DESCRIPTION = (
    "Read one or more web pages by URL and return their clean content. Without `query` "
    "each result carries the page's `full_content`; this is what you want in almost all "
    "cases. Only pass `query` when you need the passages about one specific topic: each "
    "result then returns ranked `highlights` INSTEAD of the full content, so the page "
    "body will be partial. Results also carry the page title, the publication and last "
    "crawl times when known, and a topical `category` and `page_structure` "
    "classification when detected.\n\n"
    "Use this when you already have the URL(s). To find pages first, use a search "
    "tool.\n\n"
    "Each URL succeeds or fails on its own: a page that could not be read comes back "
    'with status "failed" and an `error_message`, while the other URLs still return '
    "their content."
)


class OctenExtractInput(BaseModel):
    """Model-facing arguments of Extract."""

    # A misspelt argument must reach the model as an error, not be dropped.
    model_config = ConfigDict(extra="forbid")

    urls: List[_Url] = Field(
        min_length=1,
        description="URLs of the pages to read, e.g. 'https://example.com/article'. Each "
        "at most 2048 characters.",
    )
    query: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=500,
        description="Optional; leave it unset in the normal case, and each result returns "
        "the page's full content. Set it only to get ranked excerpts about one specific "
        "topic: each result then returns `highlights` and omits `full_content`. Max 500 "
        "characters.",
    )


class OctenExtract(_OctenVerticalTool):
    """Octen Extract: clean content of web pages, read by URL.

    Returns ``{"data": ..., "request_id": ..., "warning"?: ...}`` where
    ``data["results"]`` holds one item per requested URL, in request order, failures
    included (``status: "failed"`` with ``error_message``; failed URLs are not billed).
    Each successful item's ``resolved_mode`` is the mode it was billed at.

    Instantiate:
        .. code-block:: python

            from langchain_octen import OctenExtract

            tool = OctenExtract(max_urls=5)
    """

    name: str = "octen_extract"
    description: str = _DESCRIPTION
    args_schema: Type[BaseModel] = OctenExtractInput

    max_urls: Annotated[int, Field(ge=1, le=_API_MAX_URLS, strict=True)] = 5
    """Most URLs the model may pass in one call (1-20). More are refused without a request."""
    mode: Optional[Literal["standard", "advanced", "auto"]] = None
    """``"advanced"`` reads hard-to-reach pages at a higher price; ``"auto"`` picks per URL.
    Unset uses the server default (``standard``)."""
    format: Optional[Literal["markdown", "text"]] = None
    """Format of the content. Unset uses the server default (``markdown``)."""
    max_age_seconds: Optional[Annotated[int, Field(ge=300, le=31_536_000, strict=True)]] = None
    """Refetch pages whose cached copy is older than this. Unset uses the server default
    (86400)."""
    url_timeout: Optional[Annotated[int, Field(ge=1, le=60, strict=True)]] = None
    """Server-side timeout per URL in seconds (the API's ``timeout``). Unset uses the server
    default (30)."""
    include_images: Optional[StrictBool] = None
    """Return the images (and cover image) found on each page."""
    include_videos: Optional[StrictBool] = None
    """Return the videos found on each page."""
    include_audio: Optional[StrictBool] = None
    """Return the audio found on each page."""
    include_links: Optional[Dict[str, Any]] = None
    """Return each page's links, e.g. ``{"scope": "prefer_external", "max_links": 50}``;
    ``{}`` uses the server defaults."""
    timeout: Optional[float] = None
    """Client-side timeout for the whole call in seconds. Unset: ``url_timeout`` (or the
    server default, 30) plus 90."""

    _sdk_method: ClassVar[Tuple[str, str]] = ("extract", "extract")
    _request_model: ClassVar[Any] = octen.ExtractRequest
    _developer_options: ClassVar[Tuple[str, ...]] = (
        "mode",
        "format",
        "max_age_seconds",
        "url_timeout",
        "include_images",
        "include_videos",
        "include_audio",
        "include_links",
    )

    @field_validator("include_links", mode="after")
    @classmethod
    def _copy_links(cls, value: Any) -> Any:
        # Keep a private copy: later changes to the caller's dict must not leak in.
        return copy.deepcopy(value)

    @model_validator(mode="after")
    def _state_cap(self) -> "OctenExtract":
        if "description" not in self.model_fields_set:
            self.description = (
                f"{_DESCRIPTION}\n\nPass at most {self.max_urls} URLs per call; split a "
                "longer list into several calls."
            )
        return self

    @staticmethod
    def _sdk_options(options: Dict[str, Any]) -> Dict[str, Any]:
        sdk = {k: v for k, v in options.items() if k != "url_timeout"}
        sdk["timeout"] = options["url_timeout"]
        return sdk

    def _build_request_model(self, options: Dict[str, Any]) -> None:
        self._request_model(urls=[_PLACEHOLDER_URL], **self._sdk_options(options))

    def _request(self, model_args: Dict[str, Any]) -> Dict[str, Any]:
        options = {key: getattr(self, key) for key in self._developer_options}
        sent = {**model_args, **self._sdk_options(options)}
        kwargs = {k: v for k, v in sent.items() if v is not None}
        if self.timeout is not None:
            kwargs["request_timeout"] = self.timeout
        else:
            url_timeout = self.url_timeout or _SERVER_URL_TIMEOUT
            kwargs["request_timeout"] = url_timeout + _REQUEST_HEADROOM
        return kwargs

    def _check_count(self, urls: List[str]) -> None:
        if len(urls) > self.max_urls:
            raise ToolException(
                f"Too many URLs: got {len(urls)} URLs, but this tool reads at most "
                f"{self.max_urls} per call. No request was made; split them into several "
                f"calls of at most {self.max_urls} URLs each."
            )

    def _run(  # type: ignore[override]
        self,
        urls: List[str],
        query: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        self._check_count(urls)
        return self._call(self._request({"urls": urls, "query": query}))

    async def _arun(  # type: ignore[override]
        self,
        urls: List[str],
        query: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        self._check_count(urls)
        return await self._acall(self._request({"urls": urls, "query": query}))
