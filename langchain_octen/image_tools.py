"""Octen Image Search and Design Search tools.

The model chooses what to search for (and the source-page domain filters of Image
Search); result counts and every option that changes the size of a response are set by
the developer. Results are trimmed to the documented fields the model can cite and show.
"""

from __future__ import annotations

import copy
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type
from urllib.parse import urlparse

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
    StrictStr,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated, Literal, get_args

from langchain_octen._client import MAY_HAVE_BEEN_SENT
from langchain_octen.vertical_tools import (
    _PLACEHOLDER_QUERY,
    Language,
    _is_rfc3339,
    _OctenVerticalTool,
)

# Limits from https://docs.octen.ai/api-reference/image-search.md
_IMAGE_MAX_COUNT = 10
_IMAGE_MAX_DOMAINS = 10

# Documented result fields; anything else the server returns is dropped.
_IMAGE_FIELDS = ("title", "url", "source_page", "description", "width", "height", "thumbnail")
_DESIGN_IMAGE_FIELDS = (
    "type", "title", "description", "url", "thumbnail", "source_page", "width", "height",
    "group", "design_facts", "snippet",
)  # fmt: skip
_GROUP_FIELDS = ("group_id", "type", "name", "position", "total", "master")
# favicon and in-body images are left out: they are page chrome, not citation material.
_DESIGN_WEB_FIELDS = (
    "title", "url", "highlight", "full_content", "authors", "time_published",
    "time_last_crawled", "cover_image",
)  # fmt: skip


class _MalformedResult(Exception):
    pass


def _pick(item: Any, fields: Tuple[str, ...]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise _MalformedResult(f"expected an object, got {type(item).__name__}")
    return {k: item[k] for k in fields if item.get(k) not in (None, "")}


def _pick_all(items: Any, fields: Tuple[str, ...]) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        raise _MalformedResult(f"expected a list, got {type(items).__name__}")
    return [_pick(item, fields) for item in items]


class _OctenTrimmedTool(_OctenVerticalTool):
    """Vertical tool whose ``data`` is trimmed to the documented result fields."""

    def _trim(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def _result(self, response: Any) -> Dict[str, Any]:  # type: ignore[override]
        result = super()._result(response)
        try:
            result["data"] = self._trim(result["data"])
        except _MalformedResult as exc:
            raise ToolException(
                f"Octen returned a malformed {self.name} response ({exc}; request_id "
                f"{result['request_id']}). {MAY_HAVE_BEEN_SENT}"
            ) from None
        return result


# --- Image Search ------------------------------------------------------------------------

_IMAGE_DESCRIPTION = (
    "Find images on the web by a text description or by a reference image. Returns "
    "ranked image results: the image `url`, the `source_page` that hosts it, `title`, "
    "`width` and `height`, a `thumbnail` URL and, when available, a text `description` "
    "of the image.\n\n"
    "Pass exactly one input: `query` (text describing the images to find, max 500 "
    "characters) or `image_url` (the public URL of an image, to find visually similar "
    "images). Never both. include_domains / exclude_domains (up to 10 each) are matched "
    "against each result's source page.\n\n"
    "Use this when the user wants pictures, photos, diagrams, screenshots or other "
    "visual references, not for general text web search. When you show or cite an "
    "image, also give its source_page."
)

_BOTH_INPUTS = (
    "Pass either `query` or `image_url`, not both: this search takes a single input. "
    "No request was made."
)
_NO_INPUT = (
    "Pass `query` (text describing the images to find) or `image_url` (the public URL "
    "of an image, to find visually similar images). No request was made."
)
_Domains = Optional[List[str]]


class OctenImageSearchInput(BaseModel):
    """Model-facing arguments of Image Search."""

    # A misspelt argument must reach the model as an error, not run an unfiltered search.
    model_config = ConfigDict(extra="forbid")

    query: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=500,
        description="Text describing the images to find, e.g. 'golden gate bridge at "
        "night'. Max 500 characters. Do not pass it together with image_url.",
    )
    image_url: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Public http(s) URL of an image, to find visually similar images on "
        "the web. Do not pass it together with query.",
    )
    include_domains: _Domains = Field(
        default=None,
        max_length=_IMAGE_MAX_DOMAINS,
        description="Only return images whose source page is on these domains, e.g. "
        "'wikipedia.org'. Up to 10.",
    )
    exclude_domains: _Domains = Field(
        default=None,
        max_length=_IMAGE_MAX_DOMAINS,
        description="Drop images whose source page is on these domains. Up to 10.",
    )

    @field_validator("image_url", mode="after")
    @classmethod
    def _check_url(cls, value: Optional[str]) -> Optional[str]:
        parsed = urlparse(value) if value is not None else None
        if parsed is not None and (
            parsed.scheme.lower() not in ("http", "https") or not parsed.netloc
        ):
            raise ValueError(
                "must be the public http(s) URL of an image; base64 data and data: URIs "
                "are not accepted"
            )
        return value

    @model_validator(mode="after")
    def _one_input(self) -> "OctenImageSearchInput":
        if self.query is not None and self.image_url is not None:
            raise ValueError(_BOTH_INPUTS)
        if self.query is None and self.image_url is None:
            raise ValueError(_NO_INPUT)
        return self


class OctenImageSearch(_OctenTrimmedTool):
    """Octen Image Search: images on the web, found by text or by a reference image.

    Returns ``{"data": {"results": [...]}, "request_id": ..., "warning"?: ...}``; each
    result keeps the documented fields ``title``, ``url``, ``source_page``,
    ``description``, ``width``, ``height`` and ``thumbnail`` (empty ones are omitted).

    Instantiate:
        .. code-block:: python

            from langchain_octen import OctenImageSearch

            tool = OctenImageSearch(count=5)
    """

    name: str = "octen_image_search"
    description: str = _IMAGE_DESCRIPTION
    args_schema: Type[BaseModel] = OctenImageSearchInput

    count: Optional[Annotated[int, Field(ge=1, le=_IMAGE_MAX_COUNT, strict=True)]] = None
    """Number of results (1-10). Unset uses the server default (5)."""
    safesearch: Optional[Literal["off", "strict"]] = None
    """Adult content filter. Unset uses the server default (``strict``)."""

    _sdk_method: ClassVar[Tuple[str, str]] = ("image_search", "search")
    _request_model: ClassVar[Any] = octen.ImageSearchRequest
    _developer_options: ClassVar[Tuple[str, ...]] = ("count", "safesearch")

    def _build_request_model(self, options: Dict[str, Any]) -> None:
        inputs = [{"type": "text", "data": _PLACEHOLDER_QUERY}]
        self._request_model(inputs=inputs, **{k: v for k, v in options.items() if v is not None})

    def _trim(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"results": _pick_all(data.get("results", []), _IMAGE_FIELDS)}

    def _run(  # type: ignore[override]
        self,
        query: Optional[str] = None,
        image_url: Optional[str] = None,
        include_domains: _Domains = None,
        exclude_domains: _Domains = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        args = {"query": query, "image_url": image_url}
        domains = {"include_domains": include_domains, "exclude_domains": exclude_domains}
        return self._call(self._request({**args, **domains}))

    async def _arun(  # type: ignore[override]
        self,
        query: Optional[str] = None,
        image_url: Optional[str] = None,
        include_domains: _Domains = None,
        exclude_domains: _Domains = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        args = {"query": query, "image_url": image_url}
        domains = {"include_domains": include_domains, "exclude_domains": exclude_domains}
        return await self._acall(self._request({**args, **domains}))


# --- Design Search -----------------------------------------------------------------------

_DESIGN_DESCRIPTION = (
    "Find design references for building or restyling a user interface or a slide deck. "
    "Each image result is a screenshot of a web page, a page section, a mobile app "
    "screen or a slide (`type`: web_page, web_section, app_screen or slide) with its "
    "`url`, `thumbnail`, `title`, a text `description` and the `source_page` it comes "
    "from (not for app screens). When returned, `design_facts` holds its design values "
    "(colors, fonts, font sizes and the like) and `snippet` reusable HTML code. Web "
    "results on the same query (title, url, highlight) come back alongside in "
    "`results`.\n\n"
    "Pass `query` (max 500 characters) describing what to find, e.g. 'pricing section "
    "for a SaaS landing page' or 'iOS onboarding screen'.\n\n"
    "Sections of one web page and slides of one deck carry a `group` (group_id, type, "
    "name, position, total). To get the images of that page or deck in group order, "
    "call again with `group_id` set to that result's group.group_id; `query` may then "
    "be omitted. With `group_id` the images come from that group instead of a search."
)

_NO_QUERY_OR_GROUP = (
    "Pass `query` to search for design references, or `group_id` (the group.group_id "
    "of an earlier result) to get the images of that web page or deck. No request was "
    "made."
)

_HIGHLIGHT_KEYS = frozenset({"enable", "max_tokens"})
_LANGUAGES = frozenset(get_args(Language))


class OctenDesignSearchInput(BaseModel):
    """Model-facing arguments of Design Search."""

    # A misspelt argument must reach the model as an error, not run an unfiltered search.
    model_config = ConfigDict(extra="forbid")

    query: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=500,
        description="What to find design references for, e.g. 'pricing section for a "
        "SaaS landing page'. Max 500 characters. May be omitted when group_id is set.",
    )
    group_id: Optional[StrictStr] = Field(
        default=None,
        min_length=1,
        description="The group.group_id of an earlier image result, to get the images of "
        "that web page or deck in group order instead of searching.",
    )

    @model_validator(mode="after")
    def _query_or_group(self) -> "OctenDesignSearchInput":
        if self.query is None and self.group_id is None:
            raise ValueError(_NO_QUERY_OR_GROUP)
        return self


class OctenDesignSearch(_OctenTrimmedTool):
    """Octen Design Search: design references as screenshots with their design values.

    Returns ``{"data": {"query"?, "images"?, "results"?}, "request_id": ...,
    "warning"?: ...}``: ``images`` are the design references (web pages, page sections,
    app screens, slides; with ``group``, ``design_facts`` and ``snippet`` when returned),
    ``results`` the web results on the same query. Each item keeps its documented fields.

    Instantiate:
        .. code-block:: python

            from langchain_octen import OctenDesignSearch

            tool = OctenDesignSearch(
                images={"count": 5, "snippet": {"max_tokens": 2000}},
                web_search={"enable": False},
            )
    """

    name: str = "octen_design_search"
    description: str = _DESIGN_DESCRIPTION
    args_schema: Type[BaseModel] = OctenDesignSearchInput

    images: Optional[Dict[str, Any]] = None
    """Image result options, e.g. ``{"count": 5, "design_facts": True, "snippet":
    {"enable": True, "max_tokens": 2000}}``. ``group_id`` is the model's, not an option."""
    web_search: Optional[Dict[str, Any]] = None
    """Web result options, e.g. ``{"enable": True, "count": 3, "highlight": {"enable":
    True, "max_tokens": 300}}``; they do not affect the image results."""

    _sdk_method: ClassVar[Tuple[str, str]] = ("design_search", "search")
    _request_model: ClassVar[Any] = octen.DesignSearchRequest
    _developer_options: ClassVar[Tuple[str, ...]] = ("images", "web_search")

    @field_validator("images", mode="after")
    @classmethod
    def _copy_images(cls, value: Any) -> Any:
        # Keep a private copy: later changes to the caller's dict must not leak in.
        return copy.deepcopy(value)

    def _build_request_model(self, options: Dict[str, Any]) -> None:
        images, web = options["images"] or {}, options["web_search"] or {}
        if "group_id" in images:
            raise ValueError(
                f"invalid {self.name} option: images.group_id is chosen per call by the "
                "model, not set on the tool"
            )
        for key in ("highlight", "full_content"):
            nested = web.get(key)
            unknown = sorted(set(nested) - _HIGHLIGHT_KEYS) if isinstance(nested, dict) else []
            if unknown:
                raise ValueError(
                    f"invalid {self.name} option: web_search.{key}: unknown option "
                    f"{', '.join(unknown)}"
                )
        self._request_model(query=_PLACEHOLDER_QUERY, **options)
        language = web.get("language") or []
        bad = sorted(set(language) - _LANGUAGES)
        if bad:
            raise ValueError(
                f"invalid {self.name} option: web_search.language: unsupported "
                f"{', '.join(bad)}; use ISO 639-1 codes from {', '.join(sorted(_LANGUAGES))}"
            )
        for key in ("start_time", "end_time"):
            if web.get(key) is not None and not _is_rfc3339(web[key]):
                raise ValueError(
                    f"invalid {self.name} option: web_search.{key}: must be an RFC 3339 "
                    "time, e.g. 2025-01-01T00:00:00Z"
                )

    def _request(self, model_args: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if model_args["query"] is not None:
            kwargs["query"] = model_args["query"]
        images = self.images
        if model_args["group_id"] is not None:
            images = {**(images or {}), "group_id": model_args["group_id"]}
        if images is not None:
            kwargs["images"] = images
        if self.web_search is not None:
            kwargs["web_search"] = self.web_search
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        return kwargs

    def _trim(self, data: Dict[str, Any]) -> Dict[str, Any]:
        trimmed: Dict[str, Any] = {}
        if data.get("query"):
            trimmed["query"] = data["query"]
        if "images" in data:
            images = _pick_all(data["images"], _DESIGN_IMAGE_FIELDS)
            for image in images:
                if "group" in image:
                    image["group"] = _pick(image["group"], _GROUP_FIELDS)
            trimmed["images"] = images
        if "results" in data:
            trimmed["results"] = _pick_all(data["results"], _DESIGN_WEB_FIELDS)
        return trimmed

    def _run(  # type: ignore[override]
        self,
        query: Optional[str] = None,
        group_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        return self._call(self._request({"query": query, "group_id": group_id}))

    async def _arun(  # type: ignore[override]
        self,
        query: Optional[str] = None,
        group_id: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        return await self._acall(self._request({"query": query, "group_id": group_id}))
