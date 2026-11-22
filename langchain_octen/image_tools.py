"""Octen Image Search and Design Search tools.

The model chooses what to search for (and the source-page domain filters of Image
Search); result counts and every option that changes the size of a response are set by
the developer. ``data`` is the response's data, unchanged.
"""

from __future__ import annotations

import copy
import ipaddress
import re
from contextvars import ContextVar
from datetime import datetime
from fractions import Fraction
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type
from urllib.parse import urlsplit

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

from langchain_octen._client import MAY_HAVE_BEEN_SENT, classify
from langchain_octen.vertical_tools import (
    _PLACEHOLDER_QUERY,
    Language,
    _is_rfc3339,
    _OctenVerticalTool,
    logger,
)

# Limits from https://docs.octen.ai/api-reference/image-search.md
_IMAGE_MAX_COUNT = 10
_IMAGE_MAX_DOMAINS = 10
# The image is the model's choice, so these rejections go back to it.
_IMAGE_INPUT_HINTS = {
    413: "The image is too large: it must be at most 5MB. Try a smaller image or another "
    "image URL.",
    415: "The image format is not supported: use a JPEG, PNG, WEBP, BMP, TIFF, ICO, DIB, "
    "ICNS or SGI image, or try another image URL.",
    429: "The rate limit was exceeded: wait before trying again.",
}
_IMAGE_URL_HINT = "Try another image URL."

# --- Image Search ------------------------------------------------------------------------

_IMAGE_DESCRIPTION = (
    "Find images on the web by a text description or by a reference image. Returns "
    "ranked image results: the image `url`, the `source_page` that hosts it, `title`, "
    "`width` and `height`, a `thumbnail` URL and, when available, a text `description` "
    "of the image.\n\n"
    "Pass exactly one input: `query` (text describing the images to find, max 500 "
    "characters) or `image_url` (a public https URL of an image with a domain name, to "
    "find visually similar images). Never both. include_domains / exclude_domains (up to "
    "10 each) are matched against each result's source page.\n\n"
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
_BAD_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
# ASCII that cannot appear in a host; non-ASCII is left to the server, since a false
# rejection of a valid internationalized host is worse than a server-side 4xx.
_NOT_IN_HOST = frozenset("\\<>\"'`@[]")


def _bad_escape_outside_query(raw: str) -> bool:
    """A malformed % escape anywhere but the query, which the server keeps raw."""
    fragment = raw.find("#")
    query = raw.find("?")
    if query != -1 and (fragment == -1 or query < fragment):
        raw = raw[:query] + (raw[fragment:] if fragment != -1 else "")
    return bool(_BAD_ESCAPE.search(raw))


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _valid_authority(netloc: str) -> bool:
    """A domain-name host: the API refuses IP addresses (so any [IPv6] literal) as hosts."""
    userinfo, _, hostport = netloc.rpartition("@")
    if any(ch in "@[]" for ch in userinfo):
        return False
    host = hostport.split(":", 1)[0]
    if not host or any(ch in _NOT_IN_HOST for ch in hostport):
        return False
    return not _is_ip(host)


# The model's image URL for the call in progress (None for a text query); a signed URL can
# carry a token, so logs name only the input kind and messages replace the URL.
_CURRENT_IMAGE_URL: ContextVar[Optional[str]] = ContextVar("_CURRENT_IMAGE_URL", default=None)


def _without_upstream_text(exc: Exception) -> Exception:
    """``exc`` rebuilt without upstream text: the known Octen classes as themselves (with
    their status, request id and retry metadata), anything else as a plain OctenError."""
    kind, rid = type(exc), getattr(exc, "request_id", None)
    text = (
        f"Octen Image Search failed for an image URL input ({kind.__name__}); the "
        "upstream message is withheld because it may contain the URL."
    )
    if kind is octen.OctenAPIError:
        return octen.OctenAPIError(
            text, status_code=getattr(exc, "status_code", None), request_id=rid
        )
    if kind is octen.OctenAuthenticationError:
        return octen.OctenAuthenticationError(text, request_id=rid)
    if kind is octen.OctenRateLimitError:
        return octen.OctenRateLimitError(
            text, retry_after=getattr(exc, "retry_after", None), request_id=rid
        )
    if kind is octen.OctenValidationError:
        return octen.OctenValidationError(text)
    if kind is octen.OctenTimeoutError:
        timeout = getattr(exc, "timeout", None)
        return octen.OctenTimeoutError(
            text, timeout=timeout, request_sent=getattr(exc, "request_sent", None)
        )
    if kind is octen.OctenConnectionError:
        return octen.OctenConnectionError(text, request_sent=getattr(exc, "request_sent", None))
    return octen.OctenError(text, request_id=rid) if rid else octen.OctenError(text)


_INSTANT = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)")


def _instant(value: str) -> Tuple[int, Fraction]:
    """The UTC second and exact fraction of an RFC 3339 time checked by ``_is_rfc3339``.

    Integer arithmetic, so an offset near year 1 or 9999 cannot overflow a datetime.
    """
    match = _INSTANT.fullmatch(value)
    assert match is not None
    base, fraction, offset = match.groups()
    local = datetime.fromisoformat(base)
    seconds = local.toordinal() * 86400 + local.hour * 3600 + local.minute * 60 + local.second
    if offset != "Z":
        shift = int(offset[1:3]) * 3600 + int(offset[4:6]) * 60
        seconds -= shift if offset[0] == "+" else -shift
    exact = Fraction(int(fraction), 10 ** len(fraction)) if fraction else Fraction(0)
    return seconds, exact


def _not_blank(value: Optional[str]) -> Optional[str]:
    if value is not None and not value.strip():
        raise ValueError("must not be blank")
    return value


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
        description="Public https URL of an image, with a domain name (no IP address), to "
        "find visually similar images on the web. Do not pass it together with query.",
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

    _check_query = field_validator("query", mode="after")(_not_blank)

    @field_validator("image_url", mode="after")
    @classmethod
    def _check_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
            raise ValueError("must be a URL without control characters")
        if " " in value:  # ASCII only: other whitespace is left to the URL parser
            raise ValueError("must be a URL without spaces")
        try:
            parsed = urlsplit(value)
            escapes_ok = not _bad_escape_outside_query(value)
            ok = parsed.scheme.lower() == "https" and _valid_authority(parsed.netloc) and escapes_ok
            parsed.port  # raises ValueError on a non-numeric or out-of-range port
        except ValueError:  # e.g. an unclosed bracket or a bad port
            ok = False
        if not ok:
            raise ValueError(
                "must be a public https URL of an image with a domain name (no IP address); "
                "base64 data and data: URIs are not accepted"
            )
        return value

    @model_validator(mode="after")
    def _one_input(self) -> "OctenImageSearchInput":
        if self.query is not None and self.image_url is not None:
            raise ValueError(_BOTH_INPUTS)
        if self.query is None and self.image_url is None:
            raise ValueError(_NO_INPUT)
        return self


class OctenImageSearch(_OctenVerticalTool):
    """Octen Image Search: images on the web, found by text or by a reference image.

    Returns ``{"data": ..., "request_id": ..., "warning"?: ...}`` where ``data`` is the
    response's data unchanged: ``results`` with ``title``, ``url``, ``source_page``,
    ``description``, ``width``, ``height`` and ``thumbnail`` per image.

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

    def _model_error(self, exc: Exception) -> Optional[str]:
        url = _CURRENT_IMAGE_URL.get()
        status = getattr(exc, "status_code", None)
        if isinstance(exc, octen.OctenAPIError) and status in _IMAGE_INPUT_HINTS:
            message: Optional[str] = f"{exc}. {_IMAGE_INPUT_HINTS[status]}"
        else:
            message = classify(exc)
        request_id = getattr(exc, "request_id", None)
        if message is None:
            kind = "image URL" if url else "text query"
            logger.error("%s (input: %s, request_id=%s)", type(exc).__name__, kind, request_id)
            return None
        if not url:
            return message
        # Upstream text can echo the signed URL in a form a replace would miss, so a fixed
        # message is built from the status and request id only.
        details = [type(exc).__name__]
        if status is not None:
            details.append(f"status {status}")
        if request_id:
            details.append(f"request_id {request_id}")
        if MAY_HAVE_BEEN_SENT in message:
            # Possibly billed: no suggestion to try again, only the warning.
            return f"Image search by image URL failed ({', '.join(details)}). {MAY_HAVE_BEEN_SENT}"
        hint = _IMAGE_INPUT_HINTS.get(status) if isinstance(status, int) else None
        fixed = (
            f"Image search by image URL failed ({', '.join(details)}). {hint or _IMAGE_URL_HINT}"
        )
        return fixed

    def _failure(self, exc: Exception) -> Tuple[Optional[Exception], bool]:
        if not _CURRENT_IMAGE_URL.get():
            return super()._failure(exc)
        # Image URL call: neither the result nor the chain may carry upstream text.
        message = self._model_error(exc)
        if message is not None:
            return ToolException(message), False
        return _without_upstream_text(exc), False

    def _build_request_model(self, options: Dict[str, Any]) -> None:
        inputs = [{"type": "text", "data": _PLACEHOLDER_QUERY}]
        self._request_model(inputs=inputs, **{k: v for k, v in options.items() if v is not None})

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
        token = _CURRENT_IMAGE_URL.set(image_url)
        try:
            return self._call(self._request({**args, **domains}))
        finally:
            _CURRENT_IMAGE_URL.reset(token)

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
        token = _CURRENT_IMAGE_URL.set(image_url)
        try:
            return await self._acall(self._request({**args, **domains}))
        finally:
            _CURRENT_IMAGE_URL.reset(token)


# --- Design Search -----------------------------------------------------------------------

_DESIGN_DESCRIPTION = (
    "Find design references for building or restyling a user interface or a slide deck. "
    "Each image result is a screenshot of a web page, a page section, a mobile app "
    "screen or a slide (`type`: web_page, web_section, app_screen or slide) with its "
    "`url`, `thumbnail`, `title`, a text `description` and the `source_page` it comes "
    "from (not for app screens). When returned, `design_facts` holds its design values "
    "(colors, fonts, font sizes and the like) and `snippet` reusable HTML code.{web}\n\n"
    "Pass `query` (max 500 characters) describing what to find, e.g. 'pricing section "
    "for a SaaS landing page' or 'iOS onboarding screen'.\n\n"
    "Sections of one web page and slides of one deck carry a `group` (group_id, type, "
    "name, position, total). Calling again with `group_id` set to that result's "
    "group.group_id returns the first images of that group in group order, {limit}, "
    "instead of a search; `query` may then be omitted. `position` and `total` show "
    "where each image sits in its group."
)
# The API's default for images.count.
_DESIGN_DEFAULT_IMAGES = 5
_DESIGN_WEB_SENTENCE = (
    " Web results on the same query (title, url, highlight) come back alongside in `results`."
)
# With images.enable false only the web results come back, so group_id is not offered.
_DESIGN_WEB_ONLY_DESCRIPTION = (
    "Search the web for pages about user interface and slide design, e.g. 'pricing "
    "section for a SaaS landing page'. Returns ranked web results (title, url, highlight) "
    "in `results`. Pass `query` (max 500 characters)."
)
_NEEDS_QUERY = (
    "This tool returns no images, so group_id cannot be used on its own: pass `query`. "
    "No request was made."
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
        description="The group.group_id of an earlier image result, to get the first "
        "images of that web page or deck in group order instead of searching.",
    )

    _check_query = field_validator("query", mode="after")(_not_blank)

    @model_validator(mode="after")
    def _query_or_group(self) -> "OctenDesignSearchInput":
        if self.query is None and self.group_id is None:
            raise ValueError(_NO_QUERY_OR_GROUP)
        return self


class OctenDesignSearchWebInput(BaseModel):
    """Model-facing arguments of Design Search when image results are turned off."""

    # A misspelt argument must reach the model as an error, not run an unfiltered search.
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=500,
        description="What to find design pages for, e.g. 'pricing section for a SaaS "
        "landing page'. Max 500 characters.",
    )

    _check_query = field_validator("query", mode="after")(_not_blank)


class OctenDesignSearch(_OctenVerticalTool):
    """Octen Design Search: design references as screenshots with their design values.

    Returns ``{"data": ..., "request_id": ..., "warning"?: ...}`` where ``data`` is the
    response's data unchanged: ``images`` are the design references (web pages, page
    sections, app screens, slides, with ``group``, ``design_facts`` and ``snippet``) and
    ``results`` the web results on the same query. By default each image carries
    ``design_facts`` and a snippet of up to 5000 tokens; ``images={"design_facts": False,
    "snippet": {"enable": False}}`` turns them off, ``{"snippet": {"max_tokens": ...}}``
    shortens the snippet.

    Instantiate:
        .. code-block:: python

            from langchain_octen import OctenDesignSearch

            tool = OctenDesignSearch(
                images={"count": 5, "snippet": {"max_tokens": 2000}},
                web_search={"enable": False},
            )
    """

    name: str = "octen_design_search"
    description: str = _DESIGN_DESCRIPTION.format(
        limit=f"up to {_DESIGN_DEFAULT_IMAGES} images (the default)", web=_DESIGN_WEB_SENTENCE
    )
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

    def _images_enabled(self) -> bool:
        return (self.images or {}).get("enable") is not False

    @model_validator(mode="after")
    def _state_group_limit(self) -> "OctenDesignSearch":
        count = (self.images or {}).get("count")
        if not self._images_enabled() and "args_schema" not in self.model_fields_set:
            self.args_schema = OctenDesignSearchWebInput
        if "description" in self.model_fields_set:
            return self
        if not self._images_enabled():
            self.description = _DESIGN_WEB_ONLY_DESCRIPTION
            return self
        limit = (
            f"up to {count} images"
            if isinstance(count, int)
            else (f"up to {_DESIGN_DEFAULT_IMAGES} images (the default)")
        )
        web_on = (self.web_search or {}).get("enable") is not False
        self.description = _DESIGN_DESCRIPTION.format(
            limit=limit, web=_DESIGN_WEB_SENTENCE if web_on else ""
        )
        return self

    def _check_query(self, query: Optional[str]) -> None:
        if query is None and not self._images_enabled():
            raise ToolException(_NEEDS_QUERY)

    def _build_request_model(self, options: Dict[str, Any]) -> None:
        images, web = options["images"] or {}, options["web_search"] or {}
        if "group_id" in images:
            raise ValueError(
                f"invalid {self.name} option: images.group_id is chosen per call by the "
                "model, not set on the tool"
            )
        for key in ("highlight", "full_content"):
            nested = web.get(key)
            extra = set(nested) - _HIGHLIGHT_KEYS if isinstance(nested, dict) else set()
            # repr first: keys of mixed or non-string types cannot be sorted or joined.
            unknown = sorted(repr(k) for k in extra)
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
        start, end = web.get("start_time"), web.get("end_time")
        if start is not None and end is not None and _instant(end) < _instant(start):
            raise ValueError(
                f"invalid {self.name} option: web_search.end_time is before start_time"
            )
        if not self._images_enabled() and web.get("enable") is False:
            raise ValueError(
                f"invalid {self.name} option: images.enable and web_search.enable are both "
                "false, so every call would return nothing"
            )

    def _request(self, model_args: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if model_args["query"] is not None:
            kwargs["query"] = model_args["query"]
        images = self.images
        if model_args["group_id"] is not None and self._images_enabled():
            images = {**(images or {}), "group_id": model_args["group_id"]}
        if images is not None:
            kwargs["images"] = images
        if self.web_search is not None:
            kwargs["web_search"] = self.web_search
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        return kwargs

    def _run(  # type: ignore[override]
        self,
        query: Optional[str] = None,
        group_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        self._check_query(query)
        return self._call(self._request({"query": query, "group_id": group_id}))

    async def _arun(  # type: ignore[override]
        self,
        query: Optional[str] = None,
        group_id: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        self._check_query(query)
        return await self._acall(self._request({"query": query, "group_id": group_id}))
