"""Octen vertical search tools: Broad, News, Business, Pulse, Place and Web Search Premier.

The model only controls the query and filters that do not change the cost of a call.
Options that decide how much is returned (and billed) are set by the developer when
the tool is constructed.
"""

from __future__ import annotations

import copy
import logging
import math
import re
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Type, Union

import octen
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated

from langchain_octen._client import async_client, classify, resolve_api_key, sync_client

TimeRange = Literal["day", "week", "month", "year", "d", "w", "m", "y"]
# Values documented at https://docs.octen.ai/api-reference/search-premier.md
Language = Literal[
    "ar", "de", "en", "es", "fr", "hi", "id", "it", "ja",
    "ko", "nl", "pl", "pt", "ru", "th", "tr", "vi", "zh",
]  # fmt: skip
_Domain = Annotated[str, StringConstraints(max_length=60)]

# RFC 3339 date-time, the "format: date-time" of the Web Search Premier API reference.
# Stricter on purpose: uppercase T and Z only, as in the reference's example, and no leap
# second (:60), which Python's datetime cannot represent.
_RFC3339 = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|[+-](\d{2}):(\d{2}))",
    re.ASCII,
)


def _is_rfc3339(value: Any) -> bool:
    match = _RFC3339.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        return False
    year, month, day, hour, minute, second, off_hour, off_minute = match.groups()
    try:
        datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    except ValueError:
        return False
    return off_hour is None or (int(off_hour) <= 23 and int(off_minute) <= 59)


_QUERY_DESCRIPTION = "Search query. Max 500 characters."
_LANGUAGE_DESCRIPTION = (
    "Only return results in these languages, as ISO 639-1 codes. Supported: ar, de, en, "
    "es, fr, hi, id, it, ja, ko, nl, pl, pt, ru, th, tr, vi, zh."
)


def _describe_details(details: Any) -> str:
    parts = []
    for err in details:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = str(err.get("msg", "invalid value"))
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)


def _describe_errors(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    return _describe_details(errors(include_url=False, include_input=False))


def _format_validation_error(exc: Exception) -> str:
    """Tell the model which argument was wrong, so it can retry with a fixed call."""
    return "Invalid tool input: " + _describe_errors(exc)


logger = logging.getLogger("langchain_octen")


def _route(exc: Exception) -> Optional[str]:
    """Return the message for the model, or log and return ``None`` if ``exc`` is raised."""
    message = classify(exc)
    if message is None:
        # Some agent runtimes catch tool exceptions; the log keeps config errors visible.
        logger.error("%s: %s", type(exc).__name__, exc)
    return message


# Stands in for the model's query when developer options are checked at construction.
_PLACEHOLDER_QUERY = "placeholder"


class OctenFilteredSearchInput(BaseModel):
    """Model-facing arguments shared by Broad, News and Business search."""

    # A misspelt argument must reach the model as an error, not run an unfiltered search.
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500, description=_QUERY_DESCRIPTION)
    include_domains: Optional[List[_Domain]] = Field(
        default=None,
        max_length=1200,
        description="Only return results from these domains, e.g. 'arxiv.org'.",
    )
    exclude_domains: Optional[List[_Domain]] = Field(
        default=None,
        max_length=1200,
        description="Drop results from these domains.",
    )
    time_range: Optional[TimeRange] = Field(
        default=None,
        description="Relative time window: day, week, month or year (or d, w, m, y).",
    )
    language: Optional[List[str]] = Field(default=None, description=_LANGUAGE_DESCRIPTION)


class OctenSearchPremierInput(BaseModel):
    """Model-facing arguments of Web Search Premier: the same as ``octen_search``."""

    # A misspelt argument must reach the model as an error, not run an unfiltered search.
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500, description=_QUERY_DESCRIPTION)
    include_domains: Optional[List[_Domain]] = Field(
        default=None,
        max_length=1200,
        description="Only return results from these domains, e.g. 'arxiv.org'.",
    )
    exclude_domains: Optional[List[_Domain]] = Field(
        default=None,
        max_length=1200,
        description="Drop results from these domains.",
    )


class OctenPulseSearchInput(BaseModel):
    """Model-facing arguments of Pulse search."""

    # A misspelt argument must reach the model as an error, not run an unfiltered search.
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=500,
        description="What to find the address for, e.g. 'github docs'. Max 500 characters.",
    )


class OctenUserLocation(BaseModel):
    """Approximate location of the user, for when coordinates are unknown."""

    # A misspelt argument must reach the model as an error, not run an unfiltered search.
    model_config = ConfigDict(extra="forbid")

    city: Optional[str] = Field(default=None, description="City name.")
    region: Optional[str] = Field(default=None, description="State, province or region.")
    country: Optional[str] = Field(default=None, description="Country name or code.")
    timezone: Optional[str] = Field(
        default=None, description="IANA timezone, e.g. 'Asia/Tokyo'."
    )


class OctenPlaceSearchInput(BaseModel):
    """Model-facing arguments of Place search."""

    # A misspelt argument must reach the model as an error, not run an unfiltered search.
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=500,
        description="What to look for, e.g. 'coffee shops' or 'ramen in Shinjuku'. "
        "Max 500 characters.",
    )
    latitude: Optional[float] = Field(
        default=None,
        ge=-90,
        le=90,
        description="WGS84 latitude to search around. Must be sent with longitude.",
    )
    longitude: Optional[float] = Field(
        default=None,
        ge=-180,
        le=180,
        description="WGS84 longitude to search around. Must be sent with latitude.",
    )
    radius: Optional[int] = Field(
        default=None,
        ge=1,
        le=50000,
        description="Keep only places within this many metres of the point. Requires "
        "latitude and longitude. Omit for no distance limit.",
    )
    user_location: Optional[OctenUserLocation] = Field(
        default=None,
        description="Approximate location, for when coordinates are unknown. No distance "
        "is returned without coordinates.",
    )

    @model_validator(mode="after")
    def _check_point(self) -> "OctenPlaceSearchInput":
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be given together")
        if self.radius is not None and self.latitude is None:
            raise ValueError("radius needs latitude and longitude; it measures from that point")
        return self


class _OctenVerticalTool(BaseTool):
    """Shared key handling, client lifecycle, result shape and error routing."""

    # A misspelt option would otherwise be ignored and silently fall back to the server default.
    # Errors never echo inputs: a misspelt key kwarg would otherwise print the API key.
    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", hide_input_in_errors=True
    )

    handle_tool_error: bool = True
    handle_validation_error: Union[bool, str, Any] = _format_validation_error

    octen_api_key: Optional[SecretStr] = None
    """Octen API key. Falls back to the ``OCTEN_API_KEY`` environment variable."""
    timeout: Optional[float] = None
    """Request timeout in seconds. Unset uses the SDK default."""

    _sdk_method: ClassVar[Tuple[str, str]]
    _developer_options: ClassVar[Tuple[str, ...]]
    _request_model: ClassVar[Any]

    def __init__(self, **kwargs: Any) -> None:
        # pydantic keeps the raw input in errors() and json(), which would carry a misspelt
        # API key; re-raise a plain ValueError built without inputs, outside the except block
        # so the original is neither the cause nor the context.
        error: Optional[ValueError] = None
        try:
            super().__init__(**kwargs)
        except ValidationError as exc:
            details = exc.errors(include_url=False, include_context=False, include_input=False)
            error = ValueError(
                f"invalid {type(self).__name__} configuration: {_describe_details(details)}"
            )
        if error is not None:
            raise error

    @field_validator(
        "highlight",
        "full_content",
        "subjects",
        "entities",
        "places",
        "web_search",
        mode="after",
        check_fields=False,
    )
    @classmethod
    def _copy_option(cls, value: Any) -> Any:
        # Keep a private copy: later changes to the caller's nested dicts must not leak in.
        return copy.deepcopy(value)

    @field_validator("response_format", mode="after")
    @classmethod
    def _check_response_format(cls, value: Any) -> Any:
        # The tools return a dict; "content_and_artifact" would only fail after a paid call.
        if value != "content":
            raise ValueError('response_format must be "content"')
        return value

    @field_validator("timeout", mode="before")
    @classmethod
    def _check_timeout(cls, value: Any) -> Any:
        if value is None:
            return value
        message = "timeout must be a positive, finite number of seconds"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{message}, got {type(value).__name__}")
        try:
            finite = math.isfinite(value)
        except OverflowError:  # an int too large for a float
            finite = False
        if not finite or value <= 0:
            raise ValueError(message)
        return value

    @model_validator(mode="after")
    def _check_developer_options(self) -> "_OctenVerticalTool":
        options = {key: getattr(self, key) for key in self._developer_options}
        try:
            self._build_request_model(options)
        except ValidationError as exc:
            raise ValueError(f"invalid {self.name} option: {_describe_errors(exc)}") from None
        return self

    def _build_request_model(self, options: Dict[str, Any]) -> None:
        """Build the request model octen will build, so bad options fail here, not per call."""
        self._request_model(query=_PLACEHOLDER_QUERY, **options)

    @model_validator(mode="before")
    @classmethod
    def _resolve_key(cls, values: Any) -> Any:
        if isinstance(values, dict):
            explicit = values.get("octen_api_key")
            if isinstance(explicit, str):
                explicit = SecretStr(explicit)
            elif explicit is not None and not isinstance(explicit, SecretStr):
                raise ValueError(
                    "octen_api_key must be a str or SecretStr, "
                    f"got {type(explicit).__name__}"
                )
            values["octen_api_key"] = resolve_api_key(explicit)
        return values

    def _api_key(self) -> SecretStr:
        # Resolved at construction; resolve again in case it was reassigned to None.
        return resolve_api_key(self.octen_api_key)

    def _request(self, model_args: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        for key, value in model_args.items():
            if value is None:
                continue
            if isinstance(value, BaseModel):
                value = value.model_dump(exclude_none=True)
            kwargs[key] = value
        for key in self._developer_options:
            value = getattr(self, key)
            if value is not None:
                kwargs[key] = value
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        return kwargs

    @staticmethod
    def _result(response: Any) -> Dict[str, Any]:
        extra = getattr(response, "model_extra", None) or {}
        result: Dict[str, Any] = {
            "data": response.data,
            "request_id": extra.get("request_id", getattr(response, "request_id", None)),
        }
        warning = response.warning
        if warning:
            result["warning"] = warning
        return result

    def _model_error(self, exc: Exception) -> Optional[str]:
        """Return the message for the model, or ``None`` if ``exc`` must be raised."""
        return _route(exc)

    def _method(self, client: Any) -> Any:
        resource, method = self._sdk_method
        return getattr(getattr(client, resource), method)

    def _run(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **model_args: Any,
    ) -> Dict[str, Any]:
        return self._call(self._request({"query": query, **model_args}))

    async def _arun(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **model_args: Any,
    ) -> Dict[str, Any]:
        return await self._acall(self._request({"query": query, **model_args}))

    def _call(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        client: Any = sync_client(self._api_key())
        try:
            response = self._method(client)(**kwargs)
        except Exception as exc:
            message = self._model_error(exc)
            if message is None:
                raise
            raise ToolException(message) from exc
        finally:
            client.close()
        return self._result(response)

    async def _acall(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        client: Any = async_client(self._api_key())
        try:
            response = await self._method(client)(**kwargs)
        except Exception as exc:
            message = self._model_error(exc)
            if message is None:
                raise
            raise ToolException(message) from exc
        finally:
            await client.close()
        return self._result(response)


class OctenBroadSearch(_OctenVerticalTool):
    """Octen Broad Search: one question fanned out into several concurrent web searches.

    Returns ``{"data": ..., "request_id": ..., "warning"?: ...}`` where ``data`` holds
    ``queries`` (the generated sub-queries) and ``search_results`` (one group per
    sub-query).

    Instantiate:
        .. code-block:: python

            from langchain_octen import OctenBroadSearch

            tool = OctenBroadSearch(max_queries=5, count=3)
    """

    name: str = "octen_broad_search"
    description: str = (
        "Search the web across many angles in one call, for comparisons, research, "
        "surveys and questions with several distinct parts. Expands the question into "
        "multiple sub-queries and runs them concurrently.\n\n"
        "USE WHEN the question has multiple distinct parts or entities that one search "
        "cannot cover:\n"
        "  - comparing vendors / products / pricing across many sources\n"
        "  - literature reviews, market or landscape surveys\n"
        '  - open-ended "what are the options for X" / "how do people solve Y"\n'
        "  - a question that clearly decomposes into 3+ independent sub-questions\n"
        '  - multi-angle questions about recent events ("what shipped across the '
        'industry this month")\n\n'
        "DO NOT USE for a single fact, entity or document, news on a single topic, a "
        "company profile, a real-world place, or the address of a site you could almost "
        "name: a focused search answers those directly. Fanning out costs several times a "
        "single search and takes notably longer. Do not call it twice for the same "
        "question; follow up on the specific gaps with a targeted search.\n\n"
        "QUERY: pass one natural-language question (max 500 characters). Resolve pronouns "
        'and references from the conversation first: "how does it compare to the other '
        'one" is a useless query. Do NOT pre-split it into sub-queries; that is this '
        "tool's job. Domain, language and time_range filters apply to every sub-query.\n\n"
        "RESULTS are grouped per sub-query and NOT deduplicated: the same URL may appear "
        "under several sub-queries."
    )
    args_schema: Type[BaseModel] = OctenFilteredSearchInput

    count: Optional[int] = None
    """Results per sub-query (1-100). Unset uses the server default."""
    highlight: Optional[Dict[str, Any]] = None
    """Highlight options, e.g. ``{"enable": True, "max_tokens": 512}``."""
    full_content: Optional[Dict[str, Any]] = None
    """Full content options, e.g. ``{"enable": True, "max_tokens": 2048}``."""
    max_queries: Optional[int] = None
    """Upper bound on generated sub-queries (1-30). Unset uses the server default."""

    _sdk_method: ClassVar[Tuple[str, str]] = ("search", "broad_search")
    _developer_options: ClassVar[Tuple[str, ...]] = (
        "count",
        "highlight",
        "full_content",
        "max_queries",
    )
    _request_model: ClassVar[Any] = octen.BroadSearchRequest

    def _build_request_model(self, options: Dict[str, Any]) -> None:
        # Same two steps as octen's broad_search: per-sub-query options, then the request.
        search_options = {k: v for k, v in options.items() if k != "max_queries"}
        octen.SearchOptions(**search_options)
        self._request_model(query=_PLACEHOLDER_QUERY, max_queries=options["max_queries"])


class OctenNewsSearch(_OctenVerticalTool):
    """Octen News Search: recent articles plus storylines (``subjects``).

    Instantiate:
        .. code-block:: python

            from langchain_octen import OctenNewsSearch

            tool = OctenNewsSearch(count=5, subjects={"enable": True, "count": 2})
    """

    name: str = "octen_news_search"
    description: str = (
        "Search recent news and return ranked articles (title, url, snippet): current "
        "events, headlines, timely reporting. Alongside the articles it returns "
        "`subjects`: the coverage grouped into storylines, each with a summary and the "
        'span it covers, so a "what happened with X" question is answerable without '
        "reading every article. Narrow with domain filters, a language filter (ISO 639-1 "
        "codes) and a relative time_range. include_domains / exclude_domains narrow the "
        "ranked articles only; the storylines are built from the open web either way. "
        'When the question restricts its sources ("what is Reuters saying about X"), '
        "answer from the articles, not the storylines.\n\n"
        "USE FOR what is being reported right now: breaking events, announcements, "
        "coverage of a company or person over the last days or weeks. NOT FOR a company "
        "or person profile (financials, key people, career history) or non-news web "
        "pages. For a multi-angle question spanning several independent subtopics, one "
        "broad search covers it better than repeated calls to this tool."
    )
    args_schema: Type[BaseModel] = OctenFilteredSearchInput

    count: Optional[int] = None
    """Number of articles (1-100). Unset uses the server default."""
    highlight: Optional[Dict[str, Any]] = None
    """Highlight options, e.g. ``{"enable": True, "max_tokens": 512}``."""
    full_content: Optional[Dict[str, Any]] = None
    """Full content options, e.g. ``{"enable": True, "max_tokens": 2048}``."""
    subjects: Optional[Dict[str, Any]] = None
    """Storyline options, e.g. ``{"enable": True, "count": 2, "max_sub_news": 5}``."""

    _sdk_method: ClassVar[Tuple[str, str]] = ("search", "news_search")
    _request_model: ClassVar[Any] = octen.NewsSearchRequest
    _developer_options: ClassVar[Tuple[str, ...]] = (
        "count",
        "highlight",
        "full_content",
        "subjects",
    )


class OctenBusinessSearch(_OctenVerticalTool):
    """Octen Business Search: company and person cards plus ranked sources.

    Instantiate:
        .. code-block:: python

            from langchain_octen import OctenBusinessSearch

            tool = OctenBusinessSearch(entities={"count": 2})
    """

    name: str = "octen_business_search"
    description: str = (
        "Look up a company or a person and get a structured card back, plus the ranked "
        "sources behind it. A company card carries the profile, identifiers, financial "
        "and stock metrics, key people and recent announcements; a person card carries "
        "the current position, career history and recent activity. Name two companies in "
        "one query and you get a card for each, same fields, ready to compare. The "
        "domain, time and language filters narrow the ranked sources only, never the "
        "cards.\n\n"
        "USE FOR any question about a company or the people in one, including the "
        "numbers: revenue, net income, gross profit by quarter and year, share price and "
        "52-week range, funding rounds and who led them, headcount and web traffic, key "
        "people and their titles, career history, recent announcements. Also \"what does "
        'this company do", due diligence, and how two of them compare. The card already '
        'holds those figures, so a question like "what was their revenue last quarter" '
        "is one call here.\n\n"
        "NOT FOR breaking coverage of an event (a news search returns articles and "
        "storylines) or general web pages.\n\n"
        'Examples: "NVIDIA company overview", "Lisa Su AMD CEO", "compare Stripe and '
        'Adyen payment processing".'
    )
    args_schema: Type[BaseModel] = OctenFilteredSearchInput

    count: Optional[int] = None
    """Number of ranked sources (1-100). Unset uses the server default."""
    highlight: Optional[Dict[str, Any]] = None
    """Highlight options, e.g. ``{"enable": True, "max_tokens": 512}``."""
    full_content: Optional[Dict[str, Any]] = None
    """Full content options, e.g. ``{"enable": True, "max_tokens": 2048}``."""
    entities: Optional[Dict[str, Any]] = None
    """Card options, e.g. ``{"enable": True, "count": 2, "max_news": 5}``."""

    _sdk_method: ClassVar[Tuple[str, str]] = ("search", "business_search")
    _request_model: ClassVar[Any] = octen.BusinessSearchRequest
    _developer_options: ClassVar[Tuple[str, ...]] = (
        "count",
        "highlight",
        "full_content",
        "entities",
    )


class OctenPulseSearch(_OctenVerticalTool):
    """Octen Pulse Search: predicts the URL a query most likely means.

    Instantiate:
        .. code-block:: python

            from langchain_octen import OctenPulseSearch

            tool = OctenPulseSearch(count=3)
    """

    name: str = "octen_pulse_search"
    description: str = (
        "When what is wanted is a web address rather than what is written on the page, "
        "this is the tool. It returns ranked candidate URLs with titles: the single most "
        "likely one, the top three, or a shortlist to choose from, whichever the question "
        'calls for. Official docs and API references ("github docs", "uv documentation", '
        '"python list methods"), help centres ("how do I return an amazon order"), a '
        'site you know exists ("youtube").\n\n'
        "Prefer it over a web search for those: one inference predicts the URLs instead "
        "of searching and ranking them, so it answers in about 10ms for a fraction of "
        "the cost.\n\n"
        "NOT FOR news or anything needing freshness, or a question that needs real "
        "retrieval across the open web: it takes no filters, sees no fresh content, and "
        "is not a cheaper substitute for search.\n\n"
        "URLs may come back without a scheme (github.com/docs); prepend https:// before "
        "fetching one."
    )
    args_schema: Type[BaseModel] = OctenPulseSearchInput

    count: Optional[int] = None
    """Number of results (1-10). Unset uses the server default."""
    highlight: Optional[Dict[str, Any]] = None
    """Highlight options; no effect in ``lite`` mode."""
    full_content: Optional[Dict[str, Any]] = None
    """Full content options; no effect in ``lite`` mode."""
    mode: Optional[Literal["lite", "standard"]] = None
    """``"lite"`` or ``"standard"``. Unset uses the server default (``lite``)."""

    _sdk_method: ClassVar[Tuple[str, str]] = ("search", "pulse_search")
    _request_model: ClassVar[Any] = octen.PulseSearchRequest
    _developer_options: ClassVar[Tuple[str, ...]] = (
        "count",
        "highlight",
        "full_content",
        "mode",
    )


class OctenPlaceSearch(_OctenVerticalTool):
    """Octen Place Search: real-world places as structured cards.

    When the place engine does not answer in time the result carries
    ``"warning": "places_unavailable"``.

    Instantiate:
        .. code-block:: python

            from langchain_octen import OctenPlaceSearch

            tool = OctenPlaceSearch(places={"count": 5, "max_reviews": 0})
    """

    name: str = "octen_place_search"
    description: str = (
        "Find real-world places and get each one back as a card: name, category, "
        "address, coordinates, phone, website, opening hours, status (open / closed / "
        "permanently closed), rating and review count. Fields that could not be "
        "resolved are omitted. Pass latitude and longitude to search around a point "
        "(each card then carries its distance in metres) and radius to cap how far. "
        "When you only know the city, pass user_location instead. A query that names "
        'its own area ("ramen in Shinjuku") works without either.\n\n'
        'USE FOR anything with a location: "coffee shops near me", "family-friendly '
        'restaurants", "museums in Lisbon", "is this place open on Sunday", "what\'s '
        'their phone number". The cards answer these directly; a web search would '
        "return pages about places that then have to be read.\n\n"
        'Pass the caller\'s location when the question is relative to them ("near me", '
        '"nearby", "closest") or names no area at all: without it such a query is not '
        "limited to anywhere. A query that names its own area does not need it.\n\n"
        'If the result carries warning "places_unavailable", the place lookup did not '
        "complete; say so instead of claiming there are no such places."
    )
    args_schema: Type[BaseModel] = OctenPlaceSearchInput

    places: Optional[Dict[str, Any]] = None
    """Per-place options, e.g. ``{"count": 5, "max_reviews": 0}``."""
    web_search: Optional[Dict[str, Any]] = None
    """Web results alongside the places, e.g. ``{"enable": True, "count": 3}``. Off by
    default."""

    _sdk_method: ClassVar[Tuple[str, str]] = ("place_search", "search")
    _request_model: ClassVar[Any] = octen.PlaceSearchRequest
    _developer_options: ClassVar[Tuple[str, ...]] = ("places", "web_search")

    def _request(self, model_args: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = super()._request(model_args)
        location = kwargs.pop("user_location", None)
        if location:
            kwargs["user_location"] = {"type": "approximate", **location}
        return kwargs


class OctenSearchPremier(_OctenVerticalTool):
    """Octen Web Search Premier: web search with deeper retrieval and reranking.

    Same request options and response shape as Web Search, for questions whose answer
    sits on one specific, hard-to-find page. Returns
    ``{"data": ..., "request_id": ..., "warning"?: ...}`` where ``data`` holds
    ``query`` and ``results``.

    Instantiate:
        .. code-block:: python

            from langchain_octen import OctenSearchPremier

            tool = OctenSearchPremier(count=5, highlight={"enable": True, "max_tokens": 1024})
    """

    name: str = "octen_search_premier"
    description: str = (
        "Search the live web with deeper retrieval and reranking than a standard web "
        "search, and return ranked results (title, url, highlight). It searches more "
        "broadly and ranks more carefully, so the page that holds the answer ranks "
        "higher, and each highlight is the passage most relevant to the query from the "
        "whole page. Narrow with include_domains / exclude_domains.\n\n"
        "USE FOR questions that need the exact page holding the answer: a specific fact, "
        "a detail deep inside a page, a niche or long-tail document, or a named report, "
        "filing or document. Pass the question as is (max 500 characters).\n\n"
        "NOT FOR quick, low-latency lookups: use a standard web search for those. If the "
        "question spans several independent subtopics, one broad search covers it "
        "better than a sequence of calls to this tool."
    )
    args_schema: Type[BaseModel] = OctenSearchPremierInput

    count: Optional[int] = None
    """Number of results (1-100); the maximum when ``result_type`` is ``"dynamic"``.
    Unset uses the server default."""
    result_type: Optional[Literal["standard", "dynamic"]] = None
    """``"dynamic"`` returns only relevant results, up to ``count``, and adjusts the
    highlight length per result. Unset uses the server default (``standard``)."""
    highlight: Optional[Dict[str, Any]] = None
    """Highlight options, e.g. ``{"enable": True, "max_tokens": 1024}``."""
    full_content: Optional[Dict[str, Any]] = None
    """Full content options, e.g. ``{"enable": True, "max_tokens": 4096}``."""
    include_text: Optional[List[str]] = None
    """Strings that must appear in the page text (up to 5, 30 characters each)."""
    exclude_text: Optional[List[str]] = None
    """Strings that must not appear in the page text (up to 5, 30 characters each)."""
    language: Optional[List[Language]] = None
    """Only return results in these languages, as ISO 639-1 codes."""
    time_basis: Optional[Literal["auto", "published", "crawled"]] = None
    """Which time field the time filters use. Unset uses the server default (``auto``)."""
    time_range: Optional[TimeRange] = None
    """Relative time window; ``start_time`` / ``end_time`` take precedence over it."""
    start_time: Optional[str] = None
    """Only results after this RFC 3339 time, e.g. ``"2025-01-01T00:00:00Z"``."""
    end_time: Optional[str] = None
    """Only results before this RFC 3339 time."""
    format: Optional[Literal["text", "markdown"]] = None
    """Format of the highlights. Unset uses the server default (``text``)."""
    safesearch: Optional[Literal["off", "strict"]] = None
    """Adult content filter. Unset uses the server default (``strict``)."""
    include_images: Optional[bool] = None
    """Include each page's cover and in-body images. Unset uses the server default (off)."""

    _sdk_method: ClassVar[Tuple[str, str]] = ("search", "search_premier")
    _request_model: ClassVar[Any] = octen.PremierSearchRequest
    _developer_options: ClassVar[Tuple[str, ...]] = (
        "count",
        "result_type",
        "highlight",
        "full_content",
        "include_text",
        "exclude_text",
        "language",
        "time_basis",
        "time_range",
        "start_time",
        "end_time",
        "format",
        "safesearch",
        "include_images",
    )

    def _build_request_model(self, options: Dict[str, Any]) -> None:
        # octen keeps unknown keys in these two, so a typo like "enabled" would do nothing.
        for key in ("highlight", "full_content"):
            unknown = sorted(set(options[key] or ()) - {"enable", "max_tokens"})
            if unknown:
                raise ValueError(
                    f"invalid {self.name} option: {key}: unknown option {', '.join(unknown)}"
                )
        super()._build_request_model(options)

    @field_validator("start_time", "end_time", mode="after")
    @classmethod
    def _check_time(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not _is_rfc3339(value):
            raise ValueError("must be an RFC 3339 time, e.g. 2025-01-01T00:00:00Z")
        return value
