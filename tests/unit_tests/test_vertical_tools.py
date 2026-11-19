"""Unit tests for the vertical search tools (mocked SDK, no network)."""

from __future__ import annotations

import copy
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Literal, Optional, Tuple, Type
from unittest.mock import AsyncMock, MagicMock, patch

import octen
import pytest
from langchain_core.tools import BaseTool
from octen import (
    OctenAPIError,
    OctenAuthenticationError,
    OctenConnectionError,
    OctenRateLimitError,
    OctenTimeoutError,
    OctenValidationError,
)
from pydantic import SecretStr

from langchain_octen import (
    OctenBroadSearch,
    OctenBusinessSearch,
    OctenNewsSearch,
    OctenPlaceSearch,
    OctenPulseSearch,
)

FAKE_API_KEY = "test-api-key-12345"
FILTER_ARGS = frozenset({"query", "include_domains", "exclude_domains", "time_range", "language"})


@dataclass(frozen=True)
class Case:
    tool_cls: Type[BaseTool]
    name: str
    sdk_path: Tuple[str, str]
    response_cls: Type[Any]
    model_args: FrozenSet[str]
    options: Dict[str, Any]


CASES = [
    Case(
        OctenBroadSearch,
        "octen_broad_search",
        ("search", "broad_search"),
        octen.BroadSearchResponse,
        FILTER_ARGS,
        {"count": 3, "highlight": {"enable": True}, "full_content": {"enable": False},
         "max_queries": 4},
    ),
    Case(
        OctenNewsSearch,
        "octen_news_search",
        ("search", "news_search"),
        octen.NewsSearchResponse,
        FILTER_ARGS,
        {"count": 3, "highlight": {"enable": True}, "full_content": {"enable": False},
         "subjects": {"enable": True}},
    ),
    Case(
        OctenBusinessSearch,
        "octen_business_search",
        ("search", "business_search"),
        octen.BusinessSearchResponse,
        FILTER_ARGS,
        {"count": 3, "highlight": {"enable": True}, "full_content": {"enable": False},
         "entities": {"count": 2}},
    ),
    Case(
        OctenPulseSearch,
        "octen_pulse_search",
        ("search", "pulse_search"),
        octen.PulseSearchResponse,
        frozenset({"query"}),
        {"count": 3, "highlight": {"enable": True}, "full_content": {"enable": False},
         "mode": "standard"},
    ),
    Case(
        OctenPlaceSearch,
        "octen_place_search",
        ("place_search", "search"),
        octen.PlaceSearchResponse,
        frozenset({"query", "latitude", "longitude", "radius", "user_location"}),
        {"places": {"count": 1}, "web_search": {"enable": True}},
    ),
]
FILTERED_CASES = [c for c in CASES if c.model_args == FILTER_ARGS]
PLACE = CASES[-1]


def _ids(case: Case) -> str:
    return case.name


def _response(case: Case, meta: Any = None, request_id: Any = "rid") -> Any:
    data = {"results": [{"title": "T", "url": "https://example.com"}], "subjects": [{"name": "S"}]}
    extra = {} if request_id is None else {"request_id": request_id}
    return case.response_cls(code=0, msg="success", data=data, meta=meta or {}, **extra)


def _sdk_method(client: MagicMock, case: Case) -> MagicMock:
    resource, method = case.sdk_path
    return getattr(getattr(client, resource), method)  # type: ignore[no-any-return]


def _tool(case: Case, **kwargs: Any) -> BaseTool:
    return case.tool_cls(octen_api_key=SecretStr(FAKE_API_KEY), **kwargs)


@pytest.fixture
def sync_client() -> Any:
    with patch("octen.Octen") as mock_cls:
        yield mock_cls


@pytest.fixture
def async_client() -> Any:
    with patch("octen.AsyncOcten") as mock_cls:
        mock_cls.return_value = AsyncMock()
        yield mock_cls


def test_tool_names() -> None:
    names = [_tool(c).name for c in CASES]
    assert names == [
        "octen_broad_search",
        "octen_news_search",
        "octen_business_search",
        "octen_pulse_search",
        "octen_place_search",
    ]


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_model_schema_exposes_only_allowed_args(case: Case) -> None:
    assert set(_tool(case).args) == case.model_args


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_developer_options_are_not_model_args(case: Case) -> None:
    tool = _tool(case)
    assert not set(tool.args) & (set(case.options) | {"timeout", "octen_api_key"})


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_missing_key_raises_at_construction(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OCTEN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OCTEN_API_KEY"):
        case.tool_cls()


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_key_read_from_env(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCTEN_API_KEY", "from-env")
    tool = case.tool_cls()
    assert tool.octen_api_key.get_secret_value() == "from-env"  # type: ignore[attr-defined]


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_developer_options_forwarded(case: Case, sync_client: MagicMock) -> None:
    client = sync_client.return_value
    _sdk_method(client, case).return_value = _response(case)

    _tool(case, timeout=12.5, **case.options).invoke({"query": "q"})

    sync_client.assert_called_once_with(api_key=FAKE_API_KEY, retry_after_send=False)
    _sdk_method(client, case).assert_called_once_with(query="q", timeout=12.5, **case.options)
    client.close.assert_called_once()


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_unset_options_not_sent(case: Case, sync_client: MagicMock) -> None:
    client = sync_client.return_value
    _sdk_method(client, case).return_value = _response(case)

    _tool(case).invoke({"query": "q"})

    _sdk_method(client, case).assert_called_once_with(query="q")


@pytest.mark.parametrize("case", FILTERED_CASES, ids=_ids)
def test_model_filters_forwarded(case: Case, sync_client: MagicMock) -> None:
    client = sync_client.return_value
    _sdk_method(client, case).return_value = _response(case)

    _tool(case, count=2).invoke(
        {
            "query": "q",
            "include_domains": ["a.com"],
            "exclude_domains": ["b.com"],
            "time_range": "week",
            "language": ["en"],
        }
    )

    _sdk_method(client, case).assert_called_once_with(
        query="q",
        count=2,
        include_domains=["a.com"],
        exclude_domains=["b.com"],
        time_range="week",
        language=["en"],
    )


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_returns_data_request_id_warning(case: Case, sync_client: MagicMock) -> None:
    response = _response(case, meta={"warning": "places_unavailable", "latency": 5})
    _sdk_method(sync_client.return_value, case).return_value = response

    result = _tool(case).invoke({"query": "q"})

    assert result == {
        "data": response.data,
        "request_id": "rid",
        "warning": "places_unavailable",
    }


@pytest.mark.parametrize("meta", [{"warning": ""}, {"warning": None}, {}], ids=str)
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_empty_warning_omitted(case: Case, meta: Dict[str, Any], sync_client: MagicMock) -> None:
    response = _response(case, meta=meta)
    _sdk_method(sync_client.return_value, case).return_value = response

    result = _tool(case).invoke({"query": "q"})

    assert result == {"data": response.data, "request_id": "rid"}


def test_missing_request_id_is_none(sync_client: MagicMock) -> None:
    case = CASES[1]
    _sdk_method(sync_client.return_value, case).return_value = _response(case, request_id=None)

    result = _tool(case).invoke({"query": "q"})

    assert result["request_id"] is None


TRANSIENT_ERRORS = [
    OctenRateLimitError(request_id="rid-429"),
    OctenAPIError("upstream unavailable", status_code=503),
    OctenAPIError("invalid language", status_code=400),
    OctenAPIError("unprocessable", status_code=422),
    OctenAPIError("request timeout", status_code=408),
    OctenAPIError("API returned code 500", status_code=200),
    OctenTimeoutError(timeout=30.0),
    OctenConnectionError(),
]
CONFIG_ERRORS = [
    OctenAuthenticationError(),
    OctenAPIError("insufficient balance", status_code=403),
    OctenValidationError("count must be <= 100"),
    OctenAPIError("not found", status_code=404),
    RuntimeError("unexpected"),
]


@pytest.mark.parametrize("exc", TRANSIENT_ERRORS, ids=repr)
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_transient_errors_go_to_model(
    case: Case, exc: Exception, sync_client: MagicMock
) -> None:
    client = sync_client.return_value
    _sdk_method(client, case).side_effect = exc

    result = _tool(case).invoke({"query": "q"})

    assert isinstance(result, str)
    assert str(exc) in result
    client.close.assert_called_once()


@pytest.mark.parametrize("exc", CONFIG_ERRORS, ids=repr)
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_config_errors_raise(case: Case, exc: Exception, sync_client: MagicMock) -> None:
    client = sync_client.return_value
    _sdk_method(client, case).side_effect = exc

    with pytest.raises(type(exc)):
        _tool(case).invoke({"query": "q"})
    client.close.assert_called_once()


@pytest.mark.parametrize("case", FILTERED_CASES, ids=_ids)
def test_bad_time_range_goes_to_model(case: Case, sync_client: MagicMock) -> None:
    result = _tool(case).invoke({"query": "x", "time_range": "last week"})

    assert isinstance(result, str)
    assert "time_range" in result
    sync_client.assert_not_called()


@pytest.mark.parametrize("case", FILTERED_CASES, ids=_ids)
def test_oversized_domain_goes_to_model(case: Case, sync_client: MagicMock) -> None:
    result = _tool(case).invoke({"query": "x", "include_domains": ["a" * 61 + ".com"]})

    assert isinstance(result, str)
    assert "include_domains" in result
    sync_client.assert_not_called()


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_oversized_query_goes_to_model(case: Case, sync_client: MagicMock) -> None:
    result = _tool(case).invoke({"query": "x" * 501})

    assert isinstance(result, str)
    assert "query" in result
    sync_client.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [{"latitude": 35.68}, {"longitude": 139.76}],
    ids=["latitude-only", "longitude-only"],
)
def test_place_half_coordinate_goes_to_model(
    args: Dict[str, Any], sync_client: MagicMock
) -> None:
    result = _tool(PLACE).invoke({"query": "coffee", **args})

    assert isinstance(result, str)
    assert "latitude" in result and "longitude" in result
    sync_client.assert_not_called()


def test_place_radius_without_point_goes_to_model(sync_client: MagicMock) -> None:
    result = _tool(PLACE).invoke({"query": "coffee", "radius": 500})

    assert isinstance(result, str)
    assert "radius" in result
    sync_client.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [
        {"latitude": 91, "longitude": 0},
        {"latitude": 0, "longitude": -181},
        {"latitude": 0, "longitude": 0, "radius": 0},
        {"latitude": 0, "longitude": 0, "radius": 50001},
    ],
    ids=["lat-91", "lon-181", "radius-0", "radius-50001"],
)
def test_place_out_of_range_goes_to_model(args: Dict[str, Any], sync_client: MagicMock) -> None:
    result = _tool(PLACE).invoke({"query": "coffee", **args})

    assert isinstance(result, str)
    sync_client.assert_not_called()


def test_place_point_and_radius_forwarded(sync_client: MagicMock) -> None:
    client = sync_client.return_value
    client.place_search.search.return_value = _response(PLACE)

    _tool(PLACE).invoke(
        {"query": "coffee", "latitude": 35.68, "longitude": 139.76, "radius": 800}
    )

    client.place_search.search.assert_called_once_with(
        query="coffee", latitude=35.68, longitude=139.76, radius=800
    )


def test_place_user_location_sets_approximate(sync_client: MagicMock) -> None:
    client = sync_client.return_value
    client.place_search.search.return_value = _response(PLACE)

    _tool(PLACE).invoke(
        {"query": "ramen", "user_location": {"city": "Tokyo", "country": "JP"}}
    )

    client.place_search.search.assert_called_once_with(
        query="ramen",
        user_location={"type": "approximate", "city": "Tokyo", "country": "JP"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=_ids)
async def test_async_uses_async_client_and_closes(case: Case, async_client: MagicMock) -> None:
    client = async_client.return_value
    response = _response(case)
    _sdk_method(client, case).return_value = response

    result = await _tool(case, timeout=5.0).ainvoke({"query": "q"})

    async_client.assert_called_once_with(api_key=FAKE_API_KEY, retry_after_send=False)
    _sdk_method(client, case).assert_awaited_once_with(query="q", timeout=5.0)
    client.close.assert_awaited_once()
    assert result == {"data": response.data, "request_id": "rid"}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=_ids)
async def test_async_closes_on_config_error(case: Case, async_client: MagicMock) -> None:
    client = async_client.return_value
    _sdk_method(client, case).side_effect = OctenAuthenticationError()

    with pytest.raises(OctenAuthenticationError):
        await _tool(case).ainvoke({"query": "q"})
    client.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=_ids)
async def test_async_transient_error_goes_to_model(case: Case, async_client: MagicMock) -> None:
    client = async_client.return_value
    exc = OctenRateLimitError()
    _sdk_method(client, case).side_effect = exc

    result = await _tool(case).ainvoke({"query": "q"})

    assert isinstance(result, str)
    assert str(exc) in result
    client.close.assert_awaited_once()


@pytest.mark.parametrize(
    ("case", "options", "bad_name"),
    [
        (CASES[0], {"max_queries": 1000}, "max_queries"),
        (CASES[0], {"count": 1000}, "count"),
        (CASES[0], {"highlight": {"max_tokens": 5}}, "highlight"),
        (CASES[1], {"count": 1000}, "count"),
        (CASES[1], {"subjects": {"count": 9}}, "subjects"),
        (CASES[2], {"entities": {"count": 1000}}, "entities"),
        (CASES[2], {"entities": {"bogus": 1}}, "entities"),
        (CASES[3], {"count": 11}, "count"),
        (CASES[3], {"mode": "turbo"}, "mode"),
        (CASES[3], {"full_content": {"max_tokens": 5}}, "full_content"),
        (PLACE, {"places": {"count": 1000}}, "places"),
        (PLACE, {"web_search": {"time_range": "last week"}}, "web_search"),
    ],
    ids=[
        "broad-max_queries",
        "broad-count",
        "broad-highlight",
        "news-count",
        "news-subjects",
        "business-entities-count",
        "business-entities-unknown-key",
        "pulse-count",
        "pulse-mode",
        "pulse-full_content",
        "place-places",
        "place-web_search",
    ],
)
def test_invalid_developer_option_raises_at_construction(
    case: Case, options: Dict[str, Any], bad_name: str, sync_client: MagicMock
) -> None:
    with pytest.raises(ValueError, match=bad_name):
        _tool(case, **options)
    sync_client.assert_not_called()


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_valid_developer_options_accepted(case: Case) -> None:
    _tool(case, timeout=10.0, **case.options)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_unknown_constructor_kwarg_rejected(case: Case) -> None:
    with pytest.raises(ValueError, match="max_querys"):
        _tool(case, max_querys=3)


def test_pulse_mode_is_enumerated() -> None:
    annotation = OctenPulseSearch.model_fields["mode"].annotation
    assert annotation == Optional[Literal["lite", "standard"]]


@pytest.mark.parametrize("exc", CONFIG_ERRORS, ids=repr)
def test_raised_errors_are_logged(
    exc: Exception, sync_client: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    case = CASES[1]
    _sdk_method(sync_client.return_value, case).side_effect = exc

    with caplog.at_level(logging.ERROR, logger="langchain_octen"):
        with pytest.raises(type(exc)):
            _tool(case).invoke({"query": "q"})

    records = [r for r in caplog.records if r.name == "langchain_octen"]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert type(exc).__name__ in records[0].getMessage()
    assert str(exc) in records[0].getMessage()
    assert FAKE_API_KEY not in caplog.text


@pytest.mark.asyncio
async def test_async_raised_error_is_logged(
    async_client: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    case = PLACE
    _sdk_method(async_client.return_value, case).side_effect = OctenAuthenticationError()

    with caplog.at_level(logging.ERROR, logger="langchain_octen"):
        with pytest.raises(OctenAuthenticationError):
            await _tool(case).ainvoke({"query": "q"})

    assert [r.levelno for r in caplog.records if r.name == "langchain_octen"] == [logging.ERROR]


def test_transient_error_is_not_logged(
    sync_client: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    case = CASES[1]
    _sdk_method(sync_client.return_value, case).side_effect = OctenRateLimitError()

    with caplog.at_level(logging.DEBUG, logger="langchain_octen"):
        _tool(case).invoke({"query": "q"})

    assert not [r for r in caplog.records if r.name == "langchain_octen"]


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_forwarded_names_are_sdk_parameters(case: Case) -> None:
    # No mocks: the names the tool forwards must exist on the real octen methods.
    from octen.async_client import AsyncPlaceSearchResource, AsyncSearchResource
    from octen.resources.place_search import PlaceSearchResource
    from octen.resources.search import SearchResource

    resource, method = case.sdk_path
    sync_cls, async_cls = (
        (PlaceSearchResource, AsyncPlaceSearchResource)
        if resource == "place_search"
        else (SearchResource, AsyncSearchResource)
    )
    tool = _tool(case)
    forwarded = (
        set(tool._developer_options)  # type: ignore[attr-defined]
        | set(tool.args)
        | {"timeout"}
    )
    for cls in (sync_cls, async_cls):
        params = set(inspect.signature(getattr(cls, method)).parameters)
        assert forwarded <= params, (cls.__name__, method, forwarded - params)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_plain_string_input(case: Case, sync_client: MagicMock) -> None:
    client = sync_client.return_value
    response = _response(case)
    _sdk_method(client, case).return_value = response

    result = _tool(case).invoke("coffee")

    _sdk_method(client, case).assert_called_once_with(query="coffee")
    assert result == {"data": response.data, "request_id": "rid"}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=_ids)
async def test_plain_string_input_async(case: Case, async_client: MagicMock) -> None:
    client = async_client.return_value
    response = _response(case)
    _sdk_method(client, case).return_value = response

    result = await _tool(case).ainvoke("coffee")

    _sdk_method(client, case).assert_awaited_once_with(query="coffee")
    assert result == {"data": response.data, "request_id": "rid"}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=_ids)
async def test_dict_input_async(case: Case, async_client: MagicMock) -> None:
    client = async_client.return_value
    _sdk_method(client, case).return_value = _response(case)

    await _tool(case).ainvoke({"query": "coffee"})

    _sdk_method(client, case).assert_awaited_once_with(query="coffee")


@pytest.mark.parametrize(
    "timeout",
    [0, -1, 0.0, float("nan"), float("inf"), float("-inf"), True, False],
    ids=repr,
)
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_invalid_timeout_rejected(case: Case, timeout: Any) -> None:
    with pytest.raises(ValueError, match="timeout"):
        _tool(case, timeout=timeout)


@pytest.mark.parametrize("timeout", [None, 0.5, 30, 120.0], ids=repr)
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_valid_timeout_accepted(case: Case, timeout: Any) -> None:
    assert _tool(case, timeout=timeout).timeout == timeout  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("case", "args", "bad_name"),
    [
        (CASES[0], {"query": "x", "include_domain": ["a.com"]}, "include_domain"),
        (CASES[1], {"query": "x", "timerange": "week"}, "timerange"),
        (CASES[2], {"query": "x", "count": 50}, "count"),
        (CASES[3], {"query": "x", "count": 10}, "count"),
        (CASES[3], {"query": "x", "language": ["en"]}, "language"),
        (PLACE, {"query": "x", "lat": 35.0, "lng": 139.0}, "lat"),
        (PLACE, {"query": "x", "user_location": {"cty": "Tokyo"}}, "cty"),
    ],
    ids=[
        "broad-include_domain",
        "news-timerange",
        "business-count",
        "pulse-count",
        "pulse-language",
        "place-lat-lng",
        "place-user_location-cty",
    ],
)
def test_unknown_model_arg_goes_to_model(
    case: Case, args: Dict[str, Any], bad_name: str, sync_client: MagicMock
) -> None:
    result = _tool(case).invoke(args)

    assert isinstance(result, str)
    assert bad_name in result
    sync_client.assert_not_called()


MAY_HAVE_BEEN_SENT = "may already have reached the server"
SENT_OR_UNKNOWN = [
    OctenTimeoutError(timeout=30.0, request_sent=True),
    OctenTimeoutError(timeout=30.0, request_sent=None),
    OctenConnectionError("connection reset", request_sent=True),
    OctenConnectionError("connection reset", request_sent=None),
]
NOT_SENT = [
    OctenTimeoutError("connect timeout", timeout=30.0, request_sent=False),
    OctenConnectionError("connection refused", request_sent=False),
]


@pytest.mark.parametrize("exc", SENT_OR_UNKNOWN, ids=repr)
def test_possibly_sent_failure_warns_against_retry(exc: Exception, sync_client: MagicMock) -> None:
    case = CASES[1]
    _sdk_method(sync_client.return_value, case).side_effect = exc

    result = _tool(case).invoke({"query": "q"})

    assert isinstance(result, str)
    assert str(exc) in result
    assert MAY_HAVE_BEEN_SENT in result
    assert "billed" in result


@pytest.mark.parametrize("exc", NOT_SENT, ids=repr)
def test_unsent_failure_keeps_plain_message(exc: Exception, sync_client: MagicMock) -> None:
    case = CASES[1]
    _sdk_method(sync_client.return_value, case).side_effect = exc

    assert _tool(case).invoke({"query": "q"}) == str(exc)


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", SENT_OR_UNKNOWN, ids=repr)
async def test_possibly_sent_failure_warns_against_retry_async(
    exc: Exception, async_client: MagicMock
) -> None:
    _sdk_method(async_client.return_value, PLACE).side_effect = exc

    result = await _tool(PLACE).ainvoke({"query": "q"})

    assert isinstance(result, str)
    assert MAY_HAVE_BEEN_SENT in result


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", NOT_SENT, ids=repr)
async def test_unsent_failure_keeps_plain_message_async(
    exc: Exception, async_client: MagicMock
) -> None:
    _sdk_method(async_client.return_value, PLACE).side_effect = exc

    assert await _tool(PLACE).ainvoke({"query": "q"}) == str(exc)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_developer_option_dicts_are_copied(case: Case, sync_client: MagicMock) -> None:
    client = sync_client.return_value
    _sdk_method(client, case).return_value = _response(case)
    options = {
        key: {"enable": True, "nested": {"max_tokens": 512}}
        for key, value in case.options.items()
        if isinstance(value, dict)
    }
    expected = copy.deepcopy(options)
    with patch.object(case.tool_cls, "_build_request_model"):
        tool = _tool(case, **options)

    for value in options.values():
        value["enable"] = False
        value["nested"]["max_tokens"] = 1
        value["added"] = "later"

    for key, value in expected.items():
        assert getattr(tool, key) == value
    tool.invoke({"query": "q"})
    _, kwargs = _sdk_method(client, case).call_args
    for key, value in expected.items():
        assert kwargs[key] == value


@pytest.mark.parametrize("bad_key", [12345, 1.5, ["k"], {"key": "k"}, b"k"], ids=repr)
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_non_string_key_rejected(case: Case, bad_key: Any) -> None:
    with pytest.raises(ValueError, match="octen_api_key") as exc_info:
        case.tool_cls(octen_api_key=bad_key)
    assert repr(bad_key) not in str(exc_info.value)


def test_failed_2xx_hint_reaches_model(sync_client: MagicMock) -> None:
    case = CASES[1]
    exc = OctenAPIError("malformed successful response", status_code=200)
    _sdk_method(sync_client.return_value, case).side_effect = exc

    result = _tool(case).invoke({"query": "q"})

    assert isinstance(result, str)
    assert MAY_HAVE_BEEN_SENT in result


SECRET = "sk-live-SECRET-0123456789"


def _assert_no_secret(exc: BaseException) -> None:
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        assert SECRET not in str(current)
        assert SECRET not in repr(current)
        errors = getattr(current, "errors", None)
        if callable(errors):
            assert SECRET not in repr(errors())
        to_json = getattr(current, "json", None)
        if callable(to_json):
            assert SECRET not in to_json()
        assert SECRET not in repr(getattr(current, "args", ()))
        current = current.__cause__ or current.__context__
    assert exc.__cause__ is None and exc.__context__ is None


@pytest.mark.parametrize("env_key", [None, "from-env"], ids=["no-env-key", "env-key"])
@pytest.mark.parametrize("misspelt", ["octen_apikey", "octen_api_ky"])
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_misspelt_key_kwarg_does_not_leak_key(
    case: Case, misspelt: str, env_key: Optional[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    if env_key is None:
        monkeypatch.delenv("OCTEN_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OCTEN_API_KEY", env_key)

    with pytest.raises(ValueError) as exc_info:
        case.tool_cls(**{misspelt: SECRET})

    _assert_no_secret(exc_info.value)
    if env_key is not None:
        assert misspelt in str(exc_info.value)
    else:
        assert "OCTEN_API_KEY" in str(exc_info.value)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_option_error_does_not_leak_key(case: Case) -> None:
    with pytest.raises(ValueError) as exc_info:
        case.tool_cls(octen_api_key=SECRET, timeout=-1)
    _assert_no_secret(exc_info.value)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_non_string_key_error_has_no_input(case: Case) -> None:
    with pytest.raises(ValueError, match="octen_api_key") as exc_info:
        case.tool_cls(octen_api_key=[SECRET])
    _assert_no_secret(exc_info.value)


@pytest.mark.parametrize("timeout", [10**400, -(10**400)], ids=["huge", "huge-negative"])
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_huge_int_timeout_rejected(case: Case, timeout: int) -> None:
    with pytest.raises(ValueError, match="timeout"):
        _tool(case, timeout=timeout)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_response_format_other_than_content_rejected(case: Case, sync_client: MagicMock) -> None:
    with pytest.raises(ValueError, match="response_format"):
        _tool(case, response_format="content_and_artifact")
    sync_client.assert_not_called()


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_response_format_content_accepted(case: Case) -> None:
    assert _tool(case, response_format="content").response_format == "content"
