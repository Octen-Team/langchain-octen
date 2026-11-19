"""LangChain standard unit tests for every Octen tool."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Type

import pytest

pytest.importorskip("langchain_tests")

from langchain_core.tools import BaseTool  # noqa: E402
from langchain_tests.unit_tests import ToolsUnitTests  # noqa: E402

from langchain_octen import (  # noqa: E402
    OctenBroadSearch,
    OctenBusinessSearch,
    OctenNewsSearch,
    OctenPlaceSearch,
    OctenPulseSearch,
    OctenSearchResults,
)


class _OctenStandardTests(ToolsUnitTests):
    @property
    def tool_constructor_params(self) -> Dict[str, Any]:
        return {"octen_api_key": "fake"}

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"query": "LangChain"}

    @property
    def init_from_env_params(self) -> Tuple[Dict[str, str], Dict[str, Any], Dict[str, Any]]:
        return {"OCTEN_API_KEY": "from-env"}, {}, {}


class TestOctenSearchResultsStandard(_OctenStandardTests):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return OctenSearchResults

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"query": "LangChain", "include_domains": ["python.langchain.com"]}


class _OctenVerticalStandardTests(_OctenStandardTests):
    @property
    def init_from_env_params(self) -> Tuple[Dict[str, str], Dict[str, Any], Dict[str, Any]]:
        return {"OCTEN_API_KEY": "from-env"}, {}, {"octen_api_key": "from-env"}


class TestOctenBroadSearchStandard(_OctenVerticalStandardTests):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return OctenBroadSearch

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"query": "compare vector databases", "time_range": "month", "language": ["en"]}


class TestOctenNewsSearchStandard(_OctenVerticalStandardTests):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return OctenNewsSearch

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"query": "EU AI Act enforcement", "time_range": "week"}


class TestOctenBusinessSearchStandard(_OctenVerticalStandardTests):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return OctenBusinessSearch

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"query": "NVIDIA company overview", "exclude_domains": ["example.com"]}


class TestOctenPulseSearchStandard(_OctenVerticalStandardTests):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return OctenPulseSearch

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"query": "github docs"}


class TestOctenPlaceSearchStandard(_OctenVerticalStandardTests):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return OctenPlaceSearch

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"query": "coffee shops", "latitude": 35.6812, "longitude": 139.7671, "radius": 1000}
