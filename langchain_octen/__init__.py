"""LangChain integration for Octen Web Search."""

from langchain_octen.tools import OctenSearchResults
from langchain_octen.vertical_tools import (
    OctenBroadSearch,
    OctenBusinessSearch,
    OctenNewsSearch,
    OctenPlaceSearch,
    OctenPulseSearch,
)

__version__ = "0.2.0"
__all__ = [
    "OctenBroadSearch",
    "OctenBusinessSearch",
    "OctenNewsSearch",
    "OctenPlaceSearch",
    "OctenPulseSearch",
    "OctenSearchResults",
    "__version__",
]
