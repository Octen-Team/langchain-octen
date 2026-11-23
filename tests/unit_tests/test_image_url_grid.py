"""Image URLs checked against the server's own parser (Go net/url, see the fixture's source).

Every URL the server rejects must be rejected locally. Every URL it accepts must be
accepted locally, except the few listed in FALSE_REJECTS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from langchain_octen.image_tools import OctenImageSearchInput

FIXTURE = Path(__file__).parent / "data" / "image_url_cases.json"
CASES: List[Dict[str, Any]] = json.loads(FIXTURE.read_text(encoding="ascii"))["cases"]

# Accepted by the server's parser, refused here on purpose: none of them can name a host
# that serves an image.
FALSE_REJECTS = {
    # A double quote is not a valid host character in any DNS or IDN name.
    'https://ex"ample.com/a.png',
    # "<" is not a valid host character in any DNS or IDN name.
    "https://ex<ample.com/a.png",
    # ">" is not a valid host character in any DNS or IDN name.
    "https://ex>ample.com/a.png",
    # A bare "]" outside an IP literal is not a valid host character.
    "https://ex]ample.com/a.png",
    # A raw "@" inside userinfo is not allowed by RFC 3986; it must be %40.
    "https://u@@example.com/a.png",
}


def _accepted(url: str) -> bool:
    try:
        OctenImageSearchInput(image_url=url)
    except ValidationError:
        return False
    return True


def test_fixture_is_the_full_grid() -> None:
    assert len(CASES) == 505
    assert FALSE_REJECTS <= {c["url"] for c in CASES if c["server_accepts"]}


@pytest.mark.parametrize("case", CASES, ids=lambda c: ascii(c["url"]))
def test_local_check_matches_the_server(case: Dict[str, Any]) -> None:
    expected = case["server_accepts"] and case["url"] not in FALSE_REJECTS
    assert _accepted(case["url"]) == expected
