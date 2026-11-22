# langchain-octen

[![PyPI version](https://badge.fury.io/py/langchain-octen.svg)](https://pypi.org/project/langchain-octen/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LangChain integration for [Octen Web Search](https://octen.ai). Use Octen's fast, real-time web search inside LangChain agents and chains.

| Tool class | Tool name | Use it for |
|---|---|---|
| `OctenSearchResults` | `octen_search` | One focused web lookup |
| `OctenSearchPremier` | `octen_search_premier` | A hard-to-find specific fact or page (deeper retrieval and reranking) |
| `OctenBroadSearch` | `octen_broad_search` | Multi-part questions, comparisons, surveys (fans out into sub-queries) |
| `OctenNewsSearch` | `octen_news_search` | Recent news, with articles grouped into storylines |
| `OctenBusinessSearch` | `octen_business_search` | Company and person cards (financials, key people, career history) |
| `OctenPulseSearch` | `octen_pulse_search` | The URL of a docs page or site you could almost name |
| `OctenPlaceSearch` | `octen_place_search` | Real-world places, near a point or in a city |
| `OctenExtract` | `octen_extract` | The clean content of pages you already have the URLs of |

## Installation

```bash
pip install langchain-octen
```

## Setup

Get an API key from [octen.ai](https://octen.ai) and set it as an environment variable:

```bash
export OCTEN_API_KEY="your-api-key"
```

Or pass it directly when creating the tool:

```python
from langchain_octen import OctenSearchResults

tool = OctenSearchResults(octen_api_key="your-api-key")
```

## Web search (`OctenSearchResults`)

### Basic Search

```python
from langchain_octen import OctenSearchResults

tool = OctenSearchResults(max_results=5)
results = tool.invoke({"query": "latest AI research papers"})
for r in results:
    print(r["title"], r["url"])
```

### With Domain Filtering

```python
# Only include results from specific domains
results = tool.invoke({
    "query": "machine learning",
    "include_domains": ["arxiv.org", "papers.nips.cc"],
})

# Exclude results from specific domains
results = tool.invoke({
    "query": "Python tutorial",
    "exclude_domains": ["youtube.com", "reddit.com"],
})
```

### In a LangChain Agent

`create_agent` comes from the `langchain` package (1.0 or newer): `pip install langchain`.
`llm` is any LangChain chat model, for example `ChatOpenAI(...)` from `langchain-openai`.

```python
from langchain.agents import create_agent
from langchain_octen import OctenSearchResults

tool = OctenSearchResults(max_results=5)
agent = create_agent(model=llm, tools=[tool])
response = agent.invoke({"messages": [{"role": "user", "content": "What happened in AI today?"}]})
```

### Async Support

```python
import asyncio
from langchain_octen import OctenSearchResults

tool = OctenSearchResults(max_results=3)

async def main():
    results = await tool.ainvoke({"query": "Python 3.13 new features"})
    print(results)

asyncio.run(main())
```

### Advanced Options

```python
from langchain_octen import OctenSearchResults

# Time-filtered search
tool = OctenSearchResults(
    max_results=5,
    time_basis="published",
    start_time="2025-01-01",
    end_time="2026-03-30",
)
results = tool.invoke({"query": "Python release notes"})

# Search with highlighted snippets
tool = OctenSearchResults(
    max_results=5,
    highlight_enabled=True,
    highlight_max_tokens=200,
)
results = tool.invoke({"query": "LangChain tutorial"})

# Search with full page content
tool = OctenSearchResults(
    max_results=3,
    full_content_enabled=True,
    full_content_max_tokens=1000,
)
results = tool.invoke({"query": "Python documentation"})
```

## Vertical search tools

`OctenSearchPremier`, `OctenBroadSearch`, `OctenNewsSearch`, `OctenBusinessSearch`,
`OctenPulseSearch` and `OctenPlaceSearch` share one design:

- **The model only controls the query and search filters; you control the billing-related
  options and limits** (result counts, highlight / full content, Broad `max_queries`, ...)
  when you construct the tool. Options you leave unset are not sent, so the server defaults
  apply. Broad Search is billed per sub-query: `max_queries` is an upper bound, and how many
  sub-queries a question actually produces depends on the query.
- **Every call returns a dict:**

  ```python
  {
      "data": {...},          # the response's data, unchanged (results, places, entities, ...)
      "request_id": "...",    # quote it when contacting support
      "warning": "...",       # only present when the API attached a non-empty warning
  }
  ```

- **Errors are split by who can fix them.**
  - Raised to your code: authentication failure (`octen.OctenAuthenticationError`, 401),
    no access or insufficient balance (`octen.OctenAPIError` with status 403) and other 4xx
    responses.
    Some agent runtimes catch exceptions raised by tools, so each of these is also logged at
    ERROR level to the `langchain_octen` logger.
  - Constructor options are checked when the tool is built: an out-of-range value
    (`count=1000`), a misspelt option name (`max_querys=3`) or a `timeout` that is not a
    positive finite number raises `ValueError` naming it.
  - Handed to the model as the tool result (a `ToolException` with `handle_tool_error=True`):
    rate limits (429), server errors (5xx), 400/408/422 responses, 2xx responses carrying an
    error envelope or a malformed body, timeouts and connection errors. For a timeout or
    connection error that may have happened after the request was sent, a 502 or 504 (a
    gateway can return these after the backend served the call), or a failed 2xx response,
    the message also tells the model the call may already have been billed and should not
    be retried automatically.
  - Invalid model arguments (an unknown `time_range`, a misspelt argument name such as
    `include_domain`, half a coordinate pair, a `radius` without a point, values out of
    range) are rejected before any request is made, and the
    model gets a message naming the argument.
- Every tool takes `octen_api_key` (falls back to `OCTEN_API_KEY`; missing key raises
  `ValueError` at construction) and `timeout` (seconds).

```python
from langchain_octen import (
    OctenBroadSearch,
    OctenBusinessSearch,
    OctenNewsSearch,
    OctenPlaceSearch,
    OctenPulseSearch,
)
from langchain.agents import create_agent

tools = [
    OctenNewsSearch(count=5),
    OctenBusinessSearch(entities={"count": 2}),
    OctenPlaceSearch(places={"count": 5}),
]
agent = create_agent(model=llm, tools=tools)
```

### Web Search Premier (`OctenSearchPremier`)

Web search with deeper retrieval and reranking, for questions that need the exact page
holding the answer: a specific fact, a detail deep inside a page, a niche document or a
named report. Same request options and response shape as Web Search (`data.query`,
`data.results`). For low-latency lookups use `OctenSearchResults` instead.

The model gets the same arguments as `octen_search`; everything else is yours.

| Model arguments | Constructor options |
|---|---|
| `query`, `include_domains`, `exclude_domains` | `count` (1-100), `result_type` (`"standard"` or `"dynamic"`), `highlight`, `full_content` (only the `enable` and `max_tokens` keys), `include_text`, `exclude_text` (up to 5, 30 characters each), `language` (ISO 639-1 codes), `time_basis`, `time_range`, `start_time`, `end_time` (RFC 3339, e.g. `2025-01-01T00:00:00Z`), `format`, `safesearch`, `include_images`, `timeout` |

`result_type="dynamic"` returns only relevant results, up to `count`, and adjusts the
highlight length per result. Unlike `OctenSearchResults`, the constructor takes no
`include_domains` / `exclude_domains`: those are the model's.

```python
tool = OctenSearchPremier(count=5, highlight={"enable": True, "max_tokens": 1024})
result = tool.invoke({"query": "Kafka default retention period for topics"})
for r in result["data"]["results"]:
    print(r["title"], r["url"])
```

### Broad Search (`OctenBroadSearch`)

Expands one question into several sub-queries, searches them concurrently, and returns the
results grouped per sub-query (`data.queries`, `data.search_results`; not deduplicated).

| Model arguments | Constructor options |
|---|---|
| `query`, `include_domains`, `exclude_domains`, `time_range`, `language` | `count` (per sub-query, 1-100), `max_queries` (1-30), `highlight`, `full_content`, `timeout` |

```python
tool = OctenBroadSearch(max_queries=5, count=3)
result = tool.invoke({"query": "compare managed Postgres providers on price and limits"})
for group in result["data"]["search_results"]:
    print(group["query"], len(group["results"]))
```

### News Search (`OctenNewsSearch`)

Recent articles plus `data.subjects`: storylines with a summary and the time span they cover.
Domain filters narrow the ranked articles only, not the storylines.

| Model arguments | Constructor options |
|---|---|
| `query`, `include_domains`, `exclude_domains`, `time_range`, `language` | `count` (1-100), `subjects` (e.g. `{"enable": True, "count": 2, "max_sub_news": 5}`), `highlight`, `full_content`, `timeout` |

```python
tool = OctenNewsSearch(count=5)
result = tool.invoke({"query": "EU AI Act enforcement", "time_range": "week"})
for storyline in result["data"].get("subjects", []):
    print(storyline["name"], storyline.get("summary"))
```

### Business Search (`OctenBusinessSearch`)

Company and person cards in `data.entities` (profile, financial and stock metrics, key
people, career history, recent announcements) plus the ranked sources in `data.results`.
The domain, time and language filters narrow the ranked sources only, never the cards.

| Model arguments | Constructor options |
|---|---|
| `query`, `include_domains`, `exclude_domains`, `time_range`, `language` | `count` (1-100), `entities` (e.g. `{"count": 2, "max_news": 5}`), `highlight`, `full_content`, `timeout` |

```python
tool = OctenBusinessSearch(entities={"count": 2})
result = tool.invoke({"query": "compare Stripe and Adyen"})
for entity in result["data"].get("entities", []):
    print(entity["name"])
```

### Pulse Search (`OctenPulseSearch`)

Predicts the URL a query most likely means, in about 10ms: official docs, API references,
help centres, sites you could almost name. It takes no filters and sees no fresh content.
In `lite` mode (the server default) URLs come back without a scheme, e.g. `github.com/docs`.

| Model arguments | Constructor options |
|---|---|
| `query` | `count` (1-10), `mode` (`"lite"` or `"standard"`), `highlight`, `full_content` (both only in `standard`), `timeout` |

```python
tool = OctenPulseSearch(count=3)
result = tool.invoke({"query": "uv documentation"})
print([r["url"] for r in result["data"]["results"]])
```

### Place Search (`OctenPlaceSearch`)

Real-world places as cards in `data.places`: name, category, address, coordinates, phone,
website, opening hours, status, rating and review count. With `latitude` and `longitude` each
card carries its `distance` in metres. When the place engine does not answer in time the
result carries `"warning": "places_unavailable"`.

| Model arguments | Constructor options |
|---|---|
| `query`, `latitude` + `longitude` (both or neither), `radius` (1-50000 m, needs a point), `user_location` (`city`, `region`, `country`, `timezone`) | `places` (e.g. `{"count": 5, "max_reviews": 0}`), `web_search` (e.g. `{"enable": True, "count": 3}`; off by default), `timeout` |

```python
tool = OctenPlaceSearch(places={"count": 5})
result = tool.invoke(
    {"query": "coffee shops", "latitude": 35.6812, "longitude": 139.7671, "radius": 1000}
)
for place in result["data"].get("places", []):
    print(place["name"], place.get("distance"))
```

The tool sets `user_location.type` to `"approximate"` itself.

## Extract (`OctenExtract`)

Reads web pages by URL and returns their clean content (markdown by default). The model
passes the URLs and, only when it wants excerpts about one topic, a `query`: each result
then carries ranked `highlights` instead of `full_content`. It returns the same dict and
splits errors the same way as the vertical search tools above.

| Model arguments | Constructor options |
|---|---|
| `urls`, `query` (max 500 characters) | `max_urls` (1-20, default 5), `mode` (`"standard"`, `"advanced"` or `"auto"`), `format` (`"markdown"` or `"text"`), `max_age_seconds` (300-31536000), `url_timeout` (1-60), `include_images`, `include_videos`, `include_audio`, `include_links` (e.g. `{"scope": "prefer_external", "max_links": 50}`; `{}` for the server defaults), `timeout` |

- **`max_urls` caps each call.** A call with more URLs is refused without a request, and
  the model is told how many it sent and to split them into several calls. The tool
  description tells the model the cap. Each URL must be 1-2048 characters.
- **Every URL comes back.** `data.results` has one item per requested URL, in request
  order: `status` is `"success"` (with `title`, `full_content` or `highlights`,
  `time_published` / `time_last_crawled` when known, `category`, `page_structure`, and
  media or links only if you enabled them) or `"failed"` with an `error_message`. Failed
  URLs are not billed; the server then attaches a `warning` such as
  `"1 URL(s) failed and were not billed"`. Content is not truncated.
- **Billing.** `resolved_mode` on each successful item is the mode it was billed at
  (`auto` picks per URL; `advanced` is priced higher). The aggregate usage counts in the
  response's `meta` are not part of the tool result.
- **Two timeouts.** `url_timeout` is the server-side limit per URL (the API's `timeout`,
  server default 30). `timeout` is the HTTP request timeout octen applies to each attempt
  (passed as its `request_timeout`); left unset it is `url_timeout` (or 30) plus 90
  seconds. It is not a deadline for the whole call: octen still retries 408, 429, 500,
  502, 503 and 504 responses (see [Retries](#retries)), so a call can take longer across
  attempts. If you set it, keep it above `url_timeout`.

```python
from langchain_octen import OctenExtract

tool = OctenExtract(max_urls=3)
result = tool.invoke({"urls": ["https://example.com/", "https://docs.octen.ai/"]})
for item in result["data"]["results"]:
    if item["status"] == "success":
        print(item["url"], item.get("title"), len(item.get("full_content") or ""))
    else:
        print(item["url"], "failed:", item.get("error_message"))
```

## Retries

The tools create their `octen` clients with `retry_after_send=False`: a request that times
out or loses its connection after it may already have reached the server is not retried,
because the first attempt may already have been served and billed. Failures known to happen
before sending are still retried, and so are 408, 429, 500, 502, 503 and 504 responses.
Those status-code retries can still be charged twice: a gateway may return 502 or 504 after
the backend has finished. If you also call the API with your own `octen.Octen` /
`octen.AsyncOcten` client, consider passing `retry_after_send=False` there too.

## Configuration

Constructor options of `OctenSearchResults`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `octen_api_key` | `str` | env `OCTEN_API_KEY` | Octen API key |
| `max_results` | `int` | `5` | Number of results (1-100) |
| `search_type` | `str` | `None` | Deprecated and ignored (emits `FutureWarning`); the API no longer accepts it |
| `include_domains` | `list[str]` | `None` | Restrict to these domains |
| `exclude_domains` | `list[str]` | `None` | Exclude these domains |
| `include_text` | `list[str]` | `None` | Text that must appear in results |
| `exclude_text` | `list[str]` | `None` | Text to exclude from results |
| `time_basis` | `str` | `None` | `auto`, `published`, or `crawled` |
| `start_time` | `str` | `None` | Start time (ISO 8601) |
| `end_time` | `str` | `None` | End time (ISO 8601) |
| `format` | `str` | `None` | `text` or `markdown` |
| `safesearch` | `str` | `None` | `off` or `strict` |
| `highlight_enabled` | `bool` | `None` | Return highlighted snippets |
| `highlight_max_tokens` | `int` | `None` | Max tokens for highlights |
| `full_content_enabled` | `bool` | `None` | Return full page content |
| `full_content_max_tokens` | `int` | `None` | Max tokens for full content |
| `timeout` | `float` | `None` | Request timeout in seconds |

## Development

```bash
# Install dependencies
poetry install --with test,dev

# Run unit tests (no API key needed)
make test

# Run live tests against the API (billed; requires OCTEN_API_KEY, sets OCTEN_LIVE=1)
export OCTEN_API_KEY="your-api-key"
make integration_tests

# Run all tests with coverage
poetry run pytest tests/ --cov=langchain_octen --cov-report=term-missing

# Lint and format
make lint
make format
```

## License

MIT
