# langchain-octen

[![PyPI version](https://badge.fury.io/py/langchain-octen.svg)](https://pypi.org/project/langchain-octen/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LangChain integration for [Octen Web Search](https://octen.ai). Use Octen's fast, real-time web search inside LangChain agents and chains.

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

## Usage

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
results = tool.invoke({
    "query": "machine learning",
    "include_domains": ["arxiv.org", "papers.nips.cc"],
})
```

### In a LangChain Agent

```python
from langchain_octen import OctenSearchResults
from langgraph.prebuilt import create_react_agent

tool = OctenSearchResults(max_results=5)
agent = create_react_agent(llm, [tool])
response = agent.invoke({"messages": [("user", "What happened in AI today?")]})
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

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `octen_api_key` | `str` | env `OCTEN_API_KEY` | Octen API key |
| `max_results` | `int` | `5` | Number of results (1-100) |
| `search_type` | `str` | `None` | `auto`, `keyword`, or `semantic` |
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

## License

MIT
