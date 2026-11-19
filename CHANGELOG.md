# Changelog

## 0.2.0

### Added

- Five vertical search tools: `OctenBroadSearch` (`octen_broad_search`), `OctenNewsSearch`
  (`octen_news_search`), `OctenBusinessSearch` (`octen_business_search`), `OctenPulseSearch`
  (`octen_pulse_search`) and `OctenPlaceSearch` (`octen_place_search`).
  - The model only gets the query and search filters (`include_domains`, `exclude_domains`,
    `time_range`, `language`; Place: `latitude`, `longitude`, `radius`, `user_location`).
    Billing-related options and limits (result counts, highlight / full content, Broad
    `max_queries`, Pulse `mode`, News `subjects`, Business `entities`, Place `places` and
    `web_search`) and `timeout` are constructor options. Broad Search is billed per
    sub-query; `max_queries` caps the number, and the actual count depends on the query.
  - `response_format` must stay `"content"`; other values raise `ValueError` at
    construction.
  - Each call returns `{"data", "request_id", "warning"}`; `warning` is omitted when empty.
  - Constructor options are validated when the tool is built: an out-of-range value, an
    unknown option name or a `timeout` that is not a positive finite number raises
    `ValueError`.
  - Authentication failures, 403 and other 4xx responses are raised; rate limits, 5xx,
    400/408/422, 2xx responses with an error envelope or a malformed body, timeouts and
    connection errors are returned to the model. Invalid or unknown model arguments are
    rejected before any request is made.
  - A timeout or connection error that may have happened after the request was sent, a 502
    or 504, or a failed 2xx response is reported to the model with a note that the call may
    already have been billed and should not be retried automatically.
  - Dict options (`highlight`, `full_content`, `subjects`, `entities`, `places`,
    `web_search`) are deep-copied at construction; later changes to the caller's dicts have
    no effect.
  - The tools also accept a plain string as input (`tool.invoke("coffee shops")`), used as
    the query.
- CI on Python 3.9-3.13, LangChain standard tool tests, and opt-in live tests
  (`OCTEN_LIVE=1`).

### Changed

- Clients are created with `retry_after_send=False`: a request that times out or loses its
  connection after it may have reached the server is no longer retried. Requests that never
  left the client are still retried, and so are 408, 429, 500, 502, 503 and 504 responses;
  those status-code retries can still lead to a second charge (a gateway may return 502 or
  504 after the backend finished). This also applies to the existing `octen_search` tool
  (`OctenSearchResults` and `OctenSearchAPIWrapper`): it no longer retries a request that
  timed out or lost its connection after being sent, as it did in 0.1.
- `octen_search` has a reworded tool description; its parameters are unchanged.
- Requires `octen>=0.12.0,<1.0.0`.

### Fixed

- `search_type` no longer breaks every call. `octen` 0.5.0 removed the parameter, so passing
  it raised `TypeError`. It is now accepted but ignored, and building the tool emits a
  `FutureWarning` (shown by default, unlike `DeprecationWarning`); it will be removed in a
  future release.

### Known issues

- `octen_search` (`OctenSearchResults`) keeps its 0.1 behaviour: it returns a list of results
  and reports errors as `[{"error": ...}]`, unlike the new tools. Both will be unified in a
  future major release.
- Building a vertical tool with `model_validate` instead of the constructor, with a
  misspelt key argument, can still include the raw key in pydantic's structured error
  (`errors()` / `json()`).
- When octen retries a 502 or 504 internally and the final attempt fails differently (for
  example with a 503), the "may already have been billed" note is not added, because the
  tool only sees the final error.

## 0.1.1

- Fixed the repository URL shown on PyPI.

## 0.1.0

- Initial release: `OctenSearchResults` (`octen_search`).
