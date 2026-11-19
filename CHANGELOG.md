# Changelog

## 0.2.0

### Added

- Five vertical search tools: `OctenBroadSearch` (`octen_broad_search`), `OctenNewsSearch`
  (`octen_news_search`), `OctenBusinessSearch` (`octen_business_search`), `OctenPulseSearch`
  (`octen_pulse_search`) and `OctenPlaceSearch` (`octen_place_search`).
  - The model only gets the query and filters that do not change the cost of a call
    (`include_domains`, `exclude_domains`, `time_range`, `language`; Place: `latitude`,
    `longitude`, `radius`, `user_location`). Result counts, highlight / full content, Broad
    `max_queries`, Pulse `mode`, News `subjects`, Business `entities`, Place `places` and
    `web_search`, and `timeout` are constructor options.
  - Each call returns `{"data", "request_id", "warning"}`; `warning` is omitted when empty.
  - Authentication failures, 403, other 4xx responses and invalid constructor options are
    raised; rate limits, 5xx, 400/422, timeouts and connection errors are returned to the
    model. Invalid model arguments are rejected before any request is made.
- CI on Python 3.9-3.13, LangChain standard tool tests, and opt-in live tests
  (`OCTEN_LIVE=1`).

### Changed

- Clients are created with `retry_after_send=False`: a request that fails after it may have
  reached the server (client-side timeout, connection dropped mid-response) is no longer
  retried, so it cannot be billed twice. Requests that never left the client and 408 / 429 /
  5xx responses are still retried.
- `octen_search` has a reworded tool description; its parameters are unchanged.
- Requires `octen>=0.12.0,<1.0.0`.

### Fixed

- `search_type` no longer breaks every call. `octen` 0.5.0 removed the parameter, so passing
  it raised `TypeError`. It is now accepted but ignored, with a `DeprecationWarning`; it will
  be removed in a future release.

### Known inconsistency

- `octen_search` (`OctenSearchResults`) keeps its 0.1 behaviour: it returns a list of results
  and reports errors as `[{"error": ...}]`, unlike the new tools. Both will be unified in a
  future major release.

## 0.1.1

- Fixed the repository URL shown on PyPI.

## 0.1.0

- Initial release: `OctenSearchResults` (`octen_search`).
