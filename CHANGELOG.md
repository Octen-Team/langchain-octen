# Changelog

## 0.4.0

### Added

- `OctenImageSearch` (`octen_image_search`): Image Search, images on the web found by a text
  description or by a reference image.
  - The model passes exactly one of `query` (max 500 characters) or `image_url` (a public
    http(s) URL), plus `include_domains` / `exclude_domains` (up to 10 each, matched against
    each result's source page). A call with both inputs or neither is refused without a
    request, and the model is told which to pass. Base64 images are not accepted from the
    model; call `octen` directly for those.
  - `count` (1-10) and `safesearch` (`"off"` or `"strict"`) and `timeout` are constructor
    options, checked when the tool is built.
- `OctenDesignSearch` (`octen_design_search`): Design Search, design references (web pages,
  page sections, app screens and slides) with their design values, plus web results on the
  same query.
  - The model passes `query` (max 500 characters), `group_id` or both. `group_id` is the
    `group.group_id` of an earlier image result and returns that page's or deck's images in
    group order instead of a search. A call with neither is refused without a request.
  - `images` (`enable`, `count`, `design_facts`, `snippet`) and `web_search` (`enable`,
    `count`, `highlight`, `full_content`, domain, language and time filters) and `timeout`
    are constructor options, checked when the tool is built: `images.group_id` is refused
    (it is the model's), `highlight` and `full_content` accept only the `enable` and
    `max_tokens` keys, `language` must be one of the documented ISO 639-1 codes and
    `start_time` / `end_time` must be RFC 3339 times. Left unset, the server defaults apply,
    so each image carries `design_facts` and a snippet of up to 5000 tokens (large output);
    `images={"design_facts": False, "snippet": {"enable": False}}` turns them off and
    `images={"snippet": {"max_tokens": ...}}` shortens the snippets.
- Both tools return `{"data", "request_id", "warning"}` with the response's data unchanged,
  and split errors like the other vertical tools.

The existing tools are unchanged: same parameters, same requests, same return shape. The
new tools are separate classes; nothing is added to an agent unless you pass them in.
`octen>=0.13.0,<1.0.0` already provides both endpoints, so the dependency range is unchanged.

## 0.3.0

### Added

- `OctenSearchPremier` (`octen_search_premier`): Web Search Premier, web search with deeper
  retrieval and reranking for hard-to-find specific facts and pages.
  - The model gets the same arguments as `octen_search`: `query`, `include_domains`,
    `exclude_domains`. Result count, `result_type`, highlight / full content, text,
    language and time filters, `format`, `safesearch`, `include_images` and `timeout` are
    constructor options, checked when the tool is built; `highlight` and `full_content`
    accept only the `enable` and `max_tokens` keys, and `start_time` / `end_time` must be
    RFC 3339 times (e.g. `2025-01-01T00:00:00Z`).
  - Return shape and error handling are the same as the vertical tools.
- `OctenExtract` (`octen_extract`): Extract, the clean content of web pages read by URL.
  - The model gets `urls` and an optional `query` (ranked highlights instead of the full
    content). `max_urls` (default 5, 1-20) caps the URLs per call: a call with more is
    refused without a request, and the model is told to split it. `mode`, `format`,
    `max_age_seconds`, `url_timeout` (the API's per-URL `timeout`), `include_images`,
    `include_videos`, `include_audio`, `include_links` and `timeout` are constructor
    options, checked when the tool is built.
  - `data.results` holds the per-URL items, failures included (`status: "failed"` with
    `error_message`; failed URLs are not billed); match them to the requested URLs by each
    item's `url`, not by position. Each successful item's `resolved_mode` is the mode it
    was billed at. Return shape and error handling are the same as the vertical tools.
  - Left unset, `timeout` (the HTTP request timeout octen applies to each attempt, passed
    as its `request_timeout`) is `url_timeout`, or the server default of 30, plus 90
    seconds, so the client does not give up while the server is still within its per-URL
    limit. It is not a deadline for the whole call: octen still retries 408, 429, 500,
    502, 503 and 504 responses.

### Changed

- Requires `octen>=0.13.0,<1.0.0` (the release that adds `search_premier`).

The existing tools are unchanged: same parameters, same requests, same return shape.

## 0.2.0

### Upgrading from 0.1.x

Existing code keeps working: `OctenSearchResults` (`octen_search`) has the same
parameters, return shape and error behaviour. Things that do change:

- **`octen` is upgraded to 0.12.0 or newer.** If your project pins `octen` below
  0.12.0, dependency resolution fails until you relax that pin.
- **No retry after a request was sent.** A request that timed out or lost its
  connection after it was sent is no longer retried, so it can't be billed a
  second time by the retry. Calls that used to succeed on such a retry now
  return `[{"error": ...}]`. Retries on status codes (408, 429, 5xx) are
  unchanged.
- **`search_type` is ignored.** With `octen` 0.5.0 or newer, passing it used to
  raise `TypeError`; it now works and emits a `FutureWarning`. If you relied on
  it with an older `octen`, it no longer has any effect.
- **Reworded tool description.** The model sees a new description for
  `octen_search`, so when it chooses the tool may shift slightly.
- The five new tools are separate classes; nothing is added to an agent unless
  you pass them in.

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
