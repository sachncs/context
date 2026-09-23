# Production operations

The published distribution is `ceng-context`; the Python import remains
`ceng`. The shorter `ceng` name is already registered on PyPI by an unrelated
project.

`ceng` is a synchronous library. Run it inside the host service that owns the
request lifecycle, credentials, timeouts, and observability.

## Backend configuration

Select a backend with `CENG_BACKEND` or `ceng.set_backend()`. Configure
provider credentials through the provider's environment variables; never put
keys in source, prompts, cache files, or OKF bundles.

The hosted adapters use a 60-second timeout by default and retry transient
failures up to three times with jittered exponential backoff. Tune these with:

```text
CENG_TIMEOUT_SECONDS=60
CENG_RETRY=3
CENG_RETRY_BASE_MS=250
```

Per-call keyword arguments take precedence over environment defaults.

## Storage and privacy

- Put `cache_dir` on durable, private storage when cache reuse matters.
- Put OKF bundles and notes in an application-owned directory with least-
  privilege filesystem permissions.
- Treat caches, notes, and bundles as potentially sensitive because they may
  contain derived or original user content.
- ceng has no telemetry; configure application logging explicitly with
  `ceng.configure_logging()` if operational logs are required.

## Reliability

Catch `RuntimeError` and `CompressError` at the service boundary, record the
operation and backend name, and return a safe application-level error. Do not
log full prompts or model responses by default. Use bounded request deadlines
at the service layer as an outer guard around provider calls.

## Release checklist

1. Run `pytest`, Ruff, mypy, coverage, and the package build locally.
2. Confirm `twine check dist/*` passes.
3. Update `CHANGELOG.md` and the package/site versions together.
4. Push a `vX.Y.Z` tag after CI is green.
5. Configure the PyPI trusted publisher for this repository's `Release`
   workflow and `pypi` environment before the first release.
