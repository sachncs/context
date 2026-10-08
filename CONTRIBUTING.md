# Contributing to Foveate

Thanks for your interest in Foveate. This document explains how to set up
the project locally, run the test suite, and submit a pull request.

## Reporting issues

Open an issue at
[`sachncs/foveate/issues`](https://github.com/sachncs/foveate/issues)
using the appropriate template (`bug`, `feature_request`, or `question` via
Discussions if enabled). For security issues, follow
[`SECURITY.md`](./SECURITY.md).

## Development setup

```
make setup
source .venv/bin/activate
```

## The local gate

```
make check     # ruff check + format check, mypy --strict, pytest with a 90% coverage gate
make format    # apply ruff fixes and formatting
```

## Code standards

We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
(80 columns, Google-style docstrings, absolute imports, full type
annotations) with these project rules:

- **No underscore-prefixed names.** No `_helper`, `_CONSTANT`, `_attr`.
  Dunders are fine. This deliberately departs from the Google guide; the
  public API is defined by `__all__` in each package `__init__`, the
  `internals` package, and the docs. `tests/test_style.py` enforces it.
- Value types are `@dataclass(frozen=True)` (with `slots=True` unless a base
  class needs otherwise). Closed sets are `enum.Enum`.
- Extension points are ABCs with a registry on the class; add behaviour by
  subclassing and registering, not by adding flags.
- No thin wrappers, shims or compatibility aliases.
- No module-level mutable state. Pass a `Runtime`.
- Catch specific exceptions; every deliberate error derives from `FoveateError`.
- Every LLM call goes through `Runtime.complete`.
- Unit tests are offline and deterministic. Use `tests/faults.py` only for
  fault injection (errors, timeouts, truncation); never use a fake model to
  assert on model behaviour. Anything that depends on what a model returns
  belongs in `tests/integration/`, which calls a real provider (set
  `NVIDIA_API_KEY` or `FOVEATE_TEST_API_KEY`) and is skipped without a key,
  or `tests/local/` for a local vLLM server (`FOVEATE_TEST_VLLM_URL`). These
  run on your machine, not in CI. Framework adapters live in `integrations/<name>/` and have their own
  no-model tests (`make integrations`). Add a regression test with every bug
  fix.

## Pull request flow

1. Fork the repository.
2. Create a topic branch off `master` (use linear history).
3. Make focused commits with clear messages.
4. Ensure `tests`, `lint`, and `format` all pass.
5. Use the [PR template](./.github/PULL_REQUEST_TEMPLATE.md).
6. Push the branch and open a pull request targeting `master`.

By submitting a pull request, you agree to follow the
[Code of Conduct](./CODE_OF_CONDUCT.md).

## Releases

Releases follow semantic versioning. A maintainer creates a `vX.Y.Z` tag after
CI is green; the release workflow builds and validates the artifacts, creates a
GitHub release, and publishes to PyPI using trusted publishing. Public API
changes require a changelog entry.
