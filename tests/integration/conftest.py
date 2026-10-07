"""Fixtures for tests that call a real LLM provider.

Configuration (environment):
    NVIDIA_API_KEY / CENG_TEST_API_KEY   key for the OpenAI-compatible API
    CENG_TEST_BASE_URL                   default https://integrate.api.nvidia.com/v1
    CENG_TEST_MODEL                      default openai/gpt-oss-20b
    CENG_TEST_RATE                       calls per second, default 0.6
    CENG_TEST_NO_CACHE=1                 bypass the persistent response cache

Real responses are cached on disk (`.ceng/test-cache`, git-ignored) so reruns
are fast and cheap; tests that assert cache or latency behaviour use unique
prompts. Without a key every test here is skipped.
"""

from __future__ import annotations

import os
import pathlib
import uuid

import pytest

from ceng import observability
from ceng import runtime as runtime_lib
from ceng.backends import OpenAIBackend, ResilientBackend, RetryPolicy
from ceng.cache import base as cache_base
from ceng.cache import sqlite as cache_sqlite

API_KEY = os.environ.get("CENG_TEST_API_KEY") or os.environ.get(
    "NVIDIA_API_KEY"
)
BASE_URL = os.environ.get(
    "CENG_TEST_BASE_URL", "https://integrate.api.nvidia.com/v1"
)
MODEL = os.environ.get("CENG_TEST_MODEL", "openai/gpt-oss-20b")
RATE = float(os.environ.get("CENG_TEST_RATE", "0.6"))
CACHE_DIR = pathlib.Path(__file__).resolve().parents[2] / ".ceng" / "test-cache"

requires_key = pytest.mark.skipif(
    not API_KEY, reason="set NVIDIA_API_KEY (or CENG_TEST_API_KEY) to run"
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(requires_key)


def make_backend(
    api_key: str | None = None, model_rate: float = RATE
) -> ResilientBackend:
    return ResilientBackend(
        OpenAIBackend(base_url=BASE_URL, api_key=api_key or API_KEY),
        retry=RetryPolicy(attempts=4, base_delay=2.0, max_delay=30.0),
        timeout=120.0,
        max_concurrency=4,
        rate_per_second=model_rate,
    )


@pytest.fixture(scope="session")
def metrics() -> observability.MetricsObserver:
    return observability.MetricsObserver()


@pytest.fixture(scope="session")
def runtime(metrics: observability.MetricsObserver):
    cache: cache_base.Cache
    if os.environ.get("CENG_TEST_NO_CACHE"):
        cache = cache_base.NullCache()
    else:
        cache = cache_sqlite.SqliteCache(CACHE_DIR)
    rt = runtime_lib.Runtime(
        backend=make_backend(),
        model=MODEL,
        cache=cache,
        observers=(metrics,),
        options={"reasoning_effort": "low"},
        concurrency=4,
    )
    yield rt
    rt.close()


@pytest.fixture
def unique() -> str:
    """A string that makes a prompt unseen by any cache."""
    return uuid.uuid4().hex[:12]
