"""Shared fixtures."""

import pytest

from ceng import runtime as runtime_lib
from ceng.backends import scripted
from ceng.cache import base as cache_base
from ceng.tokenizers import HeuristicTokenizer


@pytest.fixture
def backend() -> scripted.ScriptedBackend:
    return scripted.ScriptedBackend()


@pytest.fixture
def runtime(backend: scripted.ScriptedBackend) -> runtime_lib.Runtime:
    return runtime_lib.Runtime(
        backend=backend,
        cache=cache_base.MemoryCache(),
        tokenizer=HeuristicTokenizer(),
    )
