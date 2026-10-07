"""Shared fixtures."""

import pytest

from ceng import runtime as runtime_lib
from ceng.cache import base as cache_base
from ceng.tokenizers import HeuristicTokenizer
from tests import faults as scripted


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
