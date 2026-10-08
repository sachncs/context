"""Small local model served by vLLM (see docker/README.md).

    docker compose -f docker/compose.yml up -d
    FOVEATE_TEST_VLLM_URL=http://localhost:8000/v1 pytest tests/local

Skipped unless FOVEATE_TEST_VLLM_URL is set.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from foveate import Foveator, Message, Role
from foveate import runtime as runtime_lib
from foveate.backends import OpenAIBackend, ResilientBackend, RetryPolicy
from foveate.bench.longdoc import pipelines
from foveate.cache import base as cache_base
from foveate.tokenizers import HeuristicTokenizer
from tests.test_longdoc import make_doc

URL = os.environ.get("FOVEATE_TEST_VLLM_URL", "")
MODEL = os.environ.get("FOVEATE_TEST_VLLM_MODEL", "openbmb/MiniCPM5-1B")
WINDOW = int(os.environ.get("FOVEATE_TEST_VLLM_WINDOW", "8192"))

pytestmark = pytest.mark.skipif(not URL, reason="set FOVEATE_TEST_VLLM_URL")


@pytest.fixture(scope="module")
def runtime():
    rt = runtime_lib.Runtime(
        backend=ResilientBackend(
            OpenAIBackend(base_url=URL, api_key="local"),
            retry=RetryPolicy(attempts=3, base_delay=1.0),
            timeout=600.0,
            max_concurrency=1,
        ),
        model=MODEL,
        cache=cache_base.NullCache(),
        tokenizer=HeuristicTokenizer(),
        context_window=WINDOW,
        concurrency=1,
    )
    yield rt
    rt.close()


def test_backend_answers_and_reasoning_is_stripped(runtime):
    out = asyncio.run(
        runtime.complete(
            (Message(Role.USER, "What is 12*12? Reply with only the number."),),
            source="t",
            namespace="vllm-basic",
            max_tokens=1024,
        )
    )
    assert "144" in out.text and "<think>" not in out.text


def test_foveate_answers_where_full_context_cannot_and_never_overclaims(
    runtime,
):
    doc = make_doc(pages=60)
    question = "What were Acme capital expenditures in fiscal 2018?"
    full = asyncio.run(
        pipelines.FullContext(runtime, 3000).answer(question, doc)
    )
    assert full.overflow  # ~60 pages cannot fit an 8k window
    answer = Foveator(runtime, budget=3500, max_rounds=1).ask(question, [doc])
    assert "1,577" in answer.text or "1577" in answer.text
    # A 1B model often omits citations. The contract is that "grounded" is
    # only ever claimed for verified evidence.
    assert answer.grounded == (
        bool(answer.citations) and all(c.verified for c in answer.citations)
    )


def test_abstains_when_the_pages_lack_the_answer(runtime):
    doc = make_doc(pages=30)
    answer = Foveator(runtime, budget=3500, max_rounds=1).ask(
        "What is the name of the CEO's dog?", [doc]
    )
    assert answer.abstained or not answer.grounded
