"""Offline runtime that proves benchmark wiring without network access."""

from __future__ import annotations

from collections.abc import Sequence

from ceng import runtime as runtime_lib
from ceng.backends import base as backend_base
from ceng.backends import scripted
from ceng.cache import base as cache_base
from ceng.evolution import grading
from ceng.tokenizers import base as tokenizer_base

MARKER = "Playbook of strategies"
FALLBACK_ANSWER = "unknown"


def wiring_runtime(samples: Sequence[grading.Sample]) -> runtime_lib.Runtime:
    """Builds a runtime whose model answers correctly only with a playbook.

    The scripted model returns the gold answer when the prompt contains an
    injected playbook and a wrong answer otherwise. A non-zero accuracy delta
    therefore proves the playbook reaches the prompt; it says nothing about
    real model quality.

    Args:
        samples: Samples whose gold answers the stub may return.
    """
    answers = {sample.question: sample.target for sample in samples}

    def respond(request: backend_base.Request) -> str:
        text = "\n".join(message.content for message in request.messages)
        if MARKER not in text:
            return FALLBACK_ANSWER
        for question, target in answers.items():
            if question in text:
                return target
        return FALLBACK_ANSWER

    return runtime_lib.Runtime(
        backend=scripted.ScriptedBackend(default=respond),
        cache=cache_base.NullCache(),
        tokenizer=tokenizer_base.HeuristicTokenizer(),
        model="offline-wiring-check",
    )
