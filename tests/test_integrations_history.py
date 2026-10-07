"""Framework-neutral history compression (no framework installed)."""

import dataclasses

import pytest

from ceng import Message, Role, errors
from ceng.integrations import history
from tests import faults
from tests.test_compression import make_runtime


@dataclasses.dataclass(frozen=True)
class Turn:
    kind: str  # "user" | "assistant" | "tool"
    text: str


class ToyAdapter(history.HistoryAdapter[Turn]):
    ROLES = {
        "user": Role.USER,
        "assistant": Role.ASSISTANT,
        "tool": Role.ASSISTANT,
    }

    def flatten(self, message):
        text = (
            f"[tool result: {message.text}]"
            if message.kind == "tool"
            else message.text
        )
        return [Message(self.ROLES[message.kind], text)]

    def rebuild(self, messages):
        return [
            Turn("user" if m.role is Role.USER else "assistant", m.content)
            for m in messages
        ]

    def starts_turn(self, message):
        return message.kind == "user"


def conversation(turns=6, words=80):
    items = []
    for i in range(turns):
        items.append(Turn("user", f"question {i} " + "word " * words))
        items.append(Turn("assistant", f"answer {i} " + "word " * words))
    return items


def compressor(rt, **kw):
    kw.setdefault("budget", 600)
    kw.setdefault("method", "extractive")
    return history.HistoryCompressor(ToyAdapter(), rt, **kw)


async def acompress(c, items):
    return await c.compress(items)


def run(coro):
    import asyncio

    return asyncio.run(coro)


def test_under_budget_returns_history_unchanged():
    rt, backend = make_runtime()
    items = conversation(1, 5)
    out = run(acompress(compressor(rt, budget=10_000), items))
    assert out == items and backend.requests == []


def test_old_turns_are_compressed_and_recent_turns_kept_verbatim():
    rt, _ = make_runtime()
    items = conversation()
    c = compressor(rt, keep_last=4)
    out = run(acompress(c, items))
    assert out[-4:] == items[-4:]
    assert c.count(out) <= 600 < c.count(items)
    assert c.reports and c.reports[-1].method == "extractive"


def test_cut_never_separates_a_tool_result_from_its_call():
    rt, _ = make_runtime()
    items = conversation(4, 80)
    items.insert(-2, Turn("tool", "result " + "word " * 80))
    c = compressor(rt, keep_last=2)
    cut = c.split_point(items)
    assert items[cut].kind == "user"  # a turn boundary, not the tool result
    out = run(acompress(c, items))
    assert out[-(len(items) - cut) :] == items[cut:]


def test_nothing_to_compress_when_no_older_turn_boundary():
    rt, _ = make_runtime()
    items = [Turn("user", "word " * 800)]
    c = compressor(rt, budget=100)
    assert c.split_point(items) == 0
    assert run(acompress(c, items)) == items


def test_tail_larger_than_budget_still_gets_minimum_room():
    rt, _ = make_runtime()
    c = compressor(rt, budget=50, keep_last=4, min_prefix_tokens=64)
    out = run(acompress(c, conversation()))
    assert out[-4:] == conversation()[-4:]
    assert c.reports[-1].budget == 64


def test_llm_method_failure_falls_back_with_default_method():
    rt, _ = make_runtime(steps=[errors.PermanentBackendError("down")])
    c = history.HistoryCompressor(ToyAdapter(), rt, budget=600)
    out = run(acompress(c, conversation()))
    assert c.count(out) <= 600
    assert c.reports[-1].method == "ushape|extractive"


@pytest.mark.parametrize(
    "bad", [{"budget": 0}, {"keep_last": 0}, {"min_prefix_tokens": 0}]
)
def test_config_validation(bad):
    rt, _ = make_runtime()
    with pytest.raises(errors.ConfigError):
        compressor(rt, **bad)


def test_faults_helper_is_importable():
    assert faults.ScriptedBackend


class TestStageOptions:
    def flat(self, *roles):
        return [Message(role, "x") for role in roles]

    def test_ushape_defaults_to_transcript_mode_keeping_leading_system(self):
        rt, _ = make_runtime()
        c = compressor(rt, method="ushape|extractive")
        flat = self.flat(Role.SYSTEM, Role.SYSTEM, Role.USER, Role.ASSISTANT)
        assert c.stage_options(flat) == {"ushape": {"head": 2, "tail": 0}}

    def test_single_ushape_method_gets_flat_options(self):
        rt, _ = make_runtime()
        c = compressor(rt, method="ushape")
        assert c.stage_options(self.flat(Role.USER)) == {"head": 0, "tail": 0}

    def test_user_options_override_defaults(self):
        rt, _ = make_runtime()
        c = compressor(
            rt, method="ushape|extractive", options={"ushape": {"tail": 1}}
        )
        assert c.stage_options(self.flat(Role.USER))["ushape"] == {
            "head": 0,
            "tail": 1,
        }

    def test_other_methods_pass_options_through(self):
        rt, _ = make_runtime()
        c = compressor(rt, method="extractive", options={"edge_bonus": 0.5})
        assert c.stage_options(self.flat(Role.USER)) == {"edge_bonus": 0.5}
