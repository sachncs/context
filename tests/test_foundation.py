import asyncio
import logging

import pytest

from ceng import errors, messages, observability, prompts, usage
from ceng.backends import classify
from ceng.internals import hashing, registry, runner
from ceng.tokenizers import (
    HeuristicTokenizer,
    TiktokenTokenizer,
    default_tokenizer,
    for_model,
)


class TestMessages:
    def test_roundtrip(self):
        m = messages.Message.from_mapping(
            {"role": "user", "content": "hi", "name": "bob"}
        )
        assert m.to_mapping() == {
            "role": "user",
            "content": "hi",
            "name": "bob",
        }

    def test_multipart_flattened(self):
        m = messages.Message.from_mapping(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    "b",
                    {"type": "img"},
                ],
            }
        )
        assert m.content == "a\nb"

    @pytest.mark.parametrize(
        "bad",
        [{"content": "x"}, {"role": "nope", "content": "x"}, {"role": "user"}],
    )
    def test_invalid(self, bad):
        with pytest.raises(errors.ValidationError):
            messages.Message.from_mapping(bad)

    def test_unsupported_content(self):
        with pytest.raises(errors.ValidationError):
            messages.flatten_content(3)
        assert messages.flatten_content(None) == ""

    def test_with_content(self):
        m = messages.Message(messages.Role.USER, "a").with_content("b")
        assert m.content == "b"


class TestTokenizers:
    def test_heuristic(self):
        t = HeuristicTokenizer()
        assert t.count("") == 0
        assert t.count("abcd") == 1
        assert t.count("abcde") == 2
        assert t.truncate("a" * 100, 5) == "a" * 20
        assert t.truncate("abc", 0) == ""
        with pytest.raises(errors.ConfigError):
            HeuristicTokenizer(0)

    def test_tiktoken(self):
        pytest.importorskip("tiktoken")
        t = TiktokenTokenizer()
        assert t.count("hello world") == 2
        assert t.count(t.truncate("hello world " * 20, 5)) <= 5
        assert t.truncate("x", 0) == ""
        with pytest.raises(errors.ConfigError):
            TiktokenTokenizer("nope-encoding")

    def test_default(self):
        assert default_tokenizer().count("hello") >= 1

    def test_for_model(self):
        pytest.importorskip("tiktoken")
        known = for_model("gpt-4o-mini")
        assert isinstance(known, TiktokenTokenizer)
        assert known.encoding_name == "o200k_base"
        unknown = for_model("some-unlisted-model")
        assert isinstance(unknown, TiktokenTokenizer)
        assert unknown.encoding_name == "cl100k_base"

    def test_for_model_without_tiktoken(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "tiktoken", None)
        assert isinstance(for_model("gpt-4o-mini"), HeuristicTokenizer)


class TestPrompts:
    template = prompts.PromptTemplate("t", "1", "sys", "x={x} y={y}")

    def test_render_and_fields(self):
        assert self.template.fields() == {"x", "y"}
        assert self.template.render_user(x="1", y="2") == "x=1 y=2"

    def test_mismatch(self):
        with pytest.raises(errors.ValidationError):
            self.template.render_user(x="1")

    def test_fingerprint_changes_with_text(self):
        other = prompts.PromptTemplate("t", "1", "sys2", "x={x} y={y}")
        assert other.fingerprint != self.template.fingerprint


class TestRegistry:
    def test_add_get(self):
        r = registry.Registry[int]("thing")

        @r.register("a")
        def a():  # pragma: no cover
            return 1

        assert "a" in r and list(r) == ["a"] and r.names() == ["a"]
        with pytest.raises(errors.ConfigError):
            r.add("a", 1)
        with pytest.raises(errors.ConfigError):
            r.add("", 1)
        with pytest.raises(errors.ConfigError, match="known"):
            r.get("zzz")


def test_hash_stable_and_order_independent():
    assert hashing.fingerprint({"a": 1, "b": 2}) == hashing.fingerprint(
        {"b": 2, "a": 1}
    )
    assert hashing.fingerprint("a") != hashing.fingerprint("b")


class TestRunner:
    async def value(self):
        return 7

    def test_plain(self):
        assert runner.run_sync(self.value()) == 7

    def test_inside_running_loop(self):
        async def outer():
            return runner.run_sync(self.value())

        assert asyncio.run(outer()) == 7


def test_usage_and_price():
    u = usage.Usage(1_000_000, 500_000) + usage.Usage(0, 500_000)
    assert u.total_tokens == 2_000_000
    table = usage.PriceTable({"m": usage.Price(1.0, 2.0)})
    assert table.cost("m", u) == pytest.approx(3.0)
    assert table.cost("other", u) == 0.0


class TestObservability:
    def test_metrics(self):
        m = observability.MetricsObserver()
        observability.emit(
            [m],
            observability.CacheLookup(source="s", hit=True),
        )
        observability.emit([m], observability.CacheLookup(source="s"))
        observability.emit(
            [m],
            observability.BackendCall(
                source="s", usage=usage.Usage(1, 2), seconds=0.5
            ),
        )
        observability.emit([m], observability.RetryScheduled(source="s"))
        observability.emit([m], observability.StepFinished(source="s"))
        assert (m.cache_hits, m.cache_misses, m.backend_calls, m.retries) == (
            1,
            1,
            1,
            1,
        )
        assert m.usage.total_tokens == 3

    def test_logging_and_isolation(self, caplog):
        class Boom(observability.Observer):
            def handle(self, event):
                raise RuntimeError("x")

        with caplog.at_level(logging.INFO, logger="ceng"):
            observability.emit(
                [Boom(), observability.LoggingObserver()],
                observability.StepFinished(source="s", step="a"),
            )
        assert "StepFinished" in caplog.text and "failed" in caplog.text


class TestClassify:
    class Status(Exception):
        def __init__(self, status, headers=None):
            self.status_code = status
            if headers:
                self.response = type("R", (), {"headers": headers})()

    def test_status_codes(self):
        assert isinstance(
            classify.classify(self.Status(429, {"retry-after": "2"})),
            errors.RateLimitError,
        )
        assert (
            classify.classify(
                self.Status(429, {"retry-after": "2"})
            ).retry_after
            == 2.0
        )
        assert isinstance(
            classify.classify(self.Status(503)), errors.TransientBackendError
        )
        assert isinstance(
            classify.classify(self.Status(401)), errors.PermanentBackendError
        )

    def test_names(self):
        class APIConnectionError(Exception):
            pass

        class RateLimitErr(Exception):
            pass

        assert isinstance(
            classify.classify(APIConnectionError()),
            errors.TransientBackendError,
        )
        assert isinstance(
            classify.classify(RateLimitErr()), errors.RateLimitError
        )
        assert isinstance(
            classify.classify(TimeoutError()), errors.TransientBackendError
        )
        assert isinstance(
            classify.classify(ValueError("bug")), errors.PermanentBackendError
        )

    def test_passthrough_and_bad_header(self):
        e = errors.PermanentBackendError("x")
        assert classify.classify(e) is e
        assert (
            classify.retry_after_seconds(
                self.Status(429, {"retry-after": "soon"})
            )
            is None
        )
        assert classify.retry_after_seconds(ValueError()) is None


def test_budget_error_fields():
    e = errors.BudgetExceededError(10, 20)
    assert (e.budget, e.actual) == (10, 20)
    c = errors.CompressionError("m", step="s", cause=ValueError())
    assert c.step == "s"
