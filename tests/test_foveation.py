import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from foveate import assembly, errors, models
from foveate.documents import Document
from foveate.documents import page as page_lib
from foveate.foveation import FoveationConfig, Tier, allocate, outline_line
from foveate.tokenizers import HeuristicTokenizer

TOK = HeuristicTokenizer()


def make_doc(pages=40, words=120, doc_id="d"):
    items = []
    for n in range(1, pages + 1):
        text = f"Page {n} discussion. " + "Detail sentence goes here. " * (
            words // 4
        )
        items.append(page_lib.Page(n, text, TOK.count(text), (f"HEADING {n}",)))
    return Document(doc_id, tuple(items))


class TestFoveation:
    def test_top_pages_are_full_neighbours_condensed_rest_outlined(self):
        doc = make_doc()
        result = allocate([(("d", 20), 9.0), (("d", 5), 3.0)], [doc], 2500, TOK)
        tiers = {p.page: p.tier for p in result.pages}
        assert tiers[20] is Tier.FULL and tiers[5] is Tier.FULL
        assert tiers[19] is Tier.CONDENSED and tiers[21] is Tier.CONDENSED
        assert tiers[1] in (Tier.OUTLINE, Tier.DROPPED)
        assert result.tokens <= 2500
        outlined = [p for p in result.tier(Tier.OUTLINE)]
        assert outlined and outlined[0].text.startswith("d p.")
        assert "HEADING" in outlined[0].text

    def test_fits_budget_for_any_ranking_and_budget(self):
        doc = make_doc(30)

        @settings(max_examples=60, deadline=None)
        @given(
            st.lists(st.integers(1, 30), unique=True, max_size=12),
            st.integers(50, 6000),
        )
        def check(pages, budget):
            ranked = [(("d", p), float(30 - i)) for i, p in enumerate(pages)]
            result = allocate(ranked, [doc], budget, TOK)
            assert result.tokens <= budget
            keys = [(p.doc_id, p.page) for p in result.pages]
            assert keys == [("d", n) for n in range(1, 31)]  # one plan per page
            assert all(p.tokens_after == 0 for p in result.tier(Tier.DROPPED))

        check()

    def test_max_full_and_oversized_page(self):
        doc = make_doc(10, words=400)
        capped = allocate(
            [(("d", n), 10.0 - n) for n in range(1, 8)],
            [doc],
            20_000,
            TOK,
            FoveationConfig(max_full=2),
        )
        assert len(capped.tier(Tier.FULL)) == 2
        tiny = allocate([(("d", 3), 1.0)], [doc], 100, TOK)
        full = tiny.tier(Tier.FULL)
        assert full and full[0].tokens_after <= 60 and tiny.tokens <= 100

    def test_ignores_unknown_pages_and_multi_document(self):
        a, b = make_doc(5, doc_id="a"), make_doc(5, doc_id="b")
        result = allocate(
            [(("zzz", 1), 5.0), (("b", 2), 4.0)], [a, b], 3000, TOK
        )
        assert {(p.doc_id, p.page) for p in result.tier(Tier.FULL)} == {
            ("b", 2)
        }
        assert len(result.pages) == 10

    def test_outline_line_prefers_heading_and_is_capped(self):
        doc = make_doc(3)
        assert outline_line(doc, 2, 24, TOK).startswith("d p.2: HEADING 2")
        bare = Document("x", (page_lib.Page(1, "word " * 200, 200),))
        assert TOK.count(outline_line(bare, 1, 12, TOK)) <= 12

    @pytest.mark.parametrize(
        "kw",
        [
            {"fovea_share": 0},
            {"fovea_share": 0.8, "parafovea_share": 0.3},
            {"neighbours": -1},
            {"condensed_tokens": 2},
            {"outline_tokens": 1},
            {"max_full": 0},
        ],
    )
    def test_config_validation(self, kw):
        with pytest.raises(errors.ConfigError):
            FoveationConfig(**kw)
        with pytest.raises(errors.ConfigError):
            allocate([], [make_doc(2)], 0, TOK)


class TestAssembly:
    def test_fixed_then_weighted_split(self):
        out = assembly.allocate(
            1000,
            [
                assembly.Slot("instructions", tokens=100),
                assembly.Slot("history", weight=1, minimum=50),
                assembly.Slot("evidence", weight=3, minimum=100),
            ],
        )
        assert out["instructions"] == 100
        assert out["history"] + out["evidence"] <= 900
        assert out["evidence"] > 2 * out["history"]

    def test_minimum_is_honoured_and_overcommit_rejected(self):
        out = assembly.allocate(
            500,
            [
                assembly.Slot("history", weight=1, minimum=300),
                assembly.Slot("evidence", weight=9, minimum=50),
            ],
        )
        assert out["history"] == 300 and out["evidence"] == 200
        with pytest.raises(errors.ConfigError):
            assembly.allocate(
                100,
                [
                    assembly.Slot("a", tokens=80),
                    assembly.Slot("b", weight=1, minimum=50),
                ],
            )
        with pytest.raises(errors.ConfigError):
            assembly.allocate(100, [assembly.Slot("a"), assembly.Slot("a")])


class TestModels:
    def test_lookup_longest_prefix(self):
        assert models.lookup("gpt-4o-mini").context_window == 128_000
        assert models.lookup("openai/gpt-oss-20b").context_window == 131_072
        assert models.lookup("gpt-4.1-mini").context_window == 1_047_576
        assert models.lookup("totally-unknown") is None

    def test_auto_budget(self):
        assert models.auto_budget("gpt-4o-mini") == int(128_000 * 0.95) - 2_000
        assert (
            models.auto_budget("anything", context_window=8_000)
            == int(8_000 * 0.95) - 2_000
        )
        with pytest.raises(errors.ConfigError):
            models.auto_budget("totally-unknown")
        with pytest.raises(errors.ConfigError):
            models.auto_budget("x", context_window=500)
