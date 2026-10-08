"""Framework-neutral document tools (no framework installed)."""

import pytest

from foveate import errors
from foveate.integrations import tools
from tests.test_longdoc import make_doc


def toolbox(**kw):
    return tools.DocumentTools([make_doc(pages=30)], **kw)


def test_read_pages_marks_each_page_for_citation():
    text = toolbox().read("7-8")
    assert "[acme_2018 p.7]" in text and "[acme_2018 p.8]" in text
    assert "1,577" in text and "p.9]" not in text


def test_read_reports_bad_requests_to_the_agent_instead_of_raising():
    assert toolbox().read("99").startswith("error:")
    assert toolbox().read("1", doc_id="nope").startswith("error:")


def test_search_finds_the_fact_page_with_a_snippet():
    out = toolbox().search("capital expenditures fiscal 2018", k=3)
    first = out.splitlines()[0]
    assert first.startswith("[acme_2018 p.7]") and "1,577" in first
    assert toolbox().search("zzzqqq") == "no matching pages"
    assert toolbox().search("capital expenditures", doc_id="other") == (
        "no matching pages"
    )


def test_outline_lists_headings_and_size():
    out = toolbox().outline()
    assert out.splitlines()[0].startswith("acme_2018: 30 pages")
    assert "p.7 SECTION 7" in out
    assert toolbox().outline("nope").startswith("error:")


def test_long_replies_are_cut_and_say_so():
    out = toolbox(max_tokens=200).read("1-30")
    assert out.endswith("request fewer pages]")
    assert toolbox().tokenizer.count(out) < 300


def test_functions_are_documented_plain_callables():
    fns = {fn.__name__: fn for fn in toolbox().functions()}
    assert set(fns) == {"read_pages", "search_document", "document_outline"}
    assert all(fn.__doc__ for fn in fns.values())
    assert "1,577" in fns["read_pages"]("7")
    assert "p.7" in fns["search_document"]("capital expenditures")


def test_config_and_ids_are_validated():
    with pytest.raises(errors.ConfigError):
        toolbox(max_tokens=10)
    with pytest.raises(errors.ValidationError):
        tools.DocumentTools([make_doc(), make_doc()])
    two = tools.DocumentTools([make_doc("a"), make_doc("b")])
    assert two.read("1").startswith("error:")  # ambiguous without doc_id
    assert "[b p.1]" in two.read("1", doc_id="b")
