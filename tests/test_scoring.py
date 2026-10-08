import pytest

from foveate.bench import scoring
from foveate.tokenizers import HeuristicTokenizer

TOK = HeuristicTokenizer()


@pytest.mark.parametrize(
    "a,b,em",
    [
        ("The Eiffel Tower!", "eiffel tower", True),
        ("an apple", "A  Apple.", True),
        ("Paris, France", "Paris", False),
        ("", "", True),
    ],
)
def test_exact_match(a, b, em):
    assert scoring.exact_match(a, b) is em


def test_token_f1_and_best_of_several():
    assert scoring.token_f1("new york city", "new york") == pytest.approx(0.8)
    assert scoring.token_f1("a b", "c d") == 0.0
    assert scoring.token_f1("", "x") == 0.0 and scoring.token_f1("", "") == 1.0
    assert scoring.best_f1("blue whale", ["whale", "blue whale"]) == 1.0
    assert scoring.best_f1("x", []) == 0.0


def test_contains():
    assert scoring.contains("It was 1,577 million dollars", "1,577 million")
    assert not scoring.contains("anything", "")


ATOMS = [
    scoring.Atom("deployment region", "eu-west-1", 3),
    scoring.Atom("budget", "42 thousand", 1),
    scoring.Atom("owner", "Priya", 2),
]


def test_atom_status_distinguishes_kept_mutated_omitted():
    text = "The deployment region was moved. Budget approved at 50 thousand."
    assert (
        scoring.status(ATOMS[0], text) == scoring.MUTATED
    )  # topic, wrong value
    assert scoring.status(ATOMS[1], text) == scoring.MUTATED
    assert scoring.status(ATOMS[2], text) == scoring.OMITTED
    assert (
        scoring.status(ATOMS[0], "region eu-west-1 deployment region eu-west-1")
        == scoring.KEPT
    )


def test_recall_family_and_density():
    text = "Deployment region: eu-west-1. Owner Priya."
    assert scoring.critical_atom_recall(ATOMS, text) == pytest.approx(2 / 3)
    assert scoring.weighted_atom_recall(ATOMS, text) == pytest.approx(5 / 6)
    assert scoring.commitment_density(ATOMS, text, TOK) > 0
    assert scoring.taxonomy(ATOMS, text) == {
        "kept": 2,
        "mutated": 0,
        "omitted": 1,
    }
    assert scoring.critical_atom_recall([], "x") == 1.0
    assert scoring.weighted_atom_recall([scoring.Atom("k", "v", 0)], "x") == 1.0
