"""Independent oracle for f10-expanded-powers-ten.
Deliberately does NOT import anything from
``mathgen.generators.p19a_expanded_powers_ten_build`` -- it parses every
`c x 10^k` / `c x 10` / bare term out of ``question.text`` with its own
regex and sums them, then compares against ``question.answer``."""

import os
import re
from types import SimpleNamespace

import pytest

from mathgen import make

TOPIC = "p19a_expanded_powers_ten"
N_SEEDS = int(os.environ.get("P19A_ORACLE_SEEDS", 250))
TIERS = ["easy", "medium", "hard", "exam"]

# One term: an integer coefficient, optionally followed by "x 10" or
# "x 10^k" (the multiplication sign is the real unicode "x", U+00D7 -- the
# same character `mathgen.core.latex_prompt` and the generator itself use).
_TERM_RE = re.compile(r"(\d+)(?:\s*×\s*10(?:\^(\d+))?)?")


def _extract_expr_terms(text: str) -> list[tuple[int, int]]:
    """Every `(coefficient, power)` term appearing anywhere in `text`,
    in the order they appear -- a bare number with no `x 10...` suffix is
    power 0; `x 10` with no `^k` is power 1."""
    terms = []
    for m in _TERM_RE.finditer(text):
        coeff = int(m.group(1))
        if m.group(0).strip() == m.group(1):
            power = 0
        elif m.group(2) is not None:
            power = int(m.group(2))
        else:
            power = 1
        terms.append((coeff, power))
    return terms


def _oracle(text: str) -> int:
    """Isolate the expanded-form expression from the surrounding sentence
    (it always runs from the first digit up to the LAST digit that precedes
    a `.`/`?`/`)` word boundary followed by prose) and sum every term found
    in it. The expression is exactly the run of `digit/×/^/+/space` tokens;
    everything else in the sentence is prose."""
    m = re.search(r"[\d×^+\s]+\d", text)
    assert m, text
    expr = m.group(0)
    terms = _extract_expr_terms(expr)
    assert terms, text
    return sum(c * 10**p for c, p in terms)


def _challenge(question) -> bool:
    return _oracle(question.text) != question.answer


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_agrees_with_the_shipped_answer(difficulty):
    checked = 0
    challenged = []
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        checked += 1
        if _challenge(q):
            challenged.append((seed, q.text, q.answer, _oracle(q.text)))
    assert checked == N_SEEDS
    assert not challenged, (
        f"{difficulty}: {len(challenged)}/{checked} questions disagree with the independent oracle: {challenged[:5]}"
    )


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_fires_on_a_corrupted_answer(difficulty):
    """Red-first proof: a real, un-modified question TEXT paired with a
    hand-corrupted answer must be challenged; the real, uncorrupted answer
    must not be."""
    real = make(TOPIC, seed=0, difficulty=difficulty)
    correct = SimpleNamespace(text=real.text, answer=real.answer)
    corrupted = SimpleNamespace(text=real.text, answer=real.answer + 1000)
    assert not _challenge(correct), "the real, uncorrupted answer must not be challenged"
    assert _challenge(corrupted), "a hand-corrupted answer must be challenged"


def test_oracle_disagreement_would_be_visible_on_a_deliberately_broken_question():
    q = make(TOPIC, seed=1, difficulty="medium")
    original = _oracle(q.text)
    m = re.search(r"(\d+)(?=\s*×\s*10)", q.text)
    assert m, q.text
    broken_text = q.text.replace(m.group(0), str(int(m.group(0)) + 5), 1)
    assert _oracle(broken_text) != original
