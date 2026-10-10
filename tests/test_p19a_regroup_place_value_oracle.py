"""Independent oracle for f10-regroup-zero-place-value. Deliberately does NOT import anything from
``mathgen.generators.p19a_regroup_place_value_build`` -- it parses
``question.text`` for the stated hundreds/tens/ones counts by their PLACE
WORD (order-independent -- the hard tier sometimes lists them out of order)
and totals + redecomposes them from scratch, then compares against the
labelled triple ``question.answer`` actually returns."""

import os
import re
from types import SimpleNamespace

import pytest

from mathgen import make

TOPIC = "p19a_regroup_place_value"
N_SEEDS = int(os.environ.get("P19A_ORACLE_SEEDS", 250))
TIERS = ["easy", "medium", "hard"]


def _oracle(text: str) -> tuple[int, int, int]:
    hundreds_m = re.search(r"(\d+) hundreds?", text)
    tens_m = re.search(r"(\d+) tens?", text)
    ones_m = re.search(r"(\d+) ones?", text)
    assert tens_m and ones_m, text
    h = int(hundreds_m.group(1)) if hundreds_m else 0
    t = int(tens_m.group(1))
    o = int(ones_m.group(1))
    total = h * 100 + t * 10 + o
    return total // 100, (total // 10) % 10, total % 10


def _triple(question) -> tuple:
    return question.answer["hundreds"], question.answer["tens"], question.answer["ones"]


def _challenge(question) -> bool:
    return _oracle(question.text) != _triple(question)


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_agrees_with_the_shipped_answer(difficulty):
    checked = 0
    challenged = []
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        checked += 1
        if _challenge(q):
            challenged.append((seed, q.text, _triple(q), _oracle(q.text)))
    assert checked == N_SEEDS
    assert not challenged, (
        f"{difficulty}: {len(challenged)}/{checked} questions disagree with the independent oracle: {challenged[:5]}"
    )


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_fires_on_a_corrupted_answer(difficulty):
    """Red-first proof: a real, un-modified question TEXT paired with a
    hand-corrupted answer triple must be challenged; the real, uncorrupted
    pair must not be."""
    real = make(TOPIC, seed=0, difficulty=difficulty)
    H, T, O = _triple(real)
    correct = SimpleNamespace(text=real.text, answer={"hundreds": H, "tens": T, "ones": O})
    corrupted = SimpleNamespace(text=real.text, answer={"hundreds": H + 1, "tens": T, "ones": O})
    assert not _challenge(correct), "the real, uncorrupted answer must not be challenged"
    assert _challenge(corrupted), "a hand-corrupted answer must be challenged"


def test_oracle_disagreement_would_be_visible_on_a_deliberately_broken_question():
    q = make(TOPIC, seed=1, difficulty="hard")
    original = _oracle(q.text)
    m = re.search(r"(\d+) hundreds?", q.text)
    assert m, q.text
    broken_text = q.text.replace(m.group(0), f"{int(m.group(1)) + 3} hundreds", 1)
    assert _oracle(broken_text) != original
