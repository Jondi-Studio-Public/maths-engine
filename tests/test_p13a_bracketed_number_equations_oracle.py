"""Independent oracle for f10-bracketed-number-equations. Deliberately does NOT import anything from
``mathgen.generators.p13a_bracketed_number_equations_build`` -- it parses
``question.text`` with its own regexes and recomputes ``?`` from the equation
shape alone, using a fresh ``sympy`` symbolic solve, then compares against
``question.answer_text``.
"""

import re
from types import SimpleNamespace

import pytest
from sympy import Eq, Symbol, solve

from mathgen import make

TOPIC = "p13a_bracketed_number"
SEEDS = range(60)

_QSYM = Symbol("blank", real=True)

# Each pattern matches the literal equation substring embedded in whichever
# of the three wordings the generator drew (`re.search`, not `re.match`, so
# it is insensitive to which sentence wraps the equation).
_EASY_RE = re.compile(r"(\d+) × \(\? \+ (\d+)\) = (-?\d+)")
_MEDIUM_RE = re.compile(r"(\d+) × \(\? - (\d+)\) \+ (\d+) = (-?\d+)")
_HARD_RE = re.compile(r"(-?\d+) - (\d+) × \(\? \+ (\d+)\) = (-?\d+)")


def _oracle_easy(text: str) -> int:
    m = _EASY_RE.search(text)
    assert m, f"could not parse easy-shape equation from: {text!r}"
    a, b, c = (int(x) for x in m.groups())
    roots = solve(Eq(a * (_QSYM + b), c), _QSYM)
    assert len(roots) == 1
    return int(roots[0])


def _oracle_medium(text: str) -> int:
    m = _MEDIUM_RE.search(text)
    assert m, f"could not parse medium-shape equation from: {text!r}"
    a, b, d, c = (int(x) for x in m.groups())
    roots = solve(Eq(a * (_QSYM - b) + d, c), _QSYM)
    assert len(roots) == 1
    return int(roots[0])


def _oracle_hard(text: str) -> int:
    m = _HARD_RE.search(text)
    assert m, f"could not parse hard-shape equation from: {text!r}"
    d, a, b, e = (int(x) for x in m.groups())
    roots = solve(Eq(d - a * (_QSYM + b), e), _QSYM)
    assert len(roots) == 1
    return int(roots[0])


_ORACLES = {"easy": _oracle_easy, "medium": _oracle_medium, "hard": _oracle_hard}


def _challenge(oracle, question) -> bool:
    """The ONE comparison both tests below share: does the oracle's own
    re-derivation from ``question.text`` disagree with the shipped
    ``question.answer_text``? Returns True on a challenge (mismatch)."""
    return oracle(question.text) != int(question.answer_text)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_oracle_agrees_with_the_shipped_answer(difficulty):
    oracle = _ORACLES[difficulty]
    checked = 0
    challenged = []
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        checked += 1
        if _challenge(oracle, q):
            challenged.append((seed, q.text, q.answer_text, oracle(q.text)))
    assert checked == len(SEEDS)
    assert not challenged, (
        f"{difficulty}: {len(challenged)}/{checked} questions disagree with the independent oracle: {challenged[:5]}"
    )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_oracle_fires_on_a_corrupted_answer(difficulty):
    """Exercise the oracle's sensitivity through the exact comparison
    ``test_oracle_agrees_with_the_shipped_answer`` uses (``_challenge``), on a
    real, un-modified question TEXT paired with a hand-corrupted
    ``answer_text`` -- the shape of the real defect this instrument exists to
    catch (a generator whose formatter disagrees with its own construction).
    ``SimpleNamespace`` stands in for ``Question``: only ``.text``/
    ``.answer_text`` are read by ``_challenge``."""
    oracle = _ORACLES[difficulty]
    real = make(TOPIC, seed=0, difficulty=difficulty)
    correct = SimpleNamespace(text=real.text, answer_text=real.answer_text)
    corrupted = SimpleNamespace(text=real.text, answer_text=str(int(real.answer_text) + 7))
    assert not _challenge(oracle, correct), "the real, uncorrupted answer must not be challenged"
    assert _challenge(oracle, corrupted), "a hand-corrupted answer_text must be challenged"


def test_oracle_disagreement_would_be_visible_on_a_deliberately_broken_question():
    """Prove the oracle actually discriminates: build a text string with a
    swapped constant (as a real generator regression might ship) and confirm
    the oracle computes a DIFFERENT root than the original, undamaged
    question -- i.e. the oracle is sensitive to the defect it exists to
    catch, not a tautology that always agrees."""
    q = make(TOPIC, seed=1, difficulty="easy")
    original = _oracle_easy(q.text)
    m = _EASY_RE.search(q.text)
    a, b, c = (int(x) for x in m.groups())
    broken_text = q.text.replace(f"= {c}", f"= {c + a * 3}", 1)
    assert _oracle_easy(broken_text) != original
