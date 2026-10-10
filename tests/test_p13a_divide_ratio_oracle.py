"""Independent oracle for divide-quantity-in-ratio.
Deliberately does NOT import anything from
``mathgen.generators.p13a_divide_ratio_build`` -- it parses ``question.text``
with its own regexes and recomputes both shares from scratch using its own
``fractions.Fraction`` arithmetic, then compares against the PAIR
``question.answer`` actually returns.
"""

import math
import re
from fractions import Fraction
from types import SimpleNamespace

import pytest

from mathgen import make

TOPIC = "p13a_divide_ratio"
SEEDS = range(60)

_RATIO_RE = re.compile(r"ratio (\d+(?:\.\d+)?(?:/\d+)?):(\d+(?:\.\d+)?(?:/\d+)?)")
_LEADING_NUMBER_RE = re.compile(r"^\D*(\d+)")


def _to_fraction(token: str) -> Fraction:
    if "/" in token:
        num, den = token.split("/")
        return Fraction(int(num), int(den))
    return Fraction(token)


def _whole_number_ratio(p_frac: Fraction, q_frac: Fraction) -> tuple[int, int]:
    lcm = math.lcm(p_frac.denominator, q_frac.denominator)
    p_whole = p_frac * lcm
    q_whole = q_frac * lcm
    assert p_whole.denominator == 1 and q_whole.denominator == 1
    return int(p_whole), int(q_whole)


def _oracle_easy(text: str) -> tuple[int, int]:
    total = int(_LEADING_NUMBER_RE.search(text).group(1))
    p_tok, q_tok = _RATIO_RE.search(text).groups()
    p, q = int(p_tok), int(q_tok)
    share_a = total * p // (p + q)
    share_b = total * q // (p + q)
    return share_a, share_b


def _oracle_medium(text: str) -> tuple[int, int]:
    p_tok, q_tok = _RATIO_RE.search(text).groups()
    p, q = int(p_tok), int(q_tok)
    after_ratio = text[_RATIO_RE.search(text).end() :]
    diff = int(re.search(r"(\d+)", after_ratio).group(1))
    k = diff // abs(p - q)
    return k * p, k * q


def _oracle_hard(text: str) -> tuple[int, int]:
    p_tok, q_tok = _RATIO_RE.search(text).groups()
    p, q = int(p_tok), int(q_tok)
    after_ratio = text[_RATIO_RE.search(text).end() :]
    share_a = int(re.search(r"(\d+)", after_ratio).group(1))
    k = share_a // p
    return share_a, k * q


def _oracle_exam(text: str) -> tuple[int, int]:
    total = int(_LEADING_NUMBER_RE.search(text).group(1))
    p_tok, q_tok = _RATIO_RE.search(text).groups()
    p_frac, q_frac = _to_fraction(p_tok), _to_fraction(q_tok)
    p_whole, q_whole = _whole_number_ratio(p_frac, q_frac)
    share_a = total * p_whole // (p_whole + q_whole)
    share_b = total * q_whole // (p_whole + q_whole)
    return share_a, share_b


_ORACLES = {"easy": _oracle_easy, "medium": _oracle_medium, "hard": _oracle_hard, "exam": _oracle_exam}


def _pair(question) -> tuple:
    return tuple(question.answer.values())


def _challenge(oracle, question) -> bool:
    """The ONE comparison both tests below share: does the oracle's own
    re-derivation from ``question.text`` disagree with the shipped answer
    pair?"""
    return oracle(question.text) != _pair(question)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "exam"])
def test_oracle_agrees_with_the_shipped_answer(difficulty):
    oracle = _ORACLES[difficulty]
    checked = 0
    challenged = []
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        checked += 1
        if _challenge(oracle, q):
            challenged.append((seed, q.text, _pair(q), oracle(q.text)))
    assert checked == len(SEEDS)
    assert not challenged, (
        f"{difficulty}: {len(challenged)}/{checked} questions disagree with the independent oracle: {challenged[:5]}"
    )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "exam"])
def test_oracle_fires_on_a_corrupted_answer(difficulty):
    """Exercise the oracle's sensitivity through the exact comparison
    ``test_oracle_agrees_with_the_shipped_answer`` uses (``_challenge``), on a
    real, un-modified question TEXT paired with a hand-corrupted answer pair
    -- the shape of the real defect this instrument exists to catch (a
    generator whose formatter disagrees with its own construction).
    ``SimpleNamespace`` stands in for ``Question``: only ``.text``/``.answer``
    are read by ``_challenge``/``_pair``."""
    oracle = _ORACLES[difficulty]
    real = make(TOPIC, seed=0, difficulty=difficulty)
    share_a, share_b = _pair(real)
    correct = SimpleNamespace(text=real.text, answer={"share_p": share_a, "share_q": share_b})
    corrupted = SimpleNamespace(text=real.text, answer={"share_p": share_a + 7, "share_q": share_b})
    assert not _challenge(oracle, correct), "the real, uncorrupted answer must not be challenged"
    assert _challenge(oracle, corrupted), "a hand-corrupted answer must be challenged"


def test_oracle_disagreement_would_be_visible_on_a_deliberately_broken_question():
    """Prove the oracle actually discriminates: build a text string with a
    swapped total (as a real generator regression might ship) and confirm the
    oracle computes a DIFFERENT pair than the original, undamaged question --
    i.e. the oracle is sensitive to the defect it exists to catch, not a
    tautology that always agrees."""
    q = make(TOPIC, seed=1, difficulty="easy")
    original = _oracle_easy(q.text)
    total = int(_LEADING_NUMBER_RE.search(q.text).group(1))
    broken_text = q.text.replace(str(total), str(total + 300), 1)
    assert _oracle_easy(broken_text) != original
