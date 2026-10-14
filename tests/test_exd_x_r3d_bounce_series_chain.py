"""Exam chain of the bouncing-ball skill: ratio, drop height, distance to the k-th landing, distance to rest."""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pytest

from tests import exd_x_r3d_support as S

TOPIC = "exd_x_r3d_bounce_series_chain"
SKILL = S.load_skill("senior-use-geometric-sequences-to-model-and-analyse-numerically")


def _two(x: Fraction) -> str:
    return str(
        Decimal(x.numerator).__truediv__(Decimal(x.denominator)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    )


def _read(q):
    h1, h2 = (Fraction(v) for v in re.findall(r"(\d+\.\d\d)", q.stem)[:2])
    k = int(re.search(r"(\d)th", q.parts[2].prompt).group(1))
    return h1, h2, k


def oracle(q):
    h1, h2, k = _read(q)
    r = h2 / h1
    drop = h1 / r
    # Term by term: the drop, then up and down for each of the k - 1 rebounds.
    heights = [h1 * r ** (i - 1) for i in range(1, k)]
    total = drop + 2 * sum(heights)
    # To rest: the same series without end, summed from the first rebound by the geometric formula.
    forever = drop + 2 * h1 / (1 - r)
    return {"a": _two(r), "b": _two(drop), "c": _two(total), "d": _two(forever)}


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_stated_heights_are_two_decimal_places(questions):
    for q in questions:
        h1, h2, k = _read(q)
        assert (h1 * 100).denominator == 1 and (h2 * 100).denominator == 1 and 0 < h2 < h1
        assert 4 <= k <= 6


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_ratio(questions):
    def chain(q):
        h1, h2, k = _read(q)
        r_text = next((d for d in q.parts[0].distractors if 0 < Fraction(d) < 1), None)
        if r_text is None:
            return None
        r = Fraction(r_text)
        drop = S.round_half_up(h1 / r, 2)
        total = S.round_half_up(drop + 2 * sum(drop * r**i for i in range(1, k)), 2)
        forever = S.round_half_up(drop * (1 + r) / (1 - r), 2)
        replies = {"a": r_text, "b": f"{float(drop):.2f}", "c": f"{float(total):.2f}", "d": f"{float(forever):.2f}"}
        if len({replies["b"], replies["c"], replies["d"], r_text}) < 4:
            return None
        return replies, {"b", "c", "d"}

    S.check_follow_through(SKILL, questions, chain, n=300, minimum=40)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "0.99", "b": "99.99", "c": "98.98", "d": "97.97"})


def test_deterministic():
    S.check_deterministic(TOPIC)
