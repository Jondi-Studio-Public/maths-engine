"""Exam chain of `ucv1-multivariate-regression`: b1, b2, b0 and a fitted value from centred summary sums."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from tests import exd_x_r3d_support as S

TOPIC = "exd_x_r3d_regression_summary_chain"
SKILL = S.load_skill("ucv1-multivariate-regression")


def _read(q):
    s = q.stem
    sums = {name: int(re.search(rf"{name} = (-?\d+)", s).group(1)) for name in ("S11", "S12", "S22", "S1y", "S2y")}
    means = re.search(r"means are (-?\d+) \(for x1\), (-?\d+) \(for x2\) and (-?\d+)", s) or re.search(
        r"mean of x1 is (-?\d+), the mean of x2 is (-?\d+) and the mean of y is (-?\d+)", s
    )
    m1, m2, my = (int(v) for v in means.groups())
    u, v = (int(t) for t in re.search(r"x1 = (\d+)[ ,]+(?:and )?x2 = (\d+)", q.parts[3].prompt).groups())
    return sums, m1, m2, my, u, v


def oracle(q):
    sums, m1, m2, my, u, v = _read(q)
    # Cramer's rule on the centred normal equations, with exact fractions.
    det = sums["S11"] * sums["S22"] - sums["S12"] ** 2
    b1 = Fraction(sums["S22"] * sums["S1y"] - sums["S12"] * sums["S2y"], det)
    b2 = Fraction(sums["S11"] * sums["S2y"] - sums["S12"] * sums["S1y"], det)
    b0 = my - b1 * m1 - b2 * m2
    fitted = b0 + b1 * u + b2 * v
    return {"a": str(b1), "b": str(b2), "c": str(b0), "d": str(fitted)}


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


def test_summary_is_realisable(questions):
    # The covariance block must be positive definite, as the sums of real data are.
    for q in questions:
        sums = _read(q)[0]
        assert sums["S11"] > 0 and sums["S22"] > 0
        assert sums["S11"] * sums["S22"] > sums["S12"] ** 2


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_b1(questions):
    def chain(q):
        sums, m1, m2, my, u, v = _read(q)
        b1 = int(q.parts[0].distractors[0])
        b2 = Fraction(sums["S2y"] - sums["S12"] * b1, sums["S22"])
        if b2.denominator != 1:
            b2 = Fraction(S.round_half_up(b2))
        b2 = int(b2)
        b0 = my - b1 * m1 - b2 * m2
        fitted = b0 + b1 * u + b2 * v
        if 0 in (b1, b2, b0) or len({b1, b2, b0, fitted}) < 4:
            return None
        return {"a": str(b1), "b": str(b2), "c": str(b0), "d": str(fitted)}, {"b", "c", "d"}

    S.check_follow_through(SKILL, questions, chain, n=300, minimum=25)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "99", "b": "98", "c": "97", "d": "96"})


def test_deterministic():
    S.check_deterministic(TOPIC)
