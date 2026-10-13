"""Exam chain for `senior-use-numerical-integration-using-technology-f-left-t`: interval, tail and conditional
probabilities of an exponential density."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

import pytest
from mpmath import mp, mpf, quad, exp as mexp

from mathgen.engine import grade
from tests import exd_x_r2j_kit as kit

SKILL = "senior-use-numerical-integration-using-technology-f-left-t"
TOPIC = "exd_x_r2j_exponential_pdf_chain"
SEEDS = range(300)


@pytest.fixture(scope="module")
def questions():
    return [kit.exam_q(SKILL, s) for s in SEEDS]


def _round(x, places: int) -> str:
    return str(Decimal(str(mp.nstr(x, 30))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def oracle(q) -> dict[str, str]:
    """Re-derive the four answers from the rendered prompts, by quadrature of the density."""
    mp.dps = 30
    lam = mpf(re.search(r"f\(t\) = ([\d.]+)e\^", q.stem).group(1))
    b = mpf(re.search(r"P\(T ≤ ([\d.]+)\)", q.parts[0].prompt).group(1))
    lo, hi = re.search(r"P\(([\d.]+) < T ≤ ([\d.]+)\)", q.parts[1].prompt).groups()
    assert mpf(lo) == b

    def f(t):
        return lam * mexp(-lam * t)

    a = quad(f, [0, b])
    mid = quad(f, [b, mpf(hi)])
    tail = quad(f, [mpf(hi), mp.inf])
    cond = mid / quad(f, [b, mp.inf])
    return {"a": _round(a, 4), "b": _round(mid, 4), "c": _round(tail, 4), "d": _round(cond, 3)}


def test_wiring_shapes_defects_determinism():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert [p.marks for p in q.parts] == [2, 2, 1, 2]


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == kit.canon(q), q.text


def test_answer_space():
    kit.check_answer_space(SKILL)


def test_grading():
    kit.check_grading(SKILL)


def test_follow_through_from_wrong_cumulative_probability(questions):
    for q in questions[:100]:
        c = kit.canon(q)
        a_wrong = f"{float(c['a']) - 0.0123:.4f}"
        c_wrong = f"{1 - float(a_wrong) - float(c['b']):.4f}"
        d_wrong = f"{float(c['b']) / (1 - float(a_wrong)):.3f}"
        result = grade(kit.skill(SKILL), q, {"a": a_wrong, "b": c["b"], "c": c_wrong, "d": d_wrong})
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == q.parts[1].marks
        for label in "cd":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label)
            assert result.parts[label].follow_through is True


def test_wrong_interval_carries_into_conditional(questions):
    for q in questions[:100]:
        c = kit.canon(q)
        b_wrong = f"{float(c['b']) + 0.0211:.4f}"
        d_wrong = f"{float(b_wrong) / (1 - float(c['a'])):.3f}"
        result = grade(kit.skill(SKILL), q, {**c, "b": b_wrong, "d": d_wrong})
        assert result.parts["b"].marks == 0
        assert result.parts["d"].marks == q.parts[3].marks
