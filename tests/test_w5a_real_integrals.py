"""``uc2-real-integrals-by-residues``: answers grade right, distractors wrong, each integral is recomputed by quadrature."""

from __future__ import annotations

import re

import mpmath
import pytest
import sympy

from mathgen import make

from w5a_support import (
    ORACLE_SEEDS,
    answer_value,
    check_follow_through,
    check_grading,
    check_text,
    close,
    coverage,
    numeric,
    parse,
    questions,
    tier_topic,
)

NAME = "real_integrals"
ALL = ("easy", "medium", "hard", "exam")
T = sympy.Symbol("t")
X = sympy.Symbol("x")


@pytest.mark.parametrize("tier", ALL)
def test_answers_and_distractors_grade(tier):
    for q in questions(NAME, tier):
        check_grading(q)
        check_text(q)


@pytest.mark.parametrize("tier", ALL)
def test_variety_and_determinism(tier):
    qs = questions(NAME, tier, range(1, 25))
    assert len({q.text for q in qs}) >= 12
    again = make(tier_topic(NAME, tier), seed=7, difficulty=tier)
    assert again.to_dict() == make(tier_topic(NAME, tier), seed=7, difficulty=tier).to_dict()


def test_exam_chain_shape():
    for q in questions(NAME, "exam"):
        assert 3 <= len(q.parts) <= 4
        assert [p.label for p in q.parts] == list("abcd")[: len(q.parts)]
        assert any(p.follow_through for p in q.parts)
        check_follow_through(q)


def test_tiers_are_distinct():
    shapes = {t: {q.text for q in questions(NAME, t, range(1, 13))} for t in ("easy", "medium", "hard")}
    assert shapes["easy"] != shapes["medium"] != shapes["hard"]
    assert not (shapes["easy"] & shapes["hard"])


def test_single_part_tiers_have_no_parts():
    for t in ("easy", "medium", "hard"):
        assert all(not q.parts for q in questions(NAME, t, range(1, 6)))


# -- the oracle: quadrature of the integrand in the stem -----------------------------------------------------------------


def _angular(text: str) -> sympy.Expr:
    text = text.replace("θ", "t")
    pass
    text = re.sub(r"(cos|sin) t\b", r"\1(t)", text)
    return parse(text)


def _quad_line(expr, lo, hi) -> float:
    fn = sympy.lambdify(X, expr, modules="mpmath")
    mpmath.mp.dps = 30
    return float(mpmath.quad(fn, [lo, 0, hi] if lo == -mpmath.inf else [lo, 1, hi]))


def _quad_angle(expr) -> float:
    fn = sympy.lambdify(T, expr, modules="mpmath")
    mpmath.mp.dps = 30
    return float(mpmath.quad(fn, [0, mpmath.pi / 2, mpmath.pi, 3 * mpmath.pi / 2, 2 * mpmath.pi]))


def _integral_in(text: str) -> float | None:
    m = re.search(r"∫_\(-∞\)\^\(∞\) (.+?) dx", text)
    if m:
        return _quad_line(parse(m.group(1)).subs({sympy.Symbol("x"): X}), -mpmath.inf, mpmath.inf)
    m = re.search(r"∫_0\^\(2π\) (.+?) dθ", text)
    if m:
        return _quad_angle(_angular(m.group(1)))
    return None


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_integral_is_recomputed_by_quadrature(tier):
    checked = 0
    qs = questions(NAME, tier, ORACLE_SEEDS)
    for q in qs:
        want = _integral_in(q.text)
        assert want is not None, q.text
        assert close(answer_value(q), want, 1e-8), (q.text, q.answer_text, want)
        checked += 1
    coverage(checked, len(qs), 1.0)


def test_exam_parts_are_recomputed_by_quadrature():
    for q in questions(NAME, "exam", ORACLE_SEEDS):
        values = [numeric(parse(p.answer_text)) for p in q.parts]
        whole = re.search(r"Let ([IJ]) = (∫.+?) (?:dx|dθ)", q.text)
        assert whole, q.text
        if whole.group(1) == "I":
            f = parse(re.search(r"∫_\(-∞\)\^\(∞\) (.+?) dx", q.text).group(1))
            total = _quad_line(f, -mpmath.inf, mpmath.inf)
            assert close(values[2], total, 1e-8), q.text
            assert close(values[3], _quad_line(f, 0, mpmath.inf), 1e-8), q.text
        else:
            total = _integral_in(q.text)
            assert close(values[2], total, 1e-8), q.text
            squared = re.findall(r"∫_0\^\(2π\) dθ/(\(\d+ [-+] \d+cos θ\))²", q.parts[3].prompt)
            assert squared, q.parts[3].prompt
            g = _angular(squared[-1]) ** -2
            assert close(values[3], _quad_angle(g), 1e-8), q.text
