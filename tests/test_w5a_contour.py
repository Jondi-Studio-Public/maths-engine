"""``uc2-contour-integrals-residue-theorem``: answers grade right, distractors wrong, the integral is recomputed from the stem."""

from __future__ import annotations

import cmath
import math
import re

import pytest
import sympy

from mathgen import make

from w5a_support import (
    ORACLE_SEEDS,
    Z,
    answer_value,
    balanced_after,
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

NAME = "contour"
ALL = ("easy", "medium", "hard", "exam")


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


def _circle(text: str):
    m = re.search(r"\|([^|]+)\| = (\d+(?:/\d+)?)", text)
    inner = parse(m.group(1))
    centre = -inner.subs(Z, 0)
    radius = sympy.Rational(m.group(2))
    return complex(sympy.N(centre)), float(radius)


def _numeric_integral(f, centre: complex, radius: float, n: int = 4000) -> complex:
    fn = sympy.lambdify(Z, f, modules=[{"conjugate": lambda w: w.conjugate()}, "cmath"])
    total = 0j
    for k in range(n):
        w = centre + radius * cmath.exp(2j * math.pi * k / n)
        total += fn(w) * 1j * (w - centre)
    return total * 2 * math.pi / n


def _poles_inside(f, centre: complex, radius: float):
    # the exponential factors have no zeros, so the poles are those of the rational part
    num, den = sympy.fraction(sympy.together(f.replace(sympy.exp, lambda _a: sympy.Integer(1))))
    roots = sympy.roots(sympy.Poly(den, Z))
    return [p for p in roots if abs(complex(sympy.N(p)) - centre) < radius]


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_integral_is_recomputed_from_the_stem(tier):
    checked = 0
    qs = questions(NAME, tier, ORACLE_SEEDS)
    for q in qs:
        f = parse(balanced_after(q.text, "f(z) = "))
        centre, radius = _circle(q.text)
        sign = -1 if ("clockwise" in q.text and "anticlockwise" not in q.text) else 1
        if f.has(sympy.conjugate):
            want = _numeric_integral(f, centre, radius)
        else:
            total = sum(sympy.residue(f, Z, p) for p in _poles_inside(f, centre, radius))
            want = numeric(2 * sympy.pi * sympy.I * total)
            assert close(want, _numeric_integral(f, centre, radius), 1e-5), q.text
        assert close(answer_value(q), sign * want, 1e-6), (q.text, q.answer_text, want)
        checked += 1
    coverage(checked, len(qs), 0.9)


def test_exam_parts_are_recomputed_from_the_stem():
    for q in questions(NAME, "exam", ORACLE_SEEDS):
        f = parse(balanced_after(q.text, "f(z) = "))
        centre, radius = _circle(q.text)
        inside = _poles_inside(f, centre, radius)
        assert round(answer_value_part(q.parts[0]).real) == len(inside), (q.text, q.parts[0].answer_text)
        for part in q.parts[1:3]:
            pole = parse(re.search(r"z = (.+?)\.", part.prompt).group(1))
            assert close(answer_value_part(part), numeric(sympy.residue(f, Z, pole))), (q.text, part.prompt)
        total = sum(sympy.residue(f, Z, p) for p in inside)
        assert close(answer_value_part(q.parts[3]), numeric(2 * sympy.pi * sympy.I * total)), q.text


def answer_value_part(part) -> complex:
    return numeric(parse(part.answer_text))
