"""``uc2-laurent-series-principal-part``: answers grade right, distractors wrong, coefficients are recomputed from the stem."""

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

NAME = "laurent"
ALL = ("easy", "medium", "hard", "exam")
_SUPERSCRIPTS = {"⁰": 0, "¹": 1, "²": 2, "³": 3, "⁴": 4, "⁵": 5, "⁶": 6, "⁷": 7, "⁸": 8, "⁹": 9}


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


# -- the oracle ----------------------------------------------------------------------------------------------------------


def _centre(text: str) -> complex:
    m = re.search(r"(?:about|its pole) z = (.+?)(?= valid| of |,|\. |\.$|$)", text)
    return complex(sympy.N(parse(m.group(1))))


def _radius(text: str, centre: complex) -> float:
    """A circle in the annulus the stem states, or a small one when it states none."""
    m = re.search(r"valid for (.+?) < \|[^|]+\| < (\S+?)(?:[?.,]|$| )", text)
    if not m:
        return 0.3
    inner = float(sympy.N(parse(m.group(1))))
    outer_text = m.group(2)
    outer = 3.0 + inner if "∞" in outer_text else float(sympy.N(parse(outer_text)))
    if "∞" in outer_text:
        return max(1.0, inner + 1.0)
    return (inner + outer) / 2


def _coefficient(f, centre: complex, radius: float, n: int, points: int = 2048) -> complex:
    fn = sympy.lambdify(Z, f, modules="cmath")
    total = 0j
    for k in range(points):
        w = radius * cmath.exp(2j * math.pi * k / points)
        total += fn(centre + w) / w**n
    return total / points


def _index(term: str) -> int:
    m = re.search(r"\^\((-?\d+)\)$", term)
    if m:
        return int(m.group(1))
    m = re.search(r"\^(\d+)$", term)
    if m:
        return int(m.group(1))
    if term and term[-1] in _SUPERSCRIPTS:
        return _SUPERSCRIPTS[term[-1]]
    return 1


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_coefficient_or_principal_part_is_recomputed_from_the_stem(tier):
    checked = 0
    qs = questions(NAME, tier, ORACLE_SEEDS)
    for q in qs:
        f = parse(balanced_after(q.text, "f(z) = "))
        centre = _centre(q.text)
        r = _radius(q.text, centre)
        m = re.search(r"coefficient of (.+?)(?= \(the constant term\)| in the| about|\?|\. |\.$)", q.text)
        if m:
            n = _index(m.group(1))
            want = _coefficient(f, centre, r, n)
            assert close(answer_value(q), want, 1e-6), (q.text, q.answer_text, want)
            checked += 1
        elif "principal part" in q.text or "negative powers" in q.text:
            z0 = centre + 0.7 + 0.4j
            want = sum(_coefficient(f, centre, r, n) * (z0 - centre) ** n for n in range(-6, 0))
            assert close(answer_value(q, z=z0), want, 1e-6), (q.text, q.answer_text, want)
            checked += 1
    coverage(checked, len(qs), 0.9)


def test_exam_parts_are_recomputed_from_the_stem():
    for q in questions(NAME, "exam", ORACLE_SEEDS):
        f = parse(balanced_after(q.text, "f(z) = "))
        centre = complex(sympy.N(parse(re.search(r"about z = (.+?)\.", q.text).group(1))))
        c2 = _coefficient(f, centre, 0.3, -2)
        c1 = _coefficient(f, centre, 0.3, -1)
        values = [numeric(parse(p.answer_text)) for p in q.parts]
        assert close(values[0], c2, 1e-6) and close(values[1], c1, 1e-6), q.text
        # (c) is the principal part about the first pole, evaluated one unit away
        assert close(values[2], c2 + c1, 1e-6), q.text
        num, den = sympy.fraction(sympy.together(f))
        others = [
            complex(sympy.N(p)) for p in sympy.roots(sympy.Poly(den, Z)) if abs(complex(sympy.N(p)) - centre) > 1e-9
        ]
        assert close(values[3].real, min(abs(p - centre) for p in others)), q.text
