"""locus-perpendicular-bisector: |z - a| = |z - b| as the equation of a line."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

import pytest
import sympy

from tests.sr2l_support import (
    bisector_line,
    check_answer_space,
    check_clean,
    check_graded_exam,
    check_graded_single,
    check_structure,
    check_tiers_differ,
    check_wired,
    complex_parts,
    expr,
    line_form,
    normal_line,
    number_close,
    questions,
    relation_in,
    relation_polynomial,
    stem_of,
    x,
    y,
)

SKILL = "locus-perpendicular-bisector"
TOPICS = {"easy": "sr2l_bisector", "medium": "sr2l_bisector", "hard": "sr2l_bisector", "exam": "sr2l_bisector_exam"}


def _line_from_stem(q) -> tuple:
    """The locus line, re-derived from the rendered stem (a printed relation, two plotted points, or a quadratic's roots)."""
    text = stem_of(q)
    if q.figure and q.figure.get("points"):
        a, b = (tuple(sympy.Rational(round(v)) for v in pt["at"]) for pt in q.figure["points"][:2])
        return bisector_line(a, b)
    roots = re.search(r"roots of (.+?) = 0", text)
    if roots:
        z = sympy.Symbol("z")
        poly = expr(roots.group(1), z=z)
        a, b = (complex_parts(str(r)) for r in sympy.solve(poly, z))
        return bisector_line(a, b)
    return normal_line(relation_polynomial(relation_in(text)))


def test_wired():
    check_wired(SKILL, TOPICS)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_line_matches_an_independent_expansion(tier):
    for q in questions("sr2l_bisector", tier, 80):
        assert line_form(q.answer_text) == _line_from_stem(q), (stem_of(q), q.answer_text)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_graded(tier):
    check_graded_single(SKILL, "sr2l_bisector", tier)


def test_the_plotted_points_are_what_the_medium_tier_reads():
    for q in questions("sr2l_bisector", "medium", 30):
        assert q.figure and len(q.figure["points"]) >= 2
        assert {pt["label"] for pt in q.figure["points"][:2]} == {"A", "B"}


def test_every_hard_form_appears():
    texts = " ".join(stem_of(q) for q in questions("sr2l_bisector", "hard", 120))
    assert all(form in texts for form in ("roots of", "z̄", "|2z", "|3z"))


def test_tiers_differ():
    check_tiers_differ(SKILL, TOPICS)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    check_answer_space("sr2l_bisector", tier)


def test_exam_matches_an_independent_expansion():
    for q in questions("sr2l_bisector_exam", "exam", 60):
        a_, b_, c_, d_ = (p.answer_text for p in q.parts)
        line = normal_line(relation_polynomial(relation_in(stem_of(q))))
        assert line_form(a_) == line, stem_of(q)
        la, lb, lc = line
        x_int = -lc / la
        y_int = -lc / lb
        assert number_close(b_, x_int, 0.0051), stem_of(q)
        assert number_close(c_, abs(x_int * y_int) / 2, 0.0051), stem_of(q)
        assert number_close(d_, abs(lc) / sympy.sqrt(la**2 + lb**2), 0.0051), stem_of(q)
        assert la != 0 and lb != 0


def test_exam_structure_and_grading():
    check_structure(
        "sr2l_bisector_exam", ["pair", "number", "number", "number"], [2, 1, 2, 2], [False, False, True, True]
    )
    check_graded_exam(SKILL, "sr2l_bisector_exam")


def test_exam_follows_through_from_a_wrong_intercept():
    from mathgen.engine import grade

    from tests.sr2l_support import skill

    sk, used = skill(SKILL), 0
    for q in questions("sr2l_bisector_exam", "exam", 40):
        pa, pb, pc, pd = q.parts
        wrong = pb.distractors[0]
        _, lb, lc = line_form(pa.answer_text)
        area = abs(float(wrong)) * abs(float(lc / lb)) / 2
        result = grade(
            sk,
            q,
            {
                "a": pa.answer_text,
                "b": wrong,
                "c": str(Decimal(str(area)).quantize(Decimal("0.01"), ROUND_HALF_UP)),
                "d": pd.answer_text,
            },
        )
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, wrong, area)
        used += 1
    assert used


def test_clean_and_deterministic():
    check_clean("sr2l_bisector", ["easy", "medium", "hard"])
    check_clean("sr2l_bisector_exam", ["exam"])
