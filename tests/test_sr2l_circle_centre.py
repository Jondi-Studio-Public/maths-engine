"""locus-circle-centre-radius: the centre of a circle given as a modulus relation, Cartesian equation or diameter."""

from __future__ import annotations

import re

import pytest
import sympy
from tests.sr2l_support import (
    check_answer_space,
    check_clean,
    check_graded_exam,
    check_graded_single,
    check_structure,
    check_tiers_differ,
    check_wired,
    circle_of,
    expr,
    pair,
    questions,
    rational,
    relation_in,
    relation_polynomial,
    stem_of,
)

SKILL = "locus-circle-centre-radius"
TOPICS = {
    "easy": "sr2l_circle_centre",
    "medium": "sr2l_circle_centre",
    "hard": "sr2l_circle_centre",
    "exam": "sr2l_circle_centre_exam",
}


def _centre_from_stem(q):
    """The centre read from the rendered stem: SymPy expands the printed relation, or takes the diameter's midpoint."""
    if "diameter" in stem_of(q):
        a, b = re.search(r"representing (.+?) and (.+?) are the ends", stem_of(q)).groups()
        mid = (expr(a) + expr(b)) / 2
        return sympy.re(mid), sympy.im(mid), None
    centre, r2 = circle_of(relation_polynomial(relation_in(stem_of(q))))
    if "is defined by" in q.text:  # the image of the circle under w = az + b is centred at the image of the centre
        z0 = centre[0] + sympy.I * centre[1]
        w = expr(re.search(r"w = (.+?), so", q.text).group(1), z=z0)
        return sympy.re(sympy.expand(w)), sympy.im(sympy.expand(w)), None
    return centre[0], centre[1], r2


def test_wired():
    check_wired(SKILL, TOPICS)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_centre_matches_an_independent_expansion(tier):
    for q in questions("sr2l_circle_centre", tier, 80):
        cx, cy, _ = _centre_from_stem(q)
        assert pair(q.answer_text) == (cx, cy), (stem_of(q), q.answer_text)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_graded(tier):
    check_graded_single(SKILL, "sr2l_circle_centre", tier)


def test_tiers_differ():
    check_tiers_differ(SKILL, TOPICS)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    check_answer_space("sr2l_circle_centre", tier)


def test_every_hard_form_appears():
    texts = " ".join(stem_of(q) for q in questions("sr2l_circle_centre", "hard", 80))
    assert all(form in texts for form in ("diameter", "x²", "z z̄", "is defined by"))


def test_exam_matches_an_independent_expansion():
    for q in questions("sr2l_circle_centre_exam", "exam", 60):
        (cx, cy), r2 = circle_of(relation_polynomial(relation_in(stem_of(q))))
        pa, pb, pc, pd = q.parts
        assert pair(pa.answer_text) == (cx, cy), stem_of(q)
        assert rational(pb.answer_text) ** 2 == r2, stem_of(q)
        px, py = (expr(m) for m in [re.search(r"The point P represents (.+?)\.", stem_of(q)).group(1)] * 2)
        point = sympy.re(px), sympy.im(px)
        dist = sympy.sqrt((point[0] - cx) ** 2 + (point[1] - cy) ** 2)
        assert abs(float(rational(pc.answer_text)) - float(dist)) < 0.0051, stem_of(q)
        assert abs(float(rational(pd.answer_text)) - abs(float(dist) - float(sympy.sqrt(r2)))) < 0.0101, stem_of(q)


def test_exam_structure_and_grading():
    check_structure(
        "sr2l_circle_centre_exam", ["pair", "number", "number", "number"], [2, 1, 2, 2], [False, False, False, True]
    )
    check_graded_exam(SKILL, "sr2l_circle_centre_exam")


def test_exam_follows_through_from_a_wrong_distance():
    from mathgen.engine import grade
    from tests.sr2l_support import skill

    sk, used = skill(SKILL), 0
    for q in questions("sr2l_circle_centre_exam", "exam", 40):
        pa, pb, pc, pd = q.parts
        wrong = pc.distractors[0]
        replies = {
            pa.label: pa.answer_text,
            pb.label: pb.answer_text,
            "c": wrong,
            "d": f"{float(wrong) - float(pb.answer_text):.2f}",
        }
        result = grade(sk, q, replies)
        assert result.parts["c"].marks == 0
        assert result.parts["d"].marks == result.parts["d"].max_marks, (q.text, replies)
        used += 1
    assert used


def test_clean_and_deterministic():
    check_clean("sr2l_circle_centre", ["easy", "medium", "hard"])
    check_clean("sr2l_circle_centre_exam", ["exam"])
