"""locus-apollonius-and-intersection: |z - a| = k|z - b| is a circle; its centre, a point on it, and where a line meets it."""

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
    complex_parts,
    number_list,
    pair,
    questions,
    rational,
    relation_in,
    relation_polynomial,
    stem_of,
    x,
    y,
)

SKILL = "locus-apollonius-and-intersection"
TOPICS = {
    "easy": "sr2l_apollonius",
    "medium": "sr2l_apollonius",
    "hard": "sr2l_apollonius",
    "exam": "sr2l_apollonius_exam",
}


def _circle(q):
    return circle_of(relation_polynomial(relation_in(stem_of(q))))


def test_wired():
    check_wired(SKILL, TOPICS)


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_centre_matches_an_independent_expansion(tier):
    for q in questions("sr2l_apollonius", tier, 80):
        centre, r2 = _circle(q)
        assert pair(q.answer_text) == centre, (stem_of(q), q.answer_text)
        assert r2 > 0


def test_the_ratio_is_never_one_and_the_locus_is_a_circle():
    """k = 1 would give a line; every question has a squared coefficient other than 1."""
    for tier in ("easy", "medium", "hard"):
        for q in questions("sr2l_apollonius", tier, 60):
            poly = relation_polynomial(relation_in(stem_of(q)))
            assert poly.coeff(x, 2) != 0 and poly.coeff(x, 2) == poly.coeff(y, 2)


def test_hard_point_is_the_only_one_on_the_segment():
    """Intersect the printed circle with the segment between the two given points: exactly one point, the key."""
    t = sympy.Symbol("t")
    for q in questions("sr2l_apollonius", "hard", 80):
        text = stem_of(q)
        poly = relation_polynomial(relation_in(text))
        a_text, b_text = re.search(r"joining (.+?) and (.+?) at exactly", text).groups()
        a, b = complex_parts(a_text), complex_parts(b_text)
        on_segment = poly.subs({x: b[0] + t * (a[0] - b[0]), y: b[1] + t * (a[1] - b[1])})
        hits = [r for r in sympy.solve(on_segment, t) if r.is_real and 0 <= r <= 1]
        assert len(hits) == 1, text
        point = (b[0] + hits[0] * (a[0] - b[0]), b[1] + hits[0] * (a[1] - b[1]))
        assert pair(q.answer_text) == point, (text, q.answer_text)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_graded(tier):
    check_graded_single(SKILL, "sr2l_apollonius", tier)


def test_tiers_differ():
    check_tiers_differ(SKILL, TOPICS)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    check_answer_space("sr2l_apollonius", tier, minimum=10)


def test_medium_uses_all_three_written_forms():
    texts = [relation_in(stem_of(q)) for q in questions("sr2l_apollonius", "medium", 80)]
    assert (
        any("²" in t for t in texts)
        and any(re.match(r"\d\|", t) for t in texts)
        and any(re.search(r"= \d\|", t) for t in texts)
    )


def test_exam_matches_an_independent_expansion():
    for q in questions("sr2l_apollonius_exam", "exam", 60):
        centre, r2 = _circle(q)
        pa, pb, pc, pd = q.parts
        assert pair(pa.answer_text) == centre, stem_of(q)
        assert rational(pb.answer_text) ** 2 == r2, stem_of(q)
        k = int(re.search(r"Re\(z\) = (-?\d+)", pc.prompt).group(1))
        ys = sorted(sympy.solve((k - centre[0]) ** 2 + (y - centre[1]) ** 2 - r2, y))
        assert number_list(pc.answer_text) == ys, stem_of(q)
        assert rational(pd.answer_text) == ys[1] - ys[0], stem_of(q)


def test_exam_structure_and_grading():
    check_structure(
        "sr2l_apollonius_exam", ["pair", "number", "list", "number"], [2, 2, 2, 2], [False, False, False, True]
    )
    check_graded_exam(SKILL, "sr2l_apollonius_exam")


def test_exam_chord_follows_through_from_a_wrong_radius():
    from mathgen.engine import grade

    from tests.sr2l_support import skill

    sk, used = skill(SKILL), 0
    for q in questions("sr2l_apollonius_exam", "exam", 40):
        pa, pb, pc, pd = q.parts
        k = int(re.search(r"Re\(z\) = (-?\d+)", pc.prompt).group(1))
        centre = pair(pa.answer_text)
        for bad in pb.distractors:
            inside = rational(bad) ** 2 - (k - centre[0]) ** 2
            if inside <= 0:
                continue
            chord = 2 * sympy.sqrt(inside)
            reply = {"a": pa.answer_text, "b": bad, "c": pc.answer_text, "d": str(round(float(chord)))}
            result = grade(sk, q, reply)
            assert result.parts["b"].marks == 0
            if round(float(chord)) != int(pd.answer_text):
                assert result.parts["d"].marks == result.parts["d"].max_marks, (q.text, reply)
                used += 1
    assert used >= 3


def test_clean_and_deterministic():
    check_clean("sr2l_apollonius", ["easy", "medium", "hard"])
    check_clean("sr2l_apollonius_exam", ["exam"])
