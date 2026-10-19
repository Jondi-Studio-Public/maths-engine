"""locus-circle-disc-pick-sketch: read a drawn circle, ring or set of points and pick the matching relation."""

from __future__ import annotations

import re

import pytest
import sympy

from mathgen import figures
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
    figure_circles,
    number_close,
    questions,
    relation_polynomial,
    stem_of,
)

SKILL = "locus-circle-disc-pick-sketch"
TOPICS = {
    "easy": "sr2l_circle_pick",
    "medium": "sr2l_circle_pick",
    "hard": "sr2l_circle_pick",
    "exam": "sr2l_circle_pick_exam",
}
_WORDS = {
    "including the inner circle but not the outer circle": ("≤", "<"),
    "including the outer circle but not the inner circle": ("<", "≤"),
    "including both circles": ("≤", "≤"),
    "not including either circle": ("<", "<"),
}


def _centre_of_modulus(inner: str):
    (cx, cy), _ = circle_of(relation_polynomial(f"|{inner}| = 0"))
    return cx, cy


def test_wired():
    check_wired(SKILL, TOPICS)


def test_easy_circle_is_the_drawn_circle():
    for q in questions("sr2l_circle_pick", "easy", 80):
        ((fx, fy, fr),) = figure_circles(q.figure)
        (cx, cy), r2 = circle_of(relation_polynomial(q.answer_text))
        assert (cx, cy, r2) == (fx, fy, fr**2), (q.text, q.answer_text, (fx, fy, fr))


def test_hard_ring_is_the_drawn_pair_of_circles():
    for q in questions("sr2l_circle_pick", "hard", 80):
        circles = figure_circles(q.figure)
        assert len(circles) == 2
        (c1x, c1y, ra), (c2x, c2y, rb) = circles
        assert (c1x, c1y) == (c2x, c2y)
        inner = re.search(r"\|([^|]+)\|", q.answer_text).group(1)
        assert _centre_of_modulus(inner) == (c1x, c1y), (q.answer_text, circles)
        nums = [sympy.Integer(n) for n in re.findall(r"(?<![\w+])\d+(?= [<≤]|$)|(?<=[<≤] )\d+", q.answer_text)]
        assert sorted(nums) == sorted([ra, rb]), (q.answer_text, circles)
        words = next(w for w in _WORDS if w in stem_of(q))
        ops = re.findall(r"[<≤]", q.answer_text)
        assert tuple(ops) == _WORDS[words], (stem_of(q), q.answer_text)


def test_medium_names_the_points_that_satisfy_the_inequality():
    seen = set()
    for q in questions("sr2l_circle_pick", "medium", 100):
        inner, op, radius = re.search(r"\|([^|]+)\|\s*(<|≤|>|≥)\s*(\d+)", stem_of(q)).groups()
        cx, cy = _centre_of_modulus(inner)
        pts = {pt["label"]: pt["at"] for pt in q.figure["points"]}
        assert set(pts) == set("ABCD")
        compare = {
            "<": lambda d, r: d < r,
            "≤": lambda d, r: d <= r,
            ">": lambda d, r: d > r,
            "≥": lambda d, r: d >= r,
        }[op]
        chosen = sorted(k for k, (px, py) in pts.items() if compare((px - cx) ** 2 + (py - cy) ** 2, int(radius) ** 2))
        if not chosen:
            expect = "None of the points"
        elif len(chosen) == 4:
            expect = "All four points"
        elif len(chosen) == 1:
            expect = f"{chosen[0]} only"
        else:
            expect = ", ".join(chosen[:-1]) + f" and {chosen[-1]} only"
        assert q.answer_text == expect, (stem_of(q), pts, q.answer_text)
        seen.add(len(chosen))
    assert len(seen) >= 3


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_graded(tier):
    check_graded_single(SKILL, "sr2l_circle_pick", tier)


def test_figure_reader_agrees_with_the_key():
    reader = figures.get_reader("sr2l_circle_pick")
    for tier in ("easy", "medium", "hard"):
        for q in questions("sr2l_circle_pick", tier, 40):
            assert reader(q) == q.answer_text


def test_tiers_differ():
    check_tiers_differ(SKILL, TOPICS)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    check_answer_space("sr2l_circle_pick", tier, minimum=8)


def test_exam_matches_an_independent_reading():
    for q in questions("sr2l_circle_pick_exam", "exam", 60):
        c_text, w_text = re.search(r"has centre (.+?) and passes through the point (.+?)\. Let D", stem_of(q)).groups()
        c, w = complex_parts(c_text), complex_parts(w_text)
        r = sympy.sqrt((w[0] - c[0]) ** 2 + (w[1] - c[1]) ** 2)
        mc = sympy.sqrt(c[0] ** 2 + c[1] ** 2)
        pa, pb, pc, pd = q.parts
        assert number_close(pa.answer_text, r, 1e-9)
        assert number_close(pb.answer_text, mc + r, 1e-9)
        assert number_close(pc.answer_text, mc - r, 1e-9)
        inside = lambda text: (lambda p: (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 < r**2)(complex_parts(text))  # noqa: E731
        assert inside(pd.answer_text)
        assert not any(inside(d) for d in pd.distractors), (pd.answer_text, pd.distractors)


def test_exam_structure_and_grading():
    check_structure(
        "sr2l_circle_pick_exam", ["number", "number", "number", "mcq"], [1, 2, 2, 1], [False, True, True, False]
    )
    check_graded_exam(SKILL, "sr2l_circle_pick_exam")


def test_exam_extremes_follow_through_from_a_wrong_radius():
    from mathgen.engine import grade

    from tests.sr2l_support import skill

    sk = skill(SKILL)
    for q in questions("sr2l_circle_pick_exam", "exam", 30):
        pa, pb, pc, pd = q.parts
        wrong = pa.distractors[0]
        c_text = re.search(r"has centre (.+?) and passes", stem_of(q)).group(1)
        cx, cy = complex_parts(c_text)
        mc = float(sympy.sqrt(cx**2 + cy**2))
        reply = {"a": wrong, "b": f"{mc + float(wrong):g}", "c": f"{mc - float(wrong):g}", "d": pd.answer_text}
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, reply, label)


def test_clean_and_deterministic():
    check_clean("sr2l_circle_pick", ["easy", "medium", "hard"])
    check_clean("sr2l_circle_pick_exam", ["exam"])
