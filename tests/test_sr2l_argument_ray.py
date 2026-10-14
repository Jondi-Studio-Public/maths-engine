"""locus-argument-ray-wedge: the line containing the ray Arg(z - a) = theta, and the wedge between two such rays."""

from __future__ import annotations

import re
from fractions import Fraction

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
    complex_parts,
    expr,
    line_form,
    normal_line,
    number_close,
    questions,
    stem_of,
    x,
    y,
)

SKILL = "locus-argument-ray-wedge"
TOPICS = {"easy": "sr2l_arg_ray", "medium": "sr2l_arg_ray", "hard": "sr2l_arg_ray", "exam": "sr2l_arg_ray_exam"}
_ARG = re.compile(r"(?:(-?[^≤]+?) ≤ )?Arg\(z ([^)]+)\)(?: = ([^.]+?)\.| ≤ ([^.]+?)\.| ≤ ([^ ]+) )")


def angle(text: str):
    """An angle such as ``3π/4``, ``-π/2`` or ``0`` as an exact SymPy value."""
    return (
        sympy.sympify(text.replace("π", "*pi").lstrip("*").replace("-*", "-"))
        if text.strip() != "0"
        else sympy.Integer(0)
    )


def ray_line(vertex: tuple, theta) -> tuple:
    """The line through the vertex at the angle theta, as normalised (A, B, C), from the direction (cos, sin)."""
    c, s = sympy.cos(theta), sympy.sin(theta)
    return normal_line(sympy.expand(sympy.nsimplify(s) * (x - vertex[0]) - sympy.nsimplify(c) * (y - vertex[1])))


def _normal(vertex, theta):
    c, s = sympy.simplify(sympy.cos(theta)), sympy.simplify(sympy.sin(theta))
    if c == 0:
        return normal_line(x - vertex[0])
    m = sympy.simplify(s / c)
    return normal_line(sympy.expand(m * (x - vertex[0]) + vertex[1] - y))


def _vertex(inner: str) -> tuple:
    """z + 6 - 3i has its vertex at -(6 - 3i)."""
    re_, im_ = complex_parts(inner.replace("z", "0"))
    return -re_, -im_


def test_wired():
    check_wired(SKILL, TOPICS)


def test_easy_matches_an_independent_reading():
    for q in questions("sr2l_arg_ray", "easy", 80):
        inner, theta = re.search(r"Arg\(z ([^)]+)\) = (-?[\dπ/]+)", stem_of(q)).groups()
        assert line_form(q.answer_text) == _normal(_vertex(inner), angle(theta)), (q.text, q.answer_text)


def test_medium_line_runs_through_the_drawn_ray():
    for q in questions("sr2l_arg_ray", "medium", 60):
        seg = q.figure["segments"][0]
        a, b, c = line_form(q.answer_text)
        for end in (seg["from"], seg["to"]):
            px, py = (sympy.Rational(*Fraction(v).limit_denominator(100).as_integer_ratio()) for v in end)
            assert a * px + b * py + c == 0, (q.text, q.answer_text, end)
        assert any(p.get("open") for p in q.figure["points"]) or seg.get("open_start"), "the vertex is shown open"


def test_hard_names_the_boundary_with_the_greater_argument():
    for q in questions("sr2l_arg_ray", "hard", 80):
        m = re.search(r"(.+?) ≤ Arg\(z ([^)]+)\) ≤ ([^.]+?)\. The boundary", stem_of(q))
        lo, inner, hi = m.group(1).split("satisfies ")[-1], m.group(2), m.group(3)
        which = angle(hi) if "greater argument" in stem_of(q) else angle(lo)
        assert line_form(q.answer_text) == _normal(_vertex(inner), which), (q.text, q.answer_text)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_graded(tier):
    check_graded_single(SKILL, "sr2l_arg_ray", tier)


def test_tiers_differ():
    check_tiers_differ(SKILL, TOPICS)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    check_answer_space("sr2l_arg_ray", tier)


def _intersection(line: tuple, axis: str, k):
    a, b, c = line
    sol = sympy.solve([a * x + b * y + c, (y if axis == "y" else x) - k], [x, y], dict=True)
    return (sol[0][x], sol[0][y]) if sol else None


def test_exam_matches_an_independent_reading():
    for q in questions("sr2l_arg_ray_exam", "exam", 60):
        text = stem_of(q)
        m = re.search(r"(.+?) ≤ Arg\(z ([^)]+)\) ≤ ([^.]+?)\. The boundary", text)
        lo, inner, hi = m.group(1).split("satisfies ")[-1], m.group(2), m.group(3)
        vertex = _vertex(inner)
        upper, lower = _normal(vertex, angle(hi)), _normal(vertex, angle(lo))
        pa, pb, pc, pd = q.parts
        assert line_form(pa.answer_text) == upper and line_form(pb.answer_text) == lower, text
        axis, k = re.search(r"line ([xy]) = (-?\d+)\.", pc.prompt + " " + text).groups()
        p1, p2 = _intersection(upper, axis, int(k)), _intersection(lower, axis, int(k))
        dist = sympy.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        assert number_close(pc.answer_text, dist, 0.0051), text
        area = abs((p1[0] - vertex[0]) * (p2[1] - vertex[1]) - (p2[0] - vertex[0]) * (p1[1] - vertex[1])) / 2
        assert number_close(pd.answer_text, area, 0.051), text


def test_exam_structure_and_grading():
    check_structure(
        "sr2l_arg_ray_exam", ["pair", "pair", "number", "number"], [2, 2, 2, 2], [False, False, False, True]
    )
    check_graded_exam(SKILL, "sr2l_arg_ray_exam")


def test_exam_follows_through_from_a_wrong_distance():
    from mathgen.engine import grade

    from tests.sr2l_support import skill

    sk, used = skill(SKILL), 0
    for q in questions("sr2l_arg_ray_exam", "exam", 30):
        pa, pb, pc, pd = q.parts
        wrong = pc.distractors[0]
        axis, k = re.search(r"line ([xy]) = (-?\d+)\.", stem_of(q) + pc.prompt).groups()
        reply = {"a": pa.answer_text, "b": pb.answer_text, "c": wrong}
        height = (
            float(expr(k) - complex_parts(re.search(r"a = (.+?)\.", stem_of(q)).group(1))[1 if axis == "y" else 0])
            if "a = " in stem_of(q)
            else None
        )
        if height is None:
            continue
        reply["d"] = f"{abs(float(wrong) * height) / 2:.1f}"
        result = grade(sk, q, reply)
        assert result.parts["c"].marks == 0
        assert result.parts["d"].marks == result.parts["d"].max_marks, (q.text, reply)
        used += 1
    assert used >= 20


def test_figure_reader_agrees_with_the_key():
    from mathgen import figures

    for q in questions("sr2l_arg_ray", "medium", 40):
        assert figures.get_reader("sr2l_arg_ray")(q) == q.answer_text


def test_clean_and_deterministic():
    check_clean("sr2l_arg_ray", ["easy", "medium", "hard"])
    check_clean("sr2l_arg_ray_exam", ["exam"])
