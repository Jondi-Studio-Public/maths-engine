"""`f10-transform-image-coordinates`: topic `fr2a_transform_image` (easy, medium, hard) and the exam chain."""

from __future__ import annotations

import re

import fr2a_support as sup
import pytest
import sympy

SKILL = "f10-transform-image-coordinates"
TOPIC = "fr2a_transform_image"
EXAM = "exd_x_fr2a_transform_chain"
SEEDS = range(70)


def corners(question):
    return {p["label"]: tuple(int(c) for c in p["at"]) for p in question.figure["points"] if p.get("label") in "ABC"}


def move(text: str):
    """Steps named in the stem, found with this file's own patterns and applied as SymPy matrices (an oracle apart
    from the generator's own helpers)."""
    steps = []
    for m in re.finditer(
        r"translated (\d+) units? (to the right|to the left|up|down)(?: and (\d+) units? (up|down))?"
        r"|reflected in the (x|y)-axis"
        r"|reflected in the line (x|y) = (-?\d+)"
        r"|rotated (180|90)° ?(clockwise|anticlockwise)? about (?:the origin|the point \((-?\d+), (-?\d+)\))",
        text,
    ):
        g = m.groups()
        if g[0]:
            dx = {"to the right": 1, "to the left": -1}.get(g[1], 0) * int(g[0])
            dy = {"up": 1, "down": -1}.get(g[1], 0) * int(g[0])
            if g[2]:
                dy += {"up": 1, "down": -1}[g[3]] * int(g[2])
            steps.append(("T", sympy.Matrix([dx, dy])))
        elif g[4]:
            steps.append(("M", sympy.diag(1, -1) if g[4] == "x" else sympy.diag(-1, 1), sympy.zeros(2, 1)))
        elif g[5]:
            k = int(g[6])
            if g[5] == "x":
                steps.append(("M", sympy.diag(-1, 1), sympy.Matrix([2 * k, 0])))
            else:
                steps.append(("M", sympy.diag(1, -1), sympy.Matrix([0, 2 * k])))
        else:
            angle = 180 if g[7] == "180" else (-90 if g[8] == "clockwise" else 90)
            c = sympy.Matrix([int(g[9] or 0), int(g[10] or 0)])
            rot = sympy.Matrix(
                [
                    [sympy.cos(sympy.rad(angle)), -sympy.sin(sympy.rad(angle))],
                    [sympy.sin(sympy.rad(angle)), sympy.cos(sympy.rad(angle))],
                ]
            ).applyfunc(sympy.nsimplify)
            steps.append(("R", rot, c))
    return steps


def apply(steps, p):
    v = sympy.Matrix(p)
    for s in steps:
        if s[0] == "T":
            v = v + s[1]
        elif s[0] == "M":
            v = s[1] * v + s[2]
        else:
            v = s[2] + s[1] * (v - s[2])
    return tuple(int(x) for x in v)


def asked(question):
    m = re.search(r"coordinates of ([ABC])('+) as", question.text)
    return m[1], len(m[2])


def test_wiring():
    sup.assert_wired(SKILL, TOPIC, EXAM, "pair", True)


def test_contract_all_tiers():
    sup.assert_single_answer_contract(SKILL, TOPIC, SEEDS)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_answer_is_the_image_of_the_drawn_corner(tier):
    for seed in SEEDS:
        question = sup.q(TOPIC, tier, seed)
        name, primes = asked(question)
        steps = move(question.text)
        assert len(steps) == primes
        x, y = apply(steps, corners(question)[name])
        assert question.answer == {"x": x, "y": y}, (tier, seed, question.text)


def test_tiers_differ_in_shape():
    easy = [move(sup.q(TOPIC, "easy", s).text) for s in SEEDS]
    medium = [move(sup.q(TOPIC, "medium", s).text) for s in SEEDS]
    hard = [move(sup.q(TOPIC, "hard", s).text) for s in SEEDS]
    assert all(len(steps) == 1 for steps in easy + medium)
    assert all(len(steps) == 2 for steps in hard)
    # easy is a translation or a reflection in an axis; medium needs a line, a turn or a centre
    assert all(s[0][0] == "T" or (s[0][0] == "M" and not any(s[0][2])) for s in easy)
    assert any(s[0][0] == "R" for s in medium) and any(s[0][0] == "M" and any(s[0][2]) for s in medium)


def test_hard_order_matters():
    for seed in SEEDS:
        question = sup.q(TOPIC, "hard", seed)
        name, _ = asked(question)
        steps = move(question.text)
        p = corners(question)[name]
        assert apply(steps, p) != apply(steps[::-1], p), (seed, question.text)


def test_mirror_lines_are_drawn_and_miss_the_triangle():
    seen = 0
    for seed in SEEDS:
        question = sup.q(TOPIC, "medium", seed)
        m = re.search(r"reflected in the line ([xy]) = (-?\d+)", question.text)
        if not m:
            continue
        seen += 1
        assert [ln["label"] for ln in question.figure["lines"]] == [f"{m[1]} = {m[2]}"]
        k, axis = int(m[2]), 0 if m[1] == "x" else 1
        sides = {(p[axis] > k) - (p[axis] < k) for p in corners(question).values()}
        assert len(sides) == 1 and 0 not in sides
    assert seen >= 10


def test_the_image_is_not_drawn_but_the_grid_covers_it():
    for tier in sup.TIERS:
        for seed in SEEDS:
            question = sup.q(TOPIC, tier, seed)
            fig = question.figure
            assert sorted(p["label"] for p in fig["points"]) == ["A", "B", "C"]
            name, primes = asked(question)
            x, y = question.answer["x"], question.answer["y"]
            assert fig["x_range"][0] <= x <= fig["x_range"][1] and fig["y_range"][0] <= y <= fig["y_range"][1]


# ------------------------------------------------------------------------------------------ exam


def test_exam_contract():
    sup.assert_exam_contract(SKILL, EXAM, range(70))


def test_exam_parts_rederived():
    for seed in range(70):
        question = sup.exam(EXAM, seed)
        stem = question.stem
        pts = {m[1]: (int(m[2]), int(m[3])) for m in re.finditer(r"([ABC])\((-?\d+), (-?\d+)\)", stem)}
        steps = move(stem)
        assert len(steps) == 2
        a1, a2 = apply(steps[:1], pts["A"]), apply(steps, pts["A"])
        got = sup.canonical(question)
        assert got["a"] == f"x = {a1[0]}, y = {a1[1]}" and got["b"] == f"x = {a2[0]}, y = {a2[1]}"
        b, c = pts["B"], pts["C"]
        sides = sorted(
            [((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5 for p, q in ((pts["A"], b), (b, c), (c, pts["A"]))]
        )
        assert float(got["c"]) == round(sides[2]) and float(got["d"]) == round(sum(sides))


def test_exam_follow_through_from_a_wrong_longest_side():
    for seed in range(40):
        question = sup.exam(EXAM, seed)
        hyp = float(question.parts[2].answer_text)
        legs = float(question.parts[3].answer_text) - hyp
        wrong = hyp + 3
        sup.assert_follow_through(SKILL, question, {"c": str(int(wrong))}, {"d": str(int(wrong + legs))})
