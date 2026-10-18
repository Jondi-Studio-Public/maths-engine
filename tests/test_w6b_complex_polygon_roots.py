"""Tests `complex-polygon-vertices-and-roots-of-unity` (topic `w6b_complex_polygon_roots`) and its exam chain.

The oracle reads the polygon, the centre, the vertex and the number of steps back out of the question TEXT and
recomputes the answer from the n-th roots of unity with SymPy and floating-point complex arithmetic.
"""

from __future__ import annotations

import cmath
import math
import re

import pytest
import sympy

from mathgen import make
from tests.w6b_support import TIERS, check_exam_generic, check_follow_through, check_tier, check_variety, check_wiring
from tests.w6b_support import parse_cx, q_of, skill

TOPIC = "w6b_complex_polygon_roots"
EXAM = "exd_x_w6b_complex_polygon_roots"
SK = skill("complex-polygon-vertices-and-roots-of-unity")
_CX_ALT = r"-?\d+ [+-] \d*i|-?\d*i|-?\d+"
_CX = re.compile(r"(?<![\w/.'^])(-?\d+ [+-] \d*i|-?\d*i|-?\d+)(?![\w/°^]|-gon| vert)")
_SIDES = {"equilateral triangle": 3, "square": 4, "regular hexagon": 6, "regular octagon": 8, "regular 12-gon": 12}
I = sympy.I


def sides_in(text):
    for name, n in _SIDES.items():
        if name in text:
            return n
    raise AssertionError(text)


def steps_in(text):
    m = re.search(r"(\d+) vertices", text)
    return int(m.group(1)) if m else 1


def test_wiring():
    check_wiring(SK, TOPIC, EXAM, "expression", ["complex-rotation-and-vertex-completion"])


@pytest.mark.parametrize("tier", TIERS)
def test_tier_grading_and_distractors(tier):
    check_tier(SK, TOPIC, tier, seeds=100)


@pytest.mark.parametrize("tier", TIERS)
def test_tier_variety(tier):
    check_variety(TOPIC, tier, min_texts=30, min_answers=8)


def test_easy_oracle():
    shapes = set()
    for seed in range(200):
        q = q_of(TOPIC, "easy", seed)
        n, k = sides_in(q.text), steps_in(q.text)
        z0 = parse_cx(_CX.findall(q.text)[0])
        expected = complex(z0) * cmath.exp(2j * math.pi * k / n)
        assert abs(complex(parse_cx(q.answer_text)) - expected) < 1e-9, (seed, q.text, q.answer_text)
        assert "anticlockwise" in q.text
        shapes.add(n)
    assert shapes == {3, 4, 6, 8, 12}


def _distinct_complexes(text):
    found = []
    for m in _CX.findall(text):
        z = parse_cx(m)
        if z not in found:
            found.append(z)
    return found


def test_medium_oracle():
    seen = set()
    for seed in range(250):
        q = q_of(TOPIC, "medium", seed)
        t = q.text
        if "cos(2π/" in t or "root of unity" in t:
            n = int(
                re.search(r"cos\(2π/(\d+)\)", t).group(1) if "cos(2π/" in t else re.search(r"is the (\d+)", t).group(1)
            )
            omega = cmath.exp(2j * math.pi / n)
            expr = re.split(r"Evaluate |the value of |calculate ", t)[-1].rstrip(".")
            lead = re.match(r"(\d+) \+ \1ω", expr)
            if lead:
                total = int(lead.group(1)) * sum(omega**k for k in range(n))
            elif expr.startswith("ω +"):
                total = sum(omega**k for k in range(1, n))
            else:
                lo = int(re.match(r"ω\^(\d+)", expr).group(1))
                total = sum(omega**k for k in range(lo, lo + n))
            assert abs(total - complex(sympy.sympify(q.answer_text))) < 1e-9, (seed, t, q.answer_text)
            seen.add("sum")
        else:
            c = parse_cx(re.search(rf"(?:centre is|centre at|centred at) ({_CX_ALT})", t).group(1))
            z0 = next(z for z in _distinct_complexes(t) if z != c)
            n, k = sides_in(t), steps_in(t)
            expected = complex(c) + (complex(z0) - complex(c)) * cmath.exp(2j * math.pi * k / n)
            assert abs(complex(parse_cx(q.answer_text)) - expected) < 1e-9, (seed, t)
            seen.add("centre")
    assert seen == {"sum", "centre"}


def _circle_radius(t, n):
    """The circumradius a hard question states: |z| = r, 'radius r', 'distance r' or the constant in z^n = w."""
    m = re.search(r"\|z\| = (\d+)|circumradius (\d+)|radius (\d+)|distance (\d+)", t)
    if m:
        return float(next(g for g in m.groups() if g))
    return int(re.search(r"z\^\d+ = (\d+)", t).group(1)) ** (1 / n)


def _polygon_size(t):
    m = re.search(r"(\d+)-gon|The (\d+) roots", t)
    return int(m.group(1) or m.group(2))


def test_hard_oracle():
    seen = set()
    for seed in range(300):
        q = q_of(TOPIC, "hard", seed)
        t = q.text
        if "product" in t:
            n = _polygon_size(t)
            r = _circle_radius(t, n)
            verts = [r * cmath.exp(2j * math.pi * k / n) for k in range(n)]
            prod = math.prod(abs(v - verts[0]) for v in verts[1:])
            assert abs(prod - float(q.answer_text)) < 1e-6 * prod, (seed, t, q.answer_text)
            seen.add("product")
        elif "squares of the distances" in t or "|V - z|" in t:
            n = _polygon_size(t)
            r = _circle_radius(t, n)
            verts = [r * cmath.exp(2j * math.pi * k / n) for k in range(n)]
            total = sum(abs(v - verts[0]) ** 2 for v in verts)
            assert abs(total - float(q.answer_text)) < 1e-6 * total, (seed, t, q.answer_text)
            seen.add("squares")
        elif "other" in t or "remaining" in t:
            n = int(re.search(r"z\^(\d+)", t).group(1))
            w = int(re.search(r"z\^\d+ = (\d+)", t).group(1))
            roots = [complex(r) for r in sympy.Poly(sympy.Symbol("z") ** n - w).nroots()]
            given = complex(
                parse_cx(
                    re.search(
                        r"(?:is|=|and) (-?(?:[\d√/ +-]|\(.*?\)i|i)+?)(?:\. |\.$| is one)", t.replace("z₀ = ", "is ")
                    )
                    .group(1)
                    .strip()
                )
            )
            assert min(abs(given - r) for r in roots) < 1e-9
            assert abs(sum(roots) - given - complex(parse_cx(q.answer_text))) < 1e-9, (seed, t, q.answer_text)
            seen.add("others")
        else:
            n = int(
                re.search(r"cos\(2π/(\d+)\)", t).group(1)
                if "cos(2π/" in t
                else re.search(r"primitive (\d+)", t).group(1)
            )
            found = re.search(r"ω\^\((\d+)k\)|1 \+ ω\^(\d+) \+", t)
            m = int(found.group(1) or found.group(2))
            omega = cmath.exp(2j * math.pi / n)
            total = sum(omega ** (m * k) for k in range(n))
            assert abs(total - float(q.answer_text)) < 1e-9, (seed, t, q.answer_text)
            seen.add("power")
    assert seen == {"product", "squares", "others", "power"}


def test_exam_shape_and_marks():
    check_exam_generic(SK, EXAM, n=80, marks=[1, 2, 1, 2], kinds=["number", "number", "number", "expression"])
    q = make(EXAM, seed=0, difficulty="exam")
    assert q.parts[1].follow_through.startswith("2*part_a*sin(pi/")
    assert q.parts[2].follow_through.endswith("*part_b")


def test_exam_oracle():
    shapes = set()
    for seed in range(150):
        q = make(EXAM, seed=seed, difficulty="exam")
        n = int(re.search(r"z\^(\d+) = ", q.stem).group(1))
        w = int(re.search(r"z\^\d+ = (\d+)", q.stem).group(1))
        shapes.add(n)
        roots = [complex(r) for r in sympy.Poly(sympy.Symbol("z") ** n - w).nroots()]
        r = abs(roots[0])
        assert q.parts[0].answer == round(r)
        side = min(abs(a - b) for a in roots for b in roots if a != b)
        assert abs(float(q.parts[1].answer_text) - side) <= 0.005 + 1e-9
        assert abs(float(q.parts[2].answer_text) - n * float(q.parts[1].answer_text)) <= 1e-9
        by_arg = min(roots, key=lambda z: abs(cmath.phase(z) - 2 * math.pi / n))
        assert abs(complex(parse_cx(q.parts[3].answer_text)) - by_arg) < 1e-9
    assert shapes == {3, 4, 6}


def test_exam_follow_through_from_a_wrong_radius():
    def wrong(q):
        return [str(int(q.parts[0].answer) + 1), str(int(q.parts[0].answer) + 2)]

    def follow(q, reply):
        n = int(re.search(r"z\^(\d+) = ", q.stem).group(1))
        return f"{2 * int(reply) * math.sin(math.pi / n):.2f}"

    assert check_follow_through(SK, EXAM, "a", wrong, "b", follow, n=40) >= 40
