"""Tests `complex-rotation-and-vertex-completion` (topic `w6b_complex_rotation`) and its exam chain.

The oracle reads the complex numbers and the angle back out of the question TEXT and recomputes the image or the
vertex with SymPy, independently of the generator's own construction.
"""

from __future__ import annotations

import math
import re

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade
from tests.w6b_support import TIERS, check_exam_generic, check_follow_through, check_tier, check_variety, check_wiring
from tests.w6b_support import parse_cx, q_of, skill

TOPIC = "w6b_complex_rotation"
EXAM = "exd_x_w6b_complex_rotation"
SK = skill("complex-rotation-and-vertex-completion")
_CX = re.compile(r"(?<![\w/.'])(-?\d+ [+-] \d*i|-?\d*i|-?\d+)(?![\w/°])")
I = sympy.I


def complexes(text):
    return [parse_cx(m) for m in _CX.findall(text)]


def turn_degrees(text):
    if "half turn" in text or "180°" in text or "π radians" in text:
        return 180
    for words, deg in (
        (("quarter turn", "90°", "π/2"), 90),
        (("2π/3", "120°"), 120),
        (("π/3", "60°"), 60),
        (("π/4", "45°"), 45),
        (("π/6", "30°"), 30),
    ):
        if any(w in text for w in words):
            break
    else:  # pragma: no cover
        raise AssertionError(text)
    clockwise = "clockwise" in text and "anticlockwise" not in text and "counter-clockwise" not in text
    return -deg if clockwise else deg


def cis(deg):
    rad = sympy.Rational(deg, 180) * sympy.pi
    return sympy.cos(rad) + I * sympy.sin(rad)


def same(a, b):
    return sympy.simplify(sympy.expand(a - b)) == 0


def test_wiring():
    check_wiring(
        SK,
        TOPIC,
        EXAM,
        "expression",
        ["senior-define-and-use-multiplication-and-division-of-complex-numbers-in-polar-form"],
    )


@pytest.mark.parametrize("tier", TIERS)
def test_tier_grading_and_distractors(tier):
    check_tier(SK, TOPIC, tier, seeds=100)


@pytest.mark.parametrize("tier", TIERS)
def test_tier_variety(tier):
    check_variety(TOPIC, tier, min_texts=50, min_answers=40)


def test_easy_oracle():
    degs = set()
    for seed in range(150):
        q = q_of(TOPIC, "easy", seed)
        z = complexes(q.text)[0]
        deg = turn_degrees(q.text)
        degs.add(deg)
        assert same(parse_cx(q.answer_text), z * cis(deg)), (seed, q.text, q.answer_text)
    assert degs == {90, -90, 180}


def test_medium_oracle_centre_and_surd():
    seen = set()
    for seed in range(200):
        q = q_of(TOPIC, "medium", seed)
        nums = complexes(q.text)
        deg = turn_degrees(q.text)
        if len(nums) == 2:
            z, c = nums
            expected = c + (z - c) * cis(deg)
            seen.add("centre")
        else:
            z = nums[0]
            expected = z * cis(deg)
            seen.add("surd")
            assert "origin" in q.text
        assert same(parse_cx(q.answer_text), expected), (seed, q.text, q.answer_text)
    assert seen == {"centre", "surd"}


def test_hard_oracle():
    seen = set()
    for seed in range(250):
        q = q_of(TOPIC, "hard", seed)
        nums = complexes(q.text)
        ans = parse_cx(q.answer_text)
        anti = "clockwise" not in q.text or "anticlockwise" in q.text or "counter-clockwise" in q.text
        turn = I if anti else -I
        if "equilateral" in q.text:
            a, b = nums
            expected = a + (b - a) * cis(60 if anti else -60)
            assert same(ans, expected), (seed, q.text, q.answer_text)
            assert all(abs(abs(complex(x) - complex(y)) - abs(complex(a - b))) < 1e-9 for x, y in ((a, ans), (b, ans)))
            seen.add("triangle")
        elif "centre" in q.text:
            pair = []
            for n in nums:
                if n not in pair:
                    pair.append(n)
            ok = any(same(ans, c + turn * (a - c)) for a, c in (tuple(pair), tuple(reversed(pair))))
            assert ok, (seed, q.text, q.answer_text)
            seen.add("centre")
        else:
            a, b = nums
            target = "C" if ("vertex C" in q.text or "C represents" in q.text) else "D"
            expected = b + turn * (b - a) if target == "C" else a + turn * (b - a)
            assert same(ans, expected), (seed, q.text, q.answer_text)
            seen.add("square")
    assert seen == {"triangle", "centre", "square"}


def test_exam_shape_and_marks():
    check_exam_generic(SK, EXAM, n=80, marks=[1, 1, 1, 2], kinds=["expression", "number", "number", "expression"])
    q = make(EXAM, seed=0, difficulty="exam")
    assert [p.follow_through for p in q.parts] == [None, None, "part_b**2", None]


def test_exam_oracle():
    for seed in range(150):
        q = make(EXAM, seed=seed, difficulty="exam")
        a, b = complexes(q.stem)[:2]
        anti = "anticlockwise" in q.stem
        turn = I if anti else -I
        u = b - a
        assert same(parse_cx(q.parts[0].answer_text), u)
        side = math.sqrt(float(u.as_real_imag()[0] ** 2 + u.as_real_imag()[1] ** 2))
        assert side == int(side) == q.parts[1].answer
        assert q.parts[2].answer == q.parts[1].answer ** 2
        target = re.search(r"vertex ([CD])", q.parts[3].prompt).group(1)
        expected = b + turn * u if target == "C" else a + turn * u
        assert same(parse_cx(q.parts[3].answer_text), expected), (seed, q.text)


def test_exam_follow_through_from_a_wrong_length():
    def wrong(q):
        return [str(int(q.parts[1].answer) + 1), str(int(q.parts[1].answer) + 3)]

    def follow(q, reply):
        return str(int(reply) ** 2)

    assert check_follow_through(SK, EXAM, "b", wrong, "c", follow, n=40) >= 40


def test_grader_reads_alternative_spellings():
    q = q_of(TOPIC, "easy", 3)
    z = parse_cx(q.answer_text)
    re_, im_ = sympy.re(z), sympy.im(z)
    assert grade(SK, q, f"{im_}*i + ({re_})").correct
    assert not grade(SK, q, f"{re_} + {-im_}i").correct or im_ == 0
