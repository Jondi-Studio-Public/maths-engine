"""Tests `complex-arg-ratio-geometry` (topic `nx_complex_arg_ratio`) and its exam chain.

The oracle reads the points back out of the question TEXT and recomputes the answer with SymPy (the argument by
``sympy.arg``, collinearity by the imaginary part of the ratio, perpendicularity by the real part of a product with a
conjugate), independently of the generator's construction.
"""

from __future__ import annotations

import math
import re

import pytest
import sympy

from mathgen import make
from tests.w6b_support import TIERS, check_exam_generic, check_follow_through, check_tier, check_variety, check_wiring
from tests.w6b_support import parse_cx, q_of, skill

TOPIC = "nx_complex_arg_ratio"
EXAM = "exd_nx_complex_arg_ratio"
SK = skill("complex-arg-ratio-geometry")
I = sympy.I
_CX = re.compile(r"(?<![\w/.'])(-?\d+ [+-] \d*i|-?\d*i|-?\d+)(?![\w/°])")
_ARG3 = (
    re.compile(r"represent (.+?), (.+?) and (.+?)(?: on|\. )"),
    re.compile(r"A = (.+?), B = (.+?) and C = (.+?)(?: on|\. )"),
    re.compile(r"A\((.+?)\), B\((.+?)\) and C\((.+?)\)(?: on|\. )"),
)


def complexes(text):
    return [parse_cx(m) for m in _CX.findall(text)]


def three_points(text):
    for pattern in _ARG3:
        m = pattern.search(text)
        if m:
            return [parse_cx(g) for g in m.groups()]
    raise AssertionError(text)  # pragma: no cover


def deg(z):
    return float(sympy.N(sympy.Abs(sympy.arg(sympy.expand(z))) * 180 / sympy.pi))


def same(a, b):
    return sympy.simplify(sympy.expand(a - b)) == 0


def test_wiring():
    check_wiring(
        SK,
        TOPIC,
        EXAM,
        "expression",
        ["complex-rotation-and-vertex-completion", "senior-consider-complex-numbers-as-points-in-a-plane"],
    )


@pytest.mark.parametrize("tier", TIERS)
def test_tier_grading_and_distractors(tier):
    check_tier(SK, TOPIC, tier, seeds=100)


@pytest.mark.parametrize("tier", TIERS)
def test_tier_variety(tier):
    check_variety(TOPIC, tier, min_texts=40, min_answers=15)


def test_easy_oracle():
    seen = set()
    for seed in range(200):
        q = q_of(TOPIC, "easy", seed)
        if "midpoint" in q.text:
            a, b = complexes(q.text)
            assert same(parse_cx(q.answer_text), (a + b) / 2), (seed, q.text)
            seen.add("midpoint")
        elif "apart" in q.text or "Find the distance" in q.text or "length of the segment" in q.text:
            a, b = complexes(q.text)
            assert math.isclose(float(sympy.N(sympy.Abs(a - b))), float(q.answer_text)), (seed, q.text)
            seen.add("distance")
        else:
            m = re.search(r"(\([^)]*\)|-?\w+)z ([+-]) (\([^)]*\)|\w+)", q.text)
            mult, const = parse_cx(m.group(1).strip("()")), parse_cx(m.group(3).strip("()"))
            const = const if m.group(2) == "+" else -const
            rest = q.text.replace(m.group(0), "")
            z = complexes(rest)[0]
            assert same(parse_cx(q.answer_text), mult * z + const), (seed, q.text, q.answer_text)
            seen.add("map")
    assert seen == {"midpoint", "distance", "map"}


def test_medium_oracle():
    seen = set()
    for seed in range(250):
        q = q_of(TOPIC, "medium", seed)
        found = re.search(r"k ([+-]) (\d+)i", q.text)
        if found:
            a, b = complexes(q.text[: found.start()])[:2]
            sign, c = found.groups()
            k = sympy.Symbol("k", real=True)
            point = k + (1 if sign == "+" else -1) * int(c) * I
            ratio = sympy.expand((point - a) * sympy.conjugate(b - a))
            (sol,) = sympy.solve(sympy.im(ratio), k)
            assert sol == int(q.answer_text), (seed, q.text, q.answer_text)
            seen.add("collinear")
        elif "maps" in q.text or "sends" in q.text:
            z1, w1, z2, w2 = complexes(q.text)
            assert same(parse_cx(q.answer_text), (w1 - w2) / (z1 - z2)), (seed, q.text)
            seen.add("multiplier")
        else:
            a, b, c = complexes(q.text)
            assert same(parse_cx(q.answer_text), (c - b) / (a - b)), (seed, q.text)
            seen.add("ratio")
    assert seen == {"collinear", "multiplier", "ratio"}


def test_hard_oracle():
    seen = set()
    for seed in range(250):
        q = q_of(TOPIC, "hard", seed)
        if not re.search(r"k [+-] \d+i", q.text):
            a, b, c = three_points(q.text)
            assert math.isclose(deg((c - b) / (a - b)), float(q.answer_text), abs_tol=1e-6), (seed, q.text)
            seen.add("angle")
        else:
            found = re.search(r"k ([+-]) (\d+)i", q.text)
            sign, c = found.groups()
            pts = complexes(q.text[: found.start()].replace("Find the real number k", ""))[:2]
            a, b = (pts[1], pts[0]) if "line from" in q.text else pts
            k = sympy.Symbol("k", real=True)
            point = k + (1 if sign == "+" else -1) * int(c) * I
            dot = sympy.expand(sympy.re(sympy.expand((a - b) * sympy.conjugate(point - b))))
            (sol,) = sympy.solve(dot, k)
            assert sol == int(q.answer_text), (seed, q.text, q.answer_text)
            seen.add("perp")
    assert seen == {"angle", "perp"}
    angles = {q_of(TOPIC, "hard", s).answer_text for s in range(250)}
    assert {"30", "45", "60", "90", "120", "135", "150"} <= angles


def test_exam_shape_and_marks():
    check_exam_generic(SK, EXAM, n=80, marks=[2, 1, 1, 2], kinds=["expression", "number", "expression", "expression"])
    q = make(EXAM, seed=0, difficulty="exam")
    assert q.parts[1].follow_through and "part_a" in q.parts[1].follow_through
    assert q.parts[3].follow_through and "part_c" in q.parts[3].follow_through


def test_exam_oracle():
    for seed in range(150):
        q = make(EXAM, seed=seed, difficulty="exam")
        a, b, c = complexes(q.text)[:3]
        ratio = (c - b) / (a - b)
        assert same(parse_cx(q.parts[0].answer_text), ratio), (seed, q.text)
        assert math.isclose(deg(ratio), q.parts[1].answer, abs_tol=1e-6)
        mid = (a + c) / 2
        assert same(parse_cx(q.parts[2].answer_text), mid)
        assert same(parse_cx(q.parts[3].answer_text), a + c - b)


def test_exam_follow_through_from_a_wrong_midpoint():
    def wrong(q):
        return [str(sympy.expand(parse_cx(q.parts[2].answer_text) + 3 + 2 * I)).replace("I", "i").replace("*", "")]

    def follow(q, reply):
        a, b, c = complexes(q.text)[:3]
        wrong_mid = parse_cx(reply)
        if wrong_mid == 0:
            return None
        return str(sympy.expand(2 * wrong_mid - b)).replace("I", "i").replace("*", "")

    assert check_follow_through(SK, EXAM, "c", wrong, "d", follow, n=40) >= 20


def _real(text):
    """The real number an option stands for, else None (a complex option has no order)."""
    try:
        value = sympy.sympify(text.replace("√", "sqrt").replace("^", "**"), rational=True)
        return float(value) if value.is_real else None
    except (sympy.SympifyError, TypeError, ValueError):
        return None


def _ranks(units):
    ranks, firsts = set(), set()
    for answer, options in units:
        firsts.add(options[0])
        values = [_real(t) for t in (answer, *options)]
        if all(v is not None for v in values):
            ranks.add(sum(v < values[0] for v in values[1:]))
    return ranks, firsts


@pytest.mark.parametrize("tier", TIERS)
def test_key_rank_and_position_vary(tier):
    """Over 240 seeds the key's rank among numeric options takes at least 3 values, and no option is always first."""
    units = [(q.answer_text, q.distractors) for q in (q_of(TOPIC, tier, s) for s in range(240))]
    ranks, firsts = _ranks(units)
    assert len(firsts) >= 3, firsts
    if tier != "easy":  # easy mixes complex answers with a distance; the numeric ones are checked below
        assert len(ranks) >= 3 or not ranks, ranks
    else:
        dist = [
            u
            for q, u in zip((q_of(TOPIC, tier, s) for s in range(240)), units)
            if "apart" in q.text or "distance" in q.text.lower() or "segment" in q.text
        ]
        assert len(_ranks(dist)[0]) >= 3, _ranks(dist)


def test_exam_key_rank_and_position_vary():
    qs = [make(EXAM, seed=s, difficulty="exam") for s in range(240)]
    for index in range(4):
        units = [(q.parts[index].answer_text, q.parts[index].distractors) for q in qs]
        ranks, firsts = _ranks(units)
        assert len(firsts) >= 3, (index, firsts)
        if index == 1:
            assert len(ranks) >= 3, ranks


def test_no_angle_option_exceeds_a_straight_angle():
    for seed in range(240):
        q = q_of(TOPIC, "hard", seed)
        if "angle" in q.text and "right angle" not in q.text:
            assert all(0 < float(t) < 180 for t in q.distractors), (q.text, q.distractors)
        e = make(EXAM, seed=seed, difficulty="exam")
        assert all(0 < float(t) < 180 for t in e.parts[1].distractors), e.parts[1].distractors


# -- what each stem rules out: every option (slips and fillers alike) is checked against its part's constraints ---------
#
# part                          constraint on every option
# easy  image under w = az + b  a complex number with whole-number parts, like the key
# easy  midpoint                a complex number with whole-number parts
# easy  distance                a whole number >= 1
# medium ratio / multiplier     a complex number of the key's shape (whole parts for whole-part keys)
# medium collinear k            a whole number
# hard  angle ABC               a whole number of degrees, 0 < v < 180 (a triangle's angle)
# hard  perpendicular k         a whole number
# exam  (a) ratio               shape of the key;  (b) angle: whole degrees 0 < v < 180
# exam  (c) midpoint            whole-part complex;  (d) fourth vertex: whole-part complex


def _whole(z) -> bool:
    z = sympy.expand(z)
    return bool(sympy.re(z).is_integer and sympy.im(z).is_integer)


def _check_complex(key, options, where):
    shape = _whole(parse_cx(key))
    for t in options:
        assert _whole(parse_cx(t)) == shape, (where, key, options)
        assert t != key, where


def _check_whole(key, options, where, lo=None, hi=None):
    for t in options:
        v = float(t)
        assert v == int(v), (where, options)
        assert (lo is None or v > lo) and (hi is None or v < hi), (where, key, options)


def test_every_option_meets_its_parts_constraints():
    for tier in TIERS:
        for seed in range(320):
            q = q_of(TOPIC, tier, seed)
            where = (tier, seed, q.text)
            if q.answer_kind == "number":
                angle = tier == "hard" and not re.search(r"k [+-] \d+i", q.text)
                _check_whole(q.answer_text, q.distractors, where, *((0, 180) if angle else (None, None)))
                if "distance" in q.text.lower() or "apart" in q.text or "length of the segment" in q.text:
                    _check_whole(q.answer_text, q.distractors, where, 0, None)
            else:
                _check_complex(q.answer_text, q.distractors, where)
    for seed in range(320):
        q = make(EXAM, seed=seed, difficulty="exam")
        for index, part in enumerate(q.parts):
            where = ("exam", seed, index)
            if index == 1:
                _check_whole(part.answer_text, part.distractors, where, 0, 180)
            else:
                _check_complex(part.answer_text, part.distractors, where)
        assert _whole(parse_cx(q.parts[2].answer_text)) and _whole(parse_cx(q.parts[3].answer_text))


def test_angle_options_do_not_leave_the_key_at_an_extreme():
    """With 5 and 10 degree fillers the key is not always the smallest at 30 degrees or the largest at 150."""
    for theta in ("30", "150"):
        ranks = set()
        for seed in range(600):
            q = q_of(TOPIC, "hard", seed)
            if q.answer_text == theta:
                ranks.add(sum(float(t) < float(theta) for t in q.distractors))
        assert len(ranks) >= 2, (theta, ranks)


def test_stems_never_show_a_bracketed_negative_constant():
    for seed in range(320):
        text = q_of(TOPIC, "easy", seed).text
        assert not re.search(r"\+ \(-\d+i?\)", text), text
