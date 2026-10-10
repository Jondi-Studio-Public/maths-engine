"""Exam tier of four complex-number skills (point in the plane, powers of i, n-th roots, equating real and
imaginary parts): prompt variety, an independent oracle that re-derives each key from the rendered prompt
alone, and the real graders accepting the key and rejecting every distractor.

The oracles share no code with ``mathgen.generators.exd_x_complex_plane``: they use exact integer-pair
arithmetic, rotation by matrix, ``cmath`` roots, or an exhaustive search over a grid of integers.
"""

from __future__ import annotations

import cmath
import math
import random
import re

import pytest

from mathgen import make
from mathgen.core import REGISTRY
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

# topic -> (answer kind, (existing topic, existing params, existing difficulty) of the skill's hardest practice tier)
TOPICS = {
    "exd_x_complex_point": ("pair", ("complex_number_as_point", {}, "hard")),
    "exd_x_complex_i_power": ("expression", ("imaginary_number_i_root", {}, "hard")),
    "exd_x_complex_nth_root": ("pair", ("nth_roots_of_complex_number", {}, "hard")),
    "exd_x_complex_equating": ("pair", ("p11c_cartesian_form_equating", {}, "hard")),
}

_C = r"(?<![A-Za-z])(?:-?\d*i|-?\d+(?: [+-] \d*i)?)(?![A-Za-z])"


def _parse_c(token: str) -> tuple[int, int]:
    token = token.strip()
    m = re.fullmatch(r"(-?\d+) ([+-]) (\d*)i", token)
    if m:
        return int(m.group(1)), int(m.group(3) or 1) * (1 if m.group(2) == "+" else -1)
    m = re.fullmatch(r"(-?)(\d*)i", token)
    if m:
        return 0, (-1 if m.group(1) else 1) * int(m.group(2) or 1)
    return int(token), 0


def _q(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam")


# -- oracles ----------------------------------------------------------------------------


def _oracle_point(text: str):
    nums = [_parse_c(t) for t in re.findall(_C, text)]
    if "parallelogram" in text:
        (ax, ay), (bx, by), (cx, cy) = nums
        # D is the vertex with AB parallel and equal to DC: B - A = C - D.
        return {"x": cx - (bx - ax), "y": cy - (by - ay)}
    (px, py), (cx, cy) = nums
    dx, dy = px - cx, py - cy
    # anticlockwise quarter turn: (dx, dy) -> (-dy, dx); clockwise: (dx, dy) -> (dy, -dx)
    rx, ry = (-dy, dx) if "anticlockwise" in text else (dy, -dx)
    return {"x": cx + rx, "y": cy + ry}


def _ipow_pair(n: int) -> tuple[int, int]:
    """i^n by repeated multiplication (a negative n multiplies by 1/i = -i)."""
    step = (0, 1) if n >= 0 else (0, -1)
    re_, im_ = 1, 0
    for _ in range(abs(n)):
        re_, im_ = re_ * step[0] - im_ * step[1], re_ * step[1] + im_ * step[0]
    return re_, im_


def _oracle_i_power(text: str):
    m = re.search(r"\(1 ([+-]) i\)\^(\d+)", text)
    if m:
        step = (1, 1 if m.group(1) == "+" else -1)
        re_, im_ = 1, 0
        for _ in range(int(m.group(2))):
            re_, im_ = re_ * step[0] - im_ * step[1], re_ * step[1] + im_ * step[0]
        return re_, im_
    body = re.match(r"^(?:Evaluate|Simplify) (.*?)(?:,| \.|\.) ", text + " ").group(1)
    total = [0, 0]
    for sign, e1, e2 in re.findall(r"([+-]?)\s*i\^(?:(-?\d+)|\((-?\d+)\))", body):
        e = int(e1 or e2)
        v = _ipow_pair(e)
        s = -1 if sign == "-" else 1
        total[0] += s * v[0]
        total[1] += s * v[1]
    return tuple(total)


def _oracle_nth_root(text: str):
    m = re.search(r"are (\d)cis\((\d+)°\) and \d+cis\((\d+)°\)", text)
    if m:
        r, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        w = cmath.rect(r, math.radians(a)) ** 3
        roots = [cmath.rect(r, math.radians(t)) for t in range(0, 360, 5)]
        cube_roots = [z for z in roots if abs(z**3 - w) < 1e-6]
        assert len(cube_roots) == 3
        third = [
            z
            for z in cube_roots
            if abs(z - cmath.rect(r, math.radians(a))) > 1e-6 and abs(z - cmath.rect(r, math.radians(b))) > 1e-6
        ]
        assert len(third) == 1
        return {"r": r, "theta": round(math.degrees(cmath.phase(third[0])) % 360)}
    m = re.search(r"w = (\d+)cis\((\d+)°\).*inside the (\w+) quadrant", text)
    R, theta, name = int(m.group(1)), int(m.group(2)), m.group(3)
    lo = ["first", "second", "third", "fourth"].index(name) * 90
    r = round(R ** (1 / 3))
    assert r**3 == R
    hits = []
    for k in range(3):
        ang = (theta + 360 * k) / 3
        if lo < ang < lo + 90:
            hits.append(ang)
    assert len(hits) == 1
    return {"r": r, "theta": round(hits[0])}


def _oracle_equating(text: str):
    cond = re.search(r"real and ([xy]) > 0", text).group(1)
    grid = range(-14, 15)
    sols = []
    if "z^2 =" in text:
        w = _parse_c(re.search(rf"z\^2 = ({_C}), find", text).group(1))
        for x in grid:
            for y in grid:
                if (x * x - y * y, 2 * x * y) == w:
                    sols.append((x, y))
    else:
        k = int(re.search(r"plus (\d+) times z", text).group(1))
        w = _parse_c(re.search(rf"equals ({_C}), find", text).group(1))
        for x in grid:
            for y in grid:
                if (x * x + y * y + k * x, k * y) == w:
                    sols.append((x, y))
    sols = [s for s in sols if (s[0] if cond == "x" else s[1]) > 0]
    assert len(sols) == 1, (text, sols)
    return {"x": sols[0][0], "y": sols[0][1]}


def _key_of(q):
    if q.topic == "exd_x_complex_i_power":
        return _parse_c(q.answer_text)
    return {k: int(v) for k, v in q.answer.items()}


_ORACLES = {
    "exd_x_complex_point": _oracle_point,
    "exd_x_complex_i_power": _oracle_i_power,
    "exd_x_complex_nth_root": _oracle_nth_root,
    "exd_x_complex_equating": _oracle_equating,
}


@pytest.mark.parametrize("topic", sorted(TOPICS))
def test_key_matches_independent_oracle_over_400_seeds(topic):
    bad = []
    for seed in range(400):
        q = _q(topic, seed)
        expected = _ORACLES[topic](q.text)
        if _key_of(q) != expected:
            bad.append((seed, q.text, q.answer_text, expected))
    assert bad == [], bad[:5]


@pytest.mark.parametrize("topic", sorted(TOPICS))
def test_prompt_variety_and_new_shapes_over_hard_tier(topic):
    shapes = {prompt_shape(_q(topic, s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    old_topic, old_params, old_diff = TOPICS[topic][1]
    old_shapes = {prompt_shape(make(old_topic, seed=s, difficulty=old_diff, **old_params)) for s in range(60)}
    assert shapes - old_shapes, (shapes, old_shapes)


@pytest.mark.parametrize("topic", sorted(TOPICS))
def test_no_text_defects_and_deterministic(topic):
    for seed in range(60):
        q = _q(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == _q(topic, seed).to_json()


@pytest.mark.parametrize("topic", sorted(TOPICS))
def test_real_grader_accepts_key_and_rejects_every_distractor(topic):
    grader = GRADERS[TOPICS[topic][0]]
    bad = []
    for seed in range(150):
        q = _q(topic, seed)
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3
        if not grader(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        for d in q.distractors:
            if grader(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:5]


def test_both_variants_of_each_topic_occur():
    """Each topic draws from more than one question shape; none is dead code."""
    point = {("parallelogram" in _q("exd_x_complex_point", s).text) for s in range(60)}
    roots = {("Two of the cube roots" in _q("exd_x_complex_nth_root", s).text) for s in range(60)}
    powers = {("(1 " in _q("exd_x_complex_i_power", s).text) for s in range(60)}
    equating = {("conjugate" in _q("exd_x_complex_equating", s).text) for s in range(60)}
    assert point == roots == powers == equating == {True, False}


def test_unknown_tier_rejected():
    for topic in TOPICS:
        with pytest.raises(ValueError):
            REGISTRY[topic][1](random.Random(0), tier="easy")
