"""Independent oracle for the `pv1d_parabola` questions: it re-derives each answer from the rendered question
text.

Imports NOTHING from ``mathgen.generators.pv1d_parabola_build``. Every question is
generated through the skill's own sources (``load_skills()`` -> ``make``); the key is then
re-derived FROM THE FIGURE by a route that is not the generator's reader: the sampled
polyline of the chosen curve is fitted EXACTLY (Lagrange through three samples, in
``Fraction``), every sample is confirmed to lie on that one quadratic, and then

- root count = sign of the discriminant B^2 - 4AC of the fitted quadratic;
- axis of symmetry = -B / (2A) of the fitted quadratic.

1000 seeds per tier, every tier, both skills. ``PV1D_ORACLE_BIAS=1`` adds one to the derived value, which must
turn the whole file RED.
"""

import functools
import os
import re
from fractions import Fraction
from math import isqrt

import pytest

from mathgen.skills import load_skills
from mathgen import figures, make

ROOT_ID = "senior-recognise-features-of-the-graph-of-the-general"
AXIS_ID = "senior-recognise-features-of-the-graphs-of-y-x"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = 1000
BIAS = int(os.environ.get("PV1D_ORACLE_BIAS", "0"))


@functools.lru_cache(maxsize=1)
def _skills():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}


def _gen(skill_id: str, tier: str, seed: int):
    skill = _skills()[skill_id]
    spec = next(s for s in skill.sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=spec.difficulty, **(spec.params or {}))


def _frac(v) -> Fraction:
    return Fraction(str(v))


def _fit(curve: dict):
    """Exact (A, B, C) with y = A x^2 + B x + C through the curve, or fail."""
    pts = [(_frac(x), _frac(y)) for piece in curve["pieces"] for x, y in piece]
    assert len(pts) >= 3
    (x0, y0), (x1, y1), (x2, y2) = pts[0], pts[len(pts) // 2], pts[-1]
    assert len({x0, x1, x2}) == 3
    d0, d1, d2 = (x0 - x1) * (x0 - x2), (x1 - x0) * (x1 - x2), (x2 - x0) * (x2 - x1)
    A = y0 / d0 + y1 / d1 + y2 / d2
    B = -(y0 * (x1 + x2) / d0 + y1 * (x0 + x2) / d1 + y2 * (x0 + x1) / d2)
    C = y0 * x1 * x2 / d0 + y1 * x0 * x2 / d1 + y2 * x0 * x1 / d2
    for x, y in pts:
        assert A * x * x + B * x + C == y, "samples do not lie on a single quadratic"
    assert A != 0
    return A, B, C


def _at(fit, x):
    A, B, C = fit
    return A * _frac(x) ** 2 + B * _frac(x) + C


def _disc(fit) -> Fraction:
    A, B, C = fit
    return B * B - 4 * A * C


def _roots(fit) -> list:
    A, B, _C = fit
    d = _disc(fit)
    if d < 0:
        return []
    num, den = d.numerator, d.denominator
    rn, rd = isqrt(num), isqrt(den)
    if rn * rn != num or rd * rd != den:
        return [None, None]  # irrational roots: off the grid by definition
    s = Fraction(rn, rd)
    return sorted({(-B - s) / (2 * A), (-B + s) / (2 * A)})


def _count(fit) -> int:
    d = _disc(fit)
    return 0 if d < 0 else (1 if d == 0 else 2)


def _vertex(fit):
    A, B, C = fit
    vx = -B / (2 * A)
    return vx, A * vx * vx + B * vx + C


def _is_int(v) -> bool:
    return Fraction(v).denominator == 1


def _letter(text: str):
    m = re.search(r"point marked ([A-Z])\b", text)
    return m.group(1) if m else None


# ---------------------------------------------------------------- row 1: root count


def _chosen_curve_and_fit(q):
    fig = q.figure
    fits = [_fit(c) for c in fig["curves"]]
    letter = _letter(q.text)
    if letter is None:
        assert len(fits) == 1
        return 0, fits
    pt = next(p["at"] for p in fig["points"] if p["label"] == letter)
    on = [i for i, f in enumerate(fits) if _at(f, pt[0]) == _frac(pt[1])]
    assert len(on) == 1, f"point {letter} lies on {len(on)} curves"
    return on[0], fits


def _root_derived(q) -> int:
    i, fits = _chosen_curve_and_fit(q)
    return _count(fits[i]) + BIAS


def _root_problems(q, tier: str) -> list:
    """Every boundary rule, asserted from the figure alone."""
    fig = q.figure
    problems = []
    i, fits = _chosen_curve_and_fit(q)
    fit = fits[i]
    n = _count(fit)
    vx, vy = _vertex(fit)
    if n == 1 and not (_is_int(vx) and vy == 0):
        problems.append("tangent touch is not on a grid intersection of the x-axis")
    for r in _roots(fit):
        if r is None or not (_is_int(r) and abs(r) <= 9):
            problems.append(f"crossing at {r} is not an integer at least one unit inside the window")
    if tier in ("easy", "medium") and not (_is_int(vx) and _is_int(vy) and abs(vx) <= 10 and abs(vy) <= 10):
        problems.append("vertex is not a grid intersection inside the window")
    if tier == "easy" and n != 1 and abs(vy) < 2:
        problems.append("easy vertex is closer than 2 units to the x-axis")
    if tier == "medium" and abs(vy) > 1:
        problems.append("medium vertex is further than 1 unit from the x-axis")
    if tier == "medium" and n != 1 and abs(vy) != 1:
        problems.append("medium non-tangent vertex is not exactly 1 unit off the axis")
    if (tier in ("easy", "medium")) and (fig["lines"] or fig["points"]):
        problems.append("easy/medium figure carries a line or a point")
    if tier == "hard":
        if len(fig["lines"]) != 1 or fig["lines"][0].get("label") != "L" or fig["points"]:
            problems.append("hard figure needs exactly one labelled line L and no points")
        else:
            (x0, y0), (x1, y1) = fig["lines"][0]["through"]
            m = _frac(y0)
            if y0 != y1 or m == 0:
                problems.append("L is not a horizontal line other than the x-axis")
            else:
                A, B, C = fit
                shifted = (A, B, C - m)
                if _count(shifted) == n:
                    problems.append("L meets the parabola as often as the x-axis does")
                if any(r is None or not (_is_int(r) and abs(r) <= 9) for r in _roots(shifted)):
                    problems.append("a crossing with L is not a visible integer x")
    if tier == "exam":
        if len(fits) != 2 or len(fig["points"]) != 2 or fig["lines"]:
            problems.append("exam figure needs two curves, two lettered points, no line")
        else:
            other = fits[1 - i]
            if _count(other) == n:
                problems.append("the other curve has the same root count")
            for p in fig["points"]:
                px, py = _frac(p["at"][0]), _frac(p["at"][1])
                if not (_is_int(px) and _is_int(py)):
                    problems.append("lettered point is off the grid")
                owners = [j for j, f in enumerate(fits) if _at(f, px) == py]
                if len(owners) != 1:
                    problems.append("lettered point is not on exactly one curve")
                    continue
                f = fits[owners[0]]
                if px == _vertex(f)[0] or py == 0:
                    problems.append("lettered point is a vertex or on the x-axis")
                if abs(_at(fits[1 - owners[0]], px) - py) < 1:
                    problems.append("lettered point is within 1 unit of the other curve")
    opts = [q.answer_text, *q.distractors]
    if sorted(opts) != ["0", "1", "2"]:
        problems.append(f"mcq options are {opts}")
    return problems


@pytest.mark.parametrize("tier", TIERS)
def test_root_count_key_rederived_from_figure_1000_seeds(tier):
    for seed in range(SEEDS):
        q = _gen(ROOT_ID, tier, seed)
        assert str(_root_derived(q)) == q.answer_text, (tier, seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_root_count_boundaries_hold_from_the_figure(tier):
    for seed in range(SEEDS):
        q = _gen(ROOT_ID, tier, seed)
        problems = _root_problems(q, tier)
        assert not problems, (tier, seed, problems)


def test_root_boundary_predicate_is_red_on_an_unconstrained_draw():
    """A tangent parabola with a half-integer vertex x, and a root just off the grid, are
    flagged: the boundary check is not vacuous."""

    def fake(a, h, c, tier):
        curve = {"pieces": figures.sample_curve(lambda x: a * (x - h) ** 2 + c, -10, 10, (-10, 10), n=161)}
        fig = figures.cartesian(
            x_range=(-10, 10), y_range=(-10, 10), alt="a parabola on a coordinate grid", curves=[curve]
        )

        class Q:
            figure = fig
            text = "The graph of the quadratic y = ax^2 + bx + c is shown."
            answer_text = "1"
            distractors = ["0", "2"]

        return Q

    assert _root_problems(fake(1, 2.5, 0, "easy"), "easy")  # half-integer tangent
    assert _root_problems(fake(1, 0, -2, "easy"), "easy")  # roots +-sqrt(2): off the grid
    assert not _root_problems(fake(1, 3, 0, "easy"), "easy")  # a legal tangent passes


# ---------------------------------------------------------------- row 2: axis of symmetry


def _axis_derived(q) -> str:
    fit = _fit(q.figure["curves"][0])
    A, B, _C = fit
    axis = -B / (2 * A) + BIAS
    assert _is_int(axis), f"axis {axis} is not an integer"
    return f"x = {int(axis)}"


def _axis_problems(q, tier: str) -> list:
    fig = q.figure
    problems = []
    fit = _fit(fig["curves"][0])
    vx, vy = _vertex(fit)
    pts = fig["points"]
    if not _is_int(vx) or vx == 0:
        problems.append("axis is not a non-zero integer")
    if len(fig["curves"]) != 1 or fig["lines"]:
        problems.append("figure must be one curve and no line")
    for p in pts:
        px, py = _frac(p["at"][0]), _frac(p["at"][1])
        if not (_is_int(px) and _is_int(py)):
            problems.append("marked point is off the grid")
        if _at(fit, px) != py:
            problems.append("marked point is not on the curve")
        if len(p["label"]) != 1 or not p["label"].isupper():
            problems.append("point label is not a single capital")
    inside = abs(vx) <= 10 and abs(vy) <= 10
    if tier in ("easy", "medium"):
        if not (inside and _is_int(vy)):
            problems.append("vertex is not a grid intersection inside the window")
        if vy == 0 or vy in (vx, -vx):
            problems.append("vertex height collides with an axis distractor")
    if tier == "easy":
        if [p["label"] for p in pts] != ["V"] or _frac(pts[0]["at"][0]) != vx or _frac(pts[0]["at"][1]) != vy:
            problems.append("easy needs the single point V at the vertex")
    if tier == "medium" and pts:
        problems.append("medium has a marked point")
    if tier in ("hard", "exam"):
        if inside:
            problems.append("vertex is inside the window")
        n_left = sum(1 for piece in fig["curves"][0]["pieces"] for x, _ in piece if _frac(x) < vx)
        n_right = sum(1 for piece in fig["curves"][0]["pieces"] for x, _ in piece if _frac(x) > vx)
        if n_left < 4 or n_right < 4:
            problems.append("an arm of the parabola is not visible")
        for p in pts:
            if _frac(p["at"][0]) == vx:
                problems.append("a marked point is the vertex")
        heights = [_frac(p["at"][1]) for p in pts]
        pairs = [(i, j) for i in range(len(pts)) for j in range(i + 1, len(pts)) if heights[i] == heights[j]]
        if len(pairs) != 1:
            problems.append(f"{len(pairs)} equal-height pairs")
        elif _frac(pts[pairs[0][0]]["at"][0]) + _frac(pts[pairs[0][1]]["at"][0]) != 2 * vx:
            problems.append("the equal-height pair is not symmetric about the axis")
        if tier == "hard" and sorted(p["label"] for p in pts) != ["P", "Q"]:
            problems.append("hard needs exactly P and Q")
        if tier == "exam":
            if sorted(p["label"] for p in pts) != ["P", "Q", "R"]:
                problems.append("exam needs exactly P, Q and R")
            odd = [k for k in range(3) if all(k not in pr for pr in pairs)]
            if len(odd) != 1 or _frac(pts[odd[0]]["at"][0]) == vx:
                problems.append("the odd point must exist and not lie on the axis")
    return problems


@pytest.mark.parametrize("tier", TIERS)
def test_axis_key_rederived_from_figure_1000_seeds(tier):
    for seed in range(SEEDS):
        q = _gen(AXIS_ID, tier, seed)
        assert _axis_derived(q) == q.answer_text, (tier, seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_axis_boundaries_hold_from_the_figure(tier):
    for seed in range(SEEDS):
        q = _gen(AXIS_ID, tier, seed)
        problems = _axis_problems(q, tier)
        assert not problems, (tier, seed, problems)


def test_axis_boundary_predicate_is_red_on_an_unconstrained_draw():
    """A hard-tier figure whose vertex is inside the window, and a marked point that is
    not on the curve, are flagged."""

    def fake(a, h, c, pts, tier):
        curve = {"pieces": figures.sample_curve(lambda x: a * (x - h) ** 2 + c, -10, 10, (-10, 10), n=161)}
        fig = figures.cartesian(
            x_range=(-10, 10), y_range=(-10, 10), alt="a parabola on a coordinate grid", curves=[curve], points=pts
        )

        class Q:
            figure = fig

        return Q

    on = [{"at": [1, 1], "label": "P"}, {"at": [5, 1], "label": "Q"}]
    assert _axis_problems(fake(1, 3, 0, on, "hard"), "hard")  # vertex in the window
    assert _axis_problems(fake(1, 3, 0, [{"at": [1, 2], "label": "P"}], "easy"), "easy")  # off the curve
    legal = [{"at": [3, -4], "label": "V"}]
    assert not _axis_problems(fake(1, 3, -4, legal, "easy"), "easy")  # a legal easy figure passes
