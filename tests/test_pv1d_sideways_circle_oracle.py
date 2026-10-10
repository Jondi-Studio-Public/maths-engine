"""Independent oracle for the sideways-parabola and circle figure skills of pv1d.

Imports nothing from ``mathgen/generators/pv1d_sideways_circle_build.py``. Questions are
generated through each skill's own sources; the key is then recomputed from the figure spec
(and, only where the prompt selects a circle by a letter, from the letter named in the
prompt text):

- sideways parabola: an exact (``Fraction``) quadratic fit ``x = A y^2 + B y + C`` through
  three decimal-exact sampled points gives the axis ``y = -B / 2A``; the marked points are
  then checked to sit on that curve, and on hard/exam the pair sharing an x is checked to
  have that midpoint.
- circle: the circumcentre of three sampled points (a closed-form solve, not the bounding
  box the reader uses) gives each circle's centre and radius.

The boundary rules are asserted from the figure alone. Set
``PV1D_CIRCLE_ORACLE_CORRUPT=1`` to add one to every derived key: the oracle must then fail
(this is how it is shown red).
"""

import functools
import math
import os
import random
import re
from fractions import Fraction
from types import SimpleNamespace

import pytest

from mathgen.skills import load_skills
from mathgen import figures, make

DIFFICULTIES = ["easy", "medium", "hard", "exam"]
PARABOLA = "senior-recognise-features-of-the-graph-of-y-2"
CIRCLE = "senior-recognise-features-of-the-graphs-of-x-2"
SEEDS = 1000
_CORRUPT = int(os.environ.get("PV1D_CIRCLE_ORACLE_CORRUPT", "0"))


@functools.lru_cache(maxsize=None)
def _skills():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make(skill_id, difficulty, seed):
    spec = next(s for s in _skills()[skill_id].sources if s.difficulty == difficulty)
    return make(spec.topic, seed=seed, difficulty=spec.difficulty, **(spec.params or {}))


def _frac(v):
    return Fraction(str(v))


def _all_samples(fig):
    return [(p[0], p[1]) for c in fig["curves"] for piece in c["pieces"] for p in piece]


# =============================================================================== parabola


def _fit_axis(fig):
    """(A, k): the exact quadratic fit x = A y^2 + B y + C through three sampled points that
    are decimal-exact (the sqrt places where a curve meets the window edge are not), and the
    axis height k = -B / 2A."""
    exact = []
    for x, y in _all_samples(fig):
        fx, fy = _frac(x), _frac(y)
        if (fx * 100).denominator == 1 and (fy * 10).denominator == 1:
            exact.append((fx, fy))
    exact.sort(key=lambda p: p[1])
    ys = sorted({y for _, y in exact})
    assert len(ys) >= 3, "fewer than three decimal-exact samples to fit"
    pick = [next(p for p in exact if p[1] == ys[i]) for i in (0, len(ys) // 2, len(ys) - 1)]
    (x1, y1), (x2, y2), (x3, y3) = pick
    f12, f23 = (x1 - x2) / (y1 - y2), (x2 - x3) / (y2 - y3)
    A = (f12 - f23) / (y1 - y3)
    B = f12 - A * (y1 + y2)
    return A, -B / (2 * A)


def _parabola_problems(q, tier):
    fig, problems = q.figure, []
    A, k = _fit_axis(fig)
    if abs(A) != 1:
        problems.append(f"the curve is not (y-k)^2 = +-x: leading coefficient {A}")
    if k.denominator != 1:
        problems.append(f"axis height {k} is not a whole number")
    x0, x1 = fig["x_range"]
    y0, y1 = fig["y_range"]
    samples = {(_frac(x), _frac(y)) for x, y in _all_samples(fig)}
    marked = [(p["label"], _frac(p["at"][0]), _frac(p["at"][1])) for p in fig["points"]]
    for label, px, py in marked:
        if px.denominator != 1 or py.denominator != 1:
            problems.append(f"{label} is not on a grid intersection")
        if px != A * (py - k) ** 2:
            problems.append(f"{label} is not on the curve")
        if (px, py) not in samples:
            problems.append(f"{label} is not one of the sampled curve points")
    vertex_inside = x0 < 0 < x1 and y0 < k < y1
    if tier in ("easy", "medium") and not vertex_inside:
        problems.append("vertex is not inside the window")
    if tier in ("hard", "exam") and x0 <= 0 <= x1:
        problems.append("vertex x = 0 is inside the window on a tier that hides it")
    pieces = [pc for c in fig["curves"] for pc in c["pieces"]]
    if tier in ("hard", "exam"):
        ys = [_frac(y) for _, y in _all_samples(fig)]
        if len(pieces) < 2 or not (any(y > k for y in ys) and any(y < k for y in ys)):
            problems.append("both arms are not visible")
    by_x = {}
    for label, px, py in marked:
        by_x.setdefault(px, []).append(py)
    shared = [ys for ys in by_x.values() if len(ys) == 2]
    if tier == "easy" and [m[0] for m in marked] != ["V"]:
        problems.append("easy should mark exactly the vertex V")
    if tier == "easy" and marked and (marked[0][1], marked[0][2]) != (0, k):
        problems.append("V is not at (0, k)")
    if tier == "medium" and marked:
        problems.append("medium marks a point")
    if tier == "hard" and not (len(marked) == 2 and len(shared) == 1):
        problems.append("hard needs two marked points sharing an x")
    if tier == "exam" and not (len(marked) == 3 and len(shared) == 1 and len(by_x) == 2):
        problems.append("exam needs three marked points with exactly one pair sharing an x")
    if tier in ("hard", "exam") and len(shared) == 1 and sum(shared[0]) / 2 != k:
        problems.append("the shared-x pair is not centred on the axis")
    return k, marked, by_x, problems


@pytest.mark.parametrize("tier", DIFFICULTIES)
def test_parabola_key_is_rederived_from_the_figure_at_1000_seeds(tier):
    for seed in range(SEEDS):
        q = _make(PARABOLA, tier, seed)
        k, marked, by_x, problems = _parabola_problems(q, tier)
        assert not problems, (tier, seed, problems, q.text)
        assert f"y = {int(k) + _CORRUPT}" == q.answer_text, (tier, seed, k, q.answer_text)
        if tier == "exam":
            odd_y = next(ys[0] for ys in by_x.values() if len(ys) == 1)
            pair_ys = next(ys for ys in by_x.values() if len(ys) == 2)
            wrong = {f"y = {int((y + odd_y) / 2)}" for y in pair_ys}
            assert wrong <= set(q.distractors), (seed, wrong, q.distractors)
            assert f"y = {-int(k)}" in q.distractors
        assert not any(d == q.answer_text for d in q.distractors)


# =============================================================================== circle


def _circumcircle(p1, p2, p3):
    (ax, ay), (bx, by), (cx, cy) = p1, p2, p3
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return ux, uy, math.hypot(ax - ux, ay - uy)


def _round_int(v, what):
    n = round(v)
    assert abs(v - n) < 1e-4, f"{what} = {v} is not a whole number"
    return int(n)


def _circles_of(fig):
    out = []
    for c in fig["curves"]:
        pts = c["pieces"][0]
        assert len(c["pieces"]) == 1
        ux, uy, r = _circumcircle(pts[5], pts[70], pts[140])
        # every sampled point is on that circle, and it closes up
        assert all(abs(math.hypot(x - ux, y - uy) - r) < 1e-4 for x, y in pts)
        assert pts[0] == pts[-1]
        out.append((_round_int(ux, "centre x"), _round_int(uy, "centre y"), _round_int(r, "radius")))
    return out


def _asked_letter(text, labels):
    found = [m.group(1) for m in re.finditer(r"\b([A-Z])\b", text) if m.group(1) in labels]
    return found[-1] if found else None


def _circle_problems(q, tier):
    fig, problems = q.figure, []
    circles = _circles_of(fig)
    x0, x1 = fig["x_range"]
    y0, y1 = fig["y_range"]
    labelled = {p["label"]: tuple(p["at"]) for p in fig["points"]}
    for a, b, r in circles:
        if r < 1:
            problems.append("radius below 1")
        if not (x0 < a - r and a + r < x1 and y0 < b - r and b + r < y1):
            problems.append(f"circle ({a}, {b}) r={r} is not wholly inside the window")
    for label, (px, py) in labelled.items():
        if px != int(px) or py != int(py):
            problems.append(f"{label} is not on a grid intersection")
        owners = [c for c in circles if abs(math.hypot(px - c[0], py - c[1]) - c[2]) < 1e-4]
        if tier == "hard" and len(owners) != 1:
            problems.append(f"{label} lies on {len(owners)} circles")
    if tier in ("easy", "medium"):
        if len(circles) != 1:
            return None, [*problems, "expected one circle"]
        key = circles[0][:2]
        if tier == "easy" and labelled != {"C": key}:
            problems.append("easy should mark exactly the centre as C")
        if tier == "medium" and labelled:
            problems.append("medium marks a point")
        return key, problems
    if len(circles) != 2:
        return None, [*problems, "expected two circles"]
    (a, b, r), (c, d, s) = circles
    if not math.hypot(a - c, b - d) > r + s:
        problems.append("circles intersect or one contains the other")
    if tier == "hard":
        if len(labelled) != 2:
            problems.append("hard needs two lettered points")
        asked = _asked_letter(q.text, labelled)
        if asked is None:
            return None, [*problems, "no lettered point named in the prompt"]
        px, py = labelled[asked]
        key = next(cc[:2] for cc in circles if abs(math.hypot(px - cc[0], py - cc[1]) - cc[2]) < 1e-4)
        other = next(cc[:2] for cc in circles if cc[:2] != key)
        return key, problems, other
    # exam: exactly one circle touches the x-axis
    if labelled:
        problems.append("exam marks a point")
    tangent = [cc for cc in circles if abs(cc[1]) == cc[2]]
    if len(tangent) != 1:
        problems.append(f"{len(tangent)} circles touch the x-axis")
        return None, problems
    for cc in circles:
        if abs(cc[1]) < cc[2]:
            gap = cc[2] ** 2 - cc[1] ** 2
            if math.isqrt(gap) ** 2 != gap:
                problems.append("a circle crosses the x-axis at a non-integer point")
    key = tangent[0][:2]
    other = next(cc[:2] for cc in circles if cc[:2] != key)
    return key, problems, other


@pytest.mark.parametrize("tier", DIFFICULTIES)
def test_circle_key_is_rederived_from_the_figure_at_1000_seeds(tier):
    for seed in range(SEEDS):
        q = _make(CIRCLE, tier, seed)
        result = _circle_problems(q, tier)
        key, problems = result[0], result[1]
        assert not problems, (tier, seed, problems, q.text)
        assert f"({key[0] + _CORRUPT}, {key[1]})" == q.answer_text, (tier, seed, key, q.answer_text)
        if tier in ("hard", "exam"):
            other = result[2]
            assert f"({other[0]}, {other[1]})" in q.distractors, (tier, seed, "other circle's centre missing")
        if tier == "exam":
            assert f"({key[0]}, 0)" in q.distractors, (seed, "the point of contact is missing")


def test_exam_tangent_circle_is_not_always_the_larger_left_or_upper_one():
    """Measured at 400 seeds: each of the three properties lands well away from 0 and 1."""
    larger = left = upper = n = 0
    for seed in range(400):
        q = _make(CIRCLE, "exam", seed)
        circles = _circles_of(q.figure)
        t = next(c for c in circles if abs(c[1]) == c[2])
        o = next(c for c in circles if c is not t)
        n += 1
        larger += t[2] > o[2]
        left += t[0] < o[0]
        upper += t[1] > o[1]
    for name, count in (("larger", larger), ("left", left), ("upper", upper)):
        assert 0.2 <= count / n <= 0.8, (name, count, n)


def test_hard_asked_circle_is_not_always_first_larger_or_left():
    larger = left = first_letter = n = 0
    for seed in range(400):
        q = _make(CIRCLE, "hard", seed)
        circles = _circles_of(q.figure)
        labels = {p["label"]: tuple(p["at"]) for p in q.figure["points"]}
        asked = _asked_letter(q.text, labels)
        px, py = labels[asked]
        t = next(c for c in circles if abs(math.hypot(px - c[0], py - c[1]) - c[2]) < 1e-4)
        o = next(c for c in circles if c is not t)
        n += 1
        larger += t[2] > o[2]
        left += t[0] < o[0]
        first_letter += asked == min(labels)
    for name, count in (("larger", larger), ("left", left), ("first letter", first_letter)):
        assert 0.2 <= count / n <= 0.8, (name, count, n)


# ============================================================ boundaries are RED when unconstrained


def _fake_circle_question(circles, labelled, text):
    curves = []
    for a, b, r in circles:
        pts = [[a + r * math.cos(2 * math.pi * i / 192), b + r * math.sin(2 * math.pi * i / 192)] for i in range(192)]
        pts[0], pts[48], pts[96], pts[144] = [a + r, b], [a, b + r], [a - r, b], [a, b - r]
        pts.append(list(pts[0]))
        curves.append({"pieces": [pts]})
    fig = figures.cartesian(
        x_range=(-12, 12),
        y_range=(-12, 12),
        alt="two circles on a coordinate grid",
        points=[{"at": list(at), "label": label} for label, at in labelled],
        curves=curves,
    )
    return SimpleNamespace(figure=fig, text=text, distractors=[])


def test_circle_boundary_rules_flag_an_unconstrained_draw():
    """An unconstrained second circle often overlaps the first: the hard checker must say so
    (while the generated questions above never do)."""
    rng = random.Random(7)
    overlapping = 0
    for _ in range(200):
        c1 = (rng.randint(-4, 4), rng.randint(-4, 4), rng.randint(1, 3))
        c2 = (rng.randint(-4, 4), rng.randint(-4, 4), rng.randint(1, 3))
        overlaps = not math.hypot(c1[0] - c2[0], c1[1] - c2[1]) > c1[2] + c2[2]
        q = _fake_circle_question(
            [c1, c2],
            [("P", (c1[0] + c1[2], c1[1])), ("Q", (c2[0], c2[1] + c2[2]))],
            "Give the centre of the circle through P as (a, b).",
        )
        try:
            flagged = bool(_circle_problems(q, "hard")[1])
        except (AssertionError, StopIteration):
            flagged = True
        if overlaps:
            overlapping += 1
            assert flagged, (c1, c2)
    assert overlapping >= 20, overlapping


def test_exam_boundary_rule_flags_two_tangent_circles_and_none():
    both = _fake_circle_question([(-4, 2, 2), (4, 1, 1)], [], "x")
    none = _fake_circle_question([(-4, 3, 2), (4, 5, 1)], [], "x")
    assert any("circles touch the x-axis" in p for p in _circle_problems(both, "exam")[1])
    assert any("circles touch the x-axis" in p for p in _circle_problems(none, "exam")[1])
    crossing = _fake_circle_question([(-4, 2, 2), (3, 1, 2)], [], "x")  # (3,1) r=2: sqrt(3) crossing
    assert any("non-integer" in p for p in _circle_problems(crossing, "exam")[1])


def test_parabola_boundary_rules_flag_unconstrained_marked_points():
    """Three marked points drawn without the one-shared-x rule fail the exam checker."""
    rng = random.Random(3)
    flagged = 0
    trials = 40
    for _ in range(trials):
        k = rng.choice((-3, -2, 2, 3))
        upper, lower = [], []
        for u10 in range(-34, 35):
            u = Fraction(u10, 10)
            if 3 <= u * u <= 12:
                (upper if u > 0 else lower).append([float(u * u), float(k + u)])
        arms = [lower, upper]
        offsets = [rng.choice((-3, -2, 2, 3)) for _ in range(3)]
        pts = [{"at": [o * o, k + o], "label": lab} for o, lab in zip(offsets, "PQR")]
        fig = figures.cartesian(
            x_range=(3, 12),
            y_range=(-8, 8),
            alt="part of a sideways parabola",
            points=pts,
            curves=[{"pieces": [a for a in arms if len(a) > 1]}],
        )
        q = SimpleNamespace(figure=fig, text="", distractors=[])
        try:
            problems = _parabola_problems(q, "exam")[3]
        except (AssertionError, ZeroDivisionError):
            problems = ["unfittable"]
        flagged += bool(problems)
    assert flagged >= trials // 4, flagged


def test_oracle_catches_a_corrupted_expectation():
    q = _make(PARABOLA, "easy", 0)
    k = _parabola_problems(q, "easy")[0]
    assert f"y = {int(k)}" == q.answer_text
    assert f"y = {int(k) + 1}" != q.answer_text
    c = _make(CIRCLE, "medium", 0)
    key = _circle_problems(c, "medium")[0]
    assert f"({key[0] + 1}, {key[1]})" != c.answer_text
