"""Independent oracle for `pv1f_ftc_derivative_read`, re-deriving F'(x0) from the drawn curve.

Imports NOTHING from ``mathgen.generators.pv1f_ftc_build``. Every question is generated
through the skill's own sources; the lower limit a and the point x0 are parsed from the
RENDERED PROMPT, the curve is the one drawn (or, when the prompt names a lettered point, the
one that point lies on), and the key is re-derived by NUMERICAL CALCULUS on the sampled
polyline, not by reading a height:

- A(x) = the integral of the polyline from a to x by the trapezoid rule (reversed limits
  give a negative area);
- F'(x0) = (A(x0 + h) - A(x0 - h)) / 2h with h = 1/8, rounded to the nearest integer;
- F(x0) = A(x0) rounded: the area the learner must NOT give, which must be one of the
  distractors and must differ from the key.

1000 seeds per tier. ``PV1F_ORACLE_BIAS=1`` adds one to the derived value, which must turn the whole file RED.
"""

import functools
import os
import re

import pytest

from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "senior-understand-and-use-the-theorem-f-left-x"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = 1000
BIAS = int(os.environ.get("PV1F_ORACLE_BIAS", "0"))
H = 0.125


@functools.lru_cache(maxsize=1)
def _skill():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _gen(tier, seed):
    spec = next(s for s in _skill().sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=spec.difficulty, **(spec.params or {}))


def _height(segment_pts, x):
    (x0, y0), (x1, y1) = segment_pts
    return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _integral(curve, lo, hi) -> float:
    """Trapezoid integral of the polyline over [lo, hi], lo <= hi."""
    total = 0.0
    for piece in curve["pieces"]:
        for (x0, y0), (x1, y1) in zip(piece, piece[1:]):
            a, b = max(lo, x0), min(hi, x1)
            if a < b:
                total += (b - a) * (_height(((x0, y0), (x1, y1)), a) + _height(((x0, y0), (x1, y1)), b)) / 2
    return total


def _area(curve, a, x) -> float:
    return _integral(curve, a, x) if a <= x else -_integral(curve, x, a)


def _on(curve, pt) -> bool:
    px, py = pt
    for piece in curve["pieces"]:
        for p, q in zip(piece, piece[1:]):
            if p[0] <= px <= q[0] and abs(_height((p, q), px) - py) < 1e-6:
                return True
    return False


def _parse(q):
    m = re.search(r"F'\((-?\d+)\)", q.text)
    x0 = int(m.group(1))
    a = int(re.search(r"t = (-?\d+)", q.text).group(1))
    return a, x0


def _curve_of(q):
    fig = q.figure
    m = re.search(r"point marked ([A-Z])\b", q.text)
    if m is None:
        assert len(fig["curves"]) == 1
        return fig["curves"][0]
    pt = next(p["at"] for p in fig["points"] if p["label"] == m.group(1))
    on = [c for c in fig["curves"] if _on(c, pt)]
    assert len(on) == 1, "the lettered point must lie on exactly one curve"
    return on[0]


def _derive(q):
    a, x0 = _parse(q)
    curve = _curve_of(q)
    rate = (_area(curve, a, x0 + H) - _area(curve, a, x0 - H)) / (2 * H)
    return round(rate) + BIAS, round(_area(curve, a, x0))


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_rederives_the_key_by_differentiating_the_area_1000_seeds(tier):
    for seed in range(SEEDS):
        q = _gen(tier, seed)
        key, _area_value = _derive(q)
        assert str(key) == q.answer_text, (tier, seed, q.text)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_the_area_distractor_is_offered_and_differs_from_the_key(tier):
    for seed in range(300):
        q = _gen(tier, seed)
        key, area = _derive(q)
        key -= BIAS
        assert area != key and area != 0
        assert str(area) in q.distractors, (tier, seed, area, q.distractors)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_derivative_does_not_depend_on_the_lower_limit(tier):
    """F'(x0) = f(x0) whatever a is: moving the limit changes the area, never the rate."""
    for seed in range(60):
        q = _gen(tier, seed)
        a, x0 = _parse(q)
        curve = _curve_of(q)
        for other in (a + 1, a - 1):
            if abs(other - x0) < 1:
                continue
            rate = (_area(curve, other, x0 + H) - _area(curve, other, x0 - H)) / (2 * H)
            assert round(rate) == int(q.answer_text), (tier, seed, other)


def test_oracle_is_red_on_a_hand_built_wrong_figure():
    q = _gen("easy", 7)
    key, _ = _derive(q)
    bent = {
        **q.figure,
        "curves": [{"pieces": [[[x, y + 2] for x, y in piece] for piece in q.figure["curves"][0]["pieces"]]}],
    }
    q.figure = bent
    assert _derive(q)[0] != key
