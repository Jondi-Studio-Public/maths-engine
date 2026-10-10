"""Independent oracle for the `pv1f_cubic` questions: it re-derives each answer from the rendered question text.

Imports NOTHING from ``mathgen.generators.pv1f_cubic_build``. Every question is generated
through the skill's own sources (``load_skills()`` -> ``make``); the key is then re-derived
FROM THE FIGURE by a route that is not the generator's reader: the sampled polyline of the
chosen curve is fitted EXACTLY (Lagrange through four samples, in ``Fraction``), every
sample is confirmed to lie on that one cubic, and the y-intercept is the constant term of
the fitted cubic. On the exam tier the curve is the one the lettered point P lies on, named
in the PROMPT TEXT.

1000 seeds per tier. ``PV1F_ORACLE_BIAS=1`` adds one to the derived value, which must turn the whole file RED.
"""

import functools
import os
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "senior-recognise-features-of-the-graphs-of-y-x-acmmm017"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = 1000
BIAS = int(os.environ.get("PV1F_ORACLE_BIAS", "0"))


@functools.lru_cache(maxsize=1)
def _skill():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _gen(tier, seed):
    spec = next(s for s in _skill().sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=spec.difficulty, **(spec.params or {}))


def _frac(v) -> Fraction:
    return Fraction(str(v))


def _fit(curve):
    """Exact (a3, a2, a1, a0) with y = a3 x^3 + a2 x^2 + a1 x + a0 through the integer-x
    samples (those are exact); the other samples are rounded to six places by the figure
    spec, so they are checked to within 1e-5."""
    pts = [(_frac(x), _frac(y)) for piece in curve["pieces"] for x, y in piece]
    ints = [(x, y) for x, y in pts if x.denominator == 1]
    assert len(ints) >= 4, "fewer than four integer-x samples visible"
    chosen = [ints[0], ints[len(ints) // 3], ints[2 * len(ints) // 3], ints[-1]]
    assert len({p[0] for p in chosen}) == 4
    # Solve the 4x4 Vandermonde system exactly (Gauss-Jordan over Fractions).
    m = [[x**3, x**2, x, Fraction(1), y] for x, y in chosen]
    for col in range(4):
        piv = next(r for r in range(col, 4) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        m[col] = [v / m[col][col] for v in m[col]]
        for r in range(4):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    coeffs = tuple(m[i][4] for i in range(4))
    a3, a2, a1, a0 = coeffs
    for x, y in pts:
        fitted = a3 * x**3 + a2 * x**2 + a1 * x + a0
        if x.denominator == 1:
            assert fitted == y, "integer-x samples do not lie on a single cubic"
        else:
            assert abs(fitted - y) < Fraction(1, 100000), "samples do not lie on a single cubic"
    assert a3 != 0
    return coeffs


def _at(fit, x):
    a3, a2, a1, a0 = fit
    x = _frac(x)
    return a3 * x**3 + a2 * x**2 + a1 * x + a0


def _chosen(q):
    """(the fitted curve the prompt asks about, the fits of the other drawn curves)."""
    fig = q.figure
    fits = [_fit(c) for c in fig["curves"]]
    m = re.search(r"point marked ([A-Z])\b", q.text)
    if m is None:
        assert len(fits) == 1
        return fits[0], []
    pt = next(p["at"] for p in fig["points"] if p["label"] == m.group(1))
    on = [f for f in fits if _at(f, pt[0]) == _frac(pt[1])]
    assert len(on) == 1, "the named point must lie on exactly one fitted cubic"
    return on[0], [f for f in fits if f is not on[0]]


def _derive(q) -> int:
    chosen, _others = _chosen(q)
    a0 = chosen[3]
    assert a0.denominator == 1, "the y-intercept is not an integer"
    return int(a0) + BIAS


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_rederives_the_key_from_the_figure_1000_seeds(tier):
    for seed in range(SEEDS):
        q = _gen(tier, seed)
        assert str(_derive(q)) == q.answer_text, (tier, seed, q.text)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_every_distractor_is_a_different_reading_of_the_picture(tier):
    """Each distractor differs from the fitted intercept, and on the hard tier the line's
    height and on the exam tier the other curve's intercept are genuinely offered."""
    for seed in range(200):
        q = _gen(tier, seed)
        key = _derive(q) - BIAS
        assert all(int(d) != key for d in q.distractors)
        if tier == "hard":
            (line,) = q.figure["lines"]
            assert str(line["through"][0][1]) in q.distractors
        if tier == "exam":
            others = [int(_fit(c)[3]) for c in q.figure["curves"]]
            assert sorted(others).count(key) == 1
            assert str(next(o for o in others if o != key)) in q.distractors


def _readings_of(fit):
    """Every wrong value the picture of THIS curve invites: the negated intercept, the height
    one and two squares off the axis, the integer x-intercepts and their sum, the height of the
    centre of the S (the inflection) and the combinations of it with the intercept."""
    a3, a2, a1, a0 = fit
    out = {-a0, _at(fit, 1), _at(fit, -1), _at(fit, 2), _at(fit, -2)}
    roots = [Fraction(x) for x in range(-10, 11) if _at(fit, x) == 0]
    out.update(roots)
    out.add(sum(roots))
    centre = _at(fit, -a2 / (3 * a3))
    out.update({centre, centre - a0, 2 * centre - a0})
    return out


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_every_distractor_is_a_reading_of_the_drawn_curve(tier):
    """Distractors must come from the curve (or line, or other curve) the learner can see:
    a lure computed from a curve that is not on the grid is a phantom reading."""
    for seed in range(300):
        q = _gen(tier, seed)
        chosen, others = _chosen(q)
        allowed = _readings_of(chosen)
        allowed |= {o[3] for o in others}
        allowed |= {_frac(ln["through"][0][1]) for ln in q.figure["lines"]}
        for d in q.distractors:
            assert _frac(d) in allowed, (tier, seed, d, sorted(allowed))


def test_oracle_is_red_on_a_hand_built_wrong_figure():
    """A figure whose curve crosses the y-axis one unit off the keyed value must be caught."""
    q = _gen("easy", 5)
    key = q.answer_text
    bent = {
        **q.figure,
        "curves": [{"pieces": [[[x, y + 1] for x, y in piece] for piece in q.figure["curves"][0]["pieces"]]}],
    }
    q.figure = bent
    assert str(_derive(q)) != key
