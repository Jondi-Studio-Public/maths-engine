"""Reads only `question.text` / `question.answer_text` / `question.distractors`
via `mathgen.make`. Imports NOTHING from
`mathgen.generators.p10a_motion_vector_build` -- the direction vector `d`,
`v(0)`, `r(0)`, the acceleration parameters and `T` are all parsed back out
of the rendered prose with regular expressions, then re-derived with fresh
`sympy`/`Fraction` vector arithmetic written here, never by calling into the
generator.
"""

from __future__ import annotations

import re

import pytest
from sympy import integrate, symbols, sympify

from mathgen import make

SEEDS = range(60)
TIERS = ("easy", "medium", "hard")

_t = symbols("t")

TOPIC_1 = "p10a_vector_motion_line_accel"
TOPIC_2 = "p10a_projectile_motion_vector"

# A single i/j vector term as the generator's own `join_terms`/literal
# f-string formatting renders it: an optional sign, optional magnitude (a
# bare "i"/"j" means a coefficient of 1), then the axis letter.
_TERM_RE = re.compile(r"([+-]?)\s*(\d*)\s*([ij])")


def _parse_vector_terms(text: str) -> tuple[int, int]:
    """(i-coefficient, j-coefficient) from a substring made ONLY of i/j
    terms, e.g. "-9i + 8j", "i + j", "3i + 4j"."""
    coeffs = {"i": 0, "j": 0}
    for sign, mag, axis in _TERM_RE.findall(text):
        val = int(mag) if mag else 1
        if sign == "-":
            val = -val
        coeffs[axis] = val
    return coeffs["i"], coeffs["j"]


# --- Skill 1: motion in a straight line -------------------------------------

_SKILL1_RE = re.compile(
    r"parallel to the vector d = (?P<d>.+?)\. "
    r"Its acceleration vector is (?P<g>.+?)·d m/s² at time t seconds \(t ≥ 0\)\. "
    r"At time t = 0, its velocity vector is (?P<f0>-?\d+)d m/s, and (?P<offset>.+?)\. "
    r"Find the (?P<axis>[ij])-component of the particle's position vector, "
    r"in metres, at time t = (?P<T>\d+) (?:second|seconds)\."
)
_R0_RE = re.compile(r"r\(0\) = (?P<r0>.+?) m")


def _parse_g_expr(prose: str):
    """A rendered acceleration coefficient ("6", "-12·t") back into a sympy
    expression in t. `·` is the generator's own stand-in for `*` (never a
    bare asterisk); reversible because it was produced from a sympy string."""
    py_text = prose.replace("·", "*")
    return sympify(py_text, locals={"t": _t})


def _skill1_oracle(question):
    m = _SKILL1_RE.search(question.text)
    assert m is not None, f"could not parse skill 1 question: {question.text!r}"
    dx, dy = _parse_vector_terms(m.group("d"))
    f0 = int(m.group("f0"))
    axis = m.group("axis")
    T = int(m.group("T"))
    g_expr = _parse_g_expr(m.group("g"))

    offset = m.group("offset")
    if offset == "it starts at the origin":
        x0 = y0 = 0
    else:
        r0_match = _R0_RE.search(offset)
        assert r0_match is not None, f"could not parse r(0) from: {offset!r}"
        x0, y0 = _parse_vector_terms(r0_match.group("r0"))

    # Fresh derivation route: integrate g(t) once (speed along d, +f0 at
    # t=0), integrate again (displacement along d, +0 at t=0), evaluate at
    # T, then add the offset for the asked axis.
    F_expr = integrate(g_expr, _t) + f0
    S_expr = integrate(F_expr, _t)  # +0 constant, since S(0) must be 0
    S_T = S_expr.subs(_t, T)
    assert S_T == int(S_T), f"S(T) not integer: {S_T}"
    S_T = int(S_T)

    d_component = dx if axis == "i" else dy
    r0_component = x0 if axis == "i" else y0
    return r0_component + S_T * d_component


@pytest.mark.parametrize("tier", TIERS)
def test_skill_1_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        oracle_answer = _skill1_oracle(q)
        assert str(oracle_answer) == q.answer_text, (
            f"seed={seed}/{tier}: oracle {oracle_answer} != rendered answer {q.answer_text}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_skill_1_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        oracle_answer = _skill1_oracle(q)
        for d in q.distractors:
            assert int(d) != oracle_answer, f"seed={seed}/{tier}: distractor {d!r} equals oracle answer"


# --- Skill 2: projectile motion in a plane ----------------------------------

_SKILL2_RE = re.compile(
    r"v\(0\) = (?P<v0>.+?) m/s, (?P<location>.+?)\. "
    r"Its acceleration vector is a\(t\) = -(?P<g>\d+)j m/s² \(gravity\)\. "
    r"Find the (?P<axis>[ij])-component of the particle's (?P<kind>velocity|position) vector, "
    r"in (?:m/s|m), at time t = (?P<T>\d+) (?:second|seconds)\."
)


def _skill2_oracle(question):
    m = _SKILL2_RE.search(question.text)
    assert m is not None, f"could not parse skill 2 question: {question.text!r}"
    vx0, vy0 = _parse_vector_terms(m.group("v0"))
    g = int(m.group("g"))
    axis = m.group("axis")
    kind = m.group("kind")
    T = int(m.group("T"))

    location = m.group("location")
    if location == "from the origin":
        x0 = y0 = 0
    else:
        r0_match = _R0_RE.search(location)
        assert r0_match is not None, f"could not parse r(0) from: {location!r}"
        x0, y0 = _parse_vector_terms(r0_match.group("r0"))

    # Fresh derivation route: symbolic antidifferentiation of the
    # acceleration vector a(t) = (0, -g), never the generator's own
    # plain-arithmetic path.
    ax_expr, ay_expr = 0, -g
    vx_expr = integrate(ax_expr, _t) + vx0
    vy_expr = integrate(ay_expr, _t) + vy0
    x_expr = integrate(vx_expr, _t) + x0
    y_expr = integrate(vy_expr, _t) + y0

    if kind == "velocity":
        value = vx_expr.subs(_t, T) if axis == "i" else vy_expr.subs(_t, T)
    else:
        value = x_expr.subs(_t, T) if axis == "i" else y_expr.subs(_t, T)
    assert value == int(value), f"oracle value not integer: {value}"
    return int(value)


@pytest.mark.parametrize("tier", TIERS)
def test_skill_2_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        oracle_answer = _skill2_oracle(q)
        assert str(oracle_answer) == q.answer_text, (
            f"seed={seed}/{tier}: oracle {oracle_answer} != rendered answer {q.answer_text}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_skill_2_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        oracle_answer = _skill2_oracle(q)
        for d in q.distractors:
            assert int(d) != oracle_answer, f"seed={seed}/{tier}: distractor {d!r} equals oracle answer"
