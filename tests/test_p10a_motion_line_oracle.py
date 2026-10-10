"""Reads only `question.text` / `answer_text` / `distractors` via `mathgen.make`.
Imports NOTHING from `mathgen.generators.p10a_motion_line_build` -- the
velocity/acceleration polynomial and the stated initial condition(s) are all
parsed back out of the rendered prose with regular expressions, then
re-derived with `sympy.integrate()` and direct substitution/`sympy.solve`,
never by calling into the generator's own hand power-rule loop.
"""

import re

import pytest
from sympy import Eq, integrate, solve, symbols, sympify

from mathgen import make
from mathgen.core import symbolically_equal

SEEDS = range(60)
TIERS = ("easy", "medium", "hard")

t = symbols("t")

TOPIC_1 = "p10a_velocity_to_displacement_line"
TOPIC_2 = "p10a_acceleration_to_position_line"


def _parse_poly(prose: str):
    """A generator-rendered polynomial in t ("-7·t^2 - 4·t + 9", "3 - 5·t")
    back into a sympy expression. `·` is the generator's own stand-in for
    `*`; `^` is its stand-in for `**`. Both substitutions are exactly
    reversible because the generator produced them from a sympy string."""
    py_text = prose.replace("^", "**").replace("·", "*")
    return sympify(py_text, locals={"t": t})


def _parse_expr(typed: str):
    """A generator-rendered answer/distractor ("-7*t^3/3 - 2*t^2 + 9*t + 41")
    back into a sympy expression: `^` -> `**`, bare `*` already sympy-valid."""
    return sympify(typed.replace("^", "**"), locals={"t": t})


# --- Skill 1: v(t) -> s(t), one antidifferentiation step -------------------

_SKILL1_PLAIN_RE = re.compile(
    r"^A particle moves along a straight line with velocity v\(t\) = (?P<v>.+?) m/s "
    r"at time t seconds \(t ≥ 0\)\. At time t = (?P<a>-?\d+), the particle's position "
    r"is (?P<s0>-?\d+) m\."
)
_SKILL1_WORDED_RE = re.compile(
    r"^A particle moves along a straight line with velocity v\(t\) = (?P<v>.+?) m/s "
    r"at time t seconds \(t ≥ 0\)\. At time t = (?P<a>-?\d+), the particle is "
    r"(?P<mag>\d+) m to the (?P<side>left|right) of the origin\."
)


def _skill1_oracle(question):
    """Re-derive skill 1's answer from the rendered text alone:
    `sympy.integrate()` of the parsed v(t) (a genuinely different
    implementation from the generator's own hand power-rule loop), then
    `sympy.solve` for the constant against the parsed initial condition."""
    m = _SKILL1_PLAIN_RE.search(question.text)
    if m:
        s0 = int(m.group("s0"))
    else:
        m = _SKILL1_WORDED_RE.search(question.text)
        assert m is not None, f"could not parse skill 1 head from: {question.text!r}"
        mag = int(m.group("mag"))
        s0 = -mag if m.group("side") == "left" else mag

    v_expr = _parse_poly(m.group("v"))
    a = int(m.group("a"))

    C = symbols("C")
    antideriv = integrate(v_expr, t) + C
    solved = solve(Eq(antideriv.subs(t, a), s0), C)
    assert len(solved) == 1, f"expected exactly one solution for C, got {solved}"
    return antideriv.subs(C, solved[0])


@pytest.mark.parametrize("tier", TIERS)
def test_skill1_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        oracle_answer = _skill1_oracle(q)
        rendered = _parse_expr(q.answer_text)
        assert symbolically_equal(oracle_answer, rendered), (
            f"seed={seed}/{tier}: oracle {oracle_answer} != rendered answer {rendered}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_skill1_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        oracle_answer = _skill1_oracle(q)
        for d in q.distractors:
            assert not symbolically_equal(oracle_answer, _parse_expr(d)), (
                f"seed={seed}/{tier}: distractor {d!r} equals oracle answer {oracle_answer}"
            )


# --- Skill 2: a(t) -> s(t), two antidifferentiation steps -------------------

_SKILL2_RE = re.compile(
    r"^A particle moves along a straight line with acceleration a\(t\) = (?P<a>.+?) m/s² "
    r"at time t seconds \(t ≥ 0\)\. At time t = (?P<a_ic>-?\d+), its velocity is "
    r"(?P<v0>-?\d+) m/s and its position is (?P<s0>-?\d+) m\."
)


def _skill2_oracle(question):
    """Re-derive skill 2's answer from the rendered text alone: integrate
    the parsed a(t) once for v(t) (solving C1 against the stated velocity),
    then integrate that v(t) again for s(t) (solving C2 against the stated
    position) -- both steps via `sympy.integrate()`/`sympy.solve`, never the
    generator's own construction."""
    m = _SKILL2_RE.search(question.text)
    assert m is not None, f"could not parse skill 2 head from: {question.text!r}"
    a_expr = _parse_poly(m.group("a"))
    a_ic = int(m.group("a_ic"))
    v0 = int(m.group("v0"))
    s0 = int(m.group("s0"))

    C1, C2 = symbols("C1 C2")
    v_antideriv = integrate(a_expr, t) + C1
    solved_c1 = solve(Eq(v_antideriv.subs(t, a_ic), v0), C1)
    assert len(solved_c1) == 1, f"expected exactly one solution for C1, got {solved_c1}"
    v_expr = v_antideriv.subs(C1, solved_c1[0])

    s_antideriv = integrate(v_expr, t) + C2
    solved_c2 = solve(Eq(s_antideriv.subs(t, a_ic), s0), C2)
    assert len(solved_c2) == 1, f"expected exactly one solution for C2, got {solved_c2}"
    return s_antideriv.subs(C2, solved_c2[0])


@pytest.mark.parametrize("tier", TIERS)
def test_skill2_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        oracle_answer = _skill2_oracle(q)
        rendered = _parse_expr(q.answer_text)
        assert symbolically_equal(oracle_answer, rendered), (
            f"seed={seed}/{tier}: oracle {oracle_answer} != rendered answer {rendered}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_skill2_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        oracle_answer = _skill2_oracle(q)
        for d in q.distractors:
            assert not symbolically_equal(oracle_answer, _parse_expr(d)), (
                f"seed={seed}/{tier}: distractor {d!r} equals oracle answer {oracle_answer}"
            )


# --- Item-4 misconceptions: never rendered (always the 4th of 4 candidates,
# dropped by `pick_distractors`' cap of 3), but still exercised here -------
#
# `misread-worded-sign` (skill 1, hard, s0 < 0 seeds) and
# `sign-error-on-first-constant` (skill 2, hard, every seed) both list AFTER
# `sign-error-on-constant`/`sign-error-on-second-constant` in the generator's
# candidate order, so on every seed where they exist they are the 4th
# candidate and `pick_distractors(n=3)` never offers them to a learner --
# measured 0/60 at both hard tiers. Model on
# `tests/test_p9c_difference_quotient_oracle.py`'s
# `test_row2_hard_tier_ignored_delta_x_sign_would_be_rejected`, which
# exercises its own never-rendered-alongside-3-others misconception the same
# way: recompute the misconception's value purely from the parsed text and
# confirm it differs from the independently-derived answer.


def test_skill1_misread_worded_sign_would_be_rejected():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty="hard")
        m = _SKILL1_WORDED_RE.search(q.text)
        assert m is not None, f"seed={seed}: hard tier did not render the worded IC form"
        if m.group("side") != "left":
            continue
        mag = int(m.group("mag"))
        s0 = -mag
        v_expr = _parse_poly(m.group("v"))
        a = int(m.group("a"))

        oracle_answer = _skill1_oracle(q)
        G = integrate(v_expr, t)
        wrong_c2 = abs(s0) - G.subs(t, a)
        misread_worded_sign = G + wrong_c2
        assert not symbolically_equal(oracle_answer, misread_worded_sign), (
            f"seed={seed}: misread-worded-sign {misread_worded_sign} equals the answer {oracle_answer}"
        )
        checked += 1
    assert checked > 0, "no hard-tier seed in 60 had a negative (left-of-origin) s0"


def test_skill2_sign_error_on_first_constant_would_be_rejected():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty="hard")
        m = _SKILL2_RE.search(q.text)
        assert m is not None, f"could not parse skill 2 head from: {q.text!r}"
        a_expr = _parse_poly(m.group("a"))
        a_ic = int(m.group("a_ic"))
        v0 = int(m.group("v0"))
        s0 = int(m.group("s0"))

        oracle_answer = _skill2_oracle(q)
        G1 = integrate(a_expr, t)
        c1_prime = v0 + G1.subs(t, a_ic)  # sign flipped
        wrong_v = G1 + c1_prime
        wrong_g2 = integrate(wrong_v, t)
        c2_double_prime = s0 - wrong_g2.subs(t, a_ic)
        sign_error_on_first_constant = wrong_g2 + c2_double_prime
        assert not symbolically_equal(oracle_answer, sign_error_on_first_constant), (
            f"seed={seed}: sign-error-on-first-constant {sign_error_on_first_constant} "
            f"equals the answer {oracle_answer}"
        )
        checked += 1
    assert checked == len(SEEDS)
