"""Tests `uc2-laplace-step-and-convolution` (topic `s2c3_laplace_step`) and its exam chain `exd_x_s2c3_laplace_step`.

The oracle reads the problem back out of the wording. A switched-on signal is transformed by SymPy's exact integral
from t = a; an inverse transform is checked by integrating the stated answer against e^(-st) and comparing it with the
stated transform; a convolution is integrated directly; and the exam equation is integrated step by step with
fourth-order Runge-Kutta, which shares nothing with the Laplace route.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2c3_laplace_step_convolution_build  # noqa: F401  (registers the topics)
from _s2c3_helpers import (
    assert_exam_grading,
    assert_follow_through,
    assert_key_accepted_and_distractors_rejected,
    assert_text_quality_and_determinism,
    skill_by_id,
    spread,
    tier_question,
)
from mathgen import make

TOPIC = "s2c3_laplace_step"
EXAM = "exd_x_s2c3_laplace_step"
SKILL = skill_by_id("uc2-laplace-step-and-convolution")
TIERS = ("easy", "medium", "hard")
t, s, tau = sympy.symbols("t s tau", positive=True)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _balanced(text: str, start: int) -> int:
    """Index just past the parenthesis group that opens at text[start] == '('."""
    depth = 0
    for i in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return i + 1
    raise ValueError(text)


def parse(text: str):
    """School-style maths in t and s: `e^(-2(t - 2))`, `(1/4)e^t - 3cos(2t)`, `e^(-s)(2s + 2)/s^3`."""
    text = text.replace("²", "^2").replace("−", "-")
    while "e^(" in text:
        i = text.index("e^(")
        j = _balanced(text, i + 2)
        text = text[:i] + "exp" + text[i + 2 : j] + text[j:]
    text = re.sub(r"e\^([ts])\b", r"exp(\1)", text)
    text = text.replace("^", "**")
    text = re.sub(r"(?<![A-Za-z])([ts])\(", r"\1*(", text)
    return parse_expr(text, local_dict={"t": t, "s": s, "tau": tau}, transformations=_TRANSFORMS)


def laplace_numeric(f_of_t, start: float, s0: float) -> float:
    fn = sympy.lambdify(t, f_of_t, "numpy")
    xs = np.linspace(start, start + 40.0 / s0, 400001)
    ys = fn(xs) * np.ones_like(xs) * np.exp(-s0 * xs)
    return float(np.trapezoid(ys, xs))


def test_skill_wiring():
    assert SKILL.answer == "expression"
    assert [(src.topic, src.difficulty) for src in SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert SKILL.sources[-1].stages == ["exam"]
    assert "ucv1-laplace-transform" in SKILL.prerequisites
    assert SKILL.year_level == ("university",)


def test_easy_is_the_second_shift_theorem_with_the_body_in_t():
    families = Counter()
    for seed in range(60):
        q = tier_question(TOPIC, "easy", seed)
        m = re.search(r"f\(t\) = ([^=]+?) for t ≥ (\d+)", q.text)
        g, a = parse(m.group(1)), int(m.group(2))
        exact = sympy.integrate(g * sympy.exp(-s * t), (t, a, sympy.oo))
        key = parse(q.answer)
        for s0 in (sympy.Rational(7, 2), 5):
            assert abs(float((exact - key).subs(s, s0))) < 1e-9, (seed, q.text, q.answer)
        families["exp" if "e^" in m.group(1) else ("quad" if "²" in m.group(1) else "lin")] += 1
    assert min(families.values()) >= 5, families


def _transform_given(text: str):
    return parse(re.search(r"F\(s\) = (.+?)(?:\. Find|\.$)", text).group(1))


def test_medium_inverts_a_delayed_transform():
    kinds = Counter()
    for seed in range(60):
        q = tier_question(TOPIC, "medium", seed)
        a = int(re.search(r"t > (\d+)", q.text).group(1))
        F = _transform_given(q.text)
        key = parse(q.answer)
        for s0 in (4.0, 6.0):
            want = float(F.subs(s, s0))
            assert abs(laplace_numeric(key, a, s0) - want) < 1e-5 * (1 + abs(want)), (seed, q.text, q.answer)
        kinds["quadratic" if "s²" in q.text else "poles"] += 1
    assert min(kinds.values()) >= 10, kinds


def _convolution_by_integration(g, h, t0: float) -> float:
    gf, hf = sympy.lambdify(t, g, "numpy"), sympy.lambdify(t, h, "numpy")
    xs = np.linspace(0.0, t0, 200001)
    return float(np.trapezoid(gf(xs) * np.ones_like(xs) * hf(t0 - xs) * np.ones_like(xs), xs))


def test_hard_convolution_and_product_inverse():
    kinds = Counter()
    for seed in range(60):
        q = tier_question(TOPIC, "hard", seed)
        key = parse(q.answer)
        if "convolution" in q.text and "inverse" not in q.text:
            m = re.search(r"g\(t\) = (.+?)(?:»| and|, ) ?(?:and )?h\(t\) = (.+?)(?: as a function|\. Find|\.)", q.text)
            g, h = parse(m.group(1)), parse(m.group(2))
            for t0 in (1.1, 2.3):
                assert abs(_convolution_by_integration(g, h, t0) - float(key.subs(t, t0))) < 1e-5, (
                    seed,
                    q.text,
                    q.answer,
                )
            kinds["convolution"] += 1
        else:
            F = parse(re.search(r"(?:F\(s\)|L\{f\(t\)\}) = (.+?)(?:\.|$)", q.text).group(1))
            for s0 in (4.0, 6.0):
                want = float(F.subs(s, s0))
                assert abs(laplace_numeric(key, 0.0, s0) - want) < 1e-5 * (1 + abs(want)), (seed, q.text, q.answer)
            kinds["product"] += 1
    assert min(kinds.values()) >= 8, kinds


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    assert_key_accepted_and_distractors_rejected(SKILL, TOPIC, tier, minimum=3)


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    assert_text_quality_and_determinism(TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    spread(TOPIC, tier, minimum_answers=20, minimum_texts=50)


# -- exam -----------------------------------------------------------------------------------------------------------


def _ode(q):
    m = re.search(r"y″ \+ (\d+)y′ \+ (\d+)y = (\d+)u\(t - (\d+)\)", q.stem)
    return tuple(int(v) for v in m.groups())


def rk4(p: int, q: int, A: int, T: int, t_end: float, h: float = 1e-3) -> float:
    """y(t_end) for y'' + p y' + q y = A u(t - T), y(0) = y'(0) = 0, by fourth-order Runge-Kutta."""
    y = v = 0.0
    tt = 0.0
    while tt < t_end - 1e-12:
        step = min(h, t_end - tt)
        if tt < T - 1e-12 < tt + step:
            step = T - tt
        force = A if tt >= T - 1e-9 else 0.0  # constant over the step: a step never straddles T
        f = lambda _tt, y_, v_, force=force: (v_, force - p * v_ - q * y_)  # noqa: E731
        k1 = f(tt, y, v)
        k2 = f(tt + step / 2, y + step / 2 * k1[0], v + step / 2 * k1[1])
        k3 = f(tt + step / 2, y + step / 2 * k2[0], v + step / 2 * k2[1])
        k4 = f(tt + step, y + step * k3[0], v + step * k3[1])
        y += step / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v += step / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        tt += step
    return y


def _exact(answer):
    """A fraction answer ({'p': .., 'q': ..} or an int) as a SymPy rational."""
    return sympy.Rational(answer["p"], answer["q"]) if isinstance(answer, dict) else sympy.Rational(answer)


def _t0(q):
    return float(sympy.Rational(re.search(r"Find y\(([\d/]+)\)", q.parts[3].prompt).group(1)))


def test_exam_shape_and_keys():
    assert_exam_grading(SKILL, EXAM, [3, 2, 3, 1], ["expression", "fraction", "expression", "number"])
    assert [p.follow_through is None for p in make(EXAM, seed=2, difficulty="exam").parts] == [
        True,
        False,
        False,
        False,
    ]
    for seed in range(25):
        q = make(EXAM, seed=seed, difficulty="exam")
        p, qq, A, T = _ode(q)
        # (a) satisfies (s^2 + ps + q)Y = A e^(-Ts)/s
        Y = parse(q.parts[0].answer)
        residual = ((s**2 + p * s + qq) * Y - A * sympy.exp(-T * s) / s).subs(s, sympy.Rational(23, 10))
        assert abs(float(residual)) < 1e-9, seed
        # (b) the long-time value, by integrating far on
        assert abs(rk4(p, qq, A, T, T + 12.0, 5e-3) - float(_exact(q.parts[1].answer))) < 1e-4, seed
        # (c) and (d) against Runge-Kutta
        y_c = parse(q.parts[2].answer)
        for dt in (0.6, 1.4):
            assert abs(float(y_c.subs(t, T + dt)) - rk4(p, qq, A, T, T + dt)) < 2e-5, (seed, dt)
        assert abs(rk4(p, qq, A, T, _t0(q)) - float(q.parts[3].answer)) < 6e-4, seed


def _wrong_amplitude(q):
    p, qq, A, T = _ode(q)
    return f"{A + 1}*exp(-{T}*s)/(s*(s^2 + {p}*s + {qq}))"


def test_exam_follow_through_from_a_wrong_transform():
    assert_follow_through(
        SKILL,
        EXAM,
        "b",
        {"a": _wrong_amplitude},
        lambda q, replies: str(sympy.Rational(_ode(q)[2] + 1, _ode(q)[1])),
    )


def test_exam_follow_through_from_a_wrong_long_time_value():
    assert_follow_through(
        SKILL,
        EXAM,
        "c",
        {"b": lambda q: str(2 * _exact(q.parts[1].answer))},
        lambda q, replies: f"2*({q.parts[2].answer})",
    )


def test_exam_follow_through_into_the_value():
    def expected(q, replies):
        y_c = parse(q.parts[2].answer)
        return f"{3 * float(y_c.subs(t, _t0(q))):.3f}"

    assert_follow_through(SKILL, EXAM, "d", {"c": lambda q: f"3*({q.parts[2].answer})"}, expected)
