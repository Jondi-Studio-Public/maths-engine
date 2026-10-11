"""Tests for the exam sources of the three integration skills (velocity -> displacement, f(x) from f'(x), and
acceleration -> position). Each key is re-derived from the rendered prompt alone with sympy."""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

_t, _x = sympy.symbols("t x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

VELOCITY = "exd_x_h5_velocity_two_positions"
ANTIDERIV = "exd_x_h5_antiderivative_linear_power"
ACCEL = "exd_x_h5_acceleration_rest_or_positions"
HARD_TOPIC = {
    VELOCITY: "p10a_velocity_to_displacement_line",
    ANTIDERIV: "p10a_antiderivative_given_fprime",
    ACCEL: "p10a_acceleration_to_position_line",
}


def _exam(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


def _parse(text):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"t": _t, "x": _x})


@pytest.mark.parametrize("topic", sorted(HARD_TOPIC))
def test_text_variety_new_shapes_and_determinism(topic):
    for seed in range(60):
        assert find_text_defects(_exam(topic, seed)) == []
    shapes = {prompt_shape(_exam(topic, s)) for s in range(60)}
    hard = {prompt_shape(make(HARD_TOPIC[topic], seed=s, difficulty="hard")) for s in range(60)}
    assert len(shapes) >= 3
    assert shapes - hard
    assert _exam(topic, 5).to_json() == _exam(topic, 5).to_json()


def _oracle_velocity(text):
    a, d = re.search(r"v\(t\) = (-?\d+)t\^2 \+ kt ([+-] \d+) m/s", text).groups()
    pts = [(int(p), int(tt)) for p, tt in re.findall(r"(-?\d+) m (?:at|when) t = (\d+)", text)]
    assert len(pts) == 2
    k, c = sympy.symbols("k c")
    s = int(a) * _t**3 / 3 + k * _t**2 / 2 + int(d.replace(" ", "")) * _t + c
    sol = sympy.solve([s.subs(_t, tt) - p for p, tt in pts], [k, c], dict=True)[0]
    return s.subs(sol)


def _oracle_antideriv(text):
    coef, lin, n = re.search(r"f'\(x\) = (-?\d*)\((.*?)\)\^(\d+)", text).groups()
    a, b = re.search(r"f\((-?\d+)\) = (-?\d+)", text).groups()
    coef = {"": 1, "-": -1}.get(coef, None) or int(coef)
    fprime = coef * _parse(lin) ** int(n)
    big_c = sympy.Symbol("C")
    g = sympy.integrate(fprime, _x) + big_c
    return g.subs(big_c, sympy.solve(g.subs(_x, int(a)) - int(b), big_c)[0])


def _oracle_accel(text):
    a_text = re.search(r"a\(t\) = (.*?) m/s", text).group(1)
    a_expr = _parse(a_text)
    c1, c2 = sympy.symbols("c1 c2")
    v = sympy.integrate(a_expr, _t) + c1
    s = sympy.integrate(v, _t) + c2
    eqs = []
    rest = re.search(r"at rest when t = (\d+)", text)
    if rest:
        eqs.append(v.subs(_t, int(rest.group(1))))
    for p, tt in re.findall(r"(-?\d+) m when t = (\d+)", text):
        eqs.append(s.subs(_t, int(tt)) - int(p))
    sol = sympy.solve(eqs, [c1, c2], dict=True)[0]
    return s.subs(sol)


ORACLES = {VELOCITY: _oracle_velocity, ANTIDERIV: _oracle_antideriv, ACCEL: _oracle_accel}


@pytest.mark.parametrize("topic", sorted(ORACLES))
def test_key_matches_independent_oracle_over_300_seeds(topic):
    bad = []
    for seed in range(300):
        q = _exam(topic, seed)
        expected = ORACLES[topic](q.text)
        if sympy.simplify(expected - _parse(q.answer_text)) != 0:
            bad.append((seed, q.text, q.answer_text))
    assert bad == []


@pytest.mark.parametrize("topic", sorted(ORACLES))
def test_grader_accepts_key_and_rejects_every_distractor(topic):
    grade = GRADERS["expression"]
    for seed in range(120):
        q = _exam(topic, seed)
        assert grade(q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, d, q.answer_text)
