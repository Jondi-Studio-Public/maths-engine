"""Tests the all-solutions trig skill (`w5_trig_all_solutions`, list answers).

The oracle reads the equation and the window back out of the question TEXT and finds the solutions with SymPy's
`solveset` on the real line (a different route from the generator's CAST rule and its half-degree scan).
"""

from __future__ import annotations

import re

import pytest
import sympy

import mathgen.generators.w5_trig_solutions_build  # noqa: F401  (registers the topics)
from tests import w5_support as sup

TOPIC, EXAM = "w5_trig_all_solutions", "exd_x_w5_trig_all_solutions"
SKILL = sup.skill_of("senior-find-all-solutions-of-mathrm-f-left-mathrm")
X = sympy.Symbol("x", real=True)
_VALUES = {
    "1/2": sympy.Rational(1, 2),
    "√2/2": sympy.sqrt(2) / 2,
    "√3/2": sympy.sqrt(3) / 2,
    "1/√3": 1 / sympy.sqrt(3),
    "1": sympy.Integer(1),
    "√3": sympy.sqrt(3),
}
_FN = {"sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan}
_EQ = re.compile(r"(sin|cos|tan)\(([^)]*(?:\([^)]*\))?[^)]*)\) = (-?)([^ ,.]+?)(?=[ .]| with| for|$)")


def _argument(text: str):
    text = text.strip()
    if text == "x":
        return X
    if m := re.fullmatch(r"x ([-+]) (\d+)", text):
        return X - int(m.group(2)) if m.group(1) == "-" else X + int(m.group(2))
    if m := re.fullmatch(r"(\d)x", text):
        return int(m.group(1)) * X
    if m := re.fullmatch(r"(\d)\(x ([-+]) (\d+)\)", text):
        a = int(m.group(1))
        return a * (X - int(m.group(3)) if m.group(2) == "-" else X + int(m.group(3)))
    raise AssertionError(text)


def _solve(equation: str, lo: int, hi: int):
    m = re.fullmatch(r"(sin|cos|tan)\((.+)\) = (-?)(.+)", equation)
    func, arg, neg, val = m.groups()
    rhs = _VALUES[val] * (-1 if neg else 1)
    u = sympy.Symbol("u", real=True)
    deg = sympy.pi / 180
    expr = _FN[func](_argument(arg) * deg) - rhs
    sols = sympy.solveset(expr, X, sympy.Interval.open(lo, hi))
    return sorted(int(sympy.nsimplify(s)) for s in sols)


def _split(text: str):
    m = re.search(r"((?:sin|cos|tan)\(.+?\) = -?[^ ]+?)(?: for| with|\.|,)", text)
    w = re.search(r"(-?\d+) ≤ x ≤ (\d+)", text)
    return m.group(1), int(w.group(1)), int(w.group(2))


def _listed(text: str):
    return [int(v) for v in text.split(", ")]


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM, "list")


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(tier):
    sup.check_single_tier(SKILL, TOPIC, tier)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_answers_are_unguessable(tier):
    assert len(sup.answer_spread(TOPIC, tier, 60)) >= 12


def test_pools_and_shapes():
    sup.check_pools(TOPIC, min_prompts=20)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_solveset(tier):
    for seed in range(60):
        q = sup.single(TOPIC, tier, seed)
        equation, lo, hi = _split(q.text)
        assert _listed(q.answer_text) == _solve(equation, lo, hi), (seed, q.text, q.answer_text)
        for d in q.distractors:
            assert _listed(d) != _solve(equation, lo, hi)


def test_hard_is_not_easy():
    easy = {len(_listed(sup.single(TOPIC, "easy", s).answer_text)) for s in range(40)}
    hard = {len(_listed(sup.single(TOPIC, "hard", s).answer_text)) for s in range(40)}
    assert easy == {2} and hard <= {4, 6} and hard


def test_exam_chain():
    sup.check_exam_chain_kinds(SKILL, EXAM, [2, 1, 3], [{"list"}, {"number", "fraction"}, {"list"}], seeds=40)
    for seed in range(40):
        q = sup.exam(EXAM, seed)
        lo, hi = (int(v) for v in re.search(r"(-?\d+) ≤ x ≤ (-?\d+)", q.text).groups())
        eqs = re.findall(
            r"(?:solutions of|solution of) ((?:sin|cos|tan)\(.+?\) = -?[^ ]+?)(?:\.| that)",
            " ".join(p.prompt for p in q.parts),
        )
        a, b, c = q.parts
        e1, e2, e3 = eqs[0], re.search(r"of ((?:sin|cos|tan)\(.+?\) = -?[^ ]+?) that", b.prompt).group(1), eqs[-1]
        assert _listed(a.answer_text) == _solve(e1, lo, hi)
        assert int(b.answer_text) == min(v for v in _solve(e2, lo, hi) if v >= 0)
        assert _listed(c.answer_text) == _solve(e3, lo, hi)
