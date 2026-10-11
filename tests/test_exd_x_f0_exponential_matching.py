"""Exam source of `f10-exponential-relations-equations`: solutions are verified by substituting back into the
rendered equation with exact arithmetic."""

from __future__ import annotations

import re
from fractions import Fraction

from sympy import Rational, simplify

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f0_exponential_matching"
_FACTOR = re.compile(r"(\(1/\d+\)|\d+)\^\(([^)]*)\)")


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _hard(seed):
    return make("p11d_exponential_common_base_solve", seed=seed, difficulty="hard")


def _base(token: str) -> Rational:
    if token.startswith("("):
        return Rational(1, int(token[3:-1]))
    return Rational(int(token))


def _exponent(expr: str, x: Rational) -> Rational:
    """'-2x + 3' evaluated at x."""
    total = Rational(0)
    for sign, body in re.findall(r"([+-]?)\s*(\d*x|\d+)", expr.replace(" ", "")):
        sgn = -1 if sign == "-" else 1
        if body.endswith("x"):
            coef = body[:-1]
            total += sgn * (int(coef) if coef else 1) * x
        else:
            total += sgn * int(body)
    return total


def _side(text: str, x: Rational) -> Rational:
    value = Rational(1)
    for base, exp in _FACTOR.findall(text):
        value *= _base(base) ** _exponent(exp, x)
    return value


def _rhs_number(text: str) -> Rational | None:
    m = re.fullmatch(r"(\d+)(?:/(\d+))?", text.strip())
    return Rational(int(m.group(1)), int(m.group(2) or 1)) if m else None


def holds(text: str, x: Fraction) -> bool:
    eq = re.match(r"Solve (.+?) for x\.", text).group(1).replace("×", "*")
    lhs, rhs = eq.split(" = ")
    xr = Rational(x.numerator, x.denominator)
    left = Rational(1)
    for part in lhs.split(" * "):
        left *= _side(part, xr)
    right = _rhs_number(rhs)
    right = right if right is not None else _side(rhs, xr)
    return simplify(left - right) == 0


def _root_is_unique(text: str, x: Fraction) -> bool:
    """Both sides are exponentials of linear functions of x with a fixed prime, so the difference of the exponents
    is linear in x: the key being a root and the slope being non-zero makes it the only root."""
    return holds(text, x) and not holds(text, x + 1) and not holds(text, x - 1)


def test_no_text_defects_and_three_shapes():
    for seed in range(60):
        assert find_text_defects(_q(seed)) == []
    assert len({prompt_shape(_q(s)) for s in range(60)}) >= 3


def test_exam_adds_shapes_over_hard_tier():
    hard = {prompt_shape(_hard(s)) for s in range(60)}
    exam = {prompt_shape(_q(s)) for s in range(60)}
    assert len(exam - hard) >= 3, (exam, hard)


def test_key_satisfies_rendered_equation_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _q(seed)
        x = Fraction(q.answer["p"], q.answer["q"])
        if not _root_is_unique(q.text, x):
            bad.append((seed, q.text, x))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor():
    grade = GRADERS["fraction"]
    for seed in range(150):
        q = _q(seed)
        assert grade(q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) >= 3
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, d, q.answer_text)
            assert not holds(q.text, Fraction(d)), (seed, d)


def test_deterministic():
    assert _q(5).to_json() == _q(5).to_json()
