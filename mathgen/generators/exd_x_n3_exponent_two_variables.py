"""Exam-tier source for `f10-exponent-laws-integer-variable`: two variables, then evaluate the simplified form.

The practice tiers simplify a chain of powers of one variable. An exam question uses two variables, a power
raised to a power that applies to a whole bracket, and a negative index somewhere in the working. Part (a)
asks for the simplified form `x^p y^q`; part (b) asks the learner to evaluate that form at stated values of `x`
and `y`, which only comes out right when every index in (a) has the right sign. Part (b) follows through from
(a): a learner whose simplified form is wrong still earns (b) for evaluating their own form correctly.

Built backwards from the printed expression: the simplified indices are computed by the index laws and
re-derived with SymPy from the printed expression, a different route.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy
from sympy import Rational, symbols

from ..core import check, draw_until, generator, latex_answer, pick_distractors, rational_text

x, y = symbols("x y")
_BASES = [2, 3, 5, -2, -3, 10]


def _pw(var: str, e: int) -> str:
    if e == 1:
        return var
    return f"{var}^{e}" if e >= 0 else f"{var}^({e})"


def _term(a: int, b: int) -> str:
    return f"{_pw('x', a)} {_pw('y', b)}"


def _outer(c: int) -> str:
    return f"^{c}" if c >= 0 else f"^({c})"


def _show(p: int, q: int) -> str:
    return "*".join(v if e == 1 else f"{v}^{e}" for v, e in (("x", p), ("y", q)))


def _expr(p: int, q: int):
    return x**p * y**q


def _draw(rng: random.Random):
    shape = rng.randrange(3)
    e = lambda: rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])  # noqa: E731
    outer = lambda: rng.choice([-2, -1, 2, 3, 4])  # noqa: E731
    a, b, d, f = e(), e(), e(), e()
    c, g = outer(), outer()
    if shape == 0:  # (x^a y^b)^c / (x^d y^f)
        text = f"({_term(a, b)}){_outer(c)} ÷ ({_term(d, f)})"
        right = (a * c - d, b * c - f)
        wrong = [(a + c - d, b + c - f), (a * c + d, b * c + f), (a * c - d, b - f), (a * c - d, -(b * c - f))]
    elif shape == 1:  # (x^a y^b)^c x (x^d y^f)^g
        text = f"({_term(a, b)}){_outer(c)} × ({_term(d, f)}){_outer(g)}"
        right = (a * c + d * g, b * c + f * g)
        wrong = [
            (a * c - d * g, b * c - f * g),
            (a + c + d + g, b + c + f + g),
            (a * c + d, b * c + f),
            (a * c * d * g, b * c + f * g),
        ]
    else:  # x^a y^b / (x^d y^f)^c
        text = f"{_term(a, b)} ÷ ({_term(d, f)}){_outer(c)}"
        right = (a - d * c, b - f * c)
        wrong = [(a - d, b - f), (a + d * c, b + f * c), (a - d + c, b - f + c), (a - d * c, -(b - f * c))]
    return text, right, wrong


def _usable(w, right) -> bool:
    return w != right and w[0] != 0 and w[1] != 0 and abs(w[0]) <= 9 and abs(w[1]) <= 9


def _value_candidates(wrong, right, bx, by, value) -> list[str]:
    """Texts of the value a learner reaches from each wrong simplified form, then the reciprocal slip."""
    out = []
    for w in wrong:
        if not _usable(w, right):
            continue
        v = Fraction(bx) ** w[0] * Fraction(by) ** w[1]
        if abs(v.numerator) <= 10**6 and v.denominator <= 10**6:
            out.append(rational_text(Rational(v.numerator, v.denominator)))
    out.append(rational_text(Rational(value.denominator, value.numerator)))  # reciprocal
    return out


@generator("exd_x_n3_exponent_two_variables", level=9, presets={"exam": dict(tier="exam")})
def exd_x_n3_exponent_two_variables(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: simplify a two-variable index expression, then evaluate the result."""
    check(tier == "exam", f"unknown tier {tier!r}")

    def draw(r):
        return _draw(r), r.choice(_BASES), r.choice(_BASES)

    def accept(t):
        (text, (p, q), wrong), bx, by = t
        if not (0 < abs(p) <= 8 and 0 < abs(q) <= 8) or bx == by:
            return False
        value = Fraction(bx) ** p * Fraction(by) ** q
        if abs(value.numerator) > 5000 or value.denominator > 5000 or abs(value) == 1:
            return False
        correct_text = rational_text(Rational(value.numerator, value.denominator))
        b_texts = set(_value_candidates(wrong, (p, q), bx, by, value)) - {correct_text}
        return len({w for w in wrong if _usable(w, (p, q))}) >= 2 and len(b_texts) >= 2

    (text, (p, q), wrong), bx, by = draw_until(rng, draw, accept, what="an expression and base values")
    printed = sympy.sympify(
        text.replace("÷", "/").replace("×", "*").replace("^", "**").replace(" y", "*y").replace(" x", "*x")
    )
    check(sympy.simplify(printed - _expr(p, q)) == 0, "index laws disagree with SymPy")
    value = Fraction(bx) ** p * Fraction(by) ** q

    a_text = _show(p, q)
    a_dist = pick_distractors(a_text, [_show(*w) for w in wrong if _usable(w, (p, q))], 3)
    b_text = rational_text(Rational(value.numerator, value.denominator))
    b_cands = _value_candidates(wrong, (p, q), bx, by, value)
    b_dist = pick_distractors(b_text, b_cands, 3)
    check(len(a_dist) >= 2 and len(b_dist) >= 2, "too few distractors")

    return dict(
        text=f"Consider the expression {text}.",
        parts=[
            dict(
                prompt="Simplify the expression, giving your answer in the form x^p y^q.",
                answer_kind="expression",
                answer=str(_expr(p, q)),
                answer_text=a_text,
                answer_latex=latex_answer(_expr(p, q)),
                marks=3,
                working=[
                    "Apply the index laws: multiply indices for a power of a power, add for a product, subtract for a quotient."
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=f"Hence find the value of the expression when x = {bx} and y = {by}.",
                answer_kind="fraction",
                answer={"p": int(value.numerator), "q": int(value.denominator)},
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                follow_through=f"Subs(part_a, (x, y), ({bx}, {by}))",
                working=[f"x^{p} y^{q} = ({bx})^{p} × ({by})^{q} = {b_text}."],
                distractors=b_dist,
            ),
        ],
    )
