"""Exam-tier source for ``senior-consider-inverses-of-one-to-one-function``.

The practice tiers invert a linear function or a translated reciprocal ``a/(x - h) + k`` where the
unknown appears once. Here ``x`` appears in the numerator AND the denominator, so the learner must
collect the unknown on one side and factor it out before it can be isolated. Two forms:

* ``ratio``:   f(x) = (a x + b) / (c x + d)
* ``shifted``: f(x) = k + m / (c x + d)  with a non-unit coefficient c on x

Both are one-to-one on their natural domain (``a d - b c != 0``, ``m != 0``, ``c != 0``). The inverse
is built from a closed form and re-derived by letting sympy solve ``y = f(x)`` for ``x``.
"""

from __future__ import annotations

import random
import re

from sympy import Eq, Symbol, expand, fraction, latex, solve, sympify, together

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    join_terms,
    phrase,
    pick_distractors,
    symbolically_equal,
)

x = Symbol("x")
y = Symbol("y")

_WORDINGS = (
    "Find the inverse function f⁻¹(x) of f(x) = {f}.",
    "The function f is defined by f(x) = {f}. Write down an expression for f⁻¹(x).",
    "Let f(x) = {f}. Determine f⁻¹(x).",
)


def _lin(coef: int, const: int) -> str:
    return join_terms([(coef, "x"), (const, "")])


def _frac(num: str, den: str) -> str:
    return f"({num})/({den})"


def _as_ratio(expr) -> str:
    """``(numerator)/(denominator)`` in the same style as the canonical answer, so no option stands out."""
    num, den = fraction(together(expr))
    text = f"({expand(num)})/({expand(den)})"
    return re.sub(r"(\d)\*x", r"\1x", text)


def _ratio(rng: random.Random) -> dict:
    for _ in range(200):
        a = rng.choice([n for n in range(-5, 6) if n != 0])
        b = rng.choice([n for n in range(-6, 7) if n != 0])
        c = rng.choice([1, 2, 3])
        d = rng.choice([n for n in range(-6, 7) if n != 0])
        if a * d - b * c == 0 or a == -d:  # a == -d would make f its own inverse
            continue
        break
    else:
        check(False, "no ratio draw")
    f = _frac(_lin(a, b), _lin(c, d))
    f_expr = (a * x + b) / (c * x + d)
    inv = _frac(_lin(d, -b), join_terms([(-c, "x"), (a, "")]))
    inv_expr = (d * x - b) / (-c * x + a)
    wrong = [
        # inverted the value instead of the function: 1 / f(x)
        (c * x + d) / (a * x + b),
        # moved b across without changing its sign
        (d * x + b) / (-c * x + a),
        # collected the x terms with the wrong sign
        (d * x - b) / (c * x + a),
        # left f unchanged
        f_expr,
    ]
    return dict(f=f, f_expr=f_expr, ans_text=inv, ans_expr=inv_expr, wrong=wrong, f_latex=latex(f_expr))


def _shifted(rng: random.Random) -> dict:
    k = rng.choice([n for n in range(-5, 6) if n != 0])
    m = rng.choice([n for n in range(-6, 7) if n not in (0, 1, -1)])
    c = rng.choice([2, 3])
    d = rng.choice([n for n in range(-5, 6) if n != 0])
    f = f"{k} {'+' if m > 0 else '-'} {abs(m)}/({_lin(c, d)})"
    f_expr = k + sympify(m) / (c * x + d)
    # y = k + m/(c x + d)  =>  x = (m - d (y - k)) / (c (y - k))
    ans_expr = (-d * x + m + d * k) / (c * x - c * k)
    inv = _frac(join_terms([(-d, "x"), (m + d * k, "")]), join_terms([(c, "x"), (-c * k, "")]))
    wrong = [
        # forgot to divide the whole expression by c
        m / (x - k) - d,
        # translated the wrong way: x + k in place of x - k
        (m / (x + k) - d) / c,
        # inverted the value instead of the function
        1 / f_expr,
        # left f unchanged
        f_expr,
    ]
    return dict(f=f, f_expr=f_expr, ans_text=inv, ans_expr=ans_expr, wrong=wrong, f_latex=latex(f_expr))


def _build(rng: random.Random) -> dict:
    for _ in range(100):
        d = _ratio(rng) if rng.random() < 0.6 else _shifted(rng)
        # Independent route: let sympy solve y = f(x) for x, then rename y to x.
        solved = solve(Eq(y, d["f_expr"].subs(x, x)), x)
        check(len(solved) == 1, "f is not one-to-one as drawn")
        check_equivalent(solved[0].subs(y, x), d["ans_expr"])
        chosen: list = []
        for w in d["wrong"]:
            if symbolically_equal(w, d["ans_expr"]) or any(symbolically_equal(w, o) for o in chosen):
                continue
            chosen.append(w)
        if len(chosen) < 3:
            continue
        chosen = chosen[:3]
        check_distinct(d["ans_expr"], *chosen)
        text = phrase(rng, _WORDINGS, f=d["f"])
        return dict(
            text=text,
            latex=rf"f(x) = {d['f_latex']},\ \text{{find }} f^{{-1}}(x)",
            answer=d["ans_text"],
            answer_text=d["ans_text"],
            answer_latex=rf"f^{{-1}}(x) = {latex(d['ans_expr'])}",
            working=[
                "Write y = f(x), multiply out the denominator, collect every term in x on one side, "
                f"factor x out and divide. This gives f⁻¹(x) = {d['ans_text']}."
            ],
            distractors=pick_distractors(d["ans_text"], [_as_ratio(w) for w in chosen]),
        )
    check(False, "could not build an inverse question")


@generator(
    "exd_x_f1_inverse_mobius",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f1_inverse_mobius(rng: random.Random, tier: str = "exam") -> dict:
    """Inverse of a function in which the unknown appears in both numerator and denominator."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _build(rng)
