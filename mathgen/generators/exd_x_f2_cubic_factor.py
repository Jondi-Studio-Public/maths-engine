"""Exam-tier source for factorising a cubic into a linear and a quadratic factor.

The practice tiers always use a linear factor ``(x - p)`` with a small integer root, which the learner finds by
testing small integers. Here the linear factor is harder to see, or has to be found by an extra step first.

Three question shapes, each built backwards from chosen integer factors so the key is exact:

- ``cubes``: a sum or difference of two perfect cubes with a scaled variable, ``(m x)^3 +- n^3``. There is no
  linear or quadratic term to guide a root search; the learner must recognise the identity
  ``a^3 +- b^3 = (a +- b)(a^2 -+ ab + b^2)``.
- ``grouping``: a cubic of the form ``m x^3 - p x^2 + m q x - p q``, which factorises by grouping as
  ``(m x - p)(x^2 + q)``. Testing small integer roots does not find the factor when ``m > 1``, and even for
  ``m = 1`` the root can be large; the structure of the coefficients is the way in.
- ``parameter``: a monic cubic with one unknown coefficient ``k`` and a stated linear factor ``(x - p)``. The
  learner must apply the factor theorem to find ``k`` before dividing.

Every key is checked by expanding the printed factored form and comparing it, term by term, with the printed
cubic (and, for the parameter shape, by substituting the root into the printed cubic with the key's ``k``).
The required form is a product of a linear factor and a quadratic factor, so the grader asks for a bracketed
answer. Distractors, each a named slip:

- ``cubes``: the wrong sign on the middle term of the quadratic factor; the wrong sign in the linear factor;
  the coefficient of ``x`` in the quadratic left as ``m`` instead of ``m^2`` for the leading term.
- ``grouping``: the sign of the constant in the linear factor reversed; the sign of ``q`` reversed in the
  quadratic factor; the constants of the two groups swapped.
- ``parameter``: the sign of the root reversed in the linear factor; the value of ``k`` mistaken for the
  coefficient of ``x`` in the quadratic factor; the sign of the quadratic's constant reversed.
"""

from __future__ import annotations

import random

from sympy import Symbol, expand
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from ..core import check, check_distinct, check_equivalent, generator, join_terms, phrase, pick_distractors

x = Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

_CUBES = (
    "Write {cubic} as a product of a linear factor and a quadratic factor.",
    "Factorise {cubic} into a linear factor and a quadratic factor.",
)
_GROUPING = (
    "Factorise {cubic} as a product of a linear factor and a quadratic factor, each with integer coefficients.",
    "Express {cubic} as the product of a linear factor and a quadratic factor with integer coefficients.",
)
_PARAMETER = (
    "The cubic {cubic} has (x {sign} {p}) as a factor, where k is a constant. Find k and hence write the cubic "
    "as a product of a linear factor and a quadratic factor.",
    "Given that (x {sign} {p}) is a factor of {cubic}, where k is a constant, find k and factorise the cubic "
    "into a linear factor and a quadratic factor.",
)


def _splits(a: int, b: int, c: int) -> bool:
    """True when a x^2 + b x + c has integer roots, so the quadratic factor would not be fully factorised."""
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    root = int(disc**0.5)
    return root * root == disc and (-b + root) % (2 * a) == 0 and (-b - root) % (2 * a) == 0


def _parse(text: str):
    return parse_expr(text, local_dict={"x": x}, transformations=_TRANSFORMS)


def _linear(a: int, b: int) -> str:
    return join_terms([(a, "x"), (b, "")])


def _quad(a: int, b: int, c: int) -> str:
    return join_terms([(a, "x^2"), (b, "x"), (c, "")])


def _cubic_text(expr) -> str:
    e = expand(expr)
    return join_terms([(int(e.coeff(x, k)), v) for k, v in ((3, "x^3"), (2, "x^2"), (1, "x"), (0, ""))])


def _product(lin: tuple[int, int], quad: tuple[int, int, int]) -> str:
    return f"({_linear(*lin)})({_quad(*quad)})"


def _result(text: str, lin, quad, cubic_text: str, candidates: list[tuple], working: list[str]) -> dict:
    answer_text = _product(lin, quad)
    key = _parse(answer_text)
    # Independent check: the printed factored key must expand to the printed cubic.
    check_equivalent(expand(key), _parse(cubic_text), "the factored key does not expand to the printed cubic")
    texts = []
    for cand_lin, cand_quad in candidates:
        cand_text = _product(cand_lin, cand_quad)
        check_distinct(key, _parse(cand_text), msg=f"distractor {cand_text!r} equals the key")
        texts.append(cand_text)
    return dict(
        text=text,
        answer=str(key),
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, texts, n=3),
        answer_form="factored",
    )


def _cubes(rng: random.Random) -> dict:
    m = rng.choice([2, 3, 4, 5])
    n = rng.randint(1, 6)
    plus = rng.random() < 0.5
    sign = 1 if plus else -1
    cubic = (m * x) ** 3 + sign * n**3
    cubic_text = _cubic_text(cubic)
    # a^3 + b^3 = (a + b)(a^2 - ab + b^2);  a^3 - b^3 = (a - b)(a^2 + ab + b^2), with a = m x, b = n
    lin = (m, sign * n)
    quad = (m * m, -sign * m * n, n * n)
    candidates = [
        (lin, (m * m, sign * m * n, n * n)),  # wrong sign on the middle term
        ((m, -sign * n), quad),  # wrong sign in the linear factor
        (lin, (m, -sign * m * n, n * n)),  # m instead of m^2 on the leading term
    ]
    text = phrase(rng, _CUBES, cubic=cubic_text)
    word = "sum" if plus else "difference"
    working = [
        f"{cubic_text} is a {word} of cubes: ({m}x)^3 {'+' if plus else '-'} {n}^3.",
        f"a^3 {'+' if plus else '-'} b^3 = (a {'+' if plus else '-'} b)(a^2 {'-' if plus else '+'} ab + b^2) with a = {m}x, b = {n}.",
        f"Answer: {_product(lin, quad)}.",
    ]
    return _result(text, lin, quad, cubic_text, candidates, working)


def _grouping(rng: random.Random) -> dict:
    m = rng.choice([1, 2, 3])
    p = rng.choice([v for v in range(-9, 10) if v != 0])
    q = rng.choice([v for v in range(-9, 10) if v != 0 and abs(v) != abs(p) and not _splits(1, 0, v)])
    lin = (m, -p)
    quad = (1, 0, q)
    cubic_text = _cubic_text(_parse(_linear(*lin)) * _parse(_quad(*quad)))
    candidates = [
        ((m, p), quad),  # sign of the constant in the linear factor reversed
        (lin, (1, 0, -q)),  # sign of q reversed in the quadratic factor
        ((m, -q), (1, 0, p)),  # the two groups' constants swapped
    ]
    text = phrase(rng, _GROUPING, cubic=cubic_text)
    working = [
        f"Group the terms in pairs: x^2({_linear(*lin)}) {'+' if q > 0 else '-'} {abs(q)}({_linear(*lin)}).",
        f"Both groups share the factor ({_linear(*lin)}), so the cubic is ({_linear(*lin)})({_quad(*quad)}).",
        f"Answer: {_product(lin, quad)}.",
    ]
    return _result(text, lin, quad, cubic_text, candidates, working)


def _parameter(rng: random.Random) -> dict:
    p = rng.choice([v for v in range(-4, 5) if v != 0])
    qq = rng.choice([v for v in range(-5, 6) if v != 0])
    r = rng.choice([v for v in range(-6, 7) if v != 0])
    lin = (1, -p)
    quad = (1, qq, r)
    expanded = expand(_parse(_linear(*lin)) * _parse(_quad(*quad)))
    k = int(expanded.coeff(x, 2))
    # Skip draws where k coincides with the quadratic's own coefficients, so the "k used as the coefficient"
    # slip is a genuinely different factorisation.
    if k == qq or k == 0 or _splits(1, qq, r):
        return _parameter(rng)
    k_symbol = Symbol("k")
    cubic_with_k = x**3 + k_symbol * x**2 + int(expanded.coeff(x, 1)) * x + int(expanded.coeff(x, 0))
    # Independent check: the factor theorem. The printed cubic must vanish at x = p exactly when k has this value.
    check(cubic_with_k.subs({x: p, k_symbol: k}) == 0, "factor theorem check failed for the derived k")
    cubic_text = join_terms(
        [(1, "x^3"), (1, "kx^2"), (int(expanded.coeff(x, 1)), "x"), (int(expanded.coeff(x, 0)), "")]
    )
    candidates = [
        ((1, p), quad),  # sign of the root reversed
        (lin, (1, k, r)),  # k used as the coefficient of x in the quadratic
        (lin, (1, qq, -r)),  # sign of the constant reversed
    ]
    text = phrase(rng, _PARAMETER, cubic=cubic_text, sign="-" if p > 0 else "+", p=abs(p))
    working = [
        f"Factor theorem: the cubic is zero at x = {p}, so {p}^3 + k({p})^2 + ({int(expanded.coeff(x, 1))})({p}) + ({int(expanded.coeff(x, 0))}) = 0.",
        f"Solving gives k = {k}.",
        f"Dividing by (x {'-' if p > 0 else '+'} {abs(p)}) gives {_quad(*quad)}, so the answer is {_product(lin, quad)}.",
    ]
    return _result(text, lin, quad, _cubic_text(expanded), candidates, working)


@generator(
    "exd_x_f2_cubic_factor",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f2_cubic_factor(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier cubic factorisation: cubes, grouping, or an unknown coefficient."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}; this source only has 'exam'")
    shape = rng.choice(["cubes", "grouping", "parameter"])
    if shape == "cubes":
        return _cubes(rng)
    if shape == "grouping":
        return _grouping(rng)
    return _parameter(rng)
