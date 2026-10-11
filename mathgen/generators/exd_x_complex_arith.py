"""Exam-tier questions for the Cartesian complex-number skills: add, conjugate, multiply, real part,
imaginary part and modulus.

The practice tiers of these skills apply one operation to one or two numbers written as ``a + bi``. An exam
question composes two ideas, or hides the operation inside a different shape, so the learner has to choose
a method:

- ``exd_x_complex_add``: solve ``p + k * conj(q) = s`` for ``p`` (an addition with a conjugate inside it,
  run backwards).
- ``exd_x_complex_conjugate``: the conjugate of a product, a square, or a quotient.
- ``exd_x_complex_multiply``: a triple product, a cube, or a difference of two squares.
- ``exd_x_complex_real_part`` / ``exd_x_complex_imaginary_part``: a part of a power, a triple product or a
  quotient (conjugate-multiply, or spot the factor).
- ``exd_x_complex_modulus``: the modulus of a product, a power-and-conjugate, or a quotient, where
  multiplying out is possible but multiplicativity of the modulus is the efficient route.

Every question is built backwards from chosen integer components, so each key is exact. All complex
arithmetic here is done on integer pairs ``(re, im)``; the ring rule ``i * i = isq`` is a parameter, and the
"treats i squared as positive one" distractors are the same computation run with ``isq = +1``.
Each key is re-derived with sympy's imaginary unit before it is returned.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy

from ..core import check, check_equivalent, generator, join_terms, pick_distractors

Gauss = tuple[int, int]

_SMALL = (-4, -3, -2, -1, 1, 2, 3, 4)
_MODULUS_TRIPLES = ((3, 4, 5), (5, 12, 13), (8, 15, 17), (6, 8, 10), (7, 24, 25), (9, 12, 15))


# -- integer-pair arithmetic ---------------------------------------------------


def _mul(p: Gauss, q: Gauss, isq: int = -1) -> Gauss:
    return (p[0] * q[0] + isq * p[1] * q[1], p[0] * q[1] + p[1] * q[0])


def _add(p: Gauss, q: Gauss) -> Gauss:
    return (p[0] + q[0], p[1] + q[1])


def _sub(p: Gauss, q: Gauss) -> Gauss:
    return (p[0] - q[0], p[1] - q[1])


def _scale(k: int, p: Gauss) -> Gauss:
    return (k * p[0], k * p[1])


def _conj(p: Gauss) -> Gauss:
    return (p[0], -p[1])


def _sym(p: Gauss):
    return p[0] + p[1] * sympy.I


def _text(p: Gauss) -> str:
    """``a + bi`` the way a student writes it: a zero part is dropped, a unit coefficient is bare ``i``."""
    return join_terms([(p[0], ""), (p[1], "i")])


def _bracket(p: Gauss) -> str:
    return f"({_text(p)})"


def _draw(rng: random.Random, lo: int = -4, hi: int = 4) -> Gauss:
    """A Gaussian integer with both parts nonzero."""
    while True:
        p = (rng.randint(lo, hi), rng.randint(lo, hi))
        if p[0] != 0 and p[1] != 0:
            return p


def _options(key: Gauss, candidates: list[Gauss]) -> tuple[str, list[str]]:
    answer_text = _text(key)
    distractors = pick_distractors(answer_text, [_text(c) for c in candidates])
    check(len(distractors) == 3, "expected three distinct distractors")
    return answer_text, distractors


def _mcq(rng, patterns, key: Gauss, candidates: list[Gauss], working: list[str], **slots) -> dict:
    from ..core import phrase

    answer_text, distractors = _options(key, candidates)
    text = phrase(rng, patterns, **slots)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


def _number(rng, patterns, key: int, candidates: list[int], working: list[str], **slots) -> dict:
    from ..core import phrase

    answer_text = str(key)
    distractors = pick_distractors(answer_text, [str(c) for c in candidates])
    check(len(distractors) == 3, "expected three distinct distractors")
    text = phrase(rng, patterns, **slots)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


# -- add: solve p + k * conj(q) = s for p -----------------------------------------


def _build_add(rng: random.Random) -> dict:
    while True:
        p = _draw(rng, -6, 6)
        q = _draw(rng, -6, 6)
        k = rng.choice((1, 2, 3))
        sign = rng.choice((1, -1))
        s = _add(p, _scale(sign * k, _conj(q)))
        wrong_noconj = _sub(s, _scale(sign * k, q))  # forgot to conjugate q
        wrong_sign = _sub(s, _scale(-sign * k, _conj(q)))  # moved the term across without changing its sign
        wrong_k = _sub(s, _scale(sign, _conj(q)))  # dropped the multiplier k
        wrong_sub = _sub(s, _conj(q)) if k != 1 else (p[0], -p[1])
        if len({p, wrong_noconj, wrong_sign, wrong_k, wrong_sub}) == 5:
            break
    check_equivalent(
        sympy.expand(_sym(p) + sign * k * sympy.conjugate(_sym(q))),
        _sym(s),
        "the constructed p must satisfy p + k*conj(q) = s",
    )
    term = "conjugate of q" if k == 1 else f"{k} times the conjugate of q"
    op = "+" if sign == 1 else "-"
    patterns = (
        "Let q = {q}. The complex number p satisfies p {op} {term} = {s}. Which is p?",
        "The complex number p is such that p {op} {term} = {s}, where q = {q}. Find p.",
        "Given q = {q}, p {op} ({term}) = {s}. Determine p.",
    )
    return _mcq(
        rng,
        patterns,
        p,
        [wrong_noconj, wrong_sign, wrong_k, wrong_sub],
        [
            f"The conjugate of q is {_text(_conj(q))}.",
            f"So p = {_text(s)} {'-' if sign == 1 else '+'} {k}({_text(_conj(q))}) = {_text(p)}.",
        ],
        q=_text(q),
        s=_text(s),
        op=op,
        term=term,
    )


# -- conjugate of a product, a square, or a quotient ------------------------------


def _build_conjugate(rng: random.Random) -> dict:
    shape = rng.choice(("product", "square", "quotient"))
    while True:
        z = _draw(rng, -4, 4)
        w = _draw(rng, -4, 4)
        if shape == "product":
            prod = _mul(z, w)
            key = _conj(prod)
            candidates = [prod, _scale(-1, prod), _mul(_conj(z), w), _mul(_conj(z), _conj(w), isq=1)]
            expr = f"{_bracket(z)}{_bracket(w)}"
            how = "multiply first, then reverse the sign of the imaginary part"
        elif shape == "square":
            prod = _mul(z, z)
            key = _conj(prod)
            candidates = [prod, _scale(-1, prod), _mul(_conj(z), z), _mul(_conj(z), _conj(z), isq=1)]
            expr = f"{_bracket(z)}^2"
            how = "square first, then reverse the sign of the imaginary part"
        else:
            # z = q * w, so the quotient z / w is the Gaussian integer q.
            q = z
            prod = _mul(q, w)
            key = _conj(q)
            candidates = [q, _scale(-1, q), _scale(-1, key), (key[1], key[0])]
            expr = f"({_text(prod)})/({_text(w)})"
            how = "divide first, then reverse the sign of the imaginary part"
        if len({key, *candidates}) == 5 and key != candidates[0]:
            break
    if shape == "quotient":
        check_equivalent(
            sympy.conjugate(sympy.simplify(_sym(prod) / _sym(w))), _sym(key), "conjugate of the quotient must match"
        )
    elif shape == "square":
        check_equivalent(sympy.conjugate(sympy.expand(_sym(z) ** 2)), _sym(key), "conjugate of the square must match")
    else:
        check_equivalent(
            sympy.conjugate(sympy.expand(_sym(z) * _sym(w))), _sym(key), "conjugate of the product must match"
        )
    patterns = (
        "Find the conjugate of {expr}.",
        "What is the complex conjugate of {expr}?",
        "Write the conjugate of {expr} in the form a + bi.",
    )
    return _mcq(rng, patterns, key, candidates, [f"To find the conjugate of {expr}: {how}."], expr=expr)


# -- multiply: triple product, cube, difference of squares ------------------------


def _build_multiply(rng: random.Random) -> dict:
    shape = rng.choice(("triple", "cube", "squares"))
    while True:
        z = _draw(rng, -3, 3)
        w = _draw(rng, -3, 3)
        u = _draw(rng, -3, 3)
        if shape == "triple":
            key = _mul(_mul(z, w), u)
            wrong_isq = _mul(_mul(z, w, 1), u, 1)
            componentwise = (z[0] * w[0] * u[0], z[1] * w[1] * u[1])
            candidates = [wrong_isq, componentwise, _conj(key), _mul(_mul(z, w), _conj(u))]
            expr = f"{_bracket(z)}{_bracket(w)}{_bracket(u)}"
        elif shape == "cube":
            key = _mul(_mul(z, z), z)
            wrong_isq = _mul(_mul(z, z, 1), z, 1)
            componentwise = (z[0] ** 3, z[1] ** 3)
            candidates = [wrong_isq, componentwise, _conj(key), _mul(_mul(z, z), _conj(z))]
            expr = f"{_bracket(z)}^3"
        else:
            key = _sub(_mul(z, z), _mul(w, w))
            wrong_isq = _sub(_mul(z, z, 1), _mul(w, w, 1))
            componentwise = (z[0] ** 2 - w[0] ** 2, z[1] ** 2 - w[1] ** 2)
            candidates = [wrong_isq, componentwise, _conj(key), _mul(_sub(z, w), _sub(z, w))]
            expr = f"{_bracket(z)}^2 - {_bracket(w)}^2"
        if len({key, *candidates}) == 5:
            break
    if shape == "triple":
        expected = sympy.expand(_sym(z) * _sym(w) * _sym(u))
    elif shape == "cube":
        expected = sympy.expand(_sym(z) ** 3)
    else:
        expected = sympy.expand(_sym(z) ** 2 - _sym(w) ** 2)
    check_equivalent(expected, _sym(key), "the constructed product must match sympy's expansion")
    patterns = (
        "Calculate {expr}.",
        "Expand and simplify {expr}.",
        "Write {expr} in the form a + bi.",
    )
    return _mcq(
        rng,
        patterns,
        key,
        candidates,
        ["Multiply step by step, using i^2 = -1 each time, then collect real and imaginary parts."],
        expr=expr,
    )


# -- real and imaginary parts of a composed expression ----------------------------


def _build_part(rng: random.Random, part: int) -> dict:
    """``part`` is 0 for the real part, 1 for the imaginary part."""
    name = "real" if part == 0 else "imaginary"
    shapes = ("cube", "triple", "quotient") if part == 0 else ("square_times", "conj_product", "quotient")
    shape = rng.choice(shapes)
    while True:
        z = _draw(rng, -3, 3)
        w = _draw(rng, -3, 3)
        u = _draw(rng, -3, 3)
        other_part = 1 - part
        if shape == "cube":
            val = _mul(_mul(z, z), z)
            wrong = _mul(_mul(z, z, 1), z, 1)
            expr = f"{_bracket(z)}^3"
        elif shape == "triple":
            val = _mul(_mul(z, w), u)
            wrong = _mul(_mul(z, w, 1), u, 1)
            expr = f"{_bracket(z)}{_bracket(w)}{_bracket(u)}"
        elif shape == "square_times":
            val = _mul(_mul(z, z), w)
            wrong = _mul(_mul(z, z, 1), w, 1)
            expr = f"{_bracket(z)}^2{_bracket(w)}"
        elif shape == "conj_product":
            val = _mul(z, _conj(w))
            wrong = _mul(z, w)  # reads the conjugated factor as w itself
            expr = f"{_bracket(z)}{_bracket(_conj(w))}"
        else:
            q = z
            val = q
            top = _mul(q, w)
            wrong = _mul(top, w)  # multiplied instead of dividing
            expr = f"({_text(top)})/({_text(w)})"
        key = val[part]
        candidates = [
            val[other_part],
            wrong[part],
            -key,
            key + 1 if shape != "quotient" else val[other_part] + val[part],
            key - 1,
        ]
        if len({key, *candidates[:3]}) >= 4 and key != 0:
            break
    sym_val = {
        "cube": lambda: sympy.expand(_sym(z) ** 3),
        "triple": lambda: sympy.expand(_sym(z) * _sym(w) * _sym(u)),
        "square_times": lambda: sympy.expand(_sym(z) ** 2 * _sym(w)),
        "conj_product": lambda: sympy.expand(_sym(z) * sympy.conjugate(_sym(w))),
        "quotient": lambda: sympy.simplify(_sym(top) / _sym(w)) if shape == "quotient" else 0,
    }[shape]()
    check_equivalent(sympy.re(sym_val) if part == 0 else sympy.im(sym_val), key, "constructed part must match sympy")
    notation = "Re" if part == 0 else "Im"
    wording = "real part" if part == 0 else "imaginary part"
    patterns = (
        f"Find {notation}({{expr}}).",
        f"Find the {wording} of {{expr}}. Give a whole number.",
        f"What is the {wording} of the complex number {{expr}}? Give the coefficient only, a whole number.",
    )
    return _number(
        rng,
        patterns,
        key,
        candidates,
        [f"Simplify {expr} to the form a + bi, then read off the {name} part: {key}."],
        expr=expr,
    )


# -- modulus ------------------------------------------------------------------------


def _triple_point(rng: random.Random) -> tuple[Gauss, int]:
    x, y, m = rng.choice(_MODULUS_TRIPLES)
    sx, sy = rng.choice((-1, 1)), rng.choice((-1, 1))
    if rng.random() < 0.5:
        x, y = y, x
    return (sx * x, sy * y), m


def _build_modulus(rng: random.Random) -> dict:
    shape = rng.choice(("product", "power_conj", "quotient"))
    while True:
        z, mz = _triple_point(rng)
        w, mw = _triple_point(rng)
        if shape == "product":
            key = Fraction(mz * mw)
            expr = f"{_bracket(z)}{_bracket(w)}"
            candidates = [mz + mw, mz * mw + 1, mz * mz + mw, abs(z[0] * w[0]) + abs(z[1] * w[1])]
            check_val = abs(complex(*z) * complex(*w))
        elif shape == "power_conj":
            key = Fraction(mz * mz * mw)
            expr = f"{_bracket(z)}^2{_bracket(_conj(w))}"
            candidates = [mz * mw, 2 * mz + mw, mz * mz + mw, mz + mw]
            check_val = abs(complex(*z) ** 2 * complex(w[0], -w[1]))
        else:
            key = Fraction(mz**3, mw)
            if key.denominator != 1:
                continue
            expr = f"({_text(z)})^3/({_text(w)})"
            candidates = [mz * 3 - mw, mz**3 + mw, mz**2 * mw, mz**3 - mw]
            check_val = abs(complex(*z) ** 3 / complex(*w))
        k = int(key)
        if len({k, *candidates}) == len(candidates) + 1 and k <= 5000:
            break
    check(abs(check_val - k) < 1e-6, "the constructed modulus must match complex-number arithmetic")
    patterns = (
        "Find the modulus of {expr}.",
        "Calculate |{expr}|.",
        "What is the value of |{expr}|?",
    )
    return _number(
        rng,
        patterns,
        k,
        candidates,
        ["The modulus of a product is the product of the moduli, and the modulus of a quotient is the quotient."],
        expr=expr,
    )


# -- registration ---------------------------------------------------------------------

_PRESETS = {"exam": dict(tier="exam")}


def _require_exam(tier: str) -> None:
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}: this topic only has an exam tier")


@generator("exd_x_complex_add", level=12, presets=_PRESETS)
def exd_x_complex_add(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: solve p + k*conj(q) = s for p."""
    _require_exam(tier)
    return _build_add(rng)


@generator("exd_x_complex_conjugate", level=12, presets=_PRESETS)
def exd_x_complex_conjugate(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the conjugate of a product, a square or a quotient."""
    _require_exam(tier)
    return _build_conjugate(rng)


@generator("exd_x_complex_multiply", level=12, presets=_PRESETS)
def exd_x_complex_multiply(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: a triple product, a cube, or a difference of two squares of complex numbers."""
    _require_exam(tier)
    return _build_multiply(rng)


@generator("exd_x_complex_real_part", level=12, presets=_PRESETS)
def exd_x_complex_real_part(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the real part of a cube, a triple product or a quotient."""
    _require_exam(tier)
    return _build_part(rng, 0)


@generator("exd_x_complex_imaginary_part", level=12, presets=_PRESETS)
def exd_x_complex_imaginary_part(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the imaginary part of a square-times-factor, a conjugate product or a quotient."""
    _require_exam(tier)
    return _build_part(rng, 1)


@generator("exd_x_complex_modulus", level=12, presets=_PRESETS)
def exd_x_complex_modulus(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the modulus of a product, a power with a conjugate, or a quotient."""
    _require_exam(tier)
    return _build_modulus(rng)
