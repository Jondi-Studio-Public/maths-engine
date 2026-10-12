"""Generates `exam_parts_implicit_tangent` questions: a multi-part exam question on a curve given implicitly.

Skill: senior-use-implicit-differentiation-to-determine-the-gradient-of (exam tier). It replaces the
single-answer `p12c_implicit_diff_gradient` exam preset (find dy/dx only) with the connected chain a senior
exam asks (see `docs/design/multi-part-questions.md`):

(a) find dy/dx in terms of x and y                              expression, 2 marks
(b) hence the gradient of the curve at P                        fraction,   1 mark,  Subs(part_a, (x, y), (x0, y0))
(c) hence the tangent at P, in the form y = mx + c              expression, 2 marks, part_b*(x - x0) + y0
(d) hence where the tangent meets the x-axis                    fraction,   1 mark,  x0 - y0/part_b

## Construction: choose the point and the coefficients first

The point P(x0, y0) has non-zero integer coordinates and the curve is built through it: the coefficients are
drawn, then the constant is set so that P lies on the curve. Three families of curve (each with a cross term, so
the product rule is always needed):

1. `a x^2 + b xy + c y^2 = d`
2. `a x^3 + b y^3 = k xy`, with k chosen so P lies on it (k = (a x0^3 + b y0^3) / (x0 y0) must be a non-zero integer)
3. `x^2 y + a x y^2 = d`

For each family dy/dx is written by hand (term by term) and re-derived as -Fx/Fy with `sympy.diff`; the two
must agree. P is accepted only when both the numerator and the denominator of dy/dx are non-zero at P, so the
gradient exists and is non-zero and the tangent meets the x-axis at one point. y0 is non-zero so that point is
not P itself.

## Grading notes

* (a) and (c) are `expression` parts graded by equivalence: any algebraically equal form is accepted. A
  reply "y = 3x/2 - 5" is read with its "y =" label removed. No `answer_form` is set.
* (b) and (d) are `fraction` parts: graded exactly, by value.
* If a learner's (a) makes (b)'s follow-through undefined (zero denominator at P) the core finds no target,
  and (b) earns nothing for a non-canonical reply; (c) and (d) then likewise.

## Distractors (the skill's misconception ids)

* (a): `sign-error-isolating-yprime` (the negative of dy/dx), `forgot-product-rule` (one half of the product
  rule on the cross term), `forgot-chain-rule-on-power-term` / `forgot-yprime-factor` (a power of y differentiated
  with no y' attached).
* (b): the same slips carried to the value at P (the negative gradient is the sign slip), and the reciprocal.
* (c): the normal instead of the tangent, the sign-flipped gradient, dropping y0, and the sign slip on x0.
* (d): the sign slip on y0/m, multiplying instead of dividing by m, forgetting the shift by x0, and giving the
  y-intercept.
"""

from __future__ import annotations

import sympy
from sympy import Rational, symbols

from ..core import (
    check,
    fmt_signed,
    generator,
    join_terms,
    latex_answer,
    paren_negative,
    phrase,
    pick_distractors,
    rational_text,
)

x, y = symbols("x y")

_COEFFS = (-4, -3, -2, -1, 1, 2, 3, 4)
_POINT = (-4, -3, -2, -1, 1, 2, 3, 4)


def _typed(expr) -> str:
    return str(expr).replace("**", "^")


def _family_quadratic(rng, x0, y0):
    """a x^2 + b xy + c y^2 = d"""
    a = rng.choice([1, 2, 3, 4])
    b = rng.choice(_COEFFS)
    c = rng.choice(_COEFFS)
    d = a * x0**2 + b * x0 * y0 + c * y0**2
    if d == 0:
        return None
    eq = f"{join_terms([(a, 'x²'), (b, 'xy'), (c, 'y²')])} = {d}"
    F = a * x**2 + b * x * y + c * y**2 - d
    N, D = 2 * a * x + b * y, b * x + 2 * c * y
    true = (-1, N, D)
    cands = [
        ("sign", 1, N, D),
        ("product", -1, 2 * a * x, D),  # drops the "+by" piece of the product rule
        ("product", -1, N, 2 * c * y),  # drops the "+bx y'" piece
        ("chain", -1, N + 2 * c * y, b * x),  # y² differentiated with no y'
    ]
    working = [
        f"Differentiate both sides with respect to x: {join_terms([(2 * a, 'x'), (b, 'y'), (b, 'x dy/dx'), (2 * c, 'y dy/dx')])} = 0, "
        "using the product rule on the xy term and the chain rule on the y² term.",
        f"Collect the dy/dx terms: ({join_terms([(b, 'x'), (2 * c, 'y')])}) dy/dx = -({join_terms([(2 * a, 'x'), (b, 'y')])}).",
    ]
    return eq, F, N, D, true, cands, working


def _family_cubic(rng, x0, y0):
    """a x^3 + b y^3 = k xy, k chosen so that P lies on the curve."""
    for _ in range(400):
        a = rng.choice([1, 1, 2, 3])
        b = rng.choice([1, 1, 2, 3])
        num = a * x0**3 + b * y0**3
        if num % (x0 * y0) == 0 and num != 0:
            k = num // (x0 * y0)
            break
    else:  # pragma: no cover - the caller redraws P
        return None
    eq = f"{join_terms([(a, 'x³'), (b, 'y³')])} = {fmt_signed(k, 'xy')}"
    F = a * x**3 + b * y**3 - k * x * y
    N, D = k * y - 3 * a * x**2, 3 * b * y**2 - k * x
    true = (1, N, D)
    cands = [
        ("sign", -1, N, D),
        ("product", 1, N, 3 * b * y**2),  # drops the "kx dy/dx" piece
        ("product", -1, 3 * a * x**2, D),  # drops the "ky" piece
        ("chain", 1, 3 * a * x**2 + 3 * b * y**2 - k * y, k * x),  # y³ differentiated with no y'
    ]
    working = [
        f"Differentiate both sides with respect to x: {join_terms([(3 * a, 'x²'), (3 * b, 'y² dy/dx')])} = "
        f"{join_terms([(k, 'y'), (k, 'x dy/dx')])}, using the product rule on the xy term and the chain rule on the y³ term.",
        f"Collect the dy/dx terms: ({join_terms([(3 * b, 'y²'), (-k, 'x')])}) dy/dx = {join_terms([(k, 'y'), (-3 * a, 'x²')])}.",
    ]
    return eq, F, N, D, true, cands, working


def _family_product(rng, x0, y0):
    """x^2 y + a xy^2 = d"""
    a = rng.choice(_COEFFS)
    d = x0**2 * y0 + a * x0 * y0**2
    if d == 0:
        return None
    eq = f"{join_terms([(1, 'x²y'), (a, 'xy²')])} = {d}"
    F = x**2 * y + a * x * y**2 - d
    N, D = 2 * x * y + a * y**2, x**2 + 2 * a * x * y
    true = (-1, N, D)
    cands = [
        ("sign", 1, N, D),
        ("product", -1, a * y**2, D),  # x²y differentiated as if x² were constant
        ("product", -1, N, x**2),  # xy² differentiated as if x were constant: drops 2axy dy/dx
        ("chain", -1, N + 2 * a * x * y, x**2),  # the y² factor differentiated with no y'
    ]
    working = [
        f"Differentiate both sides with respect to x, using the product rule on each term: "
        f"{join_terms([(2, 'xy'), (1, 'x² dy/dx'), (a, 'y²'), (2 * a, 'xy dy/dx')])} = 0.",
        f"Collect the dy/dx terms: ({join_terms([(1, 'x²'), (2 * a, 'xy')])}) dy/dx = -({join_terms([(2, 'xy'), (a, 'y²')])}).",
    ]
    return eq, F, N, D, true, cands, working


_FAMILIES = (_family_quadratic, _family_cubic, _family_product)


#: Points where two rational expressions in x, y are compared exactly. Every dy/dx here has degree at most 4 over
#: degree 4, so agreement at this many generic rational points means they are the same function.
_PROBES = tuple(
    (Rational(a, 3), Rational(b, 7))
    for a, b in (
        (2, 5),
        (-5, 3),
        (7, -4),
        (11, 9),
        (-8, -13),
        (13, 2),
        (4, 17),
        (-9, 6),
        (5, -11),
        (17, 10),
        (-3, 8),
        (9, 19),
    )
)


def _same(left, right) -> bool:
    """Whether two rational expressions in x, y are the same function: equal, in exact arithmetic, at twelve
    generic rational points. Cheap (microseconds) where `simplify`, as `check_equivalent` runs it, is tens of
    milliseconds; a finite point where one side is undefined and the other is not counts as a difference."""
    for px, py in _PROBES:
        sub = {x: px, y: py}
        a, b = left.xreplace(sub), right.xreplace(sub)
        if a != b:
            return False
    return True


def _wrap(e, *, always: bool) -> str:
    text = _typed(e)
    needs = isinstance(e, sympy.Add) or (always and not (e.is_Symbol or e.is_Integer or e.is_Pow))
    return f"({text})" if needs else text


def _frac(sign, num, den) -> str:
    """ "-(2*x + y)/(x + 2*y)": the written form of sign * num / den, bracketing so it reads back correctly."""
    num, den = sympy.expand(num), sympy.expand(den)
    # Lead each of numerator and denominator with a positive coefficient, moving the sign out to the front.
    if num.as_ordered_terms()[0].as_coeff_Mul()[0] < 0:
        sign, num = -sign, sympy.expand(-num)
    if den.as_ordered_terms()[0].as_coeff_Mul()[0] < 0:
        sign, den = -sign, sympy.expand(-den)
    return ("-" if sign < 0 else "") + f"{_wrap(num, always=False)}/{_wrap(den, always=True)}"


def _line_text(slope, intercept) -> str:
    return "y = " + _typed(slope * x + intercept)


def _fraction_part(prompt, value, marks, working, candidates, follow_through=None):
    """A fraction part whose answer is the exact Rational `value`."""
    value = Rational(value)
    answer_text = rational_text(value)
    cands = [Rational(c) for c in candidates if c is not None and c.is_finite]
    texts = pick_distractors(answer_text, [rational_text(c) for c in cands])
    for t in texts:
        check(Rational(t) != value, "a fraction distractor equals the answer")
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer={"p": int(value.p), "q": int(value.q)},
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        marks=marks,
        working=working,
        distractors=texts,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


@generator("exam_parts_implicit_tangent", level=17, presets={"exam": {}})
def exam_parts_implicit_tangent(rng):
    """A curve with a cross term and a point P on it: dy/dx, the gradient at P, the tangent, its x-intercept."""
    for _ in range(200):
        x0, y0 = rng.choice(_POINT), rng.choice(_POINT)
        eq, F, N, D, true, cands, a_working = rng.choice(_FAMILIES)(rng, x0, y0) or (None,) * 7
        if eq is None:
            continue
        n0, d0 = N.subs({x: x0, y: y0}), D.subs({x: x0, y: y0})
        if n0 != 0 and d0 != 0:
            break
    else:  # pragma: no cover
        check(False, "no usable point found")

    # P lies on the curve, and dy/dx agrees with the implicit-differentiation route -Fx/Fy.
    check(F.subs({x: x0, y: y0}) == 0, "P is not on the curve")
    sign, num, den = true
    true = sign * num / den
    check(_same(true, -sympy.diff(F, x) / sympy.diff(F, y)), "dy/dx disagrees with -Fx/Fy")

    # ---- exact values for every part ---------------------------------------------------------------------
    m = Rational(true.subs({x: x0, y: y0}))
    check(m != 0, "gradient at P is zero")
    c = y0 - m * x0
    tangent = m * x + c
    x_int = x0 - Rational(y0) / m
    check(tangent.subs(x, x_int) == 0, "x-intercept is not on the tangent")
    check(tangent.subs(x, x0) == y0, "tangent does not pass through P")

    # ---- part (a) ------------------------------------------------------------------------------------------
    a_text = _frac(sign, num, den)
    a_cands = []
    # One slip of each kind first (sign, chain rule, product rule), so the three offered are three different errors.
    for kind, sg, nu, de in sorted(cands, key=lambda c: {"sign": 0, "chain": 1, "product": 2}[c[0]]):
        e = sg * nu / de
        if not _same(e, true):
            a_cands.append((kind, e, _frac(sg, nu, de)))
    check(all(not _same(true, e) for _, e, _ in a_cands), "a dy/dx distractor equals the answer")
    a_dist = pick_distractors(a_text, [t for _, _, t in a_cands])
    part_a = dict(
        prompt="Find dy/dx in terms of x and y.",
        answer_kind="expression",
        answer=str(true),
        answer_text=a_text,
        answer_latex=latex_answer(true),
        marks=2,
        working=a_working + [f"So dy/dx = {a_text.replace('*', '')}."],
        distractors=a_dist,
    )

    # ---- part (b) ------------------------------------------------------------------------------------------
    b_cands = [-m, 1 / m]
    for _, e, _t in a_cands:
        try:
            v = e.subs({x: x0, y: y0})
            if v.is_Rational:
                b_cands.append(v)
        except (ZeroDivisionError, TypeError):
            continue
    part_b = _fraction_part(
        f"Hence find the gradient of the curve at P({x0}, {y0}).",
        m,
        1,
        [f"At ({x0}, {y0}): dy/dx = {rational_text(m)}."],
        b_cands,
        follow_through=f"Subs(part_a, (x, y), ({x0}, {y0}))",
    )

    # ---- part (c) ------------------------------------------------------------------------------------------
    c_text = _line_text(m, c)
    c_cands = [
        ("normal line", (-1 / m) * (x - x0) + y0),
        ("sign-flipped gradient", -m * (x - x0) + y0),
        ("dropped y0", m * (x - x0)),
        ("sign slip on x0", m * (x + x0) + y0),
    ]
    c_cands = [(k, sympy.expand(e)) for k, e in c_cands]
    check(all(sympy.expand(tangent - e) != 0 for _, e in c_cands), "a tangent distractor equals the answer")
    c_dist = pick_distractors(c_text, ["y = " + _typed(e) for _, e in c_cands])
    part_c = dict(
        prompt=f"Hence find the equation of the tangent to the curve at P({x0}, {y0}), in the form y = mx + c.",
        answer_kind="expression",
        answer=str(tangent),
        answer_text=c_text,
        answer_latex=latex_answer(c_text),
        marks=2,
        working=[
            f"y - {paren_negative(y0)} = {rational_text(m)}(x - {paren_negative(x0)}).",
            f"So {c_text.replace('*', '')}.",
        ],
        distractors=c_dist,
        follow_through=f"part_b*(x - {paren_negative(x0)}) + {paren_negative(y0)}",
    )

    # ---- part (d) ------------------------------------------------------------------------------------------
    d_cands = [x0 + Rational(y0) / m, x0 - y0 * m, Rational(-y0) / m, c]
    part_d = _fraction_part(
        "Hence find the x-coordinate of the point where this tangent meets the x-axis.",
        x_int,
        1,
        [f"Put y = 0 in the tangent: 0 = {join_terms([(m, 'x'), (c, '')])}, so x = {rational_text(x_int)}."],
        d_cands,
        follow_through=f"{paren_negative(x0)} - {paren_negative(y0)}/part_b",
    )

    stem = phrase(
        rng,
        (
            "The curve C has equation {eq}. The point P({x0}, {y0}) lies on C.",
            "A curve C is defined implicitly by {eq}, and P({x0}, {y0}) is a point on C.",
            "Consider the curve C given by {eq}. The point P({x0}, {y0}) lies on this curve.",
        ),
        eq=eq,
        x0=x0,
        y0=y0,
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
