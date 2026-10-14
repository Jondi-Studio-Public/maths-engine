"""Generates `r3h_families` questions (easy, medium, hard) and the exam chain `exd_x_r3h_families`.

``families-of-curves-and-linearity`` (``differentiation``, answers ``expression`` and ``number``): the derivative of a
function fixes it only up to a constant, so ``f'(x)`` gives a family ``f(x) = F(x) + C`` and one point picks the member;
and differentiation and integration are linear, so a combination ``r f + s g`` can be handled from what is known about
``f`` and ``g`` separately, without ever finding them.

* easy: the derivative of a combination ``r f(x) + s g(x) + u x^2 + c`` at a point, from ``f'`` and ``g'`` at that point
  (number). The constant ``c`` has no derivative.
* medium: the integral of a combination over an interval, from the integrals of ``f`` and ``g`` over it (number).
* hard: two derivatives ``f'`` and ``g'`` are given with one value of each; the learner finds both members of their
  families and writes the combination ``r f(x) + s g(x)`` (expression).
* exam (``exd_x_r3h_families``): (a) the antiderivative with zero constant, (b) the constant from a point, (c) a value of
  the function, (d) an integral of ``r g + s f'`` over an interval from a given integral of ``g`` (follows through
  from (a)).

Every answer is checked a second way: the hard answer is differentiated and compared with the combination of the given
derivatives, and its values at the given point are compared with the given values; the other numbers are recomputed from
the stem.
"""

from __future__ import annotations

import sympy

from ..core import attempts, check, generator, rounds_unambiguously
from ._r3h_kit import (
    X,
    Redraw,
    build_with_redraws,
    expr_options,
    expr_part,
    fraction_part,
    frac_text,
    machine,
    number_options,
    number_part,
    plain,
    same_function,
    say_marked,
)

_R = sympy.Rational
x = X

_FP = "f'(x)"
_GP = "g'(x)"
_NONZERO = (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)


def _sgn(value) -> str:
    return "-" if value < 0 else "+"


def _comb(r, s, left="f(x)", right="g(x)") -> str:
    """``r f(x) + s g(x)`` written the way a person would."""

    def term(c, name, first):
        mag = abs(c)
        body = name if mag == 1 else f"{mag}{name}"
        if first:
            return f"-{body}" if c < 0 else body
        return f" {_sgn(c)} {body}"

    return term(r, left, True) + term(s, right, False)


def _signed(*terms) -> str:
    """``(coefficient, body)`` pairs joined as ``a - b + c`` with no ``+ -`` anywhere."""
    out = ""
    for i, (c, body) in enumerate(terms):
        mag = abs(c)
        text = body if mag == 1 and not body[0].isdigit() and body[0] != "(" else f"{mag}{body}"
        out += (f"-{text}" if c < 0 else text) if i == 0 else f" {_sgn(c)} {text}"
    return out


# ---------------------------------------------------------------------------------------------------------------
# derivative families with an exact antiderivative


def _draw_fprime(rng, kind: str):
    """``(f'(x), F(x))`` for a derivative of the given kind, with ``F`` having whole-number coefficients.

    ``F`` is written down from the form of ``f'``, not integrated by SymPy (which is the slow step); the tests
    differentiate it and check it against SymPy's own integral.
    """
    if kind == "poly":
        a = 3 * rng.choice([-3, -2, -1, 1, 2, 3])
        b = 2 * rng.choice([-3, -2, -1, 0, 1, 2, 3])
        c = rng.choice(_NONZERO)
        return a * x**2 + b * x + c, (a // 3) * x**3 + (b // 2) * x**2 + c * x
    if kind == "exp":
        k = rng.choice([1, 2, 3, -1, -2])
        a = k * rng.choice([-3, -2, -1, 1, 2, 3])
        b = rng.choice([-4, -3, -2, 2, 3, 4, 5])
        return a * sympy.exp(k * x) + b, (a // k) * sympy.exp(k * x) + b * x
    k = rng.choice([1, 2, 3])
    a = k * rng.choice([-3, -2, -1, 1, 2, 3])
    b = rng.choice([-4, -3, -2, 2, 3, 4, 5])
    if kind == "cos":
        return a * sympy.cos(k * x) + b, (a // k) * sympy.sin(k * x) + b * x
    return a * sympy.sin(k * x) + b, -(a // k) * sympy.cos(k * x) + b * x


def _draw_member(rng, fp, F, x0):
    """The constant of the family and the value ``f(x0)`` that picks the member."""
    for _ in attempts(60, "a nonzero constant of integration"):
        c = rng.randint(-9, 9)
        if c != 0:
            break
    y0 = sympy.nsimplify(F.subs(x, x0) + c)
    check(y0.is_Integer, "the value at the point is not a whole number")
    return c, int(y0)


# ---------------------------------------------------------------------------------------------------------------
# easy: linearity of the derivative at a point


def _easy(rng):
    x0 = rng.randint(-3, 4)
    p = rng.choice(_NONZERO)
    q = rng.choice(_NONZERO)
    r = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
    s = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
    u = rng.choice([-3, -2, -1, 1, 2, 3])
    c = rng.choice(_NONZERO)
    value = r * p + s * q + 2 * u * x0
    # an independent route: differentiate a concrete pair of functions with those slopes at x0
    xs = sympy.Symbol("x")
    f = p * (xs - x0) + 1
    g = q * (xs - x0) - 2
    h = r * f + s * g + u * xs**2 + c
    check(sympy.diff(h, xs).subs(xs, x0) == value, "the derivative of the combination fails the check")
    wrongs = [
        ("r3h-families-constant-has-derivative", value + c),
        ("r3h-families-square-term-forgotten", r * p + s * q),
        ("r3h-families-scalar-not-applied", p + q + 2 * u * x0),
        ("r3h-families-derivatives-multiplied", r * p * s * q + 2 * u * x0),
        ("r3h-families-power-rule-not-lowered", r * p + s * q + u * x0**2),
        ("r3h-families-sign-of-second-term", r * p - s * q + 2 * u * x0),
    ]
    rest = f" {_sgn(u)} {abs(u) if abs(u) != 1 else ''}x^2 {_sgn(c)} {abs(c)}"
    stem, latex = say_marked(
        f"The functions «f» and «g» are differentiable and «f'({x0}) = {p}», «g'({x0}) = {q}». "
        f"Let «h(x) = {_comb(r, s)}{rest}». Find «h'({x0})»."
    )
    dist = number_options(float(value), 0, [(t, float(v)) for t, v in wrongs], gap=1)
    check(len(dist) >= 2, "too few distractors")
    return dict(
        text=stem,
        latex=latex,
        answer=int(value),
        answer_text=str(int(value)),
        answer_kind="number",
        distractors=dist,
        working=[
            f"h'(x) = {_signed((r, _FP), (s, _GP), (2 * u, 'x'))}, because the constant {c} has derivative 0.",
            f"h'({x0}) = {_signed((r, f'({p})'), (s, f'({q})'), (2 * u, f'({x0})'))} = {value}.",
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# medium: linearity of the definite integral


def _medium(rng):
    a = rng.randint(-2, 2)
    b = a + rng.randint(2, 4)
    P = rng.choice(_NONZERO)
    Q = rng.choice(_NONZERO)
    r = rng.choice([-4, -3, -2, 2, 3, 4])
    s = rng.choice([-4, -3, -2, 2, 3, 4])
    use_x = rng.random() < 0.6
    if use_x:
        u = rng.choice([-4, -2, 2, 4])
        extra = _R(u * (b * b - a * a), 2)
        extra_text = f" {_sgn(u)} {abs(u) if abs(u) != 1 else ''}x"
        wrong_extra = [
            ("r3h-families-integral-of-term-not-integrated", u * (b - a)),
            ("r3h-families-lower-limit-ignored", _R(u * b * b, 2)),
        ]
    else:
        u = rng.choice([-5, -3, -2, 2, 3, 5])
        extra = u * (b - a)
        extra_text = f" {_sgn(u)} {abs(u)}"
        wrong_extra = [
            ("r3h-families-integral-of-term-not-integrated", u),
            ("r3h-families-lower-limit-ignored", u * b),
        ]
    check(sympy.Rational(extra).is_Integer, "the extra integral is not whole")
    base = r * P + s * Q
    value = base + extra
    # an independent route: integrate concrete functions whose integrals over [a, b] are P and Q
    xs = sympy.Symbol("x")
    f = sympy.Rational(P, b - a) + 0 * xs
    g = sympy.Rational(Q, b - a) + 0 * xs
    ext = u * xs if use_x else sympy.Integer(u)
    check(sympy.integrate(r * f + s * g + ext, (xs, a, b)) == value, "the integral of the combination fails the check")
    wrongs = [
        ("r3h-families-scalar-not-applied", P + Q + extra),
        ("r3h-families-sign-of-second-term", r * P - s * Q + extra),
        ("r3h-families-integral-of-product", r * P * s * Q + extra),
        *[(tag, base + w) for tag, w in wrong_extra],
        ("r3h-families-extra-term-dropped", base),
    ]
    stem, latex = say_marked(
        f"For two functions «f» and «g» it is known that «∫ f(x) dx = {P}» and «∫ g(x) dx = {Q}», where each integral "
        f"is taken from «x = {a}» to «x = {b}». Find the value of «∫ ({_comb(r, s)}{extra_text}) dx» over the same interval."
    )
    dist = number_options(float(value), 0, [(t, float(v)) for t, v in wrongs], gap=1)
    check(len(dist) >= 2, "too few distractors")
    return dict(
        text=stem,
        latex=latex,
        answer=int(value),
        answer_text=str(int(value)),
        answer_kind="number",
        distractors=dist,
        working=[
            f"The integral of a sum is the sum of the integrals, and constants come out: {_signed((r, f'({P})'), (s, f'({Q})'))} = {base}.",
            f"The remaining term integrates to {frac_text(extra)}, so the total is {value}.",
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# hard: two families, two members, one combination


def _hard(rng):
    kinds = rng.choice(
        [("poly", "poly"), ("poly", "exp"), ("exp", "poly"), ("poly", "cos"), ("sin", "poly"), ("exp", "cos")]
    )
    x0 = rng.randint(-2, 2) if kinds == ("poly", "poly") else 0
    fp, F = _draw_fprime(rng, kinds[0])
    gp, G = _draw_fprime(rng, kinds[1])
    cf, yf = _draw_member(rng, fp, F, x0)
    cg, yg = _draw_member(rng, gp, G, x0)
    r = rng.choice([-3, -2, 2, 3, 4])
    s = rng.choice([-3, -2, 2, 3, 4])
    f = F + cf
    g = G + cg
    h = sympy.expand(r * f + s * g)
    # the independent route: h differentiates to the combination of the given derivatives and takes the right value at x0
    check(same_function(sympy.diff(h, x), r * fp + s * gp, x), "the combination has the wrong derivative")
    check(sympy.simplify(h.subs(x, x0) - (r * yf + s * yg)) == 0, "the combination has the wrong value at the point")
    wrongs = [
        ("r3h-families-constant-dropped", sympy.expand(r * F + s * G)),
        ("r3h-families-constant-not-scaled", sympy.expand(r * F + s * G + cf + cg)),
        ("r3h-families-derivative-combined", sympy.expand(r * fp + s * gp)),
        ("r3h-families-sign-of-second-term", sympy.expand(r * f - s * g)),
        ("r3h-families-scalar-not-applied", sympy.expand(r * f + g)),
        ("r3h-families-values-not-combined", sympy.expand(r * F + s * G + r * yf + s * yg)),
    ]
    stem, latex = say_marked(
        f"The functions «f» and «g» satisfy «f'(x) = {plain(fp)}», «f({x0}) = {yf}», «g'(x) = {plain(gp)}» and "
        f"«g({x0}) = {yg}». Find «h(x)» where «h(x) = {_comb(r, s)}»."
    )
    dist = expr_options(h, wrongs, 4, x)
    check(len(dist) >= 2, "too few distractors")
    return dict(
        text=stem,
        latex=latex,
        answer=machine(h),
        answer_text=plain(h),
        answer_kind="expression",
        distractors=dist,
        working=[
            f"f(x) = {plain(F)} + C, and f({x0}) = {yf} gives C = {cf}; g(x) = {plain(G)} + D, and g({x0}) = {yg} gives D = {cg}.",
            f"h(x) = {_comb(r, s)} = {plain(h)}.",
        ],
    )


@generator(
    "r3h_families",
    level=15,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3h_families(rng, tier="easy"):
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    return build_with_redraws(rng, build)


# ---------------------------------------------------------------------------------------------------------------
# exam chain


def _exam_once(rng):
    kind = rng.choice(["poly", "poly", "exp"])
    fp, F = _draw_fprime(rng, kind)
    x0 = rng.randint(-2, 2) if kind == "poly" else 0
    c, y0 = _draw_member(rng, fp, F, x0)
    f = F + c
    dp = 0 if kind == "poly" else 2
    x1 = rng.choice([v for v in (-1, 1, 2, 3) if v != x0]) if kind == "poly" else 1
    a = rng.randint(-1, 1)
    b = a + rng.randint(1, 2)
    if kind == "exp":
        a, b = 0, 1
    m = rng.choice(_NONZERO)
    r = rng.choice([-3, -2, 2, 3, 4])
    s = rng.choice([-3, -2, 2, 3, 4])
    val_c = float(sympy.N(f.subs(x, x1), 30))
    span = float(sympy.N(F.subs(x, b) - F.subs(x, a), 30))
    val_d = r * m + s * span
    if dp and not (rounds_unambiguously(val_c, dp, 0.1) and rounds_unambiguously(val_d, dp, 0.1)):
        raise Redraw
    if abs(val_c) > 400 or abs(val_d) > 400:
        raise Redraw
    # independent route for (d): integrate numerically (midpoint rule) the combination of the known pieces
    n = 4000
    h = (b - a) / n
    fpn = sympy.lambdify(x, fp, "math")
    approx = s * sum(fpn(a + (i + 0.5) * h) for i in range(n)) * h + r * m
    check(abs(approx - val_d) < 1e-3 * max(1, abs(val_d)), "the integral fails the numerical check")
    stem, latex = say_marked(f"A function «f» satisfies «f'(x) = {plain(fp)}» and «f({x0}) = {y0}».")

    def at(p, point):
        return f"Subs({p}, (x, e), ({point}, E))"

    parts = [
        expr_part(
            "Find «F(x)», the antiderivative of «f'(x)» whose constant of integration is 0.",
            F,
            2,
            [f"F(x) = {plain(F)}."],
            [
                ("r3h-antiderivative-differentiated", sympy.diff(fp, x)),
                ("r3h-antiderivative-sign", -F),
                ("r3h-antiderivative-times-x", sympy.expand(x * fp)),
                ("r3h-families-integrated-without-dividing", _undivided(fp)),
            ],
        ),
        fraction_part(
            f"Then «f(x) = F(x) + C» for a constant «C». Use «f({x0}) = {y0}» to find «C».",
            c,
            1,
            [f"C = {y0} - F({x0}) = {c}."],
            [
                ("r3h-families-constant-sign", -c),
                ("r3h-families-constant-is-value", y0),
                ("r3h-families-constant-adds-antiderivative", y0 + F.subs(x, x0)),
            ],
            follow_through=f"{y0} - {at('part_a', frac_text(x0))}",
        ),
        number_part(
            f"Find «f({x1})»" + (", correct to 2 decimal places." if dp else "."),
            val_c,
            dp,
            1,
            [f"f({x1}) = F({x1}) + C = {val_c:.{dp}f}."],
            [
                ("r3h-families-constant-dropped", float(sympy.N(F.subs(x, x1), 30))),
                ("r3h-families-constant-sign", float(sympy.N(F.subs(x, x1) - c, 30))),
                ("r3h-families-value-is-derivative", float(sympy.N(fp.subs(x, x1), 30))),
                ("r3h-families-constant-is-value", float(sympy.N(F.subs(x, x1) + y0, 30))),
            ],
            follow_through=f"{at('part_a', x1)} + {frac_text(c)}",
            gap=1 if not dp else None,
        ),
        number_part(
            f"A second function «g» has «∫ g(x) dx = {m}», taken from «x = {a}» to «x = {b}». Find «∫ ({_comb(r, s, 'g(x)', _FP)}) dx» over the same interval"
            + (", correct to 2 decimal places." if dp else "."),
            val_d,
            dp,
            2,
            [
                f"The integral of f'(x) over [{a}, {b}] is F({b}) - F({a}) = {span:.{dp}f}, so the result is {_signed((r, f'({m})'), (s, f'({span:.{dp}f})'))} = {val_d:.{dp}f}."
            ],
            [
                ("r3h-families-scalar-not-applied", m + s * span),
                ("r3h-families-lower-limit-ignored", r * m + s * float(sympy.N(F.subs(x, b), 30))),
                ("r3h-families-derivative-evaluated", r * m + s * float(sympy.N(fp.subs(x, b) - fp.subs(x, a), 30))),
                ("r3h-families-sign-of-second-term", r * m - s * span),
                ("r3h-families-integral-of-product", r * m * s * span if abs(r * m * s * span) < 1e4 else None),
            ],
            follow_through=f"{r}*{m} + {s}*({at('part_a', b)} - {at('part_a', a)})",
            gap=1 if not dp else None,
        ),
    ]
    return dict(text=stem, latex=latex, parts=parts)


def _undivided(fp):
    """``fp`` integrated term by term, raising a power but never dividing by the new power or the inner coefficient."""
    out = 0
    for term in sympy.Add.make_args(sympy.expand(fp)):
        if term.has(sympy.exp):
            out += term
        else:
            coeff, power = term.as_coeff_exponent(x)
            out += coeff * x ** (power + 1)
    return out


@generator("exd_x_r3h_families", level=18, presets={"exam": dict(tier="exam")})
def exd_x_r3h_families(rng, tier="exam"):
    """Exam tier: antiderivative, constant, value, integral by linearity."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_with_redraws(rng, _exam_once)
