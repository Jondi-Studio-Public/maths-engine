"""Generates `r3i_parametric_conics` questions (easy, medium, hard) and the exam chain `r3i_exam_parametric_conics`.

Skill ``parametric-to-cartesian-conics`` (``vectors``, answer ``expression``): eliminating the parameter from
parametric equations to reach a Cartesian relation, for lines and parabolas, circles and ellipses (cos and sin),
hyperbolas (sec and tan) and a polar line. The answer is always one expression in ``x`` (``y`` or ``y²`` or
``(y - k)²`` in terms of ``x``), because a two-variable equation is not graded.

* easy: ``x = pt + q``, ``y = rt² + s`` (or linear in t); y in terms of x.
* medium: ``x = a cos(t), y = b sin(t)`` (y² in terms of x), ``x = a sec(t), y = b tan(t)`` (y² in terms of x), or the
  polar line ``r(cos(θ) + m sin(θ)) = p`` (y in terms of x).
* hard: a shifted ellipse (``(y - k)²``), ``x = p(t + 1/t), y = q(t - 1/t)``, ``x = a sin(t), y = c + d cos(2t)``, or the
  rational circle/ellipse ``x = a(1 - t²)/(1 + t²), y = 2bt/(1 + t²)``.
* exam (``r3i_exam_parametric_conics``): an ellipse ``x = a cos(t), y = b sin(t)``; (a) x when t = π/3, (b) y² when
  t = π/3, (c) y² in terms of x, (d) y² at the point whose x is one more than the answer to (a) (follows through from (a)).

Every key is checked numerically against the parametric equations at several values of the parameter, with the
``math`` module (the independent route), and every distractor is a different function of x from the key.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, phrase
from ._r3i_kit import expr_dist, expr_fields, nice, num_cands, num_fields, pick_text, retry_short

X = sympy.Symbol("x")
_TS = (0.3, 0.7, 1.1, 1.9, 2.4, 2.9, 4.1, 5.3)


def _sg(n: int) -> str:
    return f" + {n}" if n >= 0 else f" - {-n}"


def _text(expr) -> str:
    """An expression of x as a learner types it, with `x²`-style powers kept as `^`."""
    return nice(expr)


def _verify(xf, yf, target, key, name: str) -> None:
    """The key, evaluated at x(t), equals the target built from y(t), at every sample of t (the independent route)."""
    fn = sympy.lambdify(X, key, "math")
    for t in _TS:
        try:
            xv, yv = xf(t), yf(t)
        except ZeroDivisionError:
            continue
        check(math.isclose(fn(xv), target(yv), rel_tol=1e-7, abs_tol=1e-7), f"{name}: the key fails at t = {t}")


def _cands(key, pairs, count=3):
    """Wrong expressions that are different functions from the key, as tagged typed answers."""
    out, seen = [], {_text(key)}
    for expr, tag in pairs:
        if expr is None:
            continue
        diff = sympy.expand(expr - key)
        text = _text(expr)
        if diff == 0 or text in seen:
            continue
        seen.add(text)
        out.append(expr_dist(text, tag))
    return out


def _ask(target: str) -> str:
    return f"Find {target} in terms of x."


def _coef(n: int) -> str:
    return "" if n == 1 else ("-" if n == -1 else str(n))


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(200, "a line or parabola with a distinct inverse"):
        p = rng.choice([2, 3, 4, -2, -3, 5])
        q = rng.choice([x for x in range(-6, 7) if x != 0])
        r = rng.choice([1, 2, 3, -1, -2])
        s = rng.choice([x for x in range(-7, 8) if x != 0])
        quad = rng.random() < 0.7
        if quad or r != 1:
            break
    t_of_x = (X - q) / p
    key = sympy.expand(r * t_of_x**2 + s) if quad else sympy.expand(r * t_of_x + s)
    xs = f"{_coef(p)}t{_sg(q)}"
    ys = f"{_coef(r)}t²{_sg(s)}" if quad else f"{_coef(r)}t{_sg(s)}"
    _verify(
        lambda t: p * t + q,
        (lambda t: r * t * t + s) if quad else (lambda t: r * t + s),
        lambda y: y,
        key,
        "easy",
    )
    fields = expr_fields(key)
    unsolved = r * X**2 + s if quad else r * X + s
    cands = _cands(
        key,
        [
            (unsolved, "r3i-parametric-parameter-not-eliminated"),
            (r * ((X + q) / p) ** 2 + s if quad else r * ((X + q) / p) + s, "r3i-parametric-solved-for-t-wrong-sign"),
            (r * ((X - q) * p) ** 2 + s if quad else r * ((X - q) * p) + s, "r3i-parametric-solved-for-t-multiplied"),
            (r * (X - q) ** 2 / p + s if quad else r * (X - q) / p**2 + s, "r3i-parametric-scale-lost-in-squaring"),
            (r * t_of_x**2 if quad else r * t_of_x, "r3i-parametric-constant-dropped"),
        ],
    )
    return dict(
        text=f"A curve has parametric equations x = {xs}, y = {ys}. {_ask('y')}",
        working=[
            f"From x = {xs}, t = {_text(t_of_x)}.",
            f"Substitute into y: y = {_text(key)}.",
        ],
        distractors=pick_text(fields, cands),
        **fields,
    )


# -- medium -------------------------------------------------------------------------------------------------------


def _medium_ellipse(rng):
    a, b = rng.sample([2, 3, 4, 5, 6], 2) if rng.random() < 0.85 else (rng.choice([2, 3, 4, 5]),) * 2
    key = sympy.expand(b**2 - sympy.Rational(b**2, a**2) * X**2)
    _verify(lambda t: a * math.cos(t), lambda t: b * math.sin(t), lambda y: y * y, key, "ellipse")
    fields = expr_fields(key)
    cands = _cands(
        key,
        [
            (b**2 - sympy.Rational(b**2, a) * X**2, "r3i-parametric-denominator-not-squared"),
            (b**2 + sympy.Rational(b**2, a**2) * X**2, "r3i-parametric-conic-sign-confused"),
            (b - sympy.Rational(b, a**2) * X**2, "r3i-parametric-coefficient-not-squared"),
            (1 - sympy.Rational(1, a**2) * X**2, "r3i-parametric-scale-dropped"),
            (b**2 * X**2 / a**2 - b**2, "r3i-parametric-conic-sign-confused"),
        ],
    )
    return dict(
        text=f"A curve has parametric equations x = {_coef(a)}cos(t), y = {_coef(b)}sin(t). {_ask('y²')}",
        working=[
            f"cos(t) = x/{a} and sin(t) = y/{b}, and cos²(t) + sin²(t) = 1.",
            f"So (x/{a})² + (y/{b})² = 1, giving y² = {_text(key)}.",
        ],
        distractors=pick_text(fields, cands),
        **fields,
    )


def _medium_hyperbola(rng):
    a, b = rng.sample([2, 3, 4, 5, 6], 2) if rng.random() < 0.85 else (rng.choice([2, 3, 4, 5]),) * 2
    key = sympy.expand(sympy.Rational(b**2, a**2) * X**2 - b**2)
    _verify(lambda t: a / math.cos(t), lambda t: b * math.tan(t), lambda y: y * y, key, "hyperbola")
    fields = expr_fields(key)
    cands = _cands(
        key,
        [
            (b**2 - sympy.Rational(b**2, a**2) * X**2, "r3i-parametric-conic-sign-confused"),
            (sympy.Rational(b**2, a**2) * X**2 + b**2, "r3i-parametric-sec-tan-identity-sign"),
            (sympy.Rational(b**2, a) * X**2 - b**2, "r3i-parametric-denominator-not-squared"),
            (sympy.Rational(b, a**2) * X**2 - b, "r3i-parametric-coefficient-not-squared"),
            (sympy.Rational(b**2, a**2) * X**2 - 1, "r3i-parametric-scale-dropped"),
        ],
    )
    return dict(
        text=f"A curve has parametric equations x = {_coef(a)}sec(t), y = {_coef(b)}tan(t). {_ask('y²')}",
        working=[
            f"sec(t) = x/{a} and tan(t) = y/{b}, and sec²(t) - tan²(t) = 1.",
            f"So (x/{a})² - (y/{b})² = 1, giving y² = {_text(key)}.",
        ],
        distractors=pick_text(fields, cands),
        **fields,
    )


def _medium_polar(rng):
    for _ in attempts(200, "a polar line with m not 1"):
        m = rng.choice([2, 3, 4, -2, -3, 5])
        p = rng.choice([x for x in range(-9, 10) if x != 0])
        if abs(m) != 1:
            break
    key = sympy.expand((p - X) / sympy.Integer(m))
    fields = expr_fields(key)
    # r cos θ = x and r sin θ = y; checked on points of the polar line r = p / (cos θ + m sin θ).
    for th in (0.2, 0.5, 0.9, 1.3, 2.0, 2.6):
        den = math.cos(th) + m * math.sin(th)
        if abs(den) < 0.2:
            continue
        r = p / den
        xv, yv = r * math.cos(th), r * math.sin(th)
        check(math.isclose(float(key.subs(X, xv)), yv, rel_tol=1e-7, abs_tol=1e-7), "polar line key fails")
    cands = _cands(
        key,
        [
            (p - X, "r3i-parametric-coefficient-of-y-ignored"),
            ((p + X) / sympy.Integer(m), "r3i-parametric-sign-lost-moving-term"),
            (sympy.Rational(p, m) - X, "r3i-parametric-divided-only-one-term"),
            ((X - p) / sympy.Integer(m), "r3i-parametric-sign-lost-moving-term"),
            (sympy.Integer(m) * (p - X), "r3i-parametric-multiplied-not-divided"),
        ],
    )
    return dict(
        text=(
            f"A line has polar equation r(cos(θ) {'+' if m > 0 else '-'} {abs(m)}sin(θ)) = {p}, where x = r cos(θ) and "
            f"y = r sin(θ). {_ask('y')}"
        ),
        working=[
            f"Expand: r cos(θ) {'+' if m > 0 else '-'} {abs(m)} r sin(θ) = {p}, so x {'+' if m > 0 else '-'} {abs(m)}y = {p}.",
            f"Solve for y: y = {_text(key)}.",
        ],
        distractors=pick_text(fields, cands),
        **fields,
    )


def _medium(rng):
    roll = rng.random()
    return _medium_ellipse(rng) if roll < 0.4 else _medium_hyperbola(rng) if roll < 0.75 else _medium_polar(rng)


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard_shifted(rng):
    a, b = rng.sample([2, 3, 4, 5], 2)
    h, k = rng.choice([x for x in range(-4, 5) if x != 0]), rng.choice([x for x in range(-4, 5) if x != 0])
    u = X - h
    key = sympy.expand(b**2 - sympy.Rational(b**2, a**2) * u**2)
    _verify(lambda t: h + a * math.cos(t), lambda t: k + b * math.sin(t), lambda y: (y - k) ** 2, key, "shifted")
    fields = expr_fields(key)
    cands = _cands(
        key,
        [
            (b**2 - sympy.Rational(b**2, a**2) * (X + h) ** 2, "r3i-parametric-shift-sign-wrong"),
            (b**2 - sympy.Rational(b**2, a**2) * X**2, "r3i-parametric-shift-dropped"),
            (b**2 - sympy.Rational(b**2, a) * u**2, "r3i-parametric-denominator-not-squared"),
            (b - sympy.Rational(b, a**2) * u**2, "r3i-parametric-coefficient-not-squared"),
            (b**2 + sympy.Rational(b**2, a**2) * u**2, "r3i-parametric-conic-sign-confused"),
        ],
    )
    return dict(
        text=(
            f"A curve has parametric equations x = {h} + {a}cos(t), y = {k} + {b}sin(t). "
            f"{_ask(f'(y {"-" if k > 0 else "+"} {abs(k)})²')}"
        ),
        working=[
            f"cos(t) = (x {'-' if h > 0 else '+'} {abs(h)})/{a} and sin(t) = (y {'-' if k > 0 else '+'} {abs(k)})/{b}.",
            f"Using cos²(t) + sin²(t) = 1, (y {'-' if k > 0 else '+'} {abs(k)})² = {_text(key)}.",
        ],
        distractors=pick_text(fields, cands),
        **fields,
    )


def _hard_reciprocal(rng):
    p, q = rng.sample([1, 2, 3, 4], 2) if rng.random() < 0.85 else (rng.choice([1, 2, 3]),) * 2
    key = sympy.expand(sympy.Rational(q**2, p**2) * X**2 - 4 * q**2)
    _verify(lambda t: p * (t + 1 / t), lambda t: q * (t - 1 / t), lambda y: y * y, key, "reciprocal")
    fields = expr_fields(key)
    cands = _cands(
        key,
        [
            (sympy.Rational(q**2, p**2) * X**2 + 4 * q**2, "r3i-parametric-sign-lost-squaring-difference"),
            (sympy.Rational(q**2, p**2) * X**2 - 4, "r3i-parametric-scale-dropped"),
            (sympy.Rational(q**2, p**2) * X**2 - 2 * q**2, "r3i-parametric-cross-term-halved"),
            (sympy.Rational(q**2, p) * X**2 - 4 * q**2, "r3i-parametric-denominator-not-squared"),
            (sympy.Rational(q, p**2) * X**2 - 4 * q, "r3i-parametric-coefficient-not-squared"),
        ],
    )
    xs = f"{_coef(p)}(t + 1/t)" if p != 1 else "t + 1/t"
    ys = f"{_coef(q)}(t - 1/t)" if q != 1 else "t - 1/t"
    return dict(
        text=f"A curve has parametric equations x = {xs}, y = {ys}. {_ask('y²')}",
        working=[
            f"x/{p} = t + 1/t and y/{q} = t - 1/t. Squaring and subtracting, (x/{p})² - (y/{q})² = 4.",
            f"So y² = {_text(key)}.",
        ],
        distractors=pick_text(fields, cands),
        **fields,
    )


def _hard_double_angle(rng):
    a = rng.choice([2, 3, 4, 5])
    c = rng.choice([x for x in range(-4, 5) if x != 0])
    d = rng.choice([2, 3, 4, -2, -3])
    key = sympy.expand(c + d - sympy.Rational(2 * d, a**2) * X**2)
    _verify(lambda t: a * math.sin(t), lambda t: c + d * math.cos(2 * t), lambda y: y, key, "double angle")
    fields = expr_fields(key)
    cands = _cands(
        key,
        [
            (c + d - sympy.Rational(d, a**2) * X**2, "r3i-parametric-double-angle-factor-lost"),
            (c + d + sympy.Rational(2 * d, a**2) * X**2, "r3i-parametric-double-angle-sign"),
            (c - sympy.Rational(2 * d, a**2) * X**2, "r3i-parametric-constant-dropped"),
            (c + d - sympy.Rational(2 * d, a) * X**2, "r3i-parametric-denominator-not-squared"),
            (c + d * (1 - 2 * X / a), "r3i-parametric-square-forgotten"),
        ],
    )
    return dict(
        text=f"A curve has parametric equations x = {_coef(a)}sin(t), y = {c} {'+' if d > 0 else '-'} {abs(d) if abs(d) != 1 else ''}cos(2t). {_ask('y')}",
        working=[
            f"cos(2t) = 1 - 2sin²(t) and sin(t) = x/{a}.",
            f"So y = {c} {'+' if d > 0 else '-'} {abs(d)}(1 - 2x²/{a**2}) = {_text(key)}.",
        ],
        distractors=pick_text(fields, cands),
        **fields,
    )


def _hard_rational(rng):
    a, b = rng.sample([1, 2, 3, 4], 2) if rng.random() < 0.85 else (rng.choice([1, 2, 3]),) * 2
    key = sympy.expand(b**2 - sympy.Rational(b**2, a**2) * X**2)
    _verify(
        lambda t: a * (1 - t * t) / (1 + t * t),
        lambda t: 2 * b * t / (1 + t * t),
        lambda y: y * y,
        key,
        "rational",
    )
    fields = expr_fields(key)
    cands = _cands(
        key,
        [
            (b**2 - sympy.Rational(b**2, a) * X**2, "r3i-parametric-denominator-not-squared"),
            (b**2 + sympy.Rational(b**2, a**2) * X**2, "r3i-parametric-conic-sign-confused"),
            (4 * b**2 - sympy.Rational(4 * b**2, a**2) * X**2, "r3i-parametric-factor-two-squared-wrongly"),
            (b - sympy.Rational(b, a**2) * X**2, "r3i-parametric-coefficient-not-squared"),
            (1 - sympy.Rational(1, a**2) * X**2, "r3i-parametric-scale-dropped"),
        ],
    )
    xs = f"{_coef(a)}(1 - t²)/(1 + t²)" if a != 1 else "(1 - t²)/(1 + t²)"
    ys = f"{2 * b}t/(1 + t²)"
    return dict(
        text=f"A curve has parametric equations x = {xs}, y = {ys}. {_ask('y²')}",
        working=[
            f"Let u = x/{a} = (1 - t²)/(1 + t²) and v = y/{b} = 2t/(1 + t²). Then u² + v² = 1, because (1 - t²)² + 4t² = (1 + t²)².",
            f"So y² = {_text(key)}.",
        ],
        distractors=pick_text(fields, cands),
        **fields,
    )


def _hard(rng):
    shapes = [_hard_shifted, _hard_reciprocal, _hard_double_angle, _hard_rational]
    return shapes[rng.randrange(4)](rng)


@generator(
    "r3i_parametric_conics",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def r3i_parametric_conics(rng, tier="easy"):
    """Eliminating the parameter from parametric (and one polar) equations; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("r3i_exam_parametric_conics", level=14, presets={"exam": dict(tier="exam")})
@retry_short
def r3i_exam_parametric_conics(rng, tier="exam"):
    """Exam chain on the ellipse x = a cos t, y = b sin t."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "an ellipse whose y² at the stated x is a whole number"):
        a, b, sh = rng.choice([4, 6, 8, 10, 12]), rng.choice(range(2, 25, 2)), rng.choice([1, 2, 3])
        at_x = Fraction(b * b) * (1 - Fraction(a // 2 + sh, a) ** 2)
        if at_x.denominator == 1 and at_x > 0 and a != b:
            break
    x0, y2 = a // 2, Fraction(3 * b * b, 4)
    key = sympy.expand(b**2 - sympy.Rational(b**2, a**2) * X**2)
    _verify(lambda t: a * math.cos(t), lambda t: b * math.sin(t), lambda y: y * y, key, "exam ellipse")
    check(
        abs(a * math.cos(math.pi / 3) - x0) < 1e-9 and abs((b * math.sin(math.pi / 3)) ** 2 - float(y2)) < 1e-9,
        "t = π/3 values",
    )
    check(abs(float(key.subs(X, a // 2 + sh)) - float(at_x)) < 1e-9, "y² at the new x")
    f_a, f_b, f_d = num_fields(x0), num_fields(y2), num_fields(at_x)
    f_c = expr_fields(key)
    cands_c = _cands(
        key,
        [
            (b**2 - sympy.Rational(b**2, a) * X**2, "r3i-parametric-denominator-not-squared"),
            (b**2 + sympy.Rational(b**2, a**2) * X**2, "r3i-parametric-conic-sign-confused"),
            (b - sympy.Rational(b, a**2) * X**2, "r3i-parametric-coefficient-not-squared"),
            (1 - sympy.Rational(1, a**2) * X**2, "r3i-parametric-scale-dropped"),
        ],
    )
    shifted = Fraction(b * b) * (1 - Fraction(a // 2, a) ** 2)
    return dict(
        text=phrase(
            rng,
            [
                "A curve has parametric equations x = {a}cos(t), y = {b}sin(t), for 0 ≤ t < 2π.",
                "The parametric equations x = {a}cos(t), y = {b}sin(t), for 0 ≤ t < 2π, describe a curve.",
                "A point moves so that its position at parameter t is given by x = {a}cos(t), y = {b}sin(t), "
                "for 0 ≤ t < 2π.",
            ],
            a=a,
            b=b,
        ),
        parts=[
            dict(
                prompt="Find the value of x when t = π/3.",
                marks=1,
                sensible_range=(-20, 20),
                working=[f"x = {a}cos(π/3) = {a} × 1/2 = {x0}."],
                distractors=pick_text(
                    f_a,
                    num_cands(
                        [
                            (a, "r3i-parametric-trig-value-misread"),
                            (2 * a, "r3i-parametric-sec-for-cos"),
                            (a // 4 if a % 4 == 0 else None, "r3i-parametric-trig-value-misread"),
                        ],
                        x0,
                    ),
                ),
                **f_a,
            ),
            dict(
                prompt="Find the value of y² when t = π/3.",
                marks=1,
                sensible_range=(0, 1000),
                working=[f"y² = {b}²sin²(π/3) = {b * b} × 3/4 = {y2}."],
                distractors=pick_text(
                    f_b,
                    num_cands(
                        [
                            (Fraction(b * b, 4), "r3i-parametric-cos-squared-used-for-sin"),
                            (b * b, "r3i-parametric-trig-value-misread"),
                            (Fraction(3 * b * b, 2), "r3i-parametric-trig-value-misread"),
                            (Fraction(3 * b, 4), "r3i-parametric-coefficient-not-squared"),
                        ],
                        y2,
                    ),
                ),
                **f_b,
            ),
            dict(
                prompt="Find y² in terms of x.",
                marks=3,
                working=[f"cos(t) = x/{a} and sin(t) = y/{b}, so (x/{a})² + (y/{b})² = 1 and y² = {_text(key)}."],
                distractors=pick_text(f_c, cands_c),
                **f_c,
            ),
            dict(
                prompt=f"Use your answer to part (c) to find y² at the point of the curve where x is {sh} more than your answer to part (a).",
                marks=2,
                follow_through=f"{b * b}*(1-(part_a+{sh})**2/{a * a})",
                sensible_range=(0, 1000),
                working=[f"x = {x0 + sh}: y² = {b * b}(1 - {x0 + sh}²/{a * a}) = {at_x}."],
                distractors=pick_text(
                    f_d,
                    num_cands(
                        [
                            (shifted, "r3i-parametric-instruction-not-followed"),
                            (
                                Fraction(b * b) * (1 + Fraction(a // 2 + sh, a) ** 2),
                                "r3i-parametric-conic-sign-confused",
                            ),
                            (
                                Fraction(b * b) * (1 - Fraction(a // 2 + sh, a)),
                                "r3i-parametric-denominator-not-squared",
                            ),
                            (y2 + 1, "r3i-parametric-instruction-not-followed"),
                        ],
                        at_x,
                    ),
                ),
                **f_d,
            ),
        ],
    )
