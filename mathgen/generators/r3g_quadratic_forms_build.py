"""Generates `r3g_quadratic_forms` questions (easy, medium, hard) and the exam chain `r3g_exam_quadratic_forms`.

Skill ``completing-square-and-discriminant`` (``quadratics``, answer ``inequality``): completing the square, the turning point, exact roots, the discriminant for a parameter and quadratic
and rational inequalities written as intervals.

* easy: the range of a monic quadratic with an even middle coefficient (or its negative), by completing the square,
  written as an inequality in y.
* medium: a quadratic inequality whose quadratic does not factorise, solved by completing the square, with surd bounds.
* hard: the values of a parameter k for which a quadratic has no or two real roots (an inequality in k, including a
  parameter in the leading coefficient); or a rational inequality, solved as intervals.
* exam (``r3g_exam_quadratic_forms``): for f(x) = a(x - p)(x - q) written out: (a) the x-coordinate of the turning
  point, (b) the turning value (follows through from (a)), (c) the discriminant (follows through from (b)) and (d) the
  solution set of f(x) ≥ 0 or f(x) < 0.

Turning points and roots are confirmed with SymPy's own derivative and root finding; sets of k and inequalities are
confirmed by counting real roots numerically at sample values of k and by testing sample points of x.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy as sp

from ..answers import Distractor
from ..core import attempts, check, generator
from ..grading.inequality import set_text
from ._r3g_kit import (
    ex,
    in_real_set,
    int_dist,
    latex,
    number_fields,
    pick,
    retry_short,
    value_set_dist,
    value_set_fields,
    TooFewDistractors,
    typed,
)

R = sp.Rational
F = Fraction
X, K = sp.symbols("x k")


def _poly_text(a, b, c, var="x") -> str:
    def term(coef, body, first):
        if coef == 0:
            return ""
        mag = abs(coef)
        shown = body if (mag == 1 and body) else f"{typed(mag)}{body}"
        if first:
            return f"-{shown}" if coef < 0 else shown
        return f" {'-' if coef < 0 else '+'} {shown}"

    t1 = term(a, f"{var}²", True)
    t2 = term(b, var, not t1)
    t3 = term(c, "", not (t1 or t2))
    return t1 + t2 + t3


def _vertex(a, b, c):
    xv = R(-b, 2 * a) if not isinstance(b, sp.Basic) else -b / (2 * a)
    return xv, a * xv**2 + b * xv + c


def _check_vertex(a, b, c):
    xv, yv = _vertex(a, b, c)
    f = a * X**2 + b * X + c
    check(sp.solve(sp.diff(f, X), X) == [xv], "turning point differs from the derivative")
    check(f.subs(X, xv) == yv, "turning value differs from direct substitution")
    return xv, yv


def _ray(value, up: bool, closed: bool = True):
    """[value, oo) or (-oo, value] (open at the end when `closed` is false)."""
    if up:
        return sp.Interval(value, sp.oo, not closed, True)
    return sp.Interval(-sp.oo, value, True, not closed)


def _easy(rng):
    """The range of a monic or negated-monic quadratic, found by completing the square."""
    a = rng.choice((1, 1, 1, -1))
    h = rng.randint(1, 6) * rng.choice((1, -1))
    b = -2 * a * h
    c = rng.randint(-9, 12)
    xv, yv = _check_vertex(a, b, c)
    up = a > 0
    solution = _ray(int(yv), up)
    # second route: the range is bounded by the value at the turning point; spot-check values around it
    f = lambda x: a * x**2 + b * x + c  # noqa: E731
    check(all(in_real_set(solution, f(t)) for t in range(-12, 13)), "a value of the quadratic fell outside its range")
    poly = _poly_text(a, b, c)
    kind = "minimum" if up else "maximum"
    cands = [
        (_ray(int(yv), not up), "r3g-quadratic-region-reversed"),
        (_ray(c, up), "r3g-quadratic-constant-term-given"),
        (_ray(c + a * h**2, up), "r3g-quadratic-square-not-subtracted"),
        (_ray(int(yv), up, False), "r3g-quadratic-equality-included"),
        (_ray(-int(yv), up), "r3g-quadratic-min-max-sign"),
    ]
    fields = _ineq_fields(solution, "y")
    return dict(
        text=f"Complete the square to find the range of y = {poly}. Write your answer as an inequality in y.",
        working=[
            f"{poly} = {typed(a)}(x - {typed(h)})² + ({typed(int(yv))}).".replace("x - -", "x + "),
            f"The {kind} value is {typed(int(yv))} and the graph goes {'up' if up else 'down'} from there, so {fields['answer_text']}.",
        ],
        distractors=pick(fields["answer_text"], [_ineq_dist(w, "y", t) for w, t in cands if w != solution], 3),
        **fields,
    )


def _medium(rng):
    """An inequality whose quadratic does not factorise: the bounds are surds found by completing the square."""
    for _ in attempts(300, "a monic quadratic with a surd root pair"):
        h = rng.randint(1, 7) * rng.choice((1, -1))
        d = rng.randint(2, 20)
        if sp.sqrt(d).is_Integer or d % 4 == 0:
            continue
        break
    b, c = -2 * h, h * h - d
    lo, hi = h - sp.sqrt(d), h + sp.sqrt(d)
    op = rng.choice(("<", ">", "≤", "≥"))
    ops = {"<": sp.Lt, ">": sp.Gt, "≤": sp.Le, "≥": sp.Ge}
    solution = _quadratic_set(1, lo, hi, op)
    pyop = {"<": lambda v: v < 0, ">": lambda v: v > 0, "≤": lambda v: v <= 0, "≥": lambda v: v >= 0}[op]
    pts = [
        R(round(float(lo)) - 1),
        R(round(float(lo))),
        R(round(float(h))),
        R(round(float(hi))),
        R(round(float(hi)) + 1),
        R(round(float(lo) * 4), 4),
        R(round(float(hi) * 4), 4),
    ]
    _check_by_sampling(lambda x: pyop(x**2 + b * x + c), solution, pts)
    closed = op in ("≤", "≥")
    inside = op in ("<", "≤")
    between = sp.Interval(lo, hi, not closed, not closed)
    outside = sp.Union(sp.Interval(-sp.oo, lo, True, not closed), sp.Interval(hi, sp.oo, not closed, True))
    cands = [
        (outside if inside else between, "r3g-quadratic-region-reversed"),
        (
            sp.Interval(-h - sp.sqrt(d), -h + sp.sqrt(d), not closed, not closed)
            if inside
            else sp.Union(
                sp.Interval(-sp.oo, -h - sp.sqrt(d), True, not closed),
                sp.Interval(-h + sp.sqrt(d), sp.oo, not closed, True),
            ),
            "r3g-quadratic-sign-of-shift",
        ),
        (
            sp.Interval(h - d, h + d, not closed, not closed)
            if inside
            else sp.Union(sp.Interval(-sp.oo, h - d, True, not closed), sp.Interval(h + d, sp.oo, not closed, True)),
            "r3g-quadratic-root-not-taken",
        ),
        (
            sp.Interval(lo, hi, closed, closed)
            if inside
            else sp.Union(sp.Interval(-sp.oo, lo, True, closed), sp.Interval(hi, sp.oo, closed, True)),
            "r3g-quadratic-equality-included",
        ),
        (
            sp.Interval(-sp.sqrt(d), sp.sqrt(d), not closed, not closed)
            if inside
            else sp.Union(
                sp.Interval(-sp.oo, -sp.sqrt(d), True, not closed), sp.Interval(sp.sqrt(d), sp.oo, not closed, True)
            ),
            "r3g-quadratic-shift-dropped",
        ),
    ]
    fields = _ineq_fields(solution, "x")
    return dict(
        text=(
            f"Solve {_poly_text(1, b, c)} {op} 0 by completing the square. "
            "Write your answer as an inequality, with the bounds exact (typed like 2 + sqrt(7))."
        ),
        working=[
            f"(x - {typed(h)})² - {d} {op} 0 gives (x - {typed(h)})² {op} {d}.".replace("- -", "+ "),
            f"So x = {typed(h)} ± √{d} are the bounds and the solution is {fields['answer_text']}.",
        ],
        distractors=pick(fields["answer_text"], [_ineq_dist(w, "x", t) for w, t in cands if w != solution], 3),
        **fields,
    )


# -- hard --------------------------------------------------------------------------------------------------------


def _real_root_count(a, b, c) -> int:
    """Distinct real roots of a x² + b x + c, by Sturm's theorem (a different route from the discriminant)."""
    if a == 0:
        return 1 if b != 0 else 0
    return sp.Poly(sp.nsimplify(a) * X**2 + sp.nsimplify(b) * X + sp.nsimplify(c), X).count_roots()


def _ineq_fields(solution, var: str):
    text = set_text(solution, var)
    return dict(answer_kind="inequality", answer=text, answer_text=text, answer_latex=_ineq_latex(solution, var))


def _ineq_latex(solution, var: str) -> str:
    text = set_text(solution, var)
    return (
        text.replace("≤", r"\le ")
        .replace("≥", r"\ge ")
        .replace("<", "<")
        .replace("√", r"\sqrt")
        .replace(" or ", r" \text{ or } ")
        .replace("∞", r"\infty")
    )


def _ineq_dist(solution, var: str, tag: str) -> Distractor:
    text = set_text(solution, var)
    return Distractor(text, tag, text)


def _discriminant_inequality(rng):
    mode = rng.choice(("none", "two", "none", "two", "lead", "const"))
    if mode in ("none", "two"):
        m = rng.randint(2, 7)
        a = rng.choice((1, 1, 2, 4))
        c = m * m // a if (m * m) % a == 0 else None
        if c is None:
            raise TooFewDistractors("a does not divide m squared")
        bound = 2 * m
        poly = f"{'' if a == 1 else a}x² + kx + {c}"
        if mode == "none":
            solution = sp.Interval.open(-bound, bound)
            word = "no real roots"
        else:
            solution = sp.Union(sp.Interval.open(-sp.oo, -bound), sp.Interval.open(bound, sp.oo))
            word = "two distinct real roots"
        samples = [-bound - 1, -bound, 0, bound, bound + 1]
        for k in samples:
            want = {"none": 0, "two": 2}[mode]
            count = _real_root_count(a, k, c)
            inside = in_real_set(solution, k)
            check((count == want) == inside, f"k = {k}: {count} roots but membership is {inside}")
        other = (
            sp.Union(sp.Interval.open(-sp.oo, -bound), sp.Interval.open(bound, sp.oo))
            if mode == "none"
            else sp.Interval.open(-bound, bound)
        )
        cands = [
            (other, "r3g-quadratic-region-reversed"),
            (
                sp.Interval.open(-sp.oo, bound) if mode == "none" else sp.Interval.open(bound, sp.oo),
                "r3g-quadratic-one-sign-only",
            ),
            (
                sp.Interval(-bound, bound)
                if mode == "none"
                else sp.Union(sp.Interval(-sp.oo, -bound), sp.Interval(bound, sp.oo)),
                "r3g-quadratic-equality-included",
            ),
            (
                sp.Interval.open(-m, m)
                if mode == "none"
                else sp.Union(sp.Interval.open(-sp.oo, -m), sp.Interval.open(m, sp.oo)),
                "r3g-quadratic-factor-two-dropped",
            ),
        ]
        var = "k"
        stem = f"Find the values of k for which the equation {poly} = 0 has {word}."
        working = [
            f"The discriminant is k² - 4({a})({c}) = k² - {4 * a * c}.",
            f"{word.capitalize()} needs it to be {'negative' if mode == 'none' else 'positive'}: {set_text(solution, 'k')}.",
        ]
    elif mode == "const":
        b = rng.choice((2, 4, 6, 8))
        bound = b * b // 4
        want_two = rng.random() < 0.5
        solution = sp.Interval.open(-sp.oo, bound) if want_two else sp.Interval.open(bound, sp.oo)
        for k in (bound - 2, bound, bound + 2):
            count = _real_root_count(1, b, k)
            check(
                (count == (2 if want_two else 0)) == (in_real_set(solution, k)), "sample k disagrees with the interval"
            )
        other = sp.Interval.open(bound, sp.oo) if want_two else sp.Interval.open(-sp.oo, bound)
        cands = [
            (other, "r3g-quadratic-region-reversed"),
            (sp.Interval(-sp.oo, bound) if want_two else sp.Interval(bound, sp.oo), "r3g-quadratic-equality-included"),
            (
                sp.Interval.open(-sp.oo, b * b) if want_two else sp.Interval.open(b * b, sp.oo),
                "r3g-quadratic-four-a-dropped",
            ),
            (
                sp.Interval.open(-sp.oo, b) if want_two else sp.Interval.open(b, sp.oo),
                "r3g-quadratic-square-root-not-taken",
            ),
        ]
        var = "k"
        stem = f"Find the values of k for which x² + {b}x + k = 0 has {'two distinct real roots' if want_two else 'no real roots'}."
        working = [
            f"The discriminant is {b}² - 4k = {b * b} - 4k.",
            f"{'Positive' if want_two else 'Negative'} discriminant: {set_text(solution, 'k')}.",
        ]
    else:
        b = rng.choice((2, 4, 6))
        c = rng.choice((1, 2, 3))
        bound = R(b * b, 4 * c)
        # k x² + b x + c = 0: discriminant b² - 4kc
        want_none = rng.random() < 0.5
        solution = (
            sp.Interval.open(bound, sp.oo)
            if want_none
            else sp.Union(sp.Interval.open(-sp.oo, 0), sp.Interval.open(0, bound))
        )
        for k in (R(-1), R(1, 2), bound - R(1, 4), bound, bound + 1):
            count = _real_root_count(k, b, c)
            want = 0 if want_none else 2
            check((count == want) == (in_real_set(solution, k)), f"k = {k}: {count} roots disagree with the set")
        other = sp.Interval.open(-sp.oo, bound) if want_none else sp.Interval.open(bound, sp.oo)
        cands = [
            (other, "r3g-quadratic-region-reversed"),
            (
                sp.Interval.open(-sp.oo, bound) if not want_none else sp.Interval.open(-bound, sp.oo),
                "r3g-quadratic-region-reversed",
            ),
            (
                sp.Interval.open(0, bound) if not want_none else sp.Interval.open(b * b, sp.oo),
                "r3g-quadratic-leading-coefficient-zero-allowed",
            ),
            (
                sp.Interval(-sp.oo, bound) if not want_none else sp.Interval(bound, sp.oo),
                "r3g-quadratic-equality-included",
            ),
        ]
        var = "k"
        stem = f"Find the values of k for which kx² + {b}x + {c} = 0 has {'no real roots' if want_none else 'two distinct real roots'}."
        working = [
            f"The discriminant is {b}² - 4k({c}) = {b * b} - {4 * c}k.",
            f"{set_text(solution, 'k')} (for two roots, k ≠ 0 because the equation must stay quadratic)."
            if not want_none
            else f"No real roots needs it negative: {set_text(solution, 'k')}.",
        ]
    fields = _ineq_fields(solution, var)
    return dict(
        text=stem + " Write your answer as an inequality or in interval notation.",
        working=working,
        distractors=pick(fields["answer_text"], [_ineq_dist(s, var, t) for s, t in cands], 3),
        **fields,
    )


def _quadratic_set(a, r1, r2, op):
    """{x : a (x - r1)(x - r2) op 0} for r1 < r2, built from the sign pattern of the parabola."""
    inside, outside = sp.Interval.open(r1, r2), sp.Union(sp.Interval.open(-sp.oo, r1), sp.Interval.open(r2, sp.oo))
    negative = inside if a > 0 else outside
    positive = outside if a > 0 else inside
    strict = negative if op in ("<", "≤") else positive
    return sp.Union(strict, sp.FiniteSet(r1, r2)) if op in ("≤", "≥") else strict


def _rational_set(r, s, op):
    """{x : (x - r) / (x - s) op 0}: the zero r is allowed when the inequality is not strict, the pole s never."""
    lo, hi = min(r, s), max(r, s)
    inside, outside = sp.Interval.open(lo, hi), sp.Union(sp.Interval.open(-sp.oo, lo), sp.Interval.open(hi, sp.oo))
    strict = inside if op in ("<", "≤") else outside
    return sp.Union(strict, sp.FiniteSet(r)) if op in ("≤", "≥") else strict


def _check_by_sampling(predicate, solution, points):
    """Every sample point is in `solution` exactly when `predicate` (a direct evaluation of the stem) holds."""
    for x in points:
        try:
            holds = predicate(x)
        except ZeroDivisionError:
            holds = False
        check(
            holds == (in_real_set(solution, x)),
            f"x = {x}: stem gives {holds} but the set says {in_real_set(solution, x)}",
        )


def _rational_inequality(rng):
    for _ in attempts(300, "a rational inequality with two critical values"):
        r, s = rng.randint(-5, 5), rng.randint(-5, 5)
        if r == s:
            continue
        p = rng.choice((1, 1, 2, 3))
        q = rng.choice((1, 1, 2))
        break
    op = rng.choice(("≥", "≤", ">", "<"))
    ops = {"<": sp.Lt, ">": sp.Gt, "≤": sp.Le, "≥": sp.Ge}
    num, den = p * X - p * r, q * X - q * s  # zero at r, pole at s
    rel = ops[op](num / den, 0)
    solution = _rational_set(r, s, op)
    pts = [min(r, s) - 1, r, s, max(r, s) + 1, R(r + s, 2), s - R(1, 2), s + R(1, 2), r - R(1, 2), r + R(1, 2)]
    pyop = {"<": lambda v: v < 0, ">": lambda v: v > 0, "≤": lambda v: v <= 0, "≥": lambda v: v >= 0}[op]
    _check_by_sampling(lambda x: pyop((p * x - p * r) / (q * x - q * s)), solution, pts)
    check(not in_real_set(solution, s), "the pole must be excluded")
    wrong_sets = []
    # cross-multiplying by the denominator without considering its sign: (x - r)(x - s) with the same relation
    quad = _quadratic_set(1, min(r, s), max(r, s), op)
    wrong_sets.append((quad, "r3g-quadratic-rational-cross-multiplied"))
    comp = sp.Complement(sp.S.Reals, solution)
    wrong_sets.append((sp.Union(comp, sp.FiniteSet(s)) if False else comp, "r3g-quadratic-region-reversed"))
    closed_pole = sp.Union(solution, sp.FiniteSet(s))
    wrong_sets.append((closed_pole, "r3g-quadratic-pole-included"))
    wrong_sets.append(
        (
            sp.Interval(min(r, s), max(r, s)) if op in ("≥", "≤") else sp.Interval.open(min(r, s), max(r, s)),
            "r3g-quadratic-region-reversed",
        )
    )
    fields = _ineq_fields(solution, "x")
    numtext = (
        f"{'' if p == 1 else p}(x - {r})".replace("- -", "+ ").replace("(x - 0)", "x")
        if p != 1
        else f"x - {r}".replace("- -", "+ ")
    )
    dentext = (
        f"{'' if q == 1 else q}(x - {s})".replace("- -", "+ ").replace("(x - 0)", "x")
        if q != 1
        else f"x - {s}".replace("- -", "+ ")
    )
    numtext = numtext.replace("x - 0", "x")
    dentext = dentext.replace("x - 0", "x")
    return dict(
        text=f"Solve ({numtext}) / ({dentext}) {op} 0. Write your answer as an inequality or in interval notation.",
        working=[
            f"The expression changes sign at x = {r} (a zero) and x = {s} (where the denominator is zero, so x ≠ {s}).",
            f"Test a point in each interval: the solution is {fields['answer_text']}.",
        ],
        distractors=pick(fields["answer_text"], [_ineq_dist(w, "x", t) for w, t in wrong_sets if w != solution], 3),
        **fields,
    )


def _hard(rng):
    return rng.choice([_discriminant_inequality, _discriminant_inequality, _rational_inequality, _rational_inequality])(
        rng
    )


@generator(
    "r3g_quadratic_forms",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def r3g_quadratic_forms(rng, tier="easy"):
    """Completing the square, the discriminant and inequalities; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("r3g_exam_quadratic_forms", level=12, presets={"exam": dict(tier="exam")})
@retry_short
def r3g_exam_quadratic_forms(rng, tier="exam"):
    """Exam chain: turning point x, turning value, discriminant and the solution set of f(x) ≥ 0 or f(x) < 0."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(300, "a quadratic with distinct integer zeros and a whole-number turning value"):
        p, q = rng.randint(-6, 4), rng.randint(-3, 7)
        a = rng.choice((1, 2, -1, -2, 3))
        if p >= q or (p + q) % 2:
            continue
        break
    b, c = -a * (p + q), a * p * q
    xv, yv = _check_vertex(a, b, c)
    disc = b * b - 4 * a * c
    check(disc == -4 * a * yv, "discriminant differs from -4a times the turning value")
    check(_real_root_count(a, b, c) == 2, "f should have two real zeros")
    op = rng.choice(("≥", "<"))
    ops = {"≥": sp.Ge, "<": sp.Lt}
    solution = _quadratic_set(a, p, q, op)
    pyop = {"≥": lambda v: v >= 0, "<": lambda v: v < 0}[op]
    _check_by_sampling(lambda x: pyop(a * x**2 + b * x + c), solution, [p - 1, p, xv, q, q + 1])
    poly = _poly_text(a, b, c)
    kind = "minimum" if a > 0 else "maximum"

    def expr_part(prompt, value, marks, working, cands, follow=None):
        text = typed(value)
        d = dict(
            prompt=prompt,
            answer_kind="expression",
            answer=text,
            answer_text=text,
            answer_latex=latex(value),
            marks=marks,
            working=working,
            distractors=pick(text, [ex(w, t) for w, t in cands if sp.sympify(w) != value], 3),
        )
        if follow:
            d["follow_through"] = follow
        return d

    fields = _ineq_fields(solution, "x")
    between = (a > 0 and op == "<") or (a < 0 and op == "≥")
    wrong = [
        (
            sp.Union(sp.Interval(-sp.oo, p, True, op != "≥"), sp.Interval(q, sp.oo, op != "≥", True))
            if between
            else sp.Interval(p, q, op != "≥", op != "≥"),
            "r3g-quadratic-region-reversed",
        ),
        (
            sp.Interval.open(p, q)
            if op == "≥" and between
            else sp.Interval(p, q)
            if op == "<" and between
            else sp.Interval.open(-sp.oo, p),
            "r3g-quadratic-equality-included",
        ),
        (sp.Interval(-q, -p), "r3g-quadratic-roots-negated"),
    ]
    return dict(
        text=f"Let f(x) = {poly}.",
        parts=[
            expr_part(
                "Find the x-coordinate of the turning point of f.",
                xv,
                1,
                [f"The turning point is halfway between the zeros {p} and {q}, or x = -b/2a = {typed(xv)}."],
                [
                    (-xv, "r3g-quadratic-sign-of-shift"),
                    (R(-b, 1), "r3g-quadratic-leading-coefficient-ignored"),
                    (R(b, 2 * a), "r3g-quadratic-sign-of-shift"),
                    (R(-b, a), "r3g-quadratic-half-not-taken"),
                ],
            ),
            expr_part(
                f"Hence find the {kind} value of f.",
                yv,
                2,
                [f"f({typed(xv)}) = {typed(yv)}."],
                [
                    (-yv, "r3g-quadratic-min-max-sign"),
                    (c, "r3g-quadratic-constant-term-given"),
                    (a * xv**2, "r3g-quadratic-constant-dropped"),
                    (a * xv * xv + b * xv, "r3g-quadratic-constant-dropped"),
                ],
                f"({a})*part_a^2 + ({b})*part_a + ({c})",
            ),
            expr_part(
                "Hence find the discriminant of f(x) = 0.",
                disc,
                1,
                [f"For a quadratic the turning value is -Δ/4a, so Δ = -4a × {typed(yv)} = {typed(disc)}."],
                [
                    (-disc, "r3g-quadratic-min-max-sign"),
                    (R(disc, 4), "r3g-quadratic-four-a-dropped"),
                    (-yv, "r3g-quadratic-four-a-dropped"),
                    (disc + 4 * a * c, "r3g-quadratic-wrong-formula"),
                ],
                f"-4*({a})*part_b",
            ),
            dict(
                prompt=f"Solve f(x) {op} 0. Write your answer as an inequality or in interval notation.",
                marks=2,
                working=[
                    f"f has zeros x = {p} and x = {q}, and the parabola opens {'up' if a > 0 else 'down'}.",
                    f"So f(x) {op} 0 for {fields['answer_text']}.",
                ],
                distractors=pick(fields["answer_text"], [_ineq_dist(w, "x", t) for w, t in wrong if w != solution], 3),
                **fields,
            ),
        ],
    )
