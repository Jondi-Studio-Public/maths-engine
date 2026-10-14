"""Generates `r3g_trig_equations` questions (easy, medium, hard) and the exam chain `r3g_exam_trig_equations`.

Skill ``trig-equations-radians-list`` (``trigonometry``, answer ``set``): every solution of a trigonometric equation
in an interval of radians, typed as multiples of pi (``5pi/6``). The degree-based skills for sine, cosine and tangent
equations stay; this one works on 0 ≤ x < 2π in exact radians.

* easy: ``sin x = a``, ``cos x = a`` or ``tan x = a`` for a positive standard value. Two solutions, one in each of the
  two quadrants where the function is positive.
* medium: a negative or zero value, the equation rearranged (``2 cos x + √3 = 0``), or a multiple angle
  (``sin 2x = a``), which doubles the number of solutions in the interval.
* hard: a quadratic in sine or cosine (``2 sin²x - sin x - 1 = 0``), one of whose roots may lie outside [-1, 1] and so
  gives nothing, ``tan²x = k``, or a shifted and scaled argument (``sin(2(x - π/6)) = a``).
* exam (``r3g_exam_trig_equations``): a quadratic in sin x or cos x; (a) the values of the function, (b) the smallest
  solution, (c) the largest, (d) their difference (follows through from (b) and (c)).

Solutions are found exactly, as fractions of pi, from the reference angle and the quadrant signs, then every solution is
substituted into the equation with SymPy and the count is confirmed by scanning the interval at multiples of pi/72.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy as sp

from ..answers import Distractor
from ..core import attempts, check, generator, phrase
from ._r3g_kit import (
    vary_symbol,
    pi_dist,
    pi_latex,
    pi_set_dist,
    pi_set_fields,
    pi_shown,
    pi_text,
    pick,
    retry_short,
    typed,
    value_set_dist,
    value_set_fields,
    ex,
)

_S2, _S3 = sp.sqrt(2), sp.sqrt(3)
F = Fraction
# standard values of |a| and the reference angle (in units of pi) at which each function takes them
_REF = {
    "sin": {sp.Rational(1, 2): F(1, 6), _S2 / 2: F(1, 4), _S3 / 2: F(1, 3)},
    "cos": {sp.Rational(1, 2): F(1, 3), _S2 / 2: F(1, 4), _S3 / 2: F(1, 6)},
    "tan": {_S3 / 3: F(1, 6), sp.Integer(1): F(1, 4), _S3: F(1, 3)},
}
_FN = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan}
_FLOAT = {"sin": math.sin, "cos": math.cos, "tan": math.tan}
# how a value is written in a stem (surds with the root sign) and as an equation  q f(x) = n  or  q f(x) + n = 0
_SHOWN = {
    sp.Rational(1, 2): ("1", "2"),
    _S2 / 2: ("√2", "2"),
    _S3 / 2: ("√3", "2"),
    _S3 / 3: ("√3", "3"),
    sp.Integer(1): ("1", "1"),
    _S3: ("√3", "1"),
}
_FNAME = {"sin": "sin", "cos": "cos", "tan": "tan"}


def solutions_in_unit_interval(fn: str, a) -> list[Fraction]:
    """The x in [0, 2) (units of pi) with f(x pi) = a, from the reference angle and the signs of the quadrants."""
    a = sp.nsimplify(a)
    if a == 0:
        return {"sin": [F(0), F(1)], "cos": [F(1, 2), F(3, 2)], "tan": [F(0), F(1)]}[fn]
    if abs(a) == 1 and fn != "tan":
        pos = {"sin": F(1, 2), "cos": F(0)}[fn]
        neg = {"sin": F(3, 2), "cos": F(1)}[fn]
        return [pos if a > 0 else neg]
    r = _REF[fn][abs(a)]
    if fn == "sin":
        out = [r, 1 - r] if a > 0 else [1 + r, 2 - r]
    elif fn == "cos":
        out = [r, 2 - r] if a > 0 else [1 - r, 1 + r]
    else:
        out = [r, 1 + r] if a > 0 else [1 - r, 2 - r]
    return sorted(out)


def solve(fn: str, a, m: int = 1, b: Fraction = F(0)) -> list[Fraction]:
    """Every x in [0, 2) (units of pi) with f(m (x pi - b pi)) = a, for a whole number m >= 1."""
    period = 1 if fn == "tan" else 2
    base = solutions_in_unit_interval(fn, a)
    found = set()
    for s in base:
        for k in range(-2 * m - 2, 2 * m + 3):
            x = (s + period * k) / m + b
            if 0 <= x < 2:
                found.add(x)
    return sorted(found)


def verify(fn: str, a, m: int, b: Fraction, sols: list[Fraction]) -> None:
    """Second route: substitute each solution into the equation with SymPy, and count the roots at multiples of pi/72
    (every solution lies on that grid) with floats."""
    for x in sols:
        arg = sp.pi * sp.Rational((m * (x - b)).numerator, (m * (x - b)).denominator)
        check(sp.simplify(_FN[fn](arg) - a) == 0, f"{fn}({arg}) is not {a}")
    target = float(a)
    hits = []
    for k in range(144):
        x = F(k, 72)
        arg = math.pi * float(m * (x - b))
        if fn == "tan" and abs(math.cos(arg)) < 1e-9:
            continue
        if abs(_FLOAT[fn](arg) - target) < 1e-9:
            hits.append(x)
    check(hits == sols, f"scan found {hits}, construction {sols}")


def _equation(fn: str, a, rng, rearranged: bool) -> str:
    """`f x = a` as a stem writes it: plain, or rearranged as  q f x = n  or  q f x + n = 0."""
    mag = abs(a)
    num, den = _SHOWN[mag]
    sign = "-" if a < 0 else ""
    if not rearranged:
        value = f"{sign}{num}/{den}" if den != "1" else f"{sign}{num}"
        return f"{fn} x = {value}"
    left = fn if den == "1" else f"{den} {fn}"
    if rng.random() < 0.5:
        return f"{left} x = {sign}{num}"
    return f"{left} x {'+' if a < 0 else '-'} {num} = 0"


_FORMAT = "Give the exact solutions in radians as multiples of pi, typed like 5pi/6 and separated by commas."


def _set_working(fn, a, sols, extra=None) -> list[str]:
    steps = []
    if extra:
        steps.append(extra)
    steps.append(
        f"The solutions in the interval are {', '.join(pi_shown(s) for s in sols)}."
        if sols
        else "There are no solutions in the interval."
    )
    return steps


def _slip_sets(fn, a, m=1, b=F(0)) -> list:
    """Wrong sets from the errors a learner makes, each as `(set of coefficients, tag)`."""
    out = []
    sols = solve(fn, a, m, b)
    base = solutions_in_unit_interval(fn, a)
    if len(base) == 2 and m == 1 and b == 0:
        first, second = base
        r = _REF[fn][abs(a)] if a != 0 else None
        if r is not None:
            for w in (r, 1 - r, 1 + r, 2 - r):
                if w not in (first, second) and 0 <= w < 2:
                    out.append(({first, w}, "r3g-trig-second-solution-wrong-quadrant"))
            out.append(({first}, "r3g-trig-principal-value-only"))
            out.append((set(solutions_in_unit_interval(fn, -a)), "r3g-trig-sign-of-value-ignored"))
            if a < 0:
                out.append(({-r, first}, "r3g-trig-calculator-value-outside-interval"))
    if m > 1:
        nodiv = {x for s0 in base for x in (s0 + b, s0 + b + (1 if fn == "tan" else 2)) if 0 <= x < 2}
        out.append((nodiv, "r3g-trig-multiple-angle-not-divided"))
        out.append((set(sols[: len(base)]), "r3g-trig-multiple-angle-first-period-only"))
        out.append((set(sols[:-1]), "r3g-trig-multiple-angle-first-period-only"))
    if b != 0:
        out.append((set(solve(fn, a, m, -b)), "r3g-trig-shift-wrong-direction"))
        out.append((set(solve(fn, a, m, F(0))), "r3g-trig-shift-ignored-in-answer"))
    # keep only genuinely different, in-order candidates
    keep, seen = [], {tuple(sorted(sols))}
    for items, tag in out:
        key = tuple(sorted(items))
        if key in seen or not items:
            continue
        seen.add(key)
        keep.append(pi_set_dist(items, tag))
    return keep


_VALUES = {
    "sin": [sp.Rational(1, 2), _S2 / 2, _S3 / 2],
    "cos": [sp.Rational(1, 2), _S2 / 2, _S3 / 2],
    "tan": [_S3 / 3, sp.Integer(1), _S3],
}


def _result(fn, a, m, b, stem, working_note=None):
    sols = solve(fn, a, m, b)
    check(len(sols) >= 1, "no solutions")
    verify(fn, a, m, b, sols)
    cands = _slip_sets(fn, a, m, b)
    fields = pi_set_fields(sols)
    return dict(
        text=stem,
        working=_set_working(fn, a, sols, working_note),
        distractors=pick(fields["answer_text"], cands, 3),
        **fields,
    )


def _easy(rng):
    fn = rng.choice(("sin", "cos", "tan"))
    a = rng.choice(_VALUES[fn])
    stem = phrase(
        rng,
        ("Solve {eq} for 0 ≤ x < 2π. {fmt}", "Find every x with 0 ≤ x < 2π that satisfies {eq}. {fmt}"),
        eq=_equation(fn, a, rng, False),
        fmt=_FORMAT,
    )
    ref = _REF[fn][a]
    note = f"The reference angle is {pi_shown(ref)}; {fn} is positive in two quadrants."
    return _result(fn, a, 1, F(0), stem, note)


def _medium(rng):
    fn = rng.choice(("sin", "cos", "tan"))
    mode = rng.choice(("negative", "rearranged", "multiple", "zero"))
    if mode == "zero":
        fn = rng.choice(("sin", "cos"))
        a = sp.Integer(0)
        stem = f"Solve {fn} x = 0 for 0 ≤ x < 2π. {_FORMAT}"
        return _result(fn, a, 1, F(0), stem, f"{fn} is zero at the points where the unit circle point is on an axis.")
    a = rng.choice(_VALUES[fn])
    if mode in ("negative", "rearranged"):
        a = -a
        eq = _equation(fn, a, rng, mode == "rearranged")
        stem = f"Solve {eq} for 0 ≤ x < 2π. {_FORMAT}"
        note = f"The reference angle is {pi_shown(_REF[fn][abs(a)])}; the sign of the value puts the solutions in the other two quadrants."
        return _result(fn, a, 1, F(0), stem, note)
    m = rng.choice((2, 2, 3))
    a = a if rng.random() < 0.5 else -a
    value = _equation(fn, a, rng, False).split("= ", 1)[1]
    stem = f"Solve {fn} {m}x = {value} for 0 ≤ x < 2π. {_FORMAT}"
    note = f"Let u = {m}x, so 0 ≤ u < {2 * m}π, and take every solution for u in that longer interval."
    return _result(fn, a, m, F(0), stem, note)


_QUAD_ROOTS = [F(-1), F(-1, 2), F(0), F(1, 2), F(1)]
_OUT_ROOTS = [F(2), F(-2), F(3, 2), F(-3, 2), F(3), F(-3)]


def _quadratic(fn, r1, r2):
    """(m s - p)(n s - q) expanded, as the stem prints it, for roots r1, r2: integer coefficients A s² + B s + C."""
    lcm = r1.denominator * r2.denominator // math.gcd(r1.denominator, r2.denominator)
    A = lcm * lcm if lcm > 1 else 1
    # (d1 s - n1)(d2 s - n2) with the roots n1/d1, n2/d2
    d1, n1, d2, n2 = r1.denominator, r1.numerator, r2.denominator, r2.numerator
    A, B, C = d1 * d2, -(d1 * n2 + d2 * n1), n1 * n2
    g = math.gcd(math.gcd(A, abs(B)), abs(C)) or 1
    return A // g, B // g, C // g


def _quad_text(fn, A, B, C) -> str:
    def term(coef, body, first):
        if coef == 0:
            return ""
        sign = "-" if coef < 0 else "+"
        mag = abs(coef)
        shown = body if (mag == 1 and body) else f"{mag}{body}"
        return (f"-{shown}" if sign == "-" else shown) if first else f" {sign} {shown}"

    s = f"{fn} x"
    pieces = [term(A, f"{fn}²x", True)]
    pieces.append(term(B, s, not pieces[0]))
    pieces.append(term(C, "", not (pieces[0] or pieces[1])))
    return "".join(pieces) + " = 0"


def _quad_roots_pair(rng, want_invalid: bool):
    for _ in attempts(200, "two distinct quadratic roots"):
        r1 = rng.choice(_QUAD_ROOTS)
        r2 = rng.choice(_OUT_ROOTS if want_invalid else _QUAD_ROOTS)
        if r1 != r2:
            break
    return r1, r2


def _quad_solutions(fn, roots) -> dict:
    return {r: solve(fn, sp.Rational(r.numerator, r.denominator)) for r in roots if abs(r) <= 1}


def _hard_quadratic(rng):
    fn = rng.choice(("sin", "cos"))
    invalid = rng.random() < 0.4
    r1, r2 = _quad_roots_pair(rng, invalid)
    A, B, C = _quadratic(fn, r1, r2)
    # check the expansion with SymPy
    s = sp.Symbol("s")
    check(
        sp.expand(
            A * s**2
            + B * s
            + C
            - A * (s - sp.Rational(r1.numerator, r1.denominator)) * (s - sp.Rational(r2.numerator, r2.denominator))
        )
        == 0,
        "quadratic differs from its roots",
    )
    per_root = _quad_solutions(fn, (r1, r2))
    sols = sorted({x for v in per_root.values() for x in v})
    check(sols, "no solutions")
    for r, v in per_root.items():
        verify(fn, sp.Rational(r.numerator, r.denominator), 1, F(0), v)
    eq = _quad_text(fn, A, B, C)
    stem = f"Solve {eq} for 0 ≤ x < 2π. {_FORMAT}"
    fields = pi_set_fields(sols)
    cands = []
    for r in per_root:
        cands.append(pi_set_dist(per_root[r], "r3g-trig-quadratic-root-dropped"))
    flipped = sorted({x for r in (r1, r2) if abs(r) <= 1 for x in solve(fn, sp.Rational(-r.numerator, r.denominator))})
    cands.append(pi_set_dist(flipped, "r3g-trig-quadratic-root-sign-slip"))
    inverted = [
        F(r.denominator, r.numerator)
        for r in (r1, r2)
        if r != 0 and abs(F(r.denominator, r.numerator)) in (F(1, 2), F(1))
    ]
    if inverted:
        cands.append(
            pi_set_dist(
                {x for r in inverted for x in solve(fn, sp.Rational(r.numerator, r.denominator))},
                "r3g-trig-quadratic-root-inverted",
            )
        )
    note = (
        f"Factorise as a quadratic in {fn} x: the roots are {fn} x = {typed(sp.Rational(r1.numerator, r1.denominator))} "
        f"and {fn} x = {typed(sp.Rational(r2.numerator, r2.denominator))}"
        + ("; a value outside [-1, 1] gives no solution." if invalid else ".")
    )
    return dict(
        text=stem,
        working=_set_working(fn, None, sols, note),
        distractors=pick(fields["answer_text"], cands, 3),
        **fields,
    )


def _hard_tan_square(rng):
    root = rng.choice((_S3 / 3, sp.Integer(1), _S3))
    k = sp.nsimplify(root**2)
    sols = sorted(set(solve("tan", root)) | set(solve("tan", -root)))
    verify("tan", root, 1, F(0), solve("tan", root))
    verify("tan", -root, 1, F(0), solve("tan", -root))
    fields = pi_set_fields(sols)
    positive = solve("tan", root)
    r = _REF["tan"][root]
    cands = [
        pi_set_dist(positive, "r3g-trig-square-root-negative-dropped"),
        pi_set_dist({r, 1 - r}, "r3g-trig-second-solution-wrong-quadrant"),
        pi_set_dist(set(solve("tan", -root)), "r3g-trig-square-root-negative-dropped"),
    ]
    stem = f"Solve tan²x = {typed(k)} for 0 ≤ x < 2π. {_FORMAT}".replace("²x", "²x")
    return dict(
        text=stem,
        working=[
            f"tan²x = {typed(k)} gives tan x = {typed(root)} or tan x = -{typed(root)}.",
            f"The solutions in the interval are {', '.join(pi_shown(s) for s in sols)}.",
        ],
        distractors=pick(fields["answer_text"], cands, 3),
        **fields,
    )


def _hard_shifted(rng):
    for _ in attempts(300, "a shifted equation with solutions"):
        fn = rng.choice(("sin", "cos", "tan"))
        a = rng.choice(_VALUES[fn]) * rng.choice((1, -1))
        m = rng.choice((1, 1, 2))
        b = rng.choice((F(1, 6), F(1, 4), F(1, 3), F(1, 12)))
        sols = solve(fn, a, m, b)
        if len(sols) >= 2:
            break
    verify(fn, a, m, b, sols)
    value = _equation(fn, a, rng, False).split("= ", 1)[1]
    inner = f"x - {pi_shown(b)}" if m == 1 else f"2(x - {pi_shown(b)})"
    stem = f"Solve {fn}({inner}) = {value} for 0 ≤ x < 2π. {_FORMAT}"
    fields = pi_set_fields(sols)
    cands = _slip_sets(fn, a, m, b)
    return dict(
        text=stem,
        working=[
            f"Let u = {'x - ' + pi_shown(b) if m == 1 else '2(x - ' + pi_shown(b) + ')'} and solve {fn} u = {value} over the interval u runs through.",
            f"Convert each u back to x. The solutions in the interval are {', '.join(pi_shown(s) for s in sols)}.",
        ],
        distractors=pick(fields["answer_text"], cands, 3),
        **fields,
    )


@generator(
    "r3g_trig_equations",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
@vary_symbol("x", ("x", "θ", "φ"))
def r3g_trig_equations(rng, tier="easy"):
    """Trigonometric equations over 0 ≤ x < 2π in exact radians; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_quadratic, _hard_quadratic, _hard_shifted, _hard_shifted, _hard_tan_square])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("r3g_exam_trig_equations", level=12, presets={"exam": dict(tier="exam")})
@retry_short
@vary_symbol("x", ("x", "θ", "φ"))
def r3g_exam_trig_equations(rng, tier="exam"):
    """Exam chain: the values of sin x (or cos x), the smallest and largest solution, and their difference."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(300, "a quadratic with a spread of solutions"):
        fn = rng.choice(("sin", "cos"))
        invalid = rng.random() < 0.35
        r1, r2 = _quad_roots_pair(rng, invalid)
        per_root = _quad_solutions(fn, (r1, r2))
        sols = sorted({x for v in per_root.values() for x in v})
        if len(sols) >= 3:
            break
    A, B, C = _quadratic(fn, r1, r2)
    for r, v in per_root.items():
        verify(fn, sp.Rational(r.numerator, r.denominator), 1, F(0), v)
    lo, hi = sols[0], sols[-1]
    values = value_set_fields([r1, r2])
    # distractors for (a): sign slip, inverted roots, one root only
    a_cands = [
        value_set_dist([-r1, -r2], "r3g-trig-quadratic-root-sign-slip"),
        value_set_dist(
            [F(r.denominator, r.numerator) if r else F(0) for r in (r1, r2)], "r3g-trig-quadratic-root-inverted"
        ),
        value_set_dist([r1], "r3g-trig-quadratic-root-dropped"),
        value_set_dist([r2], "r3g-trig-quadratic-root-dropped"),
    ]
    # (b) smallest, (c) largest: other solutions of the same equation, and the solution for the sign-flipped roots
    flipped = sorted({x for r in (r1, r2) if abs(r) <= 1 for x in solve(fn, sp.Rational(-r.numerator, r.denominator))})
    b_cands = [pi_dist(s, "r3g-trig-solution-not-smallest") for s in sols[1:2]]
    b_cands += [pi_dist(flipped[0], "r3g-trig-quadratic-root-sign-slip")] if flipped else []
    b_cands += [
        pi_dist(per_root_min, "r3g-trig-quadratic-root-dropped")
        for per_root_min in [min(v) for v in per_root.values()]
        if per_root_min != lo
    ]
    c_cands = [pi_dist(s, "r3g-trig-solution-not-largest") for s in sols[-2:-1]]
    c_cands += [pi_dist(flipped[-1], "r3g-trig-quadratic-root-sign-slip")] if flipped else []
    c_cands += [
        pi_dist(per_root_max, "r3g-trig-quadratic-root-dropped")
        for per_root_max in [max(v) for v in per_root.values()]
        if per_root_max != hi
    ]
    c_cands.append(pi_dist(2 - hi, "r3g-trig-second-solution-wrong-quadrant"))
    diff = hi - lo
    d_cands = [
        pi_dist(hi + lo, "r3g-trig-sum-for-difference"),
        pi_dist(2 - diff, "r3g-trig-interval-complement"),
        pi_dist(diff / 2, "r3g-trig-difference-halved"),
        pi_dist(hi - sols[1], "r3g-trig-solution-not-smallest"),
    ]
    eq = _quad_text(fn, A, B, C)
    return dict(
        text=f"Consider the equation {eq}, where 0 ≤ x < 2π.",
        parts=[
            dict(
                prompt=f"Find the values of {fn} x that satisfy the equation. Type each as a fraction or whole number, separated by commas.",
                marks=2,
                working=[f"Factorise: the roots in {fn} x are {rational(r1)} and {rational(r2)}."],
                distractors=pick(values["answer_text"], a_cands, 3),
                **values,
            ),
            dict(
                prompt="Find the smallest solution x. Give it exactly in radians, typed like 5pi/6.",
                answer_kind="expression",
                answer=pi_text(lo),
                answer_text=pi_text(lo),
                answer_latex=pi_latex(lo),
                marks=2,
                working=[
                    f"Solving each valid value of {fn} x in the interval, the smallest solution is {pi_shown(lo)}."
                ],
                distractors=pick(pi_text(lo), b_cands, 3),
            ),
            dict(
                prompt="Find the largest solution x. Give it exactly in radians.",
                answer_kind="expression",
                answer=pi_text(hi),
                answer_text=pi_text(hi),
                answer_latex=pi_latex(hi),
                marks=1,
                working=[f"The largest solution in the interval is {pi_shown(hi)}."],
                distractors=pick(pi_text(hi), c_cands, 3),
            ),
            dict(
                prompt="Hence find the difference between the largest and smallest solutions. Give it exactly in radians.",
                answer_kind="expression",
                answer=pi_text(diff),
                answer_text=pi_text(diff),
                answer_latex=pi_latex(diff),
                marks=1,
                follow_through="part_c - part_b",
                working=[f"{pi_shown(hi)} - {pi_shown(lo)} = {pi_shown(diff)}."],
                distractors=pick(pi_text(diff), d_cands, 3),
            ),
        ],
    )


def rational(r: Fraction) -> str:
    from ..core import rational_text

    return rational_text(r)
