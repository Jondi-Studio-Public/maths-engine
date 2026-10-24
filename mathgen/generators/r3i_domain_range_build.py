"""Generates `r3i_domain_range` questions (easy, medium, hard) and the exam chain `r3i_exam_domain_range`.

Skill ``domain-range-standard-functions`` (``functions-sequences``, answer ``inequality``): the largest domain and the
range of a function built from a square root, a reciprocal, a logarithm or a quadratic, written as an inequality
(or, for a set with a hole, ``x ≠ h``). Nothing is drawn.

* easy: ``√(ax + b) + c`` (or ``c - √(...)``, or ``√(b - ax)``) asked for its domain or its range, and
  ``k/(x - h) + c`` asked for its domain or its range.
* medium: the domain of ``ln(ax + b)``; the domain or range of ``1/√(ax + b)``; the range of a quadratic on a closed
  interval, where the vertex may or may not lie inside it; the domain or range of ``(ax + b)/(cx + d)``.
* hard: the domain of the square root or the logarithm of a quadratic (a union of two rays, or a closed interval); a
  semicircle ``k√(r² - x²) + c`` asked for its domain or its range; the range of a quadratic on a half-open interval,
  where one end is not attained.
* exam (``r3i_exam_domain_range``): ``g(x) = √(ax + b) - c``; (a) the least x in the domain, (b) the least value of g,
  (c) the x-intercept (follows through from (a)), (d) the domain of 1/g.

Each domain and range is kept as a list of intervals; the learner-facing text is built from the intervals, and the
function is then evaluated on a fine set of sample points (and next to every endpoint) to check that it is defined
exactly on the claimed domain and takes values only in the claimed range, reaching each finite end of it.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator, phrase, rational_text
from ._r3i_kit import ineq_dist, ineq_fields, num_cands, num_fields, pick, pick_text, retry_short

INF = None

#: An interval is (low, low_closed, high, high_closed); None for low or high means unbounded.


def _b(value) -> str:
    return rational_text(Fraction(value))


def ivtext(var: str, intervals) -> str:
    """The text of a set of reals given as intervals."""
    iv = sorted(intervals, key=lambda i: (i[0] is not None, i[0] if i[0] is not None else 0))
    if len(iv) == 1:
        lo, lc, hi, hc = iv[0]
        if lo is None and hi is None:
            return "all real numbers"
        if hi is None:
            return f"{var} {'≥' if lc else '>'} {_b(lo)}"
        if lo is None:
            return f"{var} {'≤' if hc else '<'} {_b(hi)}"
        return f"{_b(lo)} {'≤' if lc else '<'} {var} {'≤' if hc else '<'} {_b(hi)}"
    if len(iv) == 2 and iv[0][0] is None and iv[1][2] is None:
        (_, _, hi1, hc1), (lo2, lc2, _, _) = iv
        if hi1 == lo2 and not hc1 and not lc2:
            return f"{var} ≠ {_b(hi1)}"
        return f"{var} {'≤' if hc1 else '<'} {_b(hi1)} or {var} {'≥' if lc2 else '>'} {_b(lo2)}"
    raise AssertionError(f"unsupported interval set {iv}")


def inside(intervals, x) -> bool:
    x = float(x)
    for lo, lc, hi, hc in intervals:
        lo = None if lo is None else float(lo)
        hi = None if hi is None else float(hi)
        if (lo is None or x > lo or (lc and x == lo)) and (hi is None or x < hi or (hc and x == hi)):
            return True
    return False


def hole(point) -> list:
    return [(None, False, point, False), (point, False, None, False)]


def ray_up(point, closed) -> list:
    return [(point, closed, None, False)]


def ray_down(point, closed) -> list:
    return [(None, False, point, closed)]


ALL = [(None, False, None, False)]


# -- second route: evaluate the function ---------------------------------------------------------------------


def _defined(f, x) -> bool:
    try:
        y = f(float(x))
    except (ValueError, ZeroDivisionError, OverflowError):
        return False
    return not (isinstance(y, complex) or math.isnan(y) or math.isinf(y))


def probe_domain(f, domain, lo=-12.0, hi=12.0):
    """The function is defined exactly where `domain` says, at a grid and just either side of every endpoint."""
    points = {k / 4 for k in range(int(lo * 4), int(hi * 4) + 1)}
    for a, _, b, _ in domain:
        for e in (a, b):
            if e is not None:
                e = float(e)
                points |= {e, e + 1e-6, e - 1e-6}
    for x in sorted(points):
        check(_defined(f, x) == inside(domain, x), f"domain disagrees with the function at x = {x}")


def probe_range(f, domain, claimed, approach=(), grid_lo=-20.0, grid_hi=20.0, far=1e7):
    """Every sampled value lies in `claimed`, and each value in `approach` is reached to within 1e-4."""
    xs = {k / 4 for k in range(int(grid_lo * 4), int(grid_hi * 4) + 1)}
    for a, _, b, _ in domain:
        for e in (a, b):
            if e is not None:
                xs |= {float(e), float(e) + 1e-9, float(e) - 1e-9}
    xs |= {far, -far, far * 10, -far * 10, far * 1e6}
    values = [f(x) for x in sorted(xs) if inside(domain, x) and _defined(f, x)]
    check(values, "no sample point lies in the domain")
    tol = 1e-4
    for y in values:
        near = any(
            (lo is None or y > float(lo) - tol) and (hi is None or y < float(hi) + tol) for lo, _, hi, _ in claimed
        )
        check(near, f"value {y} lies outside the claimed range")
    for target in approach:
        check(min(abs(y - float(target)) for y in values) < tol, f"the range end {target} is not reached")


# -- distractor helpers -------------------------------------------------------------------------------------


def _cands(var: str, pairs) -> list:
    out = []
    for intervals, tag in pairs:
        if intervals is not None:
            out.append(ineq_dist(ivtext(var, intervals), tag))
    return out


def _q(rng, domain_ask: bool, fn_text: str):
    if domain_ask:
        return phrase(
            rng,
            (
                "Find the largest possible domain of f(x) = {f}. Give your answer as an inequality in x.",
                "State the largest possible domain of the function f(x) = {f}, as an inequality in x.",
            ),
            f=fn_text,
        )
    return phrase(
        rng,
        (
            "Find the range of f(x) = {f}. Give your answer as an inequality in y.",
            "State the range of the function f(x) = {f}, as an inequality in y.",
        ),
        f=fn_text,
    )


def _lin(a, b) -> str:
    """ax + b as written in a stem: 2x + 5, -3x + 1, x - 4."""
    head = "x" if a == 1 else "-x" if a == -1 else f"{a}x"
    if b == 0:
        return head
    return f"{head} {'+' if b > 0 else '-'} {abs(b)}"


def _shift(c) -> str:
    return "" if c == 0 else f" {'+' if c > 0 else '-'} {abs(c)}"


# -- easy ---------------------------------------------------------------------------------------------------


def _easy_sqrt(rng):
    a = rng.choice([1, 2, 3, 4, 5])
    x0 = rng.randint(-6, 6)
    b = -a * x0 if rng.random() < 0.75 else rng.randint(-9, 9)
    c = rng.choice([x for x in range(-6, 7) if x != 0])
    form = rng.choice(["plus", "plus", "minus", "reflect"])
    bound = Fraction(-b, a)
    if form == "reflect":
        # sqrt(b - ax): the domain is a ray pointing the other way
        text_arg = f"{b} - {a if a != 1 else ''}x" if b != 0 else f"-{a if a != 1 else ''}x"
        bound = Fraction(b, a)
        domain = ray_down(bound, True)
        f = lambda x: math.sqrt(b - a * x) + c  # noqa: E731
        fn_text = f"√({text_arg}){_shift(c)}"
        rng_set = ray_up(Fraction(c), True)
    elif form == "minus":
        domain = ray_up(bound, True)
        f = lambda x: c - math.sqrt(a * x + b)  # noqa: E731
        fn_text = f"{c} - √({_lin(a, b)})"
        rng_set = ray_down(Fraction(c), True)
    else:
        domain = ray_up(bound, True)
        f = lambda x: math.sqrt(a * x + b) + c  # noqa: E731
        fn_text = f"√({_lin(a, b)}){_shift(c)}"
        rng_set = ray_up(Fraction(c), True)
    probe_domain(f, domain)
    probe_range(f, domain, rng_set, approach=[c])
    ask_domain = rng.random() < 0.5
    if ask_domain:
        flipped = ray_up(bound, True) if form == "reflect" else ray_down(bound, True)
        cands = [
            (
                ray_up(Fraction(-b), True) if form != "reflect" else ray_down(Fraction(b), True),
                "r3i-domain-no-division-by-coefficient",
            ),
            (ray_up(-bound, True) if form != "reflect" else ray_down(-bound, True), "r3i-domain-bound-sign"),
            (ray_up(bound, False) if form != "reflect" else ray_down(bound, False), "r3i-domain-strict-for-root"),
            (flipped, "r3i-domain-inequality-direction"),
        ]
        ans = ivtext("x", domain)
        working = [
            "The expression under the square root must be at least 0 for the function to be defined.",
            f"Solving gives {ans}.",
        ]
    else:
        cands = [
            (
                ray_up(Fraction(0), True) if form != "minus" else ray_down(Fraction(0), True),
                "r3i-range-ignores-vertical-shift",
            ),
            (ray_up(Fraction(-c), True) if form != "minus" else ray_down(Fraction(-c), True), "r3i-range-shift-sign"),
            (ray_down(Fraction(c), True) if form != "minus" else ray_up(Fraction(c), True), "r3i-range-direction"),
            (
                ray_up(Fraction(c), False) if form != "minus" else ray_down(Fraction(c), False),
                "r3i-range-end-not-attained",
            ),
        ]
        ans = ivtext("y", rng_set)
        working = [
            "A square root is never negative, so its smallest value is 0, reached at the end of the domain."
            if form != "minus"
            else "A square root is never negative, so subtracting it gives values no larger than the constant.",
            f"Shifting by {c} gives {ans}.",
        ]
    fields = ineq_fields(ans)
    var = "x" if ask_domain else "y"
    return dict(
        text=_q(rng, ask_domain, fn_text), working=working, distractors=pick_text(fields, _cands(var, cands)), **fields
    )


def _easy_reciprocal(rng):
    k = rng.choice([1, 2, 3, 4, 5, -1, -2, -3])
    h = rng.choice([x for x in range(-6, 7) if x != 0])
    c = rng.choice([x for x in range(-5, 6) if x != 0])
    f = lambda x: k / (x - h) + c  # noqa: E731
    domain, rng_set = hole(Fraction(h)), hole(Fraction(c))
    probe_domain(f, domain)
    probe_range(f, domain, rng_set, approach=[c])
    # the value y = c is never taken: k/(x - h) = 0 has no solution
    denominator = f"x {'-' if h > 0 else '+'} {abs(h)}"
    fn_text = f"{k}/({denominator}){_shift(c)}"
    ask_domain = rng.random() < 0.5
    if ask_domain:
        cands = [
            (hole(Fraction(-h)), "r3i-domain-hole-sign"),
            (hole(Fraction(c)), "r3i-domain-range-confused"),
            (hole(Fraction(0)), "r3i-domain-hole-at-zero"),
            (hole(Fraction(k)), "r3i-domain-hole-from-numerator"),
        ]
        ans = ivtext("x", domain)
        working = [f"The denominator x {'-' if h > 0 else '+'} {abs(h)} is 0 when x = {h}, so x ≠ {h}."]
        var = "x"
    else:
        cands = [
            (hole(Fraction(-c)), "r3i-range-shift-sign"),
            (hole(Fraction(0)), "r3i-range-hole-at-zero"),
            (hole(Fraction(h)), "r3i-domain-range-confused"),
            (ALL, "r3i-range-all-reals"),
        ]
        ans = ivtext("y", rng_set)
        working = [f"k/(x - h) is never 0, so the function never takes the value {c}: y ≠ {c}."]
        var = "y"
    fields = ineq_fields(ans)
    return dict(
        text=_q(rng, ask_domain, fn_text), working=working, distractors=pick_text(fields, _cands(var, cands)), **fields
    )


def _easy(rng):
    return _easy_sqrt(rng) if rng.random() < 0.6 else _easy_reciprocal(rng)


# -- medium -------------------------------------------------------------------------------------------------


def _medium_ln(rng):
    a = rng.choice([1, 2, 3, 4, 5])
    x0 = rng.randint(-6, 6)
    b = -a * x0 if rng.random() < 0.7 else rng.randint(-9, 9)
    c = rng.choice([0, 0, 1, -1, 2, -3])
    bound = Fraction(-b, a)
    domain = ray_up(bound, False)
    f = lambda x: math.log(a * x + b) + c  # noqa: E731
    probe_domain(f, domain)
    ans = ivtext("x", domain)
    cands = [
        (ray_up(bound, True), "r3i-domain-nonstrict-for-log"),
        (ray_up(Fraction(-b), False), "r3i-domain-no-division-by-coefficient"),
        (ray_up(-bound, False), "r3i-domain-bound-sign"),
        (ray_down(bound, False), "r3i-domain-inequality-direction"),
    ]
    fields = ineq_fields(ans)
    return dict(
        text=_q(rng, True, f"ln({_lin(a, b)}){_shift(c)}"),
        working=["The argument of a logarithm must be positive.", f"{_lin(a, b)} > 0 gives {ans}."],
        distractors=pick_text(fields, _cands("x", cands)),
        **fields,
    )


def _medium_inv_sqrt(rng):
    a = rng.choice([1, 2, 3, 4])
    x0 = rng.randint(-5, 5)
    b = -a * x0
    c = rng.choice([0, 0, 1, -2, 3])
    f = lambda x: 1 / math.sqrt(a * x + b) + c  # noqa: E731
    domain = ray_up(Fraction(x0), False)
    probe_domain(f, domain)
    ask_domain = rng.random() < 0.5
    fn_text = f"1/√({_lin(a, b)}){_shift(c)}"
    if ask_domain:
        ans = ivtext("x", domain)
        cands = [
            (ray_up(Fraction(x0), True), "r3i-domain-strict-for-denominator"),
            (ray_up(Fraction(-b), False), "r3i-domain-no-division-by-coefficient"),
            (ray_up(Fraction(-x0), False), "r3i-domain-bound-sign"),
            (hole(Fraction(x0)), "r3i-domain-only-zero-removed"),
        ]
        working = [
            "The square root must be positive, not just non-negative, because it is in the denominator.",
            f"{_lin(a, b)} > 0 gives {ans}.",
        ]
        var = "x"
    else:
        rng_set = ray_up(Fraction(c), False)
        probe_range(f, domain, rng_set, approach=[c])
        ans = ivtext("y", rng_set)
        cands = [
            (ray_up(Fraction(c), True), "r3i-range-end-attained"),
            (ray_up(Fraction(0), False), "r3i-range-ignores-vertical-shift"),
            (hole(Fraction(c)), "r3i-range-hole-not-ray"),
            (ray_up(Fraction(-c), False), "r3i-range-shift-sign"),
        ]
        working = [
            "1/√(...) takes every positive value, and never 0.",
            f"Shifting by {c} gives {ans}.",
        ]
        var = "y"
    fields = ineq_fields(ans)
    return dict(
        text=_q(rng, ask_domain, fn_text), working=working, distractors=pick_text(fields, _cands(var, cands)), **fields
    )


def _quad_text(a, b, c) -> str:
    out = "x²" if a == 1 else "-x²" if a == -1 else f"{a}x²"
    if b:
        out += f" {'+' if b > 0 else '-'} {'' if abs(b) == 1 else abs(b)}x"
    if c:
        out += f" {'+' if c > 0 else '-'} {abs(c)}"
    return out


def _medium_quad_interval(rng):
    for _ in attempts(500, "a quadratic and a closed interval with distinct end values"):
        a = rng.choice([1, 1, 2, 3, -1, -2])
        h = rng.randint(-4, 4)
        k = rng.randint(-8, 8)
        b, c = -2 * a * h, a * h * h + k
        p = rng.randint(-6, 3)
        q = p + rng.randint(3, 8)
        fp, fq = a * p * p + b * p + c, a * q * q + b * q + c
        if fp == fq or b == 0:
            continue
        break
    f = lambda x: a * x * x + b * x + c  # noqa: E731
    vertex_in = p <= h <= q
    ends = sorted([fp, fq])
    if a > 0:
        lo, hi = (k if vertex_in else ends[0]), ends[1]
    else:
        lo, hi = ends[0], (k if vertex_in else ends[1])
    domain = [(Fraction(p), True, Fraction(q), True)]
    rng_set = [(Fraction(lo), True, Fraction(hi), True)]
    probe_range(f, domain, rng_set, approach=[lo, hi], grid_lo=p, grid_hi=q)
    # the vertex itself is among the sample points when it lies inside the interval
    if vertex_in:
        check(min(f(x / 16) for x in range(p * 16, q * 16 + 1)) >= min(lo, hi) - 1e-9, "range misses the vertex")
    ans = ivtext("y", rng_set)
    cands = [
        ([(Fraction(ends[0]), True, Fraction(ends[1]), True)], "r3i-range-endpoints-only"),
        (
            [(Fraction(k), True, Fraction(ends[1]), True)] if a > 0 else [(Fraction(ends[0]), True, Fraction(k), True)],
            "r3i-range-vertex-always-used",
        ),
        ([(Fraction(fp), True, Fraction(fq), True)] if fp < fq else None, "r3i-range-values-in-domain-order"),
        ([(Fraction(p), True, Fraction(q), True)], "r3i-range-domain-confused"),
    ]
    fields = ineq_fields(ans)
    return dict(
        text=(
            f"Find the range of f(x) = {_quad_text(a, b, c)} for {p} ≤ x ≤ {q}. Give your answer as an inequality in y."
        ),
        working=[
            f"The vertex is at x = {h} and f({h}) = {k}; it {'lies inside' if vertex_in else 'does not lie inside'} the interval.",
            f"f({p}) = {fp} and f({q}) = {fq}, so the range is {ans}.",
        ],
        distractors=pick_text(fields, _cands("y", [c for c in cands if c[0] is not None])),
        **fields,
    )


def _medium_rational(rng):
    for _ in attempts(500, "a Mobius function that does not simplify"):
        a, b = rng.choice([1, 2, 3, 4, 5, -1, -2, -3]), rng.randint(-6, 6)
        c, d = rng.choice([1, 2, 3, -1, -2]), rng.randint(-6, 6)
        if a * d - b * c != 0 and d != 0:
            break
    f = lambda x: (a * x + b) / (c * x + d)  # noqa: E731
    x_hole, y_hole = Fraction(-d, c), Fraction(a, c)
    domain, rng_set = hole(x_hole), hole(y_hole)
    probe_domain(f, domain)
    probe_range(f, domain, rng_set, approach=[y_hole])
    ask_domain = rng.random() < 0.5
    fn_text = f"({_lin(a, b)})/({_lin(c, d)})"
    if ask_domain:
        ans = ivtext("x", domain)
        cands = [
            (hole(Fraction(d, c)), "r3i-domain-hole-sign"),
            (hole(Fraction(-d)), "r3i-domain-no-division-by-coefficient"),
            (hole(y_hole), "r3i-domain-range-confused"),
            (hole(Fraction(-b, a)), "r3i-domain-hole-from-numerator"),
        ]
        working = [f"The denominator {_lin(c, d)} is 0 when x = {_b(x_hole)}.", f"So the domain is {ans}."]
        var = "x"
    else:
        ans = ivtext("y", rng_set)
        cands = [
            (hole(x_hole), "r3i-domain-range-confused"),
            (hole(Fraction(a)), "r3i-range-ratio-without-denominator-coefficient"),
            (hole(Fraction(c, a)), "r3i-range-ratio-inverted"),
            (hole(-y_hole), "r3i-range-shift-sign"),
        ]
        working = [
            f"As x grows large, ({_lin(a, b)})/({_lin(c, d)}) approaches {_b(y_hole) if f'{a}/{c}' == _b(y_hole) else f'{a}/{c} = ' + _b(y_hole)} but never reaches it.",
            f"Every other value is taken, so the range is {ans}.",
        ]
        var = "y"
    fields = ineq_fields(ans)
    return dict(
        text=_q(rng, ask_domain, fn_text), working=working, distractors=pick_text(fields, _cands(var, cands)), **fields
    )


def _medium(rng):
    return rng.choice([_medium_ln, _medium_inv_sqrt, _medium_quad_interval, _medium_rational])(rng)


# -- hard ---------------------------------------------------------------------------------------------------


def _hard_root_of_quadratic(rng):
    for _ in attempts(200, "two distinct integer roots"):
        p = rng.randint(-6, 4)
        q = p + rng.randint(1, 7)
        if p != q:
            break
    kind = rng.choice(["sqrt_up", "sqrt_down", "ln_up", "ln_down"])
    sign = 1 if kind.endswith("up") else -1
    s, pr = p + q, p * q
    poly = f"{_quad_text(sign, -sign * s, sign * pr)}"
    if kind.startswith("sqrt"):
        f = lambda x: math.sqrt(sign * (x - p) * (x - q))  # noqa: E731
        outer = [(None, False, Fraction(p), True), (Fraction(q), True, None, False)]
        inner = [(Fraction(p), True, Fraction(q), True)]
        fn = f"√({poly})"
        why = "The expression under the square root must be at least 0."
    else:
        f = lambda x: math.log(sign * (x - p) * (x - q))  # noqa: E731
        outer = [(None, False, Fraction(p), False), (Fraction(q), False, None, False)]
        inner = [(Fraction(p), False, Fraction(q), False)]
        fn = f"ln({poly})"
        why = "The argument of a logarithm must be positive."
    domain = outer if sign == 1 else inner
    probe_domain(f, domain)
    ans = ivtext("x", domain)
    wrong_shape = inner if sign == 1 else outer
    strict = [(lo, not lc if lo is not None else lc, hi, not hc if hi is not None else hc) for lo, lc, hi, hc in domain]
    cands = [
        (wrong_shape, "r3i-domain-quadratic-sign-chart-reversed"),
        (strict, "r3i-domain-strictness-slip"),
        (
            [(None, False, Fraction(-q), True), (Fraction(-p), True, None, False)]
            if sign == 1
            else [(Fraction(-q), True, Fraction(-p), True)],
            "r3i-domain-bound-sign",
        ),
        (
            [(Fraction(p), True, None, False)] if sign == 1 else [(Fraction(p), True, Fraction(p + q), True)],
            "r3i-domain-quadratic-one-root-only",
        ),
    ]
    fields = ineq_fields(ans)
    return dict(
        text=_q(rng, True, fn),
        working=[why, f"{poly} {'≥' if kind.startswith('sqrt') else '>'} 0 has roots x = {p} and x = {q}, so {ans}."],
        distractors=pick_text(fields, _cands("x", cands)),
        **fields,
    )


def _hard_semicircle(rng):
    r = rng.randint(2, 8)
    k = rng.choice([1, 1, 2, 3])
    c = rng.choice([0, 1, -2, 3, -1])
    f = lambda x: k * math.sqrt(r * r - x * x) + c  # noqa: E731
    domain = [(Fraction(-r), True, Fraction(r), True)]
    rng_set = [(Fraction(c), True, Fraction(c + k * r), True)]
    probe_domain(f, domain, lo=-r - 3, hi=r + 3)
    probe_range(f, domain, rng_set, approach=[c, c + k * r], grid_lo=-r, grid_hi=r)
    fn_text = f"{'' if k == 1 else k}√({r * r} - x²){_shift(c)}"
    ask_domain = rng.random() < 0.5
    if ask_domain:
        ans = ivtext("x", domain)
        cands = [
            ([(Fraction(-r * r), True, Fraction(r * r), True)], "r3i-domain-radius-squared-used"),
            (ray_down(Fraction(r), True), "r3i-domain-one-sided"),
            ([(Fraction(-r), False, Fraction(r), False)], "r3i-domain-strictness-slip"),
            (
                ray_up(Fraction(r), True) + [(None, False, Fraction(-r), True)],
                "r3i-domain-quadratic-sign-chart-reversed",
            ),
        ]
        working = [f"{r * r} - x² ≥ 0 gives x² ≤ {r * r}, so {ans}."]
        var = "x"
    else:
        ans = ivtext("y", rng_set)
        cands = [
            ([(Fraction(0), True, Fraction(k * r), True)], "r3i-range-ignores-vertical-shift"),
            ([(Fraction(c), True, Fraction(c + r), True)], "r3i-range-scale-factor-ignored"),
            ([(Fraction(c), True, Fraction(c + k * r * r), True)], "r3i-range-radius-squared-used"),
            ([(Fraction(c - k * r), True, Fraction(c + k * r), True)], "r3i-range-symmetric-about-shift"),
        ]
        working = [
            f"The root √({r * r} - x²) takes values from 0 (at x = ±{r}) up to {r} (at x = 0).",
            f"Scaling by {k} and shifting by {c} gives {ans}.",
        ]
        var = "y"
    # the cands list may contain a two-ray union written for a closed interval: only valid sets are used
    cands = [
        cand
        for cand in cands
        if len(cand[0]) == 1 or (len(cand[0]) == 2 and cand[0][0][0] is None and cand[0][1][2] is None)
    ]
    fields = ineq_fields(ans)
    return dict(
        text=_q(rng, ask_domain, fn_text), working=working, distractors=pick_text(fields, _cands(var, cands)), **fields
    )


def _hard_half_open(rng):
    for _ in attempts(500, "a quadratic with its vertex inside a half-open interval"):
        a = rng.choice([1, 1, 2, -1, -2])
        h = rng.randint(-3, 3)
        k = rng.randint(-6, 6)
        b, c = -2 * a * h, a * h * h + k
        p, q = h - rng.randint(1, 5), h + rng.randint(1, 5)
        if p == q or b == 0:
            continue
        fp, fq = a * p * p + b * p + c, a * q * q + b * q + c
        if fp != fq:
            break
    f = lambda x: a * x * x + b * x + c  # noqa: E731
    open_left = rng.random() < 0.5
    far_end_value = max(fp, fq) if a > 0 else min(fp, fq)
    far_is_left = fp == far_end_value
    # the end whose value sets the far end of the range: open if that end is the open end of the interval
    far_open = (far_is_left and open_left) or ((not far_is_left) and (not open_left))
    domain = [(Fraction(p), not open_left, Fraction(q), open_left)]
    if a > 0:
        rng_set = [(Fraction(k), True, Fraction(far_end_value), not far_open)]
    else:
        rng_set = [(Fraction(far_end_value), not far_open, Fraction(k), True)]
    probe_range(f, domain, rng_set, approach=[k, far_end_value], grid_lo=p, grid_hi=q)
    # the far end is only reached when the interval includes it
    ans = ivtext("y", rng_set)
    closed_both = [(rng_set[0][0], True, rng_set[0][2], True)]
    open_both = [(rng_set[0][0], False, rng_set[0][2], False)]
    sets = [
        (closed_both, "r3i-range-open-end-treated-as-closed"),
        (open_both, "r3i-range-vertex-treated-as-open"),
        ([(Fraction(min(fp, fq)), True, Fraction(max(fp, fq)), True)], "r3i-range-endpoints-only"),
        ([(Fraction(p), True, Fraction(q), True)], "r3i-range-domain-confused"),
    ]
    fields = ineq_fields(ans)
    interval_text = f"{p} < x ≤ {q}" if open_left else f"{p} ≤ x < {q}"
    return dict(
        text=(
            f"Find the range of f(x) = {_quad_text(a, b, c)} for {interval_text}. "
            "Give your answer as an inequality in y."
        ),
        working=[
            f"The vertex is at ({h}, {k}), inside the interval, so y = {k} is attained.",
            f"f({p}) = {fp} and f({q}) = {fq}; the larger end value is only attained if its end of the interval is included.",
            f"The range is {ans}.",
        ],
        distractors=pick_text(fields, _cands("y", sets)),
        **fields,
    )


def _hard(rng):
    return rng.choice([_hard_root_of_quadratic, _hard_semicircle, _hard_half_open])(rng)


@generator(
    "r3i_domain_range",
    level=13,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def r3i_domain_range(rng, tier="easy"):
    """Domain and range of standard functions; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------


@generator("r3i_exam_domain_range", level=13, presets={"exam": dict(tier="exam")})
@retry_short
def r3i_exam_domain_range(rng, tier="exam"):
    """Exam chain on g(x) = √(ax + b) - c: the end of the domain, the least value, the x-intercept, the domain of 1/g."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a root function whose intercept is a whole number"):
        c = rng.randint(1, 6)
        a = rng.choice([x for x in (1, 2, 3, 4, 5, 6, 8, 9) if (c * c) % x == 0])
        x0 = rng.randint(-6, 4)
        m = c * c // a
        if m >= 1:
            break
    b = -a * x0
    xi = x0 + m
    g = lambda x: math.sqrt(a * x + b) - c  # noqa: E731
    domain = ray_up(Fraction(x0), True)
    probe_domain(g, domain)
    check(abs(g(xi)) < 1e-12, "the x-intercept is not a zero of g")
    check(all(g(x0 + t / 7) > -c - 1e-12 for t in range(0, 200)), "g dips below -c")
    # the domain of 1/g: x >= x0 with the zero of g taken out
    dom_inv = [(Fraction(x0), True, Fraction(xi), False), (Fraction(xi), False, None, False)]

    def inv(x):
        return 1 / g(x)

    probe_domain(inv, dom_inv)
    ans_d = f"x ≥ {x0} and x ≠ {xi}"
    f_a, f_b, f_c = num_fields(x0), num_fields(-c), num_fields(xi)
    f_d = ineq_fields(ans_d)
    return dict(
        text=phrase(
            rng,
            [
                "Let g(x) = √({lin}) - {c}.",
                "Consider the function g(x) = √({lin}) - {c}.",
                "A function g is defined by g(x) = √({lin}) - {c}.",
            ],
            lin=_lin(a, b),
            c=c,
        ),
        parts=[
            dict(
                prompt="Find the smallest value of x for which g(x) is defined.",
                marks=1,
                sensible_range=(-100, 100),
                working=[f"{_lin(a, b)} ≥ 0 gives x ≥ {x0}, so the smallest value is {x0}."],
                distractors=pick(
                    f_a,
                    num_cands(
                        [
                            (-x0, "r3i-domain-bound-sign"),
                            (-b, "r3i-domain-no-division-by-coefficient"),
                            (b, "r3i-domain-bound-sign"),
                        ],
                        x0,
                    ),
                ),
                **f_a,
            ),
            dict(
                prompt="Find the smallest value that g(x) takes.",
                marks=1,
                sensible_range=(-100, 100),
                working=[f"The square root is at least 0, so g(x) ≥ -{c}; the value -{c} is reached at x = {x0}."],
                distractors=pick(
                    f_b,
                    num_cands(
                        [
                            (0, "r3i-range-ignores-vertical-shift"),
                            (c, "r3i-range-shift-sign"),
                            (x0 - c, "r3i-range-shift-sign"),
                        ],
                        -c,
                    ),
                ),
                **f_b,
            ),
            dict(
                prompt="Find the x-coordinate of the point where the graph of y = g(x) meets the x-axis.",
                marks=2,
                follow_through=f"part_a+{m}",
                sensible_range=(-100, 100),
                working=[f"√({_lin(a, b)}) = {c} gives {_lin(a, b)} = {c * c}, so x = {xi}."],
                distractors=pick(
                    f_c,
                    num_cands(
                        [
                            (Fraction(c - b, a), "r3i-intercept-root-not-squared"),
                            (Fraction(c * c + b, a), "r3i-intercept-sign-slip"),
                            (m, "r3i-intercept-measured-from-origin"),
                        ],
                        xi,
                    ),
                ),
                **f_c,
            ),
            dict(
                prompt="Write down the largest possible domain of h(x) = 1/g(x), as an inequality.",
                marks=3,
                working=[
                    f"h is defined where g is defined (x ≥ {x0}) and g(x) ≠ 0, which removes x = {xi}.",
                    f"The domain is {ans_d}.",
                ],
                distractors=pick_text(
                    f_d,
                    [
                        ineq_dist(f"x > {x0}", "r3i-domain-only-endpoint-removed"),
                        ineq_dist(f"x ≥ {x0}", "r3i-domain-zero-of-denominator-missed"),
                        ineq_dist(f"x ≥ {x0} and x ≠ {-xi}", "r3i-domain-hole-sign"),
                        ineq_dist(f"x ≠ {xi}", "r3i-domain-root-condition-missed"),
                    ],
                ),
                **f_d,
            ),
        ],
    )
