"""Quadratic and rational inequalities: topics `w5_quadratic_inequality` and `w5_rational_inequality` (easy, medium,
hard) and their exam chains `exd_x_w5_quadratic_inequality` and `exd_x_w5_rational_inequality`.

Skills `quadratic-inequalities` and `rational-inequalities` (domain `inequalities`, answer `inequality`): the learner
types the solution set, as inequalities joined with "or" or as intervals joined with a union sign, and the real
inequality grader compares sets, so `x < -2 or x > 3`, `(-inf, -2) U (3, inf)` and `x < -2, x > 3` are one answer.

Quadratic inequalities
* easy: x^2 + bx + c compared with 0 (expanded, factorised, or with the constant on the other side), whole roots.
* medium: a leading coefficient that is not 1 (often negative) with fractional roots, a constant moved to the other
  side before the roots show, and squares such as x^2 >= 9.
* hard: a repeated root, no real roots (so every x or no x), surd bounds from completing the square, and a height or
  profit model with a stated level to reach.
* exam: the turning point, the solution set, how many whole numbers satisfy a strict or closed version, and the
  greatest or least value (the last two follow through from the turning point).

Rational inequalities
* easy: (x - a)/(x - b) compared with 0; the zero may be included, the pole never.
* medium: coefficients that are not 1 (including a negative one) and a fraction compared with a number that is not 0,
  which must first be rearranged into one fraction.
* hard: three critical values (a quadratic over a linear, or a linear over a quadratic) and a squared numerator.
* exam: the excluded value, two solution sets (one against 0, one against a number) and the whole numbers in a window.

Every answer comes from testing the sign of the expression in each region between the critical values, and is checked a
second way by evaluating the ORIGINAL statement at many exact points either side of every critical value.
"""

from __future__ import annotations

import re
from fractions import Fraction as Fr

import sympy

from ..core import attempts, check, generator, join_terms, paren_negative, phrase
from ._w5_kit import (
    TooFewDistractors,
    enough,
    in_set,
    sinter,
    sunion,
    scomp,
    ineq_cands,
    ineq_fields,
    number_cands,
    number_fields,
    part,
    retry_draw,
    rt,
    sym,
)

QUAD = "w5_quadratic_inequality"
RATL = "w5_rational_inequality"
QUAD_EXAM = "exd_x_w5_quadratic_inequality"
RATL_EXAM = "exd_x_w5_rational_inequality"

_OPS = ("<", "≤", ">", "≥")
_FLIP = {"<": ">", "≤": "≥", ">": "<", "≥": "≤"}
_STRICTNESS = {"<": "≤", "≤": "<", ">": "≥", "≥": ">"}
_OO = sympy.oo
_R = sympy.S.Reals
_EMPTY = sympy.S.EmptySet

_ASK = (
    "Write your answer as inequalities joined with 'or', or in interval notation.",
    "Give the solution set as inequalities (join separate pieces with 'or') or as intervals.",
    "State the solution using inequalities (use 'or' between separate pieces) or interval notation.",
)
_LEAD = ("Solve {e}.", "Find all values for which {e}.", "Solve the inequality {e}.")


def _holds(value, op: str) -> bool:
    return {"<": value < 0, "≤": value <= 0, ">": value > 0, "≥": value >= 0}[op]


def solve_sign(f, zeros, poles, op: str):
    """The set where `f(x) op 0`, where `f` returns a Fraction (None at a pole). One test point in each region between
    the critical values decides that region; a zero is added back for a non-strict relation, a pole never is."""
    points = sorted({Fr(z) for z in zeros} | {Fr(p) for p in poles})
    edges = [None, *points, None]
    pieces = []
    for lo, hi in zip(edges, edges[1:]):
        if lo is None and hi is None:
            probe = Fr(0)
        elif lo is None:
            probe = hi - 1
        elif hi is None:
            probe = lo + 1
        else:
            probe = (lo + hi) / 2
        value = f(probe)
        check(value is not None, "the test point sits on a pole")
        if _holds(value, op):
            pieces.append(sympy.Interval.open(-_OO if lo is None else sym(lo), _OO if hi is None else sym(hi)))
    if op in ("≤", "≥"):
        pieces += [sympy.FiniteSet(sym(Fr(z))) for z in zeros]
    return sunion(*pieces)


def _near(c) -> Fr:
    return Fr(c) if isinstance(c, (int, Fr)) else Fr(round(float(c) * 16), 16)


def probe_points(critical):
    """Exact test points: each critical value, points either side of it, and midpoints of neighbours."""
    cs = sorted(critical, key=float)
    pts = set()
    for c in cs:
        base = _near(c)
        for d in (Fr(0), Fr(1, 100), Fr(-1, 100), Fr(1, 2), Fr(-1, 2), Fr(3), Fr(-3)):
            pts.add(base + d)
    for a, b in zip(cs, cs[1:]):
        pts.add((_near(a) + _near(b)) / 2)
    return pts


def verify(stmt, op: str, solution, critical, extra=(), floor=None) -> None:
    """`solution` and the original statement `stmt(x) op 0` agree at every probe point (at or above `floor`, if given)."""
    for p in probe_points(critical) | set(extra):
        if floor is not None and p < floor:
            continue
        value = stmt(p)
        theirs = value is not None and _holds(value, op)
        check(in_set(solution, p) == theirs, f"the answer disagrees with the statement at x = {p}")


def _lin(a, b, var="x") -> str:
    return join_terms([(a, var), (b, "")])


def _wrap(expr: str) -> str:
    """Brackets round everything except a lone letter or whole number, so a numerator or denominator is never ambiguous."""
    return expr if re.fullmatch(r"[a-z]|\d+|\([^()]*\)(?:\^\d)?", expr) else f"({expr})"


def _frac(num: str, den: str) -> str:
    return f"{_wrap(num)}/{_wrap(den)}"


def _nz(rng, lo, hi, *avoid):
    return rng.choice([v for v in range(lo, hi + 1) if v != 0 and v not in avoid])


def _stem(rng, e: str) -> str:
    return f"{phrase(rng, _LEAD, e=e)} {rng.choice(_ASK)}"


def _ray(op: str, bound, var_side="x"):
    """The set where x op bound (op one of the four relations)."""
    b = sym(bound)
    return {
        "<": sympy.Interval.open(-_OO, b),
        "≤": sympy.Interval(-_OO, b, True, False),
        ">": sympy.Interval.open(b, _OO),
        "≥": sympy.Interval(b, _OO, False, True),
    }[op]


# -- quadratic inequalities -------------------------------------------------------------------------------------------


def _quad_slips(f, zeros, op, extra=()):
    """Tagged wrong solution sets for a quadratic with two real roots `zeros` (sorted) and the relation `op`."""
    lo, hi = zeros
    out = [
        (solve_sign(f, zeros, [], _FLIP[op]), "w5qi-region-reversed"),
        (solve_sign(f, zeros, [], _STRICTNESS[op]), "w5qi-strictness-swapped"),
        *extra,
    ]
    if (-hi, -lo) != (lo, hi):
        out.append((solve_sign(lambda x: f(-x), [-hi, -lo], [], op), "w5qi-roots-negated"))
    # each factor given the sign of the whole product: one ray instead of two regions
    out.append((_ray(op, lo if op in ("<", "≤") else hi), "w5qi-one-factor-only"))
    return out


def _quad_answer(f, zeros, op, extra=()):
    zeros = sorted(Fr(z) for z in zeros)
    solution = solve_sign(f, zeros, [], op)
    verify(f, op, solution, zeros)
    fields = ineq_fields(solution)
    cands = ineq_cands(fields, _quad_slips(f, zeros, op, extra))
    enough(cands)
    return solution, fields, cands


def _sign_working(zeros, lead_sign, op, answer):
    lo, hi = zeros
    shape = (
        "opens upward, so it is below zero between the roots"
        if lead_sign > 0
        else ("opens downward, so it is above zero between the roots")
    )
    return [
        f"Roots: x = {rt(lo)} and x = {rt(hi)}.",
        f"The parabola {shape} and has the opposite sign outside them.",
        f"Keep the region(s) where the expression {op} 0: {answer}.",
    ]


def _quad_easy(rng):
    p, q = sorted(rng.sample(range(-8, 9), 2))
    op = rng.choice(_OPS)
    form = rng.choice(("expanded", "expanded", "factorised", "moved"))
    f = lambda x: (x - p) * (x - q)  # noqa: E731
    solution, fields, cands = _quad_answer(f, [p, q], op)
    b, c = -(p + q), p * q
    if form == "expanded":
        expr = f"{join_terms([(1, 'x^2'), (b, 'x'), (c, '')])} {op} 0"
    elif form == "factorised":
        expr = f"({_lin(1, -p)})({_lin(1, -q)}) {op} 0"
    else:
        expr = f"{join_terms([(1, 'x^2'), (b, 'x')])} {op} {-c}"
    return dict(
        text=_stem(rng, expr), working=_sign_working([p, q], 1, op, fields["answer_text"]), distractors=cands, **fields
    )


def _quad_medium(rng):
    kind = rng.choice(("lead", "lead", "moved", "square"))
    op = rng.choice(_OPS)
    extra = []
    lead_sign = 1
    if kind == "lead":
        for _ in attempts(200, "distinct rational roots"):
            m, n = rng.choice((1, 2, 3)), rng.choice((2, 3, 4))
            p, q = _nz(rng, -7, 7), _nz(rng, -7, 7)
            if Fr(p, m) != Fr(q, n):
                break
        a = rng.choice((1, -1, -1))
        A, B, C = a * m * n, -a * (m * q + n * p), a * p * q  # a (mx - p)(nx - q)
        f = lambda x: A * x * x + B * x + C  # noqa: E731
        zeros = [Fr(p, m), Fr(q, n)]
        expr = f"{join_terms([(A, 'x^2'), (B, 'x'), (C, '')])} {op} 0"
        lead_sign = a
        if a < 0:
            extra.append((solve_sign(lambda x: -f(x), zeros, [], op), "w5qi-ignores-negative-leading"))
    elif kind == "moved":
        p, q = sorted(rng.sample(range(-6, 7), 2))
        b, c = -(p + q), p * q
        d = _nz(rng, -9, 9)
        shown = c + d  # x^2 + bx + shown  op  d   is   (x - p)(x - q) op 0
        f = lambda x: (x - p) * (x - q)  # noqa: E731
        zeros = [Fr(p), Fr(q)]
        expr = f"{join_terms([(1, 'x^2'), (b, 'x'), (shown, '')])} {op} {d}"
        disc = b * b - 4 * shown
        if disc > 0 and int(disc**0.5) ** 2 == disc:
            r = int(disc**0.5)
            wrong_roots = [Fr(-b - r, 2), Fr(-b + r, 2)]
            extra.append(
                (solve_sign(lambda x: x * x + b * x + shown, wrong_roots, [], op), "w5qi-forgot-to-move-constant")
            )
    else:
        k = rng.randint(2, 12)
        f = lambda x: x * x - k * k  # noqa: E731
        zeros = [Fr(-k), Fr(k)]
        expr = f"x^2 {op} {k * k}"
        extra.append((_ray(op, k), "w5qi-positive-root-only"))
    solution, fields, cands = _quad_answer(f, zeros, op, extra)
    return dict(
        text=_stem(rng, expr),
        working=_sign_working(sorted(zeros), lead_sign, op, fields["answer_text"]),
        distractors=cands,
        **fields,
    )


def _hard_repeated(rng):
    p = _nz(rng, -8, 8)
    op = rng.choice((">", "<", "≥"))
    f = lambda x: (x - p) ** 2  # noqa: E731
    solution = solve_sign(f, [Fr(p)], [], op)
    verify(f, op, solution, [Fr(p)])
    fields = ineq_fields(solution)
    if rng.random() < 0.5:
        expr = f"({_lin(1, -p)})^2 {op} 0"
    else:
        expr = f"{join_terms([(1, 'x^2'), (-2 * p, 'x'), (p * p, '')])} {op} 0"
    other = _EMPTY if solution == _R else _R
    cands = ineq_cands(
        fields,
        [
            (solve_sign(f, [Fr(p)], [], _FLIP[op]), "w5qi-region-reversed"),
            (_ray(op if op != "≥" else ">", p) if op != "<" else _ray("<", p), "w5qi-one-factor-only"),
            (_R if op == ">" else _EMPTY, "w5qi-ignores-repeated-root"),
            (other, "w5qi-no-solution-confusion"),
        ],
    )
    enough(cands)
    work = [
        f"The left side is the square ({_lin(1, -p)})^2, which is 0 at x = {p} and positive everywhere else.",
        f"So the relation {op} 0 holds for: {fields['answer_text']}.",
    ]
    return dict(text=_stem(rng, expr), working=work, distractors=cands, **fields)


def _hard_no_roots(rng):
    h, k = rng.randint(-6, 6), rng.randint(1, 9)
    a = rng.choice((1, 2, 3)) * rng.choice((1, -1))
    A, B, C = a, -2 * a * h, a * (h * h + k)
    op = rng.choice(_OPS)
    f = lambda x: A * x * x + B * x + C  # noqa: E731
    disc = B * B - 4 * A * C
    check(disc < 0, "expected no real roots")
    solution = _R if _holds(Fr(A), op) else _EMPTY
    verify(f, op, solution, [Fr(h)])
    fields = ineq_fields(solution)
    fake = (
        sunion(sympy.Interval.open(-_OO, sym(h - 1)), sympy.Interval.open(sym(h + 1), _OO))
        if op in (">", "≥")
        else sympy.Interval.open(sym(h - 1), sym(h + 1))
    )
    cands = ineq_cands(
        fields,
        [
            (_EMPTY if solution == _R else _R, "w5qi-no-roots-wrong-sign"),
            (fake, "w5qi-invents-roots"),
            (_ray("<" if op in ("<", "≤") else ">", h), "w5qi-vertex-as-boundary"),
        ],
    )
    enough(cands)
    expr = f"{join_terms([(A, 'x^2'), (B, 'x'), (C, '')])} {op} 0"
    work = [
        f"Discriminant = {paren_negative(B)}^2 - 4({A})({C}) = {disc}, which is negative, so the parabola never reaches the x-axis.",
        f"It opens {'upward' if A > 0 else 'downward'}, so the expression is {'positive' if A > 0 else 'negative'} for every x.",
        f"The relation {op} 0 therefore holds for {'every x' if solution == _R else 'no x'}.",
    ]
    return dict(
        text=f"{phrase(rng, _LEAD, e=expr)} If every real number works write 'all real numbers'; if none does write "
        "'no solution'.",
        working=work,
        distractors=cands,
        **fields,
    )


def _hard_surd(rng):
    h = rng.randint(-5, 5)
    d = rng.choice((2, 3, 5, 6, 7, 10, 11, 13))
    a = rng.choice((1, 1, 2, 3))
    A, B, C = a, -2 * a * h, a * (h * h - d)
    op = rng.choice(_OPS)
    sq = sympy.sqrt(d)
    lo, hi = sym(h) - sq, sym(h) + sq
    f = lambda x: A * x * x + B * x + C  # noqa: E731
    if op in ("<", "≤"):
        solution = sympy.Interval(lo, hi, op == "<", op == "<")
    else:
        solution = sunion(sympy.Interval(-_OO, lo, True, op == ">"), sympy.Interval(hi, _OO, op == ">", True))
    verify(f, op, solution, [lo, hi])
    fields = ineq_fields(solution)

    def region(centre, spread):
        a_, b_ = centre - spread, centre + spread
        if op in ("<", "≤"):
            return sympy.Interval(a_, b_, op == "<", op == "<")
        return sunion(sympy.Interval(-_OO, a_, True, op == ">"), sympy.Interval(b_, _OO, op == ">", True))

    cands = ineq_cands(
        fields,
        [
            (region(sym(h), sym(d)), "w5qi-forgets-square-root"),
            (scomp(solution), "w5qi-region-reversed"),
            (region(sym(-h), sq), "w5qi-roots-negated"),
            (region(sym(h), sym(max(1, round(float(sq))))), "w5qi-rounds-surd-bounds"),
        ],
    )
    enough(cands)
    expr = f"{join_terms([(A, 'x^2'), (B, 'x'), (C, '')])} {op} 0"
    work = [
        f"Divide by {A} if needed and complete the square: (x {'-' if h > 0 else '+'} {abs(h)})^2 = {d} gives x = {h} ± √{d}."
        if h
        else f"x^2 = {d} gives x = ±√{d}.",
        f"The parabola opens upward, so the relation {op} 0 gives: {fields['answer_text']}.",
    ]
    return dict(
        text=f"{phrase(rng, _LEAD, e=expr)} Give exact bounds (type a square root as sqrt(7)). {rng.choice(_ASK)}",
        working=work,
        distractors=cands,
        **fields,
    )


_MODELS = (
    (
        "A ball is thrown upward from a platform. Its height in metres, t seconds after the throw (0 ≤ t ≤ {T}, until it lands), is h = {e}.",
        "h",
        "t",
        "metres",
    ),
    (
        "A stone is launched from a cliff. Its height above the sea in metres, after t seconds (0 ≤ t ≤ {T}, until it lands in the sea), is h = {e}.",
        "h",
        "t",
        "metres",
    ),
    ("A small business sells n items (n ≥ 0). Its profit in dollars is P = {e}.", "P", "n", "dollars"),
)


def _hard_model(rng):
    pat, out_var, in_var, unit = rng.choice(_MODELS)
    height = out_var == "h"
    if height:
        # model = -a(t - r1)(t - r2) + level, with the landing time T (an integer) beyond r1 + r2 so that the height at
        # t = 0 is positive: level = a(T - r1)(T - r2) makes the height zero at t = T.
        r1 = rng.randint(1, 5)
        r2 = r1 + rng.randint(3, 6)
        T = r1 + r2 + rng.randint(1, 3)
        a = rng.choice((1, 2))
        level = a * (T - r1) * (T - r2)
        domain = sympy.Interval(0, T)
    else:
        r1 = rng.randint(3, 8)
        r2 = r1 + rng.randint(3, 6)
        T = None
        a = rng.choice((1, 2, 3, 5))
        level = rng.choice((5, 10, 15, 20, 30, 40))
        domain = sympy.Interval(0, _OO)
    A, B, C = -a, a * (r1 + r2), -a * r1 * r2 + level  # model = -a(t - r1)(t - r2) + level
    above = rng.random() < 0.65
    op = ">" if above else "<"
    f = lambda t: -a * (t - r1) * (t - r2)  # noqa: E731  (the model minus the level)
    solution = sinter(solve_sign(f, [Fr(r1), Fr(r2)], [], op), domain)
    if height:
        check(C > 0 and A * T * T + B * T + C == 0, "the height at the start or the landing time is wrong")
        bounded = lambda t: f(t) if t <= T else None  # noqa: E731
    else:
        bounded = f
    verify(bounded, op, solution, [Fr(r1), Fr(r2)], extra=[Fr(0), Fr(1, 3)], floor=Fr(0))
    fields = ineq_fields(solution, in_var)
    cands = ineq_cands(
        fields,
        [
            (sinter(domain, scomp(solution)), "w5qi-region-reversed"),
            (
                sympy.Interval(sym(r1), sym(r2))
                if above
                else sunion(sympy.Interval(0, sym(r1)), sinter(sympy.Interval(sym(r2), _OO), domain)),
                "w5qi-strictness-swapped",
            ),
            (sinter(_ray(">", r2), domain) if above else sympy.Interval.open(0, sym(r1)), "w5qi-one-factor-only"),
        ],
        var=in_var,
    )
    enough(cands)
    expr = join_terms([(A, f"{in_var}^2"), (B, in_var), (C, "")])
    word = "more than" if above else "less than"
    limit = f"0 ≤ {in_var} ≤ {T}" if height else f"{in_var} ≥ 0"
    work = [
        f"Reaching {level} {unit} means {expr} = {level}, which is {-a}({in_var} - {r1})({in_var} - {r2}) = 0.",
        f"The roots are {in_var} = {r1} and {in_var} = {r2} and the parabola opens downward.",
        f"{out_var} is {word} {level} when: {fields['answer_text']} (with {limit}).",
    ]
    stem = (
        f"{pat.format(e=expr, T=T)} For which values of {in_var} is {out_var} {word} {level} {unit}? {rng.choice(_ASK)}"
    )
    return dict(text=stem, working=work, distractors=cands, **fields)


_QUAD_HARD = (_hard_repeated, _hard_no_roots, _hard_surd, _hard_model, _hard_surd, _hard_model)


@generator(
    QUAD,
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_quadratic_inequality(rng, tier="easy"):
    """Quadratic inequalities; the tiers are described in the module docstring."""
    pool = {"easy": (_quad_easy,), "medium": (_quad_medium,), "hard": _QUAD_HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# -- quadratic exam chain ---------------------------------------------------------------------------------------------


def _quad_exam(rng):
    for _ in attempts(500, "a quadratic with a wide enough gap"):
        p, q = sorted(rng.sample(range(-9, 10), 2))
        if q - p >= 4:
            break
    a = rng.choice((1, 1, 2, -1, -2))
    A, B, C = a, -a * (p + q), a * p * q
    fn = lambda x: A * x * x + B * x + C  # noqa: E731
    shown = join_terms([(A, "x^2"), (B, "x"), (C, "")])
    v = Fr(p + q, 2)
    # (a) turning point
    v_fields = number_fields(v)
    v_dist = number_cands(
        v,
        [
            (Fr(p + q), "w5qi-vertex-sum-not-half"),
            (Fr(q - p, 2), "w5qi-vertex-half-gap"),
            (-v, "w5qi-vertex-sign-flipped"),
            (Fr(q), "w5qi-root-for-turning-point"),
        ],
    )
    # (b) a solution set
    op_b = rng.choice(_OPS)
    sol_b = solve_sign(fn, [Fr(p), Fr(q)], [], op_b)
    verify(fn, op_b, sol_b, [Fr(p), Fr(q)])
    fields_b = ineq_fields(sol_b)
    extra = [(solve_sign(lambda x: -fn(x), [Fr(p), Fr(q)], [], op_b), "w5qi-ignores-negative-leading")] if a < 0 else []
    dist_b = ineq_cands(fields_b, _quad_slips(fn, [Fr(p), Fr(q)], op_b, extra))
    # (c) whole numbers between the roots
    finite_op = rng.choice(("<", "≤")) if a > 0 else rng.choice((">", "≥"))
    strict = finite_op in ("<", ">")
    count = sum(1 for x in range(p - 2, q + 3) if _holds(Fr(fn(x)), finite_op))
    check(count == (q - p - 1 if strict else q - p + 1), "integer count differs")
    c_fields = number_fields(count, fraction=True)
    c_dist = number_cands(
        count,
        [
            (q - p + 1 if strict else q - p - 1, "integer-solutions-endpoint-miscount"),
            (q - p, "integer-solutions-endpoint-miscount"),
            (q - p + 2 if strict else q - p - 2, "integer-solutions-endpoint-miscount"),
            (q + p, "w5qi-count-sums-roots"),
        ],
    )
    # (d) greatest or least value
    extreme = fn(v)
    check(extreme == -A * (v - p) ** 2, "extreme value differs")
    d_fields = number_fields(extreme, fraction=True)
    d_dist = number_cands(
        extreme,
        [
            (C, "w5qi-intercept-not-extreme"),
            (-extreme, "w5qi-extreme-sign-flipped"),
            (v, "w5qi-gives-position-not-value"),
            (A * v * v, "w5qi-extreme-forgets-linear-term"),
        ],
    )
    for d in (v_dist, dist_b, c_dist, d_dist):
        enough(d)
    word = "least" if a > 0 else "greatest"
    sign_word = "positive" if a > 0 else "negative"
    del sign_word
    return dict(
        text=f"Let f(x) = {shown}, which has two real roots.",
        parts=[
            part(
                "Find the x-coordinate of the turning point of y = f(x).",
                v_fields["answer_kind"],
                {k: val for k, val in v_fields.items() if k != "answer_kind"},
                1,
                [
                    f"The turning point is halfway between the roots {p} and {q}: x = ({p} + {paren_negative(q)})/2 = {rt(v)}."
                ],
                v_dist,
            ),
            part(
                f"Solve f(x) {op_b} 0. Give the solution set using inequalities (join pieces with 'or') or intervals.",
                "inequality",
                {k: val for k, val in fields_b.items() if k != "answer_kind"},
                2,
                _sign_working([p, q], a, op_b, fields_b["answer_text"]),
                dist_b,
            ),
            part(
                f"How many integers x satisfy f(x) {finite_op} 0?",
                c_fields["answer_kind"],
                {k: val for k, val in c_fields.items() if k != "answer_kind"},
                1,
                [f"The solution set runs from x = {p} to x = {q}; the integers in it number {count}."],
                c_dist,
                follow_through=f"2*(part_a - ({p})) {'-' if strict else '+'} 1",
                sensible_range=(0, None),
            ),
            part(
                f"Find the {word} value of f(x).",
                d_fields["answer_kind"],
                {k: val for k, val in d_fields.items() if k != "answer_kind"},
                1,
                [f"Substitute x = {rt(v)}: f({rt(v)}) = {rt(extreme)}."],
                d_dist,
                follow_through=f"-({A})*(part_a - ({p}))**2",
            ),
        ],
    )


@generator(QUAD_EXAM, level=20, presets={"exam": dict(tier="exam")})
def exd_x_w5_quadratic_inequality(rng, tier="exam"):
    """Exam chain: turning point, solution set, whole numbers satisfying a closed region, and the extreme value."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _quad_exam)


# -- rational inequalities --------------------------------------------------------------------------------------------


def _ratl_slips(g, zeros, poles, op, num, den, extra=()):
    """Tagged wrong sets for the rational statement g(x) op 0, where `num` and `den` evaluate the pieces."""
    out = [
        (solve_sign(g, zeros, poles, _FLIP[op]), "w5ri-region-reversed"),
        (solve_sign(g, zeros, poles, _STRICTNESS[op]), "w5ri-zero-strictness"),
    ]
    out += list(extra)
    # multiplies both sides by the denominator without asking whether it is negative: only the numerator is left
    out.append((solve_sign(num, zeros, [], op), "w5ri-multiplied-by-denominator"))
    if op in ("≤", "≥") and poles:
        out.append(
            (
                sunion(solve_sign(g, zeros, poles, op), *[sympy.FiniteSet(sym(Fr(p))) for p in poles]),
                "w5ri-pole-included",
            )
        )
    out.append((solve_sign(lambda x: g(-x), [-z for z in zeros], [-p for p in poles], op), "w5ri-roots-negated"))
    return out


def _ratl_answer(g, zeros, poles, op, num, den, stmt=None, extra=()):
    zeros = sorted(Fr(z) for z in zeros)
    poles = sorted(Fr(p) for p in poles)
    solution = solve_sign(g, zeros, poles, op)
    verify(stmt or g, op, solution, zeros + poles)
    fields = ineq_fields(solution)
    cands = ineq_cands(fields, _ratl_slips(g, zeros, poles, op, num, den, extra))
    enough(cands)
    return solution, fields, cands


def _ratl_working(zeros, poles, op, answer):
    z = " and ".join(f"x = {rt(v)}" for v in zeros) or "none"
    pl = " and ".join(f"x = {rt(v)}" for v in poles)
    return [
        f"Critical values: the numerator is zero at {z}; the denominator is zero at {pl} (never allowed).",
        "Test the sign of the fraction in each region between them (a sign diagram).",
        f"Keep the regions where the fraction {op} 0: {answer}.",
    ]


def _ratl_easy(rng):
    a, b = rng.sample(range(-8, 9), 2)
    op = rng.choice(_OPS)
    num = lambda x: x - a  # noqa: E731
    den = lambda x: x - b  # noqa: E731
    g = lambda x: None if den(x) == 0 else Fr(num(x)) / den(x)  # noqa: E731
    solution, fields, cands = _ratl_answer(g, [a], [b], op, num, den)
    expr = f"{_frac(_lin(1, -a), _lin(1, -b))} {op} 0"
    return dict(
        text=_stem(rng, expr), working=_ratl_working([a], [b], op, fields["answer_text"]), distractors=cands, **fields
    )


def _ratl_medium(rng):
    kind = rng.choice(("coeff", "against_k", "against_k"))
    op = rng.choice(_OPS)
    for _ in attempts(500, "distinct zero and pole"):
        alpha = rng.choice((1, 1, 2, -1, -2, 3, -3))
        r, s = rng.sample(range(-7, 8), 2)
        gamma = rng.choice((1, 1, -1, 2, -2, 3))
        if Fr(r) != Fr(s):
            break
    A = (alpha, -alpha * r)  # alpha (x - r)
    D = (gamma, -gamma * s)  # gamma (x - s)
    if kind == "coeff":
        expr = f"{_frac(_lin(*A), _lin(*D))} {op} 0"
        stmt = lambda x: None if D[0] * x + D[1] == 0 else Fr(A[0] * x + A[1]) / (D[0] * x + D[1])  # noqa: E731
        k = 0
    else:
        k = rng.choice((-3, -2, -1, 1, 2, 3, 4))
        N = (A[0] + k * D[0], A[1] + k * D[1])
        if N[0] == 0 and N[1] == 0:
            raise TooFewDistractors("degenerate draw")
        num_text = _lin(*N) if N[0] else str(N[1])
        expr = f"{_frac(num_text, _lin(*D))} {op} {k}"
        stmt = lambda x: None if D[0] * x + D[1] == 0 else Fr(N[0] * x + N[1]) / (D[0] * x + D[1]) - k  # noqa: E731
    g = lambda x: None if D[0] * x + D[1] == 0 else Fr(A[0] * x + A[1]) / (D[0] * x + D[1])  # noqa: E731
    num = lambda x: A[0] * x + A[1]  # noqa: E731
    den = lambda x: D[0] * x + D[1]  # noqa: E731
    extra = []
    if k and N[0]:
        # reads the fraction against 0 although the inequality compares it with k
        extra.append(
            (
                solve_sign(lambda x: None if den(x) == 0 else Fr(N[0] * x + N[1]) / den(x), [Fr(-N[1], N[0])], [s], op),
                "w5ri-compares-with-zero-not-k",
            )
        )
    solution, fields, cands = _ratl_answer(g, [r], [s], op, num, den, stmt=stmt, extra=extra)
    return dict(
        text=_stem(rng, expr),
        working=_ratl_working([r], [s], op, fields["answer_text"])
        if not k
        else [
            f"Subtract {k}: write the inequality as one fraction compared with 0.",
            *_ratl_working([r], [s], op, fields["answer_text"]),
        ],
        distractors=cands,
        **fields,
    )


def _hard_three(rng):
    shape = rng.choice(("two_over_one", "one_over_two", "square_over_one", "two_over_one", "one_over_two"))
    op = rng.choice(_OPS)
    if shape == "two_over_one":
        a, b, c = rng.sample(range(-7, 8), 3)
        zeros, poles = [a, b], [c]
        num = lambda x: (x - a) * (x - b)  # noqa: E731
        den = lambda x: x - c  # noqa: E731
        expr = f"{_frac(f'({_lin(1, -a)})({_lin(1, -b)})', _lin(1, -c))} {op} 0"
    elif shape == "one_over_two":
        a, b, c = rng.sample(range(-7, 8), 3)
        zeros, poles = [a], [b, c]
        num = lambda x: x - a  # noqa: E731
        den = lambda x: (x - b) * (x - c)  # noqa: E731
        expr = f"{_frac(_lin(1, -a), f'({_lin(1, -b)})({_lin(1, -c)})')} {op} 0"
    else:
        a, b = rng.sample(range(-7, 8), 2)
        zeros, poles = [a], [b]
        num = lambda x: (x - a) ** 2  # noqa: E731
        den = lambda x: x - b  # noqa: E731
        expr = f"{_frac(f'({_lin(1, -a)})^2', _lin(1, -b))} {op} 0"
    g = lambda x: None if den(x) == 0 else Fr(num(x)) / den(x)  # noqa: E731
    extra = []
    if shape == "square_over_one":
        # the squared numerator never changes the sign of the fraction, so the only sign change is at the pole
        extra.append(
            (
                solve_sign(lambda x: None if x == b else Fr(x - a) / (x - b), [Fr(a)], [Fr(b)], op),
                "w5ri-square-treated-as-sign-change",
            )
        )
    solution, fields, cands = _ratl_answer(g, zeros, poles, op, num, den, extra=extra)
    return dict(
        text=_stem(rng, expr),
        working=_ratl_working(sorted(zeros), sorted(poles), op, fields["answer_text"]),
        distractors=cands,
        **fields,
    )


@generator(
    RATL,
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_rational_inequality(rng, tier="easy"):
    """Rational inequalities; the tiers are described in the module docstring."""
    pool = {"easy": (_ratl_easy,), "medium": (_ratl_medium,), "hard": (_hard_three,)}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# -- rational exam chain ----------------------------------------------------------------------------------------------


def _ratl_exam(rng):
    for _ in attempts(500, "a zero, a pole and a usable level"):
        a, b = rng.sample(range(-8, 9), 2)
        k = rng.choice((-3, -2, -1, 2, 3))
        if k != 1:
            break
    g = lambda x: None if x == b else Fr(x - a) / (x - b)  # noqa: E731
    shown = f"g(x) = {_frac(_lin(1, -a), _lin(1, -b))}"
    # (a) the value g is undefined at
    a_fields = number_fields(b)
    a_dist = number_cands(b, [(a, "w5ri-zero-for-pole"), (-b, "w5ri-pole-sign-flipped"), (0, "w5ri-pole-zero-guess")])
    # (b) g(x) op 0
    op_b = rng.choice(_OPS)
    num = lambda x: x - a  # noqa: E731
    den = lambda x: x - b  # noqa: E731
    sol_b = solve_sign(g, [Fr(a)], [Fr(b)], op_b)
    verify(g, op_b, sol_b, [Fr(a), Fr(b)])
    fields_b = ineq_fields(sol_b)
    dist_b = ineq_cands(fields_b, _ratl_slips(g, [Fr(a)], [Fr(b)], op_b, num, den))
    # (c) g(x) op k : one fraction, (1 - k)x + (kb - a) over (x - b)
    op_c = rng.choice(_OPS)
    h = lambda x: None if x == b else Fr(x - a) / (x - b) - k  # noqa: E731
    zero_c = Fr(a - k * b, 1 - k)
    check(h(zero_c) == 0, "level crossing differs")
    sol_c = solve_sign(h, [zero_c], [Fr(b)], op_c)
    verify(h, op_c, sol_c, [zero_c, Fr(b)])
    fields_c = ineq_fields(sol_c)
    num_c = lambda x: (1 - k) * x + (k * b - a)  # noqa: E731
    extra_c = [
        (solve_sign(g, [Fr(a)], [Fr(b)], op_c), "w5ri-compares-with-zero-not-k"),
    ]
    dist_c = ineq_cands(fields_c, _ratl_slips(h, [zero_c], [Fr(b)], op_c, num_c, den, extra_c))
    # (d) whole numbers from -10 to 10 satisfying (c)
    count = sum(1 for x in range(-10, 11) if x != b and _holds(h(x), op_c))
    wrong_cross = sum(1 for x in range(-10, 11) if _holds(Fr(num_c(x)), op_c))
    wrong_comp = sum(1 for x in range(-10, 11) if x != b and not _holds(h(x), op_c))
    d_fields = number_fields(count)
    d_dist = number_cands(
        count,
        [
            (wrong_cross, "w5ri-multiplied-by-denominator"),
            (wrong_comp, "w5ri-region-reversed"),
            (21 - count, "w5ri-counts-the-pole"),
            (count + 1, "integer-solutions-endpoint-miscount"),
            (count - 1, "integer-solutions-endpoint-miscount"),
        ],
    )
    for d in (a_dist, dist_b, dist_c, d_dist):
        enough(d)
    return dict(
        text=f"Let {shown}.",
        parts=[
            part(
                "For which value of x is g(x) undefined?",
                a_fields["answer_kind"],
                {kk: v for kk, v in a_fields.items() if kk != "answer_kind"},
                1,
                [
                    f"The denominator x - {b} is zero when x = {b}."
                    if b > 0
                    else f"The denominator is zero when x = {b}."
                ],
                a_dist,
            ),
            part(
                f"Solve g(x) {op_b} 0. Give the solution set using inequalities (join pieces with 'or') or intervals.",
                "inequality",
                {kk: v for kk, v in fields_b.items() if kk != "answer_kind"},
                2,
                _ratl_working([a], [b], op_b, fields_b["answer_text"]),
                dist_b,
            ),
            part(
                f"Solve g(x) {op_c} {k}. Give the solution set using inequalities or intervals.",
                "inequality",
                {kk: v for kk, v in fields_c.items() if kk != "answer_kind"},
                3,
                [
                    f"Subtract {k} and combine: g(x) - ({k}) = ({_lin(1 - k, k * b - a)})/({_lin(1, -b)}).",
                    *_ratl_working([zero_c], [b], op_c, fields_c["answer_text"]),
                ],
                dist_c,
            ),
            part(
                f"How many integers from -10 to 10 inclusive satisfy g(x) {op_c} {k}?",
                d_fields["answer_kind"],
                {kk: v for kk, v in d_fields.items() if kk != "answer_kind"},
                2,
                [f"Count the integers in {fields_c['answer_text']} within the window: {count}."],
                d_dist,
            ),
        ],
    )


@generator(RATL_EXAM, level=20, presets={"exam": dict(tier="exam")})
def exd_x_w5_rational_inequality(rng, tier="exam"):
    """Exam chain: the excluded value, two solution sets and the whole numbers satisfying the second."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _ratl_exam)
