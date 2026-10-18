"""Generates `r3g_absolute_value` questions (easy, medium, hard) and the exam chain `r3g_exam_absolute_value`.

Skill ``absolute-value-equations-inequalities`` (``inequalities``, answer ``set``): equations and
inequalities with the absolute value (modulus) of a linear expression.

* easy: ``|ax + b| = c`` (or ``|ax + b| + d = e``) as a set of two solutions.
* medium: ``|ax + b| < c``, ``≤``, ``>`` or ``≥`` (also with a multiple and a constant around the modulus); the answer
  is the set of integers from -6 to 6 that satisfy it, which is an interval for ``<`` and two rays for ``>``.
* hard: a two-modulus equation ``|ax + b| = |cx + d|``; ``|ax + b| = cx + d``, where a candidate may have to be rejected
  because ``cx + d`` is negative there; or the integers from -6 to 6 satisfying ``|ax + b| < cx + d`` (or ``>``).
* exam (``r3g_exam_absolute_value``): a line meets the V-shaped graph y = |ax + e|. (a) The smaller and (b) the larger
  x-coordinate of the intersections, (c) the distance between them (follows through from (a) and (b)) and (d) the values
  of x for which the V lies below the line.

Solutions are found by cases and confirmed by testing sample points directly in the stated equation or inequality; the
intersections of the exam chain are confirmed against the roots found by SymPy (squared, keeping mx + d >= 0).
"""

from __future__ import annotations

import math

import sympy as sp

from ..answers import Distractor
from ..core import attempts, check, generator
from ..grading.inequality import set_text
from ._r3g_kit import (
    TooFewDistractors,
    ex,
    in_real_set,
    latex,
    pick,
    retry_short,
    typed,
    value_set_dist,
    value_set_fields,
)

R = sp.Rational
X = sp.Symbol("x", real=True)


def lin(a, b, var="x") -> str:
    """a x + b as written: 2x - 3, -2x + 5, x + 4, 5 - x."""
    head = {1: var, -1: f"-{var}"}.get(a, f"{a}{var}")
    if b == 0:
        return head
    return f"{head} {'+' if b > 0 else '-'} {abs(b)}"


def _ineq_fields(solution, var="x"):
    text = set_text(solution, var)
    latex_text = (
        text.replace("≤", r"\le ").replace("≥", r"\ge ").replace(" or ", r" \text{ or } ").replace("∞", r"\infty")
    )
    return dict(answer_kind="inequality", answer=text, answer_text=text, answer_latex=latex_text)


def _ineq_dist(solution, tag):
    text = set_text(solution, "x")
    return Distractor(text, tag, text)


WINDOW = 6


def _positive(coef, const):
    """{x : coef x + const > 0} as a real set."""
    coef, const = sp.Rational(coef), sp.Rational(const)
    if coef == 0:
        return sp.S.Reals if const > 0 else sp.S.EmptySet
    edge = -const / coef
    return sp.Interval.open(edge, sp.oo) if coef > 0 else sp.Interval.open(-sp.oo, edge)


def _modulus_vs_linear(a, b, c, d, op):
    """The solution of |ax + b| < cx + d (op "<") or > cx + d (op ">"), by splitting the modulus into its two branches."""
    if op == "<":
        return sp.Intersection(_positive(c - a, d - b), _positive(c + a, d + b))
    return sp.Union(_positive(a - c, b - d), _positive(-a - c, -b - d))


def _members(region) -> list[int]:
    """The integers x with -6 <= x <= 6 that lie in the real set `region`."""
    return [n for n in range(-WINDOW, WINDOW + 1) if in_real_set(region, sp.Integer(n))]


def _region_dist(region, tag):
    return value_set_dist(_members(region), tag)


def _window_text() -> str:
    return f"-{WINDOW} ≤ x ≤ {WINDOW}"


def _easy(rng):
    for _ in attempts(300, "a modulus equation with distinct, not-both-integer-free solutions"):
        a = rng.choice((1, 2, 3, 4, 5, -2, -3))
        b = rng.choice([v for v in range(-9, 10) if v != 0])
        c = rng.randint(1, 12)
        sols = sorted({R(-b + c, a), R(-b - c, a)})
        if len(sols) == 2:
            break
    shifted = rng.random() < 0.4
    d = rng.randint(1, 6) if shifted else 0
    stem_rhs = c + d
    for s in sols:
        check(abs(a * s + b) == c, f"{s} does not satisfy |{a}x + {b}| = {c}")
    # no other solution: the function |ax + b| - c has a sign change only at these points (checked by cases)
    check(set(sp.roots(sp.Poly((a * X + b) ** 2 - c**2, X))) == set(sols), "squaring gives different solutions")
    fields = value_set_fields(sols)
    other = lambda s: value_set_dist(s, "")  # noqa: E731
    cands = [
        value_set_dist([R(-b + c, a)], "r3g-abs-second-case-missed"),
        value_set_dist([R(-b - c, a)], "r3g-abs-second-case-missed"),
        value_set_dist([R(b + c, a), R(b - c, a)], "r3g-abs-sign-of-constant"),
        value_set_dist([R(c - b, 1), R(-c - b, 1)], "r3g-abs-coefficient-not-divided"),
        value_set_dist([R(c, a) - b, R(-c, a) - b], "r3g-abs-constant-moved-before-dividing"),
    ]
    del other
    if shifted:
        stem = f"Solve |{lin(a, b)}| + {d} = {stem_rhs}."
    else:
        stem = f"Solve |{lin(a, b)}| = {c}."
    return dict(
        text=stem + " Give the solutions separated by commas, for example -3, 7/2.",
        working=([f"|{lin(a, b)}| = {c}."] if shifted else [])
        + [f"{lin(a, b)} = {c} or {lin(a, b)} = {-c}.", "The solutions are " + fields["answer_text"] + "."],
        distractors=pick(fields["answer_text"], cands, 3),
        **fields,
    )


_RELS = {"<": sp.Lt, "≤": sp.Le, ">": sp.Gt, "≥": sp.Ge}
_TEST = {"<": lambda v, w: v < w, "≤": lambda v, w: v <= w, ">": lambda v, w: v > w, "≥": lambda v, w: v >= w}


def _sample_points(*critical):
    pts = set()
    for c in critical:
        for d in (R(-1, 2), R(0), R(1, 2), R(-3), R(3)):
            pts.add(R(c) + d)
    return sorted(pts)


def _medium(rng):
    for _ in attempts(300, "a modulus inequality with distinct boundaries"):
        a = rng.choice((1, 2, 3, 4, -1, -2, -3))
        b = rng.choice([v for v in range(-9, 10) if v != 0])
        c = rng.randint(1, 10)
        k = rng.choice((1, 1, 2, 3))
        d = rng.choice((0, 0, 1, -1, 2, -2, 3))
        op = rng.choice(("<", "≤", ">", "≥"))
        lo, hi = sorted((R(-b - c, a), R(-b + c, a)))
        if not (-WINDOW + 1 < lo < hi < WINDOW - 1):
            continue
        k_inside = len(_members(sp.Interval(lo, hi)))
        count = k_inside if op in ("<", "≤") else 2 * WINDOW + 1 - k_inside
        if 3 <= count <= 9:
            break
    else:
        raise TooFewDistractors("no draw with a usable integer count")
    # k |ax+b| + d  op  k c + d
    rhs = k * c + d
    inner = f"|{lin(a, b)}|"
    left = inner if k == 1 else f"{k}{inner}"
    if d:
        left += f" {'+' if d > 0 else '-'} {abs(d)}"
    solution = {
        "<": sp.Interval.open(lo, hi),
        "≤": sp.Interval(lo, hi),
        ">": sp.Union(sp.Interval.open(-sp.oo, lo), sp.Interval.open(hi, sp.oo)),
        "≥": sp.Union(sp.Interval(-sp.oo, lo), sp.Interval(hi, sp.oo)),
    }[op]
    for x in _sample_points(lo, hi):
        holds = _TEST[op](k * abs(a * x + b) + d, rhs)
        check(bool(holds) == in_real_set(solution, x), f"x = {x}: stem {holds} but set says {in_real_set(solution, x)}")
    closed = op in ("≤", "≥")
    between = op in ("<", "≤")
    reverse = (
        sp.Union(sp.Interval(-sp.oo, lo, True, not closed), sp.Interval(hi, sp.oo, not closed, True))
        if between
        else sp.Interval(lo, hi, not closed, not closed)
    )
    sign_slip = sorted((R(b - c, a), R(b + c, a)))
    cands = [
        (reverse, "r3g-abs-region-reversed"),
        (
            sp.Interval(sign_slip[0], sign_slip[1], not closed, not closed)
            if between
            else sp.Union(
                sp.Interval(-sp.oo, sign_slip[0], True, not closed), sp.Interval(sign_slip[1], sp.oo, not closed, True)
            ),
            "r3g-abs-sign-of-constant",
        ),
        (
            (sp.Interval(-sp.oo, hi, True, not closed) if between else sp.Interval(lo, sp.oo, not closed, True)),
            "r3g-abs-one-bound-only",
        ),
        (
            (
                sp.Interval(lo, hi, closed is False, closed is False)
                if between
                else sp.Union(sp.Interval(-sp.oo, lo, True, closed), sp.Interval(hi, sp.oo, closed, True))
            ),
            "r3g-abs-equality-included",
        ),
    ]
    if a < 0:
        cands.append(
            (
                sp.Interval(hi, lo, not closed, not closed)
                if between and False
                else sp.Union(sp.Interval(-sp.oo, hi, True, not closed), sp.Interval(lo, sp.oo, not closed, True))
                if False
                else reverse,
                "r3g-abs-region-reversed",
            )
        )
    fields = value_set_fields(_members(solution))
    steps = []
    if k != 1 or d:
        steps.append(f"Isolate the modulus: |{lin(a, b)}| {op} {c}.")
    steps.append(
        f"{lin(a, b)} {'between' if between else 'outside'} -{c} and {c}, so the boundary values are x = {typed(lo)} and x = {typed(hi)}."
    )
    steps.append(
        f"The solution is {set_text(solution, 'x')}, so the integers from {-WINDOW} to {WINDOW} are {fields['answer_text']}."
    )
    return dict(
        text=(
            f"Find all the integers x with {_window_text()} that satisfy {left} {op} {rhs}. "
            "Give them separated by commas, for example -3, 0, 4."
        ),
        working=steps,
        distractors=pick(
            fields["answer_text"], [_region_dist(s, t) for s, t in cands if _members(s) != _members(solution)], 3
        ),
        **fields,
    )


def _two_modulus(rng):
    for _ in attempts(300, "a two-modulus equation with distinct solutions"):
        a, c = rng.choice((1, 2, 3, 4)), rng.choice((1, 2, 3, -1, -2, -3, 4))
        b, d = rng.randint(-8, 8), rng.randint(-8, 8)
        if abs(a) == abs(c) or b == d or b == 0 or d == 0:
            continue
        s1, s2 = R(d - b, a - c), R(-b - d, a + c)
        if s1 != s2:
            break
    sols = sorted({s1, s2})
    for s in sols:
        check(abs(a * s + b) == abs(c * s + d), f"{s} fails |{a}x + {b}| = |{c}x + {d}|")
    check(
        set(sp.roots(sp.Poly((a * X + b) ** 2 - (c * X + d) ** 2, X))) == set(sols),
        "squaring gives different solutions",
    )
    fields = value_set_fields(sols)
    cands = [
        value_set_dist([s1], "r3g-abs-second-case-missed"),
        value_set_dist([s2], "r3g-abs-second-case-missed"),
        value_set_dist([R(b + d, a + c), R(b - d, a - c)], "r3g-abs-sign-of-constant"),
        value_set_dist([R(d - b, a + c), R(b + d, a - c)], "r3g-abs-negative-case-sign"),
    ]
    return dict(
        text=f"Solve |{lin(a, b)}| = |{lin(c, d)}|. Give the solutions separated by commas, for example -3, 7/2.",
        working=[
            f"Either {lin(a, b)} = {lin(c, d)} or {lin(a, b)} = -({lin(c, d)}).",
            f"The first gives x = {typed(s1)} and the second gives x = {typed(s2)}.",
        ],
        distractors=pick(fields["answer_text"], cands, 3),
        **fields,
    )


def _modulus_equals_linear(rng):
    for _ in attempts(500, "a modulus equation with one candidate rejected"):
        a, c = rng.choice((1, 2, 3, -1, -2)), rng.choice((1, 2, -1, -2, 3))
        b, d = rng.randint(-9, 9), rng.randint(-9, 9)
        if a == c or a == -c or b == 0:
            continue
        s1, s2 = R(d - b, a - c), R(-b - d, a + c)
        v1, v2 = c * s1 + d, c * s2 + d
        valid = sorted({s for s, v in ((s1, v1), (s2, v2)) if v >= 0})
        if len(valid) == 1 and s1 != s2:
            break
    for s in valid:
        check(abs(a * s + b) == c * s + d, f"{s} fails |{a}x + {b}| = {c}x + {d}")
    check(
        {r for r in sp.roots(sp.Poly((a * X + b) ** 2 - (c * X + d) ** 2, X)) if c * r + d >= 0} == set(valid),
        "squaring gives different solutions",
    )
    rejected = [s for s in (s1, s2) if s not in valid][0]
    fields = value_set_fields(valid)
    cands = [
        value_set_dist([s1, s2], "r3g-abs-extraneous-root-kept"),
        value_set_dist([rejected], "r3g-abs-valid-root-rejected"),
        value_set_dist([R(d - b, a + c)], "r3g-abs-negative-case-sign"),
        value_set_dist([-valid[0]], "r3g-abs-sign-of-constant"),
    ]
    return dict(
        text=f"Solve |{lin(a, b)}| = {lin(c, d)}. Give the solutions separated by commas, for example -3, 7/2.",
        working=[
            f"{lin(a, b)} = {lin(c, d)} gives x = {typed(s1)}; {lin(a, b)} = -({lin(c, d)}) gives x = {typed(s2)}.",
            f"The modulus cannot equal a negative number, so check {lin(c, d)} ≥ 0: x = {typed(rejected)} does not satisfy it.",
            f"The only solution is x = {typed(valid[0])}.",
        ],
        distractors=pick(fields["answer_text"], cands, 3),
        **fields,
    )


def _modulus_less_linear(rng):
    for _ in attempts(500, "an inequality with a modulus on the small side"):
        a, c = rng.choice((1, 2, 3, -1, -2)), rng.choice((1, 2, 3))
        b, d = rng.randint(-8, 8), rng.randint(-8, 8)
        if abs(a) == abs(c) or b == 0:
            continue
        s1, s2 = R(d - b, a - c), R(-b - d, a + c)
        if s1 == s2:
            continue
        op = rng.choice(("<", ">"))
        solution = _modulus_vs_linear(a, b, c, d, op)
        count = len(_members(solution))
        if (
            solution not in (sp.S.Reals, sp.S.EmptySet)
            and 2 <= count <= 9
            and -WINDOW + 1 < min(s1, s2)
            and max(s1, s2) < WINDOW - 1
        ):
            break
    else:
        raise TooFewDistractors("no draw with a usable integer count")
    for x in _sample_points(s1, s2):
        holds = _TEST[op](abs(a * x + b), c * x + d)
        check(bool(holds) == in_real_set(solution, x), f"x = {x}: stem {holds} but set says {in_real_set(solution, x)}")
    lo, hi = sorted((s1, s2))
    cands = [
        (
            sp.Interval.open(lo, hi)
            if op == "<"
            else sp.Union(sp.Interval.open(-sp.oo, lo), sp.Interval.open(hi, sp.oo)),
            "r3g-abs-region-reversed",
        ),
        (sp.Interval.open(-sp.oo, hi), "r3g-abs-one-bound-only"),
        (sp.Interval.open(lo, sp.oo), "r3g-abs-one-bound-only"),
        (sp.Complement(sp.S.Reals, solution), "r3g-abs-region-reversed"),
    ]
    fields = value_set_fields(_members(solution))
    return dict(
        text=(
            f"Find all the integers x with {_window_text()} that satisfy |{lin(a, b)}| {op} {lin(c, d)}. "
            "Give them separated by commas, for example -3, 0, 4."
        ),
        working=[
            f"Solve |{lin(a, b)}| = {lin(c, d)}: x = {typed(s1)} or x = {typed(s2)}, then test a point in each interval (the right side must also be positive for '<').",
            f"The solution is {set_text(solution, 'x')}, so the integers from {-WINDOW} to {WINDOW} are {fields['answer_text']}.",
        ],
        distractors=pick(
            fields["answer_text"],
            [_region_dist(s, t) for s, t in cands if s != solution and _members(s) != _members(solution)],
            3,
        ),
        **fields,
    )


def _hard(rng):
    return rng.choice(
        [_two_modulus, _modulus_equals_linear, _modulus_equals_linear, _modulus_less_linear, _modulus_less_linear]
    )(rng)


@generator(
    "r3g_absolute_value",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def r3g_absolute_value(rng, tier="easy"):
    """Absolute value equations and inequalities; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("r3g_exam_absolute_value", level=12, presets={"exam": dict(tier="exam")})
@retry_short
def r3g_exam_absolute_value(rng, tier="exam"):
    """Exam chain: where a line meets the V-shaped graph y = |ax + e|, the distance between the points and where the V is below."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a line meeting a V-shaped modulus graph at two points"):
        a = rng.choice((2, 3, 4))
        m = rng.choice([v for v in (-3, -2, -1, 1, 2, 3) if abs(v) < a])
        r1, r2 = rng.randint(-6, 2), rng.randint(-1, 7)
        if r1 >= r2 or (r2 - r1) < 2:
            continue
        # left root on the branch -(ax + e) = mx + d, right root on ax + e = mx + d
        two_d, two_e = r2 * (a - m) - r1 * (a + m), -r1 * (a + m) - r2 * (a - m)
        if two_d % 2 or two_e % 2:
            continue
        d, e = two_d // 2, two_e // 2
        if e == 0 or d == 0 or a * r1 + e >= 0 or a * r2 + e <= 0:
            continue
        break
    # the roots of |ax + e| = mx + d, found independently by squaring and keeping those with mx + d >= 0
    xs = sorted(r for r in sp.roots(sp.Poly((a * X + e) ** 2 - (m * X + d) ** 2, X)) if m * r + d >= 0)
    check(xs == [r1, r2], "intersections differ from SymPy")
    dist = (r2 - r1) * sp.sqrt(1 + m * m)
    p1 = (r1, m * r1 + d)
    p2 = (r2, m * r2 + d)
    check(abs(a * r1 + e) == p1[1] and abs(a * r2 + e) == p2[1], "a point is not on the modulus graph")
    check(sp.simplify(sp.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) - dist) == 0, "distance differs")
    solution = sp.Interval.open(r1, r2)
    for x in (r1 - 1, r1, R(r1 + r2, 2), r2, r2 + 1, R(r1 + r2, 2) + R(1, 3)):
        below = abs(a * x + e) < m * x + d
        check(below == (x in solution), "the V is not below the line exactly between the intersections")
    fields = _ineq_fields(solution)

    def part(prompt, value, marks, working, cands, follow=None):
        text = typed(value)
        dct = dict(
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
            dct["follow_through"] = follow
        return dct

    graph = f"y = |{lin(a, e)}|"
    line = f"y = {lin(m, d)}"
    return dict(
        text=f"The graph {graph} and the line {line} intersect at two points.",
        parts=[
            part(
                "Find the x-coordinate of the left-hand point of intersection.",
                r1,
                2,
                [
                    f"Left of the vertex {lin(a, e)} is negative, so |{lin(a, e)}| = -({lin(a, e)}). "
                    f"Then -({lin(a, e)}) = {lin(m, d)} gives x = {r1}."
                ],
                [
                    (-r1, "r3g-abs-sign-of-constant"),
                    (r2, "r3g-abs-wrong-root"),
                    (R(e - d, a + m), "r3g-abs-negative-case-sign"),
                    (R(d - e, a - m) if r1 != R(d - e, a - m) else r1 + 1, "r3g-abs-second-case-missed"),
                ],
            ),
            part(
                "Find the x-coordinate of the right-hand point of intersection.",
                r2,
                1,
                [f"Right of the vertex, {lin(a, e)} = {lin(m, d)} gives x = {r2}."],
                [
                    (-r2, "r3g-abs-sign-of-constant"),
                    (r1, "r3g-abs-wrong-root"),
                    (r1 + r2, "r3g-abs-roots-added"),
                    (r1 * r2, "r3g-abs-roots-added"),
                ],
            ),
            part(
                "Find the distance between the two points of intersection. Give it exactly, for example 3sqrt(10).",
                dist,
                2,
                [
                    f"The points are ({r1}, {typed(p1[1])}) and ({r2}, {typed(p2[1])}), so the distance is √(({r2 - r1})² + ({typed(p2[1] - p1[1])})²) = {typed(dist)}."
                ],
                [
                    (r2 - r1, "r3g-abs-distance-gradient-ignored"),
                    ((r2 - r1) * (1 + abs(m)), "r3g-abs-distance-not-pythagoras"),
                    ((r2 - r1) * sp.sqrt(1 + abs(m)), "r3g-abs-distance-gradient-ignored"),
                    (sp.sqrt(abs(p2[1] - p1[1])), "r3g-abs-distance-not-pythagoras"),
                ],
                f"sqrt(1 + ({m})^2)*(part_b - part_a)",
            ),
            dict(
                prompt="Find the values of x for which the graph of y = |"
                + lin(a, e)
                + "| lies below the line. Write your answer as an inequality or in interval notation.",
                marks=2,
                working=[
                    f"|{lin(a, e)}| < {lin(m, d)} holds between the intersections, because the V is steeper than the line: {fields['answer_text']}."
                ],
                distractors=pick(
                    fields["answer_text"],
                    [
                        _ineq_dist(
                            sp.Union(sp.Interval.open(-sp.oo, r1), sp.Interval.open(r2, sp.oo)),
                            "r3g-abs-region-reversed",
                        ),
                        _ineq_dist(sp.Interval(r1, r2), "r3g-abs-equality-included"),
                        _ineq_dist(sp.Interval.open(-r2, -r1), "r3g-abs-sign-of-constant"),
                    ],
                    3,
                ),
                **fields,
            ),
        ],
    )
