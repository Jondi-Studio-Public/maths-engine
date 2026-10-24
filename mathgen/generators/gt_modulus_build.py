"""Generates `gt_modulus` (easy, medium, hard) and the exam chain `gt_modulus_exam`: typed key points of the graphs of
``y = |f(x)|`` and ``y = f(|x|)`` for a quadratic ``f`` given by its rule.

Skill ``graph-modulus-key-features`` (``functions-sequences``, answer ``pair``: every tier asks for the coordinates of
one point, typed as ``x = ..., y = ...``). The existing ``senior-graph-reciprocal-abs-compare`` skill only matches a drawn
graph to its equation from a list; here the learner finds the points.

* easy: ``f(x) = (x - p)(x - q)`` given expanded. Either the local maximum of ``y = |f(x)|`` (the vertex of f, flipped
  above the axis: ``(h, m^2)``) or, for zeros of opposite sign, the y-intercept of ``y = |f(x)|`` (``-pq``, positive).
* medium: ``y = f(|x|)``. Either the minimum point with positive x (the vertex ``(h, -m^2)``, since both zeros are
  positive and the left half of the graph is the mirror of the right), or, when one zero is negative, the x-intercept
  with positive x (the negative zero gives none, because ``|x|`` is never negative).
* hard: ``y = |f(x)|`` for ``f(x) = a(x - p)(x - q)`` with a leading coefficient other than 1, given expanded: the local
  maximum is ``(h, |a| m^2)`` whether f opens up (the vertex is flipped) or down (it is already above the axis).
* exam (``gt_modulus_exam``): for ``f(x) = x^2 - (p + q)x + pq`` with both zeros positive, (a) the x-intercepts of
  ``y = f(|x|)`` (four of them), (b) the least value of f, (c) the local maximum value of ``y = |f(x)|``, which follows
  through from (b) by the absolute value, and (d) how many solutions ``|f(x)| = c`` has for a given c: four below the
  local maximum, three at it, two above.

Each key is built from the chosen zeros, then checked by scanning the modulus function numerically for its turning
points or, for the exam counts, by solving ``f(x) = c`` and ``f(x) = -c`` with SymPy.
"""

from __future__ import annotations

from fractions import Fraction as Frac

import numpy as np
import sympy as sp

from ..answers import Distractor
from ..core import attempts, check, generator, phrase
from ._gt_quad import quad_text, scan_turning_points
from ._sr2b_kit import Pair, int_dist, number_fields, pair_candidates, pick, retry_short

X = sp.Symbol("x")


def _pair(x, y) -> Pair:
    return Pair("x", x, "y", y)


def _poly(a, p, q):
    """a(x - p)(x - q) as numpy-friendly function and expanded coefficients."""
    A, B, C = a, -a * (p + q), a * p * q
    return (lambda x: A * x * x + B * x + C), (A, B, C)


def _draw_same_parity(rng, lo: int, hi: int, gap: int = 2, positive: bool = False):
    for _ in attempts(1000, "two zeros of the same parity"):
        p, q = rng.randint(lo, hi), rng.randint(lo, hi)
        if q - p >= gap and (p + q) % 2 == 0 and (not positive or p > 0):
            return p, q
    raise AssertionError("unreachable")


def _verify_modulus_points(fn, expected, lo, hi) -> None:
    """The turning points of fn(x), scanned numerically, are exactly `expected` [(x, y, "max" or "min")]."""
    found = scan_turning_points(fn, lo, hi)
    check(len(found) == len(expected), f"{len(found)} turning points found, {len(expected)} expected")
    for (x, y, kind), (ex, ey, ek) in zip(found, sorted(expected)):
        check(abs(x - ex) < 2e-2 and abs(y - ey) < 2e-2 * (1 + abs(ey)) and kind == ek, "turning point differs")


# ------------------------------------------------------------------------------------------------------- easy


def _easy_max(rng):
    p, q = _draw_same_parity(rng, -7, 7)
    h, m = (p + q) // 2, (q - p) // 2
    f, (A, B, C) = _poly(1, p, q)
    _verify_modulus_points(
        lambda x: abs(f(x)),
        [(float(p), 0.0, "min"), (float(h), float(m * m), "max"), (float(q), 0.0, "min")],
        p - 3.0,
        q + 3.0,
    )
    ans = _pair(h, m * m)
    wrong = [
        ("gtm-reflection-skipped", h, -m * m),
        ("gtm-negates-x-instead-of-y", -h, m * m),
        ("gtm-y-intercept-as-turning-point", 0, abs(p * q)),
        ("gtm-negates-both-coordinates", -h, -m * m),
    ]
    options = pick(ans.text, pair_candidates(ans, wrong))
    stem = phrase(
        rng,
        (
            "Let f(x) = {f}. State the coordinates of the local maximum of the graph of y = |f(x)|.",
            "The function f is given by f(x) = {f}. Find the coordinates of the local maximum point of y = |f(x)|.",
        ),
        f=quad_text(A, B, C),
    )
    return dict(
        text=stem,
        distractors=options,
        working=[
            f"Factorise: f(x) = (x {_sg(-p)})(x {_sg(-q)}), so the zeros are x = {p} and x = {q} and the vertex is at x = {h}.",
            f"f({h}) = -{m * m}. The part of the graph below the x-axis is reflected above it, so the vertex becomes ({h}, {m * m}).",
            f"The local maximum of y = |f(x)| is ({h}, {m * m}).",
        ],
        **ans.fields(),
    )


def _sg(n: int) -> str:
    """+ 3 or - 3."""
    return f"{'+' if n >= 0 else '-'} {abs(n)}"


def _easy_intercept(rng):
    for _ in attempts(500, "zeros of opposite sign"):
        p, q = rng.randint(-6, -1), rng.randint(1, 6)
        if abs(p * q) >= 3:
            break
    f, (A, B, C) = _poly(1, p, q)
    y0 = -p * q
    check(abs(abs(f(0.0)) - y0) < 1e-12 and f(0.0) < 0, "f(0) is not the negative of the answer")
    ans = _pair(0, y0)
    wrong = [
        ("gtm-intercept-keeps-sign", 0, p * q),
        ("gtm-reads-linear-coefficient-as-intercept", 0, B),
        ("gtm-gives-an-x-intercept", q, 0),
        ("gtm-gives-an-x-intercept", p, 0),
    ]
    options = pick(ans.text, pair_candidates(ans, wrong))
    stem = phrase(
        rng,
        (
            "Let f(x) = {f}. State the coordinates of the y-intercept of the graph of y = |f(x)|.",
            "The function f is given by f(x) = {f}. Find the coordinates of the point where the graph of y = |f(x)| meets the y-axis.",
        ),
        f=quad_text(A, B, C),
    )
    return dict(
        text=stem,
        distractors=options,
        working=[
            f"f(0) = {C}, which is negative, so |f(0)| = {y0}.",
            f"The graph of y = |f(x)| crosses the y-axis at (0, {y0}).",
        ],
        **ans.fields(),
    )


# ----------------------------------------------------------------------------------------------------- medium


def _medium_min(rng):
    p, q = _draw_same_parity(rng, 1, 9, gap=2, positive=True)
    h, m = (p + q) // 2, (q - p) // 2
    f, (A, B, C) = _poly(1, p, q)
    found = scan_turning_points(lambda x: f(np.abs(x)), -q - 3.0, q + 3.0)
    mins = sorted(t for t in found if t[2] == "min")
    check(len(mins) == 2 and abs(mins[0][0] + h) < 2e-2 and abs(mins[1][0] - h) < 2e-2, "f(|x|) minimums differ")
    check(all(abs(t[1] + m * m) < 2e-2 for t in mins), "f(|x|) minimum heights differ")
    ans = _pair(h, -m * m)
    wrong = [
        ("gtm-confuses-absolute-forms", h, m * m),
        ("gtm-sum-of-zeros-as-vertex", p + q, -m * m),
        ("gtm-y-intercept-as-turning-point", 0, p * q),
        ("gtm-uses-a-zero-as-the-minimum", p, 0),
    ]
    options = pick(ans.text, pair_candidates(ans, wrong))
    stem = phrase(
        rng,
        (
            "Let f(x) = {f}. The graph of y = f(|x|) has two minimum points. State the coordinates of the one with positive x-coordinate.",
            "The function f is given by f(x) = {f}. Find the coordinates of the minimum point of the graph of y = f(|x|) that has a positive x-coordinate.",
        ),
        f=quad_text(A, B, C),
    )
    return dict(
        text=stem,
        distractors=options,
        working=[
            f"For x ≥ 0, |x| = x, so y = f(|x|) is the graph of f. f has its minimum at ({h}, {-m * m}), with zeros x = {p} and x = {q}.",
            "The graph of y = f(|x|) for x < 0 is the mirror image in the y-axis, so there is a second minimum at the mirror point.",
            f"The minimum with positive x is ({h}, {-m * m}).",
        ],
        **ans.fields(),
    )


def _medium_intercept(rng):
    for _ in attempts(500, "zeros of opposite sign"):
        p, q = rng.randint(-6, -1), rng.randint(1, 6)
        if abs(p) != q:
            break
    f, (A, B, C) = _poly(1, p, q)
    xs = np.linspace(-9, 9, 180001) + 1.234e-6
    g = f(np.abs(xs))
    crossings = xs[:-1][np.nonzero(g[:-1] * g[1:] < 0)[0]]
    check(
        len(crossings) == 2 and all(abs(abs(c) - q) < 1e-3 for c in crossings), "f(|x|) should cross at x = +/- q only"
    )
    ans = _pair(q, 0)
    wrong = [
        ("gtm-mirrors-the-negative-zero", -p, 0),
        ("gtm-uses-the-negative-zero", p, 0),
        ("gtm-y-intercept-as-turning-point", 0, p * q),
        ("gtm-confuses-absolute-forms", p + q, 0),
    ]
    options = pick(ans.text, pair_candidates(ans, wrong))
    stem = phrase(
        rng,
        (
            "Let f(x) = {f}. The graph of y = f(|x|) has two x-intercepts. State the coordinates of the one with positive x-coordinate.",
            "The function f is given by f(x) = {f}. Find the coordinates of the x-intercept of the graph of y = f(|x|) that has a positive x-coordinate.",
        ),
        f=quad_text(A, B, C),
    )
    return dict(
        text=stem,
        distractors=options,
        working=[
            f"f(x) = (x {_sg(-p)})(x {_sg(-q)}), so f(|x|) = 0 when |x| = {p} or |x| = {q}.",
            f"|x| is never negative, so |x| = {p} has no solution; |x| = {q} gives x = {q} or x = {-q}.",
            f"The x-intercept with positive x-coordinate is ({q}, 0).",
        ],
        **ans.fields(),
    )


# ------------------------------------------------------------------------------------------------------- hard


def _hard(rng):
    a = rng.choice((2, 3, -1, -2))
    p, q = _draw_same_parity(rng, -6, 6)
    h, m = (p + q) // 2, (q - p) // 2
    f, (A, B, C) = _poly(a, p, q)
    k = -a * m * m
    top = abs(k)
    pts = [(float(p), 0.0, "min"), (float(h), float(top), "max"), (float(q), 0.0, "min")]
    _verify_modulus_points(lambda x: abs(f(x)), pts, p - 3.0, q + 3.0)
    ans = _pair(h, top)
    wrong = [
        ("gtm-reflection-skipped" if a > 0 else "gtm-reflects-when-already-positive", h, -top),
        ("gtm-forgets-leading-coefficient", h, m * m),
        ("gtm-negates-x-instead-of-y", -h, top),
        ("gtm-y-intercept-as-turning-point", 0, abs(C)),
    ]
    options = pick(ans.text, pair_candidates(ans, wrong))
    where = (
        "opens upward, so its vertex is below the axis and is reflected"
        if a > 0
        else "opens downward, so its vertex is already above the axis and stays"
    )
    stem = phrase(
        rng,
        (
            "Let f(x) = {f}. State the coordinates of the local maximum of the graph of y = |f(x)|.",
            "The function f is given by f(x) = {f}. Find the coordinates of the local maximum point of y = |f(x)|.",
        ),
        f=quad_text(A, B, C),
    )
    return dict(
        text=stem,
        distractors=options,
        working=[
            f"Factorise: f(x) = {'' if a == 1 else '-' if a == -1 else a}(x {_sg(-p)})(x {_sg(-q)}). The zeros are x = {p} and x = {q}, so the vertex is at x = {h}.",
            f"f({h}) = {k}. The parabola {where}, so the height of the local maximum of |f| is |{k}| = {top}.",
            f"The local maximum of y = |f(x)| is ({h}, {top}).",
        ],
        **ans.fields(),
    )


@generator(
    "gt_modulus",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def gt_modulus(rng, tier="easy"):
    """A key point of y = |f(x)| or y = f(|x|); the tiers are described in the module docstring."""
    if tier == "easy":
        return rng.choice([_easy_max, _easy_max, _easy_intercept])(rng)
    if tier == "medium":
        return rng.choice([_medium_min, _medium_intercept])(rng)
    if tier != "hard":
        raise ValueError(f"unknown tier {tier!r}")
    return _hard(rng)


# -------------------------------------------------------------------------------------------------------- exam


def _distinct_real_roots(poly) -> set:
    return {r for r in sp.Poly(poly, X).real_roots()}


def _solutions_of_abs(p: int, q: int, c: int) -> int:
    """The number of x with |x^2 - (p + q)x + pq| = c, from SymPy's real roots of f = c and f = -c."""
    f = X**2 - (p + q) * X + p * q
    return len(_distinct_real_roots(f - c) | _distinct_real_roots(f + c))


def _list_text(values) -> str:
    return ", ".join(str(v) for v in values)


def _list_dist(values, tag) -> Distractor:
    values = sorted(values)
    return Distractor(_list_text(values), tag, values)


@generator("gt_modulus_exam", level=15, presets={"exam": dict(tier="exam")})
@retry_short
def gt_modulus_exam(rng, tier="exam"):
    """Exam chain on a quadratic with both zeros positive: intercepts of f(|x|), least value, peak of |f|, solutions of |f| = c."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    p, q = _draw_same_parity(rng, 1, 20, gap=4, positive=True)
    h, m = (p + q) // 2, (q - p) // 2
    peak = m * m
    kind = rng.choice(("below", "at", "above"))
    c = {
        "below": rng.choice([v for v in range(1, peak) if v != peak]),
        "at": peak,
        "above": peak + rng.choice((1, 2, 3, 5)),
    }[kind]
    count = {"below": 4, "at": 3, "above": 2}[kind]
    check(_solutions_of_abs(p, q, c) == count, f"|f(x)| = {c} does not have {count} solutions")
    key_a = sorted({-q, -p, p, q})
    check(len(key_a) == 4, "four intercepts expected")
    # the intercepts of f(|x|), second route: a non-negative root of f gives x and -x, a negative root gives none
    roots = [r for r in sp.solve(X**2 - (p + q) * X + p * q, X) if r >= 0]
    check(sorted({s_ * r for r in roots for s_ in (-1, 1)}) == key_a, "intercepts differ")
    a_text = _list_text(key_a)
    a_wrong = [
        _list_dist([p, q], "gtm-misses-the-mirror-image"),
        _list_dist([-q, -p], "gtm-keeps-only-the-mirror-image"),
        _list_dist([-p, p], "gtm-uses-one-zero"),
        _list_dist([p, q, p + q], "gtm-sum-of-zeros-as-vertex"),
    ]
    b_fields = number_fields(Frac(-peak), 0)
    b_wrong = [
        int_dist(peak, "gtm-minimum-sign"),
        int_dist(p * q, "gtm-reads-constant-as-minimum"),
        int_dist(-(p + q), "gtm-reads-linear-coefficient-as-minimum"),
        int_dist(-h * h, "gtm-completing-square-drops-constant"),
    ]
    c_fields = number_fields(Frac(peak), 0)
    c_wrong = [
        int_dist(-peak, "gtm-reflection-skipped"),
        int_dist(h, "gtm-gives-the-x-coordinate"),
        int_dist(p * q, "gtm-y-intercept-as-turning-point"),
    ]
    c_wrong = [w for w in c_wrong if int(w.text) != peak]
    d_wrong = [
        int_dist(n, tag)
        for n, tag in (
            (2, "gtm-counts-only-outer-sections"),
            (3, "gtm-counts-the-peak-as-a-solution"),
            (4, "gtm-treats-the-line-as-below-the-peak"),
            (6, "gtm-counts-every-section-twice"),
        )
        if n != count
    ]
    why = {
        "below": f"The local maximum of |f| is {peak} and its minimums are 0. A horizontal line y = {c} with 0 < {c} < {peak} cuts each of the four sections once: 4 solutions.",
        "at": f"The line y = {c} passes through the local maximum {peak}, touching there, and cuts the two outer sections: 3 solutions.",
        "above": f"The line y = {c} is above the local maximum {peak}, so it cuts only the two outer sections: 2 solutions.",
    }[kind]
    parts = [
        dict(
            prompt="Find the x-intercepts of the graph of y = f(|x|). List them from least to greatest.",
            marks=2,
            working=[
                f"f(x) = (x - {p})(x - {q}), so f(|x|) = 0 when |x| = {p} or |x| = {q}.",
                f"Each of these has two values of x: {a_text}.",
            ],
            distractors=pick(a_text, a_wrong),
            answer_kind="list",
            answer=key_a,
            answer_text=a_text,
            answer_latex="plain:" + a_text,
        ),
        dict(
            prompt="Find the least value of f.",
            marks=2,
            sensible_range=(-100, 100),
            working=[f"f(x) = (x {_sg(-h)})^2 - {peak}, so the least value of f is {-peak}, at x = {h}."],
            distractors=pick(b_fields["answer_text"], b_wrong),
            **b_fields,
        ),
        dict(
            prompt="Hence find the y-coordinate of the local maximum of the graph of y = |f(x)|.",
            marks=1,
            sensible_range=(0, 10000),
            follow_through="Abs(part_b)",
            working=[
                f"The vertex of f, at height {-peak}, is below the axis, so it is reflected: the height becomes |{-peak}| = {peak}."
            ],
            distractors=pick(c_fields["answer_text"], c_wrong),
            **c_fields,
        ),
        dict(
            prompt=f"How many solutions does the equation |f(x)| = {c} have?",
            marks=2,
            sensible_range=(0, 20),
            working=[why],
            distractors=pick(str(count), d_wrong),
            **number_fields(Frac(count), 0),
        ),
    ]
    stem = phrase(
        rng,
        (
            "The function f is given by f(x) = {f}.",
            "Consider the function f(x) = {f}.",
            "Let f be the quadratic function with rule f(x) = {f}.",
            "A quadratic function is defined by f(x) = {f}.",
        ),
        f=quad_text(1, -(p + q), p * q),
    )
    return dict(text=stem, parts=parts)
