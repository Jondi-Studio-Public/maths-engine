"""Generates `gt_f_squared` (easy, medium, hard) and the exam chain `gt_f_squared_exam`: key points of the graph of
``y = (f(x))^2`` from the graph or the rule of ``f``.

Skill ``graph-f-squared-features`` (``functions-sequences``, answer ``pair``: every tier asks for the coordinates of one
point, typed as ``x = ..., y = ...``).

* easy (drawn): ``f`` is a straight line ``m(x - r)`` crossing the x-axis at a whole number. Squaring never goes below
  the axis, so ``y = f(x)^2`` touches it at the zero of f: the turning point is ``(r, 0)``.
* medium (drawn): ``f`` is a parabola ``a(x - h)^2 + k`` with ``|k|`` at least 2. With two zeros (``a k < 0``) the square
  has a local maximum ``(h, k^2)`` between two minimums on the axis; with no zeros (``a k > 0``) it has the single
  turning point ``(h, k^2)``. The y-coordinate is squared, so a negative vertex height becomes positive.
* hard (no graph): either the local maximum of ``y = f(x)^2`` for ``f(x) = (x - p)(x - q)`` given expanded (complete the
  square, then square the height) or, for a straight line given by its rule, the point where ``y = f(x)^2`` meets
  ``y = f(x)`` away from the axis (where ``f(x) = 1``).
* exam (``gt_f_squared_exam``): for ``f(x) = x^2 - 2hx + h^2 - m^2``, (a) the least value of f, (b) the local maximum
  value of ``y = f(x)^2``, which follows through from (a) by squaring, (c) how many turning points the square has, and
  (d) how many solutions ``(f(x))^2 = c`` has for a given c between 0 and the local maximum, which is four.

Each key is built from the chosen vertex and zeros and checked by scanning the squared function numerically for its
turning points (and, for (d), counting sign changes of ``(f(x))^2 - c``).
"""

from __future__ import annotations

import math
from fractions import Fraction as Frac

import numpy as np

from .. import figures
from ..core import attempts, check, generator, phrase
from ._gt_quad import curve_points, para, plane, quad_text, verify_square, vertex_of
from ._sr2b_kit import Pair, int_dist, number_fields, pair_candidates, pick, retry_short

# --------------------------------------------------------------------------------------------------------- shared


def _pair(x, y) -> Pair:
    return Pair("x", x, "y", y)


def _square_points(h, m, k):
    """The turning points of (a(x - h)^2 + k)^2 when the zeros are h +/- m (m = 0 means there are none)."""
    if m == 0:
        return [(float(h), float(k * k), "min")]
    return [(float(h - m), 0.0, "min"), (float(h), float(k * k), "max"), (float(h + m), 0.0, "min")]


def _vertex_wrongs(h, k, m):
    """Wrong points for the vertex of the square: (tag, x, y)."""
    out = [
        ("gtf-turning-point-unsquared", h, k),
        ("gtf-square-keeps-sign", h, -k * k),
        ("gtf-absolute-value-for-square", h, abs(k)),
    ]
    if m:
        out.append(("gtf-turning-point-at-zero", h - m, 0))
    return out


def _num(v) -> str:
    v = Frac(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def _shifted(var: str, h: int) -> str:
    """x - 3, x + 3, or x for 0."""
    return var if h == 0 else f"{var} {'-' if h > 0 else '+'} {abs(h)}"


# ------------------------------------------------------------------------------------------------------- easy


def _easy(rng):
    for _ in attempts(500, "a line with a whole-number zero off the y-axis and a visible slope"):
        m = rng.choice([Frac(v) for v in (1, 2, 3, -1, -2, -3)] + [Frac(1, 2), Frac(-1, 2)])
        r = rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
        if abs(m * r) <= 7:
            break
    f = lambda x: float(m) * (x - r)  # noqa: E731
    fig = plane(f, "f", "the graph of the straight line y = f(x) on a coordinate grid, one unit per grid square")
    verify_square(f, [(float(r), 0.0, "min")])
    ans = _pair(r, 0)
    f0 = -m * r
    wrong = [
        ("gtf-turning-point-at-y-intercept", 0, f0),
        ("gtf-zero-sign", -r, 0),
        ("gtf-turning-point-at-y-intercept", 0, f0 * f0),
        ("gtf-invariant-point-as-turning-point", r + 1 / m, 1),
    ]
    options = pick(ans.text, pair_candidates(ans, wrong))
    stem = phrase(
        rng,
        (
            "The graph of the straight line y = f(x) is shown. State the coordinates of the turning point of the graph of y = (f(x))^2.",
            "The graph of y = f(x) is a straight line, as drawn. Find the coordinates of the turning point of the graph of y = (f(x))^2.",
        ),
    )
    figures.assert_no_leak(fig, [ans.text, *[d.text for d in options]])
    return dict(
        text=stem,
        figure=fig,
        distractors=options,
        working=[
            f"The line crosses the x-axis at x = {r}, so f({r}) = 0.",
            "Squaring never gives a negative value, so (f(x))^2 touches the x-axis there and has its lowest point at that zero.",
            f"The turning point is ({r}, 0).",
        ],
        **ans.fields(),
    )


# ----------------------------------------------------------------------------------------------------- medium


def _draw_parabola(rng, roots: bool):
    for _ in attempts(500, "a parabola with a vertex height of at least 2 inside the grid"):
        a = rng.choice((1, 1, -1, 2, -2))
        h = rng.choice((-3, -2, -1, 0, 1, 2, 3))
        if roots:
            m = rng.choice((1, 2))
            k = -a * m * m
        else:
            m = 0
            k = rng.choice((2, 3)) * (1 if a > 0 else -1)
        if 2 <= abs(k) <= 6 and abs(h) + m <= 6:
            return a, h, m, k
    raise AssertionError("unreachable")


def _medium(rng):
    roots = rng.random() < 0.6
    a, h, m, k = _draw_parabola(rng, roots)
    f = para(a, h, k)
    fig = plane(f, "f", "the graph of the parabola y = f(x) on a coordinate grid, one unit per grid square")
    verify_square(f, _square_points(h, m, k))
    ans = _pair(h, k * k)
    options = pick(ans.text, pair_candidates(ans, _vertex_wrongs(h, k, m)))
    if roots:
        stem = phrase(
            rng,
            (
                "The graph of the parabola y = f(x) is shown. The graph of y = (f(x))^2 has three turning points. State the coordinates of the local maximum.",
                "The graph of y = f(x) is drawn. Find the coordinates of the local maximum point of the graph of y = (f(x))^2.",
            ),
        )
        working = [
            f"f has its turning point at ({h}, {k}), between its zeros x = {h - m} and x = {h + m}.",
            f"At the zeros (f(x))^2 touches the axis; between them it is largest where |f| is largest, at x = {h}: y = ({k})^2 = {k * k}.",
            f"The local maximum is ({h}, {k * k}).",
        ]
    else:
        stem = phrase(
            rng,
            (
                "The graph of the parabola y = f(x) is shown and never meets the x-axis. State the coordinates of the turning point of the graph of y = (f(x))^2.",
                "The graph of y = f(x) is a parabola that does not cross the x-axis, as drawn. Find the coordinates of the turning point of y = (f(x))^2.",
            ),
        )
        working = [
            f"f has its turning point at ({h}, {k}) and never equals 0, so (f(x))^2 has one turning point, at the same x.",
            f"y = ({k})^2 = {k * k}, so the turning point is ({h}, {k * k}).",
        ]
    figures.assert_no_leak(fig, [ans.text, *[d.text for d in options]])
    return dict(text=stem, figure=fig, distractors=options, working=working, **ans.fields())


# ------------------------------------------------------------------------------------------------------- hard


def _draw_factors(rng, gap: int = 4):
    for _ in attempts(500, "two zeros of the same parity"):
        p, q = rng.randint(1, 9), rng.randint(1, 9)
        if q - p >= gap and (p + q) % 2 == 0:
            return p, q
    raise AssertionError("unreachable")


def _hard_max(rng):
    p, q = _draw_factors(rng)
    h, m = (p + q) // 2, (q - p) // 2
    k = -m * m
    f = para(1, h, k)
    verify_square(f, _square_points(h, m, k))
    ans = _pair(h, k * k)
    options = pick(ans.text, pair_candidates(ans, _vertex_wrongs(h, k, m)))
    stem = phrase(
        rng,
        (
            "Let f(x) = {f}. The graph of y = (f(x))^2 has three turning points. Find the coordinates of its local maximum.",
            "The function f is given by f(x) = {f}. State the coordinates of the local maximum of the graph of y = (f(x))^2.",
        ),
        f=quad_text(1, -(p + q), p * q),
    )
    return dict(
        text=stem,
        distractors=options,
        working=[
            f"Complete the square: f(x) = (x - {h})^2 - {m * m}, so f has its minimum value {k} at x = {h}.",
            f"The zeros of f are x = {p} and x = {q}, where (f(x))^2 has minimums; between them (f(x))^2 has a local maximum.",
            f"At x = {h}: y = ({k})^2 = {k * k}. The local maximum is ({h}, {k * k}).",
        ],
        **ans.fields(),
    )


def _linear_text(m: Frac, c: int) -> str:
    """m x + c as written in a stem: 2x - 6, x/2 + 3, -x + 4."""
    sign = "-" if m < 0 else ""
    num, den = abs(m.numerator), m.denominator
    lead = "" if num == 1 else str(num)
    head = f"{sign}{lead}x" if den == 1 else f"{sign}{lead}x/{den}"
    return head if c == 0 else f"{head} {'+' if c > 0 else '-'} {abs(c)}"


def _hard_meet(rng):
    for _ in attempts(500, "a line with whole-number intercept data"):
        m = rng.choice([Frac(v) for v in (2, 3, -2, -3, 4)] + [Frac(1, 2), Frac(-1, 2)])
        r = rng.choice((-5, -4, -3, -2, -1, 1, 2, 3, 4, 5))
        c = -m * r
        if c.denominator == 1 and abs(c) <= 12:
            break
    c = int(c)
    f = lambda x: float(m) * x + c  # noqa: E731
    x1 = r + 1 / m
    # the second route: (f(x))^2 - f(x) changes sign at exactly two places, f = 0 and f = 1
    xs = np.linspace(float(min(r, x1)) - 3, float(max(r, x1)) + 3, 24001) + 1.234e-6
    g = f(xs) ** 2 - f(xs)
    crossings = xs[:-1][np.nonzero(g[:-1] * g[1:] < 0)[0]]
    check(len(crossings) == 2, "y = f^2 and y = f should cross at two points")
    check(abs(f(float(x1)) - 1) < 1e-9 and abs(f(float(r))) < 1e-9, "the meeting points are not f = 0 and f = 1")
    check(
        all(min(abs(cx - float(r)), abs(cx - float(x1))) < 2e-3 for cx in crossings),
        "a crossing is not at a zero of f or where f = 1",
    )
    ans = _pair(x1, 1)
    wrong = [
        ("gtf-gives-the-zero", r, 0),
        ("gtf-uses-f-equals-minus-one", r - 1 / m, 1),
        ("gtf-forgets-to-divide-by-gradient", r + 1, 1),
        ("gtf-gives-the-zero", x1, 0),
    ]
    options = pick(ans.text, pair_candidates(ans, wrong))
    stem = phrase(
        rng,
        (
            "Let f(x) = {f}. The graphs of y = f(x) and y = (f(x))^2 meet at two points. One of them is on the x-axis. Find the coordinates of the other.",
            "The line y = f(x) has rule f(x) = {f}. The graph of y = (f(x))^2 meets the line at two points, one on the x-axis. State the coordinates of the other point.",
        ),
        f=_linear_text(m, c),
    )
    return dict(
        text=stem,
        distractors=options,
        working=[
            "The graphs meet where (f(x))^2 = f(x), so f(x) = 0 or f(x) = 1.",
            f"f(x) = 0 gives x = {r}, the point on the x-axis. f(x) = 1 gives x = {_num(x1)}, with y = 1.",
            f"The other point is ({_num(x1)}, 1).",
        ],
        **ans.fields(),
    )


def _read(question) -> str:
    """Recompute the answer from the drawn graph alone."""
    pts = curve_points(question.figure)
    if "straight line" in question.text:
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        m = Frac(round((y2 - y1) * 12), round((x2 - x1) * 12)).limit_denominator(6)
        r = Frac(round((x1 - y1 / float(m)) * 12), 12).limit_denominator(12)
        return _pair(r, 0).text
    vx, vy = vertex_of(pts)
    h, k = Frac(round(vx * 12), 12), Frac(round(vy * 12), 12)
    return _pair(h, k * k).text


figures.register_reader("gt_f_squared", _read)


@generator(
    "gt_f_squared",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def gt_f_squared(rng, tier="easy"):
    """A turning or meeting point of y = f(x)^2; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_max, _hard_max, _hard_meet])(rng)


# -------------------------------------------------------------------------------------------------------- exam


def _count_solutions(fn, c: float, lo: float, hi: float) -> int:
    """Sign changes of fn(x)^2 - c on [lo, hi], on a grid that avoids the turning points (the second route)."""
    xs = np.linspace(lo, hi, 200001) + 1.234e-6
    g = fn(xs) ** 2 - c
    return int(np.count_nonzero(g[:-1] * g[1:] < 0))


@generator("gt_f_squared_exam", level=15, presets={"exam": dict(tier="exam")})
@retry_short
def gt_f_squared_exam(rng, tier="exam"):
    """Exam chain on f(x) = x^2 - 2hx + h^2 - m^2: least value, local maximum of f^2, turning points, solutions of f^2 = c."""
    check(tier == "exam", f"unknown tier {tier!r}")
    h = rng.choice((-3, -2, -1, 1, 2, 3, 4))
    n = rng.choice((2, 3, 4, 5, 6, 7, 8, 9, 10, 12))  # f(x) = (x - h)^2 - n, so f has two zeros
    k = -n
    f = para(1, h, k)
    verify_square(f, _square_points(h, math.sqrt(n), k))
    top = n * n
    case = rng.choice(("below", "at", "above"))
    c = {"below": rng.randint(1, top - 1), "at": top, "above": top + rng.choice((1, 2, 3, 5, 8, 20))}[case]
    expect = {"below": 4, "at": 3, "above": 2}[case]
    span = math.sqrt(n) + math.sqrt(c) + 6.0
    if case == "at":
        check(
            _count_solutions(f, c + 1e-3, h - span, h + span) == 2
            and _count_solutions(f, c - 1e-3, h - span, h + span) == 4,
            "peak line",
        )
    else:
        check(_count_solutions(f, c, h - span, h + span) == expect, f"(f(x))^2 = {c} does not have {expect} solutions")
    a_fields = number_fields(Frac(k), 0)
    a_wrong = [
        int_dist(-2 * h, "gtf-reads-linear-coefficient-as-minimum"),
        int_dist(h * h - n, "gtf-reads-constant-as-minimum"),
        int_dist(n, "gtf-minimum-sign"),
        int_dist(-h * h, "gtf-completing-square-drops-constant"),
    ]
    a_wrong = [d for d in a_wrong if d.text != str(k)]
    b_fields = number_fields(Frac(top), 0)
    b_wrong = [
        int_dist(k, "gtf-turning-point-unsquared"),
        int_dist(-top, "gtf-square-keeps-sign"),
        int_dist(2 * abs(k), "gtf-doubles-instead-of-squares"),
    ]
    c_wrong = [int_dist(2, "gtf-count-ignores-max"), int_dist(1, "gtf-count-as-f"), int_dist(4, "gtf-count-extra")]
    d_wrong = {
        "below": [
            int_dist(2, "gtf-solutions-above-the-peak"),
            int_dist(3, "gtf-solutions-at-the-peak"),
            int_dist(6, "gtf-solutions-both-branches-doubled"),
        ],
        "at": [
            int_dist(4, "gtf-four-solutions-for-line-at-the-peak"),
            int_dist(2, "gtf-two-solutions-for-line-at-the-peak"),
            int_dist(6, "gtf-solutions-both-branches-doubled"),
        ],
        "above": [
            int_dist(4, "gtf-four-solutions-for-line-above-the-peak"),
            int_dist(3, "gtf-three-solutions-for-line-above-the-peak"),
            int_dist(1, "gtf-one-solution-for-line-above-the-peak"),
        ],
    }[case]
    d_working = {
        "below": [
            f"The local maximum of (f(x))^2 is {top} and its minimums are 0, and 0 < {c} < {top}.",
            f"Each of the four rising and falling sections of the graph is crossed once by the horizontal line y = {c}, so there are 4 solutions.",
        ],
        "at": [
            f"The local maximum of (f(x))^2 is {top}, and {c} = {top}.",
            f"The line y = {c} touches the graph at its peak and cuts each of the two outer sections once, so there are 3 solutions.",
        ],
        "above": [
            f"The local maximum of (f(x))^2 is {top}, and {c} > {top}.",
            f"The line y = {c} is above the peak, so it cuts only the two outer sections, once each: 2 solutions.",
        ],
    }[case]
    parts = [
        dict(
            prompt="Find the least value of f, by completing the square.",
            marks=2,
            sensible_range=(-100, 100),
            working=[f"f(x) = ({_shifted('x', h)})^2 - {n}.", f"The least value of f is {k}, at x = {h}."],
            distractors=pick(a_fields["answer_text"], a_wrong),
            **a_fields,
        ),
        dict(
            prompt="Hence find the y-coordinate of the local maximum of the graph of y = (f(x))^2.",
            marks=2,
            sensible_range=(0, 10000),
            follow_through="part_a^2",
            working=[
                f"At x = {h}, f(x) = {k}, between the zeros, so (f(x))^2 has a local maximum there.",
                f"y = ({k})^2 = {top}.",
            ],
            distractors=pick(b_fields["answer_text"], b_wrong),
            **b_fields,
        ),
        dict(
            prompt="How many turning points does the graph of y = (f(x))^2 have?",
            marks=1,
            sensible_range=(0, 20),
            working=[
                f"f has two zeros, where (f(x))^2 has minimums on the axis, and a turning point between them at x = {h}.",
                "So there are 3 turning points.",
            ],
            distractors=pick("3", c_wrong),
            **number_fields(Frac(3), 0),
        ),
        dict(
            prompt=f"How many solutions does the equation (f(x))^2 = {c} have?",
            marks=2,
            sensible_range=(0, 20),
            working=d_working,
            distractors=pick(str(expect), d_wrong),
            **number_fields(Frac(expect), 0),
        ),
    ]
    stem = phrase(
        rng,
        (
            "The function f is given by f(x) = {f}.",
            "Consider the function f(x) = {f}.",
        ),
        f=quad_text(1, -2 * h, h * h - n),
    )
    return dict(text=stem, parts=parts)
