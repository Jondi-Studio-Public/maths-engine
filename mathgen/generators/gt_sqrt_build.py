"""Generates `gt_sqrt_function` (easy, medium, hard) and the exam chain `gt_sqrt_function_exam`: the domain and range of
``y = sqrt(f(x))`` from the graph or the rule of ``f``.

Skill ``graph-sqrt-of-function-domain-range`` (``functions-sequences``, answer ``inequality``: every tier asks for a set
of numbers, typed as inequalities such as ``x <= 1 or x >= 5``, ``0 <= y <= 3``, ``all real numbers`` or ``no solution``).

* easy (drawn): ``f`` is a straight line crossing the x-axis at a whole number; the root is defined where the line is
  on or above the axis, one side of the zero.
* medium (drawn): ``f`` is a parabola ``a(x - h)^2 + k``. With zeros ``h +/- m`` the domain is outside them (``a > 0``)
  or between them (``a < 0``); a parabola that misses the axis gives all real numbers (opens up, above the axis) or no
  solution (opens down, below it).
* hard (no graph): the domain of ``sqrt(f(x))`` for ``f(x) = a(x - p)(x - q)`` given expanded, or the range of
  ``y = sqrt(f(x))`` for ``f(x) = s^2 - (x - h)^2`` (from 0 up to s) or ``f(x) = (x - h)^2 + s^2`` (from s up).
* exam (``gt_sqrt_function_exam``): for ``f(x) = s^2 - (x - h)^2`` given expanded, (a) the greatest value of f, (b) the
  greatest value of ``y = sqrt(f(x))``, which follows through from (a) by a square root, (c) the domain, and (d) the
  exact y-intercept, a surd.

Each key is built from the chosen zeros or vertex and checked by scanning ``f`` (and ``sqrt(f)``) numerically.
"""

from __future__ import annotations

import math
from fractions import Fraction as Frac

import numpy as np
import sympy as sp

from .. import figures
from ..core import attempts, check, generator, phrase, rational_text
from ._gt_quad import (
    curve_points,
    ineq_dist,
    ineq_text,
    interval_set,
    para,
    plane,
    quad_text,
    verify_domain,
    vertex_of,
)
from ._sr2b_kit import int_dist, number_fields, pick, retry_short, text_dist

ALL = sp.Interval(-sp.oo, sp.oo)
NONE = sp.EmptySet


def _strict(spec) -> sp.Set:
    """The same set with its end points left out (what a learner writes with < or >)."""
    kind = spec[0]
    r = lambda v: sp.Rational(Frac(v).numerator, Frac(v).denominator)  # noqa: E731
    if kind == "ge":
        return sp.Interval.open(r(spec[1]), sp.oo)
    if kind == "le":
        return sp.Interval.open(-sp.oo, r(spec[1]))
    if kind == "outside":
        return sp.Union(sp.Interval.open(-sp.oo, r(spec[1])), sp.Interval.open(r(spec[2]), sp.oo))
    return sp.Interval.open(r(spec[1]), r(spec[2]))


def _finish(text, answer_set, wrong_sets, var="x"):
    """The answer fields and the picked wrong sets for one inequality question."""
    options = pick(text, [ineq_dist(s, tag, var) for s, tag in wrong_sets])
    return dict(
        answer_kind="inequality",
        answer=text,
        answer_text=text,
        answer_latex="plain:" + text,
        distractors=options,
    )


# ------------------------------------------------------------------------------------------------------- easy


def _easy(rng):
    for _ in attempts(500, "a line with a whole-number zero and a visible slope"):
        m = rng.choice([Frac(v) for v in (1, 2, 3, -1, -2, -3)] + [Frac(1, 2), Frac(-1, 2)])
        r = rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
        if abs(m * r) <= 7:
            break
    f = lambda x: float(m) * (x - r)  # noqa: E731
    fig = plane(f, "f", "the graph of the straight line y = f(x) on a coordinate grid, one unit per grid square")
    spec = ("ge", r) if m > 0 else ("le", r)
    s = interval_set(spec)
    verify_domain(f, s, -9.0, 9.0)
    text = ineq_text(s)
    wrong = [
        (interval_set(("le", r) if m > 0 else ("ge", r)), "gtf-domain-wrong-side"),
        (_strict(spec), "gtf-domain-excludes-zeros"),
        (ALL, "gtf-domain-all-reals"),
        (interval_set(("ge", -r) if m > 0 else ("le", -r)), "gtf-zero-sign"),
    ]
    out = _finish(text, s, wrong)
    stem = phrase(
        rng,
        (
            "The graph of the straight line y = f(x) is shown. Find the domain of the function y = sqrt(f(x)), as an inequality.",
            "The graph of y = f(x) is a straight line, as drawn. State the largest domain of the function y = sqrt(f(x)).",
            "A straight line is drawn as the graph of y = f(x). For which values of x is the function y = sqrt(f(x)) defined? Give your answer as an inequality.",
        ),
    )
    figures.assert_no_leak(fig, [text, *[d.text for d in out["distractors"]]])
    side = "right" if m > 0 else "left"
    out.update(
        text=stem,
        figure=fig,
        working=[
            "The square root is defined only where f(x) is not negative: on or above the x-axis.",
            f"The line is on or above the axis at x = {r} and to the {side} of it, so the domain is {text}.",
        ],
    )
    return out


# ----------------------------------------------------------------------------------------------------- medium


def _draw_parabola(rng):
    for _ in attempts(500, "a parabola inside the grid"):
        a = rng.choice((1, 1, -1, -1, 2, -2))
        h = rng.choice((-3, -2, -1, 0, 1, 2, 3))
        if rng.random() < 0.2:
            m = 0
            k = rng.choice((1, 2, 3)) * (1 if a > 0 else -1)
        else:
            m = rng.choice((1, 2))
            k = -a * m * m
        if abs(k) <= 6 and abs(h) + m <= 6:
            return a, h, m, k
    raise AssertionError("unreachable")


def _parabola_domain(a, h, m):
    """The domain of sqrt(a(x - h)^2 + k) when the zeros are h +/- m (m = 0: f has no zeros)."""
    if m == 0:
        return ALL if a > 0 else NONE
    return interval_set(("outside", h - m, h + m) if a > 0 else ("between", h - m, h + m))


def _domain_wrongs(a, h, m, k):
    if m == 0:
        if a > 0:  # above the axis: every x works
            return [
                (NONE, "gts-no-zeros-no-domain"),
                (interval_set(("ge", h)), "gts-domain-from-vertex"),
                (interval_set(("le", h)), "gts-domain-from-vertex"),
            ]
        return [  # below the axis: no x works
            (ALL, "gtf-domain-all-reals"),
            (interval_set(("ge", h)), "gts-domain-from-vertex"),
            (interval_set(("le", h)), "gts-domain-from-vertex"),
        ]
    low, high = h - m, h + m
    if a > 0:
        spec, other = ("outside", low, high), ("between", low, high)
    else:
        spec, other = ("between", low, high), ("outside", low, high)
    return [
        (interval_set(other), "gtf-domain-wrong-side"),
        (_strict(spec), "gtf-domain-excludes-zeros"),
        (ALL, "gtf-domain-all-reals"),
        (interval_set(("outside", -high, -low)) if a > 0 else interval_set(("between", -high, -low)), "gtf-zero-sign"),
    ]


def _medium(rng):
    a, h, m, k = _draw_parabola(rng)
    f = para(a, h, k)
    fig = plane(f, "f", "the graph of the parabola y = f(x) on a coordinate grid, one unit per grid square")
    s = _parabola_domain(a, h, m)
    verify_domain(f, s, -10.0, 10.0)
    text = ineq_text(s)
    out = _finish(text, s, _domain_wrongs(a, h, m, k))
    stem = phrase(
        rng,
        (
            "The graph of the parabola y = f(x) is shown. Find the domain of the function y = sqrt(f(x)), as an inequality.",
            "The graph of y = f(x) is drawn. State the largest domain of the function y = sqrt(f(x)).",
            "Part of the graph of y = f(x) is drawn. For which values of x is the function y = sqrt(f(x)) defined? Give your answer as an inequality.",
        ),
    )
    figures.assert_no_leak(fig, [text, *[d.text for d in out["distractors"]]])
    if m == 0:
        why = (
            "The whole graph of f is above the x-axis, so f(x) is never negative."
            if a > 0
            else "The whole graph of f is below the x-axis, so f(x) is negative for every x."
        )
    else:
        why = (
            f"f is zero at x = {h - m} and x = {h + m}, and not negative {'outside them' if a > 0 else 'between them'}."
        )
    out.update(
        text=stem,
        figure=fig,
        working=["The square root is defined only where f(x) is not negative.", why, f"So the domain is {text}."],
    )
    return out


# ------------------------------------------------------------------------------------------------------- hard


def _shifted(h: int) -> str:
    """x - 3, x + 3, or x for 0."""
    return "x" if h == 0 else f"x {'-' if h > 0 else '+'} {abs(h)}"


def _hard_domain(rng):
    for _ in attempts(500, "two different zeros"):
        p, q = rng.randint(-6, 6), rng.randint(-6, 6)
        if p < q:
            break
    a = rng.choice((1, 1, -1, 2, -2))
    # a(x - p)(x - q) expanded
    A, B, C = a, -a * (p + q), a * p * q
    f = lambda x: A * x * x + B * x + C  # noqa: E731
    if a > 0:
        spec, other = ("outside", p, q), ("between", p, q)
    else:
        spec, other = ("between", p, q), ("outside", p, q)
    s = interval_set(spec)
    verify_domain(f, s, -14.0, 14.0)
    text = ineq_text(s)
    wrong = [
        (interval_set(other), "gtf-domain-wrong-side"),
        (_strict(spec), "gtf-domain-excludes-zeros"),
        (ALL, "gtf-domain-all-reals"),
        (interval_set(("outside", -q, -p) if a > 0 else ("between", -q, -p)), "gtf-zero-sign"),
    ]
    out = _finish(text, s, wrong)
    where = "outside the zeros" if a > 0 else "between the zeros"
    out.update(
        text=phrase(
            rng,
            (
                "Let f(x) = {f}. Find the largest domain of the function y = sqrt(f(x)), as an inequality.",
                "For f(x) = {f}, state the domain of y = sqrt(f(x)).",
            ),
            f=quad_text(A, B, C),
        ),
        working=[
            f"Factorise: f(x) = {'' if a == 1 else '-' if a == -1 else str(a)}({_shifted(p)})({_shifted(q)}), with zeros x = {p} and x = {q}.",
            f"f(x) is not negative {where}, so the domain is {text}.",
        ],
    )
    return out


def _range_text(lo, hi, var="y") -> str:
    if hi is None:
        return ineq_text(sp.Interval(sp.Integer(lo), sp.oo), var)
    return ineq_text(sp.Interval(sp.Integer(lo), sp.Integer(hi)), var)


def _hard_range(rng):
    s_ = rng.choice((2, 3, 4))
    h = rng.choice((-3, -2, -1, 1, 2, 3))
    if rng.random() < 0.5:
        # f(x) = s^2 - (x - h)^2 = -x^2 + 2hx + (s^2 - h^2)
        f = lambda x: s_ * s_ - (x - h) ** 2  # noqa: E731
        xs = np.linspace(h - s_, h + s_, 40001)
        ys = np.sqrt(np.maximum(f(xs), 0))
        check(abs(ys.max() - s_) < 1e-9 and abs(ys.min()) < 1e-3, "range scan differs from [0, s]")
        text = _range_text(0, s_)
        wrong = [
            (sp.Interval(0, s_ * s_), "gts-range-of-f-not-root"),
            (sp.Interval(0, sp.oo), "gts-range-forgets-upper-bound"),
            (sp.Interval(-s_, s_), "gtf-root-negative-branch"),
            (sp.Interval(0, 2 * s_), "gts-range-doubled"),
        ]
        ftext = quad_text(-1, 2 * h, s_ * s_ - h * h)
        working = [
            f"f(x) = {s_ * s_} - ({_shifted(h)})^2, so the greatest value of f is {s_ * s_}, at x = {h}, and f can be as low as 0 where the root is defined.",
            f"So sqrt(f(x)) takes every value from sqrt(0) = 0 up to sqrt({s_ * s_}) = {s_}: the range is {text}.",
        ]
    else:
        # f(x) = (x - h)^2 + s^2
        f = lambda x: (x - h) ** 2 + s_ * s_  # noqa: E731
        xs = np.linspace(h - 60, h + 60, 120001)
        ys = np.sqrt(f(xs))
        check(abs(ys.min() - s_) < 1e-6, "range scan differs from [s, oo)")
        text = _range_text(s_, None)
        wrong = [
            (sp.Interval(s_ * s_, sp.oo), "gts-range-of-f-not-root"),
            (sp.Interval(0, sp.oo), "gts-range-starts-at-zero"),
            (sp.Interval(-s_, sp.oo), "gtf-root-negative-branch"),
            (sp.Interval(s_, s_ * s_), "gts-range-forgets-upper-bound"),
        ]
        ftext = quad_text(1, -2 * h, h * h + s_ * s_)
        working = [
            f"f(x) = ({_shifted(h)})^2 + {s_ * s_}, so the least value of f is {s_ * s_}, and f has no greatest value.",
            f"So sqrt(f(x)) takes every value from sqrt({s_ * s_}) = {s_} upwards: the range is {text}.",
        ]
    out = _finish(text, None, [(s, t) for s, t in wrong], "y")
    out.update(
        text=phrase(
            rng,
            (
                "Let f(x) = {f}. Find the range of the function y = sqrt(f(x)), as an inequality in y.",
                "For f(x) = {f}, state the range of y = sqrt(f(x)).",
            ),
            f=ftext,
        ),
        working=working,
    )
    return out


def _read(question) -> str:
    """Recompute the answer from the drawn graph alone."""
    pts = curve_points(question.figure)
    if "straight line" in question.text:
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        m = Frac(round((y2 - y1) * 12), round((x2 - x1) * 12)).limit_denominator(6)
        r = Frac(round((x1 - y1 / float(m)) * 12), 12).limit_denominator(12)
        return ineq_text(interval_set(("ge", r) if m > 0 else ("le", r)))
    vx, vy = vertex_of(pts)
    h, k = Frac(round(vx * 12), 12), Frac(round(vy * 12), 12)
    opens_up = pts[0][1] > vy
    if (opens_up and k > 0) or (not opens_up and k < 0):
        return ineq_text(ALL if opens_up else NONE)
    right = next(y for x, y in pts if abs(x - (float(h) + 1)) < 1e-9)
    a = Frac(round((right - float(k)) * 12), 12)
    m = Frac(int(round(math.sqrt(float(-k / a)))))
    return ineq_text(interval_set(("outside", h - m, h + m) if a > 0 else ("between", h - m, h + m)))


figures.register_reader("gt_sqrt_function", _read)


@generator(
    "gt_sqrt_function",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def gt_sqrt_function(rng, tier="easy"):
    """The domain or range of y = sqrt(f(x)); the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier != "hard":
        raise ValueError(f"unknown tier {tier!r}")
    return rng.choice([_hard_domain, _hard_range])(rng)


# -------------------------------------------------------------------------------------------------------- exam


@generator("gt_sqrt_function_exam", level=15, presets={"exam": dict(tier="exam")})
@retry_short
def gt_sqrt_function_exam(rng, tier="exam"):
    """Exam chain on f(x) = s^2 - (x - h)^2: greatest value of f and of sqrt(f), the domain, the exact y-intercept."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a chain with a surd y-intercept"):
        s_ = rng.choice(range(2, 16))
        h = rng.choice([v for v in range(-s_ + 1, s_) if v])
        t = s_ * s_
        c0 = t - h * h
        if c0 > 0 and math.isqrt(c0) ** 2 != c0:
            break
    f = lambda x: t - (x - h) ** 2  # noqa: E731
    dom_set = interval_set(("between", h - s_, h + s_))
    verify_domain(f, dom_set, -40.0, 40.0)
    dom = ineq_text(dom_set)
    ys = np.sqrt(np.maximum(f(np.linspace(h - s_, h + s_, 40001)), 0))
    check(abs(ys.max() - s_) < 1e-9, "greatest value of the root differs")
    a_fields = number_fields(Frac(t), 0)
    a_wrong = [
        int_dist(c0, "gts-reads-constant-as-greatest-value"),
        int_dist(2 * h, "gts-reads-linear-coefficient-as-greatest-value"),
        int_dist(s_, "gts-greatest-value-of-root-for-f"),
        int_dist(t + h * h, "gts-completing-square-adds-instead-of-subtracts"),
    ]
    b_fields = number_fields(Frac(s_), 0)
    b_wrong = [
        int_dist(t, "gts-range-of-f-not-root"),
        int_dist(t // 2 if t % 2 == 0 else t + 1, "gtf-root-as-half"),
        int_dist(-s_, "gtf-root-negative-branch"),
        int_dist(t * t, "gtf-root-squared-instead"),
    ]
    outer = max(d for d in range(1, c0 + 1) if c0 % (d * d) == 0)
    inner = c0 // (outer * outer)
    ans = f"{outer}*sqrt({inner})" if outer > 1 else f"sqrt({inner})"
    root = sp.sqrt(c0)
    check(abs(float(root) - float(sp.sympify(ans))) < 1e-12, "typed surd differs")
    d_wrong = [
        text_dist(str(c0), "gtf-root-not-taken"),
        text_dist(rational_text(Frac(c0, 2)), "gtf-root-as-half"),
        text_dist(f"-{ans}", "gtf-root-negative-branch"),
        text_dist(f"sqrt({t})", "gts-ignores-the-shift"),
    ]
    d_wrong = [w for w in d_wrong if sp.simplify(sp.sympify(w.text) - root) != 0]
    c_wrong = [
        ineq_dist(interval_set(("outside", h - s_, h + s_)), "gtf-domain-wrong-side"),
        ineq_dist(sp.Interval.open(h - s_, h + s_), "gtf-domain-excludes-zeros"),
        ineq_dist(interval_set(("between", -h - s_, -h + s_)), "gtf-zero-sign"),
    ]
    latex = f"{outer}\\sqrt{{{inner}}}" if outer > 1 else f"\\sqrt{{{inner}}}"
    parts = [
        dict(
            prompt="Find the greatest value of f.",
            marks=2,
            sensible_range=(-1000, 1000),
            working=[f"f(x) = {t} - ({_shifted(h)})^2, so its greatest value is {t}, at x = {h}."],
            distractors=pick(a_fields["answer_text"], a_wrong),
            **a_fields,
        ),
        dict(
            prompt="Hence find the greatest value of the function y = sqrt(f(x)).",
            marks=1,
            sensible_range=(0, 100),
            follow_through="sqrt(part_a)",
            working=[f"The greatest value of f is {t}, so the greatest value of sqrt(f(x)) is sqrt({t}) = {s_}."],
            distractors=pick(b_fields["answer_text"], b_wrong),
            **b_fields,
        ),
        dict(
            prompt="Find the domain of the function y = sqrt(f(x)), as an inequality.",
            marks=2,
            working=[
                f"f(x) = {t} - ({_shifted(h)})^2 is not negative between its zeros x = {h - s_} and x = {h + s_}.",
                f"The domain is {dom}.",
            ],
            distractors=pick(dom, c_wrong),
            answer_kind="inequality",
            answer=dom,
            answer_text=dom,
            answer_latex="plain:" + dom,
        ),
        dict(
            prompt="Find the exact value of y when x = 0 on the graph of y = sqrt(f(x)).",
            marks=1,
            working=[f"f(0) = {t} - ({-h})^2 = {c0}.", f"y = sqrt({c0}) = {ans.replace('*', '')}."],
            distractors=pick(ans, d_wrong),
            answer_kind="expression",
            answer=ans,
            answer_text=ans,
            answer_latex=latex,
        ),
    ]
    stem = phrase(
        rng,
        ("The function f is given by f(x) = {f}.", "Consider the function f(x) = {f}."),
        f=quad_text(-1, 2 * h, t - h * h),
    )
    return dict(text=stem, parts=parts)
