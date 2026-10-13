"""Generates `sr2b_trig_graph_features` questions (easy, medium, hard) and the exam chain
`sr2b_exam_trig_graph_features`.

Skill ``reciprocal-and-trig-graph-features`` (``trigonometry``, answer ``expression``; a count is a whole number, which
the expression kind also grades): features of the graphs of ``y = a sec(bx)``, ``a cosec(bx)`` and ``a tan(bx)``, read
from the equation without drawing anything. Angles are in radians and exact values are typed in terms of pi.

* easy: the period of ``y = a f(bx)`` for ``f`` one of sec, cosec or tan and ``b`` a whole number or a fraction. Sec and
  cosec repeat every ``2 pi / b`` and tan every ``pi / b``, so the base period has to be recalled as well as divided.
* medium: how many vertical asymptotes the graph has for ``0 <= x <= 2 pi``. A cosec graph has one at each end of the
  interval; sec and tan do not reach theirs there.
* hard: either the smallest positive ``x`` at which ``y = a sec(b(x - s)) + d`` or ``a cosec(b(x - s)) + d`` has a
  vertical asymptote (the shift moves it, and an earlier branch of the pattern may already lie in positive x), or the
  larger of the two values between which a ``y = a sec(bx) + d`` or ``a cosec(bx) + d`` graph takes no value.
* exam (``sr2b_exam_trig_graph_features``): for ``f(x) = a cosec(bx) + d``, (a) the period, (b) the smallest positive
  asymptote (half a period; follows through), (c) the number of asymptotes for ``0 <= x <= 2 pi`` (follows through
  from (b)), (d) the larger of the two values f never takes between.

Each answer is drawn from exact `Fraction` multiples of pi, then confirmed by a numeric scan of the function itself
(sign changes of the cosine or sine, a shifted copy of the graph).
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..answers import Distractor
from ..core import attempts, check, generator, phrase
from ._sr2b_kit import int_dist, pi_latex, pi_shown, pi_text, pick, retry_short

_FN = ("sec", "cosec", "tan")


def _bx(b: Fraction) -> str:
    if b.denominator == 1:
        return "x" if b == 1 else f"{b.numerator}x"
    return f"x/{b.denominator}" if b.numerator == 1 else f"{b.numerator}x/{b.denominator}"


def _lead(a: int) -> str:
    return "" if a == 1 else "-" if a == -1 else f"{a} "


def _num(fn: str, theta: float) -> float:
    """The function at an angle, by floats (second route; no pi-multiple bookkeeping)."""
    s, c = math.sin(theta), math.cos(theta)
    return {
        "sec": 1 / c if abs(c) > 1e-12 else math.inf,
        "cosec": 1 / s if abs(s) > 1e-12 else math.inf,
        "tan": s / c if abs(c) > 1e-12 else math.inf,
    }[fn]


def _base(fn: str) -> int:
    """The period of sec and cosec is 2 pi; tan's is pi (in units of pi)."""
    return 1 if fn == "tan" else 2


def _check_period(fn: str, a: int, b: Fraction, period: Fraction) -> None:
    """The graph repeats after the period and not after any smaller simple fraction of it."""
    g = lambda x: a * _num(fn, float(b) * x)  # noqa: E731
    samples = [0.123 + 0.377 * i for i in range(7)]
    p = math.pi * float(period)
    for x in samples:
        check(abs(g(x + p) - g(x)) < 1e-6 * (1 + abs(g(x))), "the graph does not repeat after the period")
    for div in (2, 3, 4):
        q = p / div
        check(any(abs(g(x + q) - g(x)) > 1e-3 * (1 + abs(g(x))) for x in samples), "a shorter period also works")


def _pdist(coeff, tag: str) -> Distractor:
    t = pi_text(coeff)
    return Distractor(t, tag, t)


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy(rng):
    fn = rng.choice(_FN)
    a = rng.choice((-4, -3, -2, 1, 2, 3, 5))
    b = rng.choice(
        [Fraction(v) for v in (2, 3, 4, 5, 6)] + [Fraction(1, 2), Fraction(1, 3), Fraction(3, 2), Fraction(2, 3)]
    )
    base = _base(fn)
    period = Fraction(base) / b
    _check_period(fn, a, b, period)
    other = Fraction(3 - base)  # the base period of the other family (1 vs 2)
    cands = [
        _pdist(other / b, "sr2b-period-wrong-base"),
        _pdist(Fraction(base) * b, "sr2b-period-multiplied-by-b"),
        _pdist(Fraction(base), "sr2b-period-b-ignored"),
        _pdist(other * b, "sr2b-period-multiplied-by-b"),
    ]
    ans = pi_text(period)
    stem = phrase(
        rng,
        (
            "State the period of the graph of y = {f}.",
            "Find the period of y = {f}. Give your answer in radians, in terms of pi.",
            "What is the period, in radians, of the function y = {f}? Give an exact value.",
        ),
        f=f"{_lead(a)}{fn}({_bx(b)})",
    )
    return dict(
        text=stem,
        answer_kind="expression",
        answer=ans,
        answer_text=ans,
        answer_latex=pi_latex(period),
        working=[
            f"The graph of y = {fn}(x) repeats every {'π' if base == 1 else '2π'}.",
            f"The period of y = {fn}({_bx(b)}) is that basic period divided by the coefficient of x, {b}, which gives {pi_shown(period)}.",
        ],
        distractors=pick(ans, cands),
    )


# -- medium -------------------------------------------------------------------------------------------------------


def _asymptote_count(fn: str, b: Fraction) -> int:
    """Vertical asymptotes of a f(bx) for 0 <= x <= 2 pi: the zeros of cos(bx) (sec, tan) or sin(bx) (cosec)."""
    offset = Fraction(0) if fn == "cosec" else Fraction(1, 2)  # bx = (k + offset) * pi
    return sum(1 for k in range(-60, 120) if 0 <= (k + offset) / b <= 2)


def _scan_count(fn: str, b: Fraction) -> int:
    """Second route: sign changes of the denominator function on an offset grid, plus its exact zeros at the ends."""
    den = math.sin if fn == "cosec" else math.cos
    n = 4000
    vals = [den(float(b) * 2 * math.pi * (i + 0.5) / n) for i in range(n)]
    count = sum(1 for u, v in zip(vals, vals[1:]) if u * v < 0)
    count += sum(1 for end in (0.0, 2 * math.pi) if abs(den(float(b) * end)) < 1e-9)
    return count


def _medium(rng):
    fn = rng.choice(_FN)
    a = rng.choice((-3, -2, 1, 2, 3))
    b = rng.choice([Fraction(v) for v in (2, 3, 4, 5)] + [Fraction(3, 2), Fraction(5, 2)])
    count = _asymptote_count(fn, b)
    check(count == _scan_count(fn, b), "asymptote count differs from the scan")
    periods = 2 / (Fraction(_base(fn)) / b)  # whole or half periods in 0..2 pi
    per_period = 1 if fn == "tan" else 2
    cands = [
        int_dist(count - 1, "sr2b-asymptote-count-endpoints-missed")
        if fn == "cosec"
        else int_dist(count + 1, "sr2b-asymptote-count-endpoints-included"),
        int_dist(count + 1, "sr2b-asymptote-count-endpoints-included")
        if fn == "cosec"
        else int_dist(count - 1, "sr2b-asymptote-count-off-by-one"),
        int_dist(per_period, "sr2b-asymptote-count-one-period-only"),
        int_dist(int(periods), "sr2b-asymptote-count-periods-not-asymptotes")
        if periods.denominator == 1
        else int_dist(count + 2, "sr2b-asymptote-count-off-by-one"),
        int_dist(2 * count, "sr2b-asymptote-count-doubled"),
    ]
    cands = [c for c in cands if int(c.text) > 0]
    stem = phrase(
        rng,
        (
            "How many vertical asymptotes does the graph of y = {f} have for 0 ≤ x ≤ 2π?",
            "For 0 ≤ x ≤ 2π, how many vertical asymptotes does y = {f} have?",
        ),
        f=f"{_lead(a)}{fn}({_bx(b)})",
    )
    where = "cos" if fn != "cosec" else "sin"
    return dict(
        text=stem,
        answer_kind="number",
        answer=count,
        answer_text=str(count),
        answer_latex=str(count),
        working=[
            f"The asymptotes occur where {where}({_bx(b)}) = 0.",
            f"Listing these values of x from 0 to 2π (including either end if it is one) gives {count} asymptotes.",
        ],
        distractors=pick(str(count), cands),
    )


# -- hard ---------------------------------------------------------------------------------------------------------


def _smallest_positive(fn: str, b: Fraction, shift: Fraction) -> Fraction:
    """Smallest positive asymptote (in units of pi) of a f(b(x - shift)): b(x - shift) = (k + offset) pi."""
    offset = Fraction(0) if fn == "cosec" else Fraction(1, 2)
    xs = [shift + (k + offset) / b for k in range(-40, 40)]
    return min(x for x in xs if x > 0)


def _scan_first(fn: str, b: Fraction, shift: Fraction, first: Fraction) -> None:
    """Second route: the denominator function is zero at `first` and changes sign nowhere in (0, first)."""
    den = math.sin if fn == "cosec" else math.cos
    f = lambda x: den(float(b) * (x - math.pi * float(shift)))  # noqa: E731
    check(abs(f(math.pi * float(first))) < 1e-9, "not an asymptote")
    n = 3000
    top = math.pi * float(first)
    vals = [f(top * (i + 0.5) / n) for i in range(n)]
    check(all(u * v > 0 for u, v in zip(vals, vals[1:])), "an earlier positive asymptote exists")
    check(abs(f(0.0)) > 1e-6, "an asymptote at x = 0")


def _hard_asymptote(rng):
    for _ in attempts(1000, "a shifted reciprocal graph whose first positive asymptote needs stepping back"):
        fn = rng.choice(("sec", "cosec"))
        b = Fraction(rng.choice((2, 3, 4)))
        shift = rng.choice(
            [Fraction(p, q) for p, q in ((1, 6), (1, 4), (1, 3), (1, 2), (2, 3), (3, 4), (5, 6), (1, 1))]
        )
        a = rng.choice((-3, -2, 2, 3))
        d = rng.choice((-3, -2, -1, 1, 2, 3))
        offset = Fraction(0) if fn == "cosec" else Fraction(1, 2)
        first = _smallest_positive(fn, b, shift)
        k0 = shift + offset / b  # the k = 0 member of the pattern
        if first == k0 and rng.random() < 0.7:
            continue  # usually require stepping back from the obvious member, so the shift does real work
        # no asymptote exactly at 0
        if any((shift + (k + offset) / b) == 0 for k in range(-40, 40)):
            continue
        cands = [
            _pdist(k0, "sr2b-asymptote-first-member-of-pattern"),
            _pdist(offset / b, "sr2b-asymptote-shift-ignored"),
            _pdist(shift - offset / b, "sr2b-asymptote-shift-direction"),
            _pdist(shift, "sr2b-asymptote-at-the-shift"),
            _pdist(shift + (offset + 1) / b, "sr2b-asymptote-next-member"),
            _pdist(first + 1 / b, "sr2b-asymptote-next-member"),
        ]
        cands = [c for c in cands if not c.text.startswith("-") and c.text != "0"]
        if len({c.text for c in cands if c.text != pi_text(first)}) >= 3:
            break
    _scan_first(fn, b, shift, first)
    nb = f"{b}(x - {pi_shown(shift)})"
    expr = f"y = {_lead(a)}{fn}({nb}) {'+' if d > 0 else '-'} {abs(d)}"
    ans = pi_text(first)
    stem = phrase(
        rng,
        (
            "Find the smallest positive value of x at which the graph of {e} has a vertical asymptote. Give an exact value.",
            "The graph of {e} has vertical asymptotes. What is the smallest positive x-value at which one occurs? Give an exact value.",
        ),
        e=expr,
    )
    zero = "cos" if fn == "sec" else "sin"
    return dict(
        text=stem,
        answer_kind="expression",
        answer=ans,
        answer_text=ans,
        answer_latex=pi_latex(first),
        working=[
            f"Asymptotes occur where {zero}({nb}) = 0, that is {nb} = {'π/2 + kπ' if fn == 'sec' else 'kπ'} for whole numbers k.",
            f"So x = {pi_shown(shift)} + ({'π/2 + kπ' if fn == 'sec' else 'kπ'})/{b}; the smallest positive value (check k = -1, -2, ...) is {pi_shown(first)}.",
        ],
        distractors=pick(ans, cands),
    )


def _hard_gap(rng):
    for _ in attempts(200, "a reciprocal gap whose edge has three wrong values"):
        fn = rng.choice(("sec", "cosec"))
        a = rng.choice((-4, -3, -2, 2, 3, 4, 5))
        b = rng.choice((2, 3, 4))
        d = rng.choice((-4, -3, -2, -1, 1, 2, 3))
        top = d + abs(a)
        if len({v for v in (d - abs(a), abs(a), d + a, d - a, a * d) if v not in (top, 0)}) >= 3:
            break
    # second route: |a f(bx)| >= |a| on a fine grid, with equality reached
    vals = [a * _num(fn, b * 0.0137 * i) for i in range(1, 600)]
    finite = [v for v in vals if math.isfinite(v)]
    check(min(abs(v) for v in finite) >= abs(a) - 1e-9, "the graph enters the gap")
    check(min(abs(v) for v in finite) < abs(a) + 0.05, "the graph does not reach the edge of the gap")
    expr = f"y = {_lead(a)}{fn}({b}x) {'+' if d > 0 else '-'} {abs(d)}"
    cands = [
        int_dist(d - abs(a), "sr2b-reciprocal-gap-wrong-edge"),
        int_dist(abs(a), "sr2b-reciprocal-gap-shift-ignored"),
        int_dist(d + a, "sr2b-reciprocal-gap-sign-of-a"),
        int_dist(a * d, "sr2b-reciprocal-gap-product"),
        int_dist(d - a, "sr2b-reciprocal-gap-sign-of-a"),
    ]
    cands = [c for c in cands if int(c.text) not in (top, 0)]
    stem = f"The graph of {expr} takes no values between two numbers. Find the larger of these two numbers."
    return dict(
        text=stem,
        answer_kind="number",
        answer=top,
        answer_text=str(top),
        answer_latex=str(top),
        working=[
            f"{fn}(θ) is never between -1 and 1, so {_lead(a)}{fn}({b}x) is never between {-abs(a)} and {abs(a)}.",
            f"The constant term moves the gap by {d}, to between {d - abs(a)} and {d + abs(a)}; the larger is {top}.",
        ],
        distractors=pick(str(top), cands),
    )


@generator(
    "sr2b_trig_graph_features",
    level=13,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def sr2b_trig_graph_features(rng, tier="easy"):
    """One feature of a reciprocal or tangent graph; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_asymptote, _hard_asymptote, _hard_gap])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("sr2b_exam_trig_graph_features", level=13, presets={"exam": dict(tier="exam")})
@retry_short
def sr2b_exam_trig_graph_features(rng, tier="exam"):
    """Exam chain on y = a cosec(bx) + d: period, first positive asymptote, asymptote count, edge of the gap."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "an exam cosecant graph whose gap edge has three wrong values"):
        a = rng.choice((-3, -2, 2, 3, 4))
        b = rng.choice((2, 3, 4))
        d = rng.choice((-3, -2, -1, 1, 2))
        wrong = {v for v in (d - abs(a), abs(a), d + a, d - a, a * d) if v != d + abs(a) and v != 0}
        if len(wrong) >= 3:
            break
    period = Fraction(2, b)
    first = Fraction(1, b)
    count = 2 * b + 1
    top = d + abs(a)
    _check_period("cosec", a, Fraction(b), period)
    check(count == _asymptote_count("cosec", Fraction(b)) == _scan_count("cosec", Fraction(b)), "count differs")
    check(first == _smallest_positive("cosec", Fraction(b), Fraction(0)), "first asymptote differs")
    expr = f"f(x) = {_lead(a)}cosec({b}x) {'+' if d > 0 else '-'} {abs(d)}"
    p_a = [
        _pdist(Fraction(1, b), "sr2b-period-wrong-base"),
        _pdist(Fraction(2 * b), "sr2b-period-multiplied-by-b"),
        _pdist(Fraction(2), "sr2b-period-b-ignored"),
    ]
    p_b = [
        _pdist(Fraction(2, b), "sr2b-asymptote-whole-period"),
        _pdist(Fraction(1, 2 * b), "sr2b-asymptote-quarter-period"),
        _pdist(Fraction(1), "sr2b-asymptote-b-ignored"),
    ]
    p_c = [
        int_dist(2 * b, "sr2b-asymptote-count-endpoints-missed"),
        int_dist(2 * b + 2, "sr2b-asymptote-count-endpoints-included"),
        int_dist(b, "sr2b-asymptote-count-periods-not-asymptotes"),
        int_dist(4 * b + 1, "sr2b-asymptote-count-doubled"),
    ]
    p_d = [
        int_dist(d - abs(a), "sr2b-reciprocal-gap-wrong-edge"),
        int_dist(abs(a), "sr2b-reciprocal-gap-shift-ignored"),
        int_dist(d + a, "sr2b-reciprocal-gap-sign-of-a"),
        int_dist(d - a, "sr2b-reciprocal-gap-sign-of-a"),
        int_dist(a * d, "sr2b-reciprocal-gap-product"),
    ]
    return dict(
        text=phrase(
            rng,
            (
                "Consider the function {expr}, for 0 ≤ x ≤ 2π.",
                "Let {expr}, where 0 ≤ x ≤ 2π.",
                "A function is given by {expr}, for 0 ≤ x ≤ 2π.",
            ),
            expr=expr,
        ),
        parts=[
            dict(
                prompt="Find the period of f. Give an exact value in radians.",
                answer_kind="expression",
                answer=pi_text(period),
                answer_text=pi_text(period),
                answer_latex=pi_latex(period),
                marks=1,
                working=[f"cosec(θ) has period 2π, so cosec({b}x) has period 2π/{b} = {pi_shown(period)}."],
                distractors=pick(pi_text(period), [c for c in p_a if c.text != pi_text(period)]),
            ),
            dict(
                prompt="Find the smallest positive value of x at which the graph has a vertical asymptote. Give an exact value.",
                answer_kind="expression",
                answer=pi_text(first),
                answer_text=pi_text(first),
                answer_latex=pi_latex(first),
                marks=2,
                follow_through="part_a/2",
                working=[
                    f"cosec has asymptotes where sin({b}x) = 0, which is twice per period; the first positive one is half a period, {pi_shown(first)}."
                ],
                distractors=pick(pi_text(first), [c for c in p_b if c.text != pi_text(first)]),
            ),
            dict(
                prompt="How many vertical asymptotes does the graph have in the interval 0 ≤ x ≤ 2π?",
                answer_kind="number",
                answer=count,
                answer_text=str(count),
                answer_latex=str(count),
                marks=2,
                sensible_range=(0, 100),
                follow_through="2*pi/part_b+1",
                working=[
                    f"The asymptotes are {pi_shown(first)} apart, starting at x = 0 and ending at x = 2π: 2π ÷ {pi_shown(first)} = {2 * b} gaps, so {count} asymptotes."
                ],
                distractors=pick(str(count), p_c),
            ),
            dict(
                prompt="The graph takes no values between two numbers. Find the larger of these two numbers.",
                answer_kind="number",
                answer=top,
                answer_text=str(top),
                answer_latex=str(top),
                marks=2,
                sensible_range=(-100, 100),
                working=[
                    f"cosec(θ) is never between -1 and 1, so the graph never lies between {d - abs(a)} and {d + abs(a)}; the larger is {top}."
                ],
                distractors=pick(str(top), [c for c in p_d if int(c.text) != top and int(c.text) != 0]),
            ),
        ],
    )
