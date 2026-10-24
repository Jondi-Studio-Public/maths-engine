"""Generates `sr2b_signed_area` questions (easy, medium, hard) and the exam chain `sr2b_exam_signed_area`.

Skill ``signed-area-definite-integral`` (``integration``, answer ``number``): the areas of the regions between a graph
and the x-axis are given in words, and the learner turns them into values of definite integrals, which count a region
below the axis as negative, and into total areas, which do not. No curve is drawn and no formula is given.

* easy: two regions, one above the axis and one below; the definite integral over the whole interval.
* medium: three regions (above, below, above); one of the integral over the whole interval, the total area, or the
  integral over the last two regions only.
* hard: either an unknown area found from the value of the integral and the total area together (a pair of equations),
  or the integral of ``k f(x) + m`` over an interval made of several regions, where the constant is integrated over the
  width of the interval.
* exam (``sr2b_exam_signed_area``): three regions; (a) the integral over the first two, (b) over the whole interval
  (follows through from (a)), (c) the total area, (d) the integral of ``k f(x) + m`` over the whole interval (follows
  through from (b)).

Each region is also modelled as a rectangle of the same width and signed area, and the answer is re-derived by
integrating those pieces, which is a second route to every value.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, join_terms, phrase
from ._sr2b_kit import int_dist, number_fields, pick, retry_short

_SQ = "square units"


def _boundaries(rng, n_regions: int) -> list[int]:
    start = rng.choice((0, 0, 0, 1, 2))
    xs = [start]
    for _ in range(n_regions):
        xs.append(xs[-1] + rng.randint(1, 3))
    return xs


def _describe(xs, signs, areas) -> str:
    zs = xs[1:-1]
    cross = (
        f"x = {zs[0]}"
        if len(zs) == 1
        else " and ".join(f"x = {z}" for z in zs)
        if len(zs) == 2
        else ", ".join(f"x = {z}" for z in zs[:-1]) + f" and x = {zs[-1]}"
    )
    out = [f"The graph of y = f(x) for {xs[0]} ≤ x ≤ {xs[-1]} crosses the x-axis at {cross}."]
    for i, (s, a) in enumerate(zip(signs, areas)):
        side = "above" if s > 0 else "below"
        out.append(
            f"Between x = {xs[i]} and x = {xs[i + 1]} the graph is {side} the x-axis and the region between it and the axis has area {a} {_SQ}."
        )
    return " ".join(out)


def _integral(xs, signs, areas, lo: int, hi: int, k: int = 1, m: int = 0) -> Fraction:
    """∫ (k f + m) dx over [xs[lo], xs[hi]] by rectangles of equal width and signed area."""
    total = Fraction(0)
    for i in range(lo, hi):
        width = xs[i + 1] - xs[i]
        height = Fraction(signs[i] * areas[i], width)
        total += (k * height + m) * width
    return total


def _signed(signs, areas, lo=0, hi=None) -> int:
    hi = len(areas) if hi is None else hi
    return sum(s * a for s, a in zip(signs[lo:hi], areas[lo:hi]))


def _total(areas, lo=0, hi=None) -> int:
    hi = len(areas) if hi is None else hi
    return sum(areas[lo:hi])


def _fields(value) -> dict:
    return number_fields(Fraction(value), 0)


def _dists(answer: int, cands: list[tuple[int, str]]):
    return pick(str(answer), [int_dist(v, tag) for v, tag in cands if v != answer])


def _draw_areas(rng, n: int, lo=2, hi=12):
    for _ in attempts(500, "region areas with distinct signed and total areas"):
        areas = [rng.randint(lo, hi) for _ in range(n)]
        signs = [1, -1, 1][:n] if rng.random() < 0.7 else [-1, 1, -1][:n]
        if _signed(signs, areas) != _total(areas) and _signed(signs, areas) != 0 and len(set(areas)) == n:
            return signs, areas
    raise AssertionError  # pragma: no cover


# -- easy -----------------------------------------------------------------------------------------------------


def _easy(rng):
    xs = _boundaries(rng, 2)
    signs, areas = _draw_areas(rng, 2)
    ans = _signed(signs, areas)
    check(_integral(xs, signs, areas, 0, 2) == ans, "integral differs from the rectangle route")
    a, b = areas
    cands = [
        (_total(areas), "sr2b-total-area-for-integral"),
        (-ans, "sr2b-signed-area-sign-reversed"),
        (areas[0], "sr2b-one-region-only"),
        (areas[1], "sr2b-one-region-only"),
    ]
    stem = (
        _describe(xs, signs, areas)
        + " "
        + phrase(
            rng,
            (
                "Find the value of ∫ from {lo} to {hi} of f(x) dx.",
                "Evaluate the definite integral of f(x) from x = {lo} to x = {hi}.",
            ),
            lo=xs[0],
            hi=xs[-1],
        )
    )
    return dict(
        text=stem,
        working=[
            "A definite integral counts area above the x-axis as positive and area below it as negative.",
            f"∫ f(x) dx = {signs[0] * a} + ({signs[1] * b}) = {ans}.",
        ],
        distractors=_dists(ans, cands),
        **_fields(ans),
    )


# -- medium ---------------------------------------------------------------------------------------------------


def _medium(rng):
    xs = _boundaries(rng, 3)
    signs, areas = _draw_areas(rng, 3)
    kind = rng.choice(("whole", "total", "last_two"))
    P, Q, R = areas
    s1, s2, s3 = signs
    if kind == "whole":
        ans = _signed(signs, areas)
        check(_integral(xs, signs, areas, 0, 3) == ans, "integral differs")
        ask = f"Find the value of ∫ from {xs[0]} to {xs[3]} of f(x) dx."
        cands = [
            (_total(areas), "sr2b-total-area-for-integral"),
            (P + Q - R, "sr2b-sign-pattern-wrong"),
            (P - Q - R if s1 > 0 else -P + Q + R, "sr2b-sign-pattern-wrong"),
            (-ans, "sr2b-signed-area-sign-reversed"),
            (s1 * P + s2 * Q, "sr2b-region-left-out"),
        ]
        working = [f"∫ f(x) dx = {s1 * P} + ({s2 * Q}) + ({s3 * R}) = {ans}."]
    elif kind == "total":
        ans = _total(areas)
        check(sum(abs(_integral(xs, signs, areas, i, i + 1)) for i in range(3)) == ans, "total area differs")
        ask = f"Find the total area enclosed between the graph of y = f(x) and the x-axis from x = {xs[0]} to x = {xs[3]}."
        cands = [
            (_signed(signs, areas), "sr2b-integral-for-total-area"),
            (abs(_signed(signs, areas)), "sr2b-integral-for-total-area"),
            (P + Q, "sr2b-region-left-out"),
            (Q + R, "sr2b-region-left-out"),
        ]
        working = [f"Area is never negative, so the total area is {P} + {Q} + {R} = {ans}."]
    else:
        ans = _signed(signs, areas, 1, 3)
        check(_integral(xs, signs, areas, 1, 3) == ans, "integral differs")
        ask = f"Find the value of ∫ from {xs[1]} to {xs[3]} of f(x) dx."
        cands = [
            (Q + R, "sr2b-total-area-for-integral"),
            (_signed(signs, areas), "sr2b-wrong-interval"),
            (-ans, "sr2b-signed-area-sign-reversed"),
            (s2 * Q, "sr2b-region-left-out"),
        ]
        working = [
            f"The interval from x = {xs[1]} to x = {xs[3]} holds the last two regions: ({s2 * Q}) + ({s3 * R}) = {ans}."
        ]
    if ans == 0:
        return _medium(rng)
    return dict(
        text=_describe(xs, signs, areas) + " " + ask, working=working, distractors=_dists(ans, cands), **_fields(ans)
    )


# -- hard -----------------------------------------------------------------------------------------------------


def _hard_unknown(rng):
    for _ in attempts(1000, "areas with an integer unknown from the integral and total"):
        xs = _boundaries(rng, 3)
        signs = [1, -1, 1] if rng.random() < 0.7 else [-1, 1, -1]
        P, Q, R = (rng.randint(2, 12) for _ in range(3))
        integral, total = _signed(signs, [P, Q, R]), _total([P, Q, R])
        if integral == 0 or integral == total or len({P, Q, R}) < 3:
            continue
        break
    ask_region = rng.choice((1, 2))  # ask the area of the middle region or the last one, the first being given
    sidewise = ["above" if s > 0 else "below" for s in signs]
    stem = (
        f"The graph of y = f(x) for {xs[0]} ≤ x ≤ {xs[3]} crosses the x-axis at x = {xs[1]} and x = {xs[2]}, and is {sidewise[0]} the x-axis "
        f"between x = {xs[0]} and x = {xs[1]}, {sidewise[1]} it between x = {xs[1]} and x = {xs[2]}, and {sidewise[2]} it between "
        f"x = {xs[2]} and x = {xs[3]}. The region between the graph and the axis from x = {xs[0]} to x = {xs[1]} has area {P} {_SQ}. "
        f"The total area between the graph and the axis from x = {xs[0]} to x = {xs[3]} is {total} {_SQ}, and ∫ from {xs[0]} to {xs[3]} of f(x) dx = {integral}. "
    )
    # second route: solve the two equations for the unknown areas directly
    unknown_sum = total - P
    diff = integral - signs[0] * P  # = s2*Q + s3*R
    q_found = Fraction(unknown_sum * signs[2] - diff, signs[2] - signs[1]) if signs[2] != signs[1] else None
    check(q_found == Q, "system solution differs from the drawn area")
    if ask_region == 1:
        ans, label = Q, f"from x = {xs[1]} to x = {xs[2]}"
        cands = [
            (R, "sr2b-regions-swapped"),
            (abs(integral), "sr2b-integral-for-area"),
            (total - P, "sr2b-two-regions-combined"),
            (Fraction(total - P, 2), "sr2b-regions-assumed-equal"),
            (total - P - abs(integral - signs[0] * P), "sr2b-signed-and-total-mixed"),
        ]
    else:
        ans, label = R, f"from x = {xs[2]} to x = {xs[3]}"
        cands = [
            (Q, "sr2b-regions-swapped"),
            (abs(integral), "sr2b-integral-for-area"),
            (total - P, "sr2b-two-regions-combined"),
            (Fraction(total - P, 2), "sr2b-regions-assumed-equal"),
            (total - P - Q - P, "sr2b-signed-and-total-mixed"),
        ]
    cands = [(int(v), t) for v, t in cands if Fraction(v).denominator == 1 and int(v) > 0 and int(v) != ans]
    return dict(
        text=stem + f"Find the area of the region between the graph and the x-axis {label}.",
        working=[
            f"Let the unknown areas be Q (middle) and R (last). Total area: {P} + Q + R = {total}, so Q + R = {total - P}.",
            f"Integral: {join_terms([(signs[0] * P, ''), (signs[1], 'Q'), (signs[2], 'R')])} = {integral}, so {join_terms([(signs[1], 'Q'), (signs[2], 'R')])} = {integral - signs[0] * P}.",
            f"Solving gives Q = {Q} and R = {R}, so the area asked for is {ans}.",
        ],
        distractors=_dists(ans, cands),
        **_fields(ans),
    )


def _hard_transform(rng):
    xs = _boundaries(rng, 3)
    signs, areas = _draw_areas(rng, 3)
    k = rng.choice((2, 3, 4, -2, -3))
    m = rng.choice((-2, -1, 1, 2, 3, 5))
    lo = rng.choice((0, 1))
    base = _signed(signs, areas, lo, 3)
    width = xs[3] - xs[lo]
    ans = k * base + m * width
    check(_integral(xs, signs, areas, lo, 3, k, m) == ans, "transformed integral differs from the rectangle route")
    if ans == 0:
        return _hard_transform(rng)
    cands = [
        (k * _total(areas, lo, 3) + m * width, "sr2b-total-area-for-integral"),
        (k * base + m, "sr2b-constant-not-integrated-over-width"),
        (k * base, "sr2b-constant-term-dropped"),
        (base + m * width, "sr2b-scale-factor-dropped"),
        (k * base - m * width, "sr2b-constant-sign-reversed"),
    ]
    stem = (
        _describe(xs, signs, areas)
        + " "
        + f"Find the value of ∫ from {xs[lo]} to {xs[3]} of ({k}f(x) {'+' if m > 0 else '-'} {abs(m)}) dx."
    )
    return dict(
        text=stem,
        working=[
            f"∫ f(x) dx from {xs[lo]} to {xs[3]} = {base}.",
            f"∫ ({k}f(x) {'+' if m > 0 else '-'} {abs(m)}) dx = {k} × ({base}) + ({m}) × {width} = {ans}.",
        ],
        distractors=_dists(ans, cands),
        **_fields(ans),
    )


@generator(
    "sr2b_signed_area",
    level=13,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def sr2b_signed_area(rng, tier="easy"):
    """Definite integrals and areas from the areas of regions; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_unknown, _hard_transform])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("sr2b_exam_signed_area", level=13, presets={"exam": dict(tier="exam")})
@retry_short
def sr2b_exam_signed_area(rng, tier="exam"):
    """Exam chain: integral over two regions, over all three, the total area, and the integral of k f + m."""
    check(tier == "exam", f"unknown tier {tier!r}")
    xs = _boundaries(rng, 3)
    for _ in attempts(500, "exam regions"):
        signs, areas = _draw_areas(rng, 3)
        if _signed(signs, areas, 0, 2) != 0 and _signed(signs, areas) != 0:
            break
    P, Q, R = areas
    k = rng.choice((2, 3, 4, -2, -3))
    m = rng.choice((-2, -1, 1, 2, 3, 5))
    width = xs[3] - xs[0]
    a = _signed(signs, areas, 0, 2)
    b = _signed(signs, areas)
    c = _total(areas)
    d = k * b + m * width
    check(_integral(xs, signs, areas, 0, 2) == a and _integral(xs, signs, areas, 0, 3) == b, "integrals differ")
    check(_integral(xs, signs, areas, 0, 3, k, m) == d, "transformed integral differs")
    cands_a = [
        (P + Q, "sr2b-total-area-for-integral"),
        (-a, "sr2b-signed-area-sign-reversed"),
        (signs[0] * P, "sr2b-one-region-only"),
        (signs[1] * Q, "sr2b-one-region-only"),
    ]
    cands_b = [
        (c, "sr2b-total-area-for-integral"),
        (a - signs[2] * R, "sr2b-sign-pattern-wrong"),
        (a, "sr2b-region-left-out"),
        (-b, "sr2b-signed-area-sign-reversed"),
    ]
    cands_c = [
        (b, "sr2b-integral-for-total-area"),
        (abs(b), "sr2b-integral-for-total-area"),
        (P + Q, "sr2b-region-left-out"),
        (Q + R, "sr2b-region-left-out"),
    ]
    cands_d = [
        (k * c + m * width, "sr2b-total-area-for-integral"),
        (k * b + m, "sr2b-constant-not-integrated-over-width"),
        (k * b, "sr2b-constant-term-dropped"),
        (b + m * width, "sr2b-scale-factor-dropped"),
        (k * b - m * width, "sr2b-constant-sign-reversed"),
    ]
    return dict(
        text=_describe(xs, signs, areas),
        parts=[
            dict(
                prompt=f"Find the value of ∫ from {xs[0]} to {xs[2]} of f(x) dx.",
                marks=1,
                sensible_range=(-100, 100),
                working=[f"{signs[0] * P} + ({signs[1] * Q}) = {a}."],
                distractors=_dists(a, cands_a),
                **_fields(a),
            ),
            dict(
                prompt=f"Hence find the value of ∫ from {xs[0]} to {xs[3]} of f(x) dx.",
                marks=1,
                sensible_range=(-100, 100),
                follow_through=f"part_a+({signs[2] * R})",
                working=[f"Add the last region, counted as {signs[2] * R}: {a} + ({signs[2] * R}) = {b}."],
                distractors=_dists(b, cands_b),
                **_fields(b),
            ),
            dict(
                prompt=f"Find the total area between the graph of y = f(x) and the x-axis from x = {xs[0]} to x = {xs[3]}.",
                marks=2,
                sensible_range=(0, None),
                working=[f"{P} + {Q} + {R} = {c}."],
                distractors=_dists(c, cands_c),
                **_fields(c),
            ),
            dict(
                prompt=f"Hence find the value of ∫ from {xs[0]} to {xs[3]} of ({k}f(x) {'+' if m > 0 else '-'} {abs(m)}) dx.",
                marks=2,
                sensible_range=(-300, 300),
                follow_through=f"({k})*part_b+({m * width})",
                working=[f"{k} × {b} + ({m}) × {width} = {d}."],
                distractors=_dists(d, cands_d),
                **_fields(d),
            ),
        ],
    )
