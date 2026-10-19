"""Generates `r4e_curve_features` questions (easy, medium, hard) and the exam chain `r4e_curve_features_chain`.

Skill `curve-features-from-derivative-signs` (answer `inequality`): read the shape of a curve from the signs of f'
and f''. Answers are the largest open intervals on which f is increasing or decreasing, or on which its graph is
concave up or concave down.

* easy: f' is given in factored form with two distinct roots, f'(x) = k(x - p)(x - q); where is f increasing, or
  decreasing?
* medium: either f' has a squared factor, k(x - p)(x - q)^2, so its sign does not change at q and f keeps
  increasing (or decreasing) through the stationary point there; or f' is a given quadratic and the concavity has to
  be found from f'' = (f')'.
* hard: both signs at once on f' = k(x - p)(x - q): where is f decreasing and concave up (or increasing and concave
  down)? The answer is an intersection with an endpoint at the vertex of f', often a half.
* exam: a cubic with integer stationary points: their x-values, the intervals of increase, the x-coordinate of the
  point of inflection and its y-coordinate, following through from the x-coordinate.

Polynomials are coefficient lists in exact arithmetic (`r4e_common.pev`, `pder`). An interval answer comes from the
sign of the polynomial at a point in each gap between its distinct real roots (adjacent gaps of the same sign are
joined, because the root between them is an even-multiplicity root where the sign does not change). It is
cross-checked by testing the sign of the polynomial on a fine grid of rational points against the finished answer.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, phrase
from .r4e_common import (
    Redraw,
    draw,
    fraction_part,
    inequality_part,
    interval_text,
    list_part,
    num,
    pder,
    pev,
    poly_text,
    single,
)



def _sign_of(k) -> str:
    """What f' has the sign of, for f' = k times a product of factors."""
    if k == 1:
        return "the product of the factors"
    if k == -1:
        return "minus the product of the factors"
    return f"{k} times the product of the factors"

def polymul(a, b) -> list:
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += Fraction(x) * Fraction(y)
    return out


def from_roots(k, roots) -> list:
    """Coefficients (highest first) of k times the product of (x - r) over `roots` (repeated roots repeated)."""
    out = [Fraction(k)]
    for r in roots:
        out = polymul(out, [Fraction(1), Fraction(-r)])
    return out


def signed_intervals(coeffs, roots, positive: bool):
    """The largest open intervals on which the polynomial is >= 0 (or <= 0) and zero only at isolated points, as
    `(low, high)` pairs with None for unbounded. `roots` are its distinct real roots."""
    xs = sorted(set(Fraction(r) for r in roots))
    edges = [None, *xs, None]
    runs, current = [], None
    for lo, hi in zip(edges, edges[1:]):
        if lo is None and hi is None:
            probe = Fraction(0)
        elif lo is None:
            probe = hi - 1
        elif hi is None:
            probe = lo + 1
        else:
            probe = (lo + hi) / 2
        value = pev(coeffs, probe)
        if value != 0 and (value > 0) == positive:
            current = (current[0], hi) if current else (lo, hi)
        elif current:
            runs.append(current)
            current = None
    if current:
        runs.append(current)
    return runs


def _check_grid(coeffs, positive: bool, ivs) -> None:
    """Second route: on a grid of quarter-integer points, the polynomial is >= 0 (or <= 0) exactly where the answer
    says, apart from the roots themselves."""
    for i in range(-48, 49):
        point = Fraction(i, 4)
        inside = any((lo is None or point > lo) and (hi is None or point < hi) for lo, hi in ivs)
        value = pev(coeffs, point)
        if value == 0:
            continue
        check(((value > 0) == positive) == inside, "an interval disagrees with the sign of the polynomial")


def quad_roots(a, b, c):
    """Rational roots of ax^2 + bx + c, or None when they are irrational or absent."""
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    root = int(disc**0.5 + 0.5)
    if root * root != disc:
        return None
    return sorted({Fraction(-b - root, 2 * a), Fraction(-b + root, 2 * a)})


def reflect(ivs):
    """The intervals negated (x to -x): every endpoint read with the wrong sign."""
    return [(None if hi is None else -hi, None if lo is None else -lo) for lo, hi in ivs]


def scaled(ivs, factor):
    return [(None if lo is None else lo * factor, None if hi is None else hi * factor) for lo, hi in ivs]


def _closed(ivs):
    return [(lo, hi, lo is not None, hi is not None) for lo, hi in ivs]


def factor_text(root, power: int = 1) -> str:
    base = "x" if root == 0 else (f"(x - {root})" if root > 0 else f"(x + {-root})")
    return base if power == 1 else f"{base}^{power}"


def product_text(k, roots) -> str:
    """k(x - r1)^m1 ... as a learner would write it: `-2(x + 1)(x - 3)`."""
    lead = "" if k == 1 else ("-" if k == -1 else str(k))
    return lead + "".join(factor_text(r, m) for r, m in roots)


_WORD = {
    "increasing": "f is increasing",
    "decreasing": "f is decreasing",
    "up": "the graph of f is concave up",
    "down": "the graph of f is concave down",
}


def _ask(what: str) -> str:
    return f"Find the largest open intervals on which {_WORD[what]}."


def _meet(a, b):
    """The single interval where both lists of intervals hold, or None."""
    best = None
    for lo1, hi1 in a:
        for lo2, hi2 in b:
            lo = lo1 if lo2 is None else (lo2 if lo1 is None else max(lo1, lo2))
            hi = hi1 if hi2 is None else (hi2 if hi1 is None else min(hi1, hi2))
            if lo is None or hi is None or lo < hi:
                best = best or []
                best.append((lo, hi))
    return best


_EASY_STEMS = (
    "The derivative of a function f is f'(x) = {d}.",
    "A function f has f'(x) = {d}.",
)


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy(rng):
    p, q = sorted(rng.sample(list(range(-4, 6)), 2))
    k = rng.choice((1, 2, 3, -1, -2))
    what = rng.choice(("increasing", "decreasing"))
    d = from_roots(k, [p, q])
    ivs = signed_intervals(d, [p, q], what == "increasing")
    _check_grid(d, what == "increasing", ivs)
    other = signed_intervals(d, [p, q], what != "increasing")
    wrong = [
        (other, "curve-increasing-decreasing-swapped"),
        ([ivs[0]], "curve-drops-one-interval"),
        (_closed(ivs), "curve-includes-stationary-points"),
        (reflect(ivs), "curve-root-sign-error"),
    ]
    stem = phrase(rng, _EASY_STEMS, d=product_text(k, [(p, 1), (q, 1)]))
    working = [
        f"f'(x) = 0 at x = {p} and x = {q}, and f' has the sign of {_sign_of(k)}.",
        f"f is {what} where f' is {'positive' if what == 'increasing' else 'negative'}.",
    ]
    return single(stem, inequality_part(_ask(what), ivs, 2, working, wrong))


# -- medium -------------------------------------------------------------------------------------------------------


def _medium_repeated(rng):
    for _ in attempts(500, "a repeated root away from the simple root, on the inside of the answer"):
        p, q = rng.sample(range(-4, 6), 2)
        k = rng.choice((1, 2, -1, -3))
        what = rng.choice(("increasing", "decreasing"))
        d = from_roots(k, [p, q, q])
        ivs = signed_intervals(d, [p, q], what == "increasing")
        # the squared root must sit inside one answer interval, so a learner can wrongly split it
        if abs(p - q) >= 2 and any((lo is None or lo < q) and (hi is None or hi > q) for lo, hi in ivs):
            break
    _check_grid(d, what == "increasing", ivs)
    other = signed_intervals(d, [p, q], what != "increasing")
    split = []
    for lo, hi in ivs:
        if (lo is None or lo < q) and (hi is None or hi > q):
            split += [(lo, q), (q, hi)]
        else:
            split.append((lo, hi))
    kept = [iv for iv in ivs if not ((iv[0] is None or iv[0] < q) and (iv[1] is None or iv[1] > q))]
    wrong = [
        (split, "curve-splits-at-inflexion-stationary-point"),
        (other, "curve-increasing-decreasing-swapped"),
        (kept, "curve-ignores-squared-factor-interval"),
        (reflect(ivs), "curve-root-sign-error"),
    ]
    stem = phrase(rng, _EASY_STEMS, d=product_text(k, sorted([(p, 1), (q, 2)])))
    working = [
        f"f'(x) = 0 at x = {p} and x = {q}. The factor {factor_text(q)}^2 is never negative, so f' does not change sign at x = {q}.",
        f"f is {what} on both sides of x = {q}, so the interval runs through it.",
    ]
    return single(stem, inequality_part(_ask(what), ivs, 3, working, wrong))


def _medium_concavity(rng):
    for _ in attempts(500, "a quadratic derivative with a nonzero linear term"):
        a = rng.choice((1, 2, 3, -1, -2))
        b = rng.choice(range(-8, 9))
        c = rng.choice(range(-6, 7))
        if b != 0:
            break
    d = [Fraction(a), Fraction(b), Fraction(c)]
    second = pder(d)
    what = rng.choice(("up", "down"))
    vertex = Fraction(-b, 2 * a)
    ivs = signed_intervals(second, [vertex], what == "up")
    _check_grid(second, what == "up", ivs)
    roots = quad_roots(a, b, c)
    from_first = signed_intervals(d, roots, what == "up") if roots else None
    wrong = [
        (signed_intervals(second, [vertex], what != "up"), "curve-concave-up-down-swapped"),
        (from_first, "curve-concavity-from-first-derivative"),
        (reflect(ivs), "curve-root-sign-error"),
        (scaled(ivs, 2), "curve-vertex-without-halving"),
    ]
    stem = phrase(rng, _EASY_STEMS, d=poly_text(d))
    working = [
        f"f''(x) = {poly_text(second)}, which is zero at x = {num(vertex)}.",
        f"The graph is concave {what} where f''(x) is {'positive' if what == 'up' else 'negative'}.",
    ]
    return single(stem, inequality_part(_ask(what), ivs, 2, working, [(w, t) for w, t in wrong if w]))


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard(rng):
    for _ in attempts(500, "an intersection that is a single interval"):
        p, q = sorted(rng.sample(range(-4, 6), 2))
        k = rng.choice((1, 2, 3, -1, -2))
        first = rng.choice(("increasing", "decreasing"))
        concave = rng.choice(("up", "down"))
        d = from_roots(k, [p, q])
        second = pder(d)
        mid = Fraction(p + q, 2)
        a = signed_intervals(d, [p, q], first == "increasing")
        b = signed_intervals(second, [mid], concave == "up")
        meet = _meet(a, b)
        if meet and len(meet) == 1:
            ivs = meet
            break
    _check_grid(d, first == "increasing", a)
    _check_grid(second, concave == "up", b)
    wrong = [
        (a, "curve-only-first-condition"),
        (b, "curve-only-second-condition"),
        (_meet(signed_intervals(d, [p, q], first != "increasing"), b), "curve-increasing-decreasing-swapped"),
        (_meet(a, signed_intervals(second, [mid], concave != "up")), "curve-concave-up-down-swapped"),
    ]
    stem = phrase(rng, _EASY_STEMS, d=poly_text(d))
    prompt = f"Find the largest open interval on which f is {first} and its graph is concave {concave}."
    working = [
        f"f'(x) = {poly_text(d)} is {'positive' if first == 'increasing' else 'negative'} on {interval_text(a)}.",
        f"f''(x) = {poly_text(second)} is {'positive' if concave == 'up' else 'negative'} on {interval_text(b)}; "
        f"both hold on {interval_text(ivs)}.",
    ]
    return single(stem, inequality_part(prompt, ivs, 3, working, [(w, t) for w, t in wrong if w]))


@generator(
    "r4e_curve_features",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4e_curve_features(rng, tier: str = "easy"):
    """Intervals of increase, decrease and concavity from f' or f''; see the module docstring for the ladder."""
    if tier == "easy":
        return draw(_easy, rng)
    if tier == "medium":
        return draw(_medium_repeated if rng.random() < 0.5 else _medium_concavity, rng)
    if tier == "hard":
        return draw(_hard, rng)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ---------------------------------------------------------------------------------------------------


def _chain(rng):
    for _ in attempts(500, "stationary points of the same parity, distinct and apart"):
        p, q = sorted(rng.sample(range(-3, 6), 2))
        if (p + q) % 2 == 0 and q - p >= 2:
            break
    lead = rng.choice((1, 1, -1))
    d0 = rng.choice([v for v in range(-6, 9) if v])
    mid = Fraction(p + q, 2)
    f = [Fraction(lead), Fraction(-3 * lead * (p + q), 2), Fraction(3 * lead * p * q), Fraction(d0)]
    first = pder(f)
    check(first == from_roots(3 * lead, [p, q]), "f' disagrees with the construction")
    check(pev(pder(first), mid) == 0, "f'' is not zero at the midpoint")
    ivs = signed_intervals(first, [p, q], True)
    _check_grid(first, True, ivs)
    y_mid = pev(f, mid)
    if y_mid == 0 or pev(f, p) == y_mid or pev(f, q) == y_mid:
        raise Redraw("an inflection value that coincides with another")
    stem = f"A function f is given by f(x) = {poly_text(f)}."
    part_a = list_part(
        "Find the x-coordinates of the stationary points of f.",
        [p, q],
        2,
        [
            f"f'(x) = {poly_text(first)} = {product_text(3 * lead, [(p, 1), (q, 1)])}.",
            f"f'(x) = 0 at x = {p} and x = {q}.",
        ],
        [
            ([-p, -q], "curve-root-sign-error"),
            ([mid], "curve-second-derivative-for-stationary"),
            ([p, q, mid], "curve-inflection-listed-as-stationary"),
        ],
    )
    part_b = inequality_part(
        "Find the largest open intervals on which f is increasing.",
        ivs,
        2,
        [f"f'(x) = {product_text(3 * lead, [(p, 1), (q, 1)])} is positive on {interval_text(ivs)}."],
        [
            (signed_intervals(first, [p, q], False), "curve-increasing-decreasing-swapped"),
            ([ivs[0]], "curve-drops-one-interval"),
            (_closed(ivs), "curve-includes-stationary-points"),
        ],
    )
    part_c = fraction_part(
        "Find the x-coordinate of the point of inflection of the graph of f.",
        mid,
        2,
        [f"f''(x) = {poly_text(pder(first))} = 0 when x = {num(mid)}."],
        [
            (Fraction(p + q), "curve-vertex-without-halving"),
            (-mid, "curve-root-sign-error"),
            (Fraction(p), "curve-second-derivative-for-stationary"),
        ],
    )
    from .r4e_common import poly_code

    part_d = fraction_part(
        "Find the y-coordinate of the point of inflection.",
        y_mid,
        3,
        [f"f({num(mid)}) = {num(y_mid)}."],
        [
            (pev(f, p), "curve-stationary-value-for-inflection"),
            (pev(f, q), "curve-stationary-value-for-inflection"),
            (Fraction(d0), "curve-constant-term-for-value"),
            (pev(f, -mid), "curve-root-sign-error"),
        ],
        follow_through=poly_code(f, "part_c"),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("r4e_curve_features_chain", level=17, presets={"exam": dict(tier="exam")})
def r4e_curve_features_chain(rng, tier: str = "exam"):
    """A cubic with integer stationary points: them, the intervals of increase, the inflection point's x and y."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_chain, rng)
