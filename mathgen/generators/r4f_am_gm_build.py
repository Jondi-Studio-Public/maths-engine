"""Generates `r4f_am_gm` questions and the exam chain `exd_x_r4f_am_gm`.

Skill `am-gm-inequality` (``inequalities``): use the inequality of arithmetic and geometric means, (u + v)/2 >= sqrt(uv)
for u, v > 0 (and its three-term form), to find a least or greatest value, the value of the variable where equality
holds, and the bound in a word problem. Every answer is a number.

Two cores recur, and each is built from its equality case so every answer is exact:

* the sum px + q/x for x > 0, with q = p x0² and x0 a whole number or a half, so that the minimum is
  2sqrt(pq) = 2 p x0 and it is reached at x = x0;
* the product xy under px + qy = s, with p x* = q y* = s/2, so that the maximum is xy = s²/(4pq) at x = x*.

Tiers (topic ``r4f_am_gm``):

* easy   -- the least value of px + q/x (x > 0), or the greatest value of xy given px + qy = s;
* medium -- the value of x where equality holds (for the sum, or for the product), or where
  (x + a)(x + b)/x is least, which needs the expansion x + (a + b) + ab/x first;
* hard   -- the bound in a word problem the stem sets up but does not reduce: the fence for a paddock beside a river,
  the running cost with a fixed charge, the open-top box of given volume (three-term form), the divided pen with a fixed
  length of fencing, or the least value of (x + a)(x + b)/x.

The exam chain is four number parts for one such problem, with follow-through from the bound to the point where
equality holds, and to a cost, a surface area or a second dimension.

Every key is derived from the construction (px* = qy*, or q = p x0²) and re-checked by a scan of a fine grid of
exact rational points around the claimed optimum, so no nearby value beats it.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r4f_ft import follow_through_clean
from .exd_x_r3c_parts import number_part


def _num(value) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else str(float(value))


def _dp(value) -> int:
    value = Fraction(value)
    return 0 if value.denominator == 1 else 1


def _single(stem, part):
    """A single-answer question dict from a `number_part` result."""
    return dict(
        text=stem,
        answer=part["answer"],
        answer_text=part["answer_text"],
        answer_latex=part["answer_latex"],
        answer_kind="number",
        working=part["working"],
        distractors=part["distractors"],
    )


def _scan_min(f, x0: Fraction) -> None:
    """f has no smaller value than f(x0) over an exact grid around x0."""
    for i in range(1, 80):
        x = x0 * Fraction(i, 20)
        check(f(x) >= f(x0), "a nearby point beats the claimed minimum")


def _scan_max(f, x0: Fraction) -> None:
    for i in range(1, 80):
        x = x0 * Fraction(i, 20)
        check(f(x) <= f(x0), "a nearby point beats the claimed maximum")


def _draw_sum(rng: random.Random):
    """(p, q, x0): q = p x0², x0 a whole number 1..12 or a half 3/2..11/2 (then p is a multiple of 4)."""
    for _ in attempts(100, "a sum px + q/x"):
        if rng.random() < 0.3:
            x0 = Fraction(rng.choice((3, 5, 7, 9, 11)), 2)
            p = 4 * rng.choice((1, 1, 2, 3))
        else:
            x0 = Fraction(rng.randint(1, 12))
            p = rng.choice((1, 1, 1, 2, 3, 4, 5, 6))
        q = p * x0 * x0
        check(q.denominator == 1, "q is not whole")
        if 2 <= q <= 400 and not (p == q):
            return p, int(q), x0
    raise AssertionError  # pragma: no cover


def _sum_text(p: int, q: int, var: str = "x", flip: bool = False) -> str:
    head = f"{'' if p == 1 else p}{var}"
    tail = f"{q}/{var}"
    return f"{tail} + {head}" if flip else f"{head} + {tail}"


def _sum_check(p, q, x0):
    f = lambda x: p * x + Fraction(q) / x  # noqa: E731
    m = 2 * p * x0
    check(f(x0) == m, "the sum at x0 is not the claimed minimum")
    _scan_min(f, x0)
    # AM-GM second route: (px)(q/x) = pq, and 2 sqrt(pq) = m
    check(m * m == 4 * p * q, "2 sqrt(pq) disagrees")
    return m


def _draw_product(rng: random.Random):
    """(p, q, xs, ys, s): px* = qy* = s/2, whole xs and ys."""
    for _ in attempts(100, "a product problem"):
        p, q = rng.choice(
            ((1, 1), (1, 2), (2, 1), (2, 3), (3, 2), (1, 3), (3, 1), (3, 4), (4, 3), (1, 4), (2, 5), (5, 2))
        )
        lcm = p * q // _gcd(p, q)
        h = lcm * rng.randint(1, 6)
        xs, ys = h // p, h // q
        s = 2 * h
        if 5 <= xs * ys <= 700:
            return p, q, xs, ys, s
    raise AssertionError  # pragma: no cover


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _product_check(p, q, xs, ys, s):
    g = lambda x: x * (Fraction(s) - p * x) / q  # y = (s - p x)/q  # noqa: E731
    best = xs * ys
    check(g(Fraction(xs)) == best, "the product at x* is not the claimed maximum")
    _scan_max(g, Fraction(xs))
    check(Fraction(s * s, 4 * p * q) == best, "s²/(4pq) disagrees")
    return best


def _c(k: int) -> str:
    """A coefficient as written before a letter: nothing for 1."""
    return "" if k == 1 else str(k)


def _lin_text(p: int, q: int) -> str:
    return f"{'' if p == 1 else p}x + {'' if q == 1 else q}y"


# ---------------------------------------------------------------------------------------------------------------
# Easy and medium


def _easy_sum(rng):
    p, q, x0 = _draw_sum(rng)
    m = _sum_check(p, q, x0)
    flip = rng.random() < 0.3
    f1 = p + q
    part = number_part(
        "",
        m,
        working=[
            f"{_sum_text(p, q)} ≥ 2√({_c(p)}x · {q}/x) = 2√{p * q} = {m}, with equality when {_c(p)}x = {q}/x, at x = {_num(x0)}."
        ],
        wrongs=[
            (f1, "evaluates-at-x-equals-1"),
            (2 * p * q, "amgm-forgets-the-root"),
            (p * x0, "amgm-forgets-the-two"),
            (x0, "gives-the-answer-for-x-not-the-value"),
            (p + q if p + q != m else m + 2, "evaluates-at-x-equals-1"),
            (m + p, "amgm-bound-miscomputed"),
        ],
    )
    return _single(f"Find the minimum value of f(x) = {_sum_text(p, q, flip=flip)} for x > 0.", part)


def _easy_product(rng):
    p, q, xs, ys, s = _draw_product(rng)
    best = _product_check(p, q, xs, ys, s)
    part = number_part(
        "",
        best,
        working=[f"({_c(p)}x)({_c(q)}y) ≤ ({s}/2)² = {s * s // 4}, so xy ≤ {s * s // 4} ÷ {p * q} = {best}."],
        wrongs=[
            (Fraction(s * s, 4), "amgm-ignores-the-coefficients"),
            (Fraction(s * s, 4 * p * q) * 2, "amgm-bound-miscomputed"),
            (Fraction(s * s, 2 * p * q), "amgm-forgets-the-four"),
            (Fraction(s, 2) * Fraction(s, 2) / q, "amgm-ignores-the-coefficients"),
            (xs + ys, "gives-the-sum-not-the-product"),
        ],
    )
    return _single(
        f"The positive numbers x and y satisfy {_lin_text(p, q)} = {s}. Find the greatest possible value of the product xy.",
        part,
    )


def _medium_sum(rng):
    p, q, x0 = _draw_sum(rng)
    m = _sum_check(p, q, x0)
    flip = rng.random() < 0.3
    part = number_part(
        "",
        x0,
        dp=_dp(x0),
        working=[
            f"Equality holds when the two terms are equal: {_c(p)}x = {q}/x, so x² = {q // p if q % p == 0 else Fraction(q, p)} and x = {_num(x0)}."
        ],
        wrongs=[
            (m, "gives-the-value-not-x"),
            (x0 * x0, "forgets-to-take-the-root"),
            (Fraction(q, p) if x0 * x0 != Fraction(q, p) else x0 + 1, "forgets-to-take-the-root"),
            (Fraction(p, 1) * x0, "term-taken-as-the-variable"),
            (2 * x0, "x-doubled-from-the-two"),
        ],
    )
    return _single(
        f"For x > 0 the function f(x) = {_sum_text(p, q, flip=flip)} has a least value. "
        "Find the value of x at which f is least.",
        part,
    )


def _medium_product(rng):
    p, q, xs, ys, s = _draw_product(rng)
    _product_check(p, q, xs, ys, s)
    part = number_part(
        "",
        xs,
        working=[
            f"The product is greatest when {_c(p)}x = {_c(q)}y, each half of {s}: {_c(p)}x = {s // 2}, so x = {xs} (and y = {ys})."
        ],
        wrongs=[
            (Fraction(s, 2), "assumes-equal-parts-ignoring-coefficients"),
            (ys, "x-and-y-swapped"),
            (xs * ys, "gives-the-value-not-x"),
            (Fraction(s, 2 * q), "uses-the-wrong-coefficient"),
            (Fraction(s, p), "forgets-the-half"),
        ],
    )
    return _single(
        f"The positive numbers x and y satisfy {_lin_text(p, q)} = {s}, and the product xy is greatest for one value of x. Find it.",
        part,
    )


def _draw_ab(rng):
    """(a, b, r): (x + a)(x + b)/x = x + (a + b) + ab/x, with sqrt(ab) = r whole, a != b."""
    for _ in attempts(200, "a pair a, b with ab a square"):
        r = rng.randint(2, 12)
        a = rng.choice([d for d in range(1, r) if r * r % d == 0])
        b = r * r // a
        if a != b and a + b + 2 * r <= 150:
            return a, b, r
    raise AssertionError  # pragma: no cover


def _medium_expand(rng):
    a, b, r = _draw_ab(rng)
    f = lambda x: (x + a) * (x + b) / x  # noqa: E731
    check(f(Fraction(r)) == a + b + 2 * r, "the value at sqrt(ab) disagrees")
    _scan_min(f, Fraction(r))
    part = number_part(
        "",
        r,
        working=[f"(x + {a})(x + {b})/x = x + {a + b} + {a * b}/x, and x = {a * b}/x gives x² = {a * b}, so x = {r}."],
        wrongs=[
            (a + b + 2 * r, "gives-the-value-not-x"),
            (a * b, "forgets-to-take-the-root"),
            (a + b, "adds-instead-of-multiplies"),
            (Fraction(a + b, 2), "adds-instead-of-multiplies"),
            (a, "takes-one-factor-for-x"),
            (b, "takes-one-factor-for-x"),
        ],
    )
    return _single(
        f"For x > 0, the function f(x) = (x + {a})(x + {b})/x has a least value. Find the value of x at which it occurs.",
        part,
    )


# ---------------------------------------------------------------------------------------------------------------
# Hard: word problems. Each returns (stem, answer, wrongs, working) pieces via small builders.


def _paddock(rng):
    """Fence on three sides (river on the fourth) round an area A = 2t²: L = 2w + A/w, minimum 4t at w = t."""
    t = rng.randint(4, 15)
    area = 2 * t * t
    f = lambda w: 2 * w + Fraction(area) / w  # noqa: E731
    check(f(Fraction(t)) == 4 * t, "the paddock minimum is wrong")
    _scan_min(f, Fraction(t))
    return t, area


def _hard_paddock(rng):
    t, area = _paddock(rng)
    m = 4 * t
    part = number_part(
        "",
        m,
        working=[f"L = 2w + {area}/w ≥ 2√(2w · {area}/w) = 2√{2 * area} = {m} m, with equality at w = {t} m."],
        wrongs=[
            (area, "gives-the-area-not-the-bound"),
            (2 + area, "evaluates-at-x-equals-1"),
            (2 * area, "amgm-forgets-the-root"),
            (2 * t, "amgm-forgets-the-two"),
            (4 * t * 2, "amgm-bound-miscomputed"),
            (2 * (2 * t * 2), "amgm-bound-miscomputed"),
        ],
    )
    return _single(
        f"A rectangular paddock of area {area} m² is to be fenced on three sides, the fourth side being a straight river. "
        "The sides at right angles to the river are each w metres long. Find the least length of fencing needed, in metres.",
        part,
    )


def _hard_cost(rng):
    p, q, x0 = _draw_sum(rng)
    fixed = rng.choice((20, 25, 30, 40, 45, 60, 75, 90))
    m = _sum_check(p, q, x0)
    total = m + fixed
    contexts = (
        (
            "A pump moves x kilolitres of water per hour, and the cost of running it is C = {sum} + {fixed} dollars per hour.",
            "dollars per hour",
        ),
        (
            "A bakery bakes x trays per hour, and its running cost is C = {sum} + {fixed} dollars per hour.",
            "dollars per hour",
        ),
    )
    stem, unit = rng.choice(contexts)
    flip = rng.random() < 0.4
    part = number_part(
        "",
        total,
        working=[
            f"{_sum_text(p, q, flip=flip)} ≥ {m}, with equality at x = {_num(x0)}, so C ≥ {m} + {fixed} = {total} dollars per hour."
        ],
        wrongs=[
            (m, "forgets-the-fixed-charge"),
            (p + q + fixed, "evaluates-at-x-equals-1"),
            (2 * p * q + fixed, "amgm-forgets-the-root"),
            (p * x0 + fixed, "amgm-forgets-the-two"),
            (m + 2 * fixed, "fixed-charge-counted-twice"),
        ],
    )
    return _single(
        stem.format(sum=_sum_text(p, q, flip=flip), fixed=fixed)
        + " Find the least possible cost, in dollars per hour.",
        part,
    )


def _box(rng):
    """Open-top square-based box of volume V = 4t³: S = x² + 4V/x, least value 12t² at x = 2t, height t."""
    t = rng.randint(1, 6)
    volume = 4 * t**3
    s = lambda x: x * x + 4 * Fraction(volume) / x  # noqa: E731
    check(s(Fraction(2 * t)) == 12 * t * t, "the box minimum is wrong")
    _scan_min(s, Fraction(2 * t))
    check(volume == (2 * t) ** 2 * t, "volume disagrees with the dimensions")
    return t, volume


def _hard_box(rng):
    t, volume = _box(rng)
    m = 12 * t * t
    part = number_part(
        "",
        m,
        working=[
            f"S = x² + {4 * volume}/x = x² + {2 * volume}/x + {2 * volume}/x ≥ 3∛(x² · ({2 * volume}/x)²) = 3 × {4 * t * t} = {m}."
        ],
        wrongs=[
            (8 * t * t, "two-term-bound-used-on-three-terms"),
            (4 * t * t, "forgets-the-three"),
            (12 * t * t + 4 * t * t, "base-and-sides-counted-twice"),
            (3 * volume, "amgm-bound-miscomputed"),
            (6 * t * t, "amgm-bound-miscomputed"),
        ],
    )
    return _single(
        f"An open-top box (no lid) has a square base of side x cm and holds {volume} cm³. Its surface area is "
        f"S = x² + {4 * volume}/x square centimetres. Find the least possible surface area, in cm².",
        part,
    )


def _pen(rng):
    """A rectangular pen with one internal partition: 2x + 3y = F, area xy, greatest F²/24 at x = F/4, y = F/6."""
    h = 6 * rng.randint(2, 7)
    fence = 2 * h
    xs, ys = h // 2, h // 3
    g = lambda x: x * (Fraction(fence) - 2 * x) / 3  # noqa: E731
    check(g(Fraction(xs)) == xs * ys, "the pen maximum is wrong")
    _scan_max(g, Fraction(xs))
    check(Fraction(fence * fence, 24) == xs * ys, "F²/24 disagrees")
    return fence, xs, ys


def _hard_pen(rng):
    fence, xs, ys = _pen(rng)
    best = xs * ys
    part = number_part(
        "",
        best,
        working=[f"(2x)(3y) ≤ ({fence}/2)² = {fence * fence // 4}, so xy ≤ {fence * fence // 4} ÷ 6 = {best} m²."],
        wrongs=[
            (Fraction(fence * fence, 16), "amgm-ignores-the-coefficients"),
            (Fraction(fence * fence, 12), "amgm-forgets-the-four"),
            (fence * fence // 4, "amgm-bound-not-divided-by-coefficients"),
            (Fraction(fence * fence, 48), "amgm-bound-miscomputed"),
            (xs + ys, "gives-the-sum-not-the-product"),
        ],
    )
    return _single(
        f"A rectangular pen is divided into two equal parts by one fence parallel to its width. The width is x metres "
        f"and the length y metres, so the pen uses 2x + 3y = {fence} metres of fencing in all. "
        "Find the greatest possible area of the pen, in square metres.",
        part,
    )


def _hard_expand(rng):
    a, b, r = _draw_ab(rng)
    f = lambda x: (x + a) * (x + b) / x  # noqa: E731
    m = a + b + 2 * r
    check(f(Fraction(r)) == m, "the value at sqrt(ab) disagrees")
    _scan_min(f, Fraction(r))
    part = number_part(
        "",
        m,
        working=[f"(x + {a})(x + {b})/x = x + {a * b}/x + {a + b} ≥ 2√{a * b} + {a + b} = {m}."],
        wrongs=[
            (2 * r, "forgets-the-constant-term"),
            (a + b, "forgets-the-am-gm-part"),
            (a + b + r, "amgm-forgets-the-two"),
            ((a + 1) * (b + 1), "evaluates-at-x-equals-1"),
            (a + b + 2 * a * b, "amgm-forgets-the-root"),
        ],
    )
    return _single(f"Find the least value of f(x) = (x + {a})(x + {b})/x for x > 0.", part)


# ---------------------------------------------------------------------------------------------------------------


@generator(
    "r4f_am_gm",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4f_am_gm(rng: random.Random, tier: str = "easy") -> dict:
    """An AM-GM question: the least or greatest value (easy), the equality point (medium), a word-problem bound (hard)."""
    pick = rng.random()
    if tier == "easy":
        return _easy_sum(rng) if pick < 0.6 else _easy_product(rng)
    if tier == "medium":
        return _medium_sum(rng) if pick < 0.4 else _medium_product(rng) if pick < 0.7 else _medium_expand(rng)
    if tier == "hard":
        builders = (_hard_paddock, _hard_cost, _hard_box, _hard_pen, _hard_expand)
        return builders[int(pick * len(builders))](rng)
    raise ValueError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------------------------------------------
# Exam chains


def _exam_paddock(rng):
    t, area = _paddock(rng)
    m = 4 * t
    divisors = [d for d in range(2, area) if area % d == 0 and d != t and d <= 3 * t]
    w1 = rng.choice(divisors)
    length1 = 2 * w1 + area // w1
    price = rng.choice((12, 15, 18, 20, 24, 30, 36))
    part_a = number_part(
        f"The sides at right angles to the river are each w metres long. Find the length of fencing needed when w = {w1}.",
        length1,
        marks=1,
        sensible_range=(0, None),
        working=[f"L = 2 × {w1} + {area} ÷ {w1} = {length1} m."],
        wrongs=[
            (w1 + area // w1, "counts-only-one-short-side"),
            (4 * w1 + area // w1, "fences-all-four-sides"),
            (2 * w1 + area, "area-not-divided-by-w"),
            (2 * (w1 + area // w1), "fences-all-four-sides"),
        ],
    )
    part_b = number_part(
        f"The length of fencing is L = 2w + {area}/w. Find its least possible value, in metres.",
        m,
        marks=2,
        sensible_range=(0, None),
        working=[f"2w + {area}/w ≥ 2√({2 * area}) = {m} m."],
        wrongs=[
            (area, "gives-the-area-not-the-bound"),
            (2 * area, "amgm-forgets-the-root"),
            (2 * t, "amgm-forgets-the-two"),
            (length1, "evaluates-at-a-trial-value"),
        ],
    )
    part_c = number_part(
        "Using the least length, find the value of w, in metres, at which it is reached.",
        t,
        marks=2,
        sensible_range=(0, None),
        working=[f"Equality: 2w = {area}/w, and 2w is half of {m}, so w = {t} m."],
        wrongs=[
            (m // 2, "term-taken-as-the-variable"),
            (m, "gives-the-value-not-x"),
            (2 * t * t, "forgets-to-take-the-root"),
            (m // 8 * 2 if m // 8 * 2 != t else t + 1, "divides-by-the-wrong-number"),
        ],
        follow_through="part_b/4",
    )
    part_d = number_part(
        f"Fencing costs ${price} per metre. Find the least possible cost of the fence, in dollars.",
        price * m,
        marks=1,
        sensible_range=(0, None),
        working=[f"{price} × {m} = {price * m} dollars."],
        wrongs=[
            (price + m, "adds-instead-of-multiplies"),
            (price * m * 2, "cost-doubled"),
            (price * (m + 2 * t), "fences-all-four-sides"),
        ],
        follow_through=f"{price}*part_b",
    )
    return dict(
        text=(
            f"A rectangular paddock of area {area} m² is to be fenced on three sides, the fourth side being a straight river."
        ),
        parts=[part_a, part_b, part_c, part_d],
    )


def _exam_cost(rng):
    p, q, x0 = _draw_sum(rng)
    fixed = rng.choice((20, 25, 30, 40, 45, 60, 75, 90))
    m = _sum_check(p, q, x0)
    candidates = [x for x in range(2, 25) if q % x == 0 and Fraction(x) != x0] or [q]
    x1 = rng.choice(candidates)
    c1 = p * x1 + q // x1 + fixed
    flip = rng.random() < 0.4
    part_a = number_part(
        f"Find the running cost when x = {x1}, in dollars per hour.",
        c1,
        marks=1,
        sensible_range=(0, None),
        working=[f"C = {p} × {x1} + {q} ÷ {x1} + {fixed} = {c1}."],
        wrongs=[
            (p * x1 + q * x1 + fixed, "multiplies-instead-of-divides"),
            (p * x1 + q // x1, "forgets-the-fixed-charge"),
            (p + q + fixed, "evaluates-at-x-equals-1"),
            (p * x1 + q // x1 + 2 * fixed, "fixed-charge-counted-twice"),
        ],
    )
    part_b = number_part(
        f"Find the least possible value of {_sum_text(p, q, flip=flip)} for x > 0.",
        m,
        marks=2,
        sensible_range=(0, None),
        working=[f"{_sum_text(p, q, flip=flip)} ≥ 2√{p * q} = {m}."],
        wrongs=[
            (p * x1 + q // x1, "evaluates-at-a-trial-value"),
            (2 * p * q, "amgm-forgets-the-root"),
            (p * x0, "amgm-forgets-the-two"),
            (p + q, "evaluates-at-x-equals-1"),
        ],
    )
    part_c = number_part(
        "Hence find the least possible running cost, in dollars per hour.",
        m + fixed,
        marks=1,
        sensible_range=(0, None),
        working=[f"{m} + {fixed} = {m + fixed}."],
        wrongs=[
            (m, "forgets-the-fixed-charge"),
            (m + 2 * fixed, "fixed-charge-counted-twice"),
            (m - fixed, "fixed-charge-subtracted"),
            (2 * m, "total-doubled"),
        ],
        follow_through=f"part_b + {fixed}",
    )
    part_d = number_part(
        "Find the value of x at which the least cost occurs.",
        x0,
        dp=_dp(x0),
        marks=2,
        sensible_range=(0, None),
        working=[
            f"At equality {_c(p)}x = {q}/x, and each of these terms is half of {m}, so {_c(p)}x = {Fraction(m, 2)} and x = {_num(x0)}."
        ],
        wrongs=[
            (m, "gives-the-value-not-x"),
            (x0 * x0, "forgets-to-take-the-root"),
            (Fraction(m, 2), "term-taken-as-the-variable"),
            (Fraction(m, 4 * p) if Fraction(m, 4 * p) != x0 else x0 + 1, "divides-by-the-wrong-number"),
        ],
        follow_through=f"part_b/(2*{p})",
    )
    return dict(
        text=(
            f"A machine is run at x units per hour, and its cost is C = {_sum_text(p, q, flip=flip)} + {fixed} dollars per hour."
        ),
        parts=[part_a, part_b, part_c, part_d],
    )


def _exam_box(rng):
    t, volume = _box(rng)
    x0, m = 2 * t, 12 * t * t
    candidates = [x for x in range(1, 40) if (4 * volume) % x == 0 and x != x0 and x0 / 2 <= x <= 3 * x0] or [1]
    x1 = rng.choice(candidates)
    s1 = x1 * x1 + 4 * volume // x1
    part_a = number_part(
        f"Find the surface area when x = {x1}, in cm².",
        s1,
        marks=1,
        sensible_range=(0, None),
        working=[f"S = {x1}² + {4 * volume} ÷ {x1} = {s1}."],
        wrongs=[
            (x1 * x1 + 4 * volume * x1, "multiplies-instead-of-divides"),
            (x1 * x1 + volume // x1 if volume % x1 == 0 else x1 * x1, "surface-area-sides-miscounted"),
            (4 * volume // x1, "surface-area-left-out-the-base"),
            (2 * x1 * x1 + 4 * volume // x1, "base-and-sides-counted-twice"),
        ],
    )
    part_b = number_part(
        (
            f"S = x² + {4 * volume}/x can be written as x² + {2 * volume}/x + {2 * volume}/x, three positive terms. "
            "The three terms are equal when S is least. Find the value of x, in cm, at which this happens."
        ),
        x0,
        marks=2,
        sensible_range=(0, None),
        working=[f"x² = {2 * volume}/x, so x³ = {2 * volume} and x = {x0}."],
        wrongs=[
            (t, "forgets-the-factor-in-the-cube"),
            (m, "gives-the-value-not-x"),
            (4 * t * t, "forgets-to-take-the-root"),
            (x1, "evaluates-at-a-trial-value"),
        ],
    )
    part_c = number_part(
        "Hence find the least possible surface area, in cm².",
        m,
        marks=2,
        sensible_range=(0, None),
        working=[f"S = 3x² = 3 × {x0}² = {m} cm² (three equal terms of x² each)."],
        wrongs=[
            (x0 * x0, "forgets-the-three"),
            (2 * x0 * x0, "two-term-bound-used-on-three-terms"),
            (s1, "evaluates-at-a-trial-value"),
            (3 * x0, "multiplies-instead-of-squares"),
        ],
        follow_through="3*part_b**2",
    )
    part_d = number_part(
        "Hence find the height of the box, in cm, when the surface area is least.",
        t,
        marks=1,
        sensible_range=(0, None),
        working=[f"Height = volume ÷ base area = {volume} ÷ {x0}² = {t} cm."],
        wrongs=[
            (volume // x0, "base-area-taken-as-the-side"),
            (x0, "height-confused-with-side"),
            (volume // (x0 * x0) * 2, "arithmetic-slip"),
            (volume // (x0 * x0) + 1, "arithmetic-slip"),
            (volume, "volume-given-as-height"),
            (x0 * x0, "base-area-given-as-height"),
        ],
        follow_through=f"{volume}/part_b**2",
    )
    return dict(
        text=(
            f"An open-top box (no lid) has a square base of side x cm and a volume of {volume} cm³. Its surface area is "
            f"S = x² + {4 * volume}/x square centimetres."
        ),
        parts=[part_a, part_b, part_c, part_d],
    )


def _exam_pen(rng):
    fence, xs, ys = _pen(rng)
    best = xs * ys
    x1 = rng.choice([x for x in range(2, fence // 2) if (fence - 2 * x) % 3 == 0 and x != xs])
    y1 = (fence - 2 * x1) // 3
    part_a = number_part(
        f"Find y, in metres, when x = {x1}.",
        y1,
        marks=1,
        sensible_range=(0, None),
        working=[f"3y = {fence} - 2 × {x1} = {fence - 2 * x1}, so y = {y1}."],
        wrongs=[
            (fence - 2 * x1, "forgets-to-divide-by-the-coefficient"),
            ((fence - x1) // 3 if (fence - x1) % 3 == 0 else (fence - 3 * x1) // 2, "wrong-coefficient-removed"),
            (fence - 2 * x1 - 3, "arithmetic-slip"),
            (x1, "x-and-y-swapped"),
        ],
    )
    part_b = number_part(
        f"Find the greatest possible area xy of the pen, using 2x + 3y = {fence}, in square metres.",
        best,
        marks=2,
        sensible_range=(0, None),
        working=[f"(2x)(3y) ≤ ({fence}/2)² = {fence * fence // 4}, so xy ≤ {best}."],
        wrongs=[
            (fence * fence // 4, "amgm-bound-not-divided-by-coefficients"),
            (Fraction(fence * fence, 16), "amgm-ignores-the-coefficients"),
            (Fraction(fence * fence, 12), "amgm-forgets-the-four"),
            (x1 * y1, "evaluates-at-a-trial-value"),
        ],
    )
    part_c = number_part(
        "Find the value of x, in metres, at which the area is greatest.",
        xs,
        marks=2,
        sensible_range=(0, None),
        working=[f"Equality: 2x = 3y, each half of {fence}, so 2x = {fence // 2} and x = {xs}."],
        wrongs=[
            (fence // 2, "assumes-equal-parts-ignoring-coefficients"),
            (ys, "x-and-y-swapped"),
            (fence // 3, "uses-the-wrong-coefficient"),
            (best, "gives-the-value-not-x"),
        ],
    )
    part_d = number_part(
        "Use your x to find the value of y, in metres, at which the area is greatest.",
        ys,
        marks=1,
        sensible_range=(0, None),
        working=[f"y = ({fence} - 2 × {xs}) ÷ 3 = {ys}."],
        wrongs=[
            (xs, "x-and-y-swapped"),
            (fence - 2 * xs, "forgets-to-divide-by-the-coefficient"),
            (fence // 4, "assumes-equal-parts-ignoring-coefficients"),
            (best // 2 if best % 2 == 0 else best + 1, "gives-the-value-not-x"),
        ],
        follow_through=f"({fence} - 2*part_c)/3",
    )
    return dict(
        text=(
            "A rectangular pen is divided into two equal parts by one fence parallel to its width. The width is x metres "
            f"and the length y metres, so the pen uses 2x + 3y = {fence} metres of fencing in all."
        ),
        parts=[part_a, part_b, part_c, part_d],
    )


@generator("exd_x_r4f_am_gm", level=19, presets={"exam": dict(tier="exam")})
@distinct_part_answers
@follow_through_clean
def exd_x_r4f_am_gm(rng: random.Random, tier: str = "exam") -> dict:
    """Exam chain: a word problem solved by AM-GM (paddock, running cost, open box or divided pen), four number parts."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return (_exam_paddock, _exam_cost, _exam_box, _exam_pen)[rng.randrange(4)](rng)
