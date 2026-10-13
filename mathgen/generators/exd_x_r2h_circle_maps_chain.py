"""Exam question on the fundamental group of the circle: induced maps of a composite of two power maps.

The stem gives the power maps f(z) = z^m and g(z) = z^n on the unit circle, their composite h(z) = g(f(z)), and the
standard generator gamma of pi_1(S^1, 1) = Z. Four parts:

(a) h_* is multiplication by k = m n                                  number (whole), 2 marks
(b) the winding number of h applied to the loop gamma^j              number (whole), 1 mark, follow_through part_a * j
(c) the index of the image of (h o h)_* = multiplication by k^2      number (whole), 1 mark, follow_through part_a^2
(d) the r for which h applied to gamma^r winds T times, r = T / k    number (whole), 2 marks, follow_through T / part_a

T is chosen as k times a whole number, so (d) has an integer answer. m and n may be negative, so dropping a sign is a
live error, and |k| is at least 2 so the image is a proper subgroup. No two parts share an answer (the index asked
for in (c) is of the doubled map, k^2, so it is not just |k| again).
"""

from __future__ import annotations

import cmath
import math
import random

from ..core import attempts, check, generator, phrase
from ._exd_distinct_parts import distinct_part_answers
from .exd_x_r2h_support import build_part

_DEGREES = (-5, -4, -3, -2, 2, 3, 4, 5)
_LOOPS = (2, 3, 4, 5, 6, 7)  # times the loop lambda goes round
_PREIMAGES = (-7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7)

_STEM = (
    "The standard generator of pi_1(S^1, 1) = Z is the loop gamma(t) = e^(2 pi i t), t in [0, 1]. The maps "
    "f, g: S^1 -> S^1 are f(z) = z^{m} and g(z) = z^{n}, and h is the composite h(z) = g(f(z)). "
    "Write h_* for the map h induces on pi_1(S^1) = Z.",
    "Let gamma(t) = e^(2 pi i t), t in [0, 1], be the generator of pi_1(S^1, 1) = Z. Define f(z) = z^{m} and "
    "g(z) = z^{n} on the unit circle S^1, and let h = g after f, that is h(z) = g(f(z)). Let h_* be the induced "
    "homomorphism pi_1(S^1) -> pi_1(S^1).",
    "In pi_1(S^1, 1) = Z the loop gamma(t) = e^(2 pi i t), t in [0, 1], represents 1. Two maps of the circle are "
    "f(z) = z^{m} and g(z) = z^{n}; their composite is h(z) = g(f(z)). The induced map of h on the "
    "fundamental group is h_*.",
)
_PROMPT_A = (
    "The map h_* is multiplication by an integer k. Find k.",
    "Find the integer k for which h_* is multiplication by k.",
)
_PROMPT_B = (
    "Let lambda be the loop gamma traversed {j} times, so [lambda] = {j} in Z. Find the integer [h o lambda], the "
    "number of times the loop h(lambda(t)) winds around the circle.",
    "The loop lambda goes {j} times round the circle, so [lambda] = {j}. Use your k to find [h o lambda], the "
    "winding number of the loop h(lambda(t)).",
)
_PROMPT_C = (
    "Let H = h o h, so H(z) = h(h(z)). The image of H_* is a subgroup of Z. Find its index in Z.",
    "Apply h twice to get H(z) = h(h(z)). Use your k to find the index in Z of the image of the induced map H_*.",
)
_PROMPT_D = (
    "Find the integer r such that the loop h(gamma^r) winds {t} times, that is, such that h_* sends r to {t}.",
    "Using your k, find the integer r for which [h o gamma^r] = {t} in Z.",
)


def _power(exponent: int) -> str:
    """An exponent as written after ^: bracketed when negative."""
    return f"({exponent})" if exponent < 0 else str(exponent)


def _turns(m: int, n: int, loops: int, steps: int = 720) -> int:
    """How many times z -> (z^m)^n carries the loop t -> e^(2 pi i loops t) round the circle: the sum of the small
    argument changes of the image point, in whole turns."""
    total = 0.0
    previous = 1 + 0j
    for i in range(1, steps + 1):
        z = cmath.exp(2j * math.pi * loops * i / steps)
        image = (z**m) ** n
        total += cmath.phase(image / previous)
        previous = image
    return round(total / (2 * math.pi))


@distinct_part_answers
def _build(rng: random.Random) -> dict:
    for _ in attempts(1000, "two power maps with a proper image"):
        m, n = rng.choice(_DEGREES), rng.choice(_DEGREES)
        j, r = rng.choice(_LOOPS), rng.choice(_PREIMAGES)
        if abs(m * n) >= 2 and r != j and abs(m) != abs(n):
            break
    k = m * n
    winding = k * j
    target = k * r
    # Independent route: count the turns of the actual loop t -> h(gamma(t)^loop) by summing small argument changes.
    check(_turns(m, n, j) == winding, "winding number of h o lambda disagrees with the argument count")
    check(_turns(m, n, r) == target, "winding number of h(gamma^r) disagrees with the argument count")
    check(target % k == 0 and target // k == r, "preimage arithmetic disagrees")

    a_wrong = [
        (m + n, "adds-degrees-in-composition"),
        (abs(k), "drops-sign-of-degree"),
        (m, "ignores-second-map"),
        (n, "ignores-second-map"),
        (m**n if n > 0 else None, "powers-for-composition"),
    ]
    b_wrong = [
        (k + j, "adds-degrees-in-composition"),
        (abs(k) * j, "drops-sign-of-degree"),
        (k, "ignores-the-loop-class"),
        (m * j, "ignores-second-map"),
        (n * j, "ignores-second-map"),
        ((m + n) * j, "adds-degrees-in-composition"),
    ]
    c_wrong = [
        (-k * k, "index-sign-kept"),
        (abs(k), "ignores-second-map"),
        (2 * abs(k), "adds-degrees-in-composition"),
        (k * k - 1, "counts-largest-residue"),
        (k * k + 1, "counts-largest-residue"),
    ]
    d_wrong = [
        (target * k, "multiplies-instead-of-dividing"),
        (target // abs(k), "drops-sign-of-degree"),
        (target, "ignores-the-map"),
        (target - k, "subtracts-instead-of-dividing"),
    ]
    stem = phrase(rng, _STEM, m=_power(m), n=_power(n))
    parts = [
        build_part(
            phrase(rng, _PROMPT_A),
            k,
            kind="number",
            marks=2,
            working=[
                f"f_* is multiplication by {m} and g_* is multiplication by {n}.",
                f"Induced maps compose, so h_* = g_* f_* is multiplication by {m} x {n} = {k}.",
            ],
            wrong=a_wrong,
        ),
        build_part(
            phrase(rng, _PROMPT_B, j=j),
            winding,
            kind="number",
            marks=1,
            working=[f"[h o lambda] = h_*({j}) = {k} x {j} = {winding}."],
            wrong=b_wrong,
            follow_through=f"part_a*{j}",
        ),
        build_part(
            phrase(rng, _PROMPT_C),
            k * k,
            kind="number",
            marks=1,
            working=[
                f"H_* is multiplication by {k} twice, that is by {k} x {k} = {k * k}.",
                f"The image is {k * k}Z, which has index {k * k} in Z.",
            ],
            wrong=c_wrong,
            follow_through="part_a^2",
        ),
        build_part(
            phrase(rng, _PROMPT_D, t=target),
            r,
            kind="number",
            marks=2,
            working=[f"h_*(r) = {k}r = {target}, so r = {target} / {k} = {r}."],
            wrong=d_wrong,
            follow_through=f"{target}/part_a",
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r2h_circle_maps_chain", level=18, presets={"exam": dict(tier="exam")})
def exd_x_r2h_circle_maps_chain(rng: random.Random, tier: str = "exam") -> dict:
    """The induced map of a composite of power maps of the circle, its action on a loop, its image, and a preimage."""
    check(tier == "exam", "exd_x_r2h_circle_maps_chain: only the exam tier exists")
    return _build(rng)
