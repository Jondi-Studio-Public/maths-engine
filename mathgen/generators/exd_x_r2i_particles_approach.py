"""Exam-tier chain for `senior-examine-the-position-of-two-particles-each-described`: closest approach and range.

Two drones fly in straight lines with position vectors ``r_A(t)`` and ``r_B(t)`` (metres, ``t`` in seconds). They
never meet, but a radio link works while they are within ``R`` metres. The chain asks for

(a) the time of closest approach,
(b) the least distance between them, from the separation at that time,
(c) the length of time they stay within range, from (b), and
(d) the time they first come within range, from (a) and (c).

## Construction: backwards from the geometry

The relative position is ``d(t) = r_B(t) - r_A(t) = m + w (t - t*)`` where the relative velocity ``w`` is one of a
few vectors of length 5 or 10 and ``m`` is perpendicular to ``w``. So ``|d|`` is least at ``t*`` and the least
distance is ``|m|``. The range is built from a Pythagorean triple ``(|m|, h, R)``: they are within range while
``|t - t*| <= h / |w|``. Every answer is exact at two decimal places.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, join_terms, paren_negative

from .exd_x_r2i_support import number_part

# (|m|, h, R) with |m| a multiple of 5 so k = |m| / |w| is a whole or half number.
_TRIPLES = [
    (5, 12, 13),
    (10, 24, 26),
    (15, 8, 17),
    (15, 20, 25),
    (20, 15, 25),
    (20, 21, 29),
    (30, 16, 34),
    (30, 40, 50),
    (35, 12, 37),
    (40, 9, 41),
    (25, 60, 65),
]
_RELATIVE = [(3, 4), (4, 3), (6, 8), (8, 6)]
_VEHICLES = [("drone", "drones"), ("robot", "robots"), ("boat", "boats")]
_INTROS = [
    "Two {plural} move in a plane. Relative to a base station, their positions in metres at time t seconds (t ≥ 0) are",
    "Two {plural} are tracked from a control point. At time t seconds (t ≥ 0) their position vectors, in metres, are",
]
_LINKS = [
    "The two {plural} can communicate while they are no more than {r} m apart.",
    "A radio link between the two {plural} works while the distance between them is at most {r} m.",
]


def _vec(x0: int, vx: int) -> str:
    """The text of one coordinate ``x0 + vx t`` (e.g. ``-3 + 2t``, ``5 - t``, ``4``)."""
    if vx == 0:
        return str(x0)
    t = "t" if abs(vx) == 1 else f"{abs(vx)}t"
    sign = "-" if vx < 0 else "+"
    if x0 == 0:
        return f"-{t}" if vx < 0 else t
    return f"{x0} {sign} {t}"


@generator("exd_x_r2i_particles_approach", level=20, presets={"exam": {"tier": "exam"}})
def exd_x_r2i_particles_approach(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: two linear trajectories that never meet: time and distance of closest approach, and range."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a closest-approach configuration with the range window after t = 0"):
        w1, w2 = rng.choice(_RELATIVE)
        w = (w1 * rng.choice([-1, 1]), w2 * rng.choice([-1, 1]))
        wlen = 5 if (abs(w[0]), abs(w[1])) in ((3, 4), (4, 3)) else 10
        mlen, h, radius = rng.choice(_TRIPLES)
        k = Fraction(mlen, wlen) * rng.choice([-1, 1])  # m = k * (-w2, w1)
        m = (-k * w[1], k * w[0])
        if m[0].denominator != 1 or m[1].denominator != 1:
            continue
        t_star = Fraction(rng.randint(1, 18), 2) if wlen == 10 else Fraction(rng.randint(2, 9))
        half_width = Fraction(h, wlen)
        if t_star <= half_width:
            continue
        d0 = (m[0] - w[0] * t_star, m[1] - w[1] * t_star)
        if d0[0].denominator != 1 or d0[1].denominator != 1:
            continue
        break
    m = (int(m[0]), int(m[1]))
    d0 = (int(d0[0]), int(d0[1]))
    va = (rng.randint(-4, 4), rng.randint(-4, 4))
    vb = (va[0] + w[0], va[1] + w[1])
    a0 = (rng.randint(-12, 12), rng.randint(-12, 12))
    b0 = (a0[0] + d0[0], a0[1] + d0[1])

    # Independent check: the squared separation at t* is |m|^2 and is larger half a second either side.
    def separation_sq(time):
        dx = b0[0] - a0[0] + (vb[0] - va[0]) * time
        dy = b0[1] - a0[1] + (vb[1] - va[1]) * time
        return dx * dx + dy * dy

    check(separation_sq(t_star) == mlen**2, "least distance disagrees")
    step = Fraction(1, 2)
    check(separation_sq(t_star - step) > mlen**2 and separation_sq(t_star + step) > mlen**2, "closest time disagrees")
    check(separation_sq(t_star - half_width) == radius**2 == separation_sq(t_star + half_width), "range disagrees")
    check(radius**2 - mlen**2 == h**2, "range triple is not Pythagorean")

    least = Fraction(mlen)
    duration = 2 * half_width
    first = t_star - half_width
    check(first > 0, "the particles are already within range at t = 0")

    _, plural = rng.choice(_VEHICLES)
    stem = (
        f"{rng.choice(_INTROS).format(plural=plural)} "
        f"r_A(t) = ({_vec(a0[0], va[0])}, {_vec(a0[1], va[1])}) and "
        f"r_B(t) = ({_vec(b0[0], vb[0])}, {_vec(b0[1], vb[1])}). "
        f"{rng.choice(_LINKS).format(plural=plural, r=radius)}"
    )
    # Separation d(t) = (dx0 + wx t, dy0 + wy t); its squared length is minimal at the closest-approach time.
    dot = d0[0] * w[0] + d0[1] * w[1]
    wsq = w[0] ** 2 + w[1] ** 2
    places_t = 1 if t_star.denominator == 2 else 0
    ft_b = f"sqrt(({d0[0]} + ({w[0]})*part_a)**2 + ({d0[1]} + ({w[1]})*part_a)**2)"
    ft_c = f"2*sqrt({radius}**2 - part_b**2)/{wlen}"
    ft_d = "part_a - part_c/2"

    # The tolerance of a "number" grader follows the displayed places, so times are shown at two places throughout.
    parts = [
        number_part(
            f"Find the time, in seconds, at which the two {plural} are closest together.",
            t_star,
            places_t,
            2,
            [
                f"Separation d(t) = r_B − r_A = ({join_terms([(d0[0], ''), (w[0], 't')])}, {join_terms([(d0[1], ''), (w[1], 't')])}).",
                f"|d|² is least when d·d′ = 0: {join_terms([(dot, ''), (wsq, 't')])} = 0 gives t = {-Fraction(dot, wsq)}.",
            ],
            [
                (Fraction(-d0[0], w[0]), "solved-one-component-only"),
                (Fraction(-d0[1], w[1]), "solved-one-component-only"),
                (Fraction(-dot, wlen), "divided-by-wrong-amount"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the least distance, in metres, between the two {plural}. Give your answer to 2 decimal places.",
            least,
            2,
            2,
            [
                f"At t = {t_star}: d = ({m[0]}, {m[1]}).",
                f"|d| = √({paren_negative(m[0])}² + {paren_negative(m[1])}²) = {mlen} m.",
            ],
            [
                (Fraction(abs(m[0]) + abs(m[1])), "added-components-not-magnitude"),
                (Fraction(round((d0[0] ** 2 + d0[1] ** 2) ** 0.5)), "evaluated-at-wrong-point"),
                (Fraction(mlen**2), "left-out-the-square-root"),
            ],
            follow_through=ft_b,
            sensible_range=(0, None),
        ),
        number_part(
            f"For how long, in seconds, are the two {plural} within {radius} m of each other? "
            "Give your answer to 2 decimal places.",
            duration,
            2,
            2,
            [
                f"|d|² = {radius}² gives {wlen}²(t − {t_star})² + {mlen}² = {radius}².",
                f"|t − {t_star}| = {h}/{wlen}, so the length of time is 2 × {h}/{wlen} = {float(duration):.2f} s.",
            ],
            [
                (half_width, "gave-half-the-interval"),
                (Fraction(2 * radius, wlen), "ignored-the-closest-distance"),
                (Fraction(2 * (radius - mlen), wlen), "subtracted-distances-not-squares"),
                (Fraction(h, 1), "divided-by-wrong-amount"),
            ],
            follow_through=ft_c,
            sensible_range=(0, None),
        ),
        number_part(
            f"At what time do the two {plural} first come within {radius} m of each other? "
            "Give your answer to 2 decimal places.",
            first,
            2,
            1,
            [f"First time = {t_star} − {float(half_width):.2f} = {float(first):.2f} s."],
            [
                (t_star + half_width, "gave-end-time-not-start"),
                (t_star - duration, "gave-half-the-interval"),
                (duration, "gave-interval-not-time"),
                (t_star, "gave-closest-approach-time"),
            ],
            follow_through=ft_d,
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)
