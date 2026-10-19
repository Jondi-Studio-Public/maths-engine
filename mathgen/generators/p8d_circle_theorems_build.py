"""Circle theorems -- angle at centre, intersecting chords.

- ``the_angle_at_the_centre_subtended_by_an_arc``
  (``senior-the-angle-at-the-centre-subtended-by-an``, ACMSM030, domain
  senior-mathematics, answer: number) -- The angle at the centre subtended by an
  arc of a circle is twice the angle at the circumference subtended by the same
  arc.

- ``when_two_chords_of_a_circle_intersect_the_product``
  (``senior-when-two-chords-of-a-circle-intersect-the``, ACMSM035, domain
  measurement, answer: number) -- When two chords of a circle intersect, the
  product of the lengths of the intervals on one chord equals the product of
  the lengths of the intervals on the other chord.

**Boundary against the neighbouring, already-shipped circle theorems**
(``mathgen/generators/p5b_circle_theorems_build.py``): ``angle_in_semicircle``
(Thales' theorem: an angle in a SEMICIRCLE, i.e. subtended by a DIAMETER, is a
right angle) and ``angles_same_arc`` (two circumference points on the SAME arc
subtend EQUAL angles) are both statements about angles at the circumference
only. ``the_angle_at_the_centre_subtended_by_an_arc`` is a different
geometric relationship: it relates a CENTRE angle to a CIRCUMFERENCE angle
(the centre angle is always twice the circumference angle subtended by the
same arc), and it deliberately excludes the diameter case (the easy/medium
tiers' central angle is bounded strictly below 180 degrees) so it never
degenerates into the semicircle theorem's territory.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, check_distinct, generator, pick_distractors


# ============================================================================
# Skill 1: the_angle_at_the_centre_subtended_by_an_arc (ACMSM030)
# ============================================================================


def _verify_angle_at_centre(minor_arc_span_deg: float, p_on_major_arc: bool, expected_deg: float) -> None:
    """Independent coordinate-geometry re-derivation of the inscribed angle.

    Places A at 0 degrees and B at ``minor_arc_span_deg`` on a unit circle
    centred at O, so the SHORT way round from A to B spans
    ``minor_arc_span_deg`` (always < 180). P is placed on the requested arc
    (the far side of AB from the minor arc's midpoint if ``p_on_major_arc``,
    otherwise the minor arc's own midpoint), and angle APB is computed via
    the dot product between vectors PA and PB -- a route that never uses the
    halve-the-central-angle formula the generator itself applies to derive
    ``expected_deg``. Raises ``VerificationError`` if the two disagree.
    """
    check(0 < minor_arc_span_deg < 180, "minor_arc_span_deg must be the short way round")
    a_rad = 0.0
    b_rad = math.radians(minor_arc_span_deg)
    if p_on_major_arc:
        p_rad = math.radians(minor_arc_span_deg + 180.0)
    else:
        p_rad = math.radians(minor_arc_span_deg / 2.0)
    ax, ay = math.cos(a_rad), math.sin(a_rad)
    bx, by = math.cos(b_rad), math.sin(b_rad)
    px, py = math.cos(p_rad), math.sin(p_rad)
    pa = (ax - px, ay - py)
    pb = (bx - px, by - py)
    dot = pa[0] * pb[0] + pa[1] * pb[1]
    mag_pa = math.hypot(*pa)
    mag_pb = math.hypot(*pb)
    cos_angle = max(-1.0, min(1.0, dot / (mag_pa * mag_pb)))
    computed_deg = math.degrees(math.acos(cos_angle))
    check(
        abs(computed_deg - expected_deg) < 1e-6,
        f"angle_at_centre: computed angle APB={computed_deg:.6f}, expected {expected_deg}",
    )


@generator(
    "the_angle_at_the_centre_subtended_by_an_arc",
    level=12,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="split"),
        "hard": dict(variant="reflex"),
    },
)
def the_angle_at_the_centre_subtended_by_an_arc(rng, variant="direct"):
    """The angle at the centre subtended by an arc is twice the angle at the
    circumference subtended by the same arc.

    Every question fully specifies which arc the circumference point P sits
    on, because the relationship flips between the ordinary central angle
    and its reflex depending on that choice.
    """

    if variant == "direct":
        # Easy: circle centre O, P on the MAJOR arc AB, angle AOB (non-reflex)
        # given directly as an even integer strictly between 20 and 160
        # (never 180 -- that is angle_in_semicircle's territory).
        while True:
            x = rng.randint(21, 159)
            if x % 2 == 0:
                break

        angle_apb = x // 2

        question_text = (
            f"A circle has centre O. Points A and B lie on the circle, and point P "
            f"lies on the major arc AB. Angle AOB, the non-reflex angle at the centre, "
            f"is {x} degrees. Find angle APB, in degrees."
        )

        _verify_angle_at_centre(x, True, angle_apb)

        reports_central = x
        doubles = 2 * x
        check_distinct(
            angle_apb, reports_central, doubles, msg=f"the_angle_at_the_centre/direct: distractor collides for x={x}"
        )

        candidates = [str(reports_central), str(doubles)]
        working = [
            "The angle at the centre is twice the angle at the circumference subtended by the same arc.",
            f"Angle APB = {x} / 2 = {angle_apb} degrees.",
        ]

    elif variant == "split":
        # Medium: central angle AOB split by a third radius OC into two named
        # parts, angle AOC = p and angle COB = q. Learner must sum then halve.
        while True:
            p = rng.randint(5, 75)
            q = rng.randint(5, 75)
            total = p + q
            if 20 < total < 160 and total % 2 == 0:
                break

        angle_apb = total // 2

        question_text = (
            f"A circle has centre O. Points A and B lie on the circle, and point P "
            f"lies on the major arc AB. A third radius OC lies between OA and OB, "
            f"splitting angle AOB into two parts: angle AOC is {p} degrees and "
            f"angle COB is {q} degrees. Find angle APB, in degrees."
        )

        _verify_angle_at_centre(total, True, angle_apb)

        reports_central = total
        doubles = 2 * total
        check_distinct(
            angle_apb,
            reports_central,
            doubles,
            msg=f"the_angle_at_the_centre/split: distractor collides for p={p}, q={q}",
        )

        candidates = [str(reports_central), str(doubles)]
        working = [
            f"First add the two parts of the central angle: angle AOB = {p} + {q} = {total} degrees.",
            "The angle at the centre is twice the angle at the circumference subtended by the same arc.",
            f"Angle APB = {total} / 2 = {angle_apb} degrees.",
        ]

    elif variant == "reflex":
        # Hard: P on the MINOR arc AB, the REFLEX angle AOB given (even integer
        # strictly between 200 and 340). angle APB = reflex / 2.
        while True:
            r = rng.randint(201, 339)
            if r % 2 == 0:
                break

        angle_apb = r // 2
        minor_arc_span = 360 - r  # strictly between 20 and 160

        question_text = (
            f"A circle has centre O. Points A and B lie on the circle, and point P "
            f"lies on the minor arc AB. The reflex angle AOB, measured the long way "
            f"round from OA to OB, is {r} degrees. Find angle APB, in degrees."
        )

        _verify_angle_at_centre(minor_arc_span, False, angle_apb)

        reports_central = r
        doubles = 2 * r
        uses_wrong_arc = (360 - r) // 2
        check_distinct(
            angle_apb,
            reports_central,
            doubles,
            uses_wrong_arc,
            msg=f"the_angle_at_the_centre/reflex: distractor collides for r={r}",
        )

        candidates = [str(reports_central), str(doubles), str(uses_wrong_arc)]
        working = [
            "Since P lies on the minor arc AB, the angle at the circumference is half "
            "of the reflex angle at the centre.",
            f"Angle APB = {r} / 2 = {angle_apb} degrees.",
        ]

    else:
        raise ValueError(f"unknown variant {variant!r}")

    answer_text = str(angle_apb)
    distractors = pick_distractors(answer_text, candidates)

    return {
        "text": question_text,
        "latex": question_text,
        "answer": angle_apb,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": working,
    }


# ============================================================================
# Skill 2: when_two_chords_of_a_circle_intersect_the_product (ACMSM035)
# ============================================================================


def _verify_chord_products(ap, pb, cp, pd) -> None:
    """Independent confirmation of AP × PB = CP × PD using exact Fraction
    arithmetic -- the route the generator's own solve-for-the-unknown
    division does not take."""
    check(
        Fraction(ap) * Fraction(pb) == Fraction(cp) * Fraction(pd),
        f"chord product check failed: {ap}*{pb} != {cp}*{pd}",
    )


def _verify_quadratic_positive_root(k: int, cd: int, x_true: int) -> None:
    """Independent resolution of x(x + k) = cd via the quadratic formula,
    using exact integer arithmetic (math.isqrt on the discriminant), never
    reusing however the generator constructed x_true."""
    disc = k * k + 4 * cd
    root = math.isqrt(disc)
    check(root * root == disc, f"discriminant {disc} is not a perfect square")
    check((root - k) % 2 == 0, "positive root is not an integer")
    positive_root = (root - k) // 2
    other_root_numerator = -k - root
    check(other_root_numerator < 0, "expected the other root to be negative")
    check(positive_root > 0, "expected the positive root to be positive")
    check(positive_root == x_true, f"quadratic formula gives x={positive_root}, expected {x_true}")


# Pools precomputed once at import time (they do not depend on rng), keeping
# each make() call to a handful of arithmetic operations well under the
# generator-speed budget.


def _build_easy_chord_pool():
    pool = []
    for u in range(3, 13):
        for v in range(3, 13):
            prod = u * v
            for cp in range(2, prod):
                if prod % cp != 0:
                    continue
                pd = prod // cp
                if pd < 2 or pd > 60:
                    continue
                if len({u, v, cp, pd}) < 4:
                    continue
                wrong_add = u + v - cp
                if wrong_add <= 0 or wrong_add == pd:
                    continue
                if (u * cp) % v != 0:
                    continue
                wrong_pair = (u * cp) // v
                if wrong_pair == pd or wrong_pair == wrong_add or wrong_pair <= 0:
                    continue
                pool.append((u, v, cp, pd, wrong_add, wrong_pair))
    return pool


def _build_medium_chord_pool():
    # The mismatched unit sits on AP (a numerator segment): AP is given in mm
    # while PB, CP, PD are all in cm. This mirrors the easy pool's own
    # (u, v, cp, pd) search exactly -- no "pd must be a multiple of 10"
    # constraint is needed, because "forgets to convert" here is simply
    # 10 x pd (using the raw mm figure keeps every length 10x too large),
    # which is always an integer regardless of pd. An earlier version put
    # the mismatched unit on CP (a divisor) and required pd % 10 == 0 to
    # keep that version's "forgets to convert" distractor an integer -- that
    # constraint starved the pool down to AP in {10, 20} for 60/60 rendered
    # seeds. This shape has no such constraint.
    pool = []
    for u in range(3, 13):
        for v in range(3, 13):
            prod = u * v
            for cp in range(2, prod):
                if prod % cp != 0:
                    continue
                pd = prod // cp
                if pd < 2 or pd > 60:
                    continue
                if len({u, v, cp, pd}) < 4:
                    continue
                wrong_add = u + v - cp
                if wrong_add <= 0 or wrong_add == pd:
                    continue
                if (u * cp) % v != 0:
                    continue
                wrong_pair = (u * cp) // v
                if wrong_pair in (pd, wrong_add) or wrong_pair <= 0:
                    continue
                wrong_convert = 10 * pd
                if wrong_convert in (pd, wrong_add, wrong_pair):
                    continue
                pool.append((u, v, cp, pd, wrong_add, wrong_pair, wrong_convert))
    return pool


def _build_hard_chord_pool():
    pool = []
    for x_true in range(2, 19):
        for k in range(2, 19):
            cd = x_true * (x_true + k)
            for c in range(2, cd):
                if cd % c != 0:
                    continue
                d = cd // c
                if c == d:
                    continue
                if c in (x_true, x_true + k) or d in (x_true, x_true + k):
                    continue
                s = c + d - k
                if s <= 0 or s % 2 != 0:
                    continue
                wrong_add = s // 2
                if wrong_add == x_true or wrong_add <= 0:
                    continue
                if not (c > d and (k * d) % (c - d) == 0):
                    continue
                wrong_pair = (k * d) // (c - d)
                if wrong_pair == x_true or wrong_pair == wrong_add or wrong_pair <= 0:
                    continue
                pool.append((x_true, k, c, d, wrong_add, wrong_pair))
    return pool


_EASY_CHORD_POOL = _build_easy_chord_pool()
_MEDIUM_CHORD_POOL = _build_medium_chord_pool()
_HARD_CHORD_POOL = _build_hard_chord_pool()


@generator(
    "when_two_chords_of_a_circle_intersect_the_product",
    level=12,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="convert"),
        "hard": dict(variant="quadratic"),
    },
)
def when_two_chords_of_a_circle_intersect_the_product(rng, variant="direct"):
    """Two chords AB and CD of a circle intersect at an interior point P.
    AP times PB = CP times PD (the product of the intervals on one chord
    equals the product of the intervals on the other chord)."""

    if variant == "direct":
        # Easy: three of the four segments given directly in cm, solve the
        # fourth (PD) via division.
        u, v, cp, pd, wrong_add, wrong_pair = rng.choice(_EASY_CHORD_POOL)

        question_text = (
            f"Two chords AB and CD of a circle intersect at a point P inside the "
            f"circle. AP is {u} cm, PB is {v} cm and CP is {cp} cm. Find the length "
            f"of PD, in cm."
        )

        _verify_chord_products(u, v, cp, pd)
        check_distinct(
            pd, wrong_add, wrong_pair, msg=f"when_two_chords/direct: distractor collides for u={u}, v={v}, cp={cp}"
        )

        candidates = [str(wrong_add), str(wrong_pair)]
        working = [
            f"By the intersecting chords rule, AP × PB = CP × PD.",
            f"{u} × {v} = {cp} × PD, so PD = {u * v} / {cp} = {pd} cm.",
        ]
        answer = pd

    elif variant == "convert":
        # Medium: AP given in a different metric unit (mm) from PB/CP (cm).
        # Learner must convert before applying the product rule.
        u, v, cp, pd, wrong_add, wrong_pair, wrong_convert = rng.choice(_MEDIUM_CHORD_POOL)
        ap_mm = u * 10

        question_text = (
            f"Two chords AB and CD of a circle intersect at a point P inside the "
            f"circle. AP is {ap_mm} mm, PB is {v} cm and CP is {cp} cm. Find the "
            f"length of PD, in cm."
        )

        _verify_chord_products(u, v, cp, pd)
        check_distinct(
            pd,
            wrong_add,
            wrong_pair,
            wrong_convert,
            msg=f"when_two_chords/convert: distractor collides for u={u}, v={v}, cp={cp}",
        )

        candidates = [str(wrong_add), str(wrong_pair), str(wrong_convert)]
        working = [
            f"Convert AP to the same unit as the other lengths: {ap_mm} mm = {u} cm.",
            f"By the intersecting chords rule, AP × PB = CP × PD.",
            f"{u} × {v} = {cp} × PD, so PD = {u * v} / {cp} = {pd} cm.",
        ]
        answer = pd

    elif variant == "quadratic":
        # Hard: AP = x, PB = x + k (same chord, unknown), CP and PD given
        # directly. x(x + k) = CP x PD, a quadratic with exactly one
        # positive root.
        x_true, k, c, d, wrong_add, wrong_pair = rng.choice(_HARD_CHORD_POOL)
        cd = c * d

        question_text = (
            f"Two chords AB and CD of a circle intersect at a point P inside the "
            f"circle. AP is x cm and PB is (x + {k}) cm, where x is a positive "
            f"integer. CP is {c} cm and PD is {d} cm. Find the length of AP, in cm."
        )

        _verify_chord_products(x_true, x_true + k, c, d)
        _verify_quadratic_positive_root(k, cd, x_true)
        check_distinct(
            x_true,
            wrong_add,
            wrong_pair,
            msg=f"when_two_chords/quadratic: distractor collides for x={x_true}, k={k}, c={c}, d={d}",
        )

        candidates = [str(wrong_add), str(wrong_pair)]
        other_root = -(x_true + k)
        working = [
            f"By the intersecting chords rule, AP × PB = CP × PD, so x(x + {k}) = {c} × {d} = {cd}.",
            f"This gives x^2 + {k}x - {cd} = 0, which factorises as (x - {x_true})(x + {x_true + k}) = 0.",
            f"So x = {x_true} or x = {other_root}; a length cannot be negative, so x = {x_true} cm.",
        ]
        answer = x_true

    else:
        raise ValueError(f"unknown variant {variant!r}")

    answer_text = str(answer)
    distractors = pick_distractors(answer_text, candidates)

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": working,
    }
