"""Unit vectors + parallel/perpendicular properties, both senior ``acara-senior-v8-4``, domain ``vectors``.

Two generators:

- ``define_and_use_unit_vectors_and_the_perpendicular`` (ACMSM015)
  -- unit vectors and perpendicular unit vectors of 2D vectors with Pythagorean-triple magnitudes.
  Easy: unit vector of a first-quadrant vector. Medium: unit vector in any quadrant.
  Hard: perpendicular unit vector (rotated 90 degrees counter-clockwise).
  Answer kind: fraction (one coefficient of i or j).

- ``examine_properties_of_parallel_and_perpendicular_vectors_and`` (ACMSM021)
  -- find missing vector component or recover scalar from parallel/perpendicular relationships.
  Easy: missing component for perpendicular (dot product = 0). Medium: missing component for
  parallel (scalar multiple). Hard: recover the scalar k.
  Answer kind: number (integer).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, pick_distractors


# Fixed Pythagorean triple pool for skill 1
# Each tuple is (leg_a, leg_b, hypotenuse) with |a| != |b|
TRIPLES = [
    (3, 4, 5),
    (6, 8, 10),
    (5, 12, 13),
    (8, 15, 17),
    (9, 12, 15),
    (7, 24, 25),
    (20, 21, 29),
    (9, 40, 41),
    (12, 16, 20),
]


# --- Skill 1: define_and_use_unit_vectors_and_the_perpendicular ----------------


def _verify_unit_vector_magnitude_one(a: Fraction, b: Fraction, expected: Fraction) -> None:
    """Independent check for unit-vector tiers: verify Pythagorean unit-length identity."""
    # Check that the pair lies on the unit circle: (a/m)^2 + (b/m)^2 = 1
    check(a**2 + b**2 == 1, f"Unit vector coefficients {a} and {b} do not satisfy a^2 + b^2 = 1")


def _verify_perpendicular_unit_vector_magnitude_one(a: Fraction, b: Fraction, expected: Fraction) -> None:
    """Independent check for perpendicular tier: verify the perpendicular coefficients lie on unit circle."""
    # For perpendicular unit vector (-b/m, a/m): (-b/m)^2 + (a/m)^2 = 1
    check(a**2 + b**2 == 1, f"Perpendicular unit vector coefficients {a} and {b} do not satisfy a^2 + b^2 = 1")


def _verify_triple(leg_a: int, leg_b: int, m: int) -> None:
    """Verify that the triple is valid: leg_a^2 + leg_b^2 = m^2."""
    check(leg_a * leg_a + leg_b * leg_b == m * m, f"Triple ({leg_a}, {leg_b}, {m}) is not a valid Pythagorean triple")


@generator(
    "define_and_use_unit_vectors_and_the_perpendicular",
    level=11,
    presets={
        "easy": dict(variant="unit_positive"),
        "medium": dict(variant="unit_signed"),
        "hard": dict(variant="perpendicular"),
    },
)
def define_and_use_unit_vectors_and_the_perpendicular(
    rng,
    variant="unit_positive",
):
    """Define and use unit vectors and the perpendicular unit vectors."""

    # Pick a triple from the fixed pool
    leg_a, leg_b, m = rng.choice(TRIPLES)
    _verify_triple(leg_a, leg_b, m)

    # Determine signs and ask-axis based on variant
    if variant == "unit_positive":
        # Easy: both positive (first quadrant)
        a, b = leg_a, leg_b
        ask_i = rng.choice([True, False])  # Randomly choose which axis to ask for

        if ask_i:
            answer_frac = Fraction(a, m)
            axis_name = "i"
            text = f"v = ({a}, {b}). The magnitude of v is {m}. Find the coefficient of {axis_name} in the unit vector in the direction of v."
        else:
            answer_frac = Fraction(b, m)
            axis_name = "j"
            text = f"v = ({a}, {b}). The magnitude of v is {m}. Find the coefficient of {axis_name} in the unit vector in the direction of v."

        answer_text = str(answer_frac)
        _verify_unit_vector_magnitude_one(Fraction(a, m), Fraction(b, m), answer_frac)

        # Misconceptions:
        # 1. forgot-to-normalise: report raw leg
        wrong1 = Fraction(a if ask_i else b, 1)
        # 2. wrong-axis-unit-component: report OTHER coefficient
        wrong2 = Fraction(b if ask_i else a, m)
        # 3. divides-by-the-other-leg: Fraction(a, b) or Fraction(b, a)
        wrong3 = Fraction(a if ask_i else b, b if ask_i else a)

        candidates = [str(wrong1), str(wrong2), str(wrong3)]

        return dict(
            text=text,
            latex=rf"\frac{{{a}}}{{{m}}} \text{{ or }} \frac{{{b}}}{{{m}}}",
            answer={"p": answer_frac.numerator, "q": answer_frac.denominator},
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"Unit vector: ({a}/{m}, {b}/{m}) = ({Fraction(a, m)}, {Fraction(b, m)}). Coefficient of {axis_name}: {answer_frac}."
            ],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif variant == "unit_signed":
        # Medium: random signs on both legs (any quadrant)
        a = leg_a * rng.choice([1, -1])
        b = leg_b * rng.choice([1, -1])
        ask_i = rng.choice([True, False])

        if ask_i:
            answer_frac = Fraction(a, m)
            axis_name = "i"
            text = f"v = ({a}, {b}). The magnitude of v is {m}. Find the coefficient of {axis_name} in the unit vector in the direction of v."
        else:
            answer_frac = Fraction(b, m)
            axis_name = "j"
            text = f"v = ({a}, {b}). The magnitude of v is {m}. Find the coefficient of {axis_name} in the unit vector in the direction of v."

        answer_text = str(answer_frac)
        _verify_unit_vector_magnitude_one(Fraction(a, m), Fraction(b, m), answer_frac)

        # Misconceptions (same as easy):
        wrong1 = Fraction(a if ask_i else b, 1)
        wrong2 = Fraction(b if ask_i else a, m)
        wrong3 = Fraction(a if ask_i else b, b if ask_i else a)

        candidates = [str(wrong1), str(wrong2), str(wrong3)]

        return dict(
            text=text,
            latex=rf"\frac{{{a}}}{{{m}}} \text{{ or }} \frac{{{b}}}{{{m}}}",
            answer={"p": answer_frac.numerator, "q": answer_frac.denominator},
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"Unit vector: ({a}/{m}, {b}/{m}). Coefficient of {axis_name}: {answer_frac}."],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif variant == "perpendicular":
        # Hard: perpendicular unit vector (rotate 90 degrees counter-clockwise: (a,b) -> (-b,a))
        a = leg_a * rng.choice([1, -1])
        b = leg_b * rng.choice([1, -1])
        ask_i = rng.choice([True, False])

        # Perpendicular unit vector: (-b/m, a/m)
        if ask_i:
            answer_frac = Fraction(-b, m)
            axis_name = "i"
            text = f"v = ({a}, {b}), with magnitude {m}. The perpendicular unit vector (rotated 90 degrees counter-clockwise from v's own unit vector) is a unit vector in the direction of (-b, a). Find the coefficient of {axis_name} in the perpendicular unit vector."
        else:
            answer_frac = Fraction(a, m)
            axis_name = "j"
            text = f"v = ({a}, {b}), with magnitude {m}. The perpendicular unit vector (rotated 90 degrees counter-clockwise from v's own unit vector) is a unit vector in the direction of (-b, a). Find the coefficient of {axis_name} in the perpendicular unit vector."

        answer_text = str(answer_frac)
        _verify_perpendicular_unit_vector_magnitude_one(Fraction(-b, m), Fraction(a, m), answer_frac)

        # Misconceptions:
        # 1. used-original-units-unit-not-perpendicular: report original unit vector coefficient
        #    For i asked: Fraction(a, m); for j asked: Fraction(b, m)
        if ask_i:
            wrong1 = Fraction(a, m)
        else:
            wrong1 = Fraction(b, m)

        # 2. wrong-rotation-direction: rotate clockwise (b, -a) instead of counter-clockwise
        #    For i asked: Fraction(b, m); for j asked: Fraction(-a, m)
        if ask_i:
            wrong2 = Fraction(b, m)
        else:
            wrong2 = Fraction(-a, m)

        # 3. wrong-axis-perpendicular: report OTHER coefficient of perpendicular vector
        #    For i asked: report j-value of perpendicular = Fraction(a, m)
        #    For j asked: report i-value of perpendicular = Fraction(-b, m)
        #    BUT: when i is asked, Fraction(a, m) is ALSO the value of used-original-units
        #    when j is asked, Fraction(-b, m) is the answer itself
        if ask_i:
            # wrong-axis-perpendicular gives Fraction(a, m), which equals used-original-units
            # So we only include it when it differs; if it's the same, pick_distractors will dedupe
            wrong3 = Fraction(a, m)
        else:
            # wrong-axis-perpendicular gives Fraction(-b, m), which is the answer
            # This is not a valid distractor for j-asked; skip it
            wrong3 = None

        if wrong3 is not None:
            candidates = [str(wrong1), str(wrong2), str(wrong3)]
        else:
            candidates = [str(wrong1), str(wrong2)]

        return dict(
            text=text,
            latex=rf"\frac{{{-b}}}{{{m}}} \text{{ or }} \frac{{{a}}}{{{m}}}",
            answer={"p": answer_frac.numerator, "q": answer_frac.denominator},
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"Perpendicular unit vector: (-b/m, a/m) = ({-b}/{m}, {a}/{m}). Coefficient of {axis_name}: {answer_frac}."
            ],
            distractors=pick_distractors(answer_text, candidates),
        )

    raise ValueError(f"unknown variant: {variant!r}")


# --- Skill 2: examine_properties_of_parallel_and_perpendicular_vectors_and ------


def _verify_perpendicular_dot_product_zero(p: int, q: int, vx: int, vy: int) -> None:
    """Independent check for easy tier: dot product equals zero."""
    check(p * vx + q * vy == 0, f"Dot product of ({p}, {q}) and ({vx}, {vy}) is not zero")


def _verify_parallel_proportionality(vx: int, vy: int, p: int, q: int) -> None:
    """Independent check for medium tier: cross-multiplication proportionality test."""
    check(vx * q == p * vy, f"Vectors ({vx}, {vy}) and ({p}, {q}) are not proportional")


def _verify_recover_scalar_from_vectors(k_answer: int, p: int, q: int, A: int, B: int) -> None:
    """Independent check for hard tier: both proportionality and scalar recomputation."""
    # Proportionality check
    check(A * q == B * p, f"Proportionality check: {A} * {q} != {B} * {p}")
    # Independent recomputation via integer division on B and q
    check(B // q == k_answer, f"Scalar recomputation: {B} // {q} = {B // q}, expected {k_answer}")


@generator(
    "examine_properties_of_parallel_and_perpendicular_vectors_and",
    level=11,
    presets={
        "easy": dict(variant="perpendicular"),
        "medium": dict(variant="parallel"),
        "hard": dict(variant="recover_scalar"),
    },
)
def examine_properties_of_parallel_and_perpendicular_vectors_and(
    rng,
    variant="perpendicular",
):
    """Examine properties of parallel and perpendicular vectors."""

    if variant == "perpendicular":
        # Easy: find missing component for perpendicular relationship (dot product = 0)
        # Sample p, q each in 1..9, forced p != q
        while True:
            p = rng.randint(1, 9)
            q = rng.randint(1, 9)
            if p != q:
                break

        # Sample nonzero scale t in {1,2,3,4} with random sign
        t = rng.choice([1, 2, 3, 4]) * rng.choice([1, -1])

        # Perpendicular vector: v = (t*q, -t*p)
        # Randomly choose which axis is given
        if rng.choice([True, False]):  # x given
            vx = t * q
            vy = -t * p
            text = f"u = ({p}, {q}). v = ({vx}, ?) is perpendicular to u (u . v = 0). Find the y-component of v."
            answer = vy
        else:  # y given
            vx = t * q
            vy = -t * p
            text = f"u = ({p}, {q}). v = (?, {vy}) is perpendicular to u (u . v = 0). Find the x-component of v."
            answer = vx

        answer_text = str(answer)
        _verify_perpendicular_dot_product_zero(p, q, vx, vy)

        # Misconceptions:
        # 1. dropped-the-sign: report blank without sign (only on -t*p blank)
        if vy == -t * p and answer == vy:
            wrong1 = t * p  # Missing sign flip
        elif vx == t * q and answer == vx:
            wrong1 = None  # Can't drop sign if it's positive
        else:
            wrong1 = None

        # 2. swapped-components-of-u: use q instead of p
        if answer == -t * p:
            wrong2 = -t * q
        else:
            wrong2 = t * p

        # 3. used-given-coordinate-directly: report the OTHER coordinate
        if answer == vx:
            wrong3 = vy
        else:
            wrong3 = vx

        candidates = []
        if wrong1 is not None:
            candidates.append(str(wrong1))
        candidates.append(str(wrong2))
        candidates.append(str(wrong3))

        return dict(
            text=text,
            latex=rf"u \cdot v = 0 \Rightarrow ({p}) \cdot ({vx}) + ({q}) \cdot (?) = 0",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"u . v = 0: {p}×{vx if answer == vy else '?'} + {q}×{vy if answer == vy else '?'} = 0. If x={vx}, then {p}×{vx} + {q}×vy = 0, so vy = {vy}."
            ],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif variant == "parallel":
        # Medium: find missing component for parallel relationship (scalar multiple)
        # Sample p, q each in 1..9, forced p != q
        while True:
            p = rng.randint(1, 9)
            q = rng.randint(1, 9)
            if p != q:
                break

        # Sample nonzero scale k (never 1 or -1)
        k = rng.choice([2, 3, 4, 5, -2, -3, -4, -5])

        # Randomly decide which axis is given
        while True:
            # Loop to handle resample in case of collision
            if rng.choice([True, False]):  # x given
                vx = k * p
                vy = k * q
                text = f"u = ({p}, {q}). v = ({vx}, ?) is parallel to u. Find the y-component of v."
                answer = vy
            else:  # y given
                vx = k * p
                vy = k * q
                text = f"u = ({p}, {q}). v = (?, {vy}) is parallel to u. Find the x-component of v."
                answer = vx

            # Check for distractor collisions and resample if needed
            # Misconception 1: wrong-component-of-u-used
            if answer == vy:
                wrong1 = k * p  # Wrong component
            else:
                wrong1 = k * q

            # Misconception 2: assumed-k-equals-one
            if answer == vy:
                wrong2 = q  # Unscaled
            else:
                wrong2 = p

            # Misconception 3: used-negative-scale
            wrong3 = -answer

            # Check if all are distinct and none equal answer
            if (
                wrong1 != answer
                and wrong2 != answer
                and wrong3 != answer
                and len({answer, wrong1, wrong2, wrong3}) == 4
            ):
                break
            # Otherwise resample this seed (p, q, k, axis)

        answer_text = str(answer)
        _verify_parallel_proportionality(vx, vy, p, q)

        candidates = [str(wrong1), str(wrong2), str(wrong3)]

        return dict(
            text=text,
            latex=rf"v = k \cdot u \Rightarrow (?, {vy if answer == vx else '?'}) = {k} \cdot ({p}, {q})",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"v = {k} × u, so v = ({k}×{p}, {k}×{q}) = ({vx}, {vy}). Missing component: {answer}."],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif variant == "recover_scalar":
        # Hard: recover the scalar k from two fully-specified vectors
        # Sample p, q each in 1..9, forced p != q
        while True:
            p = rng.randint(1, 9)
            q = rng.randint(1, 9)
            if p != q:
                break

        # Sample k (same set as medium)
        while True:
            k = rng.choice([2, 3, 4, 5, -2, -3, -4, -5])
            A = k * p
            B = k * q

            # Misconceptions:
            # 1. subtracted-instead-of-divided: A - p = (k-1)*p
            wrong1 = A - p

            # 2. confused-k-with-a-component: report p (or q)
            wrong2 = q  # Use q to avoid collision when p == k

            # Check for collisions: resample if:
            # - wrong1 == k (collision with answer)
            # - wrong1 == q (collision with wrong2)
            # - p == k or q == k (collision with confused misconception)
            if wrong1 != k and wrong1 != q and p != k and q != k:
                break

        answer = k
        answer_text = str(answer)
        _verify_recover_scalar_from_vectors(k, p, q, A, B)

        text = f"u = ({p}, {q}). v = ({A}, {B}) is a scalar multiple of u (v = k × u for some integer k). Find k."
        candidates = [str(wrong1), str(wrong2)]

        return dict(
            text=text,
            latex=rf"v = k \cdot u \Rightarrow ({A}, {B}) = k \cdot ({p}, {q})",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"v = k × u, so ({A}, {B}) = k × ({p}, {q}). Therefore k = {A} / {p} = {k}."],
            distractors=pick_distractors(answer_text, candidates),
        )

    raise ValueError(f"unknown variant: {variant!r}")
