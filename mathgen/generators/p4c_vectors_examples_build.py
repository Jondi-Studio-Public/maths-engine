"""Examples of vectors including displacement and velocity, senior ``acara-senior-v8-4``, domain ``vectors``.

Single generator:

- ``examine_examples_of_vectors_including_displacement_and_velocity`` (ACMSM010)
  -- magnitude of displacement or velocity from perpendicular components.
  Easy: single displacement from two legs, direct Pythagorean computation.
  Medium: velocity with irrelevant elapsed-time number that must be ignored.
  Hard: two-leg journey requiring vector addition before magnitude computation.
  Answer kind: number (integer).
"""

from __future__ import annotations

from ..core import check, generator, pick_distractors


# Fixed Pythagorean triple pool (leg_a, leg_b, hypotenuse) with both legs positive
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


def _verify_magnitude_from_legs(leg_a: int, leg_b: int, expected: int) -> None:
    """Independent check: verify Pythagorean identity from first principles."""
    check(
        expected * expected == leg_a * leg_a + leg_b * leg_b,
        f"Magnitude {expected}^2 != {leg_a}^2 + {leg_b}^2 ({leg_a * leg_a + leg_b * leg_b})",
    )


def _verify_triple(leg_a: int, leg_b: int, m: int) -> None:
    """Verify that the triple is valid: leg_a^2 + leg_b^2 = m^2."""
    check(leg_a * leg_a + leg_b * leg_b == m * m, f"Triple ({leg_a}, {leg_b}, {m}) is not a valid Pythagorean triple")


@generator(
    "examine_examples_of_vectors_including_displacement_and_velocity",
    level=11,
    presets={
        "easy": dict(variant="single_leg"),
        "medium": dict(variant="velocity_with_red_herring"),
        "hard": dict(variant="two_leg_journey"),
    },
)
def examine_examples_of_vectors_including_displacement_and_velocity(
    rng,
    variant="single_leg",
):
    """Examine examples of vectors including displacement and velocity."""

    if variant == "single_leg":
        # Easy: pick a triple, ask for magnitude of single displacement
        leg_a, leg_b, m = rng.choice(TRIPLES)
        _verify_triple(leg_a, leg_b, m)

        # Pick noun/direction pair
        noun = rng.choice(["cyclist", "hiker", "drone"])
        directions = rng.choice([("east", "north"), ("west", "south"), ("east", "south"), ("west", "north")])
        dir1, dir2 = directions
        noun_poss = f"{noun}'s" if noun != "drone" else "drone's"

        text = f"A {noun} travels {leg_a} km {dir1} then {leg_b} km {dir2}. Find the magnitude of the {noun_poss} displacement, in km."
        answer = m
        answer_text = str(answer)

        # Independent check: verify Pythagorean identity
        _verify_magnitude_from_legs(leg_a, leg_b, m)

        # Misconceptions:
        # 1. added-instead-of-pythagoras: leg_a + leg_b
        wrong1 = leg_a + leg_b
        # 2. forgot-one-leg: report larger leg alone
        wrong2 = max(leg_a, leg_b)

        candidates = [str(wrong1), str(wrong2)]

        return dict(
            text=text,
            latex=text,
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"Displacement: sqrt({leg_a}^2 + {leg_b}^2) = sqrt({leg_a * leg_a + leg_b * leg_b}) = {m} km."],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif variant == "velocity_with_red_herring":
        # Medium: pick a triple, add irrelevant time scalar t in 2..9
        leg_a, leg_b, m = rng.choice(TRIPLES)
        _verify_triple(leg_a, leg_b, m)

        # Sample irrelevant time t in 2..9 (never 1)
        t = rng.randint(2, 9)

        text = f"Over {t} seconds, a {rng.choice(['drone', 'glider'])}'s velocity has an eastward component of {leg_a} m/s and a northward component of {leg_b} m/s. Find the magnitude of the velocity (the speed), in m/s."
        answer = m
        answer_text = str(answer)

        # Independent check: verify Pythagorean identity (ignoring the time)
        _verify_magnitude_from_legs(leg_a, leg_b, m)

        # Misconceptions:
        # 1. added-instead-of-pythagoras: leg_a + leg_b
        wrong1 = leg_a + leg_b
        # 2. forgot-one-leg: report larger leg alone
        wrong2 = max(leg_a, leg_b)
        # 3. multiplies-by-the-irrelevant-number: m * t
        wrong3 = m * t

        candidates = [str(wrong1), str(wrong2), str(wrong3)]

        return dict(
            text=text,
            latex=text,
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"Speed: sqrt({leg_a}^2 + {leg_b}^2) = sqrt({leg_a * leg_a + leg_b * leg_b}) = {m} m/s. The {t} seconds is not used."
            ],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif variant == "two_leg_journey":
        # Hard: pick final triple (P, Q, m), split P and Q into two legs each
        # Then ask for magnitude of the summed resultant
        P, Q, m = rng.choice(TRIPLES)
        _verify_triple(P, Q, m)

        # Split P into leg_a1 + leg_a2 = P
        leg_a1 = rng.randint(1, P - 1)
        leg_a2 = P - leg_a1

        # Split Q into leg_b1 + leg_b2 = Q
        leg_b1 = rng.randint(1, Q - 1)
        leg_b2 = Q - leg_b1

        text = f"A hiker walks {leg_a1} km east then {leg_b1} km north, then walks a further {leg_a2} km east and {leg_b2} km north. Find the magnitude of the hiker's total displacement, in km."
        answer = m
        answer_text = str(answer)

        # Independent check: verify Pythagorean identity using the SUMMED legs
        _verify_magnitude_from_legs(P, Q, m)

        # Misconceptions:
        # 1. added-instead-of-pythagoras: (leg_a1 + leg_a2) + (leg_b1 + leg_b2) = P + Q
        wrong1 = P + Q
        # 2. dropped-the-second-leg: sqrt(leg_a1^2 + leg_b1^2), formatted to 1 decimal place
        partial_magnitude_squared = leg_a1 * leg_a1 + leg_b1 * leg_b1
        partial_magnitude = partial_magnitude_squared**0.5
        wrong2 = f"{partial_magnitude:.1f}"
        # 3. forgot-one-leg: report P alone (full x-direction total)
        wrong3 = P
        # 4. forgot-one-leg variant: report Q alone (full y-direction total)
        wrong4 = Q

        # We have 4 misconceptions, but pick_distractors will dedupe and keep 3
        candidates = [str(wrong1), wrong2, str(wrong3), str(wrong4)]

        return dict(
            text=text,
            latex=text,
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"Total displacement: east total = {leg_a1} + {leg_a2} = {P} km, north total = {leg_b1} + {leg_b2} = {Q} km. Magnitude = sqrt({P}^2 + {Q}^2) = sqrt({P * P + Q * Q}) = {m} km."
            ],
            distractors=pick_distractors(answer_text, candidates),
        )

    raise ValueError(f"unknown variant: {variant!r}")
