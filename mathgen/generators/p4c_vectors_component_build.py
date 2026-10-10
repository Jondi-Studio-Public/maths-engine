"""Vectors component form and scalar multiple generators.

Senior Specialist Mathematics Unit 1, domain `vectors`, framework `acara-senior-v8-4`.

- express_a_vector_in_component_form_using_the (ACMSM016): given a vector described
  as horizontal/vertical movements, express it in component form using unit vectors i/j
  and report one named coefficient (never the full expression string, due to a real
  i/j-notation grading defect in mathgen/grading/typed.py).

- define_and_use_multiplication_by_a_scalar_of (ACMSM018): given a vector in
  component form and an integer scalar, compute one named coefficient of the scalar
  multiple (forward), or recover the original coefficient from the scaled value (inverse).
"""

from __future__ import annotations

from ..core import check, generator, join_terms, pick_distractors, paren_negative


# --- express_a_vector_in_component_form_using_the (ACMSM016) ---------------


def _plural(n: int) -> str:
    """Return 'unit' or 'units' based on magnitude."""
    return "unit" if abs(n) == 1 else "units"


def _direction_to_sign_x(direction: str) -> int:
    """Map horizontal direction word to sign: right -> +1, left -> -1."""
    return 1 if direction == "right" else -1


def _direction_to_sign_y(direction: str) -> int:
    """Map vertical direction word to sign: up -> +1, down -> -1."""
    return 1 if direction == "up" else -1


def _verify_single_leg_component(direction: str, magnitude: int, answer: int) -> None:
    """Verify single-leg coefficient computation independently.

    Two separate checks:
    1. Sign correctness: sign(answer) matches direction (right/up -> positive, left/down -> negative)
    2. Magnitude correctness: |answer| equals magnitude

    Neither check re-runs the sign-application code construction used.
    """
    sign_map = {"right": 1, "left": -1, "up": 1, "down": -1}
    expected_sign = sign_map[direction]
    check((answer > 0) == (expected_sign > 0), f"answer sign does not match direction {direction}")
    check(abs(answer) == magnitude, f"answer magnitude {abs(answer)} does not match stated magnitude {magnitude}")


def _verify_two_leg_component(directions: list[str], magnitudes: list[int], asked_axis: str, answer: int) -> None:
    """Verify two-leg coefficient by running independent total over both legs.

    Not calling the same construction helper; uses a fresh loop.
    """
    axis_index = 0 if asked_axis == "i" else 1  # i is x-axis, j is y-axis
    direction_keys = ["right", "left"] if asked_axis == "i" else ["up", "down"]

    total = 0
    for direction, magnitude in zip(directions, magnitudes):
        sign_map = {direction_keys[0]: 1, direction_keys[1]: -1}
        sign = sign_map[direction]
        total += sign * magnitude

    check(total == answer, f"two-leg total {total} does not equal reported answer {answer}")


def _verify_scalar_forward(k: int, component: int, answer: int) -> None:
    """Verify scalar multiplication forward tier via repeated addition.

    Does not re-run the multiplication; instead uses repeated addition.
    """
    if k > 0:
        total = sum([component] * k)
    else:
        total = sum([-component] * (-k))
    check(total == answer, f"repeated addition {total} does not equal multiplied answer {answer}")


def _verify_scalar_inverse(k: int, product: int, answer: int) -> None:
    """Verify scalar division via multiplication back.

    Does not re-run the division; instead multiplies back.
    """
    check(k * answer == product, f"k * answer ({k} * {answer} = {k * answer}) does not equal product {product}")


@generator(
    "express_a_vector_in_component_form_using_the",
    level=11,
    presets={
        "easy": dict(variant="single_leg_positive"),
        "medium": dict(variant="single_leg_signed"),
        "hard": dict(variant="two_leg"),
    },
)
def express_a_vector_in_component_form_using_the(rng, variant="single_leg_positive"):
    """Express a vector in component form using unit vectors i and j.

    Ask for one named coefficient (of i or j), never the full expression string.
    """

    if variant == "single_leg_positive":
        # Easy: positive directions (right, up) only, no sign handling needed
        p = rng.randint(1, 9)
        q = rng.randint(1, 9)
        while p == q:
            q = rng.randint(1, 9)

        x_direction = "right"
        y_direction = "up"

        # Answer is the coefficient asked for
        x_coeff = p
        y_coeff = q

        asked_axis = rng.choice(["i", "j"])
        answer = x_coeff if asked_axis == "i" else y_coeff

        # Verification: independent check
        direction = x_direction if asked_axis == "i" else y_direction
        magnitude = p if asked_axis == "i" else q
        _verify_single_leg_component(direction, magnitude, answer)

        question_text = (
            f"A vector represents a movement of {p} {_plural(p)} to the {x_direction} "
            f"and {q} {_plural(q)} {y_direction}. Write this vector in component form using "
            f"the unit vectors i and j (v = xi + yj). What is the coefficient of {asked_axis}?"
        )

        # Distractors (always distinct at easy since p != q):
        # 1. wrong-axis-component: the OTHER axis
        wrong_axis = y_coeff if asked_axis == "i" else x_coeff
        # 2. adds-both-components-together: x + y
        adds_both = x_coeff + y_coeff
        # (sign-ignored is never offered at easy tier)

        candidates = [wrong_axis, adds_both]

    elif variant == "single_leg_signed":
        # Medium: random directions, sign handling required
        # Use reconstruction loop to ensure valid distractors
        max_attempts = 100
        for attempt in range(max_attempts):
            p = rng.randint(1, 9)
            q = rng.randint(1, 9)
            while p == q:
                q = rng.randint(1, 9)

            x_direction = rng.choice(["right", "left"])
            y_direction = rng.choice(["up", "down"])

            # Compute signed coefficients
            x_coeff = p * _direction_to_sign_x(x_direction)
            y_coeff = q * _direction_to_sign_y(y_direction)

            asked_axis = rng.choice(["i", "j"])
            answer = x_coeff if asked_axis == "i" else y_coeff

            # Build candidates
            wrong_axis = y_coeff if asked_axis == "i" else x_coeff
            magnitude = p if asked_axis == "i" else q
            adds_both = x_coeff + y_coeff

            # Check for valid distractors
            candidates_list = [wrong_axis, magnitude, adds_both]
            survivors = [c for c in candidates_list if c != answer]
            if len(survivors) >= 2 and len(set(survivors)) == len(survivors):
                # All survivors distinct and at least 2
                break
        else:
            raise RuntimeError("Failed to generate valid medium-tier single-leg after 100 attempts")

        # Verification: independent check
        direction = x_direction if asked_axis == "i" else y_direction
        magnitude_val = p if asked_axis == "i" else q
        _verify_single_leg_component(direction, magnitude_val, answer)

        question_text = (
            f"A vector represents a movement of {p} {_plural(p)} to the {x_direction} "
            f"and {q} {_plural(q)} {y_direction}. Write this vector in component form using "
            f"the unit vectors i and j (v = xi + yj). What is the coefficient of {asked_axis}?"
        )

        candidates = [wrong_axis, magnitude, adds_both]

    else:  # variant == "two_leg"
        # Hard: two movements combined, net must be != 0 and nets must differ
        # Use reconstruction loop to ensure valid distractors
        max_attempts = 100
        for attempt in range(max_attempts):
            p1 = rng.randint(1, 9)
            q1 = rng.randint(1, 9)

            p2 = rng.randint(1, 9)
            q2 = rng.randint(1, 9)

            x_direction1 = rng.choice(["right", "left"])
            y_direction1 = rng.choice(["up", "down"])
            x_direction2 = rng.choice(["right", "left"])
            y_direction2 = rng.choice(["up", "down"])

            # Compute signed components for each leg
            x1 = p1 * _direction_to_sign_x(x_direction1)
            y1 = q1 * _direction_to_sign_y(y_direction1)
            x2 = p2 * _direction_to_sign_x(x_direction2)
            y2 = q2 * _direction_to_sign_y(y_direction2)

            # Net coefficients
            net_x = x1 + x2
            net_y = y1 + y2

            # Force net != 0 on the asked axis
            asked_axis = rng.choice(["i", "j"])
            if asked_axis == "i":
                # Force net_x != 0 by resampling p2 if needed
                attempts_inner = 0
                while net_x == 0 and attempts_inner < 10:
                    p2 = rng.randint(1, 9)
                    x2 = p2 * _direction_to_sign_x(x_direction2)
                    net_x = x1 + x2
                    attempts_inner += 1
                if net_x == 0:
                    continue  # Restart outer loop
            else:
                # Force net_y != 0 by resampling q2 if needed
                attempts_inner = 0
                while net_y == 0 and attempts_inner < 10:
                    q2 = rng.randint(1, 9)
                    y2 = q2 * _direction_to_sign_y(y_direction2)
                    net_y = y1 + y2
                    attempts_inner += 1
                if net_y == 0:
                    continue

            # Force net_x != net_y by resampling
            attempts_inner = 0
            while net_x == net_y and attempts_inner < 10:
                if asked_axis == "i":
                    # Resample q2 to force nets to differ
                    q2 = rng.randint(1, 9)
                    y2 = q2 * _direction_to_sign_y(y_direction2)
                    net_y = y1 + y2
                else:
                    # Resample p2 to force nets to differ
                    p2 = rng.randint(1, 9)
                    x2 = p2 * _direction_to_sign_x(x_direction2)
                    net_x = x1 + x2
                attempts_inner += 1
            if net_x == net_y:
                continue

            answer = net_x if asked_axis == "i" else net_y

            # Build candidate distractors
            first_leg = x1 if asked_axis == "i" else y1
            other_net = net_y if asked_axis == "i" else net_x
            mag_sum = p1 + p2 if asked_axis == "i" else q1 + q2

            # Check for valid distractors (pairwise distinct and not equal to answer)
            candidates_list = [first_leg, other_net, mag_sum]
            survivors = [c for c in candidates_list if c != answer]
            if len(survivors) >= 2 and len(set(survivors)) == len(survivors):
                # All survivors distinct and at least 2
                break
        else:
            raise RuntimeError("Failed to generate valid hard-tier two-leg after 100 attempts")

        # Verification: independent check using fresh loop
        directions = [x_direction1, x_direction2] if asked_axis == "i" else [y_direction1, y_direction2]
        magnitudes = [p1, p2] if asked_axis == "i" else [q1, q2]
        _verify_two_leg_component(directions, magnitudes, asked_axis, answer)

        # Build question text
        p1_unit = _plural(p1)
        q1_unit = _plural(q1)
        p2_unit = _plural(p2)
        q2_unit = _plural(q2)

        question_text = (
            f"A vector is built from two movements: first {p1} {p1_unit} {x_direction1} "
            f"and {q1} {q1_unit} {y_direction1}, then {p2} {p2_unit} {x_direction2} "
            f"and {q2} {q2_unit} {y_direction2}. Combine these into a single resultant "
            f"vector, then write it in component form using the unit vectors i and j. "
            f"What is the coefficient of {asked_axis}?"
        )

        candidates = [first_leg, other_net, mag_sum]

    # Format answer as string for answer_text
    answer_text = str(int(answer))

    # Format the vector representation for working
    if variant == "two_leg":
        if asked_axis == "i":
            x_expr = join_terms([(x1, ""), (x2, "")])
            working_line = f"Net i-coefficient = {x1} + {paren_negative(x2)} = {x_expr}."
        else:
            y_expr = join_terms([(y1, ""), (y2, "")])
            working_line = f"Net j-coefficient = {y1} + {paren_negative(y2)} = {y_expr}."
    else:
        working_line = f"Coefficient of {asked_axis} = {answer_text}."

    return dict(
        text=question_text,
        latex=question_text,  # TODO: could add vector notation if needed
        answer=int(answer),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, candidates),
        working=[working_line],
    )


# --- define_and_use_multiplication_by_a_scalar_of (ACMSM018) ----------------


def _format_vector_component(a: int, b: int) -> str:
    """Format vector v = ai + bj using join_terms to handle signs."""
    return f"v = {join_terms([(a, 'i'), (b, 'j')])}"


@generator(
    "define_and_use_multiplication_by_a_scalar_of",
    level=11,
    presets={
        "easy": dict(variant="forward_positive"),
        "medium": dict(variant="forward_signed"),
        "hard": dict(variant="inverse"),
    },
)
def define_and_use_multiplication_by_a_scalar_of(rng, variant="forward_positive"):
    """Multiply a vector by a scalar or recover the original coefficient from a scaled value."""

    if variant == "forward_positive":
        # Easy: positive scalar k, positive coefficients a, b
        # Use a reconstruction loop to ensure distractor pairwise distinctness
        max_attempts = 100
        for attempt in range(max_attempts):
            k = rng.choice([2, 3, 4, 5])

            a = rng.randint(1, 9)
            b = rng.randint(1, 9)
            while a == b:
                b = rng.randint(1, 9)

            # Special case: if k==2, resample a away from 2 to prevent collision
            # with added-instead-of-multiplied (component + k == component * k when component == 2)
            if k == 2:
                while a == 2:
                    a = rng.randint(1, 9)
                while a == b:
                    b = rng.randint(1, 9)
                # Also resample b away from 2 to prevent collision
                while b == 2:
                    b = rng.randint(1, 9)
                while a == b:
                    b = rng.randint(1, 9)

            asked_axis = rng.choice(["i", "j"])
            component = a if asked_axis == "i" else b
            answer = k * component

            # Build candidates
            other_component = b if asked_axis == "i" else a
            added = component + k
            original = component
            wrong_axis = k * other_component

            # Check for collisions between candidates and answer
            candidates_set = {added, original, wrong_axis, answer}

            # We need at least 2 distractors that survive (are distinct from answer and each other)
            # Check if we have at least 2 survivors
            survivors = [c for c in [added, original, wrong_axis] if c != answer]
            if len(survivors) >= 2 and len(set(survivors)) == len(survivors):
                # All three distinct and at least 2 don't collide with answer
                break
        else:
            raise RuntimeError("Failed to generate valid easy-tier scalar-multiple after 100 attempts")

        # Verification: repeated addition
        _verify_scalar_forward(k, component, answer)

        question_text = f"v = {a}i + {b}j. Find the coefficient of {asked_axis} in {k}v."

        candidates = [added, original, wrong_axis]

    elif variant == "forward_signed":
        # Medium: signed scalar, signed coefficients
        # Use a reconstruction loop to ensure distractor pairwise distinctness
        max_attempts = 100
        for attempt in range(max_attempts):
            k = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])

            a = rng.randint(-9, 9)
            while a == 0:
                a = rng.randint(-9, 9)

            b = rng.randint(-9, 9)
            while b == 0 or b == a:
                b = rng.randint(-9, 9)

            # Special case: if k==2, resample both a and b away from 2
            if k == 2:
                while a == 2:
                    a = rng.randint(-9, 9)
                    while a == 0:
                        a = rng.randint(-9, 9)
                while b == 2 or b == a:
                    b = rng.randint(-9, 9)
                    while b == 0:
                        b = rng.randint(-9, 9)

            asked_axis = rng.choice(["i", "j"])
            component = a if asked_axis == "i" else b
            answer = k * component

            # Build candidates
            other_component = b if asked_axis == "i" else a
            added = component + k
            original = component
            wrong_axis = k * other_component

            # Check for valid distractors
            survivors = [c for c in [added, original, wrong_axis] if c != answer]
            if len(survivors) >= 2 and len(set(survivors)) == len(survivors):
                # All survivors distinct and at least 2
                break
        else:
            raise RuntimeError("Failed to generate valid medium-tier scalar-multiple after 100 attempts")

        # Verification: repeated addition
        _verify_scalar_forward(k, component, answer)

        # Format vector with proper sign formatting
        v_text = _format_vector_component(a, b)
        question_text = f"{v_text}. Find the coefficient of {asked_axis} in {k}v."

        candidates = [added, original, wrong_axis]

    else:  # variant == "inverse"
        # Hard: inverse - given scaled coefficient, recover original
        # Use a reconstruction loop to ensure distractors are distinct from answer
        max_attempts = 100
        for attempt in range(max_attempts):
            k = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])

            # Original coefficient, sampled to be nonzero
            original = rng.randint(-9, 9)
            while original == 0:
                original = rng.randint(-9, 9)

            # Scaled product
            product = k * original

            # Answer is the original (recovered by division)
            answer = product // k  # Guaranteed exact division

            # Build candidates
            subtracted = product - k
            direct = product

            # Check if added-instead-of-divided collides with answer
            # If it does, resample the original
            if subtracted == answer or direct == answer:
                continue
            if subtracted == direct:
                continue

            # Valid candidates found
            break
        else:
            raise RuntimeError("Failed to generate valid hard-tier scalar-multiple after 100 attempts")

        # Verification: multiply back
        _verify_scalar_inverse(k, product, answer)

        asked_axis = rng.choice(["i", "j"])

        question_text = (
            f"{k}v has a coefficient of {asked_axis} equal to {product}. "
            f"What is the original vector's coefficient of {asked_axis}?"
        )

        candidates = [subtracted, direct]

    # Format answer as string for answer_text
    answer_text = str(int(answer))

    return dict(
        text=question_text,
        latex=question_text,
        answer=int(answer),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, candidates),
        working=[],
    )
