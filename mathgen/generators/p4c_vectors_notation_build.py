"""Vectors notation cluster (ACMSM014, ACMSM012).

Two generators, both senior ``acara-senior-v8-4``, domain ``vectors``, ``answer: number``:

- ``use_ordered_pair_notation_and_column_vector_notation``
  (``senior-use-ordered-pair-notation-and-column-vector-notation``, ACMSM014)
  -- given two labelled points A and B, compute one named component of the vector from A to B,
  or (inverse tier) given one endpoint and one component, find the other endpoint's coordinate.

- ``represent_a_scalar_multiple_of_a_vector``
  (``senior-represent-a-scalar-multiple-of-a-vector``, ACMSM012)
  -- given a vector and integer scalar k, compute one named component of k*v,
  or (inverse tier) given the scaled component and k, recover the original component.

Both generators use exact integer arithmetic; no decimals anywhere in answers or distractors.
All candidates and answers are distinct over 500-seed sweeps with resample-on-collision.
"""

from __future__ import annotations

from ..core import check, generator, pick_distractors, paren_negative


# --- use_ordered_pair_notation_and_column_vector_notation -------------------


def _verify_forward_simple(A_axis: int, B_axis: int, expected: int, /) -> None:
    """Independent check for forward tiers: verify by addition.

    Construction computes answer = B_axis - A_axis.
    Check verifies: A_axis + answer == B_axis (opposite direction).
    """
    recomputed = A_axis + expected
    check(recomputed == B_axis, f"addition check failed: {A_axis} + {expected} != {B_axis}")


def _verify_inverse_simple(A_axis: int, component: int, expected: int, /) -> None:
    """Independent check for inverse tier: verify by subtraction.

    Construction computes answer = A_axis + component.
    Check verifies: answer - A_axis == component (opposite direction).
    """
    recomputed = expected - A_axis
    check(recomputed == component, f"subtraction check failed: {expected} - {A_axis} != {component}")


@generator(
    "use_ordered_pair_notation_and_column_vector_notation",
    level=11,
    presets={
        "easy": dict(variant="forward_positive"),
        "medium": dict(variant="forward_signed"),
        "hard": dict(variant="inverse"),
    },
)
def use_ordered_pair_notation_and_column_vector_notation(
    rng,
    variant="forward_positive",
):
    """Use ordered pair and column vector notation to represent a vector.

    Forward tiers: given A and B, compute one component of the vector from A to B.
    Inverse tier: given A and one component of the vector, find the corresponding
    coordinate of B (run the definition backwards).
    """

    if variant == "forward_positive":
        # Easy: Bx, By > Ax, Ay; all in 1..9
        # Constraint: both components nonzero (guaranteed by range > 0)
        # Constraint: (Bx - Ax) != (By - Ay)
        while True:
            Ax = rng.randint(1, 8)  # 1..8 to ensure room for Bx > Ax in range 1..9
            Ay = rng.randint(1, 8)  # 1..8 to ensure room for By > Ay in range 1..9
            Bx = rng.randint(Ax + 1, 9)  # Force Bx > Ax
            By = rng.randint(Ay + 1, 9)  # Force By > Ay

            diff_x = Bx - Ax
            diff_y = By - Ay

            # Resample if differences are equal
            if diff_x == diff_y:
                continue

            # Randomly choose which component to ask for
            axis = rng.choice(["x", "y"])

            if axis == "x":
                answer = diff_x
            else:  # axis == "y"
                answer = diff_y

            # Build candidates for misconceptions
            wrong_reversed = -diff_x if axis == "x" else -diff_y
            wrong_other = diff_y if axis == "x" else diff_x
            wrong_endpoint = Bx if axis == "x" else By

            candidates = [str(wrong_reversed), str(wrong_other), str(wrong_endpoint)]

            # Check all candidates and answer are distinct
            all_values = [str(answer)] + candidates
            if len(set(all_values)) == len(all_values):
                break

        answer_text = str(answer)
        text = (
            f"Point A is ({Ax}, {Ay}). Point B is ({Bx}, {By}). "
            f"The vector from A to B can be written in ordered pair notation, "
            f"or equivalently as the column vector with top entry equal to its "
            f"x-component and bottom entry equal to its y-component. "
            f"What is the {axis}-component of the vector from A to B?"
        )

        _verify_forward_simple(Ax if axis == "x" else Ay, Bx if axis == "x" else By, answer)

        return dict(
            text=text,
            latex=rf"A = ({Ax}, {Ay}), B = ({Bx}, {By})",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"Vector from A to B: ({diff_x}, {diff_y}). The {axis}-component is {answer}."],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif variant == "forward_signed":
        # Medium: Ax, Ay, Bx, By in -9..9; no B > A constraint
        # Constraint: both A_axis and B_axis nonzero (for endpoint-only distinctness)
        # Constraint: asked component nonzero
        # Constraint: (Bx - Ax) != (By - Ay)
        while True:
            Ax = rng.randint(-9, 9)
            if Ax == 0:
                Ax = rng.choice([-1, 1])  # Resample if 0
            Ay = rng.randint(-9, 9)
            if Ay == 0:
                Ay = rng.choice([-1, 1])

            Bx = rng.randint(-9, 9)
            By = rng.randint(-9, 9)

            diff_x = Bx - Ax
            diff_y = By - Ay

            # Constraint: differences must be distinct
            if diff_x == diff_y:
                continue

            # Randomly choose which component to ask for
            axis = rng.choice(["x", "y"])

            if axis == "x":
                answer = diff_x
                # For endpoint-only distinctness, answer != B_axis, so need Bx != 0
                # This is enforced by sampling Bx after Ax; if Ax != 0 and we want answer != Bx,
                # we need Bx - Ax != Bx, so Ax != 0 (enforced above).
                # However, endpoint-only fails only if B_axis == B_axis - A_axis,
                # which means A_axis == 0. We already force Ax != 0.
                if answer == 0:
                    continue  # Resample if asked component is 0
            else:  # axis == "y"
                answer = diff_y
                if answer == 0:
                    continue

            # Build candidates for misconceptions
            wrong_reversed = -answer
            wrong_other = diff_y if axis == "x" else diff_x
            wrong_endpoint = Bx if axis == "x" else By

            candidates = [str(wrong_reversed), str(wrong_other), str(wrong_endpoint)]

            # Check all candidates and answer are distinct
            all_values = [str(answer)] + candidates
            if len(set(all_values)) == len(all_values):
                break

        answer_text = str(answer)
        text = (
            f"Point A is ({Ax}, {Ay}). Point B is ({Bx}, {By}). "
            f"The vector from A to B can be written in ordered pair notation, "
            f"or equivalently as the column vector with top entry equal to its "
            f"x-component and bottom entry equal to its y-component. "
            f"What is the {axis}-component of the vector from A to B?"
        )

        _verify_forward_simple(Ax if axis == "x" else Ay, Bx if axis == "x" else By, answer)

        return dict(
            text=text,
            latex=rf"A = ({Ax}, {Ay}), B = ({Bx}, {By})",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"Vector from A to B: ({diff_x}, {diff_y}). The {axis}-component is {answer}."],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif variant == "inverse":
        # Hard: inverse direction. Given A and component v, find B's coordinate.
        # Ax, Ay in -9..9, forced Ax != Ay
        # v (the component) in -9..9, nonzero
        # B_axis = A_axis + v
        # Constraint: A_axis != A_other (forced)
        # Constraint: v != 0 (forced)
        # Constraint: A_axis != 0 (for endpoint-only distinctness)
        while True:
            Ax = rng.randint(-9, 9)
            if Ax == 0:
                Ax = rng.choice([-1, 1])
            Ay = rng.randint(-9, 9)
            if Ay == 0:
                Ay = rng.choice([-1, 1])

            # Force Ax != Ay
            if Ax == Ay:
                Ay = rng.randint(-9, 9)
                if Ay == 0 or Ay == Ax:
                    Ay = (Ax + 1) % 19 - 9  # Ensure different and nonzero
                if Ay == 0:
                    Ay = (Ax + 2) % 19 - 9

            # Randomly choose which component to ask for
            axis = rng.choice(["x", "y"])

            # Sample v nonzero
            v = rng.randint(-9, 9)
            if v == 0:
                v = rng.choice([-1, 1])

            if axis == "x":
                answer = Ax + v
                A_axis = Ax
                A_other = Ay
            else:  # axis == "y"
                answer = Ay + v
                A_axis = Ay
                A_other = Ax

            # Build candidates for misconceptions
            wrong_reversed = A_axis - v
            wrong_axis = A_other + v
            wrong_endpoint = v

            candidates = [str(wrong_reversed), str(wrong_axis), str(wrong_endpoint)]

            # Check all candidates and answer are distinct
            all_values = [str(answer)] + candidates
            if len(set(all_values)) == len(all_values):
                break

        answer_text = str(answer)

        # Compose the question text for inverse tier
        # Format: "kv = ({component asked}, ?) for k = {k}. If the {x/y}-component of kv is {P}, find..."
        # But this is actually: "Given point A and the component v, find B's coordinate"
        # The scalar-multiple inverse uses a similar format; this is the vector inverse.
        # Let me use a consistent format.
        if axis == "x":
            text = (
                f"Point A is ({Ax}, {Ay}). The x-component of the vector from A to B is {v}. "
                f"What is the x-coordinate of B?"
            )
        else:
            text = (
                f"Point A is ({Ax}, {Ay}). The y-component of the vector from A to B is {v}. "
                f"What is the y-coordinate of B?"
            )

        _verify_inverse_simple(A_axis, v, answer)

        return dict(
            text=text,
            latex=rf"A = ({Ax}, {Ay})",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"Vector from A to B has {axis}-component {v}. "
                f"So the {axis}-coordinate of B is {A_axis} + {paren_negative(v)} = {answer}."
            ],
            distractors=pick_distractors(answer_text, candidates),
        )

    else:
        raise ValueError(f"Unknown variant: {variant}")


# --- represent_a_scalar_multiple_of_a_vector --------------------------------


def _verify_forward_multiplication(component: int, k: int, expected: int, /) -> None:
    """Independent check for forward tiers: verify by repeated addition.

    Construction computes product = k * component.
    Check verifies by repeated addition: sum([component] * |k|) = product (with sign).
    """
    if k > 0:
        recomputed = sum([component] * k)
    else:
        recomputed = sum([-component] * (-k))
    check(recomputed == expected, f"repeated-addition check failed: sum([{component}] * {k}) != {expected}")


def _verify_inverse_division(P: int, k: int, expected: int, /) -> None:
    """Independent check for inverse tier: verify by multiplication.

    Construction computes a = P // k.
    Check verifies: k * a == P (opposite operation from division).
    """
    recomputed = k * expected
    check(recomputed == P, f"multiplication check failed: {k} * {expected} != {P}")


@generator(
    "represent_a_scalar_multiple_of_a_vector",
    level=11,
    presets={
        "easy": dict(variant="forward_positive"),
        "medium": dict(variant="forward_signed"),
        "hard": dict(variant="inverse"),
    },
)
def represent_a_scalar_multiple_of_a_vector(
    rng,
    variant="forward_positive",
):
    """Represent a scalar multiple of a vector.

    Forward tiers: given vector v = (a, b) and scalar k, compute one component of k*v.
    Inverse tier: given the scaled component P = k*a and scalar k, recover original a.
    """

    if variant == "forward_positive":
        # Easy: k in {2,3,4,5}, a,b in 1..9, forced a != b
        # Additional constraint: if k==2, force a != 2 and b != 2
        # (to prevent a + k == a * k collision for both components)
        while True:
            k = rng.choice([2, 3, 4, 5])
            a = rng.randint(1, 9)
            b = rng.randint(1, 9)

            # Force a != b
            if a == b:
                continue

            # If k == 2, force both a != 2 and b != 2
            if k == 2 and (a == 2 or b == 2):
                continue

            # Randomly choose which component to ask for
            axis = rng.choice(["x", "y"])

            if axis == "x":
                component = a
                answer = k * component
                other_component = b
            else:  # axis == "y"
                component = b
                answer = k * component
                other_component = a

            # Build candidates for misconceptions
            wrong_added = component + k
            wrong_original = component
            wrong_axis = k * other_component

            candidates = [str(wrong_added), str(wrong_original), str(wrong_axis)]

            # Check all candidates and answer are distinct
            all_values = [str(answer)] + candidates
            if len(set(all_values)) == len(all_values):
                break

        answer_text = str(answer)
        text = f"v = ({a}, {b}). Find the {axis}-component of {k}v."

        _verify_forward_multiplication(component, k, answer)

        return dict(
            text=text,
            latex=rf"\mathbf{{v}} = \begin{{pmatrix}} {a} \\ {b} \end{{pmatrix}}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"{k}v has components ({k * a}, {k * b}). The {axis}-component is {answer}."],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif variant == "forward_signed":
        # Medium: k in {-5,-4,-3,-2,2,3,4,5}, a,b in -9..9 excluding 0
        # Constraint: a != b
        # Constraint: if k==2, also exclude 2 from a and b (same collision prevention)
        while True:
            k = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
            a = rng.randint(-9, 9)
            if a == 0:
                a = rng.choice([-1, 1])
            b = rng.randint(-9, 9)
            if b == 0:
                b = rng.choice([-1, 1])

            # Force a != b
            if a == b:
                continue

            # If k == 2, force both a != 2 and b != 2
            if k == 2 and (a == 2 or b == 2):
                continue

            # Randomly choose which component to ask for
            axis = rng.choice(["x", "y"])

            if axis == "x":
                component = a
                answer = k * component
                other_component = b
            else:  # axis == "y"
                component = b
                answer = k * component
                other_component = a

            # Build candidates for misconceptions
            wrong_added = component + k
            wrong_original = component
            wrong_axis = k * other_component

            candidates = [str(wrong_added), str(wrong_original), str(wrong_axis)]

            # Check all candidates and answer are distinct
            all_values = [str(answer)] + candidates
            if len(set(all_values)) == len(all_values):
                break

        answer_text = str(answer)
        text = f"v = ({a}, {b}). Find the {axis}-component of {k}v."

        _verify_forward_multiplication(component, k, answer)

        return dict(
            text=text,
            latex=rf"\mathbf{{v}} = \begin{{pmatrix}} {a} \\ {b} \end{{pmatrix}}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"{k}v has components ({k * a}, {k * b}). The {axis}-component is {answer}."],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif variant == "inverse":
        # Hard: inverse direction. Given P = k*a and k, find a.
        # k in {-5,-4,-3,-2,2,3,4,5}
        # a in -9..9 excluding 0
        # P = k * a (guaranteed to divide evenly)
        # answer = a = P // k (integer division, exact)
        # Constraint: if k==2, resample if a==2 (collision with added-instead-of-divided)
        while True:
            k = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
            a = rng.randint(-9, 9)
            if a == 0:
                a = rng.choice([-1, 1])

            P = k * a
            answer = a

            # If k == 2 and a == 2, then P - k == a (4 - 2 == 2), so resample
            if k == 2 and a == 2:
                continue

            # Randomly choose which component to ask for (but we're really asking for the original)
            axis = rng.choice(["x", "y"])

            # Build candidates for misconceptions
            wrong_divided = P - k  # added-instead-of-divided (subtracts k)
            wrong_product = P  # used-product-directly

            candidates = [str(wrong_divided), str(wrong_product)]

            # Check all candidates and answer are distinct
            all_values = [str(answer)] + candidates
            if len(set(all_values)) == len(all_values):
                break

        answer_text = str(answer)

        # Compose question text for inverse tier
        # The format is: "kv = ({component asked}, ?) for k = {k}. If the {x/y}-component of kv is {P}, find..."
        if axis == "x":
            text = (
                f"kv = ({P}, ?) for k = {k}. If the x-component of kv is {P}, find the original vector's x-component."
            )
        else:
            text = (
                f"kv = (?, {P}) for k = {k}. If the y-component of kv is {P}, find the original vector's y-component."
            )

        _verify_inverse_division(P, k, answer)

        return dict(
            text=text,
            latex=rf"k = {k}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"The {axis}-component of kv is {P}. The original {axis}-component is {P} / {k} = {answer}."],
            distractors=pick_distractors(answer_text, candidates),
        )

    else:
        raise ValueError(f"Unknown variant: {variant}")
