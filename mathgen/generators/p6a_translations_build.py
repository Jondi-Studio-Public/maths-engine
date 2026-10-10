"""Translations as column vectors (ACMSM054).

Skill: Translations and their representation as column vectors
- BOUNDARY: This skill goes FORWARD (given point and column vector, find image).
  The existing skill ``f10-coordinate-movement`` (AC9M6SP02, prerequisites
  ``coordinate-plane-readiness``) goes REVERSE (given before and after
  coordinates, find the change/column vector). This skill's prerequisite
  ``vector-addition`` reflects the forward direction and vector-sum reasoning
  for the hard tier (two successive translations).
- Easy: Point P(Px, Py) with positive coords; translate by column vector
  (a, b) with positive components; find image P' = (Px + a, Py + b).
- Medium: Mixed-sign coords and vector components; same addition formula.
- Hard: Two successive translations (requires vector addition to combine them):
  apply (a1, b1) then (a2, b2) to point P; final image P' = (Px + a1 + a2, Py + b1 + b2).
  Image only depends on the vector sum; translations commute; this generator
  does not generalise to non-commuting transformations.
"""

from __future__ import annotations

from ..core import check, check_distinct, check_equivalent, generator, pick_distractors


def _signed_paren(n: int) -> str:
    """Wrap a negative number in parentheses for use right after a literal
    "-" in a worked-steps subtraction/addition expression, e.g. "0 + (-6)"
    instead of "0 + -6" (a double-plus the shared text-quality test
    rejects on every topic). Non-negative values are returned unchanged."""
    return f"({n})" if n < 0 else str(n)


def _pair_collision_check(answer_pair, candidate_pairs):
    """Check if any candidate pair collides with answer (both components exactly match).

    Returns True if all candidates are safe (no collisions), False if collision detected.
    Integer-valued pairs differing by any nonzero amount clear grade_pair's tolerance,
    so exact integer equality per component is the collision threshold.
    """
    ax, ay = answer_pair
    for cx, cy in candidate_pairs:
        if cx == ax and cy == ay:
            return False  # Collision: both components exactly match
    return True  # All candidates safe


@generator(
    "translations_and_their_representation_as_column_vectors",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def translations_and_their_representation_as_column_vectors(rng, tier="easy"):
    """Given a point and a column vector translation, find the image point."""

    if tier == "easy":
        for attempt in range(200):
            # Point P with positive coordinates
            px, py = rng.randint(1, 9), rng.randint(1, 9)
            # Column vector (a, b) with positive components
            a, b = rng.randint(1, 9), rng.randint(1, 9)

            # Image point P'
            px_prime = px + a
            py_prime = py + b

            # Build distractors
            # subtracts-instead-of-adds: (Px - a, Py - b)
            sub_x = px - a
            sub_y = py - b
            # swapped-vector-components: (Px + b, Py + a)
            swap_x = px + b
            swap_y = py + a
            # one-axis-only: (Px + a, Py)
            one_axis_x = px + a
            one_axis_y = py

            # Check for pair collisions with answer
            candidates = [
                (sub_x, sub_y),
                (swap_x, swap_y),
                (one_axis_x, one_axis_y),
            ]
            if not _pair_collision_check((px_prime, py_prime), candidates):
                continue

            # Verify construction via independent route: check displacement
            displacement_x = a
            displacement_y = b
            check_equivalent(px + displacement_x, px_prime, f"easy: x image check failed")
            check_equivalent(py + displacement_y, py_prime, f"easy: y image check failed")

            # All checks passed; break out of redraw loop
            break
        else:
            raise ValueError("easy tier redraw loop exceeded 200 attempts")

        answer_text = f"x = {px_prime}, y = {py_prime}"
        distractor_strs = [
            f"x = {sub_x}, y = {sub_y}",
            f"x = {swap_x}, y = {swap_y}",
            f"x = {one_axis_x}, y = {one_axis_y}",
        ]

        return dict(
            text=(
                f"Point P({px}, {py}) is translated by the column vector (a, b) = ({a}, {b}). "
                f"Find the coordinates of the image point P'."
            ),
            latex=rf"P({px}, {py}); (a,b) = ({a},{b}); \text{{find }} P'",
            answer={"x": float(px_prime), "y": float(py_prime)},
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"Translation by column vector (a, b) = ({a}, {b}) means:",
                f"x' = x + a = {px} + {a} = {px_prime}",
                f"y' = y + b = {py} + {b} = {py_prime}",
                f"Therefore: P' = ({px_prime}, {py_prime})",
            ],
            distractors=pick_distractors(answer_text, distractor_strs),
        )

    elif tier == "medium":
        for attempt in range(200):
            # Point P with mixed-sign coordinates
            px, py = rng.randint(-9, 9), rng.randint(-9, 9)
            # Column vector (a, b) with mixed-sign components, excluding 0
            a = rng.choice([v for v in range(-9, 10) if v != 0])
            b = rng.choice([v for v in range(-9, 10) if v != 0])

            # Image point P'
            px_prime = px + a
            py_prime = py + b

            # Build distractors
            # subtracts-instead-of-adds: (Px - a, Py - b)
            sub_x = px - a
            sub_y = py - b
            # swapped-vector-components: (Px + b, Py + a)
            swap_x = px + b
            swap_y = py + a
            # one-axis-only: (Px + a, Py)
            one_axis_x = px + a
            one_axis_y = py

            # Check for pair collisions with answer
            candidates = [
                (sub_x, sub_y),
                (swap_x, swap_y),
                (one_axis_x, one_axis_y),
            ]
            if not _pair_collision_check((px_prime, py_prime), candidates):
                continue

            # Verify construction via independent route: check displacement
            displacement_x = a
            displacement_y = b
            check_equivalent(px + displacement_x, px_prime, f"medium: x image check failed")
            check_equivalent(py + displacement_y, py_prime, f"medium: y image check failed")

            # All checks passed
            break
        else:
            raise ValueError("medium tier redraw loop exceeded 200 attempts")

        answer_text = f"x = {px_prime}, y = {py_prime}"
        distractor_strs = [
            f"x = {sub_x}, y = {sub_y}",
            f"x = {swap_x}, y = {swap_y}",
            f"x = {one_axis_x}, y = {one_axis_y}",
        ]

        return dict(
            text=(
                f"Point P({px}, {py}) is translated by the column vector (a, b) = ({a}, {b}). "
                f"Find the coordinates of the image point P'."
            ),
            latex=rf"P({px}, {py}); (a,b) = ({a},{b}); \text{{find }} P'",
            answer={"x": float(px_prime), "y": float(py_prime)},
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"Translation by column vector (a, b) = ({a}, {b}) means:",
                f"x' = x + a = {px} + {_signed_paren(a)} = {px_prime}",
                f"y' = y + b = {py} + {_signed_paren(b)} = {py_prime}",
                f"Therefore: P' = ({px_prime}, {py_prime})",
            ],
            distractors=pick_distractors(answer_text, distractor_strs),
        )

    elif tier == "hard":
        for attempt in range(200):
            # Point P with mixed-sign coordinates
            px, py = rng.randint(-9, 9), rng.randint(-9, 9)
            # First column vector (a1, b1) with mixed-sign components, excluding 0
            a1 = rng.choice([v for v in range(-9, 10) if v != 0])
            b1 = rng.choice([v for v in range(-9, 10) if v != 0])
            # Second column vector (a2, b2) with mixed-sign components, excluding 0
            a2 = rng.choice([v for v in range(-9, 10) if v != 0])
            b2 = rng.choice([v for v in range(-9, 10) if v != 0])

            # Image after first translation
            px_intermediate = px + a1
            py_intermediate = py + b1

            # Final image after second translation (= Px + a1 + a2, Py + b1 + b2)
            px_prime = px + a1 + a2
            py_prime = py + b1 + b2

            # Total displacement vector
            total_a = a1 + a2
            total_b = b1 + b2

            # Build distractors
            # only-first-translation: (Px + a1, Py + b1)
            only_first_x = px + a1
            only_first_y = py + b1
            # only-second-translation: (Px + a2, Py + b2)
            only_second_x = px + a2
            only_second_y = py + b2
            # subtracts-second-translation: (Px + a1 - a2, Py + b1 - b2)
            sub_second_x = px + a1 - a2
            sub_second_y = py + b1 - b2

            # Check for pair collisions with answer
            candidates = [
                (only_first_x, only_first_y),
                (only_second_x, only_second_y),
                (sub_second_x, sub_second_y),
            ]
            if not _pair_collision_check((px_prime, py_prime), candidates):
                continue

            # Verify construction via independent route: compute total displacement first
            total_displacement_x = total_a
            total_displacement_y = total_b
            check_equivalent(px + total_displacement_x, px_prime, f"hard: x image check failed via total displacement")
            check_equivalent(py + total_displacement_y, py_prime, f"hard: y image check failed via total displacement")

            # All checks passed
            break
        else:
            raise ValueError("hard tier redraw loop exceeded 200 attempts")

        answer_text = f"x = {px_prime}, y = {py_prime}"
        distractor_strs = [
            f"x = {only_first_x}, y = {only_first_y}",
            f"x = {only_second_x}, y = {only_second_y}",
            f"x = {sub_second_x}, y = {sub_second_y}",
        ]

        return dict(
            text=(
                f"Point P({px}, {py}) is translated by the column vector (a1, b1) = ({a1}, {b1}), "
                f"then by the column vector (a2, b2) = ({a2}, {b2}). "
                f"Find the coordinates of the final image point."
            ),
            latex=rf"P({px}, {py}); (a_1,b_1) = ({a1},{b1}); (a_2,b_2) = ({a2},{b2}); \text{{find final}} P'",
            answer={"x": float(px_prime), "y": float(py_prime)},
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"Translate by (a1, b1) = ({a1}, {b1}) first:",
                f"After first translation: ({px} + {_signed_paren(a1)}, {py} + {_signed_paren(b1)}) = ({px_intermediate}, {py_intermediate})",
                f"Then translate by (a2, b2) = ({a2}, {b2}):",
                f"After second translation: ({px_intermediate} + {_signed_paren(a2)}, {py_intermediate} + {_signed_paren(b2)}) = ({px_prime}, {py_prime})",
                f"Equivalently, total displacement: ({a1} + {_signed_paren(a2)}, {b1} + {_signed_paren(b2)}) = ({total_a}, {total_b})",
                f"Final image: P' = ({px_prime}, {py_prime})",
            ],
            distractors=pick_distractors(answer_text, distractor_strs),
        )
