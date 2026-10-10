"""Vector equations of lines and planes (ACMSM105, ACMSM108).

Skill 1 (ACMSM105): Determine vector equation of a straight line (2D direction vector)
- Easy: Two points A, B with positive coords; find direction vector B - A
- Medium: Two points with mixed-sign coords; same formula
- Hard: Line through point A and midpoint M of segment CD; find direction vector M - A

Skill 2 (ACMSM108): Determine Cartesian equation of a plane (one missing scalar)
- Easy: Normal vector and point all positive; find d = a*x0 + b*y0 + c*z0
- Medium: Normal and point with mixed signs; same formula
- Hard: Which-quantity-unknown: given a, b, d, x0, y0, z0; find c = (d - a*x0 - b*y0) / z0
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, check_distinct, check_equivalent, generator, pick_distractors


def _pair_collision_check(answer_pair, candidate_pairs):
    """Check if any candidate pair collides with answer (both components within tolerance).

    Returns True if all candidates are safe (no collisions), False if collision detected.
    Integer-valued pairs differing by any nonzero amount clear grade_pair's tolerance,
    so exact integer equality per component is the collision threshold.
    """
    ax, ay = answer_pair
    for cx, cy in candidate_pairs:
        if cx == ax and cy == ay:
            return False  # Collision: both components exactly match
    return True  # All candidates safe


def _signed_paren(n: int) -> str:
    """Wrap a negative number in parentheses for use right after a literal
    "-" in a worked-steps subtraction/addition expression, e.g. "0 - (-6)"
    instead of "0 - -6" (a double-minus the shared text-quality test
    rejects on every topic). Non-negative values are returned unchanged."""
    return f"({n})" if n < 0 else str(n)


def _join_terms_with_signs(a: int, b: int, var_a: str = "x", var_b: str = "y") -> str:
    """Format two terms like '3x + 2y', '3x - 2y', etc. (never '+ -' or '- -')."""
    term_a = f"{a}{var_a}" if a != 1 else var_a if a > 0 else f"-{var_a}"
    if a == -1:
        term_a = f"-{var_a}"
    elif a == 1:
        term_a = var_a
    else:
        term_a = f"{a}{var_a}"

    if b >= 0:
        term_b = f"+ {b}{var_b}" if b != 1 else f"+ {var_b}" if b > 0 else "+ 0"
        if b == 1:
            term_b = f"+ {var_b}"
        else:
            term_b = f"+ {b}{var_b}"
    else:
        term_b = f"- {abs(b)}{var_b}" if b != -1 else f"- {var_b}"
        if b == -1:
            term_b = f"- {var_b}"
        else:
            term_b = f"- {abs(b)}{var_b}"

    return f"{term_a} {term_b}"


@generator(
    "determine_a_vector_equation_of_a_straight_line",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def determine_a_vector_equation_of_a_straight_line(rng, tier="easy"):
    """Determine the direction vector of a line given two points or equivalent info."""

    if tier == "easy":
        for attempt in range(200):
            # Two points with positive coordinates
            ax, ay = rng.randint(1, 9), rng.randint(1, 9)
            bx, by = rng.randint(1, 9), rng.randint(1, 9)

            # Reject if A == B
            if ax == bx and ay == by:
                continue

            dx = bx - ax
            dy = by - ay

            # Build distractors: wrong-direction, swapped-components, added-instead-of-subtracted
            wrong_dir_x = ax - bx
            wrong_dir_y = ay - by
            swapped_x = dy
            swapped_y = dx
            added_x = ax + bx
            added_y = ay + by

            # Check for pair collisions with answer
            candidates = [
                (wrong_dir_x, wrong_dir_y),
                (swapped_x, swapped_y),
                (added_x, added_y),
            ]
            if not _pair_collision_check((dx, dy), candidates):
                continue

            # All candidate pairs are safe; verify construction via inverse relation
            check_equivalent(ax + dx, bx, f"easy: forward check failed (x): {ax} + {dx} != {bx}")
            check_equivalent(ay + dy, by, f"easy: forward check failed (y): {ay} + {dy} != {by}")

            # All checks passed, break out of redraw loop
            break
        else:
            raise ValueError("easy tier redraw loop exceeded 200 attempts")

        answer_text = f"x = {dx}, y = {dy}"
        distractor_strs = [
            f"x = {wrong_dir_x}, y = {wrong_dir_y}",
            f"x = {swapped_x}, y = {swapped_y}",
            f"x = {added_x}, y = {added_y}",
        ]

        # Build working steps
        working = [
            f"A = ({ax}, {ay}), B = ({bx}, {by})",
            f"Direction vector = B - A = ({bx} - {_signed_paren(ax)}, {by} - {_signed_paren(ay)})",
            f"= ({dx}, {dy})",
        ]

        # Pick distractors and parse them to detect scalar multiples
        picked_distractors = pick_distractors(answer_text, distractor_strs)

        # Parse distractor strings and detect scalar multiples
        for distractor_str in picked_distractors:
            # Parse format "x = ..., y = ..."
            parts = distractor_str.replace("x = ", "").replace("y = ", "").split(", ")
            if len(parts) == 2:
                try:
                    dist_x = int(parts[0])
                    dist_y = int(parts[1])
                    # Check if distractor is a positive scalar multiple of answer
                    if dx * dist_y - dy * dist_x == 0 and dx * dist_x + dy * dist_y > 0:
                        # Compute scalar
                        scalar = Fraction(dist_x, dx) if dx != 0 else Fraction(dist_y, dy)
                        scale_phrase = "scaled down" if scalar < 1 else "scaled up"
                        working.append(
                            f"Note: ({dist_x}, {dist_y}) is also a direction vector for this line -- "
                            f"it is the same direction {scale_phrase}, so that instinct is right. "
                            f"This question asked for B minus A specifically, which is ({dx}, {dy})."
                        )
                except (ValueError, ZeroDivisionError):
                    pass

        return dict(
            text=(
                f"A straight line passes through the points A({ax}, {ay}) and B({bx}, {by}). "
                f"Find the direction vector of the line, B minus A."
            ),
            latex=rf"A({ax}, {ay}), B({bx}, {by}); \text{{find direction vector}}",
            answer={"x": float(dx), "y": float(dy)},
            answer_text=answer_text,
            distractors=picked_distractors,
            working=working,
        )

    elif tier == "medium":
        for attempt in range(200):
            # Two points with mixed-sign coordinates
            ax, ay = rng.randint(-9, 9), rng.randint(-9, 9)
            bx, by = rng.randint(-9, 9), rng.randint(-9, 9)

            # Reject if A == B
            if ax == bx and ay == by:
                continue

            dx = bx - ax
            dy = by - ay

            # Build distractors
            wrong_dir_x = ax - bx
            wrong_dir_y = ay - by
            swapped_x = dy
            swapped_y = dx
            added_x = ax + bx
            added_y = ay + by

            # Check for pair collisions with answer
            candidates = [
                (wrong_dir_x, wrong_dir_y),
                (swapped_x, swapped_y),
                (added_x, added_y),
            ]
            if not _pair_collision_check((dx, dy), candidates):
                continue

            # All candidate pairs are safe; verify construction via inverse relation
            check_equivalent(ax + dx, bx, f"medium: forward check failed (x): {ax} + {dx} != {bx}")
            check_equivalent(ay + dy, by, f"medium: forward check failed (y): {ay} + {dy} != {by}")

            # All checks passed
            break
        else:
            raise ValueError("medium tier redraw loop exceeded 200 attempts")

        answer_text = f"x = {dx}, y = {dy}"
        distractor_strs = [
            f"x = {wrong_dir_x}, y = {wrong_dir_y}",
            f"x = {swapped_x}, y = {swapped_y}",
            f"x = {added_x}, y = {added_y}",
        ]

        # Build working steps
        working = [
            f"A = ({ax}, {ay}), B = ({bx}, {by})",
            f"Direction vector = B - A = ({bx} - {_signed_paren(ax)}, {by} - {_signed_paren(ay)})",
            f"= ({dx}, {dy})",
        ]

        # Pick distractors and parse them to detect scalar multiples
        picked_distractors = pick_distractors(answer_text, distractor_strs)

        # Parse distractor strings and detect scalar multiples
        for distractor_str in picked_distractors:
            # Parse format "x = ..., y = ..."
            parts = distractor_str.replace("x = ", "").replace("y = ", "").split(", ")
            if len(parts) == 2:
                try:
                    dist_x = int(parts[0])
                    dist_y = int(parts[1])
                    # Check if distractor is a positive scalar multiple of answer
                    if dx * dist_y - dy * dist_x == 0 and dx * dist_x + dy * dist_y > 0:
                        # Compute scalar
                        scalar = Fraction(dist_x, dx) if dx != 0 else Fraction(dist_y, dy)
                        scale_phrase = "scaled down" if scalar < 1 else "scaled up"
                        working.append(
                            f"Note: ({dist_x}, {dist_y}) is also a direction vector for this line -- "
                            f"it is the same direction {scale_phrase}, so that instinct is right. "
                            f"This question asked for B minus A specifically, which is ({dx}, {dy})."
                        )
                except (ValueError, ZeroDivisionError):
                    pass

        return dict(
            text=(
                f"A straight line passes through the points A({ax}, {ay}) and B({bx}, {by}). "
                f"Find the direction vector of the line, B minus A."
            ),
            latex=rf"A({ax}, {ay}), B({bx}, {by}); \text{{find direction vector}}",
            answer={"x": float(dx), "y": float(dy)},
            answer_text=answer_text,
            distractors=picked_distractors,
            working=working,
        )

    elif tier == "hard":
        for attempt in range(200):
            # Point A
            ax, ay = rng.randint(-9, 9), rng.randint(-9, 9)

            # Segment CD with same parity per component
            cx, cy = rng.randint(-9, 9), rng.randint(-9, 9)
            dx_val = cx + 2 * rng.randint(-4, 4)
            dy_val = cy + 2 * rng.randint(-4, 4)

            # Reject if D == C
            if dx_val == cx and dy_val == cy:
                continue

            # Compute midpoint (guaranteed to be integer pair)
            mx_frac = Fraction(cx + dx_val, 2)
            my_frac = Fraction(cy + dy_val, 2)

            # Verify midpoint is exact integer
            check(mx_frac.denominator == 1, f"hard: midpoint x not integer: {mx_frac}")
            check(my_frac.denominator == 1, f"hard: midpoint y not integer: {my_frac}")

            mx = int(mx_frac)
            my = int(my_frac)

            # Answer: direction vector from A to M
            dx = mx - ax
            dy = my - ay

            # Build distractors
            # wrong-direction: A - M
            wrong_dir_x = ax - mx
            wrong_dir_y = ay - my
            # swapped-components: (dy, dx)
            swapped_x = dy
            swapped_y = dx
            # added-instead-of-subtracted: A + M
            added_x = ax + mx
            added_y = ay + my
            # used-second-point-not-midpoint: C - A (hard tier only)
            used_c_x = cx - ax
            used_c_y = cy - ay

            # Check for pair collisions with answer
            candidates = [
                (wrong_dir_x, wrong_dir_y),
                (swapped_x, swapped_y),
                (added_x, added_y),
                (used_c_x, used_c_y),
            ]
            if not _pair_collision_check((dx, dy), candidates):
                continue

            # All candidate pairs are safe; verify direction via Fraction independent check
            check_equivalent(ax + dx, int(mx_frac), f"hard: forward check failed (x): {ax} + {dx} != {int(mx_frac)}")
            check_equivalent(ay + dy, int(my_frac), f"hard: forward check failed (y): {ay} + {dy} != {int(my_frac)}")

            # All checks passed
            break
        else:
            raise ValueError("hard tier redraw loop exceeded 200 attempts")

        answer_text = f"x = {dx}, y = {dy}"
        distractor_strs = [
            f"x = {wrong_dir_x}, y = {wrong_dir_y}",
            f"x = {swapped_x}, y = {swapped_y}",
            f"x = {added_x}, y = {added_y}",
            f"x = {used_c_x}, y = {used_c_y}",
        ]

        # Build working steps
        working = [
            f"Midpoint of C({cx}, {cy}) and D({dx_val}, {dy_val}) = (({_signed_paren(cx)} + {_signed_paren(dx_val)}) / 2, ({_signed_paren(cy)} + {_signed_paren(dy_val)}) / 2) = ({mx}, {my})",
            f"Direction vector = midpoint - A = ({mx} - {_signed_paren(ax)}, {my} - {_signed_paren(ay)})",
            f"= ({dx}, {dy})",
        ]

        # Pick distractors and parse them to detect scalar multiples
        picked_distractors = pick_distractors(answer_text, distractor_strs)

        # Parse distractor strings and detect scalar multiples
        for distractor_str in picked_distractors:
            # Parse format "x = ..., y = ..."
            parts = distractor_str.replace("x = ", "").replace("y = ", "").split(", ")
            if len(parts) == 2:
                try:
                    dist_x = int(parts[0])
                    dist_y = int(parts[1])
                    # Check if distractor is a positive scalar multiple of answer
                    if dx * dist_y - dy * dist_x == 0 and dx * dist_x + dy * dist_y > 0:
                        # Compute scalar
                        scalar = Fraction(dist_x, dx) if dx != 0 else Fraction(dist_y, dy)
                        scale_phrase = "scaled down" if scalar < 1 else "scaled up"
                        working.append(
                            f"Note: ({dist_x}, {dist_y}) is also a direction vector for this line -- "
                            f"it is the same direction {scale_phrase}, so that instinct is right. "
                            f"This question asked for the midpoint minus A specifically, which is ({dx}, {dy})."
                        )
                except (ValueError, ZeroDivisionError):
                    pass

        return dict(
            text=(
                f"A straight line passes through the point A({ax}, {ay}) and through the midpoint "
                f"of the segment joining C({cx}, {cy}) and D({dx_val}, {dy_val}). "
                f"Find the direction vector of the line, the midpoint minus A."
            ),
            latex=rf"A({ax}, {ay}); \text{{midpoint of }} C({cx}, {cy}), D({dx_val}, {dy_val}); \text{{find direction vector (midpoint}} - A\text{{)}}",
            answer={"x": float(dx), "y": float(dy)},
            answer_text=answer_text,
            distractors=picked_distractors,
            working=working,
        )


@generator(
    "determine_vector_and_cartesian_equations_of_a_plane",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def determine_vector_and_cartesian_equations_of_a_plane(rng, tier="easy"):
    """Determine a scalar component of the Cartesian equation of a plane."""

    if tier == "easy":
        for attempt in range(200):
            # Normal vector (a, b, c) with positive components
            a = rng.randint(1, 9)
            b = rng.randint(1, 9)
            c = rng.randint(1, 9)

            # Point (x0, y0, z0) with positive components
            x0 = rng.randint(1, 9)
            y0 = rng.randint(1, 9)
            z0 = rng.randint(1, 9)

            # Answer: d = a*x0 + b*y0 + c*z0
            d = a * x0 + b * y0 + c * z0

            # Independent check: different computation path
            d_check = sum(coef * coord for coef, coord in zip((a, b, c), (x0, y0, z0)))
            check_equivalent(d, d_check, f"easy: d computation mismatch")

            # Build distractors
            forgot_last = a * x0 + b * y0
            sign_error_middle = a * x0 - b * y0 + c * z0
            sum_not_dot = (a + b + c) + (x0 + y0 + z0)

            # Check for collisions with answer (redraw if any distractor equals answer)
            if d == forgot_last or d == sign_error_middle or d == sum_not_dot:
                continue

            # Check all candidates are distinct integers
            check_distinct(d, forgot_last, msg=f"easy: forgot-last-term collides")
            check_distinct(d, sign_error_middle, msg=f"easy: sign-error-middle-term collides")
            check_distinct(d, sum_not_dot, msg=f"easy: sum-not-dot-product collides")

            break
        else:
            raise ValueError("easy tier redraw loop exceeded 200 attempts")

        answer_text = str(d)
        distractor_strs = [str(forgot_last), str(sign_error_middle), str(sum_not_dot)]

        working = [
            "d = a × x0 + b × y0 + c × z0",
            f"d = {a} × {x0} + {b} × {y0} + {c} × {z0}",
            f"d = {a * x0} + {b * y0} + {c * z0}",
            f"d = {d}",
        ]

        return dict(
            text=(
                f"A plane has normal vector (a, b, c) = ({a}, {b}, {c}) and passes through "
                f"the point ({x0}, {y0}, {z0}). The plane's Cartesian equation has the form "
                f"ax + by + cz = d. Find d."
            ),
            latex=rf"({a})x + ({b})y + ({c})z = d; \text{{point}} ({x0}, {y0}, {z0})",
            answer=float(d),
            answer_text=answer_text,
            distractors=pick_distractors(answer_text, distractor_strs),
            working=working,
        )

    elif tier == "medium":
        for attempt in range(200):
            # Normal vector and point with mixed-sign components
            a = rng.randint(-9, 9)
            b = rng.randint(-9, 9)
            c = rng.randint(-9, 9)

            # Reject if normal vector is (0, 0, 0) -- does not define a plane
            if a == 0 and b == 0 and c == 0:
                continue

            x0 = rng.randint(-9, 9)
            y0 = rng.randint(-9, 9)
            z0 = rng.randint(-9, 9)

            # Answer: d = a*x0 + b*y0 + c*z0
            d = a * x0 + b * y0 + c * z0

            # Build distractors
            forgot_last = a * x0 + b * y0
            sign_error_middle = a * x0 - b * y0 + c * z0
            sum_not_dot = (a + b + c) + (x0 + y0 + z0)

            # Check for collisions with answer (redraw if any distractor equals answer)
            if d == forgot_last or d == sign_error_middle or d == sum_not_dot:
                continue

            # Independent check: different computation path
            d_check = sum(coef * coord for coef, coord in zip((a, b, c), (x0, y0, z0)))
            check_equivalent(d, d_check, f"medium: d computation mismatch")

            # Verify each distractor is distinct
            check_distinct(d, forgot_last, msg=f"medium: forgot-last-term collides")
            check_distinct(d, sign_error_middle, msg=f"medium: sign-error-middle-term collides")
            check_distinct(d, sum_not_dot, msg=f"medium: sum-not-dot-product collides")

            break
        else:
            raise ValueError("medium tier redraw loop exceeded 200 attempts")

        answer_text = str(d)
        distractor_strs = [str(forgot_last), str(sign_error_middle), str(sum_not_dot)]

        working = [
            "d = a × x0 + b × y0 + c × z0",
            f"d = {_signed_paren(a)} × {_signed_paren(x0)} + {_signed_paren(b)} × {_signed_paren(y0)} + {_signed_paren(c)} × {_signed_paren(z0)}",
            f"d = {_signed_paren(a * x0)} + {_signed_paren(b * y0)} + {_signed_paren(c * z0)}",
            f"d = {d}",
        ]

        return dict(
            text=(
                f"A plane has normal vector (a, b, c) = ({a}, {b}, {c}) and passes through "
                f"the point ({x0}, {y0}, {z0}). The plane's Cartesian equation has the form "
                f"ax + by + cz = d. Find d."
            ),
            latex=rf"({a})x + ({b})y + ({c})z = d; \text{{point}} ({x0}, {y0}, {z0})",
            answer=float(d),
            answer_text=answer_text,
            distractors=pick_distractors(answer_text, distractor_strs),
            working=working,
        )

    elif tier == "hard":
        for attempt in range(200):
            # a, b drawn from [-9, 9], excluding 0
            a = rng.choice([v for v in range(-9, 10) if v != 0])
            b = rng.choice([v for v in range(-9, 10) if v != 0])

            # True c (never shown), drawn from [-9, 9], excluding 0
            c_true = rng.choice([v for v in range(-9, 10) if v != 0])

            # x0, y0 from [-9, 9]
            x0 = rng.randint(-9, 9)
            y0 = rng.randint(-9, 9)

            # z0 from [-9, 9], excluding 0 and 1
            # (0 makes equation independent of c; 1 makes forgot-to-divide distractor collide)
            z0 = rng.choice([v for v in range(-9, 10) if v not in (0, 1)])

            # Compute d using true c
            d = a * x0 + b * y0 + c_true * z0

            # Answer: c = (d - a*x0 - b*y0) / z0
            # We need to compute this exactly via Fraction
            c_frac = Fraction(d - a * x0 - b * y0, z0)
            check(c_frac.denominator == 1, f"hard: c not integer: {c_frac} (z0={z0})")
            c = int(c_frac)

            # Independent check: recompute c via Fraction
            c_check_frac = Fraction(d - a * x0 - b * y0, z0)
            check_equivalent(c_check_frac, c, f"hard: c computation mismatch")

            # Build distractors
            forgot_to_divide = d - a * x0 - b * y0
            negated_answer = -c
            off_by_one = c + 1

            # Check for collisions with answer (redraw if any distractor equals answer)
            if c == forgot_to_divide or c == negated_answer or c == off_by_one:
                continue

            check_distinct(c, forgot_to_divide, msg=f"hard: forgot-to-divide collides")
            check_distinct(c, negated_answer, msg=f"hard: negated-answer collides")
            check_distinct(c, off_by_one, msg=f"hard: off-by-one collides")

            break
        else:
            raise ValueError("hard tier redraw loop exceeded 200 attempts")

        answer_text = str(c)
        distractor_strs = [str(forgot_to_divide), str(negated_answer), str(off_by_one)]

        # Format the equation text with correct signs (no "+ -" or "- -")
        # Handle coefficient of 1 or -1 correctly (1 -> "x", -1 -> "-x")
        if a == 1:
            a_term = "x"
            a_latex = "x"
        elif a == -1:
            a_term = "-x"
            a_latex = "-x"
        else:
            a_term = f"{a}x"
            a_latex = f"{a}x"

        # Format b magnitude for text and latex (1 -> "y", -1 -> "y", etc.)
        if abs(b) == 1:
            b_magnitude = "y"
        else:
            b_magnitude = f"{abs(b)}y"

        # Format b_term for text (keeping existing +/- logic)
        if b > 0:
            b_term = f"+ {b_magnitude}"
        else:
            b_term = f"- {b_magnitude}"

        working = [
            f"{a_term} {b_term} + cz = {d}",
            f"c × {_signed_paren(z0)} = {d} - {_signed_paren(a)} × {_signed_paren(x0)} - {_signed_paren(b)} × {_signed_paren(y0)}",
            f"c × {_signed_paren(z0)} = {_signed_paren(d - a * x0 - b * y0)}",
            f"c = {_signed_paren(d - a * x0 - b * y0)} / {_signed_paren(z0)} = {c}",
        ]

        return dict(
            text=(
                f"A plane has Cartesian equation {a_term} {b_term} + cz = {d} (c unknown) "
                f"and passes through the point ({x0}, {y0}, {z0}). Find c."
            ),
            latex=rf"{a_latex} {'+' if b > 0 else '-'} {b_magnitude} + cz = {d}; \text{{point}} ({x0}, {y0}, {z0})",
            answer=float(c),
            answer_text=answer_text,
            distractors=pick_distractors(answer_text, distractor_strs),
            working=working,
        )
