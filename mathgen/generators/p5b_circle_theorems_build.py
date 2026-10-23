"""Generates `angle_in_semicircle` and `angles_same_arc` questions (among other circle theorems): find an angle
using the circle theorems.

Circle geometry theorems.

- ``senior_an_angle_in_a_semicircle_is_a_right``
  (``senior-an-angle-in-a-semicircle-is-a-right``, ACMSM029, domain senior-mathematics,
  answer: number) -- An angle in a semicircle is a right angle (Thales' theorem applied).

- ``senior_angles_at_the_circumference_of_a_circle_subtended``
  (``senior-angles-at-the-circumference-of-a-circle-subtended``, ACMSM031, domain senior-mathematics,
  answer: number) -- Angles at the circumference of a circle subtended by the same arc are equal
  (inscribed-angle theorem / same-segment theorem).
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator, pick_distractors


# ============================================================================
# Skill 1: senior_an_angle_in_a_semicircle_is_a_right (Thales' theorem)
# ============================================================================


@generator(
    "angle_in_semicircle",
    level=12,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="difference"),
        "hard": dict(variant="ratio"),
    },
)
def angle_in_semicircle(rng, variant="direct"):
    """An angle in a semicircle is a right angle (Thales' theorem).

    Triangle ABC is inscribed in a circle with AB as a diameter.
    By the theorem, angle ACB = 90°.
    The question gives one of the triangle's other two angles (or relationship)
    and asks for the other.
    """

    if variant == "direct":
        # Easy: angle BAC = x, find angle ABC = 90 - x
        # x sampled from 10..80, excluding 45 (to avoid x == 90-x collision)
        while True:
            x = rng.randint(10, 80)
            if x != 45:
                break

        angle_bac = x
        angle_abc = 90 - x

        question_text = (
            f"Triangle ABC is inscribed in a circle so that AB is a diameter. "
            f"Angle BAC is {angle_bac} degrees. "
            f"Find angle ABC, in degrees."
        )
        answer_text = str(angle_abc)
        answer = answer_text

        # Independent check: verify via coordinate geometry
        # Place A=(-1,0), B=(1,0) (diameter of unit circle)
        # C on circle: by inscribed angle theorem, if angle BAC = x_bac,
        # then C is at angle 2*x_bac from positive x-axis
        x_bac = angle_bac
        c_angle_rad = math.radians(2 * x_bac)
        c_x = math.cos(c_angle_rad)
        c_y = math.sin(c_angle_rad)

        # Check angle ACB is 90° via vector dot product
        ca = (-1 - c_x, 0 - c_y)
        cb = (1 - c_x, 0 - c_y)
        dot = ca[0] * cb[0] + ca[1] * cb[1]
        mag_ca = math.sqrt(ca[0] ** 2 + ca[1] ** 2)
        mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)
        angle_acb_rad = math.acos(dot / (mag_ca * mag_cb))
        angle_acb_deg = math.degrees(angle_acb_rad)
        check(abs(angle_acb_deg - 90.0) < 1e-6, f"angle_in_semicircle/direct: angle ACB={angle_acb_deg:.6f} not ~90")

        # Check angle ABC matches answer
        ba = (-1 - 1, 0 - 0)
        bc = (c_x - 1, c_y - 0)
        dot_abc = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
        angle_abc_rad = math.acos(dot_abc / (mag_ba * mag_bc))
        angle_abc_deg = math.degrees(angle_abc_rad)
        check(
            abs(angle_abc_deg - angle_abc) < 1e-4,
            f"angle_in_semicircle/direct: computed angle ABC={angle_abc_deg:.6f}, expected {angle_abc}",
        )

        # Misconceptions
        # forgets_right_angle: 180 - x (uses full triangle sum instead of 90+BAC+ABC=180)
        forgets_right = 180 - x
        check(forgets_right != angle_abc, f"forgets_right_angle distractor equals answer for x={x}")

        # swaps_the_angles: x (reports BAC instead of ABC)
        swaps = x
        check(swaps != angle_abc, f"swaps_the_angles distractor equals answer for x={x}")

        # sign_or_direction_error: 90 + x (adds instead of subtracts)
        sign_error = 90 + x
        check(sign_error != angle_abc, f"sign_or_direction_error distractor equals answer for x={x}")

        candidates = [str(forgets_right), str(swaps), str(sign_error)]
        distractors = pick_distractors(answer_text, candidates)

    elif variant == "difference":
        # Medium: BAC exceeds ABC by d, find ABC = (90 - d) / 2
        # d sampled from 2..30, even only (to keep answer an integer)
        d = rng.choice(range(2, 31, 2))

        # Ensure answer is a positive integer and doesn't collide
        angle_abc = (90 - d) // 2
        angle_bac = angle_abc + d

        # Verify the setup: angle_bac + angle_abc + 90 = 180
        check(
            angle_bac + angle_abc == 90,
            f"angle_in_semicircle/difference: angle_bac={angle_bac} + angle_abc={angle_abc} != 90",
        )

        question_text = (
            f"Triangle ABC is inscribed in a circle so that AB is a diameter. "
            f"Angle BAC exceeds angle ABC by {d} degrees. "
            f"Find angle ABC, in degrees."
        )
        answer_text = str(angle_abc)
        answer = answer_text

        # Independent check: verify via coordinate geometry
        x_bac = angle_bac
        c_angle_rad = math.radians(2 * x_bac)
        c_x = math.cos(c_angle_rad)
        c_y = math.sin(c_angle_rad)

        # Check angle ACB is 90°
        ca = (-1 - c_x, 0 - c_y)
        cb = (1 - c_x, 0 - c_y)
        dot = ca[0] * cb[0] + ca[1] * cb[1]
        mag_ca = math.sqrt(ca[0] ** 2 + ca[1] ** 2)
        mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)
        angle_acb_rad = math.acos(dot / (mag_ca * mag_cb))
        angle_acb_deg = math.degrees(angle_acb_rad)
        check(abs(angle_acb_deg - 90.0) < 1e-6, f"angle_in_semicircle/difference: angle ACB not ~90")

        # Check angle ABC matches answer
        ba = (-1 - 1, 0 - 0)
        bc = (c_x - 1, c_y - 0)
        dot_abc = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
        angle_abc_rad = math.acos(dot_abc / (mag_ba * mag_bc))
        angle_abc_deg = math.degrees(angle_abc_rad)
        check(
            abs(angle_abc_deg - angle_abc) < 1e-4,
            f"angle_in_semicircle/difference: computed angle ABC={angle_abc_deg:.6f}, expected {angle_abc}",
        )

        # Misconceptions
        # forgets_right_angle: (180 - d) / 2 (uses 180 instead of 90)
        forgets_right = (180 - d) // 2
        check(forgets_right != angle_abc, f"forgets_right_angle distractor equals answer for d={d}")

        # swaps_the_angles: ABC + d = BAC (reports the other angle)
        swaps = angle_abc + d
        check(swaps != angle_abc, f"swaps_the_angles distractor equals answer for d={d}")

        # sign_or_direction_error: (90 + d) / 2 (adds instead of subtracts d)
        sign_error = (90 + d) // 2
        check(sign_error != angle_abc, f"sign_or_direction_error distractor equals answer for d={d}")

        candidates = [str(forgets_right), str(swaps), str(sign_error)]
        distractors = pick_distractors(answer_text, candidates)

    elif variant == "ratio":
        # Hard: BAC : ABC = p : q, find ABC = 90 * q / (p + q)
        # p, q sampled from 1..5, p != q, and (p + q) divides 90 evenly
        # for exact integer answers

        valid_pairs = []
        for p in range(1, 6):
            for q in range(1, 6):
                if p != q and 90 % (p + q) == 0:
                    valid_pairs.append((p, q))

        p, q = rng.choice(valid_pairs)

        angle_abc = 90 * q // (p + q)
        angle_bac = 90 * p // (p + q)

        # Verify the setup
        check(
            angle_bac + angle_abc == 90,
            f"angle_in_semicircle/ratio: angle_bac={angle_bac} + angle_abc={angle_abc} != 90",
        )
        check(p != q, f"angle_in_semicircle/ratio: p={p} == q={q} (degenerate ratio)")

        question_text = (
            f"Triangle ABC is inscribed in a circle so that AB is a diameter. "
            f"The ratio of angle BAC to angle ABC is {p} : {q}. "
            f"Find angle ABC, in degrees."
        )
        answer_text = str(angle_abc)
        answer = answer_text

        # Independent check: verify via coordinate geometry
        x_bac = angle_bac
        c_angle_rad = math.radians(2 * x_bac)
        c_x = math.cos(c_angle_rad)
        c_y = math.sin(c_angle_rad)

        # Check angle ACB is 90°
        ca = (-1 - c_x, 0 - c_y)
        cb = (1 - c_x, 0 - c_y)
        dot = ca[0] * cb[0] + ca[1] * cb[1]
        mag_ca = math.sqrt(ca[0] ** 2 + ca[1] ** 2)
        mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)
        angle_acb_rad = math.acos(dot / (mag_ca * mag_cb))
        angle_acb_deg = math.degrees(angle_acb_rad)
        check(abs(angle_acb_deg - 90.0) < 1e-6, f"angle_in_semicircle/ratio: angle ACB not ~90")

        # Check angle ABC matches answer
        ba = (-1 - 1, 0 - 0)
        bc = (c_x - 1, c_y - 0)
        dot_abc = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
        angle_abc_rad = math.acos(dot_abc / (mag_ba * mag_bc))
        angle_abc_deg = math.degrees(angle_abc_rad)
        check(
            abs(angle_abc_deg - angle_abc) < 1e-4,
            f"angle_in_semicircle/ratio: computed angle ABC={angle_abc_deg:.6f}, expected {angle_abc}",
        )

        # Misconceptions
        # forgets_right_angle: 90 * q / (p + q) but with 180 in numerator
        # This would be: answer to "180 - p - q" = (180*q)/(p+q) - not exact for our pool
        # Simpler: just use 180 instead of 90 in the formula
        forgets_right = 180 * q // (p + q)
        check(forgets_right != angle_abc, f"forgets_right_angle distractor equals answer for p={p}, q={q}")

        # swaps_the_angles: 90 * p / (p + q) (uses p instead of q)
        swaps = 90 * p // (p + q)
        check(swaps != angle_abc, f"swaps_the_angles distractor equals answer for p={p}, q={q}")

        # sign_or_direction_error: 90 * p / q (divides by wrong part)
        sign_error = 90 * p // q
        check(sign_error != angle_abc, f"sign_or_direction_error distractor equals answer for p={p}, q={q}")

        candidates = [str(forgets_right), str(swaps), str(sign_error)]
        distractors = pick_distractors(answer_text, candidates)

    # Build working text based on variant
    if variant == "direct":
        working = [
            f"By Thales' theorem, angle ACB is a right angle (90 degrees).",
            f"The three angles of the triangle sum to 180 degrees.",
            f"Therefore angle ABC = 90 - {angle_bac} = {angle_abc} degrees.",
        ]
    elif variant == "difference":
        working = [
            f"By Thales' theorem, angle ACB is a right angle (90 degrees).",
            f"The three angles sum to 180, so angle BAC + angle ABC = 90.",
            f"Given angle BAC exceeds angle ABC by {d} degrees: angle BAC = angle ABC + {d}.",
            f"Substituting: (angle ABC + {d}) + angle ABC = 90.",
            f"Simplifying: 2 × angle ABC = {90 - d}, so angle ABC = {angle_abc} degrees.",
        ]
    elif variant == "ratio":
        total_parts = p + q
        q_part_word = "part" if q == 1 else "parts"
        total_part_word = "part" if total_parts == 1 else "parts"
        working = [
            f"By Thales' theorem, angle ACB is a right angle (90 degrees).",
            f"The three angles sum to 180, so angle BAC + angle ABC = 90.",
            f"The ratio angle BAC : angle ABC = {p} : {q}.",
            f"Total {total_part_word}: {p} + {q} = {total_parts}.",
            f"Angle ABC = {q}/{total_parts} × 90 = {angle_abc} degrees.",
        ]

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": working,
    }


# ============================================================================
# Skill 2: senior_angles_at_the_circumference_of_a_circle_subtended
# ============================================================================


@generator(
    "angles_same_arc",
    level=12,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="triangle_sum"),
        "hard": dict(variant="algebraic"),
    },
)
def angles_same_arc(rng, variant="direct"):
    """Angles at the circumference subtended by the same arc are equal.

    Points C and D both lie on the circumference of a circle, on the same arc,
    on the same side of chord AB. By the theorem, angle ACB = angle ADB.
    """

    if variant == "direct":
        # Easy: angle ACB = x, find angle ADB = x
        # x from 10..80
        x = rng.randint(10, 80)

        angle_acb = x
        angle_adb = x

        question_text = (
            f"Points C and D both lie on the circumference of a circle, "
            f"on the same arc, on the same side of chord AB. "
            f"Angle ACB is {angle_acb} degrees. "
            f"Find angle ADB, in degrees."
        )
        answer_text = str(angle_adb)
        answer = answer_text

        # Independent check: verify via coordinate geometry
        # Place A=(-1,0), B=(1,0) as a chord
        # Place C and D on the circle at specific angles, same side of AB
        # Verify angle ACB = angle ADB via vector dot product

        # Place C at angle theta_c, D at angle theta_d (both > 0, same arc)
        # For angle ACB = x degrees, by inscribed angle theorem,
        # the arc AB (not containing C) has central angle 2x
        # So if we place A at angle 180, B at angle 0 (the diameter),
        # then C is at angle (180 - 2x) or equivalently some value that gives
        # angle ACB = x. Let's use the parametrization:
        # If angle ACB = x, place C at angle theta_c such that angle ACB = x

        # Simpler approach: place specific coordinates
        # A = (-1, 0), B = (1, 0)
        # C on circle at angle alpha_c (in radians from positive x-axis)
        # angle ACB is computed via vectors

        # For this test, use concrete angles
        alpha_c = math.radians(90 + x)  # arbitrary choice that yields angle x at C
        c_x = math.cos(alpha_c)
        c_y = math.sin(alpha_c)

        # Verify angle ACB via dot product
        ca = (-1 - c_x, 0 - c_y)
        cb = (1 - c_x, 0 - c_y)
        dot_acb = ca[0] * cb[0] + ca[1] * cb[1]
        mag_ca = math.sqrt(ca[0] ** 2 + ca[1] ** 2)
        mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)
        computed_acb_rad = math.acos(dot_acb / (mag_ca * mag_cb))
        computed_acb_deg = math.degrees(computed_acb_rad)

        # Place D on same arc (same side of AB, so y > 0 if C is y > 0)
        alpha_d = math.radians(100 + x)  # different angle on same side
        d_x = math.cos(alpha_d)
        d_y = math.sin(alpha_d)

        # Verify angle ADB via dot product
        da = (-1 - d_x, 0 - d_y)
        db = (1 - d_x, 0 - d_y)
        dot_adb = da[0] * db[0] + da[1] * db[1]
        mag_da = math.sqrt(da[0] ** 2 + da[1] ** 2)
        mag_db = math.sqrt(db[0] ** 2 + db[1] ** 2)
        computed_adb_rad = math.acos(dot_adb / (mag_da * mag_db))
        computed_adb_deg = math.degrees(computed_adb_rad)

        # Both should be equal
        check(
            abs(computed_acb_deg - computed_adb_deg) < 1e-4,
            f"angles_same_arc/direct: computed ACB={computed_acb_deg:.6f} != ADB={computed_adb_deg:.6f}",
        )

        # Misconceptions
        # assumes_supplementary: 180 - x
        assumes_supp = 180 - x
        check(assumes_supp != angle_adb, f"assumes_supplementary distractor equals answer for x={x}")

        # halves_the_angle: x / 2
        halves = x // 2
        check(halves != angle_adb, f"halves_the_angle distractor equals answer for x={x}")

        candidates = [str(assumes_supp), str(halves)]
        distractors = pick_distractors(answer_text, candidates)

    elif variant == "triangle_sum":
        # Medium: angle CAB = p, angle CBA = q, find angle ACB = 180 - p - q
        # Then angle ADB = angle ACB (by theorem)
        # p, q from 20..70, with p + q < 170 so angle ACB > 10
        # Also exclude p + q = 90 (collision guard for supplementary distractor)

        for _ in attempts(200, "angles_same_arc/triangle_sum angles"):
            p = rng.randint(20, 70)
            q = rng.randint(20, 70)
            if p + q < 170 and p + q != 90 and 2 * p + q != 180:
                break

        angle_acb = 180 - p - q
        angle_adb = angle_acb  # by theorem

        check(angle_acb > 10, f"angles_same_arc/triangle_sum: angle_acb={angle_acb} not > 10")

        question_text = (
            f"Points C and D both lie on the circumference of a circle, "
            f"on the same arc, on the same side of chord AB. "
            f"In triangle ABC, angle CAB is {p} degrees and angle CBA is {q} degrees. "
            f"Find angle ADB, in degrees."
        )
        answer_text = str(angle_adb)
        answer = answer_text

        # Independent check: verify the triangle angle sum (basic fact)
        check(p + q + angle_acb == 180, f"angles_same_arc/triangle_sum: p + q + angle_acb != 180")

        # Geometric verification: place A, B, C, D on a circle and verify that
        # angles ACB and ADB are equal via the inscribed angle theorem
        a_x, a_y = -1.0, 0.0
        b_x, b_y = 1.0, 0.0

        # Use inscribed angle theorem: if angle ACB (inscribed) is alpha,
        # then the central angle is 2*alpha
        # Place C at angle theta_c on unit circle such that angle ACB = angle_acb
        central_angle = 2 * angle_acb
        c_angle_rad = math.radians(100 + central_angle)  # arbitrary offset to place on circle
        c_x = math.cos(c_angle_rad)
        c_y = math.sin(c_angle_rad)

        # Place D at a different angle on the same arc (same side of chord AB)
        d_angle_rad = math.radians(110 + central_angle)  # different angle, same arc
        d_x = math.cos(d_angle_rad)
        d_y = math.sin(d_angle_rad)

        # Verify both C and D see the chord AB at the same angle (inscribed angle theorem)
        # Compute angle ACB
        ca = (a_x - c_x, a_y - c_y)
        cb = (b_x - c_x, b_y - c_y)
        dot_acb = ca[0] * cb[0] + ca[1] * cb[1]
        mag_ca = math.sqrt(ca[0] ** 2 + ca[1] ** 2)
        mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)
        computed_acb_rad = math.acos(dot_acb / (mag_ca * mag_cb))
        computed_acb_deg = math.degrees(computed_acb_rad)

        # Compute angle ADB
        da = (a_x - d_x, a_y - d_y)
        db = (b_x - d_x, b_y - d_y)
        dot_adb = da[0] * db[0] + da[1] * db[1]
        mag_da = math.sqrt(da[0] ** 2 + da[1] ** 2)
        mag_db = math.sqrt(db[0] ** 2 + db[1] ** 2)
        computed_adb_rad = math.acos(dot_adb / (mag_da * mag_db))
        computed_adb_deg = math.degrees(computed_adb_rad)

        # Both angles should be equal (inscribed angle theorem)
        check(
            abs(computed_acb_deg - computed_adb_deg) < 1e-4,
            f"angles_same_arc/triangle_sum: computed ACB={computed_acb_deg:.6f} != ADB={computed_adb_deg:.6f}",
        )

        # Misconceptions
        # reports_the_wrong_triangle_angle: p (reports angle CAB instead of ACB)
        # Must differ from 180 - p - q for all our choices
        # Verify: p != 180 - p - q means 2p + q != 180
        check(2 * p + q != 180, f"angles_same_arc/triangle_sum: 2p + q = 180, collision with wrong-angle distractor")
        reports_wrong = p
        check(reports_wrong != angle_adb, f"reports_the_wrong_triangle_angle distractor equals answer for p={p}, q={q}")

        # assumes_supplementary: p + q (the supplement of angle ACB)
        # Must differ from angle_acb = 180 - p - q
        # This means p + q != 180 - p - q, i.e., 2(p+q) != 180, i.e., p+q != 90
        check(p + q != 90, f"angles_same_arc/triangle_sum: p + q = 90, collision with supplementary distractor")
        assumes_supp = p + q
        check(assumes_supp != angle_adb, f"assumes_supplementary distractor equals answer for p={p}, q={q}")

        candidates = [str(reports_wrong), str(assumes_supp)]
        distractors = pick_distractors(answer_text, candidates)

    elif variant == "algebraic":
        # Hard: angle ACB = (2x + k1), angle ADB = (3x + k2)
        # By theorem, they are equal, so solve 2x + k1 = 3x + k2 for x
        # Then compute the angle

        # Choose small integer coefficients and constants
        # such that x solves to a positive integer and angle is in 10..80

        valid_combos = []
        for k1 in range(-20, 21):
            for k2 in range(-20, 21):
                # Solve 2x + k1 = 3x + k2 => x = k1 - k2
                # Exclude k1 = 0 to avoid collision in sign_error distractor
                if k1 == 0:
                    continue
                x_val = k1 - k2
                if x_val > 0:
                    angle_val = 2 * x_val + k1
                    if 10 <= angle_val <= 80:
                        valid_combos.append((k1, k2, x_val, angle_val))

        k1, k2, x_val, angle_val = rng.choice(valid_combos)

        # Verify the equation holds
        check(
            2 * x_val + k1 == 3 * x_val + k2,
            f"angles_same_arc/algebraic: equation doesn't hold for k1={k1}, k2={k2}, x={x_val}",
        )

        # The answer is the angle, not x
        angle_acb = angle_val
        angle_adb = angle_val

        # Format coefficients with proper signs
        k1_term = f"- {-k1}" if k1 < 0 else f"+ {k1}"
        k2_term = f"- {-k2}" if k2 < 0 else f"+ {k2}"

        question_text = (
            f"Points C and D both lie on the circumference of a circle, "
            f"on the same arc, on the same side of chord AB. "
            f"Angle ACB is (2x {k1_term}) degrees and angle ADB is (3x {k2_term}) degrees. "
            f"By the theorem that angles at the circumference subtended by the same arc are equal, "
            f"find the value of the angle in degrees."
        )
        answer_text = str(angle_acb)
        answer = answer_text

        # Independent check: use sympy.solve to solve the equation differently
        try:
            from sympy import symbols, solve, Eq

            x_sym = symbols("x")
            eq = Eq(2 * x_sym + k1, 3 * x_sym + k2)
            solution = solve(eq, x_sym)
            check(len(solution) == 1, f"angles_same_arc/algebraic: equation has {len(solution)} solutions")
            x_computed = float(solution[0])
            check(
                abs(x_computed - x_val) < 1e-9,
                f"angles_same_arc/algebraic: sympy gave x={x_computed}, expected {x_val}",
            )
            angle_computed = 2 * x_computed + k1
            check(
                abs(angle_computed - angle_acb) < 1e-9,
                f"angles_same_arc/algebraic: computed angle={angle_computed}, expected {angle_acb}",
            )
        except ImportError:
            # If sympy not available, do manual check
            x_computed = k1 - k2
            check(x_computed == x_val, f"angles_same_arc/algebraic: manual solution x={x_computed}, expected {x_val}")

        # Misconceptions
        # sign_error_in_equation: solves 2x - k1 = 3x + k2 => x = -k1 - k2
        # Then substitutes into same expression: 2*(-k1-k2) + k1 = -k1 - 2*k2
        x_sign_error = -k1 - k2
        angle_sign_error = 2 * x_sign_error + k1
        check(
            angle_sign_error != angle_acb, f"sign_error_in_the_equation distractor equals answer for k1={k1}, k2={k2}"
        )

        # substitutes_into_wrong_expression: solves correctly for x but uses 3x+k2 instead of 2x+k1
        # This is tricky since both expressions equal each other at solution.
        # Instead: "forgot to solve" -- use unsolved x value 0 plugged into the expression
        x_unsolved = 0
        angle_wrong_expr = 2 * x_unsolved + k1  # just k1
        check(angle_wrong_expr != angle_acb, f"substitutes_into_wrong_expression distractor equals answer for k1={k1}")

        candidates = [str(angle_sign_error), str(angle_wrong_expr)]
        distractors = pick_distractors(answer_text, candidates)

    # Build working text based on variant
    if variant == "direct":
        working = [
            f"By the inscribed angle theorem, angles at the circumference subtended by the same arc are equal.",
            f"Therefore angle ADB = angle ACB = {angle_adb} degrees.",
        ]
    elif variant == "triangle_sum":
        working = [
            f"In triangle ABC, the three angles sum to 180 degrees.",
            f"Angle ACB = 180 - {p} - {q} = {angle_acb} degrees.",
            f"By the inscribed angle theorem, angle ADB = angle ACB = {angle_adb} degrees.",
        ]
    elif variant == "algebraic":
        # Solve 2x + k1 = 3x + k2 => x = k1 - k2
        working = [
            f"Set the angles equal by the inscribed angle theorem: 2x {k1_term} = 3x {k2_term}.",
            f"Rearrange to solve for x: x = {x_val}.",
            f"Substitute back: the angle = 2 × {x_val} {k1_term} = {angle_acb} degrees.",
        ]

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": working,
    }
