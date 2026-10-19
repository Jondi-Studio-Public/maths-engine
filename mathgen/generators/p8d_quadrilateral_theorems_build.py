"""Generates `the_opposite_angles_of_a_cyclic_quadrilateral_are` and
`the_diagonals_of_a_parallelogram_meet_at_right` questions on quadrilateral theorems.

Quadrilateral theorems.

- ``the_opposite_angles_of_a_cyclic_quadrilateral_are``
  (``senior-the-opposite-angles-of-a-cyclic-quadrilateral-are``, ACMSM032,
  domain senior-mathematics, answer: number) -- The opposite angles of a
  cyclic quadrilateral are supplementary.

- ``the_diagonals_of_a_parallelogram_meet_at_right``
  (``senior-the-diagonals-of-a-parallelogram-meet-at-right``, ACMSM039,
  domain senior-mathematics, answer: number) -- The diagonals of a
  parallelogram meet at right angles if and only if it is a rhombus.

Every question is computational (a single checkable numeric value); none
asks the learner to prove anything.
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Eq, solve, symbols

from ..core import check, check_distinct, check_equivalent, generator, join_terms, pick_distractors

_X = symbols("x")


def _expr_text(coef: int, const: int) -> str:
    """``2x + 10`` / ``3x - 5`` style text for a linear expression in x."""
    return join_terms([(coef, "x"), (const, "")])


def _bx(coef: int, const: int) -> str:
    """`_expr_text` in brackets, except a single term ("3x"), which needs none."""
    text = _expr_text(coef, const)
    return f"({text})" if const else text


def _deg_word(n: int) -> str:
    """``degree`` for a magnitude of exactly 1, ``degrees`` otherwise."""
    return "degree" if abs(n) == 1 else "degrees"


# ============================================================================
# Skill 1: the_opposite_angles_of_a_cyclic_quadrilateral_are (ACMSM032)
# ============================================================================


@generator(
    "the_opposite_angles_of_a_cyclic_quadrilateral_are",
    level=12,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="linear_pair"),
        "hard": dict(variant="two_pair_chain"),
    },
)
def the_opposite_angles_of_a_cyclic_quadrilateral_are(rng, variant="direct"):
    """Opposite angles of a cyclic quadrilateral ABCD are supplementary:
    angle A + angle C = 180, angle B + angle D = 180."""

    if variant == "direct":
        while True:
            a = rng.randint(21, 159)
            if a != 90:
                break
        c = 180 - a

        question_text = (
            f"ABCD is a cyclic quadrilateral (a quadrilateral with all four vertices "
            f"on a circle). Angle A and angle C are opposite angles of the "
            f"quadrilateral. Angle A is {a} degrees. Find angle C, in degrees."
        )
        answer_text = str(c)

        # Independent check: solve the pair-supplementary equation with sympy
        # instead of the direct subtraction construction used.
        c_sym = symbols("c")
        sol = solve(Eq(a + c_sym, 180), c_sym)
        check(len(sol) == 1, "the_opposite_angles.../direct: sympy solve did not return one root")
        check_equivalent(sol[0], c, "the_opposite_angles.../direct: sympy solve disagrees with construction")

        d_360 = 360 - a
        d_identity = a
        check_distinct(c, d_360, msg="uses-360-instead-of-180 distractor equals the answer")
        check_distinct(c, d_identity, msg="reports-non-requested-angle distractor equals the answer")

        distractors = pick_distractors(answer_text, [str(d_360), str(d_identity)])

        working = [
            "Opposite angles of a cyclic quadrilateral are supplementary: angle A + angle C = 180 degrees.",
            f"angle C = 180 - {a} = {c} degrees.",
        ]

    elif variant == "linear_pair":
        divisors = [2, 3, 4, 5, 6, 9, 10]
        total_coef = rng.choice(divisors)
        a1 = rng.randint(1, total_coef - 1)
        c1 = total_coef - a1

        # Plausibility bound: every angle a real cyclic quadrilateral could show must stay well
        # clear of a degenerate sliver. 30-150 keeps both A and C (and, since C = 180 - A's kin
        # here, both members of the pair) comfortably within a figure a learner could believe --
        # found necessary... < 180 bound.
        found = False
        for _ in range(1000):
            a2 = rng.randint(-15, 15)
            c2 = rng.randint(-15, 15)
            total_const = a2 + c2
            remaining = 180 - total_const
            if remaining <= 0 or remaining % total_coef != 0:
                continue
            x = remaining // total_coef
            if x <= 0:
                continue
            a_val = a1 * x + a2
            c_val = c1 * x + c2
            if not (30 <= a_val <= 150 and 30 <= c_val <= 150):
                continue
            if a_val == c_val:
                continue
            found = True
            break
        check(found, "the_opposite_angles.../linear_pair: no valid combo found")

        ask_target = rng.choice(["A", "C"])
        answer_val = a_val if ask_target == "A" else c_val

        question_text = (
            f"ABCD is a cyclic quadrilateral. Angle A and angle C are opposite "
            f"angles of the quadrilateral. Angle A = {_bx(a1, a2)} degrees "
            f"and angle C = {_bx(c1, c2)} degrees. "
            f"Find angle {ask_target}, in degrees."
        )
        answer_text = str(answer_val)

        # Independent check: solve the linear equation with sympy.solve rather
        # than the direct-division construction used above.
        x_sym = symbols("x")
        sol = solve(Eq((a1 * x_sym + a2) + (c1 * x_sym + c2), 180), x_sym)
        check(len(sol) == 1, "the_opposite_angles.../linear_pair: sympy solve did not return one root")
        x_check = sol[0]
        check_equivalent(x_check, x, "the_opposite_angles.../linear_pair: sympy x disagrees")
        a_check = a1 * x_check + a2
        c_check = c1 * x_check + c2
        check_equivalent(a_check, a_val, "the_opposite_angles.../linear_pair: sympy angle A disagrees")
        check_equivalent(c_check, c_val, "the_opposite_angles.../linear_pair: sympy angle C disagrees")

        # uses-360-instead-of-180: solving the same equation with 360 on the RHS.
        # Since total_coef divides 180 exactly (it was drawn from `divisors`),
        # x_360 = x + 180 // total_coef is an exact integer.
        x_360 = x + 180 // total_coef
        val_360 = (a1 * x_360 + a2) if ask_target == "A" else (c1 * x_360 + c2)
        # reports-non-requested-angle: reports the OTHER given angle's value.
        other_val = c_val if ask_target == "A" else a_val

        check_distinct(answer_val, val_360, msg="uses-360-instead-of-180 distractor equals the answer")
        check_distinct(answer_val, other_val, msg="reports-non-requested-angle distractor equals the answer")

        distractors = pick_distractors(answer_text, [str(val_360), str(other_val)])

        working = [
            "Opposite angles of a cyclic quadrilateral are supplementary: angle A + angle C = 180 degrees.",
            f"{_bx(a1, a2)} + {_bx(c1, c2)} = 180",
            f"x = {x}",
            f"angle {ask_target} = {answer_val} {_deg_word(answer_val)}.",
        ]

    elif variant == "two_pair_chain":
        # Plausibility bound, same reasoning as `linear_pair` above: with
        # A = f(x), B = g(x) and "C exceeds D by k" reducing to B - A = k,
        # nothing previously stopped A/B (hence C/D) landing on a sliver
        # found 47/60 hard-tier seeds with an angle under 20 degrees, one as
        # low as 1. 30-150 on every one of the four angles keeps every drawn
        # figure a learner could believe is a real cyclic quadrilateral.
        found = False
        for _ in range(1000):
            a1 = rng.randint(1, 4)
            b1 = rng.randint(1, 4)
            if a1 == b1:
                continue
            denom = b1 - a1
            diff_mult = rng.randint(-6, 6)
            a2 = rng.randint(-15, 15)
            b2 = a2 - denom * diff_mult
            if not (-30 <= b2 <= 30):
                continue
            k_mult = rng.randint(1, 6)
            k = abs(denom) * k_mult
            x0 = diff_mult
            sign_denom = 1 if denom > 0 else -1
            x = sign_denom * k_mult + diff_mult
            if x <= 0:
                continue
            a_val = a1 * x + a2
            b_val = b1 * x + b2
            c_val = 180 - a_val
            d_val = 180 - b_val
            if not (30 <= a_val <= 150 and 30 <= b_val <= 150 and 30 <= c_val <= 150 and 30 <= d_val <= 150):
                continue
            if a_val == 90 or b_val == 90:
                continue
            a_val0 = a1 * x0 + a2
            b_val0 = b1 * x0 + b2
            c_val0 = 180 - a_val0
            d_val0 = 180 - b_val0
            found = True
            break
        check(found, "the_opposite_angles.../two_pair_chain: no valid combo found")

        check(
            a_val + b_val + c_val + d_val == 360, "the_opposite_angles.../two_pair_chain: four angles do not sum to 360"
        )
        check(c_val - d_val == k, "the_opposite_angles.../two_pair_chain: C - D != k by construction")

        partner = {"A": "C", "C": "A", "B": "D", "D": "B"}
        cross_map = {"A": "B", "B": "A", "C": "D", "D": "C"}
        angles = {"A": a_val, "B": b_val, "C": c_val, "D": d_val}
        angles0 = {"A": a_val0, "B": b_val0, "C": c_val0, "D": d_val0}

        order = ["A", "B", "C", "D"]
        rng.shuffle(order)
        asked = None
        for cand in order:
            same_pair_val = angles[partner[cand]]
            cross_val = angles0[cross_map[cand]]
            if same_pair_val != angles[cand] and cross_val != angles[cand]:
                asked = cand
                break
        check(
            asked is not None,
            "the_opposite_angles.../two_pair_chain: every asked-angle choice collides with a distractor",
        )

        answer_val = angles[asked]
        same_pair_val = angles[partner[asked]]
        cross_val = angles0[cross_map[asked]]

        k_word = "degree" if k == 1 else "degrees"
        question_text = (
            f"ABCD is a cyclic quadrilateral. Angle A and angle C are opposite "
            f"angles; angle B and angle D are opposite angles. "
            f"Angle A = {_bx(a1, a2)} degrees and angle B = "
            f"{_bx(b1, b2)} degrees. Angle C exceeds angle D by {k} {k_word}. "
            f"Find angle {asked}, in degrees."
        )
        answer_text = str(answer_val)

        # Independent check: re-derive all four angles from scratch with
        # sympy.solve over the full system, rather than construction's own
        # closed-form substitution order.
        x_sym = symbols("x")
        eq = Eq((b1 * x_sym + b2) - (a1 * x_sym + a2), k)
        sol = solve(eq, x_sym)
        check(len(sol) == 1, "the_opposite_angles.../two_pair_chain: sympy solve did not return one root")
        x_check = sol[0]
        check_equivalent(x_check, x, "the_opposite_angles.../two_pair_chain: sympy x disagrees")
        a_check = a1 * x_check + a2
        b_check = b1 * x_check + b2
        c_check = 180 - a_check
        d_check = 180 - b_check
        check_equivalent(a_check, a_val, "the_opposite_angles.../two_pair_chain: sympy angle A disagrees")
        check_equivalent(b_check, b_val, "the_opposite_angles.../two_pair_chain: sympy angle B disagrees")
        check_equivalent(c_check, c_val, "the_opposite_angles.../two_pair_chain: sympy angle C disagrees")
        check_equivalent(d_check, d_val, "the_opposite_angles.../two_pair_chain: sympy angle D disagrees")

        check_distinct(answer_val, same_pair_val, msg="reports-non-requested-angle distractor equals the answer")
        check_distinct(answer_val, cross_val, msg="drops-difference-relationship distractor equals the answer")

        distractors = pick_distractors(answer_text, [str(same_pair_val), str(cross_val)])

        working = [
            "Opposite angles of a cyclic quadrilateral are supplementary: "
            "angle A + angle C = 180 and angle B + angle D = 180.",
            f"angle C = 180 - {_bx(a1, a2)}, angle D = 180 - {_bx(b1, b2)}.",
            f"Angle C exceeds angle D by {k}: (180 - angle A) - (180 - angle B) = {k}, so angle B - angle A = {k}.",
            f"x = {x}",
            f"angle {asked} = {answer_val} {_deg_word(answer_val)}.",
        ]

    else:
        raise ValueError(f"unknown variant {variant!r}")

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer_text,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": working,
    }


# ============================================================================
# Skill 2: the_diagonals_of_a_parallelogram_meet_at_right (ACMSM039)
# ============================================================================

_TRIPLES = [
    (3, 4, 5),
    (6, 8, 10),
    (5, 12, 13),
    (8, 15, 17),
    (7, 24, 25),
    (9, 12, 15),
    (12, 16, 20),
    (20, 21, 29),
    (9, 40, 41),
]


@generator(
    "the_diagonals_of_a_parallelogram_meet_at_right",
    level=12,
    presets={
        "easy": dict(variant="diagonals_given"),
        "medium": dict(variant="one_diagonal_given"),
        "hard": dict(variant="converse_rhombus"),
    },
)
def the_diagonals_of_a_parallelogram_meet_at_right(rng, variant="diagonals_given"):
    """The diagonals of a parallelogram meet at right angles iff it is a rhombus."""

    if variant == "diagonals_given":
        while True:
            leg1, leg2, hyp = rng.choice(_TRIPLES)
            scale = rng.randint(1, 3)
            half_p, half_q, side = leg1 * scale, leg2 * scale, hyp * scale
            if rng.random() < 0.5:
                half_p, half_q = half_q, half_p
            p, q = 2 * half_p, 2 * half_q
            if p == side or q == side:
                continue
            break

        check(half_p**2 + half_q**2 == side**2, "the_diagonals.../diagonals_given: triple does not satisfy Pythagoras")

        question_text = (
            f"A rhombus has diagonals that bisect each other at right angles "
            f"(a property of every rhombus). The diagonals have lengths {p} cm "
            f"and {q} cm. Find the side length of the rhombus, in cm."
        )
        answer_text = str(side)

        # Independent check: recompute from p, q directly with exact Fraction
        # arithmetic (never float ==), rather than reusing the half_p/half_q/
        # side variables construction already derived them from.
        hp_check = Fraction(p, 2)
        hq_check = Fraction(q, 2)
        side_sq_check = hp_check**2 + hq_check**2
        check(
            side_sq_check == Fraction(side) ** 2,
            "the_diagonals.../diagonals_given: independent Pythagoras check failed",
        )

        d_sum_of_squares = 2 * side  # uses full diagonals as legs, forgets to halve
        d_reports_given = p  # reports a given diagonal instead of the side
        check_distinct(side, d_sum_of_squares, msg="sum-of-squares-of-diagonals-confusion distractor equals the answer")
        check_distinct(side, d_reports_given, msg="reports-given-diagonal-as-answer distractor equals the answer")

        distractors = pick_distractors(answer_text, [str(d_sum_of_squares), str(d_reports_given)])

        working = [
            "The diagonals bisect each other at right angles, forming a right "
            "triangle with legs half of each diagonal and hypotenuse the side.",
            f"side = sqrt(({p}/2)^2 + ({q}/2)^2) = sqrt({half_p}^2 + {half_q}^2) = {side} cm.",
        ]

    elif variant == "one_diagonal_given":
        while True:
            leg1, leg2, hyp = rng.choice(_TRIPLES)
            scale = rng.randint(1, 3)
            half_p, half_q, side = leg1 * scale, leg2 * scale, hyp * scale
            if rng.random() < 0.5:
                half_p, half_q = half_q, half_p
            p = 2 * half_p
            q_correct = 2 * half_q
            if p == side or q_correct == side or p == q_correct:
                continue
            break

        check(
            half_p**2 + half_q**2 == side**2, "the_diagonals.../one_diagonal_given: triple does not satisfy Pythagoras"
        )

        question_text = (
            f"A rhombus has diagonals that bisect each other at right angles "
            f"(a property of every rhombus). One diagonal is {p} cm and the "
            f"side length is {side} cm. Find the length of the other diagonal, in cm."
        )
        answer_text = str(q_correct)

        # Independent check: recompute from p, side directly with exact
        # Fraction arithmetic, solving for the leg rather than reusing
        # construction's own half_q.
        half_p_check = Fraction(p, 2)
        half_q_sq_check = Fraction(side) ** 2 - half_p_check**2
        check(
            half_q_sq_check > 0, "the_diagonals.../one_diagonal_given: side is not greater than half the given diagonal"
        )
        q_sq_check = half_q_sq_check * 4
        check(
            q_sq_check == Fraction(q_correct) ** 2,
            "the_diagonals.../one_diagonal_given: independent Pythagoras check failed",
        )

        d_omits_doubling = half_q  # forgets to double the found half-diagonal
        d_reports_given = p  # reports the given diagonal instead of the other one
        check_distinct(q_correct, d_omits_doubling, msg="omits-doubling-the-half-diagonal distractor equals the answer")
        check_distinct(q_correct, d_reports_given, msg="reports-given-diagonal-as-answer distractor equals the answer")

        distractors = pick_distractors(answer_text, [str(d_omits_doubling), str(d_reports_given)])

        working = [
            "The diagonals bisect each other at right angles, forming a right "
            "triangle with legs half of each diagonal and hypotenuse the side.",
            f"({p}/2)^2 + (q/2)^2 = {side}^2, so (q/2)^2 = {side}^2 - {half_p}^2 = {half_q}^2.",
            f"q/2 = {half_q}, so q = {q_correct} cm.",
        ]

    elif variant == "converse_rhombus":
        found = False
        for _ in range(500):
            a1 = rng.randint(1, 6)
            b1 = rng.randint(1, 6)
            if a1 == b1:
                continue
            denom = a1 - b1
            x = rng.randint(1, 10)
            a2 = rng.randint(-20, 20)
            b2 = a2 + denom * x
            if not (-30 <= b2 <= 30):
                continue
            side = a1 * x + a2
            if side <= 0:
                continue
            denom_w = a1 + b1
            numer_w = -b2 - a2
            if numer_w % denom_w != 0:
                continue
            x_wrong = numer_w // denom_w
            side_wrong = a1 * x_wrong + a2
            if side_wrong == side or x_wrong == x or x == side:
                continue
            found = True
            break
        check(found, "the_diagonals.../converse_rhombus: no valid combo found")
        check(a1 * x + a2 == b1 * x + b2, "the_diagonals.../converse_rhombus: AB != BC at the solved x")

        question_text = (
            f"A parallelogram ABCD has adjacent sides AB = {_bx(a1, a2)} cm "
            f"and BC = {_bx(b1, b2)} cm. Its diagonals are given to meet at "
            f"right angles. By the theorem that a parallelogram's diagonals meet at "
            f"right angles if and only if it is a rhombus, all four sides are equal. "
            f"Find the length of side AB, in cm."
        )
        answer_text = str(side)

        # Independent check: re-solve AB = BC with sympy.solve rather than
        # reusing construction's own closed-form division.
        x_sym = symbols("x")
        sol = solve(Eq(a1 * x_sym + a2, b1 * x_sym + b2), x_sym)
        check(len(sol) == 1, "the_diagonals.../converse_rhombus: sympy solve did not return one root")
        x_check = sol[0]
        check_equivalent(x_check, x, "the_diagonals.../converse_rhombus: sympy x disagrees")
        side_check = a1 * x_check + a2
        check_equivalent(side_check, side, "the_diagonals.../converse_rhombus: sympy side disagrees")

        check_distinct(side, side_wrong, msg="wrong-sign-equation distractor equals the answer")
        check_distinct(side, x, msg="reports-x-not-side distractor equals the answer")

        distractors = pick_distractors(answer_text, [str(side_wrong), str(x)])

        working = [
            "Perpendicular diagonals imply the parallelogram is a rhombus, so all four sides are equal: AB = BC.",
            f"{_bx(a1, a2)} = {_bx(b1, b2)}",
            f"x = {x}",
            f"AB = {side} cm.",
        ]

    else:
        raise ValueError(f"unknown variant {variant!r}")

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer_text,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": working,
    }
