"""The alternate segment theorem + suitable converses of some of the above circle theorems.

- ``alternate_segment_theorem`` (``senior-the-alternate-segment-theorem``,
  ACMSM034, domain senior-mathematics, answer: number) -- A tangent line
  touches a circle at T. A chord TP is drawn from T. The angle between the
  tangent and the chord TP, measured at T, equals the inscribed angle TQP,
  where Q is any point on the circle in the alternate segment (the arc on
  the other side of chord TP from where the tangent-chord angle was
  measured).

- ``circle_theorem_converses_classify`` (
  ``senior-suitable-converses-of-some-of-the-above-results``, ACMSM037,
  domain senior-mathematics, answer: mcq) -- Given a fully specified
  numeric configuration, classify which one (if any) of four fixed
  converse conclusions -- drawn from the four already-shipped circle
  theorems ACMSM029 (angle in a semicircle), ACMSM031 (angles subtended by
  ACMSM034 (alternate segment) -- the stated numbers actually establish.
  ACMSM033 (chords of equal length) is deliberately excluded: its own
  shipped skill already exercises its own converse internally. ACMSM036
  (tangent/secant power of a point) is excluded because it has no shipped
  forward theorem yet.
"""

from __future__ import annotations

import math

from ..core import check, check_distinct, generator, join_terms, latex_answer, pick_distractors


def _linear_expr_text(p: int, q: int, var: str = "x") -> str:
    """Render ``p*x + q`` (or ``p*n + q``) the way a student would write it,
    e.g. ``_linear_expr_text(2, 10) == "(2x + 10)"``, using the shared
    ``join_terms`` helper so a negative constant renders as a visible
    subtraction rather than ``+ -15``."""
    return f"({join_terms([(p, var), (q, '')])})"


# ============================================================================
# Skill 1: alternate_segment_theorem (ACMSM034)
# ============================================================================


def _verify_alternate_segment(theta_deg: float, q_arc_fractions: tuple = (0.25, 0.65)) -> None:
    """Independent coordinate-geometry re-derivation of the alternate segment
    theorem on a unit circle, using vector dot products -- never the
    theorem's own equality restated.

    T = (1, 0); the tangent line at T is vertical (perpendicular to radius
    OT). P is placed at azimuth ``-2 * theta_deg``. An earlier worked
    example asserted this makes the tangent-chord angle (measured from the
    UPWARD tangent ray ``(0, 1)``) equal ``theta_deg`` directly -- verifying
    that numerically (as instructed, rather than asserting it) showed it is
    actually ``180 - theta_deg`` at every seed tried; the DOWNWARD ray
    ``(0, -1)`` is the one that gives ``theta_deg`` exactly, because P sits
    on the clockwise side of T for this placement. This function uses the
    downward ray and confirms the angle numerically every call.

    Q is placed on the "far" arc from T to P (the alternate segment) at one
    or more fractions of that arc's own span (``360 - 2 * theta_deg``)
    rather than a fixed degree offset, so the check stays valid up to (but
    not including) ``theta_deg == 180`` -- a fixed offset such as "+70
    degrees" (the earlier example) collides with P once
    ``2 * theta_deg`` passes roughly 220, which the hard/algebraic tier's
    own bound of 145 degrees can reach. The inscribed-angle theorem (the
    central angle over the near arc, halved) independently confirms angle
    TQP equals ``theta_deg`` at every fraction tested, proving the check is
    not special-cased to one point.
    """
    check(0 < theta_deg < 180, "theta_deg out of range for the coordinate check")

    t = (1.0, 0.0)
    p_rad = math.radians(-2.0 * theta_deg)
    p = (math.cos(p_rad), math.sin(p_rad))

    tangent_dir = (0.0, -1.0)
    tp = (p[0] - t[0], p[1] - t[1])
    mag_tp = math.hypot(*tp)
    dot_t = tangent_dir[0] * tp[0] + tangent_dir[1] * tp[1]
    cos_t = max(-1.0, min(1.0, dot_t / mag_tp))
    tangent_chord_deg = math.degrees(math.acos(cos_t))
    check(
        abs(tangent_chord_deg - theta_deg) < 1e-6,
        f"alternate_segment: tangent-chord angle {tangent_chord_deg:.6f} != {theta_deg}",
    )

    remaining_arc = 360.0 - 2.0 * theta_deg
    for frac in q_arc_fractions:
        check(0.0 < frac < 1.0, "alternate_segment: q arc fraction must be strictly between 0 and 1")
        q_rad = math.radians(frac * remaining_arc)
        q = (math.cos(q_rad), math.sin(q_rad))
        qt = (t[0] - q[0], t[1] - q[1])
        qp = (p[0] - q[0], p[1] - q[1])
        mag_qt = math.hypot(*qt)
        mag_qp = math.hypot(*qp)
        dot_q = qt[0] * qp[0] + qt[1] * qp[1]
        cos_q = max(-1.0, min(1.0, dot_q / (mag_qt * mag_qp)))
        inscribed_deg = math.degrees(math.acos(cos_q))
        check(
            abs(inscribed_deg - theta_deg) < 1e-6,
            f"alternate_segment: inscribed angle TQP={inscribed_deg:.6f} at frac={frac} != {theta_deg}",
        )


def _verify_linear_intersection(p1: int, q1: int, p2: int, q2: int, x_expected: int, angle_expected: int) -> int:
    """Independently resolve ``p1*x + q1 = p2*x + q2`` for x via plain
    integer arithmetic (never reusing however the pool constructed
    ``x_expected``), and confirm the two expressions agree at that x and
    equal ``angle_expected``. Returns the solved angle."""
    check(p1 != p2, "alternate_segment/algebraic: coefficients must differ for a unique solution")
    numerator = q2 - q1
    denominator = p1 - p2
    check(numerator % denominator == 0, "alternate_segment/algebraic: x does not resolve to an integer")
    x_solved = numerator // denominator
    check(x_solved == x_expected, f"alternate_segment/algebraic: solved x={x_solved} != expected {x_expected}")
    check(x_solved > 0, "alternate_segment/algebraic: x must be a positive integer")
    angle_solved = p1 * x_solved + q1
    check(
        angle_solved == p2 * x_solved + q2,
        "alternate_segment/algebraic: the two expressions do not agree at the solved x",
    )
    check(
        angle_solved == angle_expected,
        f"alternate_segment/algebraic: solved angle={angle_solved} != expected {angle_expected}",
    )
    return angle_solved


def _build_alternate_segment_hard_pool():
    """Small-integer coefficient pairs for the algebraic tier, built once at
    import time (never rebuilt per call -- see check_generator_speed.py's
    250ms budget). Each entry guarantees: a unique positive-integer x, both
    linear expressions landing on the SAME angle (within [15, 145]) at that
    x, a "sign-error-in-algebra" distractor (forgets the sign when isolating
    x) that resolves to a different, still-plausible positive angle, and an
    "assumes-supplementary" distractor (180 - answer) -- all three pairwise
    distinct from each other and from the answer."""
    pool = []
    for x_true in range(3, 21):
        for p1 in range(1, 6):
            for p2 in range(1, 6):
                if p1 == p2:
                    continue
                for angle_value in range(20, 141, 5):
                    if angle_value == 90:
                        continue
                    q1 = angle_value - p1 * x_true
                    q2 = angle_value - p2 * x_true
                    if abs(q1) > 80 or abs(q2) > 80:
                        continue
                    if q1 == q2:
                        continue
                    supplementary = 180 - angle_value

                    # "sign-error-in-algebra": isolates x with the numerator
                    # sign flipped (q1 + q2 instead of q2 - q1).
                    alt_den = p1 - p2
                    alt_num = q1 + q2
                    if alt_den == 0 or alt_num % alt_den != 0:
                        continue
                    alt_x = alt_num // alt_den
                    if alt_x == x_true:
                        continue
                    alt_angle = p1 * alt_x + q1
                    if not (5 <= alt_angle <= 200):
                        continue
                    if alt_angle in (angle_value, supplementary):
                        continue
                    if x_true in (angle_value, alt_angle, supplementary):
                        continue

                    pool.append((p1, q1, p2, q2, x_true, angle_value, alt_angle, supplementary))
    return pool


_ALTERNATE_SEGMENT_HARD_POOL = _build_alternate_segment_hard_pool()


@generator(
    "alternate_segment_theorem",
    level=12,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="composed"),
        "hard": dict(variant="algebraic"),
    },
)
def alternate_segment_theorem(rng, variant="direct"):
    """The angle between a tangent and a chord drawn from the point of
    tangency equals the inscribed angle subtended by that chord in the
    alternate segment (the alternate segment theorem)."""

    if variant == "direct":
        # Easy: the tangent-chord angle is stated directly, bounded well
        # clear of both 0 and 90 degrees. theta_deg == 45 is also excluded:
        # 90 - 45 == 45, which would make the "assumes-complementary"
        # distractor collide with the answer itself.
        while True:
            theta_deg = rng.randint(20, 80)
            if theta_deg != 45:
                break

        question_text = (
            "A tangent line touches a circle at point T. A chord TP is drawn from T. "
            f"The angle between the tangent and the chord TP, measured at T, is {theta_deg} degrees. "
            "Point Q lies on the circle in the alternate segment (the arc on the other side of "
            "chord TP from where the angle was measured). Find angle TQP, in degrees."
        )

        _verify_alternate_segment(theta_deg)

        supplementary = 180 - theta_deg
        complementary = 90 - theta_deg
        check_distinct(
            theta_deg,
            supplementary,
            complementary,
            msg=f"alternate_segment_theorem/direct: distractor collides for theta={theta_deg}",
        )

        candidates = [str(supplementary), str(complementary)]
        working = [
            "By the alternate segment theorem, the tangent-chord angle at T equals the inscribed "
            "angle in the alternate segment.",
            f"Angle TQP = {theta_deg} degrees.",
        ]
        answer = theta_deg

    elif variant == "composed":
        # Medium: the triangle's own two OTHER interior angles (at T and P,
        # not the tangent-chord angle) are given; the learner sums then
        # subtracts from 180 to get angle TQP, then applies the theorem to
        # get the tangent-chord angle asked for. Every value -- a, b and the
        # resolved answer -- stays inside [15, 145].
        while True:
            theta_deg = rng.randint(15, 145)
            if theta_deg == 90:
                continue
            total = 180 - theta_deg
            lo = max(15, total - 145)
            hi = min(145, total - 15)
            if hi < lo:
                continue
            a = rng.randint(lo, hi)
            b = total - a
            if not (15 <= b <= 145):
                continue
            if a == b or a == theta_deg or b == theta_deg or total == theta_deg:
                continue
            break

        question_text = (
            "A tangent line touches a circle at point T. A chord TP is drawn from T, and a point Q "
            "lies on the circle in the alternate segment (the arc on the other side of chord TP from "
            "the tangent-chord angle). In triangle TPQ, angle QTP is "
            f"{a} degrees and angle TPQ is {b} degrees. Find the angle between the tangent and the "
            "chord TP, measured at T, in degrees."
        )

        _verify_alternate_segment(theta_deg)
        check(a + b == total, "alternate_segment_theorem/composed: angle sum bookkeeping error")
        check_distinct(
            theta_deg, a, b, total, msg=f"alternate_segment_theorem/composed: distractor collides for a={a}, b={b}"
        )

        candidates = [str(a), str(b), str(total)]
        working = [
            f"By the triangle angle sum, angle TQP = 180 - {a} - {b} = {theta_deg} degrees.",
            "By the alternate segment theorem, the tangent-chord angle at T equals angle TQP.",
            f"So the tangent-chord angle = {theta_deg} degrees.",
        ]
        answer = theta_deg

    elif variant == "algebraic":
        # Hard: the tangent-chord angle and the alternate-segment angle are
        # each a different linear expression in x; the theorem sets them
        # equal, so the learner solves for x and evaluates.
        p1, q1, p2, q2, x_true, angle_value, alt_angle, supplementary = rng.choice(_ALTERNATE_SEGMENT_HARD_POOL)

        expr1 = _linear_expr_text(p1, q1)
        expr2 = _linear_expr_text(p2, q2)

        question_text = (
            "A tangent line touches a circle at point T. A chord TP is drawn from T. The angle "
            f"between the tangent and the chord TP, measured at T, is {expr1} degrees. Point Q lies "
            "on the circle in the alternate segment, and angle TQP is "
            f"{expr2} degrees, where x is a positive integer. By the alternate segment theorem these "
            "two angles are equal. Find the resulting angle, in degrees."
        )

        angle_solved = _verify_linear_intersection(p1, q1, p2, q2, x_true, angle_value)
        _verify_alternate_segment(angle_solved)
        check_distinct(
            angle_value,
            alt_angle,
            supplementary,
            msg=f"alternate_segment_theorem/algebraic: distractor collides for x={x_true}",
        )

        candidates = [str(alt_angle), str(supplementary)]
        working = [
            f"Setting the two expressions equal: {expr1} = {expr2}.",
            f"Solving gives x = {x_true}.",
            f"Substituting back: angle = {join_terms([(p1, f'({x_true})' if x_true < 0 else str(x_true)), (q1, '')])} = {angle_value} degrees.",
        ]
        answer = angle_value

    else:
        raise ValueError(f"unknown variant {variant!r}")

    answer_text = str(answer)
    distractors = pick_distractors(answer_text, candidates)

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "working": working,
    }


# ============================================================================
# Skill 2: circle_theorem_converses_classify (ACMSM037)
# ============================================================================

# Fixed MCQ options, verbatim, every seed.
OPTION_DIAMETER = "The stated side is a diameter of the circle (converse of: an angle in a semicircle is a right angle)"
OPTION_CONCYCLIC = "The four points lie on a common circle (converse of: angles subtended by the same arc are equal)"
OPTION_CYCLIC_QUAD = "The quadrilateral is a cyclic quadrilateral (converse of: opposite angles of a cyclic quadrilateral are supplementary)"
OPTION_TANGENT = "The line is a tangent to the circle at that point (converse of: the alternate segment theorem)"
OPTION_NONE = "None of these conclusions can be established from the information given"
ALL_CONVERSE_OPTIONS = (OPTION_DIAMETER, OPTION_CONCYCLIC, OPTION_CYCLIC_QUAD, OPTION_TANGENT, OPTION_NONE)


def _verify_classification(satisfied: bool, label: str) -> None:
    """Confirm the construct-to-target draw actually landed on the class it
    was decided to realise -- the boolean was chosen first, by ``rng``,
    before any number was built."""
    if satisfied:
        check(
            label != OPTION_NONE,
            "circle_theorem_converses_classify: constructed to satisfy the hypothesis but got OPTION_NONE",
        )
    else:
        check(
            label == OPTION_NONE,
            "circle_theorem_converses_classify: constructed to violate the hypothesis but got a named option",
        )


# ---- Family A: semicircle converse (ACMSM029) ------------------------------

_FAMILY_A_NOT_SATISFIED_POOL = tuple(t for t in range(60, 121) if t != 90)


def _make_family_a(rng, satisfied: bool):
    # OPTION_DIAMETER's own wording is "the stated side is a diameter", so
    # the question must actually name a side for that phrase to have a
    # referent -- side XZ, opposite the stated angle XYZ.
    theta = 90 if satisfied else rng.choice(_FAMILY_A_NOT_SATISFIED_POOL)
    text = (
        f"Triangle XYZ has all three vertices on a circle, with side XZ opposite angle XYZ. "
        f"Angle XYZ = {theta} degrees. Which of these conclusions follows?"
    )
    return text, theta


def _classify_family_a(theta: int) -> str:
    return OPTION_DIAMETER if theta == 90 else OPTION_NONE


# ---- Family B: concyclic converse (ACMSM031) -------------------------------


def _make_family_b(rng, satisfied: bool):
    if satisfied:
        alpha = rng.randint(20, 140)
        beta = alpha
    else:
        while True:
            alpha = rng.randint(20, 140)
            beta = rng.randint(20, 140)
            if abs(alpha - beta) >= 5:
                break
    text = (
        "Points A and B are fixed. Points C and D lie on the same side of line AB. "
        f"Angle ACB = {alpha} degrees and angle ADB = {beta} degrees. Which of these conclusions follows?"
    )
    return text, alpha, beta


def _classify_family_b(alpha: int, beta: int) -> str:
    return OPTION_CONCYCLIC if alpha == beta else OPTION_NONE


# ---- Family C: cyclic-quadrilateral converse (ACMSM032) --------------------


def _make_family_c(rng, satisfied: bool):
    if satisfied:
        while True:
            a = rng.randint(35, 145)
            b = 180 - a
            if 30 <= b <= 145:
                break
    else:
        while True:
            a = rng.randint(30, 145)
            b = rng.randint(30, 145)
            if abs(a + b - 180) >= 6:
                break
    text = (
        f"Quadrilateral PQRS has angle P = {a} degrees and angle R = {b} degrees. P and R are "
        "opposite angles of the quadrilateral. Which of these conclusions follows?"
    )
    return text, a, b


def _classify_family_c(a: int, b: int) -> str:
    return OPTION_CYCLIC_QUAD if a + b == 180 else OPTION_NONE


# ---- Family D: alternate-segment/tangent converse (ACMSM034), hard only ---


def _build_family_d_pools():
    """Precomputed pools (satisfied / not-satisfied) of small-integer
    coefficients for the hard tier's linear-expression form, built once at
    import time. n is solved from x's own stated expression and numeric
    value; y is evaluated at that n and compared against x."""
    satisfied_pool = []
    not_satisfied_pool = []
    for n in range(3, 16):
        for p1 in range(1, 6):
            for q1 in range(-30, 41, 5):
                x_value = p1 * n + q1
                if not (15 <= x_value <= 140):
                    continue
                for p2 in range(1, 6):
                    if p2 == p1:
                        continue
                    q2_sat = x_value - p2 * n
                    if -60 <= q2_sat <= 80:
                        satisfied_pool.append((n, p1, q1, p2, q2_sat, x_value, x_value))
                    for delta in (-12, -8, -5, 5, 8, 12):
                        y_value = x_value + delta
                        if not (15 <= y_value <= 140):
                            continue
                        q2_not = y_value - p2 * n
                        if -60 <= q2_not <= 80:
                            not_satisfied_pool.append((n, p1, q1, p2, q2_not, x_value, y_value))
    return satisfied_pool, not_satisfied_pool


_FAMILY_D_SATISFIED_POOL, _FAMILY_D_NOT_SATISFIED_POOL = _build_family_d_pools()


def _make_family_d(rng, satisfied: bool):
    pool = _FAMILY_D_SATISFIED_POOL if satisfied else _FAMILY_D_NOT_SATISFIED_POOL
    n, p1, q1, p2, q2, x_value, y_value = rng.choice(pool)
    expr1 = _linear_expr_text(p1, q1, var="n")
    expr2 = _linear_expr_text(p2, q2, var="n")
    text = (
        "A line meets a circle at point T. A chord TP is drawn. The angle between the line and "
        f"TP, measured at T, is {expr1} degrees, where n is a positive integer, and this angle is "
        f"known to be {x_value} degrees. A point Q lies on the circle in the alternate segment, and "
        f"angle TQP = {expr2} degrees. Which of these conclusions follows?"
    )
    return text, n, p1, q1, x_value, p2, q2, y_value


def _verify_family_d(n: int, p1: int, q1: int, x_value: int, p2: int, q2: int, y_value: int) -> None:
    """Independently re-solve n from x's own expression and numeric value
    (never reusing construction's own n), then re-evaluate y at that n."""
    check((x_value - q1) % p1 == 0, "family_d: n does not resolve to an integer from x's own equation")
    n_solved = (x_value - q1) // p1
    check(n_solved == n, f"family_d: solved n={n_solved} != constructed n={n}")
    check(n_solved > 0, "family_d: n must be a positive integer")
    y_solved = p2 * n_solved + q2
    check(y_solved == y_value, f"family_d: solved y={y_solved} != constructed y={y_value}")


def _classify_family_d(x_value: int, y_value: int) -> str:
    return OPTION_TANGENT if x_value == y_value else OPTION_NONE


@generator(
    "circle_theorem_converses_classify",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def circle_theorem_converses_classify(rng, tier="easy"):
    """Classify a fully specified circle configuration by which one of four
    fixed converse conclusions (drawn from four already-shipped circle
    theorems) the stated numbers actually establish, or that none can be
    established. The satisfied/not-satisfied class is always decided by
    ``rng`` FIRST, then the numbers are built to realise it."""
    satisfied = rng.choice([True, False])

    if tier == "easy":
        text, theta = _make_family_a(rng, satisfied)
        label = _classify_family_a(theta)

    elif tier == "medium":
        family = rng.choice(["B", "C"])
        if family == "B":
            text, alpha, beta = _make_family_b(rng, satisfied)
            label = _classify_family_b(alpha, beta)
        else:
            text, a, b = _make_family_c(rng, satisfied)
            label = _classify_family_c(a, b)

    elif tier == "hard":
        text, n, p1, q1, x_value, p2, q2, y_value = _make_family_d(rng, satisfied)
        _verify_family_d(n, p1, q1, x_value, p2, q2, y_value)
        label = _classify_family_d(x_value, y_value)

    else:
        raise ValueError(f"unknown tier {tier!r}")

    _verify_classification(satisfied, label)

    answer = label
    distractors = [opt for opt in ALL_CONVERSE_OPTIONS if opt != answer]

    working = [f"Checking whether the stated hypothesis is satisfied: {answer}"]

    return {
        "text": text,
        "latex": f"plain:{text}",
        "answer": answer,
        "answer_text": answer,
        "answer_latex": latex_answer(answer, plain=True),
        "distractors": distractors,
        "working": working,
    }
