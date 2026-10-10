"""The parallelogram diagonal law (ACMSM041, ``senior-the-sum-of-the-squares-of-the-lengths``).

For a parallelogram with side lengths ``a``, ``b`` and diagonal lengths ``d1``, ``d2``: ``d1^2 + d2^2 = 2(a^2
+ b^2)``. Every question states the parallelogram fully in text -- side lengths and which segments are the
diagonals -- never relying on an implied figure.

So every quadruple this generator can draw is tabulated ONCE at import time (never solved on the fly per seed
-- this ``make`` is invoked ~1000 times across the shared suite sweeps) from a small set of base integer
quadruples found by brute-force search over ``1 <= a <= b <= 20`` (testing, for each candidate diagonal
``d1``, whether ``2(a^2+b^2) - d1^2`` is a perfect square), widened by scaling each base quadruple by integer
factors ``k = 1, 2, 3`` -- and filtered so EVERY quadruple in the table is plausible, not merely
constructible:

- the angle ``theta`` between sides ``a`` and ``b``, derived from the law of
  cosines (``d1^2 = a^2 + b^2 + 2ab*cos(theta)``, so
  ``cos(theta) = (d1^2 - a^2 - b^2) / (2ab)``), must sit in ``[20, 160]``
- neither diagonal may be within half the longer side's length of it
  (``min(d1, d2) > 0.5 * max(a, b)``) -- a thin/degenerate-looking
  parallelogram survives even inside the angle bound; the measured minimum
  ratio is ~0.517, at the angle bound's own
  edge, which makes 0.5 a real, checked threshold rather than a number copied
  without justifying it against this table.

Difficulty is a genuine reasoning escalation, not span or magnitude:

- **easy**: given both sides and ONE diagonal, find the other diagonal
  (direct application of the law) -- drawn from the pre-tabulated
  exact-integer quadruples, so the unknown diagonal is always a whole
  number, never a surd needing rounding.
- **medium**: given both diagonals and ONE side, find the other side -- a
  genuinely different unknown position in the formula (rearranged for a
  side, not a diagonal) from easy's direct diagonal-for-diagonal
  application.
- **hard**: a word-problem context that genuinely exercises the stated
  ``convert-metric-length`` prerequisite -- one length is stated in
  different units from the rest of the problem (one side in metres while
  the rest of the problem is in centimetres, or the asked diagonal in
  millimetres), so the learner must convert before or after applying the
  law. The underlying quadruple is always exact-integer in a common base
  unit (centimetres), and every conversion is by a power of ten, so the
  conversion itself never introduces a rounding decision on top of the
  geometry.

Independent check, every tier: the STATED side/diagonal values (after unit
conversion to a common base for the hard tier) are re-squared and summed
directly (``given^2 + unknown^2 == 2*(side1^2+side2^2)``), independent of
which unknown the construction actually solved for -- this verifies the
full identity holds for the whole quadruple, not merely that the one
rearranged formula construction used was applied correctly.

Misconceptions:

- ``uses-linear-law``: applies ``d1+d2 = 2(a+b)`` (linear, not squared)
  instead of the real law -- ALWAYS exercisable (the table is filtered so
  this distractor is well-defined, positive and distinct from the true
  answer in every direction the generator draws it in; see
  ``_validate_row``).
- ``forgets-factor-of-two``: applies ``d1^2+d2^2 = a^2+b^2`` (drops the
  factor of 2) when solving for the unknown diagonal. Only well-defined when
  the unknown diagonal is the LARGER of the two true diagonals (otherwise
  the discriminant is non-positive by construction -- see
``_sumsq_diagonal_wrong``'s docstring) -- opportunistic
  precedent for a misconception that is not always exercisable.
- ``uses-sum-then-squares``: computes ``(d1+d2)^2`` instead of
  ``d1^2+d2^2`` when solving for the unknown side. Opportunistic for the
  same discriminant-sign reason (``_sumsq_side_wrong``).
- ``forgets-unit-conversion`` (hard tier only): uses a stated length as if
  it were already in the target unit, or reports the answer in the wrong
  unit's numeral -- a genuinely wrong numeric value, not a display issue.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import (
    check,
    check_conversion,
    check_pairwise_distinct,
    exact_decimal_string,
    generator,
    pick_distractors,
)

# ---------------------------------------------------------------------------
# Quadruple table -- built ONCE at import time. See module docstring.

_A_B_LIMIT = 20
_RATIO_THRESH = Fraction(1, 2)
_ANGLE_LO_DEG = 20.0
_ANGLE_HI_DEG = 160.0
_SCALE_FACTORS = (1, 2, 3)
_HARD_SCALE = 10  # cm; guarantees an exact whole-centimetre-to-metres
# conversion (never a rounding decision) for the hard tier.


def _isqrt_exact(n: int) -> int | None:
    if n < 0:
        return None
    r = math.isqrt(n)
    return r if r * r == n else None


def raw_quadruples(limit: int = _A_B_LIMIT) -> list[tuple[int, int, int, int]]:
    """Every primitive integer ``(a, b, d1, d2)`` with ``a<=b``, ``d1<=d2``,
    satisfying the parallelogram diagonal law ``d1^2+d2^2 = 2(a^2+b^2)``
    EXACTLY over ``1 <= a <= b <= limit`` -- deliberately with NO
    plausibility filter, so ``tests/test_p11c_parallelogram_build.py`` can
    draw straight from this to prove the red run."""
    out: list[tuple[int, int, int, int]] = []
    for a in range(1, limit + 1):
        for b in range(a, limit + 1):
            total = 2 * (a * a + b * b)
            max_d1 = math.isqrt(total // 2) + 1
            for d1 in range(1, max_d1 + 1):
                d2 = _isqrt_exact(total - d1 * d1)
                if d2 is None or d2 < d1:
                    continue
                if math.gcd(math.gcd(a, b), math.gcd(d1, d2)) != 1:
                    continue
                out.append((a, b, d1, d2))
    return out


def angle_between_sides_deg(a: int, b: int, given_diagonal: int) -> float:
    """The angle theta between sides a,b implied by ONE diagonal, from the
    law of cosines: ``d^2 = a^2+b^2+2ab*cos(theta)``, so
    ``cos(theta) = (d^2-a^2-b^2)/(2ab)``. Clamped to ``[-1, 1]`` before
    ``acos`` purely to absorb float rounding at the extremes; every quadruple
    reaching this function already satisfies the parallelogram law exactly
    as an integer identity, so a clamp is never masking a real defect."""
    cos_theta = Fraction(given_diagonal * given_diagonal - a * a - b * b, 2 * a * b)
    cos_f = max(-1.0, min(1.0, float(cos_theta)))
    return math.degrees(math.acos(cos_f))


def is_plausible(a: int, b: int, d1: int, d2: int) -> bool:
    """The plausibility gate. See the
    module docstring for the angle and ratio bounds and their justification."""
    theta = angle_between_sides_deg(a, b, d1)
    if not (_ANGLE_LO_DEG <= theta <= _ANGLE_HI_DEG):
        return False
    if min(d1, d2) <= _RATIO_THRESH * max(a, b):
        return False
    return True


def _linear_diagonal_wrong(a: int, b: int, given: int) -> int:
    """Misconception ``uses-linear-law``, solved for the unknown diagonal:
    applies ``d1+d2=2(a+b)`` instead of the real (squared) law."""
    return 2 * (a + b) - given


def _sumsq_diagonal_wrong(a: int, b: int, given: int) -> int | None:
    """Misconception ``forgets-factor-of-two``, solved for the unknown
    diagonal: applies ``d1^2+d2^2=a^2+b^2``. Undefined (returns ``None``)
    whenever ``a^2+b^2 <= given^2`` -- which is exactly when ``given`` is
    already the LARGER of the two true diagonals (the discriminant reduces
    to ``(other^2-given^2)/2``, non-positive in that case) -- included only
    when positive
    not always exercisable."""
    disc = a * a + b * b - given * given
    if disc <= 0:
        return None
    return math.floor(math.sqrt(disc) + 0.5)


def _linear_side_wrong(known: int, d1: int, d2: int) -> int:
    """Misconception ``uses-linear-law``, solved for the unknown side."""
    return math.floor((d1 + d2) / 2 - known + 0.5)


def _sumsq_side_wrong(known: int, d1: int, d2: int) -> int | None:
    """Misconception ``uses-sum-then-squares``, solved for the unknown side:
    computes ``(d1+d2)^2`` instead of ``d1^2+d2^2``. Undefined (returns
    ``None``) if the resulting discriminant is non-positive."""
    disc = (d1 + d2) ** 2 / 2 - known * known
    if disc <= 0:
        return None
    return math.floor(math.sqrt(disc) + 0.5)


def verify_diagonal_law(given: int, unknown: int, side1: int, side2: int) -> None:
    """The independent check, used identically by all three tiers: re-square
    and sum the STATED side/diagonal values directly
    (``given^2 + unknown^2 == 2*(side1^2+side2^2)``), independent of
    whichever unknown position the construction actually solved for (a sqrt
    for the diagonal tiers, a different rearrangement for the side tier) --
    this verifies the full identity holds for the whole quadruple, not
    merely that the one rearranged formula was applied correctly."""
    check(
        given * given + unknown * unknown == 2 * (side1 * side1 + side2 * side2),
        "parallelogram diagonal law failed (independent re-square)",
    )


def _validate_row(a: int, b: int, d1: int, d2: int) -> bool:
    """A row is kept in the drawable table only if the ALWAYS-exercisable
    ``uses-linear-law`` misconception is well-defined (positive) and
    genuinely different from the true value in EVERY direction the
    generator can draw it in (both diagonal-unknown directions, both
    side-unknown directions). The sum-of-squares misconceptions are
    opportunistic (only added at draw time when their own discriminant is
    positive -- see their docstrings) and do not gate table membership."""
    for given, other in ((d1, d2), (d2, d1)):
        wrong = _linear_diagonal_wrong(a, b, given)
        if wrong <= 0 or wrong == other:
            return False
    for known, unknown in ((a, b), (b, a)):
        wrong = _linear_side_wrong(known, d1, d2)
        if wrong <= 0 or wrong == unknown:
            return False
    return True


def _build_table() -> list[tuple[int, int, int, int]]:
    base = [q for q in raw_quadruples(_A_B_LIMIT) if is_plausible(*q)]
    table: list[tuple[int, int, int, int]] = []
    for a, b, d1, d2 in base:
        for k in _SCALE_FACTORS:
            row = (a * k, b * k, d1 * k, d2 * k)
            if _validate_row(*row):
                table.append(row)
    return table


_TABLE: list[tuple[int, int, int, int]] = _build_table()
check(len(_TABLE) >= 20, f"parallelogram quadruple table too small: {len(_TABLE)} rows")

# Pivot-unit factor tables for the hard tier's independent conversion checks
# (`check_conversion`), each expressed in the smallest unit named so the
# pivot arithmetic never needs a fraction of a fraction.
_CM_M_FACTORS = {"cm": Fraction(1), "m": Fraction(100)}
_CM_MM_FACTORS = {"cm": Fraction(10), "mm": Fraction(1)}

# Sentence-template variety. Three worded templates ask the IDENTICAL mathematical task per tier
# (still: given both sides and one diagonal, find the other diagonal for easy; given both diagonals
# and one side, find the other side for medium) -- no change to what is computed, to the answer's
# shape, or to which misconceptions are exercisable.
_EASY_TEMPLATES = (
    "A parallelogram has sides of length {side1} cm and {side2} cm. "
    "One of its diagonals is {given} cm long. What is the length of the "
    "other diagonal, in cm?",
    "In a parallelogram, one side is {side1} cm and the adjacent side is "
    "{side2} cm. One diagonal is {given} cm long. Find the length of the "
    "other diagonal, in cm.",
    "A parallelogram's two sides measure {side1} cm and {side2} cm. Given "
    "that one diagonal is {given} cm, calculate the length of the other "
    "diagonal, in cm.",
)
_MEDIUM_TEMPLATES = (
    "A parallelogram has diagonals of length {diag1} cm and {diag2} cm. "
    "One of its sides is {known} cm long. What is the length of the other "
    "side, in cm?",
    "In a parallelogram, the diagonals measure {diag1} cm and {diag2} cm. "
    "One side is {known} cm long. Find the length of the other side, in cm.",
    "A parallelogram has diagonal lengths {diag1} cm and {diag2} cm. Given "
    "that one side is {known} cm, calculate the length of the other side, "
    "in cm.",
)


def _build_easy(rng) -> dict:
    a, b, d1, d2 = rng.choice(_TABLE)
    if rng.choice([True, False]):
        given, unknown = d1, d2
    else:
        given, unknown = d2, d1
    if rng.choice([True, False]):
        side1, side2 = a, b
    else:
        side1, side2 = b, a

    verify_diagonal_law(given, unknown, side1, side2)

    answer_text = str(unknown)
    candidates = [str(_linear_diagonal_wrong(side1, side2, given))]
    sumsq = _sumsq_diagonal_wrong(side1, side2, given)
    if sumsq is not None:
        candidates.append(str(sumsq))
    # Safety-net filler (not itself a named misconception/043
    # precedent): repeats the given diagonal, guaranteeing a floor of
    # distractors if the two named misconceptions above happen to collide.
    candidates.append(str(given))

    distractors = pick_distractors(answer_text, candidates)
    check_pairwise_distinct(answer_text, *distractors, msg="parallelogram easy-tier distractor collision")

    text = rng.choice(_EASY_TEMPLATES).format(side1=side1, side2=side2, given=given)
    working = [
        f"d1^2 + d2^2 = 2(a^2 + b^2): {given}^2 + d2^2 = 2({side1}^2 + {side2}^2).",
        f"d2 = {answer_text} cm.",
    ]
    return dict(
        text=text,
        answer=unknown,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


def _build_medium(rng) -> dict:
    a, b, d1, d2 = rng.choice(_TABLE)
    if rng.choice([True, False]):
        known, unknown = a, b
    else:
        known, unknown = b, a
    if rng.choice([True, False]):
        diag1, diag2 = d1, d2
    else:
        diag1, diag2 = d2, d1

    verify_diagonal_law(diag1, diag2, known, unknown)

    answer_text = str(unknown)
    candidates = [str(_linear_side_wrong(known, diag1, diag2))]
    sumsq = _sumsq_side_wrong(known, diag1, diag2)
    if sumsq is not None:
        candidates.append(str(sumsq))
    candidates.append(str(known))  # safety-net filler: repeats the given side

    distractors = pick_distractors(answer_text, candidates)
    check_pairwise_distinct(answer_text, *distractors, msg="parallelogram medium-tier distractor collision")

    text = rng.choice(_MEDIUM_TEMPLATES).format(diag1=diag1, diag2=diag2, known=known)
    working = [
        f"d1^2 + d2^2 = 2(a^2 + b^2): {diag1}^2 + {diag2}^2 = 2({known}^2 + b^2).",
        f"b = {answer_text} cm.",
    ]
    return dict(
        text=text,
        answer=unknown,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


def _build_hard(rng) -> dict:
    a, b, d1, d2 = rng.choice(_TABLE)
    # Scale to a common base unit of centimetres large enough that a
    # side-in-metres conversion reads naturally (never below 1 dp).
    a, b, d1, d2 = (a * _HARD_SCALE, b * _HARD_SCALE, d1 * _HARD_SCALE, d2 * _HARD_SCALE)

    if rng.choice([True, False]):
        given, unknown = d1, d2
    else:
        given, unknown = d2, d1
    if rng.choice([True, False]):
        side1, side2 = a, b
    else:
        side1, side2 = b, a

    verify_diagonal_law(given, unknown, side1, side2)

    linear_cm = _linear_diagonal_wrong(side1, side2, given)  # always clean in cm (table validation)
    # Converting the GIVEN rather than a SIDE or the ANSWER is a real different conversion target,
    # not a relabelling of either existing variant.
    variant = rng.choice(["side_in_metres", "answer_in_millimetres", "given_diagonal_in_millimetres"])

    if variant == "side_in_metres":
        side1_m = Fraction(side1, 100)
        check_conversion(
            Fraction(side1),
            "cm",
            "m",
            side1_m,
            _CM_M_FACTORS,
            label="parallelogram hard-tier side-to-metres conversion",
        )
        side1_m_text = exact_decimal_string(side1_m, 2)

        answer_text = str(unknown)
        text = (
            f"A parallelogram has one side {side1_m_text} m long and the adjacent side "
            f"{side2} cm long. One of its diagonals is {given} cm long. What is the length of "
            f"the other diagonal, in cm?"
        )
        working = [
            f"Convert {side1_m_text} m to centimetres: {side1_m_text} m = {side1} cm.",
            f"d1^2 + d2^2 = 2(a^2 + b^2): {given}^2 + d2^2 = 2({side1}^2 + {side2}^2).",
            f"d2 = {answer_text} cm.",
        ]

        candidates = [str(linear_cm)]
        # forgets-unit-conversion: plugs the bare metre numeral into the law
        # as though it were already in centimetres.
        forgot_disc = 2 * (side1_m * side1_m + side2 * side2) - given * given
        if forgot_disc > 0:
            forgot_val = math.floor(math.sqrt(float(forgot_disc)) + 0.5)
            if forgot_val > 0:
                candidates.append(str(forgot_val))
        candidates.append(str(given))  # safety-net filler
    elif variant == "answer_in_millimetres":
        unknown_mm = unknown * 10
        check_conversion(
            Fraction(unknown),
            "cm",
            "mm",
            Fraction(unknown_mm),
            _CM_MM_FACTORS,
            label="parallelogram hard-tier answer-to-millimetres conversion",
        )

        answer_text = str(unknown_mm)
        text = (
            f"A parallelogram has sides of length {side1} cm and {side2} cm. One of its "
            f"diagonals is {given} cm long. What is the length of the other diagonal, in "
            f"millimetres?"
        )
        working = [
            f"d1^2 + d2^2 = 2(a^2 + b^2): {given}^2 + d2^2 = 2({side1}^2 + {side2}^2).",
            f"d2 = {unknown} cm = {answer_text} mm.",
        ]

        # forgets-unit-conversion: reports the correct CENTIMETRE numeral as
        # though it already answered in millimetres.
        candidates = [str(unknown), str(linear_cm * 10)]
    else:  # "given_diagonal_in_millimetres"
        given_mm = given * 10
        check_conversion(
            Fraction(given),
            "cm",
            "mm",
            Fraction(given_mm),
            _CM_MM_FACTORS,
            label="parallelogram hard-tier given-diagonal-to-millimetres conversion",
        )

        answer_text = str(unknown)
        text = (
            f"A parallelogram has sides of length {side1} cm and {side2} cm. One of its "
            f"diagonals is {given_mm} mm long. What is the length of the other diagonal, "
            f"in cm?"
        )
        working = [
            f"Convert {given_mm} mm to centimetres: {given_mm} mm = {given} cm.",
            f"d1^2 + d2^2 = 2(a^2 + b^2): {given}^2 + d2^2 = 2({side1}^2 + {side2}^2).",
            f"d2 = {answer_text} cm.",
        ]

        candidates = [str(linear_cm)]
        # forgets-unit-conversion: plugs the bare MILLIMETRE numeral into
        # the law as though it were already in centimetres.
        forgot_disc = 2 * (side1 * side1 + side2 * side2) - given_mm * given_mm
        if forgot_disc > 0:
            forgot_val = math.floor(math.sqrt(float(forgot_disc)) + 0.5)
            if forgot_val > 0 and forgot_val != unknown:
                candidates.append(str(forgot_val))
        candidates.append(str(given))  # safety-net filler

    distractors = pick_distractors(answer_text, candidates)
    check_pairwise_distinct(answer_text, *distractors, msg="parallelogram hard-tier distractor collision")

    return dict(
        text=text,
        answer=int(answer_text),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


@generator(
    "p11c_parallelogram_diagonal_law",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p11c_parallelogram_diagonal_law(rng, tier="easy"):
    """The sum of the squares of the lengths of the diagonals of a
    parallelogram equals the sum of the squares of the lengths of the sides
    (ACMSM041): ``d1^2+d2^2 = 2(a^2+b^2)``. ``tier`` selects which unknown
    position is solved for -- see the module docstring for the full
    difficulty progression."""
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    if tier == "hard":
        return _build_hard(rng)
    raise ValueError(f"unknown tier {tier!r}")
