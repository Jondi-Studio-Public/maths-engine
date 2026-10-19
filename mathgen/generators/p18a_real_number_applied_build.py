"""Generates applied real-number questions: compute an answer involving square roots or pi from a bounded applied
problem, as `answer: expression`.

Scope: "Given a bounded applied problem stated in terms of
rational and/or irrational real numbers (e.g. involving square roots or
pi), compute the requested numeric or exact-form answer."

`answer: expression` at every tier: a decimal key is graded EXACTLY at the precision it displays
(`mathgen.grading.grade_expression` parses both sides with sympy and compares the resulting values -- no
separate tolerance path the way `grade_number` has one); an exact key (a rational multiple of pi, or `a +
sqrt(n)`) accepts every algebraically-equal spelling via `symbolically_equal`. `answer_alt` is never set (it
would let a rounded decimal answer a question asking for an exact form).

Four tiers, each a composite "rational quantity + irrational quantity"
applied problem, escalating by REASONING, never by size alone:

- easy: a composite perimeter (two straight sides of a rectangle, one
  remaining straight end, and a semicircular curve replacing the other end)
  -- every length is GIVEN, the learner only has to recognise which term is
  rational (2 x length + the straight end) and which is irrational (the
  semicircular arc, pi x radius), then add them. Exact answer, form
  `a + b*pi`.
- medium: the irrational length is not given -- it is a rectangle's diagonal,
  which the learner must derive with Pythagoras first, then add to the two
  straight sides adjacent to it. Exact answer, form `a + sqrt(n)`. The
  (p, q) pair is resampled away from every Pythagorean triple under 26 (see
  the module's own EXCLUDED_TRIPLES) so the diagonal is never accidentally
  rational -- the whole point of the tier is a genuine surd.
- hard: the same composite-perimeter shape as easy, but now stacking TWO
  decisions -- the semicircular diameter is NOT given: it equals the
  diagonal of a rectangular plot, so the learner must derive it via
  Pythagoras first (medium's own decision), and only then carry true,
unrounded pi
  through the whole calculation and round only the final total. The straight
  side is an independent decimal. `rounds-pi-early` (a fixed rational pi,
  22/7 or 3.14), `forgets-to-halve-diagonal` (uses the whole diagonal as the
  radius) and `treats-one-leg-as-radius` (skips the Pythagorean derivation
  altogether) are the on-theme distractors.
- exam: the REVERSE of easy -- the exact total perimeter (`R + C*pi`) and the
  semicircular radius (`C`, matching the pi-coefficient already stated) are
  given; the learner must isolate the unknown straight-side length `L` by
  recognising that the stated pi-coefficient already accounts for the
  semicircular term, so only the rational part `R` needs halving. A new
  decision (working backward from a stated exact total to a single unknown
  linear dimension), not merely bigger numbers.

Independent check for every tier: `check_equivalent`/`check_pairwise_distinct` re-derive the exact/rounded
value from the raw stated inputs by a differently -arranged formula (never by re-using an intermediate
variable the construction itself already computed); the true independence check is the SEPARATE oracle test
file, which parses `question.text` and re-derives the answer with its own regexes and its own sympy formulas.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy as sp

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    pairwise_distinct,
    phrase,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)

_MAX_TRIES = 500


def _numeric_pairwise_ok(*exprs, tol: float = 1e-9) -> bool:
    """Fast NUMERIC pre-filter for pairwise distinctness (a float `.evalf()`
    comparison, not a symbolic one) -- used only to decide whether to
    RESAMPLE inside a draw loop. `check_pairwise_distinct` (the symbolic,
    slow check `mathgen.core` names) is still called exactly once,
    unconditionally, on every draw this function accepts, as the real proof
    of distinctness; this is purely a speed optimisation; `symbolically_equal`
    on a pi/sqrt-bearing sympy `Add` costs ~6 ms a call (measured, this
    module's own easy/medium candidates, 4-way pairwise = 6 comparisons), and
    calling it twice per draw (once to decide whether to resample, once to
    verify) pushed this generator's own median `make()` time over the
    project's 15 ms soft-flag threshold (measured 14-18 ms before this
    change). Two genuinely different numeric values for these constructions
    are always at least ~0.1 apart, so `tol=1e-9` cannot mask a real
    collision this generator could actually draw."""
    values = [float(e.evalf()) for e in exprs]
    for i, left in enumerate(values):
        for right in values[i + 1 :]:
            if abs(left - right) <= tol:
                return False
    return True


# ---------------------------------------------------------------------------
# easy: composite perimeter (rectangle + semicircular end), exact form a+b*pi.
# Every length is given; the learner only has to identify and combine the
# rational term (2*L + W) and the irrational term (pi * W/2).
# ---------------------------------------------------------------------------

_EASY_TEMPLATES = [
    "A running track is shaped like a rectangle {L} m by {W} m, but one of "
    "the {W} m ends is replaced by a semicircular curve of diameter {W} m. "
    "Using the true, unrounded value of π, find the EXACT perimeter of "
    "the track -- the two straight sides of length {L} m, the one remaining "
    "straight end of length {W} m, and the semicircular curve -- in the form "
    "a + b·π (in m).",
    "A garden bed is a rectangle {L} m by {W} m with one of its {W} m ends "
    "replaced by a semicircular curve of diameter {W} m. Give the EXACT "
    "perimeter of the bed -- the two straight sides of {L} m, the remaining "
    "straight {W} m end, and the semicircular curve, using the true "
    "unrounded value of π -- in the form a + b·π, in metres.",
    "A swimming lane is bounded by two straight sides of length {L} m, one "
    "straight end of length {W} m, and a semicircular end of diameter {W} m. "
    "State the EXACT total distance around the lane, using the true "
    "unrounded value of π, in the form a + b·π (m).",
]


def _easy(rng):
    for _ in range(_MAX_TRIES):
        L = rng.randint(4, 24)
        W = rng.randint(4, 24)
        r = Fraction(W, 2)
        a = 2 * L + W
        b = r
        answer = sp.Integer(a) + sp.Rational(b.numerator, b.denominator) * sp.pi

        # Misconceptions, all kept in the same a+b*pi SHAPE as the answer.
        uses_diameter_not_radius = sp.Integer(a) + sp.Rational(W) * sp.pi
        rsq = r * r
        uses_area_formula = sp.Integer(a) + sp.Rational(rsq.numerator, rsq.denominator) * sp.pi
        # Keep the SAME a+b*pi shape as the key (never a bare b*pi term with
        # no rational part, which would be identifiable by form): drops the
        # one remaining straight end (W) from the rational part but keeps
        # the 2*L straight sides and the correct semicircular pi term.
        drops_straight_end = sp.Integer(2 * L) + sp.Rational(b.numerator, b.denominator) * sp.pi
        candidates = [uses_diameter_not_radius, uses_area_formula, drops_straight_end]

        if not _numeric_pairwise_ok(answer, *candidates):
            continue

        # Independent re-derivation: start from the FULL rectangle's own
        # perimeter (2*L + 2*W), remove the one straight W-end that was
        # replaced, and add half the circumference of a circle of diameter W
        # -- a differently-arranged formula from the construction above,
        # never reusing `a`/`b` directly.
        rederived = (sp.Integer(2 * L + 2 * W) - sp.Integer(W)) + (sp.Rational(W, 2) * sp.pi)
        check_equivalent(answer, rederived, "easy construction disagrees with the full-rectangle re-derivation")
        check_pairwise_distinct(answer, *candidates, msg="easy misconception distractor collides")

        answer_text = str(answer)
        text = phrase(rng, _EASY_TEMPLATES, L=L, W=W)
        distractors = pick_distractors(answer_text, [str(c) for c in candidates])
        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=answer_text,
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text, evaluate=False),
            distractors=distractors,
        )
    check(False, "could not draw a non-degenerate easy real-number-applied seed")


# ---------------------------------------------------------------------------
# medium: rectangle diagonal (derived first via Pythagoras) + two sides,
# exact form a + sqrt(n).
# ---------------------------------------------------------------------------

_MEDIUM_TEMPLATES = [
    "A path runs along the diagonal of a rectangular park {p} m by {q} m, "
    "then continues along the park's two adjacent sides back to the start. "
    "Find the EXACT total length of the path -- the diagonal plus the two "
    "sides -- as an exact value (you may leave a surd in your answer), "
    "in m.",
    "Fencing is needed along the diagonal of a rectangular block {p} m by "
    "{q} m, plus its {p} m side and its {q} m side. State the EXACT "
    "total length of fencing as an exact value (you may leave a surd in "
    "your answer), in metres.",
    "A cable runs along the diagonal of a rectangular field {p} m by {q} m, "
    "then two more cables run along the field's {p} m side and its {q} m "
    "side. Give the EXACT total cable length as an exact value (you "
    "may leave a surd in your answer), in m.",
]

# Every (p, q), p <= q, p, q in [3, 25] with p*p + q*q a perfect square -- excluded so the diagonal
# is never accidentally rational. Confirmed by direct trial-division sweep.
_PERFECT_SQUARE_PAIRS = {
    (3, 4),
    (5, 12),
    (6, 8),
    (7, 24),
    (8, 15),
    (9, 12),
    (10, 24),
    (12, 16),
    (15, 20),
    (18, 24),
    (20, 21),
}


def _is_perfect_square_pair(p: int, q: int) -> bool:
    lo, hi = min(p, q), max(p, q)
    if (lo, hi) in _PERFECT_SQUARE_PAIRS:
        return True
    n = p * p + q * q
    root = math.isqrt(n)
    return root * root == n


def _medium(rng):
    for _ in range(_MAX_TRIES):
        p = rng.randint(3, 25)
        q = rng.randint(3, 25)
        if p == q or _is_perfect_square_pair(p, q):
            continue
        # The wrong-radicand distractor below (forgetting to square q) must
        # itself stay irrational, so it keeps the key's own a+b*sqrt(c)
        # shape (never a bare integer, identifiable by form) -- resample
        # away from the rare seed where p*p + q happens to be a perfect
        # square.
        wrong_radicand_n = p * p + q
        wrong_root = math.isqrt(wrong_radicand_n)
        if wrong_root * wrong_root == wrong_radicand_n:
            continue

        answer = sp.Integer(p + q) + sp.sqrt(p * p + q * q)

        # All three distractors keep the SAME a + b*sqrt(c) shape as the key
        # (never a bare integer): one adjacent side missed, the diagonal's
        # radicand built wrong (forgot to square one side), and both sides
        # doubled.
        only_one_side_plus_diagonal = sp.Integer(q) + sp.sqrt(p * p + q * q)
        wrong_radicand = sp.Integer(p + q) + sp.sqrt(wrong_radicand_n)
        both_sides_doubled = sp.Integer(2 * p + 2 * q) + sp.sqrt(p * p + q * q)
        candidates = [only_one_side_plus_diagonal, wrong_radicand, both_sides_doubled]

        if not _numeric_pairwise_ok(answer, *candidates):
            continue

        # Independent re-derivation: rebuild the diagonal length directly
        # from the raw stated dimensions via sqrt(q*q + p*p) (operand order
        # swapped, never reusing the answer's own sqrt(p*p+q*q) term).
        rederived = sp.Integer(q) + sp.Integer(p) + sp.sqrt(q * q + p * p)
        check_equivalent(answer, rederived, "medium construction disagrees with the swapped-order re-derivation")
        check_pairwise_distinct(answer, *candidates, msg="medium misconception distractor collides")

        answer_text = str(answer)
        text = phrase(rng, _MEDIUM_TEMPLATES, p=p, q=q)
        distractors = pick_distractors(answer_text, [str(c) for c in candidates])
        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=answer_text,
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text, evaluate=False),
            distractors=distractors,
        )
    check(False, "could not draw a non-degenerate medium real-number-applied seed")


# ---------------------------------------------------------------------------
# hard: the same composite-perimeter shape as easy, but now stacking TWO
# decisions instead of one -- (1) the semicircular diameter is not given
# directly: it equals the diagonal of a rectangular plot, so the learner
# must derive it with Pythagoras first (medium's own decision), and only
# THEN (2) carry the true, unrounded value of pi through the calculation
# and round the total once, at the very end. The
# straight side L is an independent decimal, so the composite total genuinely
# needs both a Pythagorean derivation AND a decision about when to round pi --
# not merely "round at the end" on its own.
# ---------------------------------------------------------------------------

_HARD_D = 2

_HARD_TEMPLATES = [
    "A cycling track has two straight sides, each {L} m long, and no other "
    "straight edge -- the far end is closed by a single semicircular end "
    "whose diameter equals the diagonal of a rectangular plot {a} m by "
    "{b} m. Using the true, unrounded value of π, find the track's total "
    "perimeter, rounded to {d} decimal places (m).",
    "A running lane has two straight sides, each {L} m long, open at the "
    "far end except for a single semicircular end whose diameter equals "
    "the diagonal of a rectangular plot {a} m by {b} m. Carry the true, "
    "unrounded value of π through your working, and give the total "
    "distance around the lane, rounded to {d} decimal places (in m).",
    "A garden path has two straight edges, each {L} m long, closed at one "
    "end only by a semicircular edge whose diameter equals the diagonal of "
    "a rectangular plot {a} m by {b} m. Using the true value of π (never a "
    "rounded approximation such as 3.14 or 22/7) throughout your working, "
    "find the total length of the path, rounded to {d} decimal "
    "places (m).",
]


def _hard_total(L: Fraction, r: float, pi_value: float) -> Fraction:
    return Fraction(float(2 * L) + r * pi_value)


def _hard(rng):
    d = _HARD_D
    for _ in range(_MAX_TRIES):
        L = Fraction(rng.randint(50, 400), 10)  # 5.0 .. 40.0, independent straight side
        a = rng.randint(3, 20)
        b = rng.randint(3, 20)
        if _is_perfect_square_pair(a, b):
            continue  # keep the derived diagonal genuinely irrational

        diagonal = math.sqrt(a * a + b * b)
        r = diagonal / 2  # the semicircular radius -- must be derived, never given

        exact_pre = _hard_total(L, r, math.pi)
        if not rounds_unambiguously(float(exact_pre), d):
            continue
        answer = round_half_up(exact_pre, d)

        wrong_pi = rng.choice([Fraction(22, 7), Fraction(314, 100)])
        rounds_pi_early_pre = _hard_total(L, r, float(wrong_pi))
        # Forgets to halve the diagonal: uses the whole diagonal as the radius.
        forgets_to_halve_diagonal_pre = _hard_total(L, diagonal, math.pi)
        # Avoids the Pythagorean derivation altogether: treats the longer
        # given side as if it were the radius.
        treats_one_leg_as_radius_pre = _hard_total(L, float(max(a, b)), math.pi)

        pre_values = [rounds_pi_early_pre, forgets_to_halve_diagonal_pre, treats_one_leg_as_radius_pre]
        if any(not rounds_unambiguously(float(v), d) for v in pre_values):
            continue

        candidates = [round_half_up(v, d) for v in pre_values]
        if not pairwise_distinct(answer, *candidates):
            continue
        if any(abs(c - answer) <= stated_precision_tolerance(d) for c in candidates):
            continue

        # Independent re-derivation: rebuild the exact pre-rounding total
        # from the raw stated L/a/b via sympy's own pi and its own
        # `sqrt(a**2 + b**2)/2` (a different route from this module's
        # float-based `_hard_total`), then round with the same half-up rule
        # and confirm it lands on the same d-decimal-place value.
        sym_radius = sp.sqrt(a * a + b * b) / 2
        sym_total = 2 * sp.Rational(L.numerator, L.denominator) + sym_radius * sp.pi
        sym_value = Fraction(str(sym_total.evalf(30)))
        check(
            abs(round_half_up(sym_value, d) - answer) <= stated_precision_tolerance(d),
            "hard construction disagrees with the sympy-evalf re-derivation",
        )
        check_pairwise_distinct(answer, *candidates, msg="hard misconception distractor collides")

        answer_text = stated_precision_string(answer, d)
        distractor_texts = [stated_precision_string(c, d) for c in candidates]
        text = phrase(rng, _HARD_TEMPLATES, L=stated_precision_string(L, 1), a=a, b=b, d=d)
        distractors = pick_distractors(answer_text, distractor_texts)
        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=answer_text,
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text, evaluate=False),
            distractors=distractors,
        )
    check(False, "could not draw a non-degenerate hard real-number-applied seed")


# ---------------------------------------------------------------------------
# exam: the REVERSE of easy -- the exact total (R + C*pi) and the
# semicircular radius C are stated; find the unknown straight-side length L.
# ---------------------------------------------------------------------------

_EXAM_TEMPLATES = [
    "A relay track is made of two straight sides, each "
    "of length L metres, and no other straight edge -- the far end is "
    "closed by a single semicircular end of radius {C} m. The track's "
    "EXACT total perimeter is stated to be {R} + {C}·π metres. Find L, in "
    "metres.",
    "A running track has two straight sides, each of "
    "length L metres, open at the far end except for a single semicircular "
    "end of radius {C} m. Its EXACT total perimeter is exactly "
    "{R} + {C}·π metres. Work out L, in metres.",
    "A cycling track consists of two straight sides, "
    "each of length L metres, closed at one end only by a semicircular end "
    "of radius {C} m. The manufacturer states the total perimeter is "
    "exactly {R} + {C}·π metres. Determine L, in metres.",
]


def _exam(rng):
    for _ in range(_MAX_TRIES):
        # R (= 2*L_true) is forced ODD -- so the key L=R/2 is ALWAYS a
        # genuine half-integer (e.g. 35/2), never an integer -- and C is
        # forced EVEN, so every distractor below stays in the SAME n/2
        # fractional form as the key (never a bare integer beside a
        # fraction, identifiable by form).
        R = rng.randrange(7, 240, 2)  # odd, 7 .. 239
        L_true = Fraction(R, 2)
        # C (the semicircular radius) is bounded below L_true so that a
        # learner's "halve then subtract the radius" misconception distractor
        # (below) never lands on an implausible negative length -- a genuine
        # physical constraint (the curved end cannot dwarf the straight
        # sides), not a construction convenience.
        c_max = max(2, int(L_true) - 1)
        c_upper = min(40, c_max)
        C = rng.randrange(2, c_upper + 1, 2)  # even

        answer = sp.Rational(R, 2)

        # All three distractors keep the SAME n/2 half-integer shape as the
        # key: adds (instead of subtracts) the radius before halving,
        # subtracts the radius before halving, and subtracts the radius
        # AFTER halving.
        adds_pi_coefficient = sp.Rational(R + C, 2)
        subtracts_pi_coefficient = sp.Rational(R - C, 2)
        halves_then_subtracts_coefficient = sp.Rational(R, 2) - sp.Integer(C)
        candidates = [adds_pi_coefficient, subtracts_pi_coefficient, halves_then_subtracts_coefficient]

        if not pairwise_distinct(answer, *candidates):
            continue

        # Independent re-derivation: substitute L back into the ORIGINAL
        # forward formula (2*L + C*pi) and confirm the rational part alone
        # (stripping the pi term symbolically) matches R -- a check on the
        # inverse relationship, not a restatement of the same division.
        forward_total = 2 * answer + sp.Integer(C) * sp.pi
        rational_part = sp.simplify(forward_total - sp.Integer(C) * sp.pi)
        check_equivalent(rational_part, sp.Integer(R), "exam construction: L does not satisfy the stated forward total")
        check_pairwise_distinct(answer, *candidates, msg="exam misconception distractor collides")

        answer_text = str(answer)
        R_text = str(R)
        text = phrase(rng, _EXAM_TEMPLATES, C=C, R=R_text)
        distractors = pick_distractors(answer_text, [str(c) for c in candidates])
        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=answer_text,
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text, evaluate=False),
            distractors=distractors,
        )
    check(False, "could not draw a non-degenerate exam real-number-applied seed")


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard, "exam": _exam}


@generator(
    "p18a_real_number_applied",
    level=10,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p18a_real_number_applied(rng, tier: str = "easy"):
    """Applied real-number problems combining a rational and an irrational
    quantity, four tiers of genuine reasoning escalation -- see the module
    docstring for the full ladder argument.

    easy   -- every length given; identify and add the rational term
              (2*L + W) and the irrational term (pi * W/2). Exact form
              a + b*pi.
    medium -- the irrational length (a rectangle's diagonal) must be derived
              first via Pythagoras, then combined with the two adjacent
              sides. Exact form a + sqrt(n).
    hard   -- the semicircular diameter is not given -- it equals the
              diagonal of a rectangular plot, derived via Pythagoras first --
              then the learner must carry true, unrounded pi through the
              whole calculation and round only the final total
              to a stated precision: two decisions
              stacked, not one.
    exam   -- the reverse of easy: given the exact total (R + C*pi) and the
              radius C, isolate the unknown straight-side length L, deciding
              that the stated pi-coefficient already accounts for the
              semicircular term.
    """
    try:
        fn = _TIERS[tier]
    except KeyError:
        raise ValueError(f"unknown tier {tier!r}; known: {', '.join(sorted(_TIERS))}") from None
    return fn(rng)
