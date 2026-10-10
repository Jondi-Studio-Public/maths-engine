"""Generates `p13c_riemann_sum_area` questions (ACMMM124): estimate an area under a curve with a sum of
rectangles.

Kernel: the descriptor's own notation Sigma f(x_i) * delta_x_i is a RIEMANN SUM
over a stated partition -- a LEFT, RIGHT or MIDPOINT rectangle approximation
to the area under y = f(x), computed as an EXACT rational NUMBER from the
stated partition. Every tier uses exactly ONE function value per rectangle
(a left endpoint, a right endpoint, or a midpoint).

**The composite-trapezoidal-polynomial boundary**
(``skills/numerical-and-computational-mathematics/composite-trapezoidal-polynomial.yaml``):
that skill's rule averages the TWO endpoint heights of every strip -- the
trapezoidal weights h/2, h, h, ..., h, h/2. This generator NEVER builds the
trapezoidal rule. Every tier here uses a single height per strip, and the
question text always names the rule ("left Riemann sum" / "right Riemann
sum" / "midpoint Riemann sum"), never "trapezoid" -- the neighbour's own
easy seed 0, "Approximate integral from 1 to 4 of f(x) = 3x^2 + 3 using 3
composite trapezoids" -> 147/2, has no equivalent here: this generator's
LEFT sum over the same interval/function uses f(1), f(2), f(3) each times
the strip width 1, summed -- a different number, computed a different way,
and the wording always says which rule, never "trapezoid".

Difficulty ladder:
- easy: LEFT Riemann sum, uniform partition, n in [2, 4], f linear or a
  simple quadratic, integer breakpoints.
- medium: RIGHT Riemann sum, uniform partition, quadratic f, n in [4, 6].
- hard: MIDPOINT Riemann sum, uniform partition -- a genuinely different
  decision from both lower tiers (computing n midpoint x-values before
  evaluating f, not just choosing which endpoint). An odd strip width makes
  the midpoint land on a half-integer, so a squared term contributes a
  quarter-fraction to the total -- the reason this skill's ``fraction``
  grader kind matters, not just the integer sums easy/medium usually give.
- exam: NON-UNIFORM partition, stated explicitly as a list of breakpoints,
  using LEFT or RIGHT endpoints (stated) -- each strip's own delta_x_i is
  computed individually, not one shared width. This is the genuine
  escalation the descriptor's own per-rectangle delta_x_i notation already
  implies, and a different decision in kind from every uniform-partition
  tier: the learner must compute each delta_x_i separately before applying
  the rule.

Misconceptions (checked distinct, every seed, unconditionally):
- wrong-endpoint: a left sum computed with right endpoints or vice versa
  (easy, medium, exam -- collapses to the OTHER named rule's value);
- reuses-endpoint-not-midpoint: the midpoint rule's sample POINTS are used
  for delta_x_i, but f is evaluated at an endpoint instead of being
  recomputed at the midpoint (hard only);
- shared-width: a non-uniform partition is summed with one shared average
  width instead of each strip's own delta_x_i (exam only -- the on-contract
  misconception for that tier);
- off-by-one: drops the last rectangle/breakpoint interval (every tier).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_distinct,
    generator,
    join_terms,
    latex_answer,
    pick_distractors,
    rational_text,
)


# --- partition helpers ----------------------------------------------------
# A partition is just its list of breakpoints (n+1 values, increasing). All
# four tiers -- uniform or not -- are expressed this way so the same helpers
# serve every tier; only how the breakpoints are DRAWN differs by tier.


def _widths(breakpoints: list) -> list:
    return [breakpoints[i + 1] - breakpoints[i] for i in range(len(breakpoints) - 1)]


def _left_xs(breakpoints: list) -> list:
    return list(breakpoints[:-1])


def _right_xs(breakpoints: list) -> list:
    return list(breakpoints[1:])


def _mid_xs(breakpoints: list) -> list:
    return [Fraction(breakpoints[i] + breakpoints[i + 1], 2) for i in range(len(breakpoints) - 1)]


def _riemann_sum(xs: list, dxs: list, f) -> Fraction:
    """Construction's own route: accumulate the sum in a running total."""
    total = Fraction(0)
    for x, dx in zip(xs, dxs):
        total = total + Fraction(f(x)) * Fraction(dx)
    return total


def _check_riemann_sum(breakpoints: list, xs: list, f, expected: Fraction, label: str) -> None:
    """Independent route: re-derive delta_x_i from the raw breakpoints (not
    the widths list construction already built) and sum f(x_i) * delta_x_i
    via ``sum(...)`` with a Fraction accumulator, comparing exactly."""
    dxs = [breakpoints[i + 1] - breakpoints[i] for i in range(len(breakpoints) - 1)]
    pairs = list(zip(xs, dxs))
    recon = sum((Fraction(f(x)) * Fraction(dx) for x, dx in pairs), Fraction(0))
    check(recon == expected, f"{label}: riemann sum independent check mismatch")


def _fraction_answer(value) -> tuple[dict, str]:
    value = Fraction(value)
    return {"p": value.numerator, "q": value.denominator}, rational_text(value)


# --- function families ------------------------------------------------


def _linear_fn(a: int, b: int):
    return lambda x: a * x + b


def _quadratic_fn(a: int, b: int, c: int):
    return lambda x: a * x * x + b * x + c


def _linear_text(a: int, b: int) -> str:
    return join_terms([(a, "x"), (b, "")])


def _quadratic_text(a: int, b: int, c: int) -> str:
    return join_terms([(a, "x^2"), (b, "x"), (c, "")])


def _drop_last(xs: list, dxs: list) -> tuple[list, list]:
    return xs[:-1], dxs[:-1]


@generator(
    "p13c_riemann_sum_area",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13c_riemann_sum_area(rng, tier="easy"):
    """See module docstring for the full ladder and the trapezoidal boundary.

    easy:   LEFT Riemann sum, uniform partition, n in [2, 4].
    medium: RIGHT Riemann sum, uniform partition, n in [4, 6], quadratic.
    hard:   MIDPOINT Riemann sum, uniform partition, quadratic.
    exam:   non-uniform partition (explicit breakpoints), LEFT or RIGHT
            (stated), quadratic.
    """
    if tier == "easy":
        family = rng.choice(("linear", "quadratic"))
        base_a = rng.randint(1, 4) if family == "linear" else rng.randint(1, 2)
        base_b = rng.randint(0, 6) if family == "linear" else rng.randint(-2, 2)
        c = None if family == "linear" else rng.randint(2, 6)
        n = rng.randint(2, 4)
        left = rng.randint(0, 3)
        w = rng.randint(1, 2)
        right = left + n * w
        breakpoints = [left + i * w for i in range(n + 1)]
        dxs = _widths(breakpoints)

        # A uniform LEFT sum and RIGHT sum telescope to the SAME value
        # whenever f(left) == f(right) -- exactly true for a quadratic whose
        # vertex sits at the interval's own midpoint. Nudge `b` by a whole
        # number (breaking that exact symmetry) on the rare seed this
        # happens, rather than let `check_distinct` fail outright.
        for nudge in range(6):
            a = base_a
            b = base_b + nudge
            f = _linear_fn(a, b) if family == "linear" else _quadratic_fn(a, b, c)
            f_text = _linear_text(a, b) if family == "linear" else _quadratic_text(a, b, c)

            xs = _left_xs(breakpoints)
            answer_value = _riemann_sum(xs, dxs, f)

            wrong_endpoint = _riemann_sum(_right_xs(breakpoints), dxs, f)
            drop_xs, drop_dxs = _drop_last(xs, dxs)
            off_by_one = _riemann_sum(drop_xs, drop_dxs, f)
            candidates = [wrong_endpoint, off_by_one]
            if all(c_val != answer_value for c_val in candidates):
                break

        _check_riemann_sum(breakpoints, xs, f, answer_value, "easy left sum")
        check_distinct(answer_value, *candidates, msg="easy: a distractor equals the answer")

        text = (
            f"Using a LEFT Riemann sum with n = {n} equal subintervals, estimate the area "
            f"under the curve y = {f_text} on [{left}, {right}]."
        )

    elif tier == "medium":
        base_a = rng.randint(1, 3)
        base_b = rng.randint(-3, 3)
        c = rng.randint(3, 8)
        n = rng.randint(4, 6)
        left = rng.randint(0, 3)
        w = rng.randint(1, 2)
        right = left + n * w
        breakpoints = [left + i * w for i in range(n + 1)]
        dxs = _widths(breakpoints)

        # Same telescoping-symmetry hazard as easy (f(left) == f(right)):
        # nudge `b` by a whole number on the rare seed it occurs.
        for nudge in range(6):
            a = base_a
            b = base_b + nudge
            f = _quadratic_fn(a, b, c)
            f_text = _quadratic_text(a, b, c)

            xs = _right_xs(breakpoints)
            answer_value = _riemann_sum(xs, dxs, f)

            wrong_endpoint = _riemann_sum(_left_xs(breakpoints), dxs, f)
            drop_xs, drop_dxs = _drop_last(xs, dxs)
            off_by_one = _riemann_sum(drop_xs, drop_dxs, f)
            candidates = [wrong_endpoint, off_by_one]
            if all(c_val != answer_value for c_val in candidates):
                break

        _check_riemann_sum(breakpoints, xs, f, answer_value, "medium right sum")
        check_distinct(answer_value, *candidates, msg="medium: a distractor equals the answer")

        text = (
            f"Using a RIGHT Riemann sum with n = {n} equal subintervals, estimate the area "
            f"under the curve y = {f_text} on [{left}, {right}]."
        )

    elif tier == "hard":
        base_a = rng.randint(1, 3)
        base_b = rng.randint(-3, 3)
        c = rng.randint(3, 9)
        n = rng.randint(3, 5)
        left = rng.randint(0, 3)
        w = rng.randint(1, 3)
        right = left + n * w
        breakpoints = [left + i * w for i in range(n + 1)]
        dxs = _widths(breakpoints)

        # Same coincidence class as easy/medium (a symmetric quadratic can
        # make the midpoint sum collide with the reuse-endpoint value): nudge
        # `b` by a whole number on the rare seed it occurs.
        for nudge in range(6):
            a = base_a
            b = base_b + nudge
            f = _quadratic_fn(a, b, c)
            f_text = _quadratic_text(a, b, c)

            xs = _mid_xs(breakpoints)
            answer_value = _riemann_sum(xs, dxs, f)

            # reuses-endpoint-not-midpoint: the strip widths are right (the
            # midpoint rule's own delta_x_i), but f is evaluated at the left
            # endpoint instead of being recomputed at the midpoint. Always an
            # integer (integer breakpoints), unlike the midpoint-derived
            # answer.
            reuse_endpoint = _riemann_sum(_left_xs(breakpoints), dxs, f)
            # off-by-one: drops one or more trailing rectangles (a
            # wording allows dropping OR duplicating "one interval" -- dropping
            # a run of k is the same misconception at a different count, still
            # an off-by-count miscount of the rectangles). Every midpoint term
            # carries the SAME quarter-fraction offset (uniform partition, same
            # width each strip), so dropping exactly one term can coincidentally
            # land back on a whole number even when the true answer is a genuine
            # quarter-fraction -- try dropping k = 1, 2, 3, ... trailing terms
            # (whichever keeps at least one rectangle) until the result keeps
            # the SAME shape (whole vs fractional) as the true answer, so this
            # distractor is never pickable on shape alone.
            answer_is_whole = Fraction(answer_value).denominator == 1
            off_by_one = None
            for k in range(1, len(xs)):
                drop_xs, drop_dxs = xs[:-k], dxs[:-k]
                candidate_value = _riemann_sum(drop_xs, drop_dxs, f)
                if (Fraction(candidate_value).denominator == 1) == answer_is_whole:
                    off_by_one = candidate_value
                    break
            if off_by_one is None:
                off_by_one = _riemann_sum(xs[:-1], dxs[:-1], f)

            candidates = [reuse_endpoint, off_by_one]
            if all(c_val != answer_value for c_val in candidates):
                break

        _check_riemann_sum(breakpoints, xs, f, answer_value, "hard midpoint sum")
        check_distinct(answer_value, *candidates, msg="hard: a distractor equals the answer")

        text = (
            f"Using a MIDPOINT Riemann sum with n = {n} equal subintervals, estimate the area "
            f"under the curve y = {f_text} on [{left}, {right}]."
        )

    else:  # exam
        base_a = rng.randint(1, 2)
        base_b = rng.randint(-2, 3)
        c = rng.randint(4, 10)

        n_seg = rng.randint(4, 5)
        start = rng.randint(0, 3)
        widths = [rng.randint(1, 4) for _ in range(n_seg)]
        if len(set(widths)) == 1:
            # Force a genuinely non-uniform partition.
            widths[-1] = widths[-1] + 1
        breakpoints = [start]
        for seg_w in widths:
            breakpoints.append(breakpoints[-1] + seg_w)
        dxs = _widths(breakpoints)
        rule = rng.choice(("left", "right"))

        # On a non-uniform partition the wrong-endpoint/shared-width/
        # off-by-one candidates are not guaranteed distinct by any algebraic
        # identity (unlike the uniform tiers' telescoping symmetry), but a
        # coincidence is still possible for specific numbers -- nudge `b` on
        # the rare seed one occurs.
        for nudge in range(6):
            a = base_a
            b = base_b + nudge
            f = _quadratic_fn(a, b, c)
            f_text = _quadratic_text(a, b, c)

            xs = _left_xs(breakpoints) if rule == "left" else _right_xs(breakpoints)
            answer_value = _riemann_sum(xs, dxs, f)

            other_xs = _right_xs(breakpoints) if rule == "left" else _left_xs(breakpoints)
            wrong_endpoint = _riemann_sum(other_xs, dxs, f)
            shared_dx = Fraction(breakpoints[-1] - breakpoints[0], len(dxs))
            shared_width = sum((Fraction(f(x)) * shared_dx for x in xs), Fraction(0))
            drop_xs, drop_dxs = _drop_last(xs, dxs)
            off_by_one = _riemann_sum(drop_xs, drop_dxs, f)

            candidates = [wrong_endpoint, shared_width, off_by_one]
            if all(c_val != answer_value for c_val in candidates):
                break

        _check_riemann_sum(breakpoints, xs, f, answer_value, f"exam {rule} sum")
        check_distinct(answer_value, *candidates, msg="exam: a distractor equals the answer")

        bp_text = ", ".join(str(bp) for bp in breakpoints)
        rule_word = "LEFT" if rule == "left" else "RIGHT"
        text = (
            f"An interval is divided at breakpoints x = {bp_text} (each subinterval its own "
            f"width). Using a {rule_word} Riemann sum over this partition, estimate the area "
            f"under the curve y = {f_text}."
        )

    answer_payload, answer_text = _fraction_answer(answer_value)
    # A distractor whose denominator is trivial (a whole number) while the
    # answer's is not (or vice versa) is pickable on SHAPE alone, without
    # doing any arithmetic (the hard/midpoint tier's own quarter-fraction
    # answers occasionally collide this way with the endpoint-reuse and
    # off-by-one candidates, which are built from integer breakpoints and so
    # are integer far more often). Prefer same-shaped candidates; only fall
    # back to a shape mismatch if every candidate would otherwise be dropped
    # (`pick_distractors` still requires at least one option).
    answer_is_whole = Fraction(answer_value).denominator == 1
    same_shape = [c for c in candidates if (Fraction(c).denominator == 1) == answer_is_whole]
    shaped_candidates = same_shape or candidates
    distractor_texts = [_fraction_answer(c)[1] for c in shaped_candidates]
    distractors = pick_distractors(answer_text, distractor_texts)

    return dict(
        text=text,
        latex=text,
        answer=answer_payload,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
    )
