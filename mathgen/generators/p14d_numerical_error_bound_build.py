"""Deterministic numerical error bound for a supplied approximation and a
certified interval.

Given an approximation ``ahat`` (a rational) and a certified interval
``[lo, hi]`` (rationals, ``lo < hi``) known to contain the true value, with
``ahat`` strictly inside ``(lo, hi)`` -- a well-posed "our current best
guess, bracketed by a certified interval" scenario -- this module computes
the worst-case ABSOLUTE error bound ``max(ahat - lo, hi - ahat)`` and the
RELATIVE error bound ``absolute_bound / abs(ahat)``.

"relative error" has a standard textbook meaning (``|error| / |true value|``) that, applied rigorously to a
CERTIFIED INTERVAL rather than a known true value, divides by the smallest possible ``|true value|`` in the
interval instead -- a different, also-defensible number. Every prompt now states the convention this skill
uses explicitly (divide by ``|ahat|``, not the worst-case true value), so exactly one reading is correct; the
rigorous-textbook reading is the ``textbook-relative-error-uses-worst-case-true-value`` distractor.

This is deliberately narrow: the learner is never asked to reason about ``ahat`` falling outside the certified
interval (an empirical-error-claim question this typed grader cannot assess), about floating-point rounding,
or about how the certified interval itself was produced (conditioning / convergence territory). Both values
are returned as EXACT ``Fraction``s, never a rounded decimal, because the ``pair`` grader keeps its ~0.051
voice tolerance on typed answers too -- every component here must differ from every distractor's matching
component by more than that tolerance, asserted explicitly below.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    GRADER_TOLERANCE,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    pick_distractors,
    rational_text,
)


def _pair_answer(absolute_bound: Fraction, relative_bound: Fraction) -> tuple[dict, str]:
    return (
        {"absolute_bound": absolute_bound, "relative_bound": relative_bound},
        f"absolute_bound = {rational_text(absolute_bound)}, relative_bound = {rational_text(relative_bound)}",
    )


def _candidate_text(absolute: Fraction, relative: Fraction) -> str:
    return f"absolute_bound = {rational_text(absolute)}, relative_bound = {rational_text(relative)}"


def _bound(ahat: Fraction, lo: Fraction, hi: Fraction):
    """The two error bounds for one (ahat, lo, hi) draw, independently
    re-proven by a genuinely different route: the direct
    ``max(ahat - lo, hi - ahat)`` computation is checked against the
    half-width-plus-off-centre-offset identity
    ``(hi - lo)/2 + abs(ahat - (lo + hi)/2)``, unconditionally, every draw.
    """
    check(lo < ahat < hi, "ahat is not strictly inside the certified interval")
    d_lo = ahat - lo
    d_hi = hi - ahat
    absolute_bound = max(d_lo, d_hi)
    half_width = (hi - lo) / 2
    centre = (lo + hi) / 2
    check_equivalent(
        absolute_bound,
        half_width + abs(ahat - centre),
        "absolute bound disagrees with the half-width + offset identity",
    )
    # The relative bound is a trivial ratio of the already-independently -confirmed absolute_bound;
    # re-deriving absolute_bound a third time here would add nothing.
    relative_bound = absolute_bound / abs(ahat)
    return absolute_bound, relative_bound, d_lo, d_hi


def _assert_margins(true_pair, candidate_pairs):
    """Every distractor is unambiguously rejected by the real pair grader -- at least one of its two components
    differs from the true answer's matching component by more than the ~0.051 tolerance, by construction.
    """
    for pair in candidate_pairs:
        differs = any(abs(t - c) > GRADER_TOLERANCE for t, c in zip(true_pair, pair))
        check(
            differs,
            "a distractor sits within the pair grader's tolerance of the true bound in every component",
        )


def _finish(text, latex, ahat, lo, hi, working_extra, candidates):
    absolute_bound, relative_bound, d_lo, d_hi = _bound(ahat, lo, hi)
    answer, answer_text = _pair_answer(absolute_bound, relative_bound)
    answer_latex = (
        rf"\text{{absolute bound}} = {latex_answer(absolute_bound)},\; "
        rf"\text{{relative bound}} = {latex_answer(relative_bound)}"
    )
    candidate_pairs = [(a, r) for a, r in candidates]
    check_pairwise_distinct(
        (absolute_bound, relative_bound),
        *candidate_pairs,
        msg="a distractor collides with the true bound or with another distractor",
    )
    _assert_margins((absolute_bound, relative_bound), candidate_pairs)
    candidate_texts = [_candidate_text(a, r) for a, r in candidates]
    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_latex,
        working=[
            f"Distances: ahat - lo = {rational_text(d_lo)}, hi - ahat = {rational_text(d_hi)}.",
            *working_extra,
            f"absolute_bound = max(ahat - lo, hi - ahat) = {rational_text(absolute_bound)}; "
            f"relative_bound = absolute_bound / |ahat| = {rational_text(relative_bound)}.",
        ],
        distractors=pick_distractors(answer_text, candidate_texts, n=3, tolerance=GRADER_TOLERANCE),
    )


_INSTRUCTION = (
    "Give the absolute error bound and the relative error bound, where the "
    "relative error bound is the absolute error bound divided by |ahat| "
    "(not by the smallest possible true value in the interval)."
)


def _make_midpoint(rng):
    """easy: ahat is EXACTLY the interval midpoint, so both distances to lo
    and hi are equal and absolute_bound = (hi - lo)/2 trivially -- a plain
    half-width computation. Because d_lo == d_hi here, the
    "uses the closer endpoint" distractor collapses onto the correct answer
    and is not offered on this tier by construction; the width-only and
    wrong-denominator distractors are used instead."""
    lo = Fraction(rng.randint(2, 15))
    # half is kept a bounded MULTIPLE of lo (not an independent small
    # constant) so the wrong-denominator distractors' relative bound stays
    # clear of the pair grader's ~0.051 tolerance regardless of lo's own
    # magnitude -- half/lo and half/(lo+half) both stay comfortably apart
    # when half is a sizeable fraction of lo, which a tiny fixed half
    # relative to a large lo would not guarantee.
    half = Fraction(rng.randint(int(lo), 2 * int(lo)))
    hi = lo + 2 * half
    ahat = lo + half

    # Wording variety only: the certified facts, in the same order, are stated with genuinely
    # different clause structure -- never with a synonym for a slot-bank word (there are none here)
    # and never by moving `ahat = {a}` after `[{lo}, {hi}]`, because
    # `test_easy_tier_distances_to_both_endpoints_are_equal` anchors on `ahat = FRAC` appearing
    # BEFORE the bracket. `_INSTRUCTION` is kept byte-identical across templates and in `latex`, so
    # `test_question_latex_instructions.py` is unaffected. Chosen AFTER `half` is drawn so the
    # numeric draw for a given seed is unchanged.
    a_txt, lo_txt, hi_txt = rational_text(ahat), rational_text(lo), rational_text(hi)
    template = rng.choice(["lie", "known", "certified"])
    if template == "lie":
        text = (
            f"An approximation ahat = {a_txt} is known to lie strictly inside the certified "
            f"interval [{lo_txt}, {hi_txt}], which is guaranteed to contain the true value. "
            f"{_INSTRUCTION}"
        )
    elif template == "known":
        text = (
            f"It is known that ahat = {a_txt} lies strictly inside the certified interval "
            f"[{lo_txt}, {hi_txt}], which is guaranteed to contain the true value. "
            f"{_INSTRUCTION}"
        )
    else:
        text = (
            f"ahat = {a_txt} is an approximation known to sit strictly within the interval "
            f"[{lo_txt}, {hi_txt}], which is certified to contain the true value. "
            f"{_INSTRUCTION}"
        )
    latex = rf"\hat a = {a_txt},\quad [{lo_txt}, {hi_txt}]. \quad \text{{{_INSTRUCTION}}}"

    absolute_bound = half
    # width-only, using the FULL certified width instead of the half-width
    # (the half-width itself equals the true answer on this tier, so it
    # cannot be offered as a distractor -- see the docstring above)
    width_full = hi - lo
    d1 = (width_full, width_full / ahat)
    # textbook-relative-error-uses-worst-case-true-value: divides by lo, the
    # endpoint nearest 0 -- the rigorous textbook |error|/|true value| bound
    # applied to the smallest possible true value in the certified interval,
    # a legitimate reading the prompt's stated convention now excludes.
    d2 = (absolute_bound, absolute_bound / lo)
    # wrong-denominator: a plain wrong-endpoint slip, divides by hi instead
    d3 = (absolute_bound, absolute_bound / hi)
    return _finish(text, latex, ahat, lo, hi, [], [d1, d2, d3])


def _make_offcentre(rng):
    """medium: ahat is OFF-CENTRE (the two distances differ by at least 3),
    so the learner must compare the two candidate distances and take the
    LARGER one rather than just halving the width -- the "uses the closer
    endpoint" distractor is exercisable here."""
    d1 = Fraction(rng.randint(2, 10))
    # lo is kept a bounded fraction OF d1 (not an independent large value)
    # so the wrong-denominator distractor's relative bound stays clear of
    # the pair grader's ~0.051 tolerance regardless of scale -- see the
    # easy-tier comment above for the same reasoning.
    lo = Fraction(rng.randint(1, int(d1)))
    gap = Fraction(rng.randint(3, 10))
    d2 = d1 + gap
    ahat = lo + d1
    hi = ahat + d2

    # Wording variety only, same technique as the easy tier's fix: the certified facts, in the same
    # order, are stated with genuinely different clause structure -- `ahat = {a}` is kept BEFORE
    # `[{lo}, {hi}]` verbatim in every template (the oracle's `_AHAT_RE`/`_BRACKET_RE` and
    # `test_medium_and_hard_tiers_exercise_the_closer_endpoint_distractor` rely on both substrings
    # being present), `_INSTRUCTION` is kept byte-identical across templates and in `latex`. Chosen
    # AFTER the numeric draw so the per-seed numbers are unchanged.
    a_txt, lo_txt, hi_txt = rational_text(ahat), rational_text(lo), rational_text(hi)
    template = rng.choice(["lie", "known", "certified"])
    if template == "lie":
        text = (
            f"An approximation ahat = {a_txt} is known to lie strictly inside the certified "
            f"interval [{lo_txt}, {hi_txt}], which is guaranteed to contain the true value. "
            f"{_INSTRUCTION}"
        )
    elif template == "known":
        text = (
            f"It is known that ahat = {a_txt} lies strictly inside the certified interval "
            f"[{lo_txt}, {hi_txt}], which is guaranteed to contain the true value. "
            f"{_INSTRUCTION}"
        )
    else:
        text = (
            f"ahat = {a_txt} is an approximation known to sit strictly within the interval "
            f"[{lo_txt}, {hi_txt}], which is certified to contain the true value. "
            f"{_INSTRUCTION}"
        )
    latex = rf"\hat a = {a_txt},\quad [{lo_txt}, {hi_txt}]. \quad \text{{{_INSTRUCTION}}}"

    closer = min(d1, d2)
    width_full = hi - lo
    absolute_bound = max(d1, d2)
    dc = (closer, closer / ahat)
    dw = (width_full, width_full / ahat)
    # textbook-relative-error-uses-worst-case-true-value: divides by lo, the
    # endpoint nearest 0 -- the smallest possible |true value| in the
    # certified interval, the rigorous textbook reading the prompt's stated
    # convention now excludes.
    dd = (absolute_bound, absolute_bound / lo)
    return _finish(text, latex, ahat, lo, hi, [], [dc, dw, dd])


def _make_fraction_near_miss(rng):
    """hard: ahat, lo and hi are all exact sixths (the common denominator of
    halves and thirds), and the two candidate distances d_lo, d_hi are a
    NEAR MISS -- exactly 1/6 apart -- so the comparison and subtraction must
    be carried through exact fraction arithmetic rather than eyeballed
    decimals (1/6 repeats, so no clean decimal rounding tells d_lo from
    d_hi apart at a glance, even though 1/6 ~ 0.1667 clears the pair
    grader's ~0.051 tolerance comfortably). ``lo`` is kept a bounded
    fraction OF d1 (not an independent value), the same scale-invariant
    device the easy/medium tiers use, so the wrong-denominator distractor's
    relative bound also stays clear of tolerance regardless of scale."""
    d1 = Fraction(rng.randint(3, 20), 6)  # a sixth
    d2 = d1 + Fraction(1, 6)  # the near miss: exactly 1/6 larger
    lo = d1 * Fraction(rng.randint(1, 6), 6)
    ahat = lo + d1
    hi = ahat + d2

    a_txt, lo_txt, hi_txt = rational_text(ahat), rational_text(lo), rational_text(hi)
    text = (
        f"An approximation ahat = {a_txt} is known to lie strictly inside the certified "
        f"interval [{lo_txt}, {hi_txt}], which is guaranteed to contain the true value. "
        f"{_INSTRUCTION}"
    )
    latex = rf"\hat a = {a_txt},\quad [{lo_txt}, {hi_txt}]. \quad \text{{{_INSTRUCTION}}}"

    closer = min(d1, d2)
    width_full = hi - lo
    absolute_bound = max(d1, d2)
    dc = (closer, closer / ahat)
    dw = (width_full, width_full / ahat)
    # textbook-relative-error-uses-worst-case-true-value: divides by lo, the
    # endpoint nearest 0 -- the smallest possible |true value| in the
    # certified interval, the rigorous textbook reading the prompt's stated
    # convention now excludes.
    dd = (absolute_bound, absolute_bound / lo)
    return _finish(text, latex, ahat, lo, hi, [], [dc, dw, dd])


def _make_centre_radius(rng):
    """exam: the certified interval is given as a CENTRE and a RADIUS
    ("the true value is certified to lie within radius of centre") instead
    of [lo, hi] directly, so the learner must first recover
    lo = centre - radius, hi = centre + radius before applying the bound
    formulas -- an extra translation step, not a bigger number. This also
    keeps the exam tier textually distinct from bisection-interval-update's
    "[left, right]" bracket phrasing: this tier's text never shows a
    bracket at all."""
    radius = Fraction(rng.randint(4, 20))
    # centre is kept a bounded amount ABOVE radius (lo = centre - radius
    # bounded by radius itself), not an independent large value, so the
    # wrong-denominator distractor's relative bound stays clear of the pair
    # grader's ~0.051 tolerance regardless of scale -- see the easy-tier
    # comment above for the same reasoning.
    lo_gap = Fraction(rng.randint(1, int(radius)))
    centre = radius + lo_gap
    offset = Fraction(rng.randint(1, int(radius) - 1))
    sign = rng.choice([1, -1])
    ahat = centre + sign * offset
    lo = centre - radius
    hi = centre + radius

    # Wording variety only, same technique as the easy tier's fix: every template keeps the literal
    # substring "certified to lie within" and never shows a bracket character
    # (`test_exam_tier_never_shows_a_bracket_and_translates_centre_radius` and the oracle's
    # `_CENTRE_RADIUS_RE` both depend on this), and keeps `_INSTRUCTION` byte-identical in `text`
    # and `latex`. Chosen AFTER the numeric draw so the per-seed numbers are unchanged.
    a_txt = rational_text(ahat)
    centre_txt, radius_txt = rational_text(centre), rational_text(radius)
    template = rng.choice(["lie", "known", "certified"])
    if template == "lie":
        text = (
            f"An approximation ahat = {a_txt} is known to lie strictly inside a certified "
            f"interval: the true value is certified to lie within {radius_txt} of {centre_txt}. "
            f"{_INSTRUCTION}"
        )
    elif template == "known":
        text = (
            f"It is known that ahat = {a_txt} lies strictly inside a certified interval: "
            f"the true value is certified to lie within {radius_txt} of {centre_txt}. "
            f"{_INSTRUCTION}"
        )
    else:
        text = (
            f"ahat = {a_txt} is an approximation known to sit strictly within a certified "
            f"interval, where the true value is certified to lie within {radius_txt} of "
            f"{centre_txt}. {_INSTRUCTION}"
        )
    latex = rf"\hat a = {a_txt},\quad |x - {centre_txt}| \le {radius_txt}. \quad \text{{{_INSTRUCTION}}}"

    d_lo, d_hi = ahat - lo, hi - ahat
    closer = min(d_lo, d_hi)
    absolute_bound = max(d_lo, d_hi)
    half_width = radius
    dc = (closer, closer / ahat)
    dw = (half_width, half_width / ahat)
    # textbook-relative-error-uses-worst-case-true-value: divides by lo, the
    # endpoint nearest 0 -- the smallest possible |true value| in the
    # certified interval, the rigorous textbook reading the prompt's stated
    # convention now excludes.
    dd = (absolute_bound, absolute_bound / lo)
    return _finish(
        text,
        latex,
        ahat,
        lo,
        hi,
        [
            f"Translate: lo = {centre_txt} - {radius_txt} = {rational_text(lo)}, "
            f"hi = {centre_txt} + {radius_txt} = {rational_text(hi)}."
        ],
        [dc, dw, dd],
    )


@generator(
    "p14d_numerical_error_bound",
    level=17,
    presets={
        "easy": dict(variant="midpoint"),
        "medium": dict(variant="offcentre"),
        "hard": dict(variant="fraction_near_miss"),
        "exam": dict(variant="centre_radius"),
    },
)
def p14d_numerical_error_bound(rng, variant="offcentre"):
    if variant == "midpoint":
        return _make_midpoint(rng)
    if variant == "offcentre":
        return _make_offcentre(rng)
    if variant == "fraction_near_miss":
        return _make_fraction_near_miss(rng)
    if variant == "centre_radius":
        return _make_centre_radius(rng)
    raise ValueError(f"unknown variant: {variant!r}")
