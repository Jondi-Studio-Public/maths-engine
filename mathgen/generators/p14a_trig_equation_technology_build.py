"""Solve trigonometric equations, technology as the platform (ACMMM043,
``senior-solve-equations-involving-trigonometric-functions-using-technology-and``).

Skill ``name`` (authoritative): "Solve equations
involving trigonometric functions using technology, and algebraically in
simple cases".

Deliberately kept SEPARATE from the sibling practical-right-triangle skill
 even though both share the ``right-triangle-angle``
prerequisite family. The prerequisite skill's whole domain is "an inverse
trig function of a SIDE RATIO in a right triangle" (answer always in
``(0deg, 90deg)`` because it is a triangle's own interior angle). This skill
never frames a question as "opposite/adjacent side of a right triangle" --
every question here is a general trigonometric EQUATION over a stated
interval (``sin(x) = k``, ``a*trig(bx) + c = d``) that can have two or more
solutions, and the unknown is a solution of an equation, not a triangle's
angle.

Four tiers, each a genuine escalation of DECISION, not magnitude:

- **easy** (``variant="direct"``): ``sin(x) = k`` or ``cos(x) = k``,
  ``x`` in ``[0, 360)`` degrees, exactly two solutions by construction
  (``k`` strictly between -1 and 1, away from 0/+-1 by a margin). Names
  "the smaller solution" or "the larger solution" -- unconditionally
  well-defined once exactly two solutions are confirmed.
- **medium** (``variant="coefficient"``): ``a * trig(x) = d``, the learner
  must isolate ``k = d / a`` before solving -- a genuinely different first
  step, same "smaller/larger of two" naming once isolated.
- **hard** (``variant="multi_period"``): ``trig(b*x) = k`` with ``b >= 2``,
  so ``[0, 360)`` contains ``2*b`` solutions -- the learner must solve for
  the general angle in ``b*x``-space first and then divide by ``b``, which
  multiplies the solution count. Named by an explicit ordinal ("the n-th
  smallest solution in [0 deg, 360 deg), counting from 1"), verified to
  exist before it is offered.
- **exam** (``variant="technology_applied"``, ``stages: [exam]``): an
  applied periodic-quantity context (tide height) states every constant
  (``a``, ``b``, ``c``, ``d``, the interval) explicitly; ``k = (d - c) / a``
  is kept away from every standard angle's sine/cosine value so the answer
  is a genuine technology/decimal solve, not a recognisable exact angle.
  Named the same way as hard (a verified ordinal); reported to 3 decimal
  places (hard/easy/medium all report to 1 -- the exam tier's own extra
  precision is a deliberate, disclosed choice, not an inconsistency).

**Naming-uniqueness discipline (the central requirement here).** A
phrase like "the solution greater than 180 deg" is NOT safe in general
(``sin(x) = k`` for ``k < 0`` puts BOTH solutions in ``(180, 360)``), so no
interval-fragment naming is used anywhere in this module. Only
"smaller/larger of exactly two" (easy/medium) and "n-th smallest of the
verified count" (hard/exam) are used, and every draw is checked, not
assumed: ``_enumerate_solutions`` (a numeric sign-change scan + bisection
refine, NEVER calling the same general-solution-formula arithmetic that
``_algebraic_solutions`` below uses to build the candidate answer) confirms
the exact solution count for every candidate draw before it is accepted;
a draw whose numeric count disagrees with the naming rule's assumption is
rejected and resampled. This is the SAME discipline
``tests/test_p14a_trig_equation_technology_oracle.py`` applies completely
independently (import nothing from this module; parse the rendered question
text and re-derive/re-enumerate from scratch).

**Independent, in-generator check** (a genuinely different route from the
algebra that built the answer): ``_algebraic_solutions`` derives every
solution symbolically (principal value from ``asin``/``acos``, reflected
into the second base solution, replicated across every full period the
domain covers). ``_enumerate_solutions`` never uses that formula -- it scans
the interval with a fine numeric step, finds every sign change of
``trig(b*x) - k``, refines each with a plain bisection loop, deduplicates
and sorts. Every accepted draw asserts the two routes agree exactly on both
the COUNT and the VALUES (within a tight numeric tolerance), via
``_confirm_against_scan``.

**Misconceptions**, each proven a genuinely different numeric value from the
answer and from every other candidate, per seed, via a plain gap check at
the tier's own stated-precision tolerance (never assumed):

- ``wrong-ordinal``: offers a different, adjacent member of the SAME
  verified solution list (the other of "smaller/larger" on easy/medium; the
  ``n-1``-th or ``n+1``-th, clamped in range, on hard/exam) -- a real root
  of the same equation, just not the one asked for.
- ``sign-flip`` (easy only): reads the stated ``k`` with the wrong sign and
  reports the corresponding base solution of ``trig(x) = -k``. Deliberately
  NOT "the raw principal value from asin/acos" -- that value is always
  numerically IDENTICAL to one of the equation's own two genuine base
  solutions (``s1`` in ``_base_solutions_deg`` is exactly the wrapped
  principal), so it can never be a distinct wrong answer; discovered and
  fixed during development, not shipped as designed.
- ``forgot-divide-by-a`` (medium/exam only): forgets only the final division
  by ``a`` -- medium solves ``trig(x) = d`` directly (never dividing by
  ``a``); exam correctly subtracts ``c`` first (``d - c``) but then also
  forgets to divide by ``a``. Only offered when that un-divided value is
  itself a valid sine/cosine value (``|value| < 0.97``); otherwise dropped
  (``asin``/``acos`` would be undefined for it), which is why medium/exam
  sometimes ship fewer than 3 distractors -- disclosed, not hidden.
- ``forgot-divide-by-b`` (hard/exam only): reports the un-recovered
  ``b * x`` angle itself instead of dividing back by ``b`` at the last
  step -- always a different physical quantity from ``x`` since ``b >= 2``.
- ``degrees-as-radians``: reports ``math.radians(answer)`` -- the numerically
  correct angle, but as if the learner forgot the whole computation was in
  degrees and read the internal radian value straight off a calculator.
  Always wildly different in magnitude from a genuine degree-valued (or
  hour-valued) answer.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import (
    check,
    generator,
    numeric_answer,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)

_DP_STANDARD = 1
_DP_EXAM = 3
_TOL_STANDARD = stated_precision_tolerance(_DP_STANDARD)
_TOL_EXAM = stated_precision_tolerance(_DP_EXAM)
_RESAMPLE_ATTEMPTS = 80
_SCAN_POINTS = 6000

# Nice/standard sine-and-cosine values the exam tier's own k must stay away
# from (by an absolute margin), so the "technology" solve is genuine and not
# a disguised standard-angle lookup.
_NICE_VALUES = [0.0, 0.5, 2**0.5 / 2, 3**0.5 / 2, 1.0]
_NICE_MARGIN = 0.03


def _to_fraction(value: float) -> Fraction:
    return Fraction(value).limit_denominator(10**9)


def _round(value: float, dp: int) -> Fraction:
    return round_half_up(_to_fraction(value), dp)


def _fmt(value: Fraction, dp: int) -> str:
    return stated_precision_string(value, dp)


def _trig(kind: str, angle_deg: float) -> float:
    rad = math.radians(angle_deg)
    return math.sin(rad) if kind == "sin" else math.cos(rad)


def _base_solutions_deg(kind: str, k: float) -> list[float]:
    """The two base solutions of ``trig(angle) = k`` in one period
    ``[0, 360)`` -- the FIRST (algebraic) route: a principal value from
    ``asin``/``acos``, reflected into its quadrant-partner base solution."""
    if kind == "sin":
        principal = math.degrees(math.asin(k))  # in [-90, 90]
        s1 = principal % 360.0
        s2 = (180.0 - principal) % 360.0
    else:
        principal = math.degrees(math.acos(k))  # in [0, 180]
        s1 = principal % 360.0
        s2 = (360.0 - principal) % 360.0
    return sorted([s1, s2])


def _algebraic_solutions(kind: str, b: int, k: float, hi: float) -> list[float]:
    """Every solution of ``trig(b*x) = k`` for ``x`` in ``[0, hi)``, built by
    the ALGEBRAIC route: the two base solutions in one ``b*x``-period,
    replicated across every full period ``b*x`` sweeps through as ``x``
    ranges over ``[0, hi)``, then divided back by ``b``. Requires
    ``b * hi / 360`` to be a whole number of periods (true for every domain
    this module ever constructs -- checked, not assumed)."""
    periods_f = b * hi / 360.0
    periods = round(periods_f)
    check(abs(periods_f - periods) < 1e-6, "domain does not cover a whole number of b*x periods")
    base = _base_solutions_deg(kind, k)
    sols = []
    for s in base:
        for p in range(periods):
            angle = s + 360.0 * p
            sols.append(angle / b)
    return sorted(sols)


def _enumerate_solutions(kind: str, b: int, k: float, hi: float, n_points: int = _SCAN_POINTS) -> list[float]:
    """Every solution of ``trig(b*x) = k`` for ``x`` in ``[0, hi)``, built by
    a SECOND, numeric route that never touches ``asin``/``acos``: a fine
    scan for sign changes of ``trig(b*x) - k``, each refined by plain
    bisection, deduplicated and sorted. This is the in-generator
    independent check's own route -- a wholly separate, hand-written
    numeric enumerator lives in the oracle test file, never importing this
    one."""
    step = hi / n_points

    def f(x: float) -> float:
        return _trig(kind, b * x) - k

    roots: list[float] = []
    prev_x = 0.0
    prev_f = f(0.0)
    for i in range(1, n_points + 1):
        x = i * step
        cur_f = f(x)
        if prev_f == 0.0:
            roots.append(prev_x)
        elif prev_f * cur_f < 0.0:
            lo_x, hi_x = prev_x, x
            lo_f, hi_f = prev_f, cur_f
            for _ in range(60):
                mid = (lo_x + hi_x) / 2.0
                mid_f = f(mid)
                if lo_f * mid_f <= 0.0:
                    hi_x, hi_f = mid, mid_f
                else:
                    lo_x, lo_f = mid, mid_f
            roots.append((lo_x + hi_x) / 2.0)
        prev_x, prev_f = x, cur_f
    roots.sort()
    deduped: list[float] = []
    for r in roots:
        if not deduped or r - deduped[-1] > 1e-4:
            deduped.append(r)
    return deduped


def _confirm_against_scan(kind: str, b: int, k: float, hi: float, algebraic: list[float], expected_count: int) -> None:
    """The in-generator independent check: the numeric scan must agree with
    the algebraic route on both COUNT and VALUES. Raises (via ``check``) on
    disagreement -- this is what a caller resamples on, not a silent
    fallback."""
    scanned = _enumerate_solutions(kind, b, k, hi)
    check(len(scanned) == expected_count, f"expected {expected_count} solutions, numeric scan found {len(scanned)}")
    check(
        len(algebraic) == expected_count, f"expected {expected_count} solutions, algebraic route found {len(algebraic)}"
    )
    for a_val, s_val in zip(sorted(algebraic), sorted(scanned)):
        check(abs(a_val - s_val) < 1e-2, f"algebraic route ({a_val}) disagrees with numeric scan ({s_val})")


def _gap_ok(values: list[float], tol: Fraction) -> bool:
    tol_f = float(tol)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if abs(values[i] - values[j]) <= tol_f:
                return False
    return True


def _near_nice_value(k: float) -> bool:
    return any(abs(abs(k) - nice) < _NICE_MARGIN for nice in _NICE_VALUES)


# =============================================================================
# easy: sin(x) = k or cos(x) = k, x in [0, 360)
# =============================================================================


def _draw_k(rng) -> float:
    magnitude = round(rng.uniform(0.15, 0.90), 2)
    return magnitude * rng.choice([1, -1])


def _build_easy(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        kind = rng.choice(["sin", "cos"])
        k = _draw_k(rng)
        which = rng.choice(["smaller", "larger"])

        algebraic = _algebraic_solutions(kind, 1, k, 360.0)
        try:
            _confirm_against_scan(kind, 1, k, 360.0, algebraic, 2)
        except Exception:
            continue

        ans_val = algebraic[0] if which == "smaller" else algebraic[1]
        other_val = algebraic[1] if which == "smaller" else algebraic[0]
        if not rounds_unambiguously(ans_val, _DP_STANDARD):
            continue
        ans = _round(ans_val, _DP_STANDARD)

        sign_flip_base = _base_solutions_deg(kind, -k)
        sign_flip_val = sign_flip_base[0] if which == "smaller" else sign_flip_base[1]
        radians_confusion = math.radians(ans_val)

        cand_floats = [other_val, sign_flip_val, radians_confusion]
        if any(not rounds_unambiguously(v, _DP_STANDARD) for v in cand_floats):
            continue
        if not _gap_ok([ans_val] + cand_floats, _TOL_STANDARD):
            continue
        cands = [_round(v, _DP_STANDARD) for v in cand_floats]
        break
    else:
        raise ValueError("p14a_solve_trig_equation_technology: no collision-free easy draw found")

    text = (
        f"Solve {kind}(x) = {k:g} for x in the interval [0°, 360°). "
        f"Give the {which} solution correct to 1 decimal place."
    )
    latex = text
    answer_text = _fmt(ans, _DP_STANDARD)
    distractors = pick_distractors(answer_text, [_fmt(c, _DP_STANDARD) for c in cands], n=3)
    check(len(distractors) >= 1, "easy tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"{kind}(x) = {k:g} has base solutions at {algebraic[0]:.4f} deg and "
            f"{algebraic[1]:.4f} deg in [0, 360). The {which} one is {answer_text} deg."
        ],
        distractors=distractors,
    )


# =============================================================================
# medium: a * trig(x) = d, isolate k = d / a first
# =============================================================================


def _build_medium(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        kind = rng.choice(["sin", "cos"])
        k = _draw_k(rng)
        a_coeff = rng.choice([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6])
        d_val = round(a_coeff * k, 2)
        which = rng.choice(["smaller", "larger"])

        algebraic = _algebraic_solutions(kind, 1, k, 360.0)
        try:
            _confirm_against_scan(kind, 1, k, 360.0, algebraic, 2)
        except Exception:
            continue

        ans_val = algebraic[0] if which == "smaller" else algebraic[1]
        other_val = algebraic[1] if which == "smaller" else algebraic[0]
        if not rounds_unambiguously(ans_val, _DP_STANDARD):
            continue
        ans = _round(ans_val, _DP_STANDARD)

        cand_floats = [other_val]
        # forgot-isolate: solves trig(x) = d directly, only meaningful (and
        # only offered) when d itself is a valid sine/cosine value.
        if abs(d_val) < 0.97:
            forgot_base = _base_solutions_deg(kind, d_val)
            forgot_val = forgot_base[0] if which == "smaller" else forgot_base[1]
            cand_floats.append(forgot_val)
        cand_floats.append(math.radians(ans_val))

        if any(not rounds_unambiguously(v, _DP_STANDARD) for v in cand_floats):
            continue
        if not _gap_ok([ans_val] + cand_floats, _TOL_STANDARD):
            continue
        cands = [_round(v, _DP_STANDARD) for v in cand_floats]
        break
    else:
        raise ValueError("p14a_solve_trig_equation_technology: no collision-free medium draw found")

    text = (
        f"Solve {a_coeff} {kind}(x) = {d_val:g} for x in the interval [0°, 360°). "
        f"Give the {which} solution correct to 1 decimal place."
    )
    latex = text
    answer_text = _fmt(ans, _DP_STANDARD)
    distractors = pick_distractors(answer_text, [_fmt(c, _DP_STANDARD) for c in cands], n=3)
    check(len(distractors) >= 1, "medium tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"{kind}(x) = {d_val:g} / {a_coeff} = {k:g}. Base solutions at "
            f"{algebraic[0]:.4f} deg and {algebraic[1]:.4f} deg; the {which} one is "
            f"{answer_text} deg."
        ],
        distractors=distractors,
    )


# =============================================================================
# hard: trig(b*x) = k, b >= 2, [0, 360) holds 2*b solutions
# =============================================================================


def _build_hard(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        kind = rng.choice(["sin", "cos"])
        b = rng.choice([2, 3])
        k = _draw_k(rng)
        expected = 2 * b

        algebraic = _algebraic_solutions(kind, b, k, 360.0)
        try:
            _confirm_against_scan(kind, b, k, 360.0, algebraic, expected)
        except Exception:
            continue

        n = rng.randint(1, expected)
        ans_val = algebraic[n - 1]
        if not rounds_unambiguously(ans_val, _DP_STANDARD):
            continue

        adjacent_n = n - 1 if n > 1 else n + 1
        wrong_ordinal_val = algebraic[adjacent_n - 1]
        forgot_divide_val = b * ans_val
        radians_confusion = math.radians(ans_val)

        cand_floats = [wrong_ordinal_val, forgot_divide_val, radians_confusion]
        if any(not rounds_unambiguously(v, _DP_STANDARD) for v in cand_floats):
            continue
        if not _gap_ok([ans_val] + cand_floats, _TOL_STANDARD):
            continue
        ans = _round(ans_val, _DP_STANDARD)
        cands = [_round(v, _DP_STANDARD) for v in cand_floats]
        break
    else:
        raise ValueError("p14a_solve_trig_equation_technology: no collision-free hard draw found")

    ordinal = _ordinal_word(n)
    text = (
        f"Solve {kind}({b}x) = {k:g} for x in the interval [0°, 360°). "
        f"Give the {ordinal} smallest solution (counting from 1) correct to 1 decimal place."
    )
    latex = text
    answer_text = _fmt(ans, _DP_STANDARD)
    distractors = pick_distractors(answer_text, [_fmt(c, _DP_STANDARD) for c in cands], n=3)
    check(len(distractors) >= 1, "hard tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"{kind}({b}x) = {k:g} has {expected} solutions for {b}x in [0, {360 * b}) deg, "
            f"giving x in [0, 360): {', '.join(f'{s:.4f}' for s in algebraic)} deg. "
            f"The {ordinal} smallest is {answer_text} deg."
        ],
        distractors=distractors,
    )


def _ordinal_word(n: int) -> str:
    words = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th"}
    return words.get(n, f"{n}th")


# =============================================================================
# exam: a * trig(b*x) + c = d, applied tide-height context, technology solve
# =============================================================================

_EXAM_PERIOD_CHOICES = [1, 2, 3]  # periods of h(t) over t in [0, 24)


def _build_exam(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        kind = rng.choice(["sin", "cos"])
        periods = rng.choice(_EXAM_PERIOD_CHOICES)
        b = 15 * periods  # degrees per hour, so b * 24 = 360 * periods
        k = _draw_k(rng)
        if _near_nice_value(k):
            continue
        a_amp = round(rng.uniform(1.2, 3.8), 1)
        c_mean = round(rng.uniform(4.0, 9.0), 1)
        d_val = round(a_amp * k + c_mean, 3)
        expected = 2 * periods

        algebraic = _algebraic_solutions(kind, b, k, 24.0)
        try:
            _confirm_against_scan(kind, b, k, 24.0, algebraic, expected)
        except Exception:
            continue

        n = rng.randint(1, expected)
        ans_val = algebraic[n - 1]
        if not rounds_unambiguously(ans_val, _DP_EXAM):
            continue

        adjacent_n = n - 1 if n > 1 else n + 1
        wrong_ordinal_val = algebraic[adjacent_n - 1]

        cand_floats = [wrong_ordinal_val]
        if abs(d_val - c_mean) < 0.97:
            forgot_iso_base = _base_solutions_deg(kind, d_val - c_mean)
            forgot_iso_val = forgot_iso_base[min(n - 1, 1)] / b
            cand_floats.append(forgot_iso_val)
        cand_floats.append(b * ans_val)

        if any(not rounds_unambiguously(v, _DP_EXAM) for v in cand_floats):
            continue
        if not _gap_ok([ans_val] + cand_floats, _TOL_EXAM):
            continue
        ans = _round(ans_val, _DP_EXAM)
        cands = [_round(v, _DP_EXAM) for v in cand_floats]
        break
    else:
        raise ValueError("p14a_solve_trig_equation_technology: no collision-free exam draw found")

    ordinal = _ordinal_word(n)
    text = (
        f"The height of the tide at a harbour, h(t) metres, t hours after midnight, is modelled by "
        f"h(t) = {a_amp:g} {kind}({b}t°) + {c_mean:g} for 0 <= t < 24. Find the {ordinal} smallest "
        f"time t (in hours, counting from 1) at which h(t) = {d_val:g} m. "
        f"Give your answer correct to 3 decimal places."
    )
    latex = text
    answer_text = _fmt(ans, _DP_EXAM)
    distractors = pick_distractors(answer_text, [_fmt(c, _DP_EXAM) for c in cands], n=3)
    check(len(distractors) >= 1, "exam tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"{kind}({b}t) = ({d_val:g} - {c_mean:g}) / {a_amp:g} = {k:.4f}. "
            f"{expected} solutions for t in [0, 24): "
            f"{', '.join(f'{s:.4f}' for s in algebraic)}. "
            f"The {ordinal} smallest is {answer_text} h."
        ],
        distractors=distractors,
    )


@generator(
    "p14a_solve_trig_equation_technology",
    level=11,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="coefficient"),
        "hard": dict(variant="multi_period"),
        "exam": dict(variant="technology_applied"),
    },
)
def p14a_solve_trig_equation_technology(rng, variant="direct"):
    """Solve a trigonometric equation over a stated interval, naming which
    solution is wanted so the answer is always unique by construction:
    easy: sin(x)=k or cos(x)=k, [0,360), "smaller"/"larger" of exactly two.
    medium: a*trig(x)=d, isolate k=d/a first, same naming.
    hard: trig(bx)=k, b>=2, [0,360) holds 2b solutions, named by a verified
    ordinal. exam: a*trig(bx)+c=d, an applied tide-height context, k kept
    away from every standard angle, named the same way as hard."""
    if variant == "direct":
        return _build_easy(rng)
    if variant == "coefficient":
        return _build_medium(rng)
    if variant == "multi_period":
        return _build_hard(rng)
    if variant == "technology_applied":
        return _build_exam(rng)
    raise ValueError(f"unknown variant: {variant!r}")
