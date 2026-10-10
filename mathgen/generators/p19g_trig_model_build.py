"""Periodic-motion modelling and practical evaluation, both ``acara-senior-v8-4``, domain ``trigonometry``.

Two generators, a deliberate BOUNDARY pair -- see each docstring:

- ``p19g_sinusoid_amplitude_period``
  (``senior-model-periodic-motion-using-sine-and-cosine-functions``,
  ACMSM050) -- given a FULLY STATED sine/cosine function
  y = A*sin(B*x + C) + D (or cosine), read off its amplitude (|A|) and
  period (2*pi/|B|). Never asks "at t = ...". `answer: pair`.
- ``p19g_sinusoid_evaluate_context``
  (``senior-identify-contexts-suitable-for-modelling-by-trigonometric-functions``,
  ACMMM042) -- given a real-world periodic SCENARIO with amplitude, period
  and starting condition stated in WORDS (never a bare function), substitute
  a stated time and compute the resulting quantity. Never asks the learner
  to state the bare amplitude or period. `answer: number`.

**GRADER-KIND DESIGN (load-bearing, decided after probing `mathgen.grading`
directly -- do not deviate):** `grade_pair` (`mathgen/grading/typed.py`) extracts
numeric literals from each labelled segment via `extract_values`; a
symbolic period like "2*pi/3" never grades correctly through it. Every
function in ``p19g_sinusoid_amplitude_period`` therefore states its
x-coefficient `B` as a multiple of pi with the FORM `pi/k` (or, for the
harder tiers, an unsimplified `m*pi/n` with `n = m*k`) -- `period =
2*pi/|B| = 2*k` is then always a plain positive integer, and `amplitude =
|A|` is always a plain integer or a terminating decimal. Both grade cleanly
through `grade_pair`. `x` is always stated in the function's own time unit
(seconds/minutes/hours) so the period's unit is never ambiguous.

Pi: real `sympy.pi` / `math.pi` throughout, round once at the end. Both generators'
independent checks go through `sympy.pi` symbolically or through an exact `Fraction`- backed `sympy.Rational`,
never a truncated numeric approximation of pi.
"""

from __future__ import annotations

import math
from fractions import Fraction

from sympy import Rational, pi as _spi, sin as _ssin, N as _sympy_N

from ..core import (
    check,
    check_equivalent,
    exact_decimal_string,
    format_pi_multiple,
    generator,
    numeric_answer,
    pick_distractors,
    rational_text,
    round_half_up,
    rounded_decimal_string,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)

# ===========================================================================
# 1. p19g_sinusoid_amplitude_period
# ===========================================================================

# Each scenario carries its OWN `k_lo`/`k_hi` range for `period = 2*k` in ITS OWN unit, so the
# period lands in a physically-realistic band for that real object -- never a single shared `k`
# range applied to every scenario regardless of unit. Real-world anchors: a Ferris wheel takes a
# couple of minutes per rotation (2-30 minutes here); a tide's period is about 12.4 hours (10/12/14
# hours here, the closest this generator's plain-integer-period design can land to that real value);
# a large pendulum swings over several seconds (4-30 seconds, unchanged -- never flagged); a piston
# (a small engine's) cycles many times per second, so its period is stated in milliseconds (20-100
# ms here, i.e. 10-50 Hz).
_CONTEXTS = [
    # A "height above..." can never be negative, so the two height contexts carry `height=True`:
    # tiers whose model has a vertical shift D >= |A| (hard/exam) keep the height noun; tiers with D
    # = 0 (easy/medium) use `signed_subject`, wording for a signed offset from a reference level.
    {
        "subject": "A Ferris wheel's height above the ground, in metres,",
        "unit": "minutes",
        "k_lo": 2,
        "k_hi": 15,
        "height": True,
        "amp_lo": 3,
        "amp_hi": 12,
        "signed_subject": "A Ferris wheel's vertical displacement from the height of its axle, in metres,",
    },
    {
        "subject": "The tide's height above the sea floor, in metres,",
        "unit": "hours",
        "k_lo": 5,
        "k_hi": 7,
        "height": True,
        "amp_lo": 1,
        "amp_hi": 6,
        "signed_subject": "The tide's displacement from mean sea level, in metres,",
    },
    {
        "subject": "A pendulum's horizontal displacement from its rest position, in centimetres,",
        "unit": "seconds",
        "k_lo": 2,
        "k_hi": 15,
        "amp_lo": 1,
        "amp_hi": 12,
    },
    {
        "subject": "A piston's distance from the centre of its cylinder, in centimetres,",
        "unit": "milliseconds",
        "k_lo": 10,
        "k_hi": 50,
        "amp_lo": 1,
        "amp_hi": 12,
    },
]

#: Phase constants offered as multiples of pi (Fraction coefficients).
_PHASE_CHOICES = [
    Fraction(0),
    Fraction(1, 6),
    Fraction(1, 4),
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(-1, 4),
    Fraction(-1, 2),
    Fraction(-1, 6),
]


def _phase_suffix(phase_frac: Fraction) -> str:
    """ "" if phase is 0, else " + pi/4" / " - pi/2" etc., built from
    `format_pi_multiple` so it can never disagree with the coefficient's own
    canonical spelling."""
    if phase_frac == 0:
        return ""
    phase_str = format_pi_multiple(phase_frac)
    if phase_str.startswith("-"):
        return f" - {phase_str[1:]}"
    return f" + {phase_str}"


def _build_function(kind: str, a_str: str, b_str: str, phase_frac: Fraction, d: int) -> str:
    """No bare multiplication signs anywhere (`mathgen/text_quality.py`'s
    `star_operator` prompt check) -- "11sin((pi/2)t)", never "11*sin((pi/2)*t)",
    matching the "no bare multiplication signs" style already established by
    `p5e_vector_calculus_build.py::_fmt_component_medium`."""
    inner = f"({b_str})t{_phase_suffix(phase_frac)}"
    fn = f"{a_str}{kind}({inner})"
    if d > 0:
        fn += f" + {d}"
    elif d < 0:
        fn += f" - {abs(d)}"
    return fn


def _verify_amplitude_period(
    amplitude_source: Fraction, places: int, b_num: int, b_den: int, period_int: int, amplitude_frac: Fraction
) -> None:
    """Independent second route: the period is recomputed here by cancelling `2*pi / (B*pi)` SYMBOLICALLY with
    real `sympy.pi`, from the exact (possibly unsimplified) `B = b_num / b_den` fraction -- never by
    re-running construction's own `period = 2*k` assignment. The amplitude is independently re-derived by
    rounding a fresh `sympy.Rational` built straight from `amplitude_source`'s own numerator/denominator (the
    pre-rounding |A|, as displayed on the harder tiers) to `places` decimals via `sympy`'s own rational
    rounding, never from any downstream formatted string."""
    b_sym = Rational(b_num, b_den) * _spi
    period_expr = 2 * _spi / b_sym
    check_equivalent(period_expr, period_int, "period disagrees with 2*pi/|B| computed via sympy pi-cancellation")
    exact_amplitude = abs(Rational(amplitude_source.numerator, amplitude_source.denominator))
    scale = Rational(10) ** places
    rounded_amplitude = Rational(math.floor(exact_amplitude * scale + Rational(1, 2)), 1) / scale
    check_equivalent(
        rounded_amplitude,
        Rational(amplitude_frac.numerator, amplitude_frac.denominator),
        "amplitude disagrees with |A| rounded via a fresh sympy Rational route",
    )


def _draw_exam_amplitude(rng, context) -> Fraction:
    """A signed amplitude with at most 2 decimal places (never a 3-d.p. third), inside the context's own
    realistic range, whose last digit is neither 0 (no rounding needed) nor 5 (a rounding tie at 1 d.p.)."""
    while True:
        n = rng.randint(100 * context["amp_lo"] + 1, 100 * context["amp_hi"] - 1)
        if n % 10 not in (0, 5):
            return Fraction(n * rng.choice([1, -1]), 100)


def _fmt_amp(value: Fraction, places: int) -> str:
    if places == 0:
        return str(int(value))
    return stated_precision_string(value, places)


@generator(
    "p19g_sinusoid_amplitude_period",
    level=19,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def p19g_sinusoid_amplitude_period(rng, variant="easy"):
    """Given a fully stated sine/cosine function modelling a periodic
    motion, state its amplitude and period as a labelled pair
    ``{"amplitude": ..., "period": ...}``. `B` is always constructed as a
    (possibly unsimplified) multiple of pi so the period is a plain
    positive integer, gradeable by `grade_pair`.

    Scope: "Given a sine or
    cosine function of the form y = A*sin(B*x + C) + D (or the cosine
    equivalent) modelling a periodic motion, state its amplitude and its
    period." `C` and `D` are always fully stated but never asked for --
    only amplitude and period. Never asks "at t = ..." (that is the sibling
    skill `p19g_sinusoid_evaluate_context`, below)."""
    context = rng.choice(_CONTEXTS)
    unit = context["unit"]
    k = rng.randint(context["k_lo"], context["k_hi"])

    if variant == "easy":
        kind = "sin"
        a_frac = Fraction(rng.randint(context["amp_lo"], context["amp_hi"]))
        phase_frac = Fraction(0)
        d = 0
        unsimplified = False
        places = 0
    elif variant == "medium":
        kind = rng.choice(["sin", "cos"])
        if kind == "sin":
            phase_frac = rng.choice([p for p in _PHASE_CHOICES if p != 0])
        else:
            phase_frac = rng.choice(_PHASE_CHOICES)
        a_frac = Fraction(
            rng.choice([v * sg for v in range(context["amp_lo"], context["amp_hi"] + 1) for sg in (1, -1)])
        )
        d = 0
        unsimplified = False
        places = 0
    elif variant == "hard":
        kind = rng.choice(["sin", "cos"])
        phase_frac = rng.choice(_PHASE_CHOICES)
        half_int = rng.choice(
            [v * sg for v in range(2 * context["amp_lo"], 2 * context["amp_hi"] + 1) for sg in (1, -1)]
        )
        a_frac = Fraction(half_int, 2)
        d = rng.randint(1, 20) * rng.choice([1, -1])
        unsimplified = True
        places = 1
    elif variant == "exam":
        kind = rng.choice(["sin", "cos"])
        phase_frac = rng.choice([p for p in _PHASE_CHOICES if p != 0])
        a_frac = _draw_exam_amplitude(rng, context)
        d = rng.randint(5, 25) * rng.choice([1, -1])
        unsimplified = True
        places = 1
    else:
        raise ValueError(f"unknown variant {variant!r}")

    if not unsimplified:
        b_num, b_den = 1, k
        b_str = format_pi_multiple(Fraction(1, k))
    else:
        m = rng.choice([2, 3, 4, 5, 6])
        n = m * k
        b_num, b_den = m, n
        b_str = f"{m}pi/{n}"  # unsimplified, no bare `*` (matches format_pi_multiple's own style)
    period_int = 2 * k

    if variant in ("easy", "medium"):
        a_str = str(int(a_frac))
        amplitude_source = abs(a_frac)
        amplitude_frac = amplitude_source
        displayed_a_frac = a_frac
    else:
        # The DISPLAYED literal in the question text is what the learner reads
        # and rounds from, so the graded answer is derived from THAT
        # literal, never from the exact untruncated fraction.
        a_display_str = rounded_decimal_string(a_frac, 3)
        a_str = a_display_str
        displayed_a_frac = Fraction(a_display_str)
        amplitude_source = abs(displayed_a_frac)
        if variant == "exam":
            for _resample in range(10):
                if rounds_unambiguously(float(amplitude_source), places):
                    break
                # The draw already excludes ties; resample rather than ship one.
                a_frac = _draw_exam_amplitude(rng, context)
                a_display_str = rounded_decimal_string(a_frac, 3)
                a_str = a_display_str
                displayed_a_frac = Fraction(a_display_str)
                amplitude_source = abs(displayed_a_frac)
            else:
                check(False, "could not draw an unambiguously-rounding amplitude in 10 attempts")
        amplitude_frac = round_half_up(amplitude_source, places)

    subject = context["subject"]
    if context.get("height"):
        if variant in ("easy", "medium"):
            # D = 0 makes the model signed: name it a displacement, not a height.
            subject = context["signed_subject"]
        elif d < math.ceil(amplitude_source):
            # A height above a fixed level must stay >= 0: D >= |A|.
            d = math.ceil(amplitude_source) + rng.randint(0, 10)
        check(
            variant in ("easy", "medium") or d - amplitude_source >= 0, "height context would model a negative height"
        )

    _verify_amplitude_period(amplitude_source, places, b_num, b_den, period_int, amplitude_frac)

    fn_expr = _build_function(kind, a_str, b_str, phase_frac, d)

    if variant == "exam":
        # Two DIFFERENT stated decimal-place counts (amplitude: 1, period:
        # 0 -- it is always a plain integer, never rounded) so
        # `mathgen.text_quality.requested_rounding` sees an intentionally
        # AMBIGUOUS whole-text precision and declines to enforce one single
        # count onto every number in `answer_text` (the same escape valve
        # `p5b_complex_plane_build.py`'s "r rounded to 2 decimal places, and
        # theta ... rounded to 1 decimal place" uses for its own two-
        # different-precisions pair answer).
        question_text = (
            f"{subject} t {unit} after tracking begins, is modelled by "
            f"y = {fn_expr}. State the amplitude, rounded to {places} decimal place"
            f"{'s' if places != 1 else ''}, and the period, to 0 decimal places, "
            f"in {unit}."
        )
    else:
        question_text = (
            f"{subject} t {unit} after tracking begins, is modelled by "
            f"y = {fn_expr}. State the amplitude and the period (in {unit}) of this model."
        )

    amplitude_str = _fmt_amp(amplitude_frac, places)
    answer_text = f"amplitude = {amplitude_str}, period = {period_int}"
    # At `places > 0` the CANONICAL always displays a decimal point (e.g.
    # "9.5", "2.0"), so the answer value is always a float there too --
    # never an `int` that would silently disagree with its own displayed
    # precision (a whole-number amplitude like 2 must still answer `2.0`,
    # matching "amplitude = 2.0" in the text).
    answer_amplitude = float(amplitude_frac) if places > 0 else numeric_answer(amplitude_frac)
    answer = {"amplitude": answer_amplitude, "period": period_int}

    # The 2127
    # set perturbed BOTH fields of every candidate, so no option shared the
    # key's amplitude (the coefficient printed in the function) and the
    # learner could pick the key without working out the period; it also
    # offered absurd reciprocal periods ("1/5"). Every distractor here is an
    # INDEPENDENT wrong route that perturbs one or both columns, always to a
    # plausible positive value:
    #   k    period = pi/B, the factor 2 dropped          (period column)
    #   d4   period = 4*pi/B, the factor 2 doubled        (period column)
    #   a2   amplitude read as peak-to-trough, 2|A|       (amplitude column)
    #   sh   vertical shift counted in the amplitude      (amplitude column)
    #   sg   the sign of A kept                           (amplitude column)
    # A "route" is an (amplitude, period) pair; the combos below always keep
    # the key's amplitude in some option and its period in another, and the
    # picker rejects any combo in which the key would be the SOLE option
    # holding the most frequent value in both columns (the 2127 tell).
    p2 = str(period_int)
    k_txt = str(k)
    d4_txt = str(period_int * 2)
    double_amp = _fmt_amp(amplitude_frac * 2, places)
    routes = {
        "k": (amplitude_str, k_txt),
        "d4": (amplitude_str, d4_txt),
        "a2p": (double_amp, p2),
        "a2k": (double_amp, k_txt),
        "a2d4": (double_amp, d4_txt),
    }
    if d != 0:
        shifted_amp = _fmt_amp(amplitude_frac + abs(d), places)
        routes["shp"] = (shifted_amp, p2)
        routes["shk"] = (shifted_amp, k_txt)
    if a_frac < 0:
        routes["sgp"] = ("-" + amplitude_str, p2)
        routes["sgk"] = ("-" + amplitude_str, k_txt)
    combos = [
        ("k", "a2p", "a2k"),
        ("k", "a2p", "a2k"),
        ("d4", "a2p", "a2d4"),
        ("k", "a2p", "a2d4"),
        ("d4", "a2p", "a2k"),
        ("k", "shp", "shk"),
        ("d4", "shp", "shk"),
        ("k", "a2p", "shk"),
        ("k", "shp", "a2k"),
        ("k", "sgp", "sgk"),
        ("d4", "sgp", "sgk"),
        ("k", "sgp", "a2k"),
    ]
    key_pair = (amplitude_str, str(period_int))

    def _combo_ok(combo):
        if any(name not in routes for name in combo):
            return False
        opts = [key_pair] + [routes[name] for name in combo]
        if len(set(opts)) != 4:
            return False
        if not any(o[0] == key_pair[0] for o in opts[1:]):
            return False
        if not any(o[1] == key_pair[1] for o in opts[1:]):
            return False
        amp_counts = [sum(x[0] == o[0] for x in opts) for o in opts]
        per_counts = [sum(x[1] == o[1] for x in opts) for o in opts]
        top = [i for i in range(4) if amp_counts[i] == max(amp_counts) and per_counts[i] == max(per_counts)]
        return top != [0]

    valid = [c for c in combos if _combo_ok(c)]
    check(bool(valid), "no independent-route distractor combo is valid")
    chosen = rng.choice(valid)
    candidates = [f"amplitude = {routes[name][0]}, period = {routes[name][1]}" for name in chosen]

    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) >= 1, "no non-colliding distractor survived")

    return dict(
        text=question_text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=rf"\text{{amplitude}} = {amplitude_str},\; \text{{period}} = {period_int}",
        working=[
            f"amplitude = |A| = |{a_str}| = {amplitude_str}",
            f"period = 2pi/|B| = 2pi/({b_str}) = {period_int} {unit}",
        ],
        distractors=distractors,
    )


# ===========================================================================
# 2. p19g_sinusoid_evaluate_context
# ===========================================================================

_CONTEXTS2 = [
    {
        "subject": "A Ferris wheel's height above the ground",
        "qty": "the height",
        "unit_len": "metres",
        "unit_time": "seconds",
        "key": "ferris",
        "amp_lo": 3,
        "amp_hi": 20,
    },
    {
        "subject": "The tide's height above the sea floor",
        "qty": "the height",
        "unit_len": "metres",
        "unit_time": "hours",
        "key": "tide",
        "amp_lo": 2,
        "amp_hi": 6,
    },
    {
        "subject": "A piston's distance from the base of its cylinder",
        "qty": "the distance",
        "unit_len": "centimetres",
        "unit_time": "seconds",
        "key": "piston",
        "amp_lo": 2,
        "amp_hi": 20,
    },
]

#: Hard-tier "convert units first" pairing, ONE per scenario (defect avoided:
#: a shared cross-scenario unit table let a "tide" draw a period in
#: MINUTES and a "Ferris wheel"/"piston" draw one in HOURS -- physically
#: absurd for all three). Each entry is
#: ``(period_unit, t_unit, factor, period_lo, period_hi)`` where
#: ``period_unit`` is the COARSER unit the period itself is stated in,
#: ``t_unit`` is the FINER unit the stated evaluation time is given in
#: (``factor`` = how many ``t_unit`` per one ``period_unit``), and the period
#: range is chosen to stay physically sane for that scenario: a tide cycles
#: over several hours (never minutes or seconds), a Ferris wheel rotates in
#: low minutes (never hours), and a piston strokes in a few seconds, so its
#: conversion partner is the sub-second ``milliseconds`` unit (never minutes
#: or hours).
_HARD_UNIT_CONFIG = {
    "ferris": ("minutes", "seconds", 60, 2, 4),
    "tide": ("hours", "minutes", 60, 8, 14),
    "piston": ("seconds", "milliseconds", 1000, 2, 5),
}

#: Closed set of starting-condition phrases, each with exactly one
#: defensible phase reading against a SINE base model
#: (`quantity(t) = D + A*sin(2*pi*t/P + phase)`). "starting at its mean
#: height and rising" pins phase=0 (sin(0)=0, rising because sin is
#: increasing at 0 for the always-positive period P); "starting at its
#: maximum"/"minimum" pin the +-pi/2 quadrantal phases directly. No other
#: reading of any of the three phrases is consistent with this fixed
#: sine-base convention -- "at its mean height and FALLING" is never one of
#: the three phrases this generator emits, so the "rising" qualifier is
#: never ambiguous.
_STARTS = [
    ("starting at its mean height and rising", Fraction(0)),
    ("starting at its maximum", Fraction(1, 2)),
    ("starting at its minimum", Fraction(-1, 2)),
]


#: Easy-tier fractions of one period, in words (odd eighths only -- see the
#: easy branch for why every pair with a `_STARTS` phase avoids sin = 0).
_EASY_FRAC_WORDS = {
    Fraction(1, 8): "one eighth",
    Fraction(3, 8): "three eighths",
    Fraction(5, 8): "five eighths",
    Fraction(7, 8): "seven eighths",
}


def _quantity(a: int, d: int, period: float, c_mult: Fraction, t: float) -> float:
    """Construction's own fast route: plain `math.sin`/`math.pi`."""
    angle = 2 * math.pi * t / period + float(c_mult) * math.pi
    return d + a * math.sin(angle)


def _quantity_independent(a: int, d: int, period_frac: Fraction, c_mult: Fraction, t_frac: Fraction) -> float:
    """Independent second route: builds the whole angle as an EXACT
    `sympy.Rational`-scaled multiple of `sympy.pi` (never a float
    approximation of pi until `N()` evaluates the final numeric value),
    and evaluates `sympy.sin` rather than `math.sin`."""
    angle = (
        2
        * _spi
        * Rational(t_frac.numerator, t_frac.denominator)
        / Rational(period_frac.numerator, period_frac.denominator)
        + Rational(c_mult.numerator, c_mult.denominator) * _spi
    )
    value = Rational(d) + Rational(a) * _ssin(angle)
    return float(_sympy_N(value, 30))


@generator(
    "p19g_sinusoid_evaluate_context",
    level=19,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def p19g_sinusoid_evaluate_context(rng, variant="easy"):
    """Given a real-world periodic scenario with amplitude, period and
    starting condition stated in WORDS (never as a bare function), compute
    the numeric value of the modelled quantity at a stated time.

    Scope: "Given a stated
    real-world periodic scenario modelled by a sinusoidal function with
    stated amplitude, period and starting condition, compute the numeric
    value of the modelled quantity (for example height or depth) at a
    stated time." Never asks the learner to state the bare amplitude or
    period (that is the sibling skill `p19g_sinusoid_amplitude_period`,
    above) -- the final answer is always the evaluated quantity.

    `D >= A` on every draw (checked explicitly, not just by construction)
    so the modelled quantity never goes negative."""
    context = rng.choice(_CONTEXTS2)
    unit_len = context["unit_len"]
    dp = 1

    if variant == "exam":
        # Inverse-solve variant does its own draws below.
        return _evaluate_context_exam(rng, context, dp)

    a = rng.randint(context["amp_lo"], context["amp_hi"])
    d = a + rng.randint(1, 15)
    check(d >= a, "D must be >= A so the modelled quantity is never negative")

    if variant == "easy":
        period = rng.choice([8, 12] if context["key"] == "tide" else [8, 12, 16, 20, 24])  # multiples of 4
        start_phrase, c_mult = rng.choice(_STARTS)
        # Odd eighths of a period: the angle is 2*pi*frac + c*pi with c in
        # {0, +-1/2}, i.e. an odd multiple of pi/4, so |sin| = sqrt(2)/2 on
        # every (start, frac) pair and the key never collapses onto the mean.
        frac = rng.choice(list(_EASY_FRAC_WORDS))
        t = period * frac
        period_unit = t_unit = context["unit_time"]
        t_frac = t
        # Redraw the amplitude/mean (never the displayed value) until the
        # irrational a*sqrt(2)/2 term does not sit on a 1-d.p. boundary.
        for _resample in range(20):
            raw_math = _quantity(a, d, period, c_mult, float(t_frac))
            raw_indep = _quantity_independent(a, d, Fraction(period), c_mult, t_frac)
            if rounds_unambiguously(raw_math, dp):
                break
            a = rng.randint(context["amp_lo"], context["amp_hi"])
            d = a + rng.randint(1, 15)
        else:
            check(False, "could not draw an unambiguously-rounding easy amplitude")
        check(d >= a, "D must be >= A so the modelled quantity is never negative")
    elif variant == "medium":
        period = rng.randint(8, 14) if context["key"] == "tide" else rng.randint(6, 30)
        start_phrase, c_mult = rng.choice(_STARTS)
        period_unit = t_unit = context["unit_time"]
        # Redraw `t` itself (never nudge the computed VALUE, which would
        # answer a different question from the one the fixed, displayed `t`
        # actually poses) until the rounded quantity is not a coin-flip on a
        # 1-d.p. boundary.
        for _resample in range(20):
            t = _draw_non_special_time(rng, period)
            t_frac = Fraction(str(t))
            raw_math = _quantity(a, d, period, c_mult, float(t_frac))
            raw_indep = _quantity_independent(a, d, Fraction(period), c_mult, t_frac)
            if rounds_unambiguously(raw_math, dp):
                break
        else:
            check(False, "could not draw an unambiguously-rounding time in 20 attempts")
    elif variant == "hard":
        period_unit, t_unit, factor, period_lo, period_hi = _pick_unit_conversion(context)
        period = rng.randint(period_lo, period_hi)
        start_phrase, c_mult = ("starting at its minimum", Fraction(-1, 2))
        for _resample in range(20):
            t_in_period_unit = _draw_non_special_time(rng, period)
            # Exact `Fraction` arithmetic throughout -- `t_in_period_unit` is
            # already rounded to 1 d.p., so multiplying by the (integer) unit
            # factor stays exact and never accumulates float noise the way
            # `t_in_period_unit * factor` (plain float) would.
            t_display_frac = Fraction(str(t_in_period_unit)) * factor  # what
            # the question STATES, in t_unit
            # `t` on the wire is in `t_unit`; convert to `period_unit` before
            # evaluating, exactly as a learner must.
            t_period_units = t_display_frac / factor
            raw_math = _quantity(a, d, period, c_mult, float(t_period_units))
            raw_indep = _quantity_independent(a, d, Fraction(period), c_mult, t_period_units)
            if rounds_unambiguously(raw_math, dp):
                break
        else:
            check(False, "could not draw an unambiguously-rounding time in 20 attempts")
    else:
        raise ValueError(f"unknown variant {variant!r}")

    check(
        abs(raw_math - raw_indep) < 1e-9, "construction's math.sin route and the independent sympy.sin route disagree"
    )

    value_frac = round_half_up(Fraction(raw_math).limit_denominator(10**6), dp)
    value = float(value_frac)
    value_str = stated_precision_string(value_frac, dp)

    if variant == "easy":
        frac_words = _EASY_FRAC_WORDS[frac]
        t_str = _fmt_time(t)
        question_text = (
            f"{context['subject']} varies sinusoidally with amplitude {a} {unit_len} "
            f"about a mean of {d} {unit_len}, with a period of {period} {period_unit}, "
            f"{start_phrase}. Find {context['qty']}, in {unit_len} (1 d.p.), at "
            f"t = {t_str} {_unit_word(t_str, t_unit)} ({frac_words} of the way through one period)."
        )
    elif variant == "medium":
        t_str = _fmt_time(t)
        question_text = (
            f"{context['subject']} varies sinusoidally with amplitude {a} {unit_len} "
            f"about a mean of {d} {unit_len}, with a period of {period} {period_unit}, "
            f"{start_phrase}. Find {context['qty']}, in {unit_len} (1 d.p.), at "
            f"t = {t_str} {_unit_word(t_str, t_unit)}."
        )
    else:  # hard
        t_str = _fmt_time(t_display_frac)
        question_text = (
            f"{context['subject']} varies sinusoidally with amplitude {a} {unit_len} "
            f"about a mean of {d} {unit_len}, with a period of {period} {period_unit}, "
            f"{start_phrase}. Find {context['qty']}, in {unit_len} (1 d.p.), at "
            f"t = {t_str} {_unit_word(t_str, t_unit)} after tracking begins "
            f"(convert to {period_unit} first)."
        )

    answer_text = f"{value_str} {unit_len}"
    if variant == "hard":
        t_eval = t_period_units
    else:
        t_eval = t_frac if variant == "easy" else Fraction(str(t))
    distractors = _build_evaluate_distractors(
        rng,
        a,
        d,
        value_frac,
        dp,
        unit_len,
        period,
        c_mult,
        t_eval,
        raw_math,
        t_wrong_unit=(t_display_frac if variant == "hard" else None),
    )

    return dict(
        text=question_text,
        answer=value,
        answer_text=answer_text,
        distractors=distractors,
    )


_SINGULAR_UNIT = {"seconds": "second", "minutes": "minute", "hours": "hour", "milliseconds": "millisecond"}


def _unit_word(value_str: str, unit: str) -> str:
    """Singular unit word when the STATED time is exactly 1 (never "1
    hours") -- `mathgen.text_quality`'s `singular_count_plural` prompt
    check, scoped to `question.text` only."""
    return _SINGULAR_UNIT.get(unit, unit) if value_str == "1" else unit


def _fmt_time(t) -> str:
    tf = t if isinstance(t, Fraction) else Fraction(str(t))
    if tf.denominator == 1:
        return str(int(tf))
    return exact_decimal_string(tf, 6)


def _draw_non_special_time(rng, period: int) -> float:
    """A decimal time within (0, period) at least 0.15*period away from
    every quarter-period boundary, so the substituted angle is never a
    multiple of pi/2 (a genuine non-special-angle evaluation)."""
    for _ in range(50):
        t = round(rng.uniform(0.05 * period, 0.95 * period), 1)
        nearest_quarter = round(t / (period / 4)) * (period / 4)
        if abs(t - nearest_quarter) > 0.1 * period / 4 + 1e-9:
            return t
    check(False, "could not draw a non-special-angle time in 50 attempts")


def _pick_unit_conversion(context):
    """The scenario's OWN fixed conversion pairing (never a shared,
    scenario-blind pool): see ``_HARD_UNIT_CONFIG``."""
    return _HARD_UNIT_CONFIG[context["key"]]


_MIN_KEY_GAP = Fraction(3, 10)
_MIN_PAIR_GAP = Fraction(2, 10)


def _choose_options(rng, key, primary, secondary, tol, is_extreme, extreme_p, attack=None):
    """Pick three distractors (rounded Fractions) from independent-route
    candidates. `primary` always sits in the pool; each `secondary` value is
    admitted with probability 0.2 (never as a fixed pair); the rest only top
    up if too few distinct values remain. The draw is repeated until the key's
    rank (extreme or not, decided once by `extreme_p`) and the optional
    `attack` heuristic do not give the key away."""

    def usable(vals):
        out = []
        for v in vals:
            # a second defensible answer: keep >= 0.3 from the key and
            # >= 0.2 from every other option
            if abs(v - key) >= _MIN_KEY_GAP and abs(v - key) > tol and all(abs(v - o) >= _MIN_PAIR_GAP for o in out):
                out.append(v)
        return out

    chosen = [v for v in secondary if rng.random() < 0.2]
    rest = [v for v in secondary if v not in chosen]
    want_extreme = rng.random() < extreme_p
    first = None
    for _attempt in range(120):
        pool = list(primary) + chosen
        rng.shuffle(pool)
        tail = list(rest)
        rng.shuffle(tail)
        opts = usable(pool + tail)[:3]
        if len(opts) < 3:
            continue
        if first is None:
            first = opts
        if is_extreme(key, opts) == want_extreme and not (attack and attack(key, opts)):
            return opts
    check(first is not None, "fewer than three distinct distractors available")
    return first


def _build_evaluate_distractors(
    rng, a, d, value_frac, dp, unit_len, period, c_mult, t_eval, raw_math, t_wrong_unit=None
):
    """Every candidate is a plausible HEIGHT/DISTANCE and lies inside the
    physical range [d - a, d + a] by construction: each PRIMARY route
    re-evaluates the periodic function under one independent slip (wrong
    start phase, period misread as half/double, a nearby time, the sine value
    rounded early, and -- hard tier -- the time left unconverted). The mean,
    the mirror about D, half-amplitude and off-by-one are SECONDARY routes,
    each admitted in a minority of rows and never as a fixed pair."""
    tol = stated_precision_tolerance(dp)
    lo, hi = Fraction(d - a), Fraction(d + a)
    key = value_frac
    dq = Fraction(d)

    def rnd(x):
        return round_half_up(Fraction(x).limit_denominator(10**6), dp)

    t0 = float(t_eval)
    primary = [
        rnd(_quantity(a, d, period, other, t0))  # wrong start phase
        for _label, other in _STARTS
        if other != c_mult
    ]
    primary.append(rnd(_quantity(a, d, period / 2, c_mult, t0)))  # period misread as half
    primary.append(rnd(_quantity(a, d, period * 2, c_mult, t0)))  # ... or double
    shift = period / 8 * (1 if rng.random() < 0.5 else -1)  # a nearby time
    primary.append(rnd(_quantity(a, d, period, c_mult, t0 + shift)))
    s_early = round((raw_math - d) / a, 1)  # sine rounded early
    primary.append(rnd(d + a * s_early))
    if t_wrong_unit is not None:  # conversion forgotten
        primary.append(rnd(_quantity(a, d, period, c_mult, float(t_wrong_unit))))

    secondary = [dq, 2 * dq - key, dq + (key - dq) / 2, key + 1 if key + 1 <= hi else key - 1]
    secondary = [rnd(v) for v in secondary]
    mirror = secondary[1]
    for c in primary + secondary:
        check(lo <= c <= hi, "distractor left the physical range [D - A, D + A]")
    # the mean is a secondary route only: odd-eighth easy draws land other
    # slips exactly on D, so keep those out of the always-available pool
    primary = [v for v in primary if abs(v - dq) > tol]

    def is_extreme(k, opts):
        return k > max(opts) or k < min(opts)

    def attack(k, opts):
        far = max(opts, key=lambda v: abs(v - dq))
        # the reflection attack, or the fixed mean + mirror pair
        return abs((2 * dq - far) - k) <= tol or (rnd(dq) in opts and mirror in opts)

    opts = _choose_options(rng, key, primary, secondary, tol, is_extreme, 0.4, attack)
    answer_text = f"{stated_precision_string(key, dp)} {unit_len}"
    candidates = [f"{stated_precision_string(c, dp)} {unit_len}" for c in opts]
    return pick_distractors(answer_text, candidates, n=3, tolerance=tol)


def _evaluate_context_exam(rng, context, dp):
    """Exam-tier kind-escalation: solve for the FIRST time t > 0 (within one
    period) at which the modelled quantity reaches a stated target value --
    an inverse operation, not a forward evaluation. The oracle re-derives
    which of (up to) two solutions in one period is the smallest positive
    one, and the prompt names that explicitly ("the smallest positive
    t")."""
    unit_len = context["unit_len"]
    a = rng.randint(context["amp_lo"], context["amp_hi"])
    d = a + rng.randint(1, 15)
    check(d >= a, "D must be >= A so the modelled quantity is never negative")
    period_unit = context["unit_time"]
    period = rng.randint(8, 14) if context["key"] == "tide" else rng.randint(6, 30)
    start_phrase, c_mult = rng.choice(_STARTS)

    for _attempt in range(30):
        s_num = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])  # avoid s == 0
        s = s_num / 5  # s in [-0.8, -0.2] U [0.2, 0.8], well clear of +-1
        target_value = d + a * s
        base_theta1 = math.asin(s)
        base_theta2 = math.pi - base_theta1
        offset = float(c_mult) * math.pi
        t1 = (period * ((base_theta1 - offset) / (2 * math.pi))) % period
        t2 = (period * ((base_theta2 - offset) / (2 * math.pi))) % period
        if t1 <= 1e-9:
            t1 += period
        if t2 <= 1e-9:
            t2 += period
        answer_t = min(t1, t2)
        other_t = max(t1, t2)
        if other_t - answer_t > 0.5 and rounds_unambiguously(answer_t, dp):
            break
    else:
        check(False, "could not draw a well-separated inverse-solve target in 30 attempts")

    # Independent re-derivation: substitute answer_t back through the exact
    # sympy route and confirm it reaches target_value.
    check_value = _quantity_independent(a, d, Fraction(period), c_mult, Fraction(answer_t).limit_denominator(10**6))
    check(
        abs(check_value - target_value) < 1e-6,
        "the derived smallest-positive-t solution does not reach the target value",
    )

    value_frac = round_half_up(Fraction(answer_t).limit_denominator(10**6), dp)
    answer = float(value_frac)
    answer_str = stated_precision_string(value_frac, dp)

    target_str = target_value if target_value == int(target_value) else round(target_value, 1)
    question_text = (
        f"{context['subject']} varies sinusoidally with amplitude {a} {unit_len} "
        f"about a mean of {d} {unit_len}, with a period of {period} {period_unit}, "
        f"{start_phrase}. Find the smallest positive time t (in {period_unit}, 1 d.p.) "
        f"at which {context['qty']} first reaches {target_str} {unit_len}."
    )

    tol = stated_precision_tolerance(dp)

    def solve(sv, cm):
        """Both solutions in (0, period] for sin-value `sv`, start phase `cm`."""
        th1 = math.asin(sv)
        th2 = math.pi - th1
        off = float(cm) * math.pi
        out = []
        for th in (th1, th2):
            tv = (period * ((th - off) / (2 * math.pi))) % period
            out.append(tv + period if tv <= 1e-9 else tv)
        return out

    def rnd(x):
        return round_half_up(Fraction(x).limit_denominator(10**6), dp)

    primary = []
    primary.extend(solve(s, 0))  # forgot the phase offset (both branches)
    for _label, other in _STARTS:  # measured from the wrong start
        if other != c_mult:
            primary.append(min(solve(s, other)))
    primary.extend(solve(-s, c_mult))  # arcsin sign slip (both branches)
    primary.append(other_t)  # gave the later solution
    primary.append(answer_t - period / 8)  # counted from an eighth-period early
    for ds in (0.2, -0.2):  # the time for a neighbouring target value
        if abs(s + ds) < 0.95:
            primary.extend(solve(s + ds, c_mult))
    primary.append(answer_t + period / 4)  # wrong quarter
    primary.append(period - other_t)  # later solution measured back from the end
    primary = [rnd(v) for v in primary]
    # P - key and P/2 are secondary slips: each in a minority of rows
    secondary = [rnd(period - answer_t), rnd(Fraction(period, 2))]
    lim = Fraction(period)
    half_dbl = (value_frac / 2, value_frac * 2)  # never derive an option from the key by 2x or /2

    def keep(v):
        return Fraction(1, 10) <= v < lim and all(abs(v - h) > Fraction(6, 100) for h in half_dbl)

    primary = [v for v in primary if keep(v)]
    secondary = [v for v in secondary if keep(v)]

    def is_min(k, opts):
        return k < min(opts)

    opts = _choose_options(rng, value_frac, primary, secondary, tol, is_min, 0.3)
    answer_text = f"{answer_str} {period_unit}"
    candidates = [f"{stated_precision_string(c, dp)} {period_unit}" for c in opts]
    distractors = pick_distractors(answer_text, candidates, n=3, tolerance=tol)

    return dict(
        text=question_text,
        answer=answer,
        answer_text=answer_text,
        distractors=distractors,
    )
