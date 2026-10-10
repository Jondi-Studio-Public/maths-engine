"""Given bounded, fully specified inputs for one named logarithmic scale
(decibels, the Richter Scale, octaves, or pH) and its stated formula, compute
the resulting value on that scale. FORWARD computation only, every tier:
never "given pH find [H+]". Every prompt states the formula and every
reference constant.

Formulas: dB: L = 10 log10(I / I0), I0 = 1e-12 W/m^2; Richter:
M = log10(A / A0); octaves: n = log2(f / f0); pH: pH = -log10[H+].

Tiers: easy evaluates a scale at an EXACT power of the base (integer
answer, graded exactly, no decimal point). medium gives a mantissa x 10^k
(or a non-power ratio), so the learner must split log(mantissa x 10^k) or
use a calculator; answer to 1 or 2 decimal places. hard states the input as
a RATIO or combination the learner must form first (a multiple of the
threshold, an amplitude requiring division by a stated reference, a
non-power frequency ratio, a multiplier on a stated reference pH). exam asks
for the DIFFERENCE in scale value between two stated inputs on the same
scale -- still a single computed scale value, via log(x1/x2).
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.core import (
    check,
    generator,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)

SCALES = ("db", "richter", "octave", "ph")


def _display(value, dp: int) -> str:
    return stated_precision_string(round_half_up(Fraction(value).limit_denominator(10**9), dp), dp)


def _finite_ok(value: float, lo: float = -1e6, hi: float = 1e6) -> bool:
    return math.isfinite(value) and lo < value < hi


def _num_disp(value, places: int = 2):
    """Display an input quantity without a spurious trailing '.0': a whole
    number renders as a plain int, anything else rounds to `places`."""
    if float(value) == int(value):
        return int(value)
    return round(float(value), places)


def _sci(mantissa, exponent: int) -> str:
    """`3.2 x 10^-5`-style scientific-notation text; `mathgen.core.latex_prompt`
    lifts `×`/`^-5` into real LaTeX automatically, so no separate latex is
    authored here."""
    return f"{mantissa} × 10^{exponent}"


# =============================================================================
# Easy: exact power of the base -- integer answer via the definition of a log
# =============================================================================

_EASY_DB_K = list(range(1, 13))
_EASY_RICHTER_K = list(range(1, 10))
# f0 = 880 = 2^4 x 55, so f0 // 2^-n stays exact only down to n = -4; the
# positive side has no such limit (f0 * 2^n is always an integer), but a
# note stays plausible only a few octaves either side of the reference.
_EASY_OCTAVE_N = [n for n in range(-4, 7) if n != 0]
_EASY_PH_K = list(range(1, 15))
_OCTAVE_F0 = 880


def _build_easy_db(rng):
    k = rng.choice(_EASY_DB_K)
    exponent = k - 12
    value = 10 * k
    templates = (
        "A sound has intensity I = {i}. Using the decibel scale L = 10 log10(I / I0), with "
        "reference intensity I0 = 1 × 10^-12 W/m^2, find L in decibels.",
        "The decibel level of a sound is given by L = 10 log10(I / I0), where I0 = "
        "1 × 10^-12 W/m^2 is the reference intensity. Find L when I = {i}.",
        "Using L = 10 log10(I / I0) with reference intensity I0 = 1 × 10^-12 W/m^2, find the "
        "decibel level L for a sound of intensity I = {i}.",
    )
    text = phrase(rng, templates, i=_sci(1, exponent) + " W/m^2")
    check(abs(10 * (math.log10(10**exponent) - math.log10(10**-12)) - value) < 1e-9, "easy(db): recompute disagrees")
    dp = 0
    candidates = [
        10 * math.log(10**exponent) - 10 * math.log(10**-12),  # used ln instead of log10
        math.log10(10**exponent) - math.log10(10**-12),  # dropped the factor 10
        10 * (math.log10(10**-12) - math.log10(10**exponent)),  # inverted the ratio
    ]
    return text, value, dp, candidates, "db"


def _build_easy_richter(rng):
    k = rng.choice(_EASY_RICHTER_K)
    value = k
    templates = (
        "An earthquake has amplitude A = {a} times the reference amplitude A0. Using the "
        "Richter Scale M = log10(A / A0), find its magnitude M.",
        "The Richter Scale magnitude is given by M = log10(A / A0), where A0 is the reference "
        "amplitude. Find M when A = {a} × A0.",
        "Using M = log10(A / A0), find the Richter magnitude M of an earthquake whose amplitude "
        "A is {a} times the reference amplitude A0.",
    )
    text = phrase(rng, templates, a=10**k)
    check(abs(math.log10(10**k) - value) < 1e-9, "easy(richter): recompute disagrees")
    dp = 0
    candidates = [
        math.log(10**k),  # used ln instead of log10
        10 * math.log10(10**k),  # wrongly applied the dB factor of 10
        math.log10(1 / (10**k)),  # inverted the ratio
    ]
    return text, value, dp, candidates, "richter"


def _build_easy_octave(rng):
    n = rng.choice(_EASY_OCTAVE_N)
    f0 = _OCTAVE_F0
    f = f0 * (2**n) if n >= 0 else f0 // (2 ** (-n))
    check(f0 * (2.0**n) == f, "easy(octave): f must land exactly on the power-of-2 grid")
    value = n
    templates = (
        "A note has frequency f = {f} Hz. Using n = log2(f / f0), with reference frequency "
        "f0 = {f0} Hz, find n, the number of octaves above (positive) or below (negative) the "
        "reference note.",
        "The number of octaves a note lies above or below a reference frequency f0 = {f0} Hz is "
        "n = log2(f / f0). Find n when f = {f} Hz.",
        "Using n = log2(f / f0) with reference frequency f0 = {f0} Hz, find n for a note of frequency f = {f} Hz.",
    )
    text = phrase(rng, templates, f=f, f0=f0)
    check(abs(math.log2(f / f0) - value) < 1e-9, "easy(octave): recompute disagrees")
    dp = 0
    candidates = [
        math.log10(f / f0),  # used log10 instead of log2
        math.log2(f0 / f),  # inverted the ratio
        math.log(f / f0),  # used ln instead of log2
    ]
    return text, value, dp, candidates, "octave"


def _build_easy_ph(rng):
    k = rng.choice(_EASY_PH_K)
    value = k
    templates = (
        "A solution has hydrogen ion concentration [H+] = {h}. Using pH = -log10[H+], find its pH.",
        "The pH of a solution is given by pH = -log10[H+]. Find the pH when [H+] = {h}.",
        "Using pH = -log10[H+], find the pH of a solution with hydrogen ion concentration [H+] = {h}.",
    )
    text = phrase(rng, templates, h=_sci(1, -k) + " mol/L")
    check(abs(-math.log10(10**-k) - value) < 1e-9, "easy(ph): recompute disagrees")
    dp = 0
    candidates = [
        -math.log(10**-k),  # used ln instead of log10
        math.log10(10**-k),  # dropped the negative sign
        -math.log10(10**k),  # inverted the ratio
    ]
    return text, value, dp, candidates, "ph"


_EASY_BUILDERS = {
    "db": _build_easy_db,
    "richter": _build_easy_richter,
    "octave": _build_easy_octave,
    "ph": _build_easy_ph,
}


def _build_easy(rng):
    scale = rng.choice(SCALES)
    text, value, dp, candidates_val, scale_used = _EASY_BUILDERS[scale](rng)
    answer_text = _display(value, dp)
    candidates_val = [v for v in candidates_val if _finite_ok(v) and abs(v - value) >= 1]
    return dict(
        text=text,
        answer=numeric_answer(round_half_up(Fraction(value), dp)),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, [_display(v, dp) for v in candidates_val], n=3),
    )


# =============================================================================
# Medium: mantissa x 10^k (or a non-power ratio) -- log of a product, 1-2 dp
# =============================================================================


def _build_medium_db(rng):
    mantissa = rng.choice([1.2, 1.5, 1.8, 2.4, 3.2, 4.5, 5.6, 6.8, 7.5, 8.1])
    k = rng.choice(range(1, 9))
    exponent = k - 12
    value = 10 * (math.log10(mantissa) + k)
    dp = 1
    templates = (
        "A sound has intensity I = {i}. Using L = 10 log10(I / I0), with reference intensity "
        "I0 = 1 × 10^-12 W/m^2, find L in decibels, to 1 decimal place.",
        "Using L = 10 log10(I / I0) with reference intensity I0 = 1 × 10^-12 W/m^2, find the "
        "decibel level L for a sound of intensity I = {i}, to 1 decimal place.",
    )
    text = phrase(rng, templates, i=_sci(mantissa, exponent) + " W/m^2")
    check(
        abs(10 * (math.log10(mantissa * 10**exponent) - math.log10(10**-12)) - value) < 1e-9,
        "medium(db): recompute disagrees",
    )
    candidates = [
        10 * (math.log(mantissa * 10**exponent) - math.log(10**-12)),  # used ln instead of log10
        math.log10(mantissa * 10**exponent) - math.log10(10**-12),  # dropped the factor 10
        10 * (math.log10(10**-12) - math.log10(mantissa * 10**exponent)),  # inverted the ratio
        10 * (math.log10(mantissa) + k - 1),  # off-by-one exponent
    ]
    return text, value, dp, candidates


def _build_medium_richter(rng):
    mantissa = rng.choice([1.2, 1.5, 1.8, 2.4, 3.2, 4.5, 5.6, 6.8, 7.5, 8.1])
    k = rng.choice(range(1, 8))
    value = math.log10(mantissa) + k
    dp = 2
    templates = (
        "An earthquake has amplitude A = {a} times the reference amplitude A0. Using "
        "M = log10(A / A0), find its magnitude M, to 2 decimal places.",
        "Using M = log10(A / A0), find the Richter magnitude M of an earthquake whose "
        "amplitude A is {a} times the reference amplitude A0, to 2 decimal places.",
    )
    a_val = mantissa * 10**k
    a_disp = _num_disp(a_val, 1)
    text = phrase(rng, templates, a=a_disp)
    check(abs(math.log10(mantissa * 10**k) - value) < 1e-9, "medium(richter): recompute disagrees")
    candidates = [
        math.log(mantissa * 10**k),
        10 * math.log10(mantissa * 10**k),
        -math.log10(mantissa * 10**k),
    ]
    return text, value, dp, candidates


_MEDIUM_OCTAVE_RATIOS = [
    (3, 2),
    (5, 4),
    (5, 3),
    (7, 4),
    (9, 8),
    (11, 8),
    (13, 8),
    (15, 8),
    (9, 5),
    (7, 5),
    (11, 6),
    (13, 6),
    (15, 4),
]


def _build_medium_octave(rng):
    f0 = _OCTAVE_F0
    e, d = rng.choice(_MEDIUM_OCTAVE_RATIOS)
    f = f0 * e // d if (f0 * e) % d == 0 else f0 * e / d
    value = math.log2(e / d)
    dp = 2
    templates = (
        "A note has frequency f = {f} Hz. Using n = log2(f / f0), with reference frequency "
        "f0 = {f0} Hz, find n, to 2 decimal places.",
        "Using n = log2(f / f0) with reference frequency f0 = {f0} Hz, find n for a note of "
        "frequency f = {f} Hz, to 2 decimal places.",
    )
    f_disp = _num_disp(f, 2)
    text = phrase(rng, templates, f=f_disp, f0=f0)
    check(abs(math.log2((f0 * e / d) / f0) - value) < 1e-9, "medium(octave): recompute disagrees")
    candidates = [
        math.log10(e / d),
        math.log2(d / e),
        math.log(e / d),
    ]
    return text, value, dp, candidates


def _build_medium_ph(rng):
    mantissa = rng.choice([1.2, 1.5, 1.8, 2.4, 3.2, 4.5, 5.6, 6.8, 7.5, 8.1])
    k = rng.choice(range(1, 14))
    value = k - math.log10(mantissa)
    dp = 2
    templates = (
        "A solution has hydrogen ion concentration [H+] = {h}. Using pH = -log10[H+], find its "
        "pH, to 2 decimal places.",
        "Using pH = -log10[H+], find the pH of a solution with hydrogen ion concentration "
        "[H+] = {h}, to 2 decimal places.",
    )
    text = phrase(rng, templates, h=_sci(mantissa, -k) + " mol/L")
    check(abs(-math.log10(mantissa * 10**-k) - value) < 1e-9, "medium(ph): recompute disagrees")
    candidates = [
        -math.log(mantissa * 10**-k),
        math.log10(mantissa * 10**-k),
        -math.log10(mantissa * 10**k),
    ]
    return text, value, dp, candidates


_MEDIUM_BUILDERS = {
    "db": _build_medium_db,
    "richter": _build_medium_richter,
    "octave": _build_medium_octave,
    "ph": _build_medium_ph,
}


def _build_medium(rng):
    for _ in range(200):
        scale = rng.choice(SCALES)
        text, value, dp, candidates_val = _MEDIUM_BUILDERS[scale](rng)
        if not (-999 < value < 999):
            continue
        if not rounds_unambiguously(value, dp):
            continue
        break
    else:
        raise RuntimeError("could not draw a valid p19e_named_log_scale_value medium question")
    margin = float(stated_precision_tolerance(dp)) + 1e-9
    answer_text = _display(value, dp)
    candidates_val = [v for v in candidates_val if _finite_ok(v) and abs(v - value) > margin]
    return dict(
        text=text,
        answer=numeric_answer(round_half_up(Fraction(value), dp)),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, [_display(v, dp) for v in candidates_val], n=3),
    )


# =============================================================================
# Hard: a ratio/combination the learner must form first
# =============================================================================


_HARD_DB_MULTS = [
    20,
    25,
    32,
    40,
    50,
    63,
    70,
    80,
    90,
    110,
    125,
    140,
    160,
    180,
    200,
    250,
    320,
    400,
    500,
    630,
    700,
    800,
    900,
]


def _build_hard_db(rng):
    mult = rng.choice(_HARD_DB_MULTS)
    value = 10 * math.log10(mult)
    dp = 1
    templates = (
        "A sound's intensity is {mult} times the threshold intensity I0 (I0 = 1 × 10^-12 "
        "W/m^2). Using L = 10 log10(I / I0), find its decibel level L, to 1 decimal place.",
        "Using L = 10 log10(I / I0), find the decibel level L of a sound whose intensity is "
        "{mult} times the threshold intensity I0 = 1 × 10^-12 W/m^2, to 1 decimal place.",
    )
    text = phrase(rng, templates, mult=mult)
    check(abs(10 * math.log10(mult) - value) < 1e-9, "hard(db): recompute disagrees")
    candidates = [10 * math.log(mult), math.log10(mult), 10 * math.log10(1 / mult)]
    return text, value, dp, candidates


_HARD_RICHTER_A_MM = [
    0.3,
    0.5,
    0.8,
    1.5,
    2.0,
    3.0,
    4.5,
    6.0,
    7.5,
    9.0,
    15.0,
    22.5,
    30.0,
    45.0,
    60.0,
    75.0,
    90.0,
    150.0,
    225.0,
    300.0,
]


def _build_hard_richter(rng):
    a_mm = rng.choice(_HARD_RICHTER_A_MM)
    a0_mm = 0.001
    value = math.log10(a_mm / a0_mm)
    dp = 2
    templates = (
        "An earthquake's amplitude is A = {a} mm, measured against a reference amplitude of "
        "A0 = {a0} mm. Using M = log10(A / A0), find its magnitude M, to 2 decimal places.",
        "Using M = log10(A / A0) with reference amplitude A0 = {a0} mm, find the Richter "
        "magnitude M of an earthquake with amplitude A = {a} mm, to 2 decimal places.",
    )
    text = phrase(rng, templates, a=a_mm, a0=a0_mm)
    check(abs(math.log10(a_mm / a0_mm) - value) < 1e-9, "hard(richter): recompute disagrees")
    candidates = [math.log10(a0_mm / a_mm), math.log(a_mm / a0_mm), 10 * math.log10(a_mm / a0_mm)]
    return text, value, dp, candidates


_HARD_OCTAVE_RATIOS = [
    (17, 5),
    (13, 4),
    (19, 5),
    (23, 8),
    (29, 8),
    (11, 3),
    (7, 2),
    (9, 2),
    (21, 5),
    (25, 6),
    (27, 7),
    (31, 7),
    (33, 8),
    (37, 9),
]


def _build_hard_octave(rng):
    e, d = rng.choice(_HARD_OCTAVE_RATIOS)
    value = math.log2(e / d)
    dp = 2
    templates = (
        "A note's frequency is {e}/{d} times a reference frequency f0. Using n = log2(f / f0), "
        "find n, to 2 decimal places.",
        "Using n = log2(f / f0), find n for a note whose frequency is {e}/{d} times the "
        "reference frequency f0, to 2 decimal places.",
    )
    text = phrase(rng, templates, e=e, d=d)
    check(abs(math.log2(e / d) - value) < 1e-9, "hard(octave): recompute disagrees")
    candidates = [math.log2(d / e), math.log10(e / d), math.log(e / d)]
    return text, value, dp, candidates


def _build_hard_ph(rng):
    ref_ph = rng.choice(range(2, 12))
    mult = rng.choice([15, 25, 40, 60, 90])
    value = ref_ph - math.log10(mult)
    dp = 2
    templates = (
        "A solution's hydrogen ion concentration is {mult} times that of a reference solution "
        "with pH = {ref_ph}. Using pH = -log10[H+], find the new solution's pH, to 2 decimal "
        "places.",
        "Using pH = -log10[H+], find the pH of a solution whose [H+] is {mult} times that of a "
        "reference solution with pH = {ref_ph}, to 2 decimal places.",
    )
    text = phrase(rng, templates, mult=mult, ref_ph=ref_ph)
    check(abs((ref_ph - math.log10(mult)) - value) < 1e-9, "hard(ph): recompute disagrees")
    candidates = [ref_ph + math.log10(mult), math.log10(mult) - ref_ph, ref_ph - math.log(mult)]
    return text, value, dp, candidates


_HARD_BUILDERS = {
    "db": _build_hard_db,
    "richter": _build_hard_richter,
    "octave": _build_hard_octave,
    "ph": _build_hard_ph,
}


def _build_hard(rng):
    for _ in range(200):
        scale = rng.choice(SCALES)
        text, value, dp, candidates_val = _HARD_BUILDERS[scale](rng)
        if not (-999 < value < 999):
            continue
        if not rounds_unambiguously(value, dp):
            continue
        break
    else:
        raise RuntimeError("could not draw a valid p19e_named_log_scale_value hard question")
    margin = float(stated_precision_tolerance(dp)) + 1e-9
    answer_text = _display(value, dp)
    candidates_val = [v for v in candidates_val if _finite_ok(v) and abs(v - value) > margin]
    return dict(
        text=text,
        answer=numeric_answer(round_half_up(Fraction(value), dp)),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, [_display(v, dp) for v in candidates_val], n=3),
    )


# =============================================================================
# Exam: DIFFERENCE in scale value between two stated inputs on the same scale
# =============================================================================


_EXAM_MANTISSAS = (1.2, 1.5, 1.8, 2.4, 3.2, 4.5, 5.6, 6.8, 7.5, 8.1)


def _build_exam_db(rng):
    m1, m2 = rng.sample(_EXAM_MANTISSAS, 2)
    k1 = rng.choice(range(1, 8))
    k2 = k1 + rng.choice(range(1, 4))
    value = 10 * (math.log10(m2 * 10**k2) - math.log10(m1 * 10**k1))
    dp = 1
    templates = (
        "One sound has intensity I1 = {i1}, and another has intensity I2 = {i2}. Using "
        "L = 10 log10(I / I0) (I0 = 1 × 10^-12 W/m^2 in both cases), find how many decibels "
        "louder the second sound is than the first, to 1 decimal place.",
        "Using L = 10 log10(I / I0), find the difference in decibel level between a sound of "
        "intensity I2 = {i2} and one of intensity I1 = {i1} (I0 = 1 × 10^-12 W/m^2), to 1 "
        "decimal place.",
    )
    text = phrase(rng, templates, i1=_sci(m1, k1 - 12) + " W/m^2", i2=_sci(m2, k2 - 12) + " W/m^2")
    check(abs(10 * (math.log10(m2 * 10**k2) - math.log10(m1 * 10**k1)) - value) < 1e-9, "exam(db): recompute disagrees")
    ratio = (m2 * 10**k2) / (m1 * 10**k1)
    candidates = [ratio, 10 * math.log(ratio), math.log10(ratio), -value]
    return text, value, dp, candidates


def _build_exam_richter(rng):
    m1, m2 = rng.sample(_EXAM_MANTISSAS, 2)
    k1 = rng.choice(range(1, 7))
    k2 = k1 + rng.choice(range(1, 3))
    value = math.log10(m2 * 10**k2) - math.log10(m1 * 10**k1)
    dp = 2
    templates = (
        "One earthquake has amplitude A1 = {a1} times the reference amplitude A0, and a "
        "second earthquake has amplitude A2 = {a2} times A0. Using M = log10(A / A0), find how "
        "many magnitude units higher the second earthquake is than the first, to 2 decimal "
        "places.",
        "Using M = log10(A / A0), find the difference in Richter magnitude between an "
        "earthquake with amplitude A2 = {a2} × A0 and one with amplitude A1 = {a1} × A0, to 2 "
        "decimal places.",
    )
    a1_disp = _num_disp(m1 * 10**k1, 1)
    a2_disp = _num_disp(m2 * 10**k2, 1)
    text = phrase(rng, templates, a1=a1_disp, a2=a2_disp)
    check(abs((math.log10(m2 * 10**k2) - math.log10(m1 * 10**k1)) - value) < 1e-9, "exam(richter): recompute disagrees")
    ratio = (m2 * 10**k2) / (m1 * 10**k1)
    candidates = [ratio, math.log(ratio), 10 * math.log10(ratio), -value]
    return text, value, dp, candidates


def _build_exam_octave(rng):
    # Both frequencies are drawn as plain whole hertz -- f2 is NOT
    # back-constructed from a target "nice" ratio (that produced visibly
    # reverse-engineered decimals like f2 = 6453.33 Hz); the resulting octave
    # difference is whatever log2(f2 / f1) naturally comes out to.
    f0 = _OCTAVE_F0
    f1 = rng.randint(80, 2000)
    f2 = rng.randint(f1 + 1, min(f1 * 16, 20000))
    value = math.log2(f2 / f1)
    dp = 2
    templates = (
        "One note has frequency f1 = {f1} Hz, and a higher note has frequency f2 = {f2} Hz. "
        "Using n = log2(f / f0), with reference frequency f0 = {f0} Hz, find how many octaves "
        "higher the second note is than the first, to 2 decimal places.",
        "Using n = log2(f / f0) with reference frequency f0 = {f0} Hz, find the difference in "
        "octaves between a note of frequency f2 = {f2} Hz and one of frequency f1 = {f1} Hz, to "
        "2 decimal places.",
    )
    text = phrase(rng, templates, f1=f1, f2=f2, f0=f0)
    check(abs(math.log2(f2 / f1) - value) < 1e-6, "exam(octave): recompute disagrees")
    check(float(f1).is_integer() and float(f2).is_integer(), "exam(octave): both frequencies must be whole hertz")
    ratio = f2 / f1
    candidates = [ratio, math.log10(ratio), math.log(ratio), -value]
    return text, value, dp, candidates


def _build_exam_ph(rng):
    m1, m2 = rng.sample(_EXAM_MANTISSAS, 2)
    k1 = rng.choice(range(1, 12))
    k2 = k1 + rng.choice(range(1, 4))
    value = (k2 - math.log10(m2)) - (k1 - math.log10(m1))
    dp = 2
    templates = (
        "One solution has hydrogen ion concentration [H+]1 = {h1}, and another has [H+]2 = "
        "{h2}. Using pH = -log10[H+], find the difference in pH between the two solutions "
        "(the smaller [H+] gives the higher pH), to 2 decimal places.",
        "Using pH = -log10[H+], find how much higher the pH of a solution with [H+]2 = {h2} is "
        "than that of a solution with [H+]1 = {h1}, to 2 decimal places.",
    )
    text = phrase(rng, templates, h1=_sci(m1, -k1) + " mol/L", h2=_sci(m2, -k2) + " mol/L")
    check(abs(((k2 - math.log10(m2)) - (k1 - math.log10(m1))) - value) < 1e-9, "exam(ph): recompute disagrees")
    ratio = (m1 * 10**-k1) / (m2 * 10**-k2)
    candidates = [ratio, -value, math.log10(ratio), math.log10(1 / ratio) * -1]
    return text, value, dp, candidates


_EXAM_BUILDERS = {
    "db": _build_exam_db,
    "richter": _build_exam_richter,
    "octave": _build_exam_octave,
    "ph": _build_exam_ph,
}


def _build_exam(rng):
    for _ in range(200):
        scale = rng.choice(SCALES)
        text, value, dp, candidates_val = _EXAM_BUILDERS[scale](rng)
        if not (0 < value < 999):
            continue
        if not rounds_unambiguously(value, dp):
            continue
        break
    else:
        raise RuntimeError("could not draw a valid p19e_named_log_scale_value exam question")
    margin = float(stated_precision_tolerance(dp)) + 1e-9
    answer_text = _display(value, dp)
    candidates_val = [v for v in candidates_val if _finite_ok(v, lo=-1e9, hi=1e9) and abs(v - value) > margin]
    return dict(
        text=text,
        answer=numeric_answer(round_half_up(Fraction(value), dp)),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, [_display(v, dp) for v in candidates_val], n=3),
    )


@generator(
    "p19e_named_log_scale_value",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19e_named_log_scale_value(rng, tier="easy"):
    """Given bounded, fully specified inputs for one named logarithmic scale
    (decibels, the Richter Scale, octaves, or pH) and its stated formula,
    compute the resulting value on that scale. FORWARD computation only,
    every tier."""
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    if tier == "hard":
        return _build_hard(rng)
    if tier == "exam":
        return _build_exam(rng)
    raise ValueError(f"unknown tier: {tier!r}")
