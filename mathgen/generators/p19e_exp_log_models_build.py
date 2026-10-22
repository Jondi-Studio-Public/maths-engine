"""Generates `p19e_exponential_model_value` and `p19e_log_model_solve_input` questions: evaluate an exponential
model in context, and solve a logarithmic model for its input, both `answer: number`.

Two senior ``answer: number`` skills, kept in one module on purpose because a similarity check flags them as a
0.82-similarity pair from shared template wording.

- ``p19e_exponential_model_value`` (ACMMM066,
  ``senior-identify-contexts-suitable-for-modelling-by-exponential-functions``):
  scope -- given a practical exponential growth or decay context with a
  stated initial value, growth or decay rate, and elapsed time, compute the
  modelled quantity AT that time. Every tier is EVALUATION: no prompt ever
  asks "after how long" or "when will".

- ``p19e_log_model_solve_input`` (ACMMM158,
  ``senior-identify-contexts-suitable-for-modelling-by-logarithmic-functions``):
  scope -- given a stated practical scenario modelled by a fully
  specified logarithmic function, compute the requested numeric value (the
  INPUT at which the modelled quantity reaches a stated level). Every tier
  SOLVES for the input; the model is always fully specified (never fitted
  from a data point).

Distinct from the shipping neighbour
``senior-solve-simple-equations-involving-logarithmic-functions-algebraically-and``
(``pv0b_log_equation``): that skill solves a bare ``log_b(ax+c) = k`` for an
EXACT algebraic x with an integer base 2..10, never a practical context.
This skill starts from a stated practical model (base 10 or base e, a and b
possibly non-integer) and a target LEVEL, and its answer is a rounded
decimal quantity with units, never an exact fraction.

Also distinct from ``senior-define-logarithms-as-indices-a-x-b-is``
(``logarithms_as_indices``): that skill assesses the bare definitional
equivalence a^x = b <-> x = log_a(b) for stated integer powers, never a
practical model and never a linear-in-log relationship y = a*log(x) + b.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import (
    check,
    generator,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
)

_E = math.e

# =============================================================================
# Shared display/verification helpers
# =============================================================================


def _display(value, dp: int) -> str:
    """Display any float/Fraction value at EXACTLY `dp` decimal places, the
    grid `grade_number`'s stated-precision tolerance is derived from."""
    return stated_precision_string(round_half_up(Fraction(value), dp), dp)


def _finite_ok(value: float, lo: float = 1e-6, hi: float = 1e7) -> bool:
    return math.isfinite(value) and lo < abs(value) < hi if value != 0 else False


def _bisection_solve(f, lo: float, hi: float, iters: int = 200) -> float:
    """Independent numeric root-find: never used by construction, only to
    re-confirm a claimed value satisfies the model equation via a route
    (interval bisection) that does not touch the closed-form algebra used to
    derive it."""
    flo, fhi = f(lo), f(hi)
    check(flo * fhi < 0, "bisection bracket does not change sign")
    for _ in range(iters):
        mid = (lo + hi) / 2
        fmid = f(mid)
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return (lo + hi) / 2


# =============================================================================
# Row 1: p19e_exponential_model_value (ACMMM066) -- evaluate at a stated time
# =============================================================================

_GROWTH_CONTEXTS = [
    # Population/cell counts stay plausible only in the thousands -- a real
    # town or colony is never "112 people"/"112 cells".
    ("The population of a small town", "thousand people", "population", ("year",)),
    ("A bacteria colony's cell count", "thousand cells", "cell count", ("day", "hour")),
    ("The number of devices infected by a computer virus", "devices", "infected device count", ("day", "week")),
    ("A wildlife park's deer population", "deer", "deer population", ("year",)),
]
_DECAY_CONTEXTS = [
    ("The mass of a radioactive sample", "mg", "mass", ("year", "day", "hour")),
    ("The volume of water in a leaking tank", "litres", "volume", ("day", "hour")),
    ("The energy stored in a large battery bank", "kWh", "stored energy", ("year", "month")),
    ("The concentration of a drug in a patient's bloodstream", "mg/L", "concentration", ("hour",)),
]
_PERIODS = ("year", "month", "week", "day", "hour")


def _period_for(allowed: tuple[str, ...], drawn: str) -> str:
    """The drawn period when the context can honestly use it, else a fixed stand-in from the context's own units
    (no extra random draw, so contexts that already suited their period keep their questions)."""
    return drawn if drawn in allowed else allowed[_PERIODS.index(drawn) % len(allowed)]


# Contexts whose quantity is quoted per year, for the question that gives a yearly rate and a time in months.
_YEARLY_STAND_IN = {
    "A bacteria colony's cell count": "A wildlife park's deer population",
    "The number of devices infected by a computer virus": "The population of a small town",
    "The volume of water in a leaking tank": "The mass of a radioactive sample",
    "The concentration of a drug in a patient's bloodstream": "The mass of a radioactive sample",
}

_EASY_TEMPLATES = (
    "{noun} starts at {A0} {unit} and grows by {r}% every {period}. "
    "Find the {short} after {t} {period}s, to the nearest whole number.",
    "{noun} starts at {A0} {unit} and decays by {r}% every {period}. "
    "Find the {short} after {t} {period}s, to the nearest whole number.",
)


def _sim_compound(A0: float, factor: float, t: int) -> float:
    """Iterative per-period multiplication -- a genuinely different route
    from the closed-form `A0 * factor ** t` used to construct the answer."""
    value = A0
    for _ in range(t):
        value *= factor
    return value


def _build_exp_easy(rng) -> dict:
    for _ in range(400):
        direction = rng.choice(("growth", "decay"))
        contexts = _GROWTH_CONTEXTS if direction == "growth" else _DECAY_CONTEXTS
        noun, unit, short, periods = rng.choice(contexts)
        period = _period_for(periods, rng.choice(_PERIODS))
        A0 = rng.randint(20, 150)
        r = rng.randint(2, 25)
        t = rng.randint(2, 8)
        factor = 1 + r / 100 if direction == "growth" else 1 - r / 100
        value = A0 * factor**t
        if not (1 <= value <= 999):
            continue
        if not rounds_unambiguously(value, 0):
            continue
        break
    else:
        raise RuntimeError("could not draw a valid p19e_exponential_model_value easy question")

    check(abs(_sim_compound(A0, factor, t) - value) < 1e-6, "easy: iterative recompute disagrees")

    template = (
        "{noun} starts at {A0} {unit} and grows by {r}% every {period}. "
        "Find the {short} after {t} {period}s, to the nearest whole number."
        if direction == "growth"
        else "{noun} starts at {A0} {unit} and decays by {r}% every {period}. "
        "Find the {short} after {t} {period}s, to the nearest whole number."
    )
    alt_template = (
        "{noun} begins at {A0} {unit} and increases by {r}% each {period}. "
        "Find the value after {t} {period}s, to the nearest whole number."
        if direction == "growth"
        else "{noun} begins at {A0} {unit} and decreases by {r}% each {period}. "
        "Find the value after {t} {period}s, to the nearest whole number."
    )
    third_template = (
        "Starting at {A0} {unit}, {noun_lower} grows by {r}% per {period}. "
        "Find the value after {t} {period}s, to the nearest whole number."
        if direction == "growth"
        else "Starting at {A0} {unit}, {noun_lower} decays by {r}% per {period}. "
        "Find the value after {t} {period}s, to the nearest whole number."
    )
    text = phrase(
        rng,
        (template, alt_template, third_template),
        noun=noun,
        noun_lower=noun[0].lower() + noun[1:],
        A0=A0,
        unit=unit,
        r=r,
        period=period,
        t=t,
        short=short,
    )

    dp = 0
    answer_text = _display(value, dp)
    candidates_val = [A0 * (1 + (r if direction == "growth" else -r) * t / 100)]
    if t > 1:
        candidates_val.append(A0 * factor ** (t - 1))
    candidates_val.append(A0 * factor ** (t + 1))
    flipped_factor = 1 - r / 100 if direction == "growth" else 1 + r / 100
    candidates_val.append(A0 * flipped_factor**t)
    candidates_val = [v for v in candidates_val if v > 0 and abs(v - value) > 0.6]

    return dict(
        text=text,
        answer=numeric_answer(round_half_up(Fraction(value), dp)),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, [_display(v, dp) for v in candidates_val], n=4),
    )


def _build_exp_medium(rng) -> dict:
    variant = rng.choice(("keyword", "fraction"))
    if variant == "keyword":
        return _build_exp_medium_keyword(rng)
    return _build_exp_medium_fraction(rng)


def _build_exp_medium_keyword(rng) -> dict:
    """Direction (growth vs decay) must be inferred from the wording, never
    stated as "grows"/"decays" directly."""
    for _ in range(400):
        direction = rng.choice(("growth", "decay"))
        contexts = _GROWTH_CONTEXTS if direction == "growth" else _DECAY_CONTEXTS
        noun, unit, short, periods = rng.choice(contexts)
        period = _period_for(periods, rng.choice(_PERIODS))
        A0 = rng.randint(20, 150)
        r = rng.randint(2, 25)
        t = rng.randint(2, 10)
        factor = 1 + r / 100 if direction == "growth" else 1 - r / 100
        value = A0 * factor**t
        if not (1 <= value <= 999):
            continue
        if not rounds_unambiguously(value, 1):
            continue
        break
    else:
        raise RuntimeError("could not draw a valid p19e_exponential_model_value medium(keyword) question")

    check(abs(_sim_compound(A0, factor, t) - value) < 1e-6, "medium(keyword): iterative recompute disagrees")

    if direction == "growth":
        templates = (
            "{noun} starts at {A0} {unit} and increases by {r}% every {period}. "
            "Find the {short} after {t} {period}s, to 1 decimal place.",
            "{noun} starts at {A0} {unit}. It gains {r}% of its value every {period}. "
            "Find the value after {t} {period}s, to 1 decimal place.",
        )
    else:
        templates = (
            "{noun} starts at {A0} {unit} and loses {r}% of its value every {period}. "
            "Find the {short} after {t} {period}s, to 1 decimal place.",
            "{noun} starts at {A0} {unit}. It shrinks by {r}% every {period}. "
            "Find the value after {t} {period}s, to 1 decimal place.",
        )
    text = phrase(rng, templates, noun=noun, A0=A0, unit=unit, r=r, period=period, t=t, short=short)

    dp = 1
    answer_text = _display(value, dp)
    candidates_val = [A0 * (1 + (r if direction == "growth" else -r) * t / 100)]
    if t > 1:
        candidates_val.append(A0 * factor ** (t - 1))
    candidates_val.append(A0 * factor ** (t + 1))
    flipped_factor = 1 - r / 100 if direction == "growth" else 1 + r / 100
    candidates_val.append(A0 * flipped_factor**t)
    candidates_val = [v for v in candidates_val if v > 0 and abs(v - value) > 0.06]

    return dict(
        text=text,
        answer=numeric_answer(round_half_up(Fraction(value), dp)),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, [_display(v, dp) for v in candidates_val], n=4),
    )


_SUB_MONTHS = (3, 6, 9, 15, 18, 21)


def _build_exp_medium_fraction(rng) -> dict:
    """Rate stated per year, elapsed time stated in MONTHS (a multiple of 3
    that is not a whole number of years), so the learner must convert to a
    fractional number of rate-periods before evaluating."""
    for _ in range(400):
        direction = rng.choice(("growth", "decay"))
        contexts = _GROWTH_CONTEXTS if direction == "growth" else _DECAY_CONTEXTS
        noun, unit, short, periods = rng.choice(contexts)
        if "year" not in periods:
            noun = _YEARLY_STAND_IN[noun]
            unit, short = next((u, sh) for n, u, sh, _p in contexts if n == noun)
        A0 = rng.randint(20, 150)
        r = rng.randint(2, 25)
        months = rng.choice(_SUB_MONTHS)
        exponent = months / 12
        factor = 1 + r / 100 if direction == "growth" else 1 - r / 100
        value = A0 * factor**exponent
        if not (1 <= value <= 999):
            continue
        if not rounds_unambiguously(value, 1):
            continue
        break
    else:
        raise RuntimeError("could not draw a valid p19e_exponential_model_value medium(fraction) question")

    check(
        abs(math.exp(exponent * math.log(factor)) - factor**exponent) < 1e-9,
        "medium(fraction): exponent recompute disagrees",
    )

    verb = "grows" if direction == "growth" else "decays"
    templates = (
        "{noun} starts at {A0} {unit} and {verb} by {r}% every year. "
        "Find the {short} after {months} months, to 1 decimal place.",
        "{noun} starts at {A0} {unit}. It {verb} by {r}% per year. "
        "Find the value after {months} months, to 1 decimal place.",
    )
    text = phrase(rng, templates, noun=noun, A0=A0, unit=unit, r=r, verb=verb, months=months, short=short)

    dp = 1
    answer_text = _display(value, dp)
    candidates_val = [A0 * factor**months]  # forgot to convert months to years at all
    candidates_val.append(A0 * factor ** (12 / months))  # inverted the ratio
    candidates_val.append(A0 * (1 + (r if direction == "growth" else -r) * exponent / 100))  # linear approx
    candidates_val = [v for v in candidates_val if v > 0 and _finite_ok(v) and abs(v - value) > 0.06]

    return dict(
        text=text,
        answer=numeric_answer(round_half_up(Fraction(value), dp)),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, [_display(v, dp) for v in candidates_val], n=4),
    )


_HARD_K = tuple(v / 100 for v in list(range(-35, -4, 5)) + list(range(5, 36, 5)))
_HALF_LIFE_H = (2, 3, 4, 5, 6, 8, 10)


def _build_exp_hard(rng) -> dict:
    variant = rng.choice(("continuous", "half_life"))
    if variant == "continuous":
        return _build_exp_hard_continuous(rng)
    return _build_exp_hard_half_life(rng)


def _build_exp_hard_continuous(rng) -> dict:
    for _ in range(400):
        direction = rng.choice(("growth", "decay"))
        contexts = _GROWTH_CONTEXTS if direction == "growth" else _DECAY_CONTEXTS
        noun, unit, short, periods = rng.choice(contexts)
        A0 = rng.randint(20, 150)
        k = rng.choice([v for v in _HARD_K if (v > 0) == (direction == "growth")])
        t = rng.randint(2, 10)
        value = A0 * math.exp(k * t)
        if not (1 <= value <= 999):
            continue
        if not rounds_unambiguously(value, 1):
            continue
        break
    else:
        raise RuntimeError("could not draw a valid p19e_exponential_model_value hard(continuous) question")

    step = math.exp(k)
    check(abs(_sim_compound(A0, step, t) - value) < 1e-6, "hard(continuous): iterative recompute disagrees")

    k_text = f"{k:.2f}"
    templates = (
        "{noun} is modelled by Q = {A0} e^({k}t), where t is measured in {period}s and Q is the {short}. "
        "Find Q when t = {t}, to 1 decimal place.",
        "The {short2} follows the model Q = {A0} e^({k}t), with t in {period}s. Find Q when t = {t}, to 1 decimal place.",
    )
    text = phrase(
        rng,
        templates,
        noun=noun,
        A0=A0,
        k=k_text,
        t=t,
        short=short,
        short2=short,
        period=periods[0],
    )

    dp = 1
    answer_text = _display(value, dp)
    candidates_val = [A0 * math.exp(-k * t)]  # sign-of-k error
    candidates_val.append(A0 * (1 + k) ** t)  # treats k as a percent-per-period rate
    if t > 1:
        candidates_val.append(A0 * math.exp(k * (t - 1)))
    candidates_val.append(A0 * math.exp(k * (t + 1)))
    candidates_val = [v for v in candidates_val if v > 0 and _finite_ok(v) and abs(v - value) > 0.06]

    return dict(
        text=text,
        answer=numeric_answer(round_half_up(Fraction(value), dp)),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, [_display(v, dp) for v in candidates_val], n=4),
    )


def _build_exp_hard_half_life(rng) -> dict:
    mode = rng.choice(("half_life", "doubling"))
    direction = "decay" if mode == "half_life" else "growth"
    contexts = _GROWTH_CONTEXTS if direction == "growth" else _DECAY_CONTEXTS
    for _ in range(400):
        noun, unit, short, periods = rng.choice(contexts)
        A0 = rng.randint(20, 150)
        h = rng.choice(_HALF_LIFE_H)
        t = rng.randint(1, 24)
        base = 0.5 if mode == "half_life" else 2.0
        value = A0 * base ** (t / h)
        if not (1 <= value <= 999):
            continue
        if not rounds_unambiguously(value, 1):
            continue
        break
    else:
        raise RuntimeError("could not draw a valid p19e_exponential_model_value hard(half-life) question")

    k_equiv = math.log(base) / h
    check(abs(A0 * math.exp(k_equiv * t) - value) < 1e-6, "hard(half-life): natural-log recompute disagrees")

    word = "half-life" if mode == "half_life" else "doubling time"
    verb = "decays" if mode == "half_life" else "grows"
    templates = (
        "{noun} starts at {A0} {unit} and {verb} with a {word} of {h} {period}s (t is also in {period}s). "
        "Find the {short} when t = {t}, to 1 decimal place.",
        "{noun} starts at {A0} {unit}. Its {word} is {h} {period}s. Find the value after t = {t} {period}s, to 1 decimal place.",
    )
    text = phrase(
        rng, templates, noun=noun, A0=A0, unit=unit, word=word, h=h, t=t, short=short, verb=verb, period=periods[0]
    )

    dp = 1
    answer_text = _display(value, dp)
    wrong_base = 2.0 if mode == "half_life" else 0.5
    candidates_val = [A0 * wrong_base ** (t / h)]  # used the opposite model
    candidates_val.append(A0 * base**t)  # forgot to divide by h
    candidates_val.append(A0 * base ** ((t + h) / h))  # off by one period
    candidates_val = [v for v in candidates_val if v > 0 and _finite_ok(v) and abs(v - value) > 0.06]

    return dict(
        text=text,
        answer=numeric_answer(round_half_up(Fraction(value), dp)),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, [_display(v, dp) for v in candidates_val], n=4),
    )


def _build_exp_exam(rng) -> dict:
    variant = rng.choice(("two_phase", "two_time_gap"))
    if variant == "two_phase":
        return _build_exp_exam_two_phase(rng)
    return _build_exp_exam_two_time_gap(rng)


def _build_exp_exam_two_phase(rng) -> dict:
    for _ in range(400):
        direction1 = rng.choice(("growth", "decay"))
        direction2 = rng.choice(("growth", "decay"))
        contexts = _GROWTH_CONTEXTS if direction1 == "growth" else _DECAY_CONTEXTS
        noun, unit, short, periods = rng.choice(contexts)
        period = _period_for(periods, rng.choice(_PERIODS))
        A0 = rng.randint(20, 150)
        r1 = rng.randint(2, 20)
        t1 = rng.randint(2, 5)
        r2 = rng.randint(2, 20)
        t2 = rng.randint(2, 5)
        factor1 = 1 + r1 / 100 if direction1 == "growth" else 1 - r1 / 100
        factor2 = 1 + r2 / 100 if direction2 == "growth" else 1 - r2 / 100
        if direction1 == direction2 and r1 == r2:
            # Not a rate change: the two phases must
            # differ, either in direction or in the per-period rate.
            continue
        mid = A0 * factor1**t1
        value = mid * factor2**t2
        if not (1 <= value <= 999):
            continue
        if not rounds_unambiguously(value, 1):
            continue
        break
    else:
        raise RuntimeError("could not draw a valid p19e_exponential_model_value exam(two-phase) question")

    check(abs(_sim_compound(A0, factor1, t1) - mid) < 1e-6, "exam(two-phase): phase-1 iterative recompute disagrees")
    check(abs(_sim_compound(mid, factor2, t2) - value) < 1e-6, "exam(two-phase): phase-2 iterative recompute disagrees")

    verb1 = "grows" if direction1 == "growth" else "decays"
    verb2 = "grows" if direction2 == "growth" else "decays"
    templates = (
        "{noun} starts at {A0} {unit} and {verb1} by {r1}% per {period} for {t1} {period}s, then "
        "{verb2} by {r2}% per {period} for a further {t2} {period}s. Find the {short} at the end, "
        "to 1 decimal place.",
        "{noun} starts at {A0} {unit}. It {verb1} by {r1}% per {period} for {t1} {period}s, and then "
        "{verb2} by {r2}% per {period} for the next {t2} {period}s. Find the value at the end, "
        "to 1 decimal place.",
    )
    text = phrase(
        rng,
        templates,
        noun=noun,
        A0=A0,
        unit=unit,
        verb1=verb1,
        verb2=verb2,
        r1=r1,
        r2=r2,
        t1=t1,
        t2=t2,
        period=period,
        short=short,
    )

    dp = 1
    answer_text = _display(value, dp)
    candidates_val = [A0 * factor2 ** (t1 + t2)]  # used only phase 2's rate throughout
    candidates_val.append(A0 * factor1 ** (t1 + t2))  # used only phase 1's rate throughout
    signed_r1 = r1 if direction1 == "growth" else -r1
    signed_r2 = r2 if direction2 == "growth" else -r2
    candidates_val.append(A0 * (1 + signed_r1 * t1 / 100 + signed_r2 * t2 / 100))  # added instead of chained
    candidates_val.append(A0 * factor1**t2 * factor2**t1)  # swapped the phase durations
    candidates_val.append(mid)  # forgot phase 2 entirely, reports the phase-1 value
    candidates_val.append(A0)  # forgot to apply either phase
    candidates_val = [v for v in candidates_val if v > 0 and _finite_ok(v) and abs(v - value) > 0.06]

    return dict(
        text=text,
        answer=numeric_answer(round_half_up(Fraction(value), dp)),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, [_display(v, dp) for v in candidates_val], n=4),
    )


def _build_exp_exam_two_time_gap(rng) -> dict:
    for _ in range(400):
        direction = rng.choice(("growth", "decay"))
        contexts = _GROWTH_CONTEXTS if direction == "growth" else _DECAY_CONTEXTS
        noun, unit, short, periods = rng.choice(contexts)
        period = _period_for(periods, rng.choice(_PERIODS))
        A0 = rng.randint(20, 150)
        r = rng.randint(2, 25)
        t1 = rng.randint(2, 6)
        t2 = t1 + rng.randint(2, 6)
        factor = 1 + r / 100 if direction == "growth" else 1 - r / 100
        val1 = A0 * factor**t1
        val2 = A0 * factor**t2
        gap = abs(val2 - val1)
        if not (1 <= val2 <= 999 and gap >= 1):
            continue
        if not rounds_unambiguously(gap, 1):
            continue
        break
    else:
        raise RuntimeError("could not draw a valid p19e_exponential_model_value exam(two-time-gap) question")

    check(abs(_sim_compound(A0, factor, t1) - val1) < 1e-6, "exam(two-time-gap): t1 iterative recompute disagrees")
    check(abs(_sim_compound(A0, factor, t2) - val2) < 1e-6, "exam(two-time-gap): t2 iterative recompute disagrees")

    verb = "grows" if direction == "growth" else "decays"
    change_word = "increase" if direction == "growth" else "decrease"
    templates = (
        "{noun} starts at {A0} {unit} and {verb} by {r}% every {period}. Find the {change_word} in the "
        "{short} between {t1} {period}s and {t2} {period}s, to 1 decimal place.",
        "{noun} starts at {A0} {unit} and {verb} by {r}% every {period}. By how much does the value "
        "{change_word} between t = {t1} {period}s and t = {t2} {period}s? Give your answer to 1 decimal place.",
    )
    text = phrase(
        rng,
        templates,
        noun=noun,
        A0=A0,
        unit=unit,
        verb=verb,
        r=r,
        period=period,
        t1=t1,
        t2=t2,
        short=short,
        change_word=change_word,
    )

    dp = 1
    answer_text = _display(gap, dp)
    candidates_val = [val2, val1]  # reports one endpoint instead of the gap
    candidates_val.append(A0 * r * (t2 - t1) / 100)  # linear-rate estimate of the gap, ignoring compounding
    candidates_val = [v for v in candidates_val if v > 0 and _finite_ok(v) and abs(v - gap) > 0.06]

    return dict(
        text=text,
        answer=numeric_answer(round_half_up(Fraction(gap), dp)),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, [_display(v, dp) for v in candidates_val], n=4),
    )


@generator(
    "p19e_exponential_model_value",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19e_exponential_model_value(rng, tier="easy"):
    """Given a practical exponential growth or decay context with a stated
    initial value, growth or decay rate, and elapsed time, compute the
    modelled quantity AT that time -- evaluation only, never solves for an
    elapsed time or a rate."""
    if tier == "easy":
        return _build_exp_easy(rng)
    if tier == "medium":
        return _build_exp_medium(rng)
    if tier == "hard":
        return _build_exp_hard(rng)
    if tier == "exam":
        return _build_exp_exam(rng)
    raise ValueError(f"unknown tier: {tier!r}")


# =============================================================================
# Row 2: p19e_log_model_solve_input (ACMMM158) -- solve a fully specified
# logarithmic model for its INPUT, given a stated target level
# =============================================================================

_LOG10_CONTEXTS = [
    # 5th field: the exam-tier "how many/much more ... is/are needed" clause,
    # named properly per quantity (countable vs a level), so that
    # "units of hours of practice" never appears as nonsense wording.
    (
        "a learner's test score after x weeks of practice",
        "score",
        "x",
        "weeks of practice",
        "many more weeks of practice are",
    ),
    (
        "the perceived loudness of a sound, in decibels, given its intensity level x",
        "loudness",
        "x",
        "the intensity level",
        "much more intensity is",
    ),
    (
        "a typist's speed, in words per minute, after x hours of practice",
        "speed",
        "x",
        "hours of practice",
        "many more hours of practice are",
    ),
]
_LN_CONTEXTS = [
    ("a seedling's height, in cm, given x grams of fertiliser applied", "height", "x", "grams of fertiliser"),
    ("the recall score of a memory task, given x minutes of study", "recall score", "x", "minutes of study"),
]
# A negative a makes y fall as x grows, so it is told only in a context where the quantity really
# falls with practice: a seedling growing less with more fertiliser read as nonsense.
_LN_FALLING_CONTEXTS = [
    (
        "the time, in seconds, a worker takes to assemble a part after x hours of practice",
        "assembly time",
        "x",
        "hours of practice",
    ),
    (
        "the number of errors per page a trainee typist makes after x hours of training",
        "error count",
        "x",
        "hours of training",
    ),
]
_SHIFTED_CONTEXTS = [
    ("a learner's test score t weeks after starting a course", "score", "t", "weeks"),
    ("a plant's height, in cm, t weeks after being planted", "height", "t", "weeks"),
]


def _model_value(x: float, a: float, b: float, base: float, c: float = 0.0) -> float:
    arg = x + c
    return a * (math.log10(arg) if base == 10 else math.log(arg)) + b


def _log10ish(x: float, base: float) -> float:
    return math.log10(x) if base == 10 else math.log(x)


_EASY_A_POOL = tuple(range(4, 31))


def _build_log_easy(rng) -> dict:
    for _ in range(400):
        a = rng.choice(_EASY_A_POOL)
        b = rng.randint(0, 25)
        e = rng.uniform(0.2, 2.4)
        y = round(a * e + b)
        exponent = (y - b) / a
        x_answer = 10**exponent
        if not (1.0 <= x_answer <= 300):
            continue
        if not rounds_unambiguously(x_answer, 1):
            continue
        break
    else:
        raise RuntimeError("could not draw a valid p19e_log_model_solve_input easy question")

    root = _bisection_solve(lambda x: _model_value(x, a, b, 10) - y, 1e-6, 1e7)
    check(abs(root - x_answer) < 1e-3 * max(1.0, x_answer), "easy: bisection recompute disagrees")

    subject, y_label, x_sym, x_label, _increase_clause = rng.choice(_LOG10_CONTEXTS)
    b_part = f" + {b}" if b > 0 else (f" - {abs(b)}" if b < 0 else "")
    model_text = f"y = {a} log10({x_sym}){b_part}"
    templates = (
        "The model y = {a} log10({x_sym}){b_part} gives {subject}. Find {x_sym} ({x_label}) "
        "when y = {y}, to 1 decimal place.",
        "{subject_cap} is modelled by {model}, where y is the {y_label}. Find {x_sym} when y = {y}, "
        "to 1 decimal place.",
    )
    text = phrase(
        rng,
        templates,
        a=a,
        x_sym=x_sym,
        b_part=b_part,
        subject=subject,
        subject_cap=subject[0].upper() + subject[1:],
        model=model_text,
        y_label=y_label,
        x_label=x_label,
        y=y,
    )

    dp = 1
    answer_text = _display(x_answer, dp)
    candidates_val = [10 ** (y / a)]  # forgot to subtract b first
    candidates_val.append((10 ** (y - b)) / a)  # divided by a after exponentiating
    candidates_val.append(math.exp((y - b) / a))  # wrong base (e instead of 10)
    candidates_val = [v for v in candidates_val if v > 0 and _finite_ok(v) and abs(v - x_answer) > 0.06]

    return dict(
        text=text,
        answer=numeric_answer(round_half_up(Fraction(x_answer), dp)),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, [_display(v, dp) for v in candidates_val], n=4),
    )


_MEDIUM_A_POOL = (
    Fraction(-9, 2),
    Fraction(-7, 2),
    Fraction(-5, 2),
    Fraction(-3, 2),
    Fraction(3, 2),
    Fraction(5, 2),
    Fraction(7, 2),
    Fraction(9, 2),
    Fraction(-4),
    Fraction(-3),
    Fraction(-2),
    Fraction(2),
    Fraction(3),
    Fraction(4),
)


def _build_log_medium(rng) -> dict:
    # The stated target y must lie in the real
    # range of the quantity the context names (never negative -- e.g. a
    # seedling's height cannot be -7 cm). Draw y DIRECTLY as a positive
    # integer target (construct to target), then derive b so the model is
    # solvable by construction: for a > 0, b is free as before; for a < 0,
    # (y - b) must be negative for x_answer = exp((y - b) / a) to land >= 1,
    # so b is drawn to guarantee b > y. This keeps the negative-a share
    # (roughly half the tier) intact by construction rather than by chance.
    for _ in range(400):
        a = rng.choice(_MEDIUM_A_POOL)
        af = float(a)
        y = rng.randint(1, 40)  # never 0: "height 0 cm" reads oddly
        if af > 0:
            b = rng.randint(-10, 10)
        else:
            # Widened from the initial (1, 15) draft: a narrow gap let almost
            # every negative-a draw clear the x_answer bounds check below
            # (the positive-a branch's free b only clears it ~1/3 of the
            # time), which pulled the negative-a share to ~73/200 -- measured
            # -- well outside the "roughly half" target. (1, 45) brings
            # the two branches' acceptance rates back in line (measured
            # ~91/200, i.e. inside [0.35, 0.65]) while still guaranteeing
            # b > y so (y - b) < 0 by construction.
            b = y + rng.randint(1, 45)
        exponent = (y - b) / af
        x_answer = math.exp(exponent)
        if not (1.0 <= x_answer <= 300):
            continue
        if not rounds_unambiguously(x_answer, 1):
            continue
        break
    else:
        raise RuntimeError("could not draw a valid p19e_log_model_solve_input medium question")

    root = _bisection_solve(lambda x: _model_value(x, af, b, math.e) - y, 1e-6, 1e7)
    check(abs(root - x_answer) < 1e-3 * max(1.0, x_answer), "medium: bisection recompute disagrees")

    subject, y_label, x_sym, x_label = rng.choice(_LN_CONTEXTS if af > 0 else _LN_FALLING_CONTEXTS)
    # Integer coefficients print bare ("2 ln(x)"); only a genuine fraction is
    # bracketed ("(5/2) ln(x)") -- a bare integer never needs the parens.
    is_fraction = a.denominator != 1
    a_text = f"{a.numerator}/{a.denominator}" if is_fraction else str(a.numerator)
    coef_display = f"({a_text})" if is_fraction else a_text
    b_part = f" + {b}" if b > 0 else (f" - {abs(b)}" if b < 0 else "")
    model_text = f"y = {coef_display} ln({x_sym}){b_part}"
    templates = (
        "The model y = {coef_display} ln({x_sym}){b_part} gives {subject}. Find {x_sym} ({x_label}) "
        "when y = {y}, to 1 decimal place.",
        "{subject_cap} is modelled by {model}, where y is the {y_label}. Find {x_sym} when y = {y}, "
        "to 1 decimal place.",
    )
    text = phrase(
        rng,
        templates,
        a_text=a_text,
        coef_display=coef_display,
        x_sym=x_sym,
        b_part=b_part,
        subject=subject,
        subject_cap=subject[0].upper() + subject[1:],
        model=model_text,
        y_label=y_label,
        x_label=x_label,
        y=y,
    )

    dp = 1
    answer_text = _display(x_answer, dp)
    candidates_val = [math.exp(y / af)]  # forgot to subtract b first
    candidates_val.append(math.exp(y - b) / af)  # divided by a after exponentiating
    candidates_val.append(10 ** ((y - b) / af))  # wrong base (10 instead of e)
    candidates_val = [v for v in candidates_val if v > 0 and _finite_ok(v) and abs(v - x_answer) > 0.06]

    return dict(
        text=text,
        answer=numeric_answer(round_half_up(Fraction(x_answer), dp)),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, [_display(v, dp) for v in candidates_val], n=4),
    )


def _build_log_hard(rng) -> dict:
    for _ in range(400):
        a = rng.choice(_EASY_A_POOL)
        b = rng.randint(0, 20)
        c = rng.randint(1, 5)
        e = rng.uniform(0.3, 2.4)
        y = round(a * e + b)
        exponent = (y - b) / a
        t_answer = 10**exponent - c
        if not (0.5 <= t_answer <= 300):
            continue
        if not rounds_unambiguously(t_answer, 1):
            continue
        break
    else:
        raise RuntimeError("could not draw a valid p19e_log_model_solve_input hard question")

    root = _bisection_solve(lambda t: _model_value(t, a, b, 10, c=c) - y, 1e-6, 1e7)
    check(abs(root - t_answer) < 1e-3 * max(1.0, t_answer), "hard: bisection recompute disagrees")

    subject, y_label, t_sym, t_label = rng.choice(_SHIFTED_CONTEXTS)
    b_part = f" + {b}" if b > 0 else (f" - {abs(b)}" if b < 0 else "")
    model_text = f"y = {a} log10({t_sym} + {c}){b_part}"
    templates = (
        "The model y = {a} log10({t_sym} + {c}){b_part} gives {subject}. Find {t_sym} ({t_label} "
        "elapsed) when y = {y}, to 1 decimal place.",
        "{subject_cap} is modelled by {model}, where y is the {y_label}. Find {t_sym} ({t_label} "
        "elapsed) when y = {y}, to 1 decimal place.",
    )
    text = phrase(
        rng,
        templates,
        a=a,
        t_sym=t_sym,
        c=c,
        b_part=b_part,
        subject=subject,
        subject_cap=subject[0].upper() + subject[1:],
        model=model_text,
        y_label=y_label,
        t_label=t_label,
        y=y,
    )

    dp = 1
    answer_text = _display(t_answer, dp)
    candidates_val = [10**exponent]  # forgot to subtract the shift c
    candidates_val.append(10 ** (y / a) - c)  # forgot to subtract b first
    candidates_val.append((10 ** (y - b)) / a - c)  # divided by a after exponentiating
    candidates_val = [v for v in candidates_val if v > 0 and _finite_ok(v) and abs(v - t_answer) > 0.06]

    return dict(
        text=text,
        answer=numeric_answer(round_half_up(Fraction(t_answer), dp)),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, [_display(v, dp) for v in candidates_val], n=4),
    )


def _build_log_exam(rng) -> dict:
    for _ in range(400):
        a = rng.choice(_EASY_A_POOL)
        b = rng.randint(0, 20)
        e1 = rng.uniform(0.3, 1.8)
        e2 = e1 + rng.uniform(0.2, 0.6)
        y1 = round(a * e1 + b)
        y2 = round(a * e2 + b)
        if y2 <= y1:
            continue
        x1 = 10 ** ((y1 - b) / a)
        x2 = 10 ** ((y2 - b) / a)
        gap = x2 - x1
        if not (1.0 <= x1 <= 300 and 1.0 <= x2 <= 300 and gap >= 1.0):
            continue
        if not rounds_unambiguously(gap, 1):
            continue
        break
    else:
        raise RuntimeError("could not draw a valid p19e_log_model_solve_input exam question")

    root1 = _bisection_solve(lambda x: _model_value(x, a, b, 10) - y1, 1e-6, 1e7)
    root2 = _bisection_solve(lambda x: _model_value(x, a, b, 10) - y2, 1e-6, 1e7)
    check(abs(root1 - x1) < 1e-3 * max(1.0, x1), "exam: bisection recompute (level 1) disagrees")
    check(abs(root2 - x2) < 1e-3 * max(1.0, x2), "exam: bisection recompute (level 2) disagrees")

    subject, y_label, x_sym, x_label, increase_clause = rng.choice(_LOG10_CONTEXTS)
    b_part = f" + {b}" if b > 0 else (f" - {abs(b)}" if b < 0 else "")
    model_text = f"y = {a} log10({x_sym}){b_part}"
    templates = (
        "The model y = {a} log10({x_sym}){b_part} gives {subject}. Find how {increase_clause} "
        "needed after y first reaches {y1} until y reaches {y2}, to 1 decimal place.",
        "{subject_cap} is modelled by {model}, where y is the {y_label}. Once y reaches {y1}, find how "
        "{increase_clause} needed for y to reach {y2}. Give your answer to 1 decimal place.",
    )
    text = phrase(
        rng,
        templates,
        a=a,
        x_sym=x_sym,
        b_part=b_part,
        subject=subject,
        subject_cap=subject[0].upper() + subject[1:],
        model=model_text,
        y_label=y_label,
        x_label=x_label,
        increase_clause=increase_clause,
        y1=y1,
        y2=y2,
    )

    dp = 1
    answer_text = _display(gap, dp)
    candidates_val = [x2, x1]  # reports one level's input instead of the gap
    candidates_val = [v for v in candidates_val if v > 0 and _finite_ok(v) and abs(v - gap) > 0.06]

    return dict(
        text=text,
        answer=numeric_answer(round_half_up(Fraction(gap), dp)),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, [_display(v, dp) for v in candidates_val], n=3),
    )


@generator(
    "p19e_log_model_solve_input",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19e_log_model_solve_input(rng, tier="easy"):
    """Given a stated practical scenario modelled by a fully specified
    logarithmic function, solve for the INPUT at which the modelled quantity
    reaches a stated target level -- the model is always fully specified,
    never fitted from a data point, and every tier solves for the input,
    never evaluates the model at a stated input."""
    if tier == "easy":
        return _build_log_easy(rng)
    if tier == "medium":
        return _build_log_medium(rng)
    if tier == "hard":
        return _build_log_hard(rng)
    if tier == "exam":
        return _build_log_exam(rng)
    raise ValueError(f"unknown tier: {tier!r}")
