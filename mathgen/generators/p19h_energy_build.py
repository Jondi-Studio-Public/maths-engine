"""Generates `p19h_energy_kwh_consumption` and `p19h_energy_kj_amount` questions on energy used by appliances and
energy amounts in kilojoules.

Senior energy computations, domain ``measurement``, ``answer: number``, two generators/topics:

- ``p19h_energy_kwh_consumption`` (skill
  ``senior-use-units-of-energy-to-describe-consumption-of``, ACMEM031) --
  given an appliance's stated POWER (kW or watts) and a stated DURATION
  (hours or minutes), compute the energy consumed in kilowatt-hours:
  ``energy_kWh = power_kW x duration_h``. Framing is always a household
  appliance / electricity-bill context.
- ``p19h_energy_kj_amount`` (skill
  ``senior-use-units-of-energy-to-describe-the-amount``, ACMEM033) -- given a
  physical activity's ENERGY-USE RATE (kJ per minute, hour or second) and a
  stated DURATION, compute the total energy used in kilojoules:
  ``energy_kJ = rate x duration`` (converted to matching units first).
  Framing is always a physical activity (walking, cycling, swimming,
  running, gym equipment) -- never food/Calories, which belongs to the
  shipping neighbour ``senior-use-units-of-energy-used-for-foods-including``.

Both topics never reduce to "convert one stated energy value between named
units" (that is the shipping neighbour
``senior-convert-from-one-unit-of-energy-to-another``'s shape): the learner
is always given TWO separate quantities (a power/rate and a duration) and
must multiply them, with a unit conversion on one or both quantities from
``medium`` tier up.

Four tiers per topic, each ADDING one step over the previous tier:

kWh consumption
- ``easy``: power in kW, duration in whole hours -- one multiplication.
- ``medium``: power in watts -- easy plus the W -> kW conversion.
- ``hard``: TWO appliances, both in watts, durations in hours -- medium
  plus a second appliance whose energy is summed.
- ``exam``: the same two appliances but the second run time is in MINUTES
  -- hard plus a minutes -> hours conversion.

kJ amount
- ``easy``: kJ/min x minutes. ``medium``: kJ/h x minutes (one conversion).
  ``hard``: two activities (one per-minute, one per-hour) summed. ``exam``:
  a kJ/s rate against an "H hours M minutes" duration (two chained
  conversions).

Realism: every appliance carries a REALISTIC power BAND (drawn
per question, so the power cannot be memorised per appliance) and a
realistic run-time range; an appliance that never runs for whole hours (a
microwave, a hair dryer) is only ever stated in minutes. Every activity
carries a rate band and a measuring source that fits it (a swim watch for
laps, the treadmill's console for a treadmill run).

Distractors: each is a genuine misconception at plausible
magnitude -- kept only when its rounded value is within a factor of 30 of
the key and not a near-zero value -- so no option can be dismissed by size.
The number of distractors BELOW the key is drawn (over the feasible ranks)
so the key's rank among the sorted options varies instead of being constant
per tier. The prompt never states the key as one of its own numbers.

Every displayed answer is rounded, via ``round_half_up``, to a precision
STATED in the question text, then formatted with ``stated_precision_string``
so the grading tolerance matches what is shown (kWh: 1 decimal place; kJ:
the nearest whole kilojoule). Every distractor is rounded to that SAME
precision before being offered. A raw answer sitting exactly halfway between
two grid values is redrawn (``_on_rounding_boundary``).

Every unit conversion is independently re-derived with ``check_conversion``
against a units-table ``factors`` dict, and the final multiplication is
checked a second way through the opposite conversion order via
``check_equivalent`` -- an independent route, not a restatement of the
construction arithmetic. Collisions are avoided by a bounded redraw loop with
``check_pairwise_distinct`` as the hard final invariant.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from ..core import (
    check,
    check_conversion,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    numeric_answer,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
    pick_distractors,
)

_MAX_ATTEMPTS = 200
_RATIO = 30  # every distractor within this factor of the key


def _on_rounding_boundary(value: Fraction, dp: int) -> bool:
    """True iff `value` sits EXACTLY halfway between two adjacent multiples
    of ``10 ** -dp`` -- the one case `round_half_up` cannot resolve without
    an arbitrary tie-break, so a learner rounding down is marked wrong for a
    defensible answer."""
    scaled = value * (10**dp) * 2
    return scaled.denominator == 1 and scaled.numerator % 2 == 1


def _plain_decimal(value: Fraction, places: int) -> str:
    """`value` (exact on the ``10**-places`` grid) at its shortest decimal
    form, at least one decimal place: 0.065, 1.2, 0.02, 3.0. NEVER interpolate
    a bare `Fraction` into text (its `str()` is "3/2")."""
    text = stated_precision_string(value, places)
    if "." in text:
        text = text.rstrip("0")
        if text.endswith("."):
            text += "0"
    return text


def _echoes(stem: str, key: Fraction) -> bool:
    """True iff the key (as a number) appears among the prompt's own numbers."""
    return any(Fraction(t) == key for t in re.findall(r"\d+(?:\.\d+)?", stem))


def _select_options(rng, key: Fraction, candidates: list, dp: int, floor: Fraction):
    """Choose three distractor VALUES (already rounded to `dp`) from the
    misconception `candidates`: each within `_RATIO` of the key, at least
    `floor`, distinct from the key and each other. The count of distractors
    below the key is drawn uniformly over the feasible ranks, so the key's
    rank among the four sorted options varies. None if fewer than three
    valid distinct candidates exist (caller redraws)."""
    unique: list[Fraction] = []
    for cand in candidates:
        value = round_half_up(cand, dp)
        if value == key or value < floor or value in unique:
            continue
        if not (key / _RATIO <= value <= key * _RATIO):
            continue
        unique.append(value)
    below = [v for v in unique if v < key]
    above = [v for v in unique if v > key]
    feasible = [r for r in range(4) if len(below) >= r and len(above) >= 3 - r]
    if not feasible:
        return None
    r = rng.choice(feasible)
    chosen = rng.sample(below, r) + rng.sample(above, 3 - r)
    rng.shuffle(chosen)
    return chosen


# ---------------------------------------------------------------------------
# kWh consumption -- household appliances, electricity-bill framing.
# ---------------------------------------------------------------------------

# (name, power band low W, high W, step W, whole-hours range or None,
#  minutes range (low, high, step) or None). Bands are typical rated powers
# for the appliance; an appliance with no hours range never runs for whole
# hours (it is only ever stated in minutes).
APPLIANCES: list[tuple] = [
    ("a laptop charger", 45, 90, 5, (4, 12), (120, 480, 30)),
    ("a desk fan", 40, 70, 5, (3, 10), (120, 420, 30)),
    ("a ceiling fan", 50, 90, 5, (4, 12), (180, 480, 30)),
    ("a set-top box", 20, 35, 5, (12, 24), (720, 1440, 60)),
    ("an LED television", 40, 150, 10, (2, 8), (90, 360, 30)),
    ("a games console", 90, 200, 10, (2, 6), (60, 240, 30)),
    ("a desktop computer", 150, 350, 10, (2, 8), (90, 360, 30)),
    ("an electric blanket", 60, 150, 10, (3, 9), (120, 480, 30)),
    ("a portable air conditioner", 900, 1500, 100, (2, 8), (90, 360, 30)),
    ("a bar heater", 1000, 2400, 200, (1, 5), (45, 240, 15)),
    ("an electric oven", 1800, 3000, 200, (1, 3), (30, 180, 15)),
    ("an electric clothes dryer", 2000, 3400, 200, (1, 3), (45, 180, 15)),
    ("a dishwasher", 1200, 2000, 100, (1, 2), (60, 150, 15)),
    ("a washing machine", 400, 900, 100, (1, 2), (45, 120, 15)),
    ("an upright vacuum cleaner", 600, 1400, 100, (1, 2), (20, 60, 10)),
    ("a microwave oven", 800, 1500, 100, None, (10, 45, 5)),
    ("a hair dryer", 1000, 2000, 100, None, (10, 30, 5)),
]

_HOURS_APPLIANCES = [a for a in APPLIANCES if a[4] is not None]

# Time-conversion units table, base unit = minutes -- shared by both topics.
_TIME_FACTORS_MIN = {"min": Fraction(1), "h": Fraction(60)}

# Power-conversion units table, base unit = watts.
_POWER_FACTORS_W = {"W": Fraction(1), "kW": Fraction(1000)}

_KWH_DP = 1
_KWH_KEY_MIN = Fraction(2, 5)  # 0.4 kWh: keeps every option well clear of 0.0
_KWH_OPTION_MIN = Fraction(2, 10)
_KWH_ASK = "To 1 decimal place, how much energy, in kilowatt-hours (kWh), does it use?"


def _kwh_text(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, _KWH_DP), _KWH_DP)


def _kwh_round(value: Fraction) -> Fraction:
    return round_half_up(value, _KWH_DP)


def _kw_text(kw: Fraction) -> str:
    return _plain_decimal(kw, 3)


def _hours_text(n: int) -> str:
    return f"{n} hour" if n == 1 else f"{n} hours"


def _cap(name: str) -> str:
    return name[0].upper() + name[1:]


def _draw_power(rng, appliance) -> int:
    _, lo, hi, step, _, _ = appliance
    return rng.choice(range(lo, hi + 1, step))


def _draw_hours(rng, appliance) -> int:
    lo, hi = appliance[4]
    return rng.randint(lo, hi)


def _draw_minutes(rng, appliance) -> int:
    lo, hi, step = appliance[5]
    return rng.choice(range(lo, hi + 1, step))


def _kwh_single_stem(rng, name: str, power_text: str, duration_text: str) -> str:
    template = rng.randrange(3)
    if template == 0:
        return f"{_cap(name)} has a power rating of {power_text}. It runs for {duration_text}."
    if template == 1:
        return f"The label on {name} states a power rating of {power_text}. It is switched on for {duration_text}."
    return f"In one day, {name} with a power rating of {power_text} is in use for {duration_text}."


def _kwh_result(rng, stem: str, key_raw: Fraction, candidates: list, working: list, ask: str = _KWH_ASK):
    """Assemble one kWh question, or None to redraw."""
    key = _kwh_round(key_raw)
    if key < _KWH_KEY_MIN or _on_rounding_boundary(key_raw, _KWH_DP):
        return None
    if _echoes(stem, key):
        return None
    options = _select_options(rng, key, candidates, _KWH_DP, _KWH_OPTION_MIN)
    if options is None:
        return None
    check_pairwise_distinct(key, *options, msg="kwh distractor collides after rounding")
    answer_text = _kwh_text(key_raw)
    text = f"{stem} {ask}"
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(key),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(
            answer_text,
            [_kwh_text(v) for v in options],
            tolerance=stated_precision_tolerance(_KWH_DP),
        ),
    )


def _build_kwh_easy(rng) -> dict:
    """easy: power already stated in kW, duration already stated in whole
    hours -- the decision this tier adds is recognising energy_kWh =
    power_kW × duration_h, with nothing to convert first."""
    for _ in range(_MAX_ATTEMPTS):
        appliance = rng.choice(_HOURS_APPLIANCES)
        name = appliance[0]
        watts = _draw_power(rng, appliance)
        hours = _draw_hours(rng, appliance)
        kw = Fraction(watts, 1000)

        answer = kw * hours
        candidates = [
            kw + hours,  # added instead of multiplied
            kw / hours,  # divided
            Fraction(hours) / kw,  # divided the other way round
            kw * (hours + 3),  # a longer run time assumed
            kw * (hours + 2),
            kw * (hours - 1),  # a shorter run time assumed
            kw * (hours - 2),
            kw * (hours - 3),
            answer * 10,  # decimal-place slip
            answer / 10,
        ]
        # Independent check: recompute via the raw-watts pivot.
        check_conversion(watts, "W", "kW", kw, _POWER_FACTORS_W, label="kwh_easy power (sanity)")
        check_equivalent(answer, Fraction(watts, 1000) * hours, "kwh_easy independent recompute failed")

        stem = _kwh_single_stem(rng, name, f"{_kw_text(kw)} kW", _hours_text(hours))
        working = [f"{_kw_text(kw)} kW × {hours} h = {_kwh_text(answer)} kWh."]
        result = _kwh_result(rng, stem, answer, candidates, working)
        if result is not None:
            return result
    check(False, "kwh_easy: could not build a valid question")


def _build_kwh_medium(rng) -> dict:
    """medium: power stated in WATTS instead of kW -- the decision this
    tier adds is converting power to kW (divide by 1000) before the same
    multiplication easy already asked for."""
    for _ in range(_MAX_ATTEMPTS):
        appliance = rng.choice(_HOURS_APPLIANCES)
        name = appliance[0]
        watts = _draw_power(rng, appliance)
        hours = _draw_hours(rng, appliance)
        kw = Fraction(watts, 1000)

        answer = kw * hours
        candidates = [
            Fraction(watts, 100) * hours,  # divided by 100 instead of 1000
            Fraction(watts, 10000) * hours,  # divided by 10000
            kw + hours,
            kw / hours,
            kw * (hours + 3),
            kw * (hours + 2),
            kw * (hours - 1),
            kw * (hours - 2),
            kw * (hours - 3),
        ]
        check_conversion(watts, "W", "kW", kw, _POWER_FACTORS_W, label="kwh_medium power")
        check_equivalent(answer, (Fraction(watts) * hours) / 1000, "kwh_medium independent recompute failed")

        stem = _kwh_single_stem(rng, name, f"{watts} W", _hours_text(hours))
        working = [f"{watts} W = {_kw_text(kw)} kW; {_kw_text(kw)} kW × {hours} h = {_kwh_text(answer)} kWh."]
        result = _kwh_result(rng, stem, answer, candidates, working)
        if result is not None:
            return result
    check(False, "kwh_medium: could not build a valid question")


def _two_appliance_stem(name1, w1, dur1, name2, w2, dur2) -> str:
    return (
        f"{_cap(name1)} has a power rating of {w1} W and runs for {dur1}. "
        f"{_cap(name2)} has a power rating of {w2} W and runs for {dur2}."
    )


_KWH_COMBINED_ASK = "To 1 decimal place, what is the combined energy, in kilowatt-hours (kWh), both appliances use?"


def _build_kwh_hard(rng) -> dict:
    """hard: TWO appliances, both stated in watts with run times in hours,
    each multiplied by its own duration and SUMMED -- the decision this tier
    adds over medium is the second appliance and the addition of the two
    energies."""
    for _ in range(_MAX_ATTEMPTS):
        a1, a2 = rng.sample(_HOURS_APPLIANCES, 2)
        w1, w2 = _draw_power(rng, a1), _draw_power(rng, a2)
        h1, h2 = _draw_hours(rng, a1), _draw_hours(rng, a2)
        kw1, kw2 = Fraction(w1, 1000), Fraction(w2, 1000)

        energy1, energy2 = kw1 * h1, kw2 * h2
        answer = energy1 + energy2
        candidates = [
            energy1,  # only the first appliance
            energy2,  # only the second appliance
            answer * 10,  # both powers divided by 100
            energy1 + energy2 * 10,  # second power divided by 100
            energy1 * 10 + energy2,
            (kw1 + kw2) * (h1 + h2),  # powers and durations each added first
            kw1 * h2 + kw2 * h1,  # the two durations swapped
            kw1 * h1 + kw2 * h1,  # first duration used for both
            kw1 * h2 + kw2 * h2,  # second duration used for both
            answer / 10,
        ]
        check_conversion(w1, "W", "kW", kw1, _POWER_FACTORS_W, label="kwh_hard power 1")
        check_conversion(w2, "W", "kW", kw2, _POWER_FACTORS_W, label="kwh_hard power 2")
        # Independent recompute: pivot both items through watt-hours.
        check_equivalent(
            answer, (Fraction(w1) * h1 + Fraction(w2) * h2) / 1000, "kwh_hard independent recompute failed"
        )

        stem = _two_appliance_stem(a1[0], w1, _hours_text(h1), a2[0], w2, _hours_text(h2))
        working = [
            f"{w1} W = {_kw_text(kw1)} kW; {_kw_text(kw1)} kW × {h1} h = {_kwh_text(energy1)} kWh; "
            f"{w2} W = {_kw_text(kw2)} kW; {_kw_text(kw2)} kW × {h2} h = {_kwh_text(energy2)} kWh; "
            f"total = {_kwh_text(answer)} kWh."
        ]
        result = _kwh_result(rng, stem, answer, candidates, working, _KWH_COMBINED_ASK)
        if result is not None:
            return result
    check(False, "kwh_hard: could not build a valid question")


def _build_kwh_exam(rng) -> dict:
    """exam: the same two-appliance sum as hard, but the second appliance's
    run time is stated in MINUTES -- the decision this tier adds over hard is
    converting minutes to hours for one of the two items."""
    for _ in range(_MAX_ATTEMPTS):
        a1 = rng.choice(_HOURS_APPLIANCES)
        a2 = rng.choice([a for a in APPLIANCES if a is not a1])
        w1, w2 = _draw_power(rng, a1), _draw_power(rng, a2)
        h1 = _draw_hours(rng, a1)
        m2 = _draw_minutes(rng, a2)
        kw1, kw2 = Fraction(w1, 1000), Fraction(w2, 1000)
        hours2 = Fraction(m2, 60)

        energy1, energy2 = kw1 * h1, kw2 * hours2
        answer = energy1 + energy2
        candidates = [
            energy1,
            energy2,
            energy1 + kw2 * m2,  # minutes used as if they were hours
            energy1 + kw2 * Fraction(m2, 100),  # minutes divided by 100
            (kw1 + kw2) * (h1 + hours2),  # powers and durations each added first
            kw1 * h1 + kw2 * h1,  # first duration used for both
            answer * 10,
            answer / 10,
            energy1 + kw2 * Fraction(m2, 30),  # minutes divided by 30
        ]
        check_conversion(w1, "W", "kW", kw1, _POWER_FACTORS_W, label="kwh_exam power 1")
        check_conversion(w2, "W", "kW", kw2, _POWER_FACTORS_W, label="kwh_exam power 2")
        check_conversion(m2, "min", "h", hours2, _TIME_FACTORS_MIN, label="kwh_exam time")
        # Independent recompute: watt-minutes pivot for the whole sum.
        check_equivalent(
            answer,
            (Fraction(w1) * h1 * 60 + Fraction(w2) * m2) / Fraction(1000 * 60),
            "kwh_exam independent recompute failed",
        )

        stem = _two_appliance_stem(a1[0], w1, _hours_text(h1), a2[0], w2, f"{m2} minutes")
        working = [
            f"{w1} W = {_kw_text(kw1)} kW; {_kw_text(kw1)} kW × {h1} h = {_kwh_text(energy1)} kWh; "
            f"{w2} W = {_kw_text(kw2)} kW; {m2} min = {stated_precision_string(round_half_up(hours2, 2), 2)} h; "
            f"{_kw_text(kw2)} kW × {stated_precision_string(round_half_up(hours2, 2), 2)} h = {_kwh_text(energy2)} kWh; "
            f"total = {_kwh_text(answer)} kWh."
        ]
        result = _kwh_result(rng, stem, answer, candidates, working, _KWH_COMBINED_ASK)
        if result is not None:
            return result
    check(False, "kwh_exam: could not build a valid question")


@generator(
    "p19h_energy_kwh_consumption",
    level=19,
    presets={
        "easy": dict(mode="kwh_easy"),
        "medium": dict(mode="kwh_medium"),
        "hard": dict(mode="kwh_hard"),
        "exam": dict(mode="kwh_exam"),
    },
)
def p19h_energy_kwh_consumption(rng, mode="kwh_easy"):
    """Given a stated appliance power (kW or watts) and a stated duration
    (hours or minutes), compute the energy consumed in kilowatt-hours; see
    module docstring for the four tiers' escalation."""
    if mode == "kwh_easy":
        return _build_kwh_easy(rng)
    if mode == "kwh_medium":
        return _build_kwh_medium(rng)
    if mode == "kwh_hard":
        return _build_kwh_hard(rng)
    if mode == "kwh_exam":
        return _build_kwh_exam(rng)
    raise ValueError(f"unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# kJ amount used in an activity -- physical-activity framing, never food.
# ---------------------------------------------------------------------------

# (activity, measuring source that fits it, rate band low, high in kJ per
# minute). The rate is drawn from the band per question.
ACTIVITIES: list[tuple[str, str, int, int]] = [
    ("walking at a brisk pace", "a smartwatch", 9, 18),
    ("cycling on a stationary bike", "the bike's console", 15, 30),
    ("swimming laps", "a swim watch", 24, 36),
    ("using a rowing machine", "the rowing machine's display", 24, 36),
    ("doing a gym circuit workout", "a fitness tracker", 18, 30),
    ("using an elliptical trainer", "the elliptical trainer's console", 21, 30),
    ("running on a treadmill", "the treadmill's console", 30, 42),
    ("playing a game of basketball", "a heart-rate monitor", 24, 36),
    ("doing a home workout video", "a fitness tracker", 15, 24),
    ("hiking uphill", "a GPS watch", 21, 33),
    ("dancing in a fitness class", "a fitness tracker", 15, 24),
    ("jogging around the block", "a running watch", 21, 33),
    ("using a cross-trainer machine", "the cross-trainer's display", 18, 27),
    ("doing a spin class", "the studio bike's console", 27, 39),
]

# Time-conversion units table, base unit = seconds -- exam tier only.
_TIME_FACTORS_SEC = {"s": Fraction(1), "min": Fraction(60), "h": Fraction(3600)}

_KJ_DP = 0
_KJ_OPTION_MIN = Fraction(5)
_KJ_ASK = "To the nearest kilojoule, how much total energy is used?"
_KJ_COMBINED_ASK = "To the nearest kilojoule, what is the combined total energy used by both activities?"


def _kj_text(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, _KJ_DP), _KJ_DP)


def _draw_rate(rng, activity, multiple_of: int = 1) -> int:
    _, _, lo, hi = activity
    return rng.choice([r for r in range(lo, hi + 1) if r % multiple_of == 0])


def _kj_result(rng, stem: str, key_raw: Fraction, candidates: list, working: list, ask: str = _KJ_ASK):
    """Assemble one kJ question, or None to redraw."""
    key = round_half_up(key_raw, _KJ_DP)
    if _echoes(stem, key):
        return None
    options = _select_options(rng, key, candidates, _KJ_DP, _KJ_OPTION_MIN)
    if options is None:
        return None
    check_pairwise_distinct(key, *options, msg="kj distractor collides after rounding")
    answer_text = _kj_text(key_raw)
    text = f"{stem} {ask}"
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(key),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(
            answer_text,
            [_kj_text(v) for v in options],
            tolerance=stated_precision_tolerance(_KJ_DP),
        ),
    )


def _build_kj_easy(rng) -> dict:
    """easy: rate already stated in kJ per minute, duration already stated
    in whole minutes -- the decision this tier adds is recognising
    energy_kJ = rate × duration, with nothing to convert first."""
    for _ in range(_MAX_ATTEMPTS):
        activity = rng.choice(ACTIVITIES)
        name, source = activity[0], activity[1]
        rate = _draw_rate(rng, activity)
        duration = rng.randint(5, 90)

        answer = Fraction(rate) * duration
        candidates = [
            Fraction(rate + duration),  # added
            Fraction(rate) * (duration + 10),
            Fraction(rate) * (duration + 5),
            Fraction(rate) * (duration - 10) if duration > 10 else Fraction(0),
            Fraction(rate),  # reported the rate itself
            answer * 10,
            answer / 10,
            Fraction(rate) * duration / 2,  # halved
        ]
        # Independent recompute: 10-minute-block decomposition.
        check_equivalent(answer, rate * 10 * Fraction(duration, 10), "kj_easy independent recompute failed")

        stem = (
            f"While {name}, {source} shows an energy-use rate of {rate} kJ per minute. "
            f"The activity lasts {duration} minutes."
        )
        working = [f"{rate} kJ/min × {duration} min = {_kj_text(answer)} kJ."]
        result = _kj_result(rng, stem, answer, candidates, working)
        if result is not None:
            return result
    check(False, "kj_easy: could not build a valid question")


def _build_kj_medium(rng) -> dict:
    """medium: rate stated in kJ per HOUR while the duration is stated in
    MINUTES -- the decision this tier adds is converting one of the two
    quantities before the same multiplication easy already asked for."""
    for _ in range(_MAX_ATTEMPTS):
        activity = rng.choice(ACTIVITIES)
        name, source = activity[0], activity[1]
        rate_min = _draw_rate(rng, activity)
        rate_hour = rate_min * 60
        duration = rng.randint(5, 90)

        hours = Fraction(duration, 60)
        answer = rate_hour * hours
        rate_per_min = Fraction(rate_hour, 60)
        check_equivalent(answer, rate_per_min * duration, "kj_medium two conversion routes disagree")
        candidates = [
            Fraction(rate_hour, 30) * hours,  # divided by 30 instead of 60
            Fraction(rate_hour) * duration / 100,  # minutes divided by 100
            Fraction(rate_hour) * math.ceil(hours),  # time rounded up to whole hours
            Fraction(rate_hour) * duration / 10,  # divided by 10 instead of 60
            rate_per_min + duration,  # added
            rate_per_min * (duration + 10),
            rate_per_min * (duration - 10) if duration > 10 else Fraction(0),
        ]
        check_conversion(duration, "min", "h", hours, _TIME_FACTORS_MIN, label="kj_medium time (duration->hours)")
        check(rate_per_min * 60 == rate_hour, "kj_medium rate (per-hour->per-minute) mismatch")

        stem = (
            f"While {name}, {source} shows an energy-use rate of {rate_hour} kJ per hour. "
            f"The activity lasts {duration} minutes."
        )
        working = [
            f"{duration} min = {stated_precision_string(round_half_up(hours, 3), 3)} h; "
            f"{rate_hour} kJ/h × {stated_precision_string(round_half_up(hours, 3), 3)} h = {_kj_text(answer)} kJ."
        ]
        result = _kj_result(rng, stem, answer, candidates, working)
        if result is not None:
            return result
    check(False, "kj_medium: could not build a valid question")


def _build_kj_hard(rng) -> dict:
    """hard: TWO activities, one stated in kJ per minute (direct) and one
    in kJ per hour (still needing medium's conversion), each multiplied by
    its own duration and SUMMED."""
    for _ in range(_MAX_ATTEMPTS):
        act1, act2 = rng.sample(ACTIVITIES, 2)
        rate1 = _draw_rate(rng, act1)
        rate2_min = _draw_rate(rng, act2)
        rate2_hour = rate2_min * 60
        d1, d2 = rng.randint(5, 60), rng.randint(5, 60)

        energy1 = Fraction(rate1) * d1
        hours2 = Fraction(d2, 60)
        energy2 = rate2_hour * hours2
        answer = energy1 + energy2
        candidates = [
            energy1,
            energy2,
            energy1 + Fraction(rate2_hour) * d2 / 100,  # second: divided by 100 not 60
            energy1 + Fraction(rate2_hour) * d2 / 10,  # second: divided by 10 not 60
            energy1 + Fraction(rate2_hour, 30) * hours2,  # second: rate divided by 30
            Fraction(rate1) * d2 + Fraction(rate2_min) * d1,  # the two durations swapped
            (rate1 + rate2_min) * (d1 + d2),  # rates and durations each added first
            answer * 10,
            answer / 10,
        ]
        check_conversion(d2, "min", "h", hours2, _TIME_FACTORS_MIN, label="kj_hard time")
        rate2_per_min = Fraction(rate2_hour, 60)
        check_equivalent(answer, Fraction(rate1) * d1 + rate2_per_min * d2, "kj_hard independent recompute failed")

        stem = (
            f"While {act1[0]}, {act1[1]} shows a rate of {rate1} kJ per minute for {d1} minutes. "
            f"While {act2[0]}, {act2[1]} shows a rate of {rate2_hour} kJ per hour for {d2} minutes."
        )
        working = [
            f"{rate1} kJ/min × {d1} min = {_kj_text(energy1)} kJ; {d2} min = "
            f"{stated_precision_string(round_half_up(hours2, 3), 3)} h, {rate2_hour} kJ/h x "
            f"{stated_precision_string(round_half_up(hours2, 3), 3)} h = {_kj_text(energy2)} kJ; "
            f"total = {_kj_text(answer)} kJ."
        ]
        result = _kj_result(rng, stem, answer, candidates, working, _KJ_COMBINED_ASK)
        if result is not None:
            return result
    check(False, "kj_hard: could not build a valid question")


def _build_kj_exam(rng) -> dict:
    """exam: ONE activity, rate stated in kJ per SECOND **and** duration
    stated in a mixed hours-and-minutes form -- the decision this tier adds
    is chaining TWO conversions (combining the duration's two parts, and
    matching that unit to the per-second rate) before the single
    multiplication."""
    for _ in range(_MAX_ATTEMPTS):
        activity = rng.choice(ACTIVITIES)
        name, source = activity[0], activity[1]
        # Per-minute rates divisible by 3 display exactly at 2 decimal places
        # as kJ per second (rate_min / 60 = k / 20).
        rate_min = _draw_rate(rng, activity, multiple_of=3)
        rate_sec = Fraction(rate_min, 60)
        hours_part = rng.choice([1, 2])
        minutes_part = rng.choice(range(5, 60))

        total_minutes = hours_part * 60 + minutes_part
        total_seconds = total_minutes * 60

        answer = rate_sec * total_seconds
        candidates = [
            rate_sec * 60 * minutes_part,  # dropped the hour part
            rate_sec * 3600 * hours_part,  # dropped the minutes part
            rate_sec * 60 * (hours_part * 100 + minutes_part),  # 1 hour = 100 minutes
            rate_sec * 3600 * (hours_part + Fraction(minutes_part, 100)),  # minutes as a decimal of an hour
            rate_sec * 100 * total_minutes,  # 1 minute = 100 seconds
            answer * 10,
            answer / 10,
        ]
        check_conversion(hours_part, "h", "min", hours_part * 60, _TIME_FACTORS_MIN, label="kj_exam hour part")
        check_conversion(total_minutes, "min", "s", total_seconds, _TIME_FACTORS_SEC, label="kj_exam total duration")
        check_equivalent(answer, (rate_sec * 60) * total_minutes, "kj_exam independent recompute failed")

        rate_sec_text = _plain_decimal(rate_sec, 2)
        stem = (
            f"While {name}, {source} shows an energy-use rate of {rate_sec_text} kJ per second. "
            f"The activity lasts {hours_part} hour" + ("s" if hours_part != 1 else "") + f" {minutes_part} minutes."
        )
        working = [
            f"{hours_part} h {minutes_part} min = {total_minutes} min = {total_seconds} s; "
            f"{rate_sec_text} kJ/s × {total_seconds} s = {_kj_text(answer)} kJ."
        ]
        result = _kj_result(rng, stem, answer, candidates, working)
        if result is not None:
            return result
    check(False, "kj_exam: could not build a valid question")


@generator(
    "p19h_energy_kj_amount",
    level=19,
    presets={
        "easy": dict(mode="kj_easy"),
        "medium": dict(mode="kj_medium"),
        "hard": dict(mode="kj_hard"),
        "exam": dict(mode="kj_exam"),
    },
)
def p19h_energy_kj_amount(rng, mode="kj_easy"):
    """Given a stated activity energy-use rate (kJ per minute, hour or
    second) and a stated duration, compute the total energy used in
    kilojoules; see module docstring for the four tiers' escalation."""
    if mode == "kj_easy":
        return _build_kj_easy(rng)
    if mode == "kj_medium":
        return _build_kj_medium(rng)
    if mode == "kj_hard":
        return _build_kj_hard(rng)
    if mode == "kj_exam":
        return _build_kj_exam(rng)
    raise ValueError(f"unknown mode: {mode!r}")
