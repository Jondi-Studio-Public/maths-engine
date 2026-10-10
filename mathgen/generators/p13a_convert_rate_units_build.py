"""Convert units of rates occurring in practical situations to solve problems --
``senior-convert-units-of-rates-occurring-in-practical-situations`` (``acara-senior-v8-4``, ACMEM015, domain
``measurement``, ``answer: number``).

Concrete scope (generic-template contract): given
a practical scenario whose rate is quoted in one unit and whose other stated
quantity or requirement is expressed in a DIFFERENT, incompatible unit,
convert the rate to a compatible unit as an intermediate step, then use the
converted rate to compute a FURTHER practical quantity (a time, a distance,
a total cost or a total consumption) that is NOT itself the converted rate.

Boundary against the two shipping neighbours (read closely, this is the
whole point of the skill):

- ``senior-convert-between-units-for-rates-for-example-km`` (``convert_units_
  for_rates``) ends with the converted rate ITSELF as the answer ("What is
  its speed in m/s?"). Every tier here converts a rate and then goes one
  step further -- the converted rate never IS the reported answer; the
  bare-converted-rate value is instead offered as a distractor at every
  tier (the "stopped after converting, forgot to finish" misconception).
- ``senior-use-rates-to-make-comparisons`` (p7d) ends in a DIFFERENCE
  between two named entities' rates or prices. No tier here ever compares
  two entities -- every question is a single scenario computing a single
  further quantity for one entity.

Four tiers, escalating by DECISION (never by magnitude alone):

- easy: a speed given in km/h converted to m/s (intermediate step), then
  multiplied by a stated time (already in the converted unit's time frame)
  to give a DISTANCE in metres. One conversion, one multiplication.
- medium: same conversion direction and context, but the final question asks
  for the TIME needed to cover a stated distance -- division instead of
  easy's multiplication, a genuinely different final operation.
- hard: a fuel-consumption rate quoted per a non-standard compound unit
  ("C litres per 100 km") needs its OWN internal reduction (divide by 100 to
  get litres per km) before a SEPARATE cross-unit conversion (the trip
  distance is stated in metres, requiring division by 1000 to reach km) --
  two reduction/conversion steps, then a total-consumption (litres) answer.
- exam: two INDEPENDENT rates -- a price rate (cents per litre -> dollars
  per litre) and a consumption rate (litres per 100 km -> litres per km) --
  are both converted and then composed into a single TOTAL COST for a
  stated trip distance. A genuinely different reasoning move (multi-rate
  composition) from hard's single-rate double-reduction.

Every intermediate value and the final answer are exact ``Fraction``
arithmetic throughout (never a re-rounded intermediate); a
division that is not guaranteed to terminate at the stated precision is
rounded exactly once, at the very end, with `round_half_up`, guarded by
`rounds_unambiguously` so no seed lands on a rounding-boundary coin flip.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    currency_from_cents,
    generator,
    latex_answer,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)

_RESAMPLE_ATTEMPTS = 200

# km/h <-> m/s conversion factor: 1 km = 1000 m, 1 h = 3600 s, so
# 1 km/h = 1000/3600 m/s = 5/18 m/s.
_KMH_TO_MS = Fraction(5, 18)
_MS_TO_KMH = Fraction(18, 5)


def _kmh_text(value: Fraction) -> str:
    return stated_precision_string(value, 1)


def _plain_int(value) -> str:
    return str(int(value))


# Phrasing banks -- the numeric slots are what the oracle/tests parse out; the wording around them
# is what `mathgen.text_quality.prompt_shape` masks and compares, so at least 3 distinct
# subject/verb combinations are needed per tier. (subject, verb, pronoun, do-verb) for the
# easy/medium tiers.
_TRAVEL_SUBJECTS = (
    ("A car", "travels", "it", "does"),
    ("A motorcyclist", "rides", "they", "do"),
    ("A train", "moves", "it", "does"),
)

# Vehicle nouns for the hard/exam fuel-consumption scenarios.
_FUEL_VEHICLES = ("car", "truck", "delivery van")


# =============================================================================
# easy: km/h -> m/s, then rate x time -> distance (metres)
# =============================================================================


def _verify_easy(speed_kmh: Fraction, t: int, speed_ms: int, distance: int):
    """Independent route: re-derive m/s from the STATED km/h value via the
    conversion factor (rather than reusing the `k` the construction started
    from), then re-derive the distance from that."""
    recomputed_ms = speed_kmh * _KMH_TO_MS
    check_equivalent(recomputed_ms, speed_ms, "easy: km/h -> m/s conversion must match construction")
    check_equivalent(recomputed_ms * t, distance, "easy: distance must equal converted speed x time")


def _easy(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        k = rng.randint(5, 25)  # speed in m/s (the converted rate)
        t = rng.randint(10, 90)  # time in seconds
        speed_kmh = Fraction(k) * _MS_TO_KMH
        distance = k * t

        wrong_skip = round_half_up(speed_kmh * t, 0)  # used km/h value as if it were m/s
        wrong_direction = round_half_up(speed_kmh * _MS_TO_KMH * t, 0)  # multiplied by 18/5 instead of 5/18
        wrong_bare_rate = k  # stopped after converting; never multiplied by time

        try:
            check_pairwise_distinct(distance, wrong_skip, wrong_direction, wrong_bare_rate)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p13a_convert_rate_units: no collision-free easy draw found")

    _verify_easy(speed_kmh, t, k, distance)

    subject, verb, pronoun, do_verb = rng.choice(_TRAVEL_SUBJECTS)
    text = (
        f"{subject} {verb} at {_kmh_text(speed_kmh)} km/h. How far, in metres, "
        f"{do_verb} {pronoun} travel in {t} seconds?"
    )
    answer_text = _plain_int(distance)
    candidates = [_plain_int(wrong_skip), _plain_int(wrong_direction), _plain_int(wrong_bare_rate)]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "easy tier expects 3 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=distance,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"Convert: {_kmh_text(speed_kmh)} km/h x 5/18 = {k} m/s.",
            f"Distance = {k} m/s x {t} s = {distance} m.",
        ],
        distractors=distractors,
    )


# =============================================================================
# medium: km/h -> m/s, then distance / rate -> time (seconds)
# =============================================================================


def _verify_medium(speed_kmh: Fraction, distance: int, speed_ms: int, t: int):
    """Independent route: re-derive m/s from the stated km/h value, then
    re-derive time by Fraction division of the stated distance -- the
    reverse operation from the multiplication that built `distance`."""
    recomputed_ms = speed_kmh * _KMH_TO_MS
    check_equivalent(recomputed_ms, speed_ms, "medium: km/h -> m/s conversion must match construction")
    check_equivalent(Fraction(distance, 1) / recomputed_ms, t, "medium: time must equal distance / converted speed")


def _medium(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        k = rng.randint(5, 25)
        t = rng.randint(10, 90)
        speed_kmh = Fraction(k) * _MS_TO_KMH
        distance = k * t

        wrong_skip = round_half_up(Fraction(distance, 1) / speed_kmh, 0)
        wrong_direction = round_half_up(Fraction(distance, 1) / (speed_kmh * _MS_TO_KMH), 0)
        wrong_bare_rate = k  # reports the converted rate itself, never divides by the distance

        try:
            check_pairwise_distinct(t, wrong_skip, wrong_direction, wrong_bare_rate)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p13a_convert_rate_units: no collision-free medium draw found")

    _verify_medium(speed_kmh, distance, k, t)

    subject, verb, pronoun, do_verb = rng.choice(_TRAVEL_SUBJECTS)
    text = (
        f"{subject} {verb} at {_kmh_text(speed_kmh)} km/h. How many seconds {do_verb} {pronoun} take "
        f"to travel {distance} metres?"
    )
    answer_text = _plain_int(t)
    candidates = [_plain_int(wrong_skip), _plain_int(wrong_direction), _plain_int(wrong_bare_rate)]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "medium tier expects 3 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=t,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"Convert: {_kmh_text(speed_kmh)} km/h x 5/18 = {k} m/s.",
            f"Time = {distance} m / {k} m/s = {t} s.",
        ],
        distractors=distractors,
    )


# =============================================================================
# hard: compound-unit reduction (L per 100 km -> L per km) THEN a cross-unit
# conversion (trip distance stated in metres -> km) -> total litres consumed
# =============================================================================


def _verify_hard(consumption_per_100km: int, distance_m: int, distance_km: int, litres: Fraction):
    """Independent route: re-derive km from the STATED metres figure (a
    separate division from the one used to derive the litres-per-km rate),
    then recombine -- two independently re-derived steps, not one route
    replayed twice."""
    recomputed_km = Fraction(distance_m, 1) / 1000
    check_equivalent(recomputed_km, distance_km, "hard: m -> km conversion must match construction")
    rate_per_km = Fraction(consumption_per_100km, 100)
    check_equivalent(rate_per_km * recomputed_km, litres, "hard: total litres must equal rate-per-km x distance-in-km")


def _hard(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        c = rng.randint(4, 20)  # litres per 100 km
        distance_km = rng.randint(40, 450)
        distance_m = distance_km * 1000

        litres = Fraction(c, 100) * distance_km

        wrong_drop_internal = Fraction(c) * distance_km  # forgot to divide by 100
        wrong_drop_cross_unit = Fraction(c, 100) * distance_m  # forgot to divide metres by 1000
        wrong_direction = Fraction(c, 100) * (distance_m * 1000)  # multiplied by 1000 instead of dividing

        try:
            check_pairwise_distinct(litres, wrong_drop_internal, wrong_drop_cross_unit, wrong_direction)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p13a_convert_rate_units: no collision-free hard draw found")

    _verify_hard(c, distance_m, distance_km, litres)

    vehicle = rng.choice(_FUEL_VEHICLES)
    text = (
        f"A {vehicle} consumes {c} litres of fuel per 100 km. For a trip of {distance_m} m, "
        f"how many litres of fuel does the {vehicle} use, to 2 decimal places?"
    )
    answer_text = stated_precision_string(litres, 2)
    candidates = [
        stated_precision_string(wrong_drop_internal, 2),
        stated_precision_string(wrong_drop_cross_unit, 2),
        stated_precision_string(wrong_direction, 2),
    ]
    tol = stated_precision_tolerance(2)
    distractors = pick_distractors(answer_text, candidates, n=3, tolerance=tol)
    check(len(distractors) == 3, "hard tier expects 3 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=float(litres),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"Reduce the compound rate: {c} L per 100 km = {c}/100 = {stated_precision_string(Fraction(c, 100), 2)} L/km.",
            f"Convert distance: {distance_m} m = {distance_m}/1000 = {distance_km} km.",
            f"Total fuel = {stated_precision_string(Fraction(c, 100), 2)} L/km x {distance_km} km = {answer_text} L.",
        ],
        distractors=distractors,
    )


# =============================================================================
# exam: two independent rates -- price (cents/L -> $/L) and consumption
# (L per 100 km -> L per km) -- composed into a total cost for a trip
# =============================================================================


def _verify_exam(
    price_cents_per_L: int,
    consumption_per_100km: int,
    distance_km: int,
    price_per_L: Fraction,
    consumption_per_km: Fraction,
    exact_dollars: Fraction,
):
    """Independent route: re-derive both converted rates from the STATED
    integers (cents-per-litre, litres-per-100km), then recombine -- the same
    two conversions the construction performed, re-derived rather than
    reused."""
    recomputed_price = Fraction(price_cents_per_L, 100)
    recomputed_consumption = Fraction(consumption_per_100km, 100)
    check_equivalent(
        recomputed_price, price_per_L, "exam: cents -> dollars-per-litre conversion must match construction"
    )
    check_equivalent(
        recomputed_consumption, consumption_per_km, "exam: L-per-100km -> L-per-km reduction must match construction"
    )
    check_equivalent(
        recomputed_price * recomputed_consumption * distance_km,
        exact_dollars,
        "exam: total cost must equal price-per-L x consumption-per-km x distance-km",
    )


def _exam(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        p = rng.randint(80, 320)  # fuel price, cents per litre
        c = rng.randint(4, 20)  # fuel consumption, litres per 100 km
        distance_km = rng.randint(100, 900)

        price_per_L = Fraction(p, 100)
        consumption_per_km = Fraction(c, 100)
        exact_dollars = price_per_L * consumption_per_km * distance_km

        if not rounds_unambiguously(float(exact_dollars), 2):
            continue
        answer_dollars = round_half_up(exact_dollars, 2)

        wrong_omit_consumption = price_per_L * distance_km  # treats distance_km as litres consumed
        wrong_omit_price = consumption_per_km * distance_km  # reports litres used, as if $1/L
        wrong_drop_reduction = price_per_L * c * distance_km  # treats c as L/km directly (drops /100)

        try:
            check_pairwise_distinct(
                answer_dollars,
                wrong_omit_consumption,
                wrong_omit_price,
                wrong_drop_reduction,
            )
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p13a_convert_rate_units: no collision-free exam draw found")

    _verify_exam(p, c, distance_km, price_per_L, consumption_per_km, exact_dollars)

    vehicle = rng.choice(_FUEL_VEHICLES)
    text = (
        f"Fuel costs {p} cents per litre. A {vehicle} consumes {c} litres of fuel per 100 km. "
        f"For a trip of {distance_km} km, find the total cost of fuel, to the nearest cent."
    )
    answer_text = currency_from_cents(int(answer_dollars * 100))
    candidates = [
        currency_from_cents(int(wrong_omit_consumption * 100)),
        currency_from_cents(int(wrong_omit_price * 100)),
        currency_from_cents(int(wrong_drop_reduction * 100)),
    ]
    tol = stated_precision_tolerance(2)
    distractors = pick_distractors(answer_text, candidates, n=3, tolerance=tol)
    check(len(distractors) == 3, "exam tier expects 3 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=float(answer_dollars),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"Price: {p} cents/L = ${stated_precision_string(price_per_L, 2)}/L.",
            f"Consumption: {c} L per 100 km = {stated_precision_string(consumption_per_km, 2)} L/km.",
            f"Total cost = ${stated_precision_string(price_per_L, 2)}/L x {stated_precision_string(consumption_per_km, 2)} L/km "
            f"x {distance_km} km = {answer_text}.",
        ],
        distractors=distractors,
    )


@generator(
    "p13a_convert_rate_units",
    level=11,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
        "exam": dict(mode="exam"),
    },
)
def p13a_convert_rate_units(rng, mode="easy"):
    """Convert units of rates occurring in practical situations to solve
    problems. Every tier converts a rate to a compatible unit as an
    intermediate step, then uses the converted rate to compute a FURTHER
    practical quantity -- never the converted rate itself.

    easy: km/h -> m/s, then rate x time -> distance.
    medium: km/h -> m/s, then distance / rate -> time (the reverse operation).
    hard: a compound-unit rate (L per 100 km) needs its own internal
    reduction, THEN a separate cross-unit conversion (trip distance in
    metres -> km), before a total-consumption answer.
    exam: two independent rates (a price rate and a consumption rate) are
    each converted, then composed into one total-cost computation.
    """
    if mode == "easy":
        return _easy(rng)
    if mode == "medium":
        return _medium(rng)
    if mode == "hard":
        return _hard(rng)
    if mode == "exam":
        return _exam(rng)
    raise ValueError(f"unknown mode: {mode!r}")
