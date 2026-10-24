"""
Generators for this stageB journey time/cost and grams-milligrams conversion.

Skill 1: senior-calculate-the-time-or-costs-for-a-journey (ACMEM087)
  - journey_time_or_cost: variants time, round_trip_cost, fuel_cost

Skill 2: senior-convert-between-grams-and-milligrams (ACMEM100)
  - grams_milligrams_convert: variants g_to_mg, mg_to_g, mixed_unit_total
"""

from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

from mathgen.core import fixed_point_string, generator, pick_distractors, rational_text


def _shown(value, places: int):
    """``value`` as a fixed-point string with exactly ``places`` decimal places,
    or ``None`` when it cannot be shown at that many places without rounding
    (such a value would give the key's shape away, so it is not offered)."""
    exact = Fraction(str(round(float(value), 6)))
    scaled = exact * 10**places
    if scaled.denominator != 1:
        return None
    return fixed_point_string(scaled.numerator, places)


def _number_question(text, answer, candidates, *, latex=None, working=None, answer_text=None):
    """Create a number-answer question with distractors.

    Every distractor is shown with the key's own number of decimal places
    (none for a whole-number key), so a learner cannot spot the key by its
    shape; a misconception value that cannot be shown that way is dropped and
    plain nearby values fill in behind the misconceptions."""
    if answer_text is None:
        answer_text = rational_text(answer) if isinstance(answer, Fraction) else str(answer)
    places = len(answer_text.partition(".")[2])
    step = Fraction(1, 10**places)
    fillers = [answer + 1, answer - 1, answer + 2, answer - 2, float(answer + 5 * step), float(answer - 5 * step)]
    shaped = []
    for c in list(candidates) + [f for f in fillers if f >= 0]:
        shown = _shown(c, places)
        if shown is not None:
            shaped.append(shown)
    d = dict(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working or [],
        distractors=pick_distractors(answer_text, shaped),
    )
    if latex is not None:
        d["latex"] = latex
    return d


@generator(
    "journey_time_or_cost",
    level=5,
    presets={
        "easy": dict(variant="time"),
        "medium": dict(variant="round_trip_cost"),
        "hard": dict(variant="fuel_cost"),
    },
)
def journey_time_or_cost(rng, variant="time"):
    """
    Calculate journey time or costs. Variants represent genuine reasoning escalation:
    - time: single-step division (distance / speed * 60)
    - round_trip_cost: two-step doubling then multiplying
    - fuel_cost: three-step chained calculation through intermediate unit
    """
    money_text = None
    if variant == "time":
        # Easy: time in minutes given distance (km) and speed (km/h)
        # d / s * 60 = minutes. Derive d from s and target minutes to ensure exact result.
        speed_options = [40, 50, 60, 80, 100, 120]
        minute_options = [15, 30, 45, 60, 90, 120]
        s = rng.choice(speed_options)
        minutes = rng.choice(minute_options)
        d = s * minutes // 60  # exact by construction

        answer = minutes
        text = f"A journey of {d} km at an average speed of {s} km/h takes how many minutes?"
        hours_unit = "hour" if d == s else "hours"
        working = [
            f"Time = distance / speed = {d} / {s} = {d / s:g} {hours_unit}.",
            f"Convert to minutes: {d / s:g} × 60 = {minutes} minutes.",
        ]

        # Misconceptions:
        # 1. inverts-the-ratio: s / d * 60
        inverted = (s * 60) // d if (s * 60) % d == 0 else (s * 60) / d
        # 2. forgets-the-minutes-conversion: d / s (hours, not minutes)
        no_conversion = Fraction(d, s)

        candidates = [
            int(inverted) if isinstance(inverted, int) else inverted,
            float(no_conversion),
            minutes + rng.randint(10, 40),
            max(0, minutes - rng.randint(5, 20)),
        ]

    elif variant == "round_trip_cost":
        # Medium: round trip cost (2 * d * c)
        # d is distance (km), c is cost per km (dollars)
        d = rng.randint(100, 500)
        c = rng.choice([0.50, 0.75, 1.00, 1.25, 1.50, 2.00])

        answer = round(2 * d * c, 2)
        text = f"A journey costs ${c:.2f} per km. What is the cost of a return trip if the one-way distance is {d} km?"
        working = [f"Round trip distance: 2 × {d} = {2 * d} km.", f"Cost: {2 * d} × ${c:.2f} = ${answer}"]

        # Misconceptions:
        # 1. forgets-round-trip: d * c
        one_way = round(d * c, 2)
        # 2. adds-a-flat-fee-error: 2 * d * c + d
        flat_fee_error = round(2 * d * c + d, 2)

        candidates = [
            one_way,
            flat_fee_error,
            round(2 * d * c / 2, 2),  # half trip
            round(2 * d * c * 1.1, 2),  # 10% more
        ]

    elif variant == "fuel_cost":
        # Hard: fuel cost ((d / 100) * k * p)
        # d = distance (km)
        # k = litres per 100 km
        # p = dollars per litre
        d = rng.choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        k = rng.choice([6, 7.5, 8, 9, 10, 12])
        p = rng.choice([1.50, 1.80, 2.00])

        # Compute exactly via Fraction, then round the money to the cent (half up) as the stem says
        fuel_used = Fraction(d, 100) * Fraction(k).limit_denominator()
        cost_fraction = fuel_used * Fraction(p).limit_denominator()
        cost = Decimal(cost_fraction.numerator) / Decimal(cost_fraction.denominator)
        cents = cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        answer = float(cents)
        money_text = f"{cents:.2f}"

        text = (
            f"A car uses {k} litres per 100 km and fuel costs ${p:.2f} per litre. "
            f"What is the total fuel cost for a {d} km journey, to the nearest cent?"
        )
        working = [
            f"Litres used: {d} / 100 × {k} = {float(fuel_used):.2f} litres.",
            f"Cost: {float(fuel_used):.2f} × ${p:.2f} = ${money_text}",
        ]

        # Misconceptions:
        # 1. skips-the-per-100-scaling: d * k * p (treats k as litres/km)
        skip_scaling = d * k * p
        # 2. prices-the-distance-not-the-fuel: d * p
        price_distance = d * p

        candidates = [
            round(skip_scaling, 2),
            round(price_distance, 2),
            round(answer * 2, 2),  # doubled
            round(answer * 0.5, 2),  # halved
        ]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return _number_question(text, answer, candidates, working=working, answer_text=money_text)


@generator(
    "grams_milligrams_convert",
    level=5,
    presets={
        "easy": dict(variant="g_to_mg"),
        "medium": dict(variant="mg_to_g"),
        "hard": dict(variant="mixed_unit_total"),
    },
)
def grams_milligrams_convert(rng, variant="g_to_mg"):
    """
    Convert between grams and milligrams. Variants represent genuine reasoning escalation:
    - g_to_mg: multiply by 1000 (forward direction)
    - mg_to_g: divide by 1000 (reverse direction)
    - mixed_unit_total: add quantities in different units (two-step)

    Conversion factor: 1 g = 1000 mg (always)
    """
    if variant == "g_to_mg":
        # Easy: grams to milligrams (multiply by 1000)
        g = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])

        if g == int(g):
            answer = int(g * 1000)
        else:
            answer = int(g * 1000)  # Always an integer since g is X.5

        # Handle singular/plural correctly
        g_unit = "gram" if g == 1 else "grams"

        text = f"Convert {g} {g_unit} to milligrams."
        working = [f"{g} g × 1000 = {answer} mg."]

        # Misconceptions:
        # 1. factor-of-100-error: multiply by 100 instead of 1000
        factor_100 = g * 100
        # 2. wrong-direction: divide by 1000
        wrong_dir = g / 1000

        candidates = [
            int(factor_100) if factor_100 == int(factor_100) else factor_100,
            wrong_dir,
            answer + rng.randint(100, 500),
            max(0, answer - rng.randint(100, 500)),
        ]

    elif variant == "mg_to_g":
        # Medium: milligrams to grams (divide by 1000, reverse direction)
        # mg must be a multiple of 100 so result is exact to at most 1 decimal place
        mg = rng.choice(
            [
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
                1500,
                2000,
                2500,
                3000,
                3500,
                4000,
                4500,
                5000,
                5500,
                6000,
                6500,
                7000,
                7500,
                8000,
                8500,
                9000,
                9500,
            ]
        )

        answer_frac = Fraction(mg, 1000)
        answer = float(answer_frac) if answer_frac.denominator != 1 else int(answer_frac)

        text = f"Convert {mg} milligrams to grams."
        working = [f"{mg} mg ÷ 1000 = {answer} g."]

        # Misconceptions:
        # 1. factor-of-100-error: divide by 100 instead of 1000
        factor_100 = mg / 100
        # 2. wrong-direction: multiply by 1000
        wrong_dir = mg * 1000

        candidates = [
            int(factor_100) if factor_100 == int(factor_100) else factor_100,
            int(wrong_dir) if isinstance(wrong_dir, float) and wrong_dir == int(wrong_dir) else wrong_dir,
            round(answer * 2, 2),
            round(answer / 2, 2),
        ]

    elif variant == "mixed_unit_total":
        # Hard: mixed-unit addition (G grams and M milligrams, total in mg or g)
        # G is 1..9, M is multiple of 100 in 100..900
        G = rng.randint(1, 9)
        M = rng.choice([100, 200, 300, 400, 500, 600, 700, 800, 900])

        # Handle singular/plural correctly
        g_unit = "gram" if G == 1 else "grams"
        mg_unit = "milligram" if M == 1 else "milligrams"

        # Alternate target unit
        target_unit = rng.choice(["mg", "g"])

        if target_unit == "mg":
            answer = G * 1000 + M
            text = f"What is the total mass in milligrams of {G} {g_unit} and {M} {mg_unit}?"
        else:
            # total in g = G + M / 1000
            answer_frac = Fraction(G) + Fraction(M, 1000)
            answer = float(answer_frac) if answer_frac.denominator != 1 else int(answer_frac)
            text = f"What is the total mass in grams of {G} {g_unit} and {M} {mg_unit}?"

        if target_unit == "mg":
            working = [f"{G} g = {G * 1000} mg, so the total is {G * 1000} + {M} = {answer} mg."]
        else:
            working = [f"{M} mg = {M / 1000:g} g, so the total is {G} + {M / 1000:g} = {answer} g."]

        # Misconceptions:
        # 1. forgets-to-convert-the-grams-component: G + M (adds raw digits)
        forgets_convert = G + M if target_unit == "mg" else G + M / 1000

        # 2. converts-the-wrong-component: G + M * 1000 (for "total in mg")
        # or M * 1000 + G (for "total in g")
        if target_unit == "mg":
            wrong_component = G + M * 1000
        else:
            wrong_component = (M * 1000 + G) / 1000

        candidates = [
            int(forgets_convert) if forgets_convert == int(forgets_convert) else forgets_convert,
            int(wrong_component)
            if isinstance(wrong_component, float) and wrong_component == int(wrong_component)
            else wrong_component,
            answer + rng.randint(10, 100) if target_unit == "mg" else answer + rng.uniform(0.1, 1),
            max(0, answer - rng.randint(10, 100)) if target_unit == "mg" else max(0, answer - rng.uniform(0.1, 1)),
        ]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return _number_question(text, answer, candidates, working=working)
