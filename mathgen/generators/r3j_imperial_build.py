"""Imperial lengths and their metric equivalents: `r3j_imperial` (easy, medium, hard) and the exam chain
`exd_x_r3j_imperial_voyage`.

``convert-imperial-length`` (``measurement``, answer ``number``). Every conversion factor the question needs is
printed in the question; none is recalled.

* easy: one imperial factor, in either direction (feet and inches, yards and feet, miles and yards).
* medium: a chain of factors ending in a metric unit, or a metric length written in feet.
* hard: a length written in two units added up and converted, or nautical miles converted to miles through
  kilometres.
* exam (``exd_x_r3j_imperial_voyage``): two legs of a voyage, one in nautical miles and one in miles, as kilometres,
  their total, and the total in nautical miles.

Each key is rebuilt a second way, by converting to inches (or metres) first and then to the asked unit.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._exd_distinct_parts import distinct_part_answers
from ._r3j_kit import PEOPLE, number_part, number_text, plain, sane, single_number

_IN = {"inch": Fraction(1), "foot": Fraction(12), "yard": Fraction(36), "mile": Fraction(63360)}
_FACT = {
    ("foot", "inch"): "1 foot = 12 inches",
    ("yard", "foot"): "1 yard = 3 feet",
    ("mile", "yard"): "1 mile = 1760 yards",
}
_PLURAL = {"inch": "inches", "foot": "feet", "yard": "yards", "mile": "miles"}


def _is_tie(value, places) -> bool:
    scaled = Fraction(value) * 10**places
    return (scaled - int(scaled)) * 2 in (1, -1)


def _easy(rng):
    (big, small) = rng.choice(list(_FACT))
    factor = int(_IN[big] / _IN[small])
    fact = _FACT[(big, small)]
    name = rng.choice(PEOPLE)
    if rng.random() < 0.5:  # larger unit to smaller: multiply
        value = Fraction(rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 20])) + rng.choice(
            [0, 0, Fraction(1, 2), Fraction(1, 4)]
        )
        if big == "mile":
            value = Fraction(rng.choice([2, 3, 4, 5, 6, 8])) + rng.choice([0, Fraction(1, 2)])
        key = value * factor
        frm, to = big, small
        slips = [
            (value / factor, "r3j-imperial-divided-for-multiplied"),
            (value * (12 if factor != 12 else 3), "r3j-imperial-wrong-factor"),
            (value + factor, "r3j-imperial-added-factor"),
            (value * (factor + 1), "r3j-imperial-factor-off-by-one"),
        ]
    else:  # smaller unit to larger: divide
        key = Fraction(rng.choice([2, 3, 4, 5, 6, 8, 9, 10, 12])) + rng.choice([0, Fraction(1, 2), Fraction(1, 4)])
        if big == "mile":
            key = Fraction(rng.choice([2, 3, 4, 5, 6])) + rng.choice([0, Fraction(1, 2), Fraction(1, 4)])
        value = key * factor
        frm, to = small, big
        slips = [
            (value * factor, "r3j-imperial-multiplied-for-divided"),
            (value / (factor + 1), "r3j-imperial-factor-off-by-one"),
            (value / (12 if factor != 12 else 3), "r3j-imperial-wrong-factor"),
            (value - factor, "r3j-imperial-subtracted-factor"),
        ]
    check(key * _IN[to] == value * _IN[frm], "conversion disagrees when built through inches")
    slips += [(key * 10, "r3j-imperial-decimal-slip"), (key / 10, "r3j-imperial-decimal-slip")]
    thing = rng.choice(
        {
            ("foot", "inch"): ["a length of timber", "a curtain rail", "a bookshelf"],
            ("yard", "foot"): ["a roll of fabric", "a garden path", "a length of rope"],
            ("mile", "yard"): ["a walking track", "a bush trail", "a cycling lap"],
        }[(big, small)]
    )
    text = (
        f"{name} measures {thing} as {plain(value)} {_PLURAL[frm]}. Using {fact}, how many {_PLURAL[to]} is that? "
        "Give your answer as a number."
    )
    working = [
        f"{fact}, so "
        + (f"multiply by {factor}" if frm == big else f"divide by {factor}")
        + f": {plain(value)} -> {plain(key)} {_PLURAL[to]}."
    ]
    return single_number(text, key, 2, working, sane(slips, key))


def _medium(rng):
    name = rng.choice(PEOPLE)
    kind = rng.choice(["feet_to_m", "yards_to_m", "m_to_feet"])
    if kind == "feet_to_m":
        value = Fraction(rng.choice([5, 6, 8, 9, 12, 15, 18, 20, 25, 30, 40]))
        key = value * 12 * Fraction("2.54") / 100
        facts = "1 foot = 12 inches and 1 inch = 2.54 centimetres"
        thing = rng.choice(["a boat", "a fence panel", "a shipping container", "a driveway"])
        text = f"{name} measures {thing} as {plain(value)} feet long. Using {facts}, find its length in metres, to 2 decimal places."
        slips = [
            (value * 12 * Fraction("2.54"), "r3j-imperial-centimetres-given"),
            (value * Fraction("2.54") / 100, "r3j-imperial-feet-treated-as-inches"),
            (value * 12 / Fraction("2.54") / 100, "r3j-imperial-divided-for-multiplied"),
            (value * 12 * Fraction("2.54") / 10, "r3j-imperial-decimal-slip"),
        ]
        working = [
            f"{plain(value)} feet = {plain(value * 12)} inches = {number_text(value * 12 * Fraction('2.54'), 2)} cm = {number_text(key, 2)} m."
        ]
        places = 2
    elif kind == "yards_to_m":
        value = Fraction(rng.choice([10, 15, 20, 22, 25, 40, 50, 75, 100, 110]))
        key = value * 3 * 12 * Fraction("2.54") / 100
        facts = "1 yard = 3 feet, 1 foot = 12 inches and 1 inch = 2.54 centimetres"
        thing = rng.choice(["a cricket pitch", "a swimming pool", "a football field", "a running track straight"])
        text = f"{name} reads that {thing} is {plain(value)} yards long. Using {facts}, find its length in metres, to 1 decimal place."
        slips = [
            (value * 3 * 12 * Fraction("2.54"), "r3j-imperial-centimetres-given"),
            (value * 12 * Fraction("2.54") / 100, "r3j-imperial-step-left-out"),
            (value * 3 * Fraction("2.54") / 100, "r3j-imperial-step-left-out"),
            (value * 3 * 12 / Fraction("2.54") / 100, "r3j-imperial-divided-for-multiplied"),
        ]
        working = [
            f"{plain(value)} yards = {plain(value * 3)} feet = {plain(value * 36)} inches = {number_text(value * 36 * Fraction('2.54'), 1)} cm = {number_text(key, 1)} m."
        ]
        places = 1
    else:
        metres = Fraction(rng.choice([3, 4, 6, 7, 9, 12, 15, 18, 24]))
        feet = metres * 100 / Fraction("2.54") / 12
        key = feet
        facts = "1 inch = 2.54 centimetres and 1 foot = 12 inches"
        thing = rng.choice(["a ceiling", "a boat", "a garden fence", "a ramp"])
        text = f"{name} measures {thing} as {plain(metres)} metres. Using {facts}, find its length in feet, to 1 decimal place."
        slips = [
            (metres * 100 / Fraction("2.54"), "r3j-imperial-inches-given"),
            (metres * 100 * Fraction("2.54") / 12, "r3j-imperial-multiplied-for-divided"),
            (metres / Fraction("2.54") / 12, "r3j-imperial-metres-treated-as-centimetres"),
            (metres * 100 / 12, "r3j-imperial-step-left-out"),
        ]
        working = [
            f"{plain(metres)} m = {plain(metres * 100)} cm = {number_text(metres * 100 / Fraction('2.54'), 2)} inches = {number_text(feet, 1)} feet."
        ]
        places = 1
    check(not _is_tie(key, places), "a rounding tie")
    return single_number(text, key, places, working, sane(slips, key))


def _hard_trail(rng):
    name = rng.choice(PEOPLE)
    for _ in attempts(500, "a total clear of a rounding tie"):
        m1, y1 = rng.randint(1, 6), rng.choice([220, 440, 880, 1100, 1320])
        m2, y2 = rng.randint(1, 5), rng.choice([220, 440, 880, 1100, 1320])
        total_yards = (m1 + m2) * 1760 + y1 + y2
        key = Fraction(total_yards, 1760) * Fraction("1.609")
        if not _is_tie(key, 2):
            break
    text = (
        f"{name} walks two sections of a trail. The first is {m1} {'mile' if m1 == 1 else 'miles'} {y1} yards and the second is {m2} {'mile' if m2 == 1 else 'miles'} "
        f"{y2} yards. Using 1 mile = 1760 yards and 1 mile = 1.609 kilometres, find the total distance in "
        "kilometres, to 2 decimal places."
    )
    slips = [
        (Fraction(m1 + m2) * Fraction("1.609") + Fraction(y1 + y2, 100), "r3j-imperial-yards-as-hundredths"),
        (Fraction(m1 + m2) * Fraction("1.609"), "r3j-imperial-yards-left-out"),
        (Fraction(total_yards) * Fraction("1.609"), "r3j-imperial-yards-treated-as-miles"),
        (Fraction(total_yards, 1760) / Fraction("1.609"), "r3j-imperial-divided-for-multiplied"),
    ]
    working = [
        f"Total yards = ({m1} + {m2}) x 1760 + {y1} + {y2} = {total_yards}.",
        f"In miles: {total_yards} / 1760 = {number_text(Fraction(total_yards, 1760), 3)}; in km: x 1.609 = {number_text(key, 2)}.",
    ]
    return single_number(text, key, 2, working, sane(slips, key))


def _hard_nautical(rng):
    name = rng.choice(PEOPLE)
    for _ in attempts(500, "a distance clear of a rounding tie"):
        nm = rng.choice([12, 18, 24, 30, 36, 45, 60, 75, 90, 120])
        key = Fraction(nm) * Fraction("1.852") / Fraction("1.609")
        if not _is_tie(key, 1):
            break
    text = (
        f"A yacht skipper tells {name} that the next harbour is {nm} nautical miles away. Using 1 nautical mile = "
        f"1.852 kilometres and 1 mile = 1.609 kilometres, find the distance in miles, to 1 decimal place."
    )
    slips = [
        (Fraction(nm) * Fraction("1.852"), "r3j-imperial-kilometres-given"),
        (Fraction(nm) * Fraction("1.609") / Fraction("1.852"), "r3j-imperial-factors-inverted"),
        (Fraction(nm) * Fraction("1.852") * Fraction("1.609"), "r3j-imperial-multiplied-for-divided"),
        (Fraction(nm), "r3j-imperial-nautical-mile-as-mile"),
    ]
    working = [
        f"{nm} nautical miles = {nm} x 1.852 = {number_text(Fraction(nm) * Fraction('1.852'), 3)} km.",
        f"In miles: {number_text(Fraction(nm) * Fraction('1.852'), 3)} / 1.609 = {number_text(key, 1)} miles.",
    ]
    return single_number(text, key, 1, working, sane(slips, key))


@generator(
    "r3j_imperial",
    level=8,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3j_imperial(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_trail, _hard_nautical])(rng)


# -- exam ---------------------------------------------------------------------------------------------------------

_NM = Fraction("1.852")
_MI = Fraction("1.609")


@generator("exd_x_r3j_imperial_voyage", level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3j_imperial_voyage(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    name = rng.choice(PEOPLE)
    for _ in attempts(500, "legs clear of rounding ties"):
        leg1 = rng.choice([12, 18, 24, 30, 36, 42, 48])
        leg2 = rng.choice([10, 14, 16, 22, 26, 28, 34])
        a, b = Fraction(leg1) * _NM, Fraction(leg2) * _MI
        total = a + b
        in_nm = total / _NM
        if not (_is_tie(a, 2) or _is_tie(b, 2) or _is_tie(total, 2) or _is_tie(in_nm, 1)):
            break
    text = phrase(
        rng,
        [
            "A yacht sails from a marina to an island and then on to a harbour. The first leg is {leg1} nautical "
            "miles and the second leg is {leg2} miles. Use 1 nautical mile = 1.852 kilometres and 1 mile = 1.609 "
            "kilometres.",
            "A charter yacht leaves a marina, calls at an island and finishes at a harbour. The first leg is {leg1} "
            "nautical miles and the second leg is {leg2} miles. Use 1 nautical mile = 1.852 kilometres and 1 mile = "
            "1.609 kilometres.",
            "A sailing crew logs two legs of a voyage: {leg1} nautical miles from a marina to an island, then {leg2} "
            "miles from the island to a harbour. Use 1 nautical mile = 1.852 kilometres and 1 mile = 1.609 "
            "kilometres.",
        ],
        leg1=leg1,
        leg2=leg2,
    )
    parts = [
        number_part(
            "Find the length of the first leg in kilometres, to 2 decimal places.",
            a,
            2,
            1,
            [f"{leg1} x 1.852 = {number_text(a, 2)} km."],
            [
                (Fraction(leg1) * _MI, "r3j-imperial-wrong-unit-factor"),
                (Fraction(leg1) / _NM, "r3j-imperial-divided-for-multiplied"),
                (Fraction(leg1), "r3j-imperial-nautical-mile-as-kilometre"),
            ],
        ),
        number_part(
            "Find the length of the second leg in kilometres, to 2 decimal places.",
            b,
            2,
            1,
            [f"{leg2} x 1.609 = {number_text(b, 2)} km."],
            [
                (Fraction(leg2) * _NM, "r3j-imperial-wrong-unit-factor"),
                (Fraction(leg2) / _MI, "r3j-imperial-divided-for-multiplied"),
                (Fraction(leg2), "r3j-imperial-mile-as-kilometre"),
            ],
        ),
        number_part(
            "Find the total distance sailed, in kilometres, to 2 decimal places.",
            total,
            2,
            2,
            [f"{number_text(a, 2)} + {number_text(b, 2)} = {number_text(total, 2)} km."],
            [
                (Fraction(leg1 + leg2) * _NM, "r3j-imperial-one-factor-for-both"),
                (Fraction(leg1 + leg2) * _MI, "r3j-imperial-one-factor-for-both"),
                (Fraction(leg1 + leg2), "r3j-imperial-units-added-without-converting"),
            ],
            follow_through="part_a + part_b",
        ),
        number_part(
            "Find the total distance sailed, in nautical miles, to 1 decimal place.",
            in_nm,
            1,
            2,
            [f"{number_text(total, 2)} / 1.852 = {number_text(in_nm, 1)} nautical miles."],
            [
                (total * _NM, "r3j-imperial-multiplied-for-divided"),
                (total / _MI, "r3j-imperial-wrong-unit-factor"),
                (Fraction(leg1 + leg2), "r3j-imperial-units-added-without-converting"),
            ],
            follow_through="part_c / 1.852",
        ),
    ]
    return dict(text=text, parts=parts)
