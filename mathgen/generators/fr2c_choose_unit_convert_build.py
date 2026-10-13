"""Generates `fr2c_choose_unit_convert` and `fr2c_choose_unit_convert_exam` questions: pick a sensible metric unit, then
compare and total measures given in different units.

One skill: ``f10-choose-unit-and-convert`` (``answer: mcq``). Three tiers, all multiple choice with four options:

- easy: the most sensible metric unit for a stated length, mass or capacity ("the length of a pencil"). The wrong options
  are a unit that is too small, a unit that is too large and a unit that measures something else.
- medium: four measures of one attribute written in different units (3.2 m, 310 cm, 3 050 mm and so on); which is the
  longest, shortest, heaviest or lightest. The largest number is never the largest measure.
- hard: a total, or an amount left over, of three measures written in two units, found in the smaller unit.

``fr2c_choose_unit_convert_exam`` is the chain: (a) the sensible unit for a small amount (mcq), (b) the total of three
ingredients in the smaller unit, (c) how many equal portions that fills (follows through from (b)), (d) what is left
over when some of them are used (follows through from (b) and (c)).

All amounts are exact fractions of the smallest unit; the printed numbers are rebuilt from them.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase

from . import _fr2c_kit as kit

_NAMES = ("Mia", "Tom", "Aiko", "Dev", "Leila", "Sam", "Jess", "Kofi", "Priya", "Marcus", "Zoe", "Ben")

# unit -> size in the smallest unit of its attribute
_UNITS = {
    "length": {"mm": 1, "cm": 10, "m": 1000, "km": 1_000_000},
    "mass": {"mg": 1, "g": 1000, "kg": 1_000_000, "t": 1_000_000_000},
    "capacity": {"mL": 1, "L": 1000, "kL": 1_000_000},
}
_NAMES_OF = {
    "mm": "millimetres",
    "cm": "centimetres",
    "m": "metres",
    "km": "kilometres",
    "mg": "milligrams",
    "g": "grams",
    "kg": "kilograms",
    "t": "tonnes",
    "mL": "millilitres",
    "L": "litres",
    "kL": "kilolitres",
}
_ATTRIBUTE_OF = {u: a for a, units in _UNITS.items() for u in units}

# (thing, attribute, best unit)
_THINGS = (
    ("the length of a pencil", "length", "cm"),
    ("the thickness of a coin", "length", "mm"),
    ("the height of a classroom door", "length", "m"),
    ("the distance by road from Adelaide to Melbourne", "length", "km"),
    ("the length of a netball court", "length", "m"),
    ("the length of an ant", "length", "mm"),
    ("the width of a mobile phone", "length", "cm"),
    ("the length of a bike ride from Perth to Fremantle", "length", "km"),
    ("the mass of a grain of salt", "mass", "mg"),
    ("the mass of a pencil", "mass", "g"),
    ("the mass of a watermelon", "mass", "kg"),
    ("the mass of a loaded semi-trailer", "mass", "t"),
    ("the mass of a slice of bread", "mass", "g"),
    ("the mass of a bag of flour", "mass", "kg"),
    ("the mass of a school bus", "mass", "t"),
    ("the amount of medicine on a teaspoon", "capacity", "mL"),
    ("the amount of water in a bath", "capacity", "L"),
    ("the amount of water in a swimming pool", "capacity", "kL"),
    ("the amount of petrol in a car's fuel tank", "capacity", "L"),
    ("the amount of water in a school's rainwater tank", "capacity", "kL"),
    ("the amount of milk in a cup of tea", "capacity", "mL"),
    ("the amount of water in a bucket", "capacity", "L"),
)


def _fmt(value: Fraction) -> str:
    """A measure as a plain number: whole, or with up to three decimal places and no trailing zeros."""
    if value.denominator == 1:
        return str(value.numerator)
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    check(Fraction(text) == value, f"{value} is not a short decimal")
    return text


def _short_decimal(value: Fraction, places: int = 3) -> bool:
    return (value * 10**places).denominator == 1


def _measure(value_in_smallest: Fraction, unit: str) -> Fraction:
    return value_in_smallest / _UNITS[_ATTRIBUTE_OF[unit]][unit]


# ------------------------------------------------------------------------------------------ easy


def _easy(rng):
    thing, attr, best = rng.choice(_THINGS)
    units = list(_UNITS[attr])
    i = units.index(best)
    others = [u for u in units if u != best]
    wrong_attr = rng.choice([u for a in _UNITS if a != attr for u in _UNITS[a]])
    wrong = []
    for u in others:
        too_small = units.index(u) < i
        wrong.append(
            ("fr2c-unit-too-small" if too_small else "fr2c-unit-too-large", _NAMES_OF[u], abs(units.index(u) - i))
        )
    wrong.sort(key=lambda t: t[2])
    wrong_opts = [(m, t) for m, t, _d in wrong]
    wrong_opts = wrong_opts[:2] + [("fr2c-unit-of-wrong-attribute", _NAMES_OF[wrong_attr])] + wrong_opts[2:]
    text = phrase(
        rng,
        (
            "Which unit is the most sensible for measuring {thing}?",
            "Mia needs to measure {thing}. Which unit should she use?",
            "Which metric unit would you choose to measure {thing}?",
        ),
        thing=thing,
    )
    fields = kit.mcq_fields(_NAMES_OF[best], wrong_opts)
    working = [f"{thing[0].upper() + thing[1:]} is best measured in {_NAMES_OF[best]}."]
    return text, fields, working


# ------------------------------------------------------------------------------------------ medium


def _medium(rng):
    for _ in attempts(500, "four measures in different units whose order differs from their numbers"):
        attr = rng.choice(list(_UNITS))
        units = list(_UNITS[attr])
        trio = {"length": ["mm", "cm", "m"], "mass": ["g", "kg"], "capacity": ["mL", "L"]}[attr]
        scale = {"length": 3000, "mass": 3_000_000, "capacity": 3000}[attr]
        measures = []
        for _i in range(4):
            u = rng.choice(trio)
            base = rng.randint(scale, int(scale * 1.7))
            num = Fraction(base, _UNITS[attr][u])
            if _UNITS[attr][u] == min(_UNITS[attr][t] for t in trio):
                num = Fraction(round(num))
            else:
                num = Fraction(round(num * 100), 100)
            if not _short_decimal(num, 2) or num <= 0:
                break
            measures.append((num, u, num * _UNITS[attr][u]))
        else:
            bases = [m[2] for m in measures]
            if len(set(bases)) < 4:
                continue
            ordered = sorted(bases)
            if any(b - a < ordered[0] * Fraction(3, 100) for a, b in zip(ordered, ordered[1:])):
                continue
            words = {
                "length": ("longest", "shortest"),
                "mass": ("heaviest", "lightest"),
                "capacity": ("largest", "smallest"),
            }
            ask_high = rng.random() < 0.5
            pick = max(measures, key=lambda m: m[2]) if ask_high else min(measures, key=lambda m: m[2])
            num_pick = max(measures, key=lambda m: m[0]) if ask_high else min(measures, key=lambda m: m[0])
            if pick is num_pick or len({m[1] for m in measures}) < min(3, len(trio)):
                continue
            break
    word = words[attr][0 if ask_high else 1]
    texts = [f"{_fmt(n)} {u}" for n, u, _b in measures]
    key = f"{_fmt(pick[0])} {pick[1]}"
    big_unit = max(measures, key=lambda m: _UNITS[attr][m[1]])
    wrong = []
    for n, u, b in measures:
        t = f"{_fmt(n)} {u}"
        if t == key:
            continue
        if (n, u) == (num_pick[0], num_pick[1]):
            wrong.append(("fr2c-compared-numbers-ignoring-units", t))
        elif (n, u) == (big_unit[0], big_unit[1]):
            wrong.append(("fr2c-chose-the-largest-unit", t))
        else:
            wrong.append(("fr2c-unit-conversion-slip", t))
    names = {"length": "lengths", "mass": "masses", "capacity": "capacities"}[attr]
    text = phrase(
        rng,
        (
            "Four {names} are written in different units: {options}. Which is the {word}?",
            "Here are four {names}: {options}. Which one is the {word}?",
        ),
        names=names,
        options=", ".join(texts[:-1]) + " and " + texts[-1],
        word=word,
    )
    smallest = min(_UNITS[attr], key=_UNITS[attr].get)
    inner = [f"{_fmt(b / _UNITS[attr][smallest])} {smallest}" for _n, _u, b in measures]
    working = [f"In {_NAMES_OF[smallest]} they are {', '.join(inner)}.", f"The {word} is {key}."]
    fields = kit.mcq_fields(key, wrong)
    # the options are the four measures themselves, so the wrong ones need no reordering
    return text, fields, working


# ------------------------------------------------------------------------------------------ hard


_PAIRS = (
    ("kg", "g", 1000),
    ("m", "cm", 100),
    ("L", "mL", 1000),
    ("km", "m", 1000),
    ("m", "mm", 1000),
    ("g", "mg", 1000),
)
_REMAINING = {
    "kg": "A bag holds {whole} {big} of flour. {name} uses {a} {small} from it and then {b} {big}. How much flour is left, in {small_name}?",
    "m": "A roll holds {whole} {big} of rope. {name} cuts off {a} {small} and then {b} {big}. How much rope is left, in {small_name}?",
    "L": "A tank holds {whole} {big} of water. {name} takes out {a} {small} and then {b} {big}. How much water is left, in {small_name}?",
    "km": "A trail is {whole} {big} long. {name} walks {a} {small} and then {b} {big}. How much of the trail is left, in {small_name}?",
    "g": "A tube holds {whole} {big} of powder. {name} uses {a} {small} and then {b} {big}. How much powder is left, in {small_name}?",
}
_CONTEXT = {
    "kg": ("parcel", "has three items weighing", "total mass"),
    "m": ("ribbon", "has three pieces measuring", "total length"),
    "L": ("jug", "is filled with three liquids measuring", "total capacity"),
    "km": ("hiker", "walks three stages measuring", "total distance"),
    "g": ("lab sample", "is made from three powders weighing", "total mass"),
}


def _big_decimal(rng, places):
    """A positive multiple of 10**-places that is not whole, as a Fraction."""
    return Fraction(rng.randint(11, 59) * 10 ** (places - 1) // 1 + rng.randint(1, 9), 10**places)


def _hard(rng):
    for _ in attempts(500, "three measures in two units"):
        big, small, f = rng.choice(_PAIRS)
        places = 2 if f == 100 else (3 if f == 1000 else 1)
        x1 = Fraction(rng.randint(12, 39), 10) if rng.random() < 0.5 else Fraction(rng.randint(2, 6))
        x3 = Fraction(rng.randint(2, 4))
        q2 = rng.randint(120, 880)
        if f == 100:
            q2 = rng.randint(15, 85)
        q1, q3 = x1 * f, x3 * f
        check(q1.denominator == 1 and q3.denominator == 1, "a measure is not whole in the small unit")
        total = q1 + q2 + q3
        if total % 1:
            continue
        remaining_kind = rng.random() < 0.4
        if remaining_kind:
            whole = Fraction(rng.randint(3, 6) * 1)  # in big units
            left = whole * f - (q2 + q3)
            if left <= 0:
                continue
            value = left
            wrong = [
                ("fr2c-forgot-to-change-one-measure", whole * f - q2 - x3),
                ("fr2c-forgot-to-change-one-measure", whole - q2 - q3),
                ("fr2c-chose-the-wrong-power-of-ten", whole * f * 10 - q2 - q3),
                ("fr2c-added-for-leftover", whole * f + q2 + q3),
                ("fr2c-chose-the-wrong-power-of-ten", whole * f - q2 * 10 - q3),
                ("fr2c-small-measure-treated-as-large", whole * f - q2 * f - q3),
            ]
            text = _REMAINING[big].format(
                name=rng.choice(_NAMES),
                whole=_fmt(whole),
                a=q2,
                small=small,
                b=_fmt(x3),
                big=big,
                small_name=_NAMES_OF[small],
            )
            working = [
                f"{_fmt(whole)} {big} = {_fmt(whole * f)} {small} and {_fmt(x3)} {big} = {_fmt(q3)} {small}.",
                f"{_fmt(whole * f)} - {q2} - {_fmt(q3)} = {_fmt(left)} {small}.",
            ]
        else:
            value = total
            wrong = [
                ("fr2c-added-numbers-ignoring-units", x1 + q2 + x3),
                ("fr2c-forgot-to-change-one-measure", q1 + q2 + x3),
                ("fr2c-forgot-to-change-one-measure", x1 + q2 + q3),
                ("fr2c-chose-the-wrong-power-of-ten", q1 * 10 + q2 + q3),
                ("fr2c-chose-the-wrong-power-of-ten", q1 + q2 + q3 * 10),
                ("fr2c-small-measure-treated-as-large", q1 + q2 * f + q3),
            ]
            ctx = _CONTEXT.get(big, ("parcel", "has three items weighing", "total"))
            noun, verb, what = ctx
            text = (
                f"A {noun} {verb} {_fmt(x1)} {big}, {q2} {small} and {_fmt(x3)} {big}. What is the {what}, in "
                f"{_NAMES_OF[small]}?"
            )
            working = [
                f"{_fmt(x1)} {big} = {_fmt(q1)} {small} and {_fmt(x3)} {big} = {_fmt(q3)} {small}.",
                f"{_fmt(q1)} + {q2} + {_fmt(q3)} = {_fmt(total)} {small}.",
            ]
        wrong = [(m, w) for m, w in wrong if w > 0 and Fraction(w).denominator == 1]
        key = f"{_fmt(Fraction(value))} {small}"
        try:
            fields = kit.mcq_fields(key, [(m, f"{_fmt(Fraction(w))} {small}") for m, w in wrong])
        except kit.Redraw:
            continue
        return text, fields, working
    raise AssertionError("unreachable")


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "fr2c_choose_unit_convert",
    level=5,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2c_choose_unit_convert(rng, tier="easy"):
    """Choose a metric unit, or compare and total measures in mixed units. See the module docstring."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    text, fields, working = _TIERS[tier](rng)
    return dict(text=text, working=working, **fields)


# ------------------------------------------------------------------------------------------ exam chain

_EXAM = (
    # attribute, big, small, factor, portion in small, thing made, portion name, a thing the small amount is measured in
    (
        "capacity",
        "L",
        "mL",
        1000,
        250,
        "fruit punch",
        "cup",
        "the soda in one glass of punch",
        "juice",
        "lemonade",
        "soda",
    ),
    (
        "mass",
        "kg",
        "g",
        1000,
        500,
        "a batch of trail mix",
        "bag",
        "the salt in one bag of trail mix",
        "nuts",
        "dried fruit",
        "seeds",
    ),
    (
        "length",
        "m",
        "cm",
        100,
        50,
        "streamers",
        "bow",
        "the width of one bow",
        "red ribbon",
        "blue ribbon",
        "gold ribbon",
    ),
)


@generator("fr2c_choose_unit_convert_exam", level=9, presets={"exam": {}})
def fr2c_choose_unit_convert_exam(rng):
    """Sensible unit, then the total of three amounts in the smaller unit, how many portions, and what is left."""
    return kit.build_with_redraw(rng, _build_fr2c_choose_unit_convert_exam, what="fr2c_choose_unit_convert_exam")


def _build_fr2c_choose_unit_convert_exam(rng):
    attr, big, small, f, portion, thing, pname, small_thing, n1, n2, n3 = rng.choice(_EXAM)
    name = rng.choice(_NAMES)
    for _ in attempts(500, "three amounts whose total makes a whole number of portions"):
        x1 = Fraction(rng.randint(15, 49), 10)  # in big units, 1.5 .. 4.9
        q2 = portion * rng.randint(1, 3) + (0 if portion == 250 else 0)
        x3 = Fraction(rng.randint(2, 9), 10)
        q1, q3 = x1 * f, x3 * f
        total = q1 + q2 + q3
        if q1.denominator != 1 or q3.denominator != 1:
            continue
        if total % portion != 0:
            continue
        count = total // portion
        used = rng.randint(2, int(count) - 2) if count >= 5 else None
        if used is None:
            continue
        break
    left = total - used * portion
    big_name, small_name = _NAMES_OF[big], _NAMES_OF[small]
    stem = (
        f"{name} is preparing {thing}. The recipe uses {_fmt(x1)} {big} of {n1}, {q2} {small} of {n2} and {_fmt(x3)} {big} "
        f"of {n3}."
    )
    best = {"capacity": "mL", "mass": "g", "length": "cm"}[attr]
    wrong_units = [
        ("fr2c-unit-too-large", _NAMES_OF[big]),
        ("fr2c-unit-too-large", _NAMES_OF[{"capacity": "kL", "mass": "t", "length": "km"}[attr]]),
        ("fr2c-unit-of-wrong-attribute", {"capacity": "metres", "mass": "litres", "length": "grams"}[attr]),
        ("fr2c-unit-too-small", _NAMES_OF[{"capacity": "mL", "mass": "mg", "length": "mm"}[attr]]),
    ]
    parts = [
        kit.mcq_part(
            f"Which unit is the most sensible for measuring {small_thing}?",
            _NAMES_OF[best],
            1,
            [f"A small amount like this is best measured in {_NAMES_OF[best]}."],
            wrong_units,
        ),
        kit.num_part(
            f"Find the total amount, in {small_name}.",
            total,
            2,
            [
                f"{_fmt(x1)} {big} = {_fmt(q1)} {small} and {_fmt(x3)} {big} = {_fmt(q3)} {small}.",
                f"{_fmt(q1)} + {q2} + {_fmt(q3)} = {_fmt(total)}.",
            ],
            [
                ("fr2c-added-numbers-ignoring-units", x1 + q2 + x3),
                ("fr2c-forgot-to-change-one-measure", q1 + q2 + x3),
                ("fr2c-forgot-to-change-one-measure", x1 + q2 + q3),
                ("fr2c-chose-the-wrong-power-of-ten", q1 * 10 + q2 + q3),
            ],
            sensible_range=(0, None),
        ),
        kit.num_part(
            f"Each {pname} uses {portion} {small}. Using your total, how many {pname}s can be made with none left over?",
            count,
            2,
            [f"{_fmt(total)} / {portion} = {count}."],
            [
                ("fr2c-divided-the-wrong-way", Fraction(portion, 1) * 0 + total * portion),
                ("fr2c-chose-the-wrong-power-of-ten", count * 10),
                ("fr2c-count-off-by-one", count + 1),
                ("fr2c-count-off-by-one", count - 1),
            ],
            follow_through=f"part_b/{portion}",
            sensible_range=(0, None),
        ),
        kit.num_part(
            f"{name} makes {used} of the {pname}s. How much is left, in {small_name}?",
            left,
            2,
            [
                f"{used} x {portion} = {used * portion} {small} is used.",
                f"{_fmt(total)} - {used * portion} = {_fmt(left)} {small}.",
            ],
            [
                ("fr2c-reports-the-amount-poured", used * portion),
                ("fr2c-added-for-leftover", total + used * portion),
                ("fr2c-took-away-the-count-not-the-amount", total - used),
                ("fr2c-chose-the-wrong-power-of-ten", left * 10),
            ],
            follow_through=f"part_b-{used * portion}",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)
