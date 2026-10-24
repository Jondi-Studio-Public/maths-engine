"""Generates `fr2c_informal_capacity_mass` questions: capacity and mass measured in informal units.

One skill: ``f10-capacity-mass-informal-units`` (``answer: number``). The measures are counts of a repeated informal unit
(cups of water, scoops of rice, blocks on the other pan of a balance), told in a short sentence.

- easy: one comparison or one total (how many more cups does the bucket hold? how many blocks balance both?).
- medium: two steps with the same unit: how many more cups fill a part-filled container, how many blocks both of two
  items balance when one is described by "3 more than" the other, how much more one item balances than two others
  together.
- hard: two units of different size. A container is measured with a big scoop and another with a small scoop, with the
  size of the big scoop given in small scoops; the learner has to work in the smaller unit that both can be counted in.

Every answer is built from the chosen counts and then recomputed from the unit sizes.
"""

from __future__ import annotations

from mathgen.core import attempts, check, generator, phrase

from . import _fr2c_kit as kit

_NAMES = ("Mia", "Tom", "Aiko", "Dev", "Leila", "Sam", "Jess", "Kofi", "Priya", "Marcus", "Zoe", "Ben")
_CONTAINERS = ("bucket", "jug", "bowl", "bottle", "tub", "watering can", "pot", "fish tank", "basin", "saucepan")
_CAP_UNITS = ("cup", "mug", "scoop", "spoon")
_ITEMS = ("pear", "book", "shoe", "melon", "teapot", "toy truck", "tin of beans", "pencil case", "apple", "lunchbox")
_MASS_UNITS = ("block", "marble", "counter", "cube")


def _plural(unit: str, n: int) -> str:
    return unit if n == 1 else unit + "s"


def _pair(rng, pool):
    a, b = rng.sample(pool, 2)
    return a, b


# ------------------------------------------------------------------------------------------ easy


def _easy(rng):
    cap = rng.random() < 0.5
    unit = rng.choice(_CAP_UNITS if cap else _MASS_UNITS)
    a_thing, b_thing = _pair(rng, _CONTAINERS if cap else _ITEMS)
    small = rng.randint(3, 9)
    big = small + rng.randint(2, 8)
    diff = rng.random() < 0.5
    if cap:
        facts = f"The {a_thing} holds {big} {_plural(unit, big)} of water and the {b_thing} holds {small} {_plural(unit, small)}."
        verb = "hold"
    else:
        facts = f"The {a_thing} balances {big} {_plural(unit, big)} and the {b_thing} balances {small} {_plural(unit, small)}."
        verb = "balance"
    if diff:
        value = big - small
        ask = phrase(
            rng,
            (
                "How many more {u} does the {a} {v} than the {b}?",
                "How many more {u} does the {a} {v} than the {b} {v}?",
            ),
            u=_plural(unit, 2),
            a=a_thing,
            v=verb,
            b=b_thing,
        )
        wrong = [
            ("added-instead-of-subtracted", big + small),
            ("fr2c-reports-the-larger-count", big),
            ("fr2c-reports-the-smaller-count", small),
            ("fr2c-count-off-by-one", value + 1),
            ("fr2c-count-off-by-one", value - 1),
        ]
        working = [f"{big} - {small} = {value}."]
    else:
        value = big + small
        ask = f"How many {_plural(unit, 2)} is that altogether for the {a_thing} and the {b_thing}?"
        wrong = [
            ("fr2c-subtracted-for-total", big - small),
            ("fr2c-reports-the-larger-count", big),
            ("fr2c-count-off-by-one", value + 1),
            ("fr2c-count-off-by-one", value - 1),
        ]
        working = [f"{big} + {small} = {value}."]
    text = f"{facts} {ask}"
    return text, value, wrong, working


# ------------------------------------------------------------------------------------------ medium


def _medium(rng):
    kind = rng.choice(["fill", "more_than", "two_items"])
    name = rng.choice(_NAMES)
    if kind == "fill":
        unit = rng.choice(_CAP_UNITS)
        thing = rng.choice(_CONTAINERS)
        first, second = rng.randint(3, 8), rng.randint(3, 8)
        full = rng.randint(max(14, first + second + 1), 30)
        value = full - first - second
        check(value > 0, "the container is already full")
        text = (
            f"The {thing} holds {full} {_plural(unit, full)} of water. {name} pours in {first} {_plural(unit, first)} and "
            f"then {second} more. How many more {_plural(unit, 2)} fill the {thing}?"
        )
        wrong = [
            ("fr2c-took-away-one-pour-only", full - first),
            ("fr2c-took-away-one-pour-only", full - second),
            ("fr2c-reports-the-amount-poured", first + second),
            ("fr2c-count-off-by-one", value + 1),
            ("fr2c-count-off-by-one", value - 1),
        ]
        working = [f"{name} has poured {first} + {second} = {first + second}.", f"{full} - {first + second} = {value}."]
    elif kind == "more_than":
        unit = rng.choice(_MASS_UNITS)
        a_thing, b_thing = _pair(rng, _ITEMS)
        base = rng.randint(5, 15)
        extra = rng.randint(2, 6)
        value = base + base + extra
        text = (
            f"The {a_thing} balances {base} {_plural(unit, base)}. The {b_thing} balances {extra} {_plural(unit, extra)} more "
            f"than the {a_thing}. How many {_plural(unit, 2)} balance the {a_thing} and the {b_thing} together?"
        )
        wrong = [
            ("fr2c-second-count-taken-as-given", base + extra),
            ("fr2c-reports-the-larger-count", base + extra),
            ("fr2c-first-count-doubled", 2 * base),
            ("fr2c-count-off-by-one", value + 1),
            ("fr2c-count-off-by-one", value - 1),
        ]
        working = [f"The {b_thing} balances {base} + {extra} = {base + extra}.", f"{base} + {base + extra} = {value}."]
    else:
        unit = rng.choice(_MASS_UNITS)
        big_thing, p, q = rng.sample(_ITEMS, 3)
        pv, qv = rng.randint(3, 9), rng.randint(3, 9)
        bigv = pv + qv + rng.randint(2, 9)
        value = bigv - pv - qv
        text = (
            f"The {big_thing} balances {bigv} {_plural(unit, bigv)}. The {p} balances {pv} {_plural(unit, pv)} and the {q} "
            f"balances {qv} {_plural(unit, qv)}. How many more {_plural(unit, 2)} does the {big_thing} balance than the {p} "
            f"and the {q} together?"
        )
        wrong = [
            ("fr2c-took-away-one-item-only", bigv - pv),
            ("fr2c-took-away-one-item-only", bigv - qv),
            ("fr2c-reports-the-amount-poured", pv + qv),
            ("fr2c-count-off-by-one", value + 1),
            ("fr2c-count-off-by-one", value - 1),
        ]
        working = [f"Together the {p} and the {q} balance {pv} + {qv} = {pv + qv}.", f"{bigv} - {pv + qv} = {value}."]
    return text, value, wrong, working


# ------------------------------------------------------------------------------------------ hard


def _hard(rng):
    name = rng.choice(_NAMES)
    cap = rng.random() < 0.6
    k = rng.randint(3, 5)  # small units in one big unit
    if cap:
        big_u, small_u = rng.choice([("mug", "spoon"), ("cup", "spoon"), ("scoop", "teaspoon"), ("jug", "cup")])
        a_thing, b_thing = _pair(rng, _CONTAINERS)
        s_verb, p_verb, of = "holds", "hold", " of water"
        fill = f"To fill the {a_thing}, {name} uses"
    else:
        big_u, small_u = rng.choice([("brick", "block"), ("big weight", "small weight"), ("tin", "marble")])
        a_thing, b_thing = _pair(rng, _ITEMS)
        s_verb, p_verb, of = "balances", "balance", ""
        fill = f"To balance the {a_thing}, {name} uses"
    b = rng.randint(3, 7)
    if rng.random() < 0.5:
        # one object counted in big units, the other in small units
        m = b * k + rng.randint(2, 2 * k + 3)
        value = m - b * k
        text = (
            f"One {big_u} {s_verb} {k} {_plural(small_u, k)}. The {a_thing} {s_verb} {b} {_plural(big_u, b)}{of}. "
            f"The {b_thing} {s_verb} {m} {_plural(small_u, m)}{of}. {name} compares them in {_plural(small_u, 2)}. "
            f"How many more {_plural(small_u, 2)} does the {b_thing} {p_verb} than the {a_thing}?"
        )
        wrong = [
            ("fr2c-mixed-unit-sizes", m - b),
            ("fr2c-mixed-unit-sizes", m + b),
            ("fr2c-big-unit-counted-as-small", value + k),
            ("fr2c-count-off-by-one", value + 1),
            ("fr2c-reports-the-larger-count", b * k),
        ]
        working = [f"The {a_thing} is {b} x {k} = {b * k} {_plural(small_u, 2)}.", f"{m} - {b * k} = {value}."]
    else:
        r = rng.randint(2, k + 2)
        value = b * k + r
        text = (
            f"One {big_u} {s_verb} {k} {_plural(small_u, k)}. {fill} {b} full {_plural(big_u, b)} and then {r} "
            f"{_plural(small_u, r)} to finish it off. How many {_plural(small_u, 2)} is that altogether?"
        )
        wrong = [
            ("fr2c-mixed-unit-sizes", b + r),
            ("fr2c-big-unit-counted-as-small", b * k),
            ("fr2c-big-unit-counted-as-small", b * k - r),
            ("fr2c-count-off-by-one", value + 1),
            ("fr2c-count-off-by-one", value - 1),
        ]
        working = [f"{b} x {k} = {b * k} {_plural(small_u, 2)}.", f"{b * k} + {r} = {value}."]
    check(value > 0, "a non-positive answer")
    return text, value, wrong, working


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "fr2c_informal_capacity_mass",
    level=2,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2c_informal_capacity_mass(rng, tier="easy"):
    """A capacity or mass told in informal units, with a number answer. See the module docstring for the tiers."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(60, "a question with three different wrong answers"):
        text, value, wrong, working = _TIERS[tier](rng)
        wrong = [(m, w) for m, w in wrong if w > 0]
        try:
            fields = kit.number_fields(value, wrong)
        except kit.Redraw:
            continue
        break
    return dict(text=text, working=working, **fields)
