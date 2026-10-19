"""Generates `f10-financial-multistep-whole-numbers` questions and its exam chain. `answer: number`.

Scope: whole-dollar money problems of two to four steps for Years 3 and 4. The learner has to decide which
operations the story needs (a total, a change, a share, a number of weeks) and carry a result from one step to the
next. Cents, percentages, rates and interest are out of scope.

Ladder (reasoning escalation, not size):
  easy:   a purchase of several equal items, paid with one note: the change. Two operations; the stem names neither.
  medium: a purchase that needs three: two kinds of item with a voucher taken off the total, a bill shared
          equally, or a total paid with a stated number of notes.
  hard:   a saving plan. How many whole weeks until a goal is reached (the last part-week still counts as a week),
          how much each equal payment is once a deposit is paid, or how much is still to save part-way through.
  exam:   a four-part chain on one story: the total of an order, the change from a note (follows through), the
          whole weeks to afford a game from that change and weekly savings (follows through), and the money left
          after buying it (follows through from both).
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, generator, phrase
from ._fr2e_kit import Redraw, draw, exact_number_part, single

_NAMES = ("Priya", "Marcus", "Aiko", "Tom", "Leila", "Sam", "Jess", "Kofi", "Mia", "Dev", "Ella", "Hamish")
# (place, item, plural item) for a purchase of equal items
_SHOPS = (
    ("the school tuckshop", "sausage roll", "sausage rolls", 3, 6),
    ("the school book fair", "bookmark", "bookmarks", 2, 5),
    ("the sports shop", "drink bottle", "drink bottles", 4, 9),
    ("the school fete", "raffle ticket", "raffle tickets", 2, 4),
    ("the craft market", "badge", "badges", 3, 8),
    ("the bakery", "pie", "pies", 4, 8),
)
# (item, plural, price lo, price hi) for a two-kind order
_ORDER_ITEMS = (
    ("pie", "pies", 4, 7),
    ("juice", "juices", 2, 4),
    ("sandwich", "sandwiches", 5, 8),
    ("fruit cup", "fruit cups", 2, 5),
    ("muffin", "muffins", 3, 5),
    ("pencil case", "pencil cases", 6, 9),
    ("notebook", "notebooks", 4, 7),
)
_GOODS = ("game", "kite", "skateboard helmet", "model kit", "football", "scooter bell set")

W_FORGOT = "fr2e-money-count-not-multiplied"
W_PRICES = "fr2e-money-prices-added-not-multiplied"
W_SWAP = "fr2e-money-prices-swapped"
W_ROUND_DOWN = "fr2e-money-part-week-dropped"
W_SKIP_START = "fr2e-money-saved-amount-ignored"


def _unit(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _easy(rng):
    place, one, many, lo, hi = rng.choice(_SHOPS)
    name = rng.choice(_NAMES)
    price, n = rng.randint(lo, hi), rng.randint(3, 7)
    note = rng.choice((20, 50))
    total = price * n
    if not (note - total >= 2 and total >= 10):
        raise Redraw("the change should be a few dollars")
    change = note - total
    stem = phrase(
        rng,
        (
            "{name} buys {k} from {place}. Each costs ${price}. {name} pays with a ${note} note.",
            "At {place}, {name} pays with a ${note} note for {k} at ${price} each.",
        ),
        name=name,
        k=_unit(n, one, many),
        place=place,
        price=price,
        note=note,
    )
    part = exact_number_part(
        "How much change does " + name + " get, in dollars?",
        change,
        1,
        [f"{n} × ${price} = ${total}.", f"${note} − ${total} = ${change}."],
        [
            (note - price, W_FORGOT),
            (total, "reported-total-not-change"),
            (note + total, "added-instead-of-subtracted"),
            (total - note, "subtracted-in-wrong-order"),
        ],
        suffix="",
    )
    check(change == note - n * price, "the change must follow from the story")
    return single(stem, part)


def _two_kind_order(rng):
    (i1, m1, lo1, hi1), (i2, m2, lo2, hi2) = rng.sample(_ORDER_ITEMS, 2)
    p, q = rng.randint(lo1, hi1), rng.randint(lo2, hi2)
    if p == q:
        raise Redraw("equal prices make the swap distractor vanish")
    a, b = rng.randint(2, 5), rng.randint(2, 5)
    return (i1, m1, p, a), (i2, m2, q, b)


def _medium(rng):
    name = rng.choice(_NAMES)
    kind = rng.choice(("voucher", "share", "notes"))
    (i1, m1, p, a), (i2, m2, q, b) = _two_kind_order(rng)
    total = a * p + b * q
    order = f"{_unit(a, i1, m1)} at ${p} each and {_unit(b, i2, m2)} at ${q} each"
    swapped = a * q + b * p
    if kind == "voucher":
        off = rng.choice((3, 4, 5, 6))
        pay = total - off
        stem = f"{name} orders {order}. A ${off} voucher is taken off the bill."
        part = exact_number_part(
            "How much does " + name + " pay, in dollars?",
            pay,
            1,
            [f"{a} × ${p} + {b} × ${q} = ${total}.", f"${total} − ${off} = ${pay}."],
            [
                (total, "fr2e-money-voucher-ignored"),
                (total + off, "added-instead-of-subtracted"),
                (swapped - off, W_SWAP),
                (a * p + b - off, W_FORGOT),
            ],
            suffix="",
        )
    elif kind == "share":
        friend = rng.choice([n for n in _NAMES if n != name])
        if total % 2:
            raise Redraw("the bill must share into whole dollars")
        each = total // 2
        stem = f"{name} and {friend} order {order} and share the bill equally."
        part = exact_number_part(
            "How much does each of them pay, in dollars?",
            each,
            1,
            [f"{a} × ${p} + {b} × ${q} = ${total}.", f"${total} ÷ 2 = ${each}."],
            [
                (total, "fr2e-money-bill-not-shared"),
                (swapped // 2 if swapped % 2 == 0 else swapped, W_SWAP),
                ((a * p + b) // 2 if (a * p + b) % 2 == 0 else a * p + b, W_FORGOT),
                (total * 2, "multiplied-instead-of-divided"),
            ],
            suffix="",
        )
    else:
        note = rng.choice((10, 20, 50))
        k = math.ceil(total / note)
        change = k * note - total
        if k < 2 or change == 0 or total < note:
            raise Redraw("the notes must not cover the bill exactly")
        stem = f"{name} orders {order}. {name} pays using only ${note} notes, as few as possible."
        part = exact_number_part(
            "How much change does " + name + " get, in dollars?",
            change,
            1,
            [
                f"{a} × ${p} + {b} × ${q} = ${total}.",
                f"{k} notes of ${note} make ${k * note}, the fewest that cover ${total}.",
                f"${k * note} − ${total} = ${change}.",
            ],
            [
                (k * note - swapped if k * note > swapped else None, W_SWAP),
                ((k - 1) * note - total if (k - 1) * note > total else None, W_ROUND_DOWN),
                (total % note, "fr2e-money-remainder-as-change"),
                (k * note, "reported-total-not-change"),
            ],
            suffix="",
        )
    return single(stem, part)


def _hard(rng):
    name = rng.choice(_NAMES)
    good = rng.choice(_GOODS)
    kind = rng.choice(("weeks", "instalment", "left"))
    goal = rng.choice(range(40, 131, 5))
    have = rng.choice(range(10, 41, 5))
    week = rng.choice((4, 5, 6, 7, 8, 9))
    if kind == "weeks":
        need = goal - have
        weeks = math.ceil(need / week)
        if need % week == 0 or weeks < 3:
            raise Redraw("the last part-week must matter")
        stem = (
            f"{name} wants a {good} that costs ${goal} and has saved ${have}. {name} saves ${week} each week from pocket money, "
            "and buys the first week there is enough."
        )
        part = exact_number_part(
            "In how many weeks from now can " + name + " buy it?",
            weeks,
            1,
            [
                f"${goal} − ${have} = ${need} still to save.",
                f"${need} ÷ ${week} = {need / week:.2f}, so {weeks} weeks.",
            ],
            [
                (need // week, W_ROUND_DOWN),
                (math.ceil(goal / week), W_SKIP_START),
                (need // week + 2, "fr2e-money-weeks-rounded-up-twice"),
                (need, "fr2e-money-dollars-as-weeks"),
            ],
            suffix="",
        )
    elif kind == "instalment":
        k = rng.choice((4, 5, 6, 8))
        deposit = rng.choice((10, 15, 20, 25))
        if (goal - deposit) % k:
            raise Redraw("the payments must be whole dollars")
        each = (goal - deposit) // k
        if each < 4:
            raise Redraw("too small a payment")
        stem = (
            f"{name}'s family buys a {good} for ${goal} on lay-by. They pay a ${deposit} deposit and then {k} equal "
            "weekly payments."
        )
        part = exact_number_part(
            "How much is each weekly payment, in dollars?",
            each,
            1,
            [f"${goal} − ${deposit} = ${goal - deposit} left to pay.", f"${goal - deposit} ÷ {k} = ${each}."],
            [
                (goal // k if goal % k == 0 else None, "fr2e-money-deposit-ignored"),
                (deposit + each, "added-instead-of-subtracted"),
                ((goal + deposit) // k if (goal + deposit) % k == 0 else None, "added-instead-of-subtracted"),
                (goal - deposit, "fr2e-money-bill-not-shared"),
            ],
            suffix="",
        )
    else:
        k = rng.choice((2, 3, 4, 5))
        spent = rng.choice((5, 8, 10, 12))
        left = goal - have - k * week + spent
        if left <= 0 or k * week + have - spent < 5:
            raise Redraw("the story must still need saving")
        stem = (
            f"{name} wants a {good} that costs ${goal} and has ${have}. {name} saves ${week} each week for {k} weeks, "
            f"but spends ${spent} on a birthday present during that time."
        )
        part = exact_number_part(
            "How much more does " + name + " need to save, in dollars?",
            left,
            1,
            [
                f"{k} × ${week} = ${k * week} saved.",
                f"${have} + ${k * week} − ${spent} = ${have + k * week - spent} in hand.",
                f"${goal} − ${have + k * week - spent} = ${left}.",
            ],
            [
                (goal - have - k * week, "fr2e-money-spending-ignored"),
                (goal - have - k * week - spent, "added-instead-of-subtracted"),
                (goal - have - week + spent, W_FORGOT),
                (goal - (k * week + spent), W_SKIP_START),
            ],
            suffix="",
        )
    return single(stem, part)


def _exam(rng):
    name = rng.choice(_NAMES)
    (i1, m1, p, a), (i2, m2, q, b) = _two_kind_order(rng)
    total = a * p + b * q
    note = next(n for n in (20, 50, 100) if n >= total + 3)
    change = note - total
    if change < 3:
        raise Redraw("the note must cover the order with a little to spare")
    good = rng.choice(_GOODS)
    goal = rng.choice(range(change + 20, change + 61, 5))
    week = rng.choice((4, 5, 6, 7, 8))
    weeks = math.ceil((goal - change) / week)
    if (goal - change) % week == 0 or weeks < 3:
        raise Redraw("the last part-week must matter")
    left = change + week * weeks - goal
    check(0 < left < week, "the money left must be less than one week's saving")
    stem = (
        f"{name} orders {_unit(a, i1, m1)} at ${p} each and {_unit(b, i2, m2)} at ${q} each for a school event, "
        f"and pays with a ${note} note. {name} puts the change towards a {good} that costs ${goal}, and saves "
        f"${week} each week from pocket money until there is enough to buy it."
    )
    part_a = exact_number_part(
        "Find the total cost of the order, in dollars.",
        total,
        2,
        [f"{a} × ${p} = ${a * p}; {b} × ${q} = ${b * q}.", f"${a * p} + ${b * q} = ${total}."],
        [
            (a * p + b, W_FORGOT),
            (a * q + b * p, W_SWAP),
            (p + q, W_PRICES),
            (a * p, "fr2e-money-second-item-left-out"),
        ],
        suffix="",
        sensible_range=(0, None),
    )
    part_b = exact_number_part(
        "Find the change from the note, in dollars.",
        change,
        1,
        [f"${note} − ${total} = ${change}."],
        [
            (total, "reported-total-not-change"),
            (note + total, "added-instead-of-subtracted"),
            (note - (a * q + b * p), W_SWAP),
            (total - note, "subtracted-in-wrong-order"),
        ],
        suffix="",
        follow_through=f"{note} - part_a",
        sensible_range=(0, None),
    )
    part_c = exact_number_part(
        f"How many whole weeks must {name} save before buying the {good}?",
        weeks,
        2,
        [
            f"${goal} − ${change} = ${goal - change} still needed.",
            f"{goal - change} ÷ {week} is not whole, so {weeks} weeks.",
        ],
        [
            ((goal - change) // week, W_ROUND_DOWN),
            (math.ceil(goal / week), W_SKIP_START),
            (goal - change, "fr2e-money-dollars-as-weeks"),
            ((goal - change) // week + 2, "fr2e-money-weeks-rounded-up-twice"),
        ],
        suffix="",
        follow_through=f"ceiling(({goal} - part_b)/{week})",
        sensible_range=(0, None),
    )
    part_d = exact_number_part(
        f"How much money does {name} have left after buying the {good}, in dollars?",
        left,
        1,
        [
            f"${change} + {weeks} × ${week} = ${change + weeks * week}.",
            f"${change + weeks * week} − ${goal} = ${left}.",
        ],
        [
            (change + weeks * week, "fr2e-money-price-not-taken-off"),
            (left + week, "fr2e-money-weeks-off-by-one"),
            (week - left, "fr2e-money-week-minus-left"),
            (goal - change, "fr2e-money-price-not-taken-off"),
        ],
        suffix="",
        follow_through=f"part_b + {week} * part_c - {goal}",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator(
    "fr2e_money_multistep",
    level=4,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2e_money_multistep(rng, tier: str = "easy"):
    """A whole-dollar money story that takes two to four steps."""
    build = {"easy": _easy, "medium": _medium, "hard": _hard}.get(tier)
    check(build is not None, f"unknown tier {tier!r}")
    return draw(build, rng)


@generator("exd_x_fr2e_money_multistep", level=4, presets={"exam": dict(tier="exam")})
def exd_x_fr2e_money_multistep(rng, tier: str = "exam"):
    """A four-part money chain: total, change, whole weeks to save, money left over."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return draw(_exam, rng)
