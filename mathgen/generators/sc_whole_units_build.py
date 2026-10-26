"""Generates `sc_whole_units` questions (easy, medium, hard) and the exam chain `sc_whole_units_chain`.

Skill `whole-units-needed-round-up-down` (answer `number`, a whole number): how many whole boxes, packs, trips or
tickets are NEEDED (the quotient rounded up) versus how many full groups or items can be MADE, CARRIED or FITTED (the
quotient rounded down). The stem never says which: the situation decides, and the same topic (paint, buses, bottles,
teams) turns up in both directions so the direction cannot be guessed from the topic.

* easy: one division, T / s, with the context deciding the direction. The remainder is chosen so that rounding to the
  nearest whole number gives a different answer from the key, so "round it normally" never scores.
* medium: two steps. A total is built first (an area from two sides and a number of coats, several groups added, a rate
  over a time, a wage plus a bonus), then divided by the pack or group size. The ask is the count, the cost of the packs
  bought, or what is left over.
* hard: four trap types. The cheapest whole-pack combination from two pack sizes (a small search); how many can be made
  when two supplies run out at different rates (round down, then take the smaller); how many items fit along a shelf
  when each needs a gap (one more than the quotient of the length by the pitch); and boards cut into pieces, where the
  number of pieces per board rounds down and then the number of boards rounds up.
* exam: one scenario in four parts that follow through: the total needed, the whole packs bought (or full groups made),
  the cost (or the value), and the leftover.

Every key is computed with integer arithmetic and re-derived by an independent route (SymPy's `ceiling` and `floor` of a
`Rational`; a dynamic programme and an exhaustive search for the cheapest combination; a brute-force count for the fits).
All answers are whole numbers, so no rounding of a working value can move one.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ._company_kit import keep_company, numeric_pool
from ..core import attempts, check, generator, round_half_up
from .r4e_common import Redraw, draw, exact_number_part, single

# -- helpers ----------------------------------------------------------------------------------------------------------

_WHOLE = "Give your answer as a whole number."
_DOLLARS = "Give your answer as a whole number of dollars."


def _ceil(t: int, s: int) -> int:
    return -(-t // s)


def _nearest(t: int, s: int) -> int:
    return int(round_half_up(Fraction(t, s)))


def _check_quotient(t: int, s: int, up: bool, key: int) -> None:
    """Independent route: SymPy's ceiling or floor of the exact quotient."""
    q = sympy.Rational(t, s)
    check(int(sympy.ceiling(q) if up else sympy.floor(q)) == key, "whole-unit count disagrees with SymPy")


def _split_ok(t: int, s: int, up: bool) -> bool:
    """A remainder that makes the nearest whole number differ from the key (never a tie, never zero)."""
    r = t % s
    if r == 0 or t < 2 * s:
        return False
    return 2 * r < s if up else 2 * r > s


def _count_wrong(t: int, s: int, up: bool):
    """Tagged wrong counts for a single division."""
    q, r = divmod(t, s)
    key = q + 1 if up else q
    return [
        (_nearest(t, s), "wu-rounds-to-nearest"),
        (q if up else q + 1, "wu-rounds-down-when-covering" if up else "wu-rounds-up-when-only-full"),
        (key + 1 if up else key - 1, "wu-adds-extra-unit" if up else "wu-drops-a-full-group"),
        (r, "wu-reports-remainder"),
        (key + 2 if up else key - 2, "wu-adds-extra-unit" if up else "wu-drops-a-full-group"),
    ]


def _cost_wrong(t: int, s: int, up: bool, price: int):
    """Tagged wrong costs, every one a whole number of packs, on both sides of the key at uneven spacing."""
    n = _ceil(t, s) if up else t // s
    over = "wu-adds-extra-unit" if up else "wu-rounds-up-when-only-full"
    under = "wu-rounds-down-when-covering" if up else "wu-drops-a-full-group"
    return [
        (_nearest(t, s) * price, "wu-rounds-to-nearest"),
        ((n - 1) * price, under),
        ((n + 1) * price, over),
        ((n + 2) * price, over),
        ((n - 2) * price, under),
        ((n + 3) * price, over),
    ]


def _part(question, value, marks, working, wrong, suffix, rng=None, **kw):
    """A whole-number `number` part whose prompt is `question` followed by the precision sentence.

    With `rng`, the wrong values are shuffled before the first three are taken, so the key sits at a different rank
    among the options from one question to the next (never reliably the central one)."""
    seen = {value}
    wrong = [w for w in wrong if w[0] >= 0 and not (w[0] in seen or seen.add(w[0]))]  # drop the key and repeats
    pool = numeric_pool(int(value), ok=lambda v: v >= 0 and v.denominator == 1)
    pool = [c for c in pool if c[0] not in seen]
    if rng is not None:
        wrong = rng.sample(wrong, len(wrong))
        # a uniformly random rank for the key: t of the three shown wrong values lie below it
        taken = {w[0] for w in wrong}
        below = [c for c in wrong if c[0] < value] + [c for c in pool if c[0] < value and c[0] not in taken]
        above = [c for c in wrong if c[0] > value] + [c for c in pool if c[0] > value and c[0] not in taken]
        same_width = lambda c: len(str(c[0])) != len(str(value))  # noqa: E731  (False sorts first)
        below, above = sorted(below, key=same_width), sorted(above, key=same_width)
        feasible = [
            t
            for t in range(4)
            if len(below) >= t
            and len(above) >= 3 - t
            and not all(same_width(c) for c in below[:t] + above[: 3 - t])  # the key keeps a digit-count twin
        ] or [t for t in range(4) if len(below) >= t and len(above) >= 3 - t]
        if feasible:
            t = rng.choice(feasible)
            pick = below[:t] + above[: 3 - t]
            wrong = pick + [c for c in wrong if c not in pick]
    wrong += [c for c in pool[:3] if c[0] not in {w[0] for w in wrong}]
    wrong = keep_company(int(value), wrong, pool=pool, shift_cost=1.5)
    part = exact_number_part("x", value, marks, working, wrong, suffix=suffix, **kw)
    part["prompt"] = f"{question} {suffix}".strip()
    return part


# -- easy -------------------------------------------------------------------------------------------------------------

# (up?, template, sizes, quotient range). {N} is the total and {s} the size of one pack, group or trip.
_EASY = (
    (
        True,
        "A school has {N} students going on an excursion. Each minibus seats {s} students. How many minibuses are needed so that everyone has a seat?",
        (12, 14, 16, 20, 22),
        (3, 12),
    ),
    (
        True,
        "{N} guests are coming to a barbecue and sausages are sold in packs of {s}. Each guest will eat one sausage. How many packs must be bought to have enough?",
        (6, 8, 10, 12),
        (4, 15),
    ),
    (
        True,
        "A fence is {N} metres long. Each panel is {s} metres wide. How many panels must be bought to cover the whole length?",
        (2, 3, 4, 5),
        (6, 25),
    ),
    (
        True,
        "A removalist must move {N} boxes and the van holds {s} boxes at a time. How many trips are needed to move them all?",
        (14, 16, 18, 24, 30),
        (3, 12),
    ),
    (
        True,
        "A hall has {N} students who will sit at tables of {s}. How many tables are needed?",
        (6, 8, 10),
        (6, 20),
    ),
    (
        True,
        "A tiler must cover {N} square metres of floor. Each pack of tiles covers {s} square metres. How many packs does she need to buy?",
        (2, 3, 4, 5),
        (5, 20),
    ),
    (
        False,
        "A baker has {N} cupcakes and puts {s} in each box for sale. How many full boxes can she make?",
        (4, 6, 8, 12),
        (4, 15),
    ),
    (
        False,
        "A rope is {N} centimetres long. Pieces {s} centimetres long are cut off to make skipping ropes. How many whole pieces can be cut?",
        (20, 25, 30, 40, 45),
        (3, 12),
    ),
    (
        False,
        "{N} students are split into teams of {s}. How many full teams can be made?",
        (6, 7, 8, 9),
        (3, 12),
    ),
    (
        False,
        "A shelf is {N} centimetres long. Each book is {s} centimetres thick and the books stand side by side. How many books fit on the shelf?",
        (3, 4, 5, 6),
        (8, 25),
    ),
    (
        False,
        "Maya has ${N} to spend on concert tickets that cost ${s} each. How many tickets can she buy?",
        (35, 45, 60, 75, 90),
        (2, 9),
    ),
    (
        False,
        "A tank holds {N} litres of water. A water cart is filled from it only in complete loads of {s} litres. How many complete loads can be taken?",
        (150, 200, 250, 400),
        (3, 12),
    ),
)


def _easy(rng):
    up, template, sizes, (qlo, qhi) = rng.choice(_EASY)
    for _ in attempts(500, "a total whose remainder separates the key from the nearest whole number"):
        s = rng.choice(sizes)
        t = s * rng.randint(qlo, qhi) + rng.randint(1, s - 1)
        if _split_ok(t, s, up):
            break
    key = _ceil(t, s) if up else t // s
    _check_quotient(t, s, up, key)
    check(key != _nearest(t, s), "the key should differ from the nearest whole number")
    stem = template.format(N=t, s=s)
    word = "up to the next whole number" if up else "down to the last whole number"
    working = [
        f"{t} / {s} = {t // s} remainder {t % s}, so the quotient lies between {t // s} and {t // s + 1}.",
        f"The situation needs the count {word} of that quotient, giving {key}.",
    ]
    wrong = _count_wrong(t, s, up)
    story, question = stem.rsplit(" How many", 1)
    return single(story, _part("How many" + question, key, 1, working, wrong, _WHOLE, rng))


# -- medium -----------------------------------------------------------------------------------------------------------
#
# Each builder returns the story, the total T of the thing counted, the size s of one unit, the direction and the text of
# the asks. `val_q` (with `price`) asks the money; builders with no sale or cost leave it None.


def _b_paint(rng):
    w, h = rng.randint(4, 12), rng.randint(2, 3)
    walls, coats = rng.choice((2, 3, 4)), rng.choice((2, 3))
    s = rng.choice((8, 10, 12, 14))
    return dict(
        story=f"A painter must paint {walls} identical walls, each {w} m wide and {h} m high, with {coats} coats. One tin covers {s} square metres for one coat.",
        T=walls * w * h * coats,
        alt=[walls * w * h, w * h * coats],
        s=s,
        up=True,
        sing="tin",
        plur="tins",
        count_q="How many tins must she buy?",
        left_q="How many square metres could the paint left over still cover?",
        total_q="How many square metres of paint coverage are needed altogether?",
        val_q="Each tin costs ${price}. What is the total cost of the tins she needs?",
        price=(24, 36, 42, 55, 68),
    )


def _b_turf(rng):
    a, b = rng.randint(6, 15), rng.randint(5, 12)
    s = rng.choice((2, 3, 4, 5))
    return dict(
        story=f"A rectangular lawn measures {a} m by {b} m. Rolls of turf each cover {s} square metres.",
        T=a * b,
        alt=[2 * (a + b), a + b],
        s=s,
        up=True,
        sing="roll",
        plur="rolls",
        count_q="How many rolls of turf must be ordered to cover the whole lawn?",
        left_q="How many square metres of turf are left over?",
        total_q="What is the area of the lawn, in square metres?",
        val_q="Each roll costs ${price}. What is the total cost of the rolls needed?",
        price=(9, 12, 14, 16, 18),
    )


def _b_buses(rng):
    g = [rng.randint(18, 60) for _ in range(3)]
    s = rng.choice((30, 40, 45, 48, 50, 52))
    return dict(
        story=f"Three year groups of {g[0]}, {g[1]} and {g[2]} students are going to a sports carnival. Each bus seats {s} students.",
        T=sum(g),
        alt=[g[0] + g[1], g[1] + g[2]],
        s=s,
        up=True,
        sing="bus",
        plur="buses",
        count_q="How many buses are needed to take every student?",
        left_q="How many seats will be empty?",
        total_q="How many students are going altogether?",
        val_q="Each bus costs ${price} to hire. What is the total hire cost?",
        price=(180, 220, 260, 310, 340),
    )


def _b_trips(rng):
    a, b = rng.randint(20, 90), rng.randint(15, 70)
    s = rng.choice((16, 18, 24, 25, 30))
    return dict(
        story=f"A ute delivers {a} bags of mulch and {b} bags of soil to a garden. It carries {s} bags on each trip.",
        T=a + b,
        alt=[a, b],
        s=s,
        up=True,
        sing="trip",
        plur="trips",
        count_q="How many trips does the ute need to make to deliver every bag?",
        left_q="How many bags of space are unused on the last trip?",
        total_q="How many bags are delivered altogether?",
        val_q="Each trip costs ${price} in fuel and wages. What is the total cost of the trips?",
        price=(35, 40, 45, 55, 60),
    )


def _b_tiles(rng):
    a, b = rng.randint(3, 7), rng.randint(3, 6)
    rooms = rng.choice((2, 3, 4))
    s = rng.choice((3, 4, 5, 6))
    return dict(
        story=f"A builder is tiling {rooms} identical rooms, each {a} m by {b} m. A pack of tiles covers {s} square metres.",
        T=rooms * a * b,
        alt=[a * b, rooms * (a + b)],
        s=s,
        up=True,
        sing="pack",
        plur="packs",
        count_q="How many packs of tiles must be bought?",
        left_q="How many square metres could the tiles left over still cover?",
        total_q="What is the total floor area to be tiled, in square metres?",
        val_q="Each pack costs ${price}. What is the total cost of the packs needed?",
        price=(28, 35, 44, 52, 63),
    )


def _b_teams(rng):
    a, b = rng.randint(14, 40), rng.randint(12, 38)
    s = rng.choice((6, 7, 8, 9, 10))
    return dict(
        story=f"A camp has {a} boys and {b} girls. They are put into teams of {s} for a relay.",
        T=a + b,
        alt=[a, b],
        s=s,
        up=False,
        sing="team",
        plur="teams",
        count_q="How many full teams can be made?",
        left_q="How many campers are not in a full team?",
        total_q="How many campers are there altogether?",
        val_q="Each full team pays an entry fee of ${price}. What is the total entry fee paid?",
        price=(15, 20, 25, 30, 35),
    )


def _b_jugs(rng):
    r, m = rng.randint(6, 18), rng.randint(9, 25)
    s = rng.choice((3, 4, 5, 6, 8))
    return dict(
        story=f"A pump fills a tank at {r} litres per minute and runs for {m} minutes. Juice is then poured into jugs that each hold {s} litres, and only jugs that are filled completely are sold.",
        T=r * m,
        alt=[r + m, r],
        s=s,
        up=False,
        sing="jug",
        plur="jugs",
        count_q="How many jugs can be completely filled?",
        left_q="How many litres are left over after the full jugs are filled?",
        total_q="How many litres does the pump deliver?",
        val_q="Each full jug sells for ${price}. What is the total value of the jugs sold?",
        price=(6, 8, 9, 12, 15),
    )


def _b_cookies(rng):
    trays, per = rng.randint(3, 9), rng.choice((12, 15, 18, 20, 24))
    s = rng.choice((5, 6, 8, 9, 10))
    return dict(
        story=f"A bakery makes {trays} trays of cookies with {per} cookies on each tray. The cookies are packed {s} to a box and only complete boxes go on sale.",
        T=trays * per,
        alt=[trays + per, per],
        s=s,
        up=False,
        sing="box",
        plur="boxes",
        count_q="How many complete boxes can be packed?",
        left_q="How many cookies are left unboxed?",
        total_q="How many cookies are made altogether?",
        val_q="Each box sells for ${price}. What is the total takings from the boxes?",
        price=(5, 6, 7, 8, 9),
    )


def _b_tickets(rng):
    rate, hrs, bonus = rng.randint(16, 28), rng.randint(5, 12), rng.choice((10, 15, 20, 25))
    s = rng.choice((12, 14, 18, 22, 25))
    return dict(
        story=f"Sam earns ${rate} an hour for {hrs} hours at a weekend job and is also paid a ${bonus} bonus. Show tickets cost ${s} each.",
        T=rate * hrs + bonus,
        alt=[rate * hrs, rate + hrs + bonus],
        s=s,
        up=False,
        sing="ticket",
        plur="tickets",
        count_q="How many tickets can Sam buy with these earnings?",
        left_q="How many dollars does Sam have left after buying as many tickets as possible?",
        total_q="How many dollars does Sam earn altogether?",
        val_q="How many dollars does Sam spend on the tickets?",
        price=None,
    )


def _b_shelves(rng):
    n, length = rng.randint(3, 6), rng.choice((60, 75, 80, 90, 120))
    s = rng.choice((4, 5, 6, 7))
    return dict(
        story=f"A bookcase has {n} shelves, each {length} centimetres long. Every book is {s} centimetres thick, and no book is stood on a shelf unless it fits completely.",
        T=n * length,
        alt=[length, n + length],
        s=s,
        up=False,
        sing="book",
        plur="books",
        count_q="How many books can the bookcase hold?",
        left_q="How many centimetres of shelf are left empty in total?",
        total_q="What is the total length of shelf, in centimetres?",
        val_q=None,
        price=None,
    )


_BUILDERS = (
    _b_paint,
    _b_turf,
    _b_buses,
    _b_trips,
    _b_tiles,
    _b_teams,
    _b_jugs,
    _b_cookies,
    _b_tickets,
    _b_shelves,
)


def _scenario(rng, builders=_BUILDERS, need_value=False):
    """One scenario whose quotient lies on the side that makes the nearest whole number wrong."""
    sc = rng.choice(builders)(rng)
    if need_value and sc["val_q"] is None:
        raise Redraw("a scenario with no money")
    if not _split_ok(sc["T"], sc["s"], sc["up"]):
        raise Redraw("a quotient the nearest whole number would get right")
    return sc


_MED_ASK_UP = ("count", "count", "cost", "left")
_MED_ASK_DOWN = ("count", "count", "left")


def _medium(rng):
    sc = _scenario(rng)
    t, s, up = sc["T"], sc["s"], sc["up"]
    key = _ceil(t, s) if up else t // s
    _check_quotient(t, s, up, key)
    r = t % s
    ask = rng.choice(_MED_ASK_UP if up else _MED_ASK_DOWN)
    if ask == "cost":
        price = rng.choice(sc["price"])
        value = key * price
        check(value == int(sympy.ceiling(sympy.Rational(t, s))) * price, "cost disagrees with SymPy")
        q = sc["val_q"].format(price=price)
        wrong = _cost_wrong(t, s, up, price)
        part = _part(q, value, 2, _working(t, s, key, up, price), wrong, _DOLLARS, rng)
    elif ask == "left":
        value = (s - r) if up else r
        check(value == (key * s - t if up else t - key * s), "leftover disagrees")
        if up:
            wrong = [
                (r, "wu-leftover-is-remainder"),
                ((key + 1) * s - t, "wu-adds-extra-unit"),
                (key, "wu-reports-count-not-leftover"),
                ((key + 2) * s - t, "wu-adds-extra-unit"),
            ]
        else:
            wrong = [
                (s - r, "wu-leftover-from-rounded-up"),
                (key, "wu-reports-count-not-leftover"),
                (t - (key - 1) * s, "wu-drops-a-full-group"),
                (t - (key - 2) * s, "wu-drops-a-full-group"),
            ]
        part = _part(sc["left_q"], value, 2, _working(t, s, key, up, None, value), wrong, _WHOLE, rng)
    else:
        part = _part(sc["count_q"], key, 2, _working(t, s, key, up, None), _count_wrong(t, s, up), _WHOLE, rng)
    return single(sc["story"], part)


def _working(t, s, key, up, price, left=None):
    steps = [f"The total is {t}. Divide by {s}: {t // s} remainder {t % s}."]
    steps.append(
        f"A part-pack still has to be bought, so the count is {key}."
        if up
        else f"Only full groups count, so the count is {key}."
    )
    if price is not None:
        steps.append(f"Cost = {key} x {price} = {key * price}.")
    if left is not None:
        steps.append(f"Left over = {left}.")
    return steps


# -- hard -------------------------------------------------------------------------------------------------------------


def _cheapest_dp(need: int, packs) -> int:
    """Cheapest cost of at least `need` units: a dynamic programme over units bought (capped at `need`)."""
    inf = 10**9
    best = [0] + [inf] * need
    for i in range(1, need + 1):
        for size, price in packs:
            best[i] = min(best[i], best[max(0, i - size)] + price)
    return best[need]


def _cheapest_search(need: int, packs):
    """Exhaustive search over whole numbers of each pack; returns (cost, small count, large count)."""
    (s1, p1), (s2, p2) = packs
    best = None
    for b in range(_ceil(need, s2) + 1):
        a = max(0, _ceil(need - b * s2, s1))
        cost = a * p1 + b * p2
        if best is None or cost < best[0]:
            best = (cost, a, b)
    return best


_H_PACK = (
    (
        "A school needs at least {T} {things}. They come in small packs of {s1} for ${p1} and large packs of {s2} for ${p2}. What is the least total cost, in dollars, of buying enough {things}?",
        ("pencils", "balloons", "name tags", "party bags"),
    ),
    (
        "A club must buy at least {T} {things}. A small box holds {s1} and costs ${p1}; a large box holds {s2} and costs ${p2}. Whole boxes only can be bought. What is the lowest possible total cost, in dollars?",
        ("bandages", "bottles of water", "paper cups", "glow sticks"),
    ),
    (
        "For an event, at least {T} {things} are needed. Small bags of {s1} cost ${p1} and large bags of {s2} cost ${p2}. What is the least total cost, in dollars?",
        ("potting mix scoops", "sandwich wraps", "stickers", "seed packets"),
    ),
)


def _hard_packs(rng):
    shape, things = rng.choice(_H_PACK)
    things = rng.choice(things)
    s1 = rng.choice((6, 8, 10, 12))
    s2 = s1 * rng.choice((3, 4, 5)) + rng.choice((-2, 0, 0, 2))
    unit = rng.choice((2, 3, 4))
    p1 = s1 * unit + rng.randint(0, 4)
    p2 = round(s2 * unit * rng.choice((0.7, 0.75, 0.8)))
    need = rng.randint(2 * s2 + 1, 4 * s2 - 1)
    packs = ((s1, p1), (s2, p2))
    cost, a, b = _cheapest_search(need, packs)
    check(cost == _cheapest_dp(need, packs), "the two cheapest-cost routes disagree")
    all_large = _ceil(need, s2) * p2
    all_small = _ceil(need, s1) * p1
    gb = need // s2
    greedy = gb * p2 + _ceil(need - gb * s2, s1) * p1
    if not (a > 0 and b > 0 and cost < all_large and cost < all_small):
        raise Redraw("a mix that is not clearly the cheapest")
    stem = shape.format(T=need, things=things, s1=s1, p1=p1, s2=s2, p2=p2)
    wrong = [
        (all_large, "wu-one-pack-size-only"),
        (greedy, "wu-greedy-largest-pack"),
        (all_small, "wu-one-pack-size-only"),
        (round_half_up(Fraction(need * p2, s2), 0), "wu-costs-fractional-pack"),
    ]
    working = [
        f"Try each number of large packs. {b} large and {a} small packs hold {b * s2 + a * s1} (at least {need}).",
        f"Cost = {b} x {p2} + {a} x {p1} = {cost}, less than {all_large} (large only), {all_small} (small only) and {greedy} (largest first).",
    ]
    return single(stem, _part("", cost, 3, working, wrong, _DOLLARS, rng))


_H_LIMIT = (
    "A cooking club makes cakes. Each cake needs {a} eggs and {b} grams of flour. They have {A} eggs and {B} grams of flour. How many whole cakes can they make?",
    "A craft group makes kits. Each kit needs {a} markers and {b} stickers. They have {A} markers and {B} stickers. How many complete kits can they make?",
    "A cafe makes rolls. Each roll uses {a} slices of ham and {b} slices of cheese. There are {A} slices of ham and {B} slices of cheese. How many rolls can be made?",
    "A garden group makes planter boxes. Each box needs {a} boards and {b} screws. They have {A} boards and {B} screws. How many complete boxes can they build?",
)


def _hard_limit(rng):
    shape = rng.choice(_H_LIMIT)
    a, b = rng.randint(2, 6), rng.choice((15, 20, 25, 40, 60))
    if "flour" not in shape:
        b = rng.randint(2, 9)
    k1, k2 = rng.randint(5, 14), rng.randint(5, 14)
    A = a * k1 + rng.randint(1, a - 1)
    B = b * k2 + rng.randint(1, b - 1)
    f1, f2 = A // a, B // b
    if f1 == f2:
        raise Redraw("both supplies run out together")
    lim_t, lim_s = (A, a) if f1 < f2 else (B, b)
    if not _split_ok(lim_t, lim_s, False):
        raise Redraw("a limiting quotient the nearest whole number would get right")
    key = min(f1, f2)
    check(key == int(min(sympy.floor(sympy.Rational(A, a)), sympy.floor(sympy.Rational(B, b)))), "limit disagrees")
    stem = shape.format(a=a, b=b, A=A, B=B)
    other = max(f1, f2)
    wrong = [
        (other, "wu-ignores-limiting-resource"),
        (_nearest(lim_t, lim_s), "wu-rounds-to-nearest"),
        (other + 1, "wu-ignores-limiting-resource"),
        (key - 1, "wu-drops-a-full-group"),
    ]
    working = [
        f"First supply: {A} / {a} gives {f1} whole sets. Second supply: {B} / {b} gives {f2} whole sets.",
        f"The scarcer supply decides, so {key}.",
    ]
    return single(stem, _part("", key, 3, working, wrong, _WHOLE, rng))


_H_GAP = (
    "A shelf is {L} centimetres long. Display stands {w} centimetres wide are placed along it with a gap of {g} centimetres between neighbouring stands (none at the ends). How many stands fit?",
    "A fence rail is {L} centimetres long and carries {w} centimetre wide brackets, with {g} centimetres between each pair of brackets and no extra gap at either end. How many brackets fit on the rail?",
    "A bench {L} centimetres long seats people who each take {w} centimetres, with {g} centimetres between neighbouring people and no gap at the ends. What is the greatest number of people who can sit on the bench?",
)


def _hard_gap(rng):
    shape = rng.choice(_H_GAP)
    for _ in attempts(500, "a length where the gap rule and the nearest whole number disagree"):
        w, g = rng.randint(20, 60), rng.randint(5, 20)
        n = rng.randint(4, 12)
        length = n * w + (n - 1) * g + rng.randint(1, w + g - 1)
        if _split_ok(length + g, w + g, False):
            break
    key = (length + g) // (w + g)
    brute = max(m for m in range(1, 200) if m * w + (m - 1) * g <= length)
    check(key == brute, "gap fit disagrees with the brute-force count")
    stem = shape.format(L=length, w=w, g=g)
    wrong = [
        (length // w, "wu-ignores-gaps"),
        (length // (w + g), "wu-gap-counted-after-last"),
        (_nearest(length + g, w + g), "wu-rounds-to-nearest"),
        (key + 1, "wu-rounds-up-when-only-full"),
    ]
    working = [
        f"n items need n x {w} + (n - 1) x {g} <= {length}, so n(w + g) <= {length} + {g}.",
        f"({length} + {g}) / {w + g} = {(length + g) / (w + g):.2f}, so at most {key} fit.",
    ]
    return single(stem, _part("", key, 3, working, wrong, _WHOLE, rng))


_H_CUT = (
    "Pieces of timber {p} centimetres long are cut from boards that are each {B} centimetres long. {P} pieces are needed and an offcut cannot be joined to another. How many boards must be bought?",
    "A florist cuts ribbons {p} centimetres long from rolls that are each {B} centimetres long. She needs {P} ribbons and the short end of a roll cannot be used again. How many rolls must she buy?",
    "A tailor needs {P} strips of fabric {p} centimetres long, cut from lengths of {B} centimetres. No strip may be joined, so each length gives only whole strips. How many lengths must be bought?",
)


def _hard_cut(rng):
    shape = rng.choice(_H_CUT)
    for _ in attempts(500, "a cutting plan where every common slip gives a different count"):
        p = rng.randint(35, 95)
        B = rng.randint(2 * p + 10, 4 * p + p // 2)
        per = B // p
        need = rng.randint(3 * per + 1, 9 * per)
        if per >= 2 and B % p != 0 and _split_ok(need, per, True):
            break
    key = _ceil(need, per)
    check(key == int(sympy.ceiling(sympy.Rational(need, sympy.floor(sympy.Rational(B, p))))), "boards disagree")
    stem = shape.format(p=p, B=B, P=need)
    wrong = [
        (_ceil(need * p, B), "wu-ignores-offcuts"),
        (_ceil(need, _ceil(B, p)), "wu-rounds-pieces-per-board-up"),
        (need // per, "wu-rounds-down-when-covering"),
        (key + 1, "wu-adds-extra-unit"),
    ]
    working = [
        f"Each {B} cm length gives {B} / {p} = {B / p:.2f}, so only {per} whole pieces.",
        f"{need} pieces need {need} / {per} = {need / per:.2f} lengths, so {key}.",
    ]
    return single(stem, _part("", key, 3, working, wrong, _WHOLE, rng))


@generator(
    "sc_whole_units",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def sc_whole_units(rng, tier: str = "easy"):
    """Whole packs needed (round up) versus full groups that can be made (round down); see the module docstring."""
    if tier == "easy":
        return draw(_easy, rng)
    if tier == "medium":
        return draw(_medium, rng)
    if tier == "hard":
        pick = rng.random()
        return draw(
            _hard_packs if pick < 0.3 else _hard_limit if pick < 0.55 else _hard_gap if pick < 0.8 else _hard_cut, rng
        )
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain -------------------------------------------------------------------------------------------------------


def _chain(rng):
    sc = _scenario(rng, need_value=True)
    t, s, up = sc["T"], sc["s"], sc["up"]
    price = s if sc["price"] is None else rng.choice(sc["price"])  # tickets: the price is the pack size
    r = t % s
    n = _ceil(t, s) if up else t // s
    _check_quotient(t, s, up, n)
    value = n * price
    left = (n * s - t) if up else (t - n * s)
    check(left == (s - r if up else r), "leftover disagrees")
    check(value == int(sympy.Rational(n) * price), "value disagrees")
    wrong_a = [(v, "wu-total-built-wrongly") for v in sc["alt"]] + [
        (t + s, "wu-total-built-wrongly"),
        (t - s, "wu-total-built-wrongly"),
    ]
    part_a = _part(sc["total_q"], t, 1, [f"The total is {t}."], wrong_a, _WHOLE, rng, sensible_range=(0, None))
    part_b = _part(
        sc["count_q"] + " Use your answer to part (a).",
        n,
        2,
        _working(t, s, n, up, None),
        _count_wrong(t, s, up),
        _WHOLE,
        rng,
        follow_through=(f"ceiling(part_a/{s})" if up else f"floor(part_a/{s})"),
        sensible_range=(0, None),
    )
    wrong_c = _cost_wrong(t, s, up, price)
    part_c = _part(
        sc["val_q"].format(price=price) + " Use your answer to part (b).",
        value,
        2,
        [f"{n} x {price} = {value}."],
        wrong_c,
        _DOLLARS,
        rng,
        follow_through=f"part_b*{price}",
        sensible_range=(0, None),
    )
    if up:
        ft_d = f"part_b*{s} - part_a"
        wrong_d = [
            (r, "wu-leftover-is-remainder"),
            ((n + 1) * s - t, "wu-adds-extra-unit"),
            (n, "wu-reports-count-not-leftover"),
            ((n + 2) * s - t, "wu-adds-extra-unit"),
        ]
    else:
        ft_d = f"part_a - part_b*{s}"
        wrong_d = [
            (s - r, "wu-leftover-from-rounded-up"),
            (n, "wu-reports-count-not-leftover"),
            (t - (n - 1) * s, "wu-drops-a-full-group"),
            (t - (n - 2) * s, "wu-drops-a-full-group"),
        ]
    part_d = _part(
        sc["left_q"] + " Use your answers to parts (a) and (b).",
        left,
        2,
        [f"Left over = {left}."],
        wrong_d,
        _WHOLE,
        rng,
        follow_through=ft_d,
        sensible_range=(0, None),
    )
    return dict(text=sc["story"], parts=[part_a, part_b, part_c, part_d])


@generator("sc_whole_units_chain", level=14, presets={"exam": dict(tier="exam")})
def sc_whole_units_chain(rng, tier: str = "exam"):
    """Whole units needed or made for one scenario: the total, the whole units, their cost or value, the leftover."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_chain, rng)
