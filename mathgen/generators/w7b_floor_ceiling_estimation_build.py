"""Generates `w7b_floor_ceiling_estimation` questions (easy, medium, hard) and the exam chain
`exd_x_w7b_floor_ceiling_estimation`.

``floor-ceiling-estimation`` (``arithmetic``, answer ``number``): practical estimates that depend on which way to round.
A quantity that must be enough (boxes, trips, packs) is rounded up; a count of whole items that fit is rounded down; a
safe upper estimate rounds every factor up and a safe lower estimate rounds every factor down. Every answer is a whole
number and none can be had by rounding to the nearest.

* easy: the boxes, packs or trips needed for a total (the quotient rounded up, sometimes an exact division where nothing
  is added), the whole items that fit in a length or a container (rounded down), or the number left over.
* medium: the vehicles needed when extra people join the groups, a safe upper or lower estimate of a product by rounding
  both factors up or down to a stated place, how far a stated answer is above the upper estimate (a reasonable-answer
  check), or how many more are needed to fill the last box.
* hard: trips when a weight limit and a count limit both apply (round the load down, then the trips up), posts for a
  straight fence or a closed loop, the vehicles when large coaches are filled first and a small one carries the rest, and
  the boxes that guarantee enough, or the full boxes that are guaranteed, for a count known only to lie in a range.
* exam (``exd_x_w7b_floor_ceiling_estimation``): boxes needed, the empty spaces in the last box, a safe upper estimate of
  the cost, and how far that estimate is above the exact cost.

Every answer is built from chosen quantities and re-derived a second way: the rounded quotient as the least (or the
greatest) whole number of groups that satisfies the condition, found by search rather than by the division formula.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._w6a_kit import enough, num_dist, num_fields, part, retry_draw

TOPIC = "w7b_floor_ceiling_estimation"
EXAM_TOPIC = "exd_x_w7b_floor_ceiling_estimation"

AREA = "floor-ceiling-estimation"
MISCONCEPTIONS = [
    (
        "fce-rounds-down",
        "Rounds the quotient down when a partly filled pack still has to be bought or a partly full trip still has to be made.",
        AREA,
    ),
    ("fce-adds-one-when-exact", "Adds one to a quotient that is already a whole number.", AREA),
    ("fce-adds-a-spare", "Rounds up and then adds a spare one that the question did not ask for.", AREA),
    (
        "fce-rounds-up-whole-items",
        "Rounds up the number of whole items that fit, counting an offcut that is too short to use.",
        AREA,
    ),
    ("fce-gives-leftover", "Gives the amount left over where the number of whole groups was asked.", AREA),
    ("fce-gives-quotient", "Gives the number of full groups where the amount left over was asked.", AREA),
    (
        "fce-gives-shortfall",
        "Gives how many more are needed to fill another group where the amount left over was asked.",
        AREA,
    ),
    ("fce-gives-count-for-leftover", "Gives the number of groups needed where the amount left over was asked.", AREA),
    (
        "fce-gives-leftover-for-shortfall",
        "Gives the amount in the last box where the amount needed to fill it was asked.",
        AREA,
    ),
    ("fce-forgets-extra", "Leaves out the extra people or items that were added to the groups.", AREA),
    (
        "fce-rounds-each-part",
        "Rounds up each part separately and adds, instead of adding first and rounding once.",
        AREA,
    ),
    (
        "fce-mixes-directions",
        "Rounds one factor up and the other down, so the estimate is neither a safe upper estimate nor a safe lower one.",
        AREA,
    ),
    ("fce-uses-exact-values", "Gives the exact value where a rounded estimate was asked.", AREA),
    (
        "fce-rounds-to-nearest-estimate",
        "Rounds each factor to the nearest place value instead of in the direction asked.",
        AREA,
    ),
    ("fce-wrong-bound", "Rounds down for the upper estimate, or up for the lower estimate.", AREA),
    (
        "fce-compares-to-lower-bound",
        "Measures the excess above the lower estimate where the upper estimate was the one to check against.",
        AREA,
    ),
    (
        "fce-gives-estimate-not-gap",
        "Gives the estimate itself where the difference between the estimate and another value was asked.",
        AREA,
    ),
    (
        "fce-gives-exact-not-gap",
        "Gives the exact value where the difference between the estimate and the exact value was asked.",
        AREA,
    ),
    ("fce-ignores-weight-limit", "Uses only the count limit for each trip and ignores the weight limit.", AREA),
    (
        "fce-ignores-whole-items",
        "Divides the total weight by the weight limit without allowing for whole items, so a part-filled load is counted.",
        AREA,
    ),
    (
        "fce-rounds-load-to-nearest",
        "Rounds the number of items one trip can carry to the nearest whole number instead of down.",
        AREA,
    ),
    ("fce-forgets-end-post", "Forgets that a straight fence needs one more post than it has gaps.", AREA),
    (
        "fce-adds-post-on-loop",
        "Adds an extra post for a closed loop, although the last gap ends at the first post.",
        AREA,
    ),
    (
        "fce-rounds-gaps-down",
        "Rounds the number of gaps down and leaves the last stretch of fence without a post.",
        AREA,
    ),
    ("fce-all-large-vehicles", "Uses only the large vehicle for everyone.", AREA),
    ("fce-all-small-vehicles", "Uses only the small vehicle for everyone.", AREA),
    ("fce-forgets-the-rest", "Counts only the full large vehicles and forgets the people left over.", AREA),
    (
        "fce-uses-lower-bound",
        "Uses the smallest possible count when the number needed must cover the largest possible one.",
        AREA,
    ),
    (
        "fce-uses-upper-bound",
        "Uses the largest possible count when the number of boxes certain to be filled was asked.",
        AREA,
    ),
    ("fce-uses-midpoint", "Uses the middle of the range instead of the end that settles the question.", AREA),
    (
        "fce-rounds-up-guaranteed-full",
        "Rounds up the number of boxes that can be filled for certain, counting a box that may not be full.",
        AREA,
    ),
]

_CEIL_CONTEXTS = (
    # (items, container, per-container phrase, verb phrase for the ask)
    ("books", "box", "boxes", "A library has {t} books to pack. Each box holds {c} books."),
    ("students", "minibus", "minibuses", "{t} students are going on an excursion. Each minibus seats {c} students."),
    ("tiles", "carton", "cartons", "A builder needs {t} tiles. Tiles are sold only in cartons of {c}."),
    (
        "bags of soil",
        "trolley load",
        "trolley loads",
        "A gardener must move {t} bags of soil. A trolley carries {c} bags at a time.",
    ),
    ("eggs", "tray", "trays", "A farm has {t} eggs to sell. Each tray holds {c} eggs."),
    ("chairs", "stack", "stacks", "A hall has {t} chairs to put away. Each stack holds {c} chairs."),
)

_FLOOR_CONTEXTS = (
    ("A ribbon is {t} cm long. Each bow needs {c} cm of ribbon.", "How many whole bows can be made?"),
    ("A plank is {t} cm long. Each shelf needs a piece {c} cm long.", "How many shelves can be cut from the plank?"),
    ("A tank holds {t} mL of juice. Each bottle takes {c} mL.", "How many bottles can be filled completely?"),
    ("A garden path is {t} cm wide. Each paver is {c} cm wide.", "How many whole pavers fit across the path?"),
    ("A roll has {t} metres of cable. Each cable run needs {c} metres.", "How many complete cable runs can be made?"),
)


def _ceil(t: int, c: int) -> int:
    return -(-t // c)


def _least_groups(t: int, c: int) -> int:
    """The least number of groups of size c that hold t, found by counting up (a second route to the ceiling)."""
    k = 0
    for k in range(t + 1):
        if k * c >= t:
            return k
    raise AssertionError  # pragma: no cover


def _most_groups(t: int, c: int) -> int:
    """The greatest number of whole groups of size c inside t, found by counting up (a second route to the floor)."""
    best = 0
    for k in range(t + 1):
        if k * c <= t:
            best = k
        else:
            break
    return best


def _answer(value: int) -> dict:
    return num_fields(value)


def _round_up(x: int, place: int) -> int:
    return -(-x // place) * place


def _round_down(x: int, place: int) -> int:
    return x // place * place


def _round_nearest(x: int, place: int) -> int:
    return (2 * x + place) // (2 * place) * place


def _name(place: int) -> str:
    return {10: "ten", 100: "hundred"}[place]


# -- easy -------------------------------------------------------------------------------------------------------------


def _easy_ceiling(rng):
    items, one, many, intro = rng.choice(_CEIL_CONTEXTS)
    c = rng.choice((6, 8, 10, 12, 15, 20, 24, 25, 30, 36, 40))
    exact = rng.random() < 0.25
    if exact:
        k = rng.randint(4, 80)
        t = k * c
    else:
        for _ in attempts(200, "a total that does not divide"):
            t = rng.randint(3 * c, 90 * c)
            if t % c:
                break
    value = _ceil(t, c)
    check(value == _least_groups(t, c), "ceiling differs")
    if exact:
        cands = [
            (value + 1, "fce-adds-one-when-exact"),
            (value - 1, "fce-rounds-down"),
            (value + 2, "fce-adds-a-spare"),
        ]
    else:
        cands = [
            (t // c, "fce-rounds-down"),
            (value + 1, "fce-adds-a-spare"),
            (t % c, "fce-gives-leftover"),
            (c - t % c, "fce-gives-shortfall"),
        ]
    dist = num_dist(value, [x for x in cands if x[0] is not None])
    enough(dist)
    ask = phrase(rng, ("How many {m} are needed?", "What is the least number of {m} that will do?"), m=many)
    stem = f"{intro.format(t=f'{t:,}', c=c)} {ask}"
    working = [
        f"{t:,} / {c} = {t / c:.2f}." if not exact else f"{t:,} / {c} = {t // c} exactly.",
        f"A partly full {one} is still needed, so round up to {value}."
        if not exact
        else f"The division is exact, so {value} {many} are enough.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


def _easy_floor(rng):
    intro, ask = rng.choice(_FLOOR_CONTEXTS)
    c = rng.choice((6, 7, 8, 12, 15, 18, 24, 35, 45, 60))
    for _ in attempts(200, "a total with a remainder of at least two"):
        t = rng.randint(3 * c, 60 * c)
        if t % c > 1:
            break
    value = _most_groups(t, c)
    check(value == t // c, "floor differs")
    cands = [
        (_ceil(t, c), "fce-rounds-up-whole-items"),
        (t % c if t % c > 1 else None, "fce-gives-leftover"),
    ]
    dist = num_dist(value, [x for x in cands if x[0] is not None])
    enough(dist)
    stem = f"{intro.format(t=f'{t:,}', c=c)} {ask}"
    working = [f"{t:,} / {c} = {t / c:.2f}.", f"Only whole items count, so round down to {value}."]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


def _easy_leftover(rng):
    items, one, many, intro = rng.choice(_CEIL_CONTEXTS)
    c = rng.choice((8, 10, 12, 15, 20, 24, 25, 30))
    for _ in attempts(200, "a total with a remainder"):
        t = rng.randint(4 * c, 60 * c)
        if t % c > 1 and (t % c) != c - 1:
            break
    r = t % c
    check(t - c * _most_groups(t, c) == r, "remainder differs")
    cands = [
        (t // c, "fce-gives-quotient"),
        (_ceil(t, c), "fce-gives-count-for-leftover"),
        (c - r, "fce-gives-shortfall"),
    ]
    dist = num_dist(r, cands)
    enough(dist)
    base = intro.format(t=f"{t:,}", c=c)
    stem = (
        f"{base} All the full {many} are filled first. How many {items} are left over for a last, partly filled {one}?"
    )
    working = [f"{t:,} = {c} x {t // c} + {r}.", f"There are {r} {items} left over."]
    return dict(text=stem, working=working, distractors=dist, **_answer(r), answer_kind="number")


# -- medium -----------------------------------------------------------------------------------------------------------


def _medium_with_extras(rng):
    cap = rng.choice((30, 35, 40, 45, 48, 50, 52, 55, 60))
    for _ in attempts(500, "a total that needs the extras to change the answer"):
        groups = rng.randint(3, 9)
        each = rng.randint(18, 34)
        extra = rng.randint(4, 15)
        t = groups * each + extra
        if t % cap and groups * each % cap:
            break
    value = _ceil(t, cap)
    check(value == _least_groups(t, cap), "ceiling differs")
    separate = _ceil(groups * each, cap) + _ceil(extra, cap)
    cands = [
        (_ceil(groups * each, cap), "fce-forgets-extra"),
        (separate, "fce-rounds-each-part"),
        (t // cap, "fce-rounds-down"),
        (value + 1, "fce-adds-a-spare"),
        (t % cap, "fce-gives-leftover"),
    ]
    dist = num_dist(value, [x for x in cands if x[0] is not None])
    enough(dist)
    stem = phrase(
        rng,
        (
            "{g} classes of {e} students are going to a concert, with {x} teachers going too. Each coach seats {c} people. "
            "What is the least number of coaches needed?",
            "A school sends {g} classes of {e} students each, plus {x} adults. One coach seats {c}. How many coaches must be booked?",
        ),
        g=groups,
        e=each,
        x=extra,
        c=cap,
    )
    working = [
        f"People = {groups} x {each} + {extra} = {t}.",
        f"{t} / {cap} = {t / cap:.2f}, so round up to {value} coaches.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


def _medium_estimate(rng):
    pa, pb = rng.choice(((10, 10), (10, 10), (100, 10)))
    upper = rng.random() < 0.6
    direct = _round_up if upper else _round_down
    against = _round_down if upper else _round_up
    for _ in attempts(500, "factors whose nearest rounding differs from the direction asked"):
        a = rng.randint(21, 98) if pa == 10 else rng.randint(210, 980)
        b = rng.randint(21, 98)
        na, nb = _round_nearest(a, pa), _round_nearest(b, pb)
        if (
            a % pa
            and b % pb
            and na * nb != direct(a, pa) * direct(b, pb)
            and against(a, pa) * against(b, pb) != direct(a, pa) * direct(b, pb)
        ):
            break
    ra, rb = direct(a, pa), direct(b, pb)
    value = ra * rb
    check(value >= a * b if upper else value <= a * b, "the estimate is not on the safe side")
    cands = [
        (na * nb, "fce-rounds-to-nearest-estimate"),
        (against(a, pa) * against(b, pb), "fce-wrong-bound"),
        (a * b, "fce-uses-exact-values"),
        (ra * against(b, pb), "fce-mixes-directions"),
        (against(a, pa) * rb, "fce-mixes-directions"),
    ]
    dist = num_dist(value, cands)
    enough(dist)
    direction = "up" if upper else "down"
    kind = "upper" if upper else "lower"
    if pa == pb:
        how = f"Round both numbers {direction} to the nearest {_name(pa)}"
    else:
        how = f"Round the first number {direction} to the nearest {_name(pa)} and the second {direction} to the nearest {_name(pb)}"
    stem = phrase(
        rng,
        (
            "A shop sells {a} boxes at ${b} each. {how} to find a safe {k} estimate of the takings, in dollars.",
            "{a} items cost ${b} each. To be sure the estimate is never {opp}, {how_l} and multiply. What is the {k} estimate, in dollars?",
        ),
        a=a,
        b=b,
        how=how,
        how_l=how[0].lower() + how[1:],
        k=kind,
        opp="too high" if not upper else "too low",
    )
    working = [
        f"Rounded {direction}: {a} becomes {ra} and {b} becomes {rb}.",
        f"{ra} x {rb} = {value:,}; the exact takings are {a * b:,}, so this is a safe {kind} estimate.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


def _medium_check_answer(rng):
    for _ in attempts(500, "a mistaken answer above the upper estimate"):
        a, b = rng.randint(21, 98), rng.randint(21, 98)
        if a % 10 and b % 10:
            break
    lo, hi = _round_down(a, 10) * _round_down(b, 10), _round_up(a, 10) * _round_up(b, 10)
    wrong = a * b * 10 + rng.randint(1, 9) * 10 if rng.random() < 0.5 else a * b + rng.randint(2, 6) * 1000
    wrong = max(wrong, hi + 500)
    value = wrong - hi
    check(value > 0, "the stated answer is not above the upper estimate")
    cands = [
        (wrong - lo, "fce-compares-to-lower-bound"),
        (wrong - a * b, "fce-gives-exact-not-gap"),
        (hi, "fce-gives-estimate-not-gap"),
        (wrong - _round_nearest(a, 10) * _round_nearest(b, 10), "fce-rounds-to-nearest-estimate"),
    ]
    dist = num_dist(value, cands)
    enough(dist)
    stem = (
        f"A student works out {a} x {b} and gets {wrong:,}. To check, round both numbers up to the nearest ten and multiply to get an upper "
        f"estimate. By how much does the student's answer exceed this upper estimate?"
    )
    working = [
        f"Upper estimate = {_round_up(a, 10)} x {_round_up(b, 10)} = {hi:,}.",
        f"{wrong:,} - {hi:,} = {value:,}, so the answer cannot be right.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


def _medium_shortfall(rng):
    items, one, many, intro = rng.choice(_CEIL_CONTEXTS)
    c = rng.choice((8, 10, 12, 15, 20, 24, 25, 30))
    for _ in attempts(200, "a total with a remainder"):
        t = rng.randint(4 * c, 60 * c)
        if 1 < t % c < c - 1:
            break
    r = t % c
    value = c - r
    check(_ceil(t, c) * c - t == value, "shortfall differs")
    cands = [
        (r, "fce-gives-leftover-for-shortfall"),
        (t // c, "fce-rounds-down"),
        (_ceil(t, c), "fce-gives-count-for-leftover"),
        (c - r + c, "fce-adds-a-spare"),
    ]
    dist = num_dist(value, cands)
    enough(dist)
    base = intro.format(t=f"{t:,}", c=c)
    stem = f"{base} They are packed into the fewest {many} possible. How many more {items} would the last {one} need to be completely full?"
    working = [
        f"{t:,} = {c} x {t // c} + {r}.",
        f"The last {one} has {r} {items}, so {c} - {r} = {value} more fill it.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


# -- hard -------------------------------------------------------------------------------------------------------------


def _hard_two_limits(rng):
    for _ in attempts(1000, "a weight-limited load"):
        cap = rng.choice((600, 750, 800, 900, 1000, 1200))
        w = rng.randint(41, 97)
        count = rng.randint(12, 20)
        per_trip = cap // w
        total = rng.randint(60, 200)
        if (
            per_trip < count
            and cap % w
            and _ceil(total, per_trip) != _ceil(total, count)
            and _ceil(total * w, cap) != _ceil(total, per_trip)
            and per_trip >= 4
        ):
            if round(Fraction(cap, w)) != per_trip:
                break
    value = _ceil(total, per_trip)
    # second route: add loads trip by trip until every crate has been taken.
    taken, trips = 0, 0
    for trips in range(1, total + 1):
        taken += min(count, cap // w)
        if taken >= total:
            break
    check(trips == value, "trip count differs")
    nearest_load = round(Fraction(cap, w))
    cands = [
        (_ceil(total, count), "fce-ignores-weight-limit"),
        (_ceil(total * w, cap), "fce-ignores-whole-items"),
        (_ceil(total, max(1, nearest_load)), "fce-rounds-load-to-nearest"),
        (total // per_trip, "fce-rounds-down"),
    ]
    dist = num_dist(value, cands)
    enough(dist)
    stem = (
        f"A van can carry at most {cap} kg and at most {count} crates on any trip. {total} crates, each weighing {w} kg, must be delivered. "
        "What is the least number of trips the van needs?"
    )
    working = [
        f"By weight, {cap} / {w} = {cap / w:.2f}, so at most {per_trip} crates (fewer than the limit of {count}).",
        f"{total} / {per_trip} = {total / per_trip:.2f}, so round up to {value} trips.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


def _hard_fence(rng):
    loop = rng.random() < 0.35
    s2 = rng.choice((4, 5, 6, 7, 8, 9, 10))  # spacing in half-metres: 2 m, 2.5 m, ...
    spacing = Fraction(s2, 2)
    for _ in attempts(500, "a fence length"):
        length = rng.randint(24, 150)
        gaps = _ceil(2 * length, s2)
        if (2 * length) % s2:
            break
    value = gaps if loop else gaps + 1
    # second route: place posts one gap at a time until the end of the fence is reached.
    reach, posts = Fraction(0), 1
    for _ in range(length + 2):
        if reach >= length:
            break
        reach += spacing
        posts += 1
    check(posts - 1 == gaps, "gap count differs")
    cands = (
        [
            (gaps + 1, "fce-adds-post-on-loop"),
            (gaps - 1, "fce-rounds-gaps-down"),
            (int(Fraction(2 * length, s2)) + 0, "fce-rounds-gaps-down"),
        ]
        if loop
        else [
            (gaps, "fce-forgets-end-post"),
            (int(Fraction(2 * length, s2)) + 1, "fce-rounds-gaps-down"),
            (gaps + 2, "fce-adds-a-spare"),
        ]
    )
    dist = num_dist(value, cands)
    enough(dist)
    sp = f"{float(spacing):g}"
    if loop:
        stem = (
            f"A circular garden bed has a border {length} m around. Posts are placed all the way round, evenly spaced no more than {sp} m apart, "
            "with the last gap ending at the first post. What is the least number of posts needed?"
        )
        working = [
            f"{length} / {sp} = {length / float(sp):.2f}, so round the number of gaps up to {gaps}.",
            f"On a closed loop the number of posts equals the number of gaps: {value}.",
        ]
    else:
        stem = (
            f"A straight fence is {length} m long. A post is placed at each end and the others are spaced no more than {sp} m apart. "
            "What is the least number of posts needed?"
        )
        working = [
            f"{length} / {sp} = {length / float(sp):.2f}, so round the number of gaps up to {gaps}.",
            f"A straight fence needs one more post than it has gaps: {gaps} + 1 = {value}.",
        ]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


def _hard_two_vehicles(rng):
    for _ in attempts(1000, "a two-vehicle draw"):
        big = rng.choice((44, 49, 52, 57, 60))
        small = rng.choice((12, 15, 16, 20))
        people = rng.randint(120, 420)
        full, rest = divmod(people, big)
        if full >= 2 and rest > 0 and _ceil(rest, small) >= 2 and _ceil(people, big) != full + _ceil(rest, small):
            break
    value = full + _ceil(rest, small)
    check(full * big + _ceil(rest, small) * small >= people > full * big, "the vehicles do not cover the group")
    cands = [
        (_ceil(people, big), "fce-all-large-vehicles"),
        (_ceil(people, small), "fce-all-small-vehicles"),
        (full, "fce-forgets-the-rest"),
        (full + rest // small, "fce-rounds-down"),
    ]
    dist = num_dist(value, cands)
    enough(dist)
    stem = (
        f"{people} people travel to a match. Coaches seat {big} and minibuses seat {small}. Coaches are filled first and minibuses carry "
        f"everyone left over. How many vehicles are needed in total?"
    )
    working = [
        f"{people} = {big} x {full} + {rest}, so {full} coaches are filled.",
        f"{rest} people remain: {rest} / {small} = {rest / small:.2f}, so {_ceil(rest, small)} minibuses.",
        f"Total vehicles = {full} + {_ceil(rest, small)} = {value}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


def _hard_range(rng):
    c = rng.choice((12, 18, 24, 30, 36))
    for _ in attempts(500, "a range with different rounded ends"):
        lo = rng.randint(30 * c, 60 * c)
        hi = lo + rng.randint(2 * c, 9 * c)
        if lo % c and hi % c and _ceil(lo, c) != _ceil(hi, c) and lo // c != hi // c:
            break
    need_enough = rng.random() < 0.6
    if need_enough:
        value = _ceil(hi, c)
        check(value == _least_groups(hi, c) and value * c >= hi >= lo, "ceiling differs")
        mid = (lo + hi) // 2
        cands = [
            (_ceil(lo, c), "fce-uses-lower-bound"),
            (hi // c, "fce-rounds-down"),
            (_ceil(mid, c), "fce-uses-midpoint"),
        ]
        ask = "How many boxes must be ordered to be certain there are enough?"
        working = [
            f"To be certain, plan for the most there could be: {hi}.",
            f"{hi} / {c} = {hi / c:.2f}, so round up to {value} boxes.",
        ]
    else:
        value = lo // c
        check(value == _most_groups(lo, c), "floor differs")
        mid = (lo + hi) // 2
        cands = [
            (_ceil(lo, c), "fce-rounds-up-guaranteed-full"),
            (hi // c, "fce-uses-upper-bound"),
            (mid // c, "fce-uses-midpoint"),
        ]
        ask = "How many boxes can be filled completely for certain?"
        working = [
            f"For certain means even the least there could be: {lo}.",
            f"{lo} / {c} = {lo / c:.2f}, so round down to {value} full boxes.",
        ]
    dist = num_dist(value, [x for x in cands if x[0] is not None])
    enough(dist)
    stem = f"A packer has between {lo:,} and {hi:,} pencils to pack (the exact number is not yet known). Each box holds {c} pencils. {ask}"
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


_EASY = (_easy_ceiling, _easy_floor, _easy_leftover)
_MEDIUM = (_medium_with_extras, _medium_estimate, _medium_check_answer, _medium_shortfall)
_HARD = (_hard_two_limits, _hard_fence, _hard_two_vehicles, _hard_range)


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w7b_floor_ceiling_estimation(rng, tier="easy"):
    """Rounding down and rounding up in practical estimates; the tiers are described in the module docstring."""
    pool = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}.get(tier)
    check(pool is not None, f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# -- exam -------------------------------------------------------------------------------------------------------------


def _exam_chain(rng):
    for _ in attempts(1000, "a boxes-and-cost chain"):
        c = rng.choice((12, 16, 24, 30, 36))
        t = rng.randint(250, 900)
        p = rng.randint(14, 48)
        boxes = _ceil(t, c)
        if (
            t % c
            and boxes % 10
            and p % 10
            and _round_nearest(boxes, 10) != _round_up(boxes, 10)
            and _round_nearest(p, 10) != _round_up(p, 10)
        ):
            break
    check(boxes == _least_groups(t, c), "boxes differ")
    empty = boxes * c - t
    check(0 < empty < c, "empty spaces differ")
    est = _round_up(boxes, 10) * _round_up(p, 10)
    exact = boxes * p
    over = est - exact
    check(est >= exact and over > 0, "the upper estimate is not above the exact cost")
    a_dist = num_dist(
        boxes,
        [
            (t // c, "fce-rounds-down"),
            (boxes + 1, "fce-adds-a-spare"),
            (t % c, "fce-gives-leftover"),
        ],
    )
    b_dist = num_dist(
        empty,
        [
            (t % c, "fce-gives-leftover-for-shortfall"),
            (t // c, "fce-rounds-down"),
            (boxes, "fce-gives-count-for-leftover"),
            (empty + c, "fce-adds-a-spare"),
        ],
    )
    c_dist = num_dist(
        est,
        [
            (_round_nearest(boxes, 10) * _round_nearest(p, 10), "fce-rounds-to-nearest-estimate"),
            (exact, "fce-uses-exact-values"),
            (_round_down(boxes, 10) * _round_down(p, 10), "fce-wrong-bound"),
            (_round_up(boxes, 10) * p, "fce-mixes-directions"),
            (boxes * _round_up(p, 10), "fce-mixes-directions"),
        ],
    )
    d_dist = num_dist(
        over,
        [
            (est, "fce-gives-estimate-not-gap"),
            (exact, "fce-gives-exact-not-gap"),
            (exact - _round_down(boxes, 10) * _round_down(p, 10), "fce-compares-to-lower-bound"),
            (est - _round_nearest(boxes, 10) * _round_nearest(p, 10), "fce-rounds-to-nearest-estimate"),
        ],
    )
    for d in (a_dist, b_dist, c_dist, d_dist):
        enough(d)
    return dict(
        text=f"A school must pack {t:,} workbooks into boxes. Each box holds {c} workbooks, and each box costs ${p} to post.",
        parts=[
            part(
                "What is the least number of boxes needed?",
                "number",
                num_fields(boxes),
                1,
                [f"{t:,} / {c} = {t / c:.2f}, so round up to {boxes} boxes."],
                a_dist,
                sensible_range=(0, None),
            ),
            part(
                "How many more workbooks would fill the last box completely?",
                "number",
                num_fields(empty),
                1,
                [f"{boxes} x {c} = {boxes * c}, and {boxes * c} - {t:,} = {empty}."],
                b_dist,
                follow_through=f"part_a*{c} - {t}",
                sensible_range=(0, None),
            ),
            part(
                "To estimate the total postage, round the number of boxes up to the nearest ten and the cost per box up to the nearest ten dollars. "
                "What is this safe upper estimate, in dollars?",
                "number",
                num_fields(est),
                2,
                [f"{_round_up(boxes, 10)} x {_round_up(p, 10)} = {est:,}."],
                c_dist,
                sensible_range=(0, None),
            ),
            part(
                "By how many dollars is the upper estimate more than the exact postage cost?",
                "number",
                num_fields(over),
                2,
                [f"Exact cost = {boxes} x {p} = {exact:,}.", f"{est:,} - {exact:,} = {over:,}."],
                d_dist,
                follow_through=f"part_c - part_a*{p}",
                sensible_range=(0, None),
            ),
        ],
    )


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w7b_floor_ceiling_estimation(rng, tier="exam"):
    """Exam chain: boxes needed, empty spaces in the last box, a safe upper estimate of the cost and its excess."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return retry_draw(rng, _exam_chain)
