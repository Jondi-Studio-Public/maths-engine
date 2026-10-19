"""Generates `w3_hcf_lcm_words` (easy, medium, hard) and the exam chain `exd_x_w3_hcf_lcm_words`.

``hcf-lcm-word-problems`` (``secondary-number``, answer ``number``): problems in context that need the highest common
factor (cutting or sharing with nothing left over) or the lowest common multiple (events that repeat, meeting again).

* easy: a pair of small quantities cut or shared into equal groups (the HCF), two repeating events (the LCM), or the
  HCF followed by one division (the size of each share).
* medium: the number of tiles on a rectangular floor, the trips made before two buses leave together again, the LCM of
  three intervals, the turns of the smaller gear before the same teeth meet, the HCF of three lengths.
* hard: no key word in the stem and a second step (the rectangles that tile the smallest square), the greatest divisor that
  leaves stated remainders, the smallest number that leaves a stated remainder, the number of meetings in a period (the
  start counts) and the other number when the HCF, the LCM and one number are known.
* exam: two runners with different lap times; the time until they next meet, the laps each has run, and the number of
  meetings in a longer period.

Each draw is built from a chosen HCF and coprime multipliers, so the key is known before the question is written, and is
re-derived with `math.gcd` and `math.lcm` from the numbers in the question.
"""

from __future__ import annotations

from math import gcd, lcm

from ..core import attempts, check, generator
from ._w3_kit import WHOLE
from ._w6a_kit import TooFewDistractors, enough, part, retry_draw

TOPIC = "w3_hcf_lcm_words"
EXAM_TOPIC = "exd_x_w3_hcf_lcm_words"
_A = "hcf-lcm-word-problems"

MISCONCEPTIONS = [
    ("hl-product", "Multiplies the two numbers instead of finding their common factor or multiple.", _A),
    ("hl-sum", "Adds the two numbers.", _A),
    ("hl-smaller", "Gives the smaller of the two numbers.", _A),
    ("hl-larger", "Gives the larger of the two numbers.", _A),
    ("hl-difference", "Gives the difference between the two numbers.", _A),
    ("hl-lcm-for-hcf", "Finds the lowest common multiple where the highest common factor was needed.", _A),
    ("hl-hcf-for-lcm", "Finds the highest common factor where the lowest common multiple was needed.", _A),
    ("hl-hcf-not-divided", "Gives the number of items in a share as the HCF, forgetting to divide by it.", _A),
    ("hl-shares-for-size", "Gives the number of groups where the size of each group was asked.", _A),
    ("hl-size-for-shares", "Gives the size of each group where the number of groups was asked.", _A),
    ("hl-one-dimension", "Counts the tiles along one side only.", _A),
    ("hl-tiles-sum", "Adds the number of tiles along each side instead of multiplying them.", _A),
    ("hl-lcm-not-divided", "Gives the time until the buses leave together where the number of trips was asked.", _A),
    ("hl-trips-other-bus", "Gives the number of trips of the other bus.", _A),
    ("hl-two-numbers-only", "Finds the lowest common multiple of two of the three numbers.", _A),
    ("hl-three-product", "Multiplies the three numbers together.", _A),
    ("hl-turns-big-gear", "Gives the turns of the larger gear where the smaller gear was asked.", _A),
    ("hl-turns-lcm", "Gives the number of teeth that pass instead of the number of turns.", _A),
    ("hl-hcf-two-only", "Finds the highest common factor of only two of the three lengths.", _A),
    ("hl-side-not-area", "Gives the side of the square where the number of tiles was asked.", _A),
    ("hl-tiles-one-way", "Counts the tiles along one side of the square only.", _A),
    ("hl-ignores-remainders", "Finds the HCF of the numbers without taking the remainders off first.", _A),
    ("hl-adds-remainders", "Adds the remainders to the numbers instead of subtracting them.", _A),
    (
        "hl-remainder-lcm",
        "Finds the lowest common multiple of the reduced numbers instead of the highest common factor.",
        _A,
    ),
    ("hl-forgets-remainder", "Gives the lowest common multiple and forgets to add the remainder.", _A),
    ("hl-subtracts-remainder", "Subtracts the remainder from the lowest common multiple instead of adding it.", _A),
    ("hl-forgets-start", "Leaves out the meeting at the start when counting how many times the events coincide.", _A),
    ("hl-counts-extra", "Counts one meeting too many.", _A),
    ("hl-time-not-meetings", "Gives the time of the first meeting where the number of meetings was asked.", _A),
    ("hl-product-for-other", "Multiplies the HCF and the LCM and does not divide by the known number.", _A),
    ("hl-lcm-over-hcf", "Divides the LCM by the HCF to find the other number.", _A),
    ("hl-other-divides-lcm", "Divides the LCM by the known number to find the other number.", _A),
]

_COPRIME_PAIRS = [(u, v) for u in range(2, 10) for v in range(u + 1, 11) if gcd(u, v) == 1]


def _pair(rng, g_lo, g_hi, u_max=10):
    g = rng.randint(g_lo, g_hi)
    u, v = rng.choice([(x, y) for x, y in _COPRIME_PAIRS if y <= u_max])
    if rng.random() < 0.5:
        u, v = v, u
    return g, u, v, g * u, g * v


def _cut_story(rng, a, b):
    kind = rng.choice(("ribbon", "pens", "teams", "path"))
    if kind == "ribbon":
        return (
            f"A florist has two ribbons, one {a} cm long and one {b} cm long. She cuts both into pieces of the same length with nothing left over.",
            "What is the greatest possible length of each piece, in centimetres?",
            "ribbon",
        )
    if kind == "pens":
        return (
            f"A teacher has {a} pens and {b} pencils. She packs them into identical bags so that every bag has the same number of pens and the same number of pencils, with none left over.",
            "What is the greatest number of bags she can make?",
            "pens",
        )
    if kind == "teams":
        return (
            f"A club has {a} boys and {b} girls. They are split into teams with the same number of boys and the same number of girls in every team, and nobody is left out.",
            "What is the greatest number of teams that can be made?",
            "teams",
        )
    return (
        f"A path is {a} m long and a second path is {b} m long. Both are to be laid with paving strips of one length, with no strip cut and none left over.",
        "What is the greatest possible length of a strip, in metres?",
        "path",
    )


_EVENTS = (
    (
        "Bus A leaves the depot every {a} minutes and Bus B leaves every {b} minutes. They leave together at 7:00 am.",
        "After how many minutes will they next leave together?",
    ),
    (
        "A red light flashes every {a} seconds and a blue light flashes every {b} seconds. They flash together at the start.",
        "After how many seconds will they next flash together?",
    ),
    (
        "A school bell rings every {a} minutes and the library bell rings every {b} minutes. Both ring at 9:00 am.",
        "After how many minutes will they next ring together?",
    ),
    (
        "Two joggers start together at the same point on a track. One takes {a} seconds for a lap and the other takes {b} seconds.",
        "After how many seconds will they next be together at the start?",
    ),
)


def _hl_hcf(rng):
    g, u, v, a, b = _pair(rng, 2, 14, 7)
    intro, ask, kind = _cut_story(rng, a, b)
    check(gcd(a, b) == g, "hcf")
    d = WHOLE.dist(
        g,
        [
            (lcm(a, b), "hl-lcm-for-hcf"),
            (min(a, b), "hl-smaller"),
            (abs(a - b), "hl-difference"),
            (a + b, "hl-sum"),
            (a * b, "hl-product"),
        ],
    )
    enough(d, 3)
    text = f"{intro} {ask}"
    working = [
        f"The HCF of {a} and {b} is {g}. Nothing is left over only when the length or number divides both exactly."
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(g), answer_kind="number")


def _hl_lcm(rng):
    g, u, v, a, b = _pair(rng, 1, 4, 9)
    L = lcm(a, b)
    check(L == g * u * v, "lcm")
    intro, ask = rng.choice(_EVENTS)
    d = WHOLE.dist(
        L,
        [
            (a * b, "hl-product") if a * b != L else None,
            (gcd(a, b), "hl-hcf-for-lcm") if gcd(a, b) > 1 else None,
            (a + b, "hl-sum"),
            (max(a, b), "hl-larger"),
            (L + min(a, b), "hl-sum"),
        ],
    )
    enough(d, 3)
    text = f"{intro.format(a=a, b=b)} {ask}"
    working = [f"The LCM of {a} and {b} is {L}."]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(L), answer_kind="number")


def _hl_share(rng):
    g, u, v, a, b = _pair(rng, 2, 8, 7)
    ask_pens = rng.random() < 0.5
    size = a // g if ask_pens else b // g
    first, second = ("pens", "pencils") if ask_pens else ("pencils", "pens")
    n_first, n_second = (a, b) if ask_pens else (b, a)
    d = WHOLE.dist(
        size,
        [
            (g, "hl-hcf-not-divided"),
            (n_first, "hl-larger" if n_first > n_second else "hl-smaller"),
            (n_second // g, "hl-size-for-shares"),
            (lcm(a, b) // n_first, "hl-shares-for-size"),
            (g + 1, "hl-hcf-not-divided"),
        ],
    )
    enough(d, 3)
    text = (
        f"A shop has {a} pens and {b} pencils. They are packed into the greatest possible number of identical gift bags, with the same number of pens and the "
        f"same number of pencils in every bag and none left over. How many {first} go in each bag?"
    )
    working = [
        f"The greatest number of bags is the HCF of {a} and {b}, which is {g}.",
        f"{first.capitalize()} in each bag = {n_first} / {g} = {size}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(size), answer_kind="number")


# -- medium -------------------------------------------------------------------------------------------------------------


def _hl_tiles(rng):
    g, u, v, a, b = _pair(rng, 5, 30, 9)
    if u == v:
        raise TooFewDistractors("equal")
    tiles = u * v
    d = WHOLE.dist(
        tiles,
        [
            (u + v, "hl-tiles-sum"),
            (u, "hl-one-dimension"),
            (v, "hl-one-dimension"),
            (g, "hl-size-for-shares"),
            (a * b // (g + 1), "hl-product"),
        ],
    )
    enough(d, 3)
    text = (
        f"A rectangular floor is {a} cm long and {b} cm wide. It is to be covered with identical square tiles, with no tile cut. "
        f"The tiles are as large as possible. How many tiles are needed?"
    )
    working = [
        f"The side of the largest tile is the HCF of {a} and {b}, which is {g} cm.",
        f"Tiles = ({a} / {g}) × ({b} / {g}) = {u} × {v} = {tiles}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(tiles), answer_kind="number")


def _hl_trips(rng):
    g, u, v, a, b = _pair(rng, 2, 6, 9)
    L = lcm(a, b)
    ask_a = rng.random() < 0.5
    trips = L // a if ask_a else L // b
    who = ("Bus A", a) if ask_a else ("Bus B", b)
    other = L // b if ask_a else L // a
    d = WHOLE.dist(
        trips,
        [
            (L, "hl-lcm-not-divided"),
            (other, "hl-trips-other-bus"),
            (a * b // who[1], "hl-product"),
            (trips + 1, "hl-counts-extra"),
            (gcd(a, b), "hl-hcf-for-lcm"),
        ],
    )
    enough(d, 3)
    text = (
        f"Bus A leaves the station every {a} minutes and Bus B leaves every {b} minutes. They leave together at 8:00 am. "
        f"How many trips does {who[0]} make from 8:00 am until the next time they leave together? (Count the trip that leaves together, but not the one at 8:00 am.)"
    )
    working = [
        f"They next leave together after the LCM of {a} and {b}, which is {L} minutes.",
        f"{who[0]}: {L} / {who[1]} = {trips} trips.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(trips), answer_kind="number")


def _hl_three_lcm(rng):
    for _ in attempts(200, "three intervals"):
        a, b, c = (rng.choice((4, 5, 6, 8, 9, 10, 12, 15, 18, 20)) for _ in range(3))
        if len({a, b, c}) == 3:
            L = lcm(a, b, c)
            if L not in (lcm(a, b), lcm(a, c), lcm(b, c)) and L <= 400:
                break
    else:
        raise TooFewDistractors("no triple")
    d = WHOLE.dist(
        L,
        [
            (lcm(a, b), "hl-two-numbers-only"),
            (lcm(b, c), "hl-two-numbers-only"),
            (a * b * c, "hl-three-product"),
            (gcd(gcd(a, b), c) or 1, "hl-hcf-for-lcm"),
            (a + b + c, "hl-sum"),
        ],
    )
    enough(d, 3)
    text = (
        f"Three lighthouses flash their lamps every {a} seconds, every {b} seconds and every {c} seconds. They all flash together at midnight. "
        f"After how many seconds do they next all flash together?"
    )
    working = [f"The LCM of {a}, {b} and {c} is {L}."]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(L), answer_kind="number")


def _hl_gears(rng):
    g, u, v, a, b = _pair(rng, 2, 6, 9)
    L = lcm(a, b)
    small, large = min(a, b), max(a, b)
    turns = L // small
    d = WHOLE.dist(
        turns,
        [
            (L // large, "hl-turns-big-gear"),
            (L, "hl-turns-lcm"),
            (large // gcd(a, b), "hl-hcf-for-lcm"),
            (large, "hl-larger"),
            (turns + 1, "hl-counts-extra"),
        ],
    )
    enough(d, 3)
    text = (
        f"Two meshed gears have {small} teeth and {large} teeth. A mark is made on a tooth of each gear where they meet. "
        f"How many complete turns must the smaller gear make before the two marked teeth meet again?"
    )
    working = [
        f"The marked teeth meet again after LCM({small}, {large}) = {L} teeth have passed.",
        f"Turns of the smaller gear = {L} / {small} = {turns}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(turns), answer_kind="number")


def _hl_three_hcf(rng):
    g = rng.randint(3, 12)
    ms = rng.choice([(2, 3, 5), (3, 4, 6), (4, 5, 7), (2, 5, 7), (3, 5, 8), (4, 6, 9), (5, 6, 8)])
    a, b, c = (g * m for m in ms)
    h = gcd(gcd(a, b), c)
    check(h == g * gcd(gcd(ms[0], ms[1]), ms[2]), "three-hcf")
    if h < 3:
        raise TooFewDistractors("small")
    d = WHOLE.dist(
        h,
        [
            (gcd(a, b), "hl-hcf-two-only"),
            (gcd(b, c), "hl-hcf-two-only"),
            (min(a, b, c), "hl-smaller"),
            (lcm(a, b, c), "hl-lcm-for-hcf"),
            (h + 1, "hl-hcf-two-only"),
        ],
    )
    enough(d, 3)
    text = (
        f"A builder has three timber lengths of {a} cm, {b} cm and {c} cm. All three are cut into pieces of one length with nothing left over. "
        f"What is the greatest possible length of a piece, in centimetres?"
    )
    working = [f"The HCF of {a}, {b} and {c} is {h}."]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(h), answer_kind="number")


# -- hard ---------------------------------------------------------------------------------------------------------------


def _hl_square_of_rectangles(rng):
    g, u, v, a, b = _pair(rng, 2, 6, 7)
    L = lcm(a, b)
    tiles = (L // a) * (L // b)
    d = WHOLE.dist(
        tiles,
        [
            (L, "hl-side-not-area"),
            (L // a, "hl-tiles-one-way"),
            (L // b, "hl-tiles-one-way"),
            ((L // a) + (L // b), "hl-tiles-sum"),
            (gcd(a, b), "hl-hcf-for-lcm"),
            (L * L // (a * b) + 1, "hl-counts-extra"),
        ],
    )
    enough(d, 3)
    text = (
        f"Rectangular tiles measuring {a} cm by {b} cm are laid, all the same way round, to make the smallest possible square with no gaps and no overlaps. "
        f"How many tiles are used?"
    )
    working = [
        f"The side of the smallest square is the LCM of {a} and {b}, which is {L} cm.",
        f"Tiles along one side: {L} / {a} = {L // a} and {L} / {b} = {L // b}; tiles = {L // a} × {L // b} = {tiles}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(tiles), answer_kind="number")


def _hl_remainders(rng):
    for _ in attempts(300, "a remainder draw"):
        g = rng.randint(7, 19)
        u, v = rng.choice([(x, y) for x, y in _COPRIME_PAIRS if y <= 9])
        r1, r2 = rng.randint(1, g - 1), rng.randint(1, g - 1)
        if r1 != r2 and max(r1, r2) < g:
            break
    a, b = g * u + r1, g * v + r2
    check(gcd(a - r1, b - r2) == g and g > max(r1, r2), "remainder hcf")
    d = WHOLE.dist(
        g,
        [
            (gcd(a, b), "hl-ignores-remainders"),
            (gcd(a + r1, b + r2), "hl-adds-remainders"),
            (lcm(a - r1, b - r2), "hl-remainder-lcm"),
            (abs((a - r1) - (b - r2)), "hl-difference"),
            (g + 1, "hl-ignores-remainders"),
        ],
    )
    enough(d, 3)
    text = (
        f"When {a} is divided by a certain whole number n, the remainder is {r1}. When {b} is divided by the same number n, the remainder is {r2}. "
        f"What is the greatest possible value of n?"
    )
    working = [
        f"n divides {a} - {r1} = {a - r1} and {b} - {r2} = {b - r2}.",
        f"The HCF of {a - r1} and {b - r2} is {g}, which is greater than both remainders, so n = {g}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(g), answer_kind="number")


def _hl_smallest_remainder(rng):
    for _ in attempts(200, "a triple"):
        xs = [rng.choice((4, 5, 6, 8, 9, 10, 12)) for _ in range(3)]
        if len(set(xs)) == 3 and lcm(*xs) <= 200:
            break
    else:
        raise TooFewDistractors("no triple")
    r = rng.randint(1, min(xs) - 1)
    L = lcm(*xs)
    answer = L + r
    d = WHOLE.dist(
        answer,
        [
            (L, "hl-forgets-remainder"),
            (L - r, "hl-subtracts-remainder"),
            (xs[0] * xs[1] * xs[2] + r, "hl-three-product"),
            (lcm(xs[0], xs[1]) + r, "hl-two-numbers-only"),
            (L + r * 2, "hl-counts-extra"),
        ],
    )
    enough(d, 3)
    text = f"What is the smallest whole number greater than {r} that leaves a remainder of {r} when divided by {xs[0]}, by {xs[1]} and by {xs[2]}?"
    working = [
        f"The number is a multiple of LCM({xs[0]}, {xs[1]}, {xs[2]}) = {L}, plus {r}.",
        f"Smallest = {L} + {r} = {answer}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(answer), answer_kind="number")


def _meetings(a: int, b: int, hours: int) -> int:
    return hours * 60 // lcm(a, b) + 1


def _hl_meetings(rng):
    for _ in attempts(300, "a meetings draw"):
        a, b = rng.choice(
            [
                (6, 8),
                (10, 15),
                (12, 20),
                (15, 18),
                (20, 30),
                (9, 12),
                (14, 21),
                (16, 24),
                (18, 24),
                (25, 40),
                (12, 18),
                (30, 45),
            ]
        )
        hours = rng.randint(2, 12)
        L = lcm(a, b)
        if 60 * hours // L >= 2 and (60 * hours) % L != 0:
            break
    else:
        raise TooFewDistractors("no draw")
    n = _meetings(a, b, hours)
    d = WHOLE.dist(
        n,
        [
            (n - 1, "hl-forgets-start"),
            (n + 1, "hl-counts-extra"),
            (L, "hl-time-not-meetings"),
            (hours * 60 // a + hours * 60 // b, "hl-sum"),
        ],
    )
    enough(d, 3)
    text = (
        f"Two machines each sound an alarm: the first every {a} minutes and the second every {b} minutes. They sound together at noon. "
        f"How many times do they sound together from noon until {noon_plus(hours)}, counting both noon and any time that falls exactly on the end of the period?"
    )
    working = [
        f"They sound together every LCM({a}, {b}) = {L} minutes.",
        f"In {hours * 60} minutes: {hours * 60} / {L} gives {hours * 60 // L} whole intervals, so {hours * 60 // L} + 1 = {n} times.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(n), answer_kind="number")


_CONTEXTS = (
    ("runner", "Runner", "circular track", "run"),
    ("cyclist", "Cyclist", "circular velodrome", "ridden"),
    ("go-kart driver", "Driver", "racing circuit", "driven"),
)


def noon_plus(hours: int) -> str:
    return f"{hours}:00 pm" if hours < 12 else "midnight"


def _hl_other_number(rng):
    g, u, v, a, b = _pair(rng, 2, 9, 9)
    h, L = g, g * u * v
    known, other = (a, b) if rng.random() < 0.5 else (b, a)
    d = WHOLE.dist(
        other,
        [
            (h * L, "hl-product-for-other"),
            (L // h, "hl-lcm-over-hcf"),
            (L // known, "hl-other-divides-lcm"),
            (h + L - known, "hl-sum"),
            (L - known, "hl-difference"),
        ],
    )
    enough(d, 3)
    text = (
        f"Two whole numbers have a highest common factor of {h} and a lowest common multiple of {L}. One of the numbers is {known}. "
        f"What is the other number?"
    )
    working = [
        f"For two numbers, HCF × LCM = product of the numbers: {h} × {L} = {h * L}.",
        f"Other number = {h * L} / {known} = {other}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(other), answer_kind="number")


_EASY = (_hl_hcf, _hl_lcm, _hl_share)
_MEDIUM = (_hl_tiles, _hl_trips, _hl_three_lcm, _hl_gears, _hl_three_hcf)
_HARD = (_hl_square_of_rectangles, _hl_remainders, _hl_smallest_remainder, _hl_meetings, _hl_other_number)


@generator(
    TOPIC,
    level=8,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w3_hcf_lcm_words(rng, tier="easy"):
    """HCF and LCM in context; the tiers are described in the module docstring."""
    pool = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


@generator(EXAM_TOPIC, level=8, presets={"exam": dict(tier="exam")})
def exd_x_w3_hcf_lcm_words(rng, tier="exam"):
    """Exam chain: two runners; the time until they next meet, the laps run by each, the meetings in a longer period."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")

    def build(rng):
        noun, who, venue, done = rng.choice(_CONTEXTS)
        for _ in attempts(300, "a runners draw"):
            g, u, v, a, b = _pair(rng, 2, 6, 9)
            if u == v or lcm(a, b) > 600:
                continue
            L = lcm(a, b)
            minutes = rng.randint(10, 40)
            if (60 * minutes) // L >= 2 and (60 * minutes) % L != 0:
                break
        else:
            raise TooFewDistractors("no draw")
        laps_a, laps_b = L // a, L // b
        n = _meetings_seconds(a, b, minutes)
        a_d = WHOLE.dist(
            L,
            [(a * b, "hl-product"), (gcd(a, b), "hl-hcf-for-lcm"), (a + b, "hl-sum"), (max(a, b), "hl-larger")],
        )
        b_d = WHOLE.dist(
            laps_a,
            [
                (L, "hl-lcm-not-divided"),
                (laps_b, "hl-trips-other-bus"),
                (laps_a + 1, "hl-counts-extra"),
                (a * b // a, "hl-product"),
            ],
        )
        c_d = WHOLE.dist(
            laps_b,
            [
                (L, "hl-lcm-not-divided"),
                (laps_a, "hl-trips-other-bus"),
                (laps_b + 1, "hl-counts-extra"),
                (a * b // b, "hl-product"),
            ],
        )
        d_d = WHOLE.dist(
            n,
            [
                (n - 1, "hl-forgets-start"),
                (n + 1, "hl-counts-extra"),
                (L, "hl-time-not-meetings"),
                (60 * minutes // a, "hl-hcf-for-lcm"),
            ],
        )
        for x in (a_d, b_d, c_d, d_d):
            enough(x, 3)
        stem = (
            f"Two {noun}s start together at the start line of a {venue}. {who} A takes {a} seconds for each lap and {who} B takes {b} seconds, "
            f"and each keeps a steady pace."
        )
        return dict(
            text=stem,
            parts=[
                part(
                    "After how many seconds are they next together at the start line?",
                    "number",
                    WHOLE.fields(L),
                    2,
                    [f"LCM({a}, {b}) = {L} seconds."],
                    a_d,
                    sensible_range=(0, None),
                ),
                part(
                    f"How many laps has {who} A {done} by then?",
                    "number",
                    WHOLE.fields(laps_a),
                    1,
                    [f"{L} / {a} = {laps_a} laps."],
                    b_d,
                    follow_through=f"part_a/{a}",
                    sensible_range=(0, None),
                ),
                part(
                    f"How many laps has {who} B {done} by then?",
                    "number",
                    WHOLE.fields(laps_b),
                    1,
                    [f"{L} / {b} = {laps_b} laps."],
                    c_d,
                    follow_through=f"part_a/{b}",
                    sensible_range=(0, None),
                ),
                part(
                    f"If they keep going for {minutes} minutes, how many times are they together at the start line, counting the start?",
                    "number",
                    WHOLE.fields(n),
                    2,
                    [
                        f"In {60 * minutes} seconds: {60 * minutes} / {L} gives {60 * minutes // L} whole intervals, so {60 * minutes // L} + 1 = {n} times."
                    ],
                    d_d,
                    sensible_range=(0, None),
                ),
            ],
        )

    return retry_draw(rng, build)


def _meetings_seconds(a: int, b: int, minutes: int) -> int:
    return minutes * 60 // lcm(a, b) + 1
