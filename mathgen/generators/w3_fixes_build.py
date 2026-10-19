"""Replacement easy tiers and extra tiers for skills whose existing questions were thin or guessable.

Topics:

* ``w3_percentage_change``   percentage-change: easy (a percentage increase with a spread of whole-number answers),
                             medium (an increase or a decrease, one decimal place), hard (a net change over two steps,
                             the fall that undoes a rise, the change in a rate itself).
* ``w3_classify_easy``       classify-data easy: variables in many settings, either classified into three types or picked
                             out of four, so the answer cannot be guessed from a three-line pool.
* ``w3_arith_recognise``     arithmetic-sequences: pick the arithmetic sequence out of four, judge a sequence whose first
                             pair of terms is misleading, find the value that makes three terms arithmetic.
* ``w3_ests_lsts``           senior-use-ests-and-lsts-to-locate-the-critical: four-option questions on a small activity
                             network (the critical activity from a table of ESTs and LSTs, the critical path, a latest
                             start time) instead of a yes/no.
* ``w3_bearings_general``    bearings (medium and hard): legs on bearings that are not 90 degrees apart and not
                             cardinal; the distance north or east, the distance and the bearing home, a leg that ends
                             due east of the start.

Every key is built from exact arithmetic or an exhaustive search (trigonometric keys are rounded and refuse a rounding
near-tie), and every tagged distractor is a believable slip at least two units of the last place from the key.
"""

from __future__ import annotations

from fractions import Fraction
from math import atan2, cos, degrees, hypot, radians, sin, tan

from ..core import VerificationError, attempts, check, generator
from ._r3j_kit import choice_question
from ._w3_kit import DP1, WHOLE, pct_text
from ._w6a_kit import TooFewDistractors, enough
from .r3j_classify_build import CONTINUOUS, DISCRETE, NOMINAL, ORDINAL, POOL

_PC = "percentage-change"
_CD = "classify-data"
_AS = "arithmetic-sequences"
_EL = "senior-use-ests-and-lsts-to-locate-the-critical"
_BR = "senior-solve-problems-involving-bearings"

MISCONCEPTIONS = [
    # percentage change
    ("w3pc-over-final", "Divides the change by the final value instead of the original value.", _PC),
    ("w3pc-change-as-percent", "Gives the amount of the change as the percentage.", _PC),
    ("w3pc-ratio-as-change", "Gives the final value as a percentage of the original and does not subtract 100.", _PC),
    ("w3pc-original-as-final", "Gives the original value as a percentage of the final value.", _PC),
    ("w3pc-points-as-percent", "Gives the rise in percentage points as the percentage change.", _PC),
    (
        "w3pc-adds-steps",
        "Adds the increase and takes off the decrease instead of multiplying the two multipliers.",
        _PC,
    ),
    ("w3pc-sums-percentages", "Adds the two percentages whatever their direction.", _PC),
    ("w3pc-reverse-same-percent", "Takes the fall that undoes a rise to be the same percentage as the rise.", _PC),
    (
        "w3pc-second-step-on-original",
        "Gives the fall in the second step as a percentage of the original price, not the overall change.",
        _PC,
    ),
    ("w3pc-wrong-direction", "Gives a percentage increase where a decrease was asked, or the reverse.", _PC),
    # classify data
    ("w3cd-category-as-number", "Treats a variable recorded in words or labels as numerical.", _CD),
    # arithmetic sequences
    ("w3as-geometric-as-arithmetic", "Takes a constant ratio for a constant difference.", _AS),
    ("w3as-one-pair", "Judges the sequence from its first two terms only.", _AS),
    ("w3as-first-term", "Gives the first term as the common difference.", _AS),
    ("w3as-growing-gaps", "Takes terms that grow by a growing amount for an arithmetic sequence.", _AS),
    ("w3as-repeating-gaps", "Takes gaps that repeat in a pattern for a constant difference.", _AS),
    ("w3as-sign", "Gives the common difference with the wrong sign.", _AS),
    ("w3as-wrong-break", "Names the wrong place where the difference changes.", _AS),
    ("w3as-halving-forgotten", "Stops at 2k = n and gives 2k as the value of k.", _AS),
    ("w3as-arithmetic-slip", "Makes an arithmetic slip when solving the equation, so the answer is one out.", _AS),
    # critical path
    ("w3el-smallest-float", "Chooses the activity with the smallest float that is not zero.", _EL),
    ("w3el-largest-est", "Chooses the activity with the latest earliest start time.", _EL),
    ("w3el-largest-lst", "Chooses the activity with the latest latest start time.", _EL),
    ("w3el-first-listed", "Chooses the first activity in the table.", _EL),
    ("w3el-longest-activity", "Follows the longest single activities instead of the longest path.", _EL),
    ("w3el-most-activities", "Takes the path with the most activities for the critical path.", _EL),
    ("w3el-one-swap", "Swaps one critical activity for a non-critical one.", _EL),
    ("w3el-est-for-lst", "Gives the earliest start time where the latest start time was asked.", _EL),
    (
        "w3el-ignores-successors",
        "Finds the latest start time from the project duration alone and ignores the activities that follow.",
        _EL,
    ),
    ("w3el-forgets-duration", "Gives the latest start time of the next activity without taking off the duration.", _EL),
    ("w3el-finish-for-start", "Gives the earliest finish time where the latest start time was asked.", _EL),
    ("w3el-float-for-lst", "Gives the float where the latest start time was asked.", _EL),
    # bearings
    ("w3br-sin-cos-swapped", "Swaps sine and cosine when finding a distance north or east.", _BR),
    ("w3br-adds-legs", "Adds the two leg lengths instead of combining the north and east distances.", _BR),
    ("w3br-one-leg", "Uses the first leg only.", _BR),
    ("w3br-no-reverse", "Gives the bearing from the start to the finish where the bearing home was asked.", _BR),
    ("w3br-tan-swapped", "Uses the north distance over the east distance for the angle.", _BR),
    ("w3br-raw-angle", "Gives the angle in the triangle and does not turn it into a three-figure bearing.", _BR),
    ("w3br-angle-from-east", "Measures the angle from east instead of from north.", _BR),
    ("w3br-sign-of-component", "Adds a distance south or west as if it were north or east.", _BR),
    (
        "w3br-uses-sin-not-cos",
        "Uses the sine of the bearing where the cosine is needed to cancel the north distance.",
        _BR,
    ),
    ("w3br-distance-as-component", "Gives the distance walked as the distance north or east.", _BR),
    ("w3br-tan-used", "Uses the tangent where the sine or cosine is needed.", _BR),
    ("w3br-divides", "Divides by the sine or cosine where it should multiply.", _BR),
    ("w3br-adds-components", "Adds the north and east distances instead of using Pythagoras.", _BR),
    ("w3br-uses-total-not-component", "Uses the whole first leg as the distance to cancel.", _BR),
]


def _draw(rng, pool):
    for _ in attempts(200, "a usable draw"):
        try:
            return rng.choice(pool)(rng)
        except (TooFewDistractors, VerificationError):
            continue
    raise AssertionError  # pragma: no cover


# -- percentage change ------------------------------------------------------------------------------------------------

_STORIES = (
    ("The price of a jacket", "$"),
    ("The weekly rent of a flat", "$"),
    ("The price of a concert ticket", "$"),
    ("The cost of a bus pass", "$"),
    ("The number of students in a choir", ""),
    ("The number of visitors to a museum each day", ""),
    ("The membership of a gym", ""),
    ("The population of a small town", ""),
    ("The number of followers of a channel", ""),
)


def _amount(unit, value):
    return f"${value}" if unit == "$" else f"{value}"


def _pc_easy(rng):
    k = rng.randint(2, 30)
    orig = 20 * k
    pct = rng.choice(range(5, 101, 5))
    final = orig + orig * pct // 100
    who, unit = rng.choice(_STORIES)
    d = WHOLE.dist(
        pct,
        [
            (Fraction(final - orig, final) * 100, "w3pc-over-final"),
            (final - orig, "w3pc-change-as-percent"),
            (Fraction(final, orig) * 100, "w3pc-ratio-as-change"),
            (Fraction(orig, final) * 100, "w3pc-original-as-final"),
        ],
    )
    enough(d, 3)
    text = f"{who} rose from {_amount(unit, orig)} to {_amount(unit, final)}. What was the percentage increase?"
    working = [
        f"Change = {final} − {orig} = {final - orig}.",
        f"Percentage increase = change ÷ original × 100 = {final - orig} ÷ {orig} × 100 = {pct}%.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(pct), answer_kind="number")


def _pc_medium(rng):
    orig = rng.randint(20, 400)
    ratio = Fraction(rng.randint(55, 180), 100)
    final = round(orig * ratio)
    check(final != orig and final > 0, "no change")
    up = final > orig
    change = abs(final - orig)
    pct = Fraction(change, orig) * 100
    who, unit = rng.choice(_STORIES)
    word = "increase" if up else "decrease"
    verb = "rose" if up else "fell"
    d = DP1.dist(
        pct,
        [
            (Fraction(change, final) * 100, "w3pc-over-final"),
            (Fraction(final, orig) * 100, "w3pc-ratio-as-change"),
            (change, "w3pc-change-as-percent"),
            (Fraction(orig, final) * 100 - 100 if up else Fraction(final, orig) * 100 - 100, "w3pc-wrong-direction"),
        ],
    )
    enough(d, 3)
    text = (
        f"{who} {verb} from {_amount(unit, orig)} to {_amount(unit, final)}. What was the percentage {word}? "
        f"Give your answer to 1 decimal place."
    )
    working = [f"Change = {change}.", f"Percentage {word} = {change} ÷ {orig} × 100 = {DP1.text(pct)}%."]
    return dict(text=text, working=working, distractors=d, **DP1.fields(pct), answer_kind="number")


def _pc_net(rng):
    a, b = rng.sample((10, 20, 30, 40, 50), 2)
    up_first = rng.random() < 0.5
    first, second = (a, b) if up_first else (b, a)
    # first step an increase of `first` percent, second a decrease of `second` percent
    net = (100 + first) * (100 - second) / Fraction(100) - 100
    check(net != 0, "no net change")
    word = "increase" if net > 0 else "decrease"
    d = DP1.dist(
        abs(net),
        [
            (abs(first - second), "w3pc-adds-steps"),
            (first + second, "w3pc-sums-percentages"),
            (Fraction(100 + first, 100) * second, "w3pc-second-step-on-original"),
        ],
    )
    enough(d, 3)
    text = (
        f"A price is increased by {first}% and the new price is then decreased by {second}%. "
        f"What is the overall percentage change in the price, as a percentage {word}? Give your answer to 1 decimal place."
    )
    working = [
        f"Multiplier = {Fraction(100 + first, 100)} × {Fraction(100 - second, 100)} = {Fraction((100 + first) * (100 - second), 10000)}.",
        f"Overall percentage {word} = {DP1.text(abs(net))}%.",
    ]
    return dict(text=text, working=working, distractors=d, **DP1.fields(abs(net)), answer_kind="number")


def _pc_undo(rng):
    a = rng.choice((10, 20, 25, 30, 40, 50, 60, 75, 100))
    orig = 100
    final = Fraction(100 + a)
    back = (final - orig) / final * 100
    d = DP1.dist(
        back,
        [
            (a, "w3pc-reverse-same-percent"),
            (Fraction(a, 2), "w3pc-reverse-same-percent"),
            (Fraction(100, 100 + a) * 100, "w3pc-original-as-final"),
        ],
    )
    enough(d, 3)
    text = (
        f"A price rises by {a}%. By what percentage must the new price then fall to return to the original price? "
        f"Give your answer to 1 decimal place."
    )
    working = [
        f"Take the original as 100; the new price is {100 + a}.",
        f"Fall needed = {a} ÷ {100 + a} × 100 = {DP1.text(back)}%.",
    ]
    return dict(text=text, working=working, distractors=d, **DP1.fields(back), answer_kind="number")


def _pc_rate(rng):
    r1 = rng.choice((2, 2.5, 3, 4, 5, 6, 8))
    r1 = Fraction(str(r1))
    r2 = r1 + rng.choice((Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2)))
    change = (r2 - r1) / r1 * 100
    d = DP1.dist(
        change,
        [
            (r2 - r1, "w3pc-points-as-percent"),
            ((r2 - r1) / r2 * 100, "w3pc-over-final"),
            (r2 / r1 * 100, "w3pc-ratio-as-change"),
            (r1 / r2 * 100, "w3pc-original-as-final"),
        ],
    )
    enough(d, 3)
    text = (
        f"An interest rate rises from {pct_text(r1)}% to {pct_text(r2)}%. By what percentage has the rate itself increased? "
        f"Give your answer to 1 decimal place."
    )
    working = [
        f"Rise = {pct_text(r2)} − {pct_text(r1)} = {pct_text(r2 - r1)} percentage points.",
        f"Percentage increase = {pct_text(r2 - r1)} ÷ {pct_text(r1)} × 100 = {DP1.text(change)}%.",
    ]
    return dict(text=text, working=working, distractors=d, **DP1.fields(change), answer_kind="number")


@generator(
    "w3_percentage_change",
    level=8,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w3_percentage_change(rng, tier="easy"):
    """Percentage change from an original and a final value; the tiers are described in the module docstring."""
    pool = {"easy": (_pc_easy,), "medium": (_pc_medium,), "hard": (_pc_net, _pc_undo, _pc_rate)}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return _draw(rng, pool)


# -- classify data ----------------------------------------------------------------------------------------------------

_CATEGORICAL = "Categorical"
_THREE = (_CATEGORICAL, DISCRETE, CONTINUOUS)
_TO_THREE = {NOMINAL: _CATEGORICAL, ORDINAL: _CATEGORICAL, DISCRETE: DISCRETE, CONTINUOUS: CONTINUOUS}
_PLAIN = [(phrase, _TO_THREE[cls]) for phrase, cls, twist in POOL if not twist]
_SLIP = {
    (_CATEGORICAL, DISCRETE): "w3cd-category-as-number",
    (_CATEGORICAL, CONTINUOUS): "w3cd-category-as-number",
    (DISCRETE, _CATEGORICAL): "r3j-classify-number-as-category",
    (DISCRETE, CONTINUOUS): "r3j-classify-count-as-continuous",
    (CONTINUOUS, _CATEGORICAL): "r3j-classify-number-as-category",
    (CONTINUOUS, DISCRETE): "r3j-classify-measure-as-discrete",
}
_WHY = {
    _CATEGORICAL: "its values are names or categories",
    DISCRETE: "its values are counted, so they are separate whole-number steps",
    CONTINUOUS: "its values are measured, so any value on a scale is possible",
}


def _cap(text):
    return text[0].upper() + text[1:]


def _classify_one(rng):
    phrase, cls = rng.choice(_PLAIN)
    lead = rng.choice(
        (
            "A survey records {p}. How is this variable best classified?",
            "Data are collected on {p}. What type of data are they?",
            "In a study, the variable is {p}. Classify the data.",
            "Classify the data: {p}.",
        )
    )
    slips = [(c, _SLIP[(cls, c)]) for c in _THREE if c != cls]
    return choice_question(lead.format(p=phrase), cls, [f"The data are {cls.lower()}: {_WHY[cls]}."], slips)


def _classify_which(rng):
    asked = rng.choice(_THREE)
    key_phrase = rng.choice([p for p, c in _PLAIN if c == asked])
    others = rng.sample([(p, c) for p, c in _PLAIN if c != asked], 3)
    check(len({c for _, c in others}) >= 2, "the wrong options should come from both other types")
    text = f"Which of these is an example of {asked.lower()} data?"
    slips = [(_cap(p), _SLIP[(c, asked)]) for p, c in others]
    return choice_question(text, _cap(key_phrase), [f"{_cap(key_phrase)} is {asked.lower()}: {_WHY[asked]}."], slips)


@generator("w3_classify_easy", level=7, presets={"easy": dict(tier="easy")})
def w3_classify_easy(rng, tier="easy"):
    """Classify a variable as categorical, discrete numerical or continuous numerical, or pick one of a stated type."""
    if tier != "easy":
        raise ValueError(f"unknown tier {tier!r}")
    return _draw(rng, (_classify_one, _classify_which))


# -- arithmetic sequences ---------------------------------------------------------------------------------------------


def _seq_text(terms):
    return ", ".join(str(t) for t in terms)


def _arith(rng, n=5):
    a = rng.randint(-12, 30)
    d = rng.choice([x for x in range(-9, 10) if x not in (0,)])
    return [a + d * i for i in range(n)], d


def _is_arithmetic(terms):
    return len({terms[i + 1] - terms[i] for i in range(len(terms) - 1)}) == 1


def _geometric(rng, n=5):
    a = rng.randint(1, 6)
    r = rng.choice((2, 3, -2, 5))
    return [a * r**i for i in range(n)]


def _growing(rng, n=5):
    a = rng.randint(0, 9)
    step = rng.randint(1, 3)
    out = [a]
    gap = step
    for _ in range(n - 1):
        out.append(out[-1] + gap)
        gap += rng.choice((1, 2))
    return out


def _repeating(rng, n=5):
    a = rng.randint(0, 20)
    g1, g2 = rng.sample(range(1, 8), 2)
    out = [a]
    for i in range(n - 1):
        out.append(out[-1] + (g1 if i % 2 == 0 else g2))
    return out


def _one_pair(rng, n=5):
    """The first difference repeats once, then changes."""
    terms, d = _arith(rng, 3)
    nxt = terms[-1] + d + rng.choice((-3, -2, -1, 1, 2, 3))
    out = terms + [nxt]
    while len(out) < n:
        out.append(out[-1] + (out[-1] - out[-2]))
    return out


def _which_arithmetic(rng):
    key, d = _arith(rng, 5)
    decoys = [
        (_geometric(rng), "w3as-geometric-as-arithmetic"),
        (_growing(rng), "w3as-growing-gaps"),
        (_repeating(rng), "w3as-repeating-gaps"),
    ]
    check(not any(_is_arithmetic(t) for t, _ in decoys), "a decoy is arithmetic")
    check(len({tuple(t) for t, _ in decoys} | {tuple(key)}) == 4, "duplicate sequences")
    text = "Which of these sequences is an arithmetic sequence?"
    slips = [(_seq_text(t), tag) for t, tag in decoys]
    working = [f"{_seq_text(key)} goes up by {d} each time, so the difference is constant."]
    return choice_question(text, _seq_text(key), working, slips)


def _statement(rng):
    terms_arith = rng.random() < 0.5
    if terms_arith:
        terms, d = _arith(rng, 5)
        key = f"It is arithmetic, with a common difference of {d}."
        slips = [
            (f"It is arithmetic, with a common difference of {-d}.", "w3as-sign"),
            (f"It is not arithmetic: the difference changes after term {rng.randint(2, 4)}.", "w3as-wrong-break"),
            (f"It is arithmetic, with a common difference of {terms[0]}.", "w3as-first-term"),
        ]
        working = [f"Every term is {d} more than the one before, so the common difference is {d}."]
    else:
        terms = _one_pair(rng, 5)
        diffs = [terms[i + 1] - terms[i] for i in range(4)]
        brk = next(i for i in range(1, 4) if diffs[i] != diffs[0]) + 1  # the term after which the difference changes
        check(brk <= 3, "the break should be early")
        key = f"It is not arithmetic: the difference changes after term {brk}."
        other = next(j for j in (2, 3, 4) if j != brk)
        slips = [
            (f"It is arithmetic, with a common difference of {diffs[0]}.", "w3as-one-pair"),
            (f"It is arithmetic, with a common difference of {diffs[-1]}.", "w3as-one-pair"),
            (f"It is not arithmetic: the difference changes after term {other}.", "w3as-wrong-break"),
        ]
        working = [f"The differences are {', '.join(str(x) for x in diffs)}, so they are not all equal."]
    text = f"Look at the sequence {_seq_text(terms)}. Which statement is correct?"
    return choice_question(text, key, working, slips)


def _solve_k(rng):
    k0 = rng.randint(2, 20)
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    c = 2 * k0 + 2 * b - a
    check(c > 0, "c must be positive")
    t1, t2, t3 = f"k + {a}", f"2k + {b}", f"k + {c}"
    slips = [
        (str(k0 * 2), "w3as-halving-forgotten"),
        (str(k0 + 1), "w3as-arithmetic-slip"),
        (str(max(1, k0 - 1)), "w3as-arithmetic-slip"),
    ]
    text = f"The expressions {t1}, {t2} and {t3} are three consecutive terms of an arithmetic sequence. What is the value of k?"
    working = [
        f"For an arithmetic sequence the middle term is the mean of its neighbours: 2({t2}) = ({t1}) + ({t3}).",
        f"4k + {2 * b} = 2k + {a + c}, so 2k = {a + c - 2 * b} and k = {k0}.",
    ]
    return choice_question(text, str(k0), working, slips)


@generator(
    "w3_arith_recognise",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w3_arith_recognise(rng, tier="easy"):
    """Recognise an arithmetic sequence: pick one, judge one, or solve for a missing value."""
    pool = {"easy": (_which_arithmetic,), "medium": (_statement,), "hard": (_solve_k,)}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return _draw(rng, pool)


# -- activity networks: ESTs and LSTs ---------------------------------------------------------------------------------

_TEMPLATES = (
    {"A": (), "B": (), "C": ("A",), "D": ("A", "B"), "E": ("C",), "F": ("D", "E")},
    {"A": (), "B": ("A",), "C": ("A",), "D": ("B",), "E": ("B", "C"), "F": ("D", "E")},
    {"A": (), "B": (), "C": ("A", "B"), "D": ("B",), "E": ("C",), "F": ("C", "D"), "G": ("E", "F")},
    {"A": (), "B": ("A",), "C": ("A",), "D": ("B", "C"), "E": ("C",), "F": ("D", "E")},
)


def _network(rng):
    net = rng.choice(_TEMPLATES)
    dur = {n: rng.randint(1, 9) for n in net}
    est = {}
    for n in net:  # templates list activities in an order where predecessors come first
        est[n] = max((est[p] + dur[p] for p in net[n]), default=0)
    total = max(est[n] + dur[n] for n in net)
    succ = {n: [m for m in net if n in net[m]] for n in net}
    lst = {}
    for n in reversed(list(net)):
        lst[n] = min((lst[s] for s in succ[n]), default=total) - dur[n]
    # all start-to-end paths and their lengths
    starts = [n for n in net if not net[n]]
    paths = []

    def walk(node, route):
        route = route + [node]
        if not succ[node]:
            paths.append(route)
        for s in succ[node]:
            walk(s, route)

    for s in starts:
        walk(s, [])
    length = {tuple(p): sum(dur[n] for n in p) for p in paths}
    longest = max(length.values())
    critical_paths = [p for p, v in length.items() if v == longest]
    check(len(critical_paths) == 1, "the critical path must be unique")
    return net, dur, est, lst, total, succ, paths, length, list(critical_paths[0])


def _el_easy(rng):
    letters = "ABCD"
    key_idx = rng.randrange(4)
    rows = []
    for i, name in enumerate(letters):
        e = rng.randint(0, 14)
        rows.append((name, e, e if i == key_idx else e + rng.randint(1, 6)))
    floats = {n: l - e for n, e, l in rows}
    non = sorted((n for n in floats if floats[n] > 0), key=lambda n: floats[n])
    first_listed = next(n for n in letters if n != letters[key_idx])
    largest_est = max((r for r in rows if r[0] != letters[key_idx]), key=lambda r: (r[1], r[0]))[0]
    largest_lst = max((r for r in rows if r[0] != letters[key_idx]), key=lambda r: (r[2], r[0]))[0]
    candidates = [
        (non[0], "w3el-smallest-float"),
        (largest_est, "w3el-largest-est"),
        (largest_lst, "w3el-largest-lst"),
        (first_listed, "w3el-first-listed"),
    ]
    options = []
    seen = {letters[key_idx]}
    for name, tag in candidates:
        if name not in seen:
            seen.add(name)
            options.append((name, tag))
    for name in letters:  # top up so there are three wrong options
        if name not in seen:
            seen.add(name)
            options.append((name, "w3el-first-listed"))
    table = "; ".join(f"Activity {n}: EST {e}, LST {l}" for n, e, l in rows)
    text = f"A project has four activities. {table}. Which activity is on the critical path?"
    key = letters[key_idx]
    return choice_question(
        text,
        key,
        [f"Activity {key} has EST = LST = {rows[key_idx][1]}, so its float is zero and it is critical."],
        options[:3],
    )


def _path_text(path):
    return "-".join(path)


def _table(net, dur, est=None, lst=None):
    rows = []
    for n in net:
        pre = ", ".join(net[n]) if net[n] else "none"
        row = f"{n} (duration {dur[n]}, after {pre}"
        if est is not None:
            row += f", EST {est[n]}, LST {lst[n]}"
        rows.append(row + ")")
    return "; ".join(rows)


def _el_medium(rng):
    net, dur, est, lst, total, succ, paths, length, crit = _network(rng)
    key = _path_text(crit)
    others = [p for p in paths if p != crit]
    check(len(others) >= 2, "too few other paths")
    non_crit = [n for n in net if n not in crit]
    cands = []
    most = max(paths, key=lambda p: (len(p), -length[tuple(p)]))
    if most != crit:
        cands.append((_path_text(most), "w3el-most-activities"))
    for p in sorted(others, key=lambda p: -length[tuple(p)]):
        cands.append((_path_text(p), "w3el-longest-activity"))
    # one critical activity swapped for a non-critical one
    for i, n in enumerate(crit[1:-1], start=1):
        for m in non_crit:
            swapped = crit[:i] + [m] + crit[i + 1 :]
            cands.append((_path_text(swapped), "w3el-one-swap"))
    seen = {key}
    slips = []
    for text, tag in cands:
        if text not in seen:
            seen.add(text)
            slips.append((text, tag))
    text = f"A project has these activities: {_table(net, dur, est, lst)}. Which path through the network is the critical path?"
    working = [
        f"Activities with EST = LST are critical: {', '.join(n for n in net if est[n] == lst[n])}.",
        f"They form the path {key}, of length {total}.",
    ]
    return choice_question(text, key, working, slips[:3])


def _el_hard(rng):
    net, dur, est, lst, total, succ, paths, length, crit = _network(rng)
    free = [n for n in net if lst[n] != est[n] and succ[n]]
    check(free, "no activity with float and a successor")
    n = rng.choice(free)
    key = lst[n]
    next_lst = min(lst[s] for s in succ[n])
    cands = [
        (est[n], "w3el-est-for-lst"),
        (total - dur[n], "w3el-ignores-successors"),
        (next_lst, "w3el-forgets-duration"),
        (lst[n] - est[n], "w3el-float-for-lst"),
        (est[n] + dur[n], "w3el-finish-for-start"),
    ]
    seen = {key}
    slips = []
    for value, tag in cands:
        if value >= 0 and value not in seen:
            seen.add(value)
            slips.append((str(value), tag))
    text = (
        f"A project has these activities: {_table(net, dur)}. The earliest the project can finish is time {total}. "
        f"What is the latest start time (LST) of activity {n} if the project is not to be delayed?"
    )
    working = [
        f"Work backwards from the finish at time {total}.",
        f"Activity {n} must finish by time {next_lst}, so its LST is {next_lst} − {dur[n]} = {key}.",
    ]
    return choice_question(text, str(key), working, slips[:3])


@generator(
    "w3_ests_lsts",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w3_ests_lsts(rng, tier="easy"):
    """Locate the critical activity or path from ESTs and LSTs, or find a latest start time, in four options."""
    pool = {"easy": (_el_easy,), "medium": (_el_medium,), "hard": (_el_hard,)}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return _draw(rng, pool)


# -- bearings ---------------------------------------------------------------------------------------------------------


def _bearing(deg):
    return f"{int(deg) % 360:03d}°"


def _clear(value, places):
    """Whether `value` is far enough from a rounding tie at `places` places."""
    scaled = abs(value) * 10**places
    return abs(scaled - int(scaled) - 0.5) > 0.06


def _fr(x):
    return Fraction(x).limit_denominator(10**9)


def _north_east(d, theta):
    return d * cos(radians(theta)), d * sin(radians(theta))


def _br_component(rng):
    ask_north = rng.random() < 0.5
    for _ in attempts(200, "a bearing with a clear component"):
        d = rng.randint(8, 60)
        if ask_north:
            theta = rng.choice(list(range(20, 71)) + list(range(290, 341)))
        else:
            theta = rng.choice(list(range(20, 71)) + list(range(110, 161)))
        n, e = _north_east(d, theta)
        key = n if ask_north else e
        if _clear(key, 1) and _clear(d * (sin if ask_north else cos)(radians(theta)), 1) and abs(abs(key) - d) > 0.6:
            break
    wrong_other = d * (sin if ask_north else cos)(radians(theta))
    word = "north" if ask_north else "east"
    slips = [
        (_fr(wrong_other), "w3br-sin-cos-swapped"),
        (_fr(d), "w3br-distance-as-component"),
        (_fr(d * tan(radians(theta))), "w3br-tan-used"),
        (_fr(d / (cos if ask_north else sin)(radians(theta))), "w3br-divides"),
    ]
    dd = DP1.dist(_fr(key), slips)
    enough(dd, 3)
    text = (
        f"A hiker walks {d} km on a bearing of {_bearing(theta)}. How far {word} of the starting point is the hiker now? "
        f"Give your answer to 1 decimal place."
    )
    working = [
        f"The {word} distance is {d} × {'cos' if ask_north else 'sin'}({theta}°) = {DP1.text(_fr(key))} km.",
    ]
    return dict(text=text, working=working, distractors=dd, **DP1.fields(_fr(key)), answer_kind="number")


def _two_legs(rng):
    for _ in attempts(300, "two legs with clear results"):
        d1, d2 = rng.randint(10, 50), rng.randint(10, 50)
        t1 = rng.randint(10, 80)
        t2 = rng.choice([x for x in range(100, 260) if x % 90])
        n1, e1 = _north_east(d1, t1)
        n2, e2 = _north_east(d2, t2)
        n, e = n1 + n2, e1 + e2
        dist = hypot(n, e)
        if abs(n) > 3 and abs(e) > 3 and _clear(dist, 1) and _clear(n, 1) and _clear(e, 1):
            break
    return d1, t1, d2, t2, n1, e1, n2, e2, n, e, dist


def _br_distance(rng):
    d1, t1, d2, t2, n1, e1, n2, e2, n, e, dist = _two_legs(rng)
    slips = [
        (_fr(d1 + d2), "w3br-adds-legs"),
        (_fr(hypot(abs(n1) + abs(n2), abs(e1) + abs(e2))), "w3br-sign-of-component"),
        (_fr(d1), "w3br-one-leg"),
        (_fr(abs(n) + abs(e)), "w3br-adds-components"),
    ]
    dd = DP1.dist(_fr(dist), slips)
    enough(dd, 3)
    text = (
        f"A ship sails {d1} km on a bearing of {_bearing(t1)} and then {d2} km on a bearing of {_bearing(t2)}. "
        f"How far is the ship from its starting point? Give your answer to 1 decimal place."
    )
    working = [
        f"North: {d1}cos({t1}°) + {d2}cos({t2}°) = {n:.1f} km. East: {d1}sin({t1}°) + {d2}sin({t2}°) = {e:.1f} km.",
        f"Distance = √(({n:.1f})² + ({e:.1f})²) = {dist:.1f} km.",
    ]
    return dict(text=text, working=working, distractors=dd, **DP1.fields(_fr(dist)), answer_kind="number")


def _br_home(rng):
    for _ in attempts(300, "a clear back bearing"):
        d1, t1, d2, t2, n1, e1, n2, e2, n, e, dist = _two_legs(rng)
        forward = degrees(atan2(e, n)) % 360
        if _clear(forward, 0):
            break
    back = round(forward + 180) % 360
    fwd = round(forward) % 360
    check(abs(forward - round(forward)) > 0.06, "near-tie bearing")
    raw = round(degrees(atan2(abs(e), abs(n))))
    swapped = round(degrees(atan2(abs(n), abs(e))))
    slips = [
        (fwd, "w3br-no-reverse"),
        (raw, "w3br-raw-angle"),
        ((fwd + 90) % 360, "w3br-angle-from-east"),
        ((round(degrees(atan2(n, e))) + 180) % 360, "w3br-tan-swapped"),
        (swapped, "w3br-tan-swapped"),
    ]
    dd = WHOLE.dist(back, [(s, t) for s, t in slips])
    enough(dd, 3)
    text = (
        f"A ship sails {d1} km on a bearing of {_bearing(t1)} and then {d2} km on a bearing of {_bearing(t2)}. "
        f"What is the bearing of the starting point from the ship? Give your answer to the nearest degree, as a whole number of degrees."
    )
    working = [
        f"North: {n:.1f} km. East: {e:.1f} km. The bearing of the ship from the start is {forward:.1f}°.",
        f"The bearing back to the start is {forward:.1f}° + 180° = {back}° (to the nearest degree).",
    ]
    return dict(text=text, working=working, distractors=dd, **WHOLE.fields(Fraction(back)), answer_kind="number")


def _br_due_east(rng):
    for _ in attempts(300, "a leg ending due east"):
        d1 = rng.randint(10, 60)
        t1 = rng.randint(15, 75)
        t2 = rng.randint(105, 165)
        d2 = d1 * cos(radians(t1)) / -cos(radians(t2))
        if _clear(d2, 1):
            break
    wrong_sin = d1 * sin(radians(t1)) / sin(radians(t2))
    wrong_total = d1 / -cos(radians(t2))
    wrong_ratio = d1 * cos(radians(t1)) / sin(radians(t2))
    slips = [
        (_fr(wrong_sin), "w3br-uses-sin-not-cos"),
        (_fr(wrong_total), "w3br-uses-total-not-component"),
        (_fr(wrong_ratio), "w3br-sin-cos-swapped"),
        (_fr(d1), "w3br-one-leg"),
    ]
    dd = DP1.dist(_fr(d2), slips)
    enough(dd, 3)
    text = (
        f"A boat sails {d1} km on a bearing of {_bearing(t1)}. It then changes course to a bearing of {_bearing(t2)} and sails "
        f"until it is due east of its starting point. How far does it sail on the second leg? Give your answer to 1 decimal place."
    )
    working = [
        f"Being due east means the total northward distance is zero: {d1}cos({t1}°) + d × cos({t2}°) = 0.",
        f"d = {d1} × cos({t1}°) ÷ {-cos(radians(t2)):.4f} = {d2:.1f} km.",
    ]
    return dict(text=text, working=working, distractors=dd, **DP1.fields(_fr(d2)), answer_kind="number")


@generator("w3_bearings_general", level=13, presets={"medium": dict(tier="medium"), "hard": dict(tier="hard")})
def w3_bearings_general(rng, tier="medium"):
    """Bearings with legs that are neither cardinal nor at right angles."""
    pool = {"medium": (_br_component,), "hard": (_br_distance, _br_home, _br_due_east)}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return _draw(rng, pool)
