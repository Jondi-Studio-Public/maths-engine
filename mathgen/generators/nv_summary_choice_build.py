"""Generates `nv_summary_choice` questions (easy, medium, hard) and the exam chain `exd_x_nv_summary_choice`.

``choose-summary-statistic-for-data`` (``statistics-probability``, answer ``mcq``): choosing mode, median or mean from the
type of variable, the shape of the data and what the figure is for, and then working out the one chosen. ``classify-data``
sorts variables into types and ``calculate-mean`` / ``calculate-median`` / ``calculate-mode`` work out a named measure;
this skill is the step between them. The existing open-response ``wa_centre_suitability`` asks for an explanation that
a rubric marks; the easy naming question is a multiple-choice question; everywhere else the choice is graded through the number it
leads to, so a wrong measure is always a different number.

* easy: which measure of centre suits a situation, answered by name (Mode for a nominal variable, Median when a few values
  are far larger, Mean when a total is wanted; the other options include the range), or the value that suits a purpose for
  five listed values (the median when one value is far larger, the mean when a total is wanted, the mode when the most
  common value is wanted).
* medium: the same purposes for ten listed values (to 1 decimal place).
* hard: a figure has been claimed that does not suit the data. The learner works out the value of the measure that does:
  the median instead of the mean for a skewed set, the mean instead of the median when a total is wanted, the mode instead
  of the mean for the most common value, the median of ordered ratings instead of their mean, and the size of the most
  common response of a variable coded with numbers (a mean code is not a category).
* exam (``exd_x_nv_summary_choice``): ten values: (a) the mean, (b) the median, (c) the mean minus the median or the
  total from the mean for a stated number of items (from (a)), (d) how far the mean overstates the median as a percentage
  of the median, or the difference between the two totals (from earlier parts).

Every number is re-derived from the listed values with ``statistics`` and by sorting by hand; each draw fixes the mean,
median and mode apart so a wrong measure is always a different number. Wrong values are ordered so that the key has no
fixed rank among the options (`_nv_kit.spread_order`).
"""

from __future__ import annotations

import statistics
from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._exd_distinct_parts import distinct_part_answers
from ._nv_kit import spread_order
from ._exd_r3f_parts import clear_of_boundary
from ._r3j_kit import choice_question, number_part, sane, single_number
from ._spread_kit import NEAR, uses_rng
from ._w6a_kit import retry_draw

TOPIC = "nv_summary_choice"
EXAM_TOPIC = "exd_x_nv_summary_choice"
AREA = "choose-summary-statistic"

MISCONCEPTIONS = [
    ("nv-mean-of-categories", "Treats the categories of a nominal variable as numbers that can be averaged.", AREA),
    (
        "nv-median-of-unordered-categories",
        "Looks for the middle category of a nominal variable, which has no order.",
        AREA,
    ),
    ("nv-range-as-centre", "Gives the range, a measure of spread, where a measure of centre was asked for.", AREA),
    (
        "nv-mean-for-skewed",
        "Uses the mean for data with one extreme value, although the extreme value drags the mean away from a typical value.",
        AREA,
    ),
    (
        "nv-mean-for-ordinal",
        "Gives the mean of ordered ratings coded 1 to 5, where the middle rating was needed.",
        AREA,
    ),
    ("nv-mode-ignores-order", "Gives the most common rating where the middle rating was needed.", AREA),
    (
        "nv-median-for-total",
        "Uses the median where a total is wanted; the total comes from the mean times the number of values.",
        AREA,
    ),
    (
        "nv-removes-outlier",
        "Throws away the extreme value and averages the rest, instead of using the median of all the values.",
        AREA,
    ),
    ("nv-mean-for-most-common", "Gives the mean where the most common value was needed.", AREA),
    ("nv-median-for-most-common", "Gives the median where the most common value was needed.", AREA),
    ("nv-largest-as-centre", "Gives the largest value as the typical value.", AREA),
    (
        "nv-mean-of-codes",
        "Gives the count for the category nearest the mean code, as if the mean code named a category.",
        AREA,
    ),
    (
        "nv-median-of-codes",
        "Gives the count for the category at the middle code, as if the middle code named a category.",
        AREA,
    ),
    ("nv-count-of-another-category", "Gives the count of a category that is not the most common one.", AREA),
    ("nv-median-unsorted", "Takes the middle of the list as written, without putting the values in order first.", AREA),
    (
        "nv-median-one-middle",
        "Takes one of the two middle values of an even count instead of the number halfway between them.",
        AREA,
    ),
    ("nv-median-wrong-position", "Takes the value one place away from the middle of the ordered list.", AREA),
    ("nv-mean-wrong-count", "Divides the total by a count that is one out.", AREA),
    ("nv-wrong-measure", "Works out a different measure of centre from the one that was asked for.", AREA),
    ("nv-difference-reversed", "Subtracts in the wrong order, giving the median minus the mean.", AREA),
    ("nv-total-of-sample", "Gives the total of the listed values as the total for the larger number of items.", AREA),
    ("nv-percent-of-mean", "Finds the excess as a percentage of the mean instead of the median.", AREA),
    ("nv-difference-not-percent", "Gives the difference itself where a percentage of the median was asked.", AREA),
    ("nv-ratio-not-excess", "Gives the mean as a percentage of the median, not the excess over it.", AREA),
]

# (what is listed, low, high, outlier low, outlier high, singular noun for a stem)
_SKEWED = [
    ("the weekly pocket money, in dollars, of ten students", 8, 30, 90, 150, "weekly pocket money"),
    ("the sale prices, in thousands of dollars, of ten houses on one street", 520, 760, 1700, 2500, "house price"),
    ("the waiting times, in minutes, of ten patients at a clinic", 6, 30, 90, 140, "waiting time"),
    ("the annual salaries, in thousands of dollars, of ten staff at a small firm", 52, 88, 260, 420, "salary"),
    ("the numbers of text messages sent in one day by ten teenagers", 12, 60, 190, 300, "daily message count"),
    ("the daily numbers of visitors to a small museum over ten days", 40, 95, 260, 400, "daily visitor count"),
]
# (what is listed, low, high, per-item noun for a total, what a total of it is called)
_BALANCED = [
    ("the masses, in grams, of ten packets of rice from one production line", 494, 508, "packet", "mass of rice"),
    ("the daily rainfall, in millimetres, recorded over ten days at a weather station", 12, 30, "day", "rainfall"),
    ("the distances, in kilometres, cycled by a club member on ten rides", 20, 38, "ride", "distance cycled"),
    ("the weekly grocery bills, in dollars, of ten households", 180, 260, "household", "grocery spending"),
    ("the daily takings, in dollars, of a market stall over ten days", 410, 560, "day", "takings"),
]
# (what is listed, low, high, what the most common value is used for)
_COUNTS = [
    ("the shoe sizes of ten players in a netball club", 5, 11, "the size the club should order most of"),
    ("the numbers of pets in each of ten households", 0, 5, "the pet count that most households have"),
    ("the numbers of goals scored by a team in ten games", 0, 6, "the score the team reaches most often"),
    ("the numbers of passengers in each of ten cars at a toll gate", 1, 6, "the passenger count that most cars carry"),
]
# (scene, what the answers are called)
_ORDINAL = [
    ("A cafe asks each customer to rate the service as Poor, Fair, Good, Very good or Excellent.", "the ratings"),
    (
        "A survey asks residents how safe they feel at night: Very unsafe, Unsafe, Neutral, Safe or Very safe.",
        "the answers",
    ),
    (
        "A school asks parents how satisfied they are with a camp: Very dissatisfied, Dissatisfied, Neutral, Satisfied or Very satisfied.",
        "the answers",
    ),
    ("A gym asks members how tiring a class was: Very easy, Easy, Moderate, Hard or Very hard.", "the answers"),
    ("A hospital asks patients to describe their pain as None, Mild, Moderate, Severe or Extreme.", "the ratings"),
    ("A library asks borrowers how often they visit: Never, Rarely, Sometimes, Often or Always.", "the answers"),
]
_NOMINAL = [
    ("A school canteen records the lunch (pie, sandwich, salad or sushi) that each student buys.", "the lunches"),
    (
        "A council records the type of dwelling (house, unit, townhouse or apartment) of each household in a suburb.",
        "the dwellings",
    ),
    (
        "A sports club records the main sport (netball, cricket, tennis or swimming) of each junior member.",
        "the sports",
    ),
    ("A tourism survey records the country each visitor came from.", "the countries"),
    ("A vet records the breed of each dog brought to a clinic in one week.", "the breeds"),
    ("A bakery records the flavour of cake that each customer orders.", "the flavours"),
]
_CODED = [
    ("how students travel to school", ["bus", "train", "walk", "car"]),
    ("the type of pet in each household", ["dog", "cat", "bird", "fish"]),
    ("the main language spoken at home by each student", ["English", "Mandarin", "Arabic", "Vietnamese"]),
    ("the flavour of pie chosen at a bakery", ["beef", "chicken", "vegetable", "apple"]),
]
_NUMERICAL = [
    "A netball club records the shoe sizes of ten of its players.",
    "A vet clinic records how many pets live in each of ten households.",
    "A coach records the goals scored by the team in each of ten games.",
    "A toll gate records the passengers in each of ten cars.",
]
_WORDS = {5: "five", 10: "ten"}


def _what(ctx, n: int) -> str:
    """The description of the listed values, written for `n` of them."""
    return ctx[0].replace("ten", _WORDS[n])


def _plural(noun: str) -> str:
    return {"person": "people"}.get(noun, noun + "s")


# -- data -------------------------------------------------------------------------------------------------------------


def _mean(values) -> Fraction:
    return Fraction(sum(values), len(values))


def _median(values) -> Fraction:
    s = sorted(values)
    n = len(s)
    return Fraction(s[n // 2]) if n % 2 else Fraction(s[n // 2 - 1] + s[n // 2], 2)


def _median_unsorted(values) -> Fraction:
    n = len(values)
    return Fraction(values[n // 2]) if n % 2 else Fraction(values[n // 2 - 1] + values[n // 2], 2)


def _shown(values) -> str:
    return ", ".join(str(v) for v in values)


def _fmt(x) -> str:
    return f"{float(x):.1f}"


def _check_stats(values, mean, median):
    check(mean == Fraction(statistics.fmean(values)).limit_denominator(1000), "mean differs")
    check(median == Fraction(statistics.median(values)).limit_denominator(1000), "median differs")


def _skewed(rng, ctx, n=10):
    _, lo, hi, olo, ohi, _ = ctx
    for _ in attempts(400, "a skewed data set"):
        base = rng.sample(range(lo, hi + 1), n - 1)
        out = rng.randint(olo, ohi)
        if n == 10:  # the mean of ten values is then a multiple of 0.5
            out += (-(sum(base) + out)) % 5
        values = base + [out]
        mean, median = _mean(values), _median(values)
        if mean - median >= Fraction(1, 4) * median:
            _check_stats(values, mean, median)
            rng.shuffle(values)
            return values, mean, median
    raise AssertionError  # pragma: no cover


def _balanced(rng, ctx, n=10):
    _, lo, hi, _, _ = ctx
    for _ in attempts(4000, "a balanced data set"):
        values = rng.sample(range(lo, hi + 1), n)
        mean, median = _mean(values), _median(values)
        gap = abs(mean - median)
        if (
            (n == 5 or sum(values) % 5 == 0)
            and Fraction(3, 10) <= gap <= Fraction(1, 25) * median
            and max(values) <= 2 * min(values)
        ):
            _check_stats(values, mean, median)
            return values, mean, median
    raise AssertionError  # pragma: no cover


def _counts(rng, ctx, n=10):
    _, lo, hi, _ = ctx
    least = 3 if n == 10 else 2
    for _ in attempts(8000, "a repeated count"):
        values = [rng.randint(lo, hi) for _ in range(n)]
        top = max(set(values), key=values.count)
        tally = sorted((values.count(v) for v in set(values)), reverse=True)
        if tally[0] < least or (len(tally) > 1 and tally[1] == tally[0]):
            continue
        mean, median = _mean(values), _median(values)
        if (
            (n == 5 or sum(values) % 5 == 0)
            and len({mean, median, Fraction(top)}) == 3
            and abs(mean - median) >= Fraction(3, 10)
            and abs(median - top) >= Fraction(1, 2)
            and abs(mean - top) >= Fraction(1, 2)
        ):
            _check_stats(values, mean, median)
            check(statistics.multimode(values) == [top], "mode differs")
            return values, mean, median, top
    raise AssertionError  # pragma: no cover


# -- wrong values -----------------------------------------------------------------------------------------------------


def _ordered(key, slips):
    """`slips` reordered so that a random number of the first three wrong values are below the key (plain near misses
    make up a side that has too few), so the key's rank among the options is not fixed."""
    key = Fraction(key)
    fillers = [(key + Fraction(d, 2), NEAR) for d in (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)]
    # the answer kit drops a wrong value far from the key (and a zero): drop it here so it cannot take a place in the order
    slips, fillers = sane(slips, key, 15), sane(fillers, key, 15)
    return spread_order(key, slips, fillers, 3, lo=-(10**6), hi=10**6)


def _median_slips(values, mean, median):
    s = sorted(values)
    n = len(s)
    out = [(mean, "nv-mean-for-skewed"), (Fraction(sum(s[:-1]), n - 1), "nv-removes-outlier")]
    out.append((_median_unsorted(values), "nv-median-unsorted"))
    if n % 2:
        out += [
            (Fraction(s[n // 2 - 1]), "nv-median-wrong-position"),
            (Fraction(s[n // 2 + 1]), "nv-median-wrong-position"),
        ]
    else:
        out += [(Fraction(s[n // 2]), "nv-median-one-middle"), (Fraction(s[n // 2 - 1]), "nv-median-one-middle")]
    return out


def _mean_slips(values, mean, median):
    n = len(values)
    return [
        (median, "nv-median-for-total"),
        (Fraction(sum(values), n - 1), "nv-mean-wrong-count"),
        (Fraction(sum(values), n + 1), "nv-mean-wrong-count"),
        (Fraction(max(values) - min(values)), "nv-range-as-centre"),
    ]


def _mode_slips(values, mean, median):
    return [
        (mean, "nv-mean-for-most-common"),
        (median, "nv-median-for-most-common"),
        (Fraction(max(values) - min(values)), "nv-range-as-centre"),
        (Fraction(max(values)), "nv-largest-as-centre"),
    ]


# -- easy -------------------------------------------------------------------------------------------------------------


def _easy_measure(rng):
    """Which measure of centre suits a variable or a purpose: the answer is a word (Mode, Median or Mean)."""
    kind = rng.choice(["nominal", "skewed", "total"])
    options = ("Mean", "Median", "Mode", "Range")
    if kind == "nominal":
        scene, thing = rng.choice(_NOMINAL)
        key = "Mode"
        text = f"{scene} Which of these is a sensible measure of centre for {thing}?"
        slips = [
            ("Mean", "nv-mean-of-categories"),
            ("Median", "nv-median-of-unordered-categories"),
            ("Range", "nv-range-as-centre"),
        ]
        working = [
            "The categories have no order and are not numbers, so they cannot be averaged or put in order.",
            "Only the most common category, the mode, can be found. The range measures spread, not centre.",
        ]
    elif kind == "skewed":
        ctx = rng.choice(_SKEWED)
        key = "Median"
        text = (
            f"A researcher lists {ctx[0]}. A few of the values are far larger than all the others. "
            f"Which measure of centre best represents a typical {ctx[5]}?"
        )
        slips = [
            ("Mean", "nv-mean-for-skewed"),
            ("Range", "nv-range-as-centre"),
            ("Mode", "nv-wrong-measure"),
        ]
        working = [
            "A few very large values drag the mean upwards, away from a typical value.",
            "The median is the middle value, so it is hardly moved by the extreme ones.",
        ]
    else:
        ctx = rng.choice(_BALANCED)
        key = "Mean"
        text = (
            f"A planner lists {ctx[0]}. The values are all close together. "
            f"The planner wants to estimate the total {ctx[4]} for many {_plural(ctx[3])} of the same kind. "
            "Which measure of centre should be scaled up for the estimate?"
        )
        slips = [
            ("Median", "nv-median-for-total"),
            ("Mode", "nv-wrong-measure"),
            ("Range", "nv-range-as-centre"),
        ]
        working = [
            "A total is the sum of all the values, and the mean is the sum shared equally.",
            "So the mean multiplied by the number of items gives the total; the median does not.",
        ]
    check(key in options and all(w in options for w, _ in slips), "unknown option")
    return choice_question(text, key, working, slips)


def _easy_median(rng):
    ctx = rng.choice(_SKEWED)
    values, mean, median = _skewed(rng, ctx, 5)
    text = phrase(
        rng,
        (
            "Here are {what}: {v}. One value is much larger than the rest. What single value best shows a typical {noun}? Give your answer to 1 decimal place.",
            "A newspaper lists {what}: {v}. Which value gives the fairest picture of a typical {noun}, not pulled up by the large one? Answer to 1 decimal place.",
        ),
        what=_what(ctx, 5),
        v=_shown(values),
        noun=ctx[5],
    )
    s = sorted(values)
    working = [
        f"Put the values in order: {_shown(s)}.",
        f"The middle value is {_fmt(median)}.",
        f"The mean ({_fmt(mean)}) is pulled up by {s[-1]}.",
    ]
    return single_number(text, median, 1, working, _ordered(median, _median_slips(values, mean, median)))


def _easy_mean(rng):
    ctx = rng.choice(_BALANCED)
    values, mean, median = _balanced(rng, ctx, 5)
    k = rng.choice([20, 30, 40])
    text = phrase(
        rng,
        (
            "The data are {what}: {v}. The figure will be scaled up to give the total {u} for {k} {n}. What single value should be scaled up? Answer to 1 decimal place.",
            "Here are {what}: {v}. A planner needs the typical value to estimate the total {u} for {k} {n}. Find it, to 1 decimal place.",
        ),
        what=_what(ctx, 5),
        v=_shown(values),
        u=ctx[4],
        k=k,
        n=_plural(ctx[3]),
    )
    working = [
        f"A total comes from the mean: the sum is {sum(values)}, so the mean is {sum(values)} / 5 = {_fmt(mean)}."
    ]
    return single_number(text, mean, 1, working, _ordered(mean, _mean_slips(values, mean, median)))


def _easy_mode(rng):
    ctx = rng.choice(_COUNTS)
    values, mean, median, mode = _counts(rng, ctx, 5)
    text = phrase(
        rng,
        (
            "Here are {what}: {v}. Which value gives {use}? Answer to 1 decimal place.",
            "The records show {what}: {v}. Find the value that tells us {use}. Give it to 1 decimal place.",
        ),
        what=_what(ctx, 5),
        v=_shown(values),
        use=ctx[3],
    )
    working = [f"The value {mode} is listed {values.count(mode)} times, more than any other value: it is the mode."]
    return single_number(text, Fraction(mode), 1, working, _ordered(Fraction(mode), _mode_slips(values, mean, median)))


_EASY = (_easy_measure, _easy_measure, _easy_median, _easy_median, _easy_mean, _easy_mean, _easy_mode)


# -- medium -----------------------------------------------------------------------------------------------------------


def _medium_median(rng):
    ctx = rng.choice(_SKEWED)
    values, mean, median = _skewed(rng, ctx)
    text = phrase(
        rng,
        (
            "These are {what}: {v}. The one very large value should not distort the figure reported. Work out the best single value for a typical {noun}. Give your answer to 1 decimal place.",
            "Data on {what}: {v}. A reporter wants a typical {noun} that one extreme value cannot pull upwards. Find that value, to 1 decimal place.",
        ),
        what=ctx[0],
        v=_shown(values),
        noun=ctx[5],
    )
    s = sorted(values)
    working = [
        f"Put the values in order: {_shown(s)}.",
        f"Ten values, so the median is halfway between the 5th ({s[4]}) and 6th ({s[5]}): {_fmt(median)}.",
        f"The mean ({_fmt(mean)}) is pulled up by {s[-1]}.",
    ]
    return single_number(text, median, 1, working, _ordered(median, _median_slips(values, mean, median)))


def _medium_mean(rng):
    ctx = rng.choice(_BALANCED)
    values, mean, median = _balanced(rng, ctx)
    n_items = rng.choice([30, 40, 50, 60])
    text = phrase(
        rng,
        (
            "These are {what}: {v}. The figure will be multiplied by {k} to estimate a total for {k} {n}. Work out the single value that should be used, to 1 decimal place.",
            "Data on {what}: {v}. A planner needs the value that gives the right total when scaled up to {k} {n}. Find it, to 1 decimal place.",
        ),
        what=ctx[0],
        v=_shown(values),
        k=n_items,
        n=_plural(ctx[3]),
    )
    working = [
        f"A total comes from the mean: the sum is {sum(values)} and there are 10 values, so the mean is {_fmt(mean)}.",
        f"The median ({_fmt(median)}) does not give the total.",
    ]
    return single_number(text, mean, 1, working, _ordered(mean, _mean_slips(values, mean, median)))


def _medium_mode(rng):
    ctx = rng.choice(_COUNTS)
    values, mean, median, mode = _counts(rng, ctx)
    text = phrase(
        rng,
        (
            "Here are {what}: {v}. Find {use}. Give your answer to 1 decimal place.",
            "The records show {what}: {v}. Work out {use}, as a number to 1 decimal place.",
        ),
        what=ctx[0],
        v=_shown(values),
        use=ctx[3],
    )
    working = [
        f"Count how often each value is listed: the value {mode} is listed {values.count(mode)} times, more than any other.",
        f"The mode is {mode}; the mean ({_fmt(mean)}) and median ({_fmt(median)}) are not the most common value.",
    ]
    return single_number(text, Fraction(mode), 1, working, _ordered(Fraction(mode), _mode_slips(values, mean, median)))


_MEDIUM = (_medium_median, _medium_median, _medium_mean, _medium_mean, _medium_mode)


# -- hard -------------------------------------------------------------------------------------------------------------


def _hard_claim_skewed(rng):
    ctx = rng.choice(_SKEWED)
    values, mean, median = _skewed(rng, ctx)
    text = (
        f"A report says the typical {ctx[5]} is {_fmt(mean)}, taken from {ctx[0]}: {_shown(values)}. "
        f"Work out the measure of centre that describes a typical {ctx[5]} better than that claim, to 1 decimal place."
    )
    working = [
        f"The claim is the mean, {_fmt(mean)}, which one very large value has pulled up.",
        f"Ordered: {_shown(sorted(values))}. The median is {_fmt(median)}.",
    ]
    return single_number(text, median, 1, working, _ordered(median, _median_slips(values, mean, median)))


def _hard_claim_total(rng):
    ctx = rng.choice(_BALANCED)
    values, mean, median = _balanced(rng, ctx)
    k = rng.choice([25, 40, 60])
    text = (
        f"A planner used the median, {_fmt(median)}, from {ctx[0]}: {_shown(values)}, to estimate the total {ctx[4]} for {k} {_plural(ctx[3])}. "
        f"Work out the figure that should be multiplied by {k} instead, to 1 decimal place."
    )
    working = [
        f"The sum of the values is {sum(values)}, so the mean is {_fmt(mean)}.",
        "A total for many items is the mean times the number of items.",
    ]
    return single_number(text, mean, 1, working, _ordered(mean, _mean_slips(values, mean, median)))


def _hard_claim_common(rng):
    ctx = rng.choice(_COUNTS)
    values, mean, median, mode = _counts(rng, ctx)
    text = (
        f"A teacher reports {_fmt(mean)} as {ctx[3]}, the mean of {ctx[0]}: {_shown(values)}. "
        "Find the value that does answer the question, to 1 decimal place."
    )
    working = [
        f"The question asks for the most common value, not the mean ({_fmt(mean)}).",
        f"The value {mode} is listed {values.count(mode)} times, more than any other value.",
    ]
    slips = [(median, "nv-median-for-most-common"), (mean + 1, NEAR)] + _mode_slips(values, mean, median)[2:]
    return single_number(text, Fraction(mode), 1, working, _ordered(Fraction(mode), slips))


def _hard_claim_ordinal(rng):
    scene = rng.choice(
        [
            "A cafe asks eleven customers to rate its service from 1 (poor) to 5 (excellent).",
            "A school asks eleven parents how satisfied they are with a camp, from 1 (very dissatisfied) to 5 (very satisfied).",
            "A council asks eleven residents how safe they feel at night, from 1 (very unsafe) to 5 (very safe).",
        ]
    )
    for _ in attempts(2000, "ordered ratings whose mean rounds away from the median"):
        codes = [rng.randint(1, 5) for _ in range(11)]
        mean, median = _mean(codes), _median(codes)
        tally = sorted((codes.count(c) for c in set(codes)), reverse=True)
        top = max(set(codes), key=codes.count)
        if (
            tally[0] >= 3
            and (len(tally) == 1 or tally[1] < tally[0])
            and len({round(mean), int(median), top}) == 3
            and abs(mean - round(mean)) < Fraction(9, 20)
        ):
            break
    text = (
        f'{scene} The ratings are: {_shown(codes)}. A manager says: "The mean rating is {_fmt(mean)}, so the typical rating is {round(mean)}." '
        "Find the typical rating using the measure that respects the order of the ratings."
    )
    working = [
        f"The ratings are ordered labels, not measured amounts, so the mean ({_fmt(mean)}) has no clear meaning.",
        f"Ordered: {_shown(sorted(codes))}. The middle (6th) rating is {int(median)}.",
    ]
    slips = [
        (Fraction(round(mean)), "nv-mean-for-ordinal"),
        (Fraction(top), "nv-mode-ignores-order"),
        (Fraction(int(median) + 1), NEAR),
        (Fraction(int(median) - 1), NEAR),
        (Fraction(_median_unsorted(codes)), "nv-median-unsorted"),
    ]
    return single_number(text, median, 0, working, _ordered_whole(median, slips))


def _ordered_whole(key, slips):
    key = Fraction(key)
    fillers = [(key + d, NEAR) for d in (-2, -1, 1, 2)]
    return spread_order(key, slips, fillers, 3, lo=1, hi=5)


def _hard_coded_count(rng):
    scene, names = rng.choice(_CODED)
    for _ in attempts(1000, "coded categories whose mean and middle codes are not the most common"):
        codes = [rng.randint(1, 4) for _ in range(rng.choice([11, 13, 15]))]
        counts = {c: codes.count(c) for c in range(1, 5)}
        top = max(counts, key=counts.get)
        if sorted(counts.values())[-1] == sorted(counts.values())[-2]:
            continue
        mean = Fraction(sum(codes), len(codes))
        near = min(range(1, 5), key=lambda c: (abs(c - mean), c))
        mid = sorted(codes)[len(codes) // 2]
        if len({top, near, mid}) == 3 and abs(mean - near) < Fraction(9, 20):
            break
    text = (
        f"A survey recorded {scene}, coded "
        + ", ".join(f"{i + 1} = {n}" for i, n in enumerate(names))
        + f". The codes for {len(codes)} people are: {_shown(codes)}. "
        f"A student finds that the mean code is {float(mean):.1f}, so says the typical response is {names[near - 1]}. "
        "How many of the people gave the most common response?"
    )
    working = [
        "The codes are labels, so a mean code has no meaning.",
        "Tally the codes: "
        + ", ".join(f"{n} {counts[i + 1]}" for i, n in enumerate(names))
        + f". The most common is {names[top - 1]}, {counts[top]} people.",
    ]
    others = [c for c in range(1, 5) if c not in (top, near, mid)]
    slips = [(Fraction(counts[near]), "nv-mean-of-codes"), (Fraction(counts[mid]), "nv-median-of-codes")] + [
        (Fraction(counts[c]), "nv-count-of-another-category") for c in others
    ]
    key = Fraction(counts[top])
    fillers = [(key + d, NEAR) for d in (-2, -1, 1, 2)]
    ordered = spread_order(key, slips, fillers, 3, lo=1, hi=len(codes) - 1)
    return single_number(text, key, 0, working, ordered)


_HARD = (
    _hard_claim_skewed,
    _hard_claim_total,
    _hard_claim_common,
    _hard_claim_ordinal,
    _hard_coded_count,
)


@generator(
    TOPIC,
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@uses_rng
def nv_summary_choice(rng, tier="easy"):
    """Choosing and working out mode, median or mean; the tiers are described in the module docstring."""
    pool = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# -- exam -------------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
@uses_rng
def exd_x_nv_summary_choice(rng, tier="exam"):
    """Exam chain on ten listed values: mean, median, a figure that follows from them, and what that shows."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _exam_chain)


def _exam_chain(rng):
    kind = "skewed" if rng.random() < 0.5 else "total"
    if kind == "skewed":
        ctx = rng.choice(_SKEWED)
        for _ in attempts(400, "a skewed set whose percentage is clear of a rounding boundary"):
            values, mean, median = _skewed(rng, ctx)
            if clear_of_boundary((mean - median) / median * 100, 1):
                break
        intro = f"A researcher lists {ctx[0]}: {_shown(values)}. The aim is to report a typical {ctx[5]}."
        c_prompt = "Find the mean minus the median."
        c_value = mean - median
        c_working = [f"{_fmt(mean)} - {_fmt(median)} = {_fmt(c_value)}."]
        c_slips = [
            (median - mean, "nv-difference-reversed"),
            (mean, "nv-wrong-measure"),
            (median, "nv-wrong-measure"),
        ]
        c_ft, c_range = "part_a - part_b", (None, None)
        d_prompt = "The median is the better measure of a typical value here. By what percentage of the median does the mean overstate it?"
        d_value = c_value / median * 100
        d_working = [f"({_fmt(mean)} - {_fmt(median)}) / {_fmt(median)} x 100 = {float(d_value):.1f}%."]
        d_slips = [
            (c_value / mean * 100, "nv-percent-of-mean"),
            (c_value, "nv-difference-not-percent"),
            (mean / median * 100, "nv-ratio-not-excess"),
            (median / mean * 100, NEAR),
        ]
        d_ft, d_range = "part_c / part_b * 100", (0, None)
    else:
        ctx = rng.choice(_BALANCED)
        values, mean, median = _balanced(rng, ctx)
        k = rng.choice([40, 50, 60, 80])
        intro = (
            f"A planner lists {ctx[0]}: {_shown(values)}. "
            f"The aim is to estimate the total {ctx[4]} for {k} {_plural(ctx[3])} of the same kind."
        )
        c_prompt = f"Use the mean to estimate the total {ctx[4]} for {k} {_plural(ctx[3])}."
        c_value = mean * k
        c_working = [f"{_fmt(mean)} x {k} = {float(c_value):.1f}."]
        c_slips = [
            (median * k, "nv-median-for-total"),
            (Fraction(sum(values)), "nv-total-of-sample"),
        ]
        c_ft, c_range = f"part_a * {k}", (0, None)
        d_prompt = f"The same estimate made from the median instead would differ from the mean's estimate. Find (estimate from the mean) minus (estimate from the median) for the {k} {_plural(ctx[3])}."
        d_value = (mean - median) * k
        d_working = [f"({_fmt(mean)} - {_fmt(median)}) x {k} = {float(d_value):.1f}."]
        d_slips = [
            ((median - mean) * k, "nv-difference-reversed"),
            (mean - median, NEAR),
            (median * k, NEAR),
            (Fraction(sum(values)) - median * k, NEAR),
        ]
        d_ft, d_range = f"(part_a - part_b) * {k}", (None, None)
    s = sorted(values)
    check(sum(s) == sum(values) and mean == Fraction(sum(values), 10), "mean differs")
    a = number_part(
        "Find the mean, to 1 decimal place.",
        mean,
        1,
        2,
        [f"The sum is {sum(values)}, so the mean is {sum(values)} / 10 = {_fmt(mean)}."],
        _ordered(
            mean,
            [
                (Fraction(sum(values), 9), "nv-mean-wrong-count"),
                (Fraction(sum(values), 11), "nv-mean-wrong-count"),
                (median, "nv-wrong-measure"),
            ],
        ),
    )
    b = number_part(
        "Find the median, to 1 decimal place.",
        median,
        1,
        2,
        [f"Ordered: {_shown(s)}.", f"The median is halfway between {s[4]} and {s[5]}: {_fmt(median)}."],
        _ordered(
            median,
            [
                (_median_unsorted(values), "nv-median-unsorted"),
                (Fraction(s[4]), "nv-median-one-middle"),
                (Fraction(s[5]), "nv-median-one-middle"),
                (mean, "nv-wrong-measure"),
            ],
        ),
    )
    c = number_part(
        c_prompt + " Give your answer to 1 decimal place.",
        c_value,
        1,
        2,
        c_working,
        _ordered(c_value, c_slips),
        follow_through=c_ft,
        sensible_range=c_range,
    )
    d = number_part(
        d_prompt + " Give your answer to 1 decimal place.",
        d_value,
        1,
        2,
        d_working,
        _ordered(d_value, d_slips),
        follow_through=d_ft,
        sensible_range=d_range,
    )
    return dict(text=intro, parts=[a, b, c, d])
