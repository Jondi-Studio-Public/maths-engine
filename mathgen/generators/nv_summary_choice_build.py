"""Generates `nv_summary_choice` questions (easy, medium, hard) and the exam chain `exd_x_nv_summary_choice`.

``choose-summary-statistic-for-data`` (``statistics-probability``): choosing mode, median or mean from the type of
variable, the shape of the data and what the figure is for, and then working out the one chosen. ``classify-data``
sorts variables into types and ``calculate-mean`` / ``calculate-median`` / ``calculate-mode`` work out a named measure;
this skill is the step between them. The existing open-response ``wa_centre_suitability`` asks for an explanation that
a rubric marks; here the choice itself is graded.

* easy: pick the measure (Mean, Median, Mode or Range) that suits a described variable: a nominal variable (mode), an
  ordinal rating (median, the one that uses the order), a skewed set with one extreme value (median), a balanced set when a
  total or an equal share is wanted (mean), or the most common value of a count when the most common is what is needed
  (mode).
* medium: the same choice, made from a stated purpose and then worked out for ten listed values (to 1 decimal place): the
  median of a skewed set, the mean of a balanced set whose total is wanted, or the mode of a repeated count.
* hard: a figure has been claimed that does not suit the data. Either choose the response that corrects the claim (four
  full statements), or work out the better measure's value, or name the typical category of a variable that was coded with
  numbers (the mean code is not a category).
* exam (``exd_x_nv_summary_choice``): ten values: (a) the mean, (b) the median, (c) the mean minus the median or the total
  for a stated number of items from the mean (from (a) and (b)), (d) which measure suits the purpose.

Every number is re-derived from the listed values with ``statistics`` and by sorting by hand; each draw fixes the mean,
median and mode apart so a wrong measure is always a different number.
"""

from __future__ import annotations

import statistics
from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._exd_distinct_parts import distinct_part_answers
from ._r3j_kit import choice_part, choice_question, number_part, single_number
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
    (
        "nv-mean-of-ordinal-labels",
        "Averages ordered labels such as poor to excellent as if they were equally spaced numbers.",
        AREA,
    ),
    (
        "nv-mode-ignores-order",
        "Gives the mode for ordered ratings, which says nothing about the order of the ratings.",
        AREA,
    ),
    ("nv-range-as-centre", "Gives the range, a measure of spread, where a measure of centre was asked for.", AREA),
    (
        "nv-mean-for-skewed",
        "Uses the mean for data with one extreme value, although the extreme value drags the mean away from a typical value.",
        AREA,
    ),
    ("nv-mean-always-best", "Believes the mean is always the best summary because it uses every value.", AREA),
    (
        "nv-median-for-total",
        "Uses the median where a total is wanted; the total comes from the mean times the number of values.",
        AREA,
    ),
    ("nv-mode-for-skewed", "Takes the most common value as the typical one, even when no value is repeated.", AREA),
    (
        "nv-removes-outlier",
        "Throws away the extreme value and averages the rest, instead of using the median of all the values.",
        AREA,
    ),
    ("nv-mean-for-most-common", "Gives the mean where the most common value was needed.", AREA),
    ("nv-median-for-most-common", "Gives the median where the most common value was needed.", AREA),
    ("nv-mode-for-equal-share", "Gives the most common value where an equal share of a total was needed.", AREA),
    (
        "nv-median-of-codes",
        "Finds the middle number of the codes used for categories and reads it as the typical category.",
        AREA,
    ),
    (
        "nv-mean-of-codes",
        "Averages the numbers used to code categories and names the category nearest that number.",
        AREA,
    ),
    (
        "nv-claim-accepted",
        "Accepts the claimed summary without checking that it suits the variable and the data.",
        AREA,
    ),
    ("nv-median-unsorted", "Takes the middle of the list as written, without putting the values in order first.", AREA),
    (
        "nv-median-one-middle",
        "Takes one of the two middle values of an even count instead of the number halfway between them.",
        AREA,
    ),
    ("nv-mean-wrong-count", "Divides the total by a count that is one out.", AREA),
    ("nv-difference-reversed", "Subtracts in the wrong order, giving the median minus the mean.", AREA),
    ("nv-total-not-scaled", "Gives the mean itself where the total for many items was asked.", AREA),
    ("nv-total-divided", "Divides the mean by the number of items where the total was asked.", AREA),
]

MEAN, MEDIAN, MODE, RANGE = "Mean", "Median", "Mode", "Range"
_MEASURES = (MEAN, MEDIAN, MODE, RANGE)

# (what is listed, low, high, outlier low, outlier high, singular noun for a stem)
_SKEWED = [
    ("the weekly pocket money, in dollars, of ten students", 8, 30, 90, 150, "weekly pocket money"),
    ("the sale prices, in thousands of dollars, of ten houses on one street", 520, 760, 1700, 2500, "house price"),
    ("the waiting times, in minutes, of ten patients at a clinic", 6, 30, 90, 140, "waiting time"),
    ("the annual salaries, in thousands of dollars, of ten staff at a small firm", 52, 88, 260, 420, "salary"),
    ("the numbers of text messages sent in one day by ten teenagers", 12, 60, 190, 300, "number of messages"),
    ("the daily numbers of visitors to a small museum over ten days", 40, 95, 260, 400, "number of daily visitors"),
]
# (what is listed, low, high, per-item noun for a total, unit of a total)
_BALANCED = [
    ("the masses, in grams, of ten packets of rice", 494, 508, "packet", "grams"),
    ("the heights, in centimetres, of ten Year 9 students", 156, 176, "student", "centimetres"),
    ("the daily maximum temperatures, in degrees Celsius, over ten days in March", 21, 33, "day", "degrees"),
    ("the times, in minutes, that ten people took to finish a puzzle", 22, 40, "person", "minutes"),
    ("the daily takings, in dollars, of a market stall over ten days", 410, 560, "day", "dollars"),
]
# (what is listed, low, high, what the most common value is used for)
_COUNTS = [
    ("the shoe sizes of ten players in a netball club", 5, 11, "the size the club should order most of"),
    ("the numbers of pets in each of ten households", 0, 5, "the number of pets that most households have"),
    ("the numbers of goals scored by a team in ten games", 0, 6, "the score the team reaches most often"),
    ("the numbers of passengers in each of ten cars at a toll gate", 1, 6, "the number of people most cars carry"),
]

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
]
_NOMINAL = [
    ("A school canteen records the lunch (pie, sandwich, salad or sushi) that each student buys.", "the typical lunch"),
    (
        "A council records the type of dwelling (house, unit, townhouse or apartment) of each household in a suburb.",
        "the typical dwelling",
    ),
    (
        "A sports club records the main sport (netball, cricket, tennis or swimming) of each junior member.",
        "the typical sport",
    ),
    ("A tourism survey records the country each visitor came from.", "the typical country"),
]
_CODED = [
    ("how students travel to school", ["bus", "train", "walk", "car"]),
    ("the type of pet in each household", ["dog", "cat", "bird", "fish"]),
    ("the main language spoken at home by each student", ["English", "Mandarin", "Arabic", "Vietnamese"]),
    ("the flavour of pie chosen at a bakery", ["beef", "chicken", "vegetable", "apple"]),
]


# -- data -------------------------------------------------------------------------------------------------------------


def _plural(noun: str) -> str:
    return {"person": "people"}.get(noun, noun + "s")


def _mean(values) -> Fraction:
    return Fraction(sum(values), len(values))


def _median(values) -> Fraction:
    s = sorted(values)
    n = len(s)
    check(n % 2 == 0, "these data sets have an even count")
    return Fraction(s[n // 2 - 1] + s[n // 2], 2)


def _shown(values) -> str:
    return ", ".join(str(v) for v in values)


def _check_stats(values, mean, median):
    check(mean == Fraction(statistics.fmean(values)).limit_denominator(1000), "mean differs")
    check(median == Fraction(statistics.median(values)).limit_denominator(1000), "median differs")


def _skewed(rng, ctx):
    _, lo, hi, olo, ohi, _ = ctx
    for _ in attempts(400, "a skewed data set"):
        base = rng.sample(range(lo, hi + 1), 9)
        out = rng.randint(olo, ohi)
        out += (-(sum(base) + out)) % 5
        values = base + [out]
        mean, median = _mean(values), _median(values)
        if mean - median >= Fraction(1, 4) * median:
            _check_stats(values, mean, median)
            rng.shuffle(values)
            return values, mean, median
    raise AssertionError  # pragma: no cover


def _balanced(rng, ctx):
    _, lo, hi, _, _ = ctx
    for _ in attempts(2000, "a balanced data set"):
        values = rng.sample(range(lo, hi + 1), 10)
        mean, median = _mean(values), _median(values)
        gap = abs(mean - median)
        if (
            sum(values) % 5 == 0
            and Fraction(3, 10) <= gap <= Fraction(1, 25) * median
            and min(values) * 1.4 > max(values) * 0.9
        ):
            _check_stats(values, mean, median)
            return values, mean, median
    raise AssertionError  # pragma: no cover


def _counts(rng, ctx):
    _, lo, hi, _ = ctx
    for _ in attempts(4000, "a repeated count"):
        values = [rng.randint(lo, hi) for _ in range(10)]
        top = max(set(values), key=values.count)
        if values.count(top) < 3 or sorted(values.count(v) for v in set(values))[-2] == values.count(top):
            continue
        mean, median = _mean(values), _median(values)
        if (
            sum(values) % 5 == 0
            and len({mean, median, Fraction(top)}) == 3
            and abs(mean - median) >= Fraction(3, 10)
            and abs(median - top) >= Fraction(1, 2)
            and abs(mean - top) >= Fraction(1, 2)
        ):
            _check_stats(values, mean, median)
            check(statistics.multimode(values) == [top], "mode differs")
            return values, mean, median, top
    raise AssertionError  # pragma: no cover


def _median_unsorted(values) -> Fraction:
    return Fraction(values[4] + values[5], 2)


def _median_slips(values, mean, median):
    s = sorted(values)
    return [
        (mean, "nv-mean-for-skewed"),
        (Fraction(sum(s[:-1]), 9), "nv-removes-outlier"),
        (_median_unsorted(values), "nv-median-unsorted"),
        (Fraction(s[5]), "nv-median-one-middle"),
        (Fraction(s[4]), "nv-median-one-middle"),
    ]


def _mean_slips(values, mean, median, mode=None):
    out = [
        (median, "nv-median-for-total"),
        (Fraction(sum(values), len(values) - 1), "nv-mean-wrong-count"),
        (Fraction(sum(values), len(values) + 1), "nv-mean-wrong-count"),
        (Fraction(max(values) - min(values)), "nv-range-as-centre"),
    ]
    if mode is not None:
        out.insert(1, (Fraction(mode), "nv-mode-for-equal-share"))
    return out


def _mode_slips(values, mean, median):
    return [
        (mean, "nv-mean-for-most-common"),
        (median, "nv-median-for-most-common"),
        (Fraction(max(values) - min(values)), "nv-range-as-centre"),
        (Fraction(max(values)), "nv-range-as-centre"),
    ]


# -- easy -------------------------------------------------------------------------------------------------------------


def _measure_choice(rng, text, key, working, slips):
    return choice_question(text, key, working, slips)


def _easy_nominal(rng):
    scene, thing = rng.choice(_NOMINAL)
    text = (
        scene
        + " "
        + phrase(
            rng,
            (
                "Which summary could be used to describe {t}?",
                "Which measure of centre makes sense for this variable?",
                "A report wants one summary of {t}. Which one can be found for these data?",
            ),
            t=thing,
        )
    )
    slips = [
        (MEAN, "nv-mean-of-categories"),
        (MEDIAN, "nv-median-of-unordered-categories"),
        (RANGE, "nv-range-as-centre"),
    ]
    return _measure_choice(
        rng,
        text,
        MODE,
        ["The categories have no order and are not numbers, so only the most common category, the mode, can be found."],
        slips,
    )


def _easy_ordinal(rng):
    scene, thing = rng.choice(_ORDINAL)
    text = (
        scene
        + " "
        + phrase(
            rng,
            (
                "Which measure of centre can be found for {t} AND uses the order of the categories?",
                "Which summary of {t} takes account of their order without treating them as numbers to be added?",
            ),
            t=thing,
        )
    )
    slips = [(MEAN, "nv-mean-of-ordinal-labels"), (MODE, "nv-mode-ignores-order"), (RANGE, "nv-range-as-centre")]
    return _measure_choice(
        rng,
        text,
        MEDIAN,
        [
            "The ratings can be put in order, so the middle one, the median, can be found.",
            "Labels cannot be added, so no mean exists, and the mode ignores the order.",
        ],
        slips,
    )


def _easy_skewed(rng):
    ctx = rng.choice(_SKEWED)
    values, mean, median = _skewed(rng, ctx)
    text = phrase(
        rng,
        (
            "Here are {what}: {v}. Which measure of centre best describes a typical {noun}?",
            "A newspaper lists {what}: {v}. One value is much larger than the rest. Which measure of centre gives the fairest picture of a typical {noun}?",
            "The values are {what}: {v}. Which summary should be used to show a typical {noun}?",
        ),
        what=ctx[0],
        v=_shown(values),
        noun=ctx[5],
    )
    slips = [(MEAN, "nv-mean-for-skewed"), (MODE, "nv-mode-for-skewed"), (RANGE, "nv-range-as-centre")]
    return _measure_choice(
        rng,
        text,
        MEDIAN,
        [
            f"The mean is {float(mean):.1f}, pulled up by the large value, but the median is {float(median):g}.",
            "The median is not distorted by one extreme value.",
        ],
        slips,
    )


def _easy_total(rng):
    ctx = rng.choice(_BALANCED)
    values, mean, median = _balanced(rng, ctx)
    share = rng.random() < 0.5
    goal = (
        "The organiser wants to estimate the total {u} for 40 {n} of the same kind."
        if not share
        else "The {n} are to be treated as one group whose total is shared equally between them."
    ).format(u=ctx[4], n=_plural(ctx[3]))
    text = f"The data are {ctx[0]}: {_shown(values)}. {goal} Which measure of centre should be used?"
    slips = [(MEDIAN, "nv-median-for-total"), (MODE, "nv-mode-for-equal-share"), (RANGE, "nv-range-as-centre")]
    return _measure_choice(
        rng,
        text,
        MEAN,
        [
            f"A total is the mean times the number of values: the mean ({float(mean):g}) uses every value, so it gives the total."
        ],
        slips,
    )


def _easy_common(rng):
    ctx = rng.choice(_COUNTS)
    values, mean, median, mode = _counts(rng, ctx)
    text = f"Here are {ctx[0]}: {_shown(values)}. Which measure of centre gives {ctx[3]}?"
    slips = [(MEAN, "nv-mean-for-most-common"), (MEDIAN, "nv-median-for-most-common"), (RANGE, "nv-range-as-centre")]
    return _measure_choice(rng, text, MODE, [f"The value that appears most often is {mode}, the mode."], slips)


_EASY = (_easy_nominal, _easy_ordinal, _easy_skewed, _easy_skewed, _easy_total, _easy_total, _easy_common)


# -- medium -----------------------------------------------------------------------------------------------------------


def _fmt(x) -> str:
    return f"{float(x):.1f}"


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
    return single_number(text, median, 1, working, _median_slips(values, mean, median))


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
    return single_number(text, mean, 1, working, _mean_slips(values, mean, median))


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
        f"Count how often each value appears: the value {mode} is listed {values.count(mode)} times, more than any other.",
        f"The mode is {mode}; the mean ({_fmt(mean)}) and median ({_fmt(median)}) are not values that were recorded most often.",
    ]
    return single_number(text, Fraction(mode), 1, working, _mode_slips(values, mean, median))


_MEDIUM = (_medium_median, _medium_median, _medium_mean, _medium_mean, _medium_mode)


# -- hard -------------------------------------------------------------------------------------------------------------

# Per situation: the measure that suits, and for each measure a reason sentence (the right one for the right measure,
# a tempting wrong one for the others), all about the same length.
_REASONS = {
    "skewed": {
        MEDIAN: "Use the median: the one very large value drags the mean upward but barely moves the middle value.",
        MEAN: "Keep the mean: it uses every value in the list, so it is always the fairest measure of a typical one.",
        MODE: "Use the mode: the value that appears most often in a list is always the most typical value in it.",
        RANGE: "Use the range: the difference between the largest and smallest values shows how big a typical one is.",
    },
    "ordinal": {
        MEDIAN: "Use the median: the ratings can be put in order, so the middle rating is meaningful and is named.",
        MEAN: "Keep the mean: giving each rating a number from 1 to 5 lets every response count in the average.",
        MODE: "Use the mode: the most common rating is the one most people chose, so it is the typical rating.",
        RANGE: "Use the range: the gap between the highest and the lowest rating shows what a typical rating is.",
    },
    "total": {
        MEAN: "Use the mean: the total is the mean times the number of values, which the median cannot give you.",
        MEDIAN: "Keep the median: large or small values do not affect it, so it is the safest measure to scale up.",
        MODE: "Use the mode: the most common value, repeated for each item, gives a simple estimate of the total.",
        RANGE: "Use the range: scaling up the difference between the largest and smallest values gives the total.",
    },
}


def _hard_claim_mcq(rng):
    kind = rng.choice(["skewed", "skewed", "ordinal", "total", "total"])
    person = rng.choice(["Ravi", "Mia", "Kofi", "Hannah", "Liam", "Zara", "Tom", "Aisha"])
    if kind == "skewed":
        ctx = rng.choice(_SKEWED)
        values, mean, median = _skewed(rng, ctx)
        claimed = MEAN
        text = f'{person} says: "The mean, {_fmt(mean)}, is the typical {ctx[5]}." The values are {ctx[0]}: {_shown(values)}. Which response is best?'
        working = [
            f"Ordered, the values are {_shown(sorted(values))}: the median is {_fmt(median)}.",
            "One very large value pulls the mean above almost all the others, so the claim does not suit the data.",
        ]
    elif kind == "ordinal":
        scene, thing = rng.choice(_ORDINAL)
        claimed = MEAN
        avg = rng.choice(["2.6", "3.1", "3.4", "3.8"])
        text = f'{scene} {person} codes the responses 1 to 5 and says: "The mean response is {avg}, so that is the typical rating." Which response is best?'
        working = [
            "The responses are ordered labels, not measured amounts: the gaps between them are not equal.",
            "The median uses the order without treating the labels as numbers.",
        ]
    else:
        ctx = rng.choice(_BALANCED)
        values, mean, median = _balanced(rng, ctx)
        claimed = MEDIAN
        text = f"{person} wants to estimate the total {ctx[4]} for 50 {_plural(ctx[3])} from {ctx[0]}: {_shown(values)}, and plans to multiply the median, {_fmt(median)}, by 50. Which response is best?"
        working = [
            f"The total of the ten values is {sum(values)}, so the mean is {_fmt(mean)}.",
            "Only the mean links a typical value to a total.",
        ]
    best = {"skewed": MEDIAN, "ordinal": MEDIAN, "total": MEAN}[kind]
    reasons = _REASONS[kind]
    wrong_tags = {
        "skewed": {MEAN: "nv-mean-always-best", MODE: "nv-mode-for-skewed", RANGE: "nv-range-as-centre"},
        "ordinal": {MEAN: "nv-mean-of-ordinal-labels", MODE: "nv-mode-ignores-order", RANGE: "nv-range-as-centre"},
        "total": {MEDIAN: "nv-median-for-total", MODE: "nv-mode-for-equal-share", RANGE: "nv-range-as-centre"},
    }[kind]
    slips = [(reasons[m], wrong_tags[m]) for m in _MEASURES if m != best]
    # the claimed measure is the most tempting wrong option, whatever its place in the list
    slips.sort(key=lambda s: s[0] != reasons[claimed])
    # the options of one question are shown in a random order by the app; the key is not always the longest or the first
    return choice_question(text, reasons[best], working, slips)


def _hard_claim_number(rng):
    kind = rng.choice(["skewed", "total", "common"])
    if kind == "skewed":
        ctx = rng.choice(_SKEWED)
        values, mean, median = _skewed(rng, ctx)
        text = f"A report says the typical {ctx[5]} is {_fmt(mean)}, taken from {ctx[0]}: {_shown(values)}. Work out the measure of centre that describes a typical {ctx[5]} better than that claim, to 1 decimal place."
        working = [
            f"The claim is the mean, {_fmt(mean)}, which one very large value has pulled up.",
            f"Ordered: {_shown(sorted(values))}. The median is {_fmt(median)}.",
        ]
        return single_number(text, median, 1, working, _median_slips(values, mean, median))
    if kind == "total":
        ctx = rng.choice(_BALANCED)
        values, mean, median = _balanced(rng, ctx)
        k = rng.choice([25, 40, 60])
        text = f"A planner used the median, {_fmt(median)}, from {ctx[0]}: {_shown(values)}, to estimate a total for {k} {_plural(ctx[3])}. Work out the figure that should be multiplied by {k} instead, to 1 decimal place."
        working = [
            f"The sum of the values is {sum(values)}, so the mean is {_fmt(mean)}.",
            "A total for many items is the mean times the number of items.",
        ]
        return single_number(text, mean, 1, working, _mean_slips(values, mean, median))
    ctx = rng.choice(_COUNTS)
    values, mean, median, mode = _counts(rng, ctx)
    text = f"A teacher reports {_fmt(mean)} as {ctx[3]}, the mean of {ctx[0]}: {_shown(values)}. Find the value that does answer the question, to 1 decimal place."
    working = [
        f"The question asks for the most common value, not the mean ({_fmt(mean)}).",
        f"The value {mode} is listed {values.count(mode)} times, more than any other value.",
    ]
    return single_number(
        text,
        Fraction(mode),
        1,
        working,
        [(median, "nv-median-for-most-common"), (mean, "nv-claim-accepted")] + _mode_slips(values, mean, median)[2:],
    )


def _hard_coded_category(rng):
    scene, names = rng.choice(_CODED)
    for _ in attempts(500, "coded categories whose mean and median codes are not the mode"):
        codes = [rng.randint(1, 4) for _ in range(rng.choice([11, 13, 15]))]
        counts = {c: codes.count(c) for c in range(1, 5)}
        top = max(counts, key=counts.get)
        if sorted(counts.values())[-1] == sorted(counts.values())[-2]:
            continue
        mean = Fraction(sum(codes), len(codes))
        near = min(range(1, 5), key=lambda c: (abs(c - mean), c))
        mid = sorted(codes)[len(codes) // 2]
        if len({top, near, mid}) == 3 and abs(mean - near) < Fraction(1, 2) and abs(mean - near) != Fraction(1, 2):
            break
    key = names[top - 1]
    text = (
        f"A survey recorded {scene}, coded "
        + ", ".join(f"{i + 1} = {n}" for i, n in enumerate(names))
        + f". The codes for {len(codes)} people are: {_shown(codes)}. "
        f"A student finds that the mean code is {float(mean):.1f}, so says the typical response is {names[near - 1]}. What is the typical (most common) response?"
    )
    others = [c for c in range(1, 5) if c not in (top, near, mid)]
    slips = [(names[near - 1], "nv-mean-of-codes"), (names[mid - 1], "nv-median-of-codes")] + [
        (names[c - 1], "nv-claim-accepted") for c in others
    ]
    working = [
        f"The codes are labels, so a mean code has no meaning.",
        "Tally the codes: "
        + ", ".join(f"{n} {codes.count(i + 1)}" for i, n in enumerate(names))
        + f". The most common is {key}.",
    ]
    return choice_question(text, key, working, slips)


_HARD = (_hard_claim_mcq, _hard_claim_mcq, _hard_claim_number, _hard_claim_number, _hard_coded_category)


@generator(
    TOPIC,
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def nv_summary_choice(rng, tier="easy"):
    """Choosing and working out mode, median or mean; the tiers are described in the module docstring."""
    pool = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# -- exam -------------------------------------------------------------------------------------------------------------

_EXAM_BEST = {
    "skewed": (
        {
            MEDIAN: "The median, because the one very large value pulls the mean up but barely moves the median",
            MEAN: "The mean, because it uses every value and so always describes the typical value best",
            MODE: "The mode, because the most common value is the typical one in any list of values",
            RANGE: "The range, because the difference between the largest and smallest values shows the typical value",
        },
        MEDIAN,
        {MEAN: "nv-mean-always-best", MODE: "nv-mode-for-skewed", RANGE: "nv-range-as-centre"},
    ),
    "total": (
        {
            MEAN: "The mean, because the total is the mean times the number of items and the values are balanced",
            MEDIAN: "The median, because it is the middle value and so it is always the safest one to scale up",
            MODE: "The mode, because the most common value repeated for every item gives the total needed",
            RANGE: "The range, because scaling up the difference between the extreme values gives the total",
        },
        MEAN,
        {MEDIAN: "nv-median-for-total", MODE: "nv-mode-for-equal-share", RANGE: "nv-range-as-centre"},
    ),
}


@generator(EXAM_TOPIC, level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_nv_summary_choice(rng, tier="exam"):
    """Exam chain on ten listed values: mean, median, a figure that follows from them, and which measure suits."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _exam_chain)


def _exam_chain(rng):
    kind = "skewed" if rng.random() < 0.5 else "total"
    if kind == "skewed":
        ctx = rng.choice(_SKEWED)
        values, mean, median = _skewed(rng, ctx)
        intro = f"A researcher lists {ctx[0]}: {_shown(values)}. The aim is to report a typical {ctx[5]}."
        c_prompt = "Find the mean minus the median."
        diff = mean - median
        c_working = [f"{_fmt(mean)} - {_fmt(median)} = {_fmt(diff)}."]
        c_slips = [
            (median - mean, "nv-difference-reversed"),
            (mean, "nv-total-not-scaled"),
            (median, "nv-total-not-scaled"),
        ]
        c_ft = "part_a - part_b"
        c_value, c_range = diff, (None, None)
    else:
        ctx = rng.choice(_BALANCED)
        values, mean, median = _balanced(rng, ctx)
        k = rng.choice([40, 50, 60, 80])
        intro = f"A planner lists {ctx[0]}: {_shown(values)}. The aim is to estimate the total {ctx[4]} for {k} {_plural(ctx[3])} of the same kind."
        c_prompt = f"Use the mean to estimate the total {ctx[4]} for {k} {_plural(ctx[3])}."
        c_value = mean * k
        c_working = [f"{_fmt(mean)} x {k} = {float(c_value):.1f}."]
        c_slips = [
            (median * k, "nv-median-for-total"),
            (mean, "nv-total-not-scaled"),
            (mean / k, "nv-total-divided"),
            (Fraction(sum(values)), "nv-total-not-scaled"),
        ]
        c_ft = f"part_a * {k}"
        c_range = (0, None)
    s = sorted(values)
    check(sum(s) == sum(values) and mean == Fraction(sum(values), 10), "mean differs")
    a = number_part(
        "Find the mean, to 1 decimal place.",
        mean,
        1,
        2,
        [f"The sum is {sum(values)}, so the mean is {sum(values)} / 10 = {_fmt(mean)}."],
        [
            (Fraction(sum(values), 9), "nv-mean-wrong-count"),
            (Fraction(sum(values), 11), "nv-mean-wrong-count"),
            (median, "nv-claim-accepted"),
            (Fraction(sum(values), 100), "nv-mean-wrong-count"),
        ],
    )
    b = number_part(
        "Find the median, to 1 decimal place.",
        median,
        1,
        2,
        [f"Ordered: {_shown(s)}.", f"The median is halfway between {s[4]} and {s[5]}: {_fmt(median)}."],
        [
            (_median_unsorted(values), "nv-median-unsorted"),
            (Fraction(s[4]), "nv-median-one-middle"),
            (Fraction(s[5]), "nv-median-one-middle"),
            (mean, "nv-claim-accepted"),
        ],
    )
    c = number_part(
        c_prompt + " Give your answer to 1 decimal place.",
        c_value,
        1,
        2,
        c_working,
        c_slips,
        follow_through=c_ft,
        sensible_range=c_range,
    )
    options, key, tags = _EXAM_BEST[kind]
    slips = [(options[m], tags[m]) for m in _MEASURES if m != key]
    d = choice_part("Which measure of centre best suits the aim?", options[key], 2, [f"{options[key]}."], slips)
    return dict(text=intro, parts=[a, b, c, d])
