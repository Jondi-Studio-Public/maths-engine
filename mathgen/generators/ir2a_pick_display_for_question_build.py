"""Generates `pick-display-for-question` questions: choose the display that answers a question, then use the data.

Each question states a question to be answered and the data collected for it (as lists, nothing is drawn), asks which
of five displays would answer it best, and then asks for one number that the chosen display would show:

  change over time              line graph             value in a month; largest rise; average yearly change
  association of two variables  scatterplot            value predicted from the stated line of fit
  share of a whole              pie chart              percentage; sector angle
  counts in many categories     column graph           difference between two columns
  centre of two groups          parallel box plots     difference in medians; difference in interquartile ranges
  shape of a spread of values   histogram              number in a class; percentage in a class

Only the right display answers the question as stated, and the stem says what the data are measured as and what is
wanted, never the name of the display.

Ladder (reasoning escalation, not size):
  easy:   time, association, share or counts; the number is one value read from the data.
  medium: adds the centre of two groups and the shape of a spread; the number needs a small calculation (a median,
          a count in a class, a value from a stated line).
  hard:   the question is disguised (years as the time axis, a count that is really the horizontal variable, a
          comparison of spread rather than centre) and the number takes two steps.
  exam:   a four-part chain on two groups (display, two medians, their difference) or on a time series (display,
          total rise, average monthly change, the next month's value).
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import VerificationError, attempts, check, generator

from ._ir2a_kit import NAMES, chain, join_and, mcq_part, median_of, number_part, quartiles, single

_TIERS = ("easy", "medium", "hard")

COLUMN, PIE, HIST, LINE, SCATTER, BOX = (
    "A column graph",
    "A pie chart",
    "A histogram",
    "A line graph",
    "A scatterplot",
    "Parallel box plots",
)
DISPLAYS = (COLUMN, PIE, HIST, LINE, SCATTER, BOX)
WRONG_TAG = {
    COLUMN: "ir2a-picks-a-column-graph-for-the-wrong-purpose",
    PIE: "ir2a-picks-a-pie-chart-for-the-wrong-purpose",
    HIST: "ir2a-picks-a-histogram-for-the-wrong-purpose",
    LINE: "ir2a-picks-a-line-graph-for-the-wrong-purpose",
    SCATTER: "ir2a-picks-a-scatterplot-for-the-wrong-purpose",
    BOX: "ir2a-picks-box-plots-for-the-wrong-purpose",
}
#: The displays most easily mistaken for each right one, most tempting first.
TEMPTING = {
    LINE: (COLUMN, SCATTER, PIE, HIST, BOX),
    SCATTER: (LINE, COLUMN, HIST, BOX, PIE),
    PIE: (COLUMN, HIST, LINE, BOX, SCATTER),
    COLUMN: (PIE, HIST, LINE, SCATTER, BOX),
    BOX: (HIST, COLUMN, SCATTER, LINE, PIE),
    HIST: (COLUMN, BOX, PIE, LINE, SCATTER),
}


def _upper_first(text: str) -> str:
    """Capitalise the first letter only, so "brand A batteries" becomes "Brand A batteries"."""
    return text[:1].upper() + text[1:]


def _display_part(rng, key: str, marks: int = 1) -> dict:
    order = list(TEMPTING[key])
    chosen = order[:3] + [rng.choice(order[3:])]
    return mcq_part(
        "Which display would answer the question best?",
        key,
        [(d, WRONG_TAG[d]) for d in chosen],
        marks=marks,
        working=[f"{key} {'show' if key == BOX else 'shows'} what the question asks about."],
        keep=4,
    )


def _nums(values) -> str:
    return ", ".join(str(v) for v in values)


# -- the six purposes -----------------------------------------------------------------------------------------------

_MONTHS = ("January", "February", "March", "April", "May", "June")
_SERIES = (
    ("the monthly sales of a bookshop, in hundreds of dollars", "sales", 20, 60),
    ("the number of visitors to a museum each month, in hundreds", "visitors", 15, 55),
    ("the monthly water use of a school, in kilolitres", "water use", 30, 80),
    ("the number of items sold at a school canteen each month, in hundreds", "items sold", 10, 40),
)


def _series(rng, n: int, lo: int, hi: int, step_multiple: int | None = None):
    for _ in attempts(500, "a time series with distinct values"):
        start = rng.randint(lo, lo + 10)
        vals = [start]
        for _i in range(n - 1):
            vals.append(max(lo, min(hi, vals[-1] + rng.randint(-6, 12))))
        if len(set(vals)) == n and (step_multiple is None or (vals[-1] - vals[0]) % step_multiple == 0):
            return vals
    raise AssertionError("unreachable")


def _time_easy(rng, name):
    desc, noun, lo, hi = rng.choice(_SERIES)
    vals = _series(rng, 6, lo, hi)
    j = rng.randrange(1, 6)
    question = f"how {noun} changed from month to month over the first half of the year"
    data = "; ".join(f"{m} {v}" for m, v in zip(_MONTHS, vals))
    stem = f"{name} has recorded {desc}: {data}. {name} wants to show {question}."
    b = number_part(
        prompt=f"What value was recorded for {_MONTHS[j]}?",
        exact=vals[j],
        places=0,
        marks=1,
        working=[f"{_MONTHS[j]}: {vals[j]}."],
        wrong=[
            (vals[j - 1], "ir2a-reads-the-neighbouring-value"),
            (vals[(j + 1) % 6], "ir2a-reads-the-neighbouring-value"),
            (vals[j] - vals[j - 1], "ir2a-reports-the-change-not-the-value"),
        ],
        sensible_range=(0, None),
    )
    return LINE, stem, b


def _time_medium(rng, name):
    desc, noun, lo, hi = rng.choice(_SERIES)
    vals = _series(rng, 6, lo, hi)
    rises = [vals[i + 1] - vals[i] for i in range(5)]
    best = max(range(5), key=lambda i: rises[i])
    check(sorted(rises)[-1] > sorted(rises)[-2], "the largest rise must be unique")
    data = "; ".join(f"{m} {v}" for m, v in zip(_MONTHS, vals))
    stem = f"{name} has recorded {desc}: {data}. {name} wants to show how {noun} changed through the first half of the year."
    b = number_part(
        prompt="What was the largest rise from one month to the next?",
        exact=rises[best],
        places=0,
        marks=1,
        working=[f"Month-to-month changes: {_nums(rises)}.", f"Largest rise = {rises[best]}."],
        wrong=[
            (vals[-1] - vals[0], "ir2a-uses-first-and-last-only"),
            (max(vals), "ir2a-reports-the-largest-value"),
            (max(vals) - min(vals), "ir2a-reports-the-range"),
        ],
        sensible_range=(0, None),
    )
    check(rises[best] > 0, "the largest change must be a rise")
    return LINE, stem, b


def _time_hard(rng, name):
    first = rng.choice([2016, 2017, 2018])
    n = rng.choice([4, 5, 6])
    last = first + n - 1
    desc, noun, lo, hi = rng.choice(_SERIES)
    vals = _series(rng, n, lo, hi, step_multiple=n - 1)
    data = "; ".join(f"{first + i}: {v}" for i, v in enumerate(vals))
    yearly = desc.replace("monthly ", "").replace(" each month", "")
    stem = (
        f"For each year from {first} to {last}, a council recorded {yearly}: {data}. "
        f"It wants to know whether the {noun} has been growing, and how fast."
    )
    avg = (vals[-1] - vals[0]) // (n - 1)
    b = number_part(
        prompt=f"What was the average change in {noun} per year from {first} to {last}?",
        exact=avg,
        places=0,
        marks=2,
        working=[f"({vals[-1]} - {vals[0]}) / {n - 1} = {avg}."],
        wrong=[
            (Fraction(vals[-1] - vals[0], n), "ir2a-divides-by-the-number-of-years-listed"),
            (vals[-1] - vals[0], "ir2a-forgets-to-divide-by-the-intervals"),
            (Fraction(vals[-1], vals[0]), "ir2a-divides-instead-of-subtracting"),
        ],
        sensible_range=None,
    )
    return LINE, stem, b


_ASSOC = (
    ("hours of study", "test mark", "hours", "marks", 40, 3, 1, 15),
    ("hours of training each week", "number of goals scored in a season", "hours", "goals", 2, 3, 2, 16),
    (
        "daily maximum temperature in degrees Celsius",
        "number of ice creams sold",
        "degrees",
        "ice creams",
        -60,
        8,
        20,
        38,
    ),
    (
        "age of a car in years",
        "advertised price in hundreds of dollars",
        "years",
        "hundreds of dollars",
        240,
        -15,
        1,
        14,
    ),
)


def _assoc(rng, name, level):
    xdesc, ydesc, xu, yu, a0, b0, xlo, xhi = rng.choice(_ASSOC)
    xs = sorted(rng.sample(range(xlo, xhi + 1), 8))
    ys = [a0 + b0 * x + rng.randint(-4, 4) for x in xs]
    pairs = "; ".join(f"({x}, {y})" for x, y in zip(xs, ys))
    free = [x for x in range(xs[1] + 1, xs[-2]) if x not in xs]
    check(bool(free), "no unused x-value inside the data range")
    x_new = rng.choice(free)
    if level == "easy":
        j = rng.randrange(8)
        stem = f"{name} recorded the {xdesc} and the {ydesc} for 8 cases, as (x, y) pairs: {pairs}. {name} wants to see whether the two are related."
        b = number_part(
            prompt=f"What was the {ydesc} when the {xdesc} was {xs[j]}?",
            exact=ys[j],
            places=0,
            marks=1,
            working=[f"The pair with x = {xs[j]} is ({xs[j]}, {ys[j]})."],
            wrong=[
                (ys[j - 1], "ir2a-reads-the-neighbouring-value"),
                (xs[j], "ir2a-reads-x-for-y"),
                (ys[(j + 1) % 8], "ir2a-reads-the-neighbouring-value"),
            ],
            sensible_range=None,
        )
        return SCATTER, stem, b
    line = f"y = {a0} {'+' if b0 > 0 else '-'} {abs(b0)}x" if a0 != 0 else f"y = {b0}x"
    pred = a0 + b0 * x_new
    stem = (
        f"{name} recorded the {xdesc} (x) and the {ydesc} (y) for 8 cases, as (x, y) pairs: {pairs}. "
        f"{name} wants to see whether the two are related, and to predict from one to the other. A line of fit for the data is {line}."
    )
    b = number_part(
        prompt=f"Use the line of fit to predict the {ydesc} when the {xdesc} is {x_new}.",
        exact=pred,
        places=0,
        marks=2,
        working=[f"y = {a0} {'+' if b0 > 0 else '-'} {abs(b0)} x {x_new} = {pred}."],
        wrong=[
            (b0 * x_new, "ir2a-forgets-the-intercept"),
            (a0 - b0 * x_new, "ir2a-flips-the-sign-of-the-slope-term"),
            (a0 + b0 + x_new, "ir2a-adds-instead-of-multiplying"),
        ],
        sensible_range=None,
    )
    return SCATTER, stem, b


_SHARE = (
    ("students at a school choosing their main way of getting to school", ["Walk", "Bus", "Car", "Bike", "Train"]),
    ("customers at a cafe choosing their drink", ["Coffee", "Tea", "Juice", "Water", "Milkshake"]),
    ("a household's yearly spending", ["Housing", "Food", "Transport", "Savings", "Other"]),
    (
        "the languages spoken at home by students in a year level",
        ["English", "Mandarin", "Arabic", "Vietnamese", "Other"],
    ),
)


def _share(rng, name, level):
    desc, cats = rng.choice(_SHARE)
    total = rng.choice([100, 200, 400, 500])
    for _ in attempts(500, "category counts"):
        k = rng.randint(4, 5)
        pcts = sorted(rng.sample(range(10, 50, 5), k - 1), reverse=True)
        rest = 100 - sum(pcts)
        if rest >= 10 and rest not in pcts:
            pcts.append(rest)
            break
    cs = cats[: len(pcts)]
    counts = [total * p // 100 for p in pcts]
    check(sum(counts) == total and all(total * p % 100 == 0 for p in pcts), "shares must be whole")
    j = rng.randrange(len(cs))
    data = "; ".join(f"{c} {n}" for c, n in zip(cs, counts))
    if level == "easy":
        stem = f"{name} has counted {desc}, out of {total}: {data}. {name} wants to show how the whole is shared among the groups."
        b = number_part(
            prompt=f"What percentage of the {total} is in the group '{cs[j]}'?",
            exact=pcts[j],
            places=0,
            marks=1,
            working=[f"{counts[j]} / {total} x 100 = {pcts[j]}%."],
            wrong=[
                (counts[j], "ir2a-reports-the-count-as-a-percentage"),
                (100 - pcts[j], "ir2a-gives-the-rest-of-the-whole"),
                (Fraction(counts[j] * 100, total - counts[j]), "ir2a-divides-by-the-rest-of-the-whole"),
            ],
            sensible_range=(0, 100),
        )
    else:
        stem = f"{name} has counted {desc}, out of {total}: {data}. {name} wants to show what part of the whole each group is, so that the shares can be compared as angles."
        angle = 360 * pcts[j] // 100
        b = number_part(
            prompt=f"What angle would the sector for '{cs[j]}' have, in degrees?",
            exact=angle,
            places=0,
            marks=2,
            working=[f"{counts[j]} / {total} x 360 = {angle}."],
            wrong=[
                (pcts[j], "ir2a-gives-the-percentage-as-the-angle"),
                (counts[j], "ir2a-reports-the-count-as-a-percentage"),
                (360 - angle, "ir2a-gives-the-rest-of-the-whole"),
            ],
            sensible_range=(0, 360),
        )
    return PIE, stem, b


_COUNTS = (
    (
        "students choosing their favourite sport",
        ["Netball", "Football", "Cricket", "Swimming", "Tennis", "Athletics", "Basketball"],
    ),
    (
        "books borrowed from a library in a week, by genre",
        ["Fantasy", "Crime", "History", "Science", "Romance", "Biography", "Poetry"],
    ),
    (
        "birds seen on a bird count, by species",
        ["Magpie", "Galah", "Kookaburra", "Rosella", "Wattlebird", "Honeyeater", "Lorikeet"],
    ),
)


def _counts(rng, name, level):
    desc, cats = rng.choice(_COUNTS)
    counts = rng.sample(range(6, 60), 7)
    data = "; ".join(f"{c} {n}" for c, n in zip(cats, counts))
    stem = f"{name} has recorded {desc}: {data}. {name} wants to compare how many fall in each of the 7 groups."
    hi, lo = max(counts), min(counts)
    b = number_part(
        prompt="How many more are in the largest group than in the smallest group?",
        exact=hi - lo,
        places=0,
        marks=1,
        working=[f"{hi} - {lo} = {hi - lo}."],
        wrong=[
            (hi + lo, "ir2a-adds-instead-of-subtracting"),
            (hi, "ir2a-reports-the-largest-value"),
            (hi - sorted(counts)[1], "ir2a-compares-with-the-wrong-group"),
        ],
        sensible_range=(0, None),
    )
    return COLUMN, stem, b


_GROUPS = (
    (
        "the time taken to finish a puzzle (in seconds)",
        ("students who practised first", "students who did not practise"),
        70,
        150,
    ),
    ("the number of words typed in one minute", ("the morning class", "the afternoon class"), 30, 75),
    (
        "the height of a plant after eight weeks (in centimetres)",
        ("plants kept in sunlight", "plants kept in shade"),
        15,
        60,
    ),
    ("the battery life of a torch (in hours)", ("brand A batteries", "brand B batteries"), 8, 30),
)


def _two_groups(rng, n, lo, hi, min_diff=2):
    for _ in attempts(500, "two groups with distinct medians"):
        a = [rng.randint(lo, hi) for _ in range(n)]
        b = [rng.randint(lo + 5, hi + 5) for _ in range(n)]
        ma, mb = median_of(a), median_of(b)
        if abs(ma - mb) >= min_diff and len(set(a)) > n - 3 and len(set(b)) > n - 3:
            return a, b
    raise AssertionError("unreachable")


def _groups(rng, name, level):
    measure, (ga, gb), lo, hi = rng.choice(_GROUPS)
    n = 9 if level == "medium" else 11
    a, b = _two_groups(rng, n, lo, hi)
    ma, mb = median_of(a), median_of(b)
    qa, qb = quartiles(a), quartiles(b)
    stem = (
        f"{name} has measured {measure} for two groups. {_upper_first(ga)}: {_nums(a)}. {_upper_first(gb)}: {_nums(b)}. "
    )
    if level == "medium":
        stem += f"{name} wants to compare the typical value of the two groups, side by side."
        diff = abs(ma - mb)
        part = number_part(
            prompt="What is the difference between the medians of the two groups?",
            exact=diff,
            places=0,
            marks=2,
            working=[
                f"Median of the first group = {int(ma)}; median of the second group = {int(mb)}.",
                f"Difference = {int(diff)}.",
            ],
            wrong=[
                (abs(Fraction(sum(a), n) - Fraction(sum(b), n)), "ir2a-compares-means-not-medians"),
                (abs(max(a) - max(b)), "ir2a-compares-the-largest-values"),
                (abs(a[n // 2] - b[n // 2]), "ir2a-takes-the-middle-of-the-list-unsorted"),
            ],
            sensible_range=(0, None),
        )
        return BOX, stem, part
    iqa, iqb = qa[2] - qa[0], qb[2] - qb[0]
    if iqa == iqb:
        raise AssertionError("retry")
    stem += f"{name} wants to know which group is more consistent, that is, which has the smaller spread in the middle half of its values."
    part = number_part(
        prompt="What is the difference between the interquartile ranges of the two groups?",
        exact=abs(iqa - iqb),
        places=0,
        marks=3,
        working=[
            f"IQR of the first group = {int(qa[2])} - {int(qa[0])} = {int(iqa)}.",
            f"IQR of the second group = {int(qb[2])} - {int(qb[0])} = {int(iqb)}.",
            f"Difference = {int(abs(iqa - iqb))}.",
        ],
        wrong=[
            (abs((max(a) - min(a)) - (max(b) - min(b))), "ir2a-compares-ranges-not-interquartile-ranges"),
            (abs(ma - mb), "ir2a-compares-medians-not-spreads"),
            (iqa + iqb, "ir2a-adds-instead-of-subtracting"),
        ],
        sensible_range=(0, None),
    )
    return BOX, stem, part


_HIST = (
    ("the waiting time of each of 24 patients at a clinic, in minutes", 0, 39, "minutes"),
    ("the time taken by each of 24 students to walk to school, in minutes", 0, 39, "minutes"),
    ("the mass of each of 24 parcels, in kilograms", 0, 39, "kilograms"),
)


def _hist(rng, name, level):
    measure, lo, hi, unit = rng.choice(_HIST)
    vals = sorted(rng.randint(2, 38) for _ in range(24))
    cls = rng.randrange(1, 6)
    lo_c, hi_c = cls * 5, cls * 5 + 5
    count = sum(1 for v in vals if lo_c <= v < hi_c)
    check(count >= 2, "the class must hold at least two values")
    rng.shuffle(vals)
    stem = f"{name} has recorded {measure}: {_nums(vals)}. {name} wants to see the shape of the spread of the values, grouped in classes of width 5."
    if level == "medium":
        b = number_part(
            prompt=f"How many values are in the class from {lo_c} up to but not including {hi_c}?",
            exact=count,
            places=0,
            marks=2,
            working=[f"Values from {lo_c} to {hi_c - 1}: {count}."],
            wrong=[
                (sum(1 for v in vals if lo_c <= v <= hi_c), "ir2a-includes-the-upper-boundary"),
                (sum(1 for v in vals if lo_c - 5 <= v < hi_c - 5), "ir2a-reads-the-neighbouring-class"),
                (count + 1, "ir2a-miscounts-by-one"),
            ],
            sensible_range=(0, 24),
        )
        return HIST, stem, b
    pct = Fraction(count * 100, 24)
    b = number_part(
        prompt=f"What percentage of the 24 values are in the class from {lo_c} up to but not including {hi_c}? Give your answer to 1 decimal place.",
        exact=pct,
        places=1,
        marks=3,
        working=[f"{count} values: {count} / 24 x 100 = {float(pct):.1f}%."],
        wrong=[
            (count, "ir2a-reports-the-count-as-a-percentage"),
            (Fraction(count * 100, 40), "ir2a-divides-by-the-wrong-total"),
            (Fraction(sum(1 for v in vals if lo_c <= v <= hi_c) * 100, 24), "ir2a-includes-the-upper-boundary"),
        ],
        sensible_range=(0, 100),
    )
    return HIST, stem, b


def _compose(rng, name, level):
    easy = [
        lambda r: _time_easy(r, name),
        lambda r: _assoc(r, name, "easy"),
        lambda r: _share(r, name, "easy"),
        lambda r: _counts(r, name, "easy"),
    ]
    medium = [
        lambda r: _time_medium(r, name),
        lambda r: _assoc(r, name, "medium"),
        lambda r: _groups(r, name, "medium"),
        lambda r: _hist(r, name, "medium"),
        lambda r: _share(r, name, "medium"),
    ]
    hard = [
        lambda r: _time_hard(r, name),
        lambda r: _groups(r, name, "hard"),
        lambda r: _hist(r, name, "hard"),
        lambda r: _share(r, name, "medium"),
    ]
    pool = {"easy": easy, "medium": medium, "hard": hard}[level]
    for _ in attempts(100, "a buildable question"):
        try:
            return rng.choice(pool)(rng)
        except (AssertionError, VerificationError):
            continue
    raise AssertionError("unreachable")


@generator("ir2a_pick_display_for_question", level=11, presets={t: dict(tier=t) for t in _TIERS})
def ir2a_pick_display_for_question(rng: random.Random, tier: str = "easy") -> dict:
    check(tier in _TIERS, f"unknown tier {tier!r}")
    name = rng.choice(NAMES)
    key, stem, b = _compose(rng, name, tier)
    return chain(stem, _display_part(rng, key), b)


@generator("exd_x_ir2a_pick_display_for_question", level=12, presets={"exam": dict(tier="exam")})
def exd_x_ir2a_pick_display_for_question(rng: random.Random, tier: str = "exam") -> dict:
    """Display, then two medians and their difference, or a time series' rise, average change and next value."""
    check(tier == "exam", "only the exam tier exists")
    name = rng.choice(NAMES)
    if rng.random() < 0.5:
        measure, (ga, gb), lo, hi = rng.choice(_GROUPS)
        n = 11
        a, b = _two_groups(rng, n, lo, hi)
        ma, mb = median_of(a), median_of(b)
        stem = (
            f"{name} has measured {measure} for two groups. {_upper_first(ga)}: {_nums(a)}. {_upper_first(gb)}: {_nums(b)}. "
            f"{name} wants to compare the typical value of the two groups, side by side."
        )
        pa = _display_part(rng, BOX)
        pb = number_part(
            prompt=f"What is the median of the {ga}?",
            exact=ma,
            places=0,
            marks=1,
            working=[f"The 6th value in order is {int(ma)}."],
            wrong=[
                (Fraction(sum(a), n), "ir2a-compares-means-not-medians"),
                (a[n // 2], "ir2a-takes-the-middle-of-the-list-unsorted"),
                (max(a), "ir2a-reports-the-largest-value"),
                (sorted(a)[n // 2 + 1], "ir2a-takes-a-value-next-to-the-median"),
                (sorted(a)[n // 2 - 1], "ir2a-takes-a-value-next-to-the-median"),
            ],
            sensible_range=(0, None),
        )
        pc = number_part(
            prompt=f"What is the median of the {gb}?",
            exact=mb,
            places=0,
            marks=1,
            working=[f"The 6th value in order is {int(mb)}."],
            wrong=[
                (Fraction(sum(b), n), "ir2a-compares-means-not-medians"),
                (b[n // 2], "ir2a-takes-the-middle-of-the-list-unsorted"),
                (max(b), "ir2a-reports-the-largest-value"),
                (sorted(b)[n // 2 + 1], "ir2a-takes-a-value-next-to-the-median"),
                (sorted(b)[n // 2 - 1], "ir2a-takes-a-value-next-to-the-median"),
            ],
            sensible_range=(0, None),
        )
        pd = number_part(
            prompt="By how much does the median of the first group differ from the median of the second group? Give the larger minus the smaller.",
            exact=abs(ma - mb),
            places=0,
            marks=2,
            working=[f"{int(max(ma, mb))} - {int(min(ma, mb))} = {int(abs(ma - mb))}."],
            wrong=[
                (ma + mb, "ir2a-adds-instead-of-subtracting"),
                (abs(max(a) - max(b)), "ir2a-compares-the-largest-values"),
                (abs(Fraction(sum(a), n) - Fraction(sum(b), n)), "ir2a-compares-means-not-medians"),
                (abs(min(a) - min(b)), "ir2a-compares-the-smallest-values"),
            ],
            follow_through="Abs(part_b - part_c)",
            sensible_range=(0, None),
        )
        return chain(stem, pa, pb, pc, pd)
    desc, noun, lo, hi = rng.choice(_SERIES)
    vals = _series(rng, 6, lo, hi, step_multiple=5)
    data = "; ".join(f"{m} {v}" for m, v in zip(_MONTHS, vals))
    stem = (
        f"{name} has recorded {desc}: {data}. {name} wants to show how {noun} changed from January to June, "
        f"and to estimate July's value by assuming the same average monthly change continues."
    )
    rise = vals[-1] - vals[0]
    avg = rise // 5
    pa = _display_part(rng, LINE)
    pb = number_part(
        prompt=f"By how much did the {noun} change from January to June?",
        exact=rise,
        places=0,
        marks=1,
        working=[f"{vals[-1]} - {vals[0]} = {rise}."],
        wrong=[
            (vals[0] - vals[-1], "ir2a-subtracts-in-the-wrong-order"),
            (vals[-1] + vals[0], "ir2a-adds-instead-of-subtracting"),
            (vals[-1], "ir2a-reports-the-last-value"),
        ],
        sensible_range=None,
    )
    pc = number_part(
        prompt="What was the average change per month over that time?",
        exact=avg,
        places=0,
        marks=2,
        working=[f"{rise} / 5 = {avg}, because there are 5 months between January and June."],
        wrong=[
            (Fraction(rise, 6), "ir2a-divides-by-the-number-of-years-listed"),
            (rise, "ir2a-forgets-to-divide-by-the-intervals"),
            (Fraction(rise, 4), "ir2a-divides-by-too-few-intervals"),
            (Fraction(vals[-1], 5), "ir2a-averages-the-last-value-not-the-change"),
            (avg * 2, "ir2a-doubles-the-average-change"),
            (avg + 1, "ir2a-miscounts-the-average-change"),
        ],
        follow_through="part_b/5",
        sensible_range=None,
    )
    pd = number_part(
        prompt="Use that average change to estimate July's value.",
        exact=vals[-1] + avg,
        places=0,
        marks=2,
        working=[f"{vals[-1]} {'+' if avg >= 0 else '-'} {abs(avg)} = {vals[-1] + avg}."],
        wrong=[
            (avg, "ir2a-reports-the-change-not-the-value"),
            (vals[-1] - avg, "ir2a-subtracts-in-the-wrong-order"),
            (vals[-1] + rise, "ir2a-adds-the-whole-change-not-one-month"),
        ],
        follow_through=f"{vals[-1]} + part_c",
        sensible_range=(0, None),
    )
    return chain(stem, pa, pb, pc, pd)
