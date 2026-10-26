"""Generates `f10-display-choice-then-read` questions (AC9M9ST04, Year 9, `answer: number`).

`f10-data-display-selection` asks the learner to name a display from four options. This skill asks for a number that
is only reachable once the right display has been chosen: how many sectors, columns, points or stems it needs, the
angle of a sector, the top of an axis. The display is never named in the stem. The data type and the purpose decide it,
and each wrong display gives a different number, so a wrong choice is diagnosed by the value it produces.

Ladder (reasoning escalation, not size):

- easy:   the purpose and the data type point to the display (shares of a whole, change over time, two groups across
          categories, two measured variables); count the marks it needs, each sector, column or point being one mark.
- medium: the display is still read from the purpose, and the number is one step of construction: the angle of the
          largest sector, the columns of a histogram with a given class width, the top of a count axis, the line
          segments of a line graph.
- hard:   the surface form points the wrong way or the construction chains. Numbers that are shirt-number labels or
          measured values with the same range; a pie chart share that must first be found from the others; the
          highest balance found from monthly changes before the axis is sized; a shared stem column for two groups
          whose stems have gaps.

Written justification of the choice stays rubric work and is not asked here. Every key is built forwards from chosen
values and then re-derived a second way (enumerating classes, stems or axis marks, or exact fractions).
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..answers import Distractor
from ..core import attempts, check, generator
from ._company_kit import keep_company, numeric_pool

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


class _Short(Exception):
    """A draw that cannot give three distinct wrong answers; the generator redraws."""


def _build(rng, answer: int, candidates, stem: str, working: list[str]) -> dict:
    """Assemble the question dict. `candidates` is a list of `(value, misconception_id)`; the ones that equal the
    answer, repeat a value or are not positive whole numbers are dropped, and three of the rest are kept."""
    seen = {answer}
    valid: list[tuple[int, str]] = []
    for value, tag in candidates:
        value = int(value) if float(value).is_integer() else value
        if isinstance(value, int) and value > 0 and value not in seen:
            seen.add(value)
            valid.append((value, tag))
    top = max([answer] + [v for v, _ in valid] + [360]) * 2
    pool = [c for c in numeric_pool(answer, lo=1, hi=top) if c[0] not in {v for v, _ in valid} and c[0] != answer]
    # the key's rank among the shown wrong options is drawn uniformly, then the key is given company in parity and
    # near-miss structure (`_company_kit`), so no feature picks it out
    below = [c for c in valid + pool if c[0] < answer]
    above = [c for c in valid + pool if c[0] > answer]
    low = max(min(rng.randrange(4), len(below)), 3 - len(above))
    if low > len(below) or 3 - low > len(above):
        raise _Short
    chosen = below[:low] + above[: 3 - low]
    rest = [c for c in valid if c not in chosen]
    ordered = keep_company(answer, chosen + rest, pool, 3, max_pool=40)
    shown = [Distractor(str(v), tag, v) for v, tag in ordered[:3]]
    check(len({d.text for d in shown}) == 3 and str(answer) not in {d.text for d in shown}, "three distinct wrong")
    return dict(
        text=stem,
        answer_kind="number",
        answer=int(answer),
        answer_text=str(int(answer)),
        working=working,
        distractors=shown,
    )


def _redraw(fn):
    def wrapper(rng, *args, **kwargs):
        for _ in attempts(200, "a draw with three distinct wrong answers"):
            try:
                return fn(rng, *args, **kwargs)
            except _Short:
                continue

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


# -- shared contexts --------------------------------------------------------------------------------------------------

_SURVEYS = (
    ("how students travel to school", ["walk", "bus", "car", "bike", "train", "scooter", "tram"]),
    (
        "the favourite sport of students",
        ["netball", "soccer", "swimming", "tennis", "cricket", "basketball", "athletics"],
    ),
    ("the kind of fruit students pack for lunch", ["apple", "banana", "orange", "grapes", "mandarin", "pear", "melon"]),
    ("the pets kept by households", ["dog", "cat", "fish", "bird", "rabbit", "lizard", "guinea pig"]),
    ("the instrument students learn", ["guitar", "piano", "violin", "drums", "flute", "trumpet", "clarinet"]),
)

_CLOSE_COUNT = (
    "Choose the display that best suits this purpose. Counting each sector, column or point as one mark, how many "
    "marks does the display need to show all of the data?",
    "Think about which display fits the data and the purpose. If every sector, column or point is one mark, how many "
    "marks are needed to show all of the data?",
    "Decide on the most suitable display. Counting one mark for each sector, column or point, how many marks does it "
    "need to show every value?",
)

# -- easy -------------------------------------------------------------------------------------------------------------


def _easy_pie(rng):
    what, cats = rng.choice(_SURVEYS)
    k = rng.randint(4, 7)
    chosen = rng.sample(cats, k)
    counts = [rng.randint(6, 40) for _ in chosen]
    total = sum(counts)
    listing = ", ".join(f"{c} {n}" for n, c in zip(chosen, counts))
    stem = (
        f"A survey of {total} people asked about {what}: {listing}. The aim is to show how the {total} people are "
        f"shared among the answers, each answer being a part of the one whole. {rng.choice(_CLOSE_COUNT)}"
    )
    answer = k
    check(len(chosen) == answer, "one mark per answer")
    working = [
        "Parts of one whole call for a pie chart, with one sector for each answer.",
        f"There are {k} answers, so {k} sectors.",
    ]
    cands = [
        (total, "ds-dc-counts-items-not-marks"),
        (k + 1, "ds-dc-extra-total-mark"),
        (k - 1, "ds-dc-misses-one-category"),
        (2 * k, "ds-dc-two-marks-per-category"),
    ]
    return _build(rng, answer, cands, stem, working)


def _easy_line(rng):
    quantity = rng.choice(
        [
            ("the rainfall, in millimetres", "weather station"),
            ("the water level, in metres", "dam"),
            ("the average daytime temperature", "weather station"),
        ]
    )
    thing, place = quantity
    n = rng.randint(4, 10)
    first = rng.randint(0, 12 - n)
    a, b = MONTHS[first], MONTHS[first + n - 1]
    series = rng.choice([1, 1, 2])
    where = "one" if series == 1 else "two"
    sites = f"A {place} records {thing} once a month from {a} to {b}."
    if series == 2:
        sites = f"Two {place}s each record {thing} once a month from {a} to {b}."
    stem = (
        f"{sites} The aim is to show how the readings change over time"
        f"{'' if series == 1 else ', with both sets of readings on the one display'}. {rng.choice(_CLOSE_COUNT)}"
    )
    answer = n * series
    check(sum(1 for _ in range(first, first + n)) * series == answer, "points on the display")
    working = [
        "Change over time calls for a line graph, with one point for each reading.",
        f"{a} to {b} is {_plural(n, 'month')}"
        + (
            f", so {n} points." if series == 1 else f", so {n} points for each of the {where} sets: {n} x 2 = {answer}."
        ),
    ]
    cands = [
        ((n - 1) * series, "ds-dc-counts-gaps-not-points"),
        (12 * series, "ds-dc-whole-year-of-months"),
        (n + 1 if series == 1 else n, "ds-dc-ignores-second-set"),
        (n + series, "ds-dc-adds-series-to-points"),
    ]
    return _build(rng, answer, cands, stem, working)


def _easy_clustered(rng):
    what, cats = rng.choice(_SURVEYS)
    k = rng.randint(3, 5)
    g = rng.choice([2, 2, 3])
    groups = rng.choice(
        [("Year 7", "Year 8", "Year 9"), ("Class 9A", "Class 9B", "Class 9C"), ("the girls", "the boys", "the staff")]
    )[:g]
    chosen = rng.sample(cats, k)
    gtext = " and ".join(groups) if g == 2 else f"{groups[0]}, {groups[1]} and {groups[2]}"
    stem = (
        f"{gtext} were each asked about {what}. The answers were {', '.join(chosen)}. The aim is to compare how many "
        f"in each group gave each answer, with the groups side by side for every answer. {rng.choice(_CLOSE_COUNT)}"
    )
    answer = k * g
    check(len([(a, grp) for a in chosen for grp in groups]) == answer, "one column per answer per group")
    working = [
        "Counts in categories, compared between groups, call for a column graph with a column for each group "
        "within each answer.",
        f"{k} answers and {g} groups give {k} x {g} = {answer} columns.",
    ]
    cands = [
        (k, "ds-dc-ignores-groups"),
        (k + g, "ds-dc-adds-categories-and-groups"),
        (g, "ds-dc-counts-groups-only"),
        (k * g + k, "ds-dc-extra-total-mark"),
    ]
    return _build(rng, answer, cands, stem, working)


def _easy_scatter(rng):
    pair = rng.choice(
        [
            ("height", "arm span", "students"),
            ("hours of practice a week", "score in a skills test", "players"),
            ("age", "height", "saplings"),
            ("length", "mass", "fish"),
        ]
    )
    a, b, who = pair
    n = rng.randint(8, 15)
    stem = (
        f"For each of {n} {who}, the {a} and the {b} were both measured. The aim is to see whether the two measured "
        f"variables are related. {rng.choice(_CLOSE_COUNT)}"
    )
    answer = n
    check(len(range(n)) == answer, "one point per pair")
    working = [
        "Two measured variables recorded together call for a scatterplot, with one point for each pair.",
        f"There are {n} pairs, so {n} points.",
    ]
    cands = [
        (2 * n, "ds-dc-points-for-each-value"),
        (n - 1, "ds-dc-counts-gaps-not-points"),
        (n + 1, "ds-dc-extra-total-mark"),
        (n + 2, "ds-dc-adds-variables-to-points"),
    ]
    return _build(rng, answer, cands, stem, working)


# -- medium -----------------------------------------------------------------------------------------------------------


def _angle_split(rng):
    """Sector angles as multiples of 12 summing to 360 with a unique largest, and counts that give them exactly."""
    for _ in attempts(500, "sector parts with a unique largest"):
        k = rng.randint(4, 6)
        cuts = sorted(rng.sample(range(1, 30), k - 1))
        parts = [b - a for a, b in zip([0, *cuts], [*cuts, 30])]
        if min(parts) >= 2 and parts.count(max(parts)) == 1:
            return parts
    raise AssertionError  # pragma: no cover


def _med_pie_angle(rng):
    what, cats = rng.choice(_SURVEYS)
    parts = _angle_split(rng)
    k = len(parts)
    m = rng.choice([2, 3, 4, 5, 6])
    counts = [p * m for p in parts]
    total = sum(counts)
    check(total == 30 * m, "the counts add to the total")
    chosen = rng.sample(cats, k)
    big = counts.index(max(counts))
    listing = ", ".join(f"{c} {n}" for n, c in zip(chosen, counts))
    stem = (
        f"A survey of {total} people asked about {what}: {listing}. The aim is to show each answer as its share of the "
        f"{total} people, as parts of one whole. On the display that suits this purpose, what is the angle, in "
        f"degrees, of the sector for the most common answer ({chosen[big]})?"
    )
    angle = Fraction(360 * counts[big], total)
    check(angle.denominator == 1 and angle == 12 * parts[big], "an exact whole-degree angle")
    answer = int(angle)
    working = [
        "Shares of one whole call for a pie chart; a sector's angle is its share of 360 degrees.",
        f"{counts[big]} / {total} x 360 = {answer} degrees.",
    ]
    cands = [
        (counts[big], "ds-dc-count-not-angle"),
        (round(100 * counts[big] / total) if (100 * counts[big]) % total == 0 else 0, "ds-dc-percent-not-angle"),
        (360 // k, "ds-dc-equal-sectors"),
        (12 * sorted(parts)[-2], "ds-dc-second-largest-sector"),
        (counts[big] * 360 // (total + counts[big]), "ds-dc-share-of-wrong-total"),
    ]
    return _build(rng, answer, cands, stem, working)


def _med_pie_percent(rng):
    what, cats = rng.choice(_SURVEYS)
    total = rng.choice([40, 60, 80, 100, 120, 200, 240])
    pct = rng.choice([10, 15, 20, 25, 30, 35, 40, 45])
    count = total * pct // 100
    check(count * 100 == total * pct, "an exact count")
    cat = rng.choice(cats)
    stem = (
        f"In a survey of {total} people about {what}, {pct}% gave the answer {cat}. The aim is to show how the "
        f"{total} people are shared among all of the answers, as parts of one whole. On the display that suits this "
        f"purpose, what is the angle, in degrees, of the sector for {cat}?"
    )
    angle = Fraction(360 * pct, 100)
    check(angle.denominator == 1 and angle == Fraction(360 * count, total), "angle by percentage and by count")
    answer = int(angle)
    working = [
        "Shares of one whole call for a pie chart; a sector's angle is its share of 360 degrees.",
        f"{pct}% of 360 = {answer} degrees.",
    ]
    cands = [
        (pct, "ds-dc-percent-not-angle"),
        (count, "ds-dc-count-not-angle"),
        (360 - answer, "ds-dc-rest-of-circle"),
        (pct * 3, "ds-dc-percent-times-three"),
    ]
    return _build(rng, answer, cands, stem, working)


_HIST_THINGS = (
    ("waiting times at a clinic, in minutes", "times"),
    ("masses of parcels, in kilograms", "masses"),
    ("heights of seedlings, in centimetres", "heights"),
    ("times for a 400 m run, in seconds", "times"),
    ("lengths of phone calls, in minutes", "lengths"),
)


def _hist_columns_by_listing(lo: int, hi: int, w: int) -> int:
    start = (lo // w) * w
    count, edge = 0, start
    for _ in attempts(500, "class edges"):
        if edge > hi:
            break
        count += 1
        edge += w
    return count


def _med_histogram(rng):
    thing, noun = rng.choice(_HIST_THINGS)
    w = rng.choice([2, 5, 5, 10])
    cols = rng.randint(5, 9)
    start = w * rng.randint(1, 8)
    top_off = 0 if rng.random() < 0.3 else rng.randint(1, w - 1)
    hi = start + (cols - 1) * w + top_off
    lo = start + rng.randint(0, w - 1)
    n = rng.randint(24, 45)
    stem = (
        f"A study recorded {n} {thing}. The measurements range from {lo} to {hi}. The aim is to show the shape of the "
        f"distribution of this measured variable. The display uses equal classes of width {w}, with the first class "
        f"starting at the multiple of {w} at or below the least value. Each class includes its lower boundary but not "
        f"its upper boundary. How many columns does the display need?"
    )
    answer = cols
    check(_hist_columns_by_listing(lo, hi, w) == answer, "columns by listing the class boundaries")
    check((lo // w) * w == start, "the first class starts at the multiple at or below the least value")
    working = [
        "A measured variable shown as a distribution calls for a histogram.",
        f"The first class starts at {start}. The classes then start at "
        f"{start}, {start + w}, ... and the last one must hold {hi}, so it starts at {start + (cols - 1) * w}.",
        f"That is {answer} columns.",
    ]
    cands = [
        ((hi - lo) // w, "ds-dc-range-over-width"),
        (math.ceil((hi - lo) / w), "ds-dc-class-start-at-minimum"),
        (n, "ds-dc-one-column-per-value"),
        (hi - lo + 1, "ds-dc-column-per-possible-value"),
        (cols + 1, "ds-dc-extra-total-mark"),
        ((hi - lo) // w + 1, "ds-dc-class-start-at-minimum"),
    ]
    return _build(rng, answer, cands, stem, working)


def _axis_top(value: int, step: int) -> int:
    top = 0
    for _ in attempts(1000, "axis marks"):
        if top >= value:
            return top
        top += step
    raise AssertionError  # pragma: no cover


def _med_axis(rng):
    what, cats = rng.choice(_SURVEYS)
    k = rng.randint(4, 6)
    chosen = rng.sample(cats, k)
    step = rng.choice([5, 10, 20])
    counts = [rng.randint(9, 95) for _ in chosen]
    big = max(counts)
    if counts.count(big) > 1 or (big % step == 0 and rng.random() < 0.7):
        raise _Short
    listing = ", ".join(f"{c} {n}" for n, c in zip(chosen, counts))
    stem = (
        f"A survey about {what} gave these counts: {listing}. The aim is to compare how many people gave each answer. "
        f"On the display that suits this purpose, the counting axis starts at 0 and is marked in steps of {step}. "
        f"The axis stops at the first mark that is at least as high as the largest count. What number is at the top "
        f"of the axis?"
    )
    answer = math.ceil(big / step) * step
    check(_axis_top(big, step) == answer, "axis top by stepping")
    working = [
        "Comparing counts in categories calls for a column graph; its counting axis must reach the tallest column.",
        f"The largest count is {big}. The first mark of {step} that is at least {big} is {answer}.",
    ]
    cands = [
        (big, "ds-dc-axis-top-is-max"),
        (answer + step, "ds-dc-axis-extra-step"),
        (answer - step, "ds-dc-axis-too-short"),
        (sum(counts), "ds-dc-axis-total"),
    ]
    return _build(rng, answer, cands, stem, working)


def _med_segments(rng):
    quantity = rng.choice(
        [
            "the number of visitors to a museum",
            "the closing price of a share, in dollars",
            "the mass of a growing puppy, in kilograms",
            "the depth of water in a tank, in centimetres",
        ]
    )
    n = rng.randint(4, 10)
    first = rng.randint(0, 12 - n)
    a, b = MONTHS[first], MONTHS[first + n - 1]
    stem = (
        f"Once a month from {a} to {b}, {quantity} was recorded. The aim is to show how it changed over "
        f"time. On the display that suits this purpose, how many straight line segments join the plotted points in "
        f"order?"
    )
    answer = n - 1
    check(len(list(zip(range(n), range(1, n)))) == answer, "segments between consecutive points")
    working = [
        "Change over time calls for a line graph.",
        f"{a} to {b} is {n} months, so {n} points. {n} points in a row are joined by {n} - 1 = {answer} segments.",
    ]
    cands = [
        (n, "ds-dc-points-not-segments"),
        (n + 1, "ds-dc-extra-total-mark"),
        (n - 2, "ds-dc-misses-one-category"),
        (11, "ds-dc-whole-year-of-months"),
    ]
    return _build(rng, answer, cands, stem, working)


# -- hard -------------------------------------------------------------------------------------------------------------


def _hard_labels_or_measures(rng):
    w = rng.choice([5, 10])
    k = rng.randint(7, 10)
    labels = rng.random() < 0.5
    for _ in attempts(500, "values whose class count differs from their distinct count"):
        lo = rng.randint(2, 15)
        values = sorted(rng.sample(range(lo, lo + rng.randint(30, 55)), k))
        cols = values[-1] // w - values[0] // w + 1
        if cols != k and cols != values[-1] - values[0] + 1 and (values[-1] - values[0]) // w != cols:
            break
    listing = ", ".join(str(v) for v in values)
    lo_v, hi_v = values[0], values[-1]
    class_note = (
        f"Classes of width {w}, starting at the multiple of {w} at or below the least value, are possible if they "
        f"suit the data."
    )
    if labels:
        stem = (
            f"{k} different players scored goals this season, wearing these shirt numbers: {listing}. The club wants "
            f"to show how many goals each shirt number scored. {class_note} How many columns does the display need?"
        )
        answer = k
        check(len(set(values)) == answer, "one column per different shirt number")
        working = [
            "Shirt numbers are labels, not measurements, so they are categories and the display is a column graph "
            "with one column for each.",
            f"There are {k} different numbers, so {k} columns, with no grouping.",
        ]
        cands = [
            (cols, "ds-dc-labels-as-measurements"),
            ((hi_v - lo_v) // w, "ds-dc-range-over-width"),
            (hi_v - lo_v + 1, "ds-dc-column-per-possible-value"),
            (k + 1, "ds-dc-extra-total-mark"),
        ]
        return _build(rng, answer, cands, stem, working)
    what, _noun = rng.choice(_HIST_THINGS)
    stem = (
        f"A study recorded {k} {what}: {listing}. The aim is to show the shape of the distribution of this measured "
        f"variable. {class_note} Each class includes its lower boundary but not its upper boundary. How many columns "
        f"does the display need?"
    )
    answer = cols
    classes = {v // w for v in values}
    check(len(range(min(classes), max(classes) + 1)) == answer, "columns by listing the classes")
    working = [
        "A measured variable shown as a distribution calls for a histogram, grouped into equal classes.",
        f"The first class starts at {(lo_v // w) * w} and the last class that holds {hi_v} starts at "
        f"{(hi_v // w) * w}, so there are {answer} classes, empty ones included.",
    ]
    cands = [
        (k, "ds-dc-measurements-as-labels"),
        ((hi_v - lo_v) // w, "ds-dc-range-over-width"),
        (hi_v - lo_v + 1, "ds-dc-column-per-possible-value"),
        (len(classes), "ds-dc-skips-empty-classes"),
        (answer + 1, "ds-dc-extra-total-mark"),
    ]
    return _build(rng, answer, cands, stem, working)


def _hard_missing_share(rng):
    what, cats = rng.choice(_SURVEYS)
    k = rng.randint(4, 6)
    chosen = rng.sample(cats, k - 1)
    for _ in attempts(500, "shares that leave a positive remainder"):
        pcts = [rng.choice([5, 10, 15, 20, 25, 30]) for _ in chosen]
        rest = 100 - sum(pcts)
        if rest >= 5:
            break
    total = rng.choice([40, 60, 80, 100, 120, 200])
    listing = ", ".join(f"{p}% {n}" for n, p in zip(chosen, pcts))
    stem = (
        f"In a survey of {total} people about {what}, the answers were {listing}, and everyone else gave other "
        f"answers. The aim is to show how the {total} people are shared among these answers and the others, as parts "
        f"of one whole. On the display that suits this purpose, what is the angle, in degrees, of the sector for the "
        f"others?"
    )
    angle = Fraction(360 * rest, 100)
    check(angle.denominator == 1, "an exact angle")
    answer = int(angle)
    check(answer + sum(Fraction(360 * p, 100) for p in pcts) == 360, "the sectors fill the circle")
    working = [
        "Parts of one whole call for a pie chart, and the sectors fill the circle.",
        f"The shares given add to {sum(pcts)}%, so the others are 100% - {sum(pcts)}% = {rest}%.",
        f"{rest}% of 360 = {answer} degrees.",
    ]
    cands = [
        (rest, "ds-dc-percent-not-angle"),
        (int(Fraction(360 * sum(pcts), 100)), "ds-dc-angles-of-given-shares"),
        (total * rest // 100 if (total * rest) % 100 == 0 else 0, "ds-dc-count-not-angle"),
        (rest * 3, "ds-dc-percent-times-three"),
    ]
    return _build(rng, answer, cands, stem, working)


def _hard_running_axis(rng):
    thing = rng.choice(
        [
            ("Zara's savings (in dollars)", "balance"),
            ("the stock of a shop's best-selling item (in units)", "stock"),
            ("the water stored in a tank (in kilolitres)", "level"),
        ]
    )
    what, word = thing
    n = rng.randint(5, 8)
    for _ in attempts(1000, "monthly changes with a peak before the last month"):
        step = rng.choice([20, 25, 50])
        v0 = rng.randint(2, 8) * 10
        changes = [rng.choice([-1, 1, 1]) * rng.randint(2, 9) * 10 for _ in range(n - 1)]
        series = [v0]
        for c in changes:
            series.append(series[-1] + c)
        peak = max(series)
        top = math.ceil(peak / step) * step
        last_top = math.ceil(series[-1] / step) * step
        if min(series) >= 10 and series.index(peak) != n - 1 and top != last_top and peak % step != 0:
            break
    first = rng.randint(0, 12 - n)
    mons = MONTHS[first : first + n]
    moves = ", ".join(f"{m[:3]} {c:+d}".replace("-", "−") for m, c in zip(mons[1:], changes))
    stem = (
        f"At the end of {mons[0]}, {what} was {v0}. In the months after, the monthly changes were: {moves}. "
        f"The aim is to show how the {word} moves from month to month. On the display that suits this purpose, the "
        f"vertical axis starts at 0 and is marked in steps of {step}. It stops at the first mark that is at least as "
        f"high as the largest {word} shown. What number is at the top of the axis?"
    )
    answer = top
    check(_axis_top(max(series), step) == answer, "axis top by stepping")
    working = [
        "Change over time calls for a line graph, whose vertical axis shows the values.",
        "End-of-month values: " + ", ".join(str(v) for v in series) + f". The largest is {peak}.",
        f"The first mark of {step} that is at least {peak} is {answer}.",
    ]
    cands = [
        (peak, "ds-dc-axis-top-is-max"),
        (answer + step, "ds-dc-axis-extra-step"),
        (last_top, "ds-dc-uses-last-value"),
        (n, "ds-dc-axis-shows-frequency"),
    ]
    return _build(rng, answer, cands, stem, working)


def _hard_stems(rng):
    context = rng.choice(
        [
            ("marks", "marks out of 100 in a test", ("Class 9A", "Class 9B")),
            ("ages", "ages, in years, of members", ("The Tigers", "The Hawks")),
            ("times", "times, in seconds, to solve a puzzle", ("Group A", "Group B")),
        ]
    )
    noun, phrase_, groups = context
    for _ in attempts(1000, "two groups whose stems have a gap and differ in size"):
        lo = rng.randint(1, 5)
        span = rng.randint(3, min(6, 9 - lo))
        stems = list(range(lo, lo + span + 1))
        a_n, b_n = rng.randint(7, 9), rng.randint(7, 9)
        vals_a = [rng.choice(stems) * 10 + rng.randint(0, 9) for _ in range(a_n)]
        vals_b = [rng.choice(stems) * 10 + rng.randint(0, 9) for _ in range(b_n)]
        all_vals = vals_a + vals_b
        tens_a, tens_b = {v // 10 for v in vals_a}, {v // 10 for v in vals_b}
        used = tens_a | tens_b
        inclusive = max(used) - min(used) + 1
        if len(used) < inclusive and inclusive >= 4 and len(tens_a) != len(tens_b) and len(tens_a) != inclusive:
            break
    va, vb = sorted(vals_a), sorted(vals_b)
    stem = (
        f"The {phrase_} of two groups were recorded. {groups[0]}: {', '.join(map(str, va))}. {groups[1]}: "
        f"{', '.join(map(str, vb))}. The aim is to compare the two groups value by value, keeping every value "
        f"visible. On the display that suits this purpose, the two groups share one column of stems, and every stem "
        f"from the least to the greatest is written, whether or not it has a leaf. How many stems does it have?"
    )
    answer = inclusive
    check(len(list(range(min(all_vals) // 10, max(all_vals) // 10 + 1))) == answer, "stems by listing")
    working = [
        "Two groups compared value by value call for a back-to-back stem-and-leaf plot.",
        f"The least value is {min(all_vals)} (stem {min(all_vals) // 10}) and the greatest is {max(all_vals)} "
        f"(stem {max(all_vals) // 10}). Every stem between is written, so {answer} stems.",
    ]
    cands = [
        (len(all_vals), "ds-dc-stems-from-values"),
        (len(used), "ds-dc-skips-empty-stems"),
        (len(tens_a), "ds-dc-stems-one-group"),
        (max(used) - min(used), "ds-dc-stems-range-not-inclusive"),
        (len(tens_a) + len(tens_b), "ds-dc-stems-for-each-group"),
    ]
    return _build(rng, answer, cands, stem, working)


# -- registration -----------------------------------------------------------------------------------------------------

_EASY = (_easy_pie, _easy_line, _easy_clustered, _easy_scatter)
_MEDIUM = (_med_pie_angle, _med_pie_percent, _med_histogram, _med_axis, _med_segments)
_HARD = (_hard_labels_or_measures, _hard_missing_share, _hard_running_axis, _hard_stems)


def _pick(rng, shapes):
    shape = rng.choice(shapes)
    return _redraw(shape)(rng)


@generator("ds_display_choice_easy", level=9)
def ds_display_choice_easy(rng):
    return _pick(rng, _EASY)


@generator("ds_display_choice_medium", level=9)
def ds_display_choice_medium(rng):
    return _pick(rng, _MEDIUM)


@generator("ds_display_choice_hard", level=9)
def ds_display_choice_hard(rng):
    return _pick(rng, _HARD)
