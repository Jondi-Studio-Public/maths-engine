"""Short written-answer generators about describing and interpreting data.

Every question gives a small dataset or table as text (8 to 15 values, or a table of counts), states the summary
numbers it needs (a mean, a median, a percentage, a correlation coefficient, a fitted line) and asks for a short
written answer marked against a rubric. All numbers a prompt states are computed in the generator from the data it
prints. The `answer` of each question is a marking dict (see `_written_answer_kit`), not a value.

Topics (one per skill):

- `wa_centre_suitability`: which measure of centre suits a dataset (an outlier, a symmetric set, categories).
- `wa_media_centre_spread`: media claims that misuse a mean, a range or a mode, judged against the data.
- `wa_presentation_suitability`: which kind of data display suits a stated dataset and purpose.
- `wa_percent_association`: describing an association from the percentages in a two-way table.
- `wa_bivariate_features`: describing the pattern, strength, form and outliers of paired data.
- `wa_interpret_relationships`: reading the slope and intercept of a fitted line in context, and using it to predict.
"""

from __future__ import annotations

import statistics
from fractions import Fraction

from ..core import VerificationError, check
from ._written_answer_kit import (
    attempts,
    NAME_SLOTS,
    NAMES,
    Case,
    Tier,
    finish,
    marking,
    mean_of,
    median_of,
    mode_of,
    num_text,
    register,
    register_custom,
    values_line,
)

# ---------------------------------------------------------------------------
# wa_centre_suitability
# ---------------------------------------------------------------------------
SKEW_CTX = (
    dict(
        thing="annual salaries (in $ thousand) of the staff at a small firm",
        lo=48,
        hi=70,
        out=(185, 260),
        total=True,
        totalword="total wage bill",
        unit="$ thousand",
        item="salary",
    ),
    dict(
        thing="sale prices (in $ thousand) of the houses sold in one street last year",
        lo=420,
        hi=540,
        out=(1400, 2100),
        total=False,
        unit="$ thousand",
        item="price",
    ),
    dict(
        thing="waiting times (in minutes) of patients at a medical clinic one morning",
        lo=6,
        hi=22,
        out=(75, 120),
        total=False,
        unit="minutes",
        item="waiting time",
    ),
    dict(
        thing="numbers of text messages sent in one day by the students in a class",
        lo=12,
        hi=40,
        out=(170, 260),
        total=True,
        totalword="total number of messages sent by the class",
        unit="messages",
        item="number of messages",
    ),
)
SYM_CTX = (
    dict(thing="marks (out of 100) in a class test", lo=55, hi=80, unit="marks", item="mark"),
    dict(thing="times (in seconds) taken by runners in a 100 m race", lo=11, hi=17, unit="seconds", item="time"),
    dict(
        thing="daily maximum temperatures (in degrees Celsius) over two weeks",
        lo=23,
        hi=31,
        unit="degrees",
        item="temperature",
    ),
    dict(thing="heights (in cm) of the players in a basketball squad", lo=182, hi=200, unit="cm", item="height"),
)
CAT_CTX = (
    ("the colours of the cars in a car park", ["red", "white", "blue", "black", "silver", "green"]),
    ("the favourite fruits chosen by a class", ["apple", "banana", "orange", "grapes", "mango", "pear"]),
    ("the ways students travel to school", ["walk", "bus", "car", "bike", "train", "scooter"]),
    ("the pets owned by the families in a street", ["dog", "cat", "bird", "fish", "rabbit", "none"]),
)


def _outlier_set(rng, ctx: dict) -> dict:
    n = rng.randint(7, 11)
    base = [rng.randint(ctx["lo"], ctx["hi"]) for _ in range(n)]
    out = rng.randint(*ctx["out"])
    vals = base + [out]
    rng.shuffle(vals)
    return _stats(vals, out=out)


def _sym_set(rng, ctx: dict) -> dict:
    for _ in attempts(400, "a symmetric-looking dataset"):
        n = rng.randint(9, 14)
        vals = [rng.randint(ctx["lo"], ctx["hi"]) for _ in range(n)]
        m, d = mean_of(vals), median_of(vals)
        if abs(m - d) <= Fraction(ctx["hi"] - ctx["lo"], 12) and len(set(vals)) >= min(6, ctx["hi"] - ctx["lo"]):
            break
    return _stats(vals)


def _stats(vals: list, out: int | None = None) -> dict:
    m, d = mean_of(vals), median_of(vals)
    return dict(
        values=vals,
        n=len(vals),
        line=values_line(vals),
        mean=num_text(m),
        median=num_text(d),
        total=sum(vals),
        above=sum(1 for v in vals if v > m),
        below=sum(1 for v in vals if v < m),
        out=out,
    )


def _cat_set(rng) -> dict:
    desc, cats = rng.choice(CAT_CTX)
    chosen = rng.sample(cats, rng.randint(4, 5))
    for _ in attempts(200, "counts with a single mode"):
        counts = [rng.randint(1, 9) for _ in chosen]
        if counts.count(max(counts)) == 1:
            break
    top = chosen[counts.index(max(counts))]
    line = ", ".join(f"{c} {k}" for c, k in zip(chosen, counts, strict=True))
    return dict(desc=desc, line=line, total=sum(counts), top=top, topn=max(counts), k=len(chosen))


def _cs_easy(rng) -> dict:
    ctx = rng.choice(SKEW_CTX)
    s = _outlier_set(rng, ctx)
    text = (
        f"The {ctx['thing']} were:\n{s['line']}\nThe mean is {s['mean']} and the median is {s['median']}. "
        f"Which measure of centre, the mean or the median, better describes a typical value? Explain, using the data."
    )
    mark = marking(
        reference=[
            f"The median ({s['median']}) is the better measure of a typical {ctx['item']}.",
            f"One value, {s['out']}, is much larger than the rest, so it pulls the mean up to {s['mean']}, which is more than {s['below']} of the {s['n']} values.",
            "The median is the middle value, so it is not pulled up by the extreme value.",
        ],
        points=[
            ("correct_idea", f"chooses the median ({s['median']}) over the mean ({s['mean']})"),
            ("correct_idea", "explains that an extreme value affects the mean much more than the median"),
            ("use_of_context", f"identifies the extreme value {s['out']} in the data and the effect on the mean"),
            ("completeness", f"supports the choice with the figures (most values are below the mean of {s['mean']})"),
            ("communication", "states the choice first, then the reason"),
        ],
        wrong=[
            (
                "chooses_the_mean_because_all_values_used",
                "does it choose the mean only because it uses every value, ignoring the extreme one?",
            ),
            ("explains_median_without_data", "does it say the median is 'the middle' without using the data?"),
        ],
    )
    return finish(text, f"the median ({s['median']})", mark)


def _cs_medium(rng) -> dict:
    kind = rng.choice(["outlier", "symmetric", "category"])
    if kind == "outlier":
        ctx = rng.choice(SKEW_CTX)
        s = _outlier_set(rng, ctx)
        text = (
            f"The {ctx['thing']} were:\n{s['line']}\nThe mean is {s['mean']} and the median is {s['median']}. "
            "Decide which measure of centre (mean, median or mode) is most suitable for summarising these data, and explain why."
        )
        ref = [
            f"The median ({s['median']}) is the most suitable.",
            f"The value {s['out']} is much larger than the others and pulls the mean up to {s['mean']}.",
            "The mode is not helpful because the values are mostly different.",
        ]
        pts = [
            ("correct_idea", f"chooses the median ({s['median']})"),
            ("use_of_context", f"explains that {s['out']} distorts the mean ({s['mean']})"),
            ("completeness", "also says why the mode is not a good choice here"),
            ("communication", "gives the choice and the reason together"),
        ]
        ans = f"the median ({s['median']})"
        wrong = [("chooses_mean", "does it choose the mean, which is pulled up by the extreme value?")]
    elif kind == "symmetric":
        ctx = rng.choice(SYM_CTX)
        s = _sym_set(rng, ctx)
        text = (
            f"The {ctx['thing']} were:\n{s['line']}\nThe mean is {s['mean']} and the median is {s['median']}. "
            "Decide which measure of centre (mean, median or mode) is most suitable for summarising these data, and explain why."
        )
        ref = [
            f"The mean ({s['mean']}) is suitable, and the median ({s['median']}) would give almost the same answer.",
            "The data have no extreme values, so the mean is not distorted and it uses every value.",
        ]
        pts = [
            ("correct_idea", f"accepts the mean ({s['mean']}) as suitable, noting the median ({s['median']}) is close"),
            ("use_of_context", "explains that there is no extreme value to distort the mean"),
            ("completeness", "gives a reason such as the mean using every value"),
            ("communication", "compares the mean and the median using the numbers"),
        ]
        ans = f"the mean ({s['mean']}); the median is close"
        wrong = [
            ("insists_on_median_only", "does it claim the mean is unsuitable even though there is no extreme value?")
        ]
    else:
        c = _cat_set(rng)
        text = (
            f"A survey recorded {c['desc']}. The numbers in each category were: {c['line']} (total {c['total']}). "
            "Which measure of centre (mean, median or mode) can be used to describe the most typical category, and why can the others not be used?"
        )
        ref = [
            f"The mode can be used: the most common category is {c['top']}, with {c['topn']}.",
            "The mean and the median need numerical values that can be added and put in order, and categories such as these are not numbers.",
        ]
        pts = [
            ("correct_idea", f"chooses the mode, which is {c['top']} ({c['topn']})"),
            ("use_of_context", f"names the most common category from the counts: {c['top']}"),
            ("completeness", "explains why the mean and the median cannot be found for categories"),
            ("communication", "answers both the choice and the reason for rejecting the others"),
        ]
        ans = f"the mode ({c['top']})"
        wrong = [
            ("averages_the_counts", f"does it find the mean of the {c['k']} counts and treat it as a typical category?")
        ]
    mark = marking(
        reference=ref,
        points=pts,
        wrong=wrong + [("no_reason_given", "does it choose a measure but give no reason from the data?")],
    )
    return finish(text, ans, mark)


def _cs_hard(rng) -> dict:
    ctx = rng.choice([c for c in SKEW_CTX if c["total"]])
    s = _outlier_set(rng, ctx)
    p, q = rng.sample(NAMES, 2)
    text = (
        f"The {ctx['thing']} were:\n{s['line']}\nThere are {s['n']} values, the mean is {s['mean']}, the median is {s['median']} and the total of all the values is {s['total']}. "
        f"{p} says the mean is the right measure to report. {q} says the median is the right measure. "
        f"For each purpose, say whether the mean or the median is more suitable and why: (a) working out the {ctx['totalword']}; (b) describing what a typical value is."
    )
    mark = marking(
        reference=[
            f"(a) The mean is suitable for the {ctx['totalword']}, because the total equals the mean times the number of values: {s['mean']} × {s['n']} is about {s['total']}.",
            f"(b) The median ({s['median']}) is better for a typical value, because the extreme value {s['out']} pulls the mean up to {s['mean']}.",
            "So each person is right for one purpose, and the measure should be chosen to match the purpose.",
        ],
        points=[
            ("correct_idea", "chooses the mean for the total (it is the total divided by the number of values)"),
            (
                "correct_idea",
                f"chooses the median ({s['median']}) for a typical value because of the extreme value {s['out']}",
            ),
            ("use_of_context", f"uses the figures: total {s['total']}, {s['n']} values, mean {s['mean']}"),
            ("completeness", "answers both purposes and links each to a different measure"),
            ("communication", "says what each person gets right and wrong"),
        ],
        wrong=[
            ("one_measure_for_both", "does it choose the same measure for both purposes?"),
            ("median_for_total", "does it use the median to find the total?"),
        ],
    )
    return finish(text, "mean for the total, median for the typical value", mark)


def _cs_exam(rng) -> dict:
    c1, c2 = rng.choice(SKEW_CTX), rng.choice(SYM_CTX)
    a, b = _outlier_set(rng, c1), _sym_set(rng, c2)
    p = rng.choice(NAMES)
    text = (
        f"Dataset A: the {c1['thing']}:\n{a['line']}\nThe mean is {a['mean']} and the median is {a['median']}.\n"
        f"Dataset B: the {c2['thing']}:\n{b['line']}\nThe mean is {b['mean']} and the median is {b['median']}.\n"
        f'{p} says: "Always use the mean, because it uses every value." Evaluate this claim using both datasets, and say which measure of centre you would report for each dataset.'
    )
    mark = marking(
        reference=[
            f"The claim is not always true. For dataset A, the value {a['out']} pulls the mean up to {a['mean']} so the median ({a['median']}) is better.",
            f"For dataset B there is no extreme value, the mean ({b['mean']}) and median ({b['median']}) are close, and the mean is a good choice.",
            "Using every value is an advantage only when no value is extreme.",
        ],
        points=[
            ("correct_idea", f"rejects the claim and chooses the median ({a['median']}) for dataset A"),
            ("correct_idea", f"accepts the mean ({b['mean']}) for dataset B and explains why"),
            (
                "use_of_context",
                f"uses the numbers from both datasets (A: {a['mean']} against {a['median']}; B: {b['mean']} against {b['median']})",
            ),
            ("completeness", "explains that using every value is a weakness when one value is extreme"),
            ("communication", "treats the two datasets separately before concluding"),
        ],
        wrong=[
            ("agrees_always_mean", "does it agree that the mean should always be used?"),
            ("always_median", "does it swing to the opposite claim and say the median should always be used?"),
        ],
    )
    return finish(text, "no: median for A, mean for B", mark)


register_custom(
    "wa_centre_suitability",
    12,
    dict(easy=_cs_easy, medium=_cs_medium, hard=_cs_hard, exam=_cs_exam),
    doc="choosing mean, median or mode.",
)

# ---------------------------------------------------------------------------
# wa_media_centre_spread
# ---------------------------------------------------------------------------
FIRMS = ("Harlow Engineering", "Brightway Logistics", "Coastline Foods", "Mapleton Printing", "Redhill Software")
PAPERS = ("The Daily Courier", "Metro News", "Channel 5 News", "The Weekend Herald", "The Evening Post")
ROUTES = (
    ("Route 12", "Route 40"),
    ("the Blue line", "the Green line"),
    ("Bus 7", "Bus 22"),
    ("the City Express", "the Coastal Express"),
)


def _salary_vals(rng, v):
    n = rng.randint(7, 10)
    return [rng.randint(46, 72) for _ in range(n)] + [rng.randint(210, 320)]


def _two_routes(rng, v):
    m = rng.randint(10, 16)
    small = [rng.randint(1, 2) for _ in range(4)]
    big = [rng.randint(6, 10) for _ in range(4)]
    a = [m + d for d in small] + [m - d for d in small]
    b = [m + d for d in big] + [m - d for d in big]
    rng.shuffle(a)
    rng.shuffle(b)
    return a, b


def _children_vals(rng, v):
    for _ in attempts(400, "a family-size dataset with a non-integer mean"):
        vals = [rng.choice([0, 1, 2, 2, 2, 3, 3, 4, 5]) for _ in range(rng.randint(10, 14))]
        m = mean_of(vals)
        if m.denominator != 1 and len(mode_of(vals)) == 1:
            return vals
    raise VerificationError("no family-size dataset")


def _range_outlier_vals(rng, v):
    n = rng.randint(9, 13)
    base = [rng.randint(24, 29) for _ in range(n)]
    return base + [rng.randint(41, 46)]


def _mode_extreme_vals(rng, v):
    low = rng.choice([5, 10, 15])
    others = rng.sample(range(25, 56), rng.randint(7, 10))
    vals = [low, low] + others
    rng.shuffle(vals)
    return vals


MEDIA_STAT_CASES = [
    Case(
        "mean_skew",
        slots={
            "paper": PAPERS,
            "firm": FIRMS,
            "vals": _salary_vals,
            "n": lambda r, v: len(v["vals"]),
            "line": lambda r, v: values_line(v["vals"]),
            "mean": lambda r, v: num_text(mean_of(v["vals"])),
            "median": lambda r, v: num_text(median_of(v["vals"])),
            "top": lambda r, v: max(v["vals"]),
            "above": lambda r, v: sum(1 for x in v["vals"] if x > mean_of(v["vals"])),
            **NAME_SLOTS,
        },
        fields=dict(
            setup="The annual salaries (in $ thousand) of the {n} employees of {firm} are:\n{line}\n{paper} reports: \"The average salary at {firm} is ${mean} thousand, so the typical employee is very well paid.\" (The 'average' is the mean.)",
            misuse="the mean ({mean}) is pulled up by one very large salary ({top}); the mean is exceeded by only {above} of the {n} salaries, and the median is {median}",
            figures="mean {mean}, median {median}, and the largest salary is {top}",
            better="The median salary at {firm} is ${median} thousand, although one employee earns ${top} thousand.",
            needs="the median, and how many employees earn near each salary",
            claim_good="the mean is distorted by the {top} salary, and the median of {median} describes the typical employee better",
            claim_bad="the mean uses every salary, so ${mean} thousand must describe the typical employee",
            bad_flaw="using every value is a weakness when one value is extreme, since it drags the mean far above what most employees earn",
        ),
        wrong=[("accepts_the_average", "does it accept the mean as the typical salary?")],
    ),
    Case(
        "same_mean_spread",
        slots={
            "paper": PAPERS,
            "routes": ROUTES,
            "r1": lambda r, v: v["routes"][0],
            "r2": lambda r, v: v["routes"][1],
            "pair": _two_routes,
            "a": lambda r, v: v["pair"][0],
            "b": lambda r, v: v["pair"][1],
            "la": lambda r, v: values_line(v["a"]),
            "lb": lambda r, v: values_line(v["b"]),
            "mean": lambda r, v: num_text(mean_of(v["a"])),
            "mean2": lambda r, v: num_text(mean_of(v["b"])),
            "ra": lambda r, v: max(v["a"]) - min(v["a"]),
            "rb": lambda r, v: max(v["b"]) - min(v["b"]),
            "maxb": lambda r, v: max(v["b"]),
            "minb": lambda r, v: min(v["b"]),
            **NAME_SLOTS,
        },
        fields=dict(
            setup='The minutes late of eight trips on each route were:\n{r1}: {la}\n{r2}: {lb}\n{paper} reports: "{r1} and {r2} both run an average of {mean} minutes late, so they are equally reliable."',
            misuse="the means are the same ({mean} and {mean2}), but the spread is very different: the trips on {r1} vary by {ra} minutes, those on {r2} by {rb} minutes (from {minb} to {maxb}), so {r2} is much less predictable",
            figures="both means are {mean}; the ranges are {ra} and {rb} minutes",
            better="{r1} and {r2} both run an average of {mean} minutes late, but {r2} is far more variable (range {rb} minutes against {ra}).",
            needs="a measure of spread such as the range",
            claim_good="the means are equal, but {r2} has a range of {rb} minutes against {ra} for {r1}, so the routes are not equally reliable",
            claim_bad="the mean is the same, so passengers will get the same service on both routes",
            bad_flaw="a mean says nothing about how much the trips vary around it, and variability is what reliability is about",
        ),
        wrong=[("ignores_spread", "does it accept the equal means as showing equal reliability?")],
    ),
    Case(
        "mean_not_a_value",
        slots={
            "paper": PAPERS,
            "vals": _children_vals,
            "n": lambda r, v: len(v["vals"]),
            "line": lambda r, v: values_line(v["vals"]),
            "mean": lambda r, v: num_text(mean_of(v["vals"]), 2),
            "mode": lambda r, v: mode_of(v["vals"])[0],
            "median": lambda r, v: num_text(median_of(v["vals"])),
            "t": lambda r, v: sum(v["vals"]),
            "town": ["Bellmore", "Corrington", "Easton Bay", "Halcombe"],
            **NAME_SLOTS,
        },
        fields=dict(
            setup='The numbers of children in {n} families in {town} were:\n{line}\n{paper} reports: "The typical {town} family has {mean} children." (The {mean} is the mean.)',
            misuse="no family can have {mean} children; the most common family size is {mode} children (the mode) and the median is {median}, which describe a typical family better than a mean of {mean}",
            figures="the {n} families have {t} children in total, giving a mean of {mean}; the mode is {mode}",
            better="The most common family size in {town} is {mode} children.",
            needs="the mode or the median, and the distribution of family sizes",
            claim_good="the mean of {mean} is not a number of children any family has, so the mode ({mode}) describes a typical family better",
            claim_bad="the mean is the average of all the families, so {mean} children is exactly what a typical family has",
            bad_flaw="a mean of whole-number data need not be a possible value, so it may not describe any actual family",
        ),
        wrong=[("accepts_mean_as_typical", "does it accept {mean} children as what a typical family has?")],
    ),
    Case(
        "range_outlier",
        slots={
            "paper": PAPERS,
            "vals": _range_outlier_vals,
            "n": lambda r, v: len(v["vals"]),
            "line": lambda r, v: values_line(v["vals"]),
            "lo": lambda r, v: min(v["vals"]),
            "hi": lambda r, v: max(v["vals"]),
            "rng": lambda r, v: max(v["vals"]) - min(v["vals"]),
            "rest_hi": lambda r, v: sorted(v["vals"])[-2],
            "rest_rng": lambda r, v: sorted(v["vals"])[-2] - min(v["vals"]),
            "m": lambda r, v: len(v["vals"]) - 1,
            "town": ["Bellmore", "Corrington", "Easton Bay", "Halcombe"],
            **NAME_SLOTS,
        },
        fields=dict(
            setup='The maximum temperatures (in degrees Celsius) in {town} on {n} summer days were:\n{line}\n{paper} reports: "Temperatures in {town} ranged wildly, from {lo} to {hi} degrees, showing how unpredictable our summers are."',
            misuse="the range ({rng}) is stretched by a single hot day ({hi}); the other {m} days lie between {lo} and {rest_hi}, a range of only {rest_rng}, so the spread is much smaller than the report suggests",
            figures="range {rng} with the hot day, but only {rest_rng} without it",
            better="On all but one of the {n} days the maximum was between {lo} and {rest_hi} degrees; one day reached {hi}.",
            needs="a spread measure not affected by one extreme day, such as the interquartile range",
            claim_good="one hot day of {hi} makes the range look large; without it the range is {rest_rng}",
            claim_bad="the range is the difference between the highest and lowest values, so a range of {rng} shows the temperatures vary wildly",
            bad_flaw="the range depends only on the two extreme values, so a single unusual day can make a settled pattern look wild",
        ),
        wrong=[
            (
                "accepts_wild_variation",
                "does it accept the range of {rng} as showing that the temperatures vary wildly?",
            )
        ],
    ),
    Case(
        "mode_extreme",
        slots={
            "paper": PAPERS,
            "vals": _mode_extreme_vals,
            "n": lambda r, v: len(v["vals"]),
            "line": lambda r, v: values_line(v["vals"]),
            "mode": lambda r, v: mode_of(v["vals"])[0],
            "mean": lambda r, v: num_text(mean_of(v["vals"])),
            "median": lambda r, v: num_text(median_of(v["vals"])),
            "below": lambda r, v: sum(1 for x in v["vals"] if x < median_of(v["vals"])),
            **NAME_SLOTS,
        },
        fields=dict(
            setup='The minutes of homework done last night by {n} students were:\n{line}\n{paper} reports: "The most common amount of homework is {mode} minutes, so that is what a typical student does."',
            misuse="the mode ({mode}) is the smallest value in the data and appears only twice; the median is {median} and the mean is {mean}, so a typical student does far more than {mode} minutes",
            figures="mode {mode}, median {median}, mean {mean}",
            better="The median amount of homework was {median} minutes; the most common single value, {mode} minutes, was done by only two students.",
            needs="the median or the mean and how the amounts are spread",
            claim_good="the mode of {mode} is the smallest value and occurs only twice, so it does not describe a typical student; the median is {median}",
            claim_bad="the mode is the most common value, so by definition it is the typical value",
            bad_flaw="the mode is the most common value, but when it occurs only a few times and sits at one end of the data it is not typical",
        ),
        wrong=[("accepts_mode_as_typical", "does it accept {mode} minutes as the typical amount?")],
    ),
]
MEDIA_STAT_TIERS = {
    "easy": Tier(
        prompt="{setup}\nWhat is wrong with the way the report uses this statistic?",
        reference=["The report misuses the statistic because {misuse}.", "Evidence from the data: {figures}."],
        points=[
            ("correct_idea", "identifies the misuse: {misuse}"),
            ("use_of_context", "uses the data: {figures}"),
            ("completeness", "explains why the statistic misleads, not only that it does"),
            ("communication", "names the measure (mean, median, mode or range) that is misused"),
        ],
        wrong=[("criticises_without_data", "does it criticise the report without using any number from the data?")],
        answer="{misuse}",
    ),
    "medium": Tier(
        prompt="{setup}\nUse the data to explain why the report's claim is not justified.",
        reference=["Figures: {figures}.", "The claim is not justified because {misuse}."],
        points=[
            ("correct_idea", "explains why the claim is not justified: {misuse}"),
            ("use_of_context", "quotes values from the data: {figures}"),
            ("completeness", "links the numbers to the weakness in the claim"),
            ("communication", "argues from the data to the conclusion"),
        ],
        wrong=[("data_not_used", "does the answer ignore the listed values and argue only in general terms?")],
        answer="{misuse}",
    ),
    "hard": Tier(
        prompt="{setup}\nRewrite the report's claim so that it uses a more suitable measure or describes the data more honestly, and say what other information a reader would want.",
        reference=[
            'A fairer statement: "{better}"',
            "A reader would want {needs}.",
            "The original misleads because {misuse}.",
        ],
        points=[
            ("correct_idea", "rewrites the claim with a suitable measure: {better}"),
            ("use_of_context", "the new claim is accurate for these data (it uses {figures})"),
            ("completeness", "names useful extra information: {needs}"),
            ("communication", "the rewrite is short and clear"),
        ],
        wrong=[
            (
                "rewrite_keeps_the_misuse",
                "does the rewritten claim still use the same misleading measure in the same way?",
            ),
            ("no_extra_information", "does it rewrite the claim but not say what other information is wanted?"),
        ],
        answer="{better}",
    ),
    "exam": Tier(
        prompt='{setup}\n{person} says: "{claim_bad}." {other} says: "{claim_good}." Evaluate both claims.',
        reference=["{other} is right: {claim_good}.", "{person} is wrong: {bad_flaw}.", "The data show {figures}."],
        points=[
            ("correct_idea", "judges {other} right: {claim_good}"),
            ("correct_idea", "judges {person} wrong: {bad_flaw}"),
            ("use_of_context", "uses the data: {figures}"),
            ("completeness", "gives a better way to summarise or describe the data"),
            ("communication", "responds to each person in turn"),
        ],
        wrong=[
            ("agrees_with_person", "does it agree with {person}?"),
            ("both_partly_right", "does it say both are partly right without deciding?"),
        ],
        answer="{other} is right",
    ),
}
register(
    "wa_media_centre_spread",
    12,
    MEDIA_STAT_CASES,
    MEDIA_STAT_TIERS,
    doc="misuse of mean, range and mode.",
)

# ---------------------------------------------------------------------------
# wa_presentation_suitability
# ---------------------------------------------------------------------------
SPORTS = ("football", "netball", "swimming", "tennis", "cricket", "basketball", "athletics", "hockey")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _counts_line(rng, names, lo=4, hi=30):
    k = len(names)
    counts = rng.sample(range(lo, hi + 1), k)
    return counts, ", ".join(f"{n} {c}" for n, c in zip(names, counts, strict=True))


def _cat_data(rng, v):
    names = rng.sample(SPORTS, rng.randint(5, 6))
    counts, line = _counts_line(rng, names)
    return dict(names=names, counts=counts, line=line, total=sum(counts), top=names[counts.index(max(counts))])


def _numeric_small(rng, v):
    vals = [rng.randint(8, 20) for _ in range(rng.randint(10, 14))]
    return dict(vals=vals, line=values_line(vals), n=len(vals))


def _rain_data(rng, v):
    vals = [rng.randint(8, 120) for _ in MONTHS]
    return dict(
        vals=vals,
        line=", ".join(f"{m} {x}" for m, x in zip(MONTHS, vals, strict=True)),
        peak=MONTHS[vals.index(max(vals))],
    )


def _pairs(rng, v):
    xs = sorted(rng.sample(range(150, 186), 12))
    pairs = [(x, x + rng.randint(-6, 6)) for x in xs]
    return dict(line="; ".join(f"({a}, {b})" for a, b in pairs), n=len(pairs))


def _many_cats(rng, v):
    cats = ["bus", "train", "car", "walk", "bike", "scooter", "tram", "ferry", "taxi", "motorbike", "skateboard"]
    names = rng.sample(cats, rng.randint(9, 11))
    counts, line = _counts_line(rng, names, 3, 40)
    return dict(names=names, line=line, total=sum(counts), k=len(names))


def _two_groups(rng, v):
    a = [rng.randint(48, 90) for _ in range(12)]
    b = [rng.randint(40, 95) for _ in range(12)]
    return dict(a=values_line(a), b=values_line(b), na=len(a), nb=len(b))


DISPLAY_CASES = [
    Case(
        "categorical",
        slots={
            "d": _cat_data,
            "line": lambda r, v: v["d"]["line"],
            "total": lambda r, v: v["d"]["total"],
            "top": lambda r, v: v["d"]["top"],
            **NAME_SLOTS,
        },
        fields=dict(
            setup="{person} asked {total} students to name their favourite sport. The number choosing each was: {line}.",
            purpose="compare how many students chose each sport",
            good="bar chart",
            good_why="each sport is a separate category and the bar lengths make the counts easy to compare",
            bad="histogram",
            bad_why="a histogram is for numerical data grouped into intervals, and sports are categories with no order",
            bad2="line graph",
            bad2_why="joining the categories with a line suggests an order or a trend from one sport to the next that does not exist",
            p2="show each sport's share of all {total} students",
            good2="pie chart",
            good2_why="a pie chart shows parts of a whole, and with only a few categories the shares can be read",
        ),
        wrong=[("draws_a_histogram", "does it recommend a histogram for categories?")],
    ),
    Case(
        "small_numeric",
        slots={"d": _numeric_small, "line": lambda r, v: v["d"]["line"], "n": lambda r, v: v["d"]["n"], **NAME_SLOTS},
        fields=dict(
            setup="{person} recorded the marks (out of 20) of {n} students on a short quiz: {line}.",
            purpose="show every mark and the shape of the distribution",
            good="dot plot",
            good_why="with only {n} values a dot plot shows every mark and any clusters or gaps",
            bad="pie chart",
            bad_why="marks are not parts of one whole, so a pie chart has nothing sensible to show",
            bad2="line graph",
            bad2_why="the students are not in a time order, so a line joining the marks would suggest a trend that is not there",
            p2="show the median and spread at a glance, so the class can be compared with another class",
            good2="box plot",
            good2_why="a box plot shows the median and quartiles and makes two classes easy to compare",
        ),
        wrong=[("pie_chart_for_marks", "does it recommend a pie chart for the marks?")],
    ),
    Case(
        "time_series",
        slots={"d": _rain_data, "line": lambda r, v: v["d"]["line"], "peak": lambda r, v: v["d"]["peak"], **NAME_SLOTS},
        fields=dict(
            setup="{person} recorded the monthly rainfall (in mm) in one town over a year: {line}. The wettest month was {peak}.",
            purpose="show how the rainfall changes through the year",
            good="line graph",
            good_why="the months are in time order, and a line shows the rises and falls over the year",
            bad="pie chart",
            bad_why="a pie chart shows shares of a whole at one moment and cannot show change over time",
            bad2="histogram",
            bad2_why="a histogram groups values into intervals and loses the order of the months",
            p2="compare the rainfall in the twelve months as separate amounts",
            good2="bar chart",
            good2_why="the bars let each month's amount be read and compared, though they are less good than a line for showing change",
        ),
        wrong=[("pie_chart_for_time", "does it recommend a pie chart for change over time?")],
    ),
    Case(
        "two_variables",
        slots={"d": _pairs, "line": lambda r, v: v["d"]["line"], "n": lambda r, v: v["d"]["n"], **NAME_SLOTS},
        fields=dict(
            setup="{person} measured the height and the arm span (both in cm) of {n} students. The pairs (height, arm span) were: {line}.",
            purpose="see whether taller students tend to have a longer arm span",
            good="scatterplot",
            good_why="a scatterplot puts one variable on each axis and shows whether the points follow a pattern",
            bad="bar chart",
            bad_why="a bar chart shows one variable for separate categories and cannot show how two numerical variables relate",
            bad2="pie chart",
            bad2_why="a pie chart cannot show two numerical variables, only shares of a total",
            p2="show only the spread of the heights, ignoring the arm spans",
            good2="dot plot",
            good2_why="a dot plot of the heights shows every value and the shape with a dozen or so values",
        ),
        wrong=[("bar_chart_for_two_variables", "does it recommend a bar chart for two numerical variables?")],
    ),
    Case(
        "many_categories",
        slots={
            "d": _many_cats,
            "line": lambda r, v: v["d"]["line"],
            "total": lambda r, v: v["d"]["total"],
            "k": lambda r, v: v["d"]["k"],
            **NAME_SLOTS,
        },
        fields=dict(
            setup="{person} asked {total} commuters how they usually travel to work. The numbers for the {k} ways of travelling were: {line}.",
            purpose="compare the numbers using each way of travelling",
            good="bar chart",
            good_why="with {k} categories the bars stay readable and are easy to compare",
            bad="pie chart",
            bad_why="with {k} slices, many of them thin, a pie chart is hard to read and similar slices cannot be compared",
            bad2="line graph",
            bad2_why="the ways of travelling have no order, so a line between them would be meaningless",
            p2="show which one or two ways are the most common",
            good2="bar chart in order of size",
            good2_why="sorting the bars from largest to smallest makes the most common ways stand out",
        ),
        wrong=[("pie_with_many_slices", "does it recommend a pie chart even though there are {k} categories?")],
    ),
    Case(
        "two_groups",
        slots={
            "d": _two_groups,
            "a": lambda r, v: v["d"]["a"],
            "b": lambda r, v: v["d"]["b"],
            "na": lambda r, v: v["d"]["na"],
            **NAME_SLOTS,
        },
        fields=dict(
            setup="{person} recorded the test marks of two classes. Class A ({na} students): {a}. Class B ({na} students): {b}.",
            purpose="compare the marks of the two classes",
            good="pair of side-by-side box plots",
            good_why="the medians and spreads of the two classes can be compared directly on the same scale",
            bad="pie chart",
            bad_why="a pie chart shows shares of one whole and cannot compare two distributions of marks",
            bad2="line graph",
            bad2_why="the students are not in a time order, so a line would suggest a trend that does not exist",
            p2="show every individual mark in both classes",
            good2="back-to-back stem plot",
            good2_why="with {na} values in each class a stem plot keeps every mark and shows both shapes",
        ),
        wrong=[("one_combined_display", "does it put both classes into a single display without separating them?")],
    ),
]
DISPLAY_TIERS = {
    "easy": Tier(
        prompt="{setup} The display is to {purpose}. Which is more suitable, a {good} or a {bad}? Give a reason that refers to these data.",
        reference=["A {good} is more suitable.", "Because {good_why}.", "A {bad} is not suitable because {bad_why}."],
        points=[
            ("correct_idea", "chooses the {good}"),
            ("use_of_context", "gives a reason that refers to these data: {good_why}"),
            ("completeness", "also says why the {bad} is unsuitable: {bad_why}"),
            ("communication", "states the choice and then the reason"),
        ],
        wrong=[
            ("chooses_the_unsuitable_display", "does it choose the {bad}?"),
            ("generic_reason", "does it give a reason like 'it looks better' with no reference to the data type?"),
        ],
        answer="a {good}",
    ),
    "medium": Tier(
        prompt="{setup} {other} suggests using a {bad2} to show these data. Explain why a {bad2} is unsuitable here, and suggest a display that would be better for this purpose: to {purpose}.",
        reference=["A {bad2} is unsuitable because {bad2_why}.", "A better display is a {good}, because {good_why}."],
        points=[
            ("correct_idea", "explains why the {bad2} is unsuitable: {bad2_why}"),
            ("correct_idea", "suggests a suitable display for the purpose: the {good}"),
            ("use_of_context", "refers to the type of data (categories, times, or numbers) in this situation"),
            ("completeness", "justifies the better display: {good_why}"),
            ("communication", "answers both parts in order"),
        ],
        wrong=[
            (
                "recommends_another_poor_display",
                "does it recommend a display that is also unsuitable for this purpose, such as a {bad}?",
            )
        ],
        answer="a {good}, not a {bad2}",
    ),
    "hard": Tier(
        prompt="{setup} The same data are to be used for two purposes. Purpose 1: {purpose}. Purpose 2: {p2}. Recommend a display for each purpose and justify each choice.",
        reference=[
            "Purpose 1: a {good}, because {good_why}.",
            "Purpose 2: a {good2}, because {good2_why}.",
            "The purposes differ, so the best display differs.",
        ],
        points=[
            ("correct_idea", "recommends the {good} for purpose 1"),
            ("correct_idea", "recommends the {good2} for purpose 2"),
            ("use_of_context", "justifies each with the data: {good_why}; {good2_why}"),
            ("completeness", "gives a reason for both purposes, not only one"),
            ("communication", "labels the answer for purpose 1 and purpose 2"),
        ],
        wrong=[
            (
                "same_display_for_both",
                "does it choose the same display for both purposes without checking that it suits both?",
            ),
            ("unsuitable_choice", "does it choose a {bad} for either purpose?"),
        ],
        answer="{good}; {good2}",
    ),
    "exam": Tier(
        prompt='{setup} The display is to {purpose}. {person} says: "A {bad2} is fine, because it uses all of the data." {other} says: "A {good} is better." Evaluate both claims, and explain what would go wrong if the {bad2} were used.',
        reference=[
            "{other} is right: a {good} is better because {good_why}.",
            "{person} is wrong: using all the data is not enough; a {bad2} is unsuitable because {bad2_why}.",
            "Using a {bad2} would mislead the reader.",
        ],
        points=[
            ("correct_idea", "judges {other} right and {person} wrong"),
            ("correct_idea", "explains why the {bad2} is unsuitable: {bad2_why}"),
            ("use_of_context", "supports the {good}: {good_why}"),
            ("completeness", "describes what a reader would wrongly conclude from the {bad2}"),
            ("communication", "responds to each person in turn"),
        ],
        wrong=[
            ("agrees_with_person", "does it agree with {person} that using all the data makes any display suitable?"),
            ("both_acceptable", "does it say both displays are acceptable?"),
        ],
        answer="{other} is right: a {good}",
    ),
}
register(
    "wa_presentation_suitability",
    12,
    DISPLAY_CASES,
    DISPLAY_TIERS,
    doc="choosing a data display for a stated purpose.",
)

# ---------------------------------------------------------------------------
# wa_percent_association
# ---------------------------------------------------------------------------
ASSOC_CTX = (
    ("students in Year 8", "students in Year 11", "buy lunch from the canteen"),
    ("passengers in the city", "passengers in regional areas", "are satisfied with the train service"),
    ("members under 40", "members aged 40 and over", "use the morning classes"),
    ("households with a garden", "households without a garden", "own a pet"),
    ("students who play a sport", "students who do not play a sport", "sleep at least eight hours on a school night"),
    (
        "learner drivers who learned in an automatic car",
        "learner drivers who learned in a manual car",
        "passed the driving test first time",
    ),
)


def pp(d: int) -> str:
    return f"{d} percentage point" + ("" if d == 1 else "s")


def _assoc(rng, kind: str) -> dict:
    g1, g2, resp = rng.choice(ASSOC_CTX)
    for _ in attempts(500, "a two-way table of the wanted strength"):
        n1, n2 = rng.choice((100, 200, 300, 400)), rng.choice((100, 200, 300, 400))
        p1, p2 = rng.randint(15, 85), rng.randint(15, 85)
        d = abs(p1 - p2)
        ok = {"strong": d >= 20, "moderate": 8 <= d <= 15, "none": 1 <= d <= 3, "flip": d >= 10}[kind]
        if kind == "flip":
            # the bigger group has more "yes" counts but a smaller percentage
            big_is_1 = n1 > n2
            pb, ps = (p1, p2) if big_is_1 else (p2, p1)
            nb, ns = (n1, n2) if big_is_1 else (n2, n1)
            ok = ok and nb != ns and pb < ps and nb * pb // 100 > ns * ps // 100
        if ok:
            break
    k1, k2 = n1 * p1 // 100, n2 * p2 // 100
    return dict(g1=g1, g2=g2, resp=resp, n1=n1, n2=n2, k1=k1, k2=k2, p1=p1, p2=p2, d=abs(p1 - p2))


def _assoc_table(a: dict) -> str:
    return f"Among {a['g1']}: {a['k1']} out of {a['n1']} {a['resp']}.\nAmong {a['g2']}: {a['k2']} out of {a['n2']} {a['resp']}."


def _sides(a: dict):
    """(big group, small group, big count, small count, big n, small n, big %, small %)."""
    if a["n1"] > a["n2"]:
        return a["g1"], a["g2"], a["k1"], a["k2"], a["n1"], a["n2"], a["p1"], a["p2"]
    return a["g2"], a["g1"], a["k2"], a["k1"], a["n2"], a["n1"], a["p2"], a["p1"]


def _pa_easy(rng) -> dict:
    a = _assoc(rng, rng.choice(["strong", "moderate"]))
    text = (
        f"A survey produced these results.\n{_assoc_table(a)}\n"
        f"Use percentages to describe the difference between the two groups in how many {a['resp']}."
    )
    hi, lo = (a["g1"], a["g2"]) if a["p1"] > a["p2"] else (a["g2"], a["g1"])
    phi, plo = max(a["p1"], a["p2"]), min(a["p1"], a["p2"])
    mark = marking(
        reference=[
            f"{a['p1']}% of the {a['g1']} ({a['k1']} out of {a['n1']}) {a['resp']}, compared with {a['p2']}% of the {a['g2']} ({a['k2']} out of {a['n2']}).",
            f"The percentage is higher for the {hi}, by {pp(a['d'])} ({phi}% against {plo}%), so there is an association between the group and the response.",
        ],
        points=[
            ("correct_idea", f"calculates both percentages correctly: {a['p1']}% and {a['p2']}%"),
            ("correct_idea", f"states the difference as {pp(a['d'])}, with the {hi} higher"),
            (
                "use_of_context",
                f"describes the result for the two groups ({a['g1']} and {a['g2']}) and the response ({a['resp']})",
            ),
            ("completeness", "interprets the difference as an association between the group and the response"),
            ("communication", "writes the comparison as a sentence using both percentages"),
        ],
        wrong=[
            (
                "compares_counts",
                f"does it compare {a['k1']} with {a['k2']} directly rather than the percentages of each group?",
            ),
            (
                "difference_as_percent",
                "does it call the difference 'percent' instead of 'percentage points', or give the wrong size?",
            ),
        ],
    )
    return finish(text, f"{phi}% against {plo}%, a difference of {pp(a['d'])}", mark)


def _pa_medium(rng) -> dict:
    a = _assoc(rng, "flip")
    big, small, kb, ks, nb, ns, pb, ps = _sides(a)
    p = rng.choice(NAMES)
    text = (
        f"A survey produced these results.\n{_assoc_table(a)}\n"
        f'{p} says: "More of the {big} {a["resp"]} than of the {small}, because {kb} is more than {ks}, so the {big} are more likely to." '
        f"Explain what is wrong with {p}'s reasoning, and describe the association using percentages."
    )
    mark = marking(
        reference=[
            f"{p} compared counts ({kb} and {ks}), but the groups are different sizes ({nb} and {ns}).",
            f"As percentages, {pb}% of the {big} {a['resp']}, compared with {ps}% of the {small}, so the {small} are more likely to, by {pp(ps - pb)}.",
        ],
        points=[
            (
                "correct_idea",
                f"explains that counts cannot be compared when the groups differ in size ({nb} against {ns})",
            ),
            (
                "correct_idea",
                f"gives the percentages {pb}% and {ps}% and concludes that the {small} are more likely to",
            ),
            ("use_of_context", f"uses the groups and the response: {a['resp']}"),
            ("completeness", f"states the difference of {pp(ps - pb)}"),
            ("communication", f"addresses {p}'s claim directly"),
        ],
        wrong=[
            ("agrees_with_counts", f"does it agree with {p} because {kb} is more than {ks}?"),
            ("percentages_of_total", "does it take percentages of the total of both groups instead of each group?"),
        ],
    )
    return finish(text, f"{p} is wrong: {pb}% against {ps}%", mark)


def _pa_hard(rng) -> dict:
    a = _assoc(rng, "none")
    text = (
        f"A survey produced these results.\n{_assoc_table(a)}\n"
        "Use percentages to decide whether the data suggest an association between the group and the response, and say what you can and cannot conclude."
    )
    mark = marking(
        reference=[
            f"{a['p1']}% of the {a['g1']} and {a['p2']}% of the {a['g2']} {a['resp']}, a difference of only {pp(a['d'])}.",
            "The percentages are nearly the same, so the data show little or no association between the group and the response.",
            "Even a small difference in a sample could be due to chance, and a survey like this cannot show that the group causes the response.",
        ],
        points=[
            ("correct_idea", f"calculates {a['p1']}% and {a['p2']}% and the small difference of {pp(a['d'])}"),
            ("correct_idea", "concludes there is little or no association"),
            ("use_of_context", "describes the result in terms of the two groups and the response"),
            ("completeness", "notes that a small difference may be chance and that association is not cause"),
            ("communication", "states the conclusion clearly"),
        ],
        wrong=[
            (
                "finds_association_from_tiny_difference",
                f"does it claim a clear association from a difference of {pp(a['d'])}?",
            ),
            ("claims_cause", "does it say the group causes the difference?"),
        ],
    )
    return finish(text, f"little or no association ({a['p1']}% against {a['p2']}%)", mark)


def _pa_exam(rng) -> dict:
    a = _assoc(rng, "flip")
    big, small, kb, ks, nb, ns, pb, ps = _sides(a)
    p, q = rng.sample(NAMES, 2)
    text = (
        f"A survey produced these results.\n{_assoc_table(a)}\n"
        f'{p} says: "There is no real difference, because about the same number in each group {a["resp"].split(" ", 1)[0] if False else ""}'
    )
    text = (
        f"A survey produced these results.\n{_assoc_table(a)}\n"
        f'{p} says: "There is no real difference between the groups, because the counts, {kb} and {ks}, are fairly close." '
        f'{q} says: "The {small} are more likely to, so being in that group makes people more likely to." '
        "Evaluate both claims, and write a correct summary of the association."
    )
    mark = marking(
        reference=[
            f"{p} is wrong: the groups are different sizes ({nb} and {ns}), so counts are not comparable.",
            f"{q} is right that the percentage is higher for the {small} ({ps}% against {pb}%, a difference of {pp(ps - pb)}), but wrong to say being in that group causes it: a survey shows association only.",
            f"A correct summary: {ps}% of the {small} {a['resp']} compared with {pb}% of the {big}, so there is an association between group and response.",
        ],
        points=[
            ("correct_idea", f"rejects {p}'s claim because the group sizes differ ({nb}, {ns}) and uses percentages"),
            ("correct_idea", f"accepts {q}'s comparison ({ps}% against {pb}%) but rejects the claim of cause"),
            ("use_of_context", "writes a summary that uses both percentages and the groups"),
            ("completeness", f"states the difference of {pp(ps - pb)}"),
            ("communication", "evaluates each person in turn before the summary"),
        ],
        wrong=[
            ("accepts_equal_counts", f"does it accept {p}'s claim that the counts show no difference?"),
            ("accepts_cause", f"does it accept {q}'s claim that the group causes the response?"),
        ],
    )
    return finish(text, f"{ps}% against {pb}%: an association, not proof of cause", mark)


register_custom(
    "wa_percent_association",
    12,
    dict(easy=_pa_easy, medium=_pa_medium, hard=_pa_hard, exam=_pa_exam),
    doc="percentages in a two-way table.",
)

# ---------------------------------------------------------------------------
# wa_bivariate_features
# ---------------------------------------------------------------------------
#: (x label, y label, x range, y range)
BIV_CTX = {
    "pos": (
        ("hours of study per week", "test mark (%)", (1, 24), (30, 95)),
        ("daily maximum temperature (degrees)", "ice creams sold", (18, 38), (20, 180)),
        ("height (cm)", "arm span (cm)", (150, 188), (148, 192)),
    ),
    "neg": (
        ("age of a used car (years)", "price ($ thousand)", (1, 16), (4, 40)),
        ("hours of television per week", "test mark (%)", (2, 36), (35, 90)),
        ("distance from the town centre (km)", "rent ($ per week)", (1, 24), (250, 700)),
    ),
    "none": (
        ("shoe size (European)", "test score (%)", (30, 44), (30, 90)),
        ("month of birth (1 to 12)", "height (cm)", (1, 12), (140, 180)),
    ),
    "curved": (
        ("speed of a car (km/h)", "fuel used per 100 km (L)", (40, 130), (5, 20), False),
        ("room temperature (degrees)", "comfort score (out of 10)", (10, 34), (1, 10), True),
    ),
}


def _r(xs, ys) -> float:
    return statistics.correlation(xs, ys)


def _clip(t: float) -> float:
    return min(1.0, max(0.0, t))


def _pairs_for(rng, kind: str, n: int, ctx):
    """x and y values for the wanted kind of relationship, scaled to the context's ranges."""
    (x0, x1), (y0, y1) = ctx[2], ctx[3]
    n = min(n, x1 - x0 + 1)
    for _ in attempts(1500, f"a {kind} dataset"):
        if kind == "curved":
            step = max(1, (x1 - x0) // (n - 1))
            xs = [x0 + i * step for i in range(n)]
            mid = (xs[0] + xs[-1]) / 2
            half = (xs[-1] - xs[0]) / 2
            ts = [0.1 + 0.8 * ((x - mid) / half) ** 2 + rng.uniform(-0.04, 0.04) for x in xs]
            if ctx[4]:
                ts = [1 - t for t in ts]
        else:
            xs = sorted(rng.sample(range(x0, x1 + 1), n))
            trend = [(x - x0) / (x1 - x0) for x in xs]
            if kind == "pos":
                ts = [0.08 + 0.84 * t + rng.uniform(-0.07, 0.07) for t in trend]
            elif kind == "neg":
                ts = [0.92 - 0.84 * t + rng.uniform(-0.2, 0.2) for t in trend]
            else:
                ts = [rng.uniform(0.05, 0.95) for _ in xs]
        ys = [round(y0 + (y1 - y0) * _clip(t)) for t in ts]
        r = _r(xs, ys)
        ok = {"pos": r >= 0.9, "neg": -0.9 <= r <= -0.55, "none": abs(r) <= 0.2, "curved": abs(r) <= 0.2}[kind]
        if ok:
            return xs, ys
    raise VerificationError("no dataset of the wanted kind")


def _fmt_pairs(xs, ys) -> str:
    return "\n".join(f"({x}, {y})" for x, y in zip(xs, ys, strict=True))


def _bv_easy(rng) -> dict:
    kind = rng.choice(["pos", "neg", "none"])
    ctx = rng.choice(BIV_CTX[kind])
    xl, yl = ctx[0], ctx[1]
    n = rng.randint(9, 13)
    xs, ys = _pairs_for(rng, kind, n, ctx)
    n = len(xs)
    r = round(_r(xs, ys), 2)
    text = (
        f"A class recorded {xl} (x) and {yl} (y) for {n} cases. The pairs (x, y) were:\n{_fmt_pairs(xs, ys)}\n"
        f"The correlation coefficient is r = {r:.2f}. Describe the relationship between x and y in terms of direction and strength, and say what the value of r tells us."
    )
    words = {
        "pos": "positive and strong",
        "neg": "negative and moderate to strong",
        "none": "very weak, with no clear association",
    }[kind]
    direction = {
        "pos": "As x increases, y tends to increase.",
        "neg": "As x increases, y tends to decrease.",
        "none": "There is no consistent increase or decrease in y as x increases.",
    }[kind]
    mark = marking(
        reference=[
            f"The relationship is {words}. {direction}",
            f"r = {r:.2f}, which is {'close to +1' if kind == 'pos' else 'between -1 and 0 and some distance from 0' if kind == 'neg' else 'close to 0'}, supporting this.",
        ],
        points=[
            ("correct_idea", f"states the direction correctly: {direction}"),
            ("correct_idea", f"describes the strength as {words}, consistent with r = {r:.2f}"),
            ("use_of_context", f"refers to the variables ({xl} and {yl}) in the description"),
            (
                "completeness",
                "says what r means: its sign gives the direction and its distance from 0 gives the strength",
            ),
            ("communication", "uses the words positive, negative, strong, moderate or weak correctly"),
        ],
        wrong=[
            ("direction_reversed", "does it describe the direction the wrong way round?"),
            (
                "r_misread",
                "does it say r = {:.2f} means {:.0f}% of the points lie on the line, or read r as a percentage?".format(
                    r, abs(r) * 100
                ),
            ),
        ],
    )
    return finish(text, words, mark)


def _bv_medium(rng) -> dict:
    ctx = rng.choice(BIV_CTX["pos"])
    xl, yl = ctx[0], ctx[1]
    n = rng.randint(10, 13)
    for _ in attempts(300, "an outlier that changes r clearly"):
        xs, ys = _pairs_for(rng, "pos", n, ctx)
        n = len(xs)
        i = rng.randrange(2, n - 2)
        ys2 = list(ys)
        ys2[i] = ctx[3][0] if i >= n // 2 else ctx[3][1]
        r_with = round(_r(xs, ys2), 2)
        r_without = round(_r(xs[:i] + xs[i + 1 :], ys2[:i] + ys2[i + 1 :]), 2)
        if r_without - r_with >= 0.1 and abs(ys2[i] - ys[i]) >= 0.4 * (ctx[3][1] - ctx[3][0]):
            break
    text = (
        f"A class recorded {xl} (x) and {yl} (y) for {n} cases. The pairs (x, y) were:\n{_fmt_pairs(xs, ys2)}\n"
        f"The correlation coefficient for all {n} pairs is r = {r_with:.2f}; without the one unusual pair it is r = {r_without:.2f}. "
        "Describe the pattern, identify the unusual pair, and explain how it affects the correlation coefficient."
    )
    mark = marking(
        reference=[
            f"Apart from one pair, y increases as x increases in a strong positive pattern (r = {r_without:.2f} without it).",
            f"The unusual pair is x = {xs[i]}, y = {ys2[i]}; it is an outlier because it lies far from the pattern of the others.",
            f"It weakens the correlation from {r_without:.2f} to {r_with:.2f}, so one point can change r a great deal.",
        ],
        points=[
            ("correct_idea", "describes the pattern of the other points as strong and positive"),
            ("correct_idea", f"identifies the outlier correctly: x = {xs[i]}, y = {ys2[i]}"),
            ("use_of_context", f"uses both values of r: {r_with:.2f} with the outlier and {r_without:.2f} without"),
            ("completeness", "explains that the outlier reduces the apparent strength of the relationship"),
            ("communication", "describes direction, strength and the outlier separately"),
        ],
        wrong=[
            ("outlier_misidentified", f"does it name a point other than ({xs[i]}, {ys2[i]}) as the outlier?"),
            ("ignores_the_effect_on_r", "does it find the outlier but not say how it changes r?"),
        ],
    )
    return finish(text, f"strong positive with one outlier ({xs[i]}, {ys2[i]})", mark)


def _bv_hard(rng) -> dict:
    ctx = rng.choice(BIV_CTX["curved"])
    xl, yl = ctx[0], ctx[1]
    n = rng.randint(10, 14)
    xs, ys = _pairs_for(rng, "curved", n, ctx)
    n = len(xs)
    r = round(_r(xs, ys), 2)
    p = rng.choice(NAMES)
    low = ys.index(max(ys) if ctx[4] else min(ys))
    text = (
        f"A study recorded {xl} (x) and {yl} (y) for {n} cases. The pairs (x, y) were:\n{_fmt_pairs(xs, ys)}\n"
        f'The correlation coefficient is r = {r:.2f}. {p} says: "r is close to 0, so there is no relationship between x and y." '
        f"Evaluate {p}'s claim and describe the pattern in the data."
    )
    mark = marking(
        reference=[
            f"{p} is wrong. y {'rises' if ctx[4] else 'falls'} as x increases to about {xs[low]} (where y is {'highest' if ctx[4] else 'lowest'}, {ys[low]}) and then {'falls' if ctx[4] else 'rises'} again, a curved (non-linear) pattern.",
            f"r = {r:.2f} measures only the strength of a straight-line relationship; a curved relationship can have r close to 0 and still be strong.",
            "The right step is to look at a plot of the data, not only at r.",
        ],
        points=[
            (
                "correct_idea",
                f"rejects {p}'s claim and describes the curve: y {'rises then falls, highest' if ctx[4] else 'falls then rises, lowest'} near x = {xs[low]}",
            ),
            ("correct_idea", "explains that r measures only linear association"),
            ("use_of_context", f"uses the data and r = {r:.2f}"),
            ("completeness", "recommends looking at the scatterplot as well as the value of r"),
            ("communication", f"responds directly to {p}'s statement"),
        ],
        wrong=[
            ("agrees_no_relationship", f"does it agree with {p} that r near 0 means no relationship?"),
            ("describes_linear_trend", "does it describe a straight-line increase or decrease?"),
        ],
    )
    return finish(text, f"{p} is wrong: the relationship is curved", mark)


_LURK = (
    (
        "daily ice cream sales",
        "daily sunburn cases",
        "hot weather",
        "Ice cream sales cause sunburn.",
        (50, 300),
        (2, 40),
    ),
    (
        "the number of fire engines at a fire",
        "the damage caused by the fire ($ thousand)",
        "the size of the fire",
        "Fire engines cause the damage.",
        (1, 12),
        (20, 400),
    ),
    (
        "shoe size of primary students (European)",
        "reading score (%)",
        "age",
        "Bigger feet make students better readers.",
        (25, 38),
        (20, 95),
    ),
)


def _bv_exam(rng) -> dict:
    xl, yl, lurk, bad, xr, yr = rng.choice(_LURK)
    n = rng.randint(9, 12)
    xs, ys = _pairs_for(rng, "pos", n, (xl, yl, xr, yr))
    n = len(xs)
    r = round(_r(xs, ys), 2)
    p, q = rng.sample(NAMES, 2)
    text = (
        f"Data were collected on {xl} (x) and {yl} (y) for {n} cases. The pairs (x, y) were:\n{_fmt_pairs(xs, ys)}\n"
        f'The correlation coefficient is r = {r:.2f}. {p} says: "{bad}" {q} says: "The two variables move together, but {lurk} may explain both." '
        "Evaluate both claims and describe what the data do show."
    )
    mark = marking(
        reference=[
            f"The data show a strong positive association (r = {r:.2f}): when x is higher, y tends to be higher.",
            f"{q} is right: a strong correlation does not show that one variable causes the other, and {lurk} could be affecting both.",
            f"{p} goes beyond the data by claiming a cause.",
        ],
        points=[
            ("correct_idea", f"describes the association as strong and positive, using r = {r:.2f}"),
            ("correct_idea", f"rejects {p}'s causal claim: correlation is not causation"),
            ("use_of_context", f"supports {q}'s explanation that {lurk} could affect both variables"),
            ("completeness", "says what the data do show (an association) as well as what they do not"),
            ("communication", "evaluates each person's claim in turn"),
        ],
        wrong=[
            ("accepts_causal_claim", f"does it accept {p}'s claim of cause because r is large?"),
            (
                "denies_any_relationship",
                "does it say there is no relationship at all, instead of an association without proof of cause?",
            ),
        ],
    )
    return finish(text, "a strong association, but not proof of cause", mark)


register_custom(
    "wa_bivariate_features",
    12,
    dict(easy=_bv_easy, medium=_bv_medium, hard=_bv_hard, exam=_bv_exam),
    doc="direction, strength, form and outliers.",
)

# ---------------------------------------------------------------------------
# wa_interpret_relationships: a fitted line read in context
# ---------------------------------------------------------------------------
MODELS = (
    dict(
        y_phrase="the cost of hiring a bike",
        y_unit="dollars",
        x_phrase="the length of hire",
        x_unit="hours",
        x_one="extra hour of hire",
        a=(5, 15),
        b=(6, 12),
        lo=1,
        hi=8,
        ext=(4, 8),
        icpt="the fixed booking fee charged before any riding",
        icpt_note="this is meaningful, because a fee can be charged before any riding",
    ),
    dict(
        y_phrase="the height of a plant",
        y_unit="cm",
        x_phrase="the time since planting",
        x_unit="weeks",
        x_one="extra week",
        a=(4, 10),
        b=(2, 5),
        lo=1,
        hi=10,
        ext=(4, 8),
        icpt="the height of the plant when it was planted",
        icpt_note="this is meaningful, because it is the plant's starting height",
    ),
    dict(
        y_phrase="a student's test mark",
        y_unit="marks",
        x_phrase="the time spent studying",
        x_unit="hours",
        x_one="extra hour of study",
        a=(35, 50),
        b=(3, 6),
        lo=1,
        hi=10,
        ext=(4, 8),
        icpt="the predicted mark for a student who does no study",
        icpt_note="this is plausible, because students can score some marks with no study, and 0 hours is close to the data",
    ),
    dict(
        y_phrase="a child's height",
        y_unit="cm",
        x_phrase="the child's age",
        x_unit="years",
        x_one="extra year of age",
        a=(70, 85),
        b=(5, 7),
        lo=5,
        hi=12,
        ext=(4, 8),
        icpt="a height at age 0",
        icpt_note="this has no real meaning, because age 0 lies far outside the data (ages 5 to 12), so the line should not be used there",
    ),
    dict(
        y_phrase="the number of ice creams sold in a day",
        y_unit="ice creams",
        x_phrase="the day's maximum temperature",
        x_unit="degrees Celsius",
        x_one="degree increase in the maximum temperature",
        a=(-60, -20),
        b=(8, 14),
        lo=18,
        hi=36,
        ext=(4, 8),
        icpt="a negative number of ice creams at a maximum of 0 degrees",
        icpt_note="this has no real meaning, because sales cannot be negative and 0 degrees lies far outside the data (18 to 36 degrees)",
    ),
    dict(
        y_phrase="the fuel left in a tank",
        y_unit="litres",
        x_phrase="the distance driven",
        x_unit="hundreds of km",
        x_one="extra 100 km driven",
        a=(55, 70),
        b=(-8, -6),
        lo=1,
        hi=3,
        ext=(1, 2),
        icpt="the fuel in a full tank before any driving",
        icpt_note="this is meaningful, because it is the starting amount of fuel",
    ),
)


def _model_slots():
    def num(key):
        return lambda r, v: r.randint(*v["m"][key])

    def field(key):
        return lambda r, v: v["m"][key]

    return {
        "m": lambda r, v: r.choice(MODELS),
        "a": num("a"),
        "b": num("b"),
        **{k: field(k) for k in ("y_phrase", "y_unit", "x_phrase", "x_unit", "x_one", "lo", "hi", "icpt", "icpt_note")},
        "eq": lambda r, v: f"y = {v['a']} {'+' if v['b'] > 0 else '-'} {abs(v['b'])}x",
        "absb": lambda r, v: abs(v["b"]),
        "dir": lambda r, v: "increases" if v["b"] > 0 else "decreases",
        "sign": lambda r, v: "+" if v["b"] > 0 else "-",
        "x_in": lambda r, v: r.randint(v["lo"] + 1, v["hi"] - 1),
        "y_in": lambda r, v: v["a"] + v["b"] * v["x_in"],
        "x_out": lambda r, v: v["hi"] + r.randint(*v["m"]["ext"]),
        "y_out": lambda r, v: v["a"] + v["b"] * v["x_out"],
        **NAME_SLOTS,
    }


MODEL_CASES = [
    Case(
        "line",
        slots=_model_slots(),
        fields=dict(
            setup="A line was fitted to data on {y_phrase} against {x_phrase}, using values of x from {lo} to {hi}. The fitted line is {eq}, where y is {y_phrase} (in {y_unit}) and x is {x_phrase} (in {x_unit}).",
            slope_text="for each {x_one}, {y_phrase} {dir} by {absb} {y_unit}, on average",
            icpt_text="when x is 0 the line predicts {a} {y_unit}, which is {icpt}",
            in_text="{y_in} {y_unit}",
            out_text="{y_out} {y_unit}",
        ),
        wrong=[
            (
                "slope_as_total",
                "does it read the slope as the value of y itself rather than the change in y for each {x_one}?",
            )
        ],
    )
]
MODEL_TIERS = {
    "easy": Tier(
        prompt="{setup} Interpret the number {absb} in the equation, in the context of the situation.",
        reference=[
            "The slope is {b}: {slope_text}.",
            "The slope is a rate of change (the change in y for one extra unit of x), not the value of y itself.",
        ],
        points=[
            ("correct_idea", "reads the number as a rate of change (the slope), not a total"),
            ("correct_idea", "gives the right direction: {y_phrase} {dir}"),
            ("use_of_context", "states the amount and the units: {absb} {y_unit}"),
            ("completeness", "links it to one extra unit of x: each {x_one}"),
            ("communication", "writes a full sentence in the context"),
        ],
        wrong=[("direction_reversed", "does it say {y_phrase} goes the wrong way for each {x_one}?")],
        answer="{slope_text}",
    ),
    "medium": Tier(
        prompt="{setup} Interpret the slope and the intercept in the context of the situation, and say whether the intercept has a sensible meaning.",
        reference=["Slope: {slope_text}.", "Intercept: {icpt_text}.", "Is the intercept meaningful? {icpt_note}."],
        points=[
            ("correct_idea", "interprets the slope as a rate of change: {slope_text}"),
            ("correct_idea", "interprets the intercept as the predicted value when x is 0: {a} {y_unit}"),
            ("use_of_context", "uses the units and the words of the situation"),
            ("completeness", "judges whether the intercept is meaningful: {icpt_note}"),
            ("communication", "treats the slope and the intercept separately"),
        ],
        wrong=[
            ("intercept_as_the_rate", "does it swap the roles of the slope and the intercept?"),
            (
                "always_meaningful",
                "does it say the intercept is meaningful without checking whether x = 0 is within the range of the data ({lo} to {hi})?",
            ),
        ],
        answer="slope: {slope_text}; intercept: {icpt_text}",
    ),
    "hard": Tier(
        prompt="{setup} Use the line to predict y when x = {x_in} and when x = {x_out}. Which prediction would you trust more, and why?",
        reference=[
            "When x = {x_in}, y = {a} {sign} {absb} × {x_in} = {y_in} {y_unit}.",
            "When x = {x_out}, y = {y_out} {y_unit}.",
            "The first is more reliable: {x_in} lies inside the range of the data ({lo} to {hi}), while {x_out} is outside it, so the second is an extrapolation and the pattern may not continue.",
        ],
        points=[
            ("correct_idea", "predicts {y_in} {y_unit} at x = {x_in}"),
            ("correct_idea", "predicts {y_out} {y_unit} at x = {x_out}"),
            ("use_of_context", "refers to the range of the data ({lo} to {hi} {x_unit})"),
            ("completeness", "trusts the first prediction and explains that the second is an extrapolation"),
            ("communication", "shows how each prediction was found"),
        ],
        wrong=[
            ("trusts_both_equally", "does it treat both predictions as equally reliable?"),
            ("arithmetic_error", "does it use the slope and intercept correctly, with the right sign?"),
        ],
        answer="{y_in} {y_unit} (reliable); {y_out} {y_unit} (extrapolation)",
    ),
    "exam": Tier(
        prompt='{setup} {person} says: "The line predicts {y_out} {y_unit} when x = {x_out}, so that is what will happen." {other} says: "x = {x_out} is outside the data, so the prediction might not hold." Evaluate both claims, and explain what the slope of the line tells us.',
        reference=[
            "{other} is right: x = {x_out} is outside the range of the data ({lo} to {hi}), so predicting {y_out} {y_unit} is an extrapolation and may be unreliable.",
            "{person} is wrong to say it 'will happen': a prediction from a line is an estimate, and more so beyond the data.",
            "The slope: {slope_text}.",
        ],
        points=[
            ("correct_idea", "judges {other} right: the prediction is an extrapolation beyond {lo} to {hi}"),
            ("correct_idea", "judges {person} wrong: a prediction from a line is only an estimate"),
            ("use_of_context", "uses the numbers: {y_out} {y_unit} at x = {x_out}, data from {lo} to {hi}"),
            ("completeness", "interprets the slope: {slope_text}"),
            ("communication", "responds to each person in turn"),
        ],
        wrong=[
            ("agrees_with_person", "does it agree that the prediction is certain?"),
            (
                "rejects_all_prediction",
                "does it say the line is useless for predicting at all, even within the range of the data?",
            ),
        ],
        answer="{other} is right; {slope_text}",
    ),
}
register(
    "wa_interpret_relationships",
    12,
    MODEL_CASES,
    MODEL_TIERS,
    doc="reading a fitted line in context.",
)

check(len(MODELS) == 6, "six fitted-line contexts")
