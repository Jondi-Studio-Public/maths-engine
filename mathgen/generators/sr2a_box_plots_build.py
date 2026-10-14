"""Generates `parallel-box-plots-outlier-fences` questions (easy, medium, hard) and the exam chain `sr2a_box_plots_chain`.

Using two drawn parallel box plots (ACMGM031, with the box-plot reading of ACMEM059), `statistics-probability`, answer
``number``. The figure is a two-plot ``box_plot`` figure: every value is an even whole number on a half tick, so each
interquartile range is even and every 1.5 x IQR fence is a whole number. The prompt states the 1.5 x IQR rule and never
a quartile.

* easy: the interquartile range of a named group, or how much higher one group's median is than the other's;
* medium: the upper or the lower fence of a named group (the value beyond which a result would be an outlier);
* hard: how far a drawn outlier lies beyond its fence, or (for a group with no outlier) the smallest whole number that
  would count as an outlier above the plot, which needs the "more than" to be read strictly;
* exam (``sr2a_box_plots_chain``): one group has a drawn outlier. (a) its interquartile range, (b) the fence on the
  outlier's side (follow-through from (a)), (c) how far the outlier lies beyond it (follow-through from (b)), (d) which
  group is more variable. The group with the outlier always has the larger range but the smaller interquartile range, so
  the range is the trap in (d).

The registered readers recompute every answer from the plotted five-number summaries and outliers.
"""

from __future__ import annotations

import re

from .. import figures
from ..core import attempts, check, generator, phrase
from ..figure_kinds.data import box_plot
from ._sr2a_kit import mcq_part, number_part, single

_CTX = (
    dict(
        lead="The box plots compare the delivery times of two couriers.",
        x="Delivery time (minutes)",
        groups=("Courier A", "Courier B"),
    ),
    dict(
        lead="The box plots compare the marks of two classes on the same test.",
        x="Mark (%)",
        groups=("Class A", "Class B"),
    ),
    dict(
        lead="The box plots compare the battery life of two brands of torch.",
        x="Battery life (hours)",
        groups=("Brand X", "Brand Y"),
    ),
    dict(
        lead="The box plots compare the weekly screen time of two year groups.",
        x="Screen time (hours per week)",
        groups=("Year 9", "Year 11"),
    ),
)

_RULE = "A value is an outlier if it is more than 1.5 times the interquartile range (IQR) above Q3 or below Q1."
HIGH = 80


def _plot(rng, *, iqr=None, outlier=None):
    """One five-number summary of even whole numbers; ``outlier`` is "high", "low" or None."""
    for _ in attempts(500, "a box plot"):
        iqr_ = iqr or rng.choice([8, 10, 12, 14, 16, 18, 20])
        q1 = rng.randrange(14, 34, 2)
        q3 = q1 + iqr_
        med = rng.randrange(q1 + 2, q3 - 1, 2)
        lo_fence = q1 - 3 * iqr_ // 2
        hi_fence = q3 + 3 * iqr_ // 2
        lo = rng.randrange(max(0, lo_fence + (lo_fence % 2)), q1 - 1, 2) if q1 - 2 >= max(0, lo_fence) else None
        hi = rng.randrange(q3 + 2, min(HIGH, hi_fence) + 1, 2) if q3 + 2 <= min(HIGH, hi_fence) else None
        if lo is None or hi is None:
            continue
        outliers = []
        if outlier == "high":
            start = hi_fence + 1 + (1 if (hi_fence + 1) % 2 else 0)
            if start > HIGH:
                continue
            outliers = [rng.randrange(start, HIGH + 1, 2)]
        elif outlier == "low":
            stop = lo_fence - 1 - (1 if (lo_fence - 1) % 2 else 0)
            if stop < 0:
                continue
            outliers = [rng.randrange(0, stop + 1, 2)]
        return dict(min=lo, q1=q1, median=med, q3=q3, max=hi, outliers=outliers)
    raise AssertionError  # pragma: no cover


def _figure(ctx, plots):
    return box_plot(
        low=0,
        high=HIGH,
        step=4,
        minor_ticks=True,
        plots=[dict(label=name, **p) for name, p in zip(ctx["groups"], plots)],
        x_label=ctx["x"],
        alt="two box plots on one axis",
    )


def _iqr(p):
    return p["q3"] - p["q1"]


def _upper(p):
    return p["q3"] + 3 * _iqr(p) // 2


def _lower(p):
    return p["q1"] - 3 * _iqr(p) // 2


def _pair(rng, *, outlier_on=None, side=None):
    """(ctx, [plot A, plot B]); ``outlier_on`` is 0 or 1 for the group that has an outlier on ``side``."""
    ctx = rng.choice(_CTX)
    plots = [
        _plot(rng, outlier=side if outlier_on == 0 else None),
        _plot(rng, outlier=side if outlier_on == 1 else None),
    ]
    return ctx, plots


# -- easy -------------------------------------------------------------------------------------------------------

_EASY_IQR = (
    "What is the interquartile range of {g}?",
    "Find the interquartile range for {g}.",
)
_EASY_MEDIAN = (
    "How much higher is the median of {a} than the median of {b}?",
    "By how many {unit} is the median of {a} greater than the median of {b}?",
)


def _unit(ctx):
    unit = ctx["x"].split("(")[1].rstrip(")").split(" per ")[0]
    return "marks" if unit == "%" else unit


def _easy(rng):
    for _ in attempts(200, "an easy box-plot question"):
        ctx, plots = _pair(rng)
        if rng.random() < 0.5:
            gi = rng.randrange(2)
            p, g = plots[gi], ctx["groups"][gi]
            key = _iqr(p)
            wrong = [
                (p["max"] - p["min"], "sr2a-bp-iqr-as-range"),
                (p["q3"], "sr2a-bp-iqr-as-quartile"),
                (p["median"] - p["q1"], "sr2a-bp-iqr-half-box"),
                (p["q3"] - p["median"], "sr2a-bp-iqr-half-box"),
                (p["max"] - p["q1"], "sr2a-bp-iqr-to-whisker"),
                (key + 2, "sr2a-bp-gridline-slip"),
            ]
            ask = phrase(rng, _EASY_IQR, g=g)
            working = [f"The box runs from Q1 = {p['q1']} to Q3 = {p['q3']}.", f"IQR = {p['q3']} - {p['q1']} = {key}."]
        else:
            hi = 0 if plots[0]["median"] > plots[1]["median"] else 1
            if plots[0]["median"] == plots[1]["median"]:
                continue
            a, b = plots[hi], plots[1 - hi]
            key = a["median"] - b["median"]
            wrong = [
                (a["q3"] - b["q3"], "sr2a-bp-median-diff-quartile"),
                (a["max"] - b["max"], "sr2a-bp-median-diff-whisker"),
                ((a["max"] - a["min"]) - (b["max"] - b["min"]), "sr2a-bp-median-diff-range"),
                (a["q1"] - b["q1"], "sr2a-bp-median-diff-quartile"),
                (key + 2, "sr2a-bp-gridline-slip"),
                (key - 2, "sr2a-bp-gridline-slip"),
            ]
            ask = phrase(rng, _EASY_MEDIAN, a=ctx["groups"][hi], b=ctx["groups"][1 - hi], unit=_unit(ctx))
            working = [
                f"The median of {ctx['groups'][hi]} is {a['median']} and the median of {ctx['groups'][1 - hi]} is {b['median']}.",
                f"{a['median']} - {b['median']} = {key}.",
            ]
        q = single(f"{ctx['lead']} {ask}", key, working, wrong, lo=1, figure=_figure(ctx, plots))
        if q:
            return q
    raise AssertionError  # pragma: no cover


# -- medium -----------------------------------------------------------------------------------------------------

_MEDIUM_UP = (
    "{rule} Find the upper fence for {g}: the value above which a result would be an outlier.",
    "{rule} What is the upper fence for {g}, the value that a result must be above to be an outlier?",
)
_MEDIUM_LOW = (
    "{rule} Find the lower fence for {g}: the value below which a result would be an outlier.",
    "{rule} What is the lower fence for {g}, the value that a result must be below to be an outlier?",
)


def _medium(rng):
    for _ in attempts(200, "a medium box-plot question"):
        ctx, plots = _pair(rng)
        gi = rng.randrange(2)
        p, g = plots[gi], ctx["groups"][gi]
        iqr = _iqr(p)
        if rng.random() < 0.5:
            key = _upper(p)
            wrong = [
                (p["q3"] + iqr, "sr2a-bp-fence-one-iqr"),
                (p["q3"] + 3 // 2 + 1, "sr2a-bp-fence-forgot-iqr"),
                (3 * iqr // 2, "sr2a-bp-fence-no-quartile"),
                (p["max"] + 3 * iqr // 2, "sr2a-bp-fence-from-whisker"),
                (p["median"] + 3 * iqr // 2, "sr2a-bp-fence-from-median"),
                (p["q1"] - 3 * iqr // 2, "sr2a-bp-fence-wrong-side"),
            ]
            ask = phrase(rng, _MEDIUM_UP, rule=_RULE, g=g)
            working = [
                f"IQR = {p['q3']} - {p['q1']} = {iqr}, so 1.5 x IQR = {3 * iqr // 2}.",
                f"Upper fence = Q3 + 1.5 x IQR = {p['q3']} + {3 * iqr // 2} = {key}.",
            ]
        else:
            key = _lower(p)
            if key < 0:
                continue
            wrong = [
                (p["q1"] - iqr, "sr2a-bp-fence-one-iqr"),
                (p["q1"] - 3 // 2 - 1, "sr2a-bp-fence-forgot-iqr"),
                (3 * iqr // 2, "sr2a-bp-fence-no-quartile"),
                (p["min"] - 3 * iqr // 2, "sr2a-bp-fence-from-whisker"),
                (p["median"] - 3 * iqr // 2, "sr2a-bp-fence-from-median"),
                (p["q3"] + 3 * iqr // 2, "sr2a-bp-fence-wrong-side"),
            ]
            ask = phrase(rng, _MEDIUM_LOW, rule=_RULE, g=g)
            working = [
                f"IQR = {p['q3']} - {p['q1']} = {iqr}, so 1.5 x IQR = {3 * iqr // 2}.",
                f"Lower fence = Q1 - 1.5 x IQR = {p['q1']} - {3 * iqr // 2} = {key}.",
            ]
        q = single(f"{ctx['lead']} {ask}", key, working, wrong, lo=-100, figure=_figure(ctx, plots))
        if q:
            return q
    raise AssertionError  # pragma: no cover


# -- hard -------------------------------------------------------------------------------------------------------

_HARD_BEYOND = (
    "{rule} The plot for {g} shows an outlier. By how much does the outlier lie beyond the fence it passes?",
    "{rule} {g} has one outlier on its plot. How far beyond the fence is the outlier?",
)
_HARD_SMALLEST = (
    "{rule} The plot for {g} shows no outlier. Every value is a whole number. What is the smallest whole number that "
    "would be an outlier above the plot for {g}?",
    "{rule} {g} has no outlier. Taking every value to be a whole number, what is the smallest whole-number value that "
    "would be an outlier for {g}, above its plot?",
)


def _hard(rng):
    for _ in attempts(300, "a hard box-plot question"):
        if rng.random() < 0.5:
            side = rng.choice(["high", "low"])
            gi = rng.randrange(2)
            ctx, plots = _pair(rng, outlier_on=gi, side=side)
            p, g = plots[gi], ctx["groups"][gi]
            o = p["outliers"][0]
            iqr = _iqr(p)
            if side == "high":
                key = o - _upper(p)
                wrong = [
                    (o - p["q3"], "sr2a-bp-beyond-from-quartile"),
                    (o - p["max"], "sr2a-bp-beyond-from-whisker"),
                    (o, "sr2a-bp-beyond-outlier-value"),
                    (o - p["q3"] - iqr, "sr2a-bp-beyond-one-iqr-fence"),
                    (_upper(p) - p["q3"], "sr2a-bp-fence-no-quartile"),
                    (key + 2, "sr2a-bp-gridline-slip"),
                ]
                working = [
                    f"IQR = {p['q3']} - {p['q1']} = {iqr}. Upper fence = {p['q3']} + {3 * iqr // 2} = {_upper(p)}.",
                    f"The outlier is at {o}, so it is {o} - {_upper(p)} = {key} beyond the fence.",
                ]
            else:
                key = _lower(p) - o
                wrong = [
                    (p["q1"] - o, "sr2a-bp-beyond-from-quartile"),
                    (p["min"] - o, "sr2a-bp-beyond-from-whisker"),
                    (o, "sr2a-bp-beyond-outlier-value"),
                    (p["q1"] - iqr - o, "sr2a-bp-beyond-one-iqr-fence"),
                    (p["q1"] - _lower(p), "sr2a-bp-fence-no-quartile"),
                    (key + 2, "sr2a-bp-gridline-slip"),
                ]
                working = [
                    f"IQR = {p['q3']} - {p['q1']} = {iqr}. Lower fence = {p['q1']} - {3 * iqr // 2} = {_lower(p)}.",
                    f"The outlier is at {o}, so it is {_lower(p)} - {o} = {key} beyond the fence.",
                ]
            ask = phrase(rng, _HARD_BEYOND, rule=_RULE, g=g)
        else:
            ctx, plots = _pair(rng)
            gi = rng.randrange(2)
            p, g = plots[gi], ctx["groups"][gi]
            fence = _upper(p)
            if fence + 1 > HIGH:
                continue
            key = fence + 1
            iqr = _iqr(p)
            wrong = [
                (fence, "sr2a-bp-outlier-at-the-fence"),
                (p["max"] + 1, "sr2a-bp-outlier-just-past-whisker"),
                (p["q3"] + iqr + 1, "sr2a-bp-fence-one-iqr"),
                (p["q3"] + 3 // 2 + 1, "sr2a-bp-fence-forgot-iqr"),
                (fence + 2, "sr2a-bp-gridline-slip"),
            ]
            ask = phrase(rng, _HARD_SMALLEST, rule=_RULE, g=g)
            working = [
                f"IQR = {p['q3']} - {p['q1']} = {iqr}. Upper fence = {p['q3']} + {3 * iqr // 2} = {fence}.",
                f"A value must be more than {fence}, so {fence} itself is not an outlier and the smallest whole number is {key}.",
            ]
        q = single(f"{ctx['lead']} {ask}", key, working, wrong, lo=1, figure=_figure(ctx, plots))
        if q:
            return q
    raise AssertionError  # pragma: no cover


@generator(
    "sr2a_box_plots",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def sr2a_box_plots(rng, tier="easy"):
    """Read two drawn box plots; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("sr2a_box_plots_chain", level=12, presets={"exam": dict(tier="exam")})
def sr2a_box_plots_chain(rng, tier="exam"):
    """Exam chain on two box plots: an IQR, a fence, the distance of an outlier beyond it, and which group varies more."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a box-plot chain with enough distractors"):
        side = rng.choice(["high", "high", "low"])
        gi = rng.randrange(2)
        ctx = rng.choice(_CTX)
        small = _plot(rng, iqr=rng.choice([8, 10, 12]), outlier=side)
        big = _plot(rng, iqr=rng.choice([16, 18, 20]))
        plots = [None, None]
        plots[gi], plots[1 - gi] = small, big
        o = small["outliers"][0]
        if not (max(small["max"], o) - min(small["min"], o) > big["max"] - big["min"]):
            continue
        g, h = ctx["groups"][gi], ctx["groups"][1 - gi]
        iqr = _iqr(small)
        up = side == "high"
        fence = _upper(small) if up else _lower(small)
        dist = o - fence if up else fence - o
        stem = f"{ctx['lead']} {_RULE} {g} has one outlier."
        part_a = number_part(
            f"What is the interquartile range of {g}?",
            iqr,
            1,
            [f"{small['q3']} - {small['q1']} = {iqr}."],
            [
                (small["max"] - small["min"], "sr2a-bp-iqr-as-range"),
                (small["q3"], "sr2a-bp-iqr-as-quartile"),
                (small["median"] - small["q1"], "sr2a-bp-iqr-half-box"),
                (iqr + 2, "sr2a-bp-gridline-slip"),
            ],
            lo=1,
        )
        if up:
            wrong_b = [
                (small["q3"] + iqr, "sr2a-bp-fence-one-iqr"),
                (3 * iqr // 2, "sr2a-bp-fence-no-quartile"),
                (small["max"] + 3 * iqr // 2, "sr2a-bp-fence-from-whisker"),
                (small["median"] + 3 * iqr // 2, "sr2a-bp-fence-from-median"),
                (small["q1"] - 3 * iqr // 2, "sr2a-bp-fence-wrong-side"),
            ]
            follow_b = f"{small['q3']} + 1.5*part_a"
            how = f"Upper fence = Q3 + 1.5 x IQR = {small['q3']} + {3 * iqr // 2} = {fence}."
            word = "upper"
        else:
            wrong_b = [
                (small["q1"] - iqr, "sr2a-bp-fence-one-iqr"),
                (3 * iqr // 2, "sr2a-bp-fence-no-quartile"),
                (small["min"] - 3 * iqr // 2, "sr2a-bp-fence-from-whisker"),
                (small["median"] - 3 * iqr // 2, "sr2a-bp-fence-from-median"),
                (small["q3"] + 3 * iqr // 2, "sr2a-bp-fence-wrong-side"),
            ]
            follow_b = f"{small['q1']} - 1.5*part_a"
            how = f"Lower fence = Q1 - 1.5 x IQR = {small['q1']} - {3 * iqr // 2} = {fence}."
            word = "lower"
        part_b = number_part(
            f"Use your answer to part (a) to find the {word} fence for {g}.",
            fence,
            2,
            [how],
            wrong_b,
            follow=follow_b,
            lo=-100,
            sensible=(None, None),
        )
        part_c = number_part(
            f"By how much does the outlier lie beyond the {word} fence? Use your answer to part (b).",
            dist,
            2,
            [
                f"The outlier is at {o}: {o} - {fence} = {dist}."
                if up
                else f"The outlier is at {o}: {fence} - {o} = {dist}."
            ],
            [
                (abs(o - (small["q3"] if up else small["q1"])), "sr2a-bp-beyond-from-quartile"),
                (abs(o - (small["max"] if up else small["min"])), "sr2a-bp-beyond-from-whisker"),
                (o, "sr2a-bp-beyond-outlier-value"),
                (dist + 2, "sr2a-bp-gridline-slip"),
            ],
            follow=f"{o} - part_b" if up else f"part_b - {o}",
            lo=1,
            sensible=(None, None),
        )
        key = f"{h} is more variable, because its interquartile range is larger."
        part_d = mcq_part(
            "Which statement about the variability of the two groups is correct?",
            key,
            1,
            [
                f"{h} has the larger IQR ({_iqr(big)} against {iqr}), so the middle half of its values is more spread out. {g} has the longer range only because of its outlier."
            ],
            [
                (f"{g} is more variable, because its range is larger.", "sr2a-bp-range-for-iqr"),
                (f"{g} is more variable, because it has an outlier.", "sr2a-bp-outlier-as-spread"),
                ("The two groups are equally variable, because the outlier is only one value.", "sr2a-bp-ignores-iqr"),
                (f"{h} is more variable, because its median is larger.", "sr2a-bp-median-for-spread"),
            ],
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        return dict(text=stem, parts=[part_a, part_b, part_c, part_d], figure=_figure(ctx, plots))
    raise AssertionError  # pragma: no cover


# -- the figure readers -----------------------------------------------------------------------------------------


def _plots(question):
    return {p["label"]: p for p in question.figure["plots"]}


def _group(question, plots):
    after = question.text.split("1.5 times the interquartile range (IQR) above Q3 or below Q1.", 1)[-1]
    found = [name for name in plots if name in after]
    return found[0] if found else None


def _read_box(question) -> str:
    plots = _plots(question)
    text = question.text
    if "interquartile range of" in text or "interquartile range for" in text:
        mark = re.search(r"interquartile range (?:of|for) ([\w ]+?)[?.]", text)
        return str(_iqr(plots[mark.group(1)]))
    if "median of" in text:
        m = re.search(r"median of ([\w ]+?) (?:than|greater than) the median of ([\w ]+?)\?", text)
        return str(plots[m.group(1)]["median"] - plots[m.group(2)]["median"])
    g = _group(question, plots)
    p = plots[g]
    after = text.split("below Q1.", 1)[-1]
    if "upper fence" in after:
        return str(_upper(p))
    if "lower fence" in after:
        return str(_lower(p))
    if p["outliers"]:
        o = p["outliers"][0]
        return str(o - _upper(p) if o > p["q3"] else _lower(p) - o)
    return str(_upper(p) + 1)


def _read_chain(question) -> str:
    plots = _plots(question)
    g = next(n for n, p in plots.items() if p["outliers"])
    h = next(n for n in plots if n != g)
    p = plots[g]
    o = p["outliers"][0]
    up = o > p["q3"]
    fence = _upper(p) if up else _lower(p)
    dist = o - fence if up else fence - o
    more = h if _iqr(plots[h]) > _iqr(p) else g
    return f"(a) {_iqr(p)}; (b) {fence}; (c) {dist}; (d) {more} is more variable, because its interquartile range is larger."


figures.register_reader("sr2a_box_plots", _read_box)
figures.register_reader("sr2a_box_plots_chain", _read_chain)
