"""Exam chain for comparing two box plots (``f10-compare-box-plots``): the two interquartile ranges, the difference
between them and a justified conclusion about the medians and the spreads.

The figure is the pair of parallel ``box_plot`` plots the skill's practice tiers draw. Set the way a school exam
sets it:

(a) the interquartile range of the first group                     number, 1 mark
(b) the interquartile range of the second group                    number, 1 mark
(c) the difference between the two interquartile ranges            number, 1 mark, follow-through (a), (b)
(d) the conclusion the medians and the spreads support             mcq,    2 marks

## Construction

The two five-number summaries are drawn so that each tempting shortcut gives the other answer. The group with the
higher median has the lower maximum, so comparing the ends of the whiskers picks the wrong centre; the group with
the larger IQR has the shorter range, so comparing the whole length of the plots picks the wrong spread. The four
conclusions of (d) are the four ways of giving each comparison to a group, one of them true, and each wrong one is
exactly the conclusion one named error produces. There is no figure reader for this topic: the sweep
checks an `mcq` skill's reader against option indexes, which a multi-part question does not have, so the tests
re-derive every answer from the figure instead.

## Distractors, each one error of its own part

* (a), (b): the range, the distance from the median to a quartile, a quartile on its own.
* (c): the sum of the two IQRs, the difference of the medians, of the ranges, or of the upper quartiles.
* (d): the centre judged by the whisker ends, the spread judged by the range, both comparisons reversed.
"""

from __future__ import annotations

from mathgen.core import attempts, check, generator, phrase
from mathgen.generators._exd_x_w12a_kit import mcq_part, number_part
from mathgen.generators.stat5_box_plot_build import _COMPARE_PAIRS, _CONTEXTS, _fig, _iqr, _rng

TOPIC = "exd_x_w12a_compare_box_plots_chain"

_LEAD = (
    "The box plots compare the {nouns} of two groups of {subj}.",
    "Two groups of {subj} are shown as box plots of their {nouns}.",
)
_IQR_ASK = (
    "Find the interquartile range of {name}.",
    "What is the interquartile range for {name}?",
)
_DIFF_ASK = (
    "Find the difference between the two interquartile ranges.",
    "By how much do the interquartile ranges of the two groups differ?",
)
_CONCLUDE_ASK = (
    "Which conclusion is best supported by the medians and the interquartile ranges?",
    "Which statement compares the two groups correctly?",
)


def conclusion(centre: str, spread: str, nouns: str) -> str:
    """The sentence that gives the higher median to ``centre`` and the larger IQR to ``spread``."""
    if centre == spread:
        return f"{centre} has both the higher median and the larger IQR, so its {nouns} are typically higher and more spread out."
    return (
        f"{centre} has the higher median, so its {nouns} are typically higher, and {spread} has the larger IQR, "
        f"so its {nouns} are more spread out."
    )


def _pair_ok(a, b) -> bool:
    """Different medians, IQRs, ranges and maxima, with the whisker ends and the range each pointing the other way."""
    if len({a["median"], b["median"]}) < 2 or len({_iqr(a), _iqr(b)}) < 2:
        return False
    if a["max"] == b["max"] or _rng(a) == _rng(b):
        return False
    return (a["median"] > b["median"]) != (a["max"] > b["max"]) and (_iqr(a) > _iqr(b)) != (_rng(a) > _rng(b))


def _gaps(rng, wide: bool) -> tuple[int, int, int, int]:
    """Steps (lower whisker, lower box half, upper box half, upper whisker) for a plot with a wide box and short
    whiskers, or a narrow box and long ones. Whiskers never pass 1.5 x IQR, so no end is an outlier."""
    if wide:
        return rng.randint(1, 2), rng.randint(2, 4), rng.randint(2, 4), rng.randint(1, 2)
    box = rng.randint(1, 3), rng.randint(1, 3)
    cap = (3 * (box[0] + box[1])) // 2
    return rng.randint(1, cap), box[0], box[1], rng.randint(1, cap)


def _plot(base: int, gaps, lo, unit) -> dict:
    marks, at = [], base
    for g in (0, *gaps):
        at += g
        marks.append(lo + at * unit)
    low, q1, median, q3, high = marks
    return dict(min=low, q1=q1, median=median, q3=q3, max=high, outliers=[])


def _draw(rng):
    """Two plots where the shortcuts disagree with the truth: the group with the higher median has the lower whisker
    end, and the group with the larger IQR has the shorter range. Half the draws give both to one group, half split
    them."""
    for _ in attempts(30, "an axis with room for both patterns"):
        ctx = rng.choice(_CONTEXTS)
        names = rng.choice(_COMPARE_PAIRS)
        minor = ctx["step"] % 2 == 0 and rng.random() < 0.5
        unit = ctx["step"] // 2 if minor else ctx["step"]
        top = (ctx["hi"] - ctx["lo"]) // unit
        # The pattern is fixed before the search, so the harder one is not crowded out. A short axis may have no room
        # for it; then the other pattern is tried.
        first = rng.random() < 0.5
        for wide_wins_centre in (first, not first):
            for _ in range(1500):
                wide, long_ = _gaps(rng, True), _gaps(rng, False)
                if max(sum(wide), sum(long_)) > top - 1:
                    continue
                a = _plot(rng.randint(1, top - sum(wide)), wide, ctx["lo"], unit)
                b = _plot(rng.randint(1, top - sum(long_)), long_, ctx["lo"], unit)
                # The group that wins the centre (higher median, lower whisker end) is the wide-box one, which also wins
                # the spread, or the long-whisker one, while the wide-box group wins the spread.
                if _pair_ok(a, b) and (a["median"] > b["median"]) == wide_wins_centre:
                    return ctx, names, minor, *((a, b) if rng.random() < 0.5 else (b, a))


@generator(TOPIC, level=13, presets={"exam": {"tier": "exam"}})
def exd_x_w12a_compare_box_plots_chain(rng, tier: str = "exam"):
    """Two IQRs, their difference and a justified conclusion, from two parallel box plots."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a comparison chain with three distractors per part"):
        ctx, names, minor, a, b = _draw(rng)
        plots = dict(zip(names, (a, b)))
        nouns = ctx["nouns"]
        parts = []
        for name, p in plots.items():
            iqr = _iqr(p)
            parts.append(
                number_part(
                    phrase(rng, _IQR_ASK, name=name),
                    iqr,
                    1,
                    [f"IQR = {p['q3']} - {p['q1']} = {iqr}."],
                    [
                        (_rng(p), "box-iqr-as-range"),
                        (p["median"] - p["q1"], "box-iqr-from-median"),
                        (p["q3"], "box-quartile-for-iqr"),
                        (p["q3"] - p["median"], "box-iqr-from-median"),
                    ],
                )
            )
        ia, ib = _iqr(a), _iqr(b)
        diff = abs(ia - ib)
        part_c = number_part(
            phrase(rng, _DIFF_ASK),
            diff,
            1,
            [f"The interquartile ranges are {ia} and {ib}, which differ by {diff}."],
            [
                (ia + ib, "box-iqr-difference-summed"),
                (abs(a["median"] - b["median"]), "box-iqr-difference-of-medians"),
                (abs(_rng(a) - _rng(b)), "box-iqr-difference-of-ranges"),
                (abs(a["q3"] - b["q3"]), "box-iqr-difference-of-upper-quartiles"),
            ],
            follow="abs(part_a-part_b)",
        )
        top, bottom = names
        higher = top if a["median"] > b["median"] else bottom
        lower = bottom if higher == top else top
        wider = top if ia > ib else bottom
        narrower = bottom if wider == top else top
        key = conclusion(higher, wider, nouns)
        part_d = mcq_part(
            phrase(rng, _CONCLUDE_ASK),
            key,
            2,
            [
                f"{top}: median {a['median']}, IQR {ia}. {bottom}: median {b['median']}, IQR {ib}.",
                f"{higher} has the higher median and {wider} has the larger IQR.",
            ],
            [
                (conclusion(lower, wider, nouns), "box-median-by-whisker-end"),
                (conclusion(higher, narrower, nouns), "box-spread-by-range"),
                (conclusion(lower, narrower, nouns), "box-conclusion-both-reversed"),
            ],
        )
        if None in (*parts, part_c, part_d):
            continue
        check(_pair_ok(a, b), "the shortcuts must disagree with the medians and the IQRs")
        fig = _fig(ctx, [dict(plots[n], label=n) for n in names], step=ctx["step"], minor=minor)
        stem = phrase(rng, _LEAD, nouns=nouns, subj=ctx["subj"])
        return dict(text=stem, parts=[*parts, part_c, part_d], figure=fig)
