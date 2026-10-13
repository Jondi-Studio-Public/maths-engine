"""Exam chain for reading a box plot (``f10-read-a-box-plot``): the interquartile range, the values between the
quartiles, the 1.5 x IQR upper fence and an outlier test.

The figure is the single ``box_plot`` the skill's practice tiers draw; no outlier is plotted, so every whisker end
is a data value. Set the way a school exam sets it:

(a) the interquartile range                                        number, 2 marks
(b) about how many of N values lie between the quartiles           number, 1 mark
(c) the upper fence, Q3 + 1.5 x IQR                                number, 2 marks, follow-through (a)
(d) whether a further value is an outlier                          mcq,    1 mark

## Construction

The five numbers are drawn first (by the same helper the practice tiers use, so a whisker end is never an
outlier), with an even IQR so that 1.5 x IQR is a whole number. The further value in (d) lies above the upper
whisker and is either beyond the fence or inside it, so the learner has to compare it with the fence they found:
the end of the whisker does not settle it. Every option of (d) is a false statement or a wrong verdict for that
value, checked from the five numbers. The reader registered for the topic re-derives (a) to (d) from the figure
and the prompts alone.

## Distractors, each one error of its own part

* (a): the range, the distance from the median to a quartile, a quartile on its own.
* (b): a quarter of the values, three quarters, all of them.
* (c): one IQR instead of 1.5, the 1.5 x IQR alone, the lower fence, the fence from the median.
* (d): the wrong verdict reached by one named error of the fence.
"""

from __future__ import annotations

import re

from mathgen import figures
from mathgen.core import attempts, check, generator, phrase
from mathgen.generators._exd_x_w12a_kit import mcq_part, number_part, split_parts
from mathgen.generators.stat5_box_plot_build import _CONTEXTS, _fig, _five_numbers

TOPIC = "exd_x_w12a_box_plot_read_chain"

_YES = "Yes, it is more than 1.5 × IQR above the upper quartile."
_NO = "No, it is less than 1.5 × IQR above the upper quartile."

_IQR_ASK = (
    "Find the interquartile range of the {nouns}.",
    "What is the interquartile range?",
)
_BETWEEN_ASK = (
    "There were {n} {subj}. About how many of them had a {noun} between the lower quartile and the upper quartile?",
    "Of the {n} {subj}, about how many had a {noun} between the quartiles?",
)
_FENCE_ASK = (
    "A value is an upper outlier if it is more than 1.5 × IQR above the upper quartile. Find the upper fence, the "
    "largest value that is not an upper outlier.",
    "Outliers lie more than 1.5 × IQR beyond a quartile. Find the upper fence, the largest value that is not an "
    "upper outlier.",
)
_VERDICT_ASK = (
    "A further {noun} of {w} is added to the data. Is it an outlier?",
    "One more {noun} is recorded, and it is {w}. Is it an outlier?",
)


def _draw(rng):
    ctx = rng.choice(_CONTEXTS)
    step = ctx["step"]
    minor = step % 2 == 0 and rng.random() < 0.5  # values may sit on the unlabelled half ticks
    unit = step // 2 if minor else step
    for _ in attempts(3000, "a box plot with an even IQR and room for a further value"):
        p = _five_numbers(rng, ctx, unit=unit)
        iqr = p["q3"] - p["q1"]
        if iqr % 2:
            continue
        fence = p["q3"] + iqr * 3 // 2
        windows = {
            True: (fence + 1, ctx["hi"]),
            False: (max(p["max"], p["q3"] + iqr) + 1, min(fence - 1, ctx["hi"])),
        }
        open_ones = [beyond for beyond, (low, high) in windows.items() if low <= high]
        if not open_ones:
            continue
        beyond = rng.choice(open_ones)
        lows, highs = windows[beyond]
        return ctx, p, minor, iqr, fence, beyond, rng.randint(lows, highs)


@generator(TOPIC, level=13, presets={"exam": {"tier": "exam"}})
def exd_x_w12a_box_plot_read_chain(rng, tier: str = "exam"):
    """The IQR, the values between the quartiles, the upper fence and an outlier verdict, from one box plot."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a box plot chain with three distractors per part"):
        ctx, p, minor, iqr, fence, beyond, w = _draw(rng)
        q1, med, q3, mx, mn = p["q1"], p["median"], p["q3"], p["max"], p["min"]
        n = rng.choice((40, 60, 80, 100, 120, 140, 160))
        part_a = number_part(
            phrase(rng, _IQR_ASK, nouns=ctx["nouns"]),
            iqr,
            2,
            [f"IQR = upper quartile - lower quartile = {q3} - {q1} = {iqr}."],
            [
                (mx - mn, "box-iqr-as-range"),
                (med - q1, "box-iqr-from-median"),
                (q3, "box-quartile-for-iqr"),
                (q3 - med, "box-iqr-from-median"),
            ],
        )
        part_b = number_part(
            phrase(rng, _BETWEEN_ASK, n=n, subj=ctx["subj"], noun=ctx["noun"]),
            n // 2,
            1,
            [
                f"The quartiles cut off a quarter of the values at each end, so half of the {n} values, {n // 2}, lie between them."
            ],
            [
                (n // 4, "box-quarter-between-quartiles"),
                (n * 3 // 4, "box-three-quarters-between-quartiles"),
                (n, "box-all-values-between-quartiles"),
            ],
        )
        part_c = number_part(
            phrase(rng, _FENCE_ASK),
            fence,
            2,
            [f"Upper fence = upper quartile + 1.5 × IQR = {q3} + 1.5 × {iqr} = {fence}."],
            [
                (q3 + iqr, "box-fence-without-factor"),
                (iqr * 3 // 2, "box-fence-omits-quartile"),
                (q1 - iqr * 3 // 2, "box-fence-wrong-side"),
                (med + iqr * 3 // 2, "box-fence-from-median"),
            ],
            follow=f"{q3}+1.5*part_a",
        )
        if beyond:
            key = _YES
            wrong = [
                (_NO, "box-outlier-fence-too-generous"),
                ("No, it is less than 1.5 × IQR above the median.", "box-fence-from-median"),
                ("No, it is less than 1 × IQR above the upper quartile.", "box-fence-without-factor"),
            ]
            note = f"{w} is above the upper fence of {fence}, so it is an outlier."
        else:
            key = _NO
            wrong = [
                (_YES, "box-outlier-fence-miscalculated"),
                ("Yes, it is above the end of the upper whisker.", "box-outlier-beyond-whisker"),
                ("Yes, it is more than 1 × IQR above the upper quartile.", "box-fence-without-factor"),
            ]
            note = f"{w} is below the upper fence of {fence}, so it is not an outlier, though it is above the whisker."
        part_d = mcq_part(phrase(rng, _VERDICT_ASK, noun=ctx["noun"], w=w), key, 1, [note], wrong)
        if None in (part_a, part_b, part_c, part_d):
            continue
        check((w > fence) == beyond and w > mx, "the further value is not placed as intended")
        fig = _fig(ctx, [dict(p, label="")], step=ctx["step"], minor=minor)
        return dict(text=f"{ctx['lead']}", parts=[part_a, part_b, part_c, part_d], figure=fig)


def _read(question) -> str:
    """Every answer re-derived from the figure and the prompts alone, in the labelled form a reply takes."""
    plot = question.figure["plots"][0]
    q1, q3 = plot["q1"], plot["q3"]
    iqr = q3 - q1
    text = question.text
    n = int(re.search(r"(?:There were|Of the) (\d+)", text).group(1))
    w = float(re.search(r"\d+(?:\.\d+)?", split_parts(text)["d"]).group())
    fence = q3 + 1.5 * iqr
    verdict = _YES if w > fence else _NO
    return f"(a) {iqr:g}; (b) {n // 2}; (c) {fence:g}; (d) {verdict}"


figures.register_reader(TOPIC, _read)
