"""Exam chain for reading a histogram (``f7-read-a-histogram``): frequencies, the total, the median class and a
percentage of the data.

The figure is the ``histogram`` kind the skill's practice tiers draw: equal-width classes that include their lower
boundary and exclude their upper one. The total is 20, 25, 40 or 50 so a percentage of it can be whole. Set the
way a school exam sets it:

(a) the frequency of one named class                              number, 1 mark
(b) the total number of observations                              number, 1 mark
(c) the lower boundary of the class that contains the median      number, 2 marks
(d) the percentage of the data in a stated range of classes       number, 2 marks, follow-through (b)

## Construction

The frequencies are drawn first, then the question is built from them, never the other way round. The two middle
positions of an even total must fall in the same class, so the median class is unambiguous; the stated range is
chosen so that its share of the total is a whole percentage. The answers are re-derived from the figure and the
prompts alone by the reader registered for the topic (``figures.register_reader``), which the regrade sweep runs.

## Distractors, each one error of its own part

* (a): the neighbouring bar, the bar read at the nearest labelled gridline, a class boundary.
* (b): the number of classes, the tallest bar, every bar but one, one bar counted twice.
* (c): the upper boundary of the median class, the modal class, the middle class on the axis, a frequency.
* (d): the count itself, the percentage outside the range, a class left out or one too many.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import figures
from mathgen.core import attempts, check, generator, phrase
from mathgen.figure_kinds.data import histogram, histogram_panel
from mathgen.generators._exd_x_w12a_kit import number_part, split_parts
from mathgen.generators.stat4_histogram_build import _CONTEXTS

TOPIC = "exd_x_w12a_histogram_chain"

_CLASS_ASK = (
    "How many {subj} {cls}?",
    "Use the histogram to find how many {subj} {cls}.",
)
_TOTAL_ASK = (
    "How many {subj} are represented in the histogram altogether?",
    "Find the total number of {subj} in the data.",
)
_MEDIAN_ASK = (
    "The median is the middle value when the data are in order. What is the lower boundary of the class that "
    "contains the median?",
    "Find the lower boundary of the class that contains the median.",
)
_PCT_ASK = (
    "What percentage of the {subj} {cls}?",
    "Work out the percentage of the {subj} that {cls}.",
)


def _fit(rng, n: int, total: int) -> list[int]:
    """``n`` frequencies between 1 and 12 adding to ``total``: one each, then the rest handed out at random."""
    freqs = [1] * n
    for _ in range(total - n):
        open_bars = [i for i in range(n) if freqs[i] < 12]
        freqs[rng.choice(open_bars)] += 1
    return freqs


def _class_of(freqs: list[int], position: int) -> int:
    """The index of the class holding the ``position``-th smallest value (counting from 1)."""
    run = 0
    for i, f in enumerate(freqs):
        run += f
        if position <= run:
            return i
    raise ValueError("position beyond the data")


def _median_class(freqs: list[int]) -> int | None:
    """The class holding the median, or None when the two middle values of an even total are in different classes."""
    n = sum(freqs)
    if n % 2:
        return _class_of(freqs, (n + 1) // 2)
    lo, hi = _class_of(freqs, n // 2), _class_of(freqs, n // 2 + 1)
    return lo if lo == hi else None


def _span(ctx, edges, i, j) -> str:
    return ctx["cls"].format(a=edges[i], b=edges[j], unit=ctx["unit"])


def _draw(rng):
    ctx = rng.choice(_CONTEXTS)
    width = ctx["width"]
    lo = rng.randint(*ctx["lo"])
    if width >= 5:
        lo -= lo % width
    n = rng.randint(6, 8)
    edges = [lo + i * width for i in range(n + 1)]
    for _ in attempts(3000, "a histogram with a clear median class and a whole percentage"):
        total = rng.choice((20, 25, 40, 50))
        freqs = _fit(rng, n, total)
        m = _median_class(freqs)
        if m is None:
            continue
        i = rng.randrange(0, n - 1)
        j = i + rng.choice((2, 3))
        if j > n or (100 * sum(freqs[i:j])) % total:
            continue
        if sum(freqs[i:j]) in (freqs[i], freqs[j - 1]):
            continue
        break
    return ctx, edges, freqs, total, m, i, j


@generator(TOPIC, level=7, presets={"exam": {"tier": "exam"}})
def exd_x_w12a_histogram_chain(rng, tier: str = "exam"):
    """Class frequency, total, median class and percentage of the data, from one drawn histogram."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a histogram chain with three distractors per part"):
        ctx, edges, freqs, total, m, i, j = _draw(rng)
        n = len(freqs)
        subj = ctx["subj"]
        k = rng.randrange(n)
        s = sum(freqs[i:j])
        pct = 100 * s // total
        top = max(freqs)
        modal = freqs.index(top) if freqs.count(top) == 1 else None

        neighbour = freqs[k + 1] if k + 1 < n else freqs[k - 1]
        other = freqs[k - 1] if k else freqs[k + 2]
        part_a = number_part(
            phrase(rng, _CLASS_ASK, subj=subj, cls=_span(ctx, edges, k, k + 1)),
            freqs[k],
            1,
            [f"The bar for the class from {edges[k]} to {edges[k + 1]} is {freqs[k]} high."],
            [
                (neighbour, "hist-neighbouring-bar"),
                (freqs[k] + 1 if freqs[k] % 2 else None, "hist-bar-to-nearest-gridline"),
                (edges[k], "hist-boundary-for-frequency"),
                (other, "hist-neighbouring-bar"),
                (freqs[k] - 1 if freqs[k] % 2 else None, "hist-bar-to-nearest-gridline"),
                (edges[k + 1], "hist-boundary-for-frequency"),
            ],
        )
        part_b = number_part(
            phrase(rng, _TOTAL_ASK, subj=subj),
            total,
            1,
            [f"Add the bars: {' + '.join(map(str, freqs))} = {total}."],
            [
                (n, "hist-classes-counted-for-total"),
                (top, "hist-tallest-bar-for-total"),
                (total - freqs[-1], "hist-total-misses-a-bar"),
                (total + top, "hist-total-counts-a-bar-twice"),
                (total - freqs[0], "hist-total-misses-a-bar"),
            ],
        )
        half = total // 2 if total % 2 == 0 else None
        part_c = number_part(
            phrase(rng, _MEDIAN_ASK),
            edges[m],
            2,
            [
                f"There are {total} values, so the median is the middle value"
                + (f" (between the {half}th and {half + 1}th)." if half else f" (the {(total + 1) // 2}th)."),
                f"Adding from the lowest class, the running total reaches that position in the class from {edges[m]} "
                f"to {edges[m + 1]}, so its lower boundary is {edges[m]}.",
            ],
            [
                (edges[m + 1], "hist-upper-boundary-for-lower"),
                (edges[modal] if modal is not None and modal != m else None, "hist-modal-class-for-median-class"),
                (edges[n // 2] if n // 2 != m else None, "hist-middle-class-for-median-class"),
                (freqs[m], "hist-frequency-for-boundary"),
                (edges[m - 1] if m else None, "hist-neighbouring-bar"),
            ],
        )
        part_d = number_part(
            phrase(rng, _PCT_ASK, subj=subj, cls=_span(ctx, edges, i, j)),
            pct,
            2,
            [
                f"The classes in the range hold {' + '.join(map(str, freqs[i:j]))} = {s} of the {total} values.",
                f"{s} out of {total} is {pct}%.",
            ],
            [
                (s, "hist-count-for-percentage"),
                (100 - pct, "hist-complement-percentage"),
                (100 * (s - freqs[i]) / total, "hist-range-leaves-out-class"),
                (100 * (s + freqs[j]) / total if j < n else None, "hist-range-adds-next-class"),
                (100 * (s - freqs[j - 1]) / total, "hist-range-leaves-out-class"),
            ],
            follow=f"100*{s}/part_b",
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        hi_axis = -(-max(freqs) // 2) * 2 + 2
        fig = histogram(
            panels=[histogram_panel(edges=edges, frequencies=freqs, freq_max=hi_axis, freq_step=2, minor_lines=True)],
            alt="a histogram",
            x_label=ctx["axis"],
            y_label="Frequency",
        )
        # The key is checked a second way, from the sorted raw values the bars stand for.
        values = [edges[c] for c, f in enumerate(freqs) for _ in range(f)]
        mid = (values[(total - 1) // 2] + values[total // 2]) / 2
        check(edges[m] <= mid < edges[m + 1], "the median class disagrees with the raw values")
        check(Fraction(s * 100, total) == pct, "the percentage is not whole")
        return dict(
            text=f"{ctx['lead']} The results are shown in the histogram.",
            parts=[part_a, part_b, part_c, part_d],
            figure=fig,
        )


def _read(question) -> str:
    """Every answer re-derived from the figure and the prompts alone, in the labelled form a reply takes."""
    panel = question.figure["panels"][0]
    edges, freqs = panel["edges"], panel["frequencies"]
    prompts = split_parts(question.text)

    def span(text):
        a, b = re.search(r"from (\d+(?:\.\d+)?) to less than (\d+(?:\.\d+)?)", text).groups()
        return edges.index(float(a)), edges.index(float(b))

    ka, _ = span(prompts["a"])
    ci, cj = span(prompts["d"])
    n = sum(freqs)
    values = [edges[c] for c, f in enumerate(freqs) for _ in range(f)]
    mid = (values[(n - 1) // 2] + values[n // 2]) / 2
    median_class = max(c for c in range(len(freqs)) if edges[c] <= mid)
    fmt = lambda v: str(int(v))  # noqa: E731
    return f"(a) {fmt(freqs[ka])}; (b) {fmt(n)}; (c) {fmt(edges[median_class])}; (d) {fmt(100 * sum(freqs[ci:cj]) / n)}"


figures.register_reader(TOPIC, _read)
