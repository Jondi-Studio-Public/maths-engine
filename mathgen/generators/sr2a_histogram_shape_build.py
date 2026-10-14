"""Generates `histogram-shape-and-spread` questions (easy, medium, hard) and the exam chain `sr2a_histogram_chain`.

Reading the modal class, the spread and the middle of a drawn histogram (ACMGM029 and the display half of ACMEM060),
`statistics-probability`, answer ``number``. The figure is a single-panel ``histogram`` of 6 to 9 equal classes; the
prompt never lists the frequencies.

* easy: the lower boundary of the modal class, or the frequency of the modal class;
* medium: the greatest possible range (upper edge of the last non-empty class minus lower edge of the first), or how
  many values lie below a class boundary. One or two empty classes are sometimes added at an end of the axis, so the
  range is not simply the axis length;
* hard: the lower boundary of the class that holds the median, or an estimate of how many values lie below a value in
  the middle of a class (the values of a class are taken to be evenly spread, and the class has an even frequency so
  the estimate is a whole number);
* exam (``sr2a_histogram_chain``): an odd total of values. (a) the total, (b) the position of the median value (follow-
  through from (a)), (c) the lower boundary of the class that holds it, (d) whether the mean is greater than, less
  than or about equal to the median, decided by the shape.

For (d) the drawn shape is accepted only when the mean estimated from class midpoints and the median interpolated in
its class differ clearly in the direction of the skew, or sit within a twentieth of a class width of each other for a
symmetric shape. The registered readers recompute every answer from the plotted frequencies and the prompt.
"""

from __future__ import annotations

import re

from .. import figures
from ..core import attempts, check, generator, phrase
from ..figure_kinds.data import histogram, histogram_panel
from ._sr2a_kit import mcq_part, number_part, single

# ---------------------------------------------------------------- shapes and contexts

_CTX = {
    "right": (
        dict(
            lead="The histogram shows how long customers waited at a service desk.",
            x="Waiting time (minutes)",
            starts=(0,),
            widths=(2, 5),
            thing="customers",
        ),
        dict(
            lead="The histogram shows how many days patients stayed in a hospital ward.",
            x="Length of stay (days)",
            starts=(0,),
            widths=(2,),
            thing="patients",
        ),
        dict(
            lead="The histogram shows the water used by households in one day.",
            x="Water used (kilolitres)",
            starts=(0,),
            widths=(0.5,),
            thing="households",
        ),
    ),
    "left": (
        dict(
            lead="The histogram shows the marks of students on an easy test.",
            x="Mark (%)",
            starts=(30, 40),
            widths=(10,),
            thing="students",
        ),
        dict(
            lead="The histogram shows the ages at which people retired.",
            x="Age at retirement (years)",
            starts=(50,),
            widths=(2,),
            thing="people",
        ),
    ),
    "symmetric": (
        dict(
            lead="The histogram shows the heights of adult women.",
            x="Height (cm)",
            starts=(140, 145),
            widths=(5,),
            thing="women",
        ),
        dict(
            lead="The histogram shows the masses of bags of flour from a packing line.",
            x="Mass (grams)",
            starts=(480, 490),
            widths=(5,),
            thing="bags",
        ),
    ),
}


def _num(v: float) -> str:
    return str(int(v)) if float(v) == int(v) else f"{v:g}"


def _classes(rng, shape, k):
    """k class frequencies: one strictly tallest class and the tail of the shape, every class non-empty."""
    peak = rng.randint(9, 15)
    if shape == "symmetric":
        mid = (k - 1) / 2
        spread = rng.uniform(1.4, 2.0)
        f = [max(1, round(peak * 2.718 ** (-(((i - mid) / spread) ** 2) / 2))) for i in range(k)]
        for i in range(k // 2):  # keep it a near mirror
            f[k - 1 - i] = f[i] + rng.choice([0, 0, 1, -1]) if f[i] > 1 else f[k - 1 - i]
        f = [max(1, v) for v in f]
        if k % 2 == 0:
            f[k // 2] += rng.choice([1, 2])
        return f
    p = rng.choice([1, 2])
    ratio = rng.uniform(0.62, 0.8)
    rise = rng.uniform(0.45, 0.65)
    f = []
    for i in range(k):
        v = peak * (rise ** (p - i) if i <= p else ratio ** (i - p))
        f.append(max(1, round(v + rng.choice([0, 0, 1, -1]))))
    f[p] = max(f)
    if f.count(max(f)) > 1:
        f[p] += 1
    return f[::-1] if shape == "left" else f


def _edge(v, edges) -> str:
    """A boundary or a difference of boundaries written to the one decimal place the axis uses (none for whole edges)."""
    return f"{v:.1f}" if any(float(e) != int(e) for e in edges) else str(int(v))


def _facts(edges, f):
    n = sum(f)
    w = edges[1] - edges[0]
    mid = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
    mean = sum(m * v for m, v in zip(mid, f)) / n
    cum, med_class, below = 0, None, 0
    half = n / 2
    for i, v in enumerate(f):
        if cum + v >= half and med_class is None:
            med_class, below = i, cum
        cum += v
    median = edges[med_class] + (half - below) / f[med_class] * w
    return n, w, mean, median, med_class


def _same_class_median(f):
    n = sum(f)
    lo, hi = (n + 1) // 2, n // 2 + 1
    cum, found = 0, []
    for i, v in enumerate(f):
        for pos in (lo, hi):
            if cum < pos <= cum + v:
                found.append(i)
        cum += v
    return found[0] if found and len(set(found)) == 1 else None


def _draw(rng, shape=None, *, odd=None, pad=True):
    """(context, edges, frequencies, shape) with a unique modal class; optionally an odd total."""
    for _ in attempts(3000, "a histogram"):
        shape_ = shape or rng.choice(["right", "left", "symmetric"])
        ctx = rng.choice(_CTX[shape_])
        k = rng.randint(6, 8)
        f = _classes(rng, shape_, k)
        n = sum(f)
        if not 36 <= n <= 90 or max(f) > 16 or f.count(max(f)) != 1:
            continue
        if odd is True and n % 2 == 0:
            continue
        zeros = [0, 0]
        if pad and rng.random() < 0.5:
            side = rng.choice(["low", "high"])
            z = rng.choice([1, 1, 2]) if k <= 7 else 1
            zeros = [z, 0] if side == "low" else [0, z]
        full = [0] * zeros[0] + f + [0] * zeros[1]
        if len(full) + 1 > 13:
            continue
        w = rng.choice(ctx["widths"])
        start = rng.choice(ctx["starts"])
        edges = [start + w * i for i in range(len(full) + 1)]
        if shape_ == "left" and edges[-1] > 100 + 1e-9 and "Mark" in ctx["x"]:
            continue
        _n, _w, mean, median, _mc = _facts(edges, full)
        gap = (mean - median) / w
        if (
            shape_ == "right"
            and gap < 0.2
            or shape_ == "left"
            and gap > -0.2
            or shape_ == "symmetric"
            and abs(gap) > 0.06
        ):
            continue
        return ctx, edges, full, shape_
    raise AssertionError  # pragma: no cover


def _figure(ctx, edges, f):
    top = max(f)
    fmax = top + (top % 2)
    return histogram(
        panels=[histogram_panel(edges=edges, frequencies=f, freq_max=max(fmax, 4), freq_step=2, minor_lines=True)],
        x_label=ctx["x"],
        y_label="Frequency",
        alt="a histogram of equal-width classes",
    )


def _lead(ctx, f):
    return ctx["lead"]


# -- easy -------------------------------------------------------------------------------------------------------

_EASY_EDGE = (
    "What is the lower boundary of the modal class?",
    "Find the lower boundary of the class with the greatest frequency (the modal class).",
)
_EASY_FREQ = (
    "What is the frequency of the modal class?",
    "How many values are in the modal class, the class with the greatest frequency?",
)


def _easy(rng):
    for _ in attempts(100, "an easy histogram question"):
        ctx, edges, f, _s = _draw(rng)
        i = f.index(max(f))
        w = edges[1] - edges[0]
        lead = _lead(ctx, f)
        dp = 1 if any(float(v) != int(v) for v in edges) else 0
        is_edge = rng.random() < 0.5
        if is_edge:
            key = edges[i]
            wrong = [
                (edges[i + 1], "sr2a-hs-modal-upper-edge"),
                (f[i], "sr2a-hs-modal-frequency-for-boundary"),
                (edges[i - 1] if i > 0 else None, "sr2a-hs-modal-neighbour-class"),
                (edges[i + 2] if i + 2 < len(edges) else None, "sr2a-hs-modal-neighbour-class"),
                ((edges[i] + edges[i + 1]) / 2, "sr2a-hs-modal-midpoint"),
            ]
            ask = phrase(rng, _EASY_EDGE)
            working = [
                f"The tallest bar is the class from {_num(edges[i])} to {_num(edges[i + 1])}.",
                f"Its lower boundary is {_num(key)}.",
            ]
        else:
            key = f[i]
            wrong = [
                (edges[i], "sr2a-hs-modal-boundary-for-frequency"),
                (f[i - 1] if i > 0 else None, "sr2a-hs-modal-neighbour-class"),
                (f[i + 1] if i + 1 < len(f) else None, "sr2a-hs-modal-neighbour-class"),
                (key + 1, "sr2a-hs-gridline-slip"),
                (key - 1, "sr2a-hs-gridline-slip"),
                (i + 1, "sr2a-hs-class-number-for-frequency"),
            ]
            ask = phrase(rng, _EASY_FREQ)
            working = [
                f"The tallest bar is the class from {_num(edges[i])} to {_num(edges[i + 1])}.",
                f"Its height is {key}.",
            ]
        q = single(f"{lead} {ask}", key, working, wrong, dp=dp if is_edge else 0, lo=0, figure=_figure(ctx, edges, f))
        if q:
            return q
    raise AssertionError  # pragma: no cover


# -- medium -----------------------------------------------------------------------------------------------------

_MEDIUM_RANGE = (
    "What is the greatest possible range of the data? Give the difference between the highest and the lowest value "
    "the histogram allows.",
    "Use the histogram to find the greatest possible range of the data, the largest difference between two values it "
    "allows.",
)
_MEDIUM_BELOW = (
    "How many values are less than {x}?",
    "How many of the values are below {x}?",
)


def _medium(rng):
    for _ in attempts(200, "a medium histogram question"):
        ctx, edges, f, _s = _draw(rng)
        w = edges[1] - edges[0]
        lead = _lead(ctx, f)
        nz = [i for i, v in enumerate(f) if v]
        lo_i, hi_i = nz[0], nz[-1]
        dp = 1 if any(float(v) != int(v) for v in edges) else 0
        if rng.random() < 0.5:
            key = edges[hi_i + 1] - edges[lo_i]
            wrong = [
                (edges[-1] - edges[0], "sr2a-hs-range-whole-axis"),
                (edges[hi_i] - edges[lo_i], "sr2a-hs-range-lower-edge-of-last"),
                (edges[hi_i + 1] - edges[lo_i + 1], "sr2a-hs-range-lowest-upper-edge"),
                (len(f), "sr2a-hs-range-number-of-classes"),
                ((hi_i - lo_i + 1) * 1.0, "sr2a-hs-range-number-of-classes"),
            ]
            ask = phrase(rng, _MEDIUM_RANGE)
            working = [
                f"The first non-empty class starts at {_num(edges[lo_i])} and the last ends at {_num(edges[hi_i + 1])}.",
                f"{_num(edges[hi_i + 1])} - {_num(edges[lo_i])} = {_num(key)}.",
            ]
            q = single(f"{lead} {ask}", key, working, wrong, dp=dp, lo=0, figure=_figure(ctx, edges, f))
        else:
            j = rng.randint(2, len(f) - 2)
            x = edges[j]
            key = sum(f[:j])
            wrong = [
                (sum(f[: j + 1]), "sr2a-hs-below-includes-next-class"),
                (sum(f[: j - 1]), "sr2a-hs-below-omits-class"),
                (f[j - 1], "sr2a-hs-below-one-class-only"),
                (sum(f[j:]), "sr2a-hs-below-for-above"),
                (j, "sr2a-hs-class-number-for-frequency"),
            ]
            ask = phrase(rng, _MEDIUM_BELOW, x=_num(x))
            working = [
                f"Add the frequencies of the classes below {_num(x)}.",
                " + ".join(str(v) for v in f[:j]) + f" = {key}.",
            ]
            q = single(f"{lead} {ask}", key, working, wrong, dp=0, lo=0, figure=_figure(ctx, edges, f))
        if q:
            return q
    raise AssertionError  # pragma: no cover


# -- hard -------------------------------------------------------------------------------------------------------

_MEDIAN_CLASS = (
    "What is the lower boundary of the class that contains the median?",
    "Find the lower boundary of the class in which the median lies.",
)
_INTERP = (
    "Assume the values in each class are spread evenly across it. Estimate how many values are less than {x}.",
    "Taking the values in each class to be evenly spread across it, estimate the number of values below {x}.",
)


def _hard(rng):
    for _ in attempts(300, "a hard histogram question"):
        ctx, edges, f, _s = _draw(rng)
        lead = _lead(ctx, f)
        w = edges[1] - edges[0]
        dp = 1 if any(float(v) != int(v) for v in edges) else 0
        if rng.random() < 0.5:
            mc = _same_class_median(f)
            if mc is None:
                continue
            n = sum(f)
            modal = f.index(max(f))
            centre = len(f) // 2
            if mc in (modal, centre):
                continue
            key = edges[mc]
            wrong = [
                (edges[modal], "sr2a-hs-median-class-modal"),
                (edges[centre], "sr2a-hs-median-class-middle-of-axis"),
                (edges[mc + 1], "sr2a-hs-modal-upper-edge"),
                (edges[mc - 1] if mc else None, "sr2a-hs-modal-neighbour-class"),
                (edges[mc + 2] if mc + 2 < len(edges) else None, "sr2a-hs-modal-neighbour-class"),
            ]
            cum = sum(f[:mc])
            working = [
                f"There are {n} values, so the median is the value in position {(n + 1) // 2 if n % 2 else str(n // 2) + ' and ' + str(n // 2 + 1)}.",
                f"{cum} values lie below the class from {_num(edges[mc])}, and {cum + f[mc]} lie below the next one, so the median is in the class starting at {_num(key)}.",
            ]
            q = single(
                f"{lead} {phrase(rng, _MEDIAN_CLASS)}", key, working, wrong, dp=dp, lo=0, figure=_figure(ctx, edges, f)
            )
        else:
            js = [j for j in range(1, len(f) - 1) if f[j] % 2 == 0 and f[j] >= 4]
            if not js:
                continue
            j = rng.choice(js)
            x = (edges[j] + edges[j + 1]) / 2
            if dp == 0 and x != int(x):
                continue
            key = sum(f[:j]) + f[j] // 2
            wrong = [
                (sum(f[: j + 1]), "sr2a-hs-interp-whole-class"),
                (sum(f[:j]), "sr2a-hs-interp-omits-class"),
                (sum(f[:j]) + f[j], "sr2a-hs-interp-whole-class"),
                (f[j] // 2, "sr2a-hs-interp-half-only"),
                (key + 1, "sr2a-hs-gridline-slip"),
            ]
            working = [
                f"{_num(x)} is halfway across the class from {_num(edges[j])} to {_num(edges[j + 1])}, which has {f[j]} values.",
                f"{sum(f[:j])} + half of {f[j]} = {key}.",
            ]
            q = single(
                f"{lead} {phrase(rng, _INTERP, x=_num(x))}",
                key,
                working,
                wrong,
                dp=0,
                lo=0,
                figure=_figure(ctx, edges, f),
            )
        if q:
            return q
    raise AssertionError  # pragma: no cover


@generator(
    "sr2a_histogram_shape",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def sr2a_histogram_shape(rng, tier="easy"):
    """Read a drawn histogram; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------

_GREATER = "The mean is greater than the median."
_LESS = "The mean is less than the median."
_ABOUT = "The mean and the median are about equal."
_CANNOT = "They cannot be compared without the original data values."
_KEY = {"right": _GREATER, "left": _LESS, "symmetric": _ABOUT}


@generator("sr2a_histogram_chain", level=12, presets={"exam": dict(tier="exam")})
def sr2a_histogram_chain(rng, tier="exam"):
    """Exam chain on one histogram: the total, the position of the median, its class, and mean against median."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a histogram chain with enough distractors"):
        ctx, edges, f, shape = _draw(rng, odd=True)
        mc = _same_class_median(f)
        if mc is None:
            continue
        n = sum(f)
        pos = (n + 1) // 2
        w = edges[1] - edges[0]
        dp = 1 if any(float(v) != int(v) for v in edges) else 0
        modal = f.index(max(f))
        stem = f"{_lead(ctx, f)}"
        part_a = number_part(
            "How many values are shown in the histogram altogether?",
            n,
            1,
            ["Add the frequency of every class: " + " + ".join(str(v) for v in f if v) + f" = {n}."],
            [
                (len([v for v in f if v]), "sr2a-hs-total-counts-bars"),
                (max(f), "sr2a-hs-total-tallest"),
                (n - min(v for v in f if v), "sr2a-hs-total-omits-class"),
                (n + 1, "sr2a-hs-gridline-slip"),
                (n - 1, "sr2a-hs-gridline-slip"),
            ],
            lo=1,
        )
        part_b = number_part(
            "The values are listed in order. In which position in the list is the median?",
            pos,
            1,
            [f"With {n} values the median is the middle one: ({n} + 1) / 2 = {pos}."],
            [
                (n // 2, "sr2a-hs-position-half-n"),
                (n // 2 + 2, "sr2a-hs-position-half-n"),
                (n, "sr2a-hs-position-n"),
                (pos + 1, "sr2a-hs-position-half-n"),
            ],
            follow="(part_a + 1)/2",
            lo=1,
        )
        part_c = number_part(
            "What is the lower boundary of the class that contains the median?",
            edges[mc],
            2,
            [
                f"{sum(f[:mc])} values lie below the class starting at {_num(edges[mc])}, and the class holds {f[mc]}, which reaches position {pos}."
            ],
            [
                (edges[modal], "sr2a-hs-median-class-modal"),
                (edges[len(f) // 2], "sr2a-hs-median-class-middle-of-axis"),
                (edges[mc + 1], "sr2a-hs-modal-upper-edge"),
                (edges[mc - 1] if mc else None, "sr2a-hs-modal-neighbour-class"),
                (edges[mc + 2] if mc + 2 < len(edges) else None, "sr2a-hs-modal-neighbour-class"),
            ],
            dp=dp,
            lo=0,
        )
        key = _KEY[shape]
        tail = {"right": "higher", "left": "lower", "symmetric": "neither"}[shape]
        wrong = {
            "right": [
                (_LESS, "sr2a-hs-mean-median-reversed"),
                (_ABOUT, "sr2a-hs-skewed-mean-equals-median"),
                (_CANNOT, "sr2a-hs-cannot-compare"),
            ],
            "left": [
                (_GREATER, "sr2a-hs-mean-median-reversed"),
                (_ABOUT, "sr2a-hs-skewed-mean-equals-median"),
                (_CANNOT, "sr2a-hs-cannot-compare"),
            ],
            "symmetric": [
                (_GREATER, "sr2a-hs-symmetric-as-skewed"),
                (_LESS, "sr2a-hs-symmetric-as-skewed"),
                (_CANNOT, "sr2a-hs-cannot-compare"),
            ],
        }[shape]
        part_d = mcq_part(
            "Which statement about the mean and the median of these values is correct?",
            key,
            2,
            [
                {
                    "right": "The tail stretches to the higher values and pulls the mean above the median.",
                    "left": "The tail stretches to the lower values and pulls the mean below the median.",
                    "symmetric": "The histogram is close to symmetric, so the mean and the median sit close together.",
                }[shape]
            ],
            wrong,
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        return dict(text=stem, parts=[part_a, part_b, part_c, part_d], figure=_figure(ctx, edges, f))
    raise AssertionError  # pragma: no cover


# -- the figure readers -----------------------------------------------------------------------------------------


def _panel(question):
    p = question.figure["panels"][0]
    return p["edges"], p["frequencies"]


def _mean_vs_median(edges, f) -> str:
    _n, w, mean, median, _mc = _facts(edges, f)
    gap = (mean - median) / w
    return _GREATER if gap > 0.12 else _LESS if gap < -0.12 else _ABOUT


def _read_hist(question) -> str:
    edges, f = _panel(question)
    ask = question.text.split(".", 1)[1] if "." in question.text else question.text
    nz = [i for i, v in enumerate(f) if v]
    if "frequency of the modal" in ask or "How many values are in the modal" in ask:
        return _num(max(f))
    if "modal class" in ask:
        return _edge(edges[f.index(max(f))], edges)
    if "range" in ask:
        return _edge(edges[nz[-1] + 1] - edges[nz[0]], edges)
    if "evenly" in ask:
        x = float(re.findall(r"(?:than|below) (\d+(?:\.\d+)?)\.", ask)[-1])
        w = edges[1] - edges[0]
        total = sum(v * min(max((x - lo) / w, 0), 1) for v, lo in zip(f, edges))
        return _num(round(total))
    if "less than" in ask or "below" in ask:
        x = float(re.findall(r"(\d+(?:\.\d+)?)\?", ask)[-1])
        return _num(sum(v for v, lo in zip(f, edges) if lo < x))
    return _edge(edges[_same_class_median(f)], edges)


def _read_chain(question) -> str:
    edges, f = _panel(question)
    n = sum(f)
    mc = _same_class_median(f)
    return f"(a) {n}; (b) {(n + 1) // 2}; (c) {_edge(edges[mc], edges)}; (d) {_mean_vs_median(edges, f)}"


figures.register_reader("sr2a_histogram_shape", _read_hist)
figures.register_reader("sr2a_histogram_chain", _read_chain)
