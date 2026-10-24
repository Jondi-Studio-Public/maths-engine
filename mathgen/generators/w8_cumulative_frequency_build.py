"""Generates the `w8_*` cumulative frequency and frequency polygon questions: reading a drawn graph.

Two skills, both `answer: number`, both drawn as a ``cartesian`` figure (``equal_scale=False``) with the plotted points
marked and joined by straight segments:

``cumulative-frequency-ogive`` (topic ``w8_ogive_read``, exam ``exd_x_w8_ogive_chain``). The graph is an ogive: the
cumulative frequency plotted against the upper boundary of each class, starting at 0 at the lower boundary of the first
class. Every plotted point sits on a gridline of both axes, and the quartile positions (a quarter, a half and three
quarters of the total) are plotted points, so a median or a quartile is read exactly and never interpolated by eye.

* easy: the number of values below a stated boundary (the height of a plotted point), or the total (the top of the graph).
* medium: the median, the number of values between two stated boundaries (a difference of two heights), the number at or
  above a boundary (the total less a height) or the percentage below a boundary.
* hard: the interquartile range (two reads and a subtraction), the lower boundary of the modal class (the steepest
  segment, found by the learner), the percentage between two boundaries, or how many more values one class holds than
  another (a difference of two differences).

``frequency-polygon-read`` (topic ``w8_polygon_read``, exam ``exd_x_w8_polygon_chain``). The polygon plots each class
frequency above the class midpoint, joined in order and closed to zero at an empty class either side. The axis is
labelled with midpoints, so the learner has to move between midpoints and class boundaries.

* easy: the frequency of the class with a stated midpoint, or the midpoint of the modal class.
* medium: the total, how many more values the modal class holds than a named class, or the lower boundary of the modal
  class (a midpoint turned into a boundary).
* hard: how many values are at or above (or below) a boundary, the percentage of the data there, or an estimate of the
  mean from the class midpoints.

The answer is built first (the cumulative or class counts are drawn, the question follows), re-derived a second way (from
the raw values the classes stand for) and re-read from the figure alone by a registered figure reader.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import figures
from mathgen.answers import Distractor
from mathgen.core import GRADER_TOLERANCE, attempts, check, generator, latex_answer, phrase, pick_distractors
from mathgen.generators._exd_x_w12a_kit import split_parts
from mathgen.generators._w6a_kit import TooFewDistractors, part, retry_draw
from mathgen.generators.stat4_histogram_build import _CONTEXTS

OGIVE = "w8_ogive_read"
OGIVE_EXAM = "exd_x_w8_ogive_chain"
POLYGON = "w8_polygon_read"
POLYGON_EXAM = "exd_x_w8_polygon_chain"
TIERS = ("easy", "medium", "hard")

_OGIVE_AREA = "cumulative-frequency-ogive"
_POLYGON_AREA = "frequency-polygon-read"
MISCONCEPTIONS = [
    (
        "ogive-boundary-for-count",
        "Gives the boundary from the horizontal axis where a number of values was asked.",
        _OGIVE_AREA,
    ),
    (
        "ogive-count-for-value",
        "Gives a cumulative frequency (a count) where a value on the horizontal axis was asked.",
        _OGIVE_AREA,
    ),
    (
        "ogive-total-for-count",
        "Gives the total number of values, the top of the graph, instead of the number below the boundary.",
        _OGIVE_AREA,
    ),
    ("ogive-neighbouring-point", "Reads the plotted point next to the one the question names.", _OGIVE_AREA),
    (
        "ogive-complement-count",
        "Gives the number below the boundary where the number at or above it was asked, or the reverse.",
        _OGIVE_AREA,
    ),
    (
        "ogive-adds-heights",
        "Adds the heights of the two plotted points instead of subtracting the lower from the upper.",
        _OGIVE_AREA,
    ),
    (
        "ogive-upper-point-only",
        "Gives the height of the upper plotted point and forgets to take away the values already below the lower boundary.",
        _OGIVE_AREA,
    ),
    (
        "ogive-lower-point-only",
        "Gives the height of the lower plotted point instead of the number between the two boundaries.",
        _OGIVE_AREA,
    ),
    (
        "ogive-width-for-count",
        "Gives the difference between the two boundaries (the class width) instead of the number of values between them.",
        _OGIVE_AREA,
    ),
    (
        "ogive-adds-all-heights",
        "Adds the heights of every plotted point to get the total instead of reading the top of the graph.",
        _OGIVE_AREA,
    ),
    ("ogive-classes-for-total", "Gives the number of classes where the total number of values was asked.", _OGIVE_AREA),
    (
        "ogive-middle-of-axis",
        "Takes the middle of the horizontal axis as the median instead of reading where the graph reaches half the total.",
        _OGIVE_AREA,
    ),
    ("ogive-quartile-sum", "Adds the lower and upper quartiles instead of subtracting.", _OGIVE_AREA),
    ("ogive-range-for-iqr", "Gives the range of the whole graph where the interquartile range was asked.", _OGIVE_AREA),
    (
        "ogive-half-spread",
        "Gives the distance from a quartile to the median where the distance between the two quartiles was asked.",
        _OGIVE_AREA,
    ),
    (
        "ogive-upper-quartile-only",
        "Gives the upper quartile alone where the interquartile range was asked.",
        _OGIVE_AREA,
    ),
    (
        "ogive-count-between-quartiles",
        "Gives the number of values between the quartiles (half of them) where the interquartile range was asked.",
        _OGIVE_AREA,
    ),
    (
        "ogive-modal-upper-boundary",
        "Gives the upper boundary of the modal class instead of its lower boundary.",
        _OGIVE_AREA,
    ),
    (
        "ogive-modal-is-last-class",
        "Takes the last class, where the cumulative frequency is highest, as the modal class.",
        _OGIVE_AREA,
    ),
    (
        "ogive-modal-is-flattest",
        "Takes the flattest part of the graph, the class with the fewest values, as the modal class.",
        _OGIVE_AREA,
    ),
    (
        "ogive-modal-is-median-class",
        "Takes the class where the graph reaches half the total as the modal class.",
        _OGIVE_AREA,
    ),
    (
        "ogive-count-for-percentage",
        "Gives the number of values where the percentage of the total was asked.",
        _OGIVE_AREA,
    ),
    (
        "ogive-complement-percentage",
        "Gives the percentage on the wrong side of the boundary (100 less the answer).",
        _OGIVE_AREA,
    ),
    (
        "ogive-percentage-of-upper-point",
        "Gives the percentage below the upper boundary and forgets to take away the percentage below the lower one.",
        _OGIVE_AREA,
    ),
    (
        "ogive-compares-cumulative",
        "Compares the cumulative frequencies at the class boundaries instead of the frequencies of the two classes.",
        _OGIVE_AREA,
    ),
    ("ogive-adds-classes", "Adds the two class frequencies where their difference was asked.", _OGIVE_AREA),
    (
        "fpoly-midpoint-for-frequency",
        "Gives the class midpoint from the horizontal axis where the frequency was asked.",
        _POLYGON_AREA,
    ),
    ("fpoly-neighbouring-point", "Reads the plotted point next to the one the question names.", _POLYGON_AREA),
    ("fpoly-misreads-gridline", "Reads the gridline above or below the plotted point.", _POLYGON_AREA),
    (
        "fpoly-frequency-for-midpoint",
        "Gives the highest frequency where the midpoint of the modal class was asked.",
        _POLYGON_AREA,
    ),
    ("fpoly-boundary-for-midpoint", "Gives a class boundary where the midpoint of the class was asked.", _POLYGON_AREA),
    (
        "fpoly-midpoint-for-boundary",
        "Gives the midpoint of the modal class where its lower boundary was asked.",
        _POLYGON_AREA,
    ),
    (
        "fpoly-upper-boundary-for-lower",
        "Gives the upper boundary of the modal class instead of its lower boundary.",
        _POLYGON_AREA,
    ),
    (
        "fpoly-counts-points",
        "Gives the number of plotted points, or the tallest point, for the total number of values.",
        _POLYGON_AREA,
    ),
    (
        "fpoly-total-misses-a-class",
        "Leaves out the lowest or highest class when adding the frequencies for the total.",
        _POLYGON_AREA,
    ),
    (
        "fpoly-total-counts-a-class-twice",
        "Counts the tallest class twice when adding the frequencies for the total.",
        _POLYGON_AREA,
    ),
    (
        "fpoly-adds-modal-and-class",
        "Adds the two class frequencies where how many more one holds than the other was asked.",
        _POLYGON_AREA,
    ),
    (
        "fpoly-modal-and-neighbour",
        "Compares the modal class with the class next to the named one, not with the named class.",
        _POLYGON_AREA,
    ),
    (
        "fpoly-boundary-class-included",
        "Includes the class just below the stated boundary when counting the values at or above it.",
        _POLYGON_AREA,
    ),
    (
        "fpoly-boundary-class-left-out",
        "Leaves out the class that starts at the stated boundary when counting the values at or above it.",
        _POLYGON_AREA,
    ),
    ("fpoly-complement-count", "Counts the values on the wrong side of the stated boundary.", _POLYGON_AREA),
    (
        "fpoly-count-for-percentage",
        "Gives the number of values where the percentage of the total was asked.",
        _POLYGON_AREA,
    ),
    (
        "fpoly-complement-percentage",
        "Gives the percentage on the wrong side of the stated boundary (100 less the answer).",
        _POLYGON_AREA,
    ),
    ("fpoly-mean-is-modal-midpoint", "Gives the midpoint of the modal class as the mean.", _POLYGON_AREA),
    ("fpoly-mean-unweighted", "Averages the class midpoints without weighting by the frequencies.", _POLYGON_AREA),
    (
        "fpoly-mean-divides-by-classes",
        "Divides the sum of midpoint times frequency by the number of classes instead of the number of values.",
        _POLYGON_AREA,
    ),
    ("fpoly-mean-of-frequencies", "Averages the frequencies instead of the midpoints weighted by them.", _POLYGON_AREA),
    (
        "fpoly-mean-uses-lower-boundaries",
        "Uses the lower boundary of each class in place of its midpoint when estimating the mean.",
        _POLYGON_AREA,
    ),
]

# ---------------------------------------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------------------------------------


def _txt(value, places: int = 0) -> str | None:
    """A reading as the answer text: whole, or a half when ``places`` is 0; one decimal place when ``places`` is 1."""
    v = Fraction(value)
    if places == 1:
        return f"{float(v):.1f}"
    if v.denominator == 1:
        return str(int(v))
    if v.denominator == 2:
        return f"{float(v):g}"
    return None


def _dist(rng, key, wrong, places: int = 0, want: int = 3) -> list[Distractor]:
    """Tagged number distractors: ``wrong`` is ``[(value, id), ...]`` in priority order; non-positive values, values the
    grader would accept for the key and values with an awkward fraction are dropped. The options are a random mix of
    values below and above the key (preferring earlier candidates), so the key's rank among them is not predictable."""
    cands = []
    for value, tag in wrong:
        if value is None or value <= 0:
            continue
        text = _txt(value, places)
        if text is None or ("." in text) != ("." in _txt(key, places)):
            continue
        cands.append(Distractor(text, tag, float(text)))
    slip_tag = (
        "fpoly-misreads-gridline" if any(tag.startswith("fpoly") for _v, tag in wrong) else "ogive-neighbouring-point"
    )
    for step in (1, 2, 3):  # a slip of a gridline or two: the last resort, so lower and higher options both exist
        for value in (key - step, key + step):
            text = _txt(value, places)
            if value > 0 and text is not None and ("." in text) == ("." in _txt(key, places)):
                cands.append(Distractor(text, slip_tag, float(text)))
    valid = pick_distractors(_txt(key, places), cands, len(cands), tolerance=GRADER_TOLERANCE)
    if len(valid) < want:
        raise TooFewDistractors(f"{len(valid)} distractors for {key}")
    lows = [d for d in valid if d.value < float(key)]
    highs = [d for d in valid if d.value > float(key)]
    lo_n = rng.randint(max(0, want - len(highs)), min(want, len(lows)))
    return lows[:lo_n] + highs[: want - lo_n]


def _fields(key, places: int = 0) -> dict:
    text = _txt(key, places)
    return dict(answer=float(text), answer_text=text, answer_latex=latex_answer(text))


def _single(rng, text, key, wrong, working, fig, places: int = 0) -> dict:
    return dict(
        text=text, working=working, distractors=_dist(rng, key, wrong, places), figure=fig, **_fields(key, places)
    )


def _part(rng, prompt, key, marks, wrong, working, *, follow=None):
    extra = dict(sensible_range=(0, None))
    if follow:
        extra["follow_through"] = follow
    return part(prompt, "number", _fields(key), marks, working, _dist(rng, key, wrong), **extra)


def _num(x) -> str:
    return _txt(x)


def _span(ctx, a, b) -> str:
    return ctx["cls"].format(a=_num(a), b=_num(b), unit=ctx["unit"])


def _ge(ctx, x) -> str:
    return ctx["ge"].format(x=_num(x), unit=ctx["unit"])


def _lt(ctx, x) -> str:
    return ctx["lt"].format(x=_num(x), unit=ctx["unit"])


def _pick_ctx(rng):
    ctx = rng.choice(_CONTEXTS)
    width = ctx["width"]
    lo = rng.randint(*ctx["lo"])
    if width > 1:
        lo -= lo % width
    return ctx, lo, width


def _walk(points):
    """The polyline through ``points`` as segments for the cartesian figure."""
    return [dict(**{"from": list(points[i]), "to": list(points[i + 1])}) for i in range(len(points) - 1)]


# ---------------------------------------------------------------------------------------------------------------------
# the ogive
# ---------------------------------------------------------------------------------------------------------------------

_U_G = ((8, 5), (12, 5), (16, 5), (20, 5), (24, 5), (12, 10), (16, 10), (20, 10))  # (grid steps in total, step)


def _ogive_data(rng, *, quartiles: bool):
    """A drawn ogive: edges, cumulative counts at each edge (in units of the grid step), and the grid step."""
    ctx, lo, width = _pick_ctx(rng)
    for _ in attempts(2000, "an ogive"):
        units, g = rng.choice(_U_G)
        k = rng.randint(6, 8)
        marks = {units // 4, units // 2, 3 * units // 4}
        free = [u for u in range(1, units) if not (quartiles and u in marks)]
        need = k - 1 - (len(marks) if quartiles else 0)
        if need < 0 or need > len(free):
            continue
        inner = sorted(rng.sample(free, need) + (sorted(marks) if quartiles else []))
        cum = [0, *inner, units]
        break
    edges = [lo + i * width for i in range(k + 1)]
    n = units * g
    return ctx, width, edges, cum, g, n, units


def _ogive_figure(ctx, edges, cum, g, n, width):
    pts = [(edges[i], cum[i] * g) for i in range(len(edges))]
    return figures.check(
        dict(
            kind="cartesian",
            alt="a cumulative frequency graph",
            x_range=[edges[0], edges[-1]],
            y_range=[0, n],
            grid=width,
            x_grid=width,
            y_grid=g,
            equal_scale=False,
            x_tick_format="number",
            x_label=ctx["axis"],
            y_label="Cumulative frequency",
            tick_labels=True,
            points=[dict(at=[x, y]) for x, y in pts],
            segments=_walk(pts),
            lines=[],
            curves=[],
        )
    )


def _quartile_indices(cum, units):
    return tuple(cum.index(units * q // 4) for q in (1, 2, 3))


def _raw_values(edges, cum, g):
    """The data the classes stand for, each value taken as its class's lower boundary (for second-way checks)."""
    out = []
    for i in range(len(edges) - 1):
        out += [edges[i]] * ((cum[i + 1] - cum[i]) * g)
    return out


def _ogive_easy(rng):
    ctx, width, edges, cum, g, n, units = _ogive_data(rng, quartiles=False)
    fig = _ogive_figure(ctx, edges, cum, g, n, width)
    k = len(edges) - 1
    if rng.random() < 0.65:
        j = rng.randint(1, k - 1)
        key = cum[j] * g
        check(len([v for v in _raw_values(edges, cum, g) if v < edges[j]]) == key, "count below differs")
        text = f"{ctx['lead']} The cumulative frequency graph shows the results. {phrase(rng, ('How many {s} {x}?', 'Use the graph to find how many {s} {x}.'), s=ctx['subj'], x=_lt(ctx, edges[j]))}"
        wrong = [
            (edges[j], "ogive-boundary-for-count"),
            (n, "ogive-total-for-count"),
            (cum[j + 1] * g, "ogive-neighbouring-point"),
            (cum[j - 1] * g, "ogive-neighbouring-point"),
            ((units - cum[j]) * g, "ogive-complement-count"),
        ]
        working = [
            f"Find {_num(edges[j])} on the horizontal axis and go up to the graph.",
            f"The height there is {key}.",
        ]
    else:
        key = n
        text = f"{ctx['lead']} The cumulative frequency graph shows the results. {phrase(rng, ('How many {s} are in the data altogether?', 'Find the total number of {s} in the data.'), s=ctx['subj'])}"
        wrong = [
            (cum[k - 1] * g, "ogive-neighbouring-point"),
            (edges[-1], "ogive-boundary-for-count"),
            (k, "ogive-classes-for-total"),
            (sum(cum) * g, "ogive-adds-all-heights"),
            ((cum[k - 1] + cum[k - 2]) * g, "ogive-adds-all-heights"),
        ]
        working = [f"The graph finishes at its highest point, which is the total: {n}."]
    return _single(rng, text, key, wrong, working, fig)


def _ogive_medium(rng):
    ctx, width, edges, cum, g, n, units = _ogive_data(rng, quartiles=True)
    fig = _ogive_figure(ctx, edges, cum, g, n, width)
    k = len(edges) - 1
    qa, qb, qc = _quartile_indices(cum, units)
    lead = f"{ctx['lead']} The cumulative frequency graph shows the results."
    variants = ["median", "between", "atleast"]
    pct_ok = [j for j in range(1, k) if (cum[j] * g * 100) % n == 0]
    if pct_ok:
        variants.append("pct")
    variant = rng.choice(variants)
    if variant == "median":
        key = edges[qb]
        values = _raw_values(edges, cum, g)
        check(values[n // 2 - 1] < key <= values[n // 2] + width, "median class differs")
        text = (
            f"{lead} The median is the value that half of the {ctx['subj']} are below. "
            f"What is the median, in {ctx['unit']}?"
        )
        wrong = [
            (n // 2, "ogive-count-for-value"),
            ((edges[0] + edges[-1]) / 2, "ogive-middle-of-axis"),
            (edges[qb + 1], "ogive-neighbouring-point"),
            (edges[qb - 1], "ogive-neighbouring-point"),
            (edges[qa], "ogive-neighbouring-point"),
        ]
        working = [
            f"Half of the {n} values is {n // 2}.",
            f"Go across from {n // 2} to the graph and down: {_num(key)}.",
        ]
    elif variant == "between":
        i = rng.randint(0, k - 2)
        j = rng.randint(i + 2, k)
        key = (cum[j] - cum[i]) * g
        text = f"{lead} {phrase(rng, ('How many of the {s} {c}?', 'Use the graph to find how many {s} {c}.'), s=ctx['subj'], c=_span(ctx, edges[i], edges[j]))}"
        wrong = [
            (cum[j] * g, "ogive-upper-point-only"),
            ((cum[j] + cum[i]) * g, "ogive-adds-heights"),
            (cum[i] * g if i else None, "ogive-lower-point-only"),
            (edges[j] - edges[i], "ogive-width-for-count"),
            ((units - cum[j] + cum[i]) * g, "ogive-complement-count"),
        ]
        working = [
            f"Height at {_num(edges[j])}: {cum[j] * g}. Height at {_num(edges[i])}: {cum[i] * g}.",
            f"{cum[j] * g} - {cum[i] * g} = {key}.",
        ]
    elif variant == "atleast":
        j = rng.randint(1, k - 1)
        key = (units - cum[j]) * g
        text = f"{lead} {phrase(rng, ('How many {s} {x}?', 'Use the graph to find how many {s} {x}.'), s=ctx['subj'], x=_ge(ctx, edges[j]))}"
        wrong = [
            (cum[j] * g, "ogive-complement-count"),
            (n, "ogive-total-for-count"),
            ((units - cum[j + 1]) * g if j + 1 < k else None, "ogive-neighbouring-point"),
            ((units - cum[j - 1]) * g, "ogive-neighbouring-point"),
            (edges[j], "ogive-boundary-for-count"),
        ]
        working = [f"The graph shows {cum[j] * g} below {_num(edges[j])}.", f"{n} - {cum[j] * g} = {key}."]
    else:
        j = rng.choice(pct_ok)
        key = Fraction(cum[j] * g * 100, n)
        text = f"{lead} {phrase(rng, ('What percentage of the {s} {x}?', 'Work out the percentage of the {s} that {x}.'), s=ctx['subj'], x=_lt(ctx, edges[j]))}"
        wrong = [
            (cum[j] * g, "ogive-count-for-percentage"),
            (100 - key, "ogive-complement-percentage"),
            (cum[j + 1] * g * 100 / n if j + 1 < k else None, "ogive-neighbouring-point"),
            (cum[j - 1] * g * 100 / n, "ogive-neighbouring-point"),
            (key + 5, "ogive-neighbouring-point"),
        ]
        working = [f"The graph shows {cum[j] * g} below {_num(edges[j])}.", f"{cum[j] * g} out of {n} is {_num(key)}%."]
    return _single(rng, text, key, wrong, working, fig)


def _ogive_hard(rng):
    ctx, width, edges, cum, g, n, units = _ogive_data(rng, quartiles=True)
    fig = _ogive_figure(ctx, edges, cum, g, n, width)
    k = len(edges) - 1
    qa, qb, qc = _quartile_indices(cum, units)
    lead = f"{ctx['lead']} The cumulative frequency graph shows the results."
    freqs = [cum[i + 1] - cum[i] for i in range(k)]
    variants = ["iqr"]
    if freqs.count(max(freqs)) == 1 and freqs.index(max(freqs)) not in (k - 1, qb - 1 if qb else -1, qb):
        variants.append("modal")
    pairs = [
        (i, j)
        for i in range(k - 1)
        for j in range(i + 2, k + 1)
        if (cum[j] - cum[i]) * g * 100 % n == 0 and 0 < (cum[j] - cum[i]) * 4 < 3 * units
    ]
    if pairs:
        variants.append("pct")
    ordered = [(i, j) for i in range(k) for j in range(k) if freqs[i] > freqs[j] and i != j]
    if ordered:
        variants.append("diff")
    variant = rng.choice(variants)
    if variant == "iqr":
        key = edges[qc] - edges[qa]
        text = (
            f"{lead} The lower quartile is the value that a quarter of the {ctx['subj']} are below, and the upper quartile "
            f"is the value that three quarters are below. What is the interquartile range, in {ctx['unit']}?"
        )
        wrong = [
            (edges[qc] + edges[qa], "ogive-quartile-sum"),
            (edges[-1] - edges[0], "ogive-range-for-iqr"),
            (edges[qc] - edges[qb], "ogive-half-spread"),
            (edges[qb] - edges[qa], "ogive-half-spread"),
            (n // 2, "ogive-count-between-quartiles"),
            (edges[qc], "ogive-upper-quartile-only"),
        ]
        working = [
            f"A quarter of {n} is {n // 4}, so the lower quartile is {_num(edges[qa])}.",
            f"Three quarters of {n} is {3 * n // 4}, so the upper quartile is {_num(edges[qc])}.",
            f"{_num(edges[qc])} - {_num(edges[qa])} = {_num(key)}.",
        ]
    elif variant == "modal":
        m = freqs.index(max(freqs))
        key = edges[m]
        check(all(freqs[m] > f for i, f in enumerate(freqs) if i != m), "modal class not unique")
        text = (
            f"{lead} The modal class is the class with the most {ctx['subj']}. "
            f"What is the lower boundary of the modal class, in {ctx['unit']}?"
        )
        flattest = freqs.index(min(freqs)) if freqs.count(min(freqs)) == 1 else None
        wrong = [
            (edges[m + 1], "ogive-modal-upper-boundary"),
            (edges[k - 1], "ogive-modal-is-last-class"),
            (edges[flattest] if flattest is not None else None, "ogive-modal-is-flattest"),
            (edges[qb - 1], "ogive-modal-is-median-class"),
            (edges[m - 1] if m else None, "ogive-neighbouring-point"),
        ]
        working = [
            "The class with the most values is where the graph rises most steeply.",
            f"That is the class from {_num(edges[m])} to {_num(edges[m + 1])}, which holds {freqs[m] * g} values.",
        ]
    elif variant == "pct":
        i, j = rng.choice(pairs)
        key = Fraction((cum[j] - cum[i]) * g * 100, n)
        text = f"{lead} {phrase(rng, ('What percentage of the {s} {c}?', 'Work out the percentage of the {s} that {c}.'), s=ctx['subj'], c=_span(ctx, edges[i], edges[j]))}"
        wrong = [
            ((cum[j] - cum[i]) * g, "ogive-count-for-percentage"),
            (100 - key, "ogive-complement-percentage"),
            (Fraction(cum[j] * g * 100, n), "ogive-percentage-of-upper-point"),
            (Fraction((cum[j] + cum[i]) * g * 100, n), "ogive-adds-heights"),
            (key + 5, "ogive-neighbouring-point"),
        ]
        working = [
            f"Height at {_num(edges[j])}: {cum[j] * g}. Height at {_num(edges[i])}: {cum[i] * g}.",
            f"{cum[j] * g} - {cum[i] * g} = {(cum[j] - cum[i]) * g} values, which is {_num(key)}% of {n}.",
        ]
    else:
        i, j = rng.choice(ordered)
        key = (freqs[i] - freqs[j]) * g
        text = f"{lead} {phrase(rng, ('How many more {s} {c1} than {c2}?', 'How many more of the {s} {c1} than {c2}?'), s=ctx['subj'], c1=_span(ctx, edges[i], edges[i + 1]), c2=_span(ctx, edges[j], edges[j + 1]))}"
        wrong = [
            ((cum[i + 1] - cum[j + 1]) * g, "ogive-compares-cumulative"),
            ((freqs[i] + freqs[j]) * g, "ogive-adds-classes"),
            (freqs[i] * g, "ogive-lower-point-only"),
            ((cum[i + 1] - cum[j]) * g, "ogive-compares-cumulative"),
            ((freqs[i] - freqs[j]) * g + g, "ogive-neighbouring-point"),
        ]
        working = [
            f"Class from {_num(edges[i])}: {cum[i + 1] * g} - {cum[i] * g} = {freqs[i] * g}.",
            f"Class from {_num(edges[j])}: {cum[j + 1] * g} - {cum[j] * g} = {freqs[j] * g}.",
            f"{freqs[i] * g} - {freqs[j] * g} = {key}.",
        ]
    return _single(rng, text, key, wrong, working, fig)


_OGIVE = {"easy": _ogive_easy, "medium": _ogive_medium, "hard": _ogive_hard}


@generator(OGIVE, level=11, presets={t: {"tier": t} for t in TIERS})
def w8_ogive_read(rng, tier="easy"):
    """Read a number from a cumulative frequency graph (ogive); the tiers are described in the module docstring."""
    if tier not in _OGIVE:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _OGIVE[tier])


@generator(OGIVE_EXAM, level=11, presets={"exam": {"tier": "exam"}})
def exd_x_w8_ogive_chain(rng, tier="exam"):
    """Exam chain on one ogive: the total, the number between two boundaries, the median and the interquartile range."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")

    def build(rng):
        ctx, width, edges, cum, g, n, units = _ogive_data(rng, quartiles=True)
        fig = _ogive_figure(ctx, edges, cum, g, n, width)
        k = len(edges) - 1
        qa, qb, qc = _quartile_indices(cum, units)
        i = rng.randint(0, k - 3)
        j = rng.randint(i + 2, k)
        between = (cum[j] - cum[i]) * g
        iqr = edges[qc] - edges[qa]
        values = _raw_values(edges, cum, g)
        check(sum(1 for v in values if edges[i] <= v < edges[j]) == between, "between differs")
        part_a = _part(
            rng,
            f"How many {ctx['subj']} are in the data altogether?",
            n,
            1,
            [
                (cum[k - 1] * g, "ogive-neighbouring-point"),
                (edges[-1], "ogive-boundary-for-count"),
                (k, "ogive-classes-for-total"),
                (sum(cum) * g, "ogive-adds-all-heights"),
            ],
            [f"The graph finishes at its highest point, which is the total: {n}."],
        )
        part_b = _part(
            rng,
            f"How many of the {ctx['subj']} {_span(ctx, edges[i], edges[j])}?",
            between,
            2,
            [
                (cum[j] * g, "ogive-upper-point-only"),
                ((cum[j] + cum[i]) * g, "ogive-adds-heights"),
                (cum[i] * g if i else None, "ogive-lower-point-only"),
                (edges[j] - edges[i], "ogive-width-for-count"),
                ((units - cum[j] + cum[i]) * g, "ogive-complement-count"),
            ],
            [f"{cum[j] * g} - {cum[i] * g} = {between}."],
        )
        part_c = _part(
            rng,
            f"The median is the value that half of the {ctx['subj']} are below. What is the median, in {ctx['unit']}?",
            edges[qb],
            1,
            [
                (n // 2, "ogive-count-for-value"),
                ((edges[0] + edges[-1]) / 2, "ogive-middle-of-axis"),
                (edges[qb + 1], "ogive-neighbouring-point"),
                (edges[qb - 1], "ogive-neighbouring-point"),
            ],
            [f"Half of {n} is {n // 2}; the graph reaches {n // 2} at {_num(edges[qb])}."],
        )
        part_d = _part(
            rng,
            "The lower quartile is the value that a quarter of the data are below, and the upper quartile is the value "
            f"that three quarters are below. Find the interquartile range, in {ctx['unit']}.",
            iqr,
            2,
            [
                (edges[qc] + edges[qa], "ogive-quartile-sum"),
                (edges[-1] - edges[0], "ogive-range-for-iqr"),
                (edges[qc] - edges[qb], "ogive-half-spread"),
                (edges[qb] - edges[qa], "ogive-half-spread"),
                (n // 2, "ogive-count-between-quartiles"),
            ],
            [
                f"Lower quartile (at {n // 4}): {_num(edges[qa])}. Upper quartile (at {3 * n // 4}): {_num(edges[qc])}.",
                f"{_num(edges[qc])} - {_num(edges[qa])} = {_num(iqr)}.",
            ],
        )
        return dict(
            text=f"{ctx['lead']} The cumulative frequency graph shows the results.",
            parts=[part_a, part_b, part_c, part_d],
            figure=fig,
        )

    return retry_draw(rng, build)


def _read_ogive(question) -> str:
    """Every answer re-derived from the plotted points and the prompts alone."""
    pts = sorted((p["at"][0], p["at"][1]) for p in question.figure["points"])
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    top = ys[-1]

    def height(x):
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1]:
                return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])
        raise AssertionError(x)

    def value(y):
        for i in range(len(xs) - 1):
            if ys[i] <= y <= ys[i + 1]:
                return xs[i] + (xs[i + 1] - xs[i]) * (y - ys[i]) / (ys[i + 1] - ys[i])
        raise AssertionError(y)

    def one(text):
        low = text.lower()
        spans = re.findall(r"from (\d+(?:\.\d+)?) to less than (\d+(?:\.\d+)?)", text)
        if "altogether" in low or "total number" in low:
            return top
        if "median" in low:
            return value(top / 2)
        if "interquartile" in low:
            return value(3 * top / 4) - value(top / 4)
        if "lower boundary of the modal class" in low:
            rises = [ys[i + 1] - ys[i] for i in range(len(xs) - 1)]
            return xs[rises.index(max(rises))]
        if "how many more" in low and len(spans) == 2:
            (a1, b1), (a2, b2) = spans
            return (height(float(b1)) - height(float(a1))) - (height(float(b2)) - height(float(a2)))
        if spans:
            a, b = spans[0]
            count = height(float(b)) - height(float(a))
        else:
            x = float(re.findall(r"(\d+(?:\.\d+)?)", text)[-1])
            count = top - height(x) if "or more" in low else height(x)
        return 100 * count / top if "percentage" in low else count

    def fmt(v):
        return _txt(Fraction(v).limit_denominator(1000))

    if question.parts:
        prompts = split_parts(question.text)
        return "; ".join(f"({lab}) {fmt(one(prompts[lab]))}" for lab in sorted(prompts))
    return fmt(one(question.text))


figures.register_reader(OGIVE, _read_ogive)
figures.register_reader(OGIVE_EXAM, _read_ogive)


# ---------------------------------------------------------------------------------------------------------------------
# the frequency polygon
# ---------------------------------------------------------------------------------------------------------------------


def _polygon_data(rng, tier, *, whole_pct=False):
    ctx, lo, width = _pick_ctx(rng)
    if width == 1:
        lo = max(lo, 4)
    g = 1 if tier == "hard" else 2
    cap = 12 if g == 1 else 16
    k = rng.randint(5, 7)
    for _ in attempts(5000, "a frequency polygon"):
        units = [rng.randint(1, cap // g) for _ in range(k)]
        freqs = [u * g for u in units]
        if freqs.count(max(freqs)) != 1:
            continue
        if len(set(freqs)) < k - 1:
            continue
        n = sum(freqs)
        if whole_pct and not any((100 * sum(freqs[:j])) % n == 0 and 0 < j < k for j in range(1, k)):
            continue
        break
    edges = [lo + i * width for i in range(k + 1)]
    mids = [Fraction(edges[i] + edges[i + 1], 2) for i in range(k)]
    return ctx, width, g, k, edges, mids, freqs, max(freqs)


def _polygon_figure(ctx, mids, freqs, width, g):
    xs = [mids[0] - width, *mids, mids[-1] + width]
    ys = [0, *freqs, 0]
    top = -(-max(freqs) // g) * g + g
    pts = [(float(x) if Fraction(x).denominator != 1 else int(x), y) for x, y in zip(xs, ys)]
    return figures.check(
        dict(
            kind="cartesian",
            alt="a frequency polygon",
            x_range=[pts[0][0], pts[-1][0]],
            y_range=[0, top],
            grid=g,
            x_grid=width,
            y_grid=g,
            equal_scale=False,
            x_tick_format="number",
            x_label=ctx["axis"],
            y_label="Frequency",
            tick_labels=True,
            points=[dict(at=[x, y]) for x, y in pts],
            segments=_walk(pts),
            lines=[],
            curves=[],
        )
    )


def _poly_lead(ctx, width) -> str:
    unit = ctx["unit"][:-1] if width == 1 and ctx["unit"].endswith("s") else ctx["unit"]
    return (
        f"{ctx['lead']} The frequency polygon shows the results, in classes {_num(width)} {unit} wide, "
        "with each point plotted above the midpoint of its class."
    )


def _polygon_wrong_total(freqs, k):
    return [
        (k + 2, "fpoly-counts-points"),
        (max(freqs), "fpoly-counts-points"),
        (sum(freqs) - freqs[0], "fpoly-total-misses-a-class"),
        (sum(freqs) - freqs[-1], "fpoly-total-misses-a-class"),
        (sum(freqs) + max(freqs), "fpoly-total-counts-a-class-twice"),
    ]


def _polygon_easy(rng):
    ctx, width, g, k, edges, mids, freqs, top = _polygon_data(rng, "easy")
    fig = _polygon_figure(ctx, mids, freqs, width, g)
    lead = _poly_lead(ctx, width)
    if rng.random() < 0.6:
        i = rng.randrange(k)
        key = freqs[i]
        text = f"{lead} {phrase(rng, ('How many {s} are in the class with midpoint {m}?', 'What is the frequency of the class with midpoint {m}?'), s=ctx['subj'], m=_num(mids[i]))}"
        wrong = [
            (mids[i], "fpoly-midpoint-for-frequency"),
            (freqs[i - 1] if i else freqs[i + 1], "fpoly-neighbouring-point"),
            (freqs[(i + 1) % k] if i + 1 < k else freqs[i - 1], "fpoly-neighbouring-point"),
            (key + g, "fpoly-misreads-gridline"),
            (key - g, "fpoly-misreads-gridline"),
        ]
        working = [f"Find {_num(mids[i])} on the horizontal axis and read the height of the point above it: {key}."]
    else:
        m = freqs.index(top)
        key = mids[m]
        text = f"{lead} What is the midpoint of the modal class, the class with the most {ctx['subj']}?"
        wrong = [
            (top, "fpoly-frequency-for-midpoint"),
            (mids[m - 1] if m else mids[m + 1], "fpoly-neighbouring-point"),
            (mids[m + 1] if m + 1 < k else mids[m - 1], "fpoly-neighbouring-point"),
            (edges[m], "fpoly-boundary-for-midpoint"),
            (edges[m + 1], "fpoly-boundary-for-midpoint"),
        ]
        working = [f"The highest point is above {_num(key)}, so the modal class has midpoint {_num(key)}."]
    return _single(rng, text, key, wrong, working, fig)


def _polygon_medium(rng):
    ctx, width, g, k, edges, mids, freqs, top = _polygon_data(rng, "medium")
    fig = _polygon_figure(ctx, mids, freqs, width, g)
    lead = _poly_lead(ctx, width)
    m = freqs.index(top)
    variant = rng.choice(("total", "more", "boundary"))
    if variant == "total":
        key = sum(freqs)
        text = f"{lead} {phrase(rng, ('How many {s} are in the data altogether?', 'Find the total number of {s} in the data.'), s=ctx['subj'])}"
        wrong = _polygon_wrong_total(freqs, k)
        working = [f"Add the heights of the points: {' + '.join(map(str, freqs))} = {key}."]
    elif variant == "more":
        i = rng.choice([x for x in range(k) if x != m])
        key = top - freqs[i]
        text = f"{lead} How many more {ctx['subj']} are in the modal class than in the class with midpoint {_num(mids[i])}?"
        other = freqs[i - 1] if i else freqs[i + 1]
        wrong = [
            (top + freqs[i], "fpoly-adds-modal-and-class"),
            (top - other, "fpoly-modal-and-neighbour"),
            (top, "fpoly-neighbouring-point"),
            (key + g, "fpoly-misreads-gridline"),
            (abs(freqs[i] - freqs[(i + 1) % k]), "fpoly-modal-and-neighbour"),
        ]
        working = [
            f"Modal class: {top}. Class with midpoint {_num(mids[i])}: {freqs[i]}.",
            f"{top} - {freqs[i]} = {key}.",
        ]
    else:
        key = edges[m]
        text = f"{lead} What is the lower boundary of the modal class, in {ctx['unit']}?"
        wrong = [
            (mids[m], "fpoly-midpoint-for-boundary"),
            (edges[m + 1], "fpoly-upper-boundary-for-lower"),
            (edges[m - 1] if m else None, "fpoly-neighbouring-point"),
            (mids[m] - width, "fpoly-neighbouring-point"),
            (top, "fpoly-frequency-for-midpoint"),
        ]
        working = [
            f"The highest point is above the midpoint {_num(mids[m])}.",
            f"The classes are {_num(width)} wide, so the class runs from {_num(mids[m])} - {_num(width / 2)} = {_num(key)} to {_num(edges[m + 1])}.",
        ]
    return _single(rng, text, key, wrong, working, fig)


def _polygon_hard(rng):
    variant = rng.choice(("ge", "lt", "pct", "mean"))
    for _ in attempts(500, "a hard polygon"):
        ctx, width, g, k, edges, mids, freqs, top = _polygon_data(rng, "hard", whole_pct=(variant == "pct"))
        n = sum(freqs)
        if variant == "mean":
            mean = Fraction(sum(m * f for m, f in zip(mids, freqs)), n)
            if mean.denominator in (1, 2):
                break
            continue
        break
    fig = _polygon_figure(ctx, mids, freqs, width, g)
    lead = _poly_lead(ctx, width)
    if variant == "mean":
        key = mean
        text = f"{lead} Use the class midpoints to estimate the mean, in {ctx['unit']}."
        wrong = [
            (mids[freqs.index(top)], "fpoly-mean-is-modal-midpoint"),
            (sum(mids) / k, "fpoly-mean-unweighted"),
            (Fraction(sum(m * f for m, f in zip(mids, freqs)), k), "fpoly-mean-divides-by-classes"),
            (Fraction(sum(edges[i] * freqs[i] for i in range(k)), n), "fpoly-mean-uses-lower-boundaries"),
            (Fraction(n, k), "fpoly-mean-of-frequencies"),
        ]
        working = [
            f"Multiply each midpoint by its frequency and add: {sum(m * f for m, f in zip(mids, freqs))}.",
            f"Divide by the total {n}: {_txt(key, 1)}.",
        ]
        values = [m for m, f in zip(mids, freqs) for _ in range(f)]
        check(Fraction(sum(values), len(values)) == key, "mean differs")
        return _single(rng, text, key, wrong, working, fig, places=1)
    j = rng.randint(1, k - 1)
    if variant == "pct":
        js = [x for x in range(1, k) if (100 * sum(freqs[:x])) % n == 0]
        j = rng.choice(js)
    below, above = sum(freqs[:j]), sum(freqs[j:])
    x = edges[j]
    if variant == "lt":
        key = below
        text = f"{lead} {phrase(rng, ('How many {s} {x}?', 'Use the graph to find how many {s} {x}.'), s=ctx['subj'], x=_lt(ctx, x))}"
        wrong = [
            (above, "fpoly-complement-count"),
            (below + freqs[j], "fpoly-boundary-class-included"),
            (below - freqs[j - 1], "fpoly-boundary-class-left-out"),
            (n, "fpoly-counts-points"),
            (x, "fpoly-midpoint-for-frequency"),
        ]
        working = [f"Classes below {_num(x)}: {' + '.join(map(str, freqs[:j]))} = {key}."]
    elif variant == "ge":
        key = above
        text = f"{lead} {phrase(rng, ('How many {s} {x}?', 'Use the graph to find how many {s} {x}.'), s=ctx['subj'], x=_ge(ctx, x))}"
        wrong = [
            (below, "fpoly-complement-count"),
            (above + freqs[j - 1], "fpoly-boundary-class-included"),
            (above - freqs[j], "fpoly-boundary-class-left-out"),
            (n, "fpoly-counts-points"),
            (x, "fpoly-midpoint-for-frequency"),
        ]
        working = [f"Classes from {_num(x)} up: {' + '.join(map(str, freqs[j:]))} = {key}."]
    else:
        key = Fraction(100 * below, n)
        text = f"{lead} {phrase(rng, ('What percentage of the {s} {x}?', 'Work out the percentage of the {s} that {x}.'), s=ctx['subj'], x=_lt(ctx, x))}"
        wrong = [
            (below, "fpoly-count-for-percentage"),
            (100 - key, "fpoly-complement-percentage"),
            (Fraction(100 * (below + freqs[j]), n), "fpoly-boundary-class-included"),
            (Fraction(100 * (below - freqs[j - 1]), n), "fpoly-boundary-class-left-out"),
            (key + 5, "fpoly-misreads-gridline"),
        ]
        working = [
            f"Classes below {_num(x)}: {' + '.join(map(str, freqs[:j]))} = {below}.",
            f"{below} out of {n} is {_num(key)}%.",
        ]
    return _single(rng, text, key, wrong, working, fig)


_POLYGON = {"easy": _polygon_easy, "medium": _polygon_medium, "hard": _polygon_hard}


@generator(POLYGON, level=11, presets={t: {"tier": t} for t in TIERS})
def w8_polygon_read(rng, tier="easy"):
    """Read a number from a frequency polygon; the tiers are described in the module docstring."""
    if tier not in _POLYGON:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _POLYGON[tier])


@generator(POLYGON_EXAM, level=11, presets={"exam": {"tier": "exam"}})
def exd_x_w8_polygon_chain(rng, tier="exam"):
    """Exam chain on one frequency polygon: a class frequency, the total, a count from a boundary up and a percentage."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")

    def build(rng):
        ctx, width, g, k, edges, mids, freqs, top = _polygon_data(rng, "medium", whole_pct=True)
        n = sum(freqs)
        fig = _polygon_figure(ctx, mids, freqs, width, g)
        i = rng.randrange(k)
        js = [x for x in range(1, k) if (100 * sum(freqs[:x])) % n == 0]
        j = rng.choice(js)
        jc = rng.randint(1, k - 1)
        above, below = sum(freqs[jc:]), sum(freqs[:j])
        pct = Fraction(100 * below, n)
        part_a = _part(
            rng,
            f"What is the frequency of the class with midpoint {_num(mids[i])}?",
            freqs[i],
            1,
            [
                (mids[i], "fpoly-midpoint-for-frequency"),
                (freqs[i - 1] if i else freqs[i + 1], "fpoly-neighbouring-point"),
                (freqs[i] + g, "fpoly-misreads-gridline"),
                (freqs[i] - g, "fpoly-misreads-gridline"),
            ],
            [f"The point above {_num(mids[i])} is at height {freqs[i]}."],
        )
        part_b = _part(
            rng,
            f"How many {ctx['subj']} are in the data altogether?",
            n,
            1,
            _polygon_wrong_total(freqs, k),
            [f"{' + '.join(map(str, freqs))} = {n}."],
        )
        part_c = _part(
            rng,
            f"How many {ctx['subj']} {_ge(ctx, edges[jc])}?",
            above,
            2,
            [
                (sum(freqs[:jc]), "fpoly-complement-count"),
                (above + freqs[jc - 1], "fpoly-boundary-class-included"),
                (above - freqs[jc], "fpoly-boundary-class-left-out"),
                (n, "fpoly-counts-points"),
            ],
            [f"Classes from {_num(edges[jc])} up: {' + '.join(map(str, freqs[jc:]))} = {above}."],
        )
        part_d = _part(
            rng,
            f"What percentage of the {ctx['subj']} {_lt(ctx, edges[j])}?",
            pct,
            2,
            [
                (below, "fpoly-count-for-percentage"),
                (100 - pct, "fpoly-complement-percentage"),
                (Fraction(100 * (below + freqs[j]), n), "fpoly-boundary-class-included"),
                (Fraction(100 * (below - freqs[j - 1]), n), "fpoly-boundary-class-left-out"),
            ],
            [f"Classes below {_num(edges[j])} hold {below} of the {n} values.", f"{below} out of {n} is {_num(pct)}%."],
            follow=f"100*{below}/part_b",
        )
        return dict(text=_poly_lead(ctx, width), parts=[part_a, part_b, part_c, part_d], figure=fig)

    return retry_draw(rng, build)


def _read_polygon(question) -> str:
    """Every answer re-derived from the plotted points and the prompts alone."""
    pts = sorted((p["at"][0], p["at"][1]) for p in question.figure["points"])
    width = question.figure["x_grid"]
    mids = [Fraction(p[0]).limit_denominator(100) for p in pts]
    freqs = [p[1] for p in pts]
    n = sum(freqs)
    top = max(freqs)
    nums = r"(\d+(?:\.\d+)?)"

    def at(m):
        return freqs[mids.index(Fraction(m).limit_denominator(100))]

    def one(text):
        text = text.split("the midpoint of its class.")[-1]
        low = text.lower()
        if "midpoint of the modal class" in low:
            return mids[freqs.index(top)]
        if "lower boundary of the modal class" in low:
            return mids[freqs.index(top)] - Fraction(width).limit_denominator(100) / 2
        if "altogether" in low or "total number" in low:
            return n
        if "estimate the mean" in low:
            return sum(m * f for m, f in zip(mids, freqs)) / n
        if "how many more" in low and "midpoint" in low:
            return top - at(re.findall(nums, text)[-1])
        if "midpoint" in low:
            return at(re.findall(nums, text.split("midpoint")[-1])[0])
        x = Fraction(re.findall(nums, text)[-1]).limit_denominator(100)
        w = Fraction(width).limit_denominator(100)
        if "or more" in low:
            count = sum(f for m, f in zip(mids, freqs) if m - w / 2 >= x)
        else:
            count = sum(f for m, f in zip(mids, freqs) if m + w / 2 <= x)
        return 100 * Fraction(count, n) if "percentage" in low else count

    def fmt(v, text):
        return _txt(v, 1 if "estimate the mean" in text.lower() else 0)

    if question.parts:
        prompts = split_parts(question.text)
        return "; ".join(f"({lab}) {fmt(one(prompts[lab]), prompts[lab])}" for lab in sorted(prompts))
    return fmt(one(question.text), question.text)


figures.register_reader(POLYGON, _read_polygon)
figures.register_reader(POLYGON_EXAM, _read_polygon)
