"""Generates figure questions on spread and on the variability of random samples (Year 11-12).

* ``ey_dot_plot_spread`` (ACMEM054, ``senior-describe-spread-from-a-dot-plot``): one dot plot. Easy says whether the dots
  are tightly packed or spread out, medium names the pattern (two clusters with a gap, one cluster with an outlier, even
  spread, tightly packed), hard picks the statement that places the biggest gap and the most crowded value correctly, and
  the exam chain (``ey_dot_plot_spread_exam``) reads the outlier, the top of the main cluster, the gap and the range.
* ``ey_sample_variability`` (ACMMM173, ``senior-simulated-samples-variability``): histograms of simulated data. Easy
  compares the spread of two samples, medium says which of two displays of sample proportions used the larger samples
  (the narrower one), hard names the source of one histogram (uniform, normal, Bernoulli or skewed), and the exam chain
  (``ey_sample_variability_exam``) counts the values in a central band of each display and compares them.

The informal words are tied to rules the reader applies to the dots: a *gap* is three or more empty ticks between
occupied ticks, a *cluster* is a run of dots with no gap inside it, *tightly packed* is one cluster no more than five
ticks wide, an *outlier* is a single dot a gap away from one cluster. A generator draws a plot of one kind by
construction and the reader classifies it again from the dots alone. For histograms the reader computes the spread of
the grouped data from the class mid-points and checks the answer by the occupied range too.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction as F

from .. import figures
from ..core import attempts, check, generator
from ..figure_kinds import data
from ._ey_senior_kit import mcq, num_part, retry_short

_TAILS = (
    "Read the values from the grid.",
    "Every value you need sits on a grid crossing.",
    "Work through the parts in order.",
)

# =============================================================================================== dot plot (ACMEM054)

DOTS = "ey_dot_plot_spread"
DOTS_EXAM = "ey_dot_plot_spread_exam"

_QUANTITIES = (
    ("The dot plot shows the number of text messages {who} sent in one day.", "Messages sent", ("some students",)),
    ("The dot plot shows the number of goals scored by a hockey team in each game of a season.", "Goals scored", ()),
    (
        "The dot plot shows the number of minutes it took {who} to get to school.",
        "Minutes to get to school",
        ("some students",),
    ),
    ("The dot plot shows the number of books borrowed by each member of a club in a month.", "Books borrowed", ()),
    ("The dot plot shows the number of customers who came into a cafe in each hour.", "Customers in an hour", ()),
)
_TICKS = 16  # the axis spans 16 steps: 17 numerals


def _plot(rows: dict, low: int, label: str, alt="a dot plot of a set of whole-number data"):
    return data.dot_plot(
        low=low, high=low + _TICKS, step=1, dots={low + k: n for k, n in sorted(rows.items())}, alt=alt, x_label=label
    )


def _features(rows: dict) -> dict:
    """Cluster / gap / outlier features of ``{tick: count}`` (a gap = three or more empty ticks)."""
    ticks = sorted(rows)
    runs = [(a, b, b - a - 1) for a, b in zip(ticks, ticks[1:]) if b - a - 1 > 0]
    clusters, cur = [], [ticks[0]]
    for a, b in zip(ticks, ticks[1:]):
        if b - a - 1 >= 3:
            clusters.append(cur)
            cur = []
        cur.append(b)
    clusters.append(cur)
    sizes = [sum(rows[t] for t in c) for c in clusters]
    return dict(
        ticks=ticks, runs=runs, clusters=clusters, sizes=sizes, span=ticks[-1] - ticks[0], total=sum(rows.values())
    )


def _lone(f: dict) -> int:
    """The tick of the single dot that sits alone in its own cluster."""
    return [t for c, n in zip(f["clusters"], f["sizes"]) if n == 1 for t in c][0]


def _dot_working(kind: str, f: dict, low: int) -> list[str]:
    if kind == "tight":
        return [f"All the dots lie within {f['span']} steps of each other, in one group with no gaps."]
    if kind == "spread":
        return [f"The dots run across {f['span']} steps with no big gap anywhere, so they are spread out."]
    if kind == "two_clusters":
        return [f"There are two groups of dots with {max(r[2] for r in f['runs'])} empty values between them."]
    return [f"One group of dots, and a single dot far from it at {_lone(f) + low}."]


def _classify(rows: dict) -> str:
    """tight, spread, two_clusters or outlier; raises if the plot is not clear-cut."""
    f = _features(rows)
    cl, sizes = f["clusters"], f["sizes"]
    if len(cl) == 1:
        if f["span"] <= 5:
            return "tight"
        check(f["span"] >= 12, "dot plot: the spread is neither tight nor wide")
        return "spread"
    if len(cl) == 2:
        if min(sizes) == 1:
            return "outlier"
        check(min(sizes) >= 3, "dot plot: the clusters are not clear")
        return "two_clusters"
    raise AssertionError("dot plot: three clusters")


_WORDS = {
    "tight": "Tightly packed",
    "spread": "Spread out",
    "two_clusters": "Two clusters with a gap between them",
    "outlier": "One cluster with an outlier",
}


def _rows_tight(rng):
    w = rng.choice((2, 3, 4))
    c0 = rng.randint(2, _TICKS - 2 - w)
    rows = {c0 + k: rng.randint(1, 5) for k in range(w + 1)}
    return rows


def _rows_spread(rng):
    ticks = [0, _TICKS]
    t = 0
    while t < _TICKS - 3:
        t += rng.choice((1, 2, 3))
        ticks.append(min(t, _TICKS))
    return {t: rng.randint(1, 3) for t in sorted(set(ticks))}


def _rows_two_clusters(rng):
    a = rng.randint(0, 3)
    left = {a + k: rng.randint(1, 5) for k in range(rng.choice((2, 3)))}
    b = rng.randint(max(left) + 5, _TICKS - 2)
    right = {b + k: rng.randint(1, 5) for k in range(rng.choice((2, 3)) if b + 2 <= _TICKS else 2)}
    return {**left, **right}


def _rows_outlier(rng):
    c0 = rng.randint(0, 4) if rng.random() < 0.5 else rng.randint(7, 10)
    cluster = {c0 + k: rng.randint(1, 5) for k in range(rng.choice((3, 4)))}
    o = (max(cluster) + rng.randint(5, 8)) if c0 < 6 else (min(cluster) - rng.randint(5, 8))
    if not 0 <= o <= _TICKS:
        raise ValueError("outlier off the axis")
    return {**cluster, o: 1}


_BUILD = {"tight": _rows_tight, "spread": _rows_spread, "two_clusters": _rows_two_clusters, "outlier": _rows_outlier}
_DOT_OPTIONS = tuple(_WORDS.values())


def _scene(rng):
    lead, label, who = rng.choice(_QUANTITIES)
    text = lead.format(who=who[0]) if who else lead
    return text, label, rng.choice((0, 5, 10, 20, 30, 40))


def _hard_rows(rng):
    """Two clusters and a unique biggest gap; the left cluster has a small internal gap; one unique most-crowded tick."""
    a = rng.randint(0, 2)
    left = [a, a + 1, a + 3]
    b = rng.randint(left[-1] + 5, _TICKS - 2)
    right = [b, b + 1, b + 2]
    rows = {t: rng.randint(1, 3) for t in left + right}
    peak = rng.choice(left + right)
    rows[peak] = rng.randint(5, 7)
    return rows


def _hard_facts(rows: dict) -> dict:
    f = _features(rows)
    runs = sorted(f["runs"], key=lambda r: -r[2])
    check(len(runs) >= 2 and runs[0][2] > runs[1][2], "dot plot: no unique biggest gap")
    top = max(rows.values())
    modes = [t for t, n in rows.items() if n == top]
    check(len(modes) == 1, "dot plot: no unique most crowded value")
    return dict(
        gap=runs[0][:2], second=runs[1][:2], mode=modes[0], others=[t for t in f["ticks"] if t != modes[0]], rows=rows
    )


def _hard_statement(g, m) -> str:
    return f"The biggest gap is between {g[0]} and {g[1]}, and the most dots are at {m}"


@generator(
    DOTS,
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def ey_dot_plot_spread(rng, tier="easy"):
    """Describe the spread of a dot plot in words that are tied to what the dots do."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    scene, label, low = _scene(rng)
    if tier in ("easy", "medium"):
        kind = rng.choice(("tight", "spread") if tier == "easy" else tuple(_BUILD))
        for _ in attempts(200, "a clear-cut dot plot"):
            try:
                rows = _BUILD[kind](rng)
                total = sum(rows.values())
                if 10 <= total <= 24 and max(rows.values()) <= 12 and _classify(rows) == kind:
                    break
            except Exception:
                continue
        fig = _plot(rows, low, label)
        if tier == "easy":
            ask = rng.choice(
                ("Which words best describe the spread of the dots?", "How would you describe the spread of the data?")
            )
            options = [_WORDS["tight"], _WORDS["spread"], _WORDS["two_clusters"], _WORDS["outlier"]]
        else:
            ask = rng.choice(
                ("Which description fits the dot plot best?", "Which words best describe the shape of the spread?")
            )
            options = list(_DOT_OPTIONS)
        key = _WORDS[kind]
        distractors = mcq(key, [o for o in options if o != key])
        text = f"{scene} {ask}"
        f = _features(rows)
        working = _dot_working(kind, f, low)
        working.append(f"So: {key.lower()}.")
    else:
        for _ in attempts(300, "a dot plot with a unique biggest gap and most crowded value"):
            rows = _hard_rows(rng)
            try:
                facts = _hard_facts(rows)
                if sum(rows.values()) <= 28:
                    break
            except Exception:
                continue
        fig = _plot(rows, low, label)
        g = (facts["gap"][0] + low, facts["gap"][1] + low)
        g2 = (facts["second"][0] + low, facts["second"][1] + low)
        m = facts["mode"] + low
        other_mode = max(facts["others"], key=lambda t: (rows[t], -t)) + low
        key = _hard_statement(g, m)
        wrong = [
            _hard_statement(g2, m),
            _hard_statement(g, other_mode),
            _hard_statement((g[0] + 1, g[1] - 1), m),
        ]
        distractors = mcq(key, wrong)
        text = f"{scene} {rng.choice(('Which statement about the dots is correct?', 'Which statement describes the dot plot correctly?'))}"
        working = [
            f"The longest empty stretch runs between the dots at {g[0]} and {g[1]}.",
            f"The tallest column of dots stands at {m}.",
        ]
    q = type("Q", (), {"figure": fig, "text": text})
    check(_read_dots(q) == key, "the dot plot reader disagrees with the key")
    figures.assert_no_leak(fig, [key, *distractors])
    return dict(
        text=text,
        answer=key,
        answer_text=key,
        answer_latex=f"plain:{key}",
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _rows_of(fig: dict) -> tuple[dict, int]:
    low = fig["range"][0]
    return {d["at"] - low: d["count"] for d in fig["dots"]}, low


def _read_dots(question) -> str:
    rows, low = _rows_of(question.figure)
    if "Which statement" in question.text:
        facts = _hard_facts(rows)
        return _hard_statement((facts["gap"][0] + low, facts["gap"][1] + low), facts["mode"] + low)
    return _WORDS[_classify(rows)]


figures.register_reader(DOTS, _read_dots)


@generator(DOTS_EXAM, level=13, presets={"exam": dict(tier="exam")})
@retry_short
def ey_dot_plot_spread_exam(rng, tier="exam"):
    """Exam chain: the outlier, the top of the main cluster, the size of the gap and the range of all the data."""
    check(tier == "exam", f"unknown tier {tier!r}")
    scene, label, low = _scene(rng)
    for _ in attempts(200, "a cluster with an outlier"):
        try:
            rows = _rows_outlier(rng)
        except ValueError:
            continue
        if _classify(rows) == "outlier" and sum(rows.values()) <= 24:
            break
    f = _features(rows)
    o = _lone(f)
    main = [t for t in f["ticks"] if t != o]
    top, bottom = (max(main), min(main))
    right_side = o > top
    far, near_edge = (o, top) if right_side else (o, bottom)
    fig = _plot(rows, low, label)
    side = "largest" if right_side else "smallest"
    ob, nb = o + low, near_edge + low
    lo_all, hi_all = min(f["ticks"]) + low, max(f["ticks"]) + low
    gap = abs(ob - nb)
    parts = [
        num_part(
            "One value is far from all the others. What is it?",
            ob,
            [nb, ob + 1, hi_all if not right_side else lo_all],
            working=[f"The single dot far from the group is at {ob}."],
        ),
        num_part(
            f"What is the {'largest' if right_side else 'smallest'} value in the main group, not counting the outlier?",
            nb,
            [ob, nb + (1 if right_side else -1), hi_all if right_side else lo_all],
            working=[f"The main group's {side} value is {nb}."],
        ),
        num_part(
            "Hence find the size of the gap between the outlier and the main group.",
            gap,
            [gap - 1, ob + nb, gap + 1],
            follow="Abs(part_a-part_b)",
            working=[f"|{ob} - {nb}| = {gap}."],
        ),
        num_part(
            "Hence find the range of all the data, including the outlier.",
            hi_all - lo_all,
            [hi_all - lo_all - gap, gap, hi_all],
            follow=(f"part_a-{lo_all}" if right_side else f"{hi_all}-part_a"),
            working=[f"Largest value - smallest value = {hi_all} - {lo_all} = {hi_all - lo_all}."],
        ),
    ]
    text = f"{scene} One value is an outlier."
    figures.assert_no_leak(fig, [str(ob), str(nb)])
    return dict(text=f"{text} {rng.choice(_TAILS)}", figure=fig, parts=parts)


def _read_dots_exam(question) -> str:
    rows, low = _rows_of(question.figure)
    check(_classify(rows) == "outlier", "the dot plot must have one outlier")
    f = _features(rows)
    o = _lone(f)
    main = [t for t in f["ticks"] if t != o]
    near_edge = max(main) if o > max(main) else min(main)
    lo, hi = min(f["ticks"]), max(f["ticks"])
    return f"(a) {o + low}; (b) {near_edge + low}; (c) {abs(o - near_edge)}; (d) {hi - lo}"


# (the exam chain of an mcq skill has no registered reader: the sweep cannot hand a labelled reply to the mcq path)

# =============================================================================================== histograms (ACMMM173)

SAMPLES = "ey_sample_variability"
SAMPLES_EXAM = "ey_sample_variability_exam"


def _spread(panel: dict) -> float:
    """The standard deviation of the grouped data (class mid-points)."""
    edges, fr = panel["edges"], panel["frequencies"]
    mids = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
    n = sum(fr)
    mean = sum(m * f for m, f in zip(mids, fr)) / n
    return math.sqrt(sum(f * (m - mean) ** 2 for m, f in zip(mids, fr)) / n)


def _width(panel: dict) -> int:
    """How many classes lie between the lowest and highest occupied class, inclusive."""
    occ = [i for i, f in enumerate(panel["frequencies"]) if f]
    return occ[-1] - occ[0] + 1


def _axis_for(*freq_lists):
    top = max(max(fr) for fr in freq_lists)
    hi = top + (top % 2)
    return hi, 2


def _apportion(weights, total: int) -> list[int]:
    """Whole-number counts in proportion to ``weights`` that add to ``total`` (largest remainders, ties to the left)."""
    s = sum(weights)
    raw = [w * total / s for w in weights]
    out = [int(r) for r in raw]
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - out[i]), i))
    for i in order[: total - sum(out)]:
        out[i] += 1
    return out


def _normal_counts(rng, centre: float, sd: float, n: int, classes: int = 10) -> list[int]:
    weights = [math.exp(-((i + 0.5 - centre) ** 2) / (2 * sd * sd)) for i in range(classes)]
    out = _apportion(weights, n)
    return _jitter(rng, out)


def _jitter(rng, counts: list[int]) -> list[int]:
    """Move one value between neighbouring classes twice, keeping every class non-negative (a sample is never exact)."""
    out = list(counts)
    for _ in range(2):
        i = rng.randrange(len(out) - 1)
        j = i + 1
        src, dst = (i, j) if rng.random() < 0.5 else (j, i)
        if out[src] >= 3:
            out[src] -= 1
            out[dst] += 1
    return out


def _histogram(panels, x_label, alt):
    hi, step = _axis_for(*[p[1] for p in panels])
    built = [
        data.histogram_panel(edges=p[2], frequencies=p[1], freq_max=hi, freq_step=step, minor_lines=True, label=p[0])
        for p in panels
    ]
    return data.histogram(panels=built, alt=alt, x_label=x_label, y_label="Frequency")


_SPREAD_STORIES = (
    (
        "Two classes each sat a test scored out of 10. The histograms show the scores of {n} students from each class.",
        "Score",
        0,
        1,
    ),
    (
        "Two machines each filled {n} bottles. The histograms show the volume in each bottle, in tenths of a litre.",
        "Volume (tenths of a litre)",
        0,
        1,
    ),
)
_SAMPLE_OPTIONS = ("A", "B", "Both are about the same")


def _two_panels(rng, spread_a: float, spread_b: float, n: int, centre_a: float, centre_b: float):
    a = _normal_counts(rng, centre_a, spread_a, n)
    b = _normal_counts(rng, centre_b, spread_b, n)
    return a, b


def _binomial_classes(n_trials: int, p: F, total: int, classes: int = 10) -> list[float]:
    weights = [0.0] * classes
    for x in range(n_trials + 1):
        k = min(classes - 1, x * classes // n_trials)
        weights[k] += math.comb(n_trials, x) * float(p) ** x * float(1 - p) ** (n_trials - x)
    return weights


def _proportions_counts(rng, n_trials: int, p: F, total: int = 50) -> list[int]:
    return _jitter(rng, _apportion(_binomial_classes(n_trials, p, total), total))


_EDGES_PROP = [round(i / 10, 1) for i in range(11)]
_EDGES_10 = list(range(11))
_SOURCES = {
    "uniform": "Uniform: every value about equally likely",
    "normal": "Normal: bell-shaped, most values near the middle",
    "bernoulli": "Bernoulli: only two possible outcomes",
    "skewed": "Skewed: mostly small values, with a long tail to the right",
}


def _source_counts(rng, kind: str, n: int):
    if kind == "uniform":
        base = _apportion([1] * 10, n)
        out = list(base)
        for _ in range(3):
            i, j = rng.sample(range(10), 2)
            if out[i] > base[i] - 2 and out[j] < base[j] + 2:
                out[i] -= 1
                out[j] += 1
        return out, _EDGES_10
    if kind == "normal":
        return _normal_counts(rng, rng.choice((4.5, 5.0, 5.5)), rng.choice((1.4, 1.6)), n), _EDGES_10
    if kind == "skewed":
        w = [math.exp(-i / rng.choice((1.8, 2.2))) for i in range(10)]
        return _apportion(w, n), _EDGES_10
    p = rng.choice((F(3, 10), F(2, 5), F(1, 2), F(3, 5)))
    first = round(n * float(1 - p))
    return [first, 0, 0, 0, 0, n - first], [round(i * 0.2, 1) for i in range(7)]


def _classify_source(panel: dict) -> str:
    fr, n = panel["frequencies"], sum(panel["frequencies"])
    nz = [i for i, f in enumerate(fr) if f]
    if len(nz) == 2 and nz[0] == 0 and nz[-1] == len(fr) - 1 and len(fr) <= 6:
        return "bernoulli"
    check(len(fr) == 10, "histogram: unexpected number of classes")
    top = max(fr)
    peak = fr.index(top)
    if min(fr) >= 1 and top <= 2 * min(fr):
        return "uniform"
    left, right = sum(fr[:5]), sum(fr[5:])
    if 3 <= peak <= 6 and fr[0] <= top / 3 and fr[-1] <= top / 3 and abs(left - right) <= 0.3 * n:
        return "normal"
    if (
        peak <= 1
        and fr[-1] <= top / 4
        and left >= 0.7 * n
        and all(a >= b - 1 for a, b in zip(fr[peak:], fr[peak + 1 :]))
    ):
        return "skewed"
    raise AssertionError("histogram: the source is not clear-cut")


@generator(
    SAMPLES,
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def ey_sample_variability(rng, tier="easy"):
    """Compare the spread of simulated samples, or name the source of a simulated sample."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    if tier == "hard":
        kind = rng.choice(tuple(_SOURCES))
        n = rng.choice((50, 60))
        for _ in attempts(200, "a histogram of a clear-cut source"):
            counts, edges = _source_counts(rng, kind, n)
            panel = dict(edges=edges, frequencies=counts)
            try:
                if _classify_source(panel) == kind:
                    break
            except Exception:
                continue
        fig = _histogram([("", counts, edges)], "Simulated value", "a histogram of simulated values from one source")
        text = rng.choice(
            (
                f"The histogram shows {n} values simulated from one source. Which source is the most likely?",
                f"{n} values were simulated from one source and graphed. Which source do they most likely come from?",
                f"A computer simulated {n} values from one source. The histogram shows them. Which source is the best guess?",
            )
        )
        key = _SOURCES[kind]
        distractors = mcq(key, [v for k, v in _SOURCES.items() if k != kind])
        working = {
            "uniform": ["All the bars are about the same height, so every value was about equally likely."],
            "normal": ["The bars are tallest in the middle and fall away evenly on both sides: a bell shape."],
            "bernoulli": ["There are only two bars, one at each end, so only two outcomes were possible."],
            "skewed": ["The tallest bar is at the left and the bars shrink slowly to the right: a long right tail."],
        }[kind]
        working.append(f"So: {key.split(':')[0].lower()}.")
    elif tier == "easy":
        story, xl, _, _ = rng.choice(_SPREAD_STORIES)
        n = rng.choice((30, 40))
        wide_first = rng.random() < 0.5
        s_wide, s_narrow = rng.choice((2.4, 2.6)), rng.choice((1.0, 1.1))
        c1, c2 = rng.choice((4.0, 5.0, 6.0)), rng.choice((4.0, 5.0, 6.0))
        for _ in attempts(200, "two samples whose spreads differ clearly"):
            a, b = _two_panels(rng, s_wide if wide_first else s_narrow, s_narrow if wide_first else s_wide, n, c1, c2)
            pa, pb = dict(edges=_EDGES_10, frequencies=a), dict(edges=_EDGES_10, frequencies=b)
            wide_ok = (_spread(pa) > 1.4 * _spread(pb)) if wide_first else (_spread(pb) > 1.4 * _spread(pa))
            if wide_ok and (_width(pa) > _width(pb)) == wide_first:
                break
        fig = _histogram(
            [("Sample A", a, _EDGES_10), ("Sample B", b, _EDGES_10)], xl, "two histograms of simulated data"
        )
        text = f"{story.format(n=n)} {rng.choice(('Which sample is more spread out?', 'Which sample varies more?', 'Which of the two samples has the greater spread?'))}"
        key = "A" if wide_first else "B"
        distractors = mcq(key, [o for o in _SAMPLE_OPTIONS if o != key])
        working = [f"Sample {key} has bars across more classes and further from the middle, so it is more spread out."]
    else:
        n_small, n_big = rng.choice(((16, 64), (25, 100), (20, 80)))
        p = F(1, 2)
        big_first = rng.random() < 0.5
        asked_big = rng.random() < 0.5
        for _ in attempts(200, "two displays whose spreads differ clearly"):
            small = _proportions_counts(rng, n_small, p)
            big = _proportions_counts(rng, n_big, p)
            ps, pb = dict(edges=_EDGES_PROP, frequencies=small), dict(edges=_EDGES_PROP, frequencies=big)
            if _spread(ps) > 1.4 * _spread(pb) and _width(ps) > _width(pb):
                break
        first, second = (big, small) if big_first else (small, big)
        fig = _histogram(
            [("Display A", first, _EDGES_PROP), ("Display B", second, _EDGES_PROP)],
            "Sample proportion of heads",
            "two histograms of sample proportions from simulated coin tosses",
        )
        size = n_big if asked_big else n_small
        lead = rng.choice(
            (
                "Each display shows 50 sample proportions of heads from simulated tosses of a fair coin.",
                "A fair coin was tossed in a simulation, and each display graphs 50 sample proportions of heads.",
                "Two simulations of a fair coin each produced 50 sample proportions of heads, graphed below.",
            )
        )
        text = (
            f"{lead} One display used samples of {n_small} tosses and the other used samples of {n_big} tosses. "
            f"Which display used samples of {size} tosses?"
        )
        big_is = "A" if big_first else "B"
        key = big_is if asked_big else ("B" if big_is == "A" else "A")
        distractors = mcq(key, [o for o in _SAMPLE_OPTIONS if o != key])
        working = [
            "Larger samples give sample proportions that vary less, so their display is narrower.",
            f"Display {big_is} is the narrower one, so it used {n_big} tosses; the other used {n_small}.",
        ]
    q = type("Q", (), {"figure": fig, "text": text})
    check(_read_samples(q) == key, "the sample variability reader disagrees with the key")
    figures.assert_no_leak(fig, [key, *distractors])
    return dict(
        text=text,
        answer=key,
        answer_text=key,
        answer_latex=f"plain:{key}",
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _read_samples(question) -> str:
    fig, text = question.figure, question.text
    panels = fig["panels"]
    if len(panels) == 1:
        return _SOURCES[_classify_source(panels[0])]
    spreads = [_spread(p) for p in panels]
    widths = [_width(p) for p in panels]
    wide = 0 if spreads[0] > spreads[1] else 1
    check(max(spreads) > 1.4 * min(spreads), "the two displays do not differ clearly in spread")
    check((widths[0] > widths[1]) == (wide == 0) and widths[0] != widths[1], "the spread and the range disagree")
    if "Which display used" not in text:
        return "AB"[wide]
    sizes = [int(s) for s in re.findall(r"samples of (\d+) tosses", text)]
    check(len(sizes) == 3, "cannot read the sample sizes")
    asked = sizes[2]
    larger = max(sizes[:2])
    narrow = 1 - wide
    return "AB"[narrow] if asked == larger else "AB"[wide]


figures.register_reader(SAMPLES, _read_samples)


@generator(SAMPLES_EXAM, level=13, presets={"exam": dict(tier="exam")})
@retry_short
def ey_sample_variability_exam(rng, tier="exam"):
    """Exam chain: the values in the central band of each display, how many more, and the proportion."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n_small, n_big = rng.choice(((16, 64), (25, 100), (20, 80)))
    p = F(1, 2)
    for _ in attempts(200, "two displays whose central bands differ"):
        small = _proportions_counts(rng, n_small, p)
        big = _proportions_counts(rng, n_big, p)
        band_s, band_b = small[4] + small[5], big[4] + big[5]
        if band_b - band_s >= 4 and band_b < 50:
            break
    fig = _histogram(
        [("Display A", small, _EDGES_PROP), ("Display B", big, _EDGES_PROP)],
        "Sample proportion of heads",
        "two histograms of sample proportions from simulated coin tosses",
    )
    text = (
        f"Each display shows 50 sample proportions of heads from simulated tosses of a fair coin. "
        f"Display A used samples of {n_small} tosses and display B used samples of {n_big} tosses."
    )
    diff = band_b - band_s
    proportion = F(band_b, 50)
    parts = [
        num_part(
            "How many values in display A are from 0.4 up to (but not including) 0.6?",
            band_s,
            [small[4], small[5], band_s + small[3]],
            working=[f"The bars from 0.4 to 0.6 have heights {small[4]} and {small[5]}: {band_s}."],
        ),
        num_part(
            "How many values in display B are from 0.4 up to (but not including) 0.6?",
            band_b,
            [big[4], big[5], band_b + big[3]],
            working=[f"The bars from 0.4 to 0.6 have heights {big[4]} and {big[5]}: {band_b}."],
        ),
        num_part(
            "Hence find how many more values display B has in that band than display A.",
            diff,
            [band_b + band_s, band_s - band_b, diff + 2],
            follow="part_b-part_a",
            working=[f"{band_b} - {band_s} = {diff}."],
        ),
        num_part(
            "Hence find the proportion of display B's 50 values in that band, as a decimal.",
            proportion,
            [F(band_s, 50), F(diff, 50), F(band_b, 100)],
            places=2,
            follow="part_b/50",
            working=[f"{band_b} / 50 = {float(proportion):.2f}."],
        ),
    ]
    figures.assert_no_leak(fig, [str(band_s), str(band_b), str(diff)])
    return dict(text=f"{text} {rng.choice(_TAILS)}", figure=fig, parts=parts)


def _read_samples_exam(question) -> str:
    panels = question.figure["panels"]
    bands = [p["frequencies"][4] + p["frequencies"][5] for p in panels]
    from ..core import round_half_up, stated_precision_string

    prop = stated_precision_string(round_half_up(F(bands[1], 50), 2), 2)
    return f"(a) {bands[0]}; (b) {bands[1]}; (c) {bands[1] - bands[0]}; (d) {prop}"


# (the exam chain of an mcq skill has no registered reader: the sweep cannot hand a labelled reply to the mcq path)
