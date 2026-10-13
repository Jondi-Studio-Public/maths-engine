"""Box plot generators: `f10-read-a-box-plot` (topic ``stat5_box_plot_read``, `answer: number`) and
`f10-compare-box-plots` (topic ``stat5_box_plot_compare``, `answer: mcq`), both AC9M10ST02 (Year 10), drawn as a
``box_plot`` figure.

The prompt never states a value that is read off the plot. A plot's five numbers sit on axis ticks (or on the
unlabelled half ticks of a finer axis), a whisker end is never an outlier, and an outlier lies beyond the usual
1.5 x IQR fences, as the ``box_plot`` validator requires.

Reading one plot (``stat5_box_plot_read``), reasoning escalation:
  easy:   read one of the five numbers (minimum, lower quartile, median, upper quartile, maximum).
  medium: combine two reads: the range, or the interquartile range. Values may sit on half ticks.
  hard:   a plot with outliers, and one of: the range of all the data (the outlier is data), the interquartile
          range with an outlier pulling the eye, or what the quartiles say about a number of values - about what
          percentage lie below a stated value that is one of the plot's quartiles or median, or about how many of
          a stated number of values lie above it.

Comparing two parallel plots (``stat5_box_plot_compare``), reasoning escalation:
  easy:   which plot has the higher median. Never a tie.
  medium: which plot has the larger interquartile range, or the larger range; built so the plots disagree on the
          OTHER measure half the time, so the overall spread of a plot does not give the answer.
  hard:   which of four statements is supported by the plots. The statements compare medians, interquartile
          ranges, ranges, and a median against the other plot's quartile; exactly one is true.

Each topic registers a figure reader that works from ``question.figure`` alone (plus the keywords, numbers and
plot labels in the prompt and the offered statements).
"""

from __future__ import annotations

import re

from mathgen import figures
from mathgen.core import attempts, check, generator, phrase
from mathgen.figure_kinds.data import box_plot
from mathgen.grading.typed import mcq_options
from mathgen.generators._stat_display_common import (
    leaks,
    number_fields,
    pick_number_distractors,
    verify_wrong_everywhere,
)

READ = "stat5_box_plot_read"
COMPARE = "stat5_box_plot_compare"
TIERS = ("easy", "medium", "hard")

_CONTEXTS = [
    dict(
        subj="students",
        lead="The box plot shows the marks of students in a test.",
        noun="mark",
        nouns="marks",
        axis="Test mark",
        lo=0,
        hi=100,
        step=10,
    ),
    dict(
        subj="students",
        lead="The box plot shows the time students took to finish a puzzle.",
        noun="time",
        nouns="times",
        axis="Time (minutes)",
        lo=0,
        hi=60,
        step=5,
    ),
    dict(
        subj="plants",
        lead="The box plot shows the heights of plants in a garden bed.",
        noun="height",
        nouns="heights",
        axis="Height (cm)",
        lo=0,
        hi=60,
        step=5,
    ),
    dict(
        subj="pumpkins",
        lead="The box plot shows the masses of pumpkins at a market.",
        noun="mass",
        nouns="masses",
        axis="Mass (kg)",
        lo=0,
        hi=24,
        step=2,
    ),
]

_COMPARE_PAIRS = (
    ("Class A", "Class B"),
    ("Year 9", "Year 10"),
    ("Group X", "Group Y"),
    ("Team Red", "Team Blue"),
)

_FIVE = ("minimum", "lower quartile", "median", "upper quartile", "maximum")
_KEYS = {"minimum": "min", "lower quartile": "q1", "median": "median", "upper quartile": "q3", "maximum": "max"}


def _pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def _fmt(v):
    return number_fields(v)[1]


def _five_numbers(rng, ctx, *, unit, outliers=0, spread=None):
    """Five numbers (multiples of ``unit``) strictly increasing, plus ``outliers`` outlier values."""
    top = (ctx["hi"] - ctx["lo"]) // unit
    for _ in attempts(2000, "box plot numbers"):
        gaps = [rng.randint(1, 4) for _ in range(4)]
        if spread == "wide":
            gaps = [rng.randint(2, 5) for _ in range(4)]
        base = rng.randint(1, max(1, top - sum(gaps) - 1))
        idx = [base]
        for g in gaps:
            idx.append(idx[-1] + g)
        mn, q1, med, q3, mx = (ctx["lo"] + i * unit for i in idx)
        outs = []
        iqr = q3 - q1
        hi_fence = q3 + 1.5 * iqr
        lo_fence = q1 - 1.5 * iqr
        if mx > hi_fence or mn < lo_fence:
            continue  # a whisker end is never an outlier
        if outliers:
            cands = []
            v = mx + unit
            while v <= ctx["hi"]:
                if v > hi_fence:
                    cands.append(v)
                v += unit
            low = []
            v = mn - unit
            while v >= ctx["lo"]:
                if v < lo_fence:
                    low.append(v)
                v -= unit
            pool = cands + low
            if len(pool) < 1:
                continue
            outs = [rng.choice(pool)]
            if outliers == 2 and len(pool) > 1:
                other = [p for p in pool if p != outs[0] and (p > mx) == (outs[0] > mx)]
                if other:
                    outs.append(rng.choice(other))
            outs = sorted(set(outs))
        if mx > ctx["hi"]:
            continue
        return dict(min=mn, q1=q1, median=med, q3=q3, max=mx, outliers=outs)


def _fig(ctx, plots, *, step, minor):
    return box_plot(
        low=ctx["lo"],
        high=ctx["hi"],
        step=step,
        minor_ticks=minor,
        plots=plots,
        alt="a box plot" if len(plots) == 1 else "two box plots on one axis",
        x_label=ctx["axis"],
    )


def _extent(p):
    pts = [p["min"], p["max"], *p["outliers"]]
    return min(pts), max(pts)


# ======================================================================= reading one plot


def _draw_read(rng, tier):
    ctx = _pick(rng, _CONTEXTS)
    step = ctx["step"]
    lead = _pick(rng, (ctx["lead"], f"Here is a box plot of the {ctx['nouns']} of the {ctx['subj']}."))
    noun = ctx["noun"]
    if tier == "easy":
        p = _five_numbers(rng, ctx, unit=step)
        fig = _fig(ctx, [dict(p, label="")], step=step, minor=False)
        name = _pick(rng, _FIVE)
        key = p[_KEYS[name]]
        ask = phrase(
            rng,
            (
                "What is the {name} {noun}?",
                "Read the {name} {noun} from the box plot.",
                "Find the {name} {noun}.",
            ),
            name=name,
            noun=noun,
        )
        others = [p[k] for k in ("min", "q1", "median", "q3", "max") if p[k] != key]
        cand = [key + step, key - step, *others]
        return fig, f"{lead} {ask}", key, cand, [f"The {name} is {_fmt(key)}."]
    minor = tier == "medium" and step % 2 == 0 and rng.random() < 0.5
    unit = step // 2 if minor else step
    if tier == "medium":
        p = _five_numbers(rng, ctx, unit=unit, spread="wide")
        fig = _fig(ctx, [dict(p, label="")], step=step, minor=minor)
        if rng.random() < 0.5:
            key = p["max"] - p["min"]
            ask = phrase(
                rng,
                (
                    "What is the range of the {nouns}?",
                    "Find the range of the {nouns}.",
                    "Work out the range of the {nouns} shown.",
                ),
                nouns=ctx["nouns"],
            )
            iqr = p["q3"] - p["q1"]
            cand = [p["max"], iqr, p["max"] - p["q1"], p["q3"] - p["min"], key + unit, key - unit]
            working = [f"Range = {_fmt(p['max'])} - {_fmt(p['min'])} = {_fmt(key)}."]
        else:
            key = p["q3"] - p["q1"]
            ask = phrase(
                rng,
                (
                    "What is the interquartile range?",
                    "Find the interquartile range of the {nouns}.",
                    "Work out the interquartile range (IQR).",
                ),
                nouns=ctx["nouns"],
            )
            cand = [p["max"] - p["min"], p["q3"], p["q1"], p["median"] - p["q1"], key + unit, key - unit]
            working = [f"IQR = {_fmt(p['q3'])} - {_fmt(p['q1'])} = {_fmt(key)}."]
        return fig, f"{lead} {ask}", key, cand, working
    # hard
    variant = _pick(rng, ("range_out", "iqr_out", "pct_below", "count_above"))
    minor = False
    if variant in ("range_out", "iqr_out"):
        p = _five_numbers(rng, ctx, unit=step, outliers=_pick(rng, (1, 1, 2)))
    else:
        p = _five_numbers(rng, ctx, unit=step, outliers=_pick(rng, (0, 1)))
    fig = _fig(ctx, [dict(p, label="")], step=step, minor=minor)
    lo_all, hi_all = _extent(p)
    if variant == "range_out":
        key = hi_all - lo_all
        ask = phrase(
            rng,
            (
                "What is the range of all the {nouns}, including any outlier?",
                "The outlier is a real value. What is the range of all the {nouns}?",
                "Find the range of all the {nouns} shown, outliers included.",
            ),
            nouns=ctx["nouns"],
        )
        cand = [p["max"] - p["min"], hi_all, p["q3"] - p["q1"], key + step, key - step, p["outliers"][0]]
        working = [f"Lowest value {_fmt(lo_all)}, highest value {_fmt(hi_all)}.", f"Range = {_fmt(key)}."]
        return fig, f"{lead} {ask}", key, cand, working
    if variant == "iqr_out":
        key = p["q3"] - p["q1"]
        ask = phrase(
            rng,
            (
                "What is the interquartile range of the {nouns}?",
                "Find the interquartile range (IQR). The outlier does not change the box.",
                "Work out the IQR of the {nouns}.",
            ),
            nouns=ctx["nouns"],
        )
        cand = [hi_all - lo_all, p["max"] - p["min"], p["q3"] - p["median"], hi_all - p["q1"], key + step, key - step]
        working = [f"IQR = {_fmt(p['q3'])} - {_fmt(p['q1'])} = {_fmt(key)}."]
        return fig, f"{lead} {ask}", key, cand, working
    name = _pick(rng, ("lower quartile", "median", "upper quartile"))
    x = p[_KEYS[name]]
    pct = {"lower quartile": 25, "median": 50, "upper quartile": 75}[name]
    if variant == "pct_below":
        key = pct
        ask = phrase(
            rng,
            (
                "A {subj1} had a {noun} of {x}. About what percentage of the {subj} had a lower {noun}?",
                "About what percentage of the {subj} had a {noun} below {x}?",
                "Roughly what percentage of the {subj} had a {noun} less than {x}?",
            ),
            subj=ctx["subj"],
            subj1=ctx["subj"].rstrip("s"),
            noun=noun,
            x=_fmt(x),
        )
        cand = [100 - pct, 50 if pct != 50 else 25, 75 if pct != 75 else 25, 100, 10, 90]
        working = [f"{_fmt(x)} is the {name}, so about {pct}% of the values are below it."]
        return fig, f"{lead} {ask}", key, cand, working
    n = _pick(rng, (40, 60, 80, 100, 120))
    key = n * (100 - pct) // 100
    ask = phrase(
        rng,
        (
            "There were {n} {subj}. About how many had a {noun} above {x}?",
            "Of the {n} {subj}, about how many had a {noun} greater than {x}?",
            "About how many of the {n} {subj} had a {noun} of more than {x}?",
        ),
        n=n,
        subj=ctx["subj"],
        noun=noun,
        x=_fmt(x),
    )
    cand = [n * pct // 100, n // 2, n // 4, n * 3 // 4, n - key + 10]
    working = [f"{_fmt(x)} is the {name}, so about {100 - pct}% of the {n} values are above it."]
    return fig, f"{lead} {ask}", key, cand, working


@generator(READ, level=13, presets={t: {"tier": t} for t in TIERS})
def stat5_box_plot_read(rng, tier="easy"):
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a box plot question"):
        fig, text, key, cand, working = _draw_read(rng, tier)
        answer, answer_text, answer_latex = number_fields(key)
        distractors = pick_number_distractors(READ, key, cand)
        verify_wrong_everywhere(READ, key, distractors)
        if leaks(fig, [answer_text, *distractors]):
            continue
        return dict(
            text=text,
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_latex,
            distractors=distractors,
            working=working,
            figure=fig,
        )


_PCT = {"lower quartile": 25, "median": 50, "upper quartile": 75}


def _read_one(question) -> str:
    fig, text = question.figure, question.text
    p = fig["plots"][0]
    low = text.lower()
    pts = [p["min"], p["max"], *p["outliers"]]
    if "interquartile" in low or " iqr" in low:
        return _fmt(p["q3"] - p["q1"])
    if "range" in low:
        if "including" in low or "included" in low or "outlier is a real value" in low:
            return _fmt(max(pts) - min(pts))
        return _fmt(p["max"] - p["min"])
    m = re.search(r"(?:about|of the) (\d+) ", low) or re.search(r"there were (\d+)", low)
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if "how many" in low:
        n, x = float(nums[0]), float(nums[-1])
        pct = _percent_at(p, x)
        return _fmt(n * (100 - pct) / 100)
    if "percentage" in low:
        x = float(nums[-1])
        return _fmt(_percent_at(p, x))
    for name in sorted(_FIVE, key=lambda s: -len(s)):
        if name in low:
            return _fmt(p[_KEYS[name]])
    raise ValueError("unreadable box plot prompt")


def _percent_at(p, x):
    for key, pct in (("q1", 25), ("median", 50), ("q3", 75)):
        if p[key] == x:
            return pct
    raise ValueError("the stated value is not a quartile or the median")


figures.register_reader(READ, _read_one)


# ======================================================================= comparing two plots


def _pair(rng, ctx, *, unit, want):
    """Two distinct five-number summaries meeting ``want`` (a predicate on (a, b))."""
    for _ in attempts(3000, "a pair of box plots"):
        a = _five_numbers(rng, ctx, unit=unit)
        b = _five_numbers(rng, ctx, unit=unit)
        if want(a, b):
            return a, b


def _opt(names, i):
    """The option naming plot ``i``: the group and where its plot is drawn (so it is not a bare figure label)."""
    return f"{names[i]} (the {'upper' if i == 0 else 'lower'} plot)"


def _iqr(p):
    return p["q3"] - p["q1"]


def _rng(p):
    return p["max"] - p["min"]


def _strict(a, b):
    return a["median"] != b["median"] and _iqr(a) != _iqr(b) and _rng(a) != _rng(b)


def _statement(kind, x, y):
    return {
        "m": f"{x} has the higher median.",
        "i": f"{x} has the larger interquartile range.",
        "r": f"{x} has the larger range.",
        "c": f"The median of {x} is higher than the upper quartile of {y}.",
        "d": f"The lower quartile of {x} is higher than the median of {y}.",
    }[kind]


def _truth(kind, x, y, plots):
    p, q = plots[x], plots[y]
    return {
        "m": p["median"] > q["median"],
        "i": _iqr(p) > _iqr(q),
        "r": _rng(p) > _rng(q),
        "c": p["median"] > q["q3"],
        "d": p["q1"] > q["median"],
    }[kind]


def _draw_compare(rng, tier):
    ctx = _pick(rng, _CONTEXTS)
    names = _pick(rng, _COMPARE_PAIRS)
    step = ctx["step"]
    lead = phrase(
        rng,
        (
            "The box plots compare the {nouns} of two groups of {subj}.",
            "Two groups of {subj} are shown as box plots of their {nouns}.",
            "These parallel box plots show the {nouns} for two groups of {subj}.",
        ),
        nouns=ctx["nouns"],
        subj=ctx["subj"],
    )
    if tier == "easy":
        a, b = _pair(rng, ctx, unit=step, want=lambda a, b: a["median"] != b["median"])
        plots = {names[0]: a, names[1]: b}
        key = _opt(names, 0 if a["median"] > b["median"] else 1)
        ask = phrase(
            rng,
            (
                "Which group has the higher median?",
                "Which plot shows the higher median {noun}?",
                "Which group has the greater median {noun}?",
                "Compare the medians. Which group's median {noun} is higher?",
            ),
            noun=ctx["noun"],
        )
        opts = [_opt(names, 0), _opt(names, 1), "The medians are the same"]
        working = [f"{names[0]} median {_fmt(a['median'])}, {names[1]} median {_fmt(b['median'])}."]
    elif tier == "medium":
        measure = _pick(rng, ("interquartile range", "range"))
        if measure == "interquartile range":
            f, g = _iqr, _rng
        else:
            f, g = _rng, _iqr
        # the other measure favours the OTHER plot half of the time
        conflict = rng.random() < 0.5

        def want(a, b):
            if f(a) == f(b) or g(a) == g(b):
                return False
            return (f(a) > f(b)) != (g(a) > g(b)) if conflict else (f(a) > f(b)) == (g(a) > g(b))

        a, b = _pair(rng, ctx, unit=step, want=want)
        plots = {names[0]: a, names[1]: b}
        key = _opt(names, 0 if f(a) > f(b) else 1)
        ask = phrase(
            rng,
            (
                "Which group has the larger {m}?",
                "Which plot has the greater {m}?",
                "Which group's {nouns} have the larger {m}?",
                "Compare the {m}s. Which group has the larger one?",
            ),
            m=measure,
            nouns=ctx["nouns"],
        )
        opts = [_opt(names, 0), _opt(names, 1), f"The {measure}s are the same"]
        working = [
            f"{names[0]}: {measure} {_fmt(f(a))}. {names[1]}: {measure} {_fmt(f(b))}.",
        ]
    else:
        a, b = _pair(rng, ctx, unit=step, want=_strict)
        plots = {names[0]: a, names[1]: b}
        x, y = names
        statements = {}
        for kind in "mirc d".replace(" ", ""):
            for p, q in ((x, y), (y, x)):
                statements[(kind, p)] = (_statement(kind, p, q), _truth(kind, p, q, plots))
        true = [v[0] for k, v in statements.items() if v[1]]
        false = [v[0] for k, v in statements.items() if not v[1]]
        # at most one statement of each kind among the four options
        for _ in attempts(200, "four statements"):
            t = _pick(rng, true)
            tk = next(k for k, v in statements.items() if v[0] == t)[0]
            pool = [s for s in false if next(k for k, v in statements.items() if v[0] == s)[0] != tk]
            rng.shuffle(pool)
            chosen, kinds = [], set()
            for s in pool:
                sk = next(k for k, v in statements.items() if v[0] == s)[0]
                if sk not in kinds:
                    kinds.add(sk)
                    chosen.append(s)
                if len(chosen) == 3:
                    break
            if len(chosen) == 3:
                break
        key = t
        opts = [t, *chosen]
        ask = _pick(
            rng,
            (
                "Which statement is supported by the box plots?",
                "Which of these statements is true?",
                "Which statement about the two groups is correct?",
                "Which statement can be made from the box plots?",
            ),
        )
        working = [
            f"{x}: median {_fmt(a['median'])}, IQR {_fmt(_iqr(a))}, range {_fmt(_rng(a))}.",
            f"{y}: median {_fmt(b['median'])}, IQR {_fmt(_iqr(b))}, range {_fmt(_rng(b))}.",
        ]
    fig = _fig(ctx, [dict(plots[n], label=n) for n in names], step=step, minor=False)
    return fig, f"{lead} {ask}", key, [o for o in opts if o != key], working


@generator(COMPARE, level=13, presets={t: {"tier": t} for t in TIERS})
def stat5_box_plot_compare(rng, tier="easy"):
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a comparison"):
        fig, text, key, others, working = _draw_compare(rng, tier)
        check(len(set(others)) == len(others) and key not in others, "options must be distinct")
        if tier == "hard":
            check(len(others) == 3, "four statements are offered")
        else:
            check(len(others) == 2, "three options are offered")
        figures.assert_no_leak(fig, [key, *others])
        return dict(
            text=text,
            answer=key,
            answer_text=key,
            answer_latex=f"plain:{key}",
            distractors=others,
            working=working,
            figure=fig,
        )


_STMT_RE = re.compile(
    r"^(?:(?P<x1>.+?) has the (?P<m>higher median|larger interquartile range|larger range)"
    r"|The (?P<a>median|lower quartile) of (?P<x2>.+?) is higher than the (?P<b>upper quartile|median) of (?P<y2>.+?))\.$"
)


def _eval_statement(plots, s):
    m = _STMT_RE.match(s)
    check(m is not None, "unreadable statement")
    if m["x1"]:
        x = m["x1"]
        y = next(n for n in plots if n != x)
        p, q = plots[x], plots[y]
        return {
            "higher median": p["median"] > q["median"],
            "larger interquartile range": _iqr(p) > _iqr(q),
            "larger range": _rng(p) > _rng(q),
        }[m["m"]]
    p, q = plots[m["x2"]], plots[m["y2"]]
    a = p["median"] if m["a"] == "median" else p["q1"]
    b = q["q3"] if m["b"] == "upper quartile" else q["median"]
    return a > b


def _read_compare(question) -> str:
    plots = {p["label"]: p for p in question.figure["plots"]}
    text = question.text.lower()
    names = list(plots)
    if "statement" in text or "true" in text or "correct" in text:
        options = mcq_options(question)
        true = [o for o in options if _eval_statement(plots, o)]
        check(len(true) == 1, "exactly one statement must be true")
        return true[0]
    a, b = (plots[n] for n in names)
    if "median" in text:
        measure = lambda p: p["median"]  # noqa: E731
    elif "interquartile" in text:
        measure = _iqr
    else:
        measure = _rng
    check(measure(a) != measure(b), "the plots tie on the measure asked")
    return _opt(names, 0 if measure(a) > measure(b) else 1)


figures.register_reader(COMPARE, _read_compare)
