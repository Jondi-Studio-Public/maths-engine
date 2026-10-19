"""Generates `f6-read-a-dot-plot` questions (AC9M7ST02, Years 5-7): read a dot plot drawn as a ``dot_plot``
figure, `answer: number`.

The dot plot is the only statement of the data; the prompt names the variable and, where a question needs it, a
value to look at, never a count and never the answer.

Ladder (reasoning escalation, not size):
  easy:   count the dots above one named value.
  medium: one of four summaries read straight off the stacks: the mode (the value with the tallest stack, with a
          unique tallest stack), the range, how many dots lie above a stated value, or the total number of dots.
  hard:   a summary that needs the whole data set rebuilt from the stacks: the median (with a half-step answer
          when the number of dots is even and the two middle values differ) or the mean (built so it is exact).

The figure reader registered for the topic re-derives every answer from ``question.figure`` and the one number
or keyword the prompt carries, never from the keyed answer.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import figures
from mathgen.core import attempts, check, generator, phrase
from mathgen.figure_kinds.data import dot_plot
from mathgen.generators._stat_display_common import (
    leaks,
    number_fields,
    noun,
    pick_number_distractors,
    verify_wrong_everywhere,
)

TOPIC = "stat3_dot_plot"
TIERS = ("easy", "medium", "hard")

# subj: what is counted; has: how a value is said ({q} quantifier, {v} value, {u} unit word);
# step/lo/hi: the number axis; x: axis title.
_CONTEXTS = [
    dict(
        subj="students",
        has="had {q}{v} {u}",
        u1="sibling",
        un="siblings",
        step=1,
        lo=0,
        hi=7,
        x="Number of siblings",
        leads=(
            "Students in a class were asked how many brothers and sisters they have.",
            "A class recorded the number of brothers and sisters each student has.",
        ),
    ),
    dict(
        subj="games",
        has="had {q}{v} {u}",
        u1="goal",
        un="goals",
        step=1,
        lo=0,
        hi=8,
        x="Goals scored",
        leads=(
            "A netball club recorded the goals its team scored in each game of a season.",
            "A soccer team kept a record of the goals it scored in every game this season.",
        ),
    ),
    dict(
        subj="students",
        has="read {q}{v} {u}",
        u1="book",
        un="books",
        step=1,
        lo=0,
        hi=9,
        x="Books read",
        leads=(
            "Students recorded how many books they read last month.",
            "A librarian counted the books each student in a class borrowed and read last month.",
        ),
    ),
    dict(
        subj="students",
        has="scored {q}{v}",
        u1="",
        un="",
        step=1,
        lo=3,
        hi=10,
        x="Test score",
        leads=(
            "A class sat a spelling test and the scores were recorded.",
            "Each student's score on a short spelling test was recorded.",
        ),
    ),
    dict(
        subj="students",
        has="took {q}{v} minutes to get to school",
        u1="minutes",
        un="minutes",
        step=5,
        lo=0,
        hi=40,
        x="Travel time (minutes)",
        leads=(
            "Students were asked how long it takes them to get to school.",
            "A class recorded the time each student takes to travel to school.",
        ),
    ),
    dict(
        subj="plants",
        has="grew {q}{v} {u}",
        u1="centimetre",
        un="centimetres",
        step=2,
        lo=0,
        hi=16,
        x="Growth (cm)",
        leads=(
            "A class measured how much each of their bean plants grew in two weeks.",
            "Students recorded the growth of their bean plants over a fortnight.",
        ),
    ),
]

_COUNT_ASK = (
    "How many {subj} {has}?",
    "Use the dot plot to find how many {subj} {has}.",
    "Count the dots: how many {subj} {has}?",
)
_TALLEST_ASK = (
    "How many {subj} are in the tallest stack of dots?",
    "Find the number of dots in the tallest stack.",
    "How many dots are in the highest stack?",
)
_ABOVE_MEAN_ASK = (
    "How many {subj} are above the mean of the data?",
    "Work out the mean, then count how many values are above it.",
    "How many of the data values are greater than the mean?",
)
_MODE_ASK = (
    "What is the mode of the data?",
    "Find the mode of the data shown.",
    "Which value is the mode?",
)
_RANGE_ASK = (
    "What is the range of the data?",
    "Find the range of the data shown.",
    "Work out the range of the data.",
)
_MORE_ASK = (
    "How many {subj} {has}?",
    "Use the dot plot to find how many {subj} {has}.",
    "Count how many {subj} {has}.",
)
_TOTAL_ASK = (
    "How many {subj} are shown on the dot plot altogether?",
    "How many data values are there in total?",
    "Find the total number of {subj} in the data.",
)
_MEDIAN_ASK = (
    "What is the median of the data?",
    "Find the median of the data shown.",
    "Work out the median.",
)
_MEAN_ASK = (
    "What is the mean of the data?",
    "Find the mean of the data shown.",
    "Work out the mean.",
)


def _pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def _fmt(v):
    return number_fields(v)[1]


def _valphrase(ctx, v, q=""):
    unit = ctx["u1"] if (v == 1 and not q) or (v == 1 and ctx["u1"] and q) else ctx["un"]
    return ctx["has"].format(q=q, v=_fmt(v), u=unit).replace("  ", " ")


def _data(rng, ctx, tier):
    """``(counts, values)``: per-tick counts and the sorted list of data values."""
    step, lo, hi = ctx["step"], ctx["lo"], ctx["hi"]
    ticks = [lo + i * step for i in range((hi - lo) // step + 1)]
    span = rng.randint(5, min(8, len(ticks)))
    start = rng.randint(0, len(ticks) - span)
    used = ticks[start : start + span]
    n_lo, n_hi = {"easy": (12, 20), "medium": (14, 22), "hard": (13, 25)}[tier]
    for _ in attempts(2000, "dot plot data"):
        counts = {v: rng.randint(0, 9 if tier == "easy" else 7) for v in used}
        counts[used[0]] = max(counts[used[0]], 1)
        counts[used[-1]] = max(counts[used[-1]], 1)
        n = sum(counts.values())
        if not n_lo <= n <= n_hi:
            continue
        return counts


def _sorted_values(counts):
    return sorted(v for v, c in counts.items() for _ in range(c))


def _median(vals):
    n = len(vals)
    return vals[n // 2] if n % 2 else Fraction(vals[n // 2 - 1] + vals[n // 2], 2)


def _figure(ctx, counts):
    # axis from one tick before the lowest data to one after the highest, within the context's own axis
    step = ctx["step"]
    used = [v for v, c in counts.items() if c]
    lo = max(ctx["lo"], min(used) - step)
    hi = min(ctx["hi"], max(used) + step)
    return dot_plot(
        low=lo,
        high=hi,
        step=step,
        dots={v: c for v, c in counts.items() if c},
        alt="a dot plot",
        x_label=ctx["x"],
    )


def _draw(rng, tier):
    ctx = _pick(rng, _CONTEXTS)
    lead = _pick(rng, ctx["leads"])
    step = ctx["step"]
    if tier == "hard":
        variant = _pick(rng, ("median", "mean", "abovemean"))
    elif tier == "medium":
        variant = _pick(rng, ("mode", "range", "more", "total", "fewer"))
    else:
        variant = _pick(rng, ("count", "count", "tallest"))
    for _ in attempts(500, "a dot plot of the right kind"):
        counts = _data(rng, ctx, tier)
        vals = _sorted_values(counts)
        n = len(vals)
        used = [v for v, c in counts.items() if c]
        top = max(counts.values())
        modes = [v for v, c in counts.items() if c == top]
        if variant == "mode" and (len(modes) != 1 or top < 3):
            continue
        if variant == "range" and max(used) - min(used) < 3 * step:
            continue
        if variant == "median":
            med = _median(vals)
            if len(set(vals)) < 5:
                continue
            if n % 2 == 0 and vals[n // 2 - 1] == vals[n // 2] and rng.random() < 0.5:
                continue  # prefer a genuinely averaged median half the time
        if variant in ("mean", "abovemean"):
            total = sum(vals)
            if total % n or len(set(vals)) < 5 or modes == [Fraction(total, n)]:
                continue
        break
    fig = _figure(ctx, counts)
    mean_unw = Fraction(sum(used), len(used))
    mid_range = Fraction(min(used) + max(used), 2)
    if variant == "count":
        v = _pick(rng, [u for u in used if counts[u] >= 1])
        key = counts[v]
        ask = phrase(rng, _COUNT_ASK, subj=ctx["subj"], has=_valphrase(ctx, v))
        nb = [counts.get(v + d, 0) for d in (-step, step)]
        cand = [*nb, v, key + 1, key - 1, sum(1 for x in vals if x >= v)]
        working = [f"The stack above {_fmt(v)} has {key} {noun(key, 'dot')}."]
    elif variant == "tallest":
        key = top
        ask = phrase(rng, _TALLEST_ASK, subj=ctx["subj"])
        cand = [modes[0], len(used), n, top + 1, top - 1, min(counts[u] for u in used)]
        working = [f"The tallest stack is above {_fmt(modes[0])} and has {top} {noun(top, 'dot')}."]
    elif variant == "fewer":
        v = _pick(rng, [u for u in used if min(used) < u < max(used)])
        key = sum(1 for x in vals if x < v)
        ask = phrase(rng, _MORE_ASK, subj=ctx["subj"], has=_valphrase(ctx, v, "fewer than "))
        cand = [
            sum(1 for x in vals if x <= v),
            sum(1 for x in vals if x > v),
            sum(x for x in vals if x < v),
            key + 1,
            key - 1,
        ]
        working = [f"Dots to the left of {_fmt(v)}: {key}."]
    elif variant == "abovemean":
        mean = Fraction(total, n)
        key = sum(1 for x in vals if x > mean)
        ask = phrase(rng, _ABOVE_MEAN_ASK, subj=ctx["subj"])
        cand = [
            sum(1 for x in vals if x >= mean),
            sum(1 for x in vals if x < mean),
            mean,
            sum(1 for x in vals if x > _median(vals)),
            sum(1 for x in used if x > mean),
            key + 1,
            key - 1,
        ]
        working = [
            f"Sum of all values = {total}; there are {n} values, so the mean is {total} / {n} = {_fmt(mean)}.",
            f"Values above {_fmt(mean)}: {key}.",
        ]
    elif variant == "mode":
        key = modes[0]
        ask = phrase(rng, _MODE_ASK)
        cand = [top, mid_range, _median(vals), Fraction(sum(vals), n), max(used)]
        working = [f"The tallest stack is above {_fmt(key)}, so the mode is {_fmt(key)}."]
    elif variant == "range":
        key = max(used) - min(used)
        ask = phrase(rng, _RANGE_ASK)
        axis_lo, axis_hi = fig["range"]
        cand = [max(used), axis_hi - axis_lo, key + step, key - step, max(counts.values()) - min(counts.values())]
        working = [
            f"Highest value {_fmt(max(used))}, lowest value {_fmt(min(used))}.",
            f"{_fmt(max(used))} - {_fmt(min(used))} = {_fmt(key)}.",
        ]
    elif variant == "more":
        cands = [u for u in used if min(used) < u < max(used)]
        v = _pick(rng, cands)
        key = sum(1 for x in vals if x > v)
        ask = phrase(rng, _MORE_ASK, subj=ctx["subj"], has=_valphrase(ctx, v, "more than "))
        cand = [
            sum(1 for x in vals if x >= v),
            sum(1 for x in vals if x < v),
            sum(x for x in vals if x > v),
            key + 1,
            key - 1,
        ]
        working = [f"Dots to the right of {_fmt(v)}: {key}."]
    elif variant == "total":
        key = n
        ask = phrase(rng, _TOTAL_ASK, subj=ctx["subj"])
        cand = [len(used), sum(vals), n + 1, n - 1, top]
        working = [f"Add the dots in every stack: {' + '.join(str(counts[u]) for u in used)} = {n}."]
    elif variant == "median":
        key = med
        ask = phrase(rng, _MEDIAN_ASK)
        i = n // 2
        cand = [
            modes[0],
            mid_range,
            Fraction(sum(vals), n),
            _median(used),
            vals[i - 1] if n % 2 else vals[i],
            vals[i + 1] if n % 2 else vals[i - 1],
        ]
        working = [
            f"There are {n} values. In order, the middle is {'value ' + str((n + 1) // 2) if n % 2 else 'between values ' + str(n // 2) + ' and ' + str(n // 2 + 1)}.",
            f"The median is {_fmt(key)}.",
        ]
    else:
        key = Fraction(total, n)
        ask = phrase(rng, _MEAN_ASK)
        cand = [modes[0], _median(vals), mean_unw, mid_range, Fraction(total, len(used))]
        working = [f"Sum of all values = {total}; there are {n} values.", f"{total} / {n} = {_fmt(key)}."]
    return fig, f"{lead} {ask}", key, cand, working


@generator(TOPIC, level=8, presets={t: {"tier": t} for t in TIERS})
def stat3_dot_plot(rng, tier="easy"):
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a dot plot question"):
        fig, text, key, cand, working = _draw(rng, tier)
        answer, answer_text, answer_latex = number_fields(key)
        distractors = pick_number_distractors(TOPIC, key, cand)
        verify_wrong_everywhere(TOPIC, key, distractors)
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


# ---------------------------------------------------------------- the figure reader


def _read(question) -> str:
    fig, text = question.figure, question.text
    counts = {d["at"]: d["count"] for d in fig["dots"]}
    vals = _sorted_values(counts)
    low = text.lower()
    if "tallest stack" in low or "highest stack" in low:
        return _fmt(max(counts.values()))
    if "mean" in low and ("above" in low or "greater" in low):
        mean = Fraction(sum(vals), len(vals))
        return _fmt(sum(1 for x in vals if x > mean))
    if "fewer than" in low:
        v = Fraction(re.findall(r"\d+", text)[-1])
        return _fmt(sum(1 for x in vals if x < v))
    if "mode" in low:
        top = max(counts.values())
        winners = [v for v, c in counts.items() if c == top]
        check(len(winners) == 1, "the figure has no single tallest stack")
        return _fmt(winners[0])
    if "range" in low:
        return _fmt(max(counts) - min(counts))
    if "median" in low:
        return _fmt(_median(vals))
    if "mean" in low:
        return _fmt(Fraction(sum(vals), len(vals)))
    if "altogether" in low or "total" in low:
        return _fmt(len(vals))
    v = Fraction(re.findall(r"\d+", text)[-1])
    if "more than" in low:
        return _fmt(sum(1 for x in vals if x > v))
    return _fmt(counts.get(int(v), 0))


figures.register_reader(TOPIC, _read)
