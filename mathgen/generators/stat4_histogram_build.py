"""Generates `f7-read-a-histogram` questions (AC9M7ST02, AC9M9ST03, Years 7 and 9): read a histogram drawn as a
``histogram`` figure, `answer: number`.

Classes are equal-width intervals that include their lower boundary and exclude their upper one, and the prompt
says so wherever a boundary matters ("20 to less than 30"). The prompt names classes and boundaries, never a
frequency.

Ladder (reasoning escalation, not size):
  easy:   the frequency of one named class; every bar ends on a gridline.
  medium: one of three reads that need more than a single bar: the total number of observations, the combined
          frequency of two named classes, or the lower boundary of the modal class (a boundary read off the
          horizontal axis, not a frequency).
  hard:   a read that turns bars into a question about the data as a whole: how many values are at or above a
          boundary, or below one (a sum of several bars), how many more are in the modal class than in the least
          common class (found by the learner), or what percentage of the data lies at or above a boundary
          (the total is 20, 25, 40 or 50, so the percentage is whole). Bars on the half lines between gridlines.

The figure reader registered for the topic re-derives every answer from ``question.figure`` and the numbers or
keywords in the prompt, never from the keyed answer.
"""

from __future__ import annotations

import re

from mathgen import figures
from mathgen.core import attempts, check, generator, phrase
from mathgen.figure_kinds.data import histogram, histogram_panel
from mathgen.generators._stat_display_common import (
    leaks,
    number_fields,
    pick_number_distractors,
    verify_wrong_everywhere,
)

TOPIC = "stat4_histogram"
TIERS = ("easy", "medium", "hard")

_CONTEXTS = [
    dict(
        subj="students",
        lead="The heights of students in a year level were measured.",
        axis="Height (cm)",
        unit="cm",
        lo=(130, 150),
        width=5,
        cls="had a height from {a} to less than {b} {unit}",
        ge="had a height of {x} {unit} or more",
        lt="had a height below {x} {unit}",
        what="heights",
    ),
    dict(
        subj="runners",
        lead="The times taken by students to run one kilometre were recorded.",
        axis="Time (minutes)",
        unit="minutes",
        lo=(4, 8),
        width=1,
        cls="ran in a time from {a} to less than {b} {unit}",
        ge="ran in a time of {x} {unit} or more",
        lt="ran in a time below {x} {unit}",
        what="times",
    ),
    dict(
        subj="households",
        lead="The electricity used by households in one week was recorded.",
        axis="Electricity used (kWh)",
        unit="kWh",
        lo=(30, 60),
        width=10,
        cls="used from {a} to less than {b} {unit}",
        ge="used {x} {unit} or more",
        lt="used less than {x} {unit}",
        what="amounts used",
    ),
    dict(
        subj="customers",
        lead="The time customers waited to be served at a cafe was recorded.",
        axis="Waiting time (minutes)",
        unit="minutes",
        lo=(0, 4),
        width=2,
        cls="waited from {a} to less than {b} {unit}",
        ge="waited {x} {unit} or more",
        lt="waited less than {x} {unit}",
        what="waiting times",
    ),
    dict(
        subj="apples",
        lead="The masses of apples picked from an orchard were recorded.",
        axis="Mass (grams)",
        unit="grams",
        lo=(100, 140),
        width=20,
        cls="had a mass from {a} to less than {b} {unit}",
        ge="had a mass of {x} {unit} or more",
        lt="had a mass below {x} {unit}",
        what="masses",
    ),
]

_CLASS_ASK = (
    "How many {subj} {cls}?",
    "Use the histogram to find how many {subj} {cls}.",
    "How many of the {subj} {cls}?",
)
_TOTAL_ASK = (
    "How many {subj} are represented in the histogram altogether?",
    "Find the total number of {subj} in the data.",
    "How many {subj} are there in total?",
)
_TWO_ASK = (
    "How many {subj} {cls1}, or {cls2}?",
    "Find the number of {subj} that {cls1} together with those that {cls2}.",
)
_MODAL_ASK = (
    "What is the lower boundary of the modal class?",
    "The modal class is the class with the highest frequency. What is its lower boundary?",
    "Find the lower boundary of the class with the highest frequency.",
)
_GE_ASK = (
    "How many {subj} {ge}?",
    "Use the histogram to find how many {subj} {ge}.",
    "Work out how many {subj} {ge}.",
)
_LT_ASK = (
    "How many {subj} {lt}?",
    "Use the histogram to find how many {subj} {lt}.",
    "Work out how many {subj} {lt}.",
)
_MODAL_LOW_ASK = (
    "How many more {subj} are in the modal class than in the class with the lowest frequency?",
    "Find the difference between the frequency of the modal class and the lowest frequency.",
)
_PCT_ASK = (
    "What percentage of the {subj} {ge}?",
    "Work out the percentage of the {subj} that {ge}.",
)


def _pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def _fmt(v):
    return number_fields(v)[1]


def _build(rng, ctx, tier, *, want_total=None):
    if want_total is not None:
        return _build_total(rng, ctx, want_total)
    width = ctx["width"]
    lo = rng.randint(*ctx["lo"])
    if width >= 5:
        lo -= lo % width  # round numbers on the axis
    n = rng.randint(6, 8)
    edges = [lo + i * width for i in range(n + 1)]
    if tier == "easy":
        step, hi_f = _pick(rng, ((1, 8), (2, 14), (5, 25)))
        unit = step
    elif tier == "medium":
        step, hi_f = _pick(rng, ((2, 14), (5, 30), (2, 16)))
        unit = step
    else:
        step, hi_f = _pick(rng, ((2, 14), (4, 24), (10, 50)))
        unit = step // 2
    top = hi_f // unit
    for _ in attempts(3000, "histogram frequencies"):
        freqs = [unit * rng.randint(1, top) for _ in range(n)]
        if tier == "hard" and sum(1 for f in freqs if f % step) < 3:
            continue
        if want_total is not None and sum(freqs) not in want_total:
            continue
        break
    vmax = max(freqs)
    axis_max = -(-vmax // step) * step
    if axis_max == vmax and tier != "easy":
        axis_max += step
    fig = histogram(
        panels=[
            histogram_panel(
                edges=edges, frequencies=freqs, freq_max=axis_max, freq_step=step, minor_lines=(tier == "hard")
            )
        ],
        alt="a histogram",
        x_label=ctx["axis"],
        y_label="Frequency",
    )
    return fig, edges, freqs, step, unit


def _build_total(rng, ctx, totals):
    """A hard histogram whose bars add to one of ``totals`` (so a percentage of it is whole)."""
    width = ctx["width"]
    lo = rng.randint(*ctx["lo"])
    if width >= 5:
        lo -= lo % width
    n = rng.randint(6, 8)
    edges = [lo + i * width for i in range(n + 1)]
    for _ in attempts(20000, "histogram frequencies with a chosen total"):
        freqs = [rng.randint(1, 12) for _ in range(n)]
        if sum(freqs) in totals and sum(1 for f in freqs if f % 2) >= 3:
            break
    axis_max = -(-max(freqs) // 2) * 2 + 2
    fig = histogram(
        panels=[histogram_panel(edges=edges, frequencies=freqs, freq_max=axis_max, freq_step=2, minor_lines=True)],
        alt="a histogram",
        x_label=ctx["axis"],
        y_label="Frequency",
    )
    return fig, edges, freqs, 2, 1


def _class_phrase(ctx, edges, i, key="cls"):
    return ctx[key].format(a=_fmt(edges[i]), b=_fmt(edges[i + 1]), unit=ctx["unit"])


def _draw(rng, tier):
    ctx = _pick(rng, _CONTEXTS)
    lead = ctx["lead"]
    subj = ctx["subj"]
    if tier == "easy":
        fig, edges, freqs, step, unit = _build(rng, ctx, tier)
        i = rng.randrange(len(freqs))
        key = freqs[i]
        ask = phrase(rng, _CLASS_ASK, subj=subj, cls=_class_phrase(ctx, edges, i))
        cand = [
            key + step,
            key - step,
            freqs[i - 1] if i else freqs[i + 1],
            freqs[(i + 1) % len(freqs)],
            edges[i],
            key + 1,
        ]
        return (
            fig,
            f"{lead} {ask}",
            key,
            cand,
            [f"The bar for the class from {_fmt(edges[i])} to {_fmt(edges[i + 1])} is {key} high."],
        )
    if tier == "medium":
        variant = _pick(rng, ("total", "two", "modal"))
        for _ in attempts(500, "a medium histogram"):
            fig, edges, freqs, step, unit = _build(rng, ctx, tier)
            if variant == "modal" and freqs.count(max(freqs)) != 1:
                continue
            break
        if variant == "total":
            key = sum(freqs)
            ask = phrase(rng, _TOTAL_ASK, subj=subj)
            cand = [len(freqs), max(freqs), key + step, key - step, sum(freqs[1:]), key + unit]
            return fig, f"{lead} {ask}", key, cand, [f"Add the bars: {' + '.join(_fmt(f) for f in freqs)} = {key}."]
        if variant == "two":
            i = rng.randrange(len(freqs) - 1)
            key = freqs[i] + freqs[i + 1]
            ask = phrase(
                rng,
                _TWO_ASK,
                subj=subj,
                cls1=_class_phrase(ctx, edges, i),
                cls2=_class_phrase(ctx, edges, i + 1),
            )
            cand = [freqs[i], freqs[i + 1], abs(freqs[i] - freqs[i + 1]), key + step, key - step]
            return fig, f"{lead} {ask}", key, cand, [f"{_fmt(freqs[i])} + {_fmt(freqs[i + 1])} = {key}."]
        m = freqs.index(max(freqs))
        key = edges[m]
        ask = phrase(rng, _MODAL_ASK)
        cand = [
            freqs[m],
            edges[m + 1],
            edges[m] + ctx["width"] / 2 if ctx["width"] % 2 == 0 else None,
            edges[m + 1] - ctx["width"],
            edges[(m + 1) % len(freqs)],
        ]
        return fig, f"{lead} {ask}", key, cand, [f"The tallest bar starts at {_fmt(key)}."]
    # hard
    variant = _pick(rng, ("ge", "lt", "modal_low", "pct"))
    for _ in attempts(500, "a hard histogram"):
        fig, edges, freqs, step, unit = _build(
            rng, ctx, tier, want_total=(20, 25, 40, 50) if variant == "pct" else None
        )
        if variant == "modal_low" and (freqs.count(max(freqs)) != 1 or freqs.count(min(freqs)) != 1):
            continue
        if variant == "pct" and not any((100 * sum(freqs[k:])) % sum(freqs) == 0 for k in range(2, len(freqs) - 1)):
            continue
        break
    n = len(freqs)
    if variant in ("ge", "lt", "pct"):
        js = list(range(2, n - 1))
        if variant == "pct":
            js = [k for k in js if (100 * sum(freqs[k:])) % sum(freqs) == 0]
            check(bool(js), "no boundary gives a whole percentage")
        j = _pick(rng, js)  # a class edge strictly inside the axis
        x = edges[j]
        above = sum(freqs[j:])
        below = sum(freqs[:j])
        if variant == "lt":
            key = below
            ask = phrase(rng, _LT_ASK, subj=subj, lt=ctx["lt"].format(x=_fmt(x), unit=ctx["unit"]))
            cand = [above, below + freqs[j], below - freqs[j - 1], freqs[j - 1], x, sum(freqs)]
            working = [f"Classes below {_fmt(x)}: {' + '.join(_fmt(f) for f in freqs[:j])} = {key}."]
        elif variant == "ge":
            key = above
            ask = phrase(rng, _GE_ASK, subj=subj, ge=ctx["ge"].format(x=_fmt(x), unit=ctx["unit"]))
            cand = [below, above + freqs[j - 1], above - freqs[j], freqs[j], x, sum(freqs)]
            working = [f"Classes from {_fmt(x)} up: {' + '.join(_fmt(f) for f in freqs[j:])} = {key}."]
        else:
            total = sum(freqs)
            key = 100 * above // total
            ask = phrase(rng, _PCT_ASK, subj=subj, ge=ctx["ge"].format(x=_fmt(x), unit=ctx["unit"]))
            cand = [above, 100 - key, key + 5, key - 5, 100 * below // total, 100 * freqs[j] // total]
            working = [f"{above} out of {total} is {key}%."]
        return fig, f"{lead} {ask}", key, cand, working
    key = max(freqs) - min(freqs)
    ask = phrase(rng, _MODAL_LOW_ASK, subj=subj)
    srt = sorted(freqs)
    cand = [max(freqs) + min(freqs), max(freqs), srt[-2] - srt[0], srt[-1] - srt[1], key + step // 2]
    return (
        fig,
        f"{lead} {ask}",
        key,
        cand,
        [
            f"Highest bar {_fmt(max(freqs))}, lowest bar {_fmt(min(freqs))}.",
            f"{_fmt(max(freqs))} - {_fmt(min(freqs))} = {_fmt(key)}.",
        ],
    )


@generator(TOPIC, level=10, presets={t: {"tier": t} for t in TIERS})
def stat4_histogram(rng, tier="easy"):
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a histogram question"):
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

_NUM = r"(\d+(?:\.\d+)?)"


def _read(question) -> str:
    fig, text = question.figure, question.text
    panel = fig["panels"][0]
    edges, freqs = panel["edges"], panel["frequencies"]
    low = text.lower()

    def klass(x):
        i = next(k for k in range(len(edges) - 1) if edges[k] == float(x))
        return i

    if (
        "lower boundary of the modal class" in low
        or "highest frequency? what" in low
        or "class with the highest frequency" in low
    ):
        return _fmt(edges[freqs.index(max(freqs))])
    if "modal class than" in low or "lowest frequency" in low:
        return _fmt(max(freqs) - min(freqs))
    if "altogether" in low or "total" in low:
        return _fmt(sum(freqs))
    spans = re.findall(rf"from {_NUM} to less than {_NUM}", text)
    if spans:
        return _fmt(sum(freqs[klass(a)] for a, _b in spans))
    x = float(re.findall(_NUM, text)[-1])
    j = next(k for k in range(len(edges)) if edges[k] == x)
    above, total = sum(freqs[j:]), sum(freqs)
    if "percentage" in low:
        return _fmt(100 * above / total)
    if "or more" in low:
        return _fmt(above)
    return _fmt(sum(freqs[:j]))


figures.register_reader(TOPIC, _read)
