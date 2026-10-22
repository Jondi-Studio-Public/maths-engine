"""Generates `f2-read-a-picture-graph` questions (AC9M2ST02, Year 2): read a picture graph drawn as a
``picture_graph`` figure, `answer: number`.

The graph is the only statement of the data: the prompt names the categories asked about and never the
counts. Every question needs the key ("Each symbol = k"), so reading the symbols as if each stood for one is
a named error.

Ladder (reasoning escalation, not size):
  easy:   read one row. The key is 1 or 2 and every row is whole symbols.
  medium: compare two named rows ("how many more ... than ..."). The key is 2, 5 or 10 and a row can end in a
          half symbol (only when the key is even, so a half is a whole number of the thing counted).
  hard:   three decisions the easy and medium tiers never ask for: add every row (the total), find the most
          and the least popular row for yourself and give the difference, or work backwards from a count to the
          number of symbols a new row needs.

The figure reader registered for the topic re-derives each answer from ``question.figure`` and the category
names in the prompt, never from the keyed answer.
"""

from __future__ import annotations

import math
import re

from mathgen import figures
from mathgen.core import attempts, check, generator, phrase
from mathgen.generators._stat_display_common import (
    leaks,
    mentioned_in_order,
    number_fields,
    noun,
    pick_number_distractors,
    verify_wrong_everywhere,
)

TOPIC = "stat1_picture_graph"
TIERS = ("easy", "medium", "hard")

# unit, singular unit, verb used in "how many <unit> <verb> <category>", lead sentences, category pool
_CONTEXTS = [
    dict(
        unit="students",
        one="student",
        verb="chose",
        leads=(
            "Students chose a favourite pet.",
            "A class voted for pets.",
            "Students picked their favourite pet.",
        ),
        cats=("Dog", "Cat", "Bird", "Fish", "Rabbit", "Horse", "Turtle"),
    ),
    dict(
        unit="students",
        one="student",
        verb="chose",
        leads=(
            "Students chose a favourite fruit.",
            "A class voted for fruits.",
            "Students picked their favourite fruit.",
        ),
        cats=("Apple", "Banana", "Orange", "Grapes", "Pear", "Mango", "Melon"),
    ),
    dict(
        unit="students",
        one="student",
        verb="chose",
        leads=(
            "Students chose a favourite sport.",
            "A school voted for sports.",
            "Students picked their favourite sport.",
        ),
        cats=("Soccer", "Netball", "Cricket", "Tennis", "Swimming", "Hockey", "Rugby"),
    ),
    dict(
        unit="days",
        one="day",
        verb="were",
        leads=(
            "A class recorded the weather.",
            "Each day's weather was recorded.",
            "Weather was noted daily.",
        ),
        cats=("Sunny", "Cloudy", "Rainy", "Windy", "Foggy", "Stormy", "Hot"),
    ),
]

_SYMBOLS = ("circle", "square", "triangle", "star", "heart", "diamond")

_EASY_ASK = (
    "How many {unit} {verb} {c}?",
    "Use the graph: how many {unit} {verb} {c}?",
    "Read the graph. How many {unit} {verb} {c}?",
    "From the graph, how many {unit} {verb} {c}?",
)
_MORE_ASK = (
    "How many more {unit} {verb} {a} than {b}?",
    "How many more {unit} {verb} {a} than {verb} {b}?",
    "Find how many more {unit} {verb} {a} than {b}.",
)
_FEWER_ASK = (
    "How many fewer {unit} {verb} {b} than {a}?",
    "Find how many fewer {unit} {verb} {b} than {a}.",
)
_TOTAL_ASK = (
    "How many {unit} are there altogether?",
    "Find the total number of {unit} on the graph.",
    "How many {unit} are there in total?",
    "How many {unit} are shown in total?",
)
_EXTREME_ASK = (
    "Find the difference between the biggest group and smallest.",
    "What is the biggest group minus the smallest?",
    "Find the biggest group minus the smallest group.",
)
_INVERSE_ASK = (
    "{c} has {n} {unit}. How many symbols is that?",
    "A new row, {c}, has {n} {unit}. How many symbols?",
    "Another group, {c}, has {n} {unit}. How many symbols?",
)


def _pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def _fmt(v):
    return number_fields(v)[1]


def _build_graph(rng, ctx, *, rows: int, each: int, half_ok: bool, sym_lo=1, sym_hi=8, distinct=False):
    cats = list(ctx["cats"])
    rng.shuffle(cats)
    cats = cats[:rows]
    for _ in attempts(200, "picture-graph rows"):
        syms = []
        for _r in range(rows):
            n = rng.randint(sym_lo, sym_hi)
            if half_ok and rng.random() < 0.4 and n < sym_hi:
                n += 0.5
            syms.append(n)
        if distinct and len(set(syms)) < rows:
            continue
        if half_ok and not any(abs(s - round(s)) > 1e-9 for s in syms):
            continue  # the medium tier should actually use a half symbol
        break
    symbol = _pick(rng, _SYMBOLS)
    fig = figures.check(
        {
            "kind": "picture_graph",
            "alt": "a picture graph",
            "symbol": symbol,
            "key": {"each": each, "unit": ctx["one"] if each == 1 else ctx["unit"]},
            "rows": [{"label": c, "symbols": s} for c, s in zip(cats, syms)],
        }
    )
    return fig, cats, syms


def _counts(fig):
    k = fig["key"]["each"]
    return {r["label"]: r["symbols"] * k for r in fig["rows"]}


def _half_as_whole(syms, each):
    return [math.ceil(s) * each for s in syms]


@generator(TOPIC, level=3, presets={t: {"tier": t} for t in TIERS})
def stat1_picture_graph(rng, tier="easy"):
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a picture graph without a key/answer collision"):
        out = _draw(rng, tier)
        fig, text, key, candidates, working = out
        answer, answer_text, answer_latex = number_fields(key)
        distractors = pick_number_distractors(TOPIC, key, candidates)
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


def _draw(rng, tier):
    ctx = _pick(rng, _CONTEXTS)
    lead = _pick(rng, ctx["leads"])
    unit, verb = ctx["unit"], ctx["verb"]
    if tier == "easy":
        each = _pick(rng, (1, 1, 2, 2, 2))
        fig, cats, syms = _build_graph(rng, ctx, rows=rng.randint(4, 5), each=each, half_ok=False)
        i = rng.randrange(len(cats))
        key = syms[i] * each
        ask = phrase(rng, _EASY_ASK, unit=unit, verb=verb, c=cats[i])
        others = [s * each for j, s in enumerate(syms) if j != i]
        cand = [syms[i] if each > 1 else None, key + each, key - each, *others, key + 1]
        working = [
            f"The {cats[i]} row has {_fmt(syms[i])} {noun(syms[i], 'symbol')} and each symbol is {each}.",
            f"{_fmt(syms[i])} × {each} = {_fmt(key)}.",
        ]
        return fig, f"{lead} {ask}", key, cand, working
    if tier == "medium":
        each = _pick(rng, (2, 5, 10))
        fig, cats, syms = _build_graph(
            rng, ctx, rows=rng.randint(4, 5), each=each, half_ok=(each % 2 == 0), sym_lo=2, sym_hi=8
        )
        order = sorted(range(len(cats)), key=lambda j: syms[j])
        a, b = rng.sample(range(len(cats)), 2)
        if syms[a] == syms[b]:
            a, b = order[-1], order[0]
        if syms[a] < syms[b]:
            a, b = b, a
        key = (syms[a] - syms[b]) * each
        fewer = rng.random() < 0.35
        if fewer:
            ask = phrase(rng, _FEWER_ASK, unit=unit, verb=verb, a=cats[a], b=cats[b])
        else:
            ask = phrase(rng, _MORE_ASK, unit=unit, verb=verb, a=cats[a], b=cats[b])
        whole = _half_as_whole(syms, each)
        cand = [
            syms[a] * each + syms[b] * each,
            syms[a] - syms[b],
            whole[a] - whole[b],
            syms[a] * each,
            key + each,
            key - each,
            syms[b] * each,
        ]
        working = [
            f"{cats[a]}: {_fmt(syms[a])} {noun(syms[a], 'symbol')} × {each} = {_fmt(syms[a] * each)}. "
            f"{cats[b]}: {_fmt(syms[b])} {noun(syms[b], 'symbol')} × {each} = {_fmt(syms[b] * each)}.",
            f"{_fmt(syms[a] * each)} - {_fmt(syms[b] * each)} = {_fmt(key)}.",
        ]
        return fig, f"{lead} {ask}", key, cand, working
    # hard
    each = _pick(rng, (2, 5, 10))
    variant = _pick(rng, ("total", "extreme", "inverse"))
    rows = rng.randint(5, 6)
    fig, cats, syms = _build_graph(
        rng, ctx, rows=rows, each=each, half_ok=(each % 2 == 0), sym_lo=1, sym_hi=9, distinct=(variant == "extreme")
    )
    if variant == "total":
        key = sum(s * each for s in syms)
        ask = phrase(rng, _TOTAL_ASK, unit=unit)
        cand = [
            sum(syms),
            sum(whole for whole in _half_as_whole(syms, each)),
            sum(math.floor(s) * each for s in syms),
            key + each,
            key - each,
            key + each // 2,
        ]
        working = [
            f"Symbols: {', '.join(_fmt(s) for s in syms)}. Total symbols = {_fmt(sum(syms))}.",
            f"{_fmt(sum(syms))} × {each} = {_fmt(key)}.",
        ]
        return fig, f"{lead} {ask}", key, cand, working
    if variant == "extreme":
        hi, lo = max(syms), min(syms)
        key = (hi - lo) * each
        ask = phrase(rng, _EXTREME_ASK, unit=unit)
        srt = sorted(syms)
        cand = [
            (hi + lo) * each,
            hi - lo,
            (srt[-2] - lo) * each,
            (hi - srt[1]) * each,
            hi * each,
            key + each,
        ]
        working = [
            f"Most: {_fmt(hi)} {noun(hi, 'symbol')}. Least: {_fmt(lo)} {noun(lo, 'symbol')}.",
            f"({_fmt(hi)} - {_fmt(lo)}) × {each} = {_fmt(key)}.",
        ]
        return fig, f"{lead} {ask}", key, cand, working
    # inverse
    free = [c for c in ctx["cats"] if c not in cats]
    new_c = _pick(rng, free)
    m = rng.randint(2, 9)
    n = m * each
    ask = phrase(rng, _INVERSE_ASK, c=new_c, n=n, unit=unit)
    key = m
    cand = [n, m + 1, m - 1, n * each, n // each + each, m * 2]
    working = [f"One symbol is {each}, so {n} needs {n} / {each} = {m} {noun(m, 'symbol')}."]
    return fig, f"{lead} {ask}", key, cand, working


# ---------------------------------------------------------------- the figure reader


def _read(question) -> str:
    fig, text = question.figure, question.text
    each = fig["key"]["each"]
    rows = {r["label"]: r["symbols"] for r in fig["rows"]}
    named = mentioned_in_order(list(rows), text)
    low = text.lower()
    if re.search(r"how many symbols|whole symbols|does the row need", low):
        n = int(re.search(r"(\d+) (?:students|days)", text).group(1))
        return _fmt(n / each)
    if re.search(r"altogether|total|in total", low) and not named:
        return _fmt(sum(rows.values()) * each)
    if "most popular" in low or "biggest group" in low:
        return _fmt((max(rows.values()) - min(rows.values())) * each)
    if len(named) == 1:
        return _fmt(rows[named[0]] * each)
    check(len(named) == 2, "the prompt must name one or two rows")
    return _fmt(abs(rows[named[0]] - rows[named[1]]) * each)


figures.register_reader(TOPIC, _read)
