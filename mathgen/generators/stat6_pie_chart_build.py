"""Generates `f6-read-a-pie-chart` questions (Years 5-6): read a pie chart drawn as a ``pie_chart`` figure,
`answer: number`.

The circle is drawn cut into equal slices (4, 5, 6, 8, 10 or 12 of them), with a radius at every slice boundary,
so a sector is a counted number of slices out of the whole circle, never an angle to be measured. The prompt
states a count of people (a whole multiple of the number of slices), never a sector's size.

Ladder (reasoning escalation, not size):
  easy:   a total is stated; find how many chose one named sector (count its slices, scale to the total).
  medium: a total is stated; how many more chose one sector than another, or how many chose one of two sectors.
  hard:   work backwards or sideways from a count: the count of one sector is stated and the learner finds the
          total, or the count of another sector, or a total is stated and the learner counts everyone outside one
          sector.

The figure reader registered for the topic re-derives every answer from ``question.figure`` and the numbers and
names in the prompt, never from the keyed answer.
"""

from __future__ import annotations

import re

from mathgen import figures
from mathgen.core import attempts, check, generator, phrase
from mathgen.figure_kinds.data import pie_chart
from mathgen.generators._stat_display_common import (
    mentioned_in_order,
    number_fields,
    pick_number_distractors,
    verify_wrong_everywhere,
)

TOPIC = "stat6_pie_chart"
TIERS = ("easy", "medium", "hard")

_CONTEXTS = [
    dict(
        subj="students",
        lead="A survey asked students to choose a favourite pet.",
        cats=("Dogs", "Cats", "Birds", "Fish", "Rabbits", "Horses"),
        verb="chose",
    ),
    dict(
        subj="students",
        lead="A survey asked students how they travel to school.",
        cats=("Walk", "Bus", "Car", "Bike", "Train", "Scooter"),
        verb="chose",
    ),
    dict(
        subj="people",
        lead="People at a market were asked to name their favourite fruit.",
        cats=("Apples", "Bananas", "Oranges", "Grapes", "Pears", "Mangoes"),
        verb="chose",
    ),
    dict(
        subj="students",
        lead="A school asked students to vote for a sport to play at the carnival.",
        cats=("Soccer", "Netball", "Cricket", "Tennis", "Hockey", "Rugby"),
        verb="chose",
    ),
]


def _pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def _fmt(v):
    return number_fields(v)[1]


def _partition(rng, parts, n_sectors):
    """``n_sectors`` positive whole numbers adding to ``parts``, not all the same size."""
    for _ in range(200):
        cuts = sorted(rng.sample(range(1, parts), n_sectors - 1))
        sizes = [b - a for a, b in zip([0, *cuts], [*cuts, parts])]
        if len(set(sizes)) > 1:
            return sizes
    return None


def _draw(rng, tier):
    ctx = _pick(rng, _CONTEXTS)
    subj, verb = ctx["subj"], ctx["verb"]
    parts_pool = {"easy": (4, 5, 8, 10), "medium": (5, 6, 8, 10, 12), "hard": (5, 6, 8, 10, 12)}[tier]
    for _ in attempts(500, "a pie chart"):
        parts = _pick(rng, parts_pool)
        n_sectors = rng.randint(3, min(5, parts - 1))
        sizes = _partition(rng, parts, n_sectors)
        if sizes is None:
            continue
        break
    cats = list(ctx["cats"])
    rng.shuffle(cats)
    cats = cats[:n_sectors]
    sectors = [{"label": c, "parts": k} for c, k in zip(cats, sizes)]
    fig = pie_chart(parts=parts, sectors=sectors, alt="a pie chart")
    unit = rng.randint(2, 9) if parts < 10 else rng.randint(2, 8)  # people per slice
    total = parts * unit
    lead = ctx["lead"]
    pairs = [(a, b) for a in range(n_sectors) for b in range(n_sectors) if a != b and sizes[a] != sizes[b]]
    i, j = _pick(rng, pairs)
    ki, kj = sizes[i], sizes[j]
    if tier == "easy":
        key = ki * unit
        ask = phrase(
            rng,
            (
                "{total} {subj} answered. The pie chart shows the results. How many {verb} {c}?",
                "The pie chart shows the answers of {total} {subj}. How many {verb} {c}?",
                "{total} {subj} were surveyed. Use the pie chart to find how many {verb} {c}.",
            ),
            total=total,
            subj=subj,
            verb=verb,
            c=cats[i],
        )
        cand = [ki, total // ki if total % ki == 0 else None, total - key, key + unit, key - unit, unit]
        working = [
            f"The circle has {parts} equal slices and {cats[i]} has {ki}.",
            f"Each slice is {total} / {parts} = {unit}, so {ki} x {unit} = {key}.",
        ]
        return fig, f"{lead} {ask}", key, cand, working
    if tier == "medium":
        if rng.random() < 0.5:
            hi, lo = (i, j) if ki > kj else (j, i)
            key = (sizes[hi] - sizes[lo]) * unit
            ask = phrase(
                rng,
                (
                    "{total} {subj} answered. How many more {verb} {a} than {b}?",
                    "The pie chart shows the answers of {total} {subj}. How many more {verb} {a} than {b}?",
                    "{total} {subj} were surveyed. Find how many more {verb} {a} than {b}.",
                ),
                total=total,
                subj=subj,
                verb=verb,
                a=cats[hi],
                b=cats[lo],
            )
            cand = [(sizes[hi] + sizes[lo]) * unit, sizes[hi] - sizes[lo], sizes[hi] * unit, key + unit, key - unit]
            working = [
                f"{cats[hi]} has {sizes[hi]} slices and {cats[lo]} has {sizes[lo]}.",
                f"({sizes[hi]} - {sizes[lo]}) x {unit} = {key}.",
            ]
        else:
            key = (ki + kj) * unit
            ask = phrase(
                rng,
                (
                    "{total} {subj} answered. How many {verb} {a} or {b}?",
                    "The pie chart shows the answers of {total} {subj}. How many {verb} {a} or {b}?",
                    "{total} {subj} were surveyed. Find how many {verb} {a} or {b}.",
                ),
                total=total,
                subj=subj,
                verb=verb,
                a=cats[i],
                b=cats[j],
            )
            cand = [abs(ki - kj) * unit, ki + kj, ki * unit, kj * unit, key + unit, key - unit]
            working = [f"{cats[i]} has {ki} slices and {cats[j]} has {kj}.", f"({ki} + {kj}) x {unit} = {key}."]
        return fig, f"{lead} {ask}", key, cand, working
    variant = _pick(rng, ("total", "other", "outside"))
    if variant == "total":
        n = ki * unit
        key = total
        ask = phrase(
            rng,
            (
                "{n} {subj} {verb} {c}. How many {subj} were surveyed altogether?",
                "{n} of the {subj} surveyed {verb} {c}. How many {subj} answered altogether?",
            ),
            n=n,
            subj=subj,
            verb=verb,
            c=cats[i],
        )
        cand = [n * ki, n * parts, n + unit, parts, n * (parts - ki), total - unit]
        working = [
            f"{cats[i]} has {ki} of the {parts} slices, so {ki} slices = {n}.",
            f"One slice = {unit}, so {parts} slices = {key}.",
        ]
        return fig, f"{lead} {ask}", key, cand, working
    if variant == "other":
        n = ki * unit
        key = kj * unit
        ask = phrase(
            rng,
            (
                "{n} {subj} {verb} {a}. Using the pie chart, how many {verb} {b}?",
                "In the same survey, {n} {subj} {verb} {a}. Using the pie chart, how many {verb} {b}?",
            ),
            n=n,
            subj=subj,
            verb=verb,
            a=cats[i],
            b=cats[j],
        )
        cand = [n * kj, kj, n + (kj - ki) * 1, n * ki // kj if (n * ki) % kj == 0 else None, abs(kj - ki) * unit, n]
        working = [
            f"{cats[i]} has {ki} slices = {n}, so one slice = {unit}.",
            f"{cats[j]} has {kj} slices: {kj} x {unit} = {key}.",
        ]
        return fig, f"{lead} {ask}", key, cand, working
    key = (parts - ki) * unit
    ask = phrase(
        rng,
        (
            "{total} {subj} answered. How many did not choose {c}?",
            "The pie chart shows the answers of {total} {subj}. How many {subj} did not choose {c}?",
        ),
        total=total,
        subj=subj,
        c=cats[i],
    )
    cand = [ki * unit, ki, parts - ki, key + unit, key - unit, total - parts]
    working = [
        f"{cats[i]} has {ki} of {parts} slices, so the other {parts - ki} slices did not choose it.",
        f"{parts - ki} x {unit} = {key}.",
    ]
    return fig, f"{lead} {ask}", key, cand, working


@generator(TOPIC, level=6, presets={t: {"tier": t} for t in TIERS})
def stat6_pie_chart(rng, tier="easy"):
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a pie chart question"):
        fig, text, key, cand, working = _draw(rng, tier)
        answer, answer_text, answer_latex = number_fields(key)
        distractors = pick_number_distractors(TOPIC, key, cand)
        verify_wrong_everywhere(TOPIC, key, distractors)
        # a pie chart draws category names only; the sweep's leak check is the usual one
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
    parts = fig["parts"]
    size = {s["label"]: s["parts"] for s in fig["sectors"]}
    named = mentioned_in_order(list(size), text)
    nums = [int(n) for n in re.findall(r"\b(\d+)\b", text)]
    low = text.lower()
    check(len(nums) == 1, "the prompt states exactly one count")
    n = nums[0]
    if "altogether" in low:
        return _fmt(n * parts // size[named[0]])
    if "using the pie chart, how many" in low:
        a, b = named[0], named[1]
        return _fmt(n * size[b] // size[a])
    unit = n // parts
    if "did not choose" in low:
        return _fmt((parts - size[named[0]]) * unit)
    if "more" in low:
        return _fmt(abs(size[named[0]] - size[named[1]]) * unit)
    if " or " in low and len(named) == 2:
        return _fmt((size[named[0]] + size[named[1]]) * unit)
    return _fmt(size[named[0]] * unit)


figures.register_reader(TOPIC, _read)
