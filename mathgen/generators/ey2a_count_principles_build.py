"""Counting scattered counters to 20: how many, one more or one fewer, and how many more make a target.

Descriptors: AC9MFN01 (count forwards and backwards to 20), AC9MFN03 (compare and quantify collections to 20), AC9M1N01.

The counters sit at random cells of a 5 by 4 grid, so the order of counting is never fixed by the layout and the
count is never printed. Difficulty escalates in what the learner does with the count, not in the size of the numbers:

- easy: count every counter (11 to 20 of them).
- medium: count, then say how many there would be with one more, or one fewer.
- hard: two kinds of counter are mixed (plain and ringed); count only the ringed ones, then say how many more are
  needed to make a stated total. Counting every counter instead is the common error and is one of the options.
"""

from __future__ import annotations

from .. import figures
from ..core import generator
from ..figure_kinds.early import counters, counters_total
from ..slots import NAMES
from ._ey2a_kit import SHAPE_WORD, number_question, scatter20, tagged_numbers

_PRESETS = {
    "easy": dict(tier="easy"),
    "medium": dict(tier="medium"),
    "hard": dict(tier="hard"),
}

_EASY_STEMS = (
    "{n} counts the {w}. How many are there?",
    "How many {w} are there?",
    "Count the {w}. How many are there?",
)
_MORE_STEMS = (
    "{n} adds one more. How many {w} are there now?",
    "One more is added. How many {w} are there now?",
)
_FEWER_STEMS = (
    "{n} takes one away. How many {w} are left?",
    "One is taken away. How many {w} are left?",
)
_NEED_STEMS = (
    "Count the ringed ones. How many more make {t} ringed?",
    "{n} needs {t} ringed {w}. How many more ringed ones are needed?",
)


@generator("ey2a_count_principles", level=1, presets=_PRESETS)
def ey2a_count_principles(rng, tier="easy"):
    shape = rng.choice(sorted(SHAPE_WORD))
    word = SHAPE_WORD[shape]
    name = rng.choice(NAMES)
    if tier == "easy":
        n = rng.randint(11, 20)
        dots = scatter20(rng, n)
        fig = counters(layout="scatter", alt="some counters scattered on a plain background", shape=shape, dots=dots)
        answer = n
        text = rng.choice(_EASY_STEMS).format(n=name, w=word)
        cands = [
            (n - 1, "ey2a-count-one-short"),
            (n + 1, "ey2a-count-one-over"),
            (n - 2, "ey2a-count-one-short"),
            (n + 2, "ey2a-count-one-over"),
            (n - 3, "ey2a-count-one-short"),
        ]
        working = [f"Touch each one once while counting. The last number you say is {n}."]
    elif tier == "medium":
        more = rng.random() < 0.5
        n = rng.randint(6, 19) if more else rng.randint(7, 20)
        dots = scatter20(rng, n)
        fig = counters(layout="scatter", alt="some counters scattered on a plain background", shape=shape, dots=dots)
        answer = n + 1 if more else n - 1
        text = rng.choice(_MORE_STEMS if more else _FEWER_STEMS).format(n=name, w=word)
        cands = [
            (n, "ey2a-reports-the-count"),
            (answer + 1 if more else answer - 1, "ey2a-count-one-over" if more else "ey2a-count-one-short"),
            (n - 1 if more else n + 1, "ey2a-moves-the-wrong-way"),
            (answer - 1 if more else answer + 1, "ey2a-count-one-short" if more else "ey2a-count-one-over"),
            (answer + 2 if more else answer - 2, "ey2a-count-one-over" if more else "ey2a-count-one-short"),
        ]
        working = [f"Count the {word}: {n}.", f"One {'more' if more else 'fewer'} is {answer}."]
    elif tier == "hard":
        ringed = rng.randint(4, 12)
        plain = rng.randint(3, 20 - ringed)
        target = rng.randint(ringed + 2, min(20, ringed + 8))
        colours = [2] * ringed + [1] * plain
        rng.shuffle(colours)
        dots = scatter20(rng, ringed + plain, colours)
        fig = counters(
            layout="scatter", alt="plain and ringed counters scattered on a plain background", shape=shape, dots=dots
        )
        answer = target - ringed
        text = rng.choice(_NEED_STEMS).format(n=name, t=target, w=word)
        cands = [
            (target - (ringed + plain), "ey2a-counts-all-kinds"),
            (ringed, "ey2a-reports-the-count"),
            (answer + 1, "ey2a-count-one-over"),
            (answer - 1, "ey2a-count-one-short"),
            (target - plain, "ey2a-counts-wrong-kind"),
            (answer + 2, "ey2a-count-one-over"),
        ]
        working = [f"Count only the ringed ones: {ringed}.", f"{target} - {ringed} = {answer} more are needed."]
    else:
        raise ValueError(f"unknown tier {tier!r}")
    distractors = tagged_numbers(answer, cands, low=1, high=30)
    figures.assert_no_leak(fig, [str(answer)])
    return number_question(text=text, answer=answer, distractors=distractors, working=working, figure=fig)


def _read(q) -> str:
    """Re-derive the answer from the drawn counters and the stem alone."""
    fig, text = q.figure, q.text
    dots = fig["dots"]
    if "ringed" in text:
        ringed = sum(1 for d in dots if d["colour"] == 2)
        target = int(next(t for t in text.replace("?", " ").replace(".", " ").split() if t.isdigit()))
        return str(target - ringed)
    n = counters_total(fig)
    if "one more" in text.lower() or "adds one" in text.lower():
        return str(n + 1)
    if "away" in text.lower():
        return str(n - 1)
    return str(n)


figures.register_reader("ey2a_count_principles", _read)
