"""Base-ten blocks: ``base_ten_read`` (skill ``f10-base-ten-blocks-number``), drawn with ``base_ten``.

The learner says the number the blocks show. Easy shows rods and cubes (two-digit numbers); medium
adds flats and, half the time, a place with no blocks (a zero to write: 304, 450); hard leaves the
blocks UNREGROUPED, with ten or more cubes (or ten or more rods), so the counts of each block are
not the digits of the number. The reader re-derives the number from the three counts in the figure.
"""

from __future__ import annotations

from .. import figures
from ..core import attempts, check, generator, pick_distractors
from ..figure_kinds.early import base_ten, base_ten_value
from ..slots import NAMES

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}

_TEXT = {
    "easy": (
        "{n} builds a number with blocks. What number is it?",
        "{n} counts the blocks. What number is shown?",
        "What number do these blocks show? {n} wants to know.",
    ),
    "medium": (
        "{n} builds a number with flats, rods and cubes. What number is it?",
        "{n} looks at the flats, rods and cubes. What number do they show?",
        "What number do these blocks make? {n} wants to know.",
    ),
    "hard": (
        "{n} did not swap any ones for tens. What number do the blocks show?",
        "{n} has blocks that are not grouped in the tidy way. What number do they show?",
        "Some blocks could be swapped for bigger ones. What number do {n}'s blocks show?",
    ),
}


def _count(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _counts(rng, tier: str):
    if tier == "easy":
        return 0, rng.randint(1, 9), rng.randint(0, 9)
    if tier == "medium":
        h = rng.randint(1, 5)
        t, o = rng.randint(0, 9), rng.randint(0, 9)
        if rng.random() < 0.5:
            if rng.random() < 0.5:
                t = 0
            else:
                o = 0
            if t == 0 and o == 0:
                o = rng.randint(1, 9)
        return h, t, o
    if rng.random() < 0.6:
        return rng.randint(0, 3), rng.randint(1, 8), rng.randint(10, 16)
    return rng.randint(0, 3), rng.randint(10, 14), rng.randint(0, 9)


def _distractors(h: int, t: int, o: int, value: int):
    unregrouped = o >= 10 or t >= 10
    cands = [h + t + o]  # counts the blocks
    if not unregrouped:
        cands += [int(f"{h or ''}{t}{o}"), int(f"{h or ''}{o}{t}"), value + 10, value - 10]
        if h and (t == 0 or o == 0):
            cands.append(int(f"{h}{t or ''}{o or ''}"))  # leaves out the place with no blocks
    else:
        cands += [100 * h + 10 * t, 100 * h + 10 * t + o % 10, 100 * h + 10 * (t % 10) + o, value + 10, value - 10]
        cands.append(100 * h + 10 * min(t, 9) + min(o, 9))
    return [str(c) for c in cands if c > 0]


@generator("base_ten_read", level=1, presets=_PRESETS)
def base_ten_read(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    h, t, o = _counts(rng, tier)
    fig = base_ten(hundreds=h, tens=t, ones=o, alt="base-ten blocks: flats, rods and cubes")
    value = 100 * h + 10 * t + o
    answer_text = str(value)
    text = rng.choice(_TEXT[tier]).format(n=rng.choice(NAMES))
    distractors = pick_distractors(answer_text, _distractors(h, t, o, value))
    check(len(distractors) >= 2, "too few distractors")
    figures.assert_no_leak(fig, [answer_text, *distractors])
    check(_read(fig) == answer_text, "the figure does not show the keyed number")
    return dict(
        text=text,
        answer=value,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=[
            f"{_count(h, 'hundred')}, {_count(t, 'ten')} and {_count(o, 'one')}: {100 * h} + {10 * t} + {o} = {value}."
        ],
        figure=fig,
    )


def _read(fig: dict) -> str:
    return str(base_ten_value(fig))


figures.register_reader("base_ten_read", lambda q: _read(q.figure))
