"""Comparing how long events take from a picture, drawn with the ``durations`` layout of ``compare_picture``.

One topic, ``compare_duration_picture`` (skill ``ey-compare-duration``). Each event is a labelled picture
card beside a row of equal boxes; every box is the same stretch of time, so the row that fills more boxes
takes longer. The card's size is decoration and never says how long an event takes. The key is always
readable from the boxes alone, and the figure reader works from ``question.figure`` and the prompt.

- easy: two events from one start line. Often the bigger card belongs to the shorter event. Equal rows
  appear only under a prompt that asks "Is one longer, or are they the same?".
- medium: three events from one start line, asked for the longest or the shortest time. The event drawn
  first and the one with the biggest card are never the key, so neither habit finds it.
- hard: three events that start at different places, so where a row finishes does not say how long it
  took. Asked for the longest or the shortest time (the row finishing last is never the key), or for the
  list from shortest to longest (the list by finishing place is always one of the wrong ones).

The event names run from very short to long, and a longer name always has more boxes, so the picture and
everyday life agree; the boxes are still the evidence.
"""

from __future__ import annotations

import itertools

from .. import figures
from ..core import attempts, check, generator, pick_distractors
from ..figure_kinds import compare as cmp

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}

_SAME = "They take the same time"

#: Events from the shortest kind to the longest. Each name is a noun, because it is shown as "The ..." in an
#: answer. Two events of one group can take equally long.
_GROUPS = (
    ("blink", "sneeze", "nod"),
    ("clap", "hop", "wave", "jump"),
    ("song", "hand wash", "tidy-up", "count"),
    ("lunch", "bath", "drawing", "reading"),
    ("playtime", "swim", "film"),
    ("sleep",),
)
_UNITS = ("sand", "tick", "dot")

_TEXT = {
    "pair_more": (
        "Each box is the same time. Which takes longer?",
        "Look at the boxes. Which one takes longer?",
        "Each box is one go. Which takes more time?",
    ),
    "pair_less": (
        "Each box is the same time. Which takes less time?",
        "Look at the boxes. Which one is quicker?",
        "Each box is one go. Which takes less time?",
    ),
    "many_more": (
        "Each box is the same time. Which takes the longest?",
        "Look at the boxes. Which takes the most time?",
        "Each box is one go. Which one takes longest?",
    ),
    "many_less": (
        "Each box is the same time. Which takes the shortest time?",
        "Look at the boxes. Which takes the least time?",
        "Each box is one go. Which one is quickest?",
    ),
    "loose_more": (
        "They start at different times. Which takes the longest?",
        "Count the boxes. Which one takes the most time?",
    ),
    "loose_less": (
        "They start at different times. Which takes the shortest time?",
        "Count the boxes. Which one takes the least time?",
    ),
    "same": (
        "Is one longer, or do they take the same time?",
        "Look at the boxes. Is one longer, or are they the same?",
    ),
    "order": (
        "Which list goes from shortest time to longest time?",
        "Count the boxes. Which list is shortest time first?",
        "Put them in order, quickest first. Which list is right?",
    ),
}


def _the(label: str) -> str:
    return f"The {label}"


def _more(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in ("longer", "longest", "more", "most"))


def _extreme(values: dict, more: bool):
    best = max(values.values()) if more else min(values.values())
    hits = [k for k, v in values.items() if v == best]
    return hits[0] if len(hits) == 1 else None


def _events(rng, n: int, equal: bool = False):
    """``n`` events, shortest kind first, each with a card size; two share a group when ``equal``."""
    if equal:
        group = rng.choice([g for g in _GROUPS if len(g) >= 2])
        names = rng.sample(group, 2)
    else:
        names = [rng.choice(_GROUPS[g]) for g in sorted(rng.sample(range(len(_GROUPS)), n))]
    return [{"label": nm, "size": rng.randint(1, 3)} for nm in names]


def _bigger_card(events: list[dict]):
    top = max(e["size"] for e in events)
    hits = [e["label"] for e in events if e["size"] == top]
    return hits[0] if len(hits) == 1 else None


def _draw(rng, tier: str):
    """(events with start and blocks, in drawn order, more, key) where key is None for equal rows, or a list for order."""
    n = 2 if tier == "easy" else 3
    more = rng.random() < 0.5
    order = tier == "hard" and rng.random() < 0.45
    equal = tier == "easy" and rng.random() < 0.15
    for _ in attempts(6000, "a duration comparison"):
        ev = _events(rng, n, equal)
        if equal:
            counts = [rng.randint(2, 7)] * 2
        else:
            counts = sorted(rng.sample(range(1, 9), n))
            if n == 2 and counts[1] - counts[0] < 3:
                continue
        for e, c in zip(ev, counts):
            e["blocks"] = c
        if tier == "hard":
            for e in ev:
                e["start"] = rng.randint(0, 12 - e["blocks"])
            if len({e["start"] for e in ev}) < 2:
                continue
        else:
            for e in ev:
                e["start"] = 0
        rng.shuffle(ev)
        values = {e["label"]: e["blocks"] for e in ev}
        ends = {e["label"]: e["start"] + e["blocks"] for e in ev}
        card = _bigger_card(ev)
        if equal:
            return ev, more, None
        if tier == "easy":
            # often the bigger card is the quicker event, so a bigger picture does not decide it
            key = _extreme(values, more)
            if card == key and rng.random() < 0.6:
                continue  # most easy questions put the bigger card on the event that is NOT the answer
            return ev, more, key
        if tier == "medium":
            key = _extreme(values, more)
            if key in (ev[0]["label"], card):
                continue
            return ev, more, key
        asc = sorted(values, key=values.get)
        if order:
            if sorted(ends, key=ends.get) == asc:
                continue
            return ev, more, asc
        key = _extreme(values, more)
        if _extreme(ends, more) == key or _extreme(ends, more) is None:
            continue
        return ev, more, key
    raise AssertionError("unreachable")


def _read(fig: dict, text: str):
    blocks = cmp.read_durations(fig)
    t = text.lower()
    if "list" in t or "in order" in t:
        return ", ".join(_the(k) for k in sorted(blocks, key=blocks.get))
    if "is one longer" in t:
        return _SAME if len(set(blocks.values())) == 1 else _the(_extreme(blocks, True))
    key = _extreme(blocks, _more(text))
    check(key is not None, "no unique longest or shortest")
    return _the(key)


@generator("compare_duration_picture", level=1, presets=_PRESETS)
def compare_duration_picture(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    ev, more, key = _draw(rng, tier)
    same_start = tier != "hard"
    unit = rng.choice(_UNITS)
    fig = cmp.compare_durations(
        events=ev,
        unit=unit,
        same_start=same_start,
        alt="Events drawn as rows of equal boxes, "
        + ("all starting at the same line." if same_start else "starting at different places."),
    )
    labels = [e["label"] for e in ev]
    blocks = {e["label"]: e["blocks"] for e in ev}
    ends = {e["label"]: e["start"] + e["blocks"] for e in ev}
    pair = len(ev) == 2
    if isinstance(key, list):
        text = rng.choice(_TEXT["order"])
        answer_text = ", ".join(_the(k) for k in key)
        perms = [", ".join(_the(k) for k in p) for p in itertools.permutations(labels)]
        by_end = ", ".join(_the(k) for k in sorted(ends, key=ends.get))
        top_down = ", ".join(_the(k) for k in labels)
        backwards = ", ".join(_the(k) for k in reversed(key))
        cands = [by_end, backwards, top_down, *perms]
        distractors = pick_distractors(answer_text, cands, n=3)
        working = [f"Count the boxes: " + ", ".join(f"{k} {blocks[k]}" for k in labels) + ". Fewest boxes goes first."]
    elif key is None:
        text = rng.choice(_TEXT["same"])
        answer_text = _SAME
        distractors = pick_distractors(answer_text, [_the(k) for k in labels], n=2)
        working = ["Both fill the same number of boxes, so they take the same time."]
    else:
        pool = ("pair_" if pair else "loose_" if tier == "hard" else "many_") + ("more" if more else "less")
        text = rng.choice(_TEXT[pool])
        answer_text = _the(key)
        cands = [_the(k) for k in labels if k != key]
        if tier == "hard":
            far = _extreme(ends, more)
            cands = [_the(far)] + [c for c in cands if c != _the(far)]
        cands.append(_SAME)  # with three or more events it is a fourth option; the counts differ, so it is wrong
        distractors = pick_distractors(answer_text, cands, n=2 if pair else min(len(labels), 3))
        working = ["Count the boxes: " + ", ".join(f"{k} {blocks[k]}" for k in labels) + ". Each box is the same time."]
    check(answer_text not in distractors, "a distractor equals the key")
    check(figures.validate(fig) == [], "the figure is invalid")
    figures.assert_no_leak(fig, [answer_text, *distractors])
    check(_read(fig, text) == answer_text, "the figure does not show the keyed answer")
    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=list(distractors),
        working=working,
        figure=fig,
    )


figures.register_reader("compare_duration_picture", lambda q: _read(q.figure, q.text))
