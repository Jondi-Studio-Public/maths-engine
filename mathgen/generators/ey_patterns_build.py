"""Repeating patterns from a picture: strips of shapes, numbers, letters and actions (``pattern_strip``).

Three topics, each with an easy, a medium and a hard preset. The strip is the only place the pattern appears:
no prompt names a cell, and every option is described in words a child hears ("an open circle", "the number 5",
"a clap", "Row B").

- ``ey_pattern_continue`` (``continue-patterns``): what comes next, or what fills the empty box. Easy is an
  AB pattern; medium adds AAB, ABB and ABC and a gap in the middle; hard varies shape AND fill together, so a
  child who watches one thing picks a look-alike.
- ``ey_pattern_match`` (``continue-patterns``, ``f10-pattern-strip-repeating-unit``): the pattern is said in
  words ("clap, stamp, clap, stamp") and the child picks the row, drawn with other things, that is the same
  pattern. Medium and hard add three-long units; hard says the model with actions and hides a row of actions
  with a different pattern as the look-alike.
- ``ey_pattern_unit`` (``f10-pattern-strip-repeating-unit``): easy counts the things in the repeating part
  (``number``); medium says what the 10th or so item is (the strip is numbered but stops short); hard finds the
  one cell that breaks the pattern.

Each topic registers a figure reader that re-derives the answer from ``question.figure`` (and, for the match
topic, from the pattern said in the prompt), never from the key.
"""

from __future__ import annotations

from .. import figures
from ..core import attempts, check, generator, pick_distractors
from ..figure_kinds import pattern as pat
from ..figure_kinds.pattern import (
    ACTIONS,
    SHAPES,
    blank_cell,
    cell_from_key,
    describe_key,
    mark_cell,
    pattern_strip,
    read_gap,
    read_mistake,
    read_nth,
    row_keys,
    smallest_period,
    structure,
)
from ..slots import NAMES

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}
_LETTERS = "DEFGHJKLMNPQRSTWXYZ"  # never A, B or C, which label the rows of a match figure
_DIGITS = "123456789"

# A family is what a pattern is made of. "mixed" varies shape and fill together.
_FAMILIES = ("shapes", "fills", "numbers", "letters", "actions")
_PLAIN = ("shapes", "fills", "letters", "actions")  # no numerals, so a count or a place number cannot collide
_ALT = {
    "shapes": "a pattern of shapes",
    "fills": "a pattern of shapes",
    "mixed": "a pattern of shapes",
    "numbers": "a pattern of numbers",
    "letters": "a pattern of letters",
    "actions": "a pattern of pictures",
}


def _check_tier(tier: str) -> None:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")


def _shape_key(shape: str, fill: str) -> str:
    return f"shape:{shape}:{fill}"


def _draw_keys(rng, family: str, n: int) -> list[str]:
    """``n`` distinct token keys from one family."""
    if family == "shapes":
        fill = rng.choice(pat.FILLS)
        return [_shape_key(s, fill) for s in rng.sample(SHAPES, n)]
    if family == "fills":
        shape = rng.choice(SHAPES)
        return [_shape_key(shape, f) for f in rng.sample(pat.FILLS, n)]
    if family == "numbers":
        return [f"text:{d}" for d in rng.sample(_DIGITS, n)]
    if family == "letters":
        return [f"text:{c}" for c in rng.sample(_LETTERS, n)]
    if family == "actions":
        return [f"action:{a}" for a in rng.sample(ACTIONS, n)]
    if family == "mixed":
        for _ in attempts(200, "tokens that vary in shape and in fill"):
            shapes = rng.sample(SHAPES, rng.randint(2, 3))
            fills = rng.sample(pat.FILLS, 2)
            combos = [(s, f) for s in shapes for f in fills]
            picked = rng.sample(combos, n)
            if len({s for s, _ in picked}) >= 2 and len({f for _, f in picked}) >= 2:
                return [_shape_key(s, f) for s, f in picked]
    raise ValueError(f"unknown family {family!r}")


def _new_keys(rng, family: str, used: list[str], count: int) -> list[str]:
    """Tokens of the same family that the pattern does not use (a wrong answer nobody could read off)."""
    out: list[str] = []
    if family in ("shapes", "mixed"):
        fills = [k.split(":")[2] for k in used]
        fill = rng.choice(fills)
        pool = [_shape_key(s, fill) for s in SHAPES]
    elif family == "fills":
        shape = used[0].split(":")[1]
        pool = [_shape_key(shape, f) for f in pat.FILLS]
    elif family == "numbers":
        pool = [f"text:{d}" for d in _DIGITS]
    elif family == "letters":
        pool = [f"text:{c}" for c in _LETTERS]
    else:
        pool = [f"action:{a}" for a in ACTIONS]
    pool = [k for k in pool if k not in used]
    rng.shuffle(pool)
    out = pool[:count]
    if len(out) < count:  # an action pattern can use all three actions: a plain shape is still not in it
        extra = [_shape_key(s, "solid") for s in SHAPES]
        rng.shuffle(extra)
        out += extra[: count - len(out)]
    return out


def _cells(keys: list[str | None]) -> list[dict]:
    return [mark_cell() if k is None else cell_from_key(k) for k in keys]


def _realize(struct: str, assignment: dict[str, str], length: int) -> list[str]:
    return [assignment[struct[i % len(struct)]] for i in range(length)]


def _assign(rng, family: str, struct: str) -> dict[str, str]:
    names = sorted(set(struct))
    keys = _draw_keys(rng, family, len(names))
    return dict(zip(names, keys))


def _words(text: str) -> int:
    return len(text.split())


def _unit_text(keys: list[str]) -> str:
    return ", ".join(describe_key(k) for k in keys)


# ====================================================================== continue

_NEXT_TEXT = (
    "What comes next in the pattern?",
    "{n} makes a pattern. What comes next?",
    "Look at the pattern. What comes next?",
    "What should come next?",
    "Which one comes next in this pattern?",
)
_GAP_TEXT = (
    "What goes in the empty box?",
    "{n} left a gap in the pattern. What goes in it?",
    "Look at the pattern. What fills the empty box?",
    "Which one fits in the empty box?",
    "What is missing from the pattern?",
)
_CONTINUE_STRUCTS = {"easy": ("AB",), "medium": ("AAB", "ABB", "ABC"), "hard": ("ABC", "ABCD")}


def _continue_distractors(rng, family, unit, answer, prev, *, hard: bool) -> list[str]:
    structural = [prev] + [u for u in unit if u != prev]
    if family in ("shapes", "mixed"):
        shape, fill = answer.split(":")[1:]
        for f in pat.FILLS:  # the right shape, drawn the wrong way
            structural.insert(1, _shape_key(shape, f))
        if family == "mixed":  # the right fill on another shape, from the shapes in the pattern
            for u in unit:
                if u.split(":")[2] == fill:
                    structural.append(_shape_key(u.split(":")[1], fill))
    structural = [k for k in dict.fromkeys(structural) if k != answer]
    chosen = structural[: 3 if hard else 2]
    chosen += _new_keys(rng, family, unit, 6)
    out = [k for k in dict.fromkeys(chosen) if k != answer]
    return [describe_key(k) for k in out]


@generator("ey_pattern_continue", level=1, presets=_PRESETS)
def ey_pattern_continue(rng, tier="easy"):
    _check_tier(tier)
    for _ in attempts(500, "a pattern with one gap"):
        struct = rng.choice(_CONTINUE_STRUCTS[tier])
        p = len(struct)
        family = "mixed" if tier == "hard" else rng.choice(_FAMILIES)
        gap = tier != "easy" and rng.random() < 0.5
        assignment = _assign(rng, family, struct)
        reps = rng.choice((2, 3)) if p <= 3 else 2
        length = p * reps + rng.randint(0, p - 1) + (1 if gap else 0)
        if length > 12:
            continue
        seq = _realize(struct, assignment, length + (0 if gap else 1))
        at = rng.randint(1, length - 2) if gap else length
        answer = seq[at]
        shown = list(seq)
        shown[at] = None  # type: ignore[call-overload]
        fig = pattern_strip(
            rows=[{"cells": [blank_cell() if (gap and i == at) else c for i, c in enumerate(_cells(shown))]}],
            alt=_ALT[family] + (" with an empty box" if gap else ""),
        )
        keys = row_keys(fig)
        try:
            if read_gap(fig) != answer or smallest_period(keys) != p:
                continue
        except ValueError:
            continue
        prev = seq[at - 1]
        unit = [assignment[c] for c in struct]
        break
    answer_text = describe_key(answer)
    distractors = pick_distractors(
        answer_text, _continue_distractors(rng, family, unit, answer, prev, hard=tier == "hard"), n=3
    )
    check(len(distractors) == 3, "too few distractors")
    figures.assert_no_leak(fig, [answer_text, *distractors])
    check(_read_continue(fig) == answer_text, "the figure does not show the keyed item")
    text = rng.choice((_GAP_TEXT if gap else _NEXT_TEXT)).format(n=rng.choice(NAMES))
    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=distractors,
        working=[
            f"The pattern repeats {_unit_text(unit)}. So {answer_text} goes in the {'empty box' if gap else 'next place'}."
        ],
        figure=fig,
    )


def _read_continue(fig: dict) -> str:
    return describe_key(read_gap(fig))


figures.register_reader("ey_pattern_continue", lambda q: _read_continue(q.figure))


# ====================================================================== match

_MATCH_TEXT = (
    "Which row has the same pattern as {m}?",
    "Which row is made the same way as {m}?",
    "Which row is a pattern like {m}?",
)
_LABELS = ("A", "B", "C")


def _model_word(key: str) -> str:
    kind, _, rest = key.partition(":")
    if kind == "shape":
        return rest.split(":")[0]
    if kind == "action":
        return rest
    return rest


def _match_plan(rng, tier: str):
    """(model structure, model family, [(structure, family) for the two look-alikes])."""
    if tier == "easy":
        decoys = rng.sample(("AAB", "ABB", "ABC"), 2)
        return "AB", rng.choice(("actions", "numbers")), decoys, None
    if tier == "medium":
        model = rng.choice(("AAB", "ABB", "ABC"))
        same_length = [s for s in ("AAB", "ABB", "ABC") if s != model]
        first = rng.choice(same_length)
        rest = [s for s in ("AB", "AAB", "ABB", "ABC", "AABB") if s not in (model, first)]
        return model, rng.choice(("numbers", "letters", "shapes", "actions")), [first, rng.choice(rest)], None
    model = rng.choice(("AAB", "ABB", "ABC", "AABB", "ABAC"))
    others = [s for s in ("AB", "AAB", "ABB", "ABC", "AABB", "ABAC") if s != model]
    return model, "actions", rng.sample(others, 2), "surface"


def _row_family(rng, taken: set[str]) -> str:
    options = [f for f in ("shapes", "fills", "numbers", "letters", "actions") if f not in taken]
    return rng.choice(options)


@generator("ey_pattern_match", level=2, presets=_PRESETS)
def ey_pattern_match(rng, tier="easy"):
    _check_tier(tier)
    for _ in attempts(500, "three rows with one matching pattern"):
        model, model_family, decoys, surface = _match_plan(rng, tier)
        model_assignment = _assign(rng, model_family, model)
        reps = 2 if len(model) >= 3 else rng.choice((2, 3))
        model_keys = _realize(model, model_assignment, len(model) * reps)
        model_text = ", ".join(_model_word(k) for k in model_keys)
        text = rng.choice(_MATCH_TEXT).format(m=model_text)
        if _words(text) > 15:
            continue
        structs = [model, *decoys]
        order = list(range(3))
        rng.shuffle(order)  # order[i] = which of the three structures row i draws
        taken: set[str] = set() if surface else {model_family}
        rows, lengths = [], []
        for slot, which in enumerate(order):
            struct = structs[which]
            if surface and which == 1:  # the look-alike: the same kind of picture as the model, a different pattern
                family = model_family
                names = sorted(set(struct))
                pool = list(model_assignment.values()) + _new_keys(rng, family, list(model_assignment.values()), 2)
                assignment = dict(zip(names, pool[: len(names)]))
            else:
                family = _row_family(rng, taken | ({model_family} if surface else set()))
                taken.add(family)
                assignment = _assign(rng, family, struct)
            row_reps = rng.choice((2, 3)) if len(struct) <= 3 else 2
            seq = _realize(struct, assignment, len(struct) * row_reps)
            rows.append({"label": _LABELS[slot], "cells": _cells(seq)})
            lengths.append(len(seq))
        key_slot = order.index(0)
        if lengths.count(lengths[key_slot]) == 1 and lengths[key_slot] in (max(lengths), min(lengths)):
            continue  # the longest or the shortest row would be a shortcut
        fig = pattern_strip(rows=rows, alt="three rows of pictures, each a pattern")
        try:
            if _read_match(fig, text) != f"Row {_LABELS[key_slot]}":
                continue
        except ValueError:
            continue
        break
    answer_text = f"Row {_LABELS[key_slot]}"
    distractors = pick_distractors(answer_text, [f"Row {lab}" for lab in _LABELS], n=2)
    check(len(distractors) == 2, "too few distractors")
    figures.assert_no_leak(fig, [answer_text, *distractors])
    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=distractors,
        working=[f"The pattern is {model_text}. Its repeating part is {len(model)} long, in the shape {model}."],
        figure=fig,
    )


def _read_match(fig: dict, text: str) -> str:
    tail = text.rstrip("?").replace(" like ", " as ").rsplit(" as ", 1)[1]
    model = structure(tail.split(", "))
    match = [
        row["label"]
        for i, row in enumerate(fig["rows"])
        if structure(row_keys(fig, i)) == model  # type: ignore[arg-type]
    ]
    check(len(match) == 1, "no single row has the said pattern")
    return f"Row {match[0]}"


figures.register_reader("ey_pattern_match", lambda q: _read_match(q.figure, q.text))


# ====================================================================== unit

_COUNT_TEXT = (
    "How many are in the part that repeats?",
    "{n} sees a pattern. How many are in the part that repeats?",
    "Look at the pattern. How many are in one repeat?",
    "How many things are in each repeat?",
)
_NTH_TEXT = (
    "What will number {k} in this pattern be?",
    "{n} keeps the pattern going. What is number {k}?",
    "Keep the pattern going. What will number {k} be?",
    "What will be at number {k}?",
)
_MISTAKE_TEXT = (
    "One is wrong. Which one breaks the pattern?",
    "{n} made a mistake. Which one is wrong?",
    "Which one does not belong in the pattern?",
    "Look closely. Which one breaks the pattern?",
)
_COUNT_STRUCTS = ("AB", "AB", "AAB", "ABB", "ABC")
_NTH_STRUCTS = ("AB", "AAB", "ABB", "ABC")
_MISTAKE_STRUCTS = ("AB", "AAB", "ABB", "ABC")


def _count_distractors(p: int, total: int, kinds: int, reps: int) -> list[str]:
    cands = [total, kinds, reps, p + 1, p - 1, 2 * p]
    return [str(c) for c in cands if c >= 1 and c != p]


def _build_count(rng):
    for _ in attempts(500, "a strip with a countable repeating part"):
        struct = rng.choice(_COUNT_STRUCTS)
        p = len(struct)
        family = rng.choice(_PLAIN)
        assignment = _assign(rng, family, struct)
        reps = rng.choice((2, 3)) if p <= 3 else 2
        seq = _realize(struct, assignment, p * reps)
        fig = pattern_strip(rows=[{"cells": _cells(seq)}], alt=_ALT[family])
        answer = str(p)
        try:
            figures.assert_no_leak(fig, [answer])
        except AssertionError:
            continue
        if smallest_period(row_keys(fig)) != p:
            continue
        return fig, p, len(seq), len(set(struct)), reps
    raise AssertionError("unreachable")


def _build_nth(rng):
    for _ in attempts(500, "a numbered strip that stops short"):
        struct = rng.choice(_NTH_STRUCTS)
        p = len(struct)
        family = rng.choice(_PLAIN)
        assignment = _assign(rng, family, struct)
        shown = 2 * p + rng.randint(0, p - 1)
        k = rng.randint(9, 12)
        if k < shown + 2:
            continue
        seq = _realize(struct, assignment, k + 1)
        fig = pattern_strip(rows=[{"cells": _cells(seq[:shown])}], alt=_ALT[family], numbered=True)
        if smallest_period(row_keys(fig)) != p:
            continue
        return fig, assignment, struct, seq, shown, k, family
    raise AssertionError("unreachable")


def _build_mistake(rng):
    for _ in attempts(1000, "a strip with exactly one mistake"):
        struct = rng.choice(_MISTAKE_STRUCTS)
        p = len(struct)
        family = rng.choice(_PLAIN)
        assignment = _assign(rng, family, struct)
        length = 3 * p + rng.randint(0, p - 1) if p > 2 else rng.randint(7, 10)
        seq = _realize(struct, assignment, length)
        at = rng.randint(p, length - 1)
        unit = [assignment[c] for c in struct]
        swap = [k for k in unit if k != seq[at]]
        new = _new_keys(rng, family, unit, 2)
        wrong = rng.choice(swap) if swap and rng.random() < 0.5 else new[0]
        seq[at] = wrong
        fig = pattern_strip(rows=[{"cells": _cells(seq)}], alt=_ALT[family], numbered=True)
        try:
            if read_mistake(fig) != at + 1:
                continue
        except ValueError:
            continue
        return fig, at + 1, length, p, unit, family
    raise AssertionError("unreachable")


@generator("ey_pattern_unit", level=2, presets=_PRESETS)
def ey_pattern_unit(rng, tier="easy"):
    _check_tier(tier)
    name = rng.choice(NAMES)
    if tier == "easy":
        fig, p, total, kinds, reps = _build_count(rng)
        answer_text = str(p)
        distractors = pick_distractors(answer_text, _count_distractors(p, total, kinds, reps), n=3)
        check(len(distractors) >= 2, "too few distractors")
        figures.assert_no_leak(fig, [answer_text, *distractors])
        check(_read_unit(fig, "") == answer_text, "the figure does not show the keyed length")
        return dict(
            text=rng.choice(_COUNT_TEXT).format(n=name),
            answer=answer_text,
            answer_text=answer_text,
            answer_latex=f"plain:{answer_text}",
            distractors=distractors,
            working=[f"The pattern repeats every {p} things, and the strip has {total} things in all."],
            figure=fig,
        )
    if tier == "medium":
        fig, assignment, struct, seq, shown, k, family = _build_nth(rng)
        unit = [assignment[c] for c in struct]
        answer = seq[k - 1]
        answer_text = describe_key(answer)
        structural = [seq[k - 2], seq[k], seq[shown - 1]]  # one place early, one late, the last one drawn
        cands = [describe_key(c) for c in dict.fromkeys(structural) if c != answer]
        cands += [describe_key(c) for c in _new_keys(rng, family, unit, 4)]
        distractors = pick_distractors(answer_text, cands, n=3)
        check(len(distractors) == 3, "too few distractors")
        text = rng.choice(_NTH_TEXT).format(n=name, k=k)
        figures.assert_no_leak(fig, [answer_text, *distractors])
        check(_read_unit(fig, text) == answer_text, "the figure does not show the keyed item")
        return dict(
            text=text,
            answer=answer_text,
            answer_text=answer_text,
            answer_latex=f"plain:{answer_text}",
            distractors=distractors,
            working=[f"The pattern repeats {_unit_text(unit)}. Keep going to number {k}: {answer_text}."],
            figure=fig,
        )
    fig, place, length, p, unit, family = _build_mistake(rng)
    answer_text = f"Number {place}"
    near = [place + 1, place - 1, length, p + 1, 2 * p + 1]
    cands = [f"Number {c}" for c in dict.fromkeys(near) if 1 <= c <= length and c != place]
    distractors = pick_distractors(answer_text, cands, n=3)
    check(len(distractors) == 3, "too few distractors")
    figures.assert_no_leak(fig, [answer_text, *distractors])
    check(_read_unit(fig, "One is wrong.") == answer_text, "the figure does not show the keyed mistake")
    return dict(
        text=rng.choice(_MISTAKE_TEXT).format(n=name),
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=distractors,
        working=[f"The pattern repeats {_unit_text(unit)}. Number {place} is not what comes in that place."],
        figure=fig,
    )


def _read_unit(fig: dict, text: str) -> str:
    """Read the answer from the strip: the length of the repeat, the item at the place the prompt names, or the odd cell."""
    keys = row_keys(fig)
    if text.startswith("One is wrong") or "mistake" in text or "belong" in text or "breaks" in text:
        return f"Number {read_mistake(fig)}"
    place = [int(w) for w in text.replace("?", " ").split() if w.isdigit()]
    if place:
        return describe_key(read_nth(fig, place[0]))
    return str(smallest_period(keys))


figures.register_reader("ey_pattern_unit", lambda q: _read_unit(q.figure, q.text))
