"""Generates `fig_venn_counts` questions: read and complete a Venn diagram of counts (AC9M8P03).

One skill: ``f10-venn-diagram-counts`` (``answer: number``). A group of people is sorted by two or three things
they do (sports they play, subjects they study) and the Venn diagram, drawn with ``venn``, is the stimulus.
Every number the learner needs is on the diagram or in the one sentence that states a total. Three tiers:

- easy (two sets): three regions show a count and one shows ``?``; the sentence gives the total of the group. The
  learner finds the missing region by subtracting the three shown from the total.
- medium (two sets): every region shows a count. The question asks for a number that is NOT a single region: all
  of one set (its only-part and the overlap), either set or both, or exactly one of the two sets. The learner has to
  choose which regions the words mean and add them.
- hard (three sets): one region of seven shows ``?`` and the sentence gives either the total of the group (the
  missing region can be anywhere) or the total in one set's circle (the missing region is one of the four inside that
  circle). The learner adds the right regions and subtracts.

The prompt never states a region count, so the diagram is the only source of the numbers. The figure reader
re-derives each answer from the diagram and the one number or phrase the sentence supplies.
"""

from __future__ import annotations

import re

from .. import figures
from ..core import attempts, check, generator, phrase, pick_distractors
from ..figure_kinds import space
from . import _space_kit as kit

_CONTEXTS = (
    ("play", ("Tennis", "Netball", "Soccer", "Cricket", "Swimming", "Hockey", "Rugby", "Cycling")),
    ("study", ("Art", "Music", "French", "Drama", "Latin", "Coding")),
)
_NOUNS = ("students", "children", "people")
_UNIVERSE = {"students": "Students", "children": "Children", "people": "People"}


def _keys(n: int) -> list[str]:
    return space.venn_region_keys(list("ABC"[:n]))


def _figure(names, counts, noun, blank=None):
    """The Venn diagram for ``names`` with a count in each region and ``?`` in region ``blank``."""
    regions = {k: ("?" if k == blank else str(v)) for k, v in counts.items()}
    return space.venn(
        alt=f"A Venn diagram of {'two' if len(names) == 2 else 'three'} overlapping sets inside a rectangle"
        + (", with one region marked ?." if blank else "."),
        sets=[{"id": "ABC"[i], "name": name} for i, name in enumerate(names)],
        regions=regions,
        universe=_UNIVERSE[noun],
    )


def _draw(rng, n_sets: int, lo: int, hi: int) -> dict[str, int]:
    return {k: rng.randint(lo, hi) for k in _keys(n_sets)}


def _context(rng, n_sets: int):
    verb, pool = rng.choice(_CONTEXTS)
    return verb, rng.sample(pool, n_sets), rng.choice(_NOUNS)


def _shown(counts, blank):
    return [v for k, v in counts.items() if k != blank]


# ------------------------------------------------------------------------------------------- tiers


def _easy(rng):
    verb, names, noun = _context(rng, 2)
    for _ in attempts(100, "an easy Venn diagram"):
        counts = _draw(rng, 2, 1, 15)
        blank = rng.choice(list(counts))
        total = sum(counts.values())
        fig = _figure(names, counts, noun, blank)
        if not kit.leaks(fig, str(counts[blank])):
            break
    u = counts[blank]
    shown = _shown(counts, blank)
    text = phrase(
        rng,
        (
            "There are {total} {noun} in total. The Venn diagram shows how many {verb} {a}, {b}, both or neither. "
            "Find the number marked ?.",
            "There are {total} {noun} in total. They were asked about {a} and {b}, and the Venn diagram shows "
            "the results. Find the number marked ?.",
        ),
        total=total,
        noun=noun,
        verb=verb,
        a=names[0],
        b=names[1],
    )
    cands = [
        sum(shown),  # adds the shown numbers and never uses the total
        total - shown[0] - shown[1],  # leaves out one region
        total - shown[0] - shown[2],
        total - shown[1] - shown[2],
        total,
    ]
    working = [f"The regions add to the total: {' + '.join(str(v) for v in shown)} + ? = {total}.", f"? = {u}."]
    return fig, text, u, cands, working


def _medium(rng):
    verb, names, noun = _context(rng, 2)
    for _ in attempts(100, "a medium Venn diagram"):
        counts = _draw(rng, 2, 2, 15)
        a, b, ab, none = counts["A"], counts["B"], counts["AB"], counts["none"]
        kind = rng.choice(("circle", "either", "exactly_one"))
        lead = f"The Venn diagram shows how many {noun} {verb} {names[0]} and {names[1]}. "
        if kind == "circle":
            which = rng.randrange(2)
            only, other = (a, b) if which == 0 else (b, a)
            answer = only + ab
            text = lead + f"How many {noun} {verb} {names[which]}?"
            cands = [only, only + ab + other, ab, a + b]  # only-part; either; overlap alone; exactly one
            working = [f"Everyone in the {names[which]} circle counts, overlap included: {only} + {ab} = {answer}."]
        elif kind == "either":
            answer = a + ab + b
            text = lead + f"How many {noun} {verb} {names[0]} or {names[1]} or both?"
            cands = [a + b, a + 2 * ab + b, a + ab, a + ab + b + none]
            working = [f"Add the three regions inside the circles: {a} + {ab} + {b} = {answer}."]
        else:
            answer = a + b
            text = lead + f"How many {noun} {verb} exactly one of {names[0]} and {names[1]}?"
            cands = [a + ab + b, a, b, a + 2 * ab + b]
            working = [f"Exactly one means the two parts without the overlap: {a} + {b} = {answer}."]
        fig = _figure(names, counts, noun)
        if not kit.leaks(fig, str(answer)):
            break
    return fig, text, answer, cands, working


def _hard(rng):
    verb, names, noun = _context(rng, 3)
    for _ in attempts(200, "a hard Venn diagram"):
        counts = _draw(rng, 3, 1, 9)
        total = sum(counts.values())
        by_total = rng.random() < 0.5
        if by_total:
            blank = rng.choice(list(counts))
        else:
            circle = rng.randrange(3)
            letter = "ABC"[circle]
            inside = [k for k in counts if letter in k]
            blank = rng.choice(inside)
        fig = _figure(names, counts, noun, blank)
        u = counts[blank]
        if not kit.leaks(fig, str(u)):
            break
    if by_total:
        text = phrase(
            rng,
            (
                "There are {total} {noun} in total. The Venn diagram shows how many {verb} {a}, {b} and {c}. "
                "Find the number marked ?.",
                "There are {total} {noun} in total. Each was asked whether they {verb} {a}, {b} or {c}, and the "
                "Venn diagram shows the results. Find the number marked ?.",
            ),
            total=total,
            noun=noun,
            verb=verb,
            a=names[0],
            b=names[1],
            c=names[2],
        )
        shown = _shown(counts, blank)
        keep = rng.sample(range(len(shown)), 3)
        cands = [sum(shown)] + [u + shown[i] for i in keep]  # adds them; or leaves one region out
        working = [
            f"The eight regions add to the total: {' + '.join(str(v) for v in shown)} + ? = {total}.",
            f"? = {total} - {sum(shown)} = {u}.",
        ]
    else:
        circle_total = sum(v for k, v in counts.items() if letter in k)
        text = phrase(
            rng,
            (
                "In total, {n} {noun} {verb} {name}. The Venn diagram shows how many {verb} {a}, {b} and {c}. "
                "Find the number marked ?.",
                "The Venn diagram shows how many {noun} {verb} {a}, {b} and {c}. In total, {n} {noun} {verb} "
                "{name}. Find the number marked ?.",
            ),
            n=circle_total,
            noun=noun,
            verb=verb,
            name=names[circle],
            a=names[0],
            b=names[1],
            c=names[2],
        )
        in_circle = [v for k, v in counts.items() if letter in k and k != blank]
        cands = [circle_total] + [u + v for v in in_circle]
        working = [
            f"Everything inside the {names[circle]} circle adds to {circle_total}: "
            f"{' + '.join(str(v) for v in in_circle)} + ? = {circle_total}.",
            f"? = {circle_total} - {sum(in_circle)} = {u}.",
        ]
    return fig, text, u, cands, working


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "fig_venn_counts",
    level=8,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fig_venn_counts(rng, tier="easy"):
    """A Venn diagram of counts with a whole-number answer. See the module docstring for the three tiers."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(40, "a Venn question with three distinct wrong answers"):
        fig, text, answer, candidates, working = _TIERS[tier](rng)
        check(answer > 0, "a Venn region count must be positive")
        pool = [str(c) for c in candidates if c > 0 and c != answer]
        distractors = pick_distractors(str(answer), pool)
        if len(distractors) == 3:
            break
    return dict(
        text=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        distractors=distractors,
        working=working,
        figure=fig,
    )


# ------------------------------------------------------------------------------------------- reader

_TOTAL = re.compile(r"There are (\d+) \w+ in total")
_CIRCLE_TOTAL = re.compile(r"In total, (\d+) \w+ (?:play|study) (\w+)\.")
_ONE_SET = re.compile(r"How many \w+ (?:play|study) (\w+)\?$")
_EITHER = re.compile(r"How many \w+ (?:play|study) (\w+) or (\w+) or both\?$")
_EXACTLY_ONE = re.compile(r"exactly one of (\w+) and (\w+)\?$")


def _read_venn(question) -> str:
    """Re-derive the answer from the diagram alone, with the one total or phrase the sentence supplies."""
    fig, text = question.figure, question.text
    names = {s["name"]: s["id"] for s in fig["sets"]}
    regions = fig["regions"]
    unknown = [k for k, v in regions.items() if v == space.BLANK]
    if unknown:
        check(len(unknown) == 1, "more than one region is marked ?")
        key = unknown[0]
        rest = {k: int(v) for k, v in regions.items() if k != key}
        if (m := _CIRCLE_TOTAL.search(text)) is not None:
            letter = names[m.group(2)]
            check(letter in key, "the stated circle does not contain the region marked ?")
            return str(int(m.group(1)) - sum(v for k, v in rest.items() if letter in k))
        m = _TOTAL.search(text)
        check(m is not None, "no total in the question")
        return str(int(m.group(1)) - sum(rest.values()))
    counts = {k: int(v) for k, v in regions.items()}
    if (m := _EXACTLY_ONE.search(text)) is not None:
        x, y = names[m.group(1)], names[m.group(2)]
        return str(counts[x] + counts[y])
    if (m := _EITHER.search(text)) is not None:
        x, y = names[m.group(1)], names[m.group(2)]
        return str(counts[x] + counts[y] + counts["".join(sorted((x, y)))])
    m = _ONE_SET.search(text)
    check(m is not None, "cannot read the question")
    letter = names[m.group(1)]
    return str(sum(v for k, v in counts.items() if letter in k))


figures.register_reader("fig_venn_counts", _read_venn)
