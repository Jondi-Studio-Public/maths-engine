"""Collect, sort and compare objects: a scatter of small pictures, the kind named under each one.

Descriptor: AC9MFST01 (collect, sort and compare data represented by objects and images).

The picture is a ``tally`` figure with objects only: up to 20 pictures (apples, cats, cars, ...) scattered over the
page, the name of the kind under each. It draws no total. Every answer is a number the learner finds by sorting and
counting, and the stem never says a count. Difficulty escalates in what is done with the counts:

- easy: two or three kinds; how many of one kind.
- medium: three kinds; how many more of one kind than another, or how many of two kinds altogether.
- hard: four kinds; the size of the biggest group, how many fewer the smallest group has than the biggest, or how
  many are not of one kind. The learner has to find the extremes before comparing.
"""

from __future__ import annotations

from .. import figures
from ..core import attempts, generator
from ..figure_kinds.tally import tally
from ..slots import NAMES
from ._ey_tally_kit import PLURAL, ICONS, item_counts, number_question, scatter_objects, tagged_numbers

_PRESETS = {"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")}
_ALT = "a scatter of small pictures to sort"

_EASY = (
    "How many {pl} are there?",
    "Count the {pl}. How many are there?",
    "{n} sorts the things. How many {pl} are there?",
)
_MORE = (
    "How many more {a} than {b} are there?",
    "{n} compares the groups. How many more {a} than {b}?",
)
_BOTH = (
    "How many {a} and {b} are there altogether?",
    "{n} counts the {a} and the {b}. How many altogether?",
)
_MOST = (
    "How many are in the group with the most?",
    "{n} finds the biggest group. How many are in it?",
)
_FEWER = (
    "How many fewer are in the smallest group than the biggest group?",
    "{n} compares the smallest and biggest groups. How many fewer does the smallest have?",
)
_NOT = (
    "How many are not {pl}?",
    "{n} puts the {pl} aside. How many things are left?",
)


def _draw_counts(rng, tier: str) -> dict[str, int]:
    """Counts per kind for the tier; the extremes are unique where the question needs them to be."""
    kinds = rng.sample(ICONS, 2 if tier == "easy" and rng.random() < 0.5 else 3 if tier != "hard" else 4)
    for _ in attempts(500, "counts that suit the tier"):
        lo, hi = (1, 8) if tier == "easy" else (2, 7)
        counts = {k: rng.randint(lo, hi) for k in kinds}
        total = sum(counts.values())
        if total > (12 if tier == "easy" else 16 if tier == "medium" else 20):
            continue
        vals = sorted(counts.values())
        if tier == "hard" and (vals[-1] == vals[-2] or vals[0] == vals[1] or vals[-1] - vals[0] < 2):
            continue
        if tier != "easy" and len(set(vals)) < len(vals):
            continue
        return counts
    raise AssertionError("unreachable")


@generator("ey_sort_objects", level=1, presets=_PRESETS)
def ey_sort_objects(rng, tier="easy"):
    if tier not in _PRESETS:
        raise ValueError(f"unknown tier {tier!r}")
    name = rng.choice(NAMES)
    counts = _draw_counts(rng, tier)
    kinds = list(counts)
    total = sum(counts.values())
    fig = tally(alt=_ALT, items=scatter_objects(rng, counts))
    if tier == "easy":
        a, *others = kinds
        answer = counts[a]
        text = rng.choice(_EASY).format(pl=PLURAL[a], n=name)
        cands = [
            (total, "eyt-counts-all-kinds"),
            (counts[others[0]], "eyt-other-group"),
            (answer + 1, "eyt-count-one-over"),
            (answer - 1, "eyt-count-one-short"),
            (answer + 2, "eyt-count-one-over"),
        ]
        working = [f"Sort the pictures. Count the {PLURAL[a]}: {answer}."]
    elif tier == "medium":
        a, b, c = kinds
        if rng.random() < 0.5:
            a, b = (a, b) if counts[a] > counts[b] else (b, a)
            answer = counts[a] - counts[b]
            text = rng.choice(_MORE).format(a=PLURAL[a], b=PLURAL[b], n=name)
            cands = [
                (counts[a], "eyt-reports-one-group"),
                (counts[a] + counts[b], "eyt-adds-instead-of-compares"),
                (counts[b], "eyt-other-group"),
                (answer + 1, "eyt-count-one-over"),
                (answer - 1, "eyt-count-one-short"),
            ]
            working = [f"{PLURAL[a]}: {counts[a]}. {PLURAL[b]}: {counts[b]}. {counts[a]} - {counts[b]} = {answer}."]
        else:
            answer = counts[a] + counts[b]
            text = rng.choice(_BOTH).format(a=PLURAL[a], b=PLURAL[b], n=name)
            cands = [
                (counts[a], "eyt-reports-one-group"),
                (total, "eyt-counts-all-kinds"),
                (counts[a] + counts[c], "eyt-other-group"),
                (answer + 1, "eyt-count-one-over"),
                (answer - 1, "eyt-count-one-short"),
            ]
            working = [f"{PLURAL[a]}: {counts[a]}. {PLURAL[b]}: {counts[b]}. {counts[a]} + {counts[b]} = {answer}."]
    elif tier == "hard":
        order = sorted(kinds, key=counts.get)
        lo_kind, hi_kind = order[0], order[-1]
        lo, hi = counts[lo_kind], counts[hi_kind]
        pick = rng.choice(("most", "fewer", "not"))
        if pick == "most":
            answer = hi
            text = rng.choice(_MOST).format(n=name)
            cands = [
                (lo, "eyt-wrong-group"),
                (total, "eyt-counts-all-kinds"),
                (counts[order[-2]], "eyt-wrong-group"),
                (answer + 1, "eyt-count-one-over"),
                (answer - 1, "eyt-count-one-short"),
            ]
            working = [f"The groups have {', '.join(str(counts[k]) for k in kinds)}. The biggest has {answer}."]
        elif pick == "fewer":
            answer = hi - lo
            text = rng.choice(_FEWER).format(n=name)
            cands = [
                (hi, "eyt-reports-one-group"),
                (lo, "eyt-reports-one-group"),
                (hi + lo, "eyt-adds-instead-of-compares"),
                (counts[order[-2]] - lo, "eyt-wrong-group"),
                (answer + 1, "eyt-count-one-over"),
                (answer - 1, "eyt-count-one-short"),
            ]
            working = [f"The biggest group has {hi} and the smallest has {lo}. {hi} - {lo} = {answer}."]
        else:
            a = rng.choice(kinds)
            answer = total - counts[a]
            text = rng.choice(_NOT).format(pl=PLURAL[a], n=name)
            cands = [
                (counts[a], "eyt-reports-one-group"),
                (total, "eyt-counts-all-kinds"),
                (answer + 1, "eyt-count-one-over"),
                (answer - 1, "eyt-count-one-short"),
                (answer + 2, "eyt-count-one-over"),
            ]
            working = [
                f"Things in all: {total}. {PLURAL[a].capitalize()}: {counts[a]}. {total} - {counts[a]} = {answer}."
            ]
    else:
        raise ValueError(f"unknown tier {tier!r}")
    figures.assert_no_leak(fig, [str(answer)])
    return number_question(
        text=text, answer=answer, distractors=tagged_numbers(answer, cands, low=0, high=40), working=working, figure=fig
    )


def _read(q) -> str:
    """Re-derive the answer from the scattered pictures and the stem alone."""
    counts = item_counts(q.figure)
    text, total = q.text, sum(counts.values())

    def named(*, first: bool) -> list[str]:
        found = sorted((text.index(PLURAL[k]), k) for k in counts if PLURAL[k] in text)
        return [k for _, k in found]

    if "more" in text and "than" in text:
        a, b = named(first=True)[:2]
        return str(counts[a] - counts[b])
    if "altogether" in text or ("counts the" in text and "and the" in text):
        a, b = named(first=True)[:2]
        return str(counts[a] + counts[b])
    if "fewer" in text:
        return str(max(counts.values()) - min(counts.values()))
    if "the most" in text or "biggest group. How many are in it" in text:
        return str(max(counts.values()))
    if "not" in text or "aside" in text:
        (a,) = named(first=True)[:1]
        return str(total - counts[a])
    (a,) = named(first=True)[:1]
    return str(counts[a])


figures.register_reader("ey_sort_objects", _read)
