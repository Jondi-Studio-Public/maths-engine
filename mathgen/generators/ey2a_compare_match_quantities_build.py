"""Comparing counters in rings: which ring has more, how many more, and matching by pairing.

Descriptors: AC9MFN03 (compare, order and make correspondences between collections), AC9M1N01.

Rings of counters are drawn left to right with no numbers. The stem never says which ring has more, and the answer
is a number, so it cannot be guessed from the wording. Difficulty escalates in what is compared:

- easy: two rings; type how many counters are in the ring that has more.
- medium: two rings; each counter in one ring is matched with one in the other. Type how many counters have no
  partner, which is how many more the bigger ring has.
- hard: three rings; type how many more the biggest ring has than the smallest, or how many counters must be added
  in all to make the other two rings match the biggest.
"""

from __future__ import annotations

from .. import figures
from ..core import generator
from ..figure_kinds.early import counters
from ..slots import NAMES
from ._ey2a_kit import SHAPE_WORD, number_question, tagged_numbers

_PRESETS = {"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")}

_ALT = {2: "two rings of counters", 3: "three rings of counters"}


@generator("ey2a_compare_match_quantities", level=1, presets=_PRESETS)
def ey2a_compare_match_quantities(rng, tier="easy"):
    shape = rng.choice(sorted(SHAPE_WORD))
    word = SHAPE_WORD[shape]
    name = rng.choice(NAMES)
    if tier in ("easy", "medium"):
        small = rng.randint(1, 7) if tier == "easy" else rng.randint(2, 8)
        gap = rng.randint(2, 4) if tier == "easy" else rng.randint(1, 4)
        big = min(9, small + gap)
        small = min(small, big - 1)
        sizes = [small, big]
        rng.shuffle(sizes)
        fig = counters(layout="groups", alt=_ALT[2], shape=shape, groups=sizes)
        diff = big - small
        if tier == "easy":
            answer = big
            text = rng.choice(
                (
                    "Look at the two rings. How many {w} are in the ring with more?",
                    "{n} looks at the two rings. How many {w} does the bigger ring have?",
                )
            ).format(n=name, w=word)
            cands = [
                (small, "ey2a-reports-smaller-group"),
                (diff, "ey2a-reports-the-difference"),
                (big + 1, "ey2a-count-one-over"),
                (big - 1, "ey2a-count-one-short"),
                (big + small, "ey2a-adds-groups"),
                (big + 2, "ey2a-count-one-over"),
            ]
            working = [f"The rings have {sizes[0]} and {sizes[1]}. The ring with more has {big}."]
        else:
            answer = diff
            text = rng.choice(
                (
                    "{n} pairs up the {w} in the two rings. How many have no partner?",
                    "Pair up the {w} in the two rings. How many are left without a partner?",
                )
            ).format(n=name, w=word)
            cands = [
                (big, "ey2a-reports-larger-group"),
                (small, "ey2a-reports-smaller-group"),
                (diff + 1, "ey2a-count-one-over"),
                (diff - 1, "ey2a-count-one-short"),
                (big + small, "ey2a-adds-groups"),
                (diff + 2, "ey2a-count-one-over"),
            ]
            working = [f"{small} pairs are made. {big} - {small} = {diff} {'has' if diff == 1 else 'have'} no partner."]
    elif tier == "hard":
        sizes = rng.sample(range(2, 10), 3)
        lo, mid, hi = sorted(sizes)
        fig = counters(layout="groups", alt=_ALT[3], shape=shape, groups=sizes)
        if rng.random() < 0.5:
            answer = hi - lo
            text = rng.choice(
                (
                    "How many more {w} does the biggest ring have than the smallest ring?",
                    "{n} compares the biggest and smallest rings. How many more {w} are in the biggest?",
                )
            ).format(n=name, w=word)
            cands = [
                (hi - mid, "ey2a-compares-wrong-rings"),
                (mid - lo, "ey2a-compares-wrong-rings"),
                (hi, "ey2a-reports-larger-group"),
                (answer + 1, "ey2a-count-one-over"),
                (answer - 1, "ey2a-count-one-short"),
                (answer + 2, "ey2a-count-one-over"),
            ]
            working = [f"The rings have {lo}, {mid} and {hi}. {hi} - {lo} = {answer}."]
        else:
            answer = (hi - lo) + (hi - mid)
            text = rng.choice(
                (
                    "{n} adds {w} until all three rings match the biggest. How many {w} are added?",
                    "How many {w} are added in all to make every ring as big as the biggest?",
                )
            ).format(n=name, w=word)
            cands = [
                (hi - lo, "ey2a-fixes-one-ring-only"),
                (hi - mid, "ey2a-fixes-one-ring-only"),
                (hi, "ey2a-reports-larger-group"),
                (answer + 1, "ey2a-count-one-over"),
                (answer - 1, "ey2a-count-one-short"),
                (answer + 2, "ey2a-count-one-over"),
            ]
            working = [f"The rings have {lo}, {mid} and {hi}. {hi - lo} + {hi - mid} = {answer} are added."]
    else:
        raise ValueError(f"unknown tier {tier!r}")
    figures.assert_no_leak(fig, [str(answer)])
    return number_question(
        text=text, answer=answer, distractors=tagged_numbers(answer, cands, low=1, high=30), working=working, figure=fig
    )


def _read(q) -> str:
    """Re-derive the answer from the drawn rings and the stem alone."""
    sizes = q.figure["groups"]
    text = q.text
    if len(sizes) == 3:
        if "biggest and smallest" in text or "than the smallest" in text:
            return str(max(sizes) - min(sizes))
        return str(sum(max(sizes) - s for s in sizes))
    if "partner" in text:
        return str(abs(sizes[0] - sizes[1]))
    return str(max(sizes))


figures.register_reader("ey2a_compare_match_quantities", _read)
