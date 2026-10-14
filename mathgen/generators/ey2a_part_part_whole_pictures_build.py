"""Part-part-whole with two kinds of counter: the whole from its parts, and the missing part from a stated whole.

Descriptors: AC9MFN04 (partition and combine collections to 20), AC9M1N04.

Plain counters are one part and ringed counters the other (a ring, not a colour, tells them apart). The counters are
drawn on ten frames and the numbers are never printed in the picture. Difficulty escalates in the reasoning:

- easy: a ten frame holds both kinds; find the whole (the parts are visible).
- medium: the whole is stated in the stem, only the ringed counters are drawn; find how many plain ones are missing
  (the inverse of combining).
- hard: two ten frames with the counters in scattered boxes, whole to 20, so the parts bridge ten. Either find the
  whole from both kinds, or find the missing part from a stated whole.
"""

from __future__ import annotations

from .. import figures
from ..core import generator
from ..figure_kinds.early import counters
from ..slots import NAMES
from ._ey2a_kit import SHAPE_WORD, number_question, tagged_numbers, ten_frame_boxes

_PRESETS = {"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")}


def _split_frames(rng, ringed: int, plain: int) -> list[list[int]]:
    """Two ten frames holding ``ringed`` ringed and ``plain`` plain counters in random boxes, neither frame full
    unless the total needs it. The counters are dealt to the frames so each holds 1 to 10."""
    total = ringed + plain
    for _ in range(200):
        first = rng.randint(max(1, total - 10), min(10, total - 1))
        kinds = [2] * ringed + [1] * plain
        rng.shuffle(kinds)
        a, b = kinds[:first], kinds[first:]
        if 1 <= len(a) <= 10 and 1 <= len(b) <= 10:
            return [ten_frame_boxes(len(a), rng=rng, colours=a), ten_frame_boxes(len(b), rng=rng, colours=b)]
    raise ValueError("could not deal the counters onto two frames")


@generator("ey2a_part_part_whole_pictures", level=1, presets=_PRESETS)
def ey2a_part_part_whole_pictures(rng, tier="easy"):
    shape = rng.choice(sorted(SHAPE_WORD))
    word = SHAPE_WORD[shape]
    name = rng.choice(NAMES)
    if tier == "easy":
        plain = rng.randint(1, 8)
        ringed = rng.randint(1, min(9, 10 - plain))
        boxes = ten_frame_boxes(plain + ringed, colours=[1] * plain + [2] * ringed)
        fig = counters(layout="ten_frame", alt="a ten frame with two kinds of counter", shape=shape, frames=[boxes])
        answer = plain + ringed
        text = rng.choice(
            (
                "{n} has plain and ringed {w}. How many {w} in all?",
                "How many {w} are there altogether?",
                "Count the plain and the ringed {w}. How many in all?",
            )
        ).format(n=name, w=word)
        cands = [
            (plain, "ey2a-reports-one-part"),
            (ringed, "ey2a-reports-one-part"),
            (answer + 1, "ey2a-count-one-over"),
            (answer - 1, "ey2a-count-one-short"),
            (10 - answer, "ey2a-counts-empty-boxes"),
            (abs(plain - ringed), "ey2a-subtracts-the-parts"),
        ]
        working = [f"{plain} plain and {ringed} ringed. {plain} + {ringed} = {answer}."]
    elif tier == "medium":
        whole = rng.randint(5, 10)
        shown = rng.randint(2, whole - 1)
        boxes = ten_frame_boxes(shown, rng=rng, colours=[2] * shown)
        fig = counters(layout="ten_frame", alt="a ten frame with some ringed counters", shape=shape, frames=[boxes])
        answer = whole - shown
        text = rng.choice(
            (
                "{n} has {whole} {w}. The ringed ones are shown. How many are plain?",
                "There are {whole} {w}. Only the ringed ones are shown. How many are plain?",
            )
        ).format(n=name, whole=whole, w=word)
        cands = [
            (shown, "ey2a-reports-shown-part"),
            (whole, "ey2a-reports-the-whole"),
            (whole + shown, "ey2a-adds-parts-for-missing"),
            (answer + 1, "ey2a-count-one-over"),
            (answer - 1, "ey2a-count-one-short"),
            (10 - shown, "ey2a-counts-empty-boxes"),
        ]
        working = [f"The picture shows {shown}. {whole} - {shown} = {answer} are plain."]
    elif tier == "hard":
        ringed = rng.randint(3, 9)
        plain = rng.randint(max(1, 11 - ringed), min(9, 20 - ringed))
        whole = ringed + plain
        if rng.random() < 0.5:
            frames = _split_frames(rng, ringed, plain)
            fig = counters(
                layout="ten_frame", alt="two ten frames with two kinds of counter", shape=shape, frames=frames
            )
            answer = whole
            text = rng.choice(
                (
                    "{n} has plain and ringed {w}. How many {w} in all?",
                    "Count the plain and the ringed {w} on both frames. How many in all?",
                )
            ).format(n=name, w=word)
            cands = [
                (max(plain, ringed), "ey2a-reports-one-part"),
                (answer + 1, "ey2a-count-one-over"),
                (answer - 1, "ey2a-count-one-short"),
                (abs(plain - ringed), "ey2a-subtracts-the-parts"),
                (20 - answer, "ey2a-counts-empty-boxes"),
                (answer + 2, "ey2a-count-one-over"),
            ]
            working = [f"{plain} plain and {ringed} ringed. {plain} + {ringed} = {answer}."]
        else:
            frames = _split_frames(rng, ringed, 0)
            fig = counters(
                layout="ten_frame", alt="two ten frames with some ringed counters", shape=shape, frames=frames
            )
            answer = plain
            text = rng.choice(
                (
                    "{n} has {whole} {w}. The ringed ones are shown. How many are plain?",
                    "There are {whole} {w}. Only the ringed ones are shown. How many are plain?",
                )
            ).format(n=name, whole=whole, w=word)
            cands = [
                (ringed, "ey2a-reports-shown-part"),
                (whole, "ey2a-reports-the-whole"),
                (answer + 1, "ey2a-count-one-over"),
                (answer - 1, "ey2a-count-one-short"),
                (20 - ringed, "ey2a-counts-empty-boxes"),
                (answer + 2, "ey2a-count-one-over"),
            ]
            working = [f"The picture shows {ringed}. {whole} - {ringed} = {answer} are plain."]
    else:
        raise ValueError(f"unknown tier {tier!r}")
    figures.assert_no_leak(fig, [str(answer)])
    return number_question(
        text=text, answer=answer, distractors=tagged_numbers(answer, cands, low=1, high=30), working=working, figure=fig
    )


def _read(q) -> str:
    """Re-derive the answer from the drawn counters and the stem alone."""
    counts = {1: 0, 2: 0}
    for frame in q.figure["frames"]:
        for box in frame:
            if box:
                counts[box] += 1
    if "plain?" in q.text:
        whole = int(next(t for t in q.text.replace(".", " ").split() if t.isdigit()))
        return str(whole - counts[2])
    return str(counts[1] + counts[2])


figures.register_reader("ey2a_part_part_whole_pictures", _read)
