"""Numbers on one or two ten frames: how many there are, how many more fill a frame, and how many in all.

Descriptors: AC9MFN04 (partition and combine collections to 20), AC9M1N02, AC9M1N04.

The counters are drawn, never counted in the stem. Difficulty escalates in the structure the learner must use:

- easy: one frame filled from the top left; say how many counters there are, or how many more fill the frame.
- medium: two frames, one full; the other holds some. Say how many in all (ten and some more), or how many must move
  from the second frame to fill the first (make ten).
- hard: two part-filled frames with the counters in scattered boxes, so no frame is full; say how many in all
  (bridging ten) or how many more would fill both frames.
"""

from __future__ import annotations

from .. import figures
from ..core import generator
from ..figure_kinds.early import counters
from ..slots import NAMES
from ._ey2a_kit import SHAPE_WORD, number_question, tagged_numbers, ten_frame_boxes

_PRESETS = {"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")}


def _filled(frame: list[int]) -> int:
    return sum(1 for b in frame if b)


@generator("ey2a_ten_frame_numbers", level=1, presets=_PRESETS)
def ey2a_ten_frame_numbers(rng, tier="easy"):
    shape = rng.choice(sorted(SHAPE_WORD))
    word = SHAPE_WORD[shape]
    name = rng.choice(NAMES)
    if tier == "easy":
        n = rng.randint(1, 9)
        boxes = ten_frame_boxes(n, by_columns=rng.random() < 0.5)
        fig = counters(layout="ten_frame", alt="a ten frame with some counters", shape=shape, frames=[boxes])
        if rng.random() < 0.5:
            answer = n
            text = rng.choice(("How many {w} are on the ten frame?", "{n} counts the {w} on the ten frame. How many?"))
            text = text.format(n=name, w=word)
            cands = [
                (10 - n, "ey2a-counts-empty-boxes"),
                (n + 1, "ey2a-count-one-over"),
                (n - 1, "ey2a-count-one-short"),
                (10, "ey2a-says-the-full-frame"),
                (n + 2, "ey2a-count-one-over"),
            ]
            working = [f"Count the {word} in the boxes: {n}."]
        else:
            answer = 10 - n
            text = rng.choice(
                ("How many more {w} fill the ten frame?", "{n} wants a full ten frame. How many more {w}?")
            )
            text = text.format(n=name, w=word)
            cands = [
                (n, "ey2a-counts-filled-boxes"),
                (answer + 1, "ey2a-count-one-over"),
                (answer - 1, "ey2a-count-one-short"),
                (10, "ey2a-says-the-full-frame"),
                (answer + 2, "ey2a-count-one-over"),
            ]
            working = [
                f"The frame has 10 boxes and {n} {'is' if n == 1 else 'are'} full. "
                f"10 - {n} = {answer} {'is' if answer == 1 else 'are'} empty."
            ]
    elif tier == "medium":
        k = rng.randint(1, 9)
        if rng.random() < 0.5:
            frames = [ten_frame_boxes(10), ten_frame_boxes(k)]
            fig = counters(layout="ten_frame", alt="two ten frames", shape=shape, frames=frames)
            answer = 10 + k
            text = rng.choice(("How many {w} are on the two ten frames?", "{n} counts all the {w}. How many in all?"))
            text = text.format(n=name, w=word)
            cands = [
                (k, "ey2a-counts-one-frame"),
                (10, "ey2a-counts-one-frame"),
                (answer + 1, "ey2a-count-one-over"),
                (answer - 1, "ey2a-count-one-short"),
                (k + 1, "ey2a-counts-one-frame"),
                (20 - k, "ey2a-counts-empty-boxes"),
            ]
            working = [f"The full frame is 10. 10 and {k} more is {answer}."]
        else:
            a = rng.randint(5, 9)
            b = rng.randint(10 - a, 9)  # enough on the second frame to fill the first
            frames = [ten_frame_boxes(a), ten_frame_boxes(b)]
            fig = counters(layout="ten_frame", alt="two ten frames", shape=shape, frames=frames)
            answer = 10 - a
            text = rng.choice(
                (
                    "{n} moves {w} from the right frame to fill the left frame. How many move?",
                    "How many {w} move from the right frame to fill the left frame?",
                )
            )
            text = text.format(n=name, w=word)
            cands = [
                (a, "ey2a-counts-filled-boxes"),
                (b, "ey2a-counts-one-frame"),
                (answer + 1, "ey2a-count-one-over"),
                (answer - 1, "ey2a-count-one-short"),
                (10 - b, "ey2a-counts-empty-boxes"),
                (a + b, "ey2a-counts-all-kinds"),
            ]
            working = [f"The left frame has {a}. It needs 10 - {a} = {answer} more."]
    elif tier == "hard":
        a = rng.randint(4, 9)
        b = rng.randint(max(2, 11 - a), 9)  # a + b is 11 to 18: bridges ten, and neither frame is full
        frames = [ten_frame_boxes(a, rng=rng), ten_frame_boxes(b, rng=rng)]
        fig = counters(
            layout="ten_frame", alt="two ten frames with counters in scattered boxes", shape=shape, frames=frames
        )
        if rng.random() < 0.5:
            answer = a + b
            text = rng.choice(
                ("How many {w} are on both ten frames?", "{n} counts the {w} on both frames. How many in all?")
            )
            text = text.format(n=name, w=word)
            cands = [
                (max(a, b), "ey2a-counts-one-frame"),
                (answer + 1, "ey2a-count-one-over"),
                (answer - 1, "ey2a-count-one-short"),
                (20 - answer, "ey2a-counts-empty-boxes"),
                (10 + min(a, b), "ey2a-assumes-a-full-frame"),
                (answer + 2, "ey2a-count-one-over"),
            ]
            working = [f"{a} on one frame and {b} on the other: {a} + {b} = {answer}."]
        else:
            answer = 20 - (a + b)
            text = rng.choice(
                ("How many more {w} fill both ten frames?", "{n} wants both frames full. How many more {w}?")
            )
            text = text.format(n=name, w=word)
            cands = [
                (a + b, "ey2a-counts-filled-boxes"),
                (answer + 1, "ey2a-count-one-over"),
                (answer - 1, "ey2a-count-one-short"),
                (10 - max(a, b), "ey2a-counts-one-frame"),
                (answer + 2, "ey2a-count-one-over"),
                (20 - max(a, b), "ey2a-counts-one-frame"),
            ]
            working = [f"{a} + {b} = {a + b} are on the frames. 20 - {a + b} = {answer} more fill them."]
    else:
        raise ValueError(f"unknown tier {tier!r}")
    figures.assert_no_leak(fig, [str(answer)])
    return number_question(
        text=text, answer=answer, distractors=tagged_numbers(answer, cands, low=1, high=30), working=working, figure=fig
    )


def _read(q) -> str:
    """Re-derive the answer from the drawn frames and the stem alone."""
    frames = q.figure["frames"]
    text = q.text
    filled = [_filled(f) for f in frames]
    if len(frames) == 1:
        return str(10 - filled[0]) if "fill" in text or "full" in text else str(filled[0])
    if "move" in text:
        return str(10 - filled[0])
    if "fill both" in text or "both frames full" in text:
        return str(20 - sum(filled))
    return str(sum(filled))


figures.register_reader("ey2a_ten_frame_numbers", _read)
