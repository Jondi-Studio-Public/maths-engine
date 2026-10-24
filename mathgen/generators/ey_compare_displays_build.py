"""Comparing two displays of the same data: a tally table with a picture column beside it.

Descriptor: AC9M2ST02 (compare different representations of the same data).

The picture is a ``tally`` figure with a ``pictures`` column: each row is shown twice, as tally marks and as pictures, and
the two displays agree everywhere except one row. Difficulty escalates in how the two displays relate:

- easy: each picture stands for 1; choose the row where the tally and the pictures disagree.
- medium: each picture stands for 2, so the pictures must be doubled before they are compared with the tally. The wrong
  row has pictures equal to its tally, so a learner who forgets the key finds that row "right" and every other row wrong.
- hard: each picture stands for 2 and the stem names the row that is short; type how many more pictures it needs, which
  means halving the tally and taking away the pictures already drawn.
"""

from __future__ import annotations

from .. import figures
from ..core import attempts, check, generator
from ..figure_kinds.data import SYMBOLS
from ..figure_kinds.tally import tally
from ..slots import NAMES
from ._ey_tally_kit import PLURAL, label_question, row_option, tagged_labels, tagged_numbers

_PRESETS = {"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")}
_ALT = "a tally table and a column of pictures for the same rows"

_WRONG = (
    "One row does not match. Which row is it?",
    "{n} checks the tally and the pictures. Which row is wrong?",
    "The tally and the pictures disagree in one row. Which row is it?",
)
_SHORT = (
    "The {lb} row has too few pictures. How many more should it have?",
    "{n} checks the {lb} row. How many more pictures does it need?",
)


def _rows(rng, tier: str):
    """``(labels, counts, pictures, wrong index, each)`` where only the row at the wrong index disagrees."""
    n = 4  # the question offers every row, so four rows always give four options
    labels = [PLURAL[k] for k in rng.sample(sorted(PLURAL), n)]
    each = 1 if tier == "easy" else 2
    for _ in attempts(1000, "two displays with one wrong row"):
        counts = [rng.randint(1, 9) * each for _ in range(n)]
        w = rng.randrange(n)
        pics = [c // each for c in counts]
        if tier == "easy":
            pics[w] = counts[w] + rng.choice((-2, -1, 1, 2))
            if not 0 <= pics[w] <= 10:
                continue
        elif tier == "medium":
            if counts[w] > 10:  # the wrong row draws one picture per tally mark
                continue
            pics[w] = counts[w]
        else:
            pics[w] = counts[w] // each - rng.randint(1, min(3, counts[w] // each))
        if len(set(counts)) < n or pics[w] == counts[w] // each or not 0 <= pics[w] <= 10:
            continue
        return labels, counts, pics, w, each
    raise AssertionError("unreachable")


@generator("ey_compare_displays", level=2, presets=_PRESETS)
def ey_compare_displays(rng, tier="easy"):
    if tier not in _PRESETS:
        raise ValueError(f"unknown tier {tier!r}")
    name = rng.choice(NAMES)
    labels, counts, pics, w, each = _rows(rng, tier)
    fig = tally(
        alt=_ALT,
        rows=list(zip(labels, counts)),
        pictures={"symbol": rng.choice(SYMBOLS), "each": each, "counts": pics},
    )
    bad = labels[w]
    if tier == "hard":
        answer = counts[w] // each - pics[w]
        text = rng.choice(_SHORT).format(lb=bad, n=name)
        cands = [
            (counts[w] - pics[w], "eyt-ignores-the-picture-key"),
            (counts[w] // each, "eyt-reports-the-whole-row"),
            (pics[w], "eyt-reports-one-group"),
            (answer * 2, "eyt-ignores-the-picture-key"),
            (answer + 1, "eyt-count-one-over"),
            (answer - 1, "eyt-count-one-short"),
        ]
        distractors = tagged_numbers(answer, cands, low=0, high=30)[:3]  # always four options
        working = [
            f"The tally shows {counts[w]}. Halved, that is {counts[w] // each}. {counts[w] // each} - {pics[w]} = {answer}."
        ]
        figures.assert_no_leak(fig, [str(answer), *(d.text for d in distractors)])
        return label_question(text=text, answer=str(answer), distractors=distractors, working=working, figure=fig)
    text = rng.choice(_WRONG).format(n=name)
    tag = "eyt-picks-a-matching-row" if tier == "easy" else "eyt-ignores-the-picture-key"
    distractors = tagged_labels(row_option(bad), [(row_option(lb), tag) for i, lb in enumerate(labels) if i != w], n=3)
    working = [f"The {bad} tally shows {counts[w]}. Its pictures stand for {pics[w] * each}."]
    return label_question(text=text, answer=row_option(bad), distractors=distractors, working=working, figure=fig)


def _read(q) -> str:
    """Re-derive the answer from the tally marks, the pictures and the key alone."""
    fig = q.figure
    each = fig["pictures"]["each"]
    rows = [(r["label"], r["count"], p) for r, p in zip(fig["rows"], fig["pictures"]["counts"])]
    if "how many" in q.text.lower():
        ((lb, count, p),) = [r for r in rows if r[0] in q.text]
        return str(count // each - p)
    bad = [lb for lb, count, p in rows if p * each != count]
    check(len(bad) == 1, "exactly one row disagrees")
    return row_option(bad[0])


figures.register_reader("ey_compare_displays", _read)
