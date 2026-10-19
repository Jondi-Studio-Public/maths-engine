"""Tally tables: read a tally, find the row that matches a count, and check a tally against the things it counts.

Descriptor: AC9M2ST01 (acquire data for categorical variables; sort into categories and display using lists and
tables).

The picture is a ``tally`` figure with the counts hidden: rows of tally marks in gates of five. Every question is
multiple choice, and none states a count that the marks show. Difficulty escalates in what the marks are used for:

- easy: read one row; choose how many it shows (up to 14).
- medium: choose the row that shows a stated count (up to 29). Wrong rows are built from the ways a tally is
  misread: the strokes counted without the slash, each gate counted as one, one more or one fewer.
- hard: the figure also draws the things that were sorted, above a tally of them, and one row of the tally is wrong.
  Choose that row, which takes counting each kind of thing and reading each row of marks.
"""

from __future__ import annotations

import re

from .. import figures
from ..answers import Distractor
from ..core import attempts, check, generator
from ..figure_kinds.tally import tally
from ..slots import NAMES
from ._ey_tally_kit import (
    ICONS,
    PLURAL,
    gates_as_ones,
    item_counts,
    label_question,
    row_option,
    scatter_objects,
    strokes_only,
    tagged_labels,
    tagged_numbers,
    tally_counts,
)

_PRESETS = {"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")}
_SINGULAR = {v: k for k, v in PLURAL.items()}

_EASY = (
    "How many {pl} are there?",
    "{n} reads the tally. How many {pl}?",
)
_MEDIUM = (
    "Which row shows {k}?",
    "{n} looks for the row that shows {k}. Which row is it?",
    "Look at the tally. Which row shows {k}?",
)
_HARD = (
    "{n} made a tally of the things. Which row is wrong?",
    "One row of the tally is wrong. Which row is it?",
    "Count the things and read the tally. Which row is wrong?",
)


def _labels(rng, n: int) -> list[str]:
    return [PLURAL[k] for k in rng.sample(sorted(PLURAL), n)]


def _build_easy(rng, name):
    n = rng.choice((3, 4))
    labels = _labels(rng, n)
    for _ in attempts(300, "distinct counts with a gate to read"):
        counts = [rng.randint(1, 14) for _ in range(n)]
        if len(set(counts)) == n and max(counts) >= 5:
            break
    rows = list(zip(labels, counts))
    i = rng.randrange(n)
    a = labels[i]
    answer = counts[i]
    fig = tally(alt="a tally table of what was counted", rows=rows)
    near = counts[i - 1] if i else counts[1]
    cands = [(near, "eyt-neighbouring-row")]
    if answer >= 5:
        cands += [(strokes_only(answer), "eyt-skips-the-slash"), (gates_as_ones(answer), "eyt-gate-counted-as-one")]
    cands += [
        (answer + 1, "eyt-count-one-over"),
        (answer - 1, "eyt-count-one-short"),
        (answer + 2, "eyt-count-one-over"),
    ]
    text = rng.choice(_EASY).format(pl=a, n=name)
    distractors = tagged_numbers(answer, cands, low=0, high=30)[:3]  # always three wrong counts
    working = [f"Count the {a} row: each gate is 5. There are {answer}."]
    return fig, text, str(answer), distractors, working


def _misread_rows(k: int) -> list[tuple[int, str]]:
    """Counts of rows that a learner who misreads the tally would take for ``k``."""
    out = []
    for g in range(1, 6):
        r = k - 4 * g
        if 0 <= r <= 4 and 5 * g + r != k:
            out.append((5 * g + r, "eyt-skips-the-slash"))
    for g in range(1, 7):
        r = k - g
        if 0 <= r <= 4 and 5 * g + r != k and 5 * g + r <= 30:
            out.append((5 * g + r, "eyt-gate-counted-as-one"))
    out += [(k + 1, "eyt-count-one-over"), (k - 1, "eyt-count-one-short")]
    return [(c, t) for c, t in out if 1 <= c <= 29]


def _build_medium(rng, name):
    labels = _labels(rng, 4)
    for _ in attempts(500, "a count with misreadable rows"):
        k = rng.randint(6, 28)
        pool = _misread_rows(k)
        rng.shuffle(pool)
        chosen: dict[int, str | None] = {}
        for c, t in pool:
            if c not in chosen and len(chosen) < 3:
                chosen[c] = t
        if len(chosen) < 2:
            continue
        spare = [c for c in range(2, 30) if c != k and c not in chosen]
        rng.shuffle(spare)
        for c in spare[: 3 - len(chosen)]:
            chosen[c] = None
        counts = [k, *chosen]
        order = list(range(4))
        rng.shuffle(order)
        break
    rows = [(labels[j], counts[order[j]]) for j in range(4)]
    key_label = row_option(next(lb for lb, c in rows if c == k))
    fig = tally(alt="a tally table of what was counted", rows=rows)
    text = rng.choice(_MEDIUM).format(k=k, n=name)
    wrong = [(row_option(lb), chosen[c]) for lb, c in rows if c != k]
    distractors = tagged_labels(key_label, [(lb, t) for lb, t in wrong], n=3)
    return fig, text, key_label, distractors, [f"Count each row. {key_label} shows {k}."]


def _build_hard(rng, name):
    kinds = rng.sample(ICONS, 4)  # four rows, so there are always three right rows to choose from besides the wrong one
    for _ in attempts(500, "counts that fit on the page"):
        true = {k: rng.randint(1, 7) for k in kinds}
        if sum(true.values()) <= 20:
            break
    wrong_kind = rng.choice(kinds)
    shown = dict(true)
    for _ in attempts(50, "a wrong count"):
        d = rng.choice((-2, -1, 1, 2))
        if shown[wrong_kind] + d >= 1:
            shown[wrong_kind] += d
            break
    rows = [(PLURAL[k], shown[k]) for k in kinds]
    rng.shuffle(rows)
    fig = tally(alt="things to sort, and a tally of them", rows=rows, items=scatter_objects(rng, true))
    key_label = row_option(PLURAL[wrong_kind])
    text = rng.choice(_HARD).format(n=name)
    others = [(row_option(PLURAL[k]), "eyt-picks-a-matching-row") for k in kinds if k != wrong_kind]
    distractors = tagged_labels(key_label, others, n=3)
    working = [f"Count the {wrong_kind} pictures: {true[wrong_kind]}. The tally shows {shown[wrong_kind]}."]
    return fig, text, key_label, distractors, working


@generator("ey_tally_tables", level=2, presets=_PRESETS)
def ey_tally_tables(rng, tier="easy"):
    if tier not in _PRESETS:
        raise ValueError(f"unknown tier {tier!r}")
    name = rng.choice(NAMES)
    build = {"easy": _build_easy, "medium": _build_medium, "hard": _build_hard}[tier]
    fig, text, answer, distractors, working = build(rng, name)
    figures.assert_no_leak(fig, [answer, *(d.text for d in distractors)])
    check(all(isinstance(d, Distractor) for d in distractors))
    return label_question(text=text, answer=answer, distractors=distractors, working=working, figure=fig)


def _read(q) -> str:
    """Re-derive the answer from the marks (and the pictures, on hard) and the stem alone."""
    rows = tally_counts(q.figure)
    text = q.text
    if q.figure["items"]:
        have = item_counts(q.figure)
        bad = [lb for lb, c in rows.items() if have.get(_SINGULAR[lb], 0) != c]
        check(len(bad) == 1, "exactly one row disagrees with the pictures")
        return row_option(bad[0])
    if "shows" in text:
        k = int(re.search(r"shows (\d+)", text).group(1))
        hit = [lb for lb, c in rows.items() if c == k]
        check(len(hit) == 1, "exactly one row shows the count")
        return row_option(hit[0])
    named = [lb for lb in rows if lb in text]
    return str(rows[named[0]])


figures.register_reader("ey_tally_tables", _read)
