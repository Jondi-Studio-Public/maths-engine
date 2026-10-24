"""One-to-one displays of data, and comparing counts: a tally table or a picture graph where one picture is one thing.

Descriptor: AC9M1ST02 (represent data for a categorical variable by creating a display using one-to-one
correspondence, and compare using frequencies).

The picture is either a ``tally`` table (counts hidden, the learner reads the marks) or a ``picture_graph`` whose key is
1, so each picture stands for exactly one thing. The stem names the categories but never states a count; every answer
is a number. Difficulty escalates in what is done with the counts:

- easy: three or four rows; how many in one row.
- medium: four rows; how many more in one row than another, or how many in two rows altogether.
- hard: five rows; how many altogether in every row, or how many fewer the row with the least has than the row with the
  most (the extremes have to be found first).
"""

from __future__ import annotations

from .. import figures
from ..core import attempts, generator
from ..figure_kinds.data import SYMBOLS, picture_graph
from ..figure_kinds.tally import tally
from ..slots import NAMES
from ._ey_tally_kit import (
    PLURAL,
    display_rows,
    gates_as_ones,
    number_question,
    strokes_only,
    tagged_numbers,
)

_PRESETS = {"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")}

_EASY = (
    "How many {pl} are there?",
    "{n} counts the {pl}. How many are there?",
)
_MORE = (
    "How many more {a} than {b} are there?",
    "{n} compares two rows. How many more {a} than {b}?",
)
_BOTH = (
    "How many {a} and {b} are there altogether?",
    "{n} adds the {a} and the {b}. How many altogether?",
)
_ALL = (
    "How many are there altogether?",
    "{n} counts every row. How many altogether?",
)
_FEWER = (
    "How many fewer does the least row have than the most row?",
    "{n} compares the least and the most. How many fewer does the least have?",
)


def _draw_rows(rng, tier: str) -> list[tuple[str, int]]:
    n = rng.choice((3, 4)) if tier == "easy" else 4 if tier == "medium" else 5
    labels = [PLURAL[k] for k in rng.sample(sorted(PLURAL), n)]
    for _ in attempts(500, "distinct counts"):
        counts = [rng.randint(1, 9 if tier == "easy" else 10) for _ in range(n)]
        if tier != "easy" and len(set(counts)) < n:
            continue
        if tier == "hard" and max(counts) - min(counts) < 3:
            continue
        return list(zip(labels, counts))
    raise AssertionError("unreachable")


def _figure(rng, rows: list[tuple[str, int]]) -> dict:
    if rng.random() < 0.5:
        return tally(alt="a tally table of what was counted", rows=rows)
    return picture_graph(
        categories=[lb for lb, _ in rows],
        symbols=[n for _, n in rows],
        each=1,
        symbol=rng.choice(SYMBOLS),
        alt="a picture graph of what was counted",
    )


@generator("ey_one_to_one_data", level=1, presets=_PRESETS)
def ey_one_to_one_data(rng, tier="easy"):
    if tier not in _PRESETS:
        raise ValueError(f"unknown tier {tier!r}")
    name = rng.choice(NAMES)
    rows = _draw_rows(rng, tier)
    fig = _figure(rng, rows)
    labels = [lb for lb, _ in rows]
    counts = dict(rows)
    total = sum(counts.values())
    is_tally = fig["kind"] == "tally"
    if tier == "easy":
        i = rng.randrange(len(rows))
        a = labels[i]
        answer = counts[a]
        text = rng.choice(_EASY).format(pl=a, n=name)
        near = counts[labels[i - 1]] if i else counts[labels[1]]
        cands = [(near, "eyt-neighbouring-row"), (total, "eyt-counts-all-kinds")]
        if is_tally and answer >= 5:
            cands += [(strokes_only(answer), "eyt-skips-the-slash"), (gates_as_ones(answer), "eyt-gate-counted-as-one")]
        cands += [
            (answer + 1, "eyt-count-one-over"),
            (answer - 1, "eyt-count-one-short"),
            (answer + 2, "eyt-count-one-over"),
        ]
        working = [f"Read the {a} row. There {'is' if answer == 1 else 'are'} {answer}."]
    elif tier == "medium":
        a, b, c = rng.sample(labels, 3)
        if rng.random() < 0.5:
            a, b = (a, b) if counts[a] > counts[b] else (b, a)
            answer = counts[a] - counts[b]
            text = rng.choice(_MORE).format(a=a, b=b, n=name)
            cands = [
                (counts[a], "eyt-reports-one-group"),
                (counts[a] + counts[b], "eyt-adds-instead-of-compares"),
                (counts[b], "eyt-other-group"),
                (answer + 1, "eyt-count-one-over"),
                (answer - 1, "eyt-count-one-short"),
            ]
            working = [f"{a}: {counts[a]}. {b}: {counts[b]}. {counts[a]} - {counts[b]} = {answer}."]
        else:
            answer = counts[a] + counts[b]
            text = rng.choice(_BOTH).format(a=a, b=b, n=name)
            cands = [
                (counts[a], "eyt-reports-one-group"),
                (total, "eyt-counts-all-kinds"),
                (counts[a] + counts[c], "eyt-other-group"),
                (answer + 1, "eyt-count-one-over"),
                (answer - 1, "eyt-count-one-short"),
            ]
            working = [f"{a}: {counts[a]}. {b}: {counts[b]}. {counts[a]} + {counts[b]} = {answer}."]
    elif tier == "hard":
        order = sorted(labels, key=counts.get)
        lo, hi = counts[order[0]], counts[order[-1]]
        if rng.random() < 0.5:
            answer = total
            text = rng.choice(_ALL).format(n=name)
            cands = [
                (total - lo, "eyt-misses-a-group"),
                (total - hi, "eyt-misses-a-group"),
                (len(rows), "eyt-counts-the-rows"),
                (answer + 1, "eyt-count-one-over"),
                (answer - 1, "eyt-count-one-short"),
            ]
            working = [" + ".join(str(counts[lb]) for lb in labels) + f" = {answer}."]
        else:
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
            working = [f"The most is {hi} and the least is {lo}. {hi} - {lo} = {answer}."]
    else:
        raise ValueError(f"unknown tier {tier!r}")
    figures.assert_no_leak(fig, [str(answer)])
    return number_question(
        text=text, answer=answer, distractors=tagged_numbers(answer, cands, low=0, high=60), working=working, figure=fig
    )


def _read(q) -> str:
    """Re-derive the answer from the drawn rows (marks or pictures) and the stem alone."""
    rows = display_rows(q.figure)
    text = q.text
    named = [lb for _, lb in sorted((text.index(lb), lb) for lb in rows if lb in text)]
    if "fewer" in text:
        return str(max(rows.values()) - min(rows.values()))
    if "more" in text and "than" in text:
        return str(rows[named[0]] - rows[named[1]])
    if "altogether" in text and len(named) >= 2:
        return str(rows[named[0]] + rows[named[1]])
    if "altogether" in text:
        return str(sum(rows.values()))
    return str(rows[named[0]])


figures.register_reader("ey_one_to_one_data", _read)
