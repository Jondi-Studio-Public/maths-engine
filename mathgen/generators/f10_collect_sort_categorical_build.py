"""Generates `f10-collect-sort-compare-categorical-data` questions (Foundation to Year 5 statistics):
sort given data into categories, count each category and fill the blank cells of a table.
`answer: table`; the grader reads the blanks as integers.

Scope: the data are always supplied as text (a list of objects, tally marks, a described survey, a list
of whole-number readings). The learner never poses the question or draws a graph; they sort, count and
compare. Every blank cell is determined by the supplied data, and the blanks differ in position from
question to question, so the answer cannot be guessed from the layout.

Ladder (reasoning escalation, not size):
  easy:   one categorical variable. A list of objects in jumbled order, or a tally-mark column, is sorted
          into categories and counted. Tally questions need the group-of-five reading; list questions
          need an organised count of a jumbled list.
  medium: a described survey (a bare list of votes, or named students with their choices). Some counts
          are blank and so are some cells of a comparison column, "fewer votes than the most popular",
          so the learner counts and then compares category counts.
  hard:   discrete numerical data (siblings, shoe sizes, goals ...). The frequency table has a row for
          every value from the lowest to the highest (a value nobody gave can have frequency 0), a
          Total row, and a derived row that adds the frequencies of the values a phrase such as "3 or
          more" picks out.
"""

from __future__ import annotations

import random
from collections import Counter

from mathgen.core import check, generator, phrase
from mathgen.slots import NAMES

from ._f10_table_kit import build_answer, finish, grid_text, pick_blanks

TIERS = ("easy", "medium", "hard")

# ---------------------------------------------------------------------------
# Banks
# ---------------------------------------------------------------------------

_EASY = [
    dict(
        header="Colour",
        topic="counter colours",
        cats=["Red", "Blue", "Green", "Yellow", "Purple"],
        intro="Counters were taken from a bag one at a time. Their colours, in the order they came out, were: {items}.",
    ),
    dict(
        header="Pet",
        topic="pets",
        cats=["Dog", "Cat", "Fish", "Bird", "Rabbit"],
        intro="Some children each named the pet they have at home. In the order they spoke: {items}.",
    ),
    dict(
        header="Fruit",
        topic="pieces of fruit",
        cats=["Apple", "Banana", "Pear", "Orange", "Grape"],
        intro="A fruit bowl was emptied one piece at a time. The pieces, in order, were: {items}.",
    ),
    dict(
        header="Shape",
        topic="shapes",
        cats=["Circle", "Square", "Triangle", "Star", "Heart"],
        intro="Shape blocks were taken from a tub one at a time. The shapes, in order, were: {items}.",
    ),
    dict(
        header="Vehicle",
        topic="vehicles",
        cats=["Car", "Bus", "Bike", "Truck", "Tram"],
        intro="A class noted each vehicle that passed the school gate. In order, they were: {items}.",
    ),
    dict(
        header="Toy",
        topic="toys",
        cats=["Ball", "Doll", "Kite", "Block", "Train"],
        intro="Toys were tidied from the floor one at a time. The toys, in order, were: {items}.",
    ),
]

_MED = [
    dict(header="Fruit", topic="fruit", cats=["Apple", "Banana", "Pear", "Orange", "Grape"]),
    dict(header="Sport", topic="sport", cats=["Soccer", "Netball", "Swimming", "Tennis", "Cricket"]),
    dict(header="Lunch", topic="lunch", cats=["Sandwich", "Pasta", "Sushi", "Salad", "Wrap"]),
    dict(header="Pet", topic="pet", cats=["Dog", "Cat", "Rabbit", "Fish", "Bird"]),
    dict(header="Colour", topic="colour", cats=["Red", "Blue", "Green", "Yellow", "Orange"]),
    dict(header="Season", topic="season", cats=["Summer", "Autumn", "Winter", "Spring"]),
]

# value range, data introduction, label for the unit in a derived row
_HARD = [
    dict(
        header="Number of siblings",
        lo=0,
        hi=5,
        intro="Students in a class were asked how many brothers and sisters they have. The answers were: {data}.",
    ),
    dict(
        header="Shoe size",
        lo=3,
        hi=10,
        intro="The shoe sizes of the members of a netball club were: {data}.",
    ),
    dict(
        header="Number of pets",
        lo=0,
        hi=5,
        intro="Each student in a class said how many pets live at their home. The answers were: {data}.",
    ),
    dict(
        header="Goals scored",
        lo=0,
        hi=6,
        intro="A soccer team recorded the goals it scored in each game this season: {data}.",
    ),
    dict(
        header="Books read",
        lo=0,
        hi=6,
        intro="Students counted the books they read last month. The answers were: {data}.",
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tally(n: int) -> str:
    groups, rest = divmod(n, 5)
    parts = ["||||/"] * groups
    if rest:
        parts.append("|" * rest)
    return " ".join(parts)


def _read_tally(text: str) -> int:
    """The number a tally string shows: each ||||/ is 5, each other | is 1."""
    return 5 * text.count("/") + text.count("|") - 4 * text.count("/")


def _jumble(rng: random.Random, items: list[str]) -> list[str]:
    """`items` shuffled so equal items are not all adjacent (a sorted list would be a pre-sorted answer)."""
    for _ in range(30):
        rng.shuffle(items)
        runs = sum(1 for a, b in zip(items, items[1:]) if a != b)
        if runs >= len(set(items)) + 1:
            break
    return items


def _counts(rng: random.Random, n: int, lo: int, hi: int, total_lo: int, total_hi: int) -> list[int]:
    for _ in range(500):
        counts = [rng.randint(lo, hi) for _ in range(n)]
        if total_lo <= sum(counts) <= total_hi and len(set(counts)) > 1:
            return counts
    raise ValueError("no counts in range")  # pragma: no cover - ranges are wide enough


def _off_by_one(blanks: dict, rng: random.Random) -> list[tuple[str, dict]]:
    """Two off-by-one edits (one count too few, one too many) on different blanks where possible."""
    keys = sorted(blanks)
    rng.shuffle(keys)
    out = []
    if keys:
        out.append(("misses-an-item", {keys[0]: blanks[keys[0]] - 1}))
        k2 = keys[1] if len(keys) > 1 else keys[0]
        out.append(("double-counts-an-item", {k2: blanks[k2] + 1}))
    return out


def _swap_two(blanks: dict, rng: random.Random) -> list[tuple[str, dict]]:
    keys = sorted(blanks)
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1 :] if blanks[a] != blanks[b]]
    if not pairs:
        return []
    a, b = rng.choice(pairs)
    return [("swaps-two-categories", {a: blanks[b], b: blanks[a]})]


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------


def _easy(rng: random.Random) -> dict:
    ctx = rng.choice(_EASY)
    tally_mode = rng.random() < 0.5
    ncat = rng.randint(3, 4)
    cats = rng.sample(ctx["cats"], ncat)
    if tally_mode:
        counts = _counts(rng, ncat, 2, 12, 12, 30)
        if max(counts) < 5:
            counts[rng.randrange(ncat)] = rng.randint(5, 12)
        columns = [ctx["header"], "Tally", "Count"]
        rows = [[c, _tally(n), n] for c, n in zip(cats, counts)]
        count_col = 2
        intro = (
            f"Tally marks were used to record the {ctx['topic']}. Each ||||/ is one group of 5 and each | is a "
            "single one."
        )
        data_check = [(c, _read_tally(row[1])) for c, row in zip(cats, rows)]
    else:
        counts = _counts(rng, ncat, 1, 7, 9, 16)
        items = [c.lower() for c, n in zip(cats, counts) for _ in range(n)]
        items = _jumble(rng, items)
        columns = [ctx["header"], "Count"]
        rows = [[c, n] for c, n in zip(cats, counts)]
        count_col = 1
        intro = ctx["intro"].format(items=", ".join(items))
        tally = Counter(items)
        data_check = [(c, tally[c.lower()]) for c in cats]
    check(data_check == [(c, n) for c, n in zip(cats, counts)], "sorted counts disagree with the supplied data")
    blank_cells = pick_blanks(rng, [(r, count_col) for r in range(ncat)], 2, ncat)
    answer = build_answer(columns, rows, blank_cells)
    kinds = "tally marks" if tally_mode else "objects"
    text = f"{intro} Sort the {kinds} into the table. Fill in each ?. {grid_text(answer)}"
    blanks = {k: v for k, v in answer["blanks"].items()}
    edits = _swap_two(blanks, rng) + _off_by_one(blanks, rng)
    if tally_mode:
        # Counts each group of five as one mark, then adds the single marks.
        as_one = {key: v // 5 + v % 5 for key, v in blanks.items()}
        edits.insert(0, ("counts-each-group-as-one", {key: val for key, val in as_one.items() if val != blanks[key]}))
    answer_text, distractors, notes = finish(answer, edits)
    working = [f"{c}: {n}" for c, n in zip(cats, counts)] + [f"Total: {sum(counts)}"]
    return dict(
        text=text, answer=answer, answer_text=answer_text, working=working, distractors=distractors, option_notes=notes
    )


def _medium(rng: random.Random) -> dict:
    ctx = rng.choice(_MED)
    ncat = rng.randint(4, min(5, len(ctx["cats"])))
    cats = rng.sample(ctx["cats"], ncat)
    for _ in range(500):
        counts = [rng.randint(1, 9) for _ in range(ncat)]
        top = max(counts)
        if counts.count(top) == 1 and 14 <= sum(counts) <= 24:
            break
    else:  # pragma: no cover - ranges are wide enough
        raise ValueError("no survey counts in range")
    top_i = counts.index(top)
    votes = [c.lower() for c, n in zip(cats, counts) for _ in range(n)]
    votes = _jumble(rng, votes)
    topic = ctx["topic"]
    if rng.random() < 0.5 or len(votes) > len(NAMES):
        intro = phrase(
            rng,
            [
                "A class voted for their favourite {topic}. The votes, in the order they were cast, were: {votes}.",
                "Each student in a group chose a favourite {topic}. The choices, in the order they were given, "
                "were: {votes}.",
            ],
            topic=topic,
            votes=", ".join(votes),
        )
    else:
        names = rng.sample(list(NAMES), len(votes))
        intro = (
            f"Each student in a group chose a favourite {topic}: "
            + "; ".join(f"{n} chose {v}" for n, v in zip(names, votes))
            + "."
        )
    columns = [ctx["header"], "Votes", f"Fewer votes than {cats[top_i]}"]
    rows = [[c, n, top - n] for c, n in zip(cats, counts)]
    survey = Counter(votes)
    check([survey[c.lower()] for c in cats] == counts, "survey counts disagree with the supplied votes")
    vote_cells = [(r, 1) for r in range(ncat)]
    diff_cells = [(r, 2) for r in range(ncat) if r != top_i]
    must = [rng.choice(vote_cells), rng.choice(diff_cells)]
    blank_cells = pick_blanks(rng, vote_cells + diff_cells, 3, 5, must=must)
    answer = build_answer(columns, rows, blank_cells)
    check(all(rows[r][2] == max(counts) - rows[r][1] for r in range(ncat)), "comparison column inconsistent")
    text = (
        f"{intro} Complete the table. The last column is how many fewer votes than the most popular "
        f"{ctx['header'].lower()}. Fill in each ?. {grid_text(answer)}"
    )
    blanks = dict(answer["blanks"])
    edits: list[tuple[str, dict]] = []
    vote_keys = sorted(k for k in blanks if k.endswith(",1"))
    diff_keys = sorted(k for k in blanks if k.endswith(",2"))
    # A miscounted category carries its error into its own comparison cell when that cell is blank too.
    for vk in vote_keys:
        r = vk.split(",")[0]
        change = {vk: blanks[vk] + 1}
        dk = f"{r},2"
        if dk in blanks and int(r) != top_i:
            change[dk] = blanks[dk] - 1
        edits.append(("miscount-carries-into-comparison", change))
        break
    if diff_keys:
        dk = diff_keys[0]
        r = int(dk.split(",")[0])
        edits.append(("reports-the-count-as-the-difference", {dk: counts[r]}))
        edits.append(("measures-from-the-least-popular", {dk: counts[r] - min(counts)}))
    edits += _off_by_one(blanks, rng)
    answer_text, distractors, notes = finish(answer, edits)
    working = [
        f"{c}: {n} {'vote' if n == 1 else 'votes'}, {top - n} fewer than {cats[top_i]}" for c, n in zip(cats, counts)
    ]
    return dict(
        text=text, answer=answer, answer_text=answer_text, working=working, distractors=distractors, option_notes=notes
    )


def _hard(rng: random.Random) -> dict:
    ctx = rng.choice(_HARD)
    k = rng.randint(5, 6)
    span = ctx["hi"] - ctx["lo"] + 1
    start = ctx["lo"] + rng.randint(0, span - k)
    values = list(range(start, start + k))
    for _ in range(800):
        freqs = [rng.randint(1, 9) for _ in range(k)]
        if rng.random() < 0.4:
            freqs[rng.randint(1, k - 2)] = 0
        if 18 <= sum(freqs) <= 30 and len(set(freqs)) > 2:
            break
    else:  # pragma: no cover - ranges are wide enough
        raise ValueError("no frequencies in range")
    data = [str(v) for v, f in zip(values, freqs) for _ in range(f)]
    data = _jumble(rng, data)
    seen = Counter(data)
    check([seen[str(v)] for v in values] == freqs, "frequencies disagree with the supplied data")
    total = len(data)
    # The derived row adds the frequencies of the values a phrase picks out; its boundary value is present.
    boundary_choices = [i for i in range(1, k - 1) if freqs[i] > 0]
    bi = rng.choice(boundary_choices)
    t = values[bi]
    form = rng.choice(["or more", "or fewer", "more than", "fewer than"])
    if form == "or more":
        label, derived, wrong, wrong_id = (
            f"{t} or more",
            sum(freqs[bi:]),
            sum(freqs[bi + 1 :]),
            "leaves-out-the-boundary",
        )
    elif form == "or fewer":
        label, derived, wrong, wrong_id = (
            f"{t} or fewer",
            sum(freqs[: bi + 1]),
            sum(freqs[:bi]),
            "leaves-out-the-boundary",
        )
    elif form == "more than":
        label, derived, wrong, wrong_id = (
            f"More than {t}",
            sum(freqs[bi + 1 :]),
            sum(freqs[bi:]),
            "includes-the-boundary",
        )
    else:
        label, derived, wrong, wrong_id = (
            f"Fewer than {t}",
            sum(freqs[:bi]),
            sum(freqs[: bi + 1]),
            "includes-the-boundary",
        )
    label = label[0].upper() + label[1:]
    check(derived == sum(f for v, f in zip(values, freqs) if _in_form(v, t, form)), "derived row disagrees")
    columns = [ctx["header"], "Frequency"]
    rows = [[str(v), f] for v, f in zip(values, freqs)] + [["Total", total], [label, derived]]
    freq_cells = [(r, 1) for r in range(k)]
    total_cell, derived_cell = (k, 1), (k + 1, 1)
    must = [derived_cell]
    if rng.random() < 0.75:
        must.append(total_cell)
    blank_cells = pick_blanks(rng, freq_cells + [total_cell, derived_cell], len(must) + 2, len(must) + k - 1, must=must)
    # keep at least one given frequency cell so the table is not entirely blank
    check(any(c in freq_cells for c in freq_cells if c not in blank_cells), "no frequency left as a given cell")
    answer = build_answer(columns, rows, blank_cells)
    text = (
        f"{phrase(rng, [ctx['intro']], data=', '.join(data))} Complete the frequency table. Every value from "
        f"{values[0]} to {values[-1]} has a row. The last row adds up the frequencies it names. "
        f"Fill in each ?. {grid_text(answer)}"
    )
    blanks = dict(answer["blanks"])
    tk, dk = f"{k},1", f"{k + 1},1"
    edits: list[tuple[str, dict]] = [(wrong_id, {dk: wrong})]
    if tk in blanks:
        edits.append(("counts-the-rows-not-the-data", {tk: k}))
        edits.append(("adds-the-values-not-the-frequencies", {tk: sum(values)}))
    f_keys = sorted(key for key in blanks if int(key.split(",")[0]) < k)
    if f_keys:
        key = rng.choice(f_keys)
        edits.append(("miscounts-one-value", {key: blanks[key] + 1}))
        if len(f_keys) > 1:
            other = rng.choice([x for x in f_keys if x != key])
            edits.append(("miscounts-one-value", {other: max(blanks[other] - 1, 0)}))
    answer_text, distractors, notes = finish(answer, edits)
    working = [f"{v}: {f}" for v, f in zip(values, freqs)] + [f"Total: {total}", f"{label}: {derived}"]
    return dict(
        text=text, answer=answer, answer_text=answer_text, working=working, distractors=distractors, option_notes=notes
    )


def _in_form(v: int, t: int, form: str) -> bool:
    return {"or more": v >= t, "or fewer": v <= t, "more than": v > t, "fewer than": v < t}[form]


@generator(
    "f10_collect_sort_categorical",
    level=2,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def f10_collect_sort_categorical(rng: random.Random, tier: str = "easy") -> dict:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)
