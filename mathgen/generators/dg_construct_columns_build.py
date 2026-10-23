"""Generates `dg_construct_columns` questions (easy, medium, hard) and the exam chain `dg_construct_columns_chain`.

``construct-column-and-picture-graphs`` (``statistics-probability``, answer ``bars``): the learner builds a column graph
or a picture graph from data and replies with the height of every column (or the number of symbols in every row), in
category order, as ``Name = height``. The `bars` grader reads that reply exactly (a half symbol is ``3.5``); the front
end has no drawing canvas for bars yet, so the typed form is the answer.

* easy: sort a list of responses (a survey, one word per student) into categories and give the column heights.
* medium: a frequency table is drawn as a picture graph where one symbol stands for 2 (or 5) and half a symbol for
  half of that, so the number of symbols is the count divided by the key and an odd count needs a half symbol.
* hard: the data is a number (pets per family, goals per game) given as a long list with one possible value that never
  occurs (a column of height 0), or a table with one count missing that has to be found from the total.
* exam (``dg_construct_columns_chain``): from one list of responses: (a) the column heights, (b) the difference between
  the tallest and the shortest column, (c) the rows of a picture graph with a key of 2, (d) how many more symbols the
  longest row has than the shortest, following through from (b).

Every height is chosen first and the list or table is built from it, then recounted from the text the learner reads.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._dg_kit import bars_dist, bars_fields, numd, numf, numtext, p_bars, p_number

AREA = "construct-column-and-picture-graphs"

MISCONCEPTIONS = [
    (
        "dg-cc-tally-miscount",
        "Miscounts one category when sorting the responses, so one column is a unit too tall or too short.",
        AREA,
    ),
    (
        "dg-cc-categories-swapped",
        "Gives two categories each other's heights, matching a count to the wrong label.",
        AREA,
    ),
    (
        "dg-cc-value-as-height",
        "Uses the data value itself as the height of its column instead of counting how many times it occurs.",
        AREA,
    ),
    (
        "dg-cc-zero-column-skipped",
        "Gives a value that never occurs the height of the next value, skipping the column that should be zero.",
        AREA,
    ),
    (
        "dg-cc-key-ignored",
        "Draws one symbol for every item and ignores the key that says a symbol stands for more than one.",
        AREA,
    ),
    (
        "dg-cc-key-multiplied",
        "Multiplies the count by the key instead of dividing, so the rows are far too long.",
        AREA,
    ),
    (
        "dg-cc-half-symbol-dropped",
        "Drops the half symbol for an odd count, so the row is half a symbol too short.",
        AREA,
    ),
    (
        "dg-cc-half-symbol-whole",
        "Draws a whole symbol where a half symbol is needed, so the row is half a symbol too long.",
        AREA,
    ),
    (
        "dg-cc-missing-total-as-count",
        "Gives the total as the missing count, forgetting to take away the counts already known.",
        AREA,
    ),
    (
        "dg-cc-missing-one-category-left-out",
        "Leaves one known count out of the subtraction when finding the missing count.",
        AREA,
    ),
    ("dg-cc-range-added", "Adds the tallest and shortest column instead of finding how much taller one is.", AREA),
    (
        "dg-cc-second-tallest",
        "Compares the tallest column with the second shortest, or the second tallest with the shortest.",
        AREA,
    ),
    (
        "dg-cc-difference-in-items-not-symbols",
        "Gives the difference in items when the difference in symbols was asked for, forgetting to divide by the key.",
        AREA,
    ),
]

_THEMES = (
    (
        "A class voted for their favourite pet. The votes were: {data}.",
        ("dog", "cat", "rabbit", "fish", "bird", "lizard"),
        "votes",
        "student",
    ),
    (
        "Students were asked how they travel to school. The answers were: {data}.",
        ("walk", "bike", "bus", "car", "scooter", "train"),
        "answers",
        "student",
    ),
    (
        "A canteen recorded the lunch each child ordered: {data}.",
        ("sandwich", "salad", "pasta", "sushi", "wrap", "soup"),
        "orders",
        "child",
    ),
    (
        "A sports teacher noted each student's chosen sport: {data}.",
        ("netball", "soccer", "tennis", "cricket", "swimming", "hockey"),
        "choices",
        "student",
    ),
    (
        "A librarian counted the types of book returned in one morning: {data}.",
        ("mystery", "fantasy", "sport", "history", "science", "comic"),
        "books",
        "book",
    ),
)
_TABLE_THEMES = (
    (
        "The table shows how many students chose each sport for the school carnival.",
        ("Netball", "Soccer", "Tennis", "Cricket", "Swimming", "Hockey"),
        "student",
    ),
    (
        "A canteen counted the drinks sold in one lunch break.",
        ("Water", "Milk", "Juice", "Smoothie", "Iced tea"),
        "drink",
    ),
    (
        "A zoo shop counted the postcards sold of each animal in a day.",
        ("Koala", "Kangaroo", "Emu", "Wombat", "Platypus"),
        "postcard",
    ),
    (
        "A bakery counted the loaves sold in the morning.",
        ("Sourdough", "Wholemeal", "Rye", "Multigrain", "White"),
        "loaf|loaves",
    ),
)


def _noun(word, n):
    """``word`` is ``"drink"`` or ``"loaf|loaves"``: the singular for 1 and the plural otherwise."""
    one, _, many = word.partition("|")
    return one if n == 1 else (many or one + "s")


def _heights(rng, k, lo, hi, distinct=True):
    for _ in attempts(200, "column heights"):
        hs = [rng.randint(lo, hi) for _ in range(k)]
        if (not distinct or len(set(hs)) == k) and max(hs) - min(hs) >= 2:
            return hs
    raise AssertionError  # pragma: no cover


def _list_text(rng, cats, heights):
    items = [c for c, h in zip(cats, heights) for _ in range(h)]
    rng.shuffle(items)
    return ", ".join(items)


def _swapped(heights):
    out = []
    for i in range(len(heights)):
        for j in range(i + 1, len(heights)):
            if heights[i] != heights[j]:
                h = list(heights)
                h[i], h[j] = h[j], h[i]
                out.append(h)
    return out


def _bump(heights, i, d):
    h = list(heights)
    h[i] += d
    return h if h[i] >= 0 else None


def _ask(cats, what):
    return f"Give the {what} for {', '.join(cats[:-1])} and {cats[-1]}, in that order, written like {cats[0]} = 2."


def _recount(text, cats):
    """The count of each category in the text's list (the oracle for the tests and the key)."""
    data = text.split(": ", 1)[1].split(". ", 1)[0].rstrip(".") if ": " in text else text
    words = Counter(w.strip() for w in data.split(","))
    return [words[c] for c in cats]


# ---------------------------------------------------------------------------------------------- easy


_EASY_TASK = (
    "The results are to be shown in a column graph, with one column for each category.",
    "Draw a column graph of the results, one column for each category.",
)


def _easy(rng):
    theme = rng.choice(_THEMES)
    k = rng.randint(3, 5)
    cats = [c.capitalize() for c in rng.sample(theme[1], k)]
    cats_l = [c.lower() for c in cats]
    heights = _heights(rng, k, 2, 8)
    if sum(heights) > 24:
        return None
    data = _list_text(rng, cats_l, heights)
    stem = theme[0].format(data=data)
    check(_recount(stem, cats_l) == heights, "recount of the list")
    cands = [(h, "dg-cc-categories-swapped") for h in _swapped(heights)[:2]]
    cands += [(_bump(heights, rng.randrange(k), rng.choice((-1, 1))), "dg-cc-tally-miscount") for _ in range(3)]
    ds = bars_dist(cats_l, heights, cands)
    if len(ds) < 3:
        return None
    key = bars_fields(cats_l, heights)
    text = f"{stem} {phrase(rng, _EASY_TASK)} {_ask(cats_l, 'height of each column')}"
    return dict(
        text=text,
        answer_kind="bars",
        **key,
        distractors=ds,
        working=[
            f"Count each category: {', '.join(f'{c} {h}' for c, h in zip(cats_l, heights))}.",
            f"That is {sum(heights)} responses in all, and each column's height is its count.",
        ],
    )


# ---------------------------------------------------------------------------------------------- medium


def _medium(rng):
    theme = rng.choice(_TABLE_THEMES)
    k = rng.randint(3, 5)
    cats = rng.sample(theme[1], k)
    keyval = rng.choice((2, 2, 4, 10))
    if keyval == 10:
        counts = [rng.choice((10, 15, 20, 25, 30, 35, 40)) for _ in range(k)]
    else:
        counts = [rng.randint(3, 15) for _ in range(k)]
        if keyval == 4:
            counts = [rng.choice((6, 8, 10, 12, 14, 16, 18, 20, 22, 24)) for _ in range(k)]
    if len(set(counts)) < k:
        return None
    hs = [Fraction(c, keyval) for c in counts]
    if any(h.denominator not in (1, 2) for h in hs) or not any(h.denominator == 2 for h in hs):
        return None
    table = ", ".join(f"{c} {n}" for c, n in zip(cats, counts))
    word = theme[2]
    half = Fraction(keyval, 2)
    stem = (
        f"{theme[0]} {table}. A picture graph shows this data with one symbol standing for {keyval} {_noun(word, keyval)} "
        f"and half a symbol standing for {half} {_noun(word, half)}."
    )
    cands = [
        ([Fraction(c) for c in counts], "dg-cc-key-ignored"),
        ([Fraction(c * keyval) for c in counts], "dg-cc-key-multiplied"),
        ([h if h.denominator == 1 else h - Fraction(1, 2) for h in hs], "dg-cc-half-symbol-dropped"),
        ([h if h.denominator == 1 else h + Fraction(1, 2) for h in hs], "dg-cc-half-symbol-whole"),
    ]
    ds = bars_dist(cats, hs, cands, snap=Fraction(1, 2))
    if len(ds) < 3:
        return None
    key = bars_fields(cats, hs, snap=Fraction(1, 2))
    ask = _ask(cats, "number of symbols in each row").replace(
        "written like", "using .5 for half a symbol, written like"
    )
    working = [
        f"Divide each count by {keyval}: "
        + ", ".join(f"{c} / {keyval} = {key_text(h)}" for c, h in zip(counts, hs))
        + "."
    ]
    working.append("An odd number of halves needs a half symbol, shown as .5.")
    return dict(text=f"{stem} {ask}", answer_kind="bars", **key, distractors=ds, working=working)


def key_text(h):
    return numtext(h)


# ---------------------------------------------------------------------------------------------- hard


_NUMERIC = (
    ("Each family in a street was asked how many pets it has. The answers were: {data}.", "pets", "families"),
    ("A netball coach recorded the goals scored in each of her team's games: {data}.", "goals", "games"),
    ("A teacher recorded the number of books each student read last month: {data}.", "books", "students"),
)


def _hard_numeric(rng):
    theme = rng.choice(_NUMERIC)
    top = rng.randint(4, 6)
    values = list(range(0, top + 1))
    zero_at = rng.randint(1, top - 1)
    counts = [0 if v == zero_at else rng.randint(1, 6) for v in values]
    if sum(counts) > 24 or sum(counts) < 14 or len({c for c in counts if c}) < 3:
        return None
    items = [str(v) for v, c in zip(values, counts) for _ in range(c)]
    rng.shuffle(items)
    cats = [str(v) for v in values]
    stem = theme[0].format(data=", ".join(items))
    cands = [
        ([float(v) for v in values], "dg-cc-value-as-height"),
        (counts[:zero_at] + counts[zero_at + 1 :] + [0], "dg-cc-zero-column-skipped"),
        (_bump(counts, rng.choice([i for i, c in enumerate(counts) if c]), -1), "dg-cc-tally-miscount"),
        (_bump(counts, rng.choice([i for i, c in enumerate(counts) if c]), 1), "dg-cc-tally-miscount"),
    ]
    ds = bars_dist(cats, counts, cands)
    if len(ds) < 3:
        return None
    key = bars_fields(cats, counts)
    ask = f"Make a column graph with one column for each value from 0 to {top} (a value nobody gave has a column of height 0). Give the height of each column, written like {cats[0]} = 2."
    return dict(
        text=f"{stem} {ask}",
        answer_kind="bars",
        **key,
        distractors=ds,
        working=[
            f"Count how many times each value occurs: " + ", ".join(f"{c}: {h}" for c, h in zip(cats, counts)) + ".",
            f"{cats[zero_at]} never occurs, so its column has height 0.",
        ],
    )


def _hard_missing(rng):
    theme = rng.choice(_TABLE_THEMES)
    k = rng.randint(4, 5)
    cats = rng.sample(theme[1], k)
    counts = _heights(rng, k, 3, 14)
    miss = rng.randrange(k)
    total = sum(counts)
    known = [f"{c} {n}" for i, (c, n) in enumerate(zip(cats, counts)) if i != miss]
    stem = f"{theme[0]} In all, {total} {_noun(theme[2], total)} were counted. {', '.join(known)}; the count for {cats[miss]} is not shown."
    others = total - counts[miss]
    cands = [
        (counts[:miss] + [total] + counts[miss + 1 :], "dg-cc-missing-total-as-count"),
        (
            counts[:miss] + [counts[miss] + counts[(miss + 1) % k]] + counts[miss + 1 :],
            "dg-cc-missing-one-category-left-out",
        ),
        (counts[:miss] + [counts[miss] - 1] + counts[miss + 1 :], "dg-cc-tally-miscount"),
        (counts[:miss] + [counts[miss] + 1] + counts[miss + 1 :], "dg-cc-tally-miscount"),
    ]
    ds = bars_dist(cats, counts, cands)
    if len(ds) < 3:
        return None
    key = bars_fields(cats, counts)
    ask = f"Find the missing count first. {_ask(cats, 'height of each column of a column graph')}"
    return dict(
        text=f"{stem} {ask}",
        answer_kind="bars",
        **key,
        distractors=ds,
        working=[
            f"The known counts add to {others}.",
            f"{cats[miss]} = {total} - {others} = {counts[miss]}.",
            "The columns are " + ", ".join(f"{c} {h}" for c, h in zip(cats, counts)) + ".",
        ],
    )


def _hard(rng):
    return _hard_numeric(rng) if rng.random() < 0.5 else _hard_missing(rng)


@generator("dg_construct_columns", level=8, presets={t: {"tier": t} for t in ("easy", "medium", "hard")})
def dg_construct_columns(rng, tier="easy"):
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(500, f"a {tier} construct-columns question"):
        d = build(rng)
        if d is not None:
            return d
    raise AssertionError  # pragma: no cover


# ---------------------------------------------------------------------------------------------- exam chain


@generator("dg_construct_columns_chain", level=8, presets={"exam": {"tier": "exam"}})
def dg_construct_columns_chain(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(500, "a construct-columns chain"):
        theme = rng.choice(_THEMES)
        k = 4
        cats = [c.capitalize() for c in rng.sample(theme[1], k)]
        cats_l = [c.lower() for c in cats]
        counts = _heights(rng, k, 2, 9)
        if not 16 <= sum(counts) <= 28:
            continue
        diff = max(counts) - min(counts)
        if diff % 2 or not any(c % 2 for c in counts):
            continue  # the symbols difference is a whole number and some row needs a half symbol
        hs = [Fraction(c, 2) for c in counts]
        data = _list_text(rng, cats_l, counts)
        stem = theme[0].format(data=data)
        check(_recount(stem, cats_l) == counts, "recount of the list")
        da = bars_dist(
            cats_l,
            counts,
            [(h, "dg-cc-categories-swapped") for h in _swapped(counts)[:2]]
            + [(_bump(counts, i, d), "dg-cc-tally-miscount") for i, d in ((0, 1), (2, -1), (1, 1))],
        )
        ordered = sorted(counts)
        db = numd(
            diff,
            [
                (max(counts) + min(counts), "dg-cc-range-added"),
                (ordered[-1] - ordered[1], "dg-cc-second-tallest"),
                (ordered[-2] - ordered[0], "dg-cc-second-tallest"),
                (diff + 1, "dg-cc-tally-miscount"),
                (diff - 1, "dg-cc-tally-miscount"),
            ],
            lo=0,
        )
        dc = bars_dist(
            cats_l,
            hs,
            [
                ([Fraction(c) for c in counts], "dg-cc-key-ignored"),
                ([h if h.denominator == 1 else h - Fraction(1, 2) for h in hs], "dg-cc-half-symbol-dropped"),
                ([h if h.denominator == 1 else h + Fraction(1, 2) for h in hs], "dg-cc-half-symbol-whole"),
                ([Fraction(c * 2) for c in counts], "dg-cc-key-multiplied"),
            ],
            snap=Fraction(1, 2),
        )
        sym = Fraction(diff, 2)
        dd = numd(
            sym,
            [
                (diff, "dg-cc-difference-in-items-not-symbols"),
                (sym + 1, "dg-cc-tally-miscount"),
                (sym - 1, "dg-cc-tally-miscount"),
                (diff * 2, "dg-cc-key-multiplied"),
            ],
            lo=0,
        )
        if min(len(x) for x in (da, db, dc, dd)) < 2:
            continue
        w = ", ".join(f"{c} {h}" for c, h in zip(cats_l, counts))
        return dict(
            text=stem,
            parts=[
                p_bars(
                    _ask(cats_l, "height of each column of a column graph"),
                    cats_l,
                    counts,
                    3,
                    [f"Count each category: {w}."],
                    da,
                ),
                p_number(
                    f"How many more {theme[2]} does the most popular category have than the least popular?",
                    diff,
                    1,
                    [f"{max(counts)} - {min(counts)} = {diff}."],
                    db,
                    sensible_range=(0, None),
                ),
                p_bars(
                    "The same data is shown in a picture graph where one symbol stands for 2 and half a symbol for 1. "
                    + _ask(cats_l, "number of symbols in each row").replace(
                        "written like", "using .5 for half a symbol, written like"
                    ),
                    cats_l,
                    hs,
                    3,
                    [
                        "Divide each count by 2: "
                        + ", ".join(f"{c} / 2 = {numf(h)['answer_text']}" for c, h in zip(counts, hs))
                        + "."
                    ],
                    dc,
                    snap=Fraction(1, 2),
                ),
                p_number(
                    "How many more symbols does the longest row have than the shortest row?",
                    sym,
                    2,
                    [f"{diff} votes is {diff} / 2 = {numf(sym)['answer_text']} symbols."],
                    dd,
                    follow_through="part_b / 2",
                    sensible_range=(0, None),
                ),
            ],
        )
    raise AssertionError  # pragma: no cover
