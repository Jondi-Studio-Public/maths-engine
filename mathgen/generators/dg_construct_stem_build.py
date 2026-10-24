"""Generates `dg_construct_stem` questions (easy, medium, hard) and the exam chain `dg_construct_stem_chain`.

``construct-stem-and-leaf-plots`` (``statistics-probability``, answer ``table``): the learner builds an ordered
stem-and-leaf plot from a jumbled list of values. The plot is a two-column table, one row for each stem, and the learner
fills the Leaf cell of every row with that stem's leaves in increasing order, a repeated value written as often as it
occurs. The `table` grader compares each cell as a label ignoring spaces, so ``2 5 8`` and ``258`` are the same reply but
``5 2 8`` is wrong.

* easy: two-digit whole numbers, one leaf per value, in three or four rows.
* medium: more values, with repeats, in four or five rows.
* hard: values to one decimal place (stem the whole number, leaf the tenth), or three-digit values (stem the hundreds
  and tens, leaf the units), with repeats.
* exam (``dg_construct_stem_chain``): from a list of two-digit values: (a) the whole plot, (b) how many values there
  are, (c) the position of the median, following through from (b), (d) the median.

Every plot is built from the list the learner reads, its leaves are re-sorted and recounted from the text, and the median
is re-derived with `statistics`.
"""

from __future__ import annotations

import statistics
from fractions import Fraction
from types import SimpleNamespace

from ..answers import Distractor, same_value
from ..core import attempts, check, generator
from ._spread_kit import uses_rng
from ..grading import GRADERS
from ..grading.typed import table_answer_text
from ._dg_kit import numd, numtext, p_number

AREA = "construct-stem-and-leaf-plots"

MISCONCEPTIONS = [
    (
        "dg-sl-leaves-unordered",
        "Writes the leaves of a row in the order the values were listed instead of in increasing order.",
        AREA,
    ),
    (
        "dg-sl-leaves-descending",
        "Writes the leaves of a row from largest to smallest instead of smallest to largest.",
        AREA,
    ),
    ("dg-sl-repeat-written-once", "Writes a repeated value's leaf only once, so the row is one leaf short.", AREA),
    ("dg-sl-leaf-dropped", "Leaves one value out of the plot, so a row has a leaf missing.", AREA),
    ("dg-sl-whole-value-as-leaf", "Writes the whole value as the leaf instead of only its last digit.", AREA),
    (
        "dg-sl-leaf-in-wrong-stem",
        "Puts a value in the row of the next stem, for example a number in the twenties in the thirties row.",
        AREA,
    ),
    ("dg-sl-count-stems", "Gives the number of stems as the number of values.", AREA),
    ("dg-sl-count-off-one", "Miscounts the leaves by one when finding how many values there are.", AREA),
    (
        "dg-sl-position-half-n",
        "Halves the number of values to find the position of the median, forgetting to add 1 first.",
        AREA,
    ),
    ("dg-sl-position-not-halved", "Gives n + 1 as the position of the median without halving it.", AREA),
    ("dg-sl-median-position-as-value", "Gives the position of the median as if it were the median itself.", AREA),
    ("dg-sl-median-of-leaves-only", "Finds the middle leaf and forgets to put its stem in front of it.", AREA),
    (
        "dg-sl-median-middle-stem",
        "Takes the middle of the stems (or the middle row's first value) as the median.",
        AREA,
    ),
    (
        "dg-sl-median-even-lower",
        "Takes the lower of the two middle values as the median when the number of values is even.",
        AREA,
    ),
]


# ---------------------------------------------------------------------------------------------- the plot


def _stem_leaf(value, mode):
    """``(stem label, leaf digit)`` of a value: ``decimal`` is tenths, ``hundreds`` stems by tens, else tens and units."""
    if mode == "decimal":
        tenths = round(value * 10)
        return str(tenths // 10), tenths % 10
    return str(value // 10), value % 10


def _rows(values, mode):
    rows: dict[str, list[int]] = {}
    for v in sorted(values):
        s, l_ = _stem_leaf(v, mode)
        rows.setdefault(s, []).append(l_)
    # every stem from the first to the last appears, even one with no leaves (those are avoided by the generators)
    return rows


def _leaves(row):
    return " ".join(str(x) for x in row)


def _table(rows):
    stems = list(rows)
    return {
        "columns": ["Stem", "Leaf"],
        "rows": [[s, None] for s in stems],
        "blanks": {f"{i},1": _leaves(rows[s]) for i, s in enumerate(stems)},
    }


def _probe(answer, text):
    return SimpleNamespace(answer=answer, answer_text=text, answer_kind="table", parts=None)


def _dists(rows, edits, n=3):
    """Tagged distractors from ``[(modified rows, tag), ...]``: kept when the real grader rejects the typed text for the
    answer and accepts it for the rows it stands for."""
    answer = _table(rows)
    text = table_answer_text(answer)
    key = _probe(answer, text)
    check(GRADERS["table"](key, text).correct, "the table grader rejects its own canonical answer")
    seen = {text}
    out = []
    for cand, tag in edits:
        if cand is None or any(not v for v in cand.values()):
            continue
        a2 = _table(cand)
        t2 = table_answer_text(a2)
        if (
            t2 in seen
            or same_value(a2, answer, "table")
            or GRADERS["table"](key, t2).correct
            or not GRADERS["table"](_probe(a2, t2), t2).correct
        ):
            continue
        seen.add(t2)
        out.append(Distractor(t2, tag, a2))
        if len(out) == n:
            break
    return out


def _edits(rng, rows):
    """The slips: unsorted or descending leaves, a repeat written once, a dropped leaf, the whole value, a wrong stem."""
    stems = list(rows)
    out = []
    multi = [s for s in stems if len(rows[s]) >= 2 and len(set(rows[s])) >= 2]
    if multi:
        s = rng.choice(multi)
        shuffled = list(rows[s])
        for _ in range(10):
            rng.shuffle(shuffled)
            if shuffled != rows[s] and shuffled != sorted(rows[s], reverse=True):
                break
        out.append(({**rows, s: shuffled}, "dg-sl-leaves-unordered"))
        out.append(({**rows, s: sorted(rows[s], reverse=True)}, "dg-sl-leaves-descending"))
    rep = [s for s in stems if len(rows[s]) != len(set(rows[s]))]
    if rep:
        s = rng.choice(rep)
        out.append(({**rows, s: sorted(set(rows[s]))}, "dg-sl-repeat-written-once"))
    long_ = [s for s in stems if len(rows[s]) >= 2]
    if long_:
        s = rng.choice(long_)
        out.append(({**rows, s: rows[s][:-1]}, "dg-sl-leaf-dropped"))
        j = stems.index(s)
        if j + 1 < len(stems):
            nxt = stems[j + 1]
            moved = {**rows, s: rows[s][:-1], nxt: sorted(rows[nxt] + [rows[s][-1]])}
            out.append((moved, "dg-sl-leaf-in-wrong-stem"))
    return out


# ---------------------------------------------------------------------------------------------- data


_THEMES = (
    ("Students timed how many seconds they could balance on one leg. The times were: {data}.", "s"),
    ("A shop recorded the number of customers each hour: {data}.", ""),
    ("The ages of the people at a swimming lesson were: {data}.", ""),
    ("A class recorded the number of pages each student read last weekend: {data}.", ""),
    ("The number of points a team scored in each game was: {data}.", ""),
)
_DEC_THEMES = (
    "The lengths of the bean seedlings in a school garden, in centimetres, were: {data}.",
    "Students recorded how long their walk to school takes, in kilometres: {data}.",
    "The masses of the apples picked by a class, in hundreds of grams, were: {data}.",
)
_BIG_THEMES = (
    "The heights, in centimetres, of the students in a Year 7 class were: {data}.",
    "A walking group recorded the steps each member took in one morning, in tens: {data}.",
)


def _tens_values(rng, n, stems, repeats):
    first = rng.randint(1, 7)
    stem_set = list(range(first, first + stems))
    values = []
    for st in stem_set:
        k = max(1, rng.randint(2, 5) if n > 15 else rng.randint(2, 4))
        leaves = [rng.randint(0, 9) for _ in range(k)] if repeats else rng.sample(range(10), k)
        values += [st * 10 + l_ for l_ in leaves]
    return values


def _data(values, rng, fmt=str):
    vs = list(values)
    rng.shuffle(vs)
    return ", ".join(fmt(v) for v in vs)


def _plot_ok(values, mode, lo, hi, need_repeat):
    rows = _rows(values, mode)
    if not lo <= len(values) <= hi:
        return False
    if need_repeat and not any(len(r) != len(set(r)) for r in rows.values()):
        return False
    if not need_repeat and any(len(r) != len(set(r)) for r in rows.values()):
        return False
    stems = [int(s) for s in rows]
    return stems == list(range(stems[0], stems[0] + len(stems))) and all(rows.values())


def _key_line(mode, stem, leaf):
    if mode == "decimal":
        return f"Key: {stem} | {leaf} means {stem}.{leaf}"
    return f"Key: {stem} | {leaf} means {stem}{leaf}"


def _question(rng, values, mode, theme, tail):
    rows = _rows(values, mode)
    stem_words = ", ".join(rows)
    a = next(iter(rows))
    leaf = rows[a][0]
    text = theme.format(data=_data(values, rng, (lambda v: f"{v:.1f}") if mode == "decimal" else str))
    text += (
        f" Make a stem-and-leaf plot with the stems {stem_words}. Write the leaves of each row in increasing order, "
        f"repeating a leaf as often as its value occurs. {_key_line(mode, a, leaf)}. {tail}"
    )
    answer = _table(rows)
    edits = _edits(rng, rows)
    ds = _dists(rows, edits)
    if mode == "tens" and ds:
        # the whole value written as the leaf in one row replaces the last distractor
        s_ = rng.choice(list(rows))
        blanks = dict(answer["blanks"])
        blanks[f"{list(rows).index(s_)},1"] = " ".join(f"{s_}{x}" for x in rows[s_])
        wrong = {**answer, "blanks": blanks}
        wrong_text = table_answer_text(wrong)
        key = _probe(answer, table_answer_text(answer))
        if (
            not same_value(wrong, answer, "table")
            and not GRADERS["table"](key, wrong_text).correct
            and GRADERS["table"](_probe(wrong, wrong_text), wrong_text).correct
        ):
            ds = ds[:2] + [Distractor(wrong_text, "dg-sl-whole-value-as-leaf", wrong)]
    return rows, answer, text, ds


def _ask_tail(first):
    return f"Give the leaves of each row, written like ({first} Leaf = x y z), with one row for each stem."


def _build(rng, values, mode, theme, working_note):
    rows, answer, text, ds = _question(rng, values, mode, theme, _ask_tail(next(iter(_rows(values, mode)))))
    if len(ds) < 3:
        return None
    return dict(
        text=text,
        answer_kind="table",
        answer=answer,
        answer_text=table_answer_text(answer),
        answer_latex="plain:" + table_answer_text(answer),
        distractors=ds,
        working=[
            "Split each value into its stem and its leaf: "
            + ", ".join(f"{s} | {_leaves(r)}" for s, r in rows.items())
            + ".",
            working_note,
        ],
    )


def _easy(rng):
    values = _tens_values(rng, 12, rng.randint(3, 4), repeats=False)
    if not _plot_ok(values, "tens", 9, 15, need_repeat=False):
        return None
    return _build(
        rng,
        values,
        "tens",
        rng.choice(_THEMES)[0],
        "Each value appears once, so each leaf appears once, smallest first.",
    )


def _medium(rng):
    values = _tens_values(rng, 18, rng.randint(4, 5), repeats=True)
    if not _plot_ok(values, "tens", 14, 24, need_repeat=True):
        return None
    return _build(
        rng, values, "tens", rng.choice(_THEMES)[0], "A value that occurs twice gives its leaf twice in the same row."
    )


def _hard(rng):
    if rng.random() < 0.5:
        first = rng.randint(1, 6)
        stems = rng.randint(3, 4)
        tenths = []
        for st in range(first, first + stems):
            tenths += [st * 10 + rng.randint(0, 9) for _ in range(rng.randint(2, 5))]
        values = [t / 10 for t in tenths]
        ints = [round(v * 10) for v in values]
        if not _plot_ok(values, "decimal", 10, 20, need_repeat=True):
            return None
        theme = rng.choice(_DEC_THEMES)
        note = "The stem is the whole number and the leaf is the tenth, so 3.4 is written 3 | 4."
        del ints
        return _build(rng, values, "decimal", theme, note)
    first = rng.randint(12, 15)
    stems = rng.randint(3, 4)
    values = []
    for st in range(first, first + stems):
        values += [st * 10 + rng.randint(0, 9) for _ in range(rng.randint(2, 5))]
    if not _plot_ok(values, "tens", 10, 20, need_repeat=True):
        return None
    theme = rng.choice(_BIG_THEMES)
    return _build(rng, values, "tens", theme, "The stem is everything but the units digit, so 147 is written 14 | 7.")


@generator("dg_construct_stem", level=12, presets={t: {"tier": t} for t in ("easy", "medium", "hard")})
@uses_rng
def dg_construct_stem(rng, tier="easy"):
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(800, f"a {tier} stem-and-leaf question"):
        d = build(rng)
        if d is not None:
            return d
    raise AssertionError  # pragma: no cover


# ---------------------------------------------------------------------------------------------- exam chain


@generator("dg_construct_stem_chain", level=12, presets={"exam": {"tier": "exam"}})
@uses_rng
def dg_construct_stem_chain(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(800, "a stem-and-leaf chain"):
        values = _tens_values(rng, 16, rng.randint(3, 5), repeats=True)
        if not _plot_ok(values, "tens", 13, 24, need_repeat=False) and not _plot_ok(
            values, "tens", 13, 24, need_repeat=True
        ):
            continue
        rows = _rows(values, "tens")
        n = len(values)
        ordered = sorted(values)
        med = Fraction(statistics.median(values)).limit_denominator(2)
        check(med == Fraction(ordered[(n - 1) // 2] + ordered[n // 2], 2), "median two ways")
        pos = Fraction(n + 1, 2)
        theme = rng.choice(_THEMES)[0]
        text = theme.format(data=_data(values, rng))
        a = next(iter(rows))
        text += (
            f" A stem-and-leaf plot is made with the stems {', '.join(rows)}, the leaves of each row in increasing order. "
            f"{_key_line('tens', a, rows[a][0])}."
        )
        answer = _table(rows)
        ds_a = _dists(rows, _edits(rng, rows))
        stems = len(rows)
        db = numd(
            n,
            [
                (stems, "dg-sl-count-stems"),
                (n + 1, "dg-sl-count-off-one"),
                (n - 1, "dg-sl-count-off-one"),
                (n + 2, "dg-sl-count-off-one"),
            ],
            lo=1,
        )
        dc = numd(
            pos,
            [
                (Fraction(n, 2), "dg-sl-position-half-n"),
                (n + 1, "dg-sl-position-not-halved"),
                (Fraction(n - 1, 2), "dg-sl-position-half-n"),
                (pos + 1, "dg-sl-count-off-one"),
            ],
            lo=1,
        )
        leaf_only = Fraction(ordered[(n - 1) // 2] % 10 + ordered[n // 2] % 10, 2)
        mid_stem = int(list(rows)[stems // 2]) * 10
        dd = numd(
            med,
            [
                (pos, "dg-sl-median-position-as-value"),
                (leaf_only, "dg-sl-median-of-leaves-only"),
                (mid_stem, "dg-sl-median-middle-stem"),
                (ordered[(n - 1) // 2] if n % 2 == 0 else None, "dg-sl-median-even-lower"),
                (med + 1, "dg-sl-count-off-one"),
            ],
            lo=0,
        )
        if len(ds_a) < 3 or min(len(x) for x in (db, dc, dd)) < 2:
            continue
        ans_text = table_answer_text(answer)
        part_a = dict(
            prompt=f"Give the leaves of each row of the plot, written like ({a} Leaf = x y z).",
            answer_kind="table",
            answer=answer,
            answer_text=ans_text,
            answer_latex="plain:" + ans_text,
            marks=3,
            working=[
                "Split each value into its stem and leaf: "
                + ", ".join(f"{s} | {_leaves(r)}" for s, r in rows.items())
                + "."
            ],
            distractors=ds_a,
        )
        return dict(
            text=text,
            parts=[
                part_a,
                p_number(
                    "How many values are in the plot?",
                    n,
                    1,
                    [f"Count the leaves: {' + '.join(str(len(r)) for r in rows.values())} = {n}."],
                    db,
                    sensible_range=(1, 60),
                ),
                p_number(
                    "At which position in the ordered list is the median?",
                    pos,
                    1,
                    [f"The median is at position ({n} + 1) / 2 = {numtext(pos)}."],
                    dc,
                    follow_through="(part_b + 1) / 2",
                    sensible_range=(1, 60),
                ),
                p_number(
                    "What is the median?",
                    med,
                    2,
                    [f"Count along the leaves to position {numtext(pos)}: the median is {numtext(med)}."],
                    dd,
                    sensible_range=(0, int(med * 2) + 10),
                ),
            ],
        )
    raise AssertionError  # pragma: no cover
