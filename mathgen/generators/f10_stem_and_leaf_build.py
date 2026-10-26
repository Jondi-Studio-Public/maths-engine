"""Generates `f10-stem-and-leaf-plots` questions (Year 7 statistics, AC9M7ST02): read and interpret a
stem-and-leaf plot of whole-number data. `answer: number`, typed.

The plot is plain text inside the one-line question, rows separated by semicolons, with a key line
("Key: 2 | 3 means 23"). Every row is `stem | leaves`; a back-to-back plot is `leaves | stem | leaves`. Nothing
is drawn.

Every answer is derived from the value list that the plot is rendered from, and re-derived from the rendered
rows (`_reread`), so a plot that does not say what the generator believes fails loudly.

Ladder (reasoning escalation, not size):
  easy:   read the plot. Count the values (leaves, not stems), name the largest or smallest value (stem and
          leaf together, using the key), or count the values in one stated stem.
  medium: summary statistics from the plot: median (odd count, or an even count whose two middle values give a
          .5), range or mode. Half the plots list the leaves of a row in no particular order, so the data must
          be put in order before the median is read off.
  hard:   reason backwards or compare. A back-to-back plot of two groups (difference of the medians, or of the
          ranges), or one leaf hidden behind "?" that the learner recovers from a stated median or range.

Wrong answers are tagged with the misconception they stand for (`option_notes`); each is checked against the
real `number` grader on every candidate.
"""

from __future__ import annotations

import random
from collections import Counter
from fractions import Fraction
from types import SimpleNamespace

from mathgen.core import generator, latex_answer, ordinal
from mathgen.grading.typed import grade_number

TIERS = ("easy", "medium", "hard")
HIDDEN = "?"
NONE_MARK = "(none)"

# stems: the range of stems a context allows (a stem of 12 with key "12 | 5 means 125" gives 100s);
# item: what is counted; val: what is measured, singular; stmt: the introduction.
_CONTEXTS = [
    dict(
        stems=(3, 9),
        item="students",
        val="score",
        stmt="The stem-and-leaf plot shows the scores out of 100 that some students got in a quiz.",
    ),
    dict(
        stems=(1, 6),
        item="students",
        val="message count",
        stmt="The stem-and-leaf plot shows the number of text messages that some students sent yesterday.",
    ),
    dict(
        stems=(1, 5),
        item="customers",
        val="waiting time",
        stmt="The stem-and-leaf plot shows how many minutes some customers waited in a queue.",
    ),
    dict(
        stems=(1, 8),
        item="members",
        val="age",
        stmt="The stem-and-leaf plot shows the ages in years of the members of a club.",
    ),
    dict(
        stems=(10, 15),
        item="seedlings",
        val="height",
        stmt="The stem-and-leaf plot shows the heights in centimetres of some seedlings.",
    ),
    dict(
        stems=(3, 9),
        item="games",
        val="points total",
        stmt="The stem-and-leaf plot shows the points a basketball team scored in each of its games.",
    ),
]

# Two groups for a back-to-back plot.
_PAIRS = [
    dict(
        stems=(3, 9),
        groups=("Class 7A", "Class 7B"),
        val="score",
        stmt="The back-to-back stem-and-leaf plot compares the scores out of 100 in a quiz for two classes.",
    ),
    dict(
        stems=(1, 6),
        groups=("Clinic North", "Clinic South"),
        val="waiting time",
        stmt="The back-to-back stem-and-leaf plot compares the waiting times in minutes at two clinics.",
    ),
    dict(
        stems=(3, 9),
        groups=("Hawks", "Eagles"),
        val="points total",
        stmt="The back-to-back stem-and-leaf plot compares the points two basketball teams scored in their games.",
    ),
    dict(
        stems=(10, 15),
        groups=("Fertiliser X", "Fertiliser Y"),
        val="height",
        stmt="The back-to-back stem-and-leaf plot compares the heights in centimetres of seedlings grown with two fertilisers.",
    ),
    dict(
        stems=(1, 8),
        groups=("Choir", "Band"),
        val="age",
        stmt="The back-to-back stem-and-leaf plot compares the ages in years of the members of a choir and a band.",
    ),
]


class _Retry(Exception):
    """This draw cannot make a well-formed question; draw again."""


# ---------------------------------------------------------------------------
# Data and rendering
# ---------------------------------------------------------------------------


def _draw(rng: random.Random, ctx_stems: tuple[int, int], nstems: int, n: int, empty_ok: bool = False) -> list[int]:
    """A sorted list of `n` whole numbers whose tens (the stem) run over `nstems` consecutive stems. The first and
    last stem always have a leaf; an interior stem may be empty when `empty_ok`."""
    lo, hi = ctx_stems
    nstems = min(nstems, hi - lo + 1)
    s0 = rng.randint(lo, hi - nstems + 1)
    stems = list(range(s0, s0 + nstems))
    counts = [1] * nstems
    if empty_ok and nstems >= 3 and rng.random() < 0.3:
        counts[rng.randint(1, nstems - 2)] = 0
    weights = [rng.randint(1, 5) for _ in stems]
    for _ in range(n - sum(counts)):
        i = rng.choices(range(nstems), weights=weights)[0]
        if counts[i] == 0 and i not in (0, nstems - 1):
            continue
        counts[i] += 1
    values = []
    for s, c in zip(stems, counts):
        values += [10 * s + rng.randint(0, 9) for _ in range(c)]
    return sorted(values)


def _rows(values: list[int]) -> dict[int, list[int]]:
    """{stem: leaves in order} for every stem from the lowest to the highest, empty stems included."""
    rows = {s: [] for s in range(min(values) // 10, max(values) // 10 + 1)}
    for v in sorted(values):
        rows[v // 10].append(v % 10)
    return rows


def _plot_text(rows: dict[int, list[int | str]], shuffled: dict[int, list[int | str]] | None = None) -> str:
    parts = []
    for s, leaves in rows.items():
        shown = (shuffled or rows)[s]
        parts.append(f"{s} | " + (" ".join(str(x) for x in shown) if leaves else NONE_MARK))
    return "; ".join(parts)


def _key(rng: random.Random, values: list[int], avoid: set[int] | None = None) -> str:
    """ "Key: 2 | 3 means 23", using a value that is in the plot and is not one the question asks about."""
    pool = [v for v in values if v not in (avoid or set())] or values
    v = rng.choice(pool)
    return f"Key: {v // 10} | {v % 10} means {v}."


def _median(values: list[int]) -> Fraction:
    s = sorted(values)
    n = len(s)
    return Fraction(s[n // 2]) if n % 2 else Fraction(s[n // 2 - 1] + s[n // 2], 2)


def _num(x) -> int | float:
    x = Fraction(x)
    if x.denominator == 1:
        return int(x)
    if x.denominator == 2:
        return float(x)
    return round(float(x), 1)


def _fmt(x) -> str:
    return str(_num(x))


def _reread(text: str) -> list[int]:
    """The values a single plot's text spells out (key and rows only), as an independent check."""
    body = text.split("rows written as stem | leaves: ", 1)[1].split(". ", 1)[0]
    out = []
    for row in body.split("; "):
        stem, _, leaves = row.partition(" | ")
        if leaves == NONE_MARK:
            continue
        out += [10 * int(stem) + int(x) for x in leaves.split()]
    return out


# ---------------------------------------------------------------------------
# Wrong answers
# ---------------------------------------------------------------------------


def _finish(answer, cands: list[tuple]) -> tuple[str, list[str], dict[str, str]]:
    """(answer_text, distractors, option_notes). `cands` is an ordered list of (value, misconception id). Each kept
    candidate is a different, non-negative value, and the real `number` grader rejects it; the canonical answer is
    accepted by the same grader."""
    answer_text = _fmt(answer)
    probe = SimpleNamespace(answer=_num(answer), answer_text=answer_text, answer_alt=None)
    if not grade_number(probe, answer_text).correct:
        raise ValueError("number grader rejects its own canonical answer")
    seen = {answer_text}
    out: list[str] = []
    notes: dict[str, str] = {}
    for value, note in cands:
        if value is None or Fraction(value) < 0:
            continue
        text = _fmt(value)
        if text in seen:
            continue
        if ("." in text) != ("." in answer_text):
            continue  # every option has the answer's shape: a lone whole number among decimals (or the reverse) gives it away
        if grade_number(probe, text).correct:
            raise ValueError(f"number grader accepts the wrong answer {text!r} for {answer_text!r}")
        seen.add(text)
        out.append(text)
        notes[text] = note
        if len(out) == 3:
            break
    if len(out) < 2:
        raise _Retry
    return answer_text, out, notes


def _result(text: str, answer, cands, working: list[str]) -> dict:
    answer_text, distractors, notes = _finish(answer, cands)
    return dict(
        text=text,
        answer=_num(answer),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
        option_notes=notes,
    )


# ---------------------------------------------------------------------------
# Easy: read the plot
# ---------------------------------------------------------------------------


def _single_text(rng, ctx, values, ask: str, avoid: set[int], shuffled=None, order_note: str = "") -> str:
    rows = _rows(values)
    n = len(values)
    return (
        f"{ctx['stmt']} {_key(rng, values, avoid)} "
        f"Plot, rows written as stem | leaves: {_plot_text(rows, shuffled)}. {order_note}{ask}"
    )


def _easy_once(rng: random.Random) -> dict:
    ctx = rng.choice(_CONTEXTS)
    n = rng.randint(9, 20)
    values = _draw(rng, ctx["stems"], rng.randint(3, 5), n)
    rows = _rows(values)
    stems = list(rows)
    kind = rng.choice(["count", "largest", "smallest", "in-stem"])
    lead = "Leaves in each row are in order, smallest first. "
    if kind == "count":
        ask = f"How many {ctx['item']} are shown in the plot?"
        avoid = set()
        cands = [
            (len(stems), "counts-the-stems-not-the-leaves"),
            (n + len(stems), "counts-the-stems-as-values"),
            (n - 1, "misses-a-leaf"),
            (n + 1, "double-counts-a-leaf"),
        ]
        if rng.random() < 0.5:
            cands[2:] = cands[:1:-1]
        answer = n
        working = [f"Count the leaves: {', '.join(str(len(rows[s])) for s in stems)}", f"Total: {n}"]
    elif kind in ("largest", "smallest"):
        big = kind == "largest"
        ans = max(values) if big else min(values)
        s, leaf = ans // 10, ans % 10
        ask = f"What is the {kind} {ctx['val']}?"
        avoid = {ans}
        all_leaves = [leaf_ for row in rows.values() for leaf_ in row]
        extreme_leaf = max(all_leaves) if big else min(all_leaves)
        cands = [
            (leaf, "reads-the-leaf-alone"),
            (s + leaf, "adds-the-stem-and-leaf"),
            (10 * s, "ignores-the-leaf"),
            (extreme_leaf, "picks-the-extreme-leaf-whatever-its-stem"),
        ]
        answer = ans
        working = [f"The {kind} value is in the {'last' if big else 'first'} row: stem {s}, leaf {leaf}", f"{ans}"]
    else:
        s = rng.choice([x for x in stems if rows[x]])
        count = len(rows[s])
        tens = f"{s}0s"
        ask = f"How many {ctx['item']} have a {ctx['val']} in the {tens}?"
        avoid = set()
        neighbour = [x for x in (s - 1, s + 1) if x in rows]
        wrong_row = rng.choice(neighbour)
        cands = [
            (n, "counts-the-whole-plot"),
            (len(rows[wrong_row]), "reads-the-wrong-row"),
            (sum(rows[s]), "adds-the-leaves"),
            (s, "gives-the-stem"),
        ]
        answer = count
        working = [f"Row {s}: {len(rows[s])} leaves"]
    text = _single_text(rng, ctx, values, ask, avoid, order_note=lead)
    if sorted(_reread(text)) != sorted(values):
        raise ValueError("plot text disagrees with the values")
    return _result(text, answer, cands, working)


# ---------------------------------------------------------------------------
# Medium: summary statistics
# ---------------------------------------------------------------------------


def _shuffle_rows(rng: random.Random, rows: dict[int, list[int]]) -> dict[int, list[int]]:
    for _ in range(20):
        out = {s: rng.sample(leaves, len(leaves)) for s, leaves in rows.items()}
        if any(out[s] != rows[s] for s in rows):
            return out
    raise _Retry


def _reading_order(shuffled: dict[int, list[int]]) -> list[int]:
    return [10 * s + leaf for s, leaves in shuffled.items() for leaf in leaves]


def _medium_once(rng: random.Random) -> dict:
    ctx = rng.choice(_CONTEXTS)
    kind = rng.choice(["median-odd", "median-even", "range", "mode"])
    unordered = rng.random() < 0.5
    if kind == "median-odd":
        n = rng.choice(range(9, 20, 2))
    elif kind == "median-even":
        n = rng.choice(range(10, 21, 2))
    else:
        n = rng.randint(10, 20)
    values = _draw(rng, ctx["stems"], rng.randint(3, 5), n, empty_ok=True)
    rows = _rows(values)
    shuffled = _shuffle_rows(rng, rows) if unordered else None
    listed = _reading_order(shuffled) if shuffled else values
    order_note = (
        "The leaves in each row are not in order. "
        if unordered
        else "Leaves in each row are in order, smallest first. "
    )
    n = len(values)
    mid_pos = Fraction(n + 1, 2)
    mean = Fraction(sum(values), n)
    if kind.startswith("median"):
        med = _median(values)
        if kind == "median-even" and med.denominator != 2:
            raise _Retry
        avoid = {int(med)} if med.denominator == 1 else {values[n // 2 - 1], values[n // 2]}
        ask = f"What is the median {ctx['val']}?"
        half = len(listed) // 2
        listed_median = (
            Fraction(listed[half]) if len(listed) % 2 else Fraction(listed[half - 1] + listed[half], 2)
        )  # the middle of the list as written, without putting it in order
        if kind == "median-odd":
            cands = [
                (med % 10, "reads-the-leaf-alone"),
                (mid_pos, "gives-the-position-not-the-value"),
                (listed_median if unordered else None, "takes-the-middle-of-the-unordered-list"),
                (_num(round(mean, 1)) if mean.denominator else None, "mean-not-median"),
                (Fraction(max(values) + min(values), 2), "midrange-not-median"),
            ]
        else:
            lo, hi = sorted(values)[n // 2 - 1], sorted(values)[n // 2]
            cands = [
                (lo, "takes-one-middle-value-not-both"),
                (hi, "takes-one-middle-value-not-both"),
                (mid_pos, "gives-the-position-not-the-value"),
                (listed_median if unordered else None, "takes-the-middle-of-the-unordered-list"),
                (_num(round(mean, 1)), "mean-not-median"),
                (Fraction(max(values) + min(values), 2), "midrange-not-median"),
            ]
        answer = med
        if mid_pos.denominator == 1:
            where = f"position {mid_pos}"
        else:
            where = f"between the {ordinal(mid_pos.numerator // 2)} and {ordinal(mid_pos.numerator // 2 + 1)} values"
        working = [f"In order, the middle is {where}", f"Median: {_fmt(med)}"]
    elif kind == "range":
        rng_ = max(values) - min(values)
        avoid = {max(values), min(values)}
        ask = f"What is the range of the {ctx['val']}s?"
        leaves_all = [v % 10 for v in values]
        cands = [
            (abs(listed[-1] - listed[0]) if unordered else None, "subtracts-the-last-listed-value-from-the-first"),
            (abs(values[-1] % 10 - values[0] % 10), "range-ignores-the-stems"),
            (10 * (max(values) // 10 - min(values) // 10), "range-from-the-stems-only"),
            (max(values), "gives-the-largest-value-not-the-range"),
            (max(leaves_all) - min(leaves_all), "range-ignores-the-stems"),
        ]
        answer = rng_
        working = [f"Largest {max(values)}, smallest {min(values)}", f"Range: {rng_}"]
    else:
        counts = Counter(values)
        top = max(counts.values())
        modes = [v for v, c in counts.items() if c == top]
        if top < 2 or len(modes) != 1:
            raise _Retry
        m = modes[0]
        avoid = {m}
        ask = f"What is the mode of the {ctx['val']}s?"
        by_stem = Counter(v // 10 for v in values)
        longest = [s for s, c in by_stem.items() if c == max(by_stem.values())]
        leaf_counts = Counter(v % 10 for v in values)
        top_leaf = [lf for lf, c in leaf_counts.items() if c == max(leaf_counts.values())]
        cands = [
            (top, "gives-the-frequency-not-the-value"),
            (10 * longest[0] if len(longest) == 1 else None, "takes-the-longest-row"),
            (
                10 * (m // 10) + top_leaf[0] if len(top_leaf) == 1 else None,
                "takes-the-commonest-leaf-whatever-its-stem",
            ),
            (m % 10, "reads-the-leaf-alone"),
        ]
        answer = m
        working = [f"{m} appears {top} times, more than any other value", f"Mode: {m}"]
    text = _single_text(rng, ctx, values, ask, avoid, shuffled=shuffled, order_note=order_note)
    if sorted(_reread(text)) != sorted(values):
        raise ValueError("plot text disagrees with the values")
    cands = [c for c in cands if c[0] is not None]
    cands.sort(key=lambda c: c[1] != "takes-the-middle-of-the-unordered-list")  # a slip this plot invites comes first
    return _result(text, answer, cands, working)


# ---------------------------------------------------------------------------
# Hard: back-to-back comparison, or a hidden leaf
# ---------------------------------------------------------------------------


def _bb_text(rows_a, rows_b) -> str:
    parts = []
    for s in rows_a:
        left = " ".join(str(x) for x in reversed(rows_a[s]))
        right = " ".join(str(x) for x in rows_b[s])
        parts.append(f"{left} | {s} | {right}")
    return "; ".join(parts)


def _back_to_back(rng: random.Random) -> dict:
    ctx = rng.choice(_PAIRS)
    lo, hi = ctx["stems"]
    nstems = min(rng.randint(3, 4), hi - lo + 1)
    s0 = rng.randint(lo, hi - nstems + 1)
    stems = list(range(s0, s0 + nstems))

    def group():
        n = rng.randint(8, 13)
        counts = [1] * nstems
        weights = [rng.randint(1, 5) for _ in stems]
        for _ in range(n - nstems):
            counts[rng.choices(range(nstems), weights=weights)[0]] += 1
        return sorted(10 * s + rng.randint(0, 9) for s, c in zip(stems, counts) for _ in range(c))

    a, b = group(), group()
    rows_a = {s: [v % 10 for v in a if v // 10 == s] for s in stems}
    rows_b = {s: [v % 10 for v in b if v // 10 == s] for s in stems}
    if any(not rows_a[s] or not rows_b[s] for s in stems):
        raise _Retry
    ga, gb = ctx["groups"]
    ask_median = rng.random() < 0.5
    ma, mb = _median(a), _median(b)
    ra, rb = max(a) - min(a), max(b) - min(b)
    ks = rng.choice(stems)
    la, lb = rng.choice(rows_a[ks]), rng.choice(rows_b[ks])
    key = f"Key: {la} | {ks} | {lb} means {10 * ks + la} for {ga} and {10 * ks + lb} for {gb}."
    plot = _bb_text(rows_a, rows_b)
    head = (
        f"{ctx['stmt']} {key} Plot, rows written as {ga} leaves | stem | {gb} leaves: {plot}. "
        "Leaves in each row are in order, smallest nearest the stem. "
    )
    if ask_median:
        ans = abs(ma - mb)
        if ans == 0:
            raise _Retry
        ask = f"What is the difference between the median {ctx['val']} of {ga} and the median of {gb}? Give the larger minus the smaller."
        mid_leaf_diff = abs(_median(a) % 10 - _median(b) % 10)
        mean_diff = abs(Fraction(sum(a), len(a)) - Fraction(sum(b), len(b)))
        low_diff = abs(sorted(a)[len(a) // 2 - 1] - sorted(b)[len(b) // 2 - 1])
        cands = [
            (abs(ra - rb), "compares-the-ranges-not-the-medians"),
            (mean_diff, "mean-not-median"),
            (low_diff, "takes-one-middle-value-not-both"),
            (mid_leaf_diff, "compares-the-leaves-of-the-medians-only"),
        ]
        working = [f"{ga} median {_fmt(ma)}, {gb} median {_fmt(mb)}", f"Difference: {_fmt(ans)}"]
    else:
        ans = abs(ra - rb)
        if ans == 0:
            raise _Retry
        ask = f"What is the difference between the range of the {ctx['val']}s for {ga} and the range for {gb}? Give the larger minus the smaller."
        cands = [
            (abs(ma - mb), "compares-the-medians-not-the-ranges"),
            (ra + rb, "adds-the-ranges"),
            (max(ra, rb), "gives-one-range-not-the-difference"),
            (abs((max(a) % 10 - min(a) % 10) - (max(b) % 10 - min(b) % 10)), "range-ignores-the-stems"),
        ]
        working = [f"{ga} range {ra}, {gb} range {rb}", f"Difference: {ans}"]
    text = head + ask
    return _result(text, ans, cands, working)


def _hidden(rng: random.Random) -> dict:
    ctx = rng.choice(_CONTEXTS)
    by_median = rng.random() < 0.5
    n = rng.choice(range(10, 15, 2)) if by_median else rng.randint(9, 14)
    values = _draw(rng, ctx["stems"], rng.randint(3, 4), n)
    n = len(values)
    if by_median:
        i = rng.choice([n // 2 - 1, n // 2])  # one of the two middle values is hidden
        j = n // 2 if i == n // 2 - 1 else n // 2 - 1
        med = _median(values)
        if med.denominator != 2:
            raise _Retry
        other_side = i - 1 if i == n // 2 - 1 else i + 1
        wrong_neighbour = values[other_side] if 0 <= other_side < n else None
        stated = f"The median is {_fmt(med)}."
        cands_core = [
            (2 * med - wrong_neighbour if wrong_neighbour is not None else None, "pairs-with-the-wrong-neighbour"),
            (int(med), "gives-the-median-as-the-missing-value"),
        ]
        working = [
            f"The two middle values add to 2 × {_fmt(med)} = {_fmt(2 * med)}",
            f"{_fmt(2 * med)} - {values[j]} = {_fmt(values[i])}",
        ]
    else:
        rng_ = max(values) - min(values)
        if rng_ < 12 or values.count(max(values)) != 1 or values.count(min(values)) != 1:
            raise _Retry
        hide_max = rng.random() < 0.5
        i = n - 1 if hide_max else 0
        known = values[:-1] if hide_max else values[1:]
        stated = f"The range of the {ctx['val']}s is {rng_}. The hidden value is the {'largest' if hide_max else 'smallest'} in the plot."
        near = max(known) if hide_max else min(known)
        cands_core = [
            (near + rng_ if hide_max else near - rng_, "applies-the-range-to-the-nearest-value"),
            (rng_, "gives-the-range-as-the-missing-value"),
            (
                (min(known) % 10 + rng_) if hide_max else (max(known) % 10 - rng_),
                "range-ignores-the-stems",
            ),
        ]
        working = [
            f"{'Smallest ' + str(min(known)) + ' + ' + str(rng_) if hide_max else 'Largest ' + str(max(known)) + ' - ' + str(rng_)} = {_fmt(values[i])}"
        ]
    hidden_value = values[i]
    rows = _rows(values)
    shown = {s: list(leaves) for s, leaves in rows.items()}
    hs = hidden_value // 10
    pos = i - sum(len(rows[x]) for x in rows if x < hs)  # the hidden leaf's own place in its row
    shown[hs][pos] = HIDDEN
    key = _key(rng, [v for k, v in enumerate(values) if k != i], None)
    answer = hidden_value
    cands = [(hidden_value % 10, "reads-the-leaf-alone")] + cands_core
    if any(
        v is not None and Fraction(v) == answer for v, _ in cands
    ):  # a wrong route lands on the answer: not a distractor
        cands = [(v, t) for v, t in cands if v is None or Fraction(v) != answer]
    plot = _plot_text(rows, shown)
    text = (
        f"{ctx['stmt']} One leaf is hidden and shown as a question mark ({HIDDEN}). {key} "
        f"Plot, rows written as stem | leaves: {plot}. Leaves in each row are in order, smallest first. "
        f"{stated} What is the hidden {ctx['val']}?"
    )
    return _result(text, answer, [c for c in cands if c[0] is not None], working)


def _hard_once(rng: random.Random) -> dict:
    return _back_to_back(rng) if rng.random() < 0.5 else _hidden(rng)


_TIER_FN = {"easy": _easy_once, "medium": _medium_once, "hard": _hard_once}


@generator(
    "f10_stem_and_leaf",
    level=7,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def f10_stem_and_leaf(rng: random.Random, tier: str = "easy") -> dict:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(200):
        try:
            return _TIER_FN[tier](rng)
        except _Retry:
            continue
    raise ValueError(f"no {tier} stem-and-leaf question found")  # pragma: no cover
