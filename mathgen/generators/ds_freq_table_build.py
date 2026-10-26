"""Generates `f10-frequency-table-discrete-data` questions (Year 3, AC9M3ST01): from a stated list, tally or set of
named observations of a discrete numerical variable (pets at home, goals in a game, shoe sizes ...), complete the
blank cells of a frequency table. `answer: table`; the grader reads every blank as an integer.

Scope: the data are always supplied. Choosing a question, collecting the data and drawing a graph are not asked.
Every blank is fixed by the supplied data, and the blank positions change from question to question.

Ladder (reasoning escalation, not size):
  easy:   count how many times each value occurs (a jumbled list, a tally column, or named observations) and,
          on most questions, total the counts. A value nobody gave has frequency 0.
  medium: the same counting, then combine counts: the Total row and a derived row that adds the frequencies of
          every value a phrase such as "at least 3" or "fewer than 3" picks out, with the boundary value
          deciding whether a count is included.
  hard:   a second column built from the counts. Either a running total (counts added from the lowest value up) or
          "this many or more" (added from the highest value down), found from the raw data; or a half-finished
          table with no raw data, where the missing count must be found from the stated total and the running
          total filled in from it.

Each distractor is a whole wrong table. Its blank cells are recomputed from the wrong count or rule, so a wrong
table is as self-consistent as the right one and the key cannot be picked out by checking the sums.
"""

from __future__ import annotations

import random
from collections import Counter

from mathgen.core import attempts, check, draw_until, generator, phrase
from mathgen.slots import NAMES

from ._f10_table_kit import build_answer, finish, grid_text

TIERS = ("easy", "medium", "hard")

# header: the column heading; lo/hi: the range of values; who: what is counted; ask: the opening sentence;
# fmt: one named observation, with {u} the unit agreeing with {v}; label: None for a person's name, else a prefix.
_CTX = [
    dict(
        header="Number of pets",
        lo=0,
        hi=4,
        who="children",
        unit=("pet", "pets"),
        ask="Children in a Year 3 class were each asked how many pets they have at home.",
        fmt="has {v} {u}",
        label=None,
    ),
    dict(
        header="Goals scored",
        lo=0,
        hi=5,
        who="games",
        unit=("goal", "goals"),
        ask="A junior soccer team kept a record of the goals it scored in each game this season.",
        fmt="{v} {u}",
        label="Game",
    ),
    dict(
        header="Brothers and sisters",
        lo=0,
        hi=4,
        who="children",
        unit=("brother or sister", "brothers and sisters"),
        ask="Children in a Year 3 class were each asked how many brothers and sisters they have.",
        fmt="has {v} {u}",
        label=None,
    ),
    dict(
        header="Books borrowed",
        lo=0,
        hi=5,
        who="children",
        unit=("book", "books"),
        ask="Children in a class counted the books they borrowed from the school library this term.",
        fmt="borrowed {v} {u}",
        label=None,
    ),
    dict(
        header="Shoe size",
        lo=1,
        hi=6,
        who="children",
        unit=("size", "sizes"),
        ask="The shoe sizes of the children in a Year 3 class were recorded.",
        fmt="wears size {v}",
        label=None,
    ),
    dict(
        header="Pieces of fruit eaten",
        lo=0,
        hi=4,
        who="children",
        unit=("piece", "pieces"),
        ask="Children in a class were each asked how many pieces of fruit they ate yesterday.",
        fmt="ate {v} {u}",
        label=None,
    ),
]

_FORMS = ("at least", "fewer than", "more than", "or fewer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tally(n: int) -> str:
    """The tally marks for n: a ||||/ for each group of 5, then single marks; "none" for 0."""
    if n == 0:
        return "none"
    groups, rest = divmod(n, 5)
    parts = ["||||/"] * groups
    if rest:
        parts.append("|" * rest)
    return " ".join(parts)


def _read_tally(text: str) -> int:
    if text == "none":
        return 0
    return 5 * text.count("/") + text.count("|") - 4 * text.count("/")


def _jumble(rng: random.Random, items: list[str]) -> list[str]:
    """`items` shuffled so equal items are not bunched together (a near-sorted list would pre-sort the count)."""
    runs_needed = len(set(items)) + 1
    for _ in range(60):
        rng.shuffle(items)
        if sum(1 for a, b in zip(items, items[1:]) if a != b) >= runs_needed:
            break
    return items


def _draw_freqs(rng: random.Random, k: int, n_lo: int, n_hi: int, zero: bool) -> list[int]:
    """k frequencies in 0..7 totalling n_lo..n_hi; the end values occur, and a value in between may not."""

    def draw(r):
        fs = [r.randint(1, 7) for _ in range(k)]
        if zero and r.random() < 0.5:
            fs[r.randint(1, k - 2)] = 0
        return fs

    def ok(fs):
        return n_lo <= sum(fs) <= n_hi and len(set(fs)) >= 3 and max(fs) - min(fs) >= 2 and max(fs) <= 7

    return draw_until(rng, draw, ok, tries=2000, what="frequencies for a table")


def _unit(ctx: dict, v: int) -> str:
    return ctx["unit"][0] if v == 1 else ctx["unit"][1]


def _data_intro(rng: random.Random, ctx: dict, values: list[int], freqs: list[int], mode: str) -> tuple[str, list[int]]:
    """(the sentences that supply the data, the observations as a list). `mode` is list, named or tally."""
    obs = [v for v, f in zip(values, freqs) for _ in range(f)]
    obs = _jumble(rng, obs)
    ask = ctx["ask"]
    if mode == "tally":
        return (
            f"{ask} The answers were recorded with tally marks. Each ||||/ is one group of 5 and each | is a single "
            "one.",
            obs,
        )
    if mode == "named":
        if ctx["label"] is None:
            labels = rng.sample(list(NAMES), len(obs))
            joiner = " "
        else:
            labels = [f"{ctx['label']} {i + 1}" for i in range(len(obs))]
            joiner = ": "
        facts = "; ".join(f"{lab}{joiner}{ctx['fmt'].format(v=v, u=_unit(ctx, v))}" for lab, v in zip(labels, obs))
        return phrase(rng, ["{ask} {facts}.", "{ask} The results were: {facts}."], ask=ask, facts=facts), obs
    data = ", ".join(str(v) for v in obs)
    return (
        phrase(
            rng,
            [
                "{ask} Their answers, in the order they were given, were: {data}.",
                "{ask} The answers were: {data}.",
                "{ask} Here is what was recorded: {data}.",
            ],
            ask=ask,
            data=data,
        ),
        obs,
    )


def _recount(obs: list[int], values: list[int], freqs: list[int]) -> None:
    seen = Counter(obs)
    check([seen[v] for v in values] == freqs, "the supplied data disagree with the frequencies")


def _in_form(v: int, t: int, form: str) -> bool:
    return {"at least": v >= t, "fewer than": v < t, "more than": v > t, "or fewer": v <= t}[form]


def _form_label(form: str, t: int) -> str:
    return f"{t} or fewer" if form == "or fewer" else f"{form[0].upper()}{form[1:]} {t}"


def _blank_set(rng: random.Random, cells: list, must: list, lo: int, hi: int, ok) -> set:
    """A set of blank positions: `must` plus a random pick from `cells`, between lo and hi in all, that `ok` accepts."""
    rest = [c for c in cells if c not in must]

    def draw(r):
        n = r.randint(max(lo, len(must)), min(hi, len(must) + len(rest)))
        return set(must) | set(r.sample(rest, n - len(must)))

    return draw_until(rng, draw, ok, tries=500, what="a set of blank cells")


class _Table:
    """The rows of one table as a function of the counts, so a wrong rule or a wrong count gives a whole wrong table."""

    def __init__(self, values, tally_strings=None, cum=None, derived=None, fixed_total=None, has_total=True):
        self.values = values
        self.tally = tally_strings  # tally strings per value, or None for no Tally column
        self.cum = cum  # None, "up" (running total) or "down" (this many or more)
        self.derived = derived  # None or (form, t)
        self.fixed_total = fixed_total
        self.has_total = has_total

    def columns(self, header):
        cols = [header] + (["Tally"] if self.tally else []) + ["Count"]
        if self.cum == "up":
            cols.append("Running total")
        elif self.cum == "down":
            cols.append("This many or more")
        return cols

    @property
    def count_col(self):
        return 2 if self.tally else 1

    def rows(self, fs, total_rule="sum", derived_flip=False, cum_rule="right"):
        k = len(self.values)
        out = []
        for i, v in enumerate(self.values):
            row = [str(v)] + ([self.tally[i]] if self.tally else []) + [fs[i]]
            if self.cum:
                up = (self.cum == "up") != (cum_rule == "wrong-end")
                if cum_rule == "copy":
                    row.append(fs[i])
                else:
                    row.append(sum(fs[: i + 1]) if up else sum(fs[i:]))
            out.append(row)
        filler = ["-"] if self.tally else []
        tail = ["-"] if self.cum else []
        if self.has_total:
            if total_rule == "rows":
                total = k
            elif total_rule == "values":
                total = sum(self.values)
            elif self.fixed_total is not None:
                total = self.fixed_total
            else:
                total = sum(fs)
            out.append(["Total"] + filler + [total] + tail)
        if self.derived:
            form, t = self.derived
            if derived_flip:
                form = {
                    "at least": "more than",
                    "more than": "at least",
                    "or fewer": "fewer than",
                    "fewer than": "or fewer",
                }[form]
            out.append(
                [_form_label(*self.derived)]
                + filler
                + [sum(f for v, f in zip(self.values, fs) if _in_form(v, t, form))]
                + tail
            )
        return out


def _edits(table: _Table, answer: dict, worlds: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Each world is (misconception id, kwargs for `table.rows` plus "fs"); its blank cells become one wrong table."""
    out = []
    for note_id, kw in worlds:
        kw = dict(kw)
        fs = kw.pop("fs")
        wrong = table.rows(fs, **kw)
        changes = {}
        for key in answer["blanks"]:
            r, c = map(int, key.split(","))
            changes[key] = wrong[r][c]
        out.append((note_id, changes))
    return out


def _count_worlds(rng, table, fs, blank_rows, tally_mode):
    """Wrong-count worlds shared by every tier; `blank_rows` are the value rows whose counts are blank."""
    worlds = []
    if blank_rows:
        r = rng.choice(blank_rows)
        worlds.append(("dsft-misses-an-observation", dict(fs=[f - 1 if i == r else f for i, f in enumerate(fs)])))
        r2 = rng.choice(blank_rows)
        worlds.append(
            ("dsft-double-counts-an-observation", dict(fs=[f + 1 if i == r2 else f for i, f in enumerate(fs)]))
        )
    pairs = [(a, b) for a in blank_rows for b in blank_rows if a < b and fs[a] != fs[b]]
    if pairs:
        a, b = rng.choice(pairs)
        swapped = list(fs)
        swapped[a], swapped[b] = fs[b], fs[a]
        worlds.append(("dsft-swaps-two-values", dict(fs=swapped)))
    if tally_mode and blank_rows:
        worlds.append(
            ("dsft-tally-gate-as-one", dict(fs=[f // 5 + f % 5 if i in blank_rows else f for i, f in enumerate(fs)]))
        )
    return worlds


def _make_question(rng, text, table, ctx, fs, blanks, extra_worlds, tally_mode, working, header):
    columns = table.columns(header)
    full = table.rows(fs)
    answer = build_answer(columns, full, blanks)
    k = len(table.values)
    blank_rows = sorted({int(key.split(",")[0]) for key in answer["blanks"] if int(key.split(",")[0]) < k})
    worlds = _count_worlds(rng, table, fs, blank_rows, tally_mode) + extra_worlds
    rng.shuffle(worlds)
    answer_text, distractors, notes = finish(answer, _edits(table, answer, worlds))
    check(len(distractors) >= 2, "fewer than two distinct wrong tables")
    return dict(
        text=f"{text} {grid_text(answer)}",
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=distractors,
        option_notes=notes,
    )


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------


def _setup(rng, k_lo, k_hi, n_lo, n_hi, zero):
    ctx = rng.choice(_CTX)
    k = rng.randint(k_lo, min(k_hi, ctx["hi"] - ctx["lo"] + 1))
    start = ctx["lo"] + rng.randint(0, ctx["hi"] - ctx["lo"] + 1 - k)
    values = list(range(start, start + k))
    return ctx, values, _draw_freqs(rng, k, n_lo, n_hi, zero)


def _easy(rng: random.Random) -> dict:
    ctx, values, fs = _setup(rng, 4, 5, 9, 18, True)
    k = len(values)
    mode = rng.choice(["list", "tally", "named"])
    intro, obs = _data_intro(rng, ctx, values, fs, mode)
    tally = [_tally(f) for f in fs] if mode == "tally" else None
    if tally:
        check([_read_tally(t) for t in tally] == fs, "tally marks disagree with the counts")
    else:
        _recount(obs, values, fs)
    has_total = rng.random() < 0.8
    table = _Table(values, tally_strings=tally, has_total=has_total)
    cc = table.count_col
    cells = [(r, cc) for r in range(k)] + ([(k, cc)] if has_total else [])
    blanks = _blank_set(rng, cells, [], 2, k + 1, lambda b: sum(1 for r, _ in b if r < k) >= 2)
    task = phrase(
        rng,
        [
            "Complete the frequency table. Every value from {lo} to {hi} has a row. Fill in each ?.",
            "Count how many times each value occurs and complete the frequency table. Every value from {lo} to {hi} "
            "has a row. Fill in each ?.",
            "Complete the frequency table: the Count column is how many {who} gave each value. Fill in each ?.",
        ],
        lo=values[0],
        hi=values[-1],
        who=ctx["who"],
    )
    total_note = f" The Total row is the number of {ctx['who']} counted in all." if has_total else ""
    blank_rows = {r for r, _ in blanks if r < k}
    worlds = [
        (
            "dsft-reads-the-value-as-the-count",
            dict(fs=[v if i in blank_rows else f for i, (v, f) in enumerate(zip(values, fs))]),
        )
    ]
    if has_total and (k, cc) in blanks:
        worlds += [
            ("dsft-total-is-number-of-rows", dict(fs=fs, total_rule="rows")),
            ("dsft-total-adds-the-values", dict(fs=fs, total_rule="values")),
        ]
    working = [f"{v}: {f}" for v, f in zip(values, fs)] + ([f"Total: {sum(fs)}"] if has_total else [])
    return _make_question(
        rng, f"{intro} {task}{total_note}", table, ctx, fs, blanks, worlds, mode == "tally", working, ctx["header"]
    )


def _medium(rng: random.Random) -> dict:
    ctx, values, fs = _setup(rng, 5, 6, 12, 22, True)
    k = len(values)
    mode = rng.choice(["list", "tally", "named"])
    intro, obs = _data_intro(rng, ctx, values, fs, mode)
    tally = [_tally(f) for f in fs] if mode == "tally" else None
    if tally:
        check([_read_tally(t) for t in tally] == fs, "tally marks disagree with the counts")
    else:
        _recount(obs, values, fs)
    form = rng.choice(_FORMS)
    bi = rng.choice([i for i in range(1, k - 1) if fs[i] > 0])
    t = values[bi]
    table = _Table(values, tally_strings=tally, derived=(form, t))
    cc = table.count_col
    total_cell, derived_cell = (k, cc), (k + 1, cc)
    counts = [(r, cc) for r in range(k)]
    blanks = _blank_set(
        rng,
        counts + [total_cell],
        [derived_cell],
        3,
        k + 2,
        lambda b: sum(1 for r, _ in b if r < k) >= 2 and sum(1 for r, _ in b if r < k) < k,
    )
    rows = table.rows(fs)
    check(rows[k + 1][cc] == sum(1 for o in obs if _in_form(o, t, form)), "derived row disagrees with the raw data")
    check(rows[k][cc] == len(obs), "total disagrees with the raw data")
    descr = f"{t} or fewer" if form == "or fewer" else f"{form} {t}"
    task = phrase(
        rng,
        [
            "Complete the frequency table. The Total row is the number of {who} counted in all, and the last row "
            "adds the counts for every value that is {descr}. Fill in each ?.",
            "Fill in each ? in the frequency table. Total is the number of {who} in the data. The last row is the "
            "number of {who} whose value is {descr}.",
            "Count the data into the frequency table. The Total row counts all the {who}; the last row counts only "
            "those with a value {descr}. Fill in each ?.",
        ],
        who=ctx["who"],
        descr=descr,
    )
    flip_id = "dsft-includes-the-boundary" if form in ("more than", "fewer than") else "dsft-leaves-out-the-boundary"
    worlds = [
        (flip_id, dict(fs=fs, derived_flip=True)),
        ("dsft-total-is-number-of-rows", dict(fs=fs, total_rule="rows")),
        ("dsft-total-adds-the-values", dict(fs=fs, total_rule="values")),
    ]
    blank_rows = sorted(r for r, _ in blanks if r < k)
    r = rng.choice(blank_rows)
    worlds.append(("dsft-miscount-carries-into-totals", dict(fs=[f + 1 if i == r else f for i, f in enumerate(fs)])))
    working = [f"{v}: {f}" for v, f in zip(values, fs)] + [f"Total: {len(obs)}", f"{rows[k + 1][0]}: {rows[k + 1][cc]}"]
    return _make_question(
        rng, f"{intro} {task}", table, ctx, fs, blanks, worlds, mode == "tally", working, ctx["header"]
    )


def _hard(rng: random.Random) -> dict:
    shape = rng.choice(["up", "down", "partial"])
    ctx, values, fs = _setup(rng, 5, 6, 14, 24, True)
    k = len(values)
    if shape != "partial":
        mode = rng.choice(["list", "named"])
        intro, obs = _data_intro(rng, ctx, values, fs, mode)
        _recount(obs, values, fs)
        table = _Table(values, cum=shape)
        cum_cells = [(r, 2) for r in range(k)]
        count_cells = [(r, 1) for r in range(k)]
        blanks = _blank_set(
            rng,
            count_cells + cum_cells + [(k, 1)],
            [rng.choice(cum_cells)],
            4,
            k + 3,
            lambda b: sum(1 for _, c in b if c == 2) >= 2 and any(c == 1 and r < k for r, c in b) and len(b) < 2 * k,
        )
        check(table.rows(fs)[k][1] == len(obs), "total disagrees with the raw data")
        if shape == "up":
            how = (
                "The Running total column adds each row's count to the counts of all the rows above it, so the "
                "last running total is the number of {who} in all."
            )
        else:
            how = (
                "The This many or more column is how many {who} gave that value or a higher one, so its first "
                "row is the number of {who} in all."
            )
        task = phrase(rng, ["Complete the frequency table. " + how + " Fill in each ?."], who=ctx["who"])
        worlds = [
            ("dsft-running-total-copies-the-count", dict(fs=fs, cum_rule="copy")),
            ("dsft-accumulates-from-the-wrong-end", dict(fs=fs, cum_rule="wrong-end")),
        ]
        blank_rows = sorted({r for r, c in blanks if c == 1 and r < k})
        r = rng.choice(blank_rows)
        worlds.append(
            ("dsft-miscount-carries-into-totals", dict(fs=[f + 1 if i == r else f for i, f in enumerate(fs)]))
        )
        rows = table.rows(fs)
        working = [f"{v}: count {f}, {rows[i][2]}" for i, (v, f) in enumerate(zip(values, fs))] + [f"Total: {len(obs)}"]
        return _make_question(rng, f"{intro} {task}", table, ctx, fs, blanks, worlds, False, working, ctx["header"])
    # partial: no raw data. One count is missing and the stated total gives it.
    n = sum(fs)
    m = rng.choice([i for i in range(k) if fs[i] > 0])
    table = _Table(values, cum="up", fixed_total=n)
    cum_cells = [(r, 2) for r in range(k)]
    blanks = _blank_set(rng, cum_cells, [(m, 1)], 3, k, lambda b: sum(1 for _, c in b if c == 2) >= 2)
    task = (
        f"{ctx['ask']} {n} {ctx['who']} were counted in all. The table below is only part finished: every count is "
        f"given except one. Use the Total to work out the missing count. The Running total column adds each row's "
        f"count to the counts of all the rows above it. Fill in each ?."
    )
    known = n - fs[m]
    worlds = [
        (
            "dsft-reports-the-known-sum-as-the-missing-count",
            dict(fs=[known if i == m else f for i, f in enumerate(fs)]),
        ),
        ("dsft-running-total-copies-the-count", dict(fs=fs, cum_rule="copy")),
        ("dsft-accumulates-from-the-wrong-end", dict(fs=fs, cum_rule="wrong-end")),
        ("dsft-miscount-carries-into-totals", dict(fs=[f + 1 if i == m else f for i, f in enumerate(fs)])),
    ]
    check(sum(fs) - fs[m] + fs[m] == n and known + fs[m] == n, "missing count disagrees with the total")
    rows = table.rows(fs)
    working = [f"Counts given add to {known}", f"{n} - {known} = {fs[m]}"] + [
        f"{v}: count {f}, {rows[i][2]}" for i, (v, f) in enumerate(zip(values, fs))
    ]
    return _make_question(rng, task, table, ctx, fs, blanks, worlds, False, working, ctx["header"])


@generator(
    "ds_freq_table_easy",
    level=2,
    presets={"easy": dict(tier="easy")},
)
def ds_freq_table_easy(rng: random.Random, tier: str = "easy") -> dict:
    if tier != "easy":
        raise ValueError(f"unknown tier {tier!r}")
    return _easy(rng)


@generator(
    "ds_freq_table_derived",
    level=2,
    presets={"medium": dict(tier="medium")},
)
def ds_freq_table_derived(rng: random.Random, tier: str = "medium") -> dict:
    if tier != "medium":
        raise ValueError(f"unknown tier {tier!r}")
    return _medium(rng)


@generator(
    "ds_freq_table_cumulative",
    level=3,
    presets={"hard": dict(tier="hard")},
)
def ds_freq_table_cumulative(rng: random.Random, tier: str = "hard") -> dict:
    if tier != "hard":
        raise ValueError(f"unknown tier {tier!r}")
    return _hard(rng)
