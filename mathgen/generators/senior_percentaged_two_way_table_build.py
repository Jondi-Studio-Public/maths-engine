"""Generates `senior-use-an-appropriately-percentaged-two-way-frequency-table` questions (ACMGM050): turn the
counts in a two-way frequency table into the percentages that suit the question being asked. `answer: table`;
every percentage blank is to one decimal place, and a missing count (exam tier) is a whole number.

Scope: a table of counts for an explanatory variable (the groups being compared) and a response variable. The
learner completes the blank percentage cells. The explanatory variable sits on the rows or on the columns, at
random, and the table carries no percentage totals, so nothing in the layout says which way to percentage:
that has to come from the question. Percentages of the group totals compare the groups; percentages of the
other variable's totals, or of the grand total, answer a different question and are the distractors.

Ladder (reasoning escalation, not size):
  easy:   2 x 2 table. The question names the groups ("as a percentage of the total of its year level group").
  medium: 2 x 3 or 3 x 2 table. The question asks for a comparison across groups of a named variable, so the
          learner must decide which total each count is divided by.
  hard:   3 x 3 or 3 x 2 table. The question asks whether one variable depends on another, so the learner must
          work out which variable defines the groups (it is as likely to be the columns as the rows).
  exam:   as hard, but some counts are missing and only the row and column totals remain. The learner recovers
          each missing count from its totals, then percentages it too.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import check, draw_until, generator, phrase

from ._senior_table_kit import build_answer, finish_wrong, grid_text, one_dp, rounds_clear

TIERS = ("easy", "medium", "hard", "exam")

# explanatory variable, its categories, response variable, its categories, the people or things counted, and
# a plain-language phrase for what is being compared.
_CONTEXTS = [
    dict(
        noun="students at a senior school",
        exp="year level",
        exp_cats=["Year 10", "Year 11", "Year 12"],
        resp="part-time work",
        resp_cats=["Has a job", "Does not work"],
    ),
    dict(
        noun="adults surveyed in a regional town",
        exp="age group",
        exp_cats=["18 to 34", "35 to 54", "55 and over"],
        resp="main source of news",
        resp_cats=["Online", "Television", "Newspaper"],
    ),
    dict(
        noun="patients in a clinical trial",
        exp="treatment",
        exp_cats=["New drug", "Standard drug"],
        resp="outcome",
        resp_cats=["Recovered", "Did not recover"],
    ),
    dict(
        noun="commuters in a capital city",
        exp="suburb",
        exp_cats=["Northside", "Eastside", "Westside"],
        resp="usual transport",
        resp_cats=["Car", "Train", "Bus"],
    ),
    dict(
        noun="customers at a supermarket",
        exp="store",
        exp_cats=["Town store", "Highway store"],
        resp="payment method",
        resp_cats=["Card", "Cash", "Phone"],
    ),
    dict(
        noun="employees of a large firm",
        exp="work arrangement",
        exp_cats=["On site", "Hybrid", "Remote"],
        resp="job satisfaction",
        resp_cats=["Satisfied", "Neutral", "Dissatisfied"],
    ),
    dict(
        noun="members of a sports club who play one of three sports",
        exp="membership type",
        exp_cats=["Junior", "Senior"],
        resp="sport played",
        resp_cats=["Cricket", "Tennis", "Netball"],
    ),
    dict(
        noun="households in a survey",
        exp="state",
        exp_cats=["Victoria", "Queensland", "Western Australia"],
        resp="main heating",
        resp_cats=["Gas", "Electric", "Wood or none"],
    ),
    dict(
        noun="workers at a distribution centre",
        exp="shift",
        exp_cats=["Day shift", "Night shift"],
        resp="punctuality",
        resp_cats=["Late at least once", "Never late"],
    ),
    dict(
        noun="members of a gym",
        exp="membership",
        exp_cats=["Basic", "Premium"],
        resp="attendance",
        resp_cats=["Weekly", "Rarely"],
    ),
    dict(
        noun="students who sat a mathematics test",
        exp="school",
        exp_cats=["Hillside High", "Riverside High"],
        resp="grade band",
        resp_cats=["A or B", "C", "D or E"],
    ),
    dict(
        noun="visitors to a museum",
        exp="visitor type",
        exp_cats=["Child", "Adult", "Senior"],
        resp="souvenir purchase",
        resp_cats=["Bought something", "Did not buy"],
    ),
    dict(
        noun="visitors to a national park",
        exp="season",
        exp_cats=["Summer", "Autumn", "Winter"],
        resp="main activity",
        resp_cats=["Walking", "Camping", "Photography"],
    ),
]

_SHAPES = {
    "easy": {(2, 2)},
    "medium": {(2, 3), (3, 2)},
    "hard": {(3, 3)},
    "exam": {(3, 3), (2, 3), (3, 2)},
}


def _counts(rng, n_exp, n_resp, total_range):
    """Interior counts [exp][resp] with unequal group sizes and percentages that round cleanly both ways."""

    def draw(r):
        return [[r.randint(6, 60) for _ in range(n_resp)] for _ in range(n_exp)]

    def ok(grid):
        grand = sum(map(sum, grid))
        if not total_range[0] <= grand <= total_range[1]:
            return False
        row_totals = [sum(r) for r in grid]
        col_totals = [sum(grid[i][j] for i in range(n_exp)) for j in range(n_resp)]
        if max(row_totals) < 1.3 * min(row_totals):
            return False
        for i in range(n_exp):
            for j in range(n_resp):
                if not rounds_clear(Fraction(grid[i][j] * 100, row_totals[i]), 1, Fraction(1, 20)):
                    return False
                if not rounds_clear(Fraction(grid[i][j] * 100, col_totals[j]), 1, Fraction(1, 20)):
                    return False
        # the percentages must differ by group, or there is nothing to compare
        for j in range(n_resp):
            pcts = [Fraction(grid[i][j], row_totals[i]) for i in range(n_exp)]
            if max(pcts) - min(pcts) < Fraction(4, 100):
                return False
        return True

    return draw_until(rng, draw, ok, tries=4000, what="counts with clean row and column percentages")


def _missing_cells(rng, n_rows, n_cols):
    """Interior cells to hide: at most one per row and per column, so each is recoverable from its totals."""
    k = rng.randint(2, min(n_rows, n_cols))
    rows = rng.sample(range(n_rows), k)
    cols = rng.sample(range(n_cols), k)
    return list(zip(rows, cols))


def _build(rng, tier):
    ctx = rng.choice([c for c in _CONTEXTS if (len(c["exp_cats"]), len(c["resp_cats"])) in _SHAPES[tier]])
    exp_cats, resp_cats = list(ctx["exp_cats"]), list(ctx["resp_cats"])
    n_exp, n_resp = len(exp_cats), len(resp_cats)
    grid = _counts(rng, n_exp, n_resp, (80, 260) if tier != "easy" else (60, 160))
    exp_on_rows = rng.random() < (0.5 if tier in ("easy", "medium") else 0.4)
    if exp_on_rows:
        row_cats, col_cats, inner = exp_cats, resp_cats, grid
        row_var, col_var = ctx["exp"], ctx["resp"]
    else:
        row_cats, col_cats = resp_cats, exp_cats
        inner = [[grid[i][j] for i in range(n_exp)] for j in range(n_resp)]
        row_var, col_var = ctx["resp"], ctx["exp"]
    n_rows, n_cols = len(row_cats), len(col_cats)
    row_totals = [sum(r) for r in inner]
    col_totals = [sum(inner[i][j] for i in range(n_rows)) for j in range(n_cols)]
    grand = sum(row_totals)

    # Percentages: of the total of the explanatory group (the right ones), and the two wrong bases.
    def right_pct(i, j):
        base = row_totals[i] if exp_on_rows else col_totals[j]
        return Fraction(inner[i][j] * 100, base)

    def other_pct(i, j):
        base = col_totals[j] if exp_on_rows else row_totals[i]
        return Fraction(inner[i][j] * 100, base)

    count_cols = [f"{c} (count)" for c in col_cats] + ["Total (count)"]
    pct_cols = [f"{c} (%)" for c in col_cats]
    columns = [f"{row_var} / {col_var}", *count_cols, *pct_cols]
    rows = []
    for i in range(n_rows):
        rows.append([row_cats[i], *inner[i], row_totals[i], *[one_dp(right_pct(i, j)) for j in range(n_cols)]])
    rows.append(["Total", *col_totals, grand, *["-"] * n_cols])
    pct_first = n_cols + 2  # column index of the first percentage column
    cells = [(i, pct_first + j) for i in range(n_rows) for j in range(n_cols)]
    missing = _missing_cells(rng, n_rows, n_cols) if tier == "exam" else []
    if tier == "easy":
        # A worked cell or two stays in view, which is safe here because the question names the groups.
        chosen = set(rng.sample(cells, rng.randint(2, 3)))
    else:
        # No percentage is shown: a shown one would reveal which total the others are divided by.
        chosen = set(cells)
    blank_cells = chosen | {(i, 1 + j) for i, j in missing}
    answer = build_answer(columns, rows, blank_cells)

    def group_base(i, j):
        return row_totals[i] if exp_on_rows else col_totals[j]

    wrong: dict[str, list] = {}
    for i in range(n_rows):
        for j in range(n_cols):
            wrong[f"{i},{pct_first + j}"] = [("gives-the-complement-of-the-percentage", 100 - right_pct(i, j))]
    for i, j in missing:
        wrong[f"{i},{1 + j}"] = [
            ("takes-the-row-total-as-the-count", Fraction(row_totals[i])),
            ("takes-the-column-total-as-the-count", Fraction(col_totals[j])),
        ]
    # A learner who percentages the wrong way gets every percentage wrong, not one cell: build those tables.
    pct_keys = [k for k in answer["blanks"] if int(k.split(",")[1]) >= pct_first]

    def cell_of(key):
        r, c = key.split(",")
        return int(r), int(c) - pct_first

    whole = [
        ("percentages-of-the-wrong-total", {k: other_pct(*cell_of(k)) for k in pct_keys}),
        (
            "percentages-of-the-grand-total",
            {k: Fraction(inner[cell_of(k)[0]][cell_of(k)[1]] * 100, grand) for k in pct_keys},
        ),
    ]

    # check every blank against a recount straight from the counts
    for key, value in answer["blanks"].items():
        r, c = map(int, key.split(","))
        if c >= pct_first:
            check(value == one_dp(right_pct(r, c - pct_first)), "percentage disagrees")
        else:
            check(value == inner[r][c - 1], "count disagrees")

    exp_name = ctx["exp"]
    resp_name = ctx["resp"]
    if tier == "easy":
        stem = phrase(
            rng,
            [
                "A survey recorded the {resp} of {noun}, grouped by {exp}. In the table each percentage is the "
                "count as a percentage of the total of its {exp} group. Complete the percentage columns, giving each "
                "percentage to 1 decimal place. Fill in each ? with a number.",
                "The {resp} of {noun} was recorded for each {exp}. Show each count as a percentage of the total of "
                "its {exp} group, to 1 decimal place, by completing the percentage columns. Fill in each ? with a "
                "number.",
            ],
            exp=exp_name,
            resp=resp_name,
            noun=ctx["noun"],
        )
    elif tier == "medium":
        stem = phrase(
            rng,
            [
                "A survey recorded the {resp} of {noun}. To compare the {resp} across the groups defined by {exp}, "
                "complete the percentage columns of the table, giving each percentage to 1 decimal place. "
                "Fill in each ? with a number.",
                "The table shows the {resp} of {noun}. The aim is to compare the {resp} across the groups defined by "
                "{exp}. Complete the percentage columns that suit this comparison, each to 1 decimal place. Fill in "
                "each ? with a number.",
            ],
            exp=exp_name,
            resp=resp_name,
            noun=ctx["noun"],
        )
    elif tier == "hard":
        stem = phrase(
            rng,
            [
                "A study of {noun} asks whether their {resp} depends on the {exp}. The table shows the counts. "
                "Complete the percentage columns so that the table can be used to answer this, giving each percentage "
                "to 1 decimal place. Fill in each ? with a number.",
                "To find out whether {resp} depends on the {exp} for {noun}, a table of counts was drawn up. "
                "Complete the percentage columns so that the table can be used to answer this, giving each "
                "percentage to 1 decimal place. Fill in each ? with a number.",
            ],
            exp=exp_name,
            resp=resp_name,
            noun=ctx["noun"],
        )
    else:
        stem = phrase(
            rng,
            [
                "A study of {noun} asks whether their {resp} depends on the {exp}. Some counts in the table are "
                "missing, but every total shown is correct. Find the missing counts, then complete the percentage "
                "columns so that the table can be used to answer the question, giving each percentage to 1 decimal "
                "place. Fill in each ? with a number.",
                "To find out whether {resp} depends on the {exp} for {noun}, a table was drawn up, but some counts "
                "are missing. Every total shown is correct. Find the missing counts, then complete the percentage "
                "columns so that the table can be used to answer the question, giving each percentage to 1 decimal "
                "place. Fill in each ? with a number.",
            ],
            exp=exp_name,
            resp=resp_name,
            noun=ctx["noun"],
        )
    working = [
        f"{row_cats[i]}: " + ", ".join(one_dp(right_pct(i, j)) for j in range(n_cols)) + " (% of its "
        f"{exp_name} group total)"
        for i in range(n_rows)
    ]
    return answer, stem, wrong, working, whole


@generator(
    "senior_percentaged_two_way_table",
    level=13,
    presets={t: dict(tier=t) for t in TIERS},
)
def senior_percentaged_two_way_table(rng: random.Random, tier: str = "easy") -> dict:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    answer, stem, wrong, working, whole = _build(rng, tier)
    answer_text, distractors, notes = finish_wrong(rng, answer, wrong, want=3, whole=whole)
    return dict(
        text=f"{stem} {grid_text(answer)}",
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=distractors,
        option_notes=notes,
    )
