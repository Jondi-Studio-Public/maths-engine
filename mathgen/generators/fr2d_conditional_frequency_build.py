"""Generators for `f10-conditional-relative-frequency`: P(A given B) from a two-way table of trial or survey counts.

Three tiers answered as fractions, plus a multi-part exam chain:

* easy: P(A given B) or P(B given A) from a table that shows its row and column totals, so the group conditioned on
  is read straight off.
* medium: a table of four counts only, conditioning on an event or on its opposite ("given that it did not ...").
* hard: how much greater the chance of A is given B than overall (a difference of two relative frequencies), from a
  table of four counts. This is the comparison that decides whether two events look independent.
* exam (`fr2d_conditional_frequency_exam`): a lost count, P(A given B), P(A given not B), and the conclusion the table
  supports (some tables are exactly independent).

Every key is re-derived a second way (a direct count over rebuilt records, exact fractions) and every option is a
proper chance or a difference of two, never equal to the key.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
    rational_text,
)

from ._fr2d_kit import FRACTION_SUFFIX, Short, frac_d, int_d, is_proper, need, redraw

_CONTEXTS = (
    dict(
        lead="A survey of {n} students asked how they get to school and whether they play a sport.",
        row_title="travel",
        rows=("Walks", "Does not walk"),
        row_ev=("the student walks to school", "the student does not walk to school"),
        col_title="sport",
        cols=("Plays a sport", "Plays no sport"),
        col_ev=("the student plays a sport", "the student plays no sport"),
    ),
    dict(
        lead="A weather station recorded, over {n} days, whether it rained and whether the wind was strong.",
        row_title="rain",
        rows=("Rain", "No rain"),
        row_ev=("it rained", "it did not rain"),
        col_title="wind",
        cols=("Strong wind", "Wind not strong"),
        col_ev=("the wind was strong", "the wind was not strong"),
    ),
    dict(
        lead="A cafe recorded what {n} customers bought: whether they bought a coffee and whether they bought a cake.",
        row_title="coffee",
        rows=("Coffee", "No coffee"),
        row_ev=("the customer bought a coffee", "the customer did not buy a coffee"),
        col_title="cake",
        cols=("Cake", "No cake"),
        col_ev=("the customer bought a cake", "the customer did not buy a cake"),
    ),
    dict(
        lead="A basketballer took {n} practice shots, some from the left of the key and some from the right.",
        row_title="side",
        rows=("From the left", "From the right"),
        row_ev=("the shot was taken from the left", "the shot was taken from the right"),
        col_title="result",
        cols=("Scored", "Missed"),
        col_ev=("the shot scored", "the shot missed"),
    ),
    dict(
        lead="In a game, a coin was tossed and a spinner was spun together, {n} times.",
        row_title="coin",
        rows=("Heads", "Tails"),
        row_ev=("the coin landed heads", "the coin landed tails"),
        col_title="spinner",
        cols=("Red", "Not red"),
        col_ev=("the spinner stopped on red", "the spinner did not stop on red"),
    ),
    dict(
        lead="A school garden club planted {n} seeds, some with fertiliser and some without, and recorded which "
        "germinated.",
        row_title="fertiliser",
        rows=("Fertiliser", "No fertiliser"),
        row_ev=("the seed had fertiliser", "the seed had no fertiliser"),
        col_title="germination",
        cols=("Germinated", "Did not germinate"),
        col_ev=("the seed germinated", "the seed did not germinate"),
    ),
)

_ASK = (
    "Estimate the probability that {x}, given that {y}.",
    "One record is chosen at random. Estimate the probability that {x}, knowing that {y}.",
    "Given that {y}, estimate the probability that {x}.",
)


def _grid_text(ctx, grid, totals=False, unknown=None):
    """The table as one line of text; `?` marks the cell `unknown` (row, column)."""
    (ab, anb), (nab, nanb) = grid
    r0, r1 = ctx["rows"]
    c0, c1 = ctx["cols"]

    def cell(i, j, v):
        return "?" if unknown == (i, j) else str(v)

    head = f"Table ({ctx['row_title']} / {ctx['col_title']} — {c0} — {c1}"
    if totals:
        return (
            f"{head} — Total): "
            f"{r0} — {cell(0, 0, ab)} — {cell(0, 1, anb)} — {ab + anb}; "
            f"{r1} — {cell(1, 0, nab)} — {cell(1, 1, nanb)} — {nab + nanb}; "
            f"Total — {ab + nab} — {anb + nanb} — {ab + anb + nab + nanb}."
        )
    return f"{head}): {r0} — {cell(0, 0, ab)} — {cell(0, 1, anb)}; {r1} — {cell(1, 0, nab)} — {cell(1, 1, nanb)}."


def _draw_grid(rng, lo=3, hi=24, total_lo=40, total_hi=120):
    for _ in attempts(3000, "a two-way table of counts"):
        cells = [rng.randint(lo, hi) for _ in range(4)]
        if total_lo <= sum(cells) <= total_hi:
            return cells
    raise AssertionError  # pragma: no cover


def _records(grid):
    """The table rebuilt as individual records (row event, column event), for the independent count."""
    (ab, anb), (nab, nanb) = grid
    return [(0, 0)] * ab + [(0, 1)] * anb + [(1, 0)] * nab + [(1, 1)] * nanb


def _cond(grid, x, y):
    """P(x | y) from the table, where x and y are (axis, index) with axis 0 = row event, 1 = column event."""
    recs = _records(grid)
    in_y = [r for r in recs if r[y[0]] == y[1]]
    both = [r for r in in_y if r[x[0]] == x[1]]
    return Fraction(len(both), len(in_y))


def _prob(grid, x):
    recs = _records(grid)
    return Fraction(sum(1 for r in recs if r[x[0]] == x[1]), len(recs))


def _cell(grid, x, y):
    return grid[x[1]][y[1]] if (x[0], y[0]) == (0, 1) else grid[y[1]][x[1]]


def _margin(grid, y):
    if y[0] == 1:
        return grid[0][y[1]] + grid[1][y[1]]
    return grid[y[1]][0] + grid[y[1]][1]


def _event_text(ctx, e):
    return (ctx["row_ev"] if e[0] == 0 else ctx["col_ev"])[e[1]]


def _cond_options(grid, x, y, p):
    """Wrong options for P(x | y): each is a chance a learner reaches by a named slip."""
    total = sum(sum(r) for r in grid)
    both = _cell(grid, x, y)
    ny = (y[0], 1 - y[1])
    nx = (x[0], 1 - x[1])
    swapped = Fraction(both, _margin(grid, x))
    cands = [
        frac_d(Fraction(both, total), "fr2d-joint-for-conditional"),
        frac_d(swapped, "fr2d-swaps-condition-and-target"),
        frac_d(Fraction(_cell(grid, nx, y), _margin(grid, y)), "fr2d-answers-for-the-opposite-event"),
        frac_d(Fraction(_margin(grid, x), total), "fr2d-ignores-the-condition"),
        frac_d(Fraction(both, _margin(grid, ny)), "fr2d-divides-by-the-wrong-group"),
        frac_d(Fraction(both, _margin(grid, y) + _margin(grid, x) - both), "fr2d-divides-by-the-wrong-group"),
    ]
    return [c for c in cands if is_proper(c)]


def _fraction_part(p, working, distractors, **extra):
    return dict(
        answer_kind="fraction",
        answer=exact_value(p),
        answer_text=rational_text(p),
        answer_latex=latex_answer(rational_text(p)),
        working=working,
        distractors=distractors,
        **extra,
    )


def _pick_events(rng, hard=False):
    """The event asked about and the one conditioned on, from different axes."""
    ax = rng.choice([0, 1])
    x = (ax, 0)
    y = (1 - ax, rng.choice([0, 1]) if not hard else 0)
    return x, y


def _grid(rng):
    cells = _draw_grid(rng)
    return [[cells[0], cells[1]], [cells[2], cells[3]]]


def _easy(rng):
    ctx = rng.choice(_CONTEXTS)
    for _ in attempts(2000, "a table with a clear answer"):
        grid = _grid(rng)
        x, y = _pick_events(rng)
        y = (y[0], 0)  # condition on the first group, so the total shown is one of the totals
        p = _cond(grid, x, y)
        if not 0 < p < 1 or p == Fraction(1, 2):
            continue
        opts = pick_distractors(rational_text(p), _cond_options(grid, x, y, p), 3)
        if len(opts) == 3:
            break
    total = sum(map(sum, grid))
    n = _margin(grid, y)
    both = _cell(grid, x, y)
    check(p == Fraction(both, n), "conditional frequency differs from the cell over the margin")
    stem = (
        f"{ctx['lead'].format(n=total)} {_grid_text(ctx, grid, totals=True)} "
        + phrase(rng, _ASK, x=_event_text(ctx, x), y=_event_text(ctx, y))
        + f" {FRACTION_SUFFIX}"
    )
    working = [
        f"Only the {n} records where {_event_text(ctx, y)} count.",
        f"Of those, {both} are records where {_event_text(ctx, x)}.",
        f"{both}/{n} = {rational_text(p)}.",
    ]
    return dict(text=stem, **_fraction_part(p, working, opts))


def _medium(rng):
    ctx = rng.choice(_CONTEXTS)
    for _ in attempts(2000, "a table with a clear answer"):
        grid = _grid(rng)
        ax = rng.choice([0, 1])
        x = (ax, rng.choice([0, 1]))
        y = (1 - ax, rng.choice([0, 1]))
        if y[1] == 0 and x[1] == 0:
            continue  # the plain "A given B" case belongs to the easy tier
        p = _cond(grid, x, y)
        if not 0 < p < 1 or p == Fraction(1, 2):
            continue
        opts = pick_distractors(rational_text(p), _cond_options(grid, x, y, p), 3)
        if len(opts) == 3:
            break
    total = sum(map(sum, grid))
    n = _margin(grid, y)
    both = _cell(grid, x, y)
    check(p == Fraction(both, n), "conditional frequency differs from the cell over the margin")
    stem = (
        f"{ctx['lead'].format(n=total)} {_grid_text(ctx, grid)} "
        + phrase(rng, _ASK, x=_event_text(ctx, x), y=_event_text(ctx, y))
        + f" {FRACTION_SUFFIX}"
    )
    working = [
        f"The records where {_event_text(ctx, y)} number {n} (the two counts in that {'row' if y[0] == 0 else 'column'} added).",
        f"Of those, {both} are records where {_event_text(ctx, x)}.",
        f"{both}/{n} = {rational_text(p)}.",
    ]
    return dict(text=stem, **_fraction_part(p, working, opts))


def _hard(rng):
    ctx = rng.choice(_CONTEXTS)
    for _ in attempts(2000, "a table with a clear answer"):
        grid = _grid(rng)
        ax = rng.choice([0, 1])
        x = (ax, rng.choice([0, 1]))
        y = (1 - ax, rng.choice([0, 1]))
        pc, pu = _cond(grid, x, y), _prob(grid, x)
        d = pc - pu
        if d <= 0 or d.denominator > 400:
            continue
        total = sum(map(sum, grid))
        n = _margin(grid, y)
        both = _cell(grid, x, y)
        other_cond = _cond(grid, y, x)
        cands = [
            frac_d(pc, "fr2d-conditional-not-compared"),
            frac_d(pu, "fr2d-overall-not-compared"),
            frac_d(pu - pc, "fr2d-difference-the-wrong-way"),
            frac_d(other_cond - _prob(grid, y), "fr2d-swaps-condition-and-target"),
            frac_d(Fraction(both, total) - pu, "fr2d-joint-for-conditional"),
        ]
        cands = [
            c
            for c in cands
            if abs(Fraction(c.value["p"], c.value["q"]) if isinstance(c.value, dict) else Fraction(c.value)) < 1
        ]
        opts = pick_distractors(rational_text(d), cands, 3)
        if len(opts) == 3:
            break
    check(d == Fraction(both, n) - Fraction(_margin(grid, x), total), "difference disagrees with the cell counts")
    stem = (
        f"{ctx['lead'].format(n=total)} {_grid_text(ctx, grid)} "
        f"How much greater is the chance that {_event_text(ctx, x)}, given that {_event_text(ctx, y)}, "
        f"than the chance that {_event_text(ctx, x)} for a record chosen at random from all {total}? "
        f"{FRACTION_SUFFIX}"
    )
    working = [
        f"Given that {_event_text(ctx, y)}: {both}/{n} = {rational_text(pc)}.",
        f"For all records: {_margin(grid, x)}/{total} = {rational_text(pu)}.",
        f"{rational_text(pc)} - {rational_text(pu)} = {rational_text(d)}.",
    ]
    return dict(text=stem, **_fraction_part(d, working, opts))


@generator(
    "fr2d_conditional_frequency",
    level=10,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
@redraw
def fr2d_conditional_frequency(rng, tier: str = "easy"):
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


# -- exam chain -------------------------------------------------------------------------------------------------

_CONCLUSIONS = (
    "P(A given B) is greater than P(A given not B), so A is more likely when B happens",
    "P(A given B) is less than P(A given not B), so A is less likely when B happens",
    "P(A given B) equals P(A given not B), so A and B look independent",
    "A and B cannot be independent, because they can both happen in the same record",
)


def _conclusion(which):
    return _CONCLUSIONS[which]


@generator("fr2d_conditional_frequency_exam", level=10, presets={"exam": {"tier": "exam"}})
@redraw
def fr2d_conditional_frequency_exam(rng, tier: str = "exam"):
    ctx = rng.choice(_CONTEXTS)
    want_independent = rng.random() < 0.25
    for _ in attempts(6000, "a table for the exam chain"):
        if want_independent:
            # rows in proportion: the cells are (a*c, a*d) and (b*c, b*d)
            a, b, c, d = (rng.randint(1, 6) for _ in range(4))
            cells = [a * c, a * d, b * c, b * d]
            if len({a / b, c / d}) < 2 or not 60 <= sum(cells) <= 160 or min(cells) < 4:
                continue
        else:
            cells = _draw_grid(rng, 4, 30, 60, 160)
        ab, anb, nab, nanb = cells
        independent = ab * nanb == anb * nab
        if want_independent != independent:
            continue
        p_b = Fraction(ab, ab + nab)
        p_nb = Fraction(anb, anb + nanb)
        if not 0 < p_b < 1 or not 0 < p_nb < 1 or p_b == Fraction(1, 2):
            continue
        break
    grid = [[ab, anb], [nab, nanb]]
    total = sum(cells)
    x, y, ny = (0, 0), (1, 0), (1, 1)
    p_ab = _cond(grid, x, y)  # P(A | B)
    p_anb = _cond(grid, x, ny)  # P(A | not B)
    check(p_ab == p_b and p_anb == p_nb, "conditional chances differ from the direct fractions")
    which = 2 if independent else (0 if p_ab > p_anb else 1)
    col_b_total = ab + nab
    col_nb_total = anb + nanb
    ra, cb = ctx["row_ev"][0], ctx["col_ev"][0]
    stem = (
        f"{ctx['lead'].format(n=total)} One count in the table has been lost. "
        f"{_grid_text(ctx, grid, unknown=(0, 0))} "
        f"Let A be the event that {ra} and B the event that {cb}."
    )
    known = anb + nab + nanb
    a_opts = pick_distractors(
        str(ab),
        [
            int_d(known, "fr2d-reports-the-known-sum"),
            int_d(total, "fr2d-reports-the-total"),
            int_d(total - anb - nab, "fr2d-left-a-count-out-of-the-sum"),
            int_d(ab + 1, "fr2d-count-off-by-one"),
            int_d(max(1, ab - 1), "fr2d-count-off-by-one"),
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    b_cands = [
        frac_d(Fraction(ab, total), "fr2d-joint-for-conditional"),
        frac_d(Fraction(ab, ab + anb), "fr2d-swaps-condition-and-target"),
        frac_d(Fraction(ab, col_nb_total + ab), "fr2d-divides-by-the-wrong-group"),
        frac_d(Fraction(nab, col_b_total), "fr2d-answers-for-the-opposite-event"),
        frac_d(Fraction(ab + anb, total), "fr2d-ignores-the-condition"),
    ]
    c_cands = [
        frac_d(Fraction(anb, total), "fr2d-joint-for-conditional"),
        frac_d(Fraction(anb, ab + anb), "fr2d-swaps-condition-and-target"),
        frac_d(Fraction(nanb, col_nb_total), "fr2d-answers-for-the-opposite-event"),
        frac_d(p_ab, "fr2d-condition-and-opposite-confused"),
        frac_d(Fraction(anb, col_b_total + anb), "fr2d-divides-by-the-wrong-group"),
    ]
    b_opts = pick_distractors(rational_text(p_ab), [c for c in b_cands if is_proper(c)], 3)
    c_opts = pick_distractors(rational_text(p_anb), [c for c in c_cands if is_proper(c)], 3)
    d_key = _conclusion(which)
    d_all = [
        Distractor(_conclusion(i), tag)
        for i, tag in (
            (0, "fr2d-reads-the-comparison-backwards"),
            (1, "fr2d-reads-the-comparison-backwards"),
            (2, "fr2d-calls-different-chances-independent"),
            (3, "fr2d-confuses-independent-with-exclusive"),
        )
        if i != which
    ]
    if independent:
        d_all[0] = Distractor(d_all[0].text, "fr2d-sees-a-link-in-equal-chances")
        d_all[1] = Distractor(d_all[1].text, "fr2d-sees-a-link-in-equal-chances")
    need(a_opts, b_opts, c_opts)
    pb, pc_ = rational_text(p_ab), rational_text(p_anb)
    parts = [
        dict(
            prompt="Find the lost count: the number of records where both A and B happened.",
            answer_kind="number",
            answer=ab,
            answer_text=str(ab),
            answer_latex=latex_answer(str(ab)),
            marks=1,
            sensible_range=(0, total),
            working=[f"{total} - {anb} - {nab} - {nanb} = {ab}."],
            distractors=a_opts,
        ),
        dict(
            prompt=f"Estimate the probability that A happens, given that B happens. {FRACTION_SUFFIX}",
            marks=2,
            sensible_range=(0, 1),
            follow_through=f"part_a/(part_a+{nab})",
            **_fraction_part(
                p_ab, [f"{ab} + {nab} = {col_b_total} records with B.", f"{ab}/{col_b_total} = {pb}."], b_opts
            ),
        ),
        dict(
            prompt=f"Estimate the probability that A happens, given that B does not happen. {FRACTION_SUFFIX}",
            marks=2,
            sensible_range=(0, 1),
            **_fraction_part(
                p_anb, [f"{anb} + {nanb} = {col_nb_total} records without B.", f"{anb}/{col_nb_total} = {pc_}."], c_opts
            ),
        ),
        dict(
            prompt="Which statement do these two chances support?",
            answer_kind="mcq",
            answer=d_key,
            answer_text=d_key,
            answer_latex=latex_answer(d_key, plain=True),
            marks=2,
            working=[f"Compare {pb} with {pc_}."],
            distractors=d_all,
        ),
    ]
    return dict(text=stem, parts=parts)
