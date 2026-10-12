"""Generates `f10-linear-conjecture-testing` questions (AC9M8A04, AC9M10A05). `answer: mcq`.

Scope: a conjecture about a linear relation is stated, and a table of values is given as
text. The learner selects the statement that is true of the table: either no entry in it breaks the conjecture, so
the table supports it, or one named entry is a counterexample. Reporting a general pattern, building a table and
proof by general symbolic derivation are out of scope; no figure is drawn.

Every option is a checkable claim about the table, and exactly one is true, by construction:
  * the table either fits the conjecture everywhere or breaks it at exactly ONE place (one row, or one step
    between neighbouring rows), decided first;
  * "no place breaks the conjecture" is offered on every question, so it is a key only when the table fits;
  * the place options each name a different place of the table, and only the broken one is true.
All four options have the same sentence shape and near-equal length, so neither the key nor its position can be
read off the wording.

Ladder (reasoning escalation, not size):
  easy:   the conjecture is a rule y = mx + c, and a counterexample is a ROW whose y-value the rule does not give.
  medium: the conjecture is about change ("y goes up by d each time x goes up by 1"), and a counterexample is a
          STEP between neighbouring rows; no single row is wrong by itself.
  hard:   the x-values are unevenly spaced and the conjecture is a rate per unit of x, so a step of 2 or 3 in x
          must be divided down; a larger change in y can still fit the conjecture.
  exam:   two tables are compared. The conjecture says one table changes k times as fast as the other over every
          step; the starting values differ, so comparing the y-values themselves would mislead.
"""

from __future__ import annotations

from mathgen.core import check, draw_until, generator, join_terms

_TIERS = ("easy", "medium", "hard", "exam")

_NAMES = ("Priya", "Marcus", "Aiko", "Tom", "Leila", "Sam", "Jess", "Kofi", "Mia", "Dev")


def _list(values) -> str:
    return ", ".join(str(v) for v in values)


def _rule_text(m: int, c: int) -> str:
    return "y = " + join_terms([(m, "x"), (c, "")])


def _change_words(d: int) -> str:
    return f"goes up by {d}" if d > 0 else f"goes down by {-d}"


class _Place:
    """One place in a table where the conjecture can be checked, with the sentence that names it."""

    def __init__(self, label: str, fits: bool, gap: int = 1):
        self.label = label
        self.fits = fits
        self.gap = gap


def _options(rng, places: list[_Place], noun: str, rule_word: str, prefer_wide: bool):
    """(key text, distractor texts, option_notes) for a table whose places fit except possibly one.

    `noun` is "row" or "step"; `rule_word` names what a broken place breaks."""
    broken = [p for p in places if not p.fits]
    check(len(broken) <= 1, "a question may have at most one broken place")
    fitting = [p for p in places if p.fits]

    def says_breaks(p: _Place) -> str:
        return f"The {noun} {p.label} breaks the {rule_word}, so the table does not support it."

    none_text = f"No {noun} breaks the {rule_word}: every {noun} agrees, so the table supports the conjecture."
    notes: dict[str, str] = {}
    if not broken:
        pool = list(fitting)
        rng.shuffle(pool)
        if prefer_wide:
            pool.sort(key=lambda p: -p.gap)
        chosen = pool[:3]
        key = none_text
        distractors = [says_breaks(p) for p in chosen]
        for p in chosen:
            notes[says_breaks(p)] = "reports-a-place-that-fits-as-a-counterexample"
            if p.gap > 1:
                notes[says_breaks(p)] = "compares-changes-without-allowing-for-the-x-step"
    else:
        key = says_breaks(broken[0])
        pool = list(fitting)
        rng.shuffle(pool)
        if prefer_wide:
            pool.sort(key=lambda p: -p.gap)
        chosen = pool[:2]
        distractors = [says_breaks(p) for p in chosen] + [none_text]
        for p in chosen:
            notes[says_breaks(p)] = "reports-a-place-that-fits-as-a-counterexample"
            if p.gap > 1:
                notes[says_breaks(p)] = "compares-changes-without-allowing-for-the-x-step"
        notes[none_text] = "accepts-the-conjecture-after-checking-only-some-places"
    check(len(set(distractors)) == len(distractors) and key not in distractors, "option texts must be distinct")
    check(len(distractors) == 3, "a question needs three distractors")
    return key, distractors, notes


def _easy(rng):
    name = rng.choice(_NAMES)
    m = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5])
    c = rng.choice([-6, -5, -3, -2, 0, 1, 2, 3, 4, 5, 7])
    start = rng.choice((0, 1))
    xs = list(range(start, start + 5))
    ys = [m * x + c for x in xs]
    holds = rng.random() < 0.4
    bad = None
    if not holds:
        bad = rng.randrange(len(xs))
        ys[bad] += rng.choice([-3, -2, -1, 1, 2, 3])
    check(
        [i for i, (x, y) in enumerate(zip(xs, ys)) if y != m * x + c] == ([] if holds else [bad]),
        "the table must break the rule at exactly the chosen place",
    )
    places = [_Place(f"x = {x}", y == m * x + c) for x, y in zip(xs, ys)]
    key, distractors, notes = _options(rng, places, "row", "rule", prefer_wide=False)
    text = (
        f"{name} conjectures that the rule {_rule_text(m, c)} gives every y-value in this table of values. "
        f"x: {_list(xs)}; y: {_list(ys)}. Which statement is true?"
    )
    working = ["For each row, work out the y-value the rule gives and compare it with the y in the table."] + [
        f"x = {x}: {_rule_text(m, c)} gives {m * x + c}; the table has {y}." for x, y in zip(xs, ys)
    ]
    return text, key, distractors, notes, working


def _step_places(xs, ys, rate):
    """Places for each step between neighbouring rows: fits when the change in y is `rate` times the change in x."""
    return [
        _Place(
            f"from x = {xs[i]} to x = {xs[i + 1]}", (ys[i + 1] - ys[i]) == rate * (xs[i + 1] - xs[i]), xs[i + 1] - xs[i]
        )
        for i in range(len(xs) - 1)
    ]


def _build_ys(rng, xs, rate, start_y, holds):
    """y-values that change by `rate` per unit of x over every step, except one step with an extra change."""
    ys = [start_y]
    bad = None if holds else rng.randrange(len(xs) - 1)
    extra = rng.choice([-3, -2, -1, 1, 2, 3])
    for i in range(len(xs) - 1):
        step = rate * (xs[i + 1] - xs[i]) + (extra if i == bad else 0)
        ys.append(ys[-1] + step)
    return ys, bad


def _medium(rng):
    name = rng.choice(_NAMES)
    d = rng.choice([-4, -3, -2, 2, 3, 4, 5, 6])
    start = rng.choice((0, 1))
    xs = list(range(start, start + rng.choice((5, 6))))
    holds = rng.random() < 0.4
    ys, bad = _build_ys(rng, xs, d, rng.randint(-5, 12), holds)
    places = _step_places(xs, ys, d)
    check(
        sum(not p.fits for p in places) == (0 if holds else 1), "the table must break the pattern at exactly one step"
    )
    key, distractors, notes = _options(rng, places, "step", "pattern", prefer_wide=False)
    text = (
        f"{name} conjectures that in this table of values y {_change_words(d)} each time x goes up by 1. "
        f"x: {_list(xs)}; y: {_list(ys)}. Which statement is true?"
    )
    working = [f"Find the change in y for each step of 1 in x; the conjecture says it is {d}."] + [
        f"x = {xs[i]} to x = {xs[i + 1]}: y changes by {ys[i + 1] - ys[i]}." for i in range(len(xs) - 1)
    ]
    return text, key, distractors, notes, working


def _uneven_xs(rng, count):
    """Strictly increasing x-values whose gaps are 1, 2 or 3, differ from one another and include at least two
    gaps wider than 1."""

    def draw(_rng):
        return [_rng.choice((1, 1, 2, 2, 3)) for _ in range(count - 1)]

    gaps = draw_until(
        rng, draw, lambda g: sum(1 for x in g if x > 1) >= 2 and len(set(g)) >= 2, what="gaps with two wide steps"
    )
    xs = [rng.choice((0, 1, 2))]
    for gap in gaps:
        xs.append(xs[-1] + gap)
    return xs


def _hard(rng):
    name = rng.choice(_NAMES)
    d = rng.choice([-3, -2, -1, 1, 2, 3, 4, 5])
    xs = _uneven_xs(rng, rng.choice((5, 6)))
    holds = rng.random() < 0.4
    ys, bad = _build_ys(rng, xs, d, rng.randint(-5, 12), holds)
    places = _step_places(xs, ys, d)
    check(
        sum(not p.fits for p in places) == (0 if holds else 1), "the table must break the pattern at exactly one step"
    )
    key, distractors, notes = _options(rng, places, "step", "pattern", prefer_wide=True)
    text = (
        f"{name} conjectures that in this table of values y changes by {d} for every 1 that x goes up, "
        f"whatever the size of the step in x. x: {_list(xs)}; y: {_list(ys)}. Which statement is true?"
    )
    working = [f"For each step, divide the change in y by the change in x; the conjecture says the result is {d}."] + [
        f"x = {xs[i]} to x = {xs[i + 1]}: y changes by {ys[i + 1] - ys[i]} over {xs[i + 1] - xs[i]} in x, "
        f"which is {ys[i + 1] - ys[i]} / {xs[i + 1] - xs[i]}."
        for i in range(len(xs) - 1)
    ]
    return text, key, distractors, notes, working


def _exam(rng):
    name = rng.choice(_NAMES)
    d_a = rng.choice([-2, -1, 1, 2, 3])
    k = rng.choice((2, 3, 4))
    xs = _uneven_xs(rng, rng.choice((5, 6)))
    holds = rng.random() < 0.4
    ys_a, _ = _build_ys(rng, xs, d_a, rng.randint(-4, 10), True)
    start_b = rng.randint(-6, 14)
    ys_b = [start_b]
    bad = None if holds else rng.randrange(len(xs) - 1)
    extra = rng.choice([-3, -2, -1, 1, 2, 3])
    for i in range(len(xs) - 1):
        change_a = ys_a[i + 1] - ys_a[i]
        ys_b.append(ys_b[-1] + k * change_a + (extra if i == bad else 0))
    places = [
        _Place(
            f"from x = {xs[i]} to x = {xs[i + 1]}",
            (ys_b[i + 1] - ys_b[i]) == k * (ys_a[i + 1] - ys_a[i]),
            xs[i + 1] - xs[i],
        )
        for i in range(len(xs) - 1)
    ]
    check(
        sum(not p.fits for p in places) == (0 if holds else 1), "the tables must break the pattern at exactly one step"
    )
    key, distractors, notes = _options(rng, places, "step", "pattern", prefer_wide=True)
    text = (
        f"{name} records y-values for the same x-values in two tables, A and B, and conjectures that over every "
        f"step in x, the change in y in table B is {k} times the change in y in table A. "
        f"x: {_list(xs)}; table A y: {_list(ys_a)}; table B y: {_list(ys_b)}. Which statement is true?"
    )
    working = [f"For each step, compare the change in y in table B with {k} times the change in y in table A."] + [
        f"x = {xs[i]} to x = {xs[i + 1]}: table A changes by {ys_a[i + 1] - ys_a[i]}, so {k} times that is "
        f"{k * (ys_a[i + 1] - ys_a[i])}; table B changes by {ys_b[i + 1] - ys_b[i]}."
        for i in range(len(xs) - 1)
    ]
    return text, key, distractors, notes, working


_BUILDERS = {"easy": _easy, "medium": _medium, "hard": _hard, "exam": _exam}


@generator("f10_linear_conjecture", level=9, presets={tier: dict(tier=tier) for tier in _TIERS})
def f10_linear_conjecture(rng, tier="easy"):
    check(tier in _BUILDERS, f"unknown tier {tier!r}")
    text, key, distractors, notes, working = _BUILDERS[tier](rng)
    return dict(
        text=text,
        latex="",
        answer=key,
        answer_text=key,
        working=working,
        distractors=distractors,
        option_notes=notes,
        answer_kind="mcq",
    )
