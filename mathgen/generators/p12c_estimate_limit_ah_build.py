"""Generates `p12c_estimate_limit_ah` questions: estimate the limit of (a^h - 1)/h as h tends to 0 using
technology, for a stated a > 0.

The contract this module implements:

    id: senior-estimate-the-limit-of-frac-a-h-1
    name: 'Estimate the limit of (a^h-1)/h as h->0 using technology, for
      various values of a>0'
    domain: differentiation
    contract: Given bounded, fully specified inputs, estimate the
      limit of (a^h-1)/h as h->0 using technology, for various values of
      a>0. Return the resulting value, expression, relation, or
      classification.
    descriptor: ACMMM098
    prerequisites: [differentiate-polynomial]
    answer kind: numeric or symbolic
    capability gate: [technology_use]
    out of scope: unbounded investigations, unstated assumptions, assessment
      modes outside the capability gate

The descriptor needs no rewrite: it is already a bounded numeric task (the estimated limit value for a stated
a), and the technology is the calculator used to reach it. ``calculator: allowed`` and the ``technology_use``
gate stays cited exactly as stated (no ``capability_gates`` override).

THE CRITICAL BOUNDARY (stated here so it survives a diff of this module alone): this skill
must NEVER present the answer as, or accept as correct-by-construction, the exact symbolic value ``ln(a)`` --
the learner is estimating from numerical evidence (a table of ``(a^h - 1)/h`` at decreasing ``h``), not
proving a limit analytically.

Table values are computed as plain floats (``math.pow``), never
``Fraction``-style rational powers (a rational power is not generally
rational) -- the point is the DISPLAYED TREND a learner reads off a
calculator, exactly as the "using technology" clause describes. The
correct MCQ option is ``round(math.log(a), 2)`` formatted as a plain signed
numeral; it is verified against an independent ``sympy.log(a).evalf()``
route (never re-using the ``math.log`` call the construction itself used for
the key) and against the table's own displayed last (smallest-``h``) row,
which must itself round to the offered correct option -- the same check a
learner actually performs.
"""

from __future__ import annotations

import math
from fractions import Fraction

from sympy import Rational, log as sympy_log

from ..core import (
    check,
    check_pairwise_distinct,
    generator,
    latex_answer,
    pick_distractors,
)

OPTION_DP = 2

# Floor on the minimum gap between any two MCQ option numerals after rounding to OPTION_DP -- two
# units in the last displayed place.
_MIN_OPTION_GAP = Fraction(1, 50)

#: h-value ladders per tier. hard adds a fourth, smaller-h row (the genuine
#: reasoning escalation: the learner must extrapolate a CONVERGING trend
#: across four rows, not read a value off three).
_H_VALUES = {
    "easy": (Fraction(1, 10), Fraction(1, 100), Fraction(1, 1000)),
    "medium": (Fraction(1, 10), Fraction(1, 100), Fraction(1, 1000)),
    "hard": (Fraction(1, 10), Fraction(1, 100), Fraction(1, 1000), Fraction(1, 10000)),
}

#: Every template is a SINGLE flowing sentence with no ``\n`` -- the front
#: end renders `question.text` into a span with no white-space rule and no
#: newline-to-``<br>`` step, so a literal
#: newline collapses to nothing and the table rows a learner is meant to
#: read run together illegibly. Table rows are joined inline with ``, ``
#: instead (`_rows_for`), matching the house style `p9c_difference_quotient_build.py`
#: already uses for the same "read a numeric trend" shape.
_SINGLE_TEMPLATES = [
    "The table below shows values of (a^h - 1)/h for a = {a}, computed using "
    "technology, as h approaches 0 from above: {rows}. Using the trend in "
    "the table, estimate the limit of (a^h - 1)/h as h -> 0, to 2 decimal "
    "places.",
    "For a = {a}, the value of (a^h - 1)/h has been calculated at several "
    "small positive values of h: {rows}. What does this table suggest is "
    "the limit of (a^h - 1)/h as h -> 0? Give your answer to 2 decimal "
    "places.",
    "A student uses technology to compute (a^h - 1)/h for a = {a} at "
    "decreasing values of h: {rows}. Based on the trend shown, estimate "
    "lim(h->0) (a^h - 1)/h to 2 decimal places.",
    "Using technology, (a^h - 1)/h is evaluated for a = {a} at each of the "
    "following values of h: {rows}. What value does the table appear to be "
    "converging to as h -> 0? Answer to 2 decimal places.",
]

_EXAM_TEMPLATES = [
    "Two tables of (a^h - 1)/h are computed using technology, for two "
    "different values of a, as h -> 0. First, for a = {a1}: {rows1}. "
    "Second, for a = {a2}: {rows2}. Which of these two estimated limits is "
    "further from zero (has the larger magnitude)?",
    "A student uses technology to estimate lim(h->0) (a^h - 1)/h for two "
    "values of a. First, for a = {a1}: {rows1}. Second, for a = "
    "{a2}: {rows2}. Based on the two tables, which estimated limit has the "
    "larger magnitude (is further from zero)?",
    "The value of (a^h - 1)/h is calculated at decreasing values of h for "
    "two different constants a, using technology. First, for a = "
    "{a1}: {rows1}. Second, for a = {a2}: {rows2}. Comparing the two "
    "tables, which estimated limit is further from zero?",
]

# exam-tier minimum gap between the two sides' |ln(a)| magnitudes -- the tie case (both sides
# equally far from zero) is EXCLUDED BY CONSTRUCTION, never offered as the key.
_EXAM_MIN_GAP = 0.10


def _fmt_a(a) -> str:
    if isinstance(a, int):
        return str(a)
    return f"{float(a):.2f}"


def _to_rational(a):
    if isinstance(a, int):
        return Rational(a)
    return Rational(a.numerator, a.denominator)


def _fmt_h(h: Fraction) -> str:
    return f"{float(h):g}"


def _fmt_signed(v: float, dp: int = OPTION_DP) -> str:
    v = round(v, dp)
    if v == 0:
        v = 0.0
    return f"{v:.{dp}f}"


def _rows_for(a, h_values) -> tuple[str, list[float]]:
    """Table rows, computed as plain floats via ``math.pow`` (never a
    ``Fraction`` power -- a rational power is not generally rational; the
    table is what a learner's calculator would show)."""
    a_float = float(a)
    lines = []
    values = []
    for h in h_values:
        v = (math.pow(a_float, float(h)) - 1.0) / float(h)
        check(math.isfinite(v), f"table value overflowed for a={a}, h={h}")
        values.append(v)
        lines.append(f"h = {_fmt_h(h)}: {v:.4f}")
    return ", ".join(lines), values


def _draw_easy_a(rng) -> int:
    return rng.randint(2, 10)


def _draw_medium_a(rng) -> Fraction:
    return Fraction(rng.randint(15, 95), 100)


def _draw_mixed_a(rng):
    return _draw_easy_a(rng) if rng.random() < 0.5 else _draw_medium_a(rng)


def _pick_ranked(rng, key_str, key_value, pool, forced):
    """Three distractors from ``pool`` ({text: value}) that put the key at a random rank among the four options.

    ``forced`` texts are always included. Returns None when no draw for the chosen rank keeps every option at
    least ``_MIN_OPTION_GAP`` apart, so the caller re-draws ``a``.
    """
    rank = rng.randint(1, 4)  # 1 = key is the smallest option
    below = [k for k, v in pool.items() if v < key_value - 1e-9]
    above = [k for k, v in pool.items() if v > key_value + 1e-9]
    need_below, need_above = rank - 1, 4 - rank
    f_below = [k for k in forced if k in below]
    f_above = [k for k in forced if k in above]
    if len(f_below) + len(f_above) != len(forced) or len(f_below) > need_below or len(f_above) > need_above:
        return None
    if len(below) < need_below or len(above) < need_above:
        return None
    rng.shuffle(below)
    rng.shuffle(above)
    chosen = f_below + [k for k in below if k not in f_below][: need_below - len(f_below)]
    chosen += f_above + [k for k in above if k not in f_above][: need_above - len(f_above)]
    options = [key_str, *chosen]
    if len(set(options)) != 4:
        return None
    for i in range(4):
        for j in range(i + 1, 4):
            if abs(Fraction(options[i]) - Fraction(options[j])) < _MIN_OPTION_GAP:
                return None
    return chosen


def _draw_valid(rng, a_pool_fn, h_values, require_h01: bool = False, max_attempts: int = 500) -> dict:
    """Draw a plausible, internally-consistent, distinctly-optioned question.

    Re-draws ``a`` (never mutates a bad draw) when any of the following
    fails:
    - plausibility: |round(ln(a), 2)| >= 0.05 (never a degenerate "limit is
      zero" question);
    - internal consistency: the table's own displayed LAST (smallest-h) row
      rounds to the same 2-d.p. value as the offered correct option -- the
      Taylor correction term h*(ln a)^2/2 can otherwise round the two
      differently right at a rounding boundary (observed for a=10 on the
      easy tier's h=0.001 row: ln(10) rounds to 2.30 but the row itself
      rounds to 2.31);
    - option distinctness: all four MCQ options (key + 3 distractors) survive
      `pick_distractors` at full count AND clear the `_MIN_OPTION_GAP` floor
      pairwise, checked with `check_pairwise_distinct`.
    """
    for _ in range(max_attempts):
        a = a_pool_fn(rng)
        a_float = float(a)
        ln_a = math.log(a_float)
        key_rounded = round(ln_a, OPTION_DP)
        if abs(key_rounded) < 0.05:
            continue
        rows_text, values = _rows_for(a, h_values)
        last_val = values[-1]
        if round(last_val, OPTION_DP) != key_rounded:
            continue

        key_str = _fmt_signed(key_rounded)
        h01_str = _fmt_signed(values[0])
        a_minus_1_str = _fmt_signed(a_float - 1.0)
        neg_key_str = _fmt_signed(-ln_a)

        # Misconception pool: the first (coarsest) row, a - 1, the sign slipped, log10 for ln, a doubled and a
        # halved estimate, the reciprocal. They fall on both sides of the key, so the key's rank among the four
        # options is drawn at random rather than fixed by the tier.
        pool = {
            h01_str: values[0],
            a_minus_1_str: a_float - 1.0,
            neg_key_str: -ln_a,
            _fmt_signed(math.log10(a_float)): math.log10(a_float),
            _fmt_signed(2 * ln_a): 2 * ln_a,
            _fmt_signed(ln_a / 2): ln_a / 2,
            _fmt_signed(1.0 / ln_a): 1.0 / ln_a,
        }
        pool = {k: v for k, v in pool.items() if k != key_str}
        forced = [h01_str] if require_h01 else []
        distractors = _pick_ranked(rng, key_str, key_rounded, pool, forced)
        if distractors is None:
            continue

        options = [key_str] + distractors
        min_gap = min(
            abs(Fraction(options[i]) - Fraction(options[j]))
            for i in range(len(options))
            for j in range(i + 1, len(options))
        )
        if min_gap < _MIN_OPTION_GAP:
            continue
        check_pairwise_distinct(*options, msg="two MCQ options collide after rounding")

        # Independent route 1: re-derive the correct option via sympy.log,
        # never re-using the math.log call above for the confirmation.
        sym_key = float(sympy_log(_to_rational(a)).evalf())
        check(
            round(sym_key, OPTION_DP) == key_rounded,
            f"sympy-derived ln(a) {sym_key} disagrees with math.log-derived key {key_rounded}",
        )
        # Independent route 2: the table's own last row (what a learner
        # actually re-checks) rounds to the offered correct option.
        check(
            round(last_val, OPTION_DP) == key_rounded,
            "the table's last row does not round to the offered correct option",
        )

        return dict(
            a=a,
            key_str=key_str,
            key_rounded=key_rounded,
            rows_text=rows_text,
            values=values,
            distractors=distractors,
            min_gap=min_gap,
        )
    raise RuntimeError("could not draw a valid estimate-limit question within max_attempts")


@generator(
    "p12c_estimate_limit_ah",
    level=16,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p12c_estimate_limit_ah(rng, tier="easy"):
    """Estimate lim(h->0) (a^h - 1)/h from a table of (a^h - 1)/h at
    decreasing positive h, for a stated a > 0 -- never the exact symbolic
    ln(a), always a rounded numeral offered as one of four MCQ options.

    easy: a a "nice" integer 2 <= a <= 10 (limit comfortably positive, trend
    unambiguous), 3-row table (h = 0.1, 0.01, 0.001).
    medium: a a simple decimal in (0, 1) -- the true limit is NEGATIVE, a
    genuinely different reading of the same 3-row table (a "pick the
    largest/first value" strategy that might accidentally work on easy now
    clearly fails).
    hard: a drawn from either pool, a FOURTH table row at h = 0.0001, and the
    displayed h=0.1 row value is always offered as a distractor (the "read
    the first, coarsest row, not the trend" misconception) -- a genuinely
    different reasoning decision (trend extrapolation vs single-row reading),
    not merely a bigger table.
    exam: ONE a drawn from the easy pool (a > 1) and ONE from the medium pool
    (a < 1) -- never two from the same pool -- two full tables, and a
    comparison of which estimated limit has the LARGER MAGNITUDE (further
    from zero); which side is "First" is independently randomised. A
    comparison across two estimates, a genuinely different decision from
    estimating one value. (Asking literally "the larger estimated limit" --
    the obvious wording -- is monotonic in a: since ln is increasing, the
    bigger a always wins outright and a learner never needs to read either
    table. Magnitude comparison requires synthesising a positive-trending
    table against a negative-trending one; forcing one side of each sign is
    what actually blocks the "just compare the two a values" shortcut --
    drawing both sides from the same pool leaves that shortcut open on
    same-sign pairs, since a larger/smaller a still predicts the larger/
    smaller magnitude within one sign.
    deviation, flagged.)
    """
    if tier == "exam":
        return _make_exam(rng)
    if tier not in _H_VALUES:
        raise ValueError(f"tier must be one of 'easy', 'medium', 'hard', 'exam'; got {tier!r}")

    a_pool_fn = {"easy": _draw_easy_a, "medium": _draw_medium_a, "hard": _draw_mixed_a}[tier]
    result = _draw_valid(rng, a_pool_fn, _H_VALUES[tier], require_h01=(tier == "hard"))

    a_str = _fmt_a(result["a"])
    template = rng.choice(_SINGLE_TEMPLATES)
    text = template.format(a=a_str, rows=result["rows_text"])

    working = [
        f"The table's values are converging to {result['key_str']} as h gets "
        f"smaller -- the smallest-h row already rounds to that value to "
        f"{OPTION_DP} d.p.",
    ]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=result["key_str"],
        answer_text=result["key_str"],
        answer_latex=latex_answer(result["key_str"], plain=True),
        working=working,
        distractors=result["distractors"],
    )


def _draw_side(rng, a_pool_fn, h_values, max_attempts: int = 500) -> dict:
    """Lighter than `_draw_valid`: one side of an exam-tier comparison --
    plausible and internally consistent (table's last row rounds to its own
    key), but no MCQ-option construction (the exam tier's options are
    comparison labels, not numerals)."""
    for _ in range(max_attempts):
        a = a_pool_fn(rng)
        a_float = float(a)
        ln_a = math.log(a_float)
        key_rounded = round(ln_a, OPTION_DP)
        if abs(key_rounded) < 0.05:
            continue
        rows_text, values = _rows_for(a, h_values)
        if round(values[-1], OPTION_DP) != key_rounded:
            continue
        return dict(a=a, key_rounded=key_rounded, rows_text=rows_text)
    raise RuntimeError("could not draw a valid exam-tier side within max_attempts")


def _make_exam(rng) -> dict:
    """One side ALWAYS from the easy (a > 1, positive limit) pool and one
    ALWAYS from the medium (a < 1, negative limit) pool -- never two sides
    from the same pool. Two sides on the same side of 1 let a learner shortcut
    the whole comparison by noting ln is increasing (bigger a always wins, or
    smaller a always wins) without reading either table; forcing one of each
    sign means the raw "bigger a" comparison no longer predicts the magnitude
    winner (a positive-trending table has to be weighed against a
    negative-trending one), so the table itself has to be read. Which side is
    "First" is independently randomised so the answer is not always the
    integer side.

    The natural tie this construction can hit is the RECIPROCAL pair
    a2 = 1/a1 (e.g. a1=4, a2=0.25: ln(4) = -ln(0.25), equal magnitudes) --
    caught generically by the `_EXAM_MIN_GAP` floor below like any other tie,
    never offered as the key.
    """
    h_values = _H_VALUES["easy"]
    for _ in range(200):
        pos_side = _draw_side(rng, _draw_easy_a, h_values)
        neg_side = _draw_side(rng, _draw_medium_a, h_values)
        m_pos, m_neg = abs(pos_side["key_rounded"]), abs(neg_side["key_rounded"])
        if abs(m_pos - m_neg) >= _EXAM_MIN_GAP:
            break
    else:
        raise RuntimeError("could not draw two distinguishable exam sides")

    if rng.random() < 0.5:
        side1, side2 = pos_side, neg_side
    else:
        side1, side2 = neg_side, pos_side
    m1, m2 = abs(side1["key_rounded"]), abs(side2["key_rounded"])

    a1_str, a2_str = _fmt_a(side1["a"]), _fmt_a(side2["a"])
    option_first = f"The first value (a = {a1_str}) gives the estimated limit further from zero"
    option_second = f"The second value (a = {a2_str}) gives the estimated limit further from zero"
    option_equal = "The two estimated limits have the same magnitude (to 2 d.p.)"
    # The classic slip: the numerator a^h - 1 tends to 0, so "the limit is 0" for both.
    option_zero = "Neither: both limits are 0, because a^h - 1 tends to 0 as h tends to 0"

    # Independent route: recompute both magnitudes via sympy, never re-using
    # the math.log calls `_draw_side` used, and confirm the winner again.
    sym_m1 = abs(float(sympy_log(_to_rational(side1["a"])).evalf()))
    sym_m2 = abs(float(sympy_log(_to_rational(side2["a"])).evalf()))
    check(
        round(sym_m1, OPTION_DP) == m1 and round(sym_m2, OPTION_DP) == m2,
        "sympy-derived magnitudes disagree with math.log-derived magnitudes",
    )
    check(m1 != m2, "exam-tier magnitudes must never tie -- excluded by construction, never offered as the key")

    answer = option_first if m1 > m2 else option_second
    other = option_second if answer == option_first else option_first
    distractors = [other, option_equal, option_zero]
    # Plain string distinctness, not `check_pairwise_distinct`: these are
    # prose comparison LABELS, not numeric/symbolic values, so
    # `symbolically_equal` (which sympifies each option) is the wrong tool --
    # it would try to parse "The first value (a = 5) gives..." as an
    # expression and raise. A label a learner reads is either the same
    # string or a different one; there is no other notion of "value" here.
    check(len({answer, *distractors}) == 4, "exam-tier MCQ options collide")

    template = rng.choice(_EXAM_TEMPLATES)
    text = template.format(a1=a1_str, rows1=side1["rows_text"], a2=a2_str, rows2=side2["rows_text"])

    working = [
        f"The first table's values converge to size {m1:.2f}; the second table's "
        f"values converge to size {m2:.2f}. The larger of the two sizes wins.",
    ]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=distractors,
    )
