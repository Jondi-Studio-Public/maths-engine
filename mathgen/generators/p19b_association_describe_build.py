"""Generates `p19b_association_describe` questions: state the direction, form and strength of the association
between two numerical variables.

Scope: given a stated scatterplot
description or a stated set of summary statistics for two numerical
variables, state the association's direction (positive or negative), form
(linear or non-linear) and strength (strong, moderate or weak), each chosen
from its own small fixed label set.

## Grader kind: `ordered_list`, not `pair`

An `answer` kind of `pair` with three named fields is not usable here, because
`mathgen/grading/typed.py::grade_pair` reads each component through `extract_values`
(numeric) -- 'values = extract_values(_strip_comma_groups(given))' -- so word
values cannot be graded there. `grade_ordered_list` compares string items
case-insensitively after stripping whitespace ('elif given_item.strip().lower()
!= str(expected_item).strip().lower()'). So `answer: ordered_list`,
`question.answer = ['positive', 'linear', 'strong']`, `answer_text`
`positive, linear, strong`. `answer` is a 3-item list; `answer_text` is the same three items,
comma-joined, exactly as every prompt tells the learner to type them.

## The non-linear-strength kernel (written before the generator)

r measures LINEAR strength only, so it is never used to set the strength of
a non-linear draw. A non-linear draw's strength comes from a STATED
closeness-to-curve description, using a fixed three-level convention stated
in every non-linear prompt: "the points lie almost exactly on the curve" ->
strong; "the points are scattered fairly close to the curve" -> moderate;
"the points are noticeably scattered around the curve" -> weak. This is an
assigned convention realised directly by construction (the strength label is
drawn FIRST; its description is the label's own stated realisation) --
exactly the same discipline the linear bands use for a computed r, just with
a stated qualitative description standing in for the computed magnitude.

## Construction (every tier)

Direction, form and strength are each drawn UNIFORMLY and independently
(construct-to-target) before any presentation text is built, so
no field is a shortcut from another. Direction realises as a stated
monotone trend (never a U-shape -- a non-monotone curve has no direction);
form realises as either a straight-line scatter/residual description or a
curved one; strength realises as a linear |r|-band (rational throughout: `r`
is drawn on a hundredths grid, so `r^2` lands on an exact 2-d.p. percentage
and `sqrt(r^2)` recovers `|r|` exactly with no irrational arithmetic at all)
or a stated closeness description, per the kernel above.

**Context is drawn from direction, never independently of it.** Each context in the bank carries its own honest
natural direction (>= 5 naturally positive, >= 5 naturally negative); the
generator draws direction first and then a context whose natural direction
matches it, never the reverse. The hard tier's least-squares line
(defect #2) is drawn to-target from that context's own plausible x/y ranges,
so a prediction inside the stated x-range is never implausible (a negative
exam score, negative sales, and so on): the slope's magnitude is drawn from
a context-scaled range, then the intercept is solved for so both endpoint
predictions land inside the context's plausible y-range (`_draw_line`), and
the line is always presented with both variables named in words and the
x-range stated (`_line_str`).

## Tiers (each changes the DECISION)

- **easy**: one scatterplot description in words giving all three clues
  (trend direction, straight-vs-curved, tightly-vs-loosely packed) without
  using the answer's own three words. DECISION: read each field straight off
  its own clause.
- **medium**: TWO branches, half the draws each. Linear branch: r is stated
  (with the linear band rule) plus a residual-plot description confirming
  "no pattern" (linear); direction and strength read from r's sign and
  |r|. Non-linear branch: NO r is given at all -- a residual-plot
  description shows a clear curved pattern, and a separate closeness
  description (with the non-linear band rule) gives strength; direction
  comes from a stated trend clause. DECISION: construct the triple from
  whichever kind of evidence the prompt actually supplies.
- **hard**: linear branch escalates -- r is NOT stated with its sign; instead
  r^2 (as a percentage) and a least-squares line `y = a + bx` are given, so
  direction comes from the line's slope sign and strength from
  `sqrt(r^2)` against the bands. Non-linear branch reuses medium's own
  closeness-description kernel under a different scenario wording (see the
  linear branch's r-from-r^2/slope derivation; the non-linear branch's
  decision is unchanged from medium, since no further
  non-linear escalation is specified).
- **exam**: r (signed, from a stated straight-line fit) AND a residual-plot
  description are BOTH given on every draw, AND both strength band rules -- the linear |r| bands and the
  non-linear curve-closeness rule -- are stated on every draw, so the
  learner must decide which rule applies rather than being told by which
  single rule happens to be printed. On half of all draws the residual
  description is genuinely random (real form is linear, r is valid); on the
  other half it shows a clear curved pattern despite a large |r| -- the TRAP:
  the learner must recognise that r's size does not
  establish linearity, and that the strength is read from the stated curve-fit
  description, not from r. DECISION: decide, from the residual description
  alone, whether r is even the right statistic to read strength from, before
  answering -- a judgement no lower tier requires (every lower tier's branch
  is unambiguous about which evidence applies).

## Grader probes ("Positive, Linear, Strong", "
positive,linear , strong ", "nonlinear", "non linear")
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, generator, latex_answer, phrase

TOPIC = "p19b_association_describe"
_TIERS = ("easy", "medium", "hard", "exam")

_DIRECTIONS = ("positive", "negative")
_FORMS = ("linear", "non-linear")
_STRENGTHS = ("strong", "moderate", "weak")

_INSTRUCTION = (
    "State your answer as three fields, in this order: direction (positive or negative), "
    "form (linear or non-linear), strength (strong, moderate or weak), separated by commas "
    "-- for example 'positive, linear, strong'. Use exactly these spellings, including the "
    "hyphen in non-linear."
)
_LINEAR_BAND_RULE = (
    "For a linear relationship, judge strength from the size of r: |r| >= 0.75 is strong, "
    "0.5 <= |r| < 0.75 is moderate, 0.25 <= |r| < 0.5 is weak."
)
_NONLINEAR_BAND_RULE = (
    "For a non-linear relationship, judge strength from how closely the points follow the "
    "curve: lying almost exactly on the curve is strong, scattered fairly close to the curve "
    "is moderate, and noticeably scattered around the curve is weak."
)
# Exam tier states BOTH strength rules on every draw: the residual-plot evidence is what decides which rule applies,
# so removing one rule would remove the decision the tier exists to test.
_EXAM_BOTH_BAND_RULES = f"{_LINEAR_BAND_RULE} {_NONLINEAR_BAND_RULE}"


# ============================================================================
# Context bank: direction is a property of
# the CONTEXT, never drawn independently of it. Each context carries its own
# natural direction plus the (x_range, y_range) it is honestly true within,
# used by the hard tier's least-squares line so predictions never leave a
# plausible range. At least 5 contexts are naturally negative.
# ============================================================================

_CONTEXTS = (
    # -- naturally positive --
    dict(
        direction="positive",
        x="hours studied per week",
        x_unit="hours",
        x_range=(2, 20),
        y="exam score",
        y_unit="marks out of 100",
        y_range=(20, 95),
        b_tenths=(10, 30),
    ),
    dict(
        direction="positive",
        x="daily maximum temperature",
        x_unit="degrees Celsius",
        x_range=(15, 35),
        y="ice-cream sales",
        y_unit="units per day",
        y_range=(20, 380),
        b_tenths=(50, 150),
    ),
    dict(
        direction="positive",
        x="weekly training hours",
        x_unit="hours",
        x_range=(2, 15),
        y="fitness test score",
        y_unit="marks out of 100",
        y_range=(35, 95),
        b_tenths=(10, 30),
    ),
    dict(
        direction="positive",
        x="monthly advertising spend",
        x_unit="thousands of dollars",
        x_range=(1, 20),
        y="monthly sales",
        y_unit="thousands of dollars",
        y_range=(0, 260),
        b_tenths=(30, 100),
    ),
    dict(
        direction="positive",
        x="a plant's daily sunlight hours",
        x_unit="hours",
        x_range=(2, 12),
        y="its height after a month",
        y_unit="centimetres",
        y_range=(2, 40),
        b_tenths=(10, 30),
    ),
    # -- naturally negative --
    dict(
        direction="negative",
        x="a car's age",
        x_unit="years",
        x_range=(1, 15),
        y="its resale value",
        y_unit="thousands of dollars",
        y_range=(0, 45),
        b_tenths=(10, 25),
    ),
    dict(
        direction="negative",
        x="altitude",
        x_unit="hundreds of metres",
        x_range=(0, 30),
        y="air temperature",
        y_unit="degrees Celsius",
        y_range=(-10, 32),
        b_tenths=(3, 10),
    ),
    dict(
        direction="negative",
        x="the number of days absent per term",
        x_unit="days",
        x_range=(0, 20),
        y="exam mark",
        y_unit="marks out of 100",
        y_range=(30, 95),
        b_tenths=(10, 25),
    ),
    dict(
        direction="negative",
        x="the price",
        x_unit="dollars",
        x_range=(5, 50),
        y="units sold per week",
        y_unit="units",
        y_range=(20, 400),
        b_tenths=(20, 60),
    ),
    dict(
        direction="negative",
        x="hours of screen time per day",
        x_unit="hours",
        x_range=(1, 9),
        y="hours of sleep per night",
        y_unit="hours",
        y_range=(3, 10),
        b_tenths=(2, 6),
    ),
)

_CONTEXTS_BY_DIRECTION = {
    "positive": tuple(c for c in _CONTEXTS if c["direction"] == "positive"),
    "negative": tuple(c for c in _CONTEXTS if c["direction"] == "negative"),
}


def _pick_context(rng, direction: str) -> dict:
    return rng.choice(_CONTEXTS_BY_DIRECTION[direction])


# Strength band ranges on a hundredths grid, each comfortably (>= 0.02) clear
# of every boundary and of the weak floor, and each an EXACT rational value
# throughout: r^2 = (hundredths/100)^2 lands on an exact 2-d.p. percentage,
# and sqrt(r^2) recovers |r| exactly by construction -- no irrational
# arithmetic anywhere in this generator.
_STRENGTH_RANGE = {"strong": (77, 98), "moderate": (52, 73), "weak": (27, 48)}

_CLOSENESS_TEXT = {
    "strong": "the points lie almost exactly on the curve",
    "moderate": "the points are scattered fairly close to the curve",
    "weak": "the points are noticeably scattered around the curve",
}


# ============================================================================
# r arithmetic (exact, hundredths grid throughout)
# ============================================================================


def _draw_r(rng, strength: str, direction: str) -> Fraction:
    lo, hi = _STRENGTH_RANGE[strength]
    hundredths = rng.randint(lo, hi)
    sign = 1 if direction == "positive" else -1
    return Fraction(sign * hundredths, 100)


def _classify_r_strength(abs_r: Fraction) -> str | None:
    if abs_r >= Fraction(75, 100):
        return "strong"
    if abs_r >= Fraction(50, 100):
        return "moderate"
    if abs_r >= Fraction(25, 100):
        return "weak"
    return None


def _r_str(r: Fraction) -> str:
    return f"{float(r):.2f}"


def _pct_str(r: Fraction) -> str:
    pct = r * r * 100
    return f"{float(pct):.2f}"


# ============================================================================
# Hard tier's least-squares line: construct-to-target. The slope's magnitude is drawn from the context's own plausible
# range, then the intercept is solved for so BOTH endpoint predictions over
# the context's stated x-range land inside its plausible y-range -- never a
# fixed a/b range applied to every context regardless of scale.
# ============================================================================


def _draw_line(rng, ctx: dict) -> tuple[int, Fraction]:
    x_lo, x_hi = ctx["x_range"]
    y_lo, y_hi = ctx["y_range"]
    b_lo, b_hi = ctx["b_tenths"]
    b_mag = Fraction(rng.randint(b_lo, b_hi), 10)
    b = b_mag if ctx["direction"] == "positive" else -b_mag
    check(
        (b > 0) == (ctx["direction"] == "positive"),
        "p19b_association_describe/hard: slope sign disagrees with the context's direction",
    )
    if b >= 0:
        a_lo, a_hi = y_lo - b * x_lo, y_hi - b * x_hi
    else:
        a_lo, a_hi = y_lo - b * x_hi, y_hi - b * x_lo
    a_lo_int, a_hi_int = math.ceil(a_lo), math.floor(a_hi)
    check(
        a_lo_int <= a_hi_int, f"p19b_association_describe/hard: no plausible intercept for {ctx['x']!r} with slope {b}"
    )
    a = rng.randint(a_lo_int, a_hi_int)
    y_at_lo, y_at_hi = a + b * x_lo, a + b * x_hi
    check(
        y_lo <= y_at_lo <= y_hi and y_lo <= y_at_hi <= y_hi,
        f"p19b_association_describe/hard: line y={a}+{b}x leaves the plausible range at "
        f"x={x_lo} ({y_at_lo}) or x={x_hi} ({y_at_hi})",
    )
    return a, b


def _line_str(ctx: dict, a: int, b: Fraction) -> str:
    x_lo, x_hi = ctx["x_range"]
    b_str = f"{float(abs(b)):.1f}"
    sign = "+" if b > 0 else "-"
    return (
        f"y = {a} {sign} {b_str}x, where x is {ctx['x']}, in {ctx['x_unit']} "
        f"(typically {x_lo} to {x_hi}), and y is {ctx['y']}, in {ctx['y_unit']}"
    )


# ============================================================================
# Distractors: exactly one field flipped from the key
# ============================================================================


def _other_direction(direction: str) -> str:
    return "negative" if direction == "positive" else "positive"


def _other_form(form: str) -> str:
    return "linear" if form == "non-linear" else "non-linear"


def _adjacent_strength(rng, strength: str) -> str:
    if strength == "strong":
        return "moderate"
    if strength == "weak":
        return "moderate"
    return rng.choice(["strong", "weak"])


def _build_candidates(rng, direction: str, form: str, strength: str):
    key = (direction, form, strength)
    candidates = [
        (_other_direction(direction), form, strength),
        (direction, _other_form(form), strength),
        (direction, form, _adjacent_strength(rng, strength)),
    ]
    triples = [key, *candidates]
    check(
        len(set(triples)) == len(triples),
        f"p19b_association_describe: duplicate triple among key/distractors: {triples}",
    )
    return candidates


def _finalise(text, direction, form, strength, candidates, working):
    answer = [direction, form, strength]
    answer_text = f"{direction}, {form}, {strength}"
    cand_texts = [f"{d}, {f}, {s}" for d, f, s in candidates]
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=cand_texts,
        working=working,
    )


# ============================================================================
# Easy: one verbal scatterplot description
# ============================================================================

_DIRECTION_PHRASE = {
    "positive": "As {x} increases, {y} tends to increase as well",
    "negative": "As {x} increases, {y} tends to decrease",
}
_FORM_PHRASE = {
    "linear": "The points follow a roughly straight-line trend",
    "non-linear": "The points follow a clear curve, not a straight line",
}
_STRENGTH_PHRASE_LINEAR = {
    "strong": "The points are packed very tightly around the trend, with little scatter.",
    "moderate": "The points sit fairly close to the trend, with a noticeable but not extreme amount of scatter.",
    "weak": "The points are loosely scattered around the trend.",
}
_STRENGTH_PHRASE_NONLINEAR = {
    "strong": "The points lie almost exactly along the curve.",
    "moderate": "The points lie fairly close to the curve, with some scatter.",
    "weak": "The points are loosely scattered around the curve.",
}

_EASY_TEMPLATES = (
    "A scatterplot of {y} against {x} is described as follows. {direction_phrase}. "
    "{form_phrase}. {strength_phrase} {instruction}",
    "For a sample of {n} cases, a scatterplot of {y} against {x} shows this pattern. "
    "{direction_phrase}. {form_phrase}. {strength_phrase} {instruction}",
    "A researcher describes a scatterplot of {y} against {x}. {direction_phrase}. "
    "{form_phrase}. {strength_phrase} {instruction}",
)


def _build_easy(rng):
    direction = rng.choice(_DIRECTIONS)
    form = rng.choice(_FORMS)
    strength = rng.choice(_STRENGTHS)
    ctx = _pick_context(rng, direction)
    n = rng.randint(20, 80)

    strength_phrase = (_STRENGTH_PHRASE_LINEAR if form == "linear" else _STRENGTH_PHRASE_NONLINEAR)[strength]
    text = phrase(
        rng,
        _EASY_TEMPLATES,
        x=ctx["x"],
        y=ctx["y"],
        n=n,
        direction_phrase=_DIRECTION_PHRASE[direction].format(x=ctx["x"], y=ctx["y"]),
        form_phrase=_FORM_PHRASE[form],
        strength_phrase=strength_phrase,
        instruction=_INSTRUCTION,
    )
    candidates = _build_candidates(rng, direction, form, strength)
    working = [
        f"Direction: {direction} (from the trend clause). Form: {form} (straight vs curved "
        f"clause). Strength: {strength} (scatter clause)."
    ]
    return _finalise(text, direction, form, strength, candidates, working)


# ============================================================================
# Medium: two branches (linear via r, non-linear via curve + closeness)
# ============================================================================

_MEDIUM_LINEAR_TEMPLATES = (
    "For {n} recorded pairs of {x} and {y}, the correlation coefficient is r = {r}. "
    "{band_rule} A residual plot for this data shows no pattern -- the residuals are "
    "randomly scattered about zero, confirming a straight-line model is appropriate. "
    "{instruction}",
    "A sample of {n} cases recorded {x} and {y}, giving r = {r}. {band_rule} The residual "
    "plot shows random scatter about zero, with no pattern. {instruction}",
)
_MEDIUM_NONLINEAR_TEMPLATES = (
    "For {n} recorded pairs of {x} and {y}, a residual plot shows a clear curved pattern "
    "(not random scatter), so a straight-line model is not appropriate. As {x} increases, "
    "{y} tends to {trend}. Relative to the fitted curve, {closeness}. {band_rule} "
    "{instruction}",
    "A sample of {n} cases recorded {x} and {y}. As {x} increases, {y} tends to {trend}. A "
    "residual plot shows a clear curved pattern rather than random scatter. Relative to the "
    "fitted curve, {closeness}. {band_rule} {instruction}",
)


def _build_medium(rng):
    direction = rng.choice(_DIRECTIONS)
    strength = rng.choice(_STRENGTHS)
    form = rng.choice(_FORMS)
    ctx = _pick_context(rng, direction)
    n = rng.randint(20, 80)
    trend = "increase" if direction == "positive" else "decrease"

    if form == "linear":
        r = _draw_r(rng, strength, direction)
        check(
            _classify_r_strength(abs(r)) == strength,
            "p19b_association_describe/medium: r does not land in the drawn strength band",
        )
        text = phrase(
            rng,
            _MEDIUM_LINEAR_TEMPLATES,
            n=n,
            x=ctx["x"],
            y=ctx["y"],
            r=_r_str(r),
            band_rule=_LINEAR_BAND_RULE,
            instruction=_INSTRUCTION,
        )
        working = [
            f"r = {_r_str(r)}: sign gives direction ({direction}); |r| = {abs(float(r)):.2f} "
            f"gives strength ({strength}) by the stated bands; the residual plot confirms "
            f"linear form."
        ]
    else:
        text = phrase(
            rng,
            _MEDIUM_NONLINEAR_TEMPLATES,
            n=n,
            x=ctx["x"],
            y=ctx["y"],
            trend=trend,
            closeness=_CLOSENESS_TEXT[strength],
            band_rule=_NONLINEAR_BAND_RULE,
            instruction=_INSTRUCTION,
        )
        working = [
            f"The trend clause gives direction ({direction}); the curved residual plot gives "
            f"non-linear form; the closeness description gives strength ({strength}) by the "
            f"stated bands -- never from r, which is not given here."
        ]

    candidates = _build_candidates(rng, direction, form, strength)
    return _finalise(text, direction, form, strength, candidates, working)


# ============================================================================
# Hard: linear branch escalates to r^2 + slope; non-linear branch unchanged
# ============================================================================

_HARD_LINEAR_TEMPLATES = (
    "For {n} recorded pairs of {x} and {y}, the least-squares line is {line}, and the "
    "coefficient of determination is r^2 = {pct}%. {band_rule} A residual plot for this data "
    "shows no pattern -- random scatter about zero, confirming a straight-line model. "
    "{instruction}",
    "A sample of {n} cases recorded {x} and {y}. The fitted least-squares line is "
    "{line}, with r^2 = {pct}%. {band_rule} The residual plot shows random scatter "
    "about zero, with no pattern. {instruction}",
)
_HARD_NONLINEAR_TEMPLATES = (
    "For {n} recorded pairs of {x} and {y}, a residual plot shows a clear curved pattern, so "
    "a straight-line model is not appropriate. As {x} increases, {y} tends to {trend}. "
    "Relative to the fitted curve, {closeness}. {band_rule} {instruction}",
    "A sample of {n} cases recorded {x} and {y}, and a residual plot shows a clear curved "
    "pattern rather than random scatter. As {x} increases, {y} tends to {trend}. Relative to "
    "the fitted curve, {closeness}. {band_rule} {instruction}",
)


def _build_hard(rng):
    direction = rng.choice(_DIRECTIONS)
    strength = rng.choice(_STRENGTHS)
    form = rng.choice(_FORMS)
    ctx = _pick_context(rng, direction)
    n = rng.randint(20, 80)
    trend = "increase" if direction == "positive" else "decrease"

    if form == "linear":
        r = _draw_r(rng, strength, direction)
        check(
            _classify_r_strength(abs(r)) == strength,
            "p19b_association_describe/hard: r does not land in the drawn strength band",
        )
        a, b = _draw_line(rng, ctx)
        line_str = _line_str(ctx, a, b)
        text = phrase(
            rng,
            _HARD_LINEAR_TEMPLATES,
            n=n,
            x=ctx["x"],
            y=ctx["y"],
            line=line_str,
            pct=_pct_str(r),
            band_rule=_LINEAR_BAND_RULE,
            instruction=_INSTRUCTION,
        )
        working = [
            f"Line: {line_str}: the slope's sign gives direction ({direction}); |r| = "
            f"sqrt(r^2) = sqrt({_pct_str(r)}%) = {abs(float(r)):.2f} gives strength "
            f"({strength}) by the stated bands; the residual plot confirms linear form."
        ]
    else:
        text = phrase(
            rng,
            _HARD_NONLINEAR_TEMPLATES,
            n=n,
            x=ctx["x"],
            y=ctx["y"],
            trend=trend,
            closeness=_CLOSENESS_TEXT[strength],
            band_rule=_NONLINEAR_BAND_RULE,
            instruction=_INSTRUCTION,
        )
        working = [
            f"The trend clause gives direction ({direction}); the curved residual plot gives "
            f"non-linear form; the closeness description gives strength ({strength})."
        ]

    candidates = _build_candidates(rng, direction, form, strength)
    return _finalise(text, direction, form, strength, candidates, working)


# ============================================================================
# Exam: r and a residual description both given -- the trap is deciding
# whether r's size actually establishes linearity
# ============================================================================

_EXAM_TEMPLATES = (
    "For {n} recorded pairs of {x} and {y}, a straight-line fit gives r = {r}. {band_rule} A "
    "residual plot for this fit shows {residual_desc}. {trend_clause}{closeness_clause}"
    "{instruction}",
    "A sample of {n} cases recorded {x} and {y}. Fitting a straight line gives r = {r}. "
    "{band_rule} The residual plot shows {residual_desc}. {trend_clause}{closeness_clause}"
    "{instruction}",
)

_RESIDUAL_RANDOM = "no pattern -- the residuals are randomly scattered about zero"
_RESIDUAL_CURVED = "a clear curved pattern, not random scatter"


def _build_exam(rng):
    direction = rng.choice(_DIRECTIONS)
    trap = rng.choice([True, False])
    ctx = _pick_context(rng, direction)
    n = rng.randint(30, 100)

    if trap:
        # Real form is non-linear despite a large |r| from a straight-line
        # fit; true strength comes from the stated closeness description,
        # which is deliberately drawn from a DIFFERENT band than |r| would
        # naively suggest, so "read strength off r" is a genuinely wrong,
        # exercisable answer.
        form = "non-linear"
        strength = rng.choice(["moderate", "weak"])
        r_hundredths = rng.randint(80, 97)
        sign = 1 if direction == "positive" else -1
        r = Fraction(sign * r_hundredths, 100)
        residual_desc = _RESIDUAL_CURVED
        trend = "increase" if direction == "positive" else "decrease"
        trend_clause = f"As {ctx['x']} increases, {ctx['y']} tends to {trend}. "
        closeness_clause = f"Relative to the fitted curve, {_CLOSENESS_TEXT[strength]}. "
    else:
        form = "linear"
        strength = rng.choice(_STRENGTHS)
        r = _draw_r(rng, strength, direction)
        check(
            _classify_r_strength(abs(r)) == strength,
            "p19b_association_describe/exam: r does not land in the drawn strength band",
        )
        residual_desc = _RESIDUAL_RANDOM
        trend_clause = ""
        closeness_clause = ""

    # Both strength rules are always stated:
    # the tier's own decision is picking WHICH rule applies from the
    # residual-plot evidence, so both must be on the page every time.
    text = phrase(
        rng,
        _EXAM_TEMPLATES,
        n=n,
        x=ctx["x"],
        y=ctx["y"],
        r=_r_str(r),
        closeness_clause=closeness_clause,
        band_rule=_EXAM_BOTH_BAND_RULES,
        residual_desc=residual_desc,
        trend_clause=trend_clause,
        instruction=_INSTRUCTION,
    )

    if trap:
        working = [
            f"The residual plot shows a curved pattern, so the model is non-linear even "
            f"though r = {_r_str(r)} looks large -- r only measures LINEAR strength. "
            f"Strength ({strength}) comes from the stated closeness description instead."
        ]
    else:
        working = [
            f"The residual plot confirms linear form; r = {_r_str(r)} gives direction "
            f"({direction}) from its sign and strength ({strength}) from |r| against the "
            f"stated bands."
        ]

    candidates = _build_candidates(rng, direction, form, strength)
    return _finalise(text, direction, form, strength, candidates, working)


_TIER_BUILDERS = {
    "easy": _build_easy,
    "medium": _build_medium,
    "hard": _build_hard,
    "exam": _build_exam,
}


@generator(
    TOPIC,
    level=19,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19b_association_describe(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIER_BUILDERS[tier](rng)
