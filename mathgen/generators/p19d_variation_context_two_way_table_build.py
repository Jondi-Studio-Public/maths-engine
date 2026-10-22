"""Normal-variation context classification and two-way table reading.

- ``p19d_variation_context``
  (``senior-identify-contexts-such-as-naturally-occurring-variation-that``,
  ACMMM168, domain statistics-probability, answer: mcq)
  -- given several short described measured-quantity contexts, choose which
  ONE is best modelled by a normal random variable (a naturally-occurring,
  continuous physical measurement subject to many small independent
  influences, symmetric about its mean). Every other offered context is
  genuinely NOT normal, drawn from at least two of three distinct
  wrong-context classes every draw: discrete-count/Poisson-like, strictly
  non-negative/skewed, or bounded/clustered-at-a-boundary. Boundary: this is
  NOT ``senior-examine-the-approximate-normality-of-the-distribution-of``
  (which diagnoses the normality of an ACTUAL sampling distribution of
  p-hat given a stated n and x -- a computed check on numbers already
  given, never a real-world-context judgement), and NOT
  ``senior-understand-the-concept-of-a-random-sample`` (which judges
  selection-method validity, never a quantity's distributional shape).

- ``p19d_two_way_table``
  (``senior-interpret-information-presented-in-two-way-tables``, ACMEM038,
  domain senior-mathematics, answer: number)
  -- given a stated two-way frequency table, rendered as plain text,
  compute a specified cell value, row/column total, or proportion (a named
  cell or margin divided by a named total). A proportion is always graded
  to a stated 1-decimal-place percentage; a cell/total is graded as an
  exact integer. Boundary: this is NOT
  ``senior-complete-a-five-number-summary-for-different-datasets`` or
  ``standard-deviation-from-data``, both of which operate on a single
  numerical dataset (a list of values), never a two-way categorical
  cross-tabulation.

Both rows ship an independent oracle in
``tests/test_p19d_variation_context_two_way_table_oracle.py``: skill 1's
re-derives the normal/not-normal classification from generic textual cues
over each rendered option, never importing this module's own class tags;
skill 2's re-parses the rendered table text and recomputes the requested
cell/total/proportion independently, never importing this module's own
2D-array construction.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    generator,
    latex_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

# ============================================================================
# Skill 1: p19d_variation_context
# ============================================================================

FAMILIAR_NORMAL = [
    "the heights, in cm, of adult male students at a large high school",
    "the masses, in grams, of apples picked from a single orchard",
    "the body temperatures, in degrees Celsius, of healthy adults",
    "the heights, in cm, of fully grown eucalyptus saplings in a nursery",
    "the diameters, in mm, of ball bearings produced by a single machine",
]

LESS_FAMILIAR_NORMAL = [
    "the systolic blood pressure readings, in mmHg, of healthy adults having a routine health check",
    "the grip strengths, in kg, of adult female patients measured by a physiotherapist",
    "the wingspans, in mm, of a particular species of moth",
    "the resting heart rates, in beats per minute, of healthy adults",
    "the birth weights, in grams, of full-term babies at a large hospital",
]

# Pinned so a future edit cannot silently reintroduce any of the three.
REMOVED_NON_NORMAL_PHRASES = [
    "the reaction times, in milliseconds, of drivers to a sudden braking light",
    "the grip strengths, in kg, of adult patients measured by a physiotherapist",
    "the systolic blood pressure readings, in mmHg, of adult patients at a large clinic",
]

ALL_NORMAL = FAMILIAR_NORMAL + LESS_FAMILIAR_NORMAL

COUNT_WRONG = [
    "the number of emails a small office receives in a given hour",
    "the number of customers who enter a corner store in a 10-minute period",
    "the number of typing errors on a randomly chosen page of a long manuscript",
    "the number of calls received by a small helpdesk in a given minute",
]

SKEWED_WRONG = [
    "the time, in hours, until a particular type of lightbulb fails",
    "the annual incomes, in dollars, of households in a large city",
    "the time, in minutes, a customer spends on hold before a call is answered",
    "the time, in days, until a smartphone battery needs its first replacement",
    "the masses, in kg, of fish caught in a river in a day",
    "the distances, in km, customers travel to reach a regional shopping centre",
]

BOUNDED_WRONG = [
    "the number of heads obtained when a fair coin is tossed 3 times",
    "the proportion of students in a class of 10 who pass a particular test",
    "the number of correct answers on a 5-question multiple-choice quiz answered by random guessing",
    "the number of defective items found when a batch of 4 items is inspected",
]

# This pool is entirely physical-unit-bearing wrong contexts (kg/km/mm/cm), so "has a physical unit"
# alone never identifies the key.
#
# A careful reader can defend it as "still just heights, roughly normal like the textbook exemplar"
# (heights are THE normal exemplar, and whether a 5-12 age span makes the mix meaningfully
# non-normal is an arguable, not unambiguous, point). Genuinely, unambiguously bimodal needs two
# EXPLICITLY DISJOINT, separately-named sub-populations measured together, so different in
# scale/kind that "roughly normal like one population's spread" is not a defensible reading -- two
# visibly separated clusters, not a spread within one group. Combining kindergarten children
# (typically ~100-120 cm) and their adult parents (typically ~155-190 cm) at the same event produces
# two clearly separated height clusters with a gap between them, not a single wide-but-continuous
# spread.
BIMODAL_WRONG = [
    "the heights, in cm, of a combined group of kindergarten children and their adult parents at a school open day",
]

NEAR_MISS_WRONG = [
    "the time, in minutes, a commuter waits for a bus that arrives exactly every 10 minutes, "
    "when the commuter arrives at a random time not knowing the schedule",
    "the position, in cm, at which a spinner's arrow comes to rest around a circular dial marked from 0 to 100",
]

WRONG_POOLS = {
    "count": COUNT_WRONG,
    "skewed": SKEWED_WRONG,
    "bounded": BOUNDED_WRONG,
    "bimodal": BIMODAL_WRONG,
    "near_miss": NEAR_MISS_WRONG,
}

# A second, orthogonal clause appended to every option at the exam tier only
# -- an extra data-collection detail that never changes the underlying
# quantity's distributional shape, forcing the learner to first read past it
# to see what is actually being measured.
EXTRA_CLAUSES = [
    ", but only values recorded on weekdays are kept",
    ", excluding any values recorded during a public holiday",
    ", but only values from individuals who consented to be measured are used",
    ", recorded only during a single data-collection session",
]

# Every clause is deliberately worded to avoid EXTRA_CLAUSES' own ", but"/", excluding"/", recorded
# only" markers, so a consumer that strips those markers off an option's text (the build test's own
# wrong-class check, and a rendering that wants to show the "base" quantity) never accidentally
# strips this clause back down to a phrase that reads as plain-normal. Each base phrase is one
# already in ALL_NORMAL (a genuine truncation of a normal-base phrase), and each pair's base must
# differ from whichever phrase the exam draw picked as the correct answer (checked in `_vc_exam`).
TRUNCATED_NORMAL = [
    (
        "the heights, in cm, of adult male students at a large high school",
        "the heights, in cm, of adult male students at a large high school, where only "
        "students taller than 190 cm are included",
    ),
    (
        "the masses, in grams, of apples picked from a single orchard",
        "the masses, in grams, of apples picked from a single orchard, where only apples "
        "heavier than 250 g are included",
    ),
    (
        "the diameters, in mm, of ball bearings produced by a single machine",
        "the diameters, in mm, of ball bearings produced by a single machine, where only "
        "bearings wider than 12 mm are included",
    ),
]

STEM = "Which of the following is best modelled by a normal random variable?"
STEM_EXAM = (
    "Each context below includes an extra detail about how the data were collected. First "
    "identify what quantity is actually being measured in each case, then decide which context "
    "is best modelled by a normal random variable."
)


def _cap_first(s: str) -> str:
    """Capitalise only the first character; ``str.capitalize()`` would
    lowercase mid-string capitals (there are none today, but this is the
    same discipline p19d_random_sample_target_population's `_cap_first`
    uses)."""
    return s[:1].upper() + s[1:]


def _option(phrase: str) -> str:
    return f"{_cap_first(phrase)}."


def _classify_local(phrase: str) -> str:
    """Internal (in-generator) consistency check: an independently-coded keyword classifier over a single context
    phrase, mirroring ``check_equivalent``'s role for a computational skill -- a second route that must agree
    with which pool construction drew from."""
    p = phrase.lower()
    # Checked BEFORE the `phrase in ALL_NORMAL` membership test below: a
    # truncated-normal distractor's base phrase IS a literal ALL_NORMAL
    # member, but the full phrase (base + this clause) is no longer normal
    # -- a truncated tail, not a symmetric unbounded measurement.
    if ", where only" in p and " than " in p and "included" in p:
        return "truncated"
    if any(k in p for k in ("tossed", "multiple-choice quiz", "batch of", "proportion of students")):
        return "bounded"
    if any(k in p for k in ("arrives exactly every", "circular dial")):
        return "near_miss"
    if any(
        k in p for k in ("emails a small office", "corner store", "typing errors", "calls received by a small helpdesk")
    ):
        return "count"
    if any(
        k in p
        for k in (
            "lightbulb fails",
            "annual incomes",
            "on hold before a call",
            "first replacement",
            "fish caught",
            "travel to reach",
        )
    ):
        return "skewed"
    if any(k in p for k in ("combined group of",)):
        return "bimodal"
    if phrase in ALL_NORMAL:
        return "normal"
    return "unknown"


def _draw_option_set(rng, correct_pool, wrong_classes, *, allow_near_miss, extra_clause):
    """Draw one correct normal-context phrase plus one wrong phrase per
    named class in ``wrong_classes`` (each class used at most once, so the same wrong class never repeats), optionally
    adding the near-miss class and/or the exam-tier extra clause. Returns
    (correct_option_text, [distractor_option_texts])."""
    for _attempt in range(30):
        correct_phrase = rng.choice(correct_pool)
        wrong_entries = [(cls, rng.choice(WRONG_POOLS[cls])) for cls in wrong_classes]
        if allow_near_miss and rng.random() < 0.5:
            wrong_entries.append(("near_miss", rng.choice(WRONG_POOLS["near_miss"])))
        raw_phrases = [correct_phrase] + [p for _, p in wrong_entries]
        if len(set(raw_phrases)) != len(raw_phrases):
            continue  # a duplicate phrase was drawn twice -- resample

        classes = [_classify_local(p) for p in raw_phrases]
        check(classes[0] == "normal", f"p19d_variation_context: correct phrase misclassified: {correct_phrase!r}")
        check(
            all(c != "normal" for c in classes[1:]),
            f"p19d_variation_context: a wrong phrase classified normal: {raw_phrases!r}",
        )
        check(len(set(classes[1:])) == len(classes[1:]), f"p19d_variation_context: repeated wrong class: {classes!r}")

        display_phrases = raw_phrases
        if extra_clause:
            display_phrases = [p + rng.choice(EXTRA_CLAUSES) for p in raw_phrases]
        options = [_option(p) for p in display_phrases]
        if len(set(options)) != len(options):
            continue  # the extra clause happened to collide two options -- resample

        correct_option = options[0]
        rng.shuffle(options)
        distractors = [o for o in options if o != correct_option]
        check(len(distractors) == len(options) - 1, "p19d_variation_context: distractor bookkeeping mismatch")
        return correct_option, distractors, options
    raise RuntimeError("p19d_variation_context: could not draw a distinct, consistent option set in 30 attempts")


_OPTION_TAGS = ["1", "2", "3", "4", "5"]


def _variation_question(stem: str, correct_option: str, distractors: list[str], display_order: list[str]) -> dict:
    """`display_order` is the (already-shuffled) full option list from construction -- unrelated to
    `grading.mcq_options`'s own independent grade-time shuffle, so listing it in the PROMPT is purely
    presentational and never leaks the answer's identity via position. Enumerating the contexts in `text`."""
    opts_text = " ".join(f"{_OPTION_TAGS[i]}) {opt}" for i, opt in enumerate(display_order))
    text = f"{stem} {opts_text}"
    return dict(
        text=text,
        latex=text,
        answer=correct_option,
        answer_text=correct_option,
        answer_latex=latex_answer(correct_option, plain=True),
        distractors=distractors,
        working=[f"Best modelled by a normal random variable: {correct_option}"],
    )


def _vc_easy(rng):
    correct, distractors, order = _draw_option_set(
        rng, FAMILIAR_NORMAL, ["count", "skewed", "bounded"], allow_near_miss=False, extra_clause=False
    )
    return _variation_question(STEM, correct, distractors, order)


def _vc_medium(rng):
    # "bounded" swapped for "bimodal": every medium draw now carries one physical-unit-bearing wrong
    # option (the combined-kindergarten-children-and-parents heights entry), so "has a physical
    # unit" alone can no longer shortcut to the key.
    correct, distractors, order = _draw_option_set(
        rng,
        LESS_FAMILIAR_NORMAL,
        ["count", "skewed", "bimodal"],
        allow_near_miss=False,
        extra_clause=False,
    )
    return _variation_question(STEM, correct, distractors, order)


def _vc_hard(rng):
    # "bimodal" added as a fourth mandatory class (guaranteeing a
    # physical-unit wrong option every draw) alongside the existing three;
    # `allow_near_miss` dropped to keep the option count fixed at 5 (4
    # wrong + the key), matching `_OPTION_TAGS`. `near_miss` still occurs at
    # the exam tier below.
    correct, distractors, order = _draw_option_set(
        rng,
        ALL_NORMAL,
        ["count", "skewed", "bounded", "bimodal"],
        allow_near_miss=False,
        extra_clause=False,
    )
    return _variation_question(STEM, correct, distractors, order)


def _vc_exam(rng):
    """Exam tier: distinct from `_draw_option_set`'s generic path because it must add one MANDATORY distractor
    built by truncating a normal-base phrase -- that distractor genuinely changes the base phrase's
    distributional shape, unlike the uniformly-applied, shape-neutral `EXTRA_CLAUSES` every other option still
    gets. It also carries its own physical unit (cm/g/mm), so it satisfies the physical-unit-distractor
    requirement on its own, every draw."""
    for _attempt in range(30):
        correct_phrase = rng.choice(ALL_NORMAL)
        wrong_entries = [(cls, rng.choice(WRONG_POOLS[cls])) for cls in ("count", "skewed")]
        # Always a fifth option: a near miss half the time, otherwise a bounded count, so the tier shows 5 options.
        extra_class = "near_miss" if rng.random() < 0.5 else "bounded"
        wrong_entries.append((extra_class, rng.choice(WRONG_POOLS[extra_class])))
        trunc_base, trunc_full = rng.choice(TRUNCATED_NORMAL)

        raw_phrases = [correct_phrase] + [p for _, p in wrong_entries]
        if trunc_base in raw_phrases:
            continue  # the truncated distractor's own base collides with an already-chosen phrase
        if len(set(raw_phrases)) != len(raw_phrases):
            continue  # a duplicate phrase was drawn twice -- resample

        classes = [_classify_local(p) for p in raw_phrases]
        check(classes[0] == "normal", f"p19d_variation_context/exam: correct phrase misclassified: {correct_phrase!r}")
        check(
            all(c != "normal" for c in classes[1:]),
            f"p19d_variation_context/exam: a wrong phrase classified normal: {raw_phrases!r}",
        )
        check(
            len(set(classes[1:])) == len(classes[1:]), f"p19d_variation_context/exam: repeated wrong class: {classes!r}"
        )
        trunc_class = _classify_local(trunc_full)
        check(
            trunc_class == "truncated",
            f"p19d_variation_context/exam: truncated distractor misclassified {trunc_class!r}: {trunc_full!r}",
        )

        all_phrases = raw_phrases + [trunc_full]
        if len(set(all_phrases)) != len(all_phrases):
            continue

        # Every option except the truncated distractor gets the generic,
        # shape-neutral exam-tier clause (including the key -- its own
        # clause must stay genuinely neutral); the truncated distractor
        # already carries its own shape-changing clause and gets no
        # further suffix.
        display_phrases = [p + rng.choice(EXTRA_CLAUSES) for p in raw_phrases] + [trunc_full]
        options = [_option(p) for p in display_phrases]
        if len(set(options)) != len(options):
            continue  # a clause happened to collide two options -- resample

        correct_option = options[0]
        rng.shuffle(options)
        distractors = [o for o in options if o != correct_option]
        check(len(distractors) == len(options) - 1, "p19d_variation_context/exam: distractor bookkeeping mismatch")
        return _variation_question(STEM_EXAM, correct_option, distractors, options)
    raise RuntimeError("p19d_variation_context/exam: could not draw a distinct, consistent option set in 30 attempts")


@generator(
    "p19d_variation_context",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19d_variation_context(rng, tier="easy"):
    """Given several short described measured-quantity contexts, choose
    which one is best modelled by a normal random variable (ACMMM168). NOT
    the same operation as
    senior-examine-the-approximate-normality-of-the-distribution-of (which
    diagnoses the normality of an ACTUAL sampling distribution of p-hat
    given stated n/x, a computed check -- never a real-world-context
    judgement), and NOT the same operation as
    senior-understand-the-concept-of-a-random-sample (which judges
    selection-method validity, never a quantity's distributional shape)."""
    if tier == "easy":
        return _vc_easy(rng)
    if tier == "medium":
        return _vc_medium(rng)
    if tier == "hard":
        return _vc_hard(rng)
    if tier == "exam":
        return _vc_exam(rng)
    raise ValueError(f"p19d_variation_context: unknown tier {tier!r}")


# ============================================================================
# Skill 2: p19d_two_way_table
# ============================================================================

CONTEXTS = [
    dict(
        subject="students",
        row_cats2=["Year 11", "Year 12"],
        row_cats3=["Year 10", "Year 11", "Year 12"],
        col_cats2=["Mathematics", "English"],
        col_cats3=["Mathematics", "English", "Science"],
    ),
    dict(
        subject="gym members",
        row_cats2=["Male", "Female"],
        row_cats3=["Under 30", "30 to 50", "Over 50"],
        col_cats2=["Morning sessions", "Evening sessions"],
        col_cats3=["Morning sessions", "Evening sessions", "Weekend sessions"],
    ),
    dict(
        subject="voters",
        row_cats2=["Urban", "Rural"],
        row_cats3=["Urban", "Suburban", "Rural"],
        col_cats2=["Support", "Oppose"],
        col_cats3=["Support", "Oppose", "Undecided"],
    ),
    dict(
        subject="customers",
        row_cats2=["New customers", "Returning customers"],
        row_cats3=["New customers", "Returning customers", "Loyalty members"],
        col_cats2=["Online order", "In-store order"],
        col_cats3=["Online order", "In-store order", "Phone order"],
    ),
    dict(
        subject="patients",
        row_cats2=["Under 40", "40 or older"],
        row_cats3=["Under 40", "40 to 65", "Over 65"],
        col_cats2=["Medication A", "Medication B"],
        col_cats3=["Medication A", "Medication B", "No medication"],
    ),
]


def _render_table(rows: list[str], cols: list[str], cell_of: dict) -> str:
    """Render a fully-given two-way table as plain text: a Rows: line, a
    Columns: line, then every cell as "{row}/{col}: {value}", comma-joined
    -- the sentence-enumeration convention, one valid
    plain-text table format."""
    parts = [f"{r}/{c}: {cell_of[(r, c)]}" for r in rows for c in cols]
    return f"Rows: {', '.join(rows)}. Columns: {', '.join(cols)}. {', '.join(parts)}."


def _render_table_with_total(rows, cols, cell_of, total_row, total_value, hide) -> str:
    """Like `_render_table`, but states one row's total directly (as its
    own sentence) and omits the `hide` cell entirely -- the medium/exam-tier
    convention where that omitted cell must be derived by subtraction from
    the stated total and the row's other given cells."""
    parts = [f"{r}/{c}: {cell_of[(r, c)]}" for r in rows for c in cols if (r, c) != hide]
    return f"Rows: {', '.join(rows)}. Columns: {', '.join(cols)}. {total_row} total: {total_value}. {', '.join(parts)}."


def _fmt_pct(frac: Fraction) -> str:
    """A proportion, displayed as a percentage to exactly 1 decimal place --
    the stated precision the whole skill's proportion questions use, so
    `grade_number`'s stated-precision tolerance (0.05 percentage points)
    applies uniformly."""
    rounded = round_half_up(frac * 100, 1)
    return f"{stated_precision_string(rounded, 1)}%"


def _pct_value(pct_text: str) -> float:
    return float(Fraction(pct_text.rstrip("%")))


def _tw_easy(rng):
    ctx = rng.choice(CONTEXTS)
    rows, cols = ctx["row_cats2"], ctx["col_cats2"]
    for _attempt in range(30):
        values = {(r, c): rng.randint(5, 45) for r in rows for c in cols}
        row_totals = {r: sum(values[(r, c)] for c in cols) for r in rows}
        col_totals = {c: sum(values[(r, c)] for r in rows) for c in cols}
        grand = sum(values.values())
        # Independent route: the grand total via row totals must match the
        # grand total via column totals -- two different summation orders.
        check(sum(row_totals.values()) == grand, "p19d_two_way_table/easy: grand total via rows mismatch")
        check(sum(col_totals.values()) == grand, "p19d_two_way_table/easy: grand total via columns mismatch")

        ask_kind = rng.choice(["cell", "row_total", "col_total"])
        if ask_kind == "cell":
            r, c = rng.choice(rows), rng.choice(cols)
            target = values[(r, c)]
            question = f"How many are both {r} and {c}?"
        elif ask_kind == "row_total":
            r = rng.choice(rows)
            target = row_totals[r]
            question = f"How many are {r} in total?"
        else:
            c = rng.choice(cols)
            target = col_totals[c]
            question = f"How many are {c} in total?"

        candidates = (
            [str(v) for v in values.values()]
            + [str(v) for v in row_totals.values()]
            + [str(v) for v in col_totals.values()]
            + [str(grand)]
        )
        distractors = pick_distractors(str(target), candidates, n=3)
        if len(distractors) >= 2:
            break
    else:
        raise RuntimeError("p19d_two_way_table/easy: no draw produced >=2 distractors in 30 attempts")

    table_text = _render_table(rows, cols, values)
    text = f"A survey of {grand} {ctx['subject']} recorded two categorical variables. {table_text} {question}"
    answer_text = str(target)
    return dict(
        text=text,
        latex=text,
        answer=int(target),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=distractors,
        working=[f"Read directly from the table: {answer_text}."],
    )


def _tw_medium(rng):
    ctx = rng.choice(CONTEXTS)
    rows, cols = ctx["row_cats2"], ctx["col_cats3"]
    for _attempt in range(30):
        values = {(rows[1], c): rng.randint(5, 40) for c in cols}
        v1, v2 = rng.randint(5, 30), rng.randint(5, 30)
        missing = rng.randint(3, 25)
        total_r0 = v1 + v2 + missing
        values[(rows[0], cols[0])] = v1
        values[(rows[0], cols[1])] = v2
        # (rows[0], cols[2]) intentionally never in `values` -- it is derived.

        row2_total = sum(values[(rows[1], c)] for c in cols)
        check(total_r0 - v1 - v2 == missing, "p19d_two_way_table/medium: row-total subtraction mismatch")

        candidates = [
            str(total_r0),
            str(v1),
            str(v2),
            str(v1 + v2),
            str(values[(rows[1], cols[2])]),
            str(row2_total),
        ]
        distractors = pick_distractors(str(missing), candidates, n=3)
        if len(distractors) >= 2 and missing > 0:
            break
    else:
        raise RuntimeError("p19d_two_way_table/medium: no draw produced >=2 distractors in 30 attempts")

    table_text = _render_table_with_total(rows, cols, values, rows[0], total_r0, hide=(rows[0], cols[2]))
    text = (
        f"A survey of {ctx['subject']} recorded two categorical variables. {table_text} "
        f"How many {rows[0]} are {cols[2]}?"
    )
    answer_text = str(missing)
    return dict(
        text=text,
        latex=text,
        answer=int(missing),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=distractors,
        working=[
            f"{rows[0]} total = {total_r0}. {cols[0]}: {v1}, {cols[1]}: {v2}.",
            f"{cols[2]} = {total_r0} - {v1} - {v2} = {missing}.",
        ],
    )


DENOM_KINDS = ["row", "col", "grand"]


def _proportion_question_text(denom_kind: str, r: str, c: str) -> str:
    if denom_kind == "row":
        return (
            f"What proportion of {r} respondents also fall into the {c} category? Give your "
            f"answer as a percentage to 1 decimal place."
        )
    if denom_kind == "col":
        return (
            f"Of those in the {c} category, what proportion are {r}? Give your answer as a "
            f"percentage to 1 decimal place."
        )
    return (
        f"What proportion of all respondents are both {r} and {c}? Give your answer as a percentage to 1 decimal place."
    )


def _tw_hard(rng):
    ctx = rng.choice(CONTEXTS)
    rows = ctx["row_cats3"] if rng.random() < 0.5 else ctx["row_cats2"]
    cols = ctx["col_cats2"]
    for _attempt in range(40):
        values = {(r, c): rng.randint(8, 50) for r in rows for c in cols}
        row_totals = {r: sum(values[(r, c)] for c in cols) for r in rows}
        col_totals = {c: sum(values[(r, c)] for r in rows) for c in cols}
        grand = sum(values.values())
        check(sum(row_totals.values()) == grand, "p19d_two_way_table/hard: grand total via rows mismatch")
        check(sum(col_totals.values()) == grand, "p19d_two_way_table/hard: grand total via columns mismatch")

        r, c = rng.choice(rows), rng.choice(cols)
        cell = values[(r, c)]
        denom_kind = rng.choice(DENOM_KINDS)
        denom_values = {"row": row_totals[r], "col": col_totals[c], "grand": grand}
        denom = denom_values[denom_kind]
        correct_frac = Fraction(cell, denom)
        # A key of exactly one-half makes the "the rest" complement distractor (`Fraction(denom -
        # cell, denom)`, below) numerically equal to the key itself -- either a silent duplicate the
        # dedup swallows, or a degenerate "obviously it's half" question either way. Reject and
        # resample.
        if correct_frac == Fraction(1, 2):
            continue
        other_fracs = {k: Fraction(cell, v) for k, v in denom_values.items() if k != denom_kind}
        # Every wrong-denominator distractor must genuinely differ from the
        # right answer -- checked directly every seed, not merely relying on
        # `pick_distractors`' own dedup.
        if any(f == correct_frac for f in other_fracs.values()):
            continue

        answer_pct = _fmt_pct(correct_frac)
        candidates = [_fmt_pct(f) for f in other_fracs.values()]
        candidates.append(_fmt_pct(Fraction(denom - cell, denom)))  # "the rest" mistake
        distractors = pick_distractors(answer_pct, candidates, tolerance=stated_precision_tolerance(1))
        if len(distractors) >= 2:
            break
    else:
        raise RuntimeError("p19d_two_way_table/hard: no draw produced a usable proportion question in 40 attempts")

    table_text = _render_table(rows, cols, values)
    question = _proportion_question_text(denom_kind, r, c)
    text = f"A survey of {ctx['subject']} recorded two categorical variables. {table_text} {question}"
    return dict(
        text=text,
        latex=text,
        answer=_pct_value(answer_pct),
        answer_text=answer_pct,
        answer_latex=latex_answer(answer_pct, plain=True),
        distractors=distractors,
        working=[f"{cell} / {denom} = {answer_pct}."],
    )


def _tw_exam(rng):
    ctx = rng.choice(CONTEXTS)
    rows, cols = ctx["row_cats3"], ctx["col_cats3"]
    for _attempt in range(40):
        values = {(r, c): rng.randint(5, 40) for r in rows[1:] for c in cols}
        v1, v2 = rng.randint(5, 30), rng.randint(5, 30)
        missing = rng.randint(3, 25)
        total_r0 = v1 + v2 + missing
        values[(rows[0], cols[0])] = v1
        values[(rows[0], cols[1])] = v2
        # (rows[0], cols[2]) intentionally never in `values` -- derived below.

        row_totals = {rows[0]: total_r0}
        for r in rows[1:]:
            row_totals[r] = sum(values[(r, c)] for c in cols)
        grand = sum(row_totals.values())

        # Independent route: the grand total via row totals must match the
        # grand total via column totals, where each column total is built
        # from the derived cell (`missing`) for cols[2] and from `v1`/`v2`
        # for cols[0]/cols[1] -- a genuinely different summation order from
        # the row-total route above.
        col_totals = {
            cols[0]: v1 + sum(values[(r, cols[0])] for r in rows[1:]),
            cols[1]: v2 + sum(values[(r, cols[1])] for r in rows[1:]),
            cols[2]: missing + sum(values[(r, cols[2])] for r in rows[1:]),
        }
        check(sum(col_totals.values()) == grand, "p19d_two_way_table/exam: grand total via rows/columns mismatch")

        correct_frac = Fraction(missing, total_r0)
        # Same exact-half guard as `_tw_hard`: a key of exactly one-half makes the "forgot the
        # derived cell" / complement-style distractors degenerate.
        if correct_frac == Fraction(1, 2):
            continue
        wrong_denom = {"grand": Fraction(missing, grand)}
        if any(f == correct_frac for f in wrong_denom.values()):
            continue

        answer_pct = _fmt_pct(correct_frac)
        cand_fracs = list(wrong_denom.values())
        cand_fracs.append(Fraction(v1 + v2, total_r0))  # forgot the derived cell
        cand_fracs.append(Fraction(missing, v1 + v2))  # wrong denominator entirely
        # A proportion of a group cannot exceed 100%: such an option is dead.
        candidates = [_fmt_pct(f) for f in cand_fracs if 0 <= f <= 1]
        distractors = pick_distractors(answer_pct, candidates, tolerance=stated_precision_tolerance(1))
        if len(distractors) >= 2 and missing > 0:
            break
    else:
        raise RuntimeError("p19d_two_way_table/exam: no draw produced a usable question in 40 attempts")

    cell_of = dict(values)
    table_text = _render_table_with_total(rows, cols, cell_of, rows[0], total_r0, hide=(rows[0], cols[2]))
    text = (
        f"A survey of {ctx['subject']} recorded two categorical variables. {table_text} "
        f"First find how many {rows[0]} are {cols[2]}, then find what proportion of {rows[0]} "
        f"respondents also fall into the {cols[2]} category. Give your answer as a percentage "
        f"to 1 decimal place."
    )
    return dict(
        text=text,
        latex=text,
        answer=_pct_value(answer_pct),
        answer_text=answer_pct,
        answer_latex=latex_answer(answer_pct, plain=True),
        distractors=distractors,
        working=[
            f"{cols[2]} = {total_r0} - {v1} - {v2} = {missing}.",
            f"{missing} / {total_r0} = {answer_pct}.",
        ],
    )


@generator(
    "p19d_two_way_table",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19d_two_way_table(rng, tier="easy"):
    """Given a stated two-way frequency table (rendered as plain text),
    compute a specified cell value, row/column total, or proportion
    (ACMEM038). NOT the same operation as
    senior-complete-a-five-number-summary-for-different-datasets or
    standard-deviation-from-data, both of which operate on a single
    numerical dataset, never a two-way categorical cross-tabulation."""
    if tier == "easy":
        return _tw_easy(rng)
    if tier == "medium":
        return _tw_medium(rng)
    if tier == "hard":
        return _tw_hard(rng)
    if tier == "exam":
        return _tw_exam(rng)
    raise ValueError(f"p19d_two_way_table: unknown tier {tier!r}")
