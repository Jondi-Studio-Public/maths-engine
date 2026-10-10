"""Generates `p16a_conditional_independence` and `p16a_estimated_distribution` questions: conditional probability
from a two-way table and probabilities estimated from data.

Senior statistics computations, part 2.

Two generators, both senior ``acara-senior-v8-4``, domain
``statistics-probability``:

- ``p16a_conditional_independence``
  (``senior-use-relative-frequencies-obtained-from-data-as-point-acmmm060``,
  ACMMM060) -- from a two-way frequency table, estimate P(A|B) and compare
  it with P(A) to indicate possible independence. ``response_family:
  scalar_expression``, ``answer: fraction``. The graded value is
  ``P(A|B) - P(A)`` (exact, from integer counts): the value 0 IS the
  "independent" verdict, so the single scalar carries both the conditional
  point estimate and the independence indication the descriptor asks for,
  never a separate binary classification. Genuinely different
  from the shipping ``conditional-probability-counts`` (which grades the
  bare P(A|B) handed four raw counts directly): this skill's table must be
  read (or partially reconstructed) before any probability is computed, and
  the graded value is always the COMPARISON, never P(A|B) alone.

- ``p16a_estimated_distribution``
  (``senior-use-relative-frequencies-obtained-from-data-to-obtain``,
  ACMMM137) -- from a frequency table of a discrete random variable,
  produce the ESTIMATED PROBABILITY DISTRIBUTION as an ordered list of
  P(X = x), one per value, in the order the prompt states.
  ``response_family: ordered_structures``, ``answer: ordered_list``. Every
  item is an exact ``Fraction`` (grade_ordered_list compares Fraction items
  exactly, never with the pair/list voice tolerance).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, pick_distractors, rational_text

# --------------------------------------------------------------------------
# p16a_conditional_independence (ACMMM060)
# --------------------------------------------------------------------------

_AB_CONTEXTS = [
    ("owns a pet", "lives in a house (not an apartment)"),
    ("plays a musical instrument", "is enrolled in the school orchestra program"),
    ("travels to school by bus", "lives more than 3 km from school"),
    ("plays a winter sport", "is a member of the PE elective"),
]


def _verify_diff(a: int, b: int, c: int, d: int, diff: Fraction) -> None:
    """Independent route: recompute P(A|B) and P(A) straight from the raw
    counts by a fresh expression (not construction's own intermediate
    variables), then check the difference. row_a = a+b (count of A), N =
    a+b+c+d."""
    n_total = a + b + c + d
    check(n_total > 0, "table has no observations")
    p_a_given_b = Fraction(a, a + c)
    p_a = Fraction(a + b, n_total)
    check(p_a_given_b - p_a == diff, "recomputation from raw counts disagrees with construction")


def _draw_2x2(rng, *, force_independent_rate=0.15, n_total=None):
    """Draw a 2x2 table {a, b, c, d} (a = A&B, b = A&notB, c = notA&B,
    d = notA&notB). With probability ``force_independent_rate`` the table is
    constructed to be EXACTLY independent (P(A|B) == P(A)), via row/column
    margins that are quarters of N -- this is construct-to-target, not left to chance: independence is a measure-zero event for
    random integer counts, so a curriculum item testing "recognise
    independence" must sometimes hand it a genuinely independent table.
    ``n_total``, when given, fixes the table's total instead of drawing one
    at random -- the percentages variant passes 100 so every percentage it
    displays is an exact integer, never a decimal rounded from a total the
    exact fraction key could not otherwise be recovered from."""
    if rng.random() < force_independent_rate:
        total = n_total if n_total is not None else rng.choice([48, 96])
        p_a = rng.choice([Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)])
        p_b = rng.choice([Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)])
        row_a = int(total * p_a)
        col_b = int(total * p_b)
        a = int(total * p_a * p_b)
        b = row_a - a
        c = col_b - a
        d = total - row_a - col_b + a
        return a, b, c, d
    for _ in range(50):
        total = n_total if n_total is not None else rng.randint(60, 140)
        row_a = rng.randint(int(0.25 * total), int(0.75 * total))
        col_b = rng.randint(int(0.25 * total), int(0.75 * total))
        lo = max(0, row_a + col_b - total)
        hi = min(row_a, col_b)
        if lo >= hi:
            continue
        a = rng.randint(lo, hi)
        indep_a = row_a * col_b  # numerator of a if exactly independent, over total
        if a * total == indep_a:
            continue  # accidentally independent -- redraw so "not independent" is genuine
        b, c, d = row_a - a, col_b - a, total - row_a - col_b + a
        if min(a, b, c, d) <= 0:
            continue
        return a, b, c, d
    raise ValueError("could not draw a non-degenerate, non-independent 2x2 table")


@generator(
    "p16a_conditional_independence",
    level=12,
    presets={
        "easy": dict(variant="direct_counts"),
        "medium": dict(variant="totals_given"),
        "hard": dict(variant="percentages"),
        "exam": dict(variant="three_way"),
    },
)
def p16a_conditional_independence(rng, variant="direct_counts"):
    """From a two-way frequency table, estimate P(A|B) as a point estimate
    of the conditional probability, then compare it with the estimate of
    P(A). The graded answer is P(A|B) - P(A): a value of 0 indicates the
    data are consistent with A and B being independent; any other value
    indicates dependence. Never grades the bare P(A|B) alone."""
    event_a, event_b = rng.choice(_AB_CONTEXTS)

    if variant == "direct_counts":
        a, b, c, d = _draw_2x2(rng, force_independent_rate=0.15)
        n_total = a + b + c + d
        diff = Fraction(a, a + c) - Fraction(a + b, n_total)
        _verify_diff(a, b, c, d, diff)
        question_text = (
            f"Of {n_total} students surveyed, where A is 'the student {event_a}' and B is "
            f"'the student {event_b}': {a} satisfy both A and B; {b} satisfy A but not B; "
            f"{c} satisfy B but not A; and {d} satisfy neither A nor B. Estimate P(A given "
            f"B) minus the estimate of P(A), as a fraction in lowest terms. A value of 0 "
            f"means the data indicate A and B are independent."
        )
        working = [
            f"P(A|B) = {a}/({a}+{c}) = {rational_text(Fraction(a, a + c))}.",
            f"P(A) = ({a}+{b})/{n_total} = {rational_text(Fraction(a + b, n_total))}.",
        ]

    elif variant == "totals_given":
        a, b, c, d = _draw_2x2(rng, force_independent_rate=0.15)
        n_total = a + b + c + d
        row_a_total = a + b
        col_b_total = a + c
        diff = Fraction(a, a + c) - Fraction(a + b, n_total)
        _verify_diff(a, b, c, d, diff)
        # Only totals + one interior cell are given; b, c, d must be derived.
        question_text = (
            f"Of {n_total} students surveyed, where A is 'the student {event_a}' and B is "
            f"'the student {event_b}': {row_a_total} satisfy A and {col_b_total} satisfy "
            f"B. Of the students who satisfy A, {a} also satisfy B. Estimate P(A given B) "
            f"minus the estimate of P(A), as a fraction in lowest terms. A value of 0 "
            f"means the data indicate A and B are independent."
        )
        working = [
            f"c (A and B) = {a}, so students who satisfy B but not A = {col_b_total} - {a} = {c}.",
            f"P(A|B) = {a}/{col_b_total} = {rational_text(Fraction(a, col_b_total))}.",
            f"P(A) = {row_a_total}/{n_total} = {rational_text(Fraction(row_a_total, n_total))}.",
        ]

    elif variant == "percentages":
        # n_total is fixed at 100 (independence is never force-constructed
        # here) so every percentage below is the exact count with no
        # rounding: the exact fraction key must stay recoverable straight
        # from the prompt, never by rounding a displayed decimal back to
        # counts.
        a, b, c, d = _draw_2x2(rng, force_independent_rate=0, n_total=100)
        n_total = a + b + c + d
        check(n_total == 100, "percentages variant must keep the fixed total of 100")
        diff = Fraction(a, a + c) - Fraction(a + b, n_total)
        _verify_diff(a, b, c, d, diff)
        pct = lambda x: str(x)
        question_text = (
            f"A survey of {n_total} students found, where A is 'the student {event_a}' "
            f"and B is 'the student {event_b}': {pct(a)}% satisfy both A and B; {pct(b)}% "
            f"satisfy A but not B; {pct(c)}% satisfy B but not A; and {pct(d)}% satisfy "
            f"neither. Estimate P(A given B) minus the estimate of P(A), as a fraction in "
            f"lowest terms. A value of 0 means the data indicate A and B are independent."
        )
        working = [
            f"Counts: A&B = {a}, A&notB = {b}, notA&B = {c}, notA&notB = {d} (out of {n_total}).",
            f"P(A|B) = {a}/({a}+{c}) = {rational_text(Fraction(a, a + c))}.",
            f"P(A) = ({a}+{b})/{n_total} = {rational_text(Fraction(a + b, n_total))}.",
        ]

    elif variant == "three_way":
        # B has three categories (B1, B2, B3); A is binary. Asks about the
        # ONE stated category, forcing the learner to select the correct
        # column among several -- a step no smaller tier requires. B's
        # categories are the travel MODE, so A must come from an event
        # unrelated to travel mode: crossing "travels to school by bus"
        # (one of _AB_CONTEXTS' own events) against "travels by
        # walking/cycling/public transport" is contradictory (bus is not
        # even one of the three listed modes), so this variant redraws its
        # own event_a from the other three contexts only.
        b_labels = ["walking", "cycling", "public transport"]
        event_a_local = rng.choice([e for e, _ in _AB_CONTEXTS if "travel" not in e])
        counts_a = [rng.randint(10, 40) for _ in range(3)]  # A&Bi for i=1,2,3
        counts_not_a = [rng.randint(10, 40) for _ in range(3)]  # notA&Bi
        col_totals = [counts_a[i] + counts_not_a[i] for i in range(3)]
        n_total = sum(col_totals)
        row_a_total = sum(counts_a)
        target = rng.randint(0, 2)
        a = counts_a[target]
        col_b_total = col_totals[target]
        diff = Fraction(a, col_b_total) - Fraction(row_a_total, n_total)
        check(
            Fraction(a, col_b_total) - Fraction(row_a_total, n_total) == diff,
            "sanity: diff must equal its own construction",
        )
        # Independent re-derivation of the SAME quantity, from the raw
        # per-column arrays, never from the intermediate `a`/`col_b_total`
        # variables above.
        p_a_given_b = Fraction(counts_a[target], counts_a[target] + counts_not_a[target])
        p_a = Fraction(sum(counts_a), sum(counts_a) + sum(counts_not_a))
        check(p_a_given_b - p_a == diff, "per-column recomputation disagrees with construction")
        rows = "; ".join(
            f"of the students who travel by {b_labels[i]}, {counts_a[i]} satisfy A and {counts_not_a[i]} do not"
            for i in range(3)
        )
        rows_cap = rows[0].upper() + rows[1:]  # this joined clause opens its own sentence
        question_text = (
            f"Students travel to school by walking, cycling or public transport, where A "
            f"is 'the student {event_a_local}'. {rows_cap}. Let B be 'the student travels by "
            f"{b_labels[target]}'. Estimate P(A given B) minus the estimate of P(A) (over "
            f"all students, however they travel), as a fraction in lowest terms. A value "
            f"of 0 means the data indicate A and B are independent."
        )
        working = [
            f"P(A|B) uses only the '{b_labels[target]}' column: {a}/{col_b_total} = "
            f"{rational_text(Fraction(a, col_b_total))}.",
            f"P(A) uses every student: {row_a_total}/{n_total} = {rational_text(Fraction(row_a_total, n_total))}.",
        ]
    else:
        raise ValueError(f"unknown variant {variant!r}")

    answer_text = rational_text(diff)

    # Misconceptions, each a genuinely different VALUE (checked below):
    reversed_conditioning = -diff  # P(A) - P(A|B) instead of P(A|B) - P(A)
    candidates = [rational_text(reversed_conditioning)]
    if variant == "direct_counts":
        wrong_denominator = Fraction(a, n_total) - Fraction(a + b, n_total)  # divides by N, not a+c
        bare_p_a_given_b = Fraction(a, a + c)  # forgets the comparison entirely
        raw_count_diff = a - (a + b)  # subtracts raw counts instead of matching probabilities
        candidates += [
            str(raw_count_diff),
            rational_text(wrong_denominator),
            rational_text(bare_p_a_given_b),
        ]
    elif variant == "totals_given":
        bare_p_a_given_b = Fraction(a, col_b_total)  # forgets the comparison with P(A)
        wrong_row_used = Fraction(a, row_a_total) - Fraction(
            row_a_total, n_total
        )  # divides by row total instead of col total
        raw_count_diff = a - row_a_total  # subtracts raw counts instead of matching probabilities
        candidates += [
            str(raw_count_diff),
            rational_text(bare_p_a_given_b),
            rational_text(wrong_row_used),
        ]
    elif variant == "percentages":
        bare_p_a_given_b = Fraction(a, a + c)
        wrong_denominator = Fraction(a, n_total) - Fraction(a + b, n_total)
        raw_count_diff = a - (a + b)  # subtracts raw counts instead of matching probabilities
        candidates += [
            str(raw_count_diff),
            rational_text(bare_p_a_given_b),
            rational_text(wrong_denominator),
        ]
    else:  # three_way
        wrong_column = rng.choice([i for i in range(3) if i != target])
        wrong_p_a_given_b = Fraction(counts_a[wrong_column], col_totals[wrong_column])
        wrong_diff = wrong_p_a_given_b - p_a
        bare_p_a_given_b = p_a_given_b
        raw_count_diff = a - row_a_total  # subtracts raw counts instead of matching probabilities
        candidates += [
            str(raw_count_diff),
            rational_text(wrong_diff),
            rational_text(bare_p_a_given_b),
        ]

    distractors = pick_distractors(answer_text, candidates, n=3)
    return dict(
        text=question_text,
        latex=question_text,
        answer={"p": diff.numerator, "q": diff.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


# --------------------------------------------------------------------------
# p16a_estimated_distribution (ACMMM137)
# --------------------------------------------------------------------------

_RV_CONTEXTS = [
    ("the number of customer complaints received per day", "day", "days"),
    ("the number of defective items in a sample batch", "batch", "batches"),
    ("the number of goals scored per match by a team", "match", "matches"),
    ("the number of calls a helpdesk receives per hour", "hour", "hours"),
]


def _verify_distribution(freqs: list[int], n_total: int, probs: list[Fraction]) -> None:
    """Independent route: recompute each probability from the RATIO of
    total-minus-others (n_total - sum of every OTHER frequency), rather than
    construction's own direct freq/n_total division."""
    check(sum(freqs) == n_total, "frequencies do not sum to the stated total")
    for i, f in enumerate(freqs):
        others = sum(freqs) - f
        recomputed = Fraction(n_total - others, n_total)
        check(recomputed == probs[i], f"complement-based recomputation disagrees at index {i}")
    check(sum(probs) == 1, "estimated probabilities must sum to 1")


@generator(
    "p16a_estimated_distribution",
    level=12,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="direct_wide"),
        "hard": dict(variant="missing_frequency"),
        "exam": dict(variant="two_missing_ratio"),
    },
)
def p16a_estimated_distribution(rng, variant="direct"):
    """From a frequency table of observed values of a discrete random
    variable X, produce the estimated probability distribution as an
    ORDERED list of P(X = x), one per value, listed in the order the
    question states. Never grades a single probability alone -- the answer
    is always the full list."""
    context, _unit, plural = rng.choice(_RV_CONTEXTS)

    if variant in ("direct", "direct_wide"):
        k = 3 if variant == "direct" else 4
        start = rng.randint(0, 1)
        values = list(range(start, start + k))
        freqs = [rng.randint(3, 20) for _ in values]
        n_total = sum(freqs)
        probs = [Fraction(f, n_total) for f in freqs]
        _verify_distribution(freqs, n_total, probs)
        table = ", ".join(f"x={v}: {f} {plural}" for v, f in zip(values, freqs))
        values_list = ", ".join(str(v) for v in values)
        question_text = (
            f"Over {n_total} {plural}, {context} (X) was recorded: {table}. Estimate the "
            f"probability distribution of X: list P(X = x) for each value of x, IN "
            f"INCREASING ORDER OF x (x = {values_list}), each as a "
            f"fraction in lowest terms."
        )
        working = [f"P(X = {v}) = {f}/{n_total} = {rational_text(p)}." for v, f, p in zip(values, freqs, probs)]

    elif variant == "missing_frequency":
        k = 4
        start = rng.randint(0, 1)
        values = list(range(start, start + k))
        given_freqs = [rng.randint(3, 20) for _ in range(k - 1)]
        n_total = sum(given_freqs) + rng.randint(3, 20)
        missing = n_total - sum(given_freqs)
        check(missing > 0, "the missing frequency must be positive")
        missing_index = rng.randint(0, k - 1)
        freqs = given_freqs[:missing_index] + [missing] + given_freqs[missing_index:]
        probs = [Fraction(f, n_total) for f in freqs]
        _verify_distribution(freqs, n_total, probs)
        shown = ", ".join(
            f"x={v}: {f if i != missing_index else '?'} {plural}" for i, (v, f) in enumerate(zip(values, freqs))
        )
        values_list = ", ".join(str(v) for v in values)
        question_text = (
            f"Over {n_total} {plural} in total, {context} (X) was recorded: {shown}. The "
            f"'?' entry is not shown but the total is {n_total}. Estimate the probability "
            f"distribution of X: list P(X = x) for each value of x, IN INCREASING ORDER OF "
            f"x (x = {values_list}), each as a "
            f"fraction in lowest terms."
        )
        working = [
            f"missing frequency = {n_total} - {sum(given_freqs)} = {missing}.",
        ] + [f"P(X = {v}) = {f}/{n_total} = {rational_text(p)}." for v, f, p in zip(values, freqs, probs)]

    elif variant == "two_missing_ratio":
        # TWO of the four frequencies are missing (not one, as in
        # missing_frequency), tied together by a stated ratio rather than
        # each being independently recoverable from the total alone -- a
        # small system of two equations (the ratio, then the total), a
        # genuine step up from missing_frequency's single subtraction.
        # (Replaces an earlier "same table, shuffled display order" design,
        # which needed no arithmetic at all and was easier than hard.)
        k = 4
        start = rng.randint(0, 1)
        values = list(range(start, start + k))
        missing_indices = sorted(rng.sample(range(k), 2))
        i_small, i_large = missing_indices[1], missing_indices[0]  # freqs[i_large] = ratio * freqs[i_small]
        ratio = rng.choice([2, 3])
        f_small = rng.randint(3, 15)
        f_large = ratio * f_small
        known_indices = [i for i in range(k) if i not in missing_indices]
        known_freqs = {i: rng.randint(3, 20) for i in known_indices}
        freqs = [0] * k
        for i, f in known_freqs.items():
            freqs[i] = f
        freqs[i_large] = f_large
        freqs[i_small] = f_small
        n_total = sum(freqs)
        probs = [Fraction(f, n_total) for f in freqs]
        _verify_distribution(freqs, n_total, probs)
        shown = ", ".join(
            f"x={v}: {freqs[i] if i not in missing_indices else '?'} {plural}" for i, v in enumerate(values)
        )
        values_list = ", ".join(str(v) for v in values)
        question_text = (
            f"Over {n_total} {plural} in total, {context} (X) was recorded: {shown}. The "
            f"two '?' entries are not shown, but the frequency for x={values[i_large]} is "
            f"{ratio} times the frequency for x={values[i_small]}, and the total is "
            f"{n_total}. Estimate the probability distribution of X: list P(X = x) for "
            f"each value of x, IN INCREASING ORDER OF x (x = {values_list}), each as a "
            f"fraction in lowest terms."
        )
        known_sum = sum(known_freqs.values())
        working = [
            f"Let a = the frequency at x={values[i_small]}; the frequency at x={values[i_large]} is {ratio}a.",
            f"{known_sum} + {ratio}a + a = {n_total}, so a = {f_small}.",
            f"frequency at x={values[i_large]} = {ratio} x {f_small} = {f_large}.",
        ] + [f"P(X = {v}) = {f}/{n_total} = {rational_text(p)}." for v, f, p in zip(values, freqs, probs)]
    else:
        raise ValueError(f"unknown variant {variant!r}")

    answer_text = ", ".join(rational_text(p) for p in probs)
    answer = list(probs)

    reversed_order = list(reversed(probs))
    off_by_one = probs[1:] + probs[:1]
    wrong_total_denominator = n_total + 1  # divides by the wrong total
    wrong_total_probs = [Fraction(f, wrong_total_denominator) for f in freqs]

    distractors = pick_distractors(
        answer_text,
        [
            ", ".join(rational_text(p) for p in reversed_order),
            ", ".join(rational_text(p) for p in off_by_one),
            ", ".join(rational_text(p) for p in wrong_total_probs),
        ],
        n=3,
        unordered=False,
    )
    return dict(
        text=question_text,
        latex=question_text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
