"""Generates `p16a_relative_frequency_estimate` and `p16a_histogram_probability` questions: probabilities
estimated from observed frequencies and from histograms.

Senior statistics computations, part 3.

Two generators, both senior ``acara-senior-v8-4``, domain
``statistics-probability``, ``response_family: scalar_expression``,
``answer: fraction``:

- ``p16a_relative_frequency_estimate``
  (``senior-use-relative-frequencies-obtained-from-data-as-point``,
  ACMMM055) -- from a frequency table of OBSERVED trials of an experiment,
  estimate the probability of a stated event (single, union or complement)
  as its relative frequency, or project that estimate onto a stated number
  of future trials. Genuinely different from the shipping
  ``simple-event-probability`` / ``finite-sample-space-probability`` (a
  theoretical probability from a stated collection of known outcomes): this
  skill's prompt is always a table of trial COUNTS, never an enumerated
  sample space, and the estimate is always empirical.

- ``p16a_histogram_probability``
  (``senior-use-relative-frequencies-and-histograms-obtained-from-data``,
  ACMMM164) -- from a histogram's class frequencies for a CONTINUOUS random
  variable, estimate the probability that the variable falls in a stated
  interval, aligned to class boundaries on the easy/medium/hard tiers and
  requiring a within-class uniform-density interpolation on the exam tier
  (the genuinely new reasoning step a continuous variable needs that a
  discrete one never does). Distinct from ``p16a_estimated_distribution``
  (ACMMM137, a DISCRETE variable's full per-value list) and from
  ``p16a_relative_frequency_estimate`` (ACMMM055, a single discrete event).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, pick_distractors, rational_text, round_half_up

# --------------------------------------------------------------------------
# p16a_relative_frequency_estimate (ACMMM055)
# --------------------------------------------------------------------------

_TRIAL_CONTEXTS = [
    # (trial clause, with a `{n}` placeholder for the trial count -- a
    # complete, correctly-conjugated sentence opener on its own, never a
    # bare noun phrase forced through one generic "{device} was used {n}
    # times" template that reads fine for a device but not for an event --
    # category word, category labels)
    ("A spinner was spun {n} times", "colour", ["red", "blue", "green", "yellow"]),
    ("A six-sided die was rolled {n} times", "outcome", ["1", "2", "3", "4", "5", "6"]),
    ("{n} customer service calls were logged", "outcome type", ["resolved", "escalated", "abandoned"]),
    ("{n} seeds from a packet were planted", "germination outcome", ["sprouted", "failed", "delayed"]),
]


def _verify_relative_frequency(freqs: dict, n_total: int, favourable_keys: list, prob: Fraction) -> None:
    """Independent route: recompute the favourable count as
    (n_total - count of every category NOT in favourable_keys), rather than
    construction's own direct sum over favourable_keys."""
    check(sum(freqs.values()) == n_total, "frequencies do not sum to the stated total")
    other_count = sum(f for k, f in freqs.items() if k not in favourable_keys)
    recomputed = Fraction(n_total - other_count, n_total)
    check(recomputed == prob, "complement-based recomputation disagrees with construction")


@generator(
    "p16a_relative_frequency_estimate",
    level=12,
    presets={
        "easy": dict(variant="single"),
        "medium": dict(variant="union"),
        "hard": dict(variant="complement"),
        "exam": dict(variant="project_future"),
    },
)
def p16a_relative_frequency_estimate(rng, variant="single"):
    """From a frequency table of OBSERVED trials, estimate the probability
    of a stated event as its relative frequency (count / total trials).
    Never a theoretical probability from a stated collection of known
    outcomes -- always an empirical estimate from recorded trial counts."""
    trial_clause, category_word, labels = rng.choice(_TRIAL_CONTEXTS)
    freqs = {label: rng.randint(8, 40) for label in labels}
    n_total = sum(freqs.values())
    table = ", ".join(f"{label}: {f}" for label, f in freqs.items())
    intro = f"{trial_clause.format(n=n_total)}, and the {category_word} of each was recorded: {table}."

    if variant == "single":
        target = rng.choice(labels)
        favourable = [target]
        prob = Fraction(freqs[target], n_total)
        _verify_relative_frequency(freqs, n_total, favourable, prob)
        question_text = (
            f"{intro} Based on these trials, estimate the probability that the next trial "
            f"gives {target}, as a fraction in lowest terms."
        )
        working = [f"P({target}) = {freqs[target]}/{n_total} = {rational_text(prob)}."]
        wrong_count = Fraction(freqs[target])  # reports the count, not the relative frequency
        wrong_denominator = Fraction(freqs[target], n_total - freqs[target])  # favourable over "other", not over total

    elif variant == "union":
        pair = rng.sample(labels, 2)
        favourable = list(pair)
        count = sum(freqs[k] for k in pair)
        prob = Fraction(count, n_total)
        _verify_relative_frequency(freqs, n_total, favourable, prob)
        question_text = (
            f"{intro} Based on these trials, estimate the probability that the next trial "
            f"gives {pair[0]} or {pair[1]}, as a fraction in lowest terms."
        )
        working = [
            f"count({pair[0]} or {pair[1]}) = {freqs[pair[0]]} + {freqs[pair[1]]} = {count}.",
            f"P = {count}/{n_total} = {rational_text(prob)}.",
        ]
        wrong_count = Fraction(count)
        wrong_denominator = Fraction(freqs[pair[0]], n_total)  # only counts one of the two categories

    elif variant == "complement":
        target = rng.choice(labels)
        favourable = [k for k in labels if k != target]
        count = n_total - freqs[target]
        prob = Fraction(count, n_total)
        _verify_relative_frequency(freqs, n_total, favourable, prob)
        question_text = (
            f"{intro} Based on these trials, estimate the probability that the next trial "
            f"does NOT give {target}, as a fraction in lowest terms."
        )
        working = [
            f"P(not {target}) = 1 - {freqs[target]}/{n_total} = {rational_text(prob)}.",
        ]
        wrong_count = Fraction(freqs[target], n_total)  # forgets to take the complement
        wrong_denominator = Fraction(n_total - freqs[target], freqs[target])  # complement count over the target count

    elif variant == "project_future":
        target = rng.choice(labels)
        favourable = [target]
        prob = Fraction(freqs[target], n_total)
        _verify_relative_frequency(freqs, n_total, favourable, prob)
        future_n = rng.choice([50, 80, 100, 150, 200])
        raw = prob * future_n
        expected_count = round_half_up(raw, 0)
        check(expected_count.denominator == 1, "rounded projected count must be a whole number")
        prob = expected_count  # the graded value: the projected count, an exact integer "fraction"
        question_text = (
            f"{intro} Based on the relative frequency of {target} in these trials, "
            f"estimate how many of the next {future_n} trials will give {target}. Round "
            f"to the nearest whole number."
        )
        working = [
            f"P({target}) = {freqs[target]}/{n_total} = {rational_text(Fraction(freqs[target], n_total))}.",
            f"Expected count = {rational_text(Fraction(freqs[target], n_total))} x {future_n} = {rational_text(prob)}.",
        ]
        wrong_count = Fraction(freqs[target])  # uses the observed count directly, ignoring future_n
        wrong_denominator = round_half_up(Fraction(freqs[target], n_total - freqs[target]) * future_n, 0)
    else:
        raise ValueError(f"unknown variant {variant!r}")

    answer_text = rational_text(prob)
    # An extra misconception shared by every tier: treats every category as
    # equally likely (1/number of categories), ignoring the data entirely.
    equal_likely = Fraction(1, len(labels))
    if variant == "project_future":
        # The graded value is a projected COUNT, not a probability, so the [0, 1] rule below does not apply.
        candidates = [rational_text(wrong_count), rational_text(wrong_denominator), rational_text(equal_likely)]
    else:
        # The answer is a probability, so every option must be one: a count, or a ratio above 1, is
        # eliminated at sight. In-range stand-ins follow the misconceptions: the complement (answers the
        # opposite question) and the count nudged by one over the right total.
        count_fav = prob * n_total
        values = [
            wrong_count,
            wrong_denominator,
            equal_likely,
            1 - prob,
            (count_fav - 1) / n_total,
            (count_fav + 1) / n_total,
        ]
        candidates = [rational_text(v) for v in values if 0 < v < 1 and v != prob]

    distractors = pick_distractors(answer_text, candidates, n=3)
    if variant != "project_future":
        check(len(distractors) == 3, "need three in-range distractors")
    return dict(
        text=question_text,
        latex=question_text,
        answer={"p": prob.numerator, "q": prob.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


# --------------------------------------------------------------------------
# p16a_histogram_probability (ACMMM164)
# --------------------------------------------------------------------------

_HIST_CONTEXTS = [
    ("the delivery time (minutes) of parcels for a courier", "parcels"),
    ("the battery life (hours) of a batch of phones", "phones"),
    ("the commute time (minutes) of employees at a company", "employees"),
    ("the fill weight (grams) of packets from a production line", "packets"),
]


def _build_histogram(rng, n_classes: int, width: int = 10):
    """Build n_classes consecutive equal-width class intervals starting at a
    random multiple of `width`, each with a random frequency, and return
    (boundaries, freqs, n_total). boundaries has n_classes+1 entries."""
    start = rng.choice([0, 20, 40])
    boundaries = [start + i * width for i in range(n_classes + 1)]
    freqs = [rng.randint(6, 30) for _ in range(n_classes)]
    n_total = sum(freqs)
    return boundaries, freqs, n_total


def _verify_histogram_probability(freqs: list[int], n_total: int, included: list[int], prob: Fraction) -> None:
    """Independent route: recompute via 1 - (probability of every class NOT
    included), rather than construction's own direct sum over `included`."""
    check(sum(freqs) == n_total, "frequencies do not sum to the stated total")
    excluded_count = sum(f for i, f in enumerate(freqs) if i not in included)
    recomputed = 1 - Fraction(excluded_count, n_total)
    check(recomputed == prob, "complement-based recomputation disagrees with construction")


@generator(
    "p16a_histogram_probability",
    level=12,
    presets={
        "easy": dict(variant="single_class"),
        "medium": dict(variant="union_classes"),
        "hard": dict(variant="tail"),
        "exam": dict(variant="interpolate"),
    },
)
def p16a_histogram_probability(rng, variant="single_class"):
    """From a histogram's class frequencies for a continuous random
    variable, estimate the probability that the variable falls in a stated
    interval. Every tier before 'exam' aligns the interval to class
    boundaries exactly; 'exam' requires assuming the variable is uniformly
    distributed WITHIN one class to interpolate a partial-class
    probability -- the genuinely new step a continuous variable needs."""
    context, subjects = rng.choice(_HIST_CONTEXTS)
    width = 20

    if variant == "single_class":
        n_classes = 4
        boundaries, freqs, n_total = _build_histogram(rng, n_classes, width)
        idx = rng.randrange(n_classes)
        included = [idx]
        prob = Fraction(freqs[idx], n_total)
        _verify_histogram_probability(freqs, n_total, included, prob)
        table = ", ".join(f"[{boundaries[i]}, {boundaries[i + 1]}): {freqs[i]} {subjects}" for i in range(n_classes))
        lo, hi = boundaries[idx], boundaries[idx + 1]
        question_text = (
            f"A histogram summarises {context} for {n_total} {subjects}, by class: {table}. "
            f"Estimate the probability that a randomly chosen one of these {subjects} has a "
            f"value in [{lo}, {hi}), as a fraction in lowest terms."
        )
        working = [f"P([{lo},{hi})) = {freqs[idx]}/{n_total} = {rational_text(prob)}."]

    elif variant == "union_classes":
        n_classes = 5
        boundaries, freqs, n_total = _build_histogram(rng, n_classes, width)
        start_idx = rng.randrange(n_classes - 1)
        included = [start_idx, start_idx + 1]
        count = freqs[start_idx] + freqs[start_idx + 1]
        prob = Fraction(count, n_total)
        _verify_histogram_probability(freqs, n_total, included, prob)
        table = ", ".join(f"[{boundaries[i]}, {boundaries[i + 1]}): {freqs[i]} {subjects}" for i in range(n_classes))
        lo, hi = boundaries[start_idx], boundaries[start_idx + 2]
        question_text = (
            f"A histogram summarises {context} for {n_total} {subjects}, by class: {table}. "
            f"Estimate the probability that a randomly chosen one of these {subjects} has a "
            f"value in [{lo}, {hi}), as a fraction in lowest terms."
        )
        working = [
            f"count = {freqs[start_idx]} + {freqs[start_idx + 1]} = {count}.",
            f"P = {count}/{n_total} = {rational_text(prob)}.",
        ]

    elif variant == "tail":
        n_classes = 5
        boundaries, freqs, n_total = _build_histogram(rng, n_classes, width)
        cut_idx = rng.randrange(1, n_classes)  # split point: classes [cut_idx, end) are "at least"
        included = list(range(cut_idx, n_classes))
        count = sum(freqs[i] for i in included)
        prob = Fraction(count, n_total)
        _verify_histogram_probability(freqs, n_total, included, prob)
        table = ", ".join(f"[{boundaries[i]}, {boundaries[i + 1]}): {freqs[i]} {subjects}" for i in range(n_classes))
        cutoff = boundaries[cut_idx]
        question_text = (
            f"A histogram summarises {context} for {n_total} {subjects}, by class: {table}. "
            f"Estimate the probability that a randomly chosen one of these {subjects} has a "
            f"value of at least {cutoff}, as a fraction in lowest terms."
        )
        working = [
            f"count(>= {cutoff}) = " + " + ".join(str(freqs[i]) for i in included) + f" = {count}.",
            f"P = {count}/{n_total} = {rational_text(prob)}.",
        ]

    elif variant == "interpolate":
        n_classes = 4
        boundaries, freqs, n_total = _build_histogram(rng, n_classes, width)
        split_idx = rng.randrange(n_classes)
        # The cutoff falls strictly INSIDE class split_idx, at a quarter
        # point of its width, so the interpolated fraction is always a nice
        # rational (1/4 or 3/4) and the resulting probability stays exact.
        quarter = rng.choice([1, 3])
        cutoff = boundaries[split_idx] + quarter * width // 4
        full_classes_above = list(range(split_idx + 1, n_classes))
        remaining_fraction_of_class = Fraction(4 - quarter, 4)  # fraction of split_idx's width above cutoff
        count_above_full = sum(freqs[i] for i in full_classes_above)
        partial_count = freqs[split_idx] * remaining_fraction_of_class
        prob = Fraction(count_above_full, n_total) + partial_count / n_total
        check(prob.denominator > 0, "interpolated probability must be a valid fraction")
        # Independent check: total probability (below cutoff + above cutoff) must be 1.
        below_full = sum(freqs[i] for i in range(split_idx))
        below_partial = freqs[split_idx] * Fraction(quarter, 4)
        prob_below = Fraction(below_full, n_total) + below_partial / n_total
        check(prob + prob_below == 1, "interpolated tail probabilities must sum to 1")
        table = ", ".join(f"[{boundaries[i]}, {boundaries[i + 1]}): {freqs[i]} {subjects}" for i in range(n_classes))
        question_text = (
            f"A histogram summarises {context} for {n_total} {subjects}, by class: {table}. "
            f"Assuming values are spread uniformly within each class, estimate the "
            f"probability that a randomly chosen one of these {subjects} has a value of at "
            f"least {cutoff}, as a fraction in lowest terms."
        )
        working = [
            f"{cutoff} is {remaining_fraction_of_class} of the way through the class "
            f"[{boundaries[split_idx]}, {boundaries[split_idx + 1]}) from its upper end.",
            f"partial count from that class = {freqs[split_idx]} x {remaining_fraction_of_class} = "
            f"{rational_text(partial_count)}.",
            f"full classes above = "
            + (" + ".join(str(freqs[i]) for i in full_classes_above) or "0")
            + f" = {count_above_full}.",
            f"P = ({count_above_full} + {rational_text(partial_count)})/{n_total} = {rational_text(prob)}.",
        ]
    else:
        raise ValueError(f"unknown variant {variant!r}")

    answer_text = rational_text(prob)

    if variant == "interpolate":
        # Misconceptions specific to the interpolation step:
        ignores_interpolation = Fraction(count_above_full, n_total)  # treats the split class as fully excluded
        wrong_fraction = (
            Fraction(count_above_full, n_total) + (freqs[split_idx] * Fraction(quarter, 4)) / n_total
        )  # uses the WRONG side's fraction of the class
        includes_whole_class = Fraction(
            count_above_full + freqs[split_idx], n_total
        )  # treats the split class as fully included
        candidates = [
            rational_text(ignores_interpolation),
            rational_text(wrong_fraction),
            rational_text(includes_whole_class),
        ]
    else:
        included_count = freqs[included[0]] if len(included) == 1 else sum(freqs[i] for i in included)
        excluded_total = n_total - included_count
        wrong_complement = Fraction(excluded_total, n_total)  # complement of the intended event
        equal_split = Fraction(1, n_classes)  # ignores the data, assumes every class equally likely
        candidates = [rational_text(wrong_complement), rational_text(equal_split)]
        if len(included) > 1:
            # only one of several classes, when a union/tail spanning more was asked
            candidates.append(rational_text(Fraction(freqs[included[0]], n_total)))
        else:
            # the ADJACENT class's own relative frequency, a plausible mis-read of the table
            other_idx = (included[0] + 1) % n_classes
            candidates.append(rational_text(Fraction(freqs[other_idx], n_total)))

    distractors = pick_distractors(answer_text, candidates, n=3)
    return dict(
        text=question_text,
        latex=question_text,
        answer={"p": prob.numerator, "q": prob.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
