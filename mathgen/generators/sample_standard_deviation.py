"""Sample standard deviation from a small data set. an exam-style item asks for a
standard deviation calculated from a list of measured values, worked step by
step, and the library has no skill that rehearses this:
``statistics-probability/finite-distribution-variance`` is the variance of a
*probability distribution* (P(X=x) weights), a different object entirely.

**Sample vs population decision: every question here computes the SAMPLE standard deviation -- dividing the
sum of squared deviations by n-1, never n -- because the data in every question is framed as a sample drawn
from a larger batch/population (bolts from a production run, a survey of runners, parts off two assembly
lines), which is exactly the situation `n-1` exists for, and is also the common real-world framing of a
"here is a sample of measurements" task. The question text says "sample standard deviation"
explicitly, every time, in the instruction sentence itself -- never left for the learner to infer -- because
the population standard deviation (divide by n instead of n-1) is close enough, for some draws, to land inside
the typed-number grading tolerance, and is exactly the wrong-divisor mistake a learner who forgot "sample"
would make. It is therefore the primary misconception distractor here, and every draw is resampled until it
clears the grading tolerance from the true answer (see `_TOL` below and each variant's resample loop) -- never
merely offered on the assumption that n vs n-1 is "obviously" far enough apart.**

Four variants:

  - ``integers``: whole numbers constructed so the mean is itself a whole
    number, so every deviation is also a whole number.
  - ``non_integer_mean``: whole numbers drawn so the mean is genuinely NOT a
    whole number, forcing decimal arithmetic on every deviation.
  - ``context``: a data set given in a measurement context, with a unit,
    using one-decimal-place values, so the learner also carries a unit and
    works with decimal *inputs* (not just a decimal mean).
  - ``compare_sets``: two labelled data sets are given side by side (a
    genuine spread-comparison framing) and the question asks for the sample
    standard deviation of ONE named set; the other set's sample standard
    deviation is offered as a "used the wrong data set" distractor -- a
    plausible misreading distinct from a n-vs-(n-1) divisor slip.

Every variant reuses the same core statistics/verification/distractor
machinery below.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, generator, pick_distractors, rounds_unambiguously

_PLACES = 2
# Half a unit in the last stated decimal place -- the SAME tolerance
# mathgen/grading/typed.py's grade_number derives from a "to 2 decimal places" prompt
# (see its `_typed_number_tolerance`). Every distractor must clear this,
# compared on the *displayed* (rounded) strings, never on raw floats (two
# raw values 0.01 apart can round to the same 2dp display).
_TOL = Fraction(1, 2 * 10**_PLACES)

_MAX_ATTEMPTS = 500


def _sample_stats(values: list) -> tuple:
    """(mean, sum_sq_dev, sample_variance) computed exactly with Fraction
    arithmetic. `sum_sq_dev` (sum of squared deviations from the mean) is
    checked against the sum-of-squares identity -- sum(x^2) - (sum x)^2/n --
    a genuinely different route to the same quantity, before being trusted.
    """
    n = len(values)
    total = sum(values)
    mean = total / n
    sum_sq_dev = sum((v - mean) ** 2 for v in values)
    identity = sum(v * v for v in values) - total * total / n
    check(abs(identity - sum_sq_dev) < Fraction(1, 10**9), "sum-of-squares identity failed for the deviations")
    sample_variance = sum_sq_dev / (n - 1)
    return mean, sum_sq_dev, sample_variance


def _fmt_value(v, *, force_one_decimal: bool = False) -> str:
    """Format one RAW input data value (an integer, or a context tier's
    exact one-decimal-place reading) the way it was given -- never rounded,
    since these are exact inputs, not computed intermediates.

    `force_one_decimal` is set by the `context` variant, whose values are all
    exact readings to one decimal place even when that decimal is zero (a
    22.0 mm bolt is still a one-decimal-place reading, and must not render as
    the bare integer "22" beside "22.9" and "21.8" in the same list)."""
    v = Fraction(v)
    if force_one_decimal:
        return f"{float(v):.1f}"
    return str(v.numerator) if v.denominator == 1 else f"{float(v):.1f}"


def _fmt_values(values, *, force_one_decimal: bool = False) -> str:
    return ", ".join(_fmt_value(v, force_one_decimal=force_one_decimal) for v in values)


def _fmt_calc(v) -> str:
    """Format one COMPUTED intermediate (mean, sum of squared deviations,
    sample variance) for the worked-solution text. Unlike `_fmt_value`
    (exact inputs), these are often non-terminating decimals, so they are
    rounded to 4dp and trimmed -- enough precision that the next working
    line's arithmetic is recognisably consistent with it, without pretending
    to more accuracy than the working actually carries. Never used for the
    graded `answer_text` (that always keeps exactly `_PLACES` decimals)."""
    v = Fraction(v)
    if v.denominator == 1:
        return str(v.numerator)
    return f"{float(v):.4f}".rstrip("0").rstrip(".")


def _fmt2(x) -> str:
    return f"{x:.2f}"


def _distinct_from(target: float, *others: float) -> bool:
    """True only when EVERY `others` value clears `_TOL` from `target`,
    compared on the strings each would actually be DISPLAYED as (2dp) -- not
    on the raw floats, which can be up to 0.01 apart and still round to the
    same displayed value."""
    target_r = Fraction(_fmt2(round(target, _PLACES)))
    for o in others:
        other_r = Fraction(_fmt2(round(o, _PLACES)))
        if abs(other_r - target_r) <= _TOL:
            return False
    return True


def _misconceptions(sum_sq_dev: Fraction, n: int, sample_variance: Fraction):
    """The three shared misconception values for a single data set: the
    population standard deviation (wrong divisor), the sample variance
    (forgot the square root), and the square root of the RAW sum of squared
    deviations with no division at all (forgot to divide before rooting)."""
    population_variance = sum_sq_dev / n
    population_sd = math.sqrt(float(population_variance))
    variance_only = float(sample_variance)
    no_divide_sd = math.sqrt(float(sum_sq_dev))
    return population_sd, variance_only, no_divide_sd


def _build_single(values, context_sentence: str, instruction: str, *, force_one_decimal: bool = False):
    """Shared construction for the three single-data-set variants. Returns
    the generator's result dict, or None if this draw must be resampled
    (caller retries with a fresh draw). `force_one_decimal` is passed through
    to the raw-value formatting (see `_fmt_value`)."""
    n = len(values)
    mean, sum_sq_dev, sample_variance = _sample_stats(values)
    if sum_sq_dev == 0:
        return None  # degenerate: every value equal, zero spread
    sample_sd = math.sqrt(float(sample_variance))
    if not rounds_unambiguously(sample_sd, _PLACES):
        return None
    sample_sd_r = round(sample_sd, _PLACES)

    population_sd, variance_only, no_divide_sd = _misconceptions(sum_sq_dev, n, sample_variance)
    if not _distinct_from(sample_sd_r, population_sd, variance_only, no_divide_sd):
        return None

    # Independent check: squaring the derived answer back must return the
    # sample variance it was rooted from (an inverse-operation route, not a
    # restatement of the sqrt call itself).
    check(
        abs(sample_sd * sample_sd - float(sample_variance)) < 1e-9,
        "sample standard deviation does not square back to the sample variance",
    )

    answer_text = _fmt2(sample_sd_r)
    candidates = [
        _fmt2(round(population_sd, _PLACES)),
        _fmt2(round(variance_only, _PLACES)),
        _fmt2(round(no_divide_sd, _PLACES)),
    ]
    distractors = pick_distractors(answer_text, candidates, n=3, tolerance=_TOL)
    if len(distractors) < 2:
        return None

    working = [
        f"Mean = ({' + '.join(_fmt_value(v, force_one_decimal=force_one_decimal) for v in values)}) "
        f"/ {n} = {_fmt_calc(mean)}.",
        f"Sum of squared deviations from the mean = {_fmt_calc(sum_sq_dev)}.",
        f"Sample variance = {_fmt_calc(sum_sq_dev)} / ({n} - 1) = {_fmt_calc(sample_variance)}.",
        f"Sample standard deviation = square root of {_fmt_calc(sample_variance)} "
        f"= {answer_text} (to 2 decimal places).",
    ]

    text = f"{context_sentence} {_fmt_values(values, force_one_decimal=force_one_decimal)}. {instruction}"

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=float(sample_sd_r),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# Variant: integers -- whole numbers, integer mean.
# ---------------------------------------------------------------------------


def _draw_integers(rng):
    n = rng.choice((5, 6, 7))
    mean_target = rng.randint(10, 40)
    offsets = [rng.randint(-6, 6) for _ in range(n - 1)]
    last = -sum(offsets)
    if abs(last) > 12:
        return None
    raw = [mean_target + o for o in (*offsets, last)]
    if min(raw) < 1:
        return None  # a measured whole-number value cannot be zero/negative
    values = [Fraction(v) for v in raw]
    return values


def _variant_integers(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        values = _draw_integers(rng)
        if values is None:
            continue
        result = _build_single(
            values,
            context_sentence=f"Here are the values in a sample of {len(values)} whole numbers:",
            instruction="Treating these as a sample, calculate the sample standard deviation, to 2 decimal places.",
        )
        if result is not None:
            return result
    raise RuntimeError(
        "standard_deviation_from_data[integers]: no draw satisfied the resample constraints in 500 attempts"
    )


# ---------------------------------------------------------------------------
# Variant: non_integer_mean -- whole numbers whose mean is NOT a whole number.
# ---------------------------------------------------------------------------


def _draw_non_integer_mean(rng):
    n = rng.choice((6, 7, 8))
    values = [rng.randint(4, 45) for _ in range(n)]
    if sum(values) % n == 0:
        return None  # mean would be a whole number -- this tier needs it not to be
    if len(set(values)) == 1:
        return None
    return [Fraction(v) for v in values]


def _variant_non_integer_mean(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        values = _draw_non_integer_mean(rng)
        if values is None:
            continue
        result = _build_single(
            values,
            context_sentence=f"Here are the values in a sample of {len(values)} whole numbers:",
            instruction="Treating these as a sample, calculate the sample standard deviation, to 2 decimal places.",
        )
        if result is not None:
            return result
    raise RuntimeError(
        "standard_deviation_from_data[non_integer_mean]: no draw satisfied the resample constraints in 500 attempts"
    )


# ---------------------------------------------------------------------------
# Variant: context -- a measurement context, one-decimal-place values, a unit.
# ---------------------------------------------------------------------------

# Each entry is (description template, base tenths range, spread in tenths)
# -- realistic magnitude bands per context, in TENTHS of the stated unit, so
# every drawn value is an exact one-decimal-place reading (never a whole
# number that happens to look like an integer's worth of decimal places).
_CONTEXTS = (
    (
        "the diameters, in millimetres, of a sample of {n} bolts taken from a production batch",
        (180, 300),
        15,
    ),  # 18.0-30.0 mm +/- 1.5 mm
    (
        "the masses, in grams, of a sample of {n} eggs taken from one carton run",
        (500, 700),
        20,
    ),  # 50.0-70.0 g +/- 2.0 g
    (
        "the times, in seconds, taken by a sample of {n} sprinters to run 100 metres",
        (110, 150),
        10,
    ),  # 11.0-15.0 s +/- 1.0 s
    (
        "the lengths, in centimetres, of a sample of {n} offcuts from one cutting run",
        (150, 350),
        15,
    ),  # 15.0-35.0 cm +/- 1.5 cm
)


def _draw_context(rng):
    n = rng.choice((6, 7, 8))
    template, (lo, hi), spread = rng.choice(_CONTEXTS)
    base_tenths = rng.randint(lo, hi)
    tenths = [base_tenths + rng.randint(-spread, spread) for _ in range(n)]
    if min(tenths) < 1:
        return None
    if len(set(tenths)) == 1:
        return None
    values = [Fraction(t, 10) for t in tenths]
    return values, template.format(n=n)


def _variant_context(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        drawn = _draw_context(rng)
        if drawn is None:
            continue
        values, description = drawn
        result = _build_single(
            values,
            context_sentence=f"Here are {description}:",
            instruction="Treating these as a sample, calculate the sample standard deviation, to 2 decimal places.",
            force_one_decimal=True,
        )
        if result is not None:
            return result
    raise RuntimeError(
        "standard_deviation_from_data[context]: no draw satisfied the resample constraints in 500 attempts"
    )


# ---------------------------------------------------------------------------
# Variant: compare_sets -- two labelled data sets, asked for one's sample SD.
# ---------------------------------------------------------------------------

_SET_LABELS = (
    ("Line A", "Line B", "the assembly times, in minutes, of parts made on"),
    ("Team 1", "Team 2", "the times, in minutes, taken by"),
    ("Morning shift", "Afternoon shift", "the number of items packed per hour by"),
)


def _draw_group(rng, n):
    mean_target = rng.randint(10, 40)
    offsets = [rng.randint(-6, 6) for _ in range(n - 1)]
    last = -sum(offsets)
    if abs(last) > 12:
        return None
    raw = [mean_target + o for o in (*offsets, last)]
    if min(raw) < 1:
        return None  # a measured whole-number value cannot be zero/negative
    values = [Fraction(v) for v in raw]
    if len(set(values)) == 1:
        return None
    return values


def _variant_compare_sets(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        n = rng.choice((5, 6, 7))
        values_a = _draw_group(rng, n)
        values_b = _draw_group(rng, n)
        if values_a is None or values_b is None:
            continue

        label_a, label_b, clause = rng.choice(_SET_LABELS)
        target_first = rng.random() < 0.5
        target_label, target_values = (label_a, values_a) if target_first else (label_b, values_b)
        other_values = values_b if target_first else values_a

        mean, sum_sq_dev, sample_variance = _sample_stats(target_values)
        if sum_sq_dev == 0:
            continue
        sample_sd = math.sqrt(float(sample_variance))
        if not rounds_unambiguously(sample_sd, _PLACES):
            continue
        sample_sd_r = round(sample_sd, _PLACES)

        _, other_sum_sq_dev, other_sample_variance = _sample_stats(other_values)
        if other_sum_sq_dev == 0:
            continue
        other_sd = math.sqrt(float(other_sample_variance))

        population_sd, variance_only, no_divide_sd = _misconceptions(sum_sq_dev, n, sample_variance)
        if not _distinct_from(sample_sd_r, population_sd, variance_only, no_divide_sd, other_sd):
            continue

        check(
            abs(sample_sd * sample_sd - float(sample_variance)) < 1e-9,
            "sample standard deviation does not square back to the sample variance",
        )

        answer_text = _fmt2(sample_sd_r)
        candidates = [
            _fmt2(round(population_sd, _PLACES)),
            _fmt2(round(variance_only, _PLACES)),
            _fmt2(round(other_sd, _PLACES)),
        ]
        distractors = pick_distractors(answer_text, candidates, n=3, tolerance=_TOL)
        if len(distractors) < 2:
            continue

        text = (
            f"A quality check recorded {clause} {label_a} and {label_b}. "
            f"{label_a}: {_fmt_values(values_a)}. "
            f"{label_b}: {_fmt_values(values_b)}. "
            f"Treating {target_label}'s data as a sample, calculate the sample "
            f"standard deviation of {target_label}'s values, to 2 decimal places."
        )

        working = [
            f"{target_label}'s mean = ({' + '.join(_fmt_value(v) for v in target_values)}) / {n} = {_fmt_calc(mean)}.",
            f"{target_label}'s sum of squared deviations from the mean = {_fmt_calc(sum_sq_dev)}.",
            f"{target_label}'s sample variance = {_fmt_calc(sum_sq_dev)} / ({n} - 1) = {_fmt_calc(sample_variance)}.",
            f"{target_label}'s sample standard deviation = square root of "
            f"{_fmt_calc(sample_variance)} = {answer_text} (to 2 decimal places).",
        ]

        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=float(sample_sd_r),
            answer_text=answer_text,
            answer_latex=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError(
        "standard_deviation_from_data[compare_sets]: no draw satisfied the resample constraints in 500 attempts"
    )


_VARIANTS = {
    "integers": _variant_integers,
    "non_integer_mean": _variant_non_integer_mean,
    "context": _variant_context,
    "compare_sets": _variant_compare_sets,
}


@generator(
    "standard_deviation_from_data",
    level=10,
    presets={
        "integers": dict(variant="integers"),
        "non_integer_mean": dict(variant="non_integer_mean"),
        "context": dict(variant="context"),
        "compare_sets": dict(variant="compare_sets"),
        # Tier names, which is what a skill source's `difficulty` must be. Each sets the variant that
        # opens that tier; a source whose `params` name another variant (the `context` source is a
        # `hard` one) overrides it.
        "easy": dict(variant="integers"),
        "medium": dict(variant="non_integer_mean"),
        "hard": dict(variant="compare_sets"),
    },
)
def standard_deviation_from_data(rng, variant="integers"):
    """Sample standard deviation of a small (5-8 value) data set, always
    labelled "sample" in the question text (see module docstring for the
    sample-vs-population decision). `variant` selects one of the four
    reasoning tiers above; each is a `sources` entry in
    `skills/statistics-probability/standard-deviation-from-data.yaml`.

    The four tier names are ALSO registered as `difficulty` presets (each
    just setting the same `variant` `params` value): `mathgen.core.make()`'s
    seed hash is `f"{topic}:{difficulty or ''}:{seed}"`, which does not
    include `params`, so without a distinct `difficulty` string per tier the
    four variants would draw byte-identical random numbers at a shared seed
    (seed 0 gave literally the same six data
    values for the `integers` tier and one labelled set of `compare_sets`).
    Registering real presets -- rather than merely passing an arbitrary
    `difficulty` string through to a topic with no presets at all, which
    would silently start raising `KeyError` the moment any preset was later
    added to this topic -- also lets any generic check over ``PRESETS`` exercise
    this tier structure.

    The skill labels its sources with the library's tiers (``easy``, ``medium``, ``hard``), and so registers
    ``easy``, ``medium`` and ``hard`` presets too. ``context`` and ``compare_sets`` are both ``hard`` there, so
    at one seed they start from the same random stream; they then draw differently (different ``n`` choices
    and construction), so their questions are not copies of each other."""
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant: {variant!r}; known: {', '.join(_VARIANTS)}")
    return _VARIANTS[variant](rng)
