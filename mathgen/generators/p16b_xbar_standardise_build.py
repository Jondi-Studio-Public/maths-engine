"""Generates `p16b_xbar_sigma_standardise` and `p16b_xbar_s_standardise` questions: standardise a simulated
sample mean using the known or the sample standard deviation.

The X-bar-standardising pair.

- ``p16b_xbar_sigma_standardise``
  (``senior-simulate-repeated-random-sampling-from-a-variety-of``, ACMSM138,
  domain statistics-probability, answer: number)
  -- a seeded, reproducible simulation of repeated random sampling of size n
  from a stated distribution with mean mu and (KNOWN) standard deviation
  sigma; read the reported sample mean X-bar for a stated one of the
  samples, then compute the standardised value (X-bar - mu) / (sigma /
  sqrt(n)).

- ``p16b_xbar_s_standardise``
  (``senior-simulate-repeated-random-sampling-from-a-variety-of-acmsm139``,
  ACMSM139, domain statistics-probability, answer: number)
  -- the same simulation shape, but sigma is never known or stated: the
  learner reads the sample mean X-bar AND the sample standard deviation s
  (n - 1 divisor) for a stated sample, and standardises with (X-bar - mu) /
  (s / sqrt(n)). Every draw has n >= 30.

The two rows are co-located deliberately (one author keeps them visibly
apart): **the sigma row never prints a sample standard deviation
or sample variance; the s row never states or names a population sigma at
any tier.** Each row's standard error is its own row's formula only --
enforced textually by ``test_p16b_xbar_standardise_build.py``'s cross-row
guard.

Design choice recorded here: every sample mean/standard deviation/raw sum a question prints is
ROUNDED to 2 decimal places and the standardised value is computed from
that PRINTED, rounded figure -- never from an unrounded internal value the
learner cannot see -- with the prompt saying so ("reported to 2 decimal
places"). This is one of two options (the other
being an exact terminating decimal for a purely discrete population); this
module always takes the rounded branch, for every population, so one
formatting/precision discipline covers every tier of both skills.

A REAL simulation with the seeded ``rng`` draws every printed sample from
the population it names -- ``mathgen.core.check`` verifies the standardised
value a second, independent way before it is offered to a learner.

Notes:

1. Greek letters as unicode (mu -> \u03bc, sigma -> \u03c3) in ``text``,
   real ``\\mu``/``\\sigma`` commands in ``latex`` (see ``_greekify`` below
   -- ``latex_prompt`` in ``mathgen/core.py`` has no trigger for these two
   letters, a frozen contract not extended here;
Every tier of both rows also states the standardising
   formula symbolically once (``_SYMBOLIC_A``/``_SYMBOLIC_B``,
   ``_FORMULA_LATEX_A``/``_FORMULA_LATEX_B``).
2. "samples 1-4"/"5-8" (row A hard) rewritten "samples 1 to 4"/"5 to 8" --
   a bare digit-hyphen-digit run is `latex_prompt`'s subtraction trigger.
3. Row B's per-sample stat lists are semicolon-separated, not
   comma-separated (a comma inside one sample's own "mean = M, s = S" pair
   made the list ambiguous).
4. The indicator population's clause reads "...whose values are 0 or 1,
   with P(value = 1) = q" -- "0/1 indicator" would have entered maths mode
   as a fraction.
5. Both hard tiers draw their trap/plain class ONCE per question (never
   re-drawn inside the resimulate loop, which is what silently skewed the
   realised split before), then scan every valid same-question pair for one
   that realises the drawn class with a >= 0.05 |z| gap, picking uniformly
   among the hits. Row B's trap test is now symmetric in which sample has
   the larger raw deviation (row A's is genuinely one-directional: samples
   1-4 are always the smaller-n group by construction).
6. Every displayed value -- the key AND every distractor -- is built from
   `Decimal` arithmetic (50 significant digits, `getcontext().prec`) on the
   PRINTED inputs, then rounded half-up via the shared
   `round_half_up`/`stated_precision_string` pair, never from a raw float
   quotient reformatted with Python's `"%.2f"` (which silently rounds the
   wrong way when the true value sits exactly on a half-boundary that the
   nearest float represents fractionally below it, e.g. 0.11 / (10/15) =
   0.165 exactly, displayed "0.16" by naive float formatting).
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, getcontext
from fractions import Fraction

from ..core import (
    check,
    check_pairwise_distinct,
    generator,
    latex_answer,
    latex_prompt,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)

DP = 2
TOL = stated_precision_tolerance(DP)

getcontext().prec = 50


# ---------------------------------------------------------------------------
# Decimal-exact display pipeline -- every KEY and every
# DISTRACTOR is built through this, never through a raw float division
# reformatted with "%.2f".
# ---------------------------------------------------------------------------


def _pdec(x) -> Decimal:
    """Exact `Decimal` for a value already fixed at (at most) 2 displayed
    decimal places, or a plain `int`/`Fraction` -- built from the value's
    own decimal STRING (never a float's binary repr)."""
    if isinstance(x, Decimal):
        return x
    if isinstance(x, int):
        return Decimal(x)
    if isinstance(x, Fraction):
        return Decimal(x.numerator) / Decimal(x.denominator)
    return Decimal(f"{x:.2f}")


def _dp2(value: Decimal) -> str:
    """Round `value` half-up to 2 d.p. via the shared `round_half_up` /
    `stated_precision_string` pair -- `Decimal` converts to `Fraction`
    exactly, so this is the same rounding rule the rest of the codebase
    uses, fed an operand computed at 50 significant digits rather than a
    raw float quotient."""
    return stated_precision_string(round_half_up(Fraction(value), DP), DP)


def _finalize(z_key: Decimal, candidates: list[Decimal]):
    """Common finish for every tier: format the key and every candidate at
    the stated 2 d.p. from exact `Decimal` values, drop any candidate that
    collides with the key within grading tolerance, and prove every
    surviving option pairwise distinct at the displayed precision."""
    answer_text = _dp2(z_key)
    candidate_texts = [_dp2(c) for c in candidates]
    distractors = pick_distractors(answer_text, candidate_texts, tolerance=TOL)
    all_fracs = [Fraction(answer_text)] + [Fraction(t) for t in distractors]
    check_pairwise_distinct(*all_fracs)
    return answer_text, distractors


def _fmt_num(x) -> str:
    """Render a population parameter the way it is naturally stated: "4" for
    4.0, "0.4" for a one-decimal proportion."""
    xf = float(x)
    if xf == int(xf):
        return str(int(xf))
    return f"{xf:g}"


def _to_2dp_fraction(x: float) -> Fraction:
    """The exact `Fraction` on the 2-d.p. grid a `"%.2f"`-formatted `x`
    displays -- the PRINTED value every downstream computation reads."""
    return Fraction(f"{x:.2f}")


def _disp2(x: float) -> str:
    return stated_precision_string(_to_2dp_fraction(x), DP)


def _verify_z(xbar: float, mu: float, se_input: float, n: int, z: float, eps: float = 1e-9) -> None:
    """Independent route: re-derive the standard error through a square and a
    single sqrt (sqrt(se_input**2 / n)) rather than reusing the caller's own
    `se_input / sqrt(n)` expression, mirroring `p8c_confidence_interval_build.
    py`'s `_verify_interval`."""
    se2 = math.sqrt((se_input**2) / n)
    z2 = (xbar - mu) / se2
    check(abs(z - z2) < eps, f"z verification failed: {z} != {z2}")


def _sign_flip_or_alt(
    rng, xbar_key_d: Decimal, mu_d: Decimal, se_key_d: Decimal, divisor_d: Decimal, sqrt_n_d: Decimal
) -> Decimal:
    """Measured over 200 seeds, the negated key ((mu - xbar)/se, the "reversed the subtraction" misconception)
    was among the distractors in 200/200 draws of every tier -- every question offered exactly one +/- pair,
    and the key was always one of its two members, so a guesser could always narrow to two options. Draw the
    sign-flip candidate with the rng on only about half of draws; the other half, offer a DIFFERENT genuine
    misconception instead -- multiplying by sqrt(n) rather than dividing by it (a real "inverted the sqrt(n)
    operation" error), so the key's negation is not always among the options."""
    if rng.random() < 0.5:
        return (mu_d - xbar_key_d) / se_key_d
    return (xbar_key_d - mu_d) * sqrt_n_d / divisor_d


# ---------------------------------------------------------------------------
# Greek letters and the symbolic formula sentence.
# ---------------------------------------------------------------------------

_TEXT_EMPTY_RE = re.compile(r"\\text\{\}")


def _greekify(latex_str: str) -> str:
    """Convert the \u03bc/\u03c3 unicode letters `latex_prompt` leaves untouched inside its `\\text{...}` prose
    spans into real `\\mu`/`\\sigma` LaTeX commands. `latex_prompt` (`mathgen/core.py`) has no trigger for
    these Greek letters -- they are plain letters to its parser, so every occurrence is still sitting inside a
    `\\text{}` span after the auto-conversion. Breaking `\\text{}` at each occurrence and emitting the bare
    command is the same idiom `latex_prompt` itself uses for a lifted operator: leave the surrounding prose in
    `\\text{}`, drop into real math for the one symbol, re-enter `\\text{}` after."""
    s = latex_str.replace("\u03bc", "}\\mu\\text{")
    s = s.replace("\u03c3", "}\\sigma\\text{")
    return _TEXT_EMPTY_RE.sub("", s)


_SYMBOLIC_A = " In symbols, z = (x\u0304 \u2212 \u03bc)/(\u03c3/\u221an)."
_SYMBOLIC_B = " In symbols, z = (x\u0304 \u2212 \u03bc)/(s/\u221an)."
_FORMULA_LATEX_A = r" \text{In symbols, } z = \frac{\bar{x} - \mu}{\sigma/\sqrt{n}}\text{.}"
_FORMULA_LATEX_B = r" \text{In symbols, } z = \frac{\bar{x} - \mu}{s/\sqrt{n}}\text{.}"


def _row_a_latex(base_text: str) -> str:
    return _greekify(latex_prompt(base_text)) + _FORMULA_LATEX_A


def _row_b_latex(base_text: str) -> str:
    return _greekify(latex_prompt(base_text)) + _FORMULA_LATEX_B


# ---------------------------------------------------------------------------
# Populations shared by both rows' easy/hard(/medium, row B) tiers -- a
# variety of distributions, each with a genuine (mu, sigma) and a real
# `draw(n)`. Row A's easy/hard tiers state BOTH mu and sigma; row B's tiers
# (and row A's medium/exam tiers) only ever use `mu` -- row B never states or
# prints a population sigma at any tier.
# ---------------------------------------------------------------------------

_INDICATOR_QS = [(0.5, 0.5), (0.2, 0.4), (0.8, 0.4), (0.1, 0.3), (0.9, 0.3)]
_EXPONENTIAL_MEANS = [4, 5, 6, 8, 10, 12]
_NORMAL_PARAMS = [(40, 6), (60, 8), (25, 4), (80, 10), (50, 5)]


def _draw_population(rng):
    """Pick one of a variety of populations; return (mu, sigma, draw_fn,
    clause) where draw_fn(n) draws n iid values as a plain list."""
    kind = rng.choice(("uniform", "indicator", "exponential", "normal"))
    if kind == "uniform":
        mu, sigma = 4.0, 2.0
        draw_fn = lambda n: [rng.randint(1, 7) for _ in range(n)]
        clause = "a population where each value is a whole number chosen uniformly from 1 to 7"
    elif kind == "indicator":
        q, sigma = rng.choice(_INDICATOR_QS)
        mu = q
        draw_fn = lambda n: [1 if rng.random() < q else 0 for _ in range(n)]
        clause = f"a population whose values are 0 or 1, with P(value = 1) = {_fmt_num(q)}"
    elif kind == "exponential":
        m = rng.choice(_EXPONENTIAL_MEANS)
        mu = sigma = float(m)
        draw_fn = lambda n: [rng.expovariate(1.0 / m) for _ in range(n)]
        clause = "an exponential population of waiting times, in minutes"
    else:
        mu, sigma = rng.choice(_NORMAL_PARAMS)
        mu, sigma = float(mu), float(sigma)
        draw_fn = lambda n: [rng.gauss(mu, sigma) for _ in range(n)]
        clause = "a normal population"
    return mu, sigma, draw_fn, clause


def _draw_sample_means(rng, draw_fn, n, k):
    return [sum(draw_fn(n)) / n for _ in range(k)]


def _sample_mean_and_sd(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


# ---------------------------------------------------------------------------
# Row A: p16b_xbar_sigma_standardise (ACMSM138), sigma known
# ---------------------------------------------------------------------------

_INTRO_A_STATED = [
    "{k} random samples, each of size {n}, were drawn from {clause}. "
    "The population has mean \u03bc = {mu} and standard deviation \u03c3 = {sigma}.",
    "A simulation drew {k} independent random samples of size {n} from {clause}, "
    "which has mean \u03bc = {mu} and standard deviation \u03c3 = {sigma}.",
    "{k} samples of size {n} were randomly and independently drawn from {clause}; "
    "the population's mean is \u03bc = {mu} and its standard deviation is \u03c3 = {sigma}.",
]

_INSTRUCTION_A = [
    "Calculate z, the standardised value of sample {j}'s mean, equal to the sample mean "
    "minus \u03bc, divided by \u03c3 divided by the square root of n, to 2 decimal places.",
    "Standardise sample {j}'s mean: subtract \u03bc, then divide by \u03c3 divided by the "
    "square root of n, giving your answer to 2 decimal places.",
    "Find the standardised value of sample {j}'s mean using (sample mean minus \u03bc) divided "
    "by (\u03c3 divided by the square root of n), to 2 decimal places.",
]


def _means_line(means):
    return (
        "The recorded sample means (reported to 2 decimal places) were "
        + ", ".join(f"sample {i}: {_disp2(m)}" for i, m in enumerate(means, start=1))
        + "."
    )


def _row_a_easy(rng):
    for _attempt in range(500):
        mu, sigma, draw_fn, clause = _draw_population(rng)
        k = rng.randint(5, 8)
        n = rng.randint(20, 60)
        means = _draw_sample_means(rng, draw_fn, n, k)
        j = rng.randint(1, k)
        xbar = round(means[j - 1], 2)
        se = sigma / math.sqrt(n)
        z_raw = (xbar - mu) / se
        if not rounds_unambiguously(z_raw, DP, margin=0.1):
            continue

        mu_d, sigma_d, n_d = _pdec(mu), _pdec(sigma), Decimal(n)
        sqrt_n_d = n_d.sqrt()
        se_d = sigma_d / sqrt_n_d
        xbar_d = _pdec(xbar)
        z_key_d = (xbar_d - mu_d) / se_d

        omits_sqrt_n = (xbar_d - mu_d) / sigma_d
        divides_by_n = (xbar_d - mu_d) / (sigma_d / n_d)
        candidates = [omits_sqrt_n, divides_by_n, _sign_flip_or_alt(rng, xbar_d, mu_d, se_d, sigma_d, sqrt_n_d)]
        if sigma != 1:
            candidates.append((xbar_d - mu_d) / ((sigma_d**2) / sqrt_n_d))  # uses-variance

        answer_text, distractors = _finalize(z_key_d, candidates)
        if len(distractors) < 2:
            continue
        _verify_z(xbar, mu, sigma, n, float(z_key_d))
        break
    else:
        raise RuntimeError("p16b_xbar_sigma_standardise easy: no draw converged")

    intro = phrase(rng, _INTRO_A_STATED, k=k, n=n, clause=clause, mu=_fmt_num(mu), sigma=_fmt_num(sigma))
    instruction = phrase(rng, _INSTRUCTION_A, j=j)
    base_text = f"{intro} {_means_line(means)} {instruction}"
    text = base_text + _SYMBOLIC_A
    latex = _row_a_latex(base_text)

    working = [
        f"Standard error = sigma / sqrt(n) = {_fmt_num(sigma)} / sqrt({n}) = {se:.4f}.",
        f"z = ({xbar} - {_fmt_num(mu)}) / {se:.4f} = {answer_text}.",
    ]
    return {
        "text": text,
        "latex": latex,
        "answer": numeric_answer(Fraction(answer_text)),
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "working": working,
    }


_PMF_TABLES = [
    ([1, 2, 3], [Fraction(1, 5), Fraction(2, 5), Fraction(2, 5)]),
    ([2, 4, 6], [Fraction(1, 5), Fraction(1, 2), Fraction(3, 10)]),
    ([0, 1, 2, 3], [Fraction(1, 10), Fraction(3, 10), Fraction(3, 10), Fraction(3, 10)]),
    ([1, 3, 5, 7], [Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)]),
    ([2, 5, 8, 11, 14], [Fraction(1, 10), Fraction(1, 5), Fraction(2, 5), Fraction(1, 5), Fraction(1, 10)]),
]
for _xs, _ps in _PMF_TABLES:
    assert sum(_ps) == 1, _xs

_INTRO_A_PMF = [
    "A population X has the distribution: {table}. {k} random samples, each of size {n}, "
    "were drawn from this population.",
    "A simulation drew {k} independent random samples, each of size {n}, from the "
    "population X with distribution: {table}.",
    "{k} samples of size {n} were randomly and independently drawn from a population X "
    "given by the distribution: {table}.",
]

_INSTRUCTION_A_PMF = [
    "First find \u03bc and \u03c3, the mean and standard deviation of X. Then calculate z, the "
    "standardised value of sample {j}'s mean, equal to the sample mean minus \u03bc, divided by "
    "\u03c3 divided by the square root of n, to 2 decimal places.",
    "Derive \u03bc and \u03c3, the population mean and standard deviation of X, then standardise "
    "sample {j}'s mean: subtract \u03bc, and divide by \u03c3 divided by the square root of n, "
    "giving your answer to 2 decimal places.",
]


def _table_clause(xs, ps):
    return "; ".join(f"P(X = {x}) = {p.numerator}/{p.denominator}" for x, p in zip(xs, ps))


def _row_a_medium(rng):
    for _attempt in range(500):
        xs, ps = rng.choice(_PMF_TABLES)
        mu_frac = sum(Fraction(x) * p for x, p in zip(xs, ps))
        var_frac = sum(Fraction(x * x) * p for x, p in zip(xs, ps)) - mu_frac * mu_frac
        mu_f, sigma = float(mu_frac), math.sqrt(float(var_frac))
        weights = [float(p) for p in ps]

        k = rng.randint(5, 8)
        n = rng.randint(20, 60)
        means = [sum(rng.choices(xs, weights=weights, k=n)) / n for _ in range(k)]
        j = rng.randint(1, k)
        xbar = round(means[j - 1], 2)
        se = sigma / math.sqrt(n)
        z_raw = (xbar - mu_f) / se
        if not rounds_unambiguously(z_raw, DP, margin=0.1):
            continue

        mu_d = Decimal(mu_frac.numerator) / Decimal(mu_frac.denominator)
        var_d = Decimal(var_frac.numerator) / Decimal(var_frac.denominator)
        sigma_d = var_d.sqrt()
        n_d = Decimal(n)
        sqrt_n_d = n_d.sqrt()
        se_d = sigma_d / sqrt_n_d
        xbar_d = _pdec(xbar)
        z_key_d = (xbar_d - mu_d) / se_d

        omits_sqrt_n = (xbar_d - mu_d) / sigma_d
        divides_by_n = (xbar_d - mu_d) / (sigma_d / n_d)
        candidates = [omits_sqrt_n, divides_by_n, _sign_flip_or_alt(rng, xbar_d, mu_d, se_d, sigma_d, sqrt_n_d)]
        if var_frac != 0 and abs(float(var_frac) - sigma) > 1e-9:
            uses_variance_as_sd = (xbar_d - mu_d) / (var_d / sqrt_n_d)
            candidates.append(uses_variance_as_sd)

        answer_text, distractors = _finalize(z_key_d, candidates)
        if len(distractors) < 2:
            continue
        _verify_z(xbar, mu_f, sigma, n, float(z_key_d))
        break
    else:
        raise RuntimeError("p16b_xbar_sigma_standardise medium: no draw converged")

    intro = phrase(rng, _INTRO_A_PMF, k=k, n=n, table=_table_clause(xs, ps))
    instruction = phrase(rng, _INSTRUCTION_A_PMF, j=j)
    base_text = f"{intro} {_means_line(means)} {instruction}"
    text = base_text + _SYMBOLIC_A
    latex = _row_a_latex(base_text)

    working = [
        f"mu = sum of x times P(X = x) = {mu_f:.4f}. "
        f"sigma = square root of (sum of x squared times P(X = x), minus mu squared) = {sigma:.4f}.",
        f"Standard error = sigma / square root of {n} = {se:.4f}.",
        f"z = ({xbar} - {mu_f:.4f}) / {se:.4f} = {answer_text}.",
    ]
    return {
        "text": text,
        "latex": latex,
        "answer": numeric_answer(Fraction(answer_text)),
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "working": working,
    }


_HARD_N_PAIRS = [(25, 100), (20, 80), (16, 64), (25, 225)]

_INTRO_A_HARD = [
    "{k} random samples were drawn from {clause}, which has mean \u03bc = {mu} and standard "
    "deviation \u03c3 = {sigma}: samples 1 to 4 of size {n1}, and samples 5 to 8 of size {n2}.",
    "A simulation drew {k} independent random samples from {clause} (mean \u03bc = {mu}, "
    "standard deviation \u03c3 = {sigma}): samples 1 to 4 of size {n1} and samples 5 to 8 of "
    "size {n2}.",
]

_INSTRUCTION_A_HARD = [
    "Which of sample {i} and sample {j} has the standardised value further from 0? Give "
    "that value, with its sign, to 2 decimal places.",
    "Between sample {i} and sample {j}, one has a standardised value further from 0 than "
    "the other. Give that value, with its sign, to 2 decimal places.",
]


def _row_a_hard(rng):
    want_trap = rng.random() < 0.5
    for _resim in range(400):
        mu, sigma, draw_fn, clause = _draw_population(rng)
        n1, n2 = rng.choice(_HARD_N_PAIRS)
        means1 = _draw_sample_means(rng, draw_fn, n1, 4)
        means2 = _draw_sample_means(rng, draw_fn, n2, 4)
        se1, se2 = sigma / math.sqrt(n1), sigma / math.sqrt(n2)

        # n1 < n2 always (_HARD_N_PAIRS), and samples 1-4 are always the n1
        # group -- so "the larger raw deviation belongs to the smaller n"
        # is genuinely one-directional here (unlike row B, same n for all).
        hits = []
        for ii in range(1, 5):
            for jj in range(5, 9):
                xbar_i = round(means1[ii - 1], 2)
                xbar_j = round(means2[jj - 5], 2)
                z_i = (xbar_i - mu) / se1
                z_j = (xbar_j - mu) / se2
                raw_i, raw_j = abs(xbar_i - mu), abs(xbar_j - mu)
                if raw_i == raw_j:
                    continue
                is_trap = (raw_i > raw_j) and (abs(z_i) < abs(z_j))
                if is_trap != want_trap:
                    continue
                gap = abs(abs(z_i) - abs(z_j))
                if gap < 0.05:
                    continue
                z_key_raw = z_i if abs(z_i) > abs(z_j) else z_j
                if not rounds_unambiguously(z_key_raw, DP, margin=0.1):
                    continue
                hits.append((ii, jj, xbar_i, xbar_j))
        if not hits:
            continue
        i, jsamp, xbar_i, xbar_j = rng.choice(hits)

        mu_d, sigma_d = _pdec(mu), _pdec(sigma)
        n1_d, n2_d = Decimal(n1), Decimal(n2)
        se1_d, se2_d = sigma_d / n1_d.sqrt(), sigma_d / n2_d.sqrt()
        xbar_i_d, xbar_j_d = _pdec(xbar_i), _pdec(xbar_j)
        z_i_d = (xbar_i_d - mu_d) / se1_d
        z_j_d = (xbar_j_d - mu_d) / se2_d
        further_is_i = abs(z_i_d) > abs(z_j_d)
        z_key_d = z_i_d if further_is_i else z_j_d
        other_z_d = z_j_d if further_is_i else z_i_d
        n_key_d = n1_d if further_is_i else n2_d
        se_key_d = se1_d if further_is_i else se2_d
        xbar_key_d = xbar_i_d if further_is_i else xbar_j_d
        n_key = n1 if further_is_i else n2

        omits_sqrt_n = (xbar_key_d - mu_d) / sigma_d
        divides_by_n = (xbar_key_d - mu_d) / (sigma_d / n_key_d)
        candidates = [
            other_z_d,
            omits_sqrt_n,
            divides_by_n,
            _sign_flip_or_alt(rng, xbar_key_d, mu_d, se_key_d, sigma_d, n_key_d.sqrt()),
        ]

        answer_text, distractors = _finalize(z_key_d, candidates)
        if len(distractors) < 2:
            continue
        _verify_z(float(xbar_key_d), mu, sigma, n_key, float(z_key_d))
        break
    else:
        raise RuntimeError("p16b_xbar_sigma_standardise hard: no draw converged")

    intro = phrase(rng, _INTRO_A_HARD, k=8, n1=n1, n2=n2, clause=clause, mu=_fmt_num(mu), sigma=_fmt_num(sigma))
    means_line = (
        "The recorded sample means (reported to 2 decimal places) were "
        + ", ".join(f"sample {t}: {_disp2(m)}" for t, m in enumerate(means1, start=1))
        + ", "
        + ", ".join(f"sample {t}: {_disp2(m)}" for t, m in enumerate(means2, start=5))
        + "."
    )
    instruction = phrase(rng, _INSTRUCTION_A_HARD, i=i, j=jsamp)
    base_text = f"{intro} {means_line} {instruction}"
    text = base_text + _SYMBOLIC_A
    latex = _row_a_latex(base_text)

    working = [
        f"z(sample {i}) = ({xbar_i} - {_fmt_num(mu)}) / (sigma/sqrt({n1})) = {float(z_i_d):.4f}.",
        f"z(sample {jsamp}) = ({xbar_j} - {_fmt_num(mu)}) / (sigma/sqrt({n2})) = {float(z_j_d):.4f}.",
        f"The further-from-0 value is {answer_text}.",
    ]
    return {
        "text": text,
        "latex": latex,
        "answer": numeric_answer(Fraction(answer_text)),
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "working": working,
    }


_BINOMIAL_PARAMS = [(10, 0.5), (20, 0.1), (15, 0.2), (8, 0.25), (12, 0.75)]

_INTRO_A_EXAM = [
    "A simulation drew {k} independent random samples, each of size {n}, from the number "
    "of successes in {m} independent trials with success probability {q} per trial.",
    "{k} random samples, each of size {n}, were drawn from the number of successes in {m} "
    "independent trials with success probability {q} per trial.",
]

_INSTRUCTION_A_EXAM = [
    "Calculate z, the standardised value of sample {j}'s mean, equal to the sample mean "
    "minus \u03bc, divided by \u03c3 divided by the square root of n, to 2 decimal places, where "
    "\u03bc and \u03c3 are the mean and standard deviation of this binomial count.",
    "Standardise sample {j}'s mean, giving your answer to 2 decimal places: subtract \u03bc, "
    "and divide by \u03c3 divided by the square root of n, where \u03bc and \u03c3 are the "
    "mean and standard deviation of this binomial count.",
]


def _row_a_exam(rng):
    for _attempt in range(500):
        m, q = rng.choice(_BINOMIAL_PARAMS)
        mu = m * q
        sigma = math.sqrt(m * q * (1 - q))
        k = rng.randint(5, 8)
        n = rng.randint(25, 60)
        means = [sum(rng.binomialvariate(m, q) for _ in range(n)) / n for _ in range(k)]
        j = rng.randint(1, k)
        xbar = round(means[j - 1], 2)
        se = sigma / math.sqrt(n)
        z_raw = (xbar - mu) / se
        if not rounds_unambiguously(z_raw, DP, margin=0.1):
            continue

        m_d = Decimal(m)
        q_d = _pdec(q)
        mu_d = m_d * q_d
        sigma_d = (m_d * q_d * (Decimal(1) - q_d)).sqrt()
        n_d = Decimal(n)
        sqrt_n_d = n_d.sqrt()
        se_d = sigma_d / sqrt_n_d
        xbar_d = _pdec(xbar)
        z_key_d = (xbar_d - mu_d) / se_d

        omits_sqrt_n = (xbar_d - mu_d) / sigma_d
        divides_by_n = (xbar_d - mu_d) / (sigma_d / n_d)
        candidates = [omits_sqrt_n, divides_by_n, _sign_flip_or_alt(rng, xbar_d, mu_d, se_d, sigma_d, sqrt_n_d)]

        answer_text, distractors = _finalize(z_key_d, candidates)
        if len(distractors) < 2:
            continue
        _verify_z(xbar, mu, sigma, n, float(z_key_d))
        break
    else:
        raise RuntimeError("p16b_xbar_sigma_standardise exam: no draw converged")

    intro = phrase(rng, _INTRO_A_EXAM, k=k, n=n, m=m, q=_fmt_num(q))
    instruction = phrase(rng, _INSTRUCTION_A_EXAM, j=j)
    base_text = f"{intro} {_means_line(means)} {instruction}"
    text = base_text + _SYMBOLIC_A
    latex = _row_a_latex(base_text)

    working = [
        f"mu = m times q = {m} times {_fmt_num(q)} = {mu:.4f}. "
        f"sigma = square root of (m times q times (1 - q)) = {sigma:.4f}.",
        f"z = ({xbar} - {mu:.4f}) / (sigma / square root of {n}) = {answer_text}.",
    ]
    return {
        "text": text,
        "latex": latex,
        "answer": numeric_answer(Fraction(answer_text)),
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "working": working,
    }


@generator(
    "p16b_xbar_sigma_standardise",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p16b_xbar_sigma_standardise(rng, tier="easy"):
    """Simulate repeated random sampling from a variety of distributions;
    standardise a stated sample's mean using the KNOWN population sigma
    (ACMSM138)."""
    if tier == "easy":
        return _row_a_easy(rng)
    if tier == "medium":
        return _row_a_medium(rng)
    if tier == "hard":
        return _row_a_hard(rng)
    if tier == "exam":
        return _row_a_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Row B: p16b_xbar_s_standardise (ACMSM139), s estimated. n >= 30 ALWAYS.
# ---------------------------------------------------------------------------

_ROW_B_N_RANGE = range(30, 81)

_INTRO_B_STATED = [
    "{k} random samples, each of size {n}, were drawn from {clause}. The population has mean \u03bc = {mu}.",
    "A simulation drew {k} independent random samples of size {n} from {clause}, which has mean \u03bc = {mu}.",
    "{k} samples of size {n} were randomly and independently drawn from {clause}; the "
    "population's mean is \u03bc = {mu}.",
]


def _row_b_easy(rng):
    for _attempt in range(500):
        mu, _sigma_unused, draw_fn, clause = _draw_population(rng)
        k = rng.randint(5, 8)
        n = rng.choice(_ROW_B_N_RANGE)
        samples = [draw_fn(n) for _ in range(k)]
        stats = [_sample_mean_and_sd(vals) for vals in samples]
        j = rng.randint(1, k)
        xbar_raw, s_raw = stats[j - 1]
        xbar, s = round(xbar_raw, 2), round(s_raw, 2)
        if s == 0:
            continue
        se = s / math.sqrt(n)
        z_raw = (xbar - mu) / se
        if not rounds_unambiguously(z_raw, DP, margin=0.1):
            continue

        mu_d, s_d, n_d = _pdec(mu), _pdec(s), Decimal(n)
        sqrt_n_d = n_d.sqrt()
        se_d = s_d / sqrt_n_d
        xbar_d = _pdec(xbar)
        z_key_d = (xbar_d - mu_d) / se_d

        omits_sqrt_n = (xbar_d - mu_d) / s_d
        divides_by_n = (xbar_d - mu_d) / (s_d / n_d)
        candidates = [omits_sqrt_n, divides_by_n, _sign_flip_or_alt(rng, xbar_d, mu_d, se_d, s_d, sqrt_n_d)]

        answer_text, distractors = _finalize(z_key_d, candidates)
        if len(distractors) < 2:
            continue
        _verify_z(xbar, mu, s, n, float(z_key_d))
        break
    else:
        raise RuntimeError("p16b_xbar_s_standardise easy: no draw converged")

    stats_r = [(round(m, 2), round(sd, 2)) for m, sd in stats]
    stats_line = (
        "For each sample (n = " + str(n) + "), the simulation recorded the sample "
        "mean and the sample standard deviation s (n - 1 divisor), reported to 2 "
        "decimal places: "
        + "; ".join(f"sample {t}: mean = {_disp2(m)}, s = {_disp2(sd)}" for t, (m, sd) in enumerate(stats_r, start=1))
        + "."
    )
    intro = phrase(rng, _INTRO_B_STATED, k=k, n=n, clause=clause, mu=_fmt_num(mu))
    instruction = (
        f"Using sample {j}'s OWN sample standard deviation s, calculate z, the standardised "
        f"value of sample {j}'s mean, equal to the sample mean minus \u03bc, divided by s "
        f"divided by the square root of n, to 2 decimal places."
    )
    base_text = f"{intro} {stats_line} {instruction}"
    text = base_text + _SYMBOLIC_B
    latex = _row_b_latex(base_text)

    working = [
        f"Standard error = s / sqrt(n) = {s} / sqrt({n}) = {se:.4f}.",
        f"z = ({xbar} - {_fmt_num(mu)}) / {se:.4f} = {answer_text}.",
    ]
    return {
        "text": text,
        "latex": latex,
        "answer": numeric_answer(Fraction(answer_text)),
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "working": working,
    }


def _row_b_medium(rng):
    for _attempt in range(500):
        mu, _sigma_unused, draw_fn, clause = _draw_population(rng)
        k = rng.randint(5, 8)
        n = rng.choice(_ROW_B_N_RANGE)
        samples = [draw_fn(n) for _ in range(k)]
        stats = [_sample_mean_and_sd(vals) for vals in samples]
        j = rng.randint(1, k)
        xbar_raw, s_raw = stats[j - 1]
        xbar, s_rounded = round(xbar_raw, 2), round(s_raw, 2)
        variance_displayed = round(s_rounded * s_rounded, 2)
        if variance_displayed <= 0:
            continue
        # The learner never sees `s_rounded` -- only the PRINTED variance -- so every downstream
        # computation reads from that displayed value, never the internal `s_rounded` that produced
        # it.
        s = math.sqrt(variance_displayed)
        se = s / math.sqrt(n)
        z_raw = (xbar - mu) / se
        if not rounds_unambiguously(z_raw, DP, margin=0.1):
            continue

        mu_d = _pdec(mu)
        variance_d = _pdec(variance_displayed)
        s_d = variance_d.sqrt()
        n_d = Decimal(n)
        sqrt_n_d = n_d.sqrt()
        se_d = s_d / sqrt_n_d
        xbar_d = _pdec(xbar)
        z_key_d = (xbar_d - mu_d) / se_d

        omits_sqrt_n = (xbar_d - mu_d) / s_d
        divides_by_n = (xbar_d - mu_d) / (s_d / n_d)
        candidates = [omits_sqrt_n, divides_by_n, _sign_flip_or_alt(rng, xbar_d, mu_d, se_d, s_d, sqrt_n_d)]
        if variance_displayed != 1:
            variance_as_sd = (xbar_d - mu_d) / (variance_d / sqrt_n_d)
            candidates.append(variance_as_sd)

        answer_text, distractors = _finalize(z_key_d, candidates)
        if len(distractors) < 2:
            continue
        _verify_z(xbar, mu, s, n, float(z_key_d))
        break
    else:
        raise RuntimeError("p16b_xbar_s_standardise medium: no draw converged")

    stats_r = [(round(m, 2), round(sd, 2)) for m, sd in stats]
    variance_r = [(m, round(sd * sd, 2)) for m, sd in stats_r]
    stats_line = (
        "For each sample (n = " + str(n) + "), the simulation recorded the sample "
        "mean and the sample VARIANCE (n - 1 divisor), reported to 2 decimal places: "
        + "; ".join(
            f"sample {t}: mean = {_disp2(m)}, variance = {_disp2(v)}" for t, (m, v) in enumerate(variance_r, start=1)
        )
        + "."
    )
    intro = phrase(rng, _INTRO_B_STATED, k=k, n=n, clause=clause, mu=_fmt_num(mu))
    instruction = (
        f"Take the square root of sample {j}'s own sample variance to get its sample "
        f"standard deviation s, then calculate z, the standardised value of sample {j}'s "
        f"mean, equal to the sample mean minus \u03bc, divided by s divided by the square root "
        f"of n, to 2 decimal places."
    )
    base_text = f"{intro} {stats_line} {instruction}"
    text = base_text + _SYMBOLIC_B
    latex = _row_b_latex(base_text)

    working = [
        f"s = sqrt({round(s * s, 2)}) = {s:.4f}.",
        f"Standard error = s / sqrt(n) = {s:.4f} / sqrt({n}) = {se:.4f}.",
        f"z = ({xbar} - {_fmt_num(mu)}) / {se:.4f} = {answer_text}.",
    ]
    return {
        "text": text,
        "latex": latex,
        "answer": numeric_answer(Fraction(answer_text)),
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "working": working,
    }


_INSTRUCTION_B_HARD = [
    "Which of sample {i} and sample {j} has the standardised value further from 0? Give "
    "that value, with its sign, to 2 decimal places.",
    "Between sample {i} and sample {j}, one has a standardised value further from 0 than "
    "the other. Give that value, with its sign, to 2 decimal places.",
]


def _row_b_hard(rng):
    want_trap = rng.random() < 0.5
    for _resim in range(400):
        mu, _sigma_unused, draw_fn, clause = _draw_population(rng)
        n = rng.choice(_ROW_B_N_RANGE)
        k = rng.randint(5, 8)
        samples = [draw_fn(n) for _ in range(k)]
        stats = [_sample_mean_and_sd(vals) for vals in samples]
        rounded = [(round(m, 2), round(sd, 2)) for m, sd in stats]

        # Same n for every sample here (unlike row A), so which sample has
        # the larger raw deviation is not fixed to a direction -- the trap
        # test must be symmetric.
        hits = []
        for a in range(1, k + 1):
            for b in range(a + 1, k + 1):
                xbar_a, s_a = rounded[a - 1]
                xbar_b, s_b = rounded[b - 1]
                if s_a == 0 or s_b == 0:
                    continue
                se_a, se_b = s_a / math.sqrt(n), s_b / math.sqrt(n)
                z_a, z_b = (xbar_a - mu) / se_a, (xbar_b - mu) / se_b
                raw_a, raw_b = abs(xbar_a - mu), abs(xbar_b - mu)
                if raw_a == raw_b:
                    continue
                is_trap = (raw_a > raw_b) != (abs(z_a) > abs(z_b))
                if is_trap != want_trap:
                    continue
                gap = abs(abs(z_a) - abs(z_b))
                if gap < 0.05:
                    continue
                z_key_raw = z_a if abs(z_a) > abs(z_b) else z_b
                if not rounds_unambiguously(z_key_raw, DP, margin=0.1):
                    continue
                hits.append((a, b, xbar_a, s_a, xbar_b, s_b))
        if not hits:
            continue
        i, jsamp, xbar_i, s_i, xbar_j, s_j = rng.choice(hits)

        mu_d = _pdec(mu)
        n_d = Decimal(n)
        sqrt_n_d = n_d.sqrt()
        xbar_i_d, s_i_d = _pdec(xbar_i), _pdec(s_i)
        xbar_j_d, s_j_d = _pdec(xbar_j), _pdec(s_j)
        se_i_d, se_j_d = s_i_d / sqrt_n_d, s_j_d / sqrt_n_d
        z_i_d = (xbar_i_d - mu_d) / se_i_d
        z_j_d = (xbar_j_d - mu_d) / se_j_d
        further_is_i = abs(z_i_d) > abs(z_j_d)
        z_key_d = z_i_d if further_is_i else z_j_d
        other_z_d = z_j_d if further_is_i else z_i_d
        xbar_key_d = xbar_i_d if further_is_i else xbar_j_d
        s_key_d = s_i_d if further_is_i else s_j_d
        se_key_d = se_i_d if further_is_i else se_j_d

        omits_sqrt_n = (xbar_key_d - mu_d) / s_key_d
        divides_by_n = (xbar_key_d - mu_d) / (s_key_d / n_d)
        candidates = [
            other_z_d,
            omits_sqrt_n,
            divides_by_n,
            _sign_flip_or_alt(rng, xbar_key_d, mu_d, se_key_d, s_key_d, sqrt_n_d),
        ]

        answer_text, distractors = _finalize(z_key_d, candidates)
        if len(distractors) < 2:
            continue
        _verify_z(float(xbar_key_d), mu, float(s_key_d), n, float(z_key_d))
        break
    else:
        raise RuntimeError("p16b_xbar_s_standardise hard: no draw converged")

    stats_line = (
        "For each sample (n = " + str(n) + "), the simulation recorded the sample "
        "mean and the sample standard deviation s (n - 1 divisor), reported to 2 "
        "decimal places: "
        + "; ".join(f"sample {t}: mean = {_disp2(m)}, s = {_disp2(sd)}" for t, (m, sd) in enumerate(rounded, start=1))
        + "."
    )
    intro = phrase(rng, _INTRO_B_STATED, k=k, n=n, clause=clause, mu=_fmt_num(mu))
    instruction = phrase(rng, _INSTRUCTION_B_HARD, i=i, j=jsamp)
    base_text = f"{intro} {stats_line} {instruction}"
    text = base_text + _SYMBOLIC_B
    latex = _row_b_latex(base_text)

    working = [
        f"z(sample {i}) = ({xbar_i} - {_fmt_num(mu)}) / ({s_i}/sqrt({n})) = {float(z_i_d):.4f}.",
        f"z(sample {jsamp}) = ({xbar_j} - {_fmt_num(mu)}) / ({s_j}/sqrt({n})) = {float(z_j_d):.4f}.",
        f"The further-from-0 value is {answer_text}.",
    ]
    return {
        "text": text,
        "latex": latex,
        "answer": numeric_answer(Fraction(answer_text)),
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "working": working,
    }


def _row_b_exam(rng):
    for _attempt in range(500):
        mu, _sigma_unused, draw_fn, clause = _draw_population(rng)
        k = rng.randint(5, 8)
        n = rng.choice(_ROW_B_N_RANGE)
        samples = [draw_fn(n) for _ in range(k)]
        j = rng.randint(1, k)
        vals = samples[j - 1]
        Sx_r = round(sum(vals), 2)
        Sxx_r = round(sum(v * v for v in vals), 2)
        xbar_calc = Sx_r / n
        var_calc = (Sxx_r - (Sx_r**2) / n) / (n - 1)
        if var_calc <= 0:
            continue
        s_calc = math.sqrt(var_calc)
        se = s_calc / math.sqrt(n)
        z_raw = (xbar_calc - mu) / se
        if not rounds_unambiguously(z_raw, DP, margin=0.1):
            continue

        mu_d = _pdec(mu)
        n_d = Decimal(n)
        Sx_d = _pdec(Sx_r)
        Sxx_d = _pdec(Sxx_r)
        xbar_d = Sx_d / n_d
        var_d = (Sxx_d - (Sx_d**2) / n_d) / (n_d - 1)
        s_d = var_d.sqrt()
        sqrt_n_d = n_d.sqrt()
        se_d = s_d / sqrt_n_d
        z_key_d = (xbar_d - mu_d) / se_d

        var_pop_d = (Sxx_d - (Sx_d**2) / n_d) / n_d
        omits_sqrt_n = (xbar_d - mu_d) / s_d
        divides_by_n = (xbar_d - mu_d) / (s_d / n_d)
        candidates = [omits_sqrt_n, divides_by_n, _sign_flip_or_alt(rng, xbar_d, mu_d, se_d, s_d, sqrt_n_d)]
        if var_pop_d > 0:
            s_pop_d = var_pop_d.sqrt()
            population_divisor = (xbar_d - mu_d) / (s_pop_d / sqrt_n_d)
            candidates.append(population_divisor)

        answer_text, distractors = _finalize(z_key_d, candidates)
        if len(distractors) < 2:
            continue
        _verify_z(xbar_calc, mu, s_calc, n, float(z_key_d))
        break
    else:
        raise RuntimeError("p16b_xbar_s_standardise exam: no draw converged")

    intro = phrase(rng, _INTRO_B_STATED, k=k, n=n, clause=clause, mu=_fmt_num(mu))
    summary_line = (
        f"For sample {j}, the simulation recorded: n = {n}, the sum of the observations = "
        f"{Sx_r}, and the sum of the squares of the observations = {Sxx_r} (s is the sample "
        f"standard deviation, with divisor n - 1)."
    )
    instruction = (
        "Find sample "
        f"{j}'s own sample mean and sample standard deviation s from these summaries, then "
        "calculate z, the standardised value of sample "
        f"{j}'s mean, equal to the sample mean minus \u03bc, divided by s divided by the square "
        "root of n, to 2 decimal places."
    )
    base_text = f"{intro} {summary_line} {instruction}"
    text = base_text + _SYMBOLIC_B
    latex = _row_b_latex(base_text)

    working = [
        f"x-bar = Sx / n = {Sx_r} / {n} = {xbar_calc:.4f}.",
        f"s = sqrt((Sxx - Sx^2/n) / (n - 1)) = {s_calc:.4f}.",
        f"z = ({xbar_calc:.4f} - {_fmt_num(mu)}) / ({s_calc:.4f}/sqrt({n})) = {answer_text}.",
    ]
    return {
        "text": text,
        "latex": latex,
        "answer": numeric_answer(Fraction(answer_text)),
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "working": working,
    }


@generator(
    "p16b_xbar_s_standardise",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p16b_xbar_s_standardise(rng, tier="easy"):
    """Simulate repeated random sampling from a variety of distributions;
    standardise a stated sample's mean using that SAMPLE's own s, n >= 30
    (ACMSM139)."""
    if tier == "easy":
        return _row_b_easy(rng)
    if tier == "medium":
        return _row_b_medium(rng)
    if tier == "hard":
        return _row_b_hard(rng)
    if tier == "exam":
        return _row_b_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
