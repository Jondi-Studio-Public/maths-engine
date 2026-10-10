"""Inverse proportion confidence interval: recover the sample from a stated
normal-approximation interval, never the other direction.

`ucv1-proportion-confidence-interval` is narrowed to the INVERSE
direction: given a stated normal-approximation confidence interval `(L, U)`
for a Bernoulli proportion and its stated critical value `z`, recover the
sample: p_hat = (L+U)/2, margin = (U-L)/2, n = z^2 . p_hat . (1-p_hat) /
margin^2, k = p_hat . n. The learner NEVER computes the forward interval from
n and k at any tier, and no tier's answer is ever an interval bound -- that
is the shipping neighbour ACMMM178
(`senior-use-the-approximate-confidence-interval-left-widehat-p`).

Every printed bound is rounded to 4 decimal places with
`stated_precision_string`, and every KEY is derived from the PRINTED strings
(parsed back as exact `Fraction`s), never from the generator's own hidden
`n`/`k` -- a draw whose printed-bound-derived key disagrees with the hidden
truth is rejected and redrawn.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import (
    check,
    check_pairwise_distinct,
    generator,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)

# level% -> z to 3 d.p., printed identically everywhere (table and stated-z
# prompts must agree byte-for-byte).
_Z_TABLE = [
    (80, "1.282"),
    (90, "1.645"),
    (95, "1.960"),
    (98, "2.326"),
    (99, "2.576"),
]

_N_LO, _N_HI = 40, 400
_MAX_ATTEMPTS = 4000


def _draw_base(rng):
    """One (level, z, n, k, p_hat, L4, U4) draw, constructed to target and
    checked for plausibility (0 < L < U < 1, p_hat in [0.10, 0.90],
    k >= 5 and n-k >= 5). Does NOT check recoverability of n/k from the
    printed bounds -- callers add that per-tier."""
    for _ in range(_MAX_ATTEMPTS):
        level, z_str = rng.choice(_Z_TABLE)
        z = Fraction(z_str)
        n = rng.randint(_N_LO, _N_HI)
        k_lo = max(5, math.ceil(0.15 * n))
        k_hi = min(n - 5, math.floor(0.85 * n))
        if k_lo > k_hi:
            continue
        k = rng.randint(k_lo, k_hi)
        p_hat = Fraction(k, n)
        variance_term = float(p_hat * (1 - p_hat)) / n
        margin_float = float(z) * math.sqrt(variance_term)
        margin_exact = Fraction(margin_float)
        L_raw = p_hat - margin_exact
        U_raw = p_hat + margin_exact
        if not rounds_unambiguously(float(L_raw), 4):
            continue
        if not rounds_unambiguously(float(U_raw), 4):
            continue
        L4 = round_half_up(L_raw, 4)
        U4 = round_half_up(U_raw, 4)
        if not (Fraction(0) < L4 < U4 < Fraction(1)):
            continue
        if not (Fraction(1, 10) <= p_hat <= Fraction(9, 10)):
            continue
        return dict(level=level, z=z, n=n, k=k, p_hat=p_hat, L4=L4, U4=U4)
    check(False, "could not construct a plausible base draw")


def _art(level: int) -> str:
    """The indefinite article for a spoken confidence level -- "an 80%"
    (eighty, vowel sound) but "a 90%"/"a 95%"/"a 98%"/"a 99%" (ninety-...,
    consonant sound). Lowercase, for mid-sentence use."""
    return "an" if level == 80 else "a"


def _fmt4(value) -> str:
    """`value` (a `Fraction` or anything `Fraction`-able) rounded and
    displayed at exactly 4 d.p. -- for WORKING text only (a plain decimal,
    never a fraction, and never the graded canonical, which stores its own
    already-rounded `Fraction` directly)."""
    return stated_precision_string(round_half_up(Fraction(value), 4), 4)


def _z_text(z: Fraction) -> str:
    """z printed to 3 d.p., identically everywhere it appears (table rows
    and stated-z prompts must agree byte-for-byte, and the oracle parses
    this decimal form, never a raw fraction)."""
    return stated_precision_string(z, 3)


def _interval_text(L4: Fraction, U4: Fraction) -> str:
    return f"lower bound {stated_precision_string(L4, 4)} and upper bound {stated_precision_string(U4, 4)}"


def _interval_clause(rng, level, L4, U4) -> str:
    """Wording variety only: the same certified facts (level, both bounds), in the same order, with genuinely
    different clause structure. `_interval_text`'s literal substring is kept inside each template so
    `_BOUNDS_RE` in the oracle still matches."""
    L, U = stated_precision_string(L4, 4), stated_precision_string(U4, 4)
    art = _art(level)
    templates = [
        f"{art.capitalize()} {level}% confidence interval for a population "
        f"proportion has lower bound {L} and upper bound {U}.",
        f"A population-proportion confidence interval at the {level}% "
        f"confidence level has lower bound {L} and upper bound {U}.",
        f"At the {level}% confidence level, a confidence interval for a "
        f"population proportion is reported with lower bound {L} and upper "
        f"bound {U}.",
    ]
    return rng.choice(templates)


def _interval_with_z_clause(rng, level, L4, U4, z: Fraction) -> str:
    """Wording variety for medium/exam (interval + stated z), same
    technique as `_interval_clause`: literal `lower bound X and upper bound
    Y` and `z = Z` (Z always 3 d.p.) substrings kept verbatim in every
    template so the oracle's `_BOUNDS_RE`/`_Z_RE` still match."""
    L, U, z_str = stated_precision_string(L4, 4), stated_precision_string(U4, 4), _z_text(z)
    art = _art(level)
    templates = [
        f"A confidence interval for a population proportion has lower bound "
        f"{L} and upper bound {U}, and was built using z = {z_str} ({art} "
        f"{level}% confidence level).",
        f"A population-proportion confidence interval was built using z = "
        f"{z_str} ({art} {level}% confidence level); it has lower bound {L} "
        f"and upper bound {U}.",
        f"Using z = {z_str} ({art} {level}% confidence level), a confidence "
        f"interval for a population proportion was reported with lower "
        f"bound {L} and upper bound {U}.",
    ]
    return rng.choice(templates)


def _assert_key_clears_tolerance(true_pair, candidate_pairs, tolerances):
    """Every distractor pair differs from the true pair by more than
    `tolerances` in at least one component -- decimal components use
    `stated_precision_tolerance`, integer components use exact `!=`
    (`tolerances` entries of `Fraction(0)`, so any nonzero difference
    clears it, matching the grader's exact-integer-match behaviour)."""
    for pair in candidate_pairs:
        differs = any(abs(Fraction(t) - Fraction(c)) > tol for t, c, tol in zip(true_pair, pair, tolerances))
        check(
            differs,
            "a distractor sits within the pair grader's tolerance of the true answer in every component",
        )


def _plausible_nk(n, k) -> bool:
    return 2 <= n and 0 <= k <= n


# --- easy: p_hat and margin from both printed bounds ------------------

_EASY_INSTRUCTION = (
    "Give your answer in the form p_hat = ..., margin = ..., both to 4 "
    "decimal places, where p_hat is the midpoint of the interval and "
    "margin is its half-width."
)


def _make_easy(rng):
    for _ in range(_MAX_ATTEMPTS):
        base = _draw_base(rng)
        L4, U4, level = base["L4"], base["U4"], base["level"]
        L4i, U4i = int(L4 * 10000), int(U4 * 10000)
        if (L4i + U4i) % 2 != 0:
            continue
        p_hat_key = Fraction(L4i + U4i, 20000)
        margin_key = Fraction(U4i - L4i, 20000)
        if abs(p_hat_key - base["p_hat"]) > Fraction(1, 100):
            continue
        break
    else:
        check(False, "easy: could not construct an exact-4dp midpoint/half-width draw")

    p_str = stated_precision_string(p_hat_key, 4)
    m_str = stated_precision_string(margin_key, 4)
    text = f"{_interval_clause(rng, level, L4, U4)} {_EASY_INSTRUCTION}"
    answer = {"p_hat": p_hat_key, "margin": margin_key}
    answer_text = f"p_hat = {p_str}, margin = {m_str}"
    answer_latex = rf"\hat p = {p_str},\; \text{{margin}} = {m_str}"

    full_width = margin_key * 2
    lower_as_phat = Fraction(L4)
    upper_as_phat = Fraction(U4)
    candidates = [
        (p_hat_key, full_width),
        (lower_as_phat, margin_key),
        (upper_as_phat, margin_key),
    ]
    check_pairwise_distinct(
        (p_hat_key, margin_key),
        *candidates,
        msg="an easy-tier distractor collides with the true answer or another distractor",
    )
    tolerances = (stated_precision_tolerance(4), stated_precision_tolerance(4))
    _assert_key_clears_tolerance((p_hat_key, margin_key), candidates, tolerances)
    candidate_texts = [
        f"p_hat = {stated_precision_string(a, 4)}, margin = {stated_precision_string(b, 4)}" for a, b in candidates
    ]
    return dict(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_latex,
        working=[
            f"p_hat = (L + U) / 2 = {p_str}; margin = (U - L) / 2 = {m_str}.",
        ],
        distractors=pick_distractors(answer_text, candidate_texts, n=3, tolerance=stated_precision_tolerance(4)),
    )


# --- shared n/k recovery (medium, hard, exam) --------------------------


def _nk_formula_candidates(z: Fraction, p_hat_d: Fraction, margin_d: Fraction):
    """The four inversion misconceptions, as (n, k) Fraction pairs, in
    misconception-priority order. Filtered for plausibility by the caller."""
    full_width_margin = margin_d * 2
    out = []
    if full_width_margin > 0:
        n_fw = z * z * p_hat_d * (1 - p_hat_d) / (full_width_margin * full_width_margin)
        out.append((n_fw, p_hat_d * n_fw))
    n_missing_zsq = z * p_hat_d * (1 - p_hat_d) / (margin_d * margin_d)
    out.append((n_missing_zsq, p_hat_d * n_missing_zsq))
    n_missing_complement = z * z * p_hat_d / (margin_d * margin_d)
    out.append((n_missing_complement, p_hat_d * n_missing_complement))
    n_no_sqrt = z * p_hat_d * (1 - p_hat_d) / margin_d
    out.append((n_no_sqrt, p_hat_d * n_no_sqrt))
    return out


def _round_nk(pair):
    n_f, k_f = pair
    n_i = int(round_half_up(n_f, 0))
    k_i = int(round_half_up(k_f, 0))
    return n_i, k_i


def _nk_text(n: int, k: int) -> str:
    return f"n = {n}, k = {k}"


def _finish_nk(text, answer, answer_text, answer_latex, working, true_pair, raw_candidates):
    rounded = [_round_nk(c) for c in raw_candidates]
    plausible = [c for c in rounded if _plausible_nk(*c)]
    check_pairwise_distinct(
        true_pair,
        *plausible,
        msg="an n/k distractor collides with the true answer or another distractor",
    )
    _assert_key_clears_tolerance(true_pair, plausible, (Fraction(0), Fraction(0)))
    candidate_texts = [_nk_text(n, k) for n, k in plausible]
    distractors = pick_distractors(answer_text, candidate_texts, n=3)
    check(len(distractors) >= 1, "no distractor candidate survived filtering")
    return dict(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_latex,
        working=working,
        distractors=distractors,
    )


# --- medium: n and k from both printed bounds and a stated z ----------

_MEDIUM_INSTRUCTION = (
    "Give your answer in the form n = ..., k = ..., where n is the "
    "whole-number sample size and k is the number of successes (a whole "
    "number) that produced this interval."
)


def _make_medium(rng):
    for _ in range(_MAX_ATTEMPTS):
        base = _draw_base(rng)
        z, L4, U4 = base["z"], base["L4"], base["U4"]
        p_hat_d = (L4 + U4) / 2
        margin_d = (U4 - L4) / 2
        if margin_d <= 0:
            continue
        n_exact = z * z * p_hat_d * (1 - p_hat_d) / (margin_d * margin_d)
        if not rounds_unambiguously(float(n_exact), 0):
            continue
        n_key = int(round_half_up(n_exact, 0))
        if n_key != base["n"]:
            continue
        k_exact = p_hat_d * n_key
        if not rounds_unambiguously(float(k_exact), 0):
            continue
        k_key = int(round_half_up(k_exact, 0))
        if k_key != base["k"]:
            continue
        break
    else:
        check(False, "medium: could not construct a recoverable n/k draw")

    level = base["level"]
    text = f"{_interval_with_z_clause(rng, level, L4, U4, z)} {_MEDIUM_INSTRUCTION}"
    answer = {"n": n_key, "k": k_key}
    answer_text = _nk_text(n_key, k_key)
    answer_latex = rf"n = {n_key},\; k = {k_key}"
    p_str, m_str, z_str = _fmt4(p_hat_d), _fmt4(margin_d), _z_text(z)
    working = [
        f"p_hat = (L + U) / 2 = {p_str}; margin = (U - L) / 2 = {m_str}.",
        f"n = z² · p_hat · (1 - p_hat) / margin² = {z_str}² · "
        f"{p_str} · (1 - {p_str}) / {m_str}² = {n_key}; "
        f"k = p_hat · n = {p_str} · {n_key} = {k_key}.",
    ]
    raw_candidates = _nk_formula_candidates(z, p_hat_d, margin_d)
    return _finish_nk(text, answer, answer_text, answer_latex, working, (n_key, k_key), raw_candidates)


# --- hard: p_hat and ONE bound, z from a table -------------------------

_HARD_INSTRUCTION = (
    "Give your answer in the form n = ..., k = ..., where n is the "
    "whole-number sample size and k is the number of successes (a whole "
    "number) that produced this interval."
)


def _make_hard(rng):
    for _ in range(_MAX_ATTEMPTS):
        base = _draw_base(rng)
        z, level = base["z"], base["level"]
        p_hat_true = base["p_hat"]
        if not rounds_unambiguously(float(p_hat_true), 4):
            continue
        p_hat_stated = round_half_up(p_hat_true, 4)
        side = rng.choice(["lower", "upper"])
        bound = base["L4"] if side == "lower" else base["U4"]
        margin_d = (p_hat_stated - bound) if side == "lower" else (bound - p_hat_stated)
        if margin_d <= 0:
            continue
        n_exact = z * z * p_hat_stated * (1 - p_hat_stated) / (margin_d * margin_d)
        if not rounds_unambiguously(float(n_exact), 0):
            continue
        n_key = int(round_half_up(n_exact, 0))
        if n_key != base["n"]:
            continue
        k_exact = p_hat_stated * n_key
        if not rounds_unambiguously(float(k_exact), 0):
            continue
        k_key = int(round_half_up(k_exact, 0))
        if k_key != base["k"]:
            continue
        break
    else:
        check(False, "hard: could not construct a recoverable single-bound draw")

    # A 3-row table including the drawn level, order randomised per draw, values always printed to 3
    # d.p. so a table lookup and a stated-z prompt (medium/exam) never disagree.
    z_by_level = dict(_Z_TABLE)
    other_levels = [row for row in _Z_TABLE if row[0] != level]
    rng.shuffle(other_levels)
    # sorted by level so the table itself is stable to read
    table_rows = sorted([(level, z_by_level[level]), *other_levels[:2]])
    row_texts = [f"{_art(lv)} {lv}% confidence level uses z = {z2}" for lv, z2 in table_rows]
    table_text = f"{', '.join(row_texts[:-1])}, and {row_texts[-1]}"

    bound_str = stated_precision_string(bound, 4)
    p_str = stated_precision_string(p_hat_stated, 4)
    # Wording variety only: the literal substrings "p_hat = {p}", "{side} bound {b}." (period
    # immediately after, per the oracle's `_LOWER_BOUND_RE`/`_UPPER_BOUND_RE`) are kept verbatim in
    # every template, and every template is a complete sentence with its own main verb.
    opening_templates = [
        f"A sample gave an approximate normal confidence interval for a "
        f"population proportion with p_hat = {p_str} and {side} bound "
        f"{bound_str}.",
        f"A population proportion's approximate normal confidence interval "
        f"has p_hat = {p_str} and {side} bound {bound_str}.",
        f"An approximate normal confidence interval for a population "
        f"proportion is known to have p_hat = {p_str} and {side} bound "
        f"{bound_str}.",
    ]
    # "The interval was built at the {level}% confidence level." (the
    # oracle's `_HARD_LEVEL_RE`) is its own complete sentence, followed by
    # the table as a second complete sentence -- no dangling "Given that".
    text = (
        f"{rng.choice(opening_templates)} The interval was built at the "
        f"{level}% confidence level. For reference, {table_text}. "
        f"{_HARD_INSTRUCTION}"
    )
    answer = {"n": n_key, "k": k_key}
    answer_text = _nk_text(n_key, k_key)
    answer_latex = rf"n = {n_key},\; k = {k_key}"
    m_str, z_str = _fmt4(margin_d), _z_text(z)
    working = [
        f"margin = |p_hat - {side} bound| = {m_str}; z = {z_str} (from the {level}% row).",
        f"n = z² · p_hat · (1 - p_hat) / margin² = {z_str}² · "
        f"{p_str} · (1 - {p_str}) / {m_str}² = {n_key}; "
        f"k = p_hat · n = {p_str} · {n_key} = {k_key}.",
    ]
    raw_candidates = _nk_formula_candidates(z, p_hat_stated, margin_d)
    # "a bound taken as p-hat" (the wrong side): substitutes the printed
    # BOUND for p_hat inside the variance term, margin unchanged.
    n_bound_as_phat = z * z * bound * (1 - bound) / (margin_d * margin_d)
    raw_candidates.append((n_bound_as_phat, bound * n_bound_as_phat))
    return _finish_nk(text, answer, answer_text, answer_latex, working, (n_key, k_key), raw_candidates)


# --- exam: observed n AND the smallest N for a tighter target ----------

_EXAM_INSTRUCTION = (
    "Give your answer in the form n = ..., N = ..., where n is the "
    "whole-number sample size that produced this interval, and N is the "
    "smallest whole number sample size (at the same sample proportion and "
    "confidence level) whose margin would be at most the stated target."
)

_TARGET_PCTS = [30, 35, 40, 45, 55, 60, 65, 70]  # never 50: no exact halving


def _make_exam(rng):
    for _ in range(_MAX_ATTEMPTS):
        base = _draw_base(rng)
        z, L4, U4, level = base["z"], base["L4"], base["U4"], base["level"]
        p_hat_d = (L4 + U4) / 2
        margin_d = (U4 - L4) / 2
        if margin_d <= 0:
            continue
        n_exact = z * z * p_hat_d * (1 - p_hat_d) / (margin_d * margin_d)
        if not rounds_unambiguously(float(n_exact), 0):
            continue
        n_key = int(round_half_up(n_exact, 0))
        if n_key != base["n"]:
            continue
        k_exact = p_hat_d * n_key
        if not rounds_unambiguously(float(k_exact), 0):
            continue
        k_key = int(round_half_up(k_exact, 0))
        if k_key != base["k"]:
            continue

        target_ok = False
        for pct in rng.sample(_TARGET_PCTS, len(_TARGET_PCTS)):
            target_raw = margin_d * Fraction(pct, 100)
            target = round_half_up(target_raw, 4)
            if target <= 0:
                continue
            N_exact = z * z * p_hat_d * (1 - p_hat_d) / (target * target)
            frac_part = N_exact - int(N_exact)
            if not (Fraction(1, 20) <= frac_part <= Fraction(19, 20)):
                continue
            N_key = int(N_exact) + 1
            if N_key >= 10000:
                continue
            # Every principled route to N must land on the same whole number:
            # the printed midpoint (the working's route), the sample's own
            # proportion k/n ("the same sample proportion"), and scaling the
            # recovered n by (margin / target)^2 (the correct form of the
            # N_linear misconception below).
            p_kn = Fraction(k_key, n_key)
            N_by_kn = z * z * p_kn * (1 - p_kn) / (target * target)
            N_by_scale = n_key * (margin_d / target) ** 2
            if any(math.floor(alt) + 1 != N_key for alt in (N_by_kn, N_by_scale)):
                continue
            target_ok = True
            break
        if not target_ok:
            continue
        break
    else:
        check(False, "exam: could not construct a recoverable n/N draw")

    target_str = stated_precision_string(target, 4)
    text = f"{_interval_with_z_clause(rng, level, L4, U4, z)} The target margin is {target_str}. {_EXAM_INSTRUCTION}"
    answer = {"n": n_key, "N": N_key}
    answer_text = f"n = {n_key}, N = {N_key}"
    answer_latex = rf"n = {n_key},\; N = {N_key}"
    p_str, m_str, z_str = _fmt4(p_hat_d), _fmt4(margin_d), _z_text(z)
    working = [
        f"p_hat = (L + U) / 2 = {p_str}; margin = (U - L) / 2 = {m_str}.",
        f"n = z² · p_hat · (1 - p_hat) / margin² = {z_str}² · {p_str} · (1 - {p_str}) / {m_str}² = {n_key}.",
        f"N is the smallest whole number with z² · p_hat · (1 - p_hat) / N "
        f"no more than target², i.e. N = ceiling of z² · p_hat · "
        f"(1 - p_hat) / target², using target = {target_str}: N = {N_key}.",
    ]

    # Distractors: n is correct in every candidate except the last two, and
    # N carries a distinct misconception each time.
    N_floor = int(N_exact)  # rounds DOWN instead of up -- "smallest" ignored
    N_linear = int(round_half_up(Fraction(n_key) * (margin_d / target), 0))  # scales
    # linearly with the margin ratio instead of its square (forgets margin ~ 1/sqrt(n))
    raw_nk_candidates = _nk_formula_candidates(z, p_hat_d, margin_d)
    rounded_nk = [_round_nk(c) for c in raw_nk_candidates]
    plausible_nk = [c for c in rounded_nk if _plausible_nk(*c)]

    candidates = [(n_key, N_floor), (n_key, N_linear)]
    for n_wrong, _k_wrong in plausible_nk:
        if n_wrong == n_key or n_wrong < 2:
            continue
        # recompute a WRONG N off the SAME wrong formula's implied margin
        # scale, consistent with that formula's own (wrong) n.
        N_wrong = int(round_half_up(Fraction(N_key) * Fraction(n_wrong, n_key), 0))
        candidates.append((n_wrong, N_wrong))

    plausible = [c for c in candidates if c[0] >= 2 and c[1] >= 1 and c[1] < 10000]
    check_pairwise_distinct(
        (n_key, N_key),
        *plausible,
        msg="an exam-tier distractor collides with the true answer or another distractor",
    )
    _assert_key_clears_tolerance((n_key, N_key), plausible, (Fraction(0), Fraction(0)))
    candidate_texts = [f"n = {n}, N = {N}" for n, N in plausible]
    distractors = pick_distractors(answer_text, candidate_texts, n=3)
    check(len(distractors) >= 1, "no exam distractor candidate survived filtering")
    return dict(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_latex,
        working=working,
        distractors=distractors,
    )


@generator(
    "proportion_confidence_interval_inverse",
    level=14,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def proportion_confidence_interval_inverse(rng, variant="easy"):
    if variant == "easy":
        return _make_easy(rng)
    if variant == "medium":
        return _make_medium(rng)
    if variant == "hard":
        return _make_hard(rng)
    if variant == "exam":
        return _make_exam(rng)
    raise ValueError(f"unknown variant: {variant!r}")
