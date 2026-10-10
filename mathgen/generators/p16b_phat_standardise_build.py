"""Simulate repeated random sampling and standardise the sample proportion.

- ``p16b_phat_standardise``
  (``senior-simulate-repeated-random-sampling-for-a-variety-of``, ACMMM176,
  domain statistics-probability, answer: number)
  -- a REAL, seeded simulation of k = 5 to 8 independent samples of n
  Bernoulli(p) trials draws the sample proportion p-hat for each sample
  (every trial is actually drawn with the generator's own ``rng``, never
  approximated); the learner reads off a stated one of the samples and
  computes the standardised value
  z = (p-hat - p) / sqrt(p-hat(1 - p-hat)/n), using p-hat's OWN standard
  error (never p's) -- the formula is always stated in the prompt because
  that convention, not memorised knowledge, is what the question is
  assessing.

Four tiers, each changing the DECISION the learner makes:

- ``easy`` -- p-hat is printed per sample; z for a STATED sample. Decision:
  substitute the stated sample's own p-hat into its own standard error.
- ``medium`` -- the NUMBER OF SUCCESSES x is printed per sample (out of a
  stated n); z for a stated sample. Decision: form p-hat = x/n first.
- ``hard`` -- two stated samples i and j, on OPPOSITE sides of p, are
  compared: "which has the standardised value further from 0?". Drawn
  construct-to-target (``trap``: the sample with the LARGER |p-hat - p| has
  the SMALLER |z|, because p-hat(1 - p-hat) differs; ``plain``: the more
  familiar case) with a tie excluded by construction. Decision: compare
  standardised values, not raw distances from p.
- ``exam`` -- p is NEVER printed: the population is a stated chance model
  (a spinner, a fair die, a well-shuffled deck) the learner turns into an
  EXACT fraction p (1/3, 3/8, 5/12, 1/6, 1/4, 1/13); printed counts x per
  sample; z for a stated sample. Decision: derive the true parameter from
  the model, then standardise.

Every draw's key is checked against `rounds_unambiguously` (never sits
within a 0.1-of-a-rounding-unit margin of a half boundary) and against an
independent re-derivation of z through a genuinely different algebraic
route (`_verify_z`: n*p-hat*(1-p-hat) rewritten as x*(n-x)/n**2, never
reusing construction's own `phat*(1-phat)` expression).
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import (
    VerificationError,
    check,
    check_pairwise_distinct,
    exact_decimal_string,
    generator,
    latex_answer,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)

# n values chosen so that x/n always terminates as a decimal (each is
# 2**a * 5**b), so a per-sample p-hat is always an exact, displayable value.
N_SIZES = [20, 25, 40, 50, 80, 100, 125, 200, 250]

# p from 0.10 to 0.90 in steps of 0.05 -- a variety of values, each a short
# 1-2 dp decimal.
P_VALUES = [Fraction(i, 20) for i in range(2, 19)]

# Contexts for the easy/medium/hard tiers, where p is stated directly.
# Deliberately different wording per context (not just different numbers),
# so `mathgen.text_quality.prompt_shape` sees a real change in shape.
CONTEXTS = [
    dict(
        agent="A horticulturalist",
        unit="seed",
        verb="germinates",
        sample_desc="seeds planted",
    ),
    dict(
        agent="A quality-control inspector",
        unit="item",
        article="an",
        verb="is defective",
        sample_desc="items inspected from a production line",
    ),
    dict(
        agent="A pollster",
        unit="voter",
        verb="intends to vote yes",
        sample_desc="voters surveyed",
    ),
    dict(
        agent="A basketball coach",
        unit="free throw",
        verb="is made",
        sample_desc="free throws attempted",
    ),
    dict(
        agent="A statistics class",
        unit="spin",
        verb="lands on red",
        sample_desc="spins of a biased spinner",
    ),
]

# Chance models for the exam tier: p is NEVER printed as a number -- the
# learner derives it as an exact fraction from the stated model.
CHANCE_MODELS = [
    dict(desc="a spinner with 3 equal sectors, 1 of which is coloured red", p=Fraction(1, 3)),
    dict(desc="a spinner with 8 equal sectors, 3 of which are coloured red", p=Fraction(3, 8)),
    dict(desc="a spinner with 12 equal sectors, 5 of which are coloured red", p=Fraction(5, 12)),
    dict(desc="a fair six-sided die, where success is rolling a 6", p=Fraction(1, 6)),
    # A physical deck only has 52 cards, so a sample of up to 250 trials is
    # impossible without replacement -- state the replace-and-reshuffle
    # protocol explicitly. The spinner and die
    # models need no such statement: spinning again or rolling again never
    # depletes anything, so every trial is independent by construction
    # already.
    dict(
        desc="a well-shuffled standard deck of 52 playing cards, where one card is drawn, then "
        "replaced and the deck reshuffled, for each trial, and success is drawing a heart",
        p=Fraction(1, 4),
    ),
    dict(
        desc="a well-shuffled standard deck of 52 playing cards, where one card is drawn, then "
        "replaced and the deck reshuffled, for each trial, and success is drawing a King",
        p=Fraction(1, 13),
    ),
]

EXAM_AGENTS = ["A statistics class", "A probability tutor", "A researcher"]

# The formula is always stated in its OWN sentence: a run-on clause wedged into the question sentence read badly, and
# splitting it out lets every tier share exactly the same statement.
FORMULA_SENTENCE = (
    "The standardised value of a sample is z = (p̂ − p) / √(p̂(1 − p̂)/n), using "
    "that sample's own p̂ in the standard error."
)

# The real-LaTeX twin of FORMULA_SENTENCE:
# prose in \text{...}, the formula itself built from \frac/\sqrt/\hat so
# `tests/test_prompt_latex_mangled.py` sees real LaTeX, never bare English
# or a literal "sqrt(...)" run inside math mode.
FORMULA_SENTENCE_LATEX = (
    r"\text{The standardised value of a sample is} \ z = "
    r"\frac{\hat{p} - p}{\sqrt{\hat{p}(1 - \hat{p})/n}}"
    r"\text{, using that sample's own} \ \hat{p} \ \text{in the standard error.}"
)


def _prose_latex(sentence: str) -> str:
    r"""Wrap one plain-English sentence for KaTeX's \text mode, escaping the
    handful of characters \text treats specially. None of this generator's
    own sentences contain any of them today (no context/agent/unit string
    does), but escape defensively rather than assume that never changes."""
    s = sentence
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("{", r"\{").replace("}", r"\}")
    s = s.replace("$", r"\$").replace("%", r"\%").replace("&", r"\&")
    s = s.replace("#", r"\#").replace("_", r"\_")
    s = s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    return r"\text{" + s + "}"


def _assemble(sentences: list[str]) -> tuple[str, str]:
    """Join `sentences` into (`text`, `latex`): `text` is plain prose
    (space-joined); `latex` wraps every sentence in `\\text{...}` EXCEPT the
    shared `FORMULA_SENTENCE`, which is swapped for its real-LaTeX twin
    `FORMULA_SENTENCE_LATEX` -- so the two stay in lockstep by construction
    rather than by two independently-hand-written strings drifting apart."""
    text = " ".join(sentences)
    latex = " ".join(FORMULA_SENTENCE_LATEX if s == FORMULA_SENTENCE else _prose_latex(s) for s in sentences)
    return text, latex


# ============================================================================
# Simulation -- every sample is genuinely drawn, trial by trial, with the
# generator's own seeded rng.
# ============================================================================


def _simulate_sample(rng, n: int, p_float: float) -> int:
    """Draw n independent Bernoulli(p) trials and return the count of
    successes, resampling (the sample only) until 0 < x < n -- the guard
    that SE can never be zero and p-hat can never sit at 0 or 1."""
    for _attempt in range(200):
        x = sum(1 for _ in range(n) if rng.random() < p_float)
        if 0 < x < n:
            return x
    raise RuntimeError("p16b_phat_standardise: a sample stuck at p-hat = 0 or 1 after 200 attempts")


def _simulate_batch(rng, n: int, p_float: float, k: int) -> list[int]:
    return [_simulate_sample(rng, n, p_float) for _ in range(k)]


def _z_value(phat: float, p: float, n: int) -> float:
    se = math.sqrt(phat * (1 - phat) / n)
    return (phat - p) / se


def _verify_z(x: int, n: int, p_float: float, z: float, eps: float = 1e-6) -> None:
    """Independent route: n*p-hat*(1-p-hat) rewritten in pure integers as
    x*(n-x)/n, so the variance term is derived from the raw count x and n
    alone -- never by reusing construction's own `phat*(1-phat)/n`
    expression."""
    variance_term = Fraction(x) * Fraction(n - x) / Fraction(n) ** 3
    se_independent = math.sqrt(float(variance_term))
    z_independent = (x / n - p_float) / se_independent
    check(
        abs(z_independent - z) < eps, f"p16b_phat_standardise: z standardisation check failed ({z_independent} != {z})"
    )


# ============================================================================
# Distractors
# ============================================================================


def _fmt_z(value: float) -> str:
    return stated_precision_string(round_half_up(Fraction(value), 2), 2)


def _misconception_candidates(
    phat: float, p: float, n: int, x: int, *, other_sample_z: float | None = None, include_count: bool = False
):
    """Priority-ordered (name, raw value) misconception candidates."""
    items = []
    if other_sample_z is not None:
        items.append(("other-sample", other_sample_z))
    se_uses_p = math.sqrt(p * (1 - p) / n)
    items.append(("se-uses-p", (phat - p) / se_uses_p))
    se_no_n = math.sqrt(phat * (1 - phat))
    items.append(("omits-n", (phat - p) / se_no_n))
    var_term = phat * (1 - phat) / n
    if var_term > 0:
        items.append(("omits-sqrt", (phat - p) / var_term))
    se_correct = math.sqrt(phat * (1 - phat) / n)
    items.append(("reversed-difference", (p - phat) / se_correct))
    if include_count:
        z_count = (x - p) / se_correct
        if abs(z_count) <= 50:
            items.append(("count-not-proportion", z_count))
    return items


def _finalise(key_z: float, candidates) -> tuple[str, list[str]]:
    """Format the key and every candidate identically, dedupe by displayed
    value (candidate-vs-candidate as well as candidate-vs-answer), and prove
    the survivors pairwise distinct as a generator-internal invariant."""
    answer_text = _fmt_z(key_z)
    seen = {answer_text}
    texts = []
    for _name, val in candidates:
        t = _fmt_z(val)
        if t in seen:
            continue
        seen.add(t)
        texts.append(t)
    distractors = pick_distractors(answer_text, texts, tolerance=stated_precision_tolerance(2))
    check(len(distractors) >= 2, "p16b_phat_standardise: fewer than 2 candidates survived dedup against the answer")
    values = [Fraction(answer_text)] + [Fraction(d) for d in distractors]
    check_pairwise_distinct(*values, msg="p16b_phat_standardise: two options share a value")
    return answer_text, distractors


# ============================================================================
# Tiers
# ============================================================================


def _draw_common(rng):
    """p, n, k and a fully-simulated batch, shared by easy/medium."""
    for _attempt in range(60):
        p = rng.choice(P_VALUES)
        n = rng.choice(N_SIZES)
        k = rng.randint(5, 8)
        xs = _simulate_batch(rng, n, float(p), k)
        stated = rng.randint(1, k)
        x = xs[stated - 1]
        phat = float(Fraction(x, n))
        z = _z_value(phat, float(p), n)
        if rounds_unambiguously(z, 2, margin=0.1):
            return p, n, k, xs, stated, x, z
    raise RuntimeError("p16b_phat_standardise: no draw cleared the rounding-ambiguity margin in 60 attempts")


def _population_sentence(context, p_str: str) -> str:
    article = context.get("article", "a")
    return (
        f"{context['agent']} records whether each {context['unit']} {context['verb']}. "
        f"In this population, p = {p_str} is the probability that {article} {context['unit']} "
        f"{context['verb']}."
    )


def _easy(rng):
    for _retry in range(20):
        p, n, k, xs, stated, x, z = _draw_common(rng)
        _verify_z(x, n, float(p), z)
        candidates = _misconception_candidates(float(Fraction(x, n)), float(p), n, x)
        try:
            answer_text, distractors = _finalise(z, candidates)
        except VerificationError:
            continue
        break
    else:
        raise RuntimeError("p16b_phat_standardise/easy: no draw produced >=2 distinct distractors in 20 attempts")

    context = rng.choice(CONTEXTS)
    p_str = exact_decimal_string(p, 2)
    phat_display = [exact_decimal_string(Fraction(xi, n), 4) for xi in xs]
    sample_lines = "; ".join(f"sample {i + 1}: {d}" for i, d in enumerate(phat_display))

    sentences = [
        _population_sentence(context, p_str),
        f"{k} independent samples of {n} {context['sample_desc']} are drawn, and the sample "
        f"proportion p̂ is recorded for each: {sample_lines}.",
        FORMULA_SENTENCE,
        f"For sample {stated}, calculate the standardised value z, to 2 decimal places.",
    ]
    text, latex = _assemble(sentences)
    working = [
        f"Sample {stated}: p̂ = {x}/{n} = {phat_display[stated - 1]}.",
        f"z = ({phat_display[stated - 1]} - {p_str}) / √({phat_display[stated - 1]} x "
        f"(1 - {phat_display[stated - 1]}) / {n}) = {answer_text}.",
    ]
    return dict(
        text=text,
        latex=latex,
        answer=float(Fraction(answer_text)),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
        working=working,
    )


def _medium(rng):
    for _retry in range(20):
        p, n, k, xs, stated, x, z = _draw_common(rng)
        _verify_z(x, n, float(p), z)
        candidates = _misconception_candidates(float(Fraction(x, n)), float(p), n, x, include_count=True)
        try:
            answer_text, distractors = _finalise(z, candidates)
        except VerificationError:
            continue
        break
    else:
        raise RuntimeError("p16b_phat_standardise/medium: no draw produced >=2 distinct distractors in 20 attempts")

    context = rng.choice(CONTEXTS)
    p_str = exact_decimal_string(p, 2)
    sample_lines = "; ".join(f"sample {i + 1}: x = {xi}" for i, xi in enumerate(xs))

    sentences = [
        _population_sentence(context, p_str),
        f"{k} independent samples of {n} {context['sample_desc']} are drawn, and the number of "
        f"successes x is recorded for each: {sample_lines}.",
        FORMULA_SENTENCE,
        f"For sample {stated}, first find p̂ = x/n, then calculate the standardised value z, to 2 decimal places.",
    ]
    text, latex = _assemble(sentences)
    phat_str = exact_decimal_string(Fraction(x, n), 4)
    working = [
        f"Sample {stated}: p̂ = {x}/{n} = {phat_str}.",
        f"z = ({phat_str} - {p_str}) / √({phat_str} x (1 - {phat_str}) / {n}) = {answer_text}.",
    ]
    return dict(
        text=text,
        latex=latex,
        answer=float(Fraction(answer_text)),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
        working=working,
    )


def _hard(rng):
    target_class = "trap" if rng.random() < 0.5 else "plain"
    result = None
    for _outer in range(200):
        p = rng.choice(P_VALUES)
        p_float = float(p)
        n = rng.choice(N_SIZES)
        k = rng.randint(5, 8)
        xs = _simulate_batch(rng, n, p_float, k)

        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        rng.shuffle(pairs)
        for i, j in pairs:
            # 2**a * 5**b, so x/n terminates within the 4dp display), so
            # compare distances from p as exact `Fraction`s -- never as
            # Python floats. `xs[i] / n` as a float can round the two
            # distances to slightly different values even when the true
            # rational distances are EXACTLY equal (e.g. p = 0.55,
            # phat = 0.66 and 0.44: both exactly 0.11 away), and that float
            # noise, not a coin flip, decided which sample was "far" --
            # measured at 30 strict-trap / 94 strict-plain / 76 exact-tie
            # over seeds 0-199 before this fix. An exact tie is excluded
            # from BOTH classes: it is never a valid trap or plain pair.
            phat_i_frac, phat_j_frac = Fraction(xs[i], n), Fraction(xs[j], n)
            if (phat_i_frac - p) * (phat_j_frac - p) >= 0:
                continue  # not on opposite sides of p
            dist_i, dist_j = abs(phat_i_frac - p), abs(phat_j_frac - p)
            if dist_i == dist_j:
                continue  # exact tie: never offer this pair in either class
            phat_i, phat_j = float(phat_i_frac), float(phat_j_frac)
            zi = _z_value(phat_i, p_float, n)
            zj = _z_value(phat_j, p_float, n)
            far_is_i = dist_i > dist_j
            far_z, near_z = (zi, zj) if far_is_i else (zj, zi)
            if abs(abs(far_z) - abs(near_z)) < 0.05:
                continue  # tie excluded by construction
            is_trap = abs(far_z) < abs(near_z)
            cls = "trap" if is_trap else "plain"
            if cls != target_class:
                continue
            winner_is_i = abs(zi) >= abs(zj)
            z_win = zi if winner_is_i else zj
            if not rounds_unambiguously(z_win, 2, margin=0.1):
                continue
            winner, loser = (i, j) if winner_is_i else (j, i)
            z_lose = zj if winner_is_i else zi
            x_win = xs[winner]
            candidates = _misconception_candidates(float(Fraction(x_win, n)), p_float, n, x_win, other_sample_z=z_lose)
            try:
                answer_text, distractors = _finalise(z_win, candidates)
            except VerificationError:
                continue
            result = (p, n, k, xs, i, j, winner, loser, z_win, z_lose, x_win, answer_text, distractors)
            break
        if result is not None:
            break
    else:
        raise RuntimeError("p16b_phat_standardise/hard: no draw realised the target class in 200 attempts")

    p, n, k, xs, i, j, winner, loser, z_win, z_lose, x_win, answer_text, distractors = result
    _verify_z(x_win, n, float(p), z_win)

    context = rng.choice(CONTEXTS)
    p_str = exact_decimal_string(p, 2)
    phat_display = [exact_decimal_string(Fraction(xi, n), 4) for xi in xs]
    sample_lines = "; ".join(f"sample {idx + 1}: {d}" for idx, d in enumerate(phat_display))

    lower_label, higher_label = sorted((i + 1, j + 1))
    sentences = [
        _population_sentence(context, p_str),
        f"{k} independent samples of {n} {context['sample_desc']} are drawn, and the sample "
        f"proportion p̂ is recorded for each: {sample_lines}.",
        FORMULA_SENTENCE,
        f"Samples {lower_label} and {higher_label} are on opposite sides of p. Which of samples "
        f"{lower_label} and {higher_label} has the standardised value further from 0? Give that "
        f"value, with its sign, to 2 decimal places.",
    ]
    text, latex = _assemble(sentences)
    working = [
        f"Sample {winner + 1}: p̂ = {phat_display[winner]}, z = {_fmt_z(z_win)}.",
        f"Sample {loser + 1}: p̂ = {phat_display[loser]}, z = {_fmt_z(z_lose)}.",
        f"|{_fmt_z(z_win)}| > |{_fmt_z(z_lose)}|, so sample {winner + 1} has the standardised "
        f"value further from 0: {answer_text}.",
    ]
    return dict(
        text=text,
        latex=latex,
        answer=float(Fraction(answer_text)),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
        working=working,
    )


def _exam(rng):
    for _attempt in range(60):
        model = rng.choice(CHANCE_MODELS)
        p = model["p"]
        p_float = float(p)
        n = rng.choice(N_SIZES)
        k = rng.randint(5, 8)
        xs = _simulate_batch(rng, n, p_float, k)
        stated = rng.randint(1, k)
        x = xs[stated - 1]
        z = _z_value(x / n, p_float, n)
        if not rounds_unambiguously(z, 2, margin=0.1):
            continue
        candidates = _misconception_candidates(x / n, p_float, n, x, include_count=True)
        try:
            answer_text, distractors = _finalise(z, candidates)
        except VerificationError:
            continue
        break
    else:
        raise RuntimeError("p16b_phat_standardise/exam: no draw produced a usable question in 60 attempts")
    _verify_z(x, n, p_float, z)

    agent = rng.choice(EXAM_AGENTS)
    sample_lines = "; ".join(f"sample {i + 1}: x = {xi}" for i, xi in enumerate(xs))

    sentences = [
        f"{agent} repeatedly draws samples using {model['desc']}; a trial counts as a success "
        f"exactly when that outcome occurs.",
        f"{k} independent samples of {n} trials each are drawn, and the number of successes x "
        f"is recorded for each: {sample_lines}.",
        FORMULA_SENTENCE,
        f"For sample {stated}, first find the true probability of success p from the chance "
        f"model, then calculate the standardised value z, to 2 decimal places.",
    ]
    text, latex = _assemble(sentences)
    phat_str = exact_decimal_string(Fraction(x, n), 4)
    working = [
        f"p = {p.numerator}/{p.denominator} (exact, from the chance model).",
        f"Sample {stated}: p̂ = {x}/{n} = {phat_str}.",
        f"z = ({phat_str} - {p.numerator}/{p.denominator}) / √({phat_str} x (1 - {phat_str}) / {n}) = {answer_text}.",
    ]
    return dict(
        text=text,
        latex=latex,
        answer=float(Fraction(answer_text)),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
        working=working,
    )


@generator(
    "p16b_phat_standardise",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p16b_phat_standardise(rng, tier="easy"):
    """Simulate repeated random sampling for a stated true proportion p and
    sample size n; read the reported sample proportion p-hat for a stated
    one of the samples, then compute the standardised value
    z = (p-hat - p) / sqrt(p-hat(1 - p-hat)/n) (ACMMM176)."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier == "hard":
        return _hard(rng)
    if tier == "exam":
        return _exam(rng)
    raise ValueError(f"p16b_phat_standardise: unknown tier {tier!r}")
