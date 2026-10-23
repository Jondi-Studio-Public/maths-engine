"""Bernoulli as a MODEL vs Bernoulli SEQUENCE probability -- two senior statistics-probability skills that sit
either side of a boundary that must stay sharp.

``senior-use-a-bernoulli-random-variable-as-a-model`` (ACMMM143, topic
``p12d_bernoulli_model``) -- the MODELLING-CHOICE step: parametrise a
two-outcome situation as Bernoulli(p) (deriving p from a described count of
equally-likely outcomes -- never handed p directly) and compute its mean
(E[X] = p) or variance (Var[X] = p(1-p)), or invert: given Var[X] and a
contextual clue about which outcome is more/less common, recover p.

``senior-use-bernoulli-random-variables-and-associated-probabilities-to`` (ACMMM146, topic
``p12d_bernoulli_sequence``) -- the PROBABILITY-COMPUTATION step once a model is already assumed: p (or, at
hard/exam, the FAILURE probability q) is GIVEN directly, and the question asks for the probability of one
SPECIFIC, STATED, ORDERED SEQUENCE of independent Bernoulli trials -- a plain product of p/(1-p) factors in
the stated order. This is deliberately narrower than the descriptor (which also permits "the expected
value" as a legal deliverable) -- restricted to the sequence-probability reading only, so it never duplicates the first skill's own E[X]/Var[X]
deliverable.

**The boundary this pair must never become**: ``binomial-point-probability``
(``skills/senior-mathematics/binomial-point-probability.yaml``, topic
``binomial_probability``) asks "the probability of exactly x successes in a
binomial distribution" -- a COMBINATORIAL count over unordered outcomes,
``nCr * p^x * (1-p)^(n-x)``. Neither topic here ever multiplies by a
binomial coefficient in its CONSTRUCTION (the coefficient only ever appears
as the ``treats-as-unordered`` misconception distractor on
``p12d_bernoulli_sequence``, deliberately offered and rejected).

**The boundary against `bernoulli-mle`**
(``mathgen/generators/university_stat_learning.py``, topic ``bernoulli_mle``,
p-hat = k/n from OBSERVED sample data): neither topic here ever asks the
learner to ESTIMATE p from data. Both take p (or a value that reduces to it)
as GIVEN.

**Construction is always exact.** Every drawn probability is a `Fraction`
from a small, denominator-controlled pool (denominators dividing 60, plus a
decimal-tenths pool for ``p12d_bernoulli_sequence``'s exam tier) so every
E[X]/Var[X]/sequence-product value is an exact `Fraction`, never a rounded
float.

**Independent check.** Every forward-tier answer is re-derived via a route
genuinely different from its own construction (see each helper's own
docstring); every inverse-tier answer is forward-substituted back into the
defining equation, never re-run through whatever root-selection/mapping
logic construction used.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    pick_distractors,
)

_MAX_ATTEMPTS = 300

# Denominators all divide 60, so p(1-p) and any inverse root stay exact and simple.
_P_POOL_CORE = [
    Fraction(1, 5),
    Fraction(1, 4),
    Fraction(1, 3),
    Fraction(2, 5),
    Fraction(1, 2),
    Fraction(3, 5),
    Fraction(2, 3),
    Fraction(3, 4),
    Fraction(4, 5),
]
# The original 19-entry _P_POOL_WIDE collapses to only 10 distinct p*(1-p) products under the
# p<->1-p symmetry that _seq_easy's length-2 mixed sequence always hits (see module docstring's
# "root causes" note) -- right at the pool-variety THIN floor (>=10 distinct), and getting
# flagged whenever a rare group's pair of draws both miss 200 seeds. Twelfths/fifteenths/twentieths
# (all divide 60, same exactness invariant as the rest of this pool) add 10 more distinct
# product-groups (2 + 4 + 4, one per coprime numerator below half the denominator), for 20 groups
# total -- comfortably clear of both the THIN floor and the CONCENTRATED top-1 ceiling.
_P_POOL_WIDE = _P_POOL_CORE + [
    Fraction(1, 6),
    Fraction(5, 6),
    Fraction(1, 8),
    Fraction(3, 8),
    Fraction(5, 8),
    Fraction(7, 8),
    Fraction(1, 10),
    Fraction(3, 10),
    Fraction(7, 10),
    Fraction(9, 10),
    Fraction(1, 12),
    Fraction(5, 12),
    Fraction(7, 12),
    Fraction(11, 12),
    Fraction(1, 15),
    Fraction(2, 15),
    Fraction(4, 15),
    Fraction(7, 15),
    Fraction(8, 15),
    Fraction(11, 15),
    Fraction(13, 15),
    Fraction(14, 15),
    Fraction(1, 20),
    Fraction(3, 20),
    Fraction(7, 20),
    Fraction(9, 20),
    Fraction(11, 20),
    Fraction(13, 20),
    Fraction(17, 20),
    Fraction(19, 20),
]
# Excludes p = 1/2 -- the only pool entry where p(1-p) = 1/4, the double
# root with no less/more-common disambiguation possible (Skill 1 hard/exam).
_P_POOL_NO_HALF = [p for p in _P_POOL_WIDE if p != Fraction(1, 2)]
# Skill 2 exam tier: a decimal probability (denominator 10), excluding the
# digit 5 (which is p = q = 1/2, breaking the hard/exam failure-probability
# trap's "convert to the OTHER value" reasoning).
_DECIMAL_DIGITS_NO_HALF = [1, 2, 3, 4, 6, 7, 8, 9]


def _frac(value: Fraction) -> tuple[dict, str]:
    """(answer dict, display text) for an exact-fraction `answer: fraction` skill -- mirrors `_fraction_payload`
    in `mathgen/generators/university_stat_learning.py`."""
    value = Fraction(value)
    text = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    return {"p": value.numerator, "q": value.denominator}, text


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


# ============================================================================
# ACMMM143 -- use a Bernoulli random variable AS A MODEL
# ============================================================================

# Easy: derive p from a described count of equally-likely outcomes, then apply E[X] = p. Three
# structurally different scenario templates (prompt variety, and distinct from medium's own three
# below). Each `setup` is a CALLABLE, not a bare format string: a count of exactly 1 needs its own
# verb form ("1 of which IS shaded", never "are") and, for a predicate-noun clause, its own
# singular/indefinite-article noun form ("1 of which is A WINNING TICKET", never the bare plural
# "winning tickets") -- a plural after a count of one is a standing wording defect.


def _spinner_easy(d, k):
    verb = "is" if k == 1 else "are"
    return (
        f"A spinner has {d} equal sectors, {k} of which {verb} shaded. The "
        "spinner is spun once; let X = 1 if it lands on a shaded sector, "
        "and X = 0 otherwise."
    )


def _raffle_easy(d, k):
    clause = "1 of which is a winning ticket" if k == 1 else f"{k} of which are winning tickets"
    return (
        f"A box contains {d} identical-looking raffle tickets, {clause}. One "
        "ticket is drawn at random; let X = 1 if the ticket drawn is a "
        "winning ticket, and X = 0 otherwise."
    )


def _committee_easy(d, k):
    verb = "has" if k == 1 else "have"
    return (
        f"A committee has {d} members, {k} of whom {verb} a science "
        "background. One member is chosen at random to present; let "
        "X = 1 if the member chosen has a science background, and "
        "X = 0 otherwise."
    )


_MODEL_EASY_CONTEXTS = [
    dict(setup=_spinner_easy, ask="What is E(X), the expected value of X?"),
    dict(setup=_raffle_easy, ask="What is E(X), the expected value of X?"),
    dict(setup=_committee_easy, ask="What is E(X), the expected value of X?"),
]


def _bag_medium(d, k):
    verb = "is" if k == 1 else "are"
    return (
        f"A bag contains {d} marbles, {k} of which {verb} red. One marble "
        "is drawn at random; let X = 1 if the marble drawn is red, and "
        "X = 0 otherwise."
    )


def _cards_medium(d, k):
    verb = "is" if k == 1 else "are"
    return (
        f"A set of {d} cards is numbered 1 to {d}, and {k} of the cards "
        f"{verb} marked with a star. One card is drawn at random; let "
        "X = 1 if the card drawn is marked with a star, and X = 0 "
        "otherwise."
    )


def _parking_medium(d, k):
    clause = "1 of which is an electric vehicle" if k == 1 else f"{k} of which are electric vehicles"
    return (
        f"A car park has {d} parked cars, {clause}. One car is chosen at "
        "random; let X = 1 if the car chosen is an electric vehicle, and "
        "X = 0 otherwise."
    )


# Medium: DIFFERENT templates from easy's, same derive-p shape, asks Var[X].
_MODEL_MEDIUM_CONTEXTS = [
    dict(setup=_bag_medium, ask="What is Var(X), the variance of X?"),
    dict(setup=_cards_medium, ask="What is Var(X), the variance of X?"),
    dict(setup=_parking_medium, ask="What is Var(X), the variance of X?"),
]

# Hard/exam: abstract inverse-root templates (three differently-worded
# framings of "success is the less/more common outcome", never repeating
# the literal words another template uses, so the hard/exam pair together
# still clears >= 3 distinct prompt shapes over 60 seeds).
_MODEL_INVERSE_TEMPLATES = [
    "A two-outcome scenario is modelled by a Bernoulli random variable X, "
    "where success is defined as the {comp} common of the two possible "
    "outcomes.",
    "In a repeated random trial modelled by a Bernoulli random variable X, "
    "the outcome labelled success occurs {comp2} often than the outcome "
    "labelled failure.",
    "A Bernoulli random variable X models a two-outcome situation in which "
    "success is the {comp3} frequent of the two possible outcomes.",
]

# Exam-tier setup wordings, each stating the indirect frequency clue (never
# the literal words "less common"/"more common" hard tier uses).
_MODEL_EXAM_SETUPS = [
    "A two-outcome scenario is modelled by a Bernoulli random variable X. "
    "Across many trials, the outcome is a failure on {freq_word} than half "
    "of all trials.",
    "A Bernoulli random variable X models a repeated two-outcome trial. Out "
    "of a large number of trials, {freq_word} than half end in failure.",
    "In a two-outcome situation modelled by a Bernoulli random variable X, "
    "failures occur on {freq_word} than half of all trials in the long run.",
]


def _draw_equally_likely(rng, d_lo, d_hi):
    """Sector/outcome counts for a genuinely constructible equally-likely
    scenario: total >= 2 sectors, shaded count strictly between 0 and the
    total (never 0 or the total -- a probability of exactly 0 or 1 is not a
    genuine two-outcome model)."""
    d = rng.randint(d_lo, d_hi)
    k = rng.randint(1, d - 1)
    check(0 < k < d, "shaded count must be strictly between 0 and the total")
    return d, k


def _model_easy(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        ctx = rng.choice(_MODEL_EASY_CONTEXTS)
        # Widened from (4, 8) -- too few distinct k/d denominators to clear the pool-variety check's
        # 200-seed CONCENTRATED bar.
        d, k = _draw_equally_likely(rng, 4, 12)
        p = Fraction(k, d)
        check(0 < p < 1, "p must be strictly between 0 and 1")
        answer = p
        # Independent route: the complement of the NOT-shaded count, never
        # the same Fraction(k, d) construction expression.
        check_equivalent(answer, 1 - Fraction(d - k, d), "E[X] via complement route disagrees with p = k/d")

        off_by_one_k = k + 1 if k + 1 <= d - 1 else k - 1
        uses_failure = 1 - p
        off_by_one = Fraction(off_by_one_k, d)
        try:
            check_pairwise_distinct(answer, uses_failure, off_by_one, msg="a model-easy distractor collides")
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a non-colliding scenario for p12d model easy")

    text = f"{ctx['setup'](d, k)} {ctx['ask']}"
    answer_dict, answer_text = _frac(answer)
    return dict(
        text=text,
        answer=answer_dict,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"p = P(X = 1) = {k}/{d}.",
            f"E(X) = p = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, [_frac(uses_failure)[1], _frac(off_by_one)[1]]),
    )


def _model_medium(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        ctx = rng.choice(_MODEL_MEDIUM_CONTEXTS)
        # Widened from (4, 12) -- Var[X] = p(1-p) is symmetric in p <-> 1-p, so the effective
        # distinct-answer count is roughly half the denominator range; (4, 18) restores enough
        # distinct Var[X] values to clear the CONCENTRATED bar (top answer 14%).
        d, k = _draw_equally_likely(rng, 4, 18)
        p = Fraction(k, d)
        check(0 < p < 1, "p must be strictly between 0 and 1")
        # Independent route: E[X^2] - E[X]^2 (X^2 = X for a 0/1 variable, so
        # E[X^2] = p), computed as p - p*p, then checked against the
        # p*(1-p) construction formula.
        answer = p - p * p
        check_equivalent(answer, Fraction(p) * (1 - Fraction(p)), "Var[X] via E[X^2]-E[X]^2 disagrees with p(1-p)")
        check(answer <= Fraction(1, 4), "Var[X] must never exceed 1/4")

        adds_instead = p + (1 - p)  # == 1 always
        check(adds_instead == 1, "adds-instead-of-multiplies must equal 1")
        off_by_one_k = k + 1 if k + 1 <= d - 1 else k - 1
        p_off = Fraction(off_by_one_k, d)
        off_by_one = p_off * (1 - p_off)
        try:
            check_pairwise_distinct(answer, adds_instead, off_by_one, msg="a model-medium distractor collides")
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a non-colliding scenario for p12d model medium")

    text = f"{ctx['setup'](d, k)} {ctx['ask']}"
    answer_dict, answer_text = _frac(answer)
    return dict(
        text=text,
        answer=answer_dict,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"p = P(X = 1) = {k}/{d}.",
            f"Var(X) = p(1-p) = ({k}/{d})(1 - {k}/{d}) = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, [_frac(adds_instead)[1], _frac(off_by_one)[1]]),
    )


def _model_hard(rng):
    """Inverse direction: Var[X] is stated; a contextual clue (which outcome
    is more/less common) disambiguates the correct root of
    p^2 - p + Var[X] = 0. Construction picks p_true from the pool FIRST
    (never p = 1/2, so the two roots -- p_true and 1 - p_true -- are always
    distinct, Var[X] != 1/4 always) and derives Var[X] = p_true(1-p_true)
    from it, so "solving the quadratic" here is "looking up which pool
    entry produces the stated Var[X]", which
    is allowed."""
    for _attempt in range(_MAX_ATTEMPTS):
        p_true = rng.choice(_P_POOL_NO_HALF)
        check(p_true != Fraction(1, 2), "hard-tier p_true must never be 1/2")
        v = p_true * (1 - p_true)
        check(v != Fraction(1, 4), "Var[X] must never be exactly 1/4 (double-root case)")

        less_common = p_true < Fraction(1, 2)
        template = rng.choice(_MODEL_INVERSE_TEMPLATES)
        clue = template.format(
            comp="less" if less_common else "more",
            comp2="less" if less_common else "more",
            comp3="less" if less_common else "more",
        )

        answer = p_true
        # Independent check: forward-substitute the CHOSEN answer back into
        # p(1-p) -- never re-runs the less/more-common selection logic above.
        check_equivalent(answer * (1 - answer), v, "chosen root does not reproduce the stated Var[X]")

        wrong_root = 1 - p_true
        treats_var_as_p = v
        try:
            check_pairwise_distinct(answer, wrong_root, treats_var_as_p, msg="a model-hard distractor collides")
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a non-colliding scenario for p12d model hard")

    v_dict, v_text = _frac(v)
    text = f"{clue} If Var(X) = {v_text}, find P(success)."
    answer_dict, answer_text = _frac(answer)
    return dict(
        text=text,
        answer=answer_dict,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Solve p^2 - p + {v_text} = 0; the two roots are {answer_text} and {_frac(wrong_root)[1]}.",
            f"The stated clue selects the root matching the less/more-common condition: P(success) = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, [_frac(wrong_root)[1], _frac(treats_var_as_p)[1]]),
    )


def _model_exam(rng):
    """Exam tier: genuine escalation over hard -- the contextual clue is
    described via an INDIRECT frequency statement about FAILURE ("a failure
    on more/fewer than half of all trials") rather than hard's literal
    "success is less/more common" wording, and it describes the FAILURE
    outcome while the question still asks for P(success). The clue directly
    identifies q = P(failure) as one of the two roots of
    p^2 - p + Var[X] = 0 (the other root is p); the learner must recognise
    the clue names q, not p, and report the COMPLEMENT (the other root) as
    the answer -- a genuine conversion step hard tier never requires (hard's
    clue already names success directly)."""
    for _attempt in range(_MAX_ATTEMPTS):
        p_true = rng.choice(_P_POOL_NO_HALF)
        check(p_true != Fraction(1, 2), "exam-tier p_true must never be 1/2")
        v = p_true * (1 - p_true)
        check(v != Fraction(1, 4), "Var[X] must never be exactly 1/4 (double-root case)")

        q_true = 1 - p_true
        more_than_half = q_true > Fraction(1, 2)
        freq_word = "more" if more_than_half else "fewer"
        setup_template = rng.choice(_MODEL_EXAM_SETUPS)
        setup = setup_template.format(freq_word=freq_word)

        answer = p_true  # the COMPLEMENT of the clue-selected root (q_true)
        check_equivalent(answer * (1 - answer), v, "chosen complement does not reproduce the stated Var[X]")

        didnt_flip = q_true  # reported the clue-selected root directly, without converting
        treats_var_as_p = v
        try:
            check_pairwise_distinct(answer, didnt_flip, treats_var_as_p, msg="a model-exam distractor collides")
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a non-colliding scenario for p12d model exam")

    v_dict, v_text = _frac(v)
    text = f"{setup} If Var(X) = {v_text}, find P(success)."
    answer_dict, answer_text = _frac(answer)
    return dict(
        text=text,
        answer=answer_dict,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Solve p^2 - p + {v_text} = 0; the two roots are {answer_text} and {_frac(didnt_flip)[1]}.",
            "The frequency statement describes FAILURE, so it selects the root that is "
            f"P(failure) = {_frac(didnt_flip)[1]}; P(success) is the OTHER root, "
            f"{answer_text}.",
        ],
        distractors=pick_distractors(answer_text, [_frac(didnt_flip)[1], _frac(treats_var_as_p)[1]]),
    )


@generator(
    "p12d_bernoulli_model",
    level=14,
    presets={
        "easy": dict(kind="easy"),
        "medium": dict(kind="medium"),
        "hard": dict(kind="hard"),
        "exam": dict(kind="exam"),
    },
)
def p12d_bernoulli_model(rng, kind="easy"):
    """ACMMM143: use a Bernoulli random variable as a model -- derive p from
    an equally-likely-outcomes description (never handed p directly), then
    compute E[X] (easy), Var[X] (medium), or invert Var[X] -> p with a
    contextual less/more-common clue (hard/exam). See module docstring."""
    if kind == "easy":
        return _model_easy(rng)
    if kind == "medium":
        return _model_medium(rng)
    if kind == "hard":
        return _model_hard(rng)
    if kind == "exam":
        return _model_exam(rng)
    raise ValueError(f"unknown kind {kind!r}")


# ============================================================================
# ACMMM146 -- Bernoulli SEQUENCE probability (narrowed kernel; never the
# expected value, which belongs to p12d_bernoulli_model above, and never a
# binomial "x successes in n trials" count, which belongs to
# binomial-point-probability)
# ============================================================================

# Three structurally different scenario templates (spinner / weighted coin /
# faulty-item inspection), each with a "state label_a's probability" phrasing
# (easy/medium: p given directly) and a "state label_b's probability"
# phrasing (hard/exam: the failure-probability trap).
_SEQ_CONTEXTS = [
    dict(
        label_a="win",
        label_b="loss",
        unit="spin",
        unit_plural="spins",
        stmt_a="A spinner lands on 'win' with probability {p} on each independent spin.",
        stmt_b="A spinner lands on 'loss' with probability {p} on each independent spin.",
        n_stmt="The spinner is spun {n} times.",
    ),
    dict(
        label_a="heads",
        label_b="tails",
        unit="toss",
        unit_plural="tosses",
        stmt_a="A biased coin lands heads with probability {p} on each independent toss.",
        stmt_b="A biased coin lands tails with probability {p} on each independent toss.",
        n_stmt="The coin is tossed {n} times.",
    ),
    dict(
        label_a="faulty",
        label_b="ok",
        unit="item",
        unit_plural="items",
        stmt_a="Each item leaving the production line is independently faulty with probability {p}.",
        stmt_b="Each item leaving the production line is independently free of faults (ok) with probability {p}.",
        n_stmt="{n} items are inspected in sequence.",
    ),
]


def _draw_sequence(rng, n):
    """A length-n sequence of 'a'/'b' flags, never all-same (excludes the
    all-success/all-failure sequence, on which the treats-as-unordered
    binomial-coefficient distractor would collide with the answer since
    C(n, 0) = C(n, n) = 1)."""
    for _attempt in range(_MAX_ATTEMPTS):
        flags = [rng.choice("ab") for _ in range(n)]
        if len(set(flags)) == 2:
            return flags
    raise RuntimeError("could not draw a mixed-outcome sequence for p12d sequence")


def _plausible_faulty_rate(ctx, p_a):
    """A production line where the majority of items are defective is
    arithmetically fine but not a plausible scenario ("a
    figure must be constructible AND sensible"): for the inspection
    context specifically (label_a = "faulty"), require the defect rate
    P(faulty) = p_a to stay at or below one half. Every other context
    (spinner, coin) has no real-world plausibility ceiling on its bias."""
    if ctx["label_a"] == "faulty":
        return p_a <= Fraction(1, 2)
    return True


def _factor(flag, p_a):
    return p_a if flag == "a" else (1 - p_a)


def _sequence_answer(flags, p_a):
    """Construction route: accumulate the running product in a loop while
    also building the label sequence text (the SAME loop the caller uses
    for text construction) -- see each tier's own caller."""
    product = Fraction(1)
    for flag in flags:
        product *= _factor(flag, p_a)
    return product


def _sequence_text(ctx, flags, stated_p, use_stmt_b, n, *, as_fraction=False):
    labels = [ctx["label_a"] if f == "a" else ctx["label_b"] for f in flags]
    joined = " then ".join(labels)
    stmt = (ctx["stmt_b"] if use_stmt_b else ctx["stmt_a"]).format(p=stated_p)
    n_stmt = ctx["n_stmt"].format(n=n)
    ask = "Find the probability, as a fraction," if as_fraction else "Find the probability"
    return f"{stmt} {n_stmt} {ask} that the sequence of outcomes, in order, is: {joined}."


def _seq_distractors(flags, p_a, answer, *, include_swap):
    n = len(flags)
    count_a = flags.count("a")
    treats_as_unordered = answer * math.comb(n, count_a)
    # Independent-of-construction corruption route lives in the test file's
    # own oracle check; here we just build the misconception value directly.
    drops_one_factor = Fraction(1)
    for flag in flags[:-1]:
        drops_one_factor *= _factor(flag, p_a)
    candidates = [treats_as_unordered, drops_one_factor]
    swap_value = None
    if include_swap:
        swap_value = Fraction(1)
        for flag in flags:
            # Swapped assignment: p_a used where 1-p_a was correct, and vice
            # versa -- literally the opposite of `_factor`.
            swap_value *= (1 - p_a) if flag == "a" else p_a
        candidates.append(swap_value)
    return treats_as_unordered, drops_one_factor, swap_value, candidates


def _seq_easy(rng):
    n = 2
    for _attempt in range(_MAX_ATTEMPTS):
        ctx = rng.choice(_SEQ_CONTEXTS)
        p_a = rng.choice(_P_POOL_WIDE)
        check(0 < p_a < 1, "p must be strictly between 0 and 1")
        if not _plausible_faulty_rate(ctx, p_a):
            continue
        flags = _draw_sequence(rng, n)
        answer = _sequence_answer(flags, p_a)
        # Independent check: build the factor list via a SEPARATE list
        # comprehension (never the accumulation loop above) and multiply
        # with math.prod.
        factors_independent = [_factor(f, p_a) for f in flags]
        check_equivalent(
            answer,
            math.prod(factors_independent),
            "sequence product disagrees with the independently-built factor list",
        )

        treats_as_unordered, drops_one_factor, _swap, candidates = _seq_distractors(
            flags, p_a, answer, include_swap=False
        )
        try:
            check_pairwise_distinct(answer, *candidates, msg="a sequence-easy distractor collides")
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a non-colliding scenario for p12d sequence easy")

    p_dict, p_text = _frac(p_a)
    text = _sequence_text(ctx, flags, p_text, use_stmt_b=False, n=n)
    answer_dict, answer_text = _frac(answer)
    return dict(
        text=text,
        answer=answer_dict,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"P = {' × '.join(_frac(_factor(f, p_a))[1] for f in flags)} = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, [_frac(treats_as_unordered)[1], _frac(drops_one_factor)[1]]),
    )


def _seq_medium(rng):
    n = 3
    for _attempt in range(_MAX_ATTEMPTS):
        ctx = rng.choice(_SEQ_CONTEXTS)
        p_a = rng.choice(_P_POOL_WIDE)
        check(0 < p_a < 1, "p must be strictly between 0 and 1")
        if not _plausible_faulty_rate(ctx, p_a):
            continue
        flags = _draw_sequence(rng, n)
        answer = _sequence_answer(flags, p_a)
        factors_independent = [_factor(f, p_a) for f in flags]
        check_equivalent(
            answer,
            math.prod(factors_independent),
            "sequence product disagrees with the independently-built factor list",
        )

        treats_as_unordered, drops_one_factor, _swap, candidates = _seq_distractors(
            flags, p_a, answer, include_swap=False
        )
        try:
            check_pairwise_distinct(answer, *candidates, msg="a sequence-medium distractor collides")
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a non-colliding scenario for p12d sequence medium")

    p_dict, p_text = _frac(p_a)
    text = _sequence_text(ctx, flags, p_text, use_stmt_b=False, n=n)
    answer_dict, answer_text = _frac(answer)
    return dict(
        text=text,
        answer=answer_dict,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"P = {' × '.join(_frac(_factor(f, p_a))[1] for f in flags)} = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, [_frac(treats_as_unordered)[1], _frac(drops_one_factor)[1]]),
    )


def _seq_hard(rng):
    """The FAILURE-probability trap, forced every seed: q (not p) is stated
    directly, for label_b. The learner must identify p = 1 - q for label_a
    positions and use q directly for label_b positions."""
    for _attempt in range(_MAX_ATTEMPTS):
        n = rng.choice([4, 5])
        ctx = rng.choice(_SEQ_CONTEXTS)
        q_a = rng.choice(_P_POOL_NO_HALF)  # stated probability of label_b
        p_a = 1 - q_a  # probability of label_a, derived
        check(p_a != q_a, "hard-tier p and q must never coincide (p = 1/2 excluded)")
        if not _plausible_faulty_rate(ctx, p_a):
            continue
        flags = _draw_sequence(rng, n)
        answer = _sequence_answer(flags, p_a)
        factors_independent = [_factor(f, p_a) for f in flags]
        check_equivalent(
            answer,
            math.prod(factors_independent),
            "sequence product disagrees with the independently-built factor list",
        )

        treats_as_unordered, drops_one_factor, swap_value, candidates = _seq_distractors(
            flags, p_a, answer, include_swap=True
        )
        try:
            check_pairwise_distinct(answer, *candidates, msg="a sequence-hard distractor collides")
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a non-colliding scenario for p12d sequence hard")

    q_dict, q_text = _frac(q_a)
    text = _sequence_text(ctx, flags, q_text, use_stmt_b=True, n=n)
    answer_dict, answer_text = _frac(answer)
    return dict(
        text=text,
        answer=answer_dict,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"P({ctx['label_a']}) = 1 - {q_text} = {_frac(p_a)[1]}; P({ctx['label_b']}) = {q_text}.",
            f"P = {' × '.join(_frac(_factor(f, p_a))[1] for f in flags)} = {answer_text}.",
        ],
        distractors=pick_distractors(
            answer_text, [_frac(treats_as_unordered)[1], _frac(drops_one_factor)[1], _frac(swap_value)[1]]
        ),
    )


def _seq_exam(rng):
    """Exam tier: genuine escalation over hard -- the FAILURE probability is
    given as a DECIMAL (forcing an exact-fraction conversion first), and
    hard's failure-probability trap is retained (label_b's probability is
    what is stated, not label_a's)."""
    for _attempt in range(_MAX_ATTEMPTS):
        n = rng.choice([5, 6])
        ctx = rng.choice(_SEQ_CONTEXTS)
        digit = rng.choice(_DECIMAL_DIGITS_NO_HALF)
        q_a = Fraction(digit, 10)
        p_a = 1 - q_a
        check(p_a != q_a, "exam-tier p and q must never coincide (digit 5 excluded)")
        if not _plausible_faulty_rate(ctx, p_a):
            continue
        flags = _draw_sequence(rng, n)
        answer = _sequence_answer(flags, p_a)
        factors_independent = [_factor(f, p_a) for f in flags]
        check_equivalent(
            answer,
            math.prod(factors_independent),
            "sequence product disagrees with the independently-built factor list",
        )

        treats_as_unordered, drops_one_factor, swap_value, candidates = _seq_distractors(
            flags, p_a, answer, include_swap=True
        )
        try:
            check_pairwise_distinct(answer, *candidates, msg="a sequence-exam distractor collides")
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a non-colliding scenario for p12d sequence exam")

    q_text = f"0.{digit}"
    text = _sequence_text(ctx, flags, q_text, use_stmt_b=True, n=n, as_fraction=True)
    answer_dict, answer_text = _frac(answer)
    return dict(
        text=text,
        answer=answer_dict,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"P({ctx['label_a']}) = 1 - {q_text} = {_frac(p_a)[1]}; P({ctx['label_b']}) = {q_text}.",
            f"P = {' × '.join(_frac(_factor(f, p_a))[1] for f in flags)} = {answer_text}.",
        ],
        distractors=pick_distractors(
            answer_text, [_frac(treats_as_unordered)[1], _frac(drops_one_factor)[1], _frac(swap_value)[1]]
        ),
    )


@generator(
    "p12d_bernoulli_sequence",
    level=14,
    presets={
        "easy": dict(kind="easy"),
        "medium": dict(kind="medium"),
        "hard": dict(kind="hard"),
        "exam": dict(kind="exam"),
    },
)
def p12d_bernoulli_sequence(rng, kind="easy"):
    """ACMMM146 (narrowed kernel -- sequence probability only): p (easy/
    medium) or the FAILURE probability q (hard/exam, a trap) is GIVEN
    directly; find the probability of one specific, stated, ORDERED
    sequence of independent Bernoulli trials -- a plain product, never a
    binomial-coefficient count. See module docstring."""
    if kind == "easy":
        return _seq_easy(rng)
    if kind == "medium":
        return _seq_medium(rng)
    if kind == "hard":
        return _seq_hard(rng)
    if kind == "exam":
        return _seq_exam(rng)
    raise ValueError(f"unknown kind {kind!r}")
