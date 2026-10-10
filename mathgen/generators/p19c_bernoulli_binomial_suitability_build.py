"""Generates `p19c_binomial_suitability` and `p19c_bernoulli_suitability` questions: decide whether a described
context suits a binomial or Bernoulli model.

Bernoulli vs. binomial context-suitability classification.

- `p19c_binomial_suitability` -- `senior-identify-contexts-suitable-for-modelling-by-
  binomial-random` (ACMMM148). Given a described random-phenomenon context, classify
  whether it is suited to a binomial model -- a FIXED number of independent trials, each
  with the same two possible outcomes and the same success probability -- or, if not,
  which single one of those four conditions it fails.
- `p19c_bernoulli_suitability` -- `senior-identify-contexts-suitable-for-modelling-by-
  bernoulli-random` (ACMMM144). Given a described SINGLE trial/draw/measurement, classify
  whether it is suited to a Bernoulli model -- one trial, two possible outcomes, a fixed
  probability of success -- or, if not, which of three named reasons it fails.

THE SHARP BOUNDARY (stated explicitly in every rendered scenario of BOTH topics): a binomial-suited scenario always names a fixed repeat count ("n trials", "40
customers") -- never a scenario missing its trial count is offered as the SUITED case. A
Bernoulli scenario always describes exactly ONE trial/draw/measurement -- never a scenario
that reads as "the first of many trials" without saying so. The one class where a
binomial-topic scenario deliberately has NO fixed trial count (`p19c_binomial_suitability`'s
"not suited -- the number of trials is not fixed" class) is the exception the classification
itself is testing, not a violation of this rule: a genuinely SUITED binomial scenario always
states its count; a scenario built to fail the fixed-count condition necessarily cannot.

Both distinct from the shipping `senior-use-a-bernoulli-random-variable-as-a-model`
(topic `p12d_bernoulli_model`) and `senior-use-bernoulli-random-variables-and-associated-
probabilities-to`, which take the model as ALREADY CHOSEN and compute E(X)/Var(X) or a
sequence probability -- never asking the suitability question itself; and from
`binomial-point-probability`, which takes n, p, x as GIVEN and computes nCx p^x (1-p)^(n-x).
This pair never computes a value -- it stops at the prior recognition step.
"""

from __future__ import annotations

from ..core import check, generator, pick_distractors

# ============================================================================
# Skill 1: binomial suitability -- fixed 5-way option set, every tier
# ============================================================================

OPT_BINOMIAL_SUITED = "Suited to a binomial model"
OPT_BINOMIAL_NOT_FIXED = "Not suited -- the number of trials is not fixed"
OPT_BINOMIAL_NOT_INDEP = "Not suited -- the trials are not independent"
OPT_BINOMIAL_NOT_CONST_P = "Not suited -- the success probability is not constant across trials"
OPT_BINOMIAL_NOT_TWO = "Not suited -- each trial does not have exactly two possible outcomes"
OPTIONS_BINOMIAL = (
    OPT_BINOMIAL_SUITED,
    OPT_BINOMIAL_NOT_FIXED,
    OPT_BINOMIAL_NOT_INDEP,
    OPT_BINOMIAL_NOT_CONST_P,
    OPT_BINOMIAL_NOT_TWO,
)

# Construct-to-target weights: class 1 (genuinely suited) lands on ~30% of draws,
# comfortably inside the composition's 25-40% band; the remaining four violation
# classes split the rest evenly.
_BINOMIAL_WEIGHTS = {
    OPT_BINOMIAL_SUITED: 30,
    # Nudged off an exact 17.5 split so every key label lands in 15-40% on each tier's 200-seed
    # sweep (the equal split left one label at 14% on medium by draw noise).
    OPT_BINOMIAL_NOT_FIXED: 18.2,
    OPT_BINOMIAL_NOT_INDEP: 16.3,
    OPT_BINOMIAL_NOT_CONST_P: 18.2,
    OPT_BINOMIAL_NOT_TWO: 17.3,
}

# ============================================================================
# Skill 2: Bernoulli suitability -- fixed 4-way option set, every tier
# ============================================================================

OPT_BERNOULLI_SUITED = "Suited to a Bernoulli model"
OPT_BERNOULLI_IS_BINOMIAL = (
    "Not suited -- this describes a FIXED NUMBER of repeated trials (a binomial setup), not a single trial"
)
OPT_BERNOULLI_CONTINUOUS = "Not suited -- the outcome is a continuous measurement, not one of two discrete categories"
OPT_BERNOULLI_DEPENDENT = (
    "Not suited -- the probability of success for this single draw is not a fixed known "
    "value (items already removed, count unstated)"
)
OPTIONS_BERNOULLI = (
    OPT_BERNOULLI_SUITED,
    OPT_BERNOULLI_IS_BINOMIAL,
    OPT_BERNOULLI_CONTINUOUS,
    OPT_BERNOULLI_DEPENDENT,
)

# Construct-to-target: class 1 (genuinely suited) at ~34%, comfortably under the
# composition's 40% ceiling; the three distractor classes split the rest evenly.
_BERNOULLI_WEIGHTS = {
    OPT_BERNOULLI_SUITED: 34,
    OPT_BERNOULLI_IS_BINOMIAL: 22,
    OPT_BERNOULLI_CONTINUOUS: 22,
    OPT_BERNOULLI_DEPENDENT: 22,
}

# ============================================================================
# Shared scenario furniture (prompt variety; masked by prompt_shape())
# ============================================================================

# `removable`: True if this context's items can plausibly be "already removed without replacement
# from a population" (the Bernoulli DEPENDENT class's mechanic) -- a free throw cannot be un-thrown,
# so that context is excluded from DEPENDENT (and from the SUITED exam-tier "already removed"
# disguise, for the same reason). `continuous_measure`: a natural continuous measurement phrase for
# this item (used verbatim by the Bernoulli CONTINUOUS class and by the hard/exam SUITED recode
# step), or None if the item has no plausible one (a raffle ticket has no natural continuous
# measurement). `staffed`: True if this context is naturally staffed by workers whose experience
# level plausibly affects the outcome (so the "novice workers"/"experienced workers" group-pair
# phrasing in `_clause_probability`'s constant-probability-violation branch fits the stated
# subject); False if the subject is the item being studied itself, with no natural worker
# distinction (a raffle ticket or a germinating seed cannot plausibly be a "novice" or "experienced"
# worker). `supply`: True if "drawn with replacement from a large supply" reads naturally. `rated`: True if
# a k-point rating scale is a natural way to record this item's outcome. `production`: True only for
# a production-line context, the only place "the same quality inspector" belongs.
_CONTEXTS = (
    dict(
        subject="a factory quality-control line",
        item="component",
        items="components",
        removable=True,
        continuous_measure="its diameter in millimetres",
        staffed=True,
        supply=True,
        rated=True,
        production=True,
    ),
    dict(
        subject="a telemarketing campaign",
        item="call",
        items="calls",
        removable=True,
        continuous_measure="its duration in seconds",
        staffed=True,
        supply=False,
        rated=True,
        production=False,
    ),
    dict(
        subject="a plant nursery",
        item="seed",
        items="seeds",
        removable=True,
        continuous_measure="its height in centimetres after germination",
        # A seed is not given a k-point rating.
        staffed=False,
        supply=True,
        rated=False,
        production=False,
    ),
    dict(
        subject="a basketball free-throw session",
        item="free throw",
        items="free throws",
        removable=False,
        continuous_measure="the time the ball spends in the air, in seconds",
        staffed=False,
        # A free throw is not "drawn from a supply" and is not rated on a points scale; a quality
        # inspector does not attend a basketball session.
        supply=False,
        rated=False,
        production=False,
    ),
    dict(
        subject="a school raffle",
        item="ticket",
        items="tickets",
        removable=True,
        continuous_measure="its mass in grams",
        staffed=False,
        supply=True,
        rated=False,
        production=False,
    ),
)

_GROUP_PAIRS = (
    ("novice workers", "experienced workers"),
    ("Monday's shift", "Friday's shift"),
    ("Group A", "Group B"),
    ("the morning session", "the afternoon session"),
)

# Every filler must bear on none of the binomial conditions. "Announces the result out loud" can
# read as breaking independence; "a single afternoon" collides with the "morning/afternoon session"
# group pair; "quality inspector" fits only a production line.
_REDHERRINGS = (
    "Every trial is recorded in the same spreadsheet.",
    "All of the work takes place in the same building.",
    "Everyone involved wears the same uniform colour.",
)
_PRODUCTION_REDHERRING = "The tests are all carried out by the same quality inspector."


def _pick_filler(rng, ctx):
    pool = list(_REDHERRINGS)
    if ctx["production"]:
        pool.append(_PRODUCTION_REDHERRING)
    return rng.choice(pool)


def _clause_trial_count(n, violated):
    if not violated:
        return f"There are exactly {n} trials, fixed in advance."
    return "This carries on as many times as it takes, until the first success."


def _clause_independence(rng, n, violated, *, exam_convention=False, ctx=None):
    """Return (clause text, population size or None)."""
    if not violated:
        if exam_convention:
            pop = n * rng.randint(30, 60)
            return (
                f"The {n} trials are drawn without replacement from a large population of "
                f"{pop} (much larger than the sample of {n}). Because the population is more "
                f"than 20 times the sample, treat the draws as independent.",
                pop,
            )
        if ctx is None or ctx["supply"]:
            return (
                "Each trial is independent of the others, drawn with replacement from a large supply.",
                None,
            )
        return (
            f"Each trial is independent of the others: the result for one {ctx['item']} has no effect on any other.",
            None,
        )
    pop = n + rng.randint(1, max(2, n // 2))
    return (
        f"The {n} trials are drawn without replacement from a small population of only {pop} "
        f"(comparable to the sample size of {n}).",
        pop,
    )


def _clause_start_share(rng, pop, items):
    """The starting success share of a small population, from an actual integer K of N.

    A percentage is stated only when it lies within 0.5 percentage points of K/N (so some whole count in the
    population really produces it); otherwise, or for variety, the count itself is stated. Deliberately avoids
    "on the first draw", which hinted that the probability then changes."""
    lo, hi = -(-pop // 10), (pop * 9) // 10
    k = rng.randint(lo, hi)
    exact = 100 * k / pop
    pct = round(exact)
    if abs(pct - exact) <= 0.5 and rng.random() < 0.7:
        return f"At the start, the probability that a draw is a success is {pct}%."
    return f"At the start, {k} of the {pop} {items} count as successes."


def _clause_probability(rng, p, violated, *, first_draw_only=False, staffed=True, pop=None, items="items"):
    if not violated:
        if first_draw_only:
            # "Not independent" class: asserting a probability fixed "on every trial" would
            # contradict the without-replacement clause, so state only the starting share.
            return _clause_start_share(rng, pop, items)
        return f"The probability of success is {p}% on every trial."
    # The "novice workers"/"experienced workers" pair is a staffing distinction: it only
    # fits contexts naturally staffed by workers. For an unstaffed context
    # (the subject is the item being studied, not a worker), draw only from the three
    # context-neutral pairs, which fit any subject.
    pairs = _GROUP_PAIRS if staffed else _GROUP_PAIRS[1:]
    g1, g2 = rng.choice(pairs)
    p1 = p
    delta = rng.choice([-1, 1]) * rng.randint(15, 30)
    p2 = max(5, min(95, p1 + delta))
    check(p1 != p2, "constant-probability violation must state two different percentages")
    return f"The probability of success is {p1}% for {g1} and {p2}% for {g2}, and the trials are a mix of both."


def _clause_outcome_binomial(rng, violated, *, tier="medium", ctx=None):
    rated = ctx is None or ctx["rated"]
    measure = None if ctx is None else ctx["continuous_measure"]
    if not violated:
        if tier in ("hard", "exam") and not rated and measure:
            return (
                f"Each trial's outcome is recorded as whether {measure} exceeds a fixed "
                f"threshold set in advance (one of exactly two outcomes), not the raw "
                f"measurement itself."
            )
        if tier in ("hard", "exam") and rated:
            # Hard/exam reasoning step: recode a multi-way rating down to two categories in
            # the question's own framing. This genuinely IS a two-outcome recording -- the
            # contrast with the NOT_TWO class below (which counts the raw multi-way rating
            # itself, uncoded) is the real distinction the learner must reason through.
            k = rng.randint(4, 6)
            cutoff = rng.randint(3, k - 1)
            return (
                f"Each trial's outcome is recorded as whether a {k}-point rating comes in at "
                f"{cutoff} or above (one outcome) or below {cutoff} (the other outcome), not "
                f"the raw rating itself."
            )
        return "Each trial's outcome is recorded as one of exactly two possible outcomes: success or failure."
    kind = rng.choice(["scale", "continuous"]) if rated else "continuous"
    if kind == "scale":
        k = rng.randint(3, 6)
        return f"Each trial's outcome is recorded as one of {k} categories on a rating scale."
    example = f" (for example, {measure})" if measure else ""
    return f"Each trial's outcome is recorded as a value on a continuous scale{example}."


def _draw_class(rng, weights):
    classes = list(weights)
    return rng.choices(classes, weights=list(weights.values()), k=1)[0]


def _build_binomial_scenario(rng, tier):
    cls = _draw_class(rng, _BINOMIAL_WEIGHTS)
    exam_pop_trick = tier == "exam" and cls in (OPT_BINOMIAL_SUITED, OPT_BINOMIAL_NOT_INDEP)
    # A small-population / large-population "without replacement" draw needs items that can
    # actually be removed; a free throw cannot be.
    pool = (
        [c for c in _CONTEXTS if c["removable"]]
        if (cls == OPT_BINOMIAL_NOT_INDEP or exam_pop_trick)
        else list(_CONTEXTS)
    )
    ctx = rng.choice(pool)
    p = rng.randint(10, 90)

    if cls == OPT_BINOMIAL_NOT_FIXED:
        n = None
        intro = f"At {ctx['subject']}, {ctx['items']} are studied one after another."
    else:
        n = rng.randint(8, 40)
        intro = f"At {ctx['subject']}, {n} {ctx['items']} are studied one after another."

    if exam_pop_trick:
        # Exam-tier escalation: the without-replacement wording is shared between the
        # genuinely-suited (large population, negligible effect) and the genuinely-violated
        # (small population, real dependence) case -- the learner must judge the
        # population-to-sample RATIO, not pattern-match on the words "without replacement".
        indep_clause, pop = _clause_independence(
            rng,
            n,
            cls == OPT_BINOMIAL_NOT_INDEP,
            exam_convention=(cls == OPT_BINOMIAL_SUITED),
            ctx=ctx,
        )
    else:
        indep_clause, pop = _clause_independence(
            rng, n if n is not None else 20, cls == OPT_BINOMIAL_NOT_INDEP, ctx=ctx
        )

    clauses = [
        _clause_trial_count(n if n is not None else 0, cls == OPT_BINOMIAL_NOT_FIXED),
        indep_clause,
        _clause_probability(
            rng,
            p,
            cls == OPT_BINOMIAL_NOT_CONST_P,
            first_draw_only=(cls == OPT_BINOMIAL_NOT_INDEP),
            staffed=ctx["staffed"],
            pop=pop,
            items=ctx["items"],
        ),
        _clause_outcome_binomial(rng, cls == OPT_BINOMIAL_NOT_TWO, tier=tier, ctx=ctx),
    ]

    if tier == "hard" or (tier == "exam" and not exam_pop_trick):
        clauses.append(_pick_filler(rng, ctx))

    scenario = intro + " " + " ".join(clauses) + " Which of the following is true?"
    return scenario, cls


@generator(
    "p19c_binomial_suitability",
    level=14,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19c_binomial_suitability(rng, tier="medium"):
    """Classify whether a described context is suited to a binomial model -- a fixed number
    of independent trials, each with the same two outcomes and the same success probability --
    or, if not, which single one of those four conditions it fails. Never computes a
    probability; stops at the modelling-recognition step (boundary against
    `binomial-point-probability`, which takes n/p/x as given)."""
    text, cls = _build_binomial_scenario(rng, tier)
    check(cls in OPTIONS_BINOMIAL, f"drawn class {cls!r} not a recognised option")
    others = [o for o in OPTIONS_BINOMIAL if o != cls]
    if cls == OPT_BINOMIAL_NOT_INDEP:
        # The violated-independence clause states only "drawn without replacement from a
        # small population" (no claim about the probability changing), so NOT_CONST_P
        # ("the success probability is not constant") is never a defensible second answer
        # for this item -- exclude it from the pool entirely rather than merely deprioritise
        # it, so it is never offered as a distractor for this class.
        others = [o for o in others if o != OPT_BINOMIAL_NOT_CONST_P]
    rng.shuffle(others)
    distractors = pick_distractors(cls, others, n=3)
    check(len(distractors) == 3, f"expected 3 distractors, got {len(distractors)}")
    if cls == OPT_BINOMIAL_NOT_INDEP:
        check(
            OPT_BINOMIAL_NOT_CONST_P not in distractors,
            "NOT_INDEP item must never offer NOT_CONST_P as an option",
        )
    # These are fixed literal option strings (never numeric/symbolic), so plain string
    # distinctness is the right check -- `check_pairwise_distinct`/`symbolically_equal`
    # are for numeric or symbolic answers and raise a sympify error on plain text.
    check(len({cls, *distractors}) == 4, "binomial suitability options must be pairwise distinct")

    return dict(
        text=text,
        latex=r"\text{Binomial suitability}",
        answer=cls,
        answer_text=cls,
        distractors=distractors,
    )


# ============================================================================
# Skill 2 construction
# ============================================================================


def _build_bernoulli_scenario(rng, tier):
    cls = _draw_class(rng, _BERNOULLI_WEIGHTS)
    p = rng.randint(10, 90)
    n_repeat = rng.randint(15, 60)

    # Exam-tier escalation: dressed in the same "already removed without replacement"
    # language as the DEPENDENT distractor, but the remaining population is large enough
    # that the effect is negligible -- the learner must judge the ratio, not pattern-match
    # on "already removed". Decided BEFORE the context is drawn, so the context pool is only
    # narrowed to removable contexts when this branch will actually use removal language.
    use_trick = cls == OPT_BERNOULLI_SUITED and tier == "exam" and rng.random() < 0.5
    # Hard/exam reasoning step (non-trick SUITED only): recode a continuous measurement down
    # to two categories via a threshold, in the question's own framing -- genuinely a
    # two-outcome recording, contrasted with the CONTINUOUS class which records the raw
    # measurement itself uncoded.
    use_recode = cls == OPT_BERNOULLI_SUITED and tier in ("hard", "exam") and not use_trick

    if cls == OPT_BERNOULLI_DEPENDENT or use_trick:
        pool = [c for c in _CONTEXTS if c["removable"]]
    elif cls == OPT_BERNOULLI_CONTINUOUS or use_recode:
        pool = [c for c in _CONTEXTS if c["continuous_measure"]]
    else:
        pool = list(_CONTEXTS)
    ctx = rng.choice(pool)

    if cls == OPT_BERNOULLI_IS_BINOMIAL:
        intro = f"At {ctx['subject']}, one {ctx['item']} is checked."
        clauses = [
            f"This is repeated {n_repeat} times, and we count how many of the {n_repeat} attempts are successful."
        ]
    else:
        intro = f"At {ctx['subject']}, one {ctx['item']} is selected at random and checked once."
        if cls == OPT_BERNOULLI_CONTINUOUS:
            clauses = [f"The outcome recorded is {ctx['continuous_measure']}, a value on a continuous scale."]
        elif cls == OPT_BERNOULLI_DEPENDENT:
            small_pop = rng.randint(4, 12)
            clauses = [
                f"Several {ctx['items']} have already been removed without replacement from a "
                f"small population of only {small_pop}, but the number removed before this "
                f"draw has not been stated."
            ]
        else:  # OPT_BERNOULLI_SUITED
            if use_trick:
                large_pop = rng.randint(200, 500)
                clauses = [
                    f"Several {ctx['items']} have already been removed without replacement from "
                    f"a large population of {large_pop}, and the probability of success for "
                    f"this one draw is {p}%."
                ]
            elif use_recode:
                clauses = [
                    f"The outcome recorded is whether {ctx['continuous_measure']} exceeds a "
                    f"fixed threshold set in advance (one of exactly two categories, not the "
                    f"raw measurement itself), and the probability of success for this draw "
                    f"is {p}%, fixed and known in advance."
                ]
            else:
                clauses = [f"The outcome is recorded as one of exactly two categories: success or failure."]
                clauses.append(f"The probability of success for this draw is {p}%, fixed and known in advance.")

    if tier == "hard":
        clauses.append(_pick_filler(rng, ctx))

    scenario = intro + " " + " ".join(clauses) + " Which of the following is true?"
    return scenario, cls


@generator(
    "p19c_bernoulli_suitability",
    level=14,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19c_bernoulli_suitability(rng, tier="medium"):
    """Classify whether a described SINGLE trial/draw/measurement is suited to a Bernoulli
    model -- one trial, exactly two possible outcomes, a fixed probability of success -- or,
    if not, which of three named reasons it fails: it is actually a fixed-repeat-count
    binomial setup; the outcome is a continuous measurement; or an unstated number of prior
    removals means this single trial's own probability of success cannot be pinned down as
    a single fixed known value (never framed as the single trial itself being "not
    independent" -- independence is a multi-trial concept that does not honestly apply to
    one trial). Never computes E(X)/Var(X); stops at the
    modelling-recognition step (boundary against `senior-use-a-bernoulli-random-variable-
    as-a-model`, which takes the model as already chosen)."""
    text, cls = _build_bernoulli_scenario(rng, tier)
    check(cls in OPTIONS_BERNOULLI, f"drawn class {cls!r} not a recognised option")
    others = [o for o in OPTIONS_BERNOULLI if o != cls]
    distractors = pick_distractors(cls, others, n=3)
    check(len(distractors) == 3, f"expected 3 distractors, got {len(distractors)}")
    check(len({cls, *distractors}) == 4, "bernoulli suitability options must be pairwise distinct")

    return dict(
        text=text,
        latex=r"\text{Bernoulli suitability}",
        answer=cls,
        answer_text=cls,
        distractors=distractors,
    )
