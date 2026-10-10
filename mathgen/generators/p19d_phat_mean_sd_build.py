"""The theoretical mean and standard deviation of the sampling distribution of the sample proportion p-hat
(ACMMM174, ``senior-understand-the-concept-of-the-sample-proportion-widehat``, ``response_family:
ordered_structures``, ``answer: pair``).

Given a population proportion p and a sample size n -- BOTH given directly
in the question, restated explicitly as "p = ..." and "n = ..." at the end
of every question's setup regardless of tier, exactly like a fully-specified
input -- report the pair (mean, standard deviation) of the sampling
distribution of p-hat, where mean = p exactly and standard deviation =
sqrt(p(1-p)/n).

**Boundary against every other shipping p-hat skill**: "Theoretical formula
substitution from a GIVEN true p, distinct from every shipping p-hat skill
which starts from an observed proportion or counts. Keep the distinction
explicit in the skill description to avoid drift toward computing a margin
of error." Concretely: this skill's question text never says a sample was
drawn and its proportion observed (that is
``senior-use-the-approximate-confidence-interval-left-widehat-p``,
``senior-the-concept-of-an-interval-estimate-for-a`` and
``senior-examine-the-approximate-normality-of-the-distribution-of``'s
territory -- all three start from an OBSERVED p-hat or raw counts and
compute a diagnostic/margin/interval). This skill's text always uses
future-tense framing ("is planned", "will sample") for a population
proportion p that is a GIVEN, fixed, true value, and it never computes a
margin of error, a z-value or an interval -- only the sampling distribution's
own two descriptive parameters.

**Grading-tolerance design.** The canonical answer_text is a LABELLED pair ("mean = 0.375, sd = 0.0645"), so
`_labelled_pair_verdict` engages and order is irrelevant (matches p8c_confidence_interval_build's own
dict-of-named-keys precedent). Once labelled, each component's tolerance is tightened from the flat 0.051 to
half a unit in ITS OWN displayed decimal place (mean: however many places p's exact terminating decimal needs,
never more than 3 here; sd: always 4dp) -- but every misconception distractor below is still verified, by
direct Fraction-based construction-time check, to clear the FULL 0.051 margin on whichever component actually
differs, the safe upper bound. On the easy and medium tiers, p is restricted to
hundredths/simple-fraction values in [0.25, 0.75] and n to [20, 45] specifically so that sd = sqrt(p(1-p)/n)
never drops below about 0.056 -- the point below which the omits-sqrt misconception (which reports the
variance p(1-p)/n as if it were the SD) would land within 0.051 of the true SD.

originally.** A 5th misconception candidate, ``sd_trap_n = sqrt(var /
wrong_n)``, plugs a wrong-but-present-in-the-question n (the pre-derivation
``base`` count in the hard tier, the OTHER study's n in the exam tier) into
the sd formula instead of the correctly-derived n. This only clears the
0.051 margin at a tighter p/n pool than the rest of the module: p in
[0.4, 0.6] (var >= 0.24) and n in [20, 30], with the wrong-n ratio
k = wrong_n / n in {9, 10, 12, 16} -- at the pool's worst case (p=0.4 or
0.6, n=30, k=9), gap = sqrt(0.24/30) * (1 - 1/sqrt(9)) ~= 0.0596,
comfortably clearing 0.051 after 4dp rounding. Verified exhaustively over
the whole tight pool (25 (p, n) pairs x 4 k values): the minimum gap
anywhere in the pool is that same 0.0596, so every combination qualifies
unconditionally -- there is no thin-pool problem here. Because
``pick_distractors`` only offers the first 4 of however many candidates it
is given (it dedupes against the correct answer, never between candidates),
a 5th candidate would otherwise be dead code -- on every trap-n tier,
`_finish`/`_finish_exam` shuffle the 5 candidate texts with the same
per-question `rng` before calling `pick_distractors`, so trap-n (like every
other misconception) is offered on most but not all draws, and which of the
5 gets dropped varies across seeds instead of it always being the same one.

1. *Sample bigger than the stated pool (hard/exam).* The hard/exam setup
   sentence described a SMALL finite chance-model pool (``total``, at most
   20) and then had the sample/inspection count (``base`` in hard, up to
   ``16 * 30 = 480``; the other study's n in exam, same range) drawn from
   the tight pool above -- routinely far exceeding ``total``, e.g.
   "a deck of 5 equally likely cards" followed by "inspect a batch of 400
   cards". Measured on the pre-fix module: 200/200 seeds violate this on
   200/200 figures came from a different, narrower extraction). Fixed via
   now explicitly reframed, in the very next sentence, as illustrative of a
   proportion that "holds across a very large supply of {unit}" -- the
   literal small container (spinner/bag/box/deck) still motivates the exact
   fraction p, but the sampled/inspected quantity is now explicitly drawn
   from the unbounded supply, never implied to come from the same handful
   of physical items. `test_sampled_count_never_exceeds_stated_pool_or_
   states_unbounded_supply` asserts this positively and unconditionally
   over 200 seeds on both tiers: every draw states the large-supply phrase
   (0 silent violations, 0 silent omissions).
2. *Medium's trap-n sd distractor.* The wide medium pool (`_NICE_FRACTIONS`
   at p in [0.25, 0.75], `_N_DERIVE`'s ratios, max k = 6) cannot clear
   GRADER_TOLERANCE for a trap-n candidate at its own worst case (p=0.25 or
   0.75, n=45): true sd ~= 0.0645, and even at the pool's largest k=6,
   gap = 0.0645 * (1 - 1/sqrt(6)) ~= 0.0382, under 0.051. The true
   break-even k at that worst case is k >= 23 (1 - 1/sqrt(23) ~= 0.7915 >
   0.051/0.0645 ~= 0.7908) -- far outside any one-step "contact base,
   include a phrase fraction" ratio the wide pool's variety was designed
   around, and reusing such a large k for every draw would erase medium's
   variety anyway. So medium now reuses hard/exam's tight pool outright
   with the same exhaustively-verified 0.0596 worst-case gap -- see the
   equals hard's; only the narrative framing still differs).
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import (
    GRADER_TOLERANCE,
    check,
    generator,
    latex_answer,
    pick_distractors,
    rounds_unambiguously,
    stated_precision_string,
)

_SD_PLACES = 4

# Easy tier only. p restricted to [0.25, 0.75] so p(1-p) >= 0.1875, and n restricted to [20, 45] so
# sd = sqrt(p(1-p)/n) >= sqrt(0.1875/45) ~= 0.0645 at the worst case -- see module docstring for why
# this specific floor matters.
_P_HUNDREDTHS = list(range(25, 76, 5))  # 0.25, 0.30, ..., 0.75
_N_RANGE = list(range(20, 46))  # easy tier only

# Simple fractions with denominators built only from 2s and 5s (so they terminate in decimal within
# 3 places), values inside [0.25, 0.75]. Unused (kept for reference/ possible future easy-tier
# reuse; _NICE_FRACTIONS/_N_DERIVE below are no longer read by any tier).
_NICE_FRACTIONS = [
    (1, 4),
    (3, 8),
    (2, 5),
    (9, 20),
    (1, 2),
    (11, 20),
    (3, 5),
    (5, 8),
    (7, 10),
    (3, 4),
]

# (base count, numerator, denominator, spelled-out fraction) for a one-step n-derivation, all
# landing inside [20, 45] and all denominators dividing the base exactly (an exact integer count,
# never a rounding). Unused (see _NICE_FRACTIONS comment above).
_N_DERIVE = [
    (80, 3, 8, "three-eighths"),
    (100, 1, 5, "one-fifth"),
    (60, 2, 3, "two-thirds"),
    (90, 1, 3, "one-third"),
    (120, 1, 4, "one-quarter"),
    (200, 1, 5, "one-fifth"),
    (160, 1, 4, "one-quarter"),
    (150, 1, 5, "one-fifth"),
    (240, 1, 6, "one-sixth"),
    (100, 2, 5, "two-fifths"),
]

# These are the _NICE_FRACTIONS entries that land in that range.
_P_TIGHT = [(2, 5), (9, 20), (1, 2), (11, 20), (3, 5)]

_N_DERIVE_TIGHT = [
    (180, 1, 9, "one-ninth"),
    (225, 1, 9, "one-ninth"),
    (270, 1, 9, "one-ninth"),
    (200, 1, 10, "one-tenth"),
    (250, 1, 10, "one-tenth"),
    (300, 1, 10, "one-tenth"),
    (240, 1, 12, "one-twelfth"),
    (288, 1, 12, "one-twelfth"),
    (360, 1, 12, "one-twelfth"),
    (320, 1, 16, "one-sixteenth"),
    (400, 1, 16, "one-sixteenth"),
    (480, 1, 16, "one-sixteenth"),
]

# Exam tier: n_target drawn directly from [20, 30], and the other study's n is k * n_target for a k.
_N_TIGHT_RANGE = list(range(20, 31))
_TRAP_K = [9, 10, 12, 16]

# Chance-model contexts for the hard/exam tiers: (thing, unit) pairs so the
# same _HARD_CHANCE (favourable, total) counts can describe a spinner's
# sectors, a bag's marbles, a box's components or a deck's cards -- the
# sample-size sentence names this same `unit`, never a generic noun.
_CHANCE_CONTEXTS = [
    ("a spinner", "sectors", "coloured red"),
    ("a bag of marbles", "marbles", "red"),
    ("a box of components", "components", "defective"),
    ("a deck of cards", "cards", "marked with a star"),
]

_TOPIC_CONTEXTS = [
    "who own a smartphone",
    "who recycle regularly",
    "who support the proposal",
    "who commute by public transport",
    "who prefer the new packaging",
]


def _dp_needed(frac: Fraction) -> int:
    """Decimal places a terminating `frac` needs to display exactly (the
    larger of the denominator's power-of-2 and power-of-5 exponents),
    raising via `check` if `frac` does not terminate in decimal at all."""
    d = frac.denominator
    twos = 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    fives = 0
    while d % 5 == 0:
        d //= 5
        fives += 1
    check(d == 1, f"{frac} does not terminate in decimal")
    return max(twos, fives, 1)


def _phat_pair(p: Fraction, n: int, wrong_n: int | None = None):
    """The verified (mean, sd) pair and its misconception candidates for a
    given true proportion `p` and sample size `n`. Returns None when the
    construction-time distractor-gap check fails (caller resamples).

    `wrong_n`, when given (hard/exam tiers only), is a plausible
    WRONG sample size actually present in the question text (the
    pre-derivation `base` count, or the other exam study's n) -- adds a 5th
    "trap-n" candidate, sqrt(var / wrong_n), on top of the usual four."""
    dp = max(_dp_needed(p), _SD_PLACES)
    mean_str = stated_precision_string(p, dp)
    p_f = float(p)
    var = p_f * (1 - p_f)
    sd_raw = math.sqrt(var / n)
    if not rounds_unambiguously(sd_raw, _SD_PLACES):
        return None
    sd_r = round(sd_raw, _SD_PLACES)
    sd_str = f"{sd_r:.{_SD_PLACES}f}"

    # Independent check: a genuinely different route from construction's
    # sqrt(var/n) -- divide by sqrt(n) as a *separate* sqrt rather than
    # inside the same radical.
    sd_independent = math.sqrt(var) / math.sqrt(n)
    check(
        abs(sd_independent - sd_raw) < 1e-9,
        f"sd identity failed: sqrt(var/n)={sd_raw} != sqrt(var)/sqrt(n)={sd_independent}",
    )

    # Misconceptions -- each a genuinely different (mean, sd) VALUE pair.
    sd_omits_n = round(math.sqrt(var), _SD_PLACES)  # (1) omits-n
    sd_omits_sqrt = round(var / n, _SD_PLACES)  # (2) omits-sqrt
    mean_as_variance = round(var, _SD_PLACES)  # (3) mean-as-variance
    # (4) swap: reports (sd, mean) instead of (mean, sd)
    candidates = [
        (p_f, sd_omits_n),
        (p_f, sd_omits_sqrt),
        (mean_as_variance, sd_r),
        (sd_r, p_f),
    ]
    if wrong_n is not None:
        # (5) trap-n: plugs a wrong-but-present-in-the-text n into the sd
        # formula instead of the correctly-derived n.
        sd_trap_n = round(math.sqrt(var / wrong_n), _SD_PLACES)
        candidates.append((p_f, sd_trap_n))

    def _gap_ok(mean_c: float, sd_c: float) -> bool:
        mean_gap = abs(Fraction(str(round(mean_c, 4))) - Fraction(str(round(p_f, 4))))
        sd_gap = abs(Fraction(str(round(sd_c, 4))) - Fraction(str(sd_r)))
        return mean_gap > GRADER_TOLERANCE or sd_gap > GRADER_TOLERANCE

    if not all(_gap_ok(mc, sc) for mc, sc in candidates):
        return None

    candidate_texts = [f"mean = {mc:.4f}, sd = {sc:.4f}" for mc, sc in candidates]
    answer_text = f"mean = {mean_str}, sd = {sd_str}"

    return dict(
        mean_str=mean_str,
        sd_str=sd_str,
        sd_r=sd_r,
        p_f=p_f,
        answer_text=answer_text,
        candidate_texts=candidate_texts,
    )


def _finish(draw_fn, instruction_fn, rng):
    """Redraw a full (p, n, setup, working_extra[, wrong_n]) tuple via
    `draw_fn` -- NOT just the pair-check -- on every attempt: `_phat_pair`
    is a pure function of (p, n, wrong_n), so retrying it alone on a fixed
    draw that failed the distractor-gap check would deterministically fail
    again 500 times. `draw_fn` returns a trailing `wrong_n` only when this question
    offers the trap-n distractor."""
    result = None
    p = n = setup = working_extra = instruction = wrong_n = None
    for _attempt in range(500):
        drawn = draw_fn()
        if len(drawn) == 5:
            p, n, setup, working_extra, wrong_n = drawn
        else:
            p, n, setup, working_extra = drawn
            wrong_n = None
        instruction = instruction_fn(p, n)
        result = _phat_pair(p, n, wrong_n=wrong_n)
        if result is not None:
            break
    else:
        raise RuntimeError("p19d_phat_mean_sd: construction failed to clear distractor gap in 500 attempts")
    answer_text = result["answer_text"]
    candidate_texts = list(result["candidate_texts"])
    if len(candidate_texts) > 4:
        # `pick_distractors` only offers the first 4 in list order (it dedupes against the answer,
        # never between candidates), so a 5th candidate would otherwise never be selected. Shuffle
        # with this question's own rng so which 4 of the 5 survive varies across seeds --
        # deterministic per seed, since `rng` is.
        rng.shuffle(candidate_texts)
    distractors = pick_distractors(answer_text, candidate_texts, n=4, tolerance=GRADER_TOLERANCE)
    check(len(distractors) == 4, "a distractor collided with the answer within grader tolerance")

    question_text = setup + " " + instruction
    working = list(working_extra) + [
        f"Mean of p-hat = p = {result['mean_str']}.",
        f"Standard deviation of p-hat = the square root of (p times (1 - p) divided by n) "
        f"= the square root of ({result['mean_str']} times "
        f"{stated_precision_string(1 - p, _dp_needed(1 - p))} divided by {n}) = {result['sd_str']}.",
    ]

    return {
        "text": question_text,
        "latex": f"plain:{question_text}",
        "answer": {"mean": result["p_f"], "sd": result["sd_r"]},
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text, plain=True),
        "distractors": distractors,
        "working": working,
    }


_INSTRUCTION = (
    "State the mean and the standard deviation of the sampling distribution of the "
    "sample proportion, p-hat, giving the standard deviation to 4 decimal places."
)


def _draw_easy(rng):
    p_hundredths = rng.choice(_P_HUNDREDTHS)
    p = Fraction(p_hundredths, 100)
    n = rng.choice(_N_RANGE)
    topic = rng.choice(_TOPIC_CONTEXTS)
    mean_str = stated_precision_string(p, _dp_needed(p))
    setup = (
        f"A large population has a true proportion of p = {mean_str} of people {topic}. "
        f"A random sample of n = {n} people is planned."
    )
    return p, n, setup, []


def _draw_medium(rng):
    # Medium now draws from the same tight _P_TIGHT/_N_DERIVE_TIGHT pool as hard/exam (was
    # _NICE_FRACTIONS/_N_DERIVE) so it can also carry the trap-n sd distractor --. Medium's own
    # framing already names "a large population" (never a small finite chance-model container), so
    # `base` (the pre-derivation contact count, returned as `wrong_n` below) never conflicts with a
    # stated finite pool the way hard/exam's chance model did -- that constraint does not apply to medium.
    num, den = rng.choice(_P_TIGHT)
    p = Fraction(num, den)
    mean_str = stated_precision_string(p, _dp_needed(p))
    base, dnum, dden, phrase = rng.choice(_N_DERIVE_TIGHT)
    n = base * dnum // dden
    topic = rng.choice(_TOPIC_CONTEXTS)
    setup = (
        f"In a large population, the true proportion of people {topic} is {num}/{den}, "
        f"so p = {mean_str}. A survey team plans to contact {base} potential respondents, "
        f"of whom only {phrase} will actually be included in the planned sample, "
        f"so n = {n}."
    )
    working_extra = [f"p = {num}/{den} = {mean_str}.", f"n = {base} times {phrase} = {n}."]
    return p, n, setup, working_extra, base


def _draw_hard(rng):
    fav, total = rng.choice(_P_TIGHT)
    p = Fraction(fav, total)
    mean_str = stated_precision_string(p, _dp_needed(p))
    thing, unit, colour_or_flaw = rng.choice(_CHANCE_CONTEXTS)
    base, dnum, dden, phrase = rng.choice(_N_DERIVE_TIGHT)
    n = base * dnum // dden
    fav_verb = "is" if fav == 1 else "are"
    # 20) only exists to justify p as an exact fraction -- the inspected
    # `base` (up to 480) is explicitly drawn from a separate, unbounded
    # "very large supply", never implied to come from the same handful of
    # physical sectors/marbles/components/cards. See module docstring
    setup = (
        f"Consider a very large supply of {unit}, of the kind found in {thing}: in every "
        f"{total} equally likely {unit}, {fav} {fav_verb} {colour_or_flaw}, so the true "
        f"proportion is p = {mean_str}. A quality-assurance team plans to "
        f"inspect a batch of {base} {unit} from that supply, of which only {phrase} will be "
        f"selected for the planned sample, so n = {n}."
    )
    working_extra = [
        f"p = {fav}/{total} = {mean_str}.",
        f"n = {base} times {phrase} = {n}.",
    ]
    return p, n, setup, working_extra, base


def _draw_exam(rng):
    fav, total = rng.choice(_P_TIGHT)
    p = Fraction(fav, total)
    mean_str = stated_precision_string(p, _dp_needed(p))
    thing, unit, colour_or_flaw = rng.choice(_CHANCE_CONTEXTS)
    n_target = rng.choice(_N_TIGHT_RANGE)
    k = rng.choice(_TRAP_K)
    other_n = k * n_target
    target = rng.choice(["A", "B"])
    n_a, n_b = (n_target, other_n) if target == "A" else (other_n, n_target)
    fav_verb = "is" if fav == 1 else "are"
    setup = (
        f"Consider a very large supply of {unit}, of the kind found in {thing}: in every "
        f"{total} equally likely {unit}, {fav} {fav_verb} {colour_or_flaw}, so the true "
        f"proportion is p = {mean_str}. A research team is planning two "
        f"independent studies of that supply. Study A will sample n = {n_a} {unit}. "
        f"Study B will sample n = {n_b} {unit}."
    )
    working_extra = [f"p = {fav}/{total} = {mean_str}.", f"Using Study {target}'s n = {n_target}."]
    # stash the target label on the closure via a mutable default -- read
    # back by the instruction_fn below through the returned setup string
    # itself (it always ends with the two "n = ..." sentences, so the
    # instruction only needs the label, carried separately).
    return p, n_target, setup, working_extra, target, other_n


def _exam_instruction(target):
    return (
        f"Report the mean and the standard deviation of the sampling distribution of the "
        f"sample proportion, p-hat, for Study {target}, giving the standard deviation to "
        f"4 decimal places."
    )


@generator(
    "p19d_phat_mean_sd",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19d_phat_mean_sd(rng, tier="easy"):
    """Understand the concept of the sample proportion p-hat as a random
    variable whose value varies between samples, and the formulas for the
    mean (p) and standard deviation (sqrt(p(1-p)/n)) of the sampling
    distribution of p-hat (ACMMM174)."""
    if tier == "easy":
        return _finish(lambda: _draw_easy(rng), lambda p, n: _INSTRUCTION, rng)
    if tier == "medium":
        return _finish(lambda: _draw_medium(rng), lambda p, n: _INSTRUCTION, rng)
    if tier == "hard":
        return _finish(lambda: _draw_hard(rng), lambda p, n: _INSTRUCTION, rng)
    if tier == "exam":
        return _finish_exam(rng)

    raise ValueError(f"unknown tier: {tier!r}")


def _finish_exam(rng):
    """Exam tier needs the chosen Study label inside its instruction sentence, which the generic
    `_finish(draw_fn, instruction_fn, rng)` shape (identical instruction across every attempt) cannot carry --
    a thin dedicated wrapper instead of forcing that shape to fit. `other_n` (the non-target study's n) is the
    trap-n distractor's wrong_n."""
    result = None
    p = n_target = setup = working_extra = target = other_n = None
    for _attempt in range(500):
        p, n_target, setup, working_extra, target, other_n = _draw_exam(rng)
        result = _phat_pair(p, n_target, wrong_n=other_n)
        if result is not None:
            break
    else:
        raise RuntimeError("p19d_phat_mean_sd: construction failed to clear distractor gap in 500 attempts")
    instruction = _exam_instruction(target)
    answer_text = result["answer_text"]
    candidate_texts = list(result["candidate_texts"])
    if len(candidate_texts) > 4:
        # See `_finish`'s matching comment -- a 5th candidate is otherwise never selected by
        # `pick_distractors`.
        rng.shuffle(candidate_texts)
    distractors = pick_distractors(answer_text, candidate_texts, n=4, tolerance=GRADER_TOLERANCE)
    check(len(distractors) == 4, "a distractor collided with the answer within grader tolerance")

    question_text = setup + " " + instruction
    working = list(working_extra) + [
        f"Mean of p-hat = p = {result['mean_str']}.",
        f"Standard deviation of p-hat = the square root of (p times (1 - p) divided by n) "
        f"= the square root of ({result['mean_str']} times "
        f"{stated_precision_string(1 - p, _dp_needed(1 - p))} divided by {n_target}) = {result['sd_str']}.",
    ]
    return {
        "text": question_text,
        "latex": f"plain:{question_text}",
        "answer": {"mean": result["p_f"], "sd": result["sd_r"]},
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text, plain=True),
        "distractors": distractors,
        "working": working,
    }
