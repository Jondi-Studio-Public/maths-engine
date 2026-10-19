"""Event independence verdict + everyday probability-language classification.

Two unrelated mcq skills, built together in one module.

- ``p19c_independence_verdict`` (senior-understand-the-notion-of-independence-of-an-event,
  ACMMM058): given P(A), P(B) and a third given value (P(A and B) directly,
  or P(A|B), recovered via P(A and B) = P(A|B) x P(B) -- this generator's own
  fixed convention), determine whether A and B are independent. mcq, 4
  options: "Independent" plus 3 "Not independent -- P(A and B) = X but
  P(A) x P(B) = Y" statements, each stating REAL computed numbers (never a
  bare "not independent"). Boundary against the shipping
  ``senior-use-the-notation-p-a-vert-b``: that skill COMPUTES the numeric
  value of P(A|B) from given counts/probabilities and grades a fraction --
  it never asks an independence verdict. This skill takes P(A), P(B) and a
  third value as GIVEN inputs and asks only for the yes/no independence
  classification with a stated reason; it never grades a bare P(A|B)
  numeric value as its answer.

- ``p19c_probability_language_classify``
  (senior-interpret-commonly-used-probability-statements-including-possible-probable,
  ACMEM148): given a probability value (direct, percentage, "N times out of
  M", or a stated complement), classify it against this skill's OWN four
  disjoint ranges for possible/probable/likely/certain -- restated verbatim
  in every rendered prompt so the classification is deterministic from the
  stated ranges, never from connotation ("probable" and "likely" are
  near-synonyms otherwise). Boundary against FOUR named neighbours (see
  senior-review-probability-as-a-measure-of-the-likelihood (compares TWO
  events, never labels one probability against a scale);
  senior-review-the-probability-scale-0-leq-p-a (three-way P=0/P=1/between
  from a sample-space rule, never the four-term vocabulary and never a
  stated probability VALUE); f10-probability-scale-and-expectation (the
  REVERSE direction, word -> fixed anchor, and a different five-word set);
  f10-chance-language-and-outcomes (raw favourable/total item counts, no
  stated probability value, and a different four-word set).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, pick_distractors

# ============================================================================
# p19c_independence_verdict
# ============================================================================

OPT_INDEPENDENT = "Independent"


def _fmt(fr: Fraction) -> str:
    """Exact decimal text for a Fraction whose denominator only has factors
    2 and 5 (always true for every value this generator builds) -- pure
    integer arithmetic, no float rounding."""
    fr = Fraction(fr)
    sign = "-" if fr < 0 else ""
    fr = abs(fr)
    for dp in range(0, 6):
        scale = 10**dp
        num = fr.numerator * scale
        den = fr.denominator
        if num % den == 0:
            digits = num // den
            if dp == 0:
                return f"{sign}{digits}"
            s = str(digits).rjust(dp + 1, "0")
            return f"{sign}{s[:-dp]}.{s[-dp:]}"
    raise ValueError(f"{fr} is not exactly decimal-representable")


def _dp_needed(fr: Fraction) -> int:
    """Minimal decimal-place count for an exact decimal representation of
    `fr` (a denominator with only factors 2 and 5, as `_fmt` assumes)."""
    fr = abs(Fraction(fr))
    for dp in range(0, 6):
        if (fr.numerator * 10**dp) % fr.denominator == 0:
            return dp
    raise ValueError(f"{fr} is not exactly decimal-representable")


def _fmt_fixed(fr: Fraction, dp: int) -> str:
    """Like `_fmt`, but zero-padded to exactly `dp` decimal places, so two
    exactly-equal-VALUE numbers that happen to need different natural
    precision (e.g. 0.4 vs 0.42) render as equal-LENGTH strings (0.40 vs
    0.42) -- closing a length tell: the
    true P(A) x P(B) product otherwise tends to need the full 2 d.p. while
    several misconception products (`pa + pb`, `abs(pa - pb)`) often round
    to a tidy 1 d.p., making the genuinely-independent-looking key string
    systematically the LONGEST option a learner could pick without doing
    the multiplication."""
    fr = Fraction(fr)
    sign = "-" if fr < 0 else ""
    fr = abs(fr)
    scale = 10**dp
    num = fr.numerator * scale
    den = fr.denominator
    check(num % den == 0, f"{fr} does not represent exactly at {dp} decimal places")
    digits = num // den
    if dp == 0:
        return f"{sign}{digits}"
    s = str(digits).rjust(dp + 1, "0")
    return f"{sign}{s[:-dp]}.{s[-dp:]}"


def _ni_option(and_value: Fraction, product: Fraction, product_dp: int) -> str:
    """`and_value` is IDENTICAL across every "not independent" option in a
    given question (it is the question's own stated P(A and B)), so it
    never contributes to a length difference between options -- only the
    `product` term (which varies per option/misconception) does, hence only
    it is padded to a shared `product_dp` (see `_fmt_fixed`)."""
    return f"Not independent -- P(A and B) = {_fmt_fixed(and_value, product_dp)} but P(A) x P(B) = {_fmt_fixed(product, product_dp)}"


def classify_independence(and_value: Fraction, true_product: Fraction) -> str:
    """Independent check: verdict is a pure equality test between the given
    P(A and B) value and the true product P(A) x P(B). Never trusts the
    construction path; recomputes from the two Fractions directly."""
    return OPT_INDEPENDENT if and_value == true_product else "not_independent"


def _misconception_products(pa: Fraction, pb: Fraction) -> list[Fraction]:
    """Candidate WRONG "P(A) x P(B)" values -- plausible near-miss
    misconceptions, not filler. Order is priority order for `pick_distractors`.

    Every candidate here must be a valid, non-degenerate probability that
    itself lies in ``(0, min(pa, pb)]``: an out-of-range value (> 1, or
    even in (0, 1] but > min(pa, pb)) states an IMPOSSIBLE "P(A and B)"
    value -- ``P(A and B)`` can never exceed either ``P(A)`` or ``P(B)``
    for any real probability space, since the intersection of two sets is
    never more probable than either set -- and a 0 value states no
    discrepancy at all; either is rejectable on sight without doing the
    actual multiplication, weakening the distractor to the point of not
    being genuine.

    This is a strictly TIGHTER filter than the old ``(0, 1]`` range (several candidates routinely exceeded ``min(pa, pb)`` even
    though they were `<= 1`, e.g. ``pa + pb`` can reach 1.6 and even
    ``pa*pa``/``pb*pb`` can exceed ``min(pa, pb)`` when ``pa`` or ``pb`` is
    close to 1). Unlike the old ``(0, 1]`` filter, at least 3 of the 5
    candidates are NOT guaranteed to survive this tighter bound for every
    ``(pa, pb)`` pair -- this is now a DATA-DEPENDENT starvation case, like
    the caller's existing `and_value`-collision filter, so this function no
    longer hard-`check()`s a minimum count; the caller redraws (`continue`)
    when the pool is too small, exactly as it already does when the
    `and_value`-collision filter shrinks the pool below 3.

    Rather than clamp or rescale an invalid value (which would silently
    produce a DIFFERENT, possibly duplicate value), it is simply dropped
    from the candidate pool.

    This does NOT check against the question's own `and_value` (this
    function only sees P(A), P(B)) -- a candidate that happens to equal
    and_value is a separate, data-dependent self-contradiction (equal
    numbers describe INDEPENDENT, not "not independent") that the caller
    filters and, if it starves the pool below 3, redraws for.
    """
    candidates = [
        pa + pb,  # adds instead of multiplies
        abs(pa - pb),  # subtracts instead of multiplies
        pa * pa,  # squares P(A), forgetting P(B) entirely
        pb * pb,  # squares P(B), forgetting P(A) entirely
        (pa + pb) / 2,  # averages instead of multiplying
        pa * (1 - pb),  # P(A and not B): multiplies by the complement
        pb * (1 - pa),  # P(B and not A): multiplies by the complement
        min(pa, pb) / 2,  # halves the smaller probability
    ]
    upper = min(pa, pb)
    valid = [c for c in candidates if Fraction(0) < c <= upper]
    # Explicit >1 check kept even though it is subsumed by `c <= upper`
    # (upper = min(pa, pb) <= 1 always), so a future change to the bound
    # logic cannot silently reopen the >1 "impossible probability" case.
    check(all(c <= Fraction(1) for c in valid), "misconception-product candidate exceeds 1 -- impossible probability")
    return valid


def _feasible(pa: Fraction, pb: Fraction, and_value: Fraction) -> bool:
    """Whether ``and_value`` can actually be P(A and B) for SOME probability
    space with these P(A), P(B) -- the Frechet/Boole bounds
    ``max(0, P(A)+P(B)-1) <= P(A and B) <= min(P(A), P(B))``. The
    "independent" branch (``and_value = P(A) x P(B)``) always satisfies this
    for 0<=P(A),P(B)<=1 -- a mathematical fact, not something that needs a
    resample loop -- so only the "not independent" branches, which pick
    ``and_value`` away from the true product, can land outside it."""
    lo = max(Fraction(0), pa + pb - 1)
    hi = min(pa, pb)
    return lo <= and_value <= hi


def _pick_not_independent_and(
    rng, pa: Fraction, pb: Fraction, true_product: Fraction, min_margin: Fraction, max_margin: Fraction, denom: int
) -> Fraction | None:
    """Draw an ``and_value`` that is NOT ``true_product``, by a margin in
    [min_margin, max_margin], decimal-exact at ``denom``, staying in (0, 1)
    AND inside the feasible range for P(A), P(B) (see `_feasible`). Returns
    ``None`` if no feasible candidate was found in 50 tries, so the caller
    can resample P(A)/P(B) themselves rather than emit an impossible
    scenario."""
    for _ in range(50):
        margin_steps = rng.randint(int(min_margin * denom), int(max_margin * denom))
        if margin_steps <= 0:
            continue
        sign = rng.choice([-1, 1])
        candidate = true_product + Fraction(sign * margin_steps, denom)
        if (
            Fraction(1, denom) <= candidate <= Fraction(denom - 1, denom)
            and candidate != true_product
            and _feasible(pa, pb, candidate)
        ):
            return candidate
    return None


@generator(
    "p19c_independence_verdict",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def independence_verdict(rng, tier="easy"):
    """Given P(A), P(B) and a third value (P(A and B) directly, or P(A|B)
    with the recovery convention P(A and B) = P(A|B) x P(B)), classify
    whether A and B are independent. mcq: "Independent" or one of 3
    "Not independent -- P(A and B) = X but P(A) x P(B) = Y" statements."""
    # Raised from 30: fixing `want_independent` once per call
    # (below) means every attempt in this loop must independently clear the
    # tightened `_misconception_products` bound (~40-45% starvation, measured)
    # plus, on the medium not-independent branch, its own extra `_feasible`
    # check -- so the per-attempt acceptance probability is materially lower
    # than it was when a starved attempt could silently re-roll the verdict
    # too. 200 attempts keeps exhaustion negligible (measured zero
    # exhaustions over 800+ seeds across all four tiers).
    max_attempts = 400
    # Every decimal in one question (stimulus AND options) is printed at the same 2 d.p. (every
    # value here is a multiple of 0.01), so "0.20" never sits beside "0.32".
    _f2 = lambda fr: _fmt_fixed(fr, 2)
    # Verdict drawn ONCE, outside the retry loop: drawing it
    # fresh on every attempt let any rejection channel that fires more often
    # on one branch (e.g. the tightened `_misconception_products` bound
    # below, or medium's own extra `_feasible` check on the not-independent
    # branch) silently re-flip this coin toward whichever branch has the
    # HIGHER per-attempt acceptance rate -- so the ACCEPTED share drifted
    # away from the stated ~30% weight even though each individual draw of
    # this line is still a fair 30% coin. Fixing the verdict once per call
    # and only redrawing P(A)/P(B)/and_value on `continue` keeps the
    # marginal accepted share equal to the stated weight (subject only to
    # the negligible probability of exhausting `max_attempts`, identical for
    # both branches).
    want_independent = rng.random() < 0.30
    target_rank = rng.randint(1, 3)
    want_key_nearest = rng.random() < 0.33
    for _attempt in range(max_attempts):
        if _attempt and _attempt % 50 == 0:
            # A rank/nearest combination some (pa, pb, and_value) shapes can
            # never satisfy (medium's and_value is fixed by its P(A|B) step)
            # is relaxed rather than left to exhaust the loop: the
            # nearest-option flag flips every 50 attempts.
            want_key_nearest = not want_key_nearest
        ka = rng.randint(2, 8)
        kb = rng.randint(2, 8)
        pa = Fraction(ka, 10)
        pb = Fraction(kb, 10)
        true_product = pa * pb  # denom divides 100

        extra_note = ""
        if tier == "easy":
            if want_independent:
                and_value = true_product
            else:
                and_value = _pick_not_independent_and(
                    rng, pa, pb, true_product, Fraction(3, 100), Fraction(8, 100), 100
                )
                if and_value is None:
                    continue
            variant = rng.choice(["list", "framed", "given"])
            if variant == "list":
                stimulus = f"Event A has P(A) = {_f2(pa)}. Event B has P(B) = {_f2(pb)}. P(A and B) = {_f2(and_value)}."
            elif variant == "framed":
                stimulus = (
                    f"Two events are being compared. P(A) = {_f2(pa)}. P(B) = {_f2(pb)}. P(A and B) = {_f2(and_value)}."
                )
            else:  # given
                stimulus = (
                    f"The following probabilities are given. P(A) = {_f2(pa)}. "
                    f"P(B) = {_f2(pb)}. P(A and B) = {_f2(and_value)}."
                )

        elif tier == "medium":
            if want_independent:
                cond = pa
                and_value = cond * pb
            else:
                margin10 = rng.choice([Fraction(2, 10), Fraction(3, 10), Fraction(4, 10)])
                sign = rng.choice([-1, 1])
                cond = pa + sign * margin10
                if not (Fraction(1, 10) <= cond <= Fraction(9, 10)):
                    cond = pa - sign * margin10
                and_value = cond * pb
                if not _feasible(pa, pb, and_value):
                    continue
            variant = rng.choice(["list", "framed", "given"])
            if variant == "list":
                stimulus = (
                    f"Event A has P(A) = {_f2(pa)}. Event B has P(B) = {_f2(pb)}. "
                    f"The conditional probability P(A|B) = {_f2(cond)} "
                    f"(convention: P(A|B) = P(A and B) / P(B), so P(A and B) = P(A|B) x P(B))."
                )
            elif variant == "framed":
                stimulus = (
                    f"Two events are being compared. P(A) = {_f2(pa)}. "
                    f"P(B) = {_f2(pb)}. The conditional probability P(A|B) = {_f2(cond)} "
                    f"(convention: P(A|B) = P(A and B) / P(B), so P(A and B) = P(A|B) x P(B))."
                )
            else:  # given
                stimulus = (
                    f"The following probabilities are given. P(A) = {_f2(pa)}. "
                    f"P(B) = {_f2(pb)}. P(A|B) = {_f2(cond)} "
                    f"(convention: P(A|B) = P(A and B) / P(B), so P(A and B) = P(A|B) x P(B))."
                )

        elif tier == "hard":
            if want_independent:
                and_value = true_product
            else:
                # >= 7 people out of 100, so "approximately independent" is never a second
                # defensible survey answer.
                and_value = _pick_not_independent_and(
                    rng, pa, pb, true_product, Fraction(7, 100), Fraction(12, 100), 100
                )
                if and_value is None:
                    continue
            n = 100
            count_a = int(pa * n)
            count_b = int(pb * n)
            count_ab = int(and_value * n)
            if count_ab == 1:
                # "1 have both" is ungrammatical; redraw instead.
                continue
            variant = rng.choice(["counts", "percent", "counts_framed"])
            if variant == "counts":
                stimulus = (
                    f"Of {n} people surveyed: {count_a} have property A, "
                    f"{count_b} have property B, and {count_ab} have both A and B."
                )
            elif variant == "percent":
                stimulus = f"P(A) = {count_a}%. P(B) = {count_b}%. P(A and B) = {count_ab}%."
            else:  # counts_framed
                stimulus = (
                    f"A survey was carried out. Of {n} people surveyed: "
                    f"{count_a} have property A, {count_b} have property B, "
                    f"and {count_ab} have both A and B."
                )

        else:  # exam
            if want_independent:
                and_value = true_product
            else:
                and_value = _pick_not_independent_and(
                    rng, pa, pb, true_product, Fraction(3, 100), Fraction(12, 100), 100
                )
                if and_value is None:
                    continue
            pc = Fraction(rng.randint(2, 8), 10)
            extra_note = f" Event C is unrelated to this question and has P(C) = {_f2(pc)}."
            variant = rng.choice(["list", "framed", "given"])
            if variant == "list":
                stimulus = (
                    f"Event A has P(A) = {_f2(pa)}. Event B has P(B) = {_f2(pb)}. "
                    f"P(A and B) = {_f2(and_value)}.{extra_note}"
                )
            elif variant == "framed":
                stimulus = (
                    f"Two events are being compared. P(A) = {_f2(pa)}. "
                    f"P(B) = {_f2(pb)}. P(A and B) = {_f2(and_value)}.{extra_note}"
                )
            else:  # given
                stimulus = (
                    f"The following probabilities are given. P(A) = {_f2(pa)}. "
                    f"P(B) = {_f2(pb)}. P(A and B) = {_f2(and_value)}.{extra_note}"
                )

        check(
            _feasible(pa, pb, and_value),
            "P(A and B) outside the feasible range max(0, P(A)+P(B)-1) <= "
            "P(A and B) <= min(P(A), P(B)) -- no such probability space exists",
        )

        verdict = classify_independence(and_value, true_product)
        check(verdict == (OPT_INDEPENDENT if want_independent else "not_independent"), "verdict/and_value mismatch")

        misconceptions = _misconception_products(pa, pb)
        # A candidate that happens to equal and_value would ALSO render as "P(A and B) = X but P(A)
        # x P(B) = X" -- self-contradictory, since equal numbers describe INDEPENDENT, not "not
        # independent". _misconception_products can't see and_value, so filter it out here too.
        # Either source of starvation is handled identically below: redraw pa/pb/and_value.
        misconceptions = [m for m in misconceptions if m != and_value]
        if len(misconceptions) < 3:
            continue
        # Shared product-side decimal-place width across EVERY "not
        # independent" option this draw could render (the true product and
        # every surviving misconception), so no option's product term is
        # ever shorter/longer than another's purely from trailing-zero
        # trimming (see `_ni_option`'s docstring).
        product_dp = max(2, _dp_needed(true_product), *(_dp_needed(m) for m in misconceptions))
        ni_candidates = [_ni_option(and_value, m, product_dp) for m in misconceptions]

        if verdict == OPT_INDEPENDENT:
            answer_text = OPT_INDEPENDENT
            distractors = pick_distractors(answer_text, ni_candidates, n=3)
            if len(distractors) < 3:
                continue
        else:
            answer_text = _ni_option(and_value, true_product, product_dp)
            # The key product's rank (1 = lowest) among the three shown products is `target_rank`,
            # drawn once per call, so the accepted rank shares stay uniform however often a pool
            # starves.
            pool = [m for m in dict.fromkeys(misconceptions) if m != true_product]
            lower = [m for m in pool if m < true_product]
            higher = [m for m in pool if m > true_product]
            if len(lower) < target_rank - 1 or len(higher) < 3 - target_rank:
                continue
            rank = target_rank
            # Also draw, once per call, whether the key product is the one
            # nearest P(A and B): otherwise "nearest option to P(A and B)"
            # would be a shortcut that skips the multiplication.
            gap_key = abs(true_product - and_value)
            chosen = None
            for _try in range(20):
                trial = rng.sample(lower, rank - 1) + rng.sample(higher, 3 - rank)
                nearer = [m for m in trial if abs(m - and_value) < gap_key]
                key_is_nearest = all(abs(m - and_value) > gap_key for m in trial)
                if key_is_nearest if want_key_nearest else bool(nearer):
                    chosen = trial
                    break
            if chosen is None:
                continue
            picked = pick_distractors(answer_text, [_ni_option(and_value, m, product_dp) for m in chosen], n=2)
            if len(picked) < 2:
                continue
            distractors = [OPT_INDEPENDENT] + picked

        check(answer_text not in distractors, "answer duplicated among distractors")
        check(len(distractors) == len(set(distractors)), "duplicate distractor strings")
        break
    else:
        raise ValueError(f"could not build a valid independence draw for tier={tier!r}")

    question_text = f"{stimulus} Are events A and B independent?"

    working = []
    if tier == "medium":
        working.append(f"P(A and B) = P(A|B) × P(B) = {_f2(cond)} × {_f2(pb)} = {_f2(and_value)}.")
    elif tier == "hard":
        working.append(
            f"P(A) = {count_a}/{n} = {_f2(pa)}, P(B) = {count_b}/{n} = {_f2(pb)}, "
            f"and P(A and B) = {count_ab}/{n} = {_f2(and_value)}."
        )
    working.append("A and B are independent exactly when P(A and B) = P(A) × P(B).")
    working.append(
        f"P(A) × P(B) = {_f2(pa)} × {_f2(pb)} = {_fmt_fixed(true_product, max(2, _dp_needed(true_product)))}."
    )
    if verdict == OPT_INDEPENDENT:
        working.append(f"This equals P(A and B) = {_f2(and_value)}, so A and B are independent.")
    else:
        working.append(f"This is not equal to P(A and B) = {_f2(and_value)}, so A and B are not independent.")

    return dict(
        text=question_text,
        working=working,
        latex=r"\text{P(A and B)} \text{ vs } \text{P(A)} \times \text{P(B)}",
        answer=answer_text,
        answer_text=answer_text,
        distractors=distractors,
    )


# ============================================================================
# p19c_probability_language_classify
# ============================================================================

B1 = Fraction(3, 10)  # possible upper bound
B2 = Fraction(11, 20)  # probable upper bound
B3 = Fraction(9, 10)  # likely upper bound / certain lower bound

CONVENTION_TEXT = (
    "Using the convention that a probability is described as 'possible' for "
    "0 < P <= 0.3, 'probable' for 0.3 < P <= 0.55, 'likely' for 0.55 < P < 0.9, "
    "and 'certain' for P >= 0.9, "
)

CATEGORIES = ("possible", "probable", "likely", "certain")

_BOUNDS = {
    "possible": (Fraction(0), B1),
    "probable": (B1, B2),
    "likely": (B2, B3),
    "certain": (B3, Fraction(1)),
}
_RELEVANT_BOUNDS = {
    "possible": [B1],
    "probable": [B1, B2],
    "likely": [B2, B3],
    "certain": [B3],
}


def classify_probability_language(value: Fraction) -> str:
    """Independent check: classify against the fixed, stated boundaries.
    Never trusts the construction path; recomputes from the Fraction directly."""
    if value <= B1:
        return "possible"
    if value <= B2:
        return "probable"
    if value < B3:
        return "likely"
    return "certain"


def _sample_in_category(rng, category: str, min_margin: Fraction, max_margin: Fraction | None, denom: int) -> Fraction:
    """Draw a value inside ``category``'s range, at least ``min_margin`` from
    every boundary that borders it (and, if ``max_margin`` is given, at most
    that far -- for the exam tier's deliberately-close-to-a-boundary draw)."""
    lo, hi = _BOUNDS[category]
    lo_step = int(lo * denom) + 1
    hi_step = int(hi * denom) - 1
    valid_steps = []
    for step in range(lo_step, hi_step + 1):
        candidate = Fraction(step, denom)
        dists = [abs(candidate - b) for b in _RELEVANT_BOUNDS[category]]
        min_dist = min(dists) if dists else Fraction(1)
        if min_dist < min_margin:
            continue
        if max_margin is not None and min_dist > max_margin:
            continue
        valid_steps.append(step)
    if not valid_steps:
        raise ValueError(f"no valid value for category={category!r} at denom={denom!r}")
    candidate = Fraction(rng.choice(valid_steps), denom)
    check(
        classify_probability_language(candidate) == category, "sampled value classifies outside its own target category"
    )
    return candidate


@generator(
    "p19c_probability_language_classify",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def probability_language_classify(rng, tier="easy"):
    """Classify a stated probability value against this skill's own stated
    four-term convention (possible/probable/likely/certain), restated
    verbatim in every prompt."""
    category = rng.choice(CATEGORIES)

    if tier == "easy":
        value = _sample_in_category(rng, category, Fraction(5, 100), None, 100)
        variant = rng.choice(["value_then_convention", "convention_then_value", "scenario"])
        if variant == "value_then_convention":
            stimulus = f"a probability is measured as P = {_fmt(value)}"
            question_text = (
                f"{stimulus}. {CONVENTION_TEXT}classify it using this convention. Which term best describes it?"
            )
        elif variant == "convention_then_value":
            stimulus = f"classify the probability P = {_fmt(value)}"
            question_text = f"{CONVENTION_TEXT}{stimulus} using this convention. Which term best describes it?"
        else:  # scenario
            stimulus = f"an event has probability P = {_fmt(value)} of occurring"
            question_text = f"{CONVENTION_TEXT}{stimulus}. Which term best describes it, using this convention?"

    elif tier == "medium":
        medium_variant = rng.choice(["percent", "percent_scenario", "count"])
        if medium_variant == "percent":
            value = _sample_in_category(rng, category, Fraction(5, 100), None, 100)
            percent = int(value * 100)
            stimulus = f"classify a probability of {percent}%"
        elif medium_variant == "percent_scenario":
            value = _sample_in_category(rng, category, Fraction(5, 100), None, 100)
            percent = int(value * 100)
            stimulus = (
                f"a weather forecast describes a probability of {percent}% chance of rain -- classify this probability"
            )
        else:
            denom_choices = [20, 25] if category == "certain" else [10, 20, 25]
            denom = rng.choice(denom_choices)
            value = _sample_in_category(rng, category, Fraction(1, denom), None, denom)
            k = int(value * denom)
            stimulus = f"an event happens {k} times out of {denom} on average -- classify its probability"

    elif tier == "hard":
        value = _sample_in_category(rng, category, Fraction(5, 100), None, 100)
        complement = Fraction(1) - value
        variant = rng.choice(["not_occur", "fails", "instead"])
        if variant == "not_occur":
            stimulus = (
                f"the probability that an event does NOT occur is {_fmt(complement)} -- "
                f"classify the probability that it DOES occur"
            )
        elif variant == "fails":
            stimulus = (
                f"it is known that the probability that an event does NOT occur is "
                f"{_fmt(complement)} -- classify the probability that it DOES occur"
            )
        else:  # instead
            stimulus = (
                f"suppose the probability that an event does NOT occur is "
                f"{_fmt(complement)}; classify the probability that it DOES occur instead"
            )

    else:  # exam -- a genuinely new shape (not merely a closer number): the
        # prompt itself flags the near-boundary trap and forbids estimation,
        # so a learner must apply the stated cut-points exactly rather than
        # pattern-match on "roughly half" / "roughly nine tenths" and so on.
        value = _sample_in_category(rng, category, Fraction(2, 200), Fraction(6, 200), 200)
        variant = rng.choice(["measured", "reported", "recorded"])
        if variant == "measured":
            stimulus = (
                f"a probability is measured as P = {_fmt(value)}, which lies close to one "
                f"of the stated cut-points -- classify it precisely rather than by rounding "
                f"or estimation"
            )
        elif variant == "reported":
            stimulus = (
                f"an analyst reports a probability of P = {_fmt(value)}, which sits close "
                f"to one of the stated cut-points, so classify it precisely rather than by "
                f"rounding or estimation"
            )
        else:  # recorded
            stimulus = (
                f"a recorded probability of P = {_fmt(value)} lies close to one of the "
                f"stated cut-points; classify it precisely rather than by rounding or "
                f"estimation"
            )

    answer_text = classify_probability_language(value)
    check(answer_text == category, "constructed value landed outside its own target category")

    distractors = pick_distractors(answer_text, [c for c in CATEGORIES if c != category], n=3)
    check(len(distractors) == 3, "expected exactly 3 distractor categories")

    if tier != "easy":
        question_text = f"{CONVENTION_TEXT}{stimulus} using this convention. Which term best describes it?"

    check(
        all(b in question_text for b in ("0.3", "0.55", "0.9")),
        "mandatory range-convention boundaries missing from rendered prompt",
    )

    if tier == "hard":
        first = f"P(the event occurs) = 1 − {_fmt(complement)} = {_fmt(value)}."
    elif tier == "medium" and medium_variant == "count":
        first = f"P = {k}/{denom} = {_fmt(value)}."
    elif tier == "medium":
        first = f"P = {percent}% = {_fmt(value)}."
    else:
        first = f"P = {_fmt(value)}."
    if answer_text == "possible":
        range_line = "The stated range for 'possible' is 0 < P ≤ 0.3."
    elif answer_text == "probable":
        range_line = "The stated range for 'probable' is 0.3 < P ≤ 0.55."
    elif answer_text == "likely":
        range_line = "The stated range for 'likely' is 0.55 < P < 0.9."
    else:
        range_line = "The stated range for 'certain' is P ≥ 0.9."
    working = [first, range_line, f"{_fmt(value)} lies in this range, so the term is '{answer_text}'."]

    return dict(
        text=question_text,
        working=working,
        latex=r"\text{P}",
        answer=answer_text,
        answer_text=answer_text,
        distractors=distractors,
    )
