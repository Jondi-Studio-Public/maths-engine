"""Calls `mathgen.make` only and reads `question.text`/`question.answer_text`/
`question.distractors` -- it never imports anything from
`mathgen.generators.p19c_bernoulli_binomial_suitability_build`. For every seed it parses the
rendered scenario's stated facts with its own keyword/regex logic and re-derives the
classification independently, then asserts the result matches `answer_text`.

The five (binomial) / four (bernoulli) option strings are typed out here again, verbatim,
rather than imported -- an oracle that imported the generator's own OPTIONS_* constants could
still be fooled by a construction bug that mislabels which class a scenario belongs to while
using the right label string; typing the strings out again is what makes this a genuinely
independent check on the CONTRACT, not merely on the generator's own internal consistency.
"""

from __future__ import annotations

import re

from mathgen import make

TIERS = ["easy", "medium", "hard", "exam"]

# --- Binomial option strings, typed out independently of the generator module ---

_B_SUITED = "Suited to a binomial model"
_B_NOT_FIXED = "Not suited -- the number of trials is not fixed"
_B_NOT_INDEP = "Not suited -- the trials are not independent"
_B_NOT_CONST_P = "Not suited -- the success probability is not constant across trials"
_B_NOT_TWO = "Not suited -- each trial does not have exactly two possible outcomes"

# --- Bernoulli option strings ---

_R_SUITED = "Suited to a Bernoulli model"
_R_IS_BINOMIAL = "Not suited -- this describes a FIXED NUMBER of repeated trials (a binomial setup), not a single trial"
_R_CONTINUOUS = "Not suited -- the outcome is a continuous measurement, not one of two discrete categories"
_R_DEPENDENT = (
    "Not suited -- the probability of success for this single draw is not a fixed known "
    "value (items already removed, count unstated)"
)

_PERCENT_RE = re.compile(r"(\d+)%")


def _derive_binomial_class(text: str) -> str:
    """Re-derive the binomial-suitability classification from the rendered scenario text
    alone, using this test's own keyword logic."""
    lower = text.lower()

    # Decisive: trial count not fixed in advance.
    if "no trial count is fixed in advance" in lower or "as many times as it takes" in lower:
        return _B_NOT_FIXED

    # Decisive: outcome is not binary (continuous, or a >2-category scale).
    if "continuous scale" in lower or re.search(r"one of \d+ categories", lower):
        return _B_NOT_TWO

    # Decisive: independence -- "without replacement" plus "small"/"large" population word.
    if "without replacement" in lower:
        if "small population" in lower:
            return _B_NOT_INDEP
        if "large population" in lower:
            pass  # negligible effect: falls through to the constant-probability check
        else:
            raise AssertionError(f"without-replacement clause missing a small/large marker: {text!r}")

    # Decisive: more than one distinct success-probability percentage stated.
    percents = set(_PERCENT_RE.findall(text))
    if len(percents) >= 2:
        return _B_NOT_CONST_P

    return _B_SUITED


def _derive_bernoulli_class(text: str) -> str:
    """Re-derive the Bernoulli-suitability classification from the rendered scenario text
    alone, using this test's own keyword logic."""
    lower = text.lower()

    # Decisive: explicitly repeated a fixed number of times, counting successes.
    if "this is repeated" in lower and "how many" in lower:
        return _R_IS_BINOMIAL

    # Decisive: continuous outcome.
    if "continuous scale" in lower:
        return _R_CONTINUOUS

    # Decisive: already-removed-without-replacement framing, disambiguated by population size.
    if "already" in lower and "removed" in lower and "without replacement" in lower:
        if "small population" in lower:
            return _R_DEPENDENT
        if "large population" in lower:
            return _R_SUITED
        raise AssertionError(f"already-removed clause missing a small/large marker: {text!r}")

    return _R_SUITED


def test_binomial_oracle_matches_over_1000_seeds_per_tier():
    class_counts = {tier: {} for tier in TIERS}
    total = 0
    for tier in TIERS:
        for seed in range(1000):
            q = make("p19c_binomial_suitability", seed=seed, difficulty=tier)
            derived = _derive_binomial_class(q.text)
            assert derived == q.answer_text, (
                f"tier={tier} seed={seed}: oracle derived {derived!r} but answer_text is "
                f"{q.answer_text!r}\ntext: {q.text}"
            )
            class_counts[tier][derived] = class_counts[tier].get(derived, 0) + 1
            total += 1
    assert total == 4000
    print("binomial suitability class counts per tier (1000 seeds each):", class_counts)


def test_bernoulli_oracle_matches_over_1000_seeds_per_tier():
    class_counts = {tier: {} for tier in TIERS}
    total = 0
    for tier in TIERS:
        for seed in range(1000):
            q = make("p19c_bernoulli_suitability", seed=seed, difficulty=tier)
            derived = _derive_bernoulli_class(q.text)
            assert derived == q.answer_text, (
                f"tier={tier} seed={seed}: oracle derived {derived!r} but answer_text is "
                f"{q.answer_text!r}\ntext: {q.text}"
            )
            class_counts[tier][derived] = class_counts[tier].get(derived, 0) + 1
            total += 1
    assert total == 4000
    print("bernoulli suitability class counts per tier (1000 seeds each):", class_counts)


def test_no_binomial_suited_scenario_text_is_a_bernoulli_suited_scenario_text():
    """Sweep both topics' SUITED scenarios
    and confirm no single generated scenario TEXT from one topic could be emitted by the
    other topic's own SUITED case -- the sharp single-trial-vs-fixed-repeat-count boundary
    must hold on every draw, not just by construction. A binomial SUITED text always states
    "trials, fixed in advance" (a repeat count); a Bernoulli SUITED text never does."""
    binomial_suited_texts = set()
    bernoulli_suited_texts = set()
    for tier in TIERS:
        for seed in range(300):
            qb = make("p19c_binomial_suitability", seed=seed, difficulty=tier)
            if qb.answer_text == _B_SUITED:
                binomial_suited_texts.add(qb.text)
            qr = make("p19c_bernoulli_suitability", seed=seed, difficulty=tier)
            if qr.answer_text == _R_SUITED:
                bernoulli_suited_texts.add(qr.text)

    assert binomial_suited_texts, "no binomial SUITED scenario drawn in 1200 seeds"
    assert bernoulli_suited_texts, "no bernoulli SUITED scenario drawn in 1200 seeds"
    assert binomial_suited_texts.isdisjoint(bernoulli_suited_texts)

    # Every binomial SUITED text names a fixed repeat count; every bernoulli SUITED text
    # describes a single trial and never the "this is repeated" binomial phrasing.
    for text in binomial_suited_texts:
        assert "trials, fixed in advance" in text or "trials are drawn without replacement" in text
    for text in bernoulli_suited_texts:
        assert "this is repeated" not in text.lower()
