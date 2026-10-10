"""Calls `mathgen.make` only, reads `question.text`/`question.answer_text`, and never imports anything from
`mathgen.generators.p19c_uniform_relative_frequency_build`.

Two separate routes, each re-deriving the answer from the rendered question
text with its OWN parsing logic (never the generator's own construction
path):

- p19c_uniform_discrete_outcome: parse n from the scenario text (count the
  described structure -- a directly stated count, a listed set of named
  items, or an arithmetic product of two stated factors -- or, for the exam
  tier, the count following the phrase "equal sectors"), and assert
  answer_text equals 1/n by value.
- p19c_relative_frequency_estimate: parse count and trials from the
  scenario text independently, compute count/trials (as a percentage for
  the exam tier), round to the precision the prompt itself states, and
  assert it matches answer_text's value.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen import make


def _round_half_up(value: Fraction) -> int:
    """Ties away from zero -- the same convention `mathgen.core.round_half_up`
    uses, re-implemented here (not imported) so the oracle's rounding route
    is independent of the generator's own helper call."""
    return math.floor(value + Fraction(1, 2))


SEEDS = range(250)
TIERS = ["easy", "medium", "hard", "exam"]

UD_TOPIC = "p19c_uniform_discrete_outcome"
RF_TOPIC = "p19c_relative_frequency_estimate"


# --------------------------------------------------------------------------
# p19c_uniform_discrete_outcome oracle
# --------------------------------------------------------------------------


def _oracle_m_over_n_uniform(text: str):
    """Independently recover (m, n) for the exam-tier "probability of
    landing on one of a stated SET of m outcomes" question: n from the
    uniform spinner's own "equal sectors" description, m by counting the
    stated label set `{a, b, c}` -- never trusting the generator's own m."""
    n_match = re.search(r"divided into (\d+) equal sectors", text)
    assert n_match, f"oracle could not parse n from exam-tier text: {text!r}"
    n = int(n_match.group(1))
    # "labelled A or B" (m == 2) or "labelled A, B, ..., or Z" (m >= 3) -- a plain
    # comma/or list, never a brace-delimited set (braces get LaTeX-escaped by
    # `latex_prompt`'s auto-wrap and are avoided in this prompt for that reason).
    set_match = re.search(r"sector labelled ([^?]+)\?", text)
    assert set_match, f"oracle could not parse the label set from: {text!r}"
    labels = [x for x in re.split(r",| or ", set_match.group(1)) if x.strip()]
    return len(labels), n


def _oracle_hard_uniform(text: str) -> Fraction:
    """A conditional 1/(n - 1), or a complement (n - 1)/n reached either from an explicit "equal-sized sectors"
    count or from an itemised list of (equal) degree values -- by parsing the rendered text alone, never
    trusting the generator's own n.
    """
    m = re.search(
        r"did NOT land on the sector labelled \d+\. Given this, what is the "
        r"probability that it landed on the sector labelled \d+\?",
        text,
    )
    if m:
        n_match = re.search(r"divided into (\d+) equal sectors", text)
        assert n_match, f"oracle could not parse n from hard-tier conditional text: {text!r}"
        n = int(n_match.group(1))
        return Fraction(1, n - 1)

    if "does NOT land" in text:
        n_match = re.search(r"divided into (\d+) equal-sized sectors", text)
        if n_match:
            n = int(n_match.group(1))
            return Fraction(n - 1, n)
        # complement_degrees: count the itemised degree-value list directly
        # and check every value is the same, rather than trusting the
        # stated "A spinner has N sectors" count.
        list_match = re.search(r"sizes are: (.+?)\. The spinner", text)
        assert list_match, f"oracle could not parse the degree list from: {text!r}"
        degrees = [int(x) for x in re.findall(r"(\d+) degrees", list_match.group(1))]
        assert degrees, f"oracle found no degree values in: {text!r}"
        assert len(set(degrees)) == 1, (
            f"oracle found unequal listed degree values, not a genuine uniform spinner: {degrees} from {text!r}"
        )
        assert sum(degrees) == 360, f"oracle found listed degree values not summing to 360: {degrees} from {text!r}"
        n = len(degrees)
        return Fraction(n - 1, n)

    raise AssertionError(f"oracle could not parse hard-tier fraction shape from: {text!r}")


def _oracle_n_uniform(text: str) -> int:
    """Independently recover n, the number of equally likely outcomes, from
    the rendered text alone."""
    # "giving N equally-sized regions/teams/stubs in total"
    m = re.search(r"giving (\d+) (?:equally-sized regions|teams|stubs) in total", text)
    if m:
        return int(m.group(1))

    # A described structure with a listed set of named items: count the
    # comma-separated list following a colon, up to the first sentence
    # boundary (a period followed by a capital letter).
    m = re.search(r": (.+?)\. [A-Z]", text)
    if m:
        items = [x.strip() for x in m.group(1).split(",")]
        return len(items)

    # Otherwise: a directly/indirectly stated count of numbered outcomes
    # ("divided into N equal sectors"/"equal-sized sectors", "N raffle
    # tickets", "N identical-looking chairs").
    for pattern in [
        r"divided into (\d+) equal(?:-sized)? sectors",
        r"contains (\d+) raffle tickets",
        r"^(\d+) identical-looking chairs",
    ]:
        m = re.search(pattern, text)
        if m:
            return int(m.group(1))

    # Medium tier: the prompt gives only the two factors of a
    # rows x columns / groups x per-group structure and never the product,
    # so n is the product of the only two integers in the text.
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if len(nums) == 2:
        return nums[0] * nums[1]

    raise AssertionError(f"oracle could not parse n from: {text!r}")


def test_uniform_discrete_outcome_oracle_matches_1_over_n():
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = make(UD_TOPIC, seed=seed, difficulty=tier)
            actual = Fraction(q.answer["p"], q.answer["q"])
            if tier == "exam":
                # exam tier's graded quantity is m/n: the
                # probability of landing on one of a stated SET of m
                # outcomes, not a bare single-outcome 1/n.
                m, n = _oracle_m_over_n_uniform(q.text)
                expected = Fraction(m, n)
            elif tier == "hard":
                # hard tier's graded quantity is a
                # complement (n-1)/n or a conditional 1/(n-1), never a bare
                # single-outcome 1/n.
                expected = _oracle_hard_uniform(q.text)
            else:
                n = _oracle_n_uniform(q.text)
                expected = Fraction(1, n)
            assert actual == expected, (
                f"[{tier}, seed {seed}] oracle expected {expected}, got {actual} from text: {q.text!r}"
            )
            checked += 1
    assert checked >= 1000, f"only checked {checked} questions"


# --------------------------------------------------------------------------
# p19c_relative_frequency_estimate oracle
# --------------------------------------------------------------------------


def _oracle_count_trials(text: str, tier: str):
    """Independently recover (count, trials, dp, is_percentage) from the
    rendered text alone. Every tier states its own rounding convention in
    the prompt; the oracle reads that convention rather than assuming it."""
    is_percentage = "PERCENTAGE" in text
    dp_match = re.search(r"rounded to (\d+) decimal place", text)
    assert dp_match, f"no stated rounding precision in: {text!r}"
    dp = int(dp_match.group(1))

    # hard/derive, percent_count sub-variant: "N {trial_noun} in total; P% of them were"
    m = re.search(r"(\d+) [\w ]+ in total; (\d+)% of them were", text)
    if m:
        trials = int(m.group(1))
        pct = int(m.group(2))
        count = _round_half_up(Fraction(pct, 100) * trials)
        return count, trials, dp, is_percentage

    # hard/derive, success_miss sub-variant: "{success} were {adj} and {miss} were not"
    m = re.search(r"(\d+) were [\w ]+ and (\d+) were not", text)
    if m:
        success, miss = int(m.group(1)), int(m.group(2))
        return success, success + miss, dp, is_percentage

    # bare / scenario / percentage: "{trials} ...; {count} of them were ..." or
    # "Out of {trials} attempts, {count} were successful." or
    # "{count} out of {trials} attempts were successful." or
    # "In {trials} attempts, {count} succeeded."
    for pattern in [
        r"Out of (\d+) attempts, (\d+) were successful",
        r"(\d+) out of (\d+) attempts were successful",
        r"In (\d+) attempts, (\d+) succeeded",
    ]:
        m = re.search(pattern, text)
        if m:
            if pattern.startswith(r"(\d+) out of"):
                count, trials = int(m.group(1)), int(m.group(2))
            else:
                trials, count = int(m.group(1)), int(m.group(2))
            return count, trials, dp, is_percentage

    m = re.search(r"recorded (\d+) [\w ]+; (\d+) of them were", text)
    if m:
        trials, count = int(m.group(1)), int(m.group(2))
        return count, trials, dp, is_percentage

    raise AssertionError(f"oracle could not parse count/trials from: {text!r}")


def test_relative_frequency_estimate_oracle_matches_count_over_trials():
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = make(RF_TOPIC, seed=seed, difficulty=tier)
            count, trials, dp, is_percentage = _oracle_count_trials(q.text, tier)
            prob = Fraction(count, trials)
            if is_percentage:
                prob = prob * 100
            scale = 10**dp
            expected = Fraction(_round_half_up(prob * scale), scale)
            actual = float(q.answer_text)
            assert abs(actual - float(expected)) < 10 ** (-dp) / 2 + 1e-9, (
                f"[{tier}, seed {seed}] oracle count={count} trials={trials} "
                f"pct={is_percentage} -> expected {expected}, got {actual} from "
                f"text: {q.text!r}"
            )
            checked += 1
    assert checked >= 1000, f"only checked {checked} questions"
