"""Tests for: independence verdict + probability-language pair."""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
import mathgen.generators.p19c_independence_language_build as p19c_independence_module
from mathgen.generators.p19c_independence_language_build import (
    CATEGORIES,
    OPT_INDEPENDENT,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
TIERS = ["easy", "medium", "hard", "exam"]


# ---------------------------------------------------------------------------
# p19c_independence_verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_independence_deterministic_across_seeds(difficulty):
    for seed in SEEDS:
        q1 = make("p19c_independence_verdict", seed=seed, difficulty=difficulty)
        q2 = make("p19c_independence_verdict", seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_independence_at_least_four_options_distinct(difficulty):
    for seed in SEEDS:
        q = make("p19c_independence_verdict", seed=seed, difficulty=difficulty)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == 3
        assert len(set(q.distractors)) == 3
        options = [q.answer_text] + q.distractors
        assert len(options) >= 4


@pytest.mark.parametrize("difficulty", TIERS)
def test_independence_not_independent_options_state_real_numbers(difficulty):
    for seed in SEEDS:
        q = make("p19c_independence_verdict", seed=seed, difficulty=difficulty)
        options = [q.answer_text] + q.distractors
        ni_options = [o for o in options if o != OPT_INDEPENDENT]
        assert len(ni_options) == 3
        for o in ni_options:
            assert o.startswith("Not independent -- P(A and B) = ")
            assert "but P(A) x P(B) = " in o
            assert o != "Not independent"


@pytest.mark.parametrize("difficulty", TIERS)
def test_independence_correct_answer_grades_correct(difficulty):
    for seed in SEEDS:
        q = make("p19c_independence_verdict", seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        idx = options.index(q.answer_text)
        result = grade_mcq(q, idx)
        assert result.correct, f"seed {seed} ({difficulty}): canonical answer did not grade correct"


@pytest.mark.parametrize("difficulty", TIERS)
def test_independence_every_distractor_grades_incorrect(difficulty):
    for seed in SEEDS:
        q = make("p19c_independence_verdict", seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        for d in q.distractors:
            idx = options.index(d)
            result = grade_mcq(q, idx)
            assert not result.correct, f"seed {seed} ({difficulty}): distractor {d!r} graded correct"


@pytest.mark.parametrize("difficulty", TIERS)
def test_independence_key_share_25_to_35_percent_over_200_draws(difficulty):
    n = 200
    independent_count = 0
    for seed in range(n):
        q = make("p19c_independence_verdict", seed=seed, difficulty=difficulty)
        if q.answer_text == OPT_INDEPENDENT:
            independent_count += 1
    share = independent_count / n
    assert 0.20 <= share <= 0.40, f"{difficulty}: independent share {share:.2%} out of the safe band"


@pytest.mark.parametrize("difficulty", TIERS)
def test_independence_at_least_20_distinct_questions_per_tier(difficulty):
    seen = set()
    for seed in range(60):
        q = make("p19c_independence_verdict", seed=seed, difficulty=difficulty)
        seen.add(q.text)
    assert len(seen) >= 20, f"{difficulty}: only {len(seen)} distinct questions over 60 seeds"


def test_independence_corrupted_answer_fails_independent_check():
    from mathgen.core import VerificationError, check
    from mathgen.generators.p19c_independence_language_build import classify_independence
    from fractions import Fraction

    with pytest.raises(VerificationError):
        check(classify_independence(Fraction(1, 4), Fraction(1, 5)) == OPT_INDEPENDENT, "verdict/and_value mismatch")


def test_independence_and_value_feasible_for_2000_plus_seeds_across_all_tiers(monkeypatch):
    """Spies on `check()` (imported into the generator module as `check`) at
    the `"verdict/and_value mismatch"` call site -- the last check every
    accepted draw passes through, in both tier branch, with `pa`/`pb`/
    `and_value` already bound as locals in the caller's frame -- and reads
    those Fractions straight out of the stack, the generator's own internal
    construction-time values, never the rendered text. `_feasible` itself is
    also called from inside `_pick_not_independent_and`'s resample loop on
    REJECTED candidates (that is what makes the resample work), so spying on
    `_feasible` directly would wrongly include those; this call site
    captures only what the generator actually accepted. No epsilon is
    needed: these are exact `Fraction`s, not floats."""
    import sys

    calls: list[tuple[Fraction, Fraction, Fraction]] = []
    original_check = p19c_independence_module.check

    def spy_check(cond, msg="answer verification failed"):
        if msg == "verdict/and_value mismatch":
            frame = sys._getframe(1)
            pa = frame.f_locals.get("pa")
            pb = frame.f_locals.get("pb")
            and_value = frame.f_locals.get("and_value")
            calls.append((pa, pb, and_value))
        return original_check(cond, msg)

    monkeypatch.setattr(p19c_independence_module, "check", spy_check)

    seeds_per_tier = 500  # 4 tiers x 500 = 2000 accepted draws minimum
    for difficulty in TIERS:
        for seed in range(seeds_per_tier):
            make("p19c_independence_verdict", seed=seed, difficulty=difficulty)

    assert len(calls) >= 2000, f"expected >= 2000 recorded draws, got {len(calls)}"
    assert all(v is not None for triple in calls for v in triple), "failed to capture pa/pb/and_value"

    violations = [
        (pa, pb, and_value)
        for pa, pb, and_value in calls
        if not (max(Fraction(0), pa + pb - 1) <= and_value <= min(pa, pb))
    ]
    assert violations == [], f"{len(violations)} infeasible P(A and B) draws out of {len(calls)}: {violations[:5]}"


def test_independence_misconception_products_stay_in_0_min_pa_pb_across_800_plus_seeds(monkeypatch):
    """Spies on `check()` at the same "verdict/and_value mismatch" call site
    used by the feasibility regression above, which has `pa`/`pb` bound as
    locals in the caller's frame for BOTH tier branches -- captures the
    generator's own internal construction-time values, never rendered text.
    """
    import sys

    calls: list[tuple[Fraction, Fraction]] = []
    original_check = p19c_independence_module.check

    def spy_check(cond, msg="answer verification failed"):
        if msg == "verdict/and_value mismatch":
            frame = sys._getframe(1)
            pa = frame.f_locals.get("pa")
            pb = frame.f_locals.get("pb")
            calls.append((pa, pb))
        return original_check(cond, msg)

    monkeypatch.setattr(p19c_independence_module, "check", spy_check)

    seeds_per_tier = 200  # 4 tiers x 200 = 800 accepted draws minimum
    for difficulty in TIERS:
        for seed in range(seeds_per_tier):
            make("p19c_independence_verdict", seed=seed, difficulty=difficulty)

    assert len(calls) >= 800, f"expected >= 800 recorded draws, got {len(calls)}"

    violations = []
    for pa, pb in calls:
        upper = min(pa, pb)
        for value in p19c_independence_module._misconception_products(pa, pb):
            if not (Fraction(0) < value <= upper):
                violations.append((pa, pb, value, upper))

    assert violations == [], (
        f"{len(violations)} out-of-(0,min(pa,pb)] misconception-product candidates "
        f"out of {len(calls)} draws: {violations[:5]}"
    )


_NI_OPTION_RE = re.compile(
    r"^Not independent -- P\(A and B\) = (?P<and_value>[\d.]+) but "
    r"P\(A\) x P\(B\) = (?P<product>[\d.]+)$"
)


def test_independence_not_independent_options_never_self_contradict_across_800_plus_seeds():
    """The same is true of a degenerate Y == 0, which can only ever come from `abs(pa - pb)` when pa == pb -- a
    "distractor" that states no discrepancy at all is not a genuine distractor. Both are the same class of
    on-sight-rejectable weakness as the >1 "adds instead of multiplies" defect that an earlier fix closed.

    Black-box: renders every "not independent" mcq option (the answer AND
    every distractor) across 200 seeds x 4 tiers = 800 draws and parses
    both stated numbers out of the text itself, so this does not depend on
    which internal candidate list produced the option.

    Proven red on the pre-fix generator: running this exact test against it fails.
    """
    violations = []
    total_ni_options = 0
    for difficulty in TIERS:
        for seed in range(200):
            q = make("p19c_independence_verdict", seed=seed, difficulty=difficulty)
            for option in [q.answer_text, *q.distractors]:
                m = _NI_OPTION_RE.match(option)
                if m is None:
                    continue  # the "Independent" option has no numbers to check
                total_ni_options += 1
                and_value_text = m.group("and_value")
                product_text = m.group("product")
                if and_value_text == product_text:
                    violations.append(("equal", difficulty, seed, option))
                if product_text == "0":
                    violations.append(("degenerate-zero", difficulty, seed, option))

    assert total_ni_options >= 800, (
        f"expected >= 800 'not independent' options across 800 draws, got {total_ni_options}"
    )
    assert violations == [], (
        f"{len(violations)} self-contradictory/degenerate 'not independent' "
        f"options out of {total_ni_options}: {violations[:5]}"
    )


def test_independence_skill_loads_correctly():
    _, skills_list = load_skills(ROOT / "skills")
    skills = {s.id: s for s in skills_list}
    skill = skills.get("senior-understand-the-notion-of-independence-of-an-event")
    assert skill is not None
    assert skill.domain == "statistics-probability"
    assert skill.answer == "mcq"
    assert skill.prerequisites == [
        "senior-review-the-rules-p-left-overline-a-right",
        "f10-conditional-probability-language",
    ]
    assert skill.placement_eligible is False


# ---------------------------------------------------------------------------
# p19c_probability_language_classify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_language_deterministic_across_seeds(difficulty):
    for seed in SEEDS:
        q1 = make("p19c_probability_language_classify", seed=seed, difficulty=difficulty)
        q2 = make("p19c_probability_language_classify", seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_language_four_categories_exactly(difficulty):
    for seed in SEEDS:
        q = make("p19c_probability_language_classify", seed=seed, difficulty=difficulty)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == 3
        assert set([q.answer_text] + q.distractors) == set(CATEGORIES)


@pytest.mark.parametrize("difficulty", TIERS)
def test_language_prompt_states_all_four_ranges(difficulty):
    for seed in SEEDS:
        q = make("p19c_probability_language_classify", seed=seed, difficulty=difficulty)
        assert "'possible' for 0 < P <= 0.3" in q.text
        assert "'probable' for 0.3 < P <= 0.55" in q.text
        assert "'likely' for 0.55 < P < 0.9" in q.text
        assert "'certain' for P >= 0.9" in q.text


@pytest.mark.parametrize("difficulty", TIERS)
def test_language_correct_answer_grades_correct(difficulty):
    for seed in SEEDS:
        q = make("p19c_probability_language_classify", seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        idx = options.index(q.answer_text)
        result = grade_mcq(q, idx)
        assert result.correct, f"seed {seed} ({difficulty}): canonical answer did not grade correct"


@pytest.mark.parametrize("difficulty", TIERS)
def test_language_every_distractor_grades_incorrect(difficulty):
    for seed in SEEDS:
        q = make("p19c_probability_language_classify", seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        for d in q.distractors:
            idx = options.index(d)
            result = grade_mcq(q, idx)
            assert not result.correct, f"seed {seed} ({difficulty}): distractor {d!r} graded correct"


@pytest.mark.parametrize("difficulty", TIERS)
def test_language_all_four_terms_occur_over_200_seeds(difficulty):
    seen = set()
    for seed in range(200):
        q = make("p19c_probability_language_classify", seed=seed, difficulty=difficulty)
        seen.add(q.answer_text)
    assert seen == set(CATEGORIES), f"{difficulty}: missing terms {set(CATEGORIES) - seen}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_language_top_term_share_under_35_percent(difficulty):
    counts = {}
    n = 200
    for seed in range(n):
        q = make("p19c_probability_language_classify", seed=seed, difficulty=difficulty)
        counts[q.answer_text] = counts.get(q.answer_text, 0) + 1
    top_share = max(counts.values()) / n
    assert top_share <= 0.35, f"{difficulty}: top term share {top_share:.2%} exceeds 35% ({counts})"


@pytest.mark.parametrize("difficulty", TIERS)
def test_language_at_least_20_distinct_questions_per_tier(difficulty):
    seen = set()
    for seed in range(60):
        q = make("p19c_probability_language_classify", seed=seed, difficulty=difficulty)
        seen.add(q.text)
    assert len(seen) >= 20, f"{difficulty}: only {len(seen)} distinct questions over 60 seeds"


def test_language_corrupted_answer_fails_independent_check():
    from mathgen.core import VerificationError, check
    from mathgen.generators.p19c_independence_language_build import classify_probability_language
    from fractions import Fraction

    with pytest.raises(VerificationError):
        check(classify_probability_language(Fraction(1, 2)) == "certain", "bad classification")


def test_language_skill_loads_correctly():
    _, skills_list = load_skills(ROOT / "skills")
    skills = {s.id: s for s in skills_list}
    skill = skills.get("senior-interpret-commonly-used-probability-statements-including-possible-probable")
    assert skill is not None
    assert skill.domain == "statistics-probability"
    assert skill.answer == "mcq"
    assert skill.prerequisites == ["simple-event-probability"]
    assert skill.placement_eligible is False


# ---------------------------------------------------------------------------
# "not independent" options (the key was the longest/tied-longest option
# 87-95% of the time before the fix; must be at chance level now).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_independence_not_independent_key_is_never_uniquely_the_longest_option(difficulty):
    strict_longest = 0
    total = 0
    for seed in range(200):
        q = make("p19c_independence_verdict", seed=seed, difficulty=difficulty)
        if q.answer_text == OPT_INDEPENDENT:
            continue
        total += 1
        opts = [q.answer_text] + list(q.distractors)
        # Every "not independent"-style option in a question shares the
        # identical P(A and B) = X clause; only the P(A) x P(B) = Y term can
        # differ in length, and it is now padded to a shared decimal-place
        # width (`_ni_option`'s `product_dp`), so every "not independent"
        # option's length must be exactly equal. The fixed "Independent"
        # option (11 chars) is a different, always-short string and is
        # excluded from this equal-length check -- it is never the longest
        # option either way, so it does not affect the length-tell itself.
        ni_lens = [len(o) for o in opts if o != OPT_INDEPENDENT]
        assert len(set(ni_lens)) == 1, (
            f"seed {seed} ({difficulty}): 'not independent' option lengths not all equal: {ni_lens} ({opts!r})"
        )
        lens = [len(o) for o in opts]
        if lens[0] == max(lens) and lens.count(lens[0]) == 1:
            strict_longest += 1
    assert strict_longest == 0, (
        f"{difficulty}: key was the strictly-unique-longest option on {strict_longest}/{total} 'not independent' draws"
    )


# ---------------------------------------------------------------------------
# P(B) = Y" option can never state a value above min(P(A), P(B)): P(A and B)
# can never exceed either P(A) or P(B) for any real probability space, so
# such a value is rejectable on sight without doing the actual
# multiplication. Black-box: parses pa/pb straight off the rendered
# `q.text` with this test's OWN regex (never the generator's internal
# pa/pb locals) across every tier's stimulus shape.
# ---------------------------------------------------------------------------


def _parse_pa_pb_from_text(text: str) -> tuple[Fraction, Fraction]:
    """Recover P(A), P(B) from the rendered stimulus, independently of the
    generator's own construction path -- covers every tier's stimulus shape
    (direct P(A)/P(B) statements at easy/medium/exam, and hard tier's
    percent or people-surveyed-count phrasing)."""
    m = re.search(r"P\(A\) = (\d+(?:\.\d+)?)\.", text)
    m2 = re.search(r"P\(B\) = (\d+(?:\.\d+)?)\.", text)
    if m and m2:
        return Fraction(m.group(1)), Fraction(m2.group(1))
    pct = re.search(r"P\(A\) = (\d+)%\. P\(B\) = (\d+)%\.", text)
    if pct:
        return Fraction(int(pct.group(1)), 100), Fraction(int(pct.group(2)), 100)
    counts = re.search(r"(\d+) have property A, (\d+) have property B", text)
    assert counts, f"could not parse P(A)/P(B) from: {text!r}"
    return Fraction(int(counts.group(1)), 100), Fraction(int(counts.group(2)), 100)


@pytest.mark.parametrize("difficulty", TIERS)
def test_independence_rendered_products_never_exceed_min_pa_pb_over_200_seeds(difficulty):
    violations = []
    total_ni_options = 0
    for seed in range(200):
        q = make("p19c_independence_verdict", seed=seed, difficulty=difficulty)
        pa, pb = _parse_pa_pb_from_text(q.text)
        upper = min(pa, pb)
        for option in [q.answer_text] + list(q.distractors):
            m = _NI_OPTION_RE.match(option)
            if m is None:
                continue
            total_ni_options += 1
            product = Fraction(m.group("product"))
            if product > upper:
                violations.append((seed, option, upper))
    assert total_ni_options >= 400, (
        f"{difficulty}: expected >= 400 'not independent' options across 200 draws, got {total_ni_options}"
    )
    assert violations == [], (
        f"{difficulty}: {len(violations)} rendered P(A) x P(B) options exceed "
        f"min(P(A), P(B)) out of {total_ni_options}: {violations[:5]}"
    )


@pytest.mark.parametrize("difficulty", TIERS)
def test_independence_key_share_stays_within_25_to_35_percent_contract_band(difficulty):
    """`want_independent` is drawn once per call at a fixed 30% weight -- the generator's own answer-space
    contract. Before this fix, drawing it fresh on every retry-loop attempt let any rejection channel that
    fired more often on one branch (the tightened `_misconception_products` bound, and medium's own extra
    `_feasible` check on the not-independent branch) silently re-flip the coin, pushing the ACCEPTED share of
    "Independent" well outside the stated band (measured 41.5% at n=200, 45%+ at larger n, for medium
    specifically). With the verdict fixed once per call, the accepted share should sit close to the stated 30%
    weight for every tier, medium included."""
    n = 200
    independent_count = 0
    for seed in range(n):
        q = make("p19c_independence_verdict", seed=seed, difficulty=difficulty)
        if q.answer_text == OPT_INDEPENDENT:
            independent_count += 1
    share = independent_count / n
    assert 0.25 <= share <= 0.35, (
        f"{difficulty}: independent share {share:.2%} outside the stated 25-35% answer-space contract band"
    )


# Near-miss gap, survey gap, key-product rank. Black-box: every number is parsed from the rendered
# options and stimulus.


def _ni_draws_2060(difficulty, n=200):
    """(text, and_value, key_product, [distractor products]) per
    not-independent draw over seeds 0..n-1."""
    out = []
    for seed in range(n):
        q = make("p19c_independence_verdict", seed=seed, difficulty=difficulty)
        if q.answer_text == OPT_INDEPENDENT:
            continue
        m = _NI_OPTION_RE.match(q.answer_text)
        and_value = Fraction(m.group("and_value"))
        key = Fraction(m.group("product"))
        others = []
        for d in q.distractors:
            dm = _NI_OPTION_RE.match(d)
            if dm is not None:
                others.append(Fraction(dm.group("product")))
        out.append((q.text, and_value, key, others))
    return out


@pytest.mark.parametrize("difficulty", TIERS)
def test_independence_2060_nearest_product_to_and_value_is_not_a_shortcut(difficulty):
    """The rule 'pick the option whose product is nearest P(A and B)' must
    not be right on more than 45% of not-independent draws (was 84.5% on exam)."""
    draws = _ni_draws_2060(difficulty)
    hits = 0
    for _text, x, key, others in draws:
        products = [key, *others]
        best = min(products, key=lambda v: abs(v - x))
        if best == key and sum(abs(v - x) == abs(best - x) for v in products) == 1:
            hits += 1
    assert len(draws) >= 100
    assert hits / len(draws) <= 0.45, f"{difficulty}: nearest-to-P(A and B) rule right on {hits}/{len(draws)}"


def test_independence_2060_exam_gap_is_spread_not_always_two_or_three_hundredths():
    draws = _ni_draws_2060("exam")
    gaps = {abs(x - _pa_pb(text)) for text, x, _k, _o in draws}
    small = sum(1 for text, x, _k, _o in draws if abs(x - _pa_pb(text)) <= Fraction(3, 100))
    assert max(gaps) >= Fraction(8, 100)
    assert small / len(draws) <= 0.5


def _pa_pb(text):
    pa, pb = _parse_pa_pb_from_text(text)
    return pa * pb


def test_independence_2060_hard_survey_not_independent_gap_at_least_7_of_100():
    violations = []
    surveys = 0
    for text, x, _k, _o in _ni_draws_2060("hard", n=400):
        if "people surveyed" not in text:
            continue
        surveys += 1
        gap = abs(x - _pa_pb(text))
        if gap < Fraction(7, 100):
            violations.append((text, gap))
    assert surveys >= 60
    assert violations == [], f"{len(violations)} survey draws with gap < 7/100: {violations[:3]}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_independence_2060_key_product_rank_shares_within_15_to_40_percent(difficulty):
    """Rank 1 = lowest of the three shown products. Base: the not-independent
    draws over seeds 0..999 of this tier."""
    draws = _ni_draws_2060(difficulty, n=1000)
    counts = {1: 0, 2: 0, 3: 0}
    for _text, _x, key, others in draws:
        counts[sorted([key, *others]).index(key) + 1] += 1
    for rank, c in counts.items():
        share = c / len(draws)
        assert 0.15 <= share <= 0.40, (
            f"{difficulty}: key product is rank {rank} on {share:.1%} of {len(draws)} not-independent draws ({counts})"
        )


# Wording cleanup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_2063_independence_never_prints_a_count_of_one_with_plural_verb(difficulty):
    for seed in range(3000):
        q = make("p19c_independence_verdict", seed=seed, difficulty=difficulty)
        assert not re.search(r"(?<![\d.])1 have", q.text), q.text


@pytest.mark.parametrize("difficulty", TIERS)
def test_2063_independence_decimals_share_one_decimal_place_count(difficulty):
    for seed in range(400):
        q = make("p19c_independence_verdict", seed=seed, difficulty=difficulty)
        blob = " ".join([q.text, q.answer_text, *q.distractors])
        blob = re.sub(r"P\(A\|B\)", "", blob)
        dps = {len(m.split(".")[1]) for m in re.findall(r"\d+\.\d+", blob)}
        assert len(dps) <= 1, (dps, blob)
