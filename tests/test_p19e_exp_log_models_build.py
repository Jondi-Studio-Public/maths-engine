"""Tests `p19e_exponential_model_value` and `p19e_log_model_solve_input`, including the cross-row separation
check between the two.

The two skills share template wording closely enough (about 0.82 similarity) to read as near-duplicates, so
this file also carries a cross-row separation check.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check

SEEDS = range(60)
DIFFS = ["easy", "medium", "hard", "exam"]

TOPIC_EXP = "p19e_exponential_model_value"
SKILL_EXP = "senior-identify-contexts-suitable-for-modelling-by-exponential-functions"
TOPIC_LOG = "p19e_log_model_solve_input"
SKILL_LOG = "senior-identify-contexts-suitable-for-modelling-by-logarithmic-functions"

_BAD_PHRASE_RE = re.compile(r"how long|when will|after how many", re.I)


# =============================================================================
# Shared per-topic property checks
# =============================================================================


def _run_common_properties(topic, min_distractors=2):
    for difficulty in DIFFS:
        for seed in SEEDS:
            q1 = make(topic, seed=seed, difficulty=difficulty)
            q2 = make(topic, seed=seed, difficulty=difficulty)
            assert q1.to_json() == q2.to_json(), (topic, difficulty, seed)

            assert q1.answer_text not in q1.distractors, (topic, difficulty, seed)
            assert len(q1.distractors) == len(set(q1.distractors)), (topic, difficulty, seed)

            assert grade_number(q1, q1.answer_text).correct is True, (topic, difficulty, seed, q1.text)
            for d in q1.distractors:
                assert grade_number(q1, d).correct is False, (topic, difficulty, seed, q1.text, d)

            target = float(q1.answer_text)
            wrong = target + 5.0
            assert grade_number(q1, f"{wrong:.1f}").correct is False


@pytest.mark.parametrize("topic", [TOPIC_EXP, TOPIC_LOG])
def test_common_properties(topic):
    _run_common_properties(topic)


@pytest.mark.parametrize("topic", [TOPIC_EXP, TOPIC_LOG])
@pytest.mark.parametrize("difficulty", DIFFS)
def test_at_least_20_distinct_texts(topic, difficulty):
    texts = {make(topic, seed=seed, difficulty=difficulty).text for seed in SEEDS}
    assert len(texts) >= 20, f"{topic} {difficulty}: only {len(texts)} distinct texts over {len(SEEDS)} seeds"


@pytest.mark.parametrize("topic", [TOPIC_EXP, TOPIC_LOG])
@pytest.mark.parametrize("difficulty", DIFFS)
def test_two_or_more_distractors_at_200_seeds_default_and_tier(topic, difficulty):
    """>=85% of draws must carry >=2 distractors, both for the default params
    (make(topic, seed=s), no difficulty) and for every named tier."""

    def _share(diff):
        n_ok = 0
        for seed in range(200):
            q = make(topic, seed=seed, difficulty=diff)
            if len(q.distractors) >= 2:
                n_ok += 1
        return n_ok / 200

    assert _share(difficulty) >= 0.85, f"{topic} {difficulty}: <2 distractors too often"
    if difficulty == "easy":
        assert _share(None) >= 0.85, f"{topic} default: <2 distractors too often"


@pytest.mark.parametrize("topic", [TOPIC_EXP, TOPIC_LOG])
@pytest.mark.parametrize("difficulty", DIFFS)
def test_answer_concentration_at_200_seeds(topic, difficulty):
    counts = Counter()
    for seed in range(200):
        q = make(topic, seed=seed, difficulty=difficulty)
        counts[q.answer_text] += 1
    total = sum(counts.values())
    top_share = counts.most_common(1)[0][1] / total
    assert top_share <= 0.15, f"{topic} {difficulty}: top answer share {top_share:.3f} exceeds 0.15"


@pytest.mark.parametrize("topic", [TOPIC_EXP, TOPIC_LOG])
@pytest.mark.parametrize("difficulty", DIFFS)
def test_instruction_phrase_in_text_and_latex(topic, difficulty):
    for seed in range(20):
        q = make(topic, seed=seed, difficulty=difficulty)
        assert "decimal place" in q.text or "whole number" in q.text
        assert "decimal place" in q.latex or "whole number" in q.latex


@pytest.mark.parametrize("topic", [TOPIC_EXP, TOPIC_LOG])
@pytest.mark.parametrize("difficulty", DIFFS)
def test_answers_and_distractors_formatted_alike(topic, difficulty):
    """No option identifiable by its shape: every option carries the same
    number of decimal places (or the same lack of one)."""
    for seed in range(30):
        q = make(topic, seed=seed, difficulty=difficulty)
        answer_dp = len(q.answer_text.split(".", 1)[1]) if "." in q.answer_text else 0
        for d in q.distractors:
            d_dp = len(d.split(".", 1)[1]) if "." in d else 0
            assert d_dp == answer_dp, (topic, difficulty, seed, q.answer_text, d)


@pytest.mark.parametrize("topic", [TOPIC_EXP, TOPIC_LOG])
def test_answer_under_four_digits(topic):
    for difficulty in DIFFS:
        for seed in range(60):
            q = make(topic, seed=seed, difficulty=difficulty)
            whole = q.answer_text.split(".")[0].lstrip("-")
            assert len(whole) <= 3, (topic, difficulty, seed, q.answer_text)


# =============================================================================
# Required cross-row check: evaluate vs solve-for-input, no overlap
# =============================================================================


def test_cross_row_separation_evaluation_vs_input_solving():
    exp_texts = set()
    log_texts = set()
    bad_exp = 0
    for difficulty in DIFFS:
        for seed in range(200):
            qe = make(TOPIC_EXP, seed=seed, difficulty=difficulty)
            exp_texts.add(qe.text)
            if _BAD_PHRASE_RE.search(qe.text):
                bad_exp += 1
            ql = make(TOPIC_LOG, seed=seed, difficulty=difficulty)
            log_texts.add(ql.text)

    assert bad_exp == 0, f"{bad_exp} exponential prompts asked for an elapsed time instead of a value"
    overlap = exp_texts & log_texts
    assert not overlap, f"{len(overlap)} prompt texts shared between the two rows: {sorted(overlap)[:3]}"
    print(
        f"cross-row check: {len(exp_texts)} exponential prompts, {len(log_texts)} logarithmic "
        f"prompts, {len(overlap)} overlapping, {bad_exp} exponential prompts with a 'how long'-style phrase."
    )


# =============================================================================
# In-generator independent-check red-first proofs
# =============================================================================


def test_exp_independent_check_fires_on_corrupted_value():
    """Mirrors the generator's own iterative-compounding residual check with a corrupted claimed value, proving
    `check` raises rather than passing silently."""
    for seed in range(10):
        q = make(TOPIC_EXP, seed=seed, difficulty="easy")
        value = float(q.answer_text)
        check(abs(value - value) < 1e-6, "sanity: the claimed value must equal itself")
        corrupted = value + 500.0
        with pytest.raises(VerificationError):
            check(abs(corrupted - value) < 1e-6, "a corrupted value must not equal the claimed answer")


def test_log_independent_check_fires_on_corrupted_value():
    """Mirrors the generator's own bisection-root residual check with a
    corrupted claimed root, proving `check()` raises (proven red first)."""
    import math

    for seed in range(10):
        q = make(TOPIC_LOG, seed=seed, difficulty="easy")
        x = float(q.answer_text)
        # Parse "y = {a} log10(x){b_part}" out of q.text and confirm the
        # claimed x satisfies it, then corrupt it and confirm check() fires.
        m = re.search(r"y = (\d+) log10\(x\)([+\- ]*\d*)", q.text)
        a = int(m.group(1))
        b_part = m.group(2).strip()
        b = int(b_part.replace(" ", "")) if b_part else 0
        m2 = re.search(r"when y = (-?\d+)", q.text)
        y = int(m2.group(1))
        residual = a * math.log10(x) + b - y
        check(abs(residual) < 0.5, "sanity: the claimed x must satisfy its own model equation")
        corrupted = x + 500.0
        corrupted_residual = a * math.log10(corrupted) + b - y
        with pytest.raises(VerificationError):
            check(abs(corrupted_residual) < 0.5, "a corrupted x must not satisfy the model equation")


# =============================================================================
# Fixes
# =============================================================================


def test_exp_small_population_and_cell_contexts_use_thousands():
    """#1: a town/colony context must not be a raw small integer count of
    people/cells (e.g. "starts at 112 people"); it must be stated in
    thousands. RED on the pre-fix generator (unit was bare "people"/"cells")."""
    for seed in range(60):
        for difficulty in DIFFS:
            q = make(TOPIC_EXP, seed=seed, difficulty=difficulty)
            assert "people" not in q.text or "thousand people" in q.text, q.text
            assert (
                "cells" not in q.text or "thousand cells" in q.text or "cell count" in q.text and "thousand" in q.text
            ), q.text


def test_exp_exam_two_phase_never_same_rate():
    """#2: the two exam-tier phases must never share BOTH direction and rate
    (that is not a rate change). RED on the pre-fix generator (8/200 hits in
    a baseline measurement before this fix)."""
    n_same = 0
    for seed in range(200):
        q = make(TOPIC_EXP, seed=seed, difficulty="exam")
        if "then" in q.text and ("grows" in q.text or "decays" in q.text):
            verbs = re.findall(r"\b(grows|decays)\b", q.text)
            rates = re.findall(r"(\d+)%", q.text)
            if len(verbs) == 2 and len(rates) == 2 and verbs[0] == verbs[1] and rates[0] == rates[1]:
                n_same += 1
    assert n_same == 0, f"{n_same}/200 two-phase exam draws used the same rate in both phases"


def test_exp_exam_distractor_floor_improved():
    """#2: >= 2 distractors on a higher share of exam draws once the
    forgot-phase-2 and no-change fallback candidates are added. RED on the
    pre-fix generator (192/200 = 96.0%)."""
    n_ok = 0
    for seed in range(200):
        q = make(TOPIC_EXP, seed=seed, difficulty="exam")
        if len(q.distractors) >= 2:
            n_ok += 1
    assert n_ok / 200 >= 0.97, f"only {n_ok}/200 exam draws carry >=2 distractors"


def test_log_medium_integer_coefficient_prints_bare():
    """#3: an integer ln coefficient prints without parens ("2 ln(x)"), only
    a genuine fraction is bracketed ("(5/2) ln(x)"). RED on the pre-fix
    generator (every coefficient was bracketed, e.g. "(2) ln(x)")."""
    seen_integer = False
    for seed in range(200):
        q = make(TOPIC_LOG, seed=seed, difficulty="medium")
        m = re.search(r"y = (\(?-?[\d/]+\)?) ln\(x\)", q.text)
        assert m is not None, q.text
        coef = m.group(1)
        if "/" not in coef:
            assert not coef.startswith("("), f"bare integer coefficient printed with parens: {q.text}"
            seen_integer = True
        else:
            assert coef.startswith("(") and coef.endswith(")"), f"fraction coefficient missing parens: {q.text}"
    assert seen_integer, "no integer-coefficient medium draw seen over 200 seeds"


def test_log_medium_negative_a_share_near_half():
    """#3: negative a should land near half the tier (the pool is 7/14
    negative), not be starved by an asymmetric rejection filter. RED on the
    pre-fix generator (2/200 in a baseline measurement before this fix)."""
    n_neg = 0
    for seed in range(200):
        q = make(TOPIC_LOG, seed=seed, difficulty="medium")
        m = re.search(r"y = \(?(-?[\d/]+)\)? ln\(x\)", q.text)
        assert m is not None, q.text
        if m.group(1).startswith("-"):
            n_neg += 1
    assert 0.35 <= n_neg / 200 <= 0.65, f"negative-a share {n_neg}/200 not near half"


def test_log_medium_target_y_never_negative():
    """The medium-tier stated target y must lie in
    the quantity's real range -- never negative (e.g. a seedling's height
    cannot reach -7 cm). RED on the pre-fix generator: 92/200 seeds
    drew a negative y in a baseline measurement before this fix."""
    n_neg = 0
    for seed in range(200):
        q = make(TOPIC_LOG, seed=seed, difficulty="medium")
        m = re.search(r"when y = (-?\d+)", q.text)
        assert m is not None, q.text
        if int(m.group(1)) < 0:
            n_neg += 1
    assert n_neg == 0, f"{n_neg}/200 medium-tier draws stated a negative target y"


_LN_RISING = ("seedling's height", "recall score")


def test_log_medium_context_direction_matches_sign_of_a():
    """A negative a makes y FALL as x grows, so it must never be told as a seedling growing
    less with more fertiliser or a recall score dropping with more study. The rising contexts go with a > 0
    and the falling ones with a < 0. RED before the fix: 22/100 medium draws paired a negative a with a rising
    context."""
    for seed in range(200):
        q = make(TOPIC_LOG, seed=seed, difficulty="medium")
        m = re.search(r"y = \(?(-?[\d/]+)\)? ln\(x\)", q.text)
        assert m is not None, q.text
        rising = any(c in q.text for c in _LN_RISING)
        assert rising == (not m.group(1).startswith("-")), q.text


def test_log_exam_wording_never_says_units_of():
    """#4: the exam prompt must name the quantity properly ("how many more
    hours of practice", "how much more intensity"), never "units of X". RED
    on the pre-fix generator (200/200 said "units of")."""
    for seed in range(60):
        q = make(TOPIC_LOG, seed=seed, difficulty="exam")
        assert "units of" not in q.text, q.text
        assert re.search(r"how (many|much) more", q.text), q.text


# =============================================================================
# Skill metadata
# =============================================================================


def test_skill_exp_loads_with_correct_metadata():
    domains, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_EXP), None)
    assert skill is not None
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == [
        "senior-review-indices-including-fractional-indices-and-the-index",
        "percentage-change",
    ]
    all_stages = set()
    for source in skill.sources:
        all_stages.update(source.stages)
    assert "exam" in all_stages


def test_skill_log_loads_with_correct_metadata():
    domains, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_LOG), None)
    assert skill is not None
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == [
        "senior-define-logarithms-as-indices-a-x-b-is",
        "senior-identify-contexts-suitable-for-modelling-by-exponential-functions",
    ]
    all_stages = set()
    for source in skill.sources:
        all_stages.update(source.stages)
    assert "exam" in all_stages
