"""Tests for `mathgen/generators/p19b_association_describe_build.py`: `p19b_association_describe` (skill
`senior-describe-the-association-between-two-numerical-variables-in`, ACMEM139/ACMGM053).

This file holds the generator-specific
proofs: the in-generator band checks firing on a corrupted value, the
answer-space contract (three independent fields, key-share cap, always >= 2
distractors), the real `grade_ordered_list` battery including the
grader probes, and each tier's own forced condition.

The independent ORACLE lives in
`tests/test_p19b_association_describe_oracle.py`.
"""

from __future__ import annotations

import collections
import re

import pytest

from mathgen.grading.typed import grade_ordered_list
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

import mathgen.generators.p19b_association_describe_build as gen

TOPIC = "p19b_association_describe"
TIERS = ("easy", "medium", "hard", "exam")
DIRECTIONS = ("positive", "negative")
FORMS = ("linear", "non-linear")
STRENGTHS = ("strong", "moderate", "weak")


def _all(tier, seeds=range(60)):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


# ============================================================================
# Text quality / variety
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_no_bad_text_patterns(tier):
    for q in _all(tier, range(30)):
        defects = find_text_defects(q)
        assert defects == [], f"{tier}: {defects} for {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_variety_at_least_3_shapes(tier):
    shapes = {prompt_shape(q) for q in _all(tier)}
    assert len(shapes) >= 3, f"{tier}: only {len(shapes)} shapes: {shapes}"


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_prompts(tier):
    texts = {q.text for q in _all(tier)}
    assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct prompts"


@pytest.mark.parametrize("tier", TIERS)
def test_determinism(tier):
    for seed in (0, 1, 2, 40, 59):
        first = make(TOPIC, seed=seed, difficulty=tier).to_json()
        second = make(TOPIC, seed=seed, difficulty=tier).to_json()
        assert first == second


# ============================================================================
# Answer-space contract
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_three_field_list(tier):
    for q in _all(tier, range(30)):
        assert isinstance(q.answer, list) and len(q.answer) == 3
        direction, form, strength = q.answer
        assert direction in DIRECTIONS
        assert form in FORMS
        assert strength in STRENGTHS
        assert q.answer_text == f"{direction}, {form}, {strength}"


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_two_distractors_every_draw(tier):
    for q in _all(tier, range(60)):
        assert len(q.distractors) >= 2, f"{tier}: only {len(q.distractors)} distractors"


@pytest.mark.parametrize("tier", TIERS)
def test_key_share_under_20_percent_at_200_seeds(tier):
    counts = collections.Counter(make(TOPIC, seed=s, difficulty=tier).answer_text for s in range(200))
    for triple, n in counts.items():
        assert n <= 40, f"{tier}: {triple} is the key on {n}/200 draws (cap 40 = 20%)"


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_differ_from_key_by_exactly_one_field(tier):
    for q in _all(tier, range(60)):
        key = tuple(q.answer)
        for d in q.distractors:
            cand = tuple(part.strip() for part in d.split(","))
            diffs = sum(1 for a, b in zip(key, cand) if a != b)
            assert diffs == 1, f"{tier}: distractor {d!r} differs from key {key} in {diffs} fields"


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_pairwise_distinct_and_not_the_key(tier):
    for q in _all(tier, range(60)):
        all_triples = [q.answer_text, *q.distractors]
        assert len(set(all_triples)) == len(all_triples), f"{tier}: duplicate among {all_triples}"


# ============================================================================
# In-generator band checks fire on a corrupted value
# ============================================================================


@pytest.mark.parametrize("tier", ("medium", "hard", "exam"))
def test_band_check_fires_on_corrupted_classifier(tier, monkeypatch):
    monkeypatch.setattr(gen, "_classify_r_strength", lambda abs_r: "not-a-real-band")
    # exam's "non-trap" branch and both linear branches of medium/hard call
    # `_classify_r_strength`; run enough seeds that at least one lands there.
    saw_failure = False
    for seed in range(30):
        try:
            make(TOPIC, seed=seed, difficulty=tier)
        except VerificationError:
            saw_failure = True
            break
    assert saw_failure, f"{tier}: corrupted classifier never fired over 30 seeds"


def test_classify_r_strength_boundaries():
    from fractions import Fraction

    assert gen._classify_r_strength(Fraction(80, 100)) == "strong"
    assert gen._classify_r_strength(Fraction(60, 100)) == "moderate"
    assert gen._classify_r_strength(Fraction(30, 100)) == "weak"
    assert gen._classify_r_strength(Fraction(10, 100)) is None


# ============================================================================
# Real grader: accepts the key, rejects every distractor and a corrupted value
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_key_rejects_distractors(tier):
    for q in _all(tier, range(30)):
        result = grade_ordered_list(q, q.answer_text)
        assert result.correct, f"{tier}: key rejected for {q.text!r}"
        for d in q.distractors:
            result = grade_ordered_list(q, d)
            assert not result.correct, f"{tier}: distractor {d!r} accepted for {q.text!r}"


def test_grader_probes_from_task_file():
    """The probe table, run through the real grader on a stub question."""

    class Stub:
        answer = ["positive", "linear", "strong"]
        answer_text = "positive, linear, strong"

    q = Stub()
    assert grade_ordered_list(q, "Positive, Linear, Strong").correct
    assert grade_ordered_list(q, " positive,linear , strong ").correct
    assert not grade_ordered_list(q, "nonlinear").correct
    assert not grade_ordered_list(q, "non linear").correct


# ============================================================================
# Tier-specific forced conditions
# ============================================================================


def test_easy_never_uses_answer_words_as_labels():
    for q in _all("easy", range(60)):
        # The description clauses never literally say "positive"/"negative"
        # or "strong"/"moderate"/"weak" -- only the trailing instruction
        # sentence (which explains the label vocabulary itself) may.
        body = q.text.split("State your answer")[0]
        for word in ("positive", "negative", "strong", "moderate", "weak"):
            assert word not in body.lower(), f"easy body leaks {word!r}: {body!r}"


def test_medium_non_linear_branch_never_states_r():
    for q in _all("medium", range(80)):
        if "non-linear" == q.answer[1]:
            assert "r =" not in q.text


def test_hard_linear_branch_never_states_r_directly():
    for q in _all("hard", range(80)):
        if q.answer[1] == "linear":
            assert "r =" not in q.text
            assert "r^2 =" in q.text


def test_exam_gives_r_and_residual_plot_on_every_draw():
    for q in _all("exam", range(60)):
        assert "r =" in q.text
        assert "residual plot" in q.text


def test_exam_trap_and_nontrap_both_occur():
    seen_nonlinear = seen_linear = False
    for q in _all("exam", range(60)):
        if q.answer[1] == "non-linear":
            seen_nonlinear = True
        else:
            seen_linear = True
    assert seen_nonlinear and seen_linear, "exam should draw both the trap and non-trap branch"


# ============================================================================
# Context is a property of direction,
# the hard tier's line is plausible over its stated range, and
# every exam prompt states both strength band rules.
# ============================================================================

# Literal, hand-written natural-direction table -- deliberately NOT read from
# `gen._CONTEXTS` (a bank edit that silently flipped a context's own
# `direction` field would not be caught by comparing the drawn scenario
# against the bank's own claim about itself; this table is the independent
# real-world ground truth).
_EXPECTED_CONTEXT_DIRECTION = {
    "hours studied per week": "positive",
    "daily maximum temperature": "positive",
    "weekly training hours": "positive",
    "monthly advertising spend": "positive",
    "a plant's daily sunlight hours": "positive",
    "a car's age": "negative",
    "altitude": "negative",
    "the number of days absent per term": "negative",
    "the price": "negative",
    "hours of screen time per day": "negative",
}


def test_context_direction_matches_bank_over_1200_draws():
    """1200 draws (300 seeds x 4 tiers), matching an earlier
    measurement -- every context's x-phrase appears with exactly ONE
    direction, and it is the context's own honest real-world natural
    direction (an independent literal table, not the bank's self-report)."""
    assert {c["x"] for c in gen._CONTEXTS} == set(_EXPECTED_CONTEXT_DIRECTION), (
        "context bank and the independent expected-direction table have diverged in their set of contexts"
    )
    for ctx in gen._CONTEXTS:
        assert ctx["direction"] == _EXPECTED_CONTEXT_DIRECTION[ctx["x"]], (
            f"bank claims {ctx['x']!r} is {ctx['direction']!r}, but its real-world "
            f"natural direction is {_EXPECTED_CONTEXT_DIRECTION[ctx['x']]!r}"
        )

    seen = collections.defaultdict(set)
    total = 0
    for tier in TIERS:
        for s in range(300):
            q = make(TOPIC, seed=s, difficulty=tier)
            total += 1
            matches = [c for c in gen._CONTEXTS if c["x"] in q.text]
            assert len(matches) == 1, f"{tier} seed={s}: ambiguous context match: {q.text!r}"
            seen[matches[0]["x"]].add(q.answer[0])
    assert total == 1200
    for x, expected_direction in _EXPECTED_CONTEXT_DIRECTION.items():
        assert seen[x] == {expected_direction}, (
            f"context {x!r} appeared with direction(s) {seen[x]}, "
            f"expected only its natural direction {expected_direction!r}"
        )


_LINE_RE = re.compile(r"y = (-?\d+) ([+-]) (\d+\.\d+)x")


def test_hard_linear_line_plausible_over_stated_range():
    """The hard tier's least-squares line must predict a plausible y at
    BOTH ends of the context's own stated x-range (construct-to-target,
    defect #2) -- never an implausible value (negative score, negative
    sales, and so on) anywhere inside the range the prompt itself gives."""
    checked = 0
    for s in range(300):
        q = make(TOPIC, seed=s, difficulty="hard")
        if q.answer[1] != "linear":
            continue
        m = _LINE_RE.search(q.text)
        assert m, f"no line found in hard-linear text: {q.text!r}"
        a = int(m.group(1))
        b = float(m.group(3)) * (1 if m.group(2) == "+" else -1)
        matches = [c for c in gen._CONTEXTS if c["x"] in q.text]
        assert len(matches) == 1
        ctx = matches[0]
        assert f"where x is {ctx['x']}" in q.text, f"seed={s}: line does not name x in words: {q.text!r}"
        assert f"and y is {ctx['y']}" in q.text, f"seed={s}: line does not name y in words: {q.text!r}"
        x_lo, x_hi = ctx["x_range"]
        y_lo, y_hi = ctx["y_range"]
        for x in (x_lo, x_hi):
            y = a + b * x
            assert y_lo - 1e-9 <= y <= y_hi + 1e-9, (
                f"seed={s}: line y={a}{'+' if b >= 0 else '-'}{abs(b)}x gives y={y} at x={x}, "
                f"outside {ctx['x']!r}'s plausible range {ctx['y_range']}"
            )
        checked += 1
    assert checked >= 20, f"only {checked} hard-linear draws seen over 300 seeds"


def test_exam_states_both_strength_band_rules():
    for q in _all("exam", range(100)):
        assert "For a linear relationship, judge strength from the size of r" in q.text
        assert "For a non-linear relationship, judge strength from how closely" in q.text


# ============================================================================
# Skill load
# ============================================================================


def test_skill_loads_with_expected_shape():
    _domains, skills = load_skills()
    by_id = {s.id: s for s in skills}
    skill = by_id["senior-describe-the-association-between-two-numerical-variables-in"]
    assert skill.prerequisites == [
        "simple-event-probability",
        "senior-identify-the-response-variable-and-the-explanatory-variable",
    ]
    assert skill.answer == "ordered_list"
    assert skill.placement_eligible is False
    stages = {s.difficulty: s.stages for s in skill.sources}
    assert stages["easy"] == [6]
    assert stages["medium"] == [6]
    assert stages["hard"] == [6, "maintenance"]
    assert stages["exam"] == ["exam"]
