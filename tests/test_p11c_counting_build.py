"""Factorial notation + restricted permutations.

senior-use-factorial-notation (ACMSM003, topic p11c_factorial_notation,
answer: expression) and
senior-solve-problems-involving-permutations-and-restrictions-with-or
(ACMSM004, topic p11c_restricted_permutations, answer: number). See the
findings (permutations/combinations/multiplication-principle never write
n! or state a restriction).
"""

import sympy

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent, check_pairwise_distinct
from mathgen.generators.p11c_counting_build import (
    _EASY_COMBO_POOL,
    _easy_combo_value,
    p11c_factorial_notation,
)

FACT_TOPIC = "p11c_factorial_notation"
FACT_SKILL = "senior-use-factorial-notation"
RESTR_TOPIC = "p11c_restricted_permutations"
RESTR_SKILL = "senior-solve-problems-involving-permutations-and-restrictions-with-or"

TIERS = ("easy", "medium", "hard")
SEEDS = range(60)

GRADE_EXPRESSION = GRADERS["expression"]
GRADE_NUMBER = GRADERS["number"]


# --- Determinism -----------------------------------------------------------


@pytest.mark.parametrize("topic", [FACT_TOPIC, RESTR_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_generator_is_deterministic(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        repeat = make(topic, seed=seed, difficulty=tier)
        assert q.to_json() == repeat.to_json()


# --- Distractor hygiene ------------------------------------------------


@pytest.mark.parametrize("topic", [FACT_TOPIC, RESTR_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.answer_text not in q.distractors, f"seed={seed}: answer in distractors"
        assert len(q.distractors) == len(set(q.distractors)), f"seed={seed}: dup distractors"
        assert len(q.distractors) >= 1


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_factorial_notation_real_grader(tier):
    for seed in SEEDS:
        q = make(FACT_TOPIC, seed=seed, difficulty=tier)
        result = GRADE_EXPRESSION(q, q.answer_text)
        assert result.correct, f"seed={seed}/{tier}: canonical not accepted: {q.answer_text!r}"
        for d in q.distractors:
            result = GRADE_EXPRESSION(q, d)
            assert not result.correct, f"seed={seed}/{tier}: distractor {d!r} accepted"
        wrong = "0"
        if wrong == q.answer_text:
            wrong = (
                str(int(q.answer_text) + 999999) if q.answer_text.lstrip("-").isdigit() else q.answer_text + " + 999"
            )
        elif not q.answer_text.lstrip("-").isdigit():
            wrong = q.answer_text + " + 999"
        result = GRADE_EXPRESSION(q, wrong)
        assert not result.correct, f"seed={seed}/{tier}: deliberately wrong {wrong!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_restricted_permutations_real_grader(tier):
    for seed in SEEDS:
        q = make(RESTR_TOPIC, seed=seed, difficulty=tier)
        result = GRADE_NUMBER(q, q.answer_text)
        assert result.correct, f"seed={seed}/{tier}: canonical not accepted: {q.answer_text!r}"
        for d in q.distractors:
            result = GRADE_NUMBER(q, d)
            assert not result.correct, f"seed={seed}/{tier}: distractor {d!r} accepted"
        wrong = str(int(q.answer_text) + 999999)
        result = GRADE_NUMBER(q, wrong)
        assert not result.correct, f"seed={seed}/{tier}: deliberately wrong {wrong!r} accepted"


# --- Bare-integer parsing through the expression grader -


def test_grade_expression_accepts_a_bare_integer_string():
    """Confirmed directly, not assumed: `_parse_expression("24")` returns a
    sympy Integer, and `grade_expression` compares it to a canonical "24" via
    `symbolically_equal`, which accepts it."""
    from mathgen.grading.typed import _parse_expression

    parsed = _parse_expression("24")
    assert isinstance(parsed, sympy.Integer)
    assert parsed == 24
    q = make(FACT_TOPIC, seed=0, difficulty="easy")
    assert q.answer_text.lstrip("-").isdigit()
    assert GRADE_EXPRESSION(q, q.answer_text).correct


# --- Answer kind ---------------------------------------------------------


def test_factorial_notation_easy_medium_answers_are_bare_integers():
    for seed in SEEDS:
        for tier in ("easy", "medium"):
            q = make(FACT_TOPIC, seed=seed, difficulty=tier)
            assert q.answer_text.isdigit(), f"seed={seed}/{tier}: not a bare integer: {q.answer_text!r}"


def test_factorial_notation_medium_answer_under_1000():
    for seed in SEEDS:
        q = make(FACT_TOPIC, seed=seed, difficulty="medium")
        assert int(q.answer_text) < 1000, f"seed={seed}: medium answer not under 4 digits: {q.answer_text}"


def test_factorial_notation_easy_never_reaches_four_digits():
    """Every value the easy tier can ever display -- rendered answer AND
    distractor -- stays under 1000, checked against real make() output over
    60 seeds."""
    for seed in SEEDS:
        q = make(FACT_TOPIC, seed=seed, difficulty="easy")
        assert int(q.answer_text) < 1000, f"seed={seed}: easy answer not under 4 digits: {q.answer_text}"
        for d in q.distractors:
            if d.lstrip("-").isdigit():
                assert int(d) < 1000, f"seed={seed}: easy distractor not under 4 digits: {d}"


def test_factorial_notation_easy_combo_pool_is_enumerated_and_under_1000():
    """The requirement: assert the bound directly against the actual
    enumerated (a, b, op) pool the generator draws from, not a sample of
    make() output. Every combination the pool contains must genuinely stay
    under 1000 -- recomputed here independently of the generator's own
    `_easy_combo_value`, via a fresh manual term-by-term factorial product
    for each operand."""
    assert len(_EASY_COMBO_POOL) >= 20, f"easy combo pool only has {len(_EASY_COMBO_POOL)} (a, b, op) entries"
    for a, b, op in _EASY_COMBO_POOL:
        assert 0 <= a <= 6 and 0 <= b <= 6, f"operand out of {{0..6}}: a={a}, b={b}"
        assert op in ("+", "−", "×"), f"unexpected operator: {op!r}"
        if op == "−":
            assert a > b, f"subtraction combo with a <= b would be undefined: a={a}, b={b}"
        manual_fa = 1
        for i in range(1, a + 1):
            manual_fa *= i
        manual_fb = 1
        for i in range(1, b + 1):
            manual_fb *= i
        if op == "+":
            value = manual_fa + manual_fb
        elif op == "−":
            value = manual_fa - manual_fb
        else:
            value = manual_fa * manual_fb
        assert value == _easy_combo_value(a, b, op), (
            f"manual recomputation disagrees with the generator's own value: a={a}, b={b}, op={op}"
        )
        assert value < 1000, f"pool contains a combo >= 1000: a={a}, b={b}, op={op}, value={value}"


def test_factorial_notation_hard_answer_is_symbolic_in_n():
    n = sympy.symbols("n")
    for seed in SEEDS:
        q = make(FACT_TOPIC, seed=seed, difficulty="hard")
        expr = sympy.sympify(q.answer_text)
        assert expr.free_symbols == {n}, f"seed={seed}: hard answer not symbolic in n alone: {expr}"


def test_restricted_permutations_answers_are_integers_under_1000():
    for seed in SEEDS:
        for tier in TIERS:
            q = make(RESTR_TOPIC, seed=seed, difficulty=tier)
            assert q.answer_text.isdigit(), f"seed={seed}/{tier}: not a bare integer: {q.answer_text!r}"
            assert int(q.answer_text) < 1000, f"seed={seed}/{tier}: answer not under 4 digits: {q.answer_text}"


def test_restricted_permutations_hard_tier_never_reaches_an_undivided_7_factorial():
    """n is no longer pinned to 6 (`_HARD_COMBOS` also draws n=7), so the ORIGINAL invariant this
    test asserted -- `q.text.startswith("6 objects are arranged")` -- no longer holds and would incorrectly
    fail every n=7 draw. What that invariant was actually FOR (an undivided 7! = 5040 never reaching the
    learner as the ANSWER, and the answer staying under the shared 4-digit ceiling) still holds:
    `_HARD_COMBOS` is hand-enumerated so the largest achievable answer is 720 (n=7, r=2, together), never an
    undivided 5040. Re-asserted directly against every seed's rendered answer, which is the invariant that
    actually matters to a learner -- the exact wording of a fixed row-count was never the point."""
    for seed in SEEDS:
        q = make(RESTR_TOPIC, seed=seed, difficulty="hard")
        assert q.text.startswith(("6 objects are arranged", "7 objects are arranged")), (
            f"seed={seed}: hard tier drew an n outside {{6, 7}}: {q.text!r}"
        )
        assert int(q.answer_text) < 1000, f"seed={seed}: hard answer reached the 4-digit ceiling: {q.answer_text}"


# --- Restriction present in every question, every tier (ACMSM004's own bar) -


@pytest.mark.parametrize("tier", TIERS)
def test_restricted_permutations_every_question_states_a_restriction(tier):
    """The widened phrase list: easy/medium can now draw a block of 3 ("must all be next to each
    other" / "must NOT all be next to each other, in any order") as well as the original pair phrasing, and
    the hard tier's new "apart" constraint type reads "must NOT be next to" without the "each other"
    easy/medium's pair phrasing always carried."""
    for seed in SEEDS:
        q = make(RESTR_TOPIC, seed=seed, difficulty=tier)
        assert (
            "must be next to each other" in q.text
            or "must NOT be next to each other" in q.text
            or "must all be next to each other" in q.text
            or "must NOT all be next to each other" in q.text
            or "must be next to" in q.text
            or "must NOT be next to" in q.text
        ), f"seed={seed}/{tier}: no restriction stated: {q.text!r}"


# --- Distinct rendered questions -----------------------------------------


# --- Formatted alike -------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_restricted_permutations_answer_and_distractors_all_bare_integers(tier):
    for seed in SEEDS:
        q = make(RESTR_TOPIC, seed=seed, difficulty=tier)
        for opt in [q.answer_text] + q.distractors:
            assert opt.lstrip("-").isdigit(), f"seed={seed}/{tier}: option not a bare integer: {opt!r}"


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_factorial_notation_easy_medium_answer_and_distractors_all_bare_integers(tier):
    for seed in SEEDS:
        q = make(FACT_TOPIC, seed=seed, difficulty=tier)
        for opt in [q.answer_text] + q.distractors:
            assert opt.lstrip("-").isdigit(), f"seed={seed}/{tier}: option not a bare integer: {opt!r}"


class _FixedChoiceRng:
    """A minimal rng stub that hands back one fixed `(a, b, op)` combo from
    `_EASY_COMBO_POOL` for `rng.choice(...)`, and raises on any other rng
    method -- the easy branch only ever calls `.choice` once."""

    def __init__(self, combo):
        self._combo = combo

    def choice(self, seq):
        return self._combo

    def __getattr__(self, name):
        raise AttributeError(f"easy tier unexpectedly used rng.{name}")


def test_factorial_notation_easy_check_pairwise_distinct_never_fires_across_full_pool():
    """`check_pairwise_distinct` is called internally at the end of every
    branch; a collision anywhere in the easy tier's operand space would raise
    `VerificationError` here. Exercises the REAL generator function directly
    for every one of the enumerated (a, b, op) combinations -- not a 60-seed
    sample, which draws from a 106-entry pool and would miss most of it."""
    for combo in _EASY_COMBO_POOL:
        rng = _FixedChoiceRng(combo)
        result = p11c_factorial_notation(rng, tier="easy")
        assert result["answer_text"] not in result["distractors"], (
            f"combo={combo}: answer appears in its own distractors"
        )
        assert len(result["distractors"]) == len(set(result["distractors"])), f"combo={combo}: duplicate distractors"
        assert 1 <= len(result["distractors"]) <= 3, (
            f"combo={combo}: expected 1-3 distractors, got {result['distractors']}"
        )


def test_factorial_notation_hard_answer_and_distractors_all_symbolic():
    """Every hard-tier option (answer + distractors) contains 'n' -- no
    option is a bare number, so no option is identifiable by that shape."""
    for seed in SEEDS:
        q = make(FACT_TOPIC, seed=seed, difficulty="hard")
        for opt in [q.answer_text] + q.distractors:
            assert "n" in opt, f"seed={seed}: hard option has no 'n': {opt!r}"


# --- A correct learner writing a different but equivalent form --------


def test_factorial_notation_hard_grader_accepts_expanded_and_implicit_mult_forms():
    """A learner who expands the product, or who writes two adjacent
    bracketed factors with no operator between them (standard maths
    notation), must not be marked wrong -- checked through the REAL grader on
    a real rendered question, not `_parse_expression` in isolation."""
    n = sympy.symbols("n")
    for seed in SEEDS:
        q = make(FACT_TOPIC, seed=seed, difficulty="hard")
        answer_expr = sympy.sympify(q.answer_text)
        expanded = str(sympy.expand(answer_expr)).replace("**", "^")
        result = GRADE_EXPRESSION(q, expanded)
        assert result.correct, f"seed={seed}: expanded form {expanded!r} rejected for answer {q.answer_text!r}"
        if answer_expr.is_Mul and len(answer_expr.args) == 2:
            implicit = "".join(f"({f})" for f in answer_expr.args)
            result = GRADE_EXPRESSION(q, implicit)
            assert result.correct, f"seed={seed}: implicit-multiplication form {implicit!r} rejected"


# --- Independent-check tripwires --------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(5, 6, "should fail")


def test_check_pairwise_distinct_fires_on_a_constructed_collision():
    """Proven to actually fire, not just imported: two equal-value candidates
    (one of them algebraically, not just textually, equal -- "24" and
    "4*3*2*1*1", which both sympify to 24) must raise."""
    check_pairwise_distinct("24", "23", "n + 1")  # no collision: passes
    with pytest.raises(VerificationError):
        check_pairwise_distinct("24", "4*3*2*1*1", "n + 1")


# --- Skill loading -----------------------------------------------------


def test_factorial_notation_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == FACT_SKILL), None)
    assert skill is not None, f"{FACT_SKILL} not found"
    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert skill.prerequisites == []
    assert len(skill.sources) == 4
    for source in skill.sources[:3]:
        assert source.topic == FACT_TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    # the exam source is covered by this skill's tests/test_exd_x_* file
    assert skill.sources[3].difficulty == "exam"
    assert set(skill.sources[3].stages) == {"exam"}


def test_restricted_permutations_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == RESTR_SKILL), None)
    assert skill is not None, f"{RESTR_SKILL} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == []
    assert len(skill.sources) == 4
    for source in skill.sources[:3]:
        assert source.topic == RESTR_TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    # the exam source is covered by this skill's tests/test_exd_x_* file
    assert skill.sources[3].difficulty == "exam"
    assert set(skill.sources[3].stages) == {"exam"}


# Answer-concentration bound ----------------------
#
# senior-solve-problems-involving-permutations-and-restrictions-with-or is `answer: number` -- a
# typed field with no options to choose between. Measured over 500 seeds against the ORIGINAL
# generator (before this rework's fix): easy 3 distinct answers (240 35.4%, 12 32.4%, 48 32.2%),
# medium 3 distinct answers (72 37.8%, 480 32.0%, 12 30.2%), hard **1** distinct answer (120 on 100%
# of draws) -- a learner can type "120" on every hard question without reading it. Swept at 500
# seeds, not sampled, because the property under test IS the draw distribution itself; a handful of
# seeds cannot show concentration.
_CONCENTRATION_SEEDS = range(500)
_CONCENTRATION_MAX_SHARE = 0.25


@pytest.mark.parametrize("tier", TIERS)
def test_restricted_permutations_answer_concentration_bounded(tier):
    """No single answer value may cover more than 25% of a tier's 500-seed draws, unless the narrowness is
    genuinely the mathematics rather than the pool. A 4-option unit-choice mcq being narrow is the
    mathematics; this skill's typed hard tier having exactly one achievable value under its own generator's
    pool was not -- see the module docstring's redesigned combo pool (object count, repeat count and
    constraint type all vary) for the fix."""
    from collections import Counter

    counts = Counter()
    for seed in _CONCENTRATION_SEEDS:
        q = make(RESTR_TOPIC, seed=seed, difficulty=tier)
        counts[q.answer_text] += 1
    total = sum(counts.values())
    assert total == len(_CONCENTRATION_SEEDS)
    worst_value, worst_count = counts.most_common(1)[0]
    share = worst_count / total
    assert share <= _CONCENTRATION_MAX_SHARE, (
        f"{tier}: answer {worst_value!r} covers {worst_count}/{total} "
        f"({share:.1%}) of draws over {total} seeds -- exceeds the 25% "
        f"concentration bound (distinct answers seen: {dict(counts)})"
    )
