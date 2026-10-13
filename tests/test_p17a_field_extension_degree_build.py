"""See the generator module docstring for the per-tier ladder and the answer-space design."""

from __future__ import annotations

from collections import Counter

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check, check_pairwise_distinct
import mathgen.generators.p17a_field_extension_degree_build  # noqa: F401 -- registers the topic
from mathgen.generators.p17a_field_extension_degree_build import (
    _eisenstein_ok,
    _has_rational_root,
    _is_perfect_square,
    _poly_expr_str,
    _poly_mul,
    _quadratic_irreducible,
)

TOPIC = "p17a_field_extension_degree"
SKILL_ID = "ucv1-field-extension-degree"
SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")
TIER_VALUE_SETS = {
    "easy": {"1", "2", "3"},
    "medium": {"2", "3", "4"},
    "hard": {"1", "2", "4"},
    "exam": {"2", "3", "4"},
}


def _skill_sources():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return skill, {src.difficulty: src for src in skill.sources}


SKILL, SOURCES = _skill_sources()


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# --- Determinism -------------------------------------------------------


# --- Distractor hygiene, exactly two distractors, values within the tier's set ---


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_two_and_within_tier_value_set(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert len(q.distractors) == 2
        assert len(set(q.distractors)) == 2
        all_values = set(q.distractors) | {q.answer_text}
        assert all_values == TIER_VALUE_SETS[tier], (tier, seed, all_values)


# --- Real grader accepts canonical, rejects every distractor and an out-of-range value ---


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        result = grade_number(q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = grade_number(q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"
        result = grade_number(q, "9")
        assert not result.correct, f"{tier} seed={seed}: out-of-range value 9 accepted"


# --- Answer kind: integer 1-4, formatted alike -----------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_1_to_4_and_formatted_alike(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert q.answer in (1, 2, 3, 4)
        assert isinstance(q.answer, int)
        assert q.answer_text in ("1", "2", "3", "4")
        for d in q.distractors:
            assert d in ("1", "2", "3", "4")


# --- Skill loads with the expected shape ------------------------------------


def test_skill_loads_with_expected_shape():
    assert SKILL.answer == "number"
    assert SKILL.placement_eligible is False
    assert SKILL.domain == "advanced-algebra"
    assert SKILL.prerequisites == ["factorise-quadratic", "simplify-surds"]


# Root naming: never "rational"/"irrational", every ---- --- polynomial prompt names alpha by
# real-root order -----------------------


@pytest.mark.parametrize("tier", TIERS)
def test_no_rational_or_irrational_word_in_polynomial_prompts(tier):
    for seed in range(200):
        q = _make(tier, seed)
        text = q.text
        if "sqrt(" in text:
            continue  # compound-field prompts state alpha explicitly, not via a polynomial root
        assert "rational" not in text and "irrational" not in text, (tier, seed, text)


@pytest.mark.parametrize("tier", TIERS)
def test_every_polynomial_prompt_names_alpha_by_real_root_order(tier):
    for seed in range(200):
        q = _make(tier, seed)
        text = q.text
        if "sqrt(" in text:
            continue
        assert "largest real root" in text or "smallest real root" in text, (tier, seed, text)
        assert "a root of p(x) = 0" not in text, (tier, seed, text)


# No naming phrase determines the key -------------------

_DIGIT_RE = __import__("re").compile(r"-?\d+")
_P_EXPR_MASK_RE = __import__("re").compile(r"p\(x\) = (?!0)[^,.]+")


def _masked_phrase(text: str) -> str:
    """The prompt with the whole polynomial (not just its digits) blanked out, then every remaining digit (a/b in
    the compound branches) masked too, so two draws that differ only ("the prompt with digits and the
    polynomial masked")."""
    masked = _P_EXPR_MASK_RE.sub("p(x) = P", text)
    return _DIGIT_RE.sub("#", masked)


@pytest.mark.parametrize("tier", TIERS)
def test_no_naming_phrase_determines_the_key(tier):
    from collections import defaultdict

    phrase_keys = defaultdict(set)
    for seed in range(200):
        q = _make(tier, seed)
        phrase_keys[_masked_phrase(q.text)].add(q.answer_text)
    assert phrase_keys, tier
    for masked, keys in phrase_keys.items():
        assert len(keys) >= 2, (tier, masked, keys)


# --- At least 3 distinct prompt shapes / >= 20 distinct questions per tier --


# --- Answer concentration at 200 seeds per tier, none above 50% ------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_concentration_at_200_seeds(tier):
    counts = Counter(_make(tier, seed).answer_text for seed in range(200))
    assert len(counts) >= 3, (tier, counts)
    for value, n in counts.items():
        assert n / 200 <= 0.50, (tier, value, n, counts)


# --- Independent in-generator check tripwires --------------------------------


def test_independent_check_fires_on_corrupted_value():
    with pytest.raises(VerificationError):
        check(False, "deliberately corrupted")
    with pytest.raises(VerificationError):
        check_pairwise_distinct(3, 3, msg="should fail: identical values")


# --- Arithmetic helpers behave as their names promise -----------------------


def test_is_perfect_square():
    for n in (0, 1, 4, 9, 16, 25, 100, 121):
        assert _is_perfect_square(n)
    for n in (2, 3, 5, 6, 7, 8, 10, 24, 26, 99, 122):
        assert not _is_perfect_square(n)
    assert not _is_perfect_square(-4)


def test_quadratic_irreducible_matches_discriminant():
    assert _quadratic_irreducible(0, -2)  # D = 8, not a perfect square
    assert not _quadratic_irreducible(-3, 2)  # D = 1, a perfect square (roots 1, 2)
    assert not _quadratic_irreducible(0, 0)  # D = 0, not > 0 -> repeated root, excluded


def test_has_rational_root():
    assert _has_rational_root([1, -3, 2])  # (x-1)(x-2)
    assert not _has_rational_root([1, 0, 0, 2])  # x^3 + 2, irreducible over Q (no rational root)


def test_eisenstein_ok():
    assert _eisenstein_ok([1, 0, 6, 0, 3], 3)  # x^4 + 6x^2 + 3, Eisenstein at 3
    assert not _eisenstein_ok([1, 0, 6, 0, 9], 3)  # constant 9 = 3^2, fails


def test_poly_mul_and_expr_str():
    # (x - 2)(x^2 + 3) = x^3 - 2x^2 + 3x - 6
    coeffs = _poly_mul([1, -2], [1, 0, 3])
    assert coeffs == [1, -2, 3, -6]
    assert _poly_expr_str(coeffs) == "x^3 - 2x^2 + 3x - 6"
    assert "**" not in _poly_expr_str(coeffs)
    assert "*" not in _poly_expr_str(coeffs)


# --- Speed: each tier's construction is cheap (retry loops bounded) ---------


def test_make_is_fast():
    import time

    start = time.perf_counter()
    for tier in TIERS:
        for seed in range(60):
            make(TOPIC, seed=seed, difficulty=tier)
    elapsed_ms = (time.perf_counter() - start) * 1000 / (4 * 60)
    assert elapsed_ms < 15, f"median make() call took {elapsed_ms:.2f}ms, over the 15ms budget"
