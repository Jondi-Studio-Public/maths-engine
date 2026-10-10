"""Focused contract tests for: ucv1-number-theory-computation (answer: number)."""

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make

NTC_ID = "ucv1-number-theory-computation"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
CONCENTRATION_SEEDS = range(200)
TIERS = ["easy", "medium", "hard", "exam"]

_NUMBER_RE = re.compile(r"-?\d+")
_M_RE = re.compile(r"m = (\d+)")
_E_RE = re.compile(r"e = (\d+)(?!\^)")
# "n = " is ambiguous in some templates ("n = {p} × {q} = {n}" has TWO
# "n = " occurrences with the wrong one first) -- always derive n from the
# stated factorisation "{p}(^2)? × {q}" instead, never from a bare
# "n = (\d+)" search.
_FACTORS_RE = re.compile(r"(\d+)(\^2)?\s*×\s*(\d+)")


def _extract_n(text: str) -> int:
    m = _FACTORS_RE.search(text)
    assert m, text
    p, squared, q = m.group(1), m.group(2), m.group(3)
    p, q = int(p), int(q)
    return (p * p * q) if squared else (p * q)


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


NTC_SKILL, NTC_SOURCES = _skill_sources(NTC_ID)


def _make(difficulty, seed):
    src = NTC_SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        wrong = str(int(q.answer) + 100000)
        assert not grade_number(q, wrong).correct, f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_an_exact_non_negative_integer_in_range(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert _NUMBER_RE.fullmatch(q.answer_text), q.answer_text
        assert isinstance(q.answer, int)
        assert q.answer >= 0
        n = _extract_n(q.text)
        assert q.answer < n


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_distractor_formatted_like_the_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        for d in q.distractors:
            assert _NUMBER_RE.fullmatch(d), d


@pytest.mark.parametrize("difficulty", TIERS)
def test_200_seed_answer_concentration_under_25_percent(difficulty):
    counts = Counter(_make(difficulty, seed).answer_text for seed in CONCENTRATION_SEEDS)
    share = max(counts.values()) / len(CONCENTRATION_SEEDS)
    assert share <= 0.25, f"{difficulty}: top answer share {share:.3f} over 200 seeds"


@pytest.mark.parametrize("difficulty", TIERS)
def test_pow_builtin_agrees_with_stated_answer_on_original_e_and_n(difficulty):
    """The independent oracle-grade check: pow(m, e, n)
    computed directly on the ORIGINAL (unreduced) e and n must equal the
    stated answer, for every seed, unconditionally."""
    for seed in SEEDS:
        q = _make(difficulty, seed)
        n = _extract_n(q.text)
        m = int(_M_RE.search(q.text).group(1))
        e_match = _E_RE.search(q.text)
        if e_match is None:
            # medium: e is given as an expression "2^a + b"; evaluate it.
            expr = re.search(r"2\^(\d+) \+ (\d+)", q.text)
            e = 2 ** int(expr.group(1)) + int(expr.group(2))
        else:
            e = int(e_match.group(1))
        assert pow(m, e, n) == int(q.answer_text), (
            f"seed {seed} {difficulty}: pow(m, e, n) disagrees with stated answer"
        )


def test_hard_uses_a_repeated_prime_factor():
    for seed in SEEDS:
        q = _make("hard", seed)
        n = _extract_n(q.text)
        p_match = re.search(r"(\d+)\^2", q.text)
        assert p_match, q.text
        p = int(p_match.group(1))
        assert n % (p * p) == 0


def test_exam_base_is_not_coprime_to_modulus():
    from math import gcd

    for seed in SEEDS:
        q = _make("exam", seed)
        n = _extract_n(q.text)
        m = int(_M_RE.search(q.text).group(1))
        assert gcd(m, n) != 1, f"seed {seed}: exam base m={m} must not be coprime to n={n}"


def test_easy_and_medium_base_is_coprime_to_modulus():
    from math import gcd

    for difficulty in ("easy", "medium", "hard"):
        for seed in SEEDS:
            q = _make(difficulty, seed)
            n = _extract_n(q.text)
            m = int(_M_RE.search(q.text).group(1))
            assert gcd(m, n) == 1, f"{difficulty} seed {seed}: base m={m} must be coprime to n={n}"


def test_never_a_bare_rsa_style_prompt():
    """This skill's own text never phrases the question as RSA
    encrypt/decrypt (the neighbour skill's exact wording)."""
    for difficulty in TIERS:
        for seed in SEEDS:
            q = _make(difficulty, seed)
            lowered = q.text.lower()
            assert "rsa" not in lowered
            assert "plaintext" not in lowered
            assert "ciphertext" not in lowered


def test_skill_loads_through_api_with_correct_properties():
    assert NTC_SKILL.domain == "number-theory"
    assert NTC_SKILL.status == "active"
    assert NTC_SKILL.placement_eligible is False
    assert NTC_SKILL.answer == "number"
    assert NTC_SKILL.prerequisites == ["modular-arithmetic-congruence"]
    assert {src.difficulty for src in NTC_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = NTC_SOURCES["exam"]
    assert exam_src.stages == ["exam"]


def test_independent_check_fires_on_corrupted_agreement():
    """Exercise the generator's own check() calls directly: a candidate
    answer that disagrees with pow(m, e, n) must raise."""
    from mathgen.core import VerificationError, check

    m, e, n = 5, 100, 33
    wrong_answer = pow(m, e, n) + 1
    with pytest.raises(VerificationError):
        check(wrong_answer == pow(m, e, n), "disagrees with pow(m, e, n)")
