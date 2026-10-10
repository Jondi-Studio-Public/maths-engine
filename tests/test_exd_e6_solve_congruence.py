"""A NEW `kind="solve_congruence"` branch on the `modular_multiplicative_inverse` topic
(`mathgen/generators/university_algebra.py`).

`kind="solve_congruence"` generalises the fixed "x ≡ 1 (mod m)" target to a general `b` (drawn independently
of `value`/`modulus`, from `range(1, modulus)` -- the generator excludes `b=0`, since the answer would be invariant
to the "find the inverse" step the escalation tests, and redraws away `value=1`, since 1's own inverse is
trivially 1): the student must find the inverse of the coefficient AND THEN multiply by `b` -- a genuine extra
reasoning step, verified independently against BOTH a direct-congruence check and a from-scratch re-derivation
of the inverse.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "modular_multiplicative_inverse"

# Matches the skill's hard source exactly: `{kind: solve_congruence,
# modulus_hi: 31}` (modulus_hi=31 also comes from the "hard" named preset via
# the merge; stated explicitly here too, matching the YAML).
_HARD_PARAMS = dict(kind="solve_congruence", modulus_hi=31)


def _make_hard(seed, **overrides):
    params = {**_HARD_PARAMS, **overrides}
    return make(TOPIC, seed=seed, difficulty="hard", **params)


# ---------------------------------------------------------------------------
# Text quality, prompt variety, determinism
# ---------------------------------------------------------------------------


def test_solve_congruence_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_hard(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_solve_congruence_distinct_raw_prompts_over_60_seeds():
    """Matches the library's own convention
    (at least `_MIN_DISTINCT = 10` distinct raw prompts over 60 seeds)."""
    texts = {_make_hard(seed).text for seed in range(60)}
    assert len(texts) >= 10, texts


def test_solve_congruence_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_hard(seed).to_json()
        second = _make_hard(seed).to_json()
        assert first == second


def test_solve_congruence_hard_adds_shape_over_easy_and_medium():
    """The new `kind="solve_congruence"` hard tier's shape is not in the
    union of the UNCHANGED easy/medium tiers' shapes (both mirror the skill
    YAML's own `kind`-default sources, unaffected by this change): a genuine
    escalation in the CONGRUENCE TARGET named (a general `b`, not the fixed
    "≡ 1"), not merely a bigger `modulus_hi`."""
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy")) for s in range(60)}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium")) for s in range(60)}
    hard_shapes = {prompt_shape(_make_hard(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


# ---------------------------------------------------------------------------
# Independent oracle: re-derive x from the rendered prompt text ALONE, using
# only Python's builtin `pow(value, -1, modulus)` -- never importing
# `mathgen.generators.university_algebra`, so this genuinely does not share a
# code path with the generator's own `check()` (which itself already
# cross-checks a direct-congruence route and a from-scratch inverse
# recomputation).
# ---------------------------------------------------------------------------

_CONGRUENCE_RE = re.compile(r"^Find x in \{0,\.\.\.,(\d+)\} such that (\d*)x ≡ (\d+) \(mod (\d+)\)\.$")


def _independent_solve_congruence(question_text: str) -> int:
    m = _CONGRUENCE_RE.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    _upper, coefficient_str, b_str, modulus_str = m.groups()
    a = int(coefficient_str) if coefficient_str else 1
    b, modulus = int(b_str), int(modulus_str)
    inverse = pow(a, -1, modulus)
    return (inverse * b) % modulus


def test_solve_congruence_answer_matches_independent_oracle_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_hard(seed)
        oracle_answer = _independent_solve_congruence(q.text)
        if oracle_answer != q.answer:
            mismatches.append((seed, q.text, oracle_answer, q.answer))
    assert mismatches == [], mismatches


def test_solve_congruence_b_zero_is_never_drawn_over_600_seeds():
    """b=0 is EXCLUDED. `b` is now drawn from `range(1, modulus)`, so b=0 must never occur."""
    zero_b_seeds = [seed for seed in range(600) if int(_CONGRUENCE_RE.match(_make_hard(seed).text).group(3)) == 0]
    assert zero_b_seeds == [], zero_b_seeds


def test_solve_congruence_value_one_is_never_drawn_over_600_seeds():
    """value=1 is EXCLUDED from the hard-tier `solve_congruence` draw. The rendered prompt's
    coefficient is a bare "x" (empty capture group) exactly when value=1."""
    value_one_seeds = [seed for seed in range(600) if _CONGRUENCE_RE.match(_make_hard(seed).text).group(2) == ""]
    assert value_one_seeds == [], value_one_seeds


import builtins  # noqa: E402

_original_pow = builtins.pow


def test_solve_congruence_corrupted_check_raises_verification_error(monkeypatch):
    """Corrupts only the SECOND `pow(value, -1, modulus)` call -- the branch's
    own `inverse_of_value` recomputation -- while leaving the FIRST (the
    shared block's `inverse`, computed before the `kind` branch is even
    reached) correct. A patch that corrupted both identically would make
    `inverse` wrong first (firing the SHARED block's own pre-existing check,
    which proves nothing new about this branch) or, since `x` and
    `x_expected` are then both built from the same wrong inverse, would let
    `x == x_expected` still pass (proving nothing at all) -- seed=1 is used
    here specifically because its `b != 0` (b=0 is now excluded
    entirely, so every seed's `b` makes the corruption visible in
    `x_expected`; the old base-branch concern that a `b=0` seed would
    silently absorb any inverse corruption no longer applies)."""
    calls = {"count": 0}

    def _wrong_pow(base, exponent, modulus=None):
        result = _original_pow(base, exponent, modulus)
        if exponent == -1 and modulus is not None:
            calls["count"] += 1
            if calls["count"] == 2:
                return (result + 1) % modulus
        return result

    # `pow` is a builtin, not a module attribute of `_ua`, so it is patched
    # at the builtins level -- the module's own `pow(value, -1, modulus)`
    # calls resolve there. Restored automatically by `monkeypatch` at test
    # teardown, so this cannot leak into any other test.
    monkeypatch.setattr(builtins, "pow", _wrong_pow)
    with pytest.raises(VerificationError, match="from-scratch"):
        _make_hard(1)
    assert calls["count"] == 2, calls


def test_solve_congruence_corrupted_check_is_not_a_false_positive_when_uncorrupted():
    _make_hard(0)  # no exception


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer.
# ---------------------------------------------------------------------------


def test_solve_congruence_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_hard(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_solve_congruence_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_hard(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


# ---------------------------------------------------------------------------
# The real grader (mathgen.grading.GRADERS["number"]) -- skill's `answer: number`
# ---------------------------------------------------------------------------

from mathgen.grading.typed import GRADERS  # noqa: E402

_NUMBER_GRADER = GRADERS["number"]


def test_solve_congruence_grader_accepts_canonical_and_rejects_every_distractor_over_200_seeds():
    bad = []
    for seed in range(200):
        q = _make_hard(seed)
        canonical_result = _NUMBER_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _NUMBER_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_solve_congruence_grader_rejects_deliberately_wrong_misconception_values():
    """The three misconception values, computed by hand against a fixed seed
    and checked they are actually wrong before asserting the grader rejects
    them (never a vacuous check).

    seed=1, difficulty=hard (the draw changed from `b=rng.randint(0,
    modulus-1)` to `b=rng.randint(1, modulus-1)`, so seed=1's `b` shifted
    from the pre-fix run's 19 to 20; `value`/`modulus` are unaffected since
    this seed's initial draw never hit `value=1`): rendered question is
    "Find x in {0,...,30} such that 17x ≡ 20 (mod 31)." -- value=17,
    modulus=31, b=20. Inverse of 17 mod 31: 17*11 = 187 = 6*31 + 1, so
    inverse = 11. x = (11*20) mod 31 = 220 mod 31 = 3 (31*7=217, 220-217=3).
    forgot_multiply = inverse itself = 11 (!= 3).
    multiplied_by_coefficient = (17*20) mod 31 = 340 mod 31 = 30
    (31*10=310, 340-310=30) (!= 3).
    bad_residue_for_b = (11 * ((20-1) mod 31)) mod 31 = (11*19) mod 31 =
    209 mod 31 = 23 (31*6=186, 209-186=23) (!= 3).
    """
    q = _make_hard(1)
    m = _CONGRUENCE_RE.match(q.text)
    coefficient_str, b_str, modulus_str = m.group(2), m.group(3), m.group(4)
    value = int(coefficient_str) if coefficient_str else 1
    b, modulus = int(b_str), int(modulus_str)
    assert (value, modulus, b) == (17, 31, 20), (value, modulus, b, q.text)
    answer = q.answer
    assert answer == 3, answer

    inverse_of_value = pow(value, -1, modulus)
    assert inverse_of_value == 11
    forgot_multiply = inverse_of_value
    multiplied_by_coefficient = (value * b) % modulus
    bad_residue_for_b = (inverse_of_value * ((b - 1) % modulus)) % modulus
    wrong_candidates = [forgot_multiply, multiplied_by_coefficient, bad_residue_for_b]
    assert forgot_multiply == 11
    assert multiplied_by_coefficient == 30
    assert bad_residue_for_b == 23
    for wrong in wrong_candidates:
        assert wrong != answer, (wrong, answer)
        result = _NUMBER_GRADER(q, str(wrong))
        assert not result.correct, (wrong, q.answer_text)


# The render baseline replay is the authoritative proof (2 stored rows, both kind-default calls);
# this is a lightweight structural regression check in addition.

_INVERSE_RE = re.compile(
    r"^Find the multiplicative inverse of (\d+) modulo (\d+); "
    r"that is, find x in \{1,\.\.\.,\d+\} with \d*x ≡ 1 \(mod \d+\)\.$"
)


def test_original_kind_none_default_shape_unaffected_over_300_seeds():
    for seed in range(300):
        for difficulty in ("easy", "medium", "hard"):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert _INVERSE_RE.match(q.text), (seed, difficulty, q.text)


def test_original_kind_none_explicit_matches_default_over_60_seeds():
    """`kind=None` passed explicitly must be byte-identical to omitting it
    (the parameter's default), proving the new knob truly defaults to the old
    behaviour rather than merely happening to look the same."""
    for seed in range(60):
        implicit = make(TOPIC, seed=seed, difficulty="hard").to_json()
        explicit = make(TOPIC, seed=seed, difficulty="hard", kind=None).to_json()
        assert implicit == explicit


# ---------------------------------------------------------------------------
# Skill loads via mathgen.skills.load_skills
# ---------------------------------------------------------------------------


def test_skill_loads_via_api_skills():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "modular-multiplicative-inverse" in ids
    skill = ids["modular-multiplicative-inverse"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kind") == "solve_congruence"
