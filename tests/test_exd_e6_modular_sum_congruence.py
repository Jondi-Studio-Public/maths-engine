"""A NEW `kind="sum_congruence"` branch on the `modular_arithmetic` topic.

`kind="sum_congruence"` draws TWO integers `a`/`b` -- each from roughly HALF of `magnitude_hi`, so their sum
lands in a comparable magnitude band to the single-value case's own `[-magnitude_hi, magnitude_hi]` draw (a
documented choice, see the generator's own docstring) -- and asks for the least non-negative residue
of `a + b`, requiring the student to add first and only then reduce.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "modular_arithmetic"

# Matches the skill's hard source exactly: `{kind: sum_congruence}`
# (modulus_hi=19, magnitude_hi=250 come from the "hard" named preset via the
# merge, unchanged).
_HARD_PARAMS = dict(kind="sum_congruence")


def _make_hard(seed, **overrides):
    params = {**_HARD_PARAMS, **overrides}
    return make(TOPIC, seed=seed, difficulty="hard", **params)


# ---------------------------------------------------------------------------
# Text quality, prompt variety, determinism
# ---------------------------------------------------------------------------


def test_sum_congruence_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_hard(seed)
        defects = find_text_defects(q, include_working=True)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_sum_congruence_distinct_raw_prompts_over_60_seeds():
    """Matches the library's own convention
    (at least `_MIN_DISTINCT = 10` distinct raw prompts over 60 seeds). `prompt_shape()` masks every digit run, and this
    generator's whole prompt is a single fixed sentence template around
    `a`/`b`/the modulus, so its own prompt_shape is a single constant string
    by design -- the escalation is that the hard tier's shape (two addends)
    differs from the easy/medium tiers' shape (one value), proved below, not
    that the hard tier is internally varied."""
    texts = {_make_hard(seed).text for seed in range(60)}
    assert len(texts) >= 10, texts


def test_sum_congruence_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_hard(seed).to_json()
        second = _make_hard(seed).to_json()
        assert first == second


def test_sum_congruence_hard_adds_shape_over_easy_and_medium():
    """The new `kind="sum_congruence"` hard tier's shape is not in the union
    of the UNCHANGED easy/medium tiers' shapes (both mirror the skill YAML's
    own `kind`-default sources, unaffected by this change): a genuine
    escalation in the STRUCTURE of the prompt (two addends that must first be
    summed), not merely a bigger `magnitude_hi`."""
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy")) for s in range(60)}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium")) for s in range(60)}
    hard_shapes = {prompt_shape(_make_hard(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


# ---------------------------------------------------------------------------
# Independent oracle: re-derive the residue from the rendered prompt text
# ALONE, using only Python's stdlib `%` operator -- never importing
# `mathgen.generators.university_foundations`, so this genuinely does not
# share a code path with the generator's own `check()` (which itself already
# cross-checks `math.floor` division AND brute-force enumeration).
# ---------------------------------------------------------------------------

_SUM_CONGRUENCE_RE = re.compile(
    r"^Find the least non-negative residue r such that "
    r"\((-?\d+) \+ (\(-?\d+\)|-?\d+)\) . r \(mod (\d+)\)\.$"
)


def _independent_residue(question_text: str) -> tuple[int, int, int, int]:
    m = _SUM_CONGRUENCE_RE.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    a = int(m.group(1))
    b = int(m.group(2).strip("()"))
    modulus = int(m.group(3))
    return a, b, modulus, (a + b) % modulus


def test_sum_congruence_answer_matches_independent_oracle_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_hard(seed)
        _a, _b, _modulus, oracle_answer = _independent_residue(q.text)
        if oracle_answer != q.answer:
            mismatches.append((seed, q.text, oracle_answer, q.answer))
    assert mismatches == [], mismatches


import mathgen.generators.university_foundations as _uf  # noqa: E402


def test_sum_congruence_corrupted_check_raises_verification_error(monkeypatch):
    original_floor = _uf.math.floor

    def _wrong_floor(*args, **kwargs):
        return original_floor(*args, **kwargs) + 1

    monkeypatch.setattr(_uf.math, "floor", _wrong_floor)
    with pytest.raises(VerificationError):
        _make_hard(0)


def test_sum_congruence_corrupted_check_is_not_a_false_positive_when_uncorrupted():
    _make_hard(0)  # no exception


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer.
# ---------------------------------------------------------------------------


def test_sum_congruence_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_hard(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_sum_congruence_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_hard(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


# ---------------------------------------------------------------------------
# The real grader (mathgen.grading.GRADERS["number"]) -- skill's `answer: number`
# ---------------------------------------------------------------------------

from mathgen.grading.typed import GRADERS  # noqa: E402

_NUMBER_GRADER = GRADERS["number"]


def test_sum_congruence_grader_accepts_canonical_and_rejects_every_distractor_over_200_seeds():
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


def test_sum_congruence_grader_rejects_deliberately_wrong_misconception_values():
    """The three misconception values, computed by hand against a fixed seed
    and checked they are actually wrong before asserting the grader rejects
    them (never a vacuous check).

    seed=5, difficulty=hard, kind=sum_congruence: (-89 + (-37)) mod 12 -> 6. `reduced_separately` = (-89 % 12)
    + (-37 % 12) = 7 + 11 = 18 (reduced each addend separately, then added the residues without re-reducing
    the sum); `one_unreduced` = (-89 % 12) + (-37) = 7 + (-37) = -30 (reduced only the first addend, left the
    second unreduced, then added); `forgot_b` = -89 % 12 = 7 (reduced only the first addend and never added
    the second at all). All three are the generator's own offered distractors at this seed, in this order --
    proving the misconceptions are genuine intermediate values, not arbitrary numbers.
    """
    q = _make_hard(5)
    answer = q.answer
    assert answer == 6
    a, b, modulus = -89, -37, 12
    reduced_separately = (a % modulus) + (b % modulus)
    one_unreduced = (a % modulus) + b
    forgot_b = a % modulus
    assert reduced_separately == 18
    assert one_unreduced == -30
    assert forgot_b == 7
    wrong_candidates = [reduced_separately, one_unreduced, forgot_b]
    assert set(str(c) for c in wrong_candidates) == set(q.distractors)
    for wrong in wrong_candidates:
        assert wrong != answer, (wrong, answer)
        result = _NUMBER_GRADER(q, str(wrong))
        assert not result.correct, (wrong, q.answer_text)


def test_sum_congruence_one_unreduced_misconception_dropped_when_it_equals_answer():
    """seed=7: a=92, b=-4, modulus=8. total=88, answer=0.
    `one_unreduced` = (92 % 8) + (-4) = 4 + (-4) = 0, which IS the answer --
    `pick_distractors` must drop it rather than offer a second correct
    option, exactly the "check-verify distinct before offering it" rule
    for these candidates."""
    q = _make_hard(7)
    assert q.answer == 0
    a, modulus = 92, 8
    one_unreduced = (a % modulus) + (-4)
    assert one_unreduced == q.answer == 0
    assert "0" not in q.distractors
    # The two surviving misconceptions (reduced_separately=8, forgot_b=4) and
    # the raw sum (88) are still offered.
    assert set(q.distractors) == {"8", "4", "88"}


# The render baseline replay is the authoritative proof (2 stored rows, both kind-default calls);
# this is a lightweight structural regression check in addition.

_SINGLE_VALUE_RE = re.compile(r"^Find the least non-negative residue r such that -?\d+ . r \(mod \d+\)\.$")


def test_original_kind_none_default_shape_unaffected_over_300_seeds():
    for seed in range(300):
        for difficulty in ("easy", "medium", "hard"):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert _SINGLE_VALUE_RE.match(q.text), (seed, difficulty, q.text)


def test_original_kind_none_explicit_matches_default_over_60_seeds():
    """`kind=None` passed explicitly must be byte-identical to omitting it
    (the parameter's default), proving the new knob truly defaults to the
    old behaviour rather than merely happening to look the same."""
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
    assert "modular-arithmetic-congruence" in ids
    skill = ids["modular-arithmetic-congruence"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kind") == "sum_congruence"
