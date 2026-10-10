"""A range-only fix for `f10-multiplication-division-inverse-unknown`'s hard tier.

Before this change, the hard preset passed the SAME four-shape `shape_choices` tuple medium does, and the
generator's tuple-identity range dispatch treated that combination as medium -- so the hard tier drew from the
medium number range, never the `_HARD_A_RANGE`/`_HARD_B_RANGE` the module's own docstring claimed it did. The
fix adds a fifth shape, `"three_factor_product"` (`a x b x ? = c`), which requires combining the two KNOWN
factors before inverting against `c` -- a genuine second reasoning step none of the four existing
single-inverse-lookup shapes need -- and replaces the tuple-identity dispatch with a per-shape-set range
lookup (`_ranges_for_shapes`), additive only: the two pre-existing combinations still map to the exact ranges
they always did.

The independent oracle below (`_oracle_three_factor_product`) parses `question.text` alone and re-derives the
missing factor from scratch with plain integer arithmetic -- it imports nothing from
`mathgen.generators.p3a_number_equations_build`, so a slip baked into the generator's own `check` call could
not also be baked into the check that is supposed to catch it. `test_oracle_catches_a_corrupted_answer` proves
the oracle is not vacuously true.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import prompt_shape

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "multiplication_division_inverse_unknown"
SKILL_ID = "f10-multiplication-division-inverse-unknown"

# Force every seed onto the new shape, for tests that need to exercise it
# exhaustively rather than wait for `rng.choice` to land on it.
TFP_ONLY = ("three_factor_product",)
# The skill's own updated hard source: the four pre-existing shapes plus the
# new one, matching the YAML exactly.
HARD_SOURCE_SHAPES = (
    "a_times_missing_equals_c",
    "missing_times_b_equals_c",
    "c_div_a_equals_missing",
    "c_div_missing_equals_a",
    "three_factor_product",
)
SEEDS = range(60)


def _mk_tfp(seed):
    return make(TOPIC, seed=seed, difficulty="hard", shape_choices=TFP_ONLY)


# ---------------------------------------------------------------------------
# Independent oracle: re-derive the missing factor from the rendered prompt
# alone, never touching the generator module.

_TFP_RE = re.compile(r"Solve: (\d+) × (\d+) × \? = (\d+)\. Use the inverse multiplication-division relationship\.")


def _oracle_three_factor_product(text: str) -> str:
    match = _TFP_RE.search(text)
    assert match, f"could not parse three_factor_product prompt: {text!r}"
    a, b, c = (int(v) for v in match.groups())
    assert c % (a * b) == 0, f"c={c} is not divisible by a*b={a * b}: {text!r}"
    return str(c // (a * b))


def test_oracle_agrees_with_generator_over_60_seeds():
    checked = 0
    for seed in SEEDS:
        q = _mk_tfp(seed)
        expected = _oracle_three_factor_product(q.text)
        assert expected == q.answer_text, (
            f"seed={seed}: oracle recomputed {expected} from the prompt, "
            f"generator answer_text is {q.answer_text}\n{q.text}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_catches_a_corrupted_answer():
    """Proves the oracle is not vacuously true: pair a real prompt with a
    deliberately wrong `answer_text` and confirm the oracle disagrees."""
    q = _mk_tfp(0)
    corrupted = str(int(q.answer_text) + 7)
    assert _oracle_three_factor_product(q.text) != corrupted


# ---------------------------------------------------------------------------
# Determinism.


def test_three_factor_product_deterministic_across_seeds():
    for seed in SEEDS:
        q1 = _mk_tfp(seed)
        q2 = _mk_tfp(seed)
        assert q1.to_json() == q2.to_json()


# ---------------------------------------------------------------------------
# Distractor hygiene -- distinct, non-empty, never equal the answer. Proved
# by construction in the generator's own docstring/comments (single-factor-
# divide == b*missing with b >= 11; multiply-all-three == (a*b)**2 * missing;
# adjacent == missing +/- 1); swept here to confirm the proof empirically.


def test_three_factor_product_exactly_three_distinct_distractors():
    for seed in SEEDS:
        q = _mk_tfp(seed)
        assert len(q.distractors) == 3, (seed, q.text, q.distractors)
        assert len(set(q.distractors)) == 3, (seed, q.distractors)
        assert q.answer_text not in q.distractors, (seed, q.text, q.distractors)
        assert all(d for d in q.distractors), (seed, q.distractors)


# ---------------------------------------------------------------------------
# Real grader: accepts canonical, rejects every distractor, rejects a
# deliberately-wrong value.


def test_real_grader_accepts_canonical_and_rejects_every_distractor():
    for seed in SEEDS:
        q = _mk_tfp(seed)
        accepted = grade_number(q, q.answer_text)
        assert accepted.correct, (seed, q.text, q.answer_text)
        for d in q.distractors:
            rejected = grade_number(q, d)
            assert not rejected.correct, (seed, q.text, "distractor wrongly accepted:", d)


def test_a_deliberately_wrong_answer_is_rejected():
    for seed in SEEDS:
        q = _mk_tfp(seed)
        wrong = str(int(q.answer_text) + 5)
        assert not grade_number(q, wrong).correct, (seed, q.text, q.answer_text, wrong)


# The independent, in-generator check firing on a corrupted value -- imports the generator module
# directly, ONLY here, to call its private verification helper with a deliberately wrong `missing`.

import mathgen.generators.p3a_number_equations_build as _p3a  # noqa: E402


def test_independent_check_raises_on_corrupted_missing():
    with pytest.raises(VerificationError):
        # 3 * 7 * 3 == 63, not 999 -- the check must catch this.
        _p3a._verify_multiplication_division_inverse("three_factor_product", 3, 7, 63, 999)


def test_independent_check_passes_on_correct_missing():
    """Sanity check on the test above: the correctly-recomputed value must
    NOT raise, otherwise the firing test would be vacuous."""
    _p3a._verify_multiplication_division_inverse("three_factor_product", 3, 7, 63, 3)


def test_hard_tier_clears_three_distinct_shapes():
    shapes = {
        prompt_shape(make(TOPIC, seed=seed, difficulty="hard", shape_choices=HARD_SOURCE_SHAPES)) for seed in SEEDS
    }
    assert len(shapes) >= 3, shapes


def test_hard_tier_at_least_3_distinct_rendered_questions_over_60_seeds():
    texts = {make(TOPIC, seed=seed, difficulty="hard", shape_choices=HARD_SOURCE_SHAPES).text for seed in SEEDS}
    assert len(texts) >= 3, f"only {len(texts)} distinct hard-tier questions over 60 seeds"


def test_named_presets_easy_medium_hard_unchanged():
    for seed in range(5):
        for difficulty in ("easy", "medium", "hard"):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert "three_factor_product" not in (q.working[0] if q.working else "")
            assert q.text.count("×") <= 1 or "÷" in q.text, (
                f"seed={seed} difficulty={difficulty}: unexpected shape in {q.text!r}"
            )


def test_default_shape_choices_still_excludes_three_factor_product():
    """A bare call (no `shape_choices`) defaults to the original four
    shapes, per the function's own `if shape_choices is None:` default --
    untouched by this change."""
    for seed in range(10):
        q = make(TOPIC, seed=seed)
        assert q.text.count("×") <= 1


# ---------------------------------------------------------------------------
# Skill contract.


def test_skill_contract_hard_source_carries_the_new_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    skill = by_id[SKILL_ID]

    assert skill.answer == "number"
    assert skill.status == "active"
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard"]

    hard_source = skill.sources[2]
    assert hard_source.stages == [5, 6, "maintenance"]
    hard_shapes = set(hard_source.params.get("shape_choices", []))
    assert "three_factor_product" in hard_shapes
    # Still ALSO offers the four existing shapes.
    assert hard_shapes == set(HARD_SOURCE_SHAPES)

    medium_source = skill.sources[1]
    medium_shapes = set(medium_source.params.get("shape_choices", []))
    assert "three_factor_product" not in medium_shapes

    misconception_ids = {m.id for m in skill.misconceptions}
    assert "single-factor-divide" in misconception_ids


def test_skill_loads_without_error():
    """`load_skills` never raises; a malformed file becomes a `status:
    deferred, deferred_reason: invalid` stand-in instead
    (`mathgen/skills.py::load_skills` docstring) -- confirm this skill is NOT
    one of those."""
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    assert SKILL_ID in by_id
    skill = by_id[SKILL_ID]
    assert skill.status == "active"
    assert getattr(skill, "deferred_reason", None) != "invalid"
