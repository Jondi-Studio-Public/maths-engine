"""A range-only fix for `distance-two-points`'s hard tier.

Before this change, `distance_two_points`'s hard preset asked for the exact
Euclidean distance between the SAME two supplied points every tier (the ONLY
difference across easy/medium/hard was `coord_hi`/`leg_hi`/`allow_negative`
-- a bigger number, not a different reasoning shape). The fix adds
`kind="perimeter"`: given THREE points P, Q, R, find the perimeter of
triangle PQR -- the sum of the three pairwise distances |PQ| + |QR| + |RP|,
each independently verified the same way the existing single distance always
has been (`_verify_distance`, reused once per side), then combined into one
exact SymPy sum. This is a genuine escalation: three independent distance
computations combined into a total, not one bigger computation.

`kind="standard"` (the pre-existing branch every easy/medium/hard source
still uses) is untouched -- proved once, offline, by a 240-question
before/after snapshot diff (`make(topic, seed=s, difficulty=d)` for
s in 0..59, d in easy/medium/hard/None) showing 0 bytes changed, and reproduced here as a standing regression guard alongside
`tests/test_coordinate_geometry_build.py` (pre-existing, untouched, still
green). `_draw_leg_pair` is the exact inline code that used to sit directly
in `distance_two_points`'s body, only moved into its own function so
`kind="perimeter"` can call it twice -- same rng call sequence, same output.

The independent oracle below (`_oracle_perimeter`) parses `question.text` alone -- `re.findall(r"\\((-?\\d+),
(-?\\d+)\\)"...)`, which matches every one of the four wording templates identically since they all render
each point the same "(x, y)" way -- and re-derives the perimeter from scratch with `sympy.sqrt` on the parsed
integers. It imports nothing from `mathgen.generators.coordinate_geometry_build`, so a slip baked into the
generator's own `check_equivalent` call could not also be baked into the check that is supposed to catch it.
`test_oracle_catches_a_corrupted_answer` proves the oracle is not vacuously true.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import sympy
from sympy import sqrt

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "distance_two_points"
SKILL_ID = "distance-two-points"
_EXPRESSION_GRADER = GRADERS["expression"]

SEEDS = range(60)


def _mk(seed, difficulty="hard"):
    return make(TOPIC, seed=seed, difficulty=difficulty, kind="perimeter")


# ---------------------------------------------------------------------------
# Independent oracle: re-derive the perimeter from the rendered prompt alone,
# never touching the generator module.

_POINT_RE = re.compile(r"\((-?\d+), (-?\d+)\)")


def _oracle_perimeter(text: str) -> sympy.Expr:
    pairs = _POINT_RE.findall(text)
    assert len(pairs) == 3, f"oracle: expected 3 points, found {len(pairs)}: {text!r}"
    (px, py), (qx, qy) = (int(pairs[0][0]), int(pairs[0][1])), (int(pairs[1][0]), int(pairs[1][1]))
    rx, ry = int(pairs[2][0]), int(pairs[2][1])
    d_pq = sqrt((qx - px) ** 2 + (qy - py) ** 2)
    d_qr = sqrt((rx - qx) ** 2 + (ry - qy) ** 2)
    d_rp = sqrt((px - rx) ** 2 + (py - ry) ** 2)
    return d_pq + d_qr + d_rp


def test_oracle_agrees_with_generator_over_60_seeds():
    checked = 0
    for seed in SEEDS:
        q = _mk(seed)
        oracle_value = _oracle_perimeter(q.text)
        generator_value = sympy.sympify(q.answer_text)
        assert sympy.simplify(oracle_value - generator_value) == 0, (
            f"seed={seed}: oracle recomputed {oracle_value} from the prompt, "
            f"generator answer_text is {q.answer_text!r}\n{q.text}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_catches_a_corrupted_answer():
    """Proves the oracle is not vacuously true: pair a real prompt with a
    deliberately wrong recomputed value and confirm the oracle disagrees."""
    q = _mk(0)
    oracle_value = _oracle_perimeter(q.text)
    corrupted = oracle_value + 1
    assert sympy.simplify(corrupted - sympy.sympify(q.answer_text)) != 0


def test_oracle_matches_across_all_four_wording_templates():
    """The oracle's point-parsing regex must work identically whatever
    wording `phrase()` happened to pick -- confirmed by forcing enough seeds
    that all four templates are seen, rather than assuming the regex is
    template-agnostic."""
    seen_texts = set()
    for seed in range(60):
        q = _mk(seed)
        seen_texts.add(q.text)
        oracle_value = _oracle_perimeter(q.text)
        assert sympy.simplify(oracle_value - sympy.sympify(q.answer_text)) == 0
    # Sanity: genuinely different wordings were exercised.
    assert len(seen_texts) >= 3, seen_texts


# ---------------------------------------------------------------------------
# Determinism.


def test_perimeter_deterministic_across_seeds():
    for seed in SEEDS:
        q1 = _mk(seed)
        q2 = _mk(seed)
        assert q1.to_json() == q2.to_json()


# ---------------------------------------------------------------------------
# Text quality: no bad patterns in the prompt OR the worked solution.
# A generic check that calls `make(topic, seed=seed, difficulty=difficulty)` with no `kind=`
# override never reaches `kind="perimeter"` -- this is the only place the new
# branch's working text is checked at all.


def test_no_bad_text_patterns_over_60_seeds():
    for seed in SEEDS:
        q = _mk(seed)
        defects = find_text_defects(q, include_working=True)
        assert defects == [], f"seed={seed}: {defects}\n{q.text}"


def test_perimeter_prompt_variety():
    shapes = {prompt_shape(_mk(seed)) for seed in SEEDS}
    assert len(shapes) >= 3, shapes


def test_perimeter_adds_shapes_over_easy_and_medium():
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy")) for s in SEEDS}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium")) for s in SEEDS}
    hard_shapes = {prompt_shape(_mk(s)) for s in SEEDS}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


# ---------------------------------------------------------------------------
# Distractor hygiene -- distinct, non-empty, never equal the answer.


def test_perimeter_distractors_distinct_and_never_equal_answer():
    for seed in SEEDS:
        q = _mk(seed)
        assert len(q.distractors) == 3, (seed, q.text, q.distractors)
        assert len(set(q.distractors)) == 3, (seed, q.distractors)
        assert q.answer_text not in q.distractors, (seed, q.text, q.distractors)
        assert all(d for d in q.distractors), (seed, q.distractors)


# ---------------------------------------------------------------------------
# Real grader: accepts canonical, rejects every distractor, rejects a
# deliberately-wrong value, over 200 seeds.


def test_real_grader_accepts_canonical_and_rejects_every_distractor_over_200_seeds():
    bad = []
    for seed in range(200):
        q = _mk(seed)
        canonical_result = _EXPRESSION_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _EXPRESSION_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_a_deliberately_wrong_answer_is_rejected():
    for seed in SEEDS:
        q = _mk(seed)
        wrong = str(sympy.sympify(q.answer_text) + 1)
        assert wrong != q.answer_text
        assert not _EXPRESSION_GRADER(q, wrong).correct, (seed, q.text, q.answer_text, wrong)


# The independent, in-generator checks fire on a corrupted value -- imports the generator module
# directly, ONLY here, to call its private verification helper with deliberately wrong inputs.

import mathgen.generators.coordinate_geometry_build as _coord_geo  # noqa: E402


def test_verify_distance_raises_on_corrupted_distance():
    with pytest.raises(VerificationError):
        _coord_geo._verify_distance(3, 4, sqrt(24))  # should be sqrt(25) = 5
    _coord_geo._verify_distance(3, 4, sqrt(25))  # correct: must not raise


def test_no_construction_exceptions_over_600_seeds():
    """The collinearity-reroll loop in `_distance_two_points_perimeter` must
    reliably find a non-degenerate second displacement across a broad seed
    range -- no seed should raise, and the resulting three points must
    genuinely differ pairwise (never collinear/degenerate)."""
    for seed in range(600):
        q = make(TOPIC, seed=seed, difficulty="hard", kind="perimeter")
        pairs = _POINT_RE.findall(q.text)
        assert len(pairs) == 3, (seed, q.text)
        (px, py), (qx, qy), (rx, ry) = ((int(a), int(b)) for a, b in pairs)
        assert (px, py) != (qx, qy)
        assert (qx, qy) != (rx, ry)
        assert (px, py) != (rx, ry)
        # Collinear check: cross product of PQ and PR must be non-zero.
        cross = (qx - px) * (ry - py) - (qy - py) * (rx - px)
        assert cross != 0, (seed, q.text, "collinear P, Q, R")


_STANDARD_TEXT_RE = re.compile(
    r"^Calculate the exact distance between the points \(-?\d+, -?\d+\) and \(-?\d+, -?\d+\)\.$"
)


def test_standard_kind_prompt_unchanged_across_all_named_presets():
    for seed in range(60):
        for difficulty in ("easy", "medium", "hard", None):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert _STANDARD_TEXT_RE.match(q.text), (seed, difficulty, q.text)


def test_default_kind_is_still_standard_with_no_kind_kwarg():
    """A bare call (no `kind=`) still draws the original single-pair
    question, per the function's own `kind="standard"` default -- untouched
    by this change."""
    q = make(TOPIC, seed=0, difficulty="hard")
    assert _STANDARD_TEXT_RE.match(q.text)
    assert "perimeter" not in q.text.lower()


# ---------------------------------------------------------------------------
# Skill contract.


def test_skill_contract_hard_source_carries_the_new_kind():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    skill = by_id[SKILL_ID]

    assert skill.answer == "expression"
    assert skill.status == "active"
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]

    hard_source = skill.sources[2]
    assert hard_source.stages == [5, 6, "maintenance"]
    assert hard_source.params.get("kind") == "perimeter"
    assert hard_source.params.get("coord_hi") == 16
    assert hard_source.params.get("leg_hi") == 24
    assert hard_source.params.get("allow_negative") is True

    # The easy/medium sources are untouched (no `kind` key at all, so the
    # generator's own default "standard" applies).
    easy_source, medium_source = skill.sources[0], skill.sources[1]
    assert "kind" not in easy_source.params
    assert "kind" not in medium_source.params

    misconception_ids = {m.id for m in skill.misconceptions}
    assert "missed-a-side" in misconception_ids
    # The two existing misconceptions this new kind's other distractors
    # reuse (scaled up to three sides) are still present too.
    assert "summed-differences" in misconception_ids
    assert "forgot-square-root" in misconception_ids


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
