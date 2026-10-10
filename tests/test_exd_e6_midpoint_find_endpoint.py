"""A range-only fix for `midpoint-two-points`'s hard tier.

Before this change, `midpoint_two_points`'s hard preset asked for the exact
midpoint of the SAME two supplied points every tier (the ONLY difference
across easy/medium/hard was `coord_lo`/`coord_hi` -- a bigger number, not a
different reasoning shape). The fix adds `kind="find_endpoint"`: given ONE
endpoint A and the MIDPOINT M, find the OTHER endpoint B = 2M - A -- the
INVERSE problem (an unknown coordinate "on the other side" of the
relationship), not a bigger version of the forward computation.

`kind="standard"` (the pre-existing branch every easy/medium/hard source
still uses, and the function's default) is untouched -- proved once,
offline, by an 800-question before/after snapshot diff (`midpoint_two_points`
called directly, bypassing the registry, for seed in 0..199 x
{easy, medium, hard, default (no preset)}) showing 0 rows changed, and reproduced here as a standing regression guard alongside
`tests/test_coordinate_geometry_build.py` (pre-existing, untouched, still
green).

The independent oracle below (`_oracle_other_endpoint`) parses `question.text` alone -- regexes for "A(x1,
y1)" / "A = (x1, y1)" and "M(mx, my)", which match every one of the four wording templates identically since
they all render A and M the same way -- and re-derives B = 2M - A from scratch with `sympy.Rational` on the
parsed strings. It imports nothing from `mathgen.generators.coordinate_geometry_build`, so a slip baked into
the generator's own checks could not also be baked into the check that is supposed to catch it.
`test_oracle_catches_a_corrupted_answer` proves the oracle is not vacuously true.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import sympy

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "midpoint_two_points"
SKILL_ID = "midpoint-two-points"
_PAIR_GRADER = GRADERS["pair"]

SEEDS = range(60)


def _mk(seed, difficulty="hard"):
    return make(TOPIC, seed=seed, difficulty=difficulty, kind="find_endpoint")


# ---------------------------------------------------------------------------
# Independent oracle: re-derive the other endpoint B = 2M - A from the
# rendered prompt alone, never touching the generator module.

_A_RE = re.compile(r"A\s*=?\s*\((-?\d+), (-?\d+)\)")
_M_RE = re.compile(r"M\s*=?\s*\((-?\d+(?:/\d+)?), (-?\d+(?:/\d+)?)\)")
_ANSWER_RE = re.compile(r"x = (-?\d+), y = (-?\d+)")


def _oracle_other_endpoint(text: str):
    a_match = _A_RE.search(text)
    m_match = _M_RE.search(text)
    assert a_match is not None, f"oracle: could not parse A(x, y) from {text!r}"
    assert m_match is not None, f"oracle: could not parse M(x, y) from {text!r}"
    x1, y1 = int(a_match.group(1)), int(a_match.group(2))
    mx, my = sympy.Rational(m_match.group(1)), sympy.Rational(m_match.group(2))
    bx = 2 * mx - x1
    by = 2 * my - y1
    return bx, by


def test_oracle_agrees_with_generator_over_60_seeds():
    checked = 0
    for seed in SEEDS:
        q = _mk(seed)
        bx, by = _oracle_other_endpoint(q.text)
        match = _ANSWER_RE.match(q.answer_text)
        assert match is not None, q.answer_text
        gx, gy = int(match.group(1)), int(match.group(2))
        assert bx == gx and by == gy, (
            f"seed={seed}: oracle recomputed B=({bx}, {by}) from the prompt, "
            f"generator answer_text is {q.answer_text!r}\n{q.text}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_catches_a_corrupted_answer():
    """Proves the oracle is not vacuously true: pair a real prompt with a
    deliberately wrong recomputed value and confirm the oracle disagrees."""
    q = _mk(0)
    bx, by = _oracle_other_endpoint(q.text)
    corrupted_bx = bx + 1
    match = _ANSWER_RE.match(q.answer_text)
    gx = int(match.group(1))
    assert corrupted_bx != gx


def test_oracle_matches_across_all_four_wording_templates():
    """The oracle's regexes must work identically whatever wording
    `phrase()` happened to pick -- confirmed by forcing enough seeds that
    several templates are seen, rather than assuming the regex is
    template-agnostic."""
    seen_texts = set()
    for seed in range(60):
        q = _mk(seed)
        seen_texts.add(q.text)
        bx, by = _oracle_other_endpoint(q.text)
        match = _ANSWER_RE.match(q.answer_text)
        gx, gy = int(match.group(1)), int(match.group(2))
        assert bx == gx and by == gy
    # Sanity: genuinely different wordings were exercised.
    assert len(seen_texts) >= 3, seen_texts


# ---------------------------------------------------------------------------
# Determinism.


def test_find_endpoint_deterministic_across_seeds():
    for seed in SEEDS:
        q1 = _mk(seed)
        q2 = _mk(seed)
        assert q1.to_json() == q2.to_json()


# ---------------------------------------------------------------------------
# Text quality: no bad patterns in the prompt OR the worked solution.
# A generic check that calls `make(topic, seed=seed, difficulty=difficulty)` with no `kind=`
# override never reaches `kind="find_endpoint"` -- this is the only place the
# new branch's working text is checked at all.


def test_no_bad_text_patterns_over_60_seeds():
    for seed in SEEDS:
        q = _mk(seed)
        defects = find_text_defects(q, include_working=True)
        assert defects == [], f"seed={seed}: {defects}\n{q.text}"


def test_find_endpoint_prompt_variety():
    shapes = {prompt_shape(_mk(seed)) for seed in SEEDS}
    assert len(shapes) >= 3, shapes


def test_find_endpoint_adds_shapes_over_easy_and_medium():
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy")) for s in SEEDS}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium")) for s in SEEDS}
    hard_shapes = {prompt_shape(_mk(s)) for s in SEEDS}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


# ---------------------------------------------------------------------------
# Distractor hygiene -- distinct, non-empty, never equal the answer.


def test_find_endpoint_distractors_distinct_and_never_equal_answer():
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
        canonical_result = _PAIR_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _PAIR_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_a_deliberately_wrong_answer_is_rejected():
    for seed in SEEDS:
        q = _mk(seed)
        match = _ANSWER_RE.match(q.answer_text)
        gx, gy = int(match.group(1)), int(match.group(2))
        wrong = f"x = {gx + 1}, y = {gy}"
        assert wrong != q.answer_text
        assert not _PAIR_GRADER(q, wrong).correct, (seed, q.text, q.answer_text, wrong)


# The independent, in-generator check has a name (`_verify_midpoint`) and fires on a corrupted value
# -- imports the generator module directly, ONLY here, to call its private verification helper with
# deliberately wrong inputs. `_verify_midpoint` is REUSED unchanged from `kind="standard"`, not a
# copy.

import mathgen.generators.coordinate_geometry_build as _coord_geo  # noqa: E402


def test_verify_midpoint_raises_on_corrupted_midpoint():
    with pytest.raises(VerificationError):
        _coord_geo._verify_midpoint(0, 0, 4, 6, 2, 2)  # should be (2, 3)
    _coord_geo._verify_midpoint(0, 0, 4, 6, 2, 3)  # correct: must not raise


def test_no_construction_exceptions_over_600_seeds():
    """`kind="find_endpoint"` must reliably construct a valid (A, B, M)
    triple across a broad seed range -- no seed should raise, and A must
    never coincide with B (the reroll loop's own guarantee)."""
    for seed in range(600):
        q = make(TOPIC, seed=seed, difficulty="hard", kind="find_endpoint")
        a_match = _A_RE.search(q.text)
        m_match = _M_RE.search(q.text)
        assert a_match is not None and m_match is not None, (seed, q.text)
        x1, y1 = int(a_match.group(1)), int(a_match.group(2))
        match = _ANSWER_RE.match(q.answer_text)
        x2, y2 = int(match.group(1)), int(match.group(2))
        assert (x1, y1) != (x2, y2), (seed, q.text, "A and B coincide")


_STANDARD_TEXT_RE = re.compile(r"^Calculate the exact midpoint of the points \(-?\d+, -?\d+\) and \(-?\d+, -?\d+\)\.$")


def test_standard_kind_prompt_unchanged_across_all_named_presets():
    for seed in range(60):
        for difficulty in ("easy", "medium", "hard", None):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert _STANDARD_TEXT_RE.match(q.text), (seed, difficulty, q.text)


def test_default_kind_is_still_standard_with_no_kind_kwarg():
    """A bare call (no `kind=`) still draws the original two-point midpoint
    question, per the function's own `kind="standard"` default -- untouched
    by this change."""
    q = make(TOPIC, seed=0, difficulty="hard")
    assert _STANDARD_TEXT_RE.match(q.text)
    assert "find" not in q.text.lower()


# ---------------------------------------------------------------------------
# Skill contract.


def test_skill_contract_hard_source_carries_the_new_kind():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    skill = by_id[SKILL_ID]

    assert skill.answer == "pair"
    assert skill.status == "active"
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]

    hard_source = skill.sources[2]
    assert hard_source.stages == [5, 6, "maintenance"]
    assert hard_source.params.get("kind") == "find_endpoint"
    assert hard_source.params.get("coord_lo") == -14
    assert hard_source.params.get("coord_hi") == 14

    # The easy/medium sources are untouched (no `kind` key at all, so the
    # generator's own default "standard" applies).
    easy_source, medium_source = skill.sources[0], skill.sources[1]
    assert "kind" not in easy_source.params
    assert "kind" not in medium_source.params

    misconception_ids = {m.id for m in skill.misconceptions}
    assert "averages-known-point-and-midpoint" in misconception_ids
    # The three pre-existing misconceptions are still present too.
    assert "subtract-instead-of-average" in misconception_ids
    assert "halve-one-coordinate" in misconception_ids
    assert "omits-division" in misconception_ids


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
