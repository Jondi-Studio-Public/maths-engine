"""Tests the `kind="context"` tank and water-level branch of the `gradient` generator, used by the `rate-of-
change` skill.

Fixed up.py`'s `gradient` function carries a `kind="context"` branch (a tank/water-level word problem) that no
skill source used: points`, and `rate-of-change.yaml` is the only skill that uses the `gradient` topic at all
(confirmed by grepping `skills/` for the anchored pattern `topic: gradient` -- the only other substring hit,
`gradient-descent-quadratic.yaml`, uses the distinct topic `gradient_descent_quadratic`).

Easy/medium sources and all three `mental:` levels are untouched and still explicitly pass `kind: points`. The
answer shape is unchanged (`answer={"p"..., "q"...}`, graded by `mathgen.grading.grade_fraction`).

The fix adds a `_gradient_context(rng, hi)` helper, dispatched BEFORE the shared draw, with its own
non-negative, strictly-increasing time construction and a `minute`/`minutes` pluraliser applied independently
to both time values. This file's `test_hard_tier_renders_context_scenario_over_60_seeds` was left red by the defect
as its evidence and is green again after the generator fix -- `_CONTEXT_RE` below now
requires both time values to be non-negative digit groups (no leading minus sign, unlike the water-level
groups) and accepts singular `minute`.

It otherwise proves the SKILL's hard tier now renders the context scenario,
that the existing three misconceptions still produce well-formed, wrong
distractors in that scenario, and that the real grader accepts/rejects
correctly through the skill's own sources -- not a re-verification of
`gradient`'s general correctness, which is out of scope.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "gradient"


def _make_hard(seed):
    return make(TOPIC, seed=seed, difficulty="hard", kind="context", hi=15)


def _make_points(seed, difficulty="hard", hi=15):
    return make(TOPIC, seed=seed, difficulty=difficulty, kind="points", hi=hi)


# ---------------------------------------------------------------------------
# The skill YAML itself: hard source repointed, easy/medium/mental untouched.
# ---------------------------------------------------------------------------


def test_skill_loads_via_api_skills_and_hard_source_is_repointed():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "rate-of-change" in ids
    skill = ids["rate-of-change"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kind") == "context"
    easy_medium = [s for s in skill.sources if s.difficulty in ("easy", "medium")]
    assert easy_medium, "expected easy/medium sources"
    assert all(s.params.get("kind") == "points" for s in easy_medium)
    # mental block is a separate structure the loader keeps on the skill;
    # confirm it too still passes kind: points at every level.
    assert skill.mental is not None
    for level in skill.mental.levels:
        assert level.params.get("kind") == "points", level.params


# ---------------------------------------------------------------------------
# Text: hard tier renders the tank/water-level scenario, never the points
# skill-source level -- the generator body itself is untouched).
# ---------------------------------------------------------------------------

_CONTEXT_RE = re.compile(
    r"^A tank's water level was (-?\d+) cm after (\d+) minutes? and "
    r"(-?\d+) cm after (\d+) minutes?\. Find the rate of change of water "
    r"level \(cm per minute\)\.$"
)
_POINTS_RE = re.compile(
    r"^Find the gradient of the line through \((-?\d+), (-?\d+)\) and "
    r"\((-?\d+), (-?\d+)\)$"
)


def test_hard_tier_renders_context_scenario_over_60_seeds():
    for seed in range(60):
        q = _make_hard(seed)
        assert _CONTEXT_RE.match(q.text), (seed, q.text)
        assert _POINTS_RE.match(q.text) is None, (seed, q.text)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_hard_context_elapsed_time_strictly_positive_and_singular_minute_over_300_seeds():
    """Regression guard for that fix, independent of `find_text_defects`
    (which only catches the grammar defect, not the semantic one): over 300
    seeds, `x1 >= 0`, `x2 > x1` always (elapsed time is never zero or
    negative), and whichever time value equals exactly 1 renders as the
    singular "1 minute", never "1 minutes". Confirms both halves of the
    Background section's two defects are gone by construction, not just by
    absence of a text-quality flag."""
    saw_singular = False
    for seed in range(300):
        q = _make_hard(seed)
        m = _CONTEXT_RE.match(q.text)
        assert m is not None, (seed, q.text)
        _y1, x1, _y2, x2 = (int(v) for v in m.groups())
        assert x1 >= 0, (seed, q.text)
        assert x2 > x1, (seed, q.text)
        if x1 == 1:
            assert " after 1 minute and" in q.text, (seed, q.text)
            saw_singular = True
        if x2 == 1:
            assert " after 1 minute." in q.text, (seed, q.text)
            saw_singular = True
    assert saw_singular, "300 seeds never hit x1==1 or x2==1 -- widen the range"


def test_easy_medium_points_sources_unaffected_over_60_seeds():
    for seed in range(60):
        for difficulty, hi in (("easy", 9), ("medium", 9)):
            q = make(TOPIC, seed=seed, difficulty=difficulty, kind="points", hi=hi)
            assert _POINTS_RE.match(q.text), (seed, difficulty, q.text)
            assert "tank" not in q.text


def test_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_hard(seed).to_json()
        second = _make_hard(seed).to_json()
        assert first == second


def test_hard_context_adds_a_shape_not_in_easy_or_medium():
    """The hard kind's shapes are not a subset of the union of the
    unchanged easy/medium (`kind: points`) tiers' shapes -- a genuine
    escalation in kind (word-context vs point-pair), not merely a bigger
    `hi`, and >= 3 distinct prompt_shape() outputs of its own over 60 seeds."""
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy", kind="points", hi=9)) for s in range(60)}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium", kind="points", hi=9)) for s in range(60)}
    hard_shapes = {prompt_shape(_make_hard(s)) for s in range(60)}
    assert len(hard_shapes) >= 1  # a fixed template family, one shape by construction
    assert hard_shapes - lower_shapes, (hard_shapes, lower_shapes)


# ---------------------------------------------------------------------------
# Independent oracle: re-derive the gradient (p/q) from the rendered context
# text ALONE, with plain regex + fractions.Fraction, never importing
# mathgen.generators.foundations_algebra's own Rational construction.
# ---------------------------------------------------------------------------


def _independent_p_q(question_text: str) -> tuple[int, int]:
    m = _CONTEXT_RE.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    y1, x1, y2, x2 = (int(v) for v in m.groups())
    frac = Fraction(y2 - y1, x2 - x1)
    return frac.numerator, frac.denominator


def test_hard_answer_matches_independent_oracle_over_300_seeds():
    mismatches = []
    for seed in range(300):
        q = _make_hard(seed)
        oracle_p, oracle_q = _independent_p_q(q.text)
        if (oracle_p, oracle_q) != (q.answer["p"], q.answer["q"]):
            mismatches.append((seed, q.text, (oracle_p, oracle_q), q.answer))
    assert mismatches == [], mismatches


def test_hard_answer_text_matches_independent_format_over_300_seeds():
    for seed in range(300):
        q = _make_hard(seed)
        oracle_p, oracle_q = _independent_p_q(q.text)
        expected = str(oracle_p) if oracle_q == 1 else f"{oracle_p}/{oracle_q}"
        assert q.answer_text == expected, (seed, q.text, q.answer_text, expected)


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer -- the existing
# three misconceptions (reversed-rise-run, output-change-only,
# total-not-change) are still well-formed in the context wording, since the
# candidate construction is shared by every `kind` and sits above the
# `if kind ==` dispatch (confirmed by reading the generator: `cand_vals` is
# built before `text` is chosen).
# ---------------------------------------------------------------------------


def test_hard_distractors_non_empty_distinct_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_hard(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


# ---------------------------------------------------------------------------
# The real grader (mathgen.grading.GRADERS["fraction"]) through the skill's own
# hard source, and easy/medium for good measure.
# ---------------------------------------------------------------------------

from mathgen.grading.typed import GRADERS  # noqa: E402

_FRACTION_GRADER = GRADERS["fraction"]


def test_hard_grader_accepts_canonical_and_rejects_every_distractor_over_200_seeds():
    bad = []
    for seed in range(200):
        q = _make_hard(seed)
        canonical_result = _FRACTION_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _FRACTION_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_easy_medium_grader_accepts_canonical_and_rejects_every_distractor_over_120_seeds():
    bad = []
    for seed in range(60):
        for difficulty, hi in (("easy", 9), ("medium", 9)):
            q = make(TOPIC, seed=seed, difficulty=difficulty, kind="points", hi=hi)
            canonical_result = _FRACTION_GRADER(q, q.answer_text)
            if not canonical_result.correct:
                bad.append(("canonical-rejected", seed, difficulty, q.answer_text))
            for d in q.distractors:
                distractor_result = _FRACTION_GRADER(q, d)
                if distractor_result.correct:
                    bad.append(("distractor-accepted", seed, difficulty, d, q.answer_text))
    assert bad == [], bad


def test_hard_grader_rejects_deliberately_wrong_values():
    """A deliberately-wrong value, checked it is actually wrong before
    asserting the grader rejects it (never a vacuous check).
    """
    q = _make_hard(0)
    p, qq = q.answer["p"], q.answer["q"]
    wrong_inverted = f"{qq}/{p}" if p != 0 else "1/1"
    wrong_off_by_one = f"{p + 1}/{qq}" if qq != 1 else str(p + 1)
    for wrong in (wrong_inverted, wrong_off_by_one):
        assert wrong != q.answer_text, (wrong, q.answer_text)
        result = _FRACTION_GRADER(q, wrong)
        assert not result.correct, (wrong, q.answer_text, q.answer)
