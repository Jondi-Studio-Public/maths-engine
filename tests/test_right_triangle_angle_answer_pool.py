"""The fix draws arbitrary integer side lengths for this skill's own "angle" draws instead, generated through the
skill's own `sources` (topic + params + difficulty), never a bare `make(topic, seed)`.

Also proves the fix is genuinely opt-in: `pythagorean-hypotenuse` and
`pythagorean-leg` (which need whole-number answers, so their triple-based
draws are correct for them) are untouched, and so is any bare
`make("trigonometry", seed)` call with no `angle_style`.
"""

import re

import pytest

from mathgen.skills import load_skills
from mathgen.core import make

MIN_DISTINCT_ANSWERS = 25
SEEDS = 40


def _source_params(skill_id, difficulty):
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    matches = [s for s in skill.sources if s.difficulty == difficulty]
    assert len(matches) == 1, f"{skill_id}: expected exactly one {difficulty!r} source"
    return matches[0].topic, matches[0].params


def _questions(difficulty, seeds=SEEDS):
    topic, params = _source_params("right-triangle-angle", difficulty)
    return [make(topic, seed=s, difficulty=difficulty, **params) for s in range(seeds)]


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_answer_pool_is_no_longer_six_values(difficulty):
    """Before this fix every tier of right-triangle-angle had exactly 6 distinct answers over 40 seeds; the fixed
    generator must clear a much wider bar."""
    qs = _questions(difficulty)
    answers = {q.answer_text for q in qs}
    assert len(answers) >= MIN_DISTINCT_ANSWERS, (
        f"{difficulty}: only {len(answers)} distinct answers over {SEEDS} seeds "
        f"(want >= {MIN_DISTINCT_ANSWERS}): {sorted(answers)}"
    )


def test_easy_tier_always_gives_both_legs_and_uses_tan():
    """Easy: both legs given (opposite and adjacent), so tan is the only
    correct ratio -- the tier that teaches the basic SOH-CAH-TOA setup."""
    for q in _questions("easy"):
        assert "opposite side" in q.text and "adjacent side" in q.text
        assert any("tan⁻¹" in w for w in q.working)


def test_medium_tier_always_gives_a_leg_and_the_hypotenuse():
    """Medium: a leg and the hypotenuse are given, never both legs, so the
    learner must choose sin or cos rather than defaulting to tan."""
    for q in _questions("medium"):
        assert "hypotenuse" in q.text
        assert "opposite side" not in q.text and "adjacent side" not in q.text
        assert any("sin⁻¹" in w or "cos⁻¹" in w for w in q.working)
        assert not any("tan⁻¹" in w for w in q.working)


def test_hard_tier_exercises_all_three_ratios():
    """Hard: mixed ratios at larger magnitude -- over enough seeds, tan, sin
    and cos must each appear (a tier that only ever gives opposite+adjacent
    only ever rehearses tan)."""
    seen = set()
    for q in _questions("hard", seeds=60):
        for w in q.working:
            for sym in ("tan⁻¹", "sin⁻¹", "cos⁻¹"):
                if sym in w:
                    seen.add(sym)
    assert seen == {"tan⁻¹", "sin⁻¹", "cos⁻¹"}, f"hard tier only exercised {seen}"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_complementary_angle_distractor_present_when_distinct(difficulty):
    """The complementary angle (90 - theta) is the classic misconception and
    must be offered as a distractor whenever it differs from the true answer
    by enough to not collide with it."""
    missing = 0
    total = 0
    for q in _questions(difficulty):
        ans = float(re.search(r"-?\d+(?:\.\d+)?", q.answer_text).group())
        if abs(ans - 45.0) < 0.15:
            continue  # complementary angle collides with the answer near 45 deg
        total += 1
        comp = round(90 - ans, 1)
        comp_text = f"θ = {comp}°"
        if comp_text not in q.distractors:
            missing += 1
    assert missing == 0, f"{difficulty}: complementary-angle distractor missing on {missing}/{total} draws"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_never_collide_with_the_answer(difficulty):
    """Every distractor must be numerically distinct from the answer by more
    than the grader's own tolerance."""
    num_re = re.compile(r"-?\d+(?:\.\d+)?")
    for q in _questions(difficulty):
        ans_val = float(num_re.search(q.answer_text).group())
        assert q.answer_text not in q.distractors
        for d in q.distractors:
            d_val = float(num_re.search(d).group())
            assert abs(d_val - ans_val) > 0.051, (
                f"{difficulty}: distractor {d!r} is within grader tolerance of answer {q.answer_text!r}"
            )


def test_pythagorean_hypotenuse_and_leg_use_default_triple_style():
    """The other two skills never set `angle_style`, so they never reach the
    new branch at all -- this is a structural regression guard, not merely a
    behavioural one: `pythagorean-hypotenuse`/`pythagorean-leg` pin
    `kinds` to `side_hyp`/`side_leg`, which never touches the "angle" branch
    regardless of `angle_style`."""
    for skill_id in ("pythagorean-hypotenuse", "pythagorean-leg"):
        for difficulty in ("easy", "medium", "hard"):
            topic, params = _source_params(skill_id, difficulty)
            assert "angle_style" not in params
            assert "angle_lo" not in params
            assert "angle_hi" not in params


def test_bare_make_default_angle_style_matches_explicit_triple():
    """`angle_style="triple"` is the function's own default, so a bare
    `make("trigonometry", seed)` call (no skill params at all, as used
    elsewhere in the test suite and by any future caller) is provably
    unaffected by this change: passing `angle_style="triple"` explicitly must
    reproduce the exact same output."""
    for seed in range(20):
        for difficulty in (None, "easy", "medium", "hard"):
            implicit = make("trigonometry", seed=seed, difficulty=difficulty)
            explicit = make("trigonometry", seed=seed, difficulty=difficulty, angle_style="triple")
            assert implicit.to_json() == explicit.to_json()


def test_unknown_angle_style_raises():
    with pytest.raises(ValueError):
        make("trigonometry", seed=0, kinds=("angle",), angle_style="bogus")
