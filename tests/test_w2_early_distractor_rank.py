"""w2: early-years drill MCQs no longer give the key away by rank; yes/no skills have four options; tiny pools are wide;
the new tiers grade; three skills read aloud within 15 words."""

from __future__ import annotations

from collections import Counter

import pytest

from mathgen import engine
from mathgen.core import mcq_options
from mathgen.engine import grade, load_skills

_SKILLS = {s.id: s for s in load_skills()[1]}

DRILL = (
    "count-to-5",
    "count-to-10",
    "order-numbers-to-10",
    "skip-count-2-5-10",
    "multiplication-facts",
    "equal-groups",
    "add-within-20",
    "subtract-within-20",
    "subtract-within-10",
    "number-bonds-to-10",
    "fair-sharing",
)

FOUR_OPTION = (
    "f10-factors-multiples-divisibility",
    "f10-odd-even-properties",
    "f10-prime-composite-square-properties",
    "f10-shape-features-classification",
)

WIDENED = (
    ("f10-fractions-half-quarter-eighth", None),
    ("f10-angle-turn-right-reference", "easy"),
    ("f10-percentage-equivalence-comparison", "easy"),
    ("f10-unit-fractions-combine-whole", "easy"),
    ("rearrange-simple-formulas", "hard"),
    ("f10-circle-measures", None),
)

READ_ALOUD = ("f10-clock-half-quarter-hour", "f2-read-a-picture-graph", "f10-measure-uniform-informal-units")


def _questions(skill_id, seeds):
    s = _SKILLS[skill_id]
    for i, src in enumerate(s.sources):
        out = []
        for seed in range(seeds):
            try:
                out.append(engine.make_for_skill(s, seed, source=i))
            except Exception:  # a seed the source cannot build is not part of this check
                continue
        yield src, out


@pytest.mark.parametrize("skill_id", [s for s in DRILL if s in _SKILLS])
def test_key_rank_is_not_a_tell(skill_id):
    for src, qs in _questions(skill_id, 120):
        ranks = Counter()
        for q in qs:
            opts = mcq_options(q)
            nums = []
            for o in opts:
                try:
                    nums.append(float(str(o).replace(",", "")))
                except ValueError:
                    nums = []
                    break
            if len(nums) == len(opts) and len(opts) >= 4:
                ranks[sorted(nums).index(float(str(q.answer_text).replace(",", "")))] += 1
        if sum(ranks.values()) >= 60:
            assert max(ranks.values()) / sum(ranks.values()) <= 0.45, (skill_id, src.difficulty, dict(ranks))


@pytest.mark.parametrize("skill_id", FOUR_OPTION)
def test_no_two_option_questions(skill_id):
    for src, qs in _questions(skill_id, 40):
        assert qs, (skill_id, src.difficulty)
        assert all(len(mcq_options(q)) >= 4 for q in qs), (skill_id, src.difficulty)


@pytest.mark.parametrize("skill_id,tier", WIDENED)
def test_widened_pools_have_twelve_distinct_questions(skill_id, tier):
    by_tier: dict[str, set[str]] = {}
    for src, qs in _questions(skill_id, 60):
        if src.difficulty != "exam" and (not tier or src.difficulty == tier):
            by_tier.setdefault(src.difficulty, set()).update(q.text for q in qs)
    assert by_tier
    for difficulty, texts in by_tier.items():
        assert len(texts) >= 12, (skill_id, difficulty, len(texts))


@pytest.mark.parametrize("skill_id", ["rearrange-simple-formulas", "calculate-median"])
def test_new_sources_grade_key_right_and_distractors_wrong(skill_id):
    s = _SKILLS[skill_id]
    for i, src in enumerate(s.sources):
        if src.topic not in ("w2_rearrange_roots_more", "w2_median_even"):
            continue
        for seed in range(40):
            q = engine.make_for_skill(s, seed, source=i)
            assert grade(s, q, q.answer_text).correct, (q.text, q.answer_text)
            for d in q.distractors:
                assert not grade(s, q, d).correct, (q.text, d)


def test_median_has_an_even_count_tier():
    s = _SKILLS["calculate-median"]
    assert any(src.topic == "w2_median_even" for src in s.sources)


@pytest.mark.parametrize("skill_id", READ_ALOUD)
def test_read_aloud_prompts_fit(skill_id):
    s = _SKILLS[skill_id]
    assert s.read_aloud
    for _src, qs in _questions(skill_id, 20):
        assert all(len(q.text.split()) <= 15 for q in qs)


def test_simple_interest_uses_dollars():
    s = _SKILLS["use-formulas"]
    for _src, qs in _questions("use-formulas", 40):
        assert not any("£" in q.text for q in qs)
