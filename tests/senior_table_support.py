"""Shared helpers for the senior `table` skill tests (dividend portfolio, reducing-balance schedule,
percentaged two-way table). Nothing here imports a generator module.
"""

from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction

from mathgen import make
from mathgen.grading.typed import grade_table, table_prompt
from mathgen.skills import load_skills

TIERS = ["easy", "medium", "hard", "exam"]


def load_skill(skill_id):
    _domains, skills = load_skills()
    return next(s for s in skills if s.id == skill_id)


def make_tier(skill, difficulty, seed):
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def blank_keys(q):
    return sorted(q.answer["blanks"], key=lambda k: tuple(map(int, k.split(","))))


def unit(blank) -> Fraction:
    """One unit in the last written place of a canonical blank."""
    text = str(blank)
    places = len(text.split(".")[1]) if "." in text else 0
    return Fraction(1, 10**places)


def bump(blank, steps=1) -> str:
    """The blank changed by `steps` units of its last written place, written to the same places."""
    value = Fraction(str(blank)) + steps * unit(blank)
    places = len(str(blank).split(".")[1]) if "." in str(blank) else 0
    scaled = value * 10**places
    assert scaled.denominator == 1
    digits = str(abs(scaled.numerator)).rjust(places + 1, "0")
    text = digits if not places else f"{digits[:-places]}.{digits[-places:]}"
    return ("-" if value < 0 else "") + text


def assert_skill_shape(skill, topic, area_tag, notes_seen):
    assert skill.status == "active"
    assert skill.answer == "table"
    assert [s.difficulty for s in skill.sources] == TIERS
    assert all(s.topic == topic for s in skill.sources)
    exam = skill.sources[-1]
    assert exam.stages == ["exam"]
    assert area_tag in skill.tags
    assert notes_seen <= {m.id for m in skill.misconceptions}
    assert {m.id for m in skill.misconceptions} <= notes_seen, "a listed misconception is never exercised"


def used_notes(skill, seeds=range(80)):
    return {n for tier in TIERS for s in seeds for n in make_tier(skill, tier, s).option_notes.values()}


def assert_grader_behaviour(skill, tier, seeds):
    miscon = {m.id for m in skill.misconceptions}
    for seed in seeds:
        q = make_tier(skill, tier, seed)
        blanks = q.answer["blanks"]
        keys = blank_keys(q)
        assert grade_table(q, q.answer_text).correct
        assert grade_table(q, json.dumps({k: str(blanks[k]) for k in keys})).correct
        assert grade_table(q, ", ".join(str(blanks[k]) for k in keys)).correct
        assert 2 <= len(q.distractors) <= 3
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert d != q.answer_text
            assert not grade_table(q, d).correct, (tier, seed, d)
            assert q.option_notes[d] in miscon


def assert_each_blank_is_graded(skill, tier, seeds):
    """Moving any one blank by one unit of its written place, or dropping it, is rejected."""
    for seed in seeds:
        q = make_tier(skill, tier, seed)
        blanks = q.answer["blanks"]
        for key in blanks:
            base = {k: str(v) for k, v in blanks.items()}
            for steps in (1, -1):
                moved = dict(base)
                moved[key] = bump(blanks[key], steps)
                assert not grade_table(q, json.dumps(moved)).correct, (tier, seed, key, steps)
            assert not grade_table(q, json.dumps({k: v for k, v in base.items() if k != key})).correct


def assert_masked_prompt(skill, tier, seeds):
    for seed in seeds:
        q = make_tier(skill, tier, seed)
        prompt = table_prompt(q.answer)
        for key in q.answer["blanks"]:
            r, c = map(int, key.split(","))
            assert prompt["rows"][r][c] is None
        assert len(set(q.answer_text.split(", "))) == len(q.answer["blanks"])
        assert "\n" not in q.text and "  " not in q.text
        assert "Table (" in q.text and "?" in q.text
        # the answer never appears in the question text
        for value in q.answer["blanks"].values():
            assert f"= {value}" not in q.text


def assert_variety(skill, tier, n=200):
    qs = [make_tier(skill, tier, s) for s in range(n)]
    assert len(Counter(q.answer_text for q in qs)) >= 0.9 * n
    pool = Counter(v for q in qs for v in q.answer["blanks"].values())
    assert max(pool.values()) <= 0.2 * sum(pool.values())
    return qs
