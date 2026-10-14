"""Proofs for `f10-geometric-proof-reasons`: give the reason for each step of a short geometric proof.

The oracle reads the question text only: it recognises which proof is set from its statement of the problem, reads
the numbered reason bank back out of the text and checks that the numbers in the key point at the reasons that
proof needs. The generator also checks every proof statement numerically on random coordinates when it builds the
question, so a false step cannot be asked.
"""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make
from mathgen.engine import grade as grade_skill
from mathgen.grading import grade
from mathgen.skills import load_skills

SKILL_ID = "f10-geometric-proof-reasons"
SEEDS = range(80)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
SOURCES = {s.difficulty: s for s in _SKILL.sources}

ALT, CORR, LINE, VERT = "Alternate angles", "Corresponding angles", "Angles on a straight line", "Vertically opposite"
TRI, ISO, SUB = "The angles in a triangle", "Angles opposite equal sides", "Combining the earlier steps"
GIVEN, MID, COMMON, PPAR, PERP = (
    "stated in the question",
    "Definition of a midpoint",
    "A side shared",
    "Opposite sides of a parallelogram",
    "Definition of perpendicular",
)
CPCT = "Matching parts of congruent triangles"


def _expected(text: str) -> list[str]:
    text = text.split("Reasons: ")[0]
    prove = re.search(r"To prove: (.*?)\. Step 1", text).group(1)
    if "perpendicular" in text:
        return [MID, PERP, COMMON, "(SAS)", CPCT]
    if "parallelogram" in text:
        return [ALT, ALT, PPAR, "(ASA)", CPCT]
    if "midpoint of" in text:
        return [GIVEN, MID, COMMON, "(SSS)", CPCT]
    if "180° − 2 ×" in prove:
        return [ISO, TRI, SUB]
    if "is extended to" in text:
        return [TRI, LINE, SUB]
    if prove.count("∠") == 3 and "= 180°" in prove:
        return [ALT, ALT, LINE, SUB]
    if "cross at" in text:
        return [LINE, LINE, SUB]
    if "is parallel to" in text and "= 180°" in prove:
        return [ALT, LINE, SUB]
    if "is parallel to" in text:
        return [VERT, CORR, SUB]
    raise AssertionError(text)


def _bank(text: str) -> dict[int, str]:
    body = text.split("Reasons: ", 1)[1].split(" Write the number", 1)[0]
    return {int(n): r for n, r in re.findall(r"(\d)\. (.*?\.)(?= \d\. |$)", body)}


def _make(tier, seed):
    src = SOURCES[tier]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_key_points_at_the_reasons_the_proof_needs(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        bank = _bank(q.text)
        assert sorted(bank) == list(range(1, 9)) and len(set(bank.values())) == 8
        expected = _expected(q.text)
        assert len(q.answer) == len(expected)
        for number, want in zip(q.answer, expected):
            assert want in bank[number], (tier, seed, number, bank[number], want)
        assert "(AAS)" not in q.text


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_grader_accepts_key_rejects_every_distractor_and_a_guess_is_unlikely(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert grade("ordered_list", q, q.answer_text).correct
        assert grade("ordered_list", q, "[" + q.answer_text + "]").correct
        assert len(q.distractors) == 3
        for d in q.distractors:
            assert not grade("ordered_list", q, d).correct, (tier, seed, d)
            differing = sum(1 for a, b in zip(d.split(", "), q.answer_text.split(", ")) if a != b)
            assert differing == 1
        assert len(set(q.answer)) >= 2
        # reversing the key is also wrong
        assert (
            not grade("ordered_list", q, ", ".join(map(str, reversed(q.answer)))).correct or q.answer == q.answer[::-1]
        )


def test_tiers_differ_in_length_and_content():
    lengths = {t: {len(_make(t, s).answer) for s in SEEDS} for t in ("easy", "medium", "hard")}
    assert lengths["easy"] == {3}
    assert lengths["medium"] <= {3, 4} and 4 in lengths["medium"]
    assert lengths["hard"] == {5}
    assert all("≅" in _make("hard", s).text for s in SEEDS)
    assert not any("≅" in _make("easy", s).text for s in SEEDS)


def test_numbering_is_reshuffled_between_questions():
    for tier in ("easy", "medium", "hard"):
        keys = {tuple(_make(tier, s).answer) for s in SEEDS}
        assert len(keys) >= 12, tier


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_determinism(tier):
    assert _make(tier, 4).to_json() == _make(tier, 4).to_json()


def _apply(text: str) -> int:
    """The numeric answer to the exam's last part, worked from the text."""
    last = text.split("(d) ", 1)[1]
    nums = [int(n) for n in re.findall(r"(\d+)", last.split("find")[0])]
    if "find ∠" in last:
        if "and ∠" in last:  # two angles given
            if "180° − 2" in text:
                return 180 - 2 * nums[0]
            third = re.search(r"find ∠([A-Z]{3})", last).group(1)
            if "is extended to" in text:
                return nums[0] + nums[1]
            return 180 - nums[0] - nums[1]
        if "180° − 2" in text:
            return 180 - 2 * nums[0]
        if "= ∠" in text.split("To prove:")[1].split(". Step 1")[0] and "parallel" in text:
            return nums[0]
        return 180 - nums[0]
    if "perpendicular" in text:
        base, h = nums
        return int(math.hypot(base / 2, h))
    if "midpoint of" in text:
        hyp, base = nums
        return int(math.sqrt(hyp**2 - (base / 2) ** 2))
    return nums[0] // 2  # half of a diagonal


def test_exam_chain():
    src = SOURCES["exam"]
    seen_templates = set()
    for seed in SEEDS:
        q = make(src.topic, seed=seed, **(src.params or {}))
        assert [p.answer_kind for p in q.parts] == ["mcq", "mcq", "mcq", "number"]
        assert [p.marks for p in q.parts] == [1, 1, 1, 2]
        for p in q.parts[:3]:
            assert len(p.distractors) >= 5 and p.answer_text not in p.distractors
            assert len(set(p.distractors)) == len(p.distractors)
        assert int(q.parts[3].answer_text) == _apply(q.text), (seed, q.text)
        full = {p.label: p.answer_text for p in q.parts}
        result = grade_skill(_SKILL, q, full)
        assert result.correct and result.marks == result.max_marks == 5
        for p in q.parts:
            for d in p.distractors:
                assert grade_skill(_SKILL, q, {**full, p.label: d}).parts[p.label].marks == 0, (seed, p.label, d)
        seen_templates.add(q.text.split("To prove: ")[1].split(". Step 1")[0][:6])
    assert len(seen_templates) >= 3


def test_exam_reasons_in_parts_are_the_ones_the_proof_needs():
    src = SOURCES["exam"]
    for seed in SEEDS:
        q = make(src.topic, seed=seed, **(src.params or {}))
        expected = _expected(q.text)
        for p in q.parts[:3]:
            step = int(re.search(r"justifies step (\d)", p.prompt).group(1))
            assert expected[step - 1] in p.answer_text, (seed, p.prompt, p.answer_text)
