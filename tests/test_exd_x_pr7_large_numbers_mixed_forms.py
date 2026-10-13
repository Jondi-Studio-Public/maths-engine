"""Exam tier of `f10-large-number-naming`: convert visitor counts written in different forms, then order them."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_pr7_large_numbers_mixed_forms"
SKILL_ID = "f10-large-number-naming"
HARD = ("large_number_ordering", {"digit_lengths": [5, 6, 7]})
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _value(item: str) -> int:
    item = item.strip()
    if " thousand and " in item:
        lead, rest = item.split(" thousand and ")
        return int(lead) * 1000 + int(rest)
    if item.endswith(" million"):
        return int(Fraction(item[: -len(" million")]) * 1_000_000)
    if "+" in item:
        return sum(int(p.replace(",", "")) for p in item.split("+"))
    return int(item.replace(",", ""))


def oracle(q):
    body = re.sub(r"^.*?(?:ways|forms)\. ", "", q.text)
    body = re.sub(r"\. (?:Write|Convert).*$", "", body)
    return sorted(_value(item) for item in body.split(";"))


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC] and exam[0].stages == ["exam"]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[0], seed=s, **HARD[1])) for s in range(60)}
    assert len(exam) >= 3 and exam - hard


def test_key_matches_independent_derivation_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _q(seed)
        want = oracle(q)
        if len(want) != 5 or want != q.answer or q.answer_text != ", ".join(f"{v:,}" for v in want):
            bad.append((seed, q.text, q.answer))
    assert bad == []


def test_every_question_needs_a_conversion():
    for seed in range(100):
        q = _q(seed)
        body = re.sub(r"^.*?(?:ways|forms)\. ", "", q.text)
        assert sum(1 for item in body.split(";") if not re.fullmatch(r"\s*[\d,]+", item.split(". ")[0])) >= 3


def test_grader_accepts_key_and_rejects_every_distractor():
    for seed in range(150):
        q = _q(seed)
        assert grade(_SKILL, q, q.answer_text).correct
        assert len(q.distractors) >= 2
        for d in q.distractors:
            assert not grade(_SKILL, q, d).correct, (seed, d)


def test_distractors_tagged():
    ids = set()
    for seed in range(60):
        q = _q(seed)
        assert set(q.distractor_values) == set(q.distractors)
        ids |= set(q.option_notes.values())
    assert {"placeholder-zero-dropped", "reversed-order"} <= ids


def test_answer_space():
    assert len({_q(s).answer_text for s in range(300)}) >= 100
