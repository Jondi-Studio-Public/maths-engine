"""Exam tier of `f10-integer-numberline-coordinates`: work out final temperatures, then order the towns."""

from __future__ import annotations

import re

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_pr7_signed_temperatures_order"
SKILL_ID = "f10-integer-numberline-coordinates"
HARD = ("integer_numberline_order", {"count": 5, "sign_mix": "mixed_trap"})
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_LINE = re.compile(r"([A-Z][a-z]+): (-?\d+) °C, then (rose|fell) by (\d+) °C\.")


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def oracle(q):
    rows = [(n, int(s) + (1 if v == "rose" else -1) * int(c)) for n, s, v, c in _LINE.findall(q.text)]
    return [n for n, _ in sorted(rows, key=lambda r: r[1])], len(rows)


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
        names, count = oracle(q)
        if count not in (4, 5) or names != q.answer or q.answer_text != ", ".join(names):
            bad.append((seed, q.text, q.answer))
    assert bad == []


def test_ties_never_possible():
    for seed in range(300):
        q = _q(seed)
        finals = [int(s) + (1 if v == "rose" else -1) * int(c) for _, s, v, c in _LINE.findall(q.text)]
        assert len(set(finals)) == len(finals)


def test_grader_accepts_key_and_rejects_every_distractor():
    for seed in range(150):
        q = _q(seed)
        assert q.answer_kind == "ordered_list"
        assert grade(_SKILL, q, q.answer_text).correct
        assert len(q.distractors) >= 2
        for d in q.distractors:
            assert not grade(_SKILL, q, d).correct, (seed, d)


def test_distractors_tagged_and_matched():
    for seed in range(60):
        q = _q(seed)
        assert set(q.distractor_values) == set(q.distractors)
        d = q.distractors[0]
        result = grade(_SKILL, q, d)
        assert not result.correct and result.matched_distractor == d


def test_answer_space():
    answers = {_q(s).answer_text for s in range(300)}
    assert len(answers) >= 100
