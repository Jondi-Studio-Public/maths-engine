"""Tests the exam tier of `cantilever-beam-reactions`: reaction, fixing moment, then a section moment."""

from __future__ import annotations

import re

import pytest
import sympy

import mathgen.generators.exd_x_u7_cantilever_chain  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u7_cantilever_chain"
SKILL_ID = "cantilever-beam-reactions"
SEEDS = range(320)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def oracle(q) -> dict[str, int]:
    """Re-derive every part from the stem alone, by integrating the loads beyond each section."""
    stem = q.text.split(" (a)")[0]
    L, w, P = (int(v) for v in re.findall(r"\d+", stem)[:3])
    d = int(re.search(r"section (\d+) m from the wall", q.text).group(1))
    s = sympy.symbols("s")

    def moment(length):  # moment of the loads over the last `length` metres, about the section
        return int(sympy.integrate(w * s, (s, 0, length)) + P * length)

    return {"a": w * L + P, "b": moment(L), "c": moment(L - d)}


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired_last():
    src = _SKILL.sources[-1]
    assert (src.topic, src.difficulty, src.stages, src.params) == (TOPIC, "exam", ["exam"], {"tier": "exam"})


def test_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]
        assert find_text_defects(q) == []


def test_prompt_shapes_beyond_the_easy_tier():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    easy = {prompt_shape(make("structural", seed=s, kinds=["cantilever_udl"])) for s in range(60)}
    assert len(exam) >= 3 and exam - easy


def test_key_matches_independent_derivation_over_320_seeds():
    bad = [s for s in SEEDS if {k: str(v) for k, v in oracle(_exam(s)).items()} != _canonical(_exam(s))]
    assert bad == []


def test_key_accepted_and_every_distractor_rejected():
    for seed in range(120):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 6
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_fixing_moment():
    checked = 0
    for seed in range(250):
        q = _exam(seed)
        stem = q.text.split(" (a)")[0]
        L, w, P = (int(v) for v in re.findall(r"\d+", stem)[:3])
        d = int(re.search(r"section (\d+) m from the wall", q.text).group(1))
        reaction = w * L + P
        b_wrong = int(q.parts[1].distractors[0])
        c_ft = b_wrong - reaction * d + w * d * d // 2
        if c_ft <= 0 or c_ft in (reaction, b_wrong):  # repeating an earlier reply is not following through
            continue
        result = grade(_SKILL, q, {"a": str(reaction), "b": str(b_wrong), "c": str(c_ft)})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True
        checked += 1
    assert checked >= 50


def test_follow_through_from_a_wrong_reaction():
    checked = 0
    for seed in range(120):
        q = _exam(seed)
        stem = q.text.split(" (a)")[0]
        L, w, P = (int(v) for v in re.findall(r"\d+", stem)[:3])
        d = int(re.search(r"section (\d+) m from the wall", q.text).group(1))
        fixing = oracle(q)["b"]
        a_wrong = int(q.parts[0].distractors[0])
        c_ft = fixing - a_wrong * d + w * d * d // 2
        if c_ft <= 0 or c_ft in (oracle(q)["c"], fixing, a_wrong):
            continue
        result = grade(_SKILL, q, {"a": str(a_wrong), "b": str(fixing), "c": str(c_ft)})
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True
        checked += 1
    assert checked >= 40


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()


@pytest.mark.parametrize("seed", [0, 1])
def test_plausible_magnitudes(seed):
    q = _exam(seed)
    assert all(1 <= int(p.answer_text) <= 800 for p in q.parts)
