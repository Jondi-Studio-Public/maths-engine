"""Tests for the exam source of senior-use-bernoulli-random-variables-and-associated-probabilities-to.

The exam tier is a three-part chain: success probability from the stated failure probability, an ordered run of
trials, and that run extended by one more trial. Every key is re-derived from the rendered prompt with exact fractions.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2l_bernoulli_chain"
SKILL_ID = "senior-use-bernoulli-random-variables-and-associated-probabilities-to"
SEEDS = range(320)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)

_FAIL_PHRASES = ("misses", "is late", "fails to germinate", "runs late", "is rejected")
_SUCCESS_PHRASES = ("scores", "arrives on time", "germinates", "runs on time", "passes inspection")
_PHRASE_RE = re.compile("|".join(re.escape(p) for p in sorted(_FAIL_PHRASES + _SUCCESS_PHRASES, key=len, reverse=True)))


def _outcomes(text: str) -> str:
    """'SFS' for a prompt listing the outcomes in order."""
    return "".join("F" if m.group(0) in _FAIL_PHRASES else "S" for m in _PHRASE_RE.finditer(text))


def oracle(q) -> dict[str, str]:
    num, den = re.search(r"probability (\d+)/(\d+)", q.text).groups()
    fail = Fraction(int(num), int(den))
    p = 1 - fail

    def run(seq):
        out = Fraction(1)
        for o in seq:
            out *= p if o == "S" else 1 - p
        return out

    seq = _outcomes(q.parts[1].prompt.split("Give")[0])
    extra = _outcomes(q.parts[2].prompt.split("Give")[0].split("(b)")[-1])
    assert len(extra) == 1, q.parts[2].prompt
    show = lambda v: f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)  # noqa: E731
    return {"a": show(p), "b": show(run(seq)), "c": show(run(seq + extra))}


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages, s.params) for s in exam] == [(TOPIC, ["exam"], {"tier": "exam"})]


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)


def test_oracle_agrees_over_300_seeds(questions):
    bad = [(q.text, oracle(q), _canonical(q)) for q in questions if oracle(q) != _canonical(q)]
    assert bad == []


def test_composed_answer_full_marks(questions):
    for q in questions[:120]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5


def test_every_distractor_rejected_and_tagged(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) == 3 and len(set(p.distractors)) == 3
            assert all(p.option_notes.get(d) in CATALOGUE for d in p.distractors), (p.label, p.option_notes)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_success_probability(questions):
    checked = 0
    for q in questions[:150]:
        # the swapped distractor is the failure probability itself, a value inside (0, 1)
        wrong = q.parts[0].distractors[0]
        assert q.parts[0].option_notes[wrong] == "swapped-success-failure"
        p_wrong = Fraction(wrong)
        seq = _outcomes(q.parts[1].prompt.split("Give")[0])
        extra = _outcomes(q.parts[2].prompt.split("Give")[0].split("(b)")[-1])
        b = Fraction(1)
        for o in seq:
            b *= p_wrong if o == "S" else 1 - p_wrong
        c = b * (p_wrong if extra == "S" else 1 - p_wrong)
        show = lambda v: f"{v.numerator}/{v.denominator}"  # noqa: E731
        result = grade(_SKILL, q, {"a": wrong, "b": show(b), "c": show(c)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2
        if show(b) != q.parts[1].answer_text:  # equal numbers of successes and failures coincide with the key
            assert result.parts["b"].follow_through is True
        assert result.parts["c"].marks == 2
        if show(c) != q.parts[2].answer_text:  # a swapped pair can coincide with the key by symmetry
            assert result.parts["c"].follow_through is True
        assert result.marks == 4 and not result.correct
        checked += 1
    assert checked == 150


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "999/1000", "b": "1/1000", "c": "3/1000"})
        assert result.marks == 0


def test_answer_space(questions):
    tuples = Counter(tuple(p.answer_text for p in q.parts) for q in questions[:300])
    assert len(tuples) >= 100, len(tuples)
    assert max(tuples.values()) <= 15, tuples.most_common(3)


def test_no_text_defects_and_shapes():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard = {prompt_shape(make("p12d_bernoulli_sequence", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3 and exam - hard
    for s in range(60):
        assert find_text_defects(make(TOPIC, seed=s)) == []


def test_deterministic():
    assert make(TOPIC, seed=7).to_dict() == make(TOPIC, seed=7).to_dict()
