"""Checks for the multi-part exam source of `use-formulas`: speed, simple interest and electricity chains.

(a) several prompt shapes, some absent from the hard tier; (b) every part's key is re-derived from the RENDERED
PROMPT with exact fractions over 300 seeds; (c) the skill's grader accepts the key and rejects every distractor;
(d) a wrong earlier answer used correctly in a later part still earns that part's marks.
"""

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_sc4_formula_chain"
SKILL = next(s for s in load_skills()[1] if s.id == "use-formulas")
SEEDS = range(300)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


@pytest.fixture(scope="module")
def questions():
    return [_q(s) for s in SEEDS]


def _num(text):
    return Fraction(text)


def _round(value, dp):
    scaled = value * 10**dp
    return Fraction(math.floor(abs(scaled) + Fraction(1, 2)), 10**dp)


def _show(value, dp):
    value = _round(value, dp)
    return str(int(value)) if dp == 0 else f"{float(value):.{dp}f}"


def oracle(q):
    """The three canonical answer texts, derived from the stem and part prompts alone."""
    text = q.text
    if "km in" in text:
        d1, t1 = (_num(x) for x in re.search(r"(\d+(?:\.\d+)?) km in (\d+(?:\.\d+)?) hours", text).groups())
        t2 = _num(re.search(r"go in (\d+(?:\.\d+)?) hours", text).group(1))
        d3 = _num(re.search(r"to go (\d+(?:\.\d+)?) km", text).group(1))
        v = d1 / t1
        return [_show(v, 0), _show(v * t2, 1), _show(d3 / v, 1)]
    if "interest" in text.split("(a)")[0]:
        stem = text.split("(a)")[0]
        principal = _num(re.search(r"\$(\d+)", stem).group(1))
        rate = _num(re.search(r"(\d+(?:\.\d+)?)%", stem).group(1))
        term = int(re.search(r"(\d+) years", stem).group(1))
        target = _num(re.search(r"earn \$(\d+\.\d\d) in interest", text).group(1))
        interest = principal * rate * term / 100
        return [_show(interest, 2), _show(principal + interest, 2), str(int(target / (principal * rate / 100)))]
    watts = _num(re.search(r"(\d+) W", text).group(1))
    hours = _num(re.search(r"run for (\d+(?:\.\d+)?) hours", text).group(1))
    rate = _num(re.search(r"costs (\d+) cents", text).group(1))
    days = int(re.search(r"for (\d+) days", text).group(1))
    energy = watts * hours / 1000
    cost = energy * rate / 100
    return [_show(energy, 2), _show(cost, 2), _show(cost * days, 2)]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_q(seed)) == [], seed


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make("formula_plugin", seed=s, difficulty="hard", unknown="input")) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, (exam, hard)


def test_shape_and_wiring():
    exam = [s for s in SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    for seed in range(40):
        q = _q(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "number" and p.marks >= 1 for p in q.parts)


def test_deterministic():
    for seed in (0, 1, 7, 123):
        assert _q(seed).to_json() == _q(seed).to_json()


def test_keys_match_independent_derivation(questions):
    bad = [
        (s, q.text, oracle(q), _canonical(q))
        for s, q in zip(SEEDS, questions)
        if oracle(q) != list(_canonical(q).values())
    ]
    assert bad == [], bad[:2]


def test_grader_accepts_key_and_rejects_every_distractor(questions):
    bad = []
    for seed, q in zip(SEEDS, questions):
        if seed >= 150:
            break
        canon = _canonical(q)
        result = grade(SKILL, q, canon)
        if not (result.correct and result.marks == result.max_marks):
            bad.append(("key rejected", seed, canon))
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for d in p.distractors:
                if grade(SKILL, q, {**canon, p.label: d}).parts[p.label].marks != 0:
                    bad.append(("distractor accepted", seed, p.label, d))
    assert bad == [], bad[:3]


def test_distractors_are_tagged(questions):
    for q in questions[:100]:
        for p in q.parts:
            tagged = [d for d in p.distractors if p.option_notes.get(d)]
            assert tagged, (q.text, p.label)
            assert all(d in p.distractor_values for d in tagged)


def test_follow_through_from_a_wrong_first_answer(questions):
    checked = 0
    for q in questions[:150]:
        text = q.text
        a_wrong = q.parts[0].distractors[0]
        a = _num(a_wrong)
        if "km in" in text:
            t2 = _num(re.search(r"go in (\d+(?:\.\d+)?) hours", text).group(1))
            d3 = _num(re.search(r"to go (\d+(?:\.\d+)?) km", text).group(1))
            b, c = _show(a * t2, 1), _show(d3 / a, 1)
        elif "kWh" in text:
            rate = _num(re.search(r"costs (\d+) cents", text).group(1))
            days = int(re.search(r"for (\d+) days", text).group(1))
            cost = a * rate / 100
            b, c = _show(cost, 2), _show(cost * days, 2)
        else:
            stem = text.split("(a)")[0]
            principal = _num(re.search(r"\$(\d+)", stem).group(1))
            term = int(re.search(r"(\d+) years", stem).group(1))
            target = _num(re.search(r"earn \$(\d+\.\d\d) in interest", text).group(1))
            b, c = _show(principal + a, 2), _show(target * term / a, 0)
        result = grade(SKILL, q, {"a": a_wrong, "b": b, "c": c})
        assert result.parts["a"].marks == 0
        # A part is checked only when the learner's own chain gives a sensible value; the grader withholds the rest.
        for label in "bc":
            if result.parts[label].marks:
                assert result.parts[label].follow_through is True
        assert result.parts["b"].marks == result.parts["b"].max_marks, (q.text, a_wrong, b)
        checked += 1
    assert checked == 150


def test_all_wrong_earns_nothing(questions):
    for q in questions[:40]:
        result = grade(SKILL, q, {"a": "-9999", "b": "-5", "c": "123456"})
        assert result.marks == 0 and not result.correct
