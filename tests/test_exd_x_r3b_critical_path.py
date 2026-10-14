"""Tests the multi-part exam tier of `senior-use-the-critical-path-to-determine-the-minimum`."""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.engine import grade
from tests import exd_x_r3b_support as S

TOPIC = "exd_x_r3b_critical_path"
SKILL_ID = "senior-use-the-critical-path-to-determine-the-minimum"
_SKILL = S.skill(SKILL_ID)
_ACTIVITY = re.compile(r"Activity ([A-Z]): duration (\d+) days, (?:no predecessors|predecessors: ([A-Z, ]+))\.")


@pytest.fixture(scope="module")
def questions():
    return S.exam_questions(TOPIC, 300)


def _network(q):
    dur, preds = {}, {}
    for name, d, p in _ACTIVITY.findall(q.stem):
        dur[name] = int(d)
        preds[name] = [s.strip() for s in p.split(",")] if p else []
    return dur, preds


def _longest_through(dur, preds, activity):
    """Longest start-to-finish route through `activity`, by memoised recursion (no forward/backward pass)."""
    succ = {a: [b for b in preds if a in preds[b]] for a in preds}

    def before(a):
        return dur[a] + max((before(p) for p in preds[a]), default=0)

    def after(a):
        return max((dur[s] + after(s) for s in succ[a]), default=0)

    return before(activity) + after(activity)


def _completion(dur, preds):
    return max(_longest_through(dur, preds, a) for a in dur)


def test_source_wired():
    S.check_wiring(TOPIC, SKILL_ID)


def test_prompt_variety_and_new_shape_over_hard_tier():
    S.check_shapes(TOPIC, SKILL_ID)


def test_structure():
    S.check_structure(TOPIC, "abc", [2, 2, 2])


def test_keys_match_independent_route_lengths_over_300_seeds(questions):
    for q in questions:
        dur, preds = _network(q)
        finish = _completion(dur, preds)
        assert int(q.parts[0].answer_text) == finish, q.stem
        x = re.search(r"float of activity ([A-Z])", q.parts[1].prompt).group(1)
        assert int(q.parts[1].answer_text) == finish - _longest_through(dur, preds, x), q.stem
        y, cut = re.search(r"Activity ([A-Z]) is shortened by (\d+) days", q.parts[2].prompt).groups()
        new = dict(dur, **{y: dur[y] - int(cut)})
        assert min(new.values()) >= 2
        new_finish = _completion(new, preds)
        assert int(q.parts[2].answer_text) == new_finish, q.stem
        assert 0 < finish - new_finish < int(cut), "the saving must be smaller than the reduction"


def test_grading_accepts_key_and_rejects_every_distractor():
    S.check_grading(TOPIC, SKILL_ID, 120)


def test_distractors_are_tagged_and_formatted():
    S.check_tags(TOPIC)


def test_follow_through_from_a_wrong_completion_time(questions):
    checked = 0
    for q in questions[:100]:
        dur, preds = _network(q)
        finish = int(q.parts[0].answer_text)
        wrong = int(q.parts[0].distractors[0])  # every sum of all durations is larger than the completion time
        x = re.search(r"float of activity ([A-Z])", q.parts[1].prompt).group(1)
        through = _longest_through(dur, preds, x)
        saving = finish - int(q.parts[2].answer_text)
        reply = {"a": str(wrong), "b": str(wrong - through), "c": str(wrong - saving)}
        if len({reply["a"], reply["b"], reply["c"]}) < 3:
            continue
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (q.stem, reply)
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (q.stem, reply)
        checked += 1
    assert checked >= 40


def test_all_wrong_earns_zero(questions):
    """Replies that are wrong for the key and also one off the follow-through value earn nothing."""
    for q in questions[:40]:
        dur, preds = _network(q)
        x = re.search(r"float of activity ([A-Z])", q.parts[1].prompt).group(1)
        through = _longest_through(dur, preds, x)
        saving = int(q.parts[0].answer_text) - int(q.parts[2].answer_text)
        reply = {"a": "101", "b": str(101 - through + 1), "c": str(101 - saving + 1)}
        result = grade(_SKILL, q, reply)
        assert result.marks == 0 and not result.correct, (q.stem, reply)


def test_answer_space_over_300_seeds():
    S.check_answer_space(TOPIC)


def test_clean_and_deterministic():
    S.check_clean_and_deterministic(TOPIC)
    assert make(TOPIC, seed=7, difficulty="exam").to_dict() == make(TOPIC, seed=7, difficulty="exam").to_dict()
