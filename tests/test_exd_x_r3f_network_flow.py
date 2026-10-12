"""Exam chain of `senior-solve-small-scale-network-flow-problems-including-the`: cut, flow, upgrade, increase."""

from __future__ import annotations

import itertools
import re

import pytest

from exd_x_r3f_support import (
    assert_all_wrong_earns_nothing,
    assert_answer_space,
    assert_clean_and_deterministic,
    assert_follow_through,
    assert_full_marks_and_distractors,
    assert_shapes,
    assert_wired,
    exam_question,
)

TOPIC = "exd_x_r3f_network_flow"
SKILL_ID = "senior-solve-small-scale-network-flow-problems-including-the"


def read_network(q):
    caps = {(u, v): int(c) for u, v, c in re.findall(r"Pipe from (\w) to (\w): capacity (\d+) kL", q.stem)}
    assert len(caps) == 8
    sets = re.search(r"separates (.+?) from (.+?), in kL", q.parts[0].prompt)
    source_side = set(re.findall(r"[A-Z]", sets.group(1)))
    upgrade = re.search(r"pipe from (\w) to (\w) is replaced by one with capacity (\d+) kL", q.parts[2].prompt)
    return caps, source_side, (upgrade.group(1), upgrade.group(2)), int(upgrade.group(3))


def min_cut(caps):
    """Smallest cut capacity over every split of the junctions: the max-flow min-cut theorem as the oracle."""
    best = None
    for picks in itertools.product((0, 1), repeat=4):
        side = {"S"} | {n for n, p in zip("ABCD", picks) if p}
        value = sum(c for (u, v), c in caps.items() if u in side and v not in side)
        best = value if best is None else min(best, value)
    return best


def oracle(q):
    caps, side, edge, bump = read_network(q)
    cut = sum(c for (u, v), c in caps.items() if u in side and v not in side)
    base = min_cut(caps)
    upgraded = dict(caps)
    upgraded[edge] += bump
    after = min_cut(upgraded)
    return cut, base, after, after - base


@pytest.fixture(scope="module")
def questions():
    return [exam_question(TOPIC, s) for s in range(300)]


def test_source_wired():
    assert_wired(SKILL_ID, TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 2, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompts_and_determinism():
    assert_shapes(SKILL_ID, TOPIC)
    assert_clean_and_deterministic(TOPIC)


def test_key_rederived_from_rendered_prompt(questions):
    for q in questions:
        assert tuple(int(p.answer_text) for p in q.parts) == oracle(q), q.text


def test_upgrade_always_matters_so_no_two_parts_share_an_answer(questions):
    """An upgrade that changed nothing would make (c) equal (b) and (d) zero (integrator, free-marks audit)."""
    increases = [int(q.parts[3].answer_text) for q in questions]
    assert min(increases) > 0
    assert len(set(increases)) >= 6
    for q in questions:
        answers = [p.answer_text for p in q.parts]
        assert len(set(answers)) == len(answers), q.text


def test_grader_accepts_key_and_rejects_distractors():
    assert_full_marks_and_distractors(SKILL_ID, TOPIC)


def test_answer_space():
    assert_answer_space(TOPIC)


def _wrong_flow_replies(q):
    cut, base, after, _inc = oracle(q)
    base_wrong = int(q.parts[1].distractors[0])
    return {"a": str(cut), "b": str(base_wrong), "c": str(after), "d": str(after - base_wrong)}


def test_follow_through_from_a_wrong_maximum_flow():
    def usable(q):
        replies = _wrong_flow_replies(q)
        return replies if int(replies["d"]) != 0 and int(replies["b"]) != int(replies["c"]) else None

    assert_follow_through(SKILL_ID, TOPIC, usable, ["b"], ["d"])


def test_all_wrong_earns_nothing():
    assert_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "991", "b": "992", "c": "993", "d": "994"})
