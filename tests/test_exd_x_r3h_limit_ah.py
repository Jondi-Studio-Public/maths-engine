"""Tests for the exam tier `exd_x_r3h_limit_ah` (senior-estimate-the-limit-of-frac-a-h-1)."""

from __future__ import annotations

import math
import re

from mathgen.engine import grade
from tests import exd_x_r3h_support as sup

TOPIC = "exd_x_r3h_limit_ah"
SKILL = "senior-estimate-the-limit-of-frac-a-h-1"


def oracle(q):
    a = float(re.search(r"([\d.]+)\^x", q.stem).group(1))
    c = sup.first_int(r"\bx = (\d)\b", q.parts[1].prompt)
    limit = float(sup.rnd(math.log(a), 2))
    return {
        "a": sup.rnd(limit, 2),
        "b": sup.rnd(limit * a**c, 2),
        "c": sup.rnd(c - 1 / limit, 1),
    }


def test_source_wired_and_shapes():
    assert sup.exam_source(SKILL).topic == TOPIC
    sup.shape_check(SKILL, TOPIC)


def test_table_rows_are_the_difference_quotients_of_the_stem():
    for seed in range(100):
        q = sup.draw(TOPIC, seed)
        a = float(re.search(r"([\d.]+)\^x", q.stem).group(1))
        rows = re.findall(r"h = ([\d.]+): (-?\d+\.\d{4})", q.stem)
        assert len(rows) == 4
        for h, shown in rows:
            assert abs(((a ** float(h)) - 1) / float(h) - float(shown)) < 6e-5


def test_keys_rederived_from_the_rendered_prompt():
    sup.run_oracle(TOPIC, oracle)


def test_answer_space():
    sup.answer_space(TOPIC)


def test_key_accepted_and_every_distractor_rejected():
    sup.full_marks_and_distractors(SKILL, TOPIC)


def test_follow_through_from_a_wrong_estimate():
    sup.follow_through_check(SKILL, TOPIC)


def test_explicit_follow_through_values():
    skill = sup.skill(SKILL)
    for seed in range(40):
        q = sup.draw(TOPIC, seed)
        a = float(re.search(r"([\d.]+)\^x", q.stem).group(1))
        c = sup.first_int(r"\bx = (\d)\b", q.parts[1].prompt)
        mine = float(q.parts[0].answer_text) + 0.07
        reply = {"a": sup.rnd(mine, 2), "b": sup.rnd(mine * a**c, 2), "c": sup.rnd(c - 1 / mine, 1)}
        if reply["b"] == reply["a"] or reply["c"] in (reply["a"], reply["b"]) or float(reply["c"]) == 0:
            continue  # a reply that merely repeats an earlier one is not follow-through
        result = grade(skill, q, reply)
        assert result.parts["b"].marks == q.parts[1].marks, (seed, reply)
        assert result.parts["c"].marks == q.parts[2].marks, (seed, reply)
