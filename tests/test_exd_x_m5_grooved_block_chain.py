"""Exam source of `senior-use-addition-of-the-area-of-the-faces`: a block with a channel, face by face."""

from __future__ import annotations

import re
from fractions import Fraction

import _exd_m5_common as c

TOPIC = "exd_x_m5_grooved_block_chain"
SKILL_ID = "senior-use-addition-of-the-area-of-the-faces"


def oracle(q):
    t = q.stem + " " + q.parts[2].prompt
    nums = re.findall(r"(\d+) cm", q.stem)
    length, width, height, cw, cd = (int(n) for n in nums[:5])
    rate = Fraction(re.search(r"\$(\d+\.\d\d)", t).group(1))
    # Face by face: two ends, two sides, the top as two shoulders, and the channel's floor and two walls.
    end = width * height - cw * cd
    faces = [
        2 * end,
        2 * height * length,
        width * length,  # underneath
        (width - cw) * length,  # the two shoulders of the top together
        cw * length,  # channel floor
        2 * cd * length,  # channel walls
    ]
    total = sum(faces)
    return {"a": str(end), "b": str(total), "c": c.show(Fraction(total) * rate / 10_000, 2)}


def test_key_rederived_from_prompt():
    c.check_oracle(TOPIC, oracle)


def test_answer_space():
    for label in "abc":
        c.check_answer_space(TOPIC, label)


def test_grader_accepts_key_and_rejects_distractors():
    c.check_grading(TOPIC, SKILL_ID)


def test_follow_through_from_wrong_end_area():
    c.check_follow_through(TOPIC, SKILL_ID)


def test_all_wrong_and_deterministic():
    c.check_all_wrong(TOPIC, SKILL_ID)
    c.check_deterministic(TOPIC)
