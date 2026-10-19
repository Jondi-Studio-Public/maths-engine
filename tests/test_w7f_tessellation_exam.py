"""Exam chain for ``f10-tessellation-vertex-angles`` (the w7f work).

The oracle reads the two polygons named in the prompt and works out the angles with the exterior-angle rule, which
the generator uses only as its second route.
"""

from __future__ import annotations

import re
from fractions import Fraction

import mathgen.generators.w7f_tessellation_exam  # noqa: F401  (registers the topic)
from tests import w7b_support as sup

SKILL = "f10-tessellation-vertex-angles"
TOPIC = "exd_x_w7f_tessellation_chain"
NAMES = {"equilateral triangle": 3, "square": 4, "regular pentagon": 5, "regular hexagon": 6}
NAMES |= {"regular octagon": 8, "regular nonagon": 9, "regular decagon": 10, "regular dodecagon": 12}


def _sides(phrase: str) -> int:
    m = re.fullmatch(r"regular polygon with (\d+) sides", phrase)
    return int(m.group(1)) if m else NAMES[phrase]


def _key(text: str) -> str:
    stem = text.split("(a)")[0]
    m = re.search(r"corner point an? (.+?), an? (.+?) and one more", stem)
    a, b = _sides(m.group(1)), _sides(m.group(2))
    ia, ib = 180 - Fraction(360, a), 180 - Fraction(360, b)
    left = 360 - ia - ib
    c = Fraction(360) / (180 - left)
    return f"(a) {ia}; (b) {ib}; (c) {left}; (d) {c}"


def test_chain():
    skill = sup.skill_of(SKILL)
    sup.check_wiring(skill, "fr2a_tessellation_vertex", TOPIC, "number")
    steps = [skill.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and steps[-1] > steps[0]
    sup.check_exam_chain(skill, TOPIC, [1, 1, 2, 2], ["number"] * 4)
    answers = set()
    for seed in range(60):
        q = sup.exam(TOPIC, seed)
        assert _key(q.text) == q.answer_text, (q.text, q.answer_text)
        answers.add(q.answer_text)
    assert len(answers) >= 12


def test_follow_through():
    skill = sup.skill_of(SKILL)

    def wrong_a(q, canon):
        # a wrong (a): (c) is then the right reply to it, and must still earn its marks
        bad = int(canon["a"]) + 12
        left = 360 - bad - int(canon["b"])
        if left == int(canon["b"]):
            return None  # a follow-through reply that repeats an earlier reply is not credited
        return {"a": str(bad), "c": str(left)}

    def wrong_c(q, canon):
        # a wrong (c): (d) is then the right reply to it
        left = int(canon["c"]) + 10
        if left >= 180 or 360 % (180 - left):
            return None
        return {"c": str(left), "d": str(360 // (180 - left))}

    assert sup.check_follow_through(skill, TOPIC, "c", wrong_a, seeds=40) > 0
    assert sup.check_follow_through(skill, TOPIC, "d", wrong_c, seeds=60) > 0
