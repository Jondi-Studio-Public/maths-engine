"""`f10-ampm-elapsed-and-24h-practical`: topic `fr2c_clock_elapsed` (easy, medium, hard) and the exam chain.

The oracle reads every time out of the question text, changes it to minutes since midnight and works the answer again.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from tests.fr2c_checks import check_follow_through, check_parts, check_single, skill

SKILL = "f10-ampm-elapsed-and-24h-practical"
TOPIC = "fr2c_clock_elapsed"
EXAM = "fr2c_clock_elapsed_exam"
TIERS = ("easy", "medium", "hard")
TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?: (am|pm))?")


def _times(text):
    out = []
    for h, m, ap in TIME.findall(text):
        h, m = int(h), int(m)
        if ap == "pm" and h != 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        out.append((h * 60 + m, bool(ap)))
    return out


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def test_skill_wiring():
    sk = skill(SKILL)
    assert sk.answer == "number" and sk.status == "active"
    assert [(s.topic, s.difficulty) for s in sk.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single(SKILL, TOPIC, tier)


def test_easy_crosses_noon_with_am_pm_times():
    for seed in range(100):
        q = _q("easy", seed)
        (s, a), (e, b) = _times(q.text)
        assert a and b and s < 12 * 60 < e < 24 * 60
        assert int(q.answer_text) == e - s, q.text


def test_medium_mixes_the_two_styles():
    styles = set()
    for seed in range(100):
        q = _q("medium", seed)
        (s, a), (e, b) = _times(q.text)
        assert a != b, q.text
        styles.add(a)
        assert int(q.answer_text) == e - s


def test_hard_total_includes_the_wait():
    for seed in range(100):
        q = _q("hard", seed)
        (s1, _), (e1, _), (s2, _), (e2, _) = _times(q.text)
        assert s1 < e1 < s2 < e2 < 24 * 60
        assert int(q.answer_text) == e2 - s1, q.text
        assert int(q.answer_text) > (e1 - s1) + (e2 - s2)


def test_nothing_crosses_midnight():
    for tier in TIERS:
        for seed in range(100):
            t = [m for m, _ in _times(_q(tier, seed).text)]
            assert t == sorted(t) and t[-1] < 24 * 60


def test_exam_chain_shape():
    check_parts(
        SKILL, EXAM, marks=[2, 1, 1, 2, 1], kinds=["number"] * 4 + ["pair"], follow=[False, False, False, True, False]
    )


def test_exam_answers_recomputed_from_the_stem():
    for seed in range(80):
        q = make(EXAM, seed=seed, difficulty="exam")
        (s1, _), (e1, _), (s2, _), (e2, _) = _times(q.text.split("(a)")[0])
        p = {x.label: x for x in q.parts}
        assert int(p["a"].answer_text) == e1 - s1
        assert int(p["b"].answer_text) == s2 - e1
        assert int(p["c"].answer_text) == e2 - s2
        assert int(p["d"].answer_text) == e2 - s1
        h, m = divmod(e2 - s1, 60)
        assert p["e"].answer == {"x": h, "y": m}
        assert p["d"].follow_through == "part_a+part_b+part_c"


def test_exam_follow_through_from_a_wrong_leg():
    def replace(q, parts):
        wrong = parts["a"].distractors[0]
        b, c = int(parts["b"].answer_text), int(parts["c"].answer_text)
        return wrong, str(int(wrong) + b + c)

    check_follow_through(SKILL, EXAM, "a", replace, "d")
