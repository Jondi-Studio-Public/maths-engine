"""Turn settings for ``f10-angle-turn-right-reference`` (the w7f work): the oracle re-reads each prompt."""

from __future__ import annotations

import re

import mathgen.generators.w7f_turn_contexts  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.grading import grade
from tests import w7b_support as sup

TOPIC = "w7f_turn_contexts"
COMPASS = ["north", "east", "south", "west"]
CLOCK = [12, 3, 6, 9]
PART = {"a quarter turn": 1, "a half turn": 2, "a three-quarter turn": 3}


def _oracle(text: str) -> int:
    if "minute hand" in text:
        a, b = (int(n) for n in re.findall(r"at the (\d+)", text))
        return ((CLOCK.index(b) - CLOCK.index(a)) % 4) or 4
    if "robot faces" in text:
        a, b = (w for w in re.findall(r"\b(north|east|south|west)\b", text))
        return ((COMPASS.index(b) - COMPASS.index(a)) % 4) or 4
    if "altogether" in text:
        n = int(re.search(r"makes (\d+) full", text).group(1))
        part = next(v for k, v in PART.items() if k in text)
        return 4 * n + part
    if "anticlockwise" in text:
        n = int(re.search(r"through (\d+) full", text).group(1))
        part = next(v for k, v in PART.items() if k in text)
        back = int(re.search(r"through (\d+) right angles", text).group(1))
        return 4 * n + part - back
    r = int(re.search(r"through (\d+) right angles", text).group(1))
    return (-r) % 4


def test_wiring_and_tiers():
    skill = sup.skill_of("f10-angle-turn-right-reference")
    topics = [s.topic for s in skill.sources]
    assert topics.count(TOPIC) == 3
    for tier, floor in (("easy", 3), ("medium", 12), ("hard", 12)):
        answers = set()
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert str(_oracle(q.text)) == q.answer_text, q.text
            answers.add(q.answer_text)
            assert len(q.distractors) >= 3
            for d in q.distractors:
                assert not grade("number", q, d).correct
            for d, tag in q.option_notes.items():
                assert tag
        assert len(answers) >= floor, (tier, len(answers))


def test_no_python_notation():
    for tier in ("easy", "medium", "hard"):
        for seed in range(30):
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert "**" not in q.text and "*" not in q.text


def test_single_tier_contract():
    skill = sup.skill_of("f10-angle-turn-right-reference")
    for tier in ("easy", "medium", "hard"):
        sup.check_single_tier(skill, TOPIC, tier, seeds=40)
