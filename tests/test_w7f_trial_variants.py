"""Coin, die, card and tile experiments for ``f10-trial-tallies-compare-outcomes`` (the w7f work).

The oracle parses the tally text in the prompt back into counts, never calling the generator's helpers.
"""

from __future__ import annotations

import re

import mathgen.generators.w7f_trial_variants  # noqa: F401  (registers the topic)
from mathgen import make
from tests import w7b_support as sup

TOPIC = "w7f_trial_variants"


def _count(marks: str) -> int:
    return sum(5 if g == "||||/" else len(g) for g in marks.split())


def _table(text: str) -> dict[str, int]:
    return {m.group(1): _count(m.group(2)) for m in re.finditer(r"(?:^|[:.] |; )([\w ]+): ([|/ ]+)", text)}


def _oracle(text: str) -> int:
    if "Group 1:" in text:
        g1 = re.search(r"Group 1: (.*?)\. Group 2: (.*?)\. ", text)
        a, b = _table(". " + g1.group(1)), _table(". " + g1.group(2))
        pooled = {k: a[k] + b[k] for k in a}
        if "more" in text.split("Group 2:")[1]:
            xs = re.search(r"more (?:times did |results were )(\S+(?: \S+)?) (?:come up )?than (\S+)\?", text)
            x = xs.group(1).replace(" come up", "")
            y = xs.group(2)
            return pooled[x] - pooled[y]
        if "not" in text.split("Group 2:")[1]:
            return sum(pooled.values()) - max(pooled.values())
        word = re.search(r"(?:did (\S+) come up|were (\S+)\?)", text)
        x = word.group(1) or word.group(2)
        return pooled[x]
    shown = re.search(r"The tally: (.*?)\. (?:How|In)", text).group(1)
    counts = _table(". " + shown)
    x = re.search(r"(?:did not give|other than) (\w+)", text).group(1)
    return sum(counts.values()) - counts[x]


def test_wiring_and_oracle():
    skill = sup.skill_of("f10-trial-tallies-compare-outcomes")
    assert [s.topic for s in skill.sources].count(TOPIC) == 3
    for tier, floor in (("easy", 12), ("medium", 12), ("hard", 12)):
        answers = set()
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert str(_oracle(q.text)) == q.answer_text, q.text
            answers.add(q.answer_text)
        assert len(answers) >= floor, (tier, len(answers))


def test_single_tier_contract():
    skill = sup.skill_of("f10-trial-tallies-compare-outcomes")
    for tier in ("easy", "medium", "hard"):
        sup.check_single_tier(skill, TOPIC, tier, seeds=40)
