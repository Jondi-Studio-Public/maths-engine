"""`f10-capacity-mass-informal-units`: topic `fr2c_informal_capacity_mass` (easy, medium, hard).

The oracle re-reads the counts from the question text and works the answer out again.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from tests.fr2c_checks import check_single, skill

SKILL = "f10-capacity-mass-informal-units"
TOPIC = "fr2c_informal_capacity_mass"
TIERS = ("easy", "medium", "hard")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _nums(text):
    return [int(n) for n in re.findall(r"\d+", text)]


def test_skill_wiring():
    sk = skill(SKILL)
    assert sk.answer == "number" and sk.status == "active"
    assert [(s.topic, s.difficulty) for s in sk.sources] == [(TOPIC, t) for t in TIERS]


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single(SKILL, TOPIC, tier)


def test_easy_is_one_comparison_or_total():
    kinds = set()
    for seed in range(80):
        q = _q("easy", seed)
        big, small = _nums(q.text)[:2]
        if "How many more" in q.text:
            kinds.add("difference")
            assert int(q.answer_text) == big - small, q.text
        else:
            kinds.add("total")
            assert int(q.answer_text) == big + small, q.text
    assert kinds == {"difference", "total"}


def test_medium_two_steps():
    seen = set()
    for seed in range(120):
        q = _q("medium", seed)
        n = _nums(q.text)
        if "pours in" in q.text:
            seen.add("fill")
            assert int(q.answer_text) == n[0] - n[1] - n[2], q.text
        elif (
            "more than the" in q.text
            and "together" in q.text
            and "balances" in q.text
            and "How many more" not in q.text
        ):
            seen.add("more_than")
            assert int(q.answer_text) == n[0] + (n[0] + n[1]), q.text
        else:
            seen.add("two_items")
            assert int(q.answer_text) == n[0] - n[1] - n[2], q.text
    assert seen == {"fill", "more_than", "two_items"}


def test_hard_needs_the_size_of_the_big_unit():
    seen = set()
    for seed in range(120):
        q = _q("hard", seed)
        n = _nums(q.text)
        k = n[0]
        if "altogether" in q.text:
            seen.add("leftover")
            big, left = n[1], n[2]
            assert int(q.answer_text) == big * k + left, q.text
        else:
            seen.add("compare")
            big, small = n[1], n[2]
            assert int(q.answer_text) == small - big * k, q.text
    assert seen == {"leftover", "compare"}


def test_tiers_differ_and_prompts_vary():
    pools = {t: {_q(t, s).text for s in range(60)} for t in TIERS}
    assert len({frozenset(p) for p in pools.values()}) == 3
    for t in TIERS:
        assert len(pools[t]) >= 20
