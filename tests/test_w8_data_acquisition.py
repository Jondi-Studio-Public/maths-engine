"""Tests `acquire-and-check-data` (topic `w8_data_acquisition`) and its exam chain.

The hard-tier oracle reads the entries back out of the question TEXT and checks, by direct inspection, that exactly one
entry is impossible (not a whole number of things, not in the unit of the rest, above the maximum mark, or not an answer
to the question) and that the key is that entry.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.w8_data_acquisition_build as gen
from tests import w7b_support as sup

TOPIC, EXAM = gen.TOPIC, gen.EXAM_TOPIC
SKILL = sup.skill_of("acquire-and-check-data")


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM, "mcq")
    assert {"AC9M4ST01", "AC9M5ST01"} <= set(SKILL.tags)
    steps = [SKILL.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and len(set(steps)) == 4


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(tier):
    sup.check_single_tier(SKILL, TOPIC, tier)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_four_options_and_a_varied_pool(tier):
    texts = set()
    for seed in range(60):
        q = sup.single(TOPIC, tier, seed)
        assert len(q.distractors) >= 3 and len({q.answer_text, *q.distractors}) == 4 + len(q.distractors) - 3
        texts.add(q.text)
    assert len(texts) >= 20, (tier, len(texts))


def test_pools_and_shapes():
    sup.check_pools(TOPIC, min_prompts=20)


def test_no_python_notation_or_unfilled_slots():
    for tier in sup.TIERS:
        for seed in range(80):
            q = sup.single(TOPIC, tier, seed)
            assert "{" not in q.text and "}" not in q.text and "**" not in q.text, q.text
            for d in q.distractors:
                assert "{" not in d, d


def test_easy_and_medium_keys_match_their_purpose():
    by_question = {p[0].format(g="X"): p for p in gen._PURPOSES}
    seen = set()
    for tier in ("easy", "medium"):
        for seed in range(120):
            q = sup.single(TOPIC, tier, seed)
            purpose = next(
                p for p in gen._PURPOSES if p[0].split("{g}")[0] in q.text and p[0].split("{g}")[-1] in q.text
            )
            key = purpose[1] if tier == "easy" else purpose[3]
            assert q.answer_text == key
            wrong_texts = {w[0] for w in purpose[2]} if tier == "easy" else set(purpose[4:])
            assert set(q.distractors) <= wrong_texts
            seen.add(purpose[0])
    assert len(seen) == len(by_question) == 12


def _entries(q):
    return [
        e.strip()
        for e in re.search(r"The entries were: (.*?)\.(?: One| \(a\)|\s*$)", q.text + " ").group(1).split(", ")
    ]


def test_hard_exactly_one_impossible_entry_and_it_is_the_key():
    kinds = set()
    for seed in range(200):
        q = sup.single(TOPIC, "hard", seed)
        entries = _entries(q)
        key = q.answer_text
        assert entries.count(key) == 1, (seed, entries, key)
        others = [e for e in entries if e != key]
        if all(re.fullmatch(r"-?\d+(\.\d+)?", e) for e in entries):
            k, rest = float(key), [float(e) for e in others]
            if "metres" in q.text:
                kinds.add("unit-slip")
                assert k > 10 and all(1.2 <= v <= 1.6 for v in rest)
            elif k < 0:
                kinds.add("negative-count")
                assert all(v == int(v) and v >= 0 for v in rest)
            else:
                kinds.add("over-maximum")
                top = int(re.search(r"out of (\d+)", q.text).group(1))
                assert k > top and all(0 <= v <= top for v in rest)
        else:
            kinds.add("wrong-category")
            assert key[0].isupper() and all(e.islower() for e in others)
        for d in q.distractors:
            assert d in others, (seed, d)
    assert kinds == {"negative-count", "unit-slip", "over-maximum", "wrong-category"}


def test_hard_valid_extremes_appear_as_options():
    flagged = Counter()
    for seed in range(200):
        q = sup.single(TOPIC, "hard", seed)
        for d, note in q.option_notes.items():
            flagged[note] += 1
    assert flagged["dq-flags-largest-valid"] > 10 and flagged["dq-flags-smallest-valid"] > 10


def test_exam_chain_shape():
    sup.check_exam_chain(SKILL, EXAM, [1, 1, 2, 2], ["mcq", "mcq", "number", "number"])


def test_exam_keys_by_independent_arithmetic():
    for seed in range(100):
        q = sup.exam(EXAM, seed)
        entries = [int(e) for e in _entries(q)]
        a, b, c, d = q.parts
        bad = [e for e in entries if e < 0]
        assert len(bad) == 1 and a.answer_text == str(bad[0])
        assert b.answer_text == "Discrete numerical"
        true = int(re.search(r"should have been (\d+)", c.prompt).group(1))
        total = sum(e for e in entries if e >= 0) + true
        assert int(c.answer_text) == total
        assert int(d.answer_text) * len(entries) == total


def test_exam_follow_through_from_a_wrong_total():
    def wrong(q, canon):
        n = len(_entries(q))
        bad_total = int(canon["c"]) + n
        return {"c": str(bad_total), "d": str((bad_total) // n)}

    assert sup.check_follow_through(SKILL, EXAM, "d", wrong) >= 30
