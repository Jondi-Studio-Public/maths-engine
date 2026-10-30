"""`ds_classify_ordered`: classify three or four variables in order (an `ordered_list` tier of `classify-data`)."""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.ds_classify_ordered_build  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.grading import grade
from mathgen.generators.r3j_classify_build import CLASSES, POOL
from mathgen.text_quality import find_text_defects

WORDS = {"nominal", "ordinal", "discrete", "continuous"}
CLASS_OF = {p[0]: p[1].split()[0].lower() for p in POOL}
TWIST_OF = {p[0]: p[2] for p in POOL}
SEEDS = range(1, 121)
SIZE = {"medium": 3, "hard": 4}


def _variables(text):
    listing = re.search(r"\(A\) (.+?)\. ", text).group(0)
    return re.split(r"; \([B-D]\) ", re.sub(r"^\(A\) ", "", listing[:-2]))


@pytest.mark.parametrize("tier", ["medium", "hard"])
def test_key_accepted_distractors_rejected_and_oracle(tier):
    for seed in SEEDS:
        q = make("ds_classify_ordered", seed=seed, difficulty=tier)
        assert q.answer_kind == "ordered_list"
        variables = _variables(q.text)
        assert len(variables) == SIZE[tier] == len(q.answer)
        # independent oracle: read the classes back out of the stem through the pool
        assert q.answer == [CLASS_OF[v] for v in variables]
        assert set(q.answer) <= WORDS
        assert grade("ordered_list", q, q.answer_text).correct
        assert grade("ordered_list", q, " ,".join(q.answer).title()).correct
        assert len(q.distractors) >= 3
        for d in q.distractors:
            assert not grade("ordered_list", q, d).correct, (seed, d)
        # no transposition or single slip is accepted
        for i in range(len(q.answer)):
            for w in WORDS - {q.answer[i]}:
                reply = q.answer[:i] + [w] + q.answer[i + 1 :]
                assert not grade("ordered_list", q, ", ".join(reply)).correct
        assert not find_text_defects(q)


def test_mix_of_classes_and_twists():
    for seed in SEEDS:
        m = make("ds_classify_ordered", seed=seed, difficulty="medium")
        assert len(set(m.answer)) >= 2
        assert not any(TWIST_OF[v] for v in _variables(m.text))
        h = make("ds_classify_ordered", seed=seed, difficulty="hard")
        assert len(set(h.answer)) >= 3
        assert sum(TWIST_OF[v] for v in _variables(h.text)) >= 2


@pytest.mark.parametrize("tier,ceiling", [("medium", 0.05), ("hard", 0.03)])
def test_no_reply_can_be_guessed(tier, ceiling):
    """The commonest key must be rare, so a fixed guess (or a pattern) wins almost never."""
    keys = Counter(", ".join(make("ds_classify_ordered", seed=s, difficulty=tier).answer) for s in range(1, 401))
    assert max(keys.values()) / 400 <= ceiling, keys.most_common(3)
    assert len(keys) >= (40 if tier == "medium" else 120)


@pytest.mark.parametrize("tier", ["medium", "hard"])
def test_at_least_three_prompt_shapes(tier):
    shapes = {
        re.sub(r"\(A\) .+?\. ", "", make("ds_classify_ordered", seed=s, difficulty=tier).text)[:25]
        for s in range(1, 80)
    }
    assert len(shapes) >= 3


def test_unknown_tier_is_rejected():
    from mathgen.generators.ds_classify_ordered_build import ds_classify_ordered
    import random

    with pytest.raises(Exception):
        ds_classify_ordered(random.Random(1), tier="easy")


def test_existing_classify_topics_unchanged():
    assert CLASSES[0] == "Nominal categorical"
