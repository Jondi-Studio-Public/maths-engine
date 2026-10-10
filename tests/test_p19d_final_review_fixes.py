"""Regression tests for two p19d findings: a possessive-plural wording bug in the random-sample exam tier and
distractor spread in the two-way table tiers."""

import re

from mathgen import make

POSSESSIVE = re.compile(r"[a-z]s's\b")


def test_random_sample_exam_no_plural_possessive():
    bad = []
    for seed in range(200):
        q = make("p19d_random_sample_valid", seed=seed, difficulty="exam")
        if POSSESSIVE.search(q.text):
            bad.append(seed)
    assert not bad, f"{len(bad)}/200 prompts carry a plural possessive: {bad}"


def _pct(s):
    return float(str(s).replace("%", "").replace("\\", "").strip())


def test_two_way_exam_and_hard_distractors_within_0_100():
    bad = []
    counts = []
    for tier in ("exam", "hard"):
        for seed in range(200):
            q = make("p19d_two_way_table", seed=seed, difficulty=tier)
            ds = list(q.distractors)
            counts.append(len(ds))
            if any(not (0 <= _pct(d) <= 100) for d in ds):
                bad.append((tier, seed))
    print("min distractor count", min(counts))
    assert not bad, f"{len(bad)} questions with a distractor outside [0,100]: {bad}"
    assert min(counts) >= 2
