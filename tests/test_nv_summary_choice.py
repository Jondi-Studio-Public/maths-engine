"""Tests `choose-summary-statistic-for-data`: every number is recomputed from the values printed in the stem."""

from __future__ import annotations

import random
import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen.engine import grade
from tests import r3b_support as sup

SKILL = "choose-summary-statistic-for-data"
TOPIC = "nv_summary_choice"
EXAM = "exd_x_nv_summary_choice"
LIST = re.compile(r": ((?:\d+, )+\d+)[.,]?")


def listed(text):
    """The ten values in the stem."""
    return [int(x) for x in LIST.search(text).group(1).split(", ")]


def mean(v):
    return Fraction(sum(v), len(v))


def median(v):
    s = sorted(v)  # ten values: the two middle ones
    return Fraction(s[4] + s[5], 2)


def mode(v):
    counts = Counter(v)
    top = max(counts.values())
    (m,) = [x for x, c in counts.items() if c == top]
    return m


def test_wired():
    sup.assert_wired(
        SKILL, TOPIC, EXAM, "mcq", ["classify-data", "calculate-mean", "calculate-median", "calculate-mode"]
    )


def test_easy_choice_matches_the_data():
    """The measure chosen is the one the printed values and purpose call for; the data itself is checked."""
    seen = Counter()
    for seed in range(300):
        q = sup.tier_question(TOPIC, "easy", seed)
        seen[q.answer_text] += 1
        t = q.text
        assert q.answer_kind == "mcq"
        assert sorted([q.answer_text, *q.distractors]) == ["Mean", "Median", "Mode", "Range"], t
        if re.search(r"records the (lunch|type of dwelling|main sport|country)", t):
            assert q.answer_text == "Mode" and not re.search(r"\d", t)
        elif re.search(r"rate the service|how safe|how satisfied", t):
            assert q.answer_text == "Median" and not re.search(r"\d", t)
        elif "most often" in t or "most of" in t or "most cars" in t or "most households" in t:
            v = listed(t)
            assert q.answer_text == "Mode" and len(set(v)) < len(v) and mean(v) != mode(v) != median(v)
        elif "total" in t or "shared equally" in t:
            v = listed(t)
            assert q.answer_text == "Mean" and len(v) == 10 and mean(v) != median(v)
        else:
            v = listed(t)
            assert q.answer_text == "Median" and max(v) > 2 * sorted(v)[-2], t
            assert mean(v) - median(v) >= median(v) / 4
    # the key is spread over the measures, not one favourite
    assert set(seen) == {"Mean", "Median", "Mode"} and max(seen.values()) / 300 <= 0.5, seen


@pytest.mark.parametrize("tier", ["medium", "hard"])
def test_numeric_keys_match_the_listed_values(tier):
    checked = 0
    for seed in range(300):
        q = sup.tier_question(TOPIC, tier, seed)
        if q.answer_kind != "number":
            continue
        v = listed(q.text)
        assert len(v) == 10
        key = Fraction(q.answer_text)
        t = q.text
        if "extreme value" in t or "distort" in t or "better than that claim" in t:
            assert key == median(v) and mean(v) != key, t
        elif "multiplied by" in t or "scaled up" in t or "multiplied by" in t or "instead" in t:
            assert key == mean(v) and median(v) != key, t
        else:
            assert key == mode(v), t
            assert mode(v) not in (mean(v), median(v))
        checked += 1
    assert checked >= 60


def test_hard_claims_never_accept_the_claimed_summary():
    """The claimed figure (stated in the stem) is an option or distractor, never the key."""
    for seed in range(300):
        q = sup.tier_question(TOPIC, "hard", seed)
        if q.answer_kind == "number":
            claimed = re.search(r"(?:is|reports|median,|used the median,) (\d+\.\d)", q.text).group(1)
            assert q.answer_text != claimed, q.text


def test_hard_mcq_key_is_not_given_away_by_position_or_length():
    longest = shortest = n = 0
    for seed in range(400):
        q = sup.tier_question(TOPIC, "hard", seed)
        if q.answer_kind == "mcq" and "Which response is best" in q.text:
            options = [q.answer_text, *q.distractors]
            assert len(options) == 4
            others = sorted(len(o) for o in options if o != q.answer_text)
            longest += len(q.answer_text) > others[-1]  # strictly longer than every other option
            shortest += len(q.answer_text) < others[0]
            n += 1
    assert n >= 100
    assert longest / n <= 0.5 and shortest / n <= 0.5, (longest, shortest, n)


def test_hard_coded_category_is_not_the_mean_code():
    checked = 0
    for seed in range(400):
        q = sup.tier_question(TOPIC, "hard", seed)
        if "coded" not in q.text:
            continue
        names = re.findall(r"\d = ([A-Za-z]+)", q.text)
        codes = [int(x) for x in re.search(r"are: ((?:\d+, )+\d+)\.", q.text).group(1).split(", ")]
        top = Counter(codes).most_common(1)[0][0]
        assert q.answer_text == names[top - 1]
        mean_code = Fraction(sum(codes), len(codes))
        near = min(range(1, 5), key=lambda c: (abs(c - mean_code), c))
        assert names[near - 1] in q.distractors and names[near - 1] != q.answer_text
        checked += 1
    assert checked >= 20


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=3, top=0.5)


def test_tiers_differ():
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_chain():
    kinds = Counter()
    for seed in range(200):
        q = sup.exam_question(EXAM, seed)
        a, b, c, d = q.parts
        v = listed(q.text)
        assert Fraction(a.answer_text) == mean(v) and Fraction(b.answer_text) == median(v)
        if "estimate the total" in q.text:
            k = int(re.search(r"for (\d+) ", q.text).group(1))
            assert Fraction(c.answer_text) == mean(v) * k
            assert d.answer_text.startswith("The mean")
            kinds["total"] += 1
        else:
            assert Fraction(c.answer_text) == mean(v) - median(v)
            assert d.answer_text.startswith("The median")
            kinds["skewed"] += 1
        assert c.follow_through and not a.follow_through and not b.follow_through and not d.follow_through
    assert kinds["total"] >= 40 and kinds["skewed"] >= 40
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_answer_space(EXAM)


def test_exam_follow_through():
    from tests.r3d_support import assert_follow_through_credit

    assert_follow_through_credit(sup.skill_object(SKILL), EXAM, range(40), min_checked=10)


def test_unknown_tier_raises():
    from mathgen.generators.nv_summary_choice_build import exd_x_nv_summary_choice, nv_summary_choice

    with pytest.raises(ValueError):
        nv_summary_choice(random.Random(1), tier="exam")
    with pytest.raises(ValueError):
        exd_x_nv_summary_choice(random.Random(1), tier="hard")
