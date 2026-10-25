"""Tests `choose-summary-statistic-for-data`: every number is recomputed from the values printed in the stem."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from fractions import Fraction

import pytest

from mathgen.core import mcq_options
from tests import r3b_support as sup

SKILL = "choose-summary-statistic-for-data"
TOPIC = "nv_summary_choice"
EXAM = "exd_x_nv_summary_choice"
LIST = re.compile(r": ((?:\d+, )+\d+)[.,]?")
BAD_TOTALS = re.compile(r"temperature|heights?|puzzle|degrees")


def listed(text):
    """The listed values in the stem."""
    return [int(x) for x in LIST.search(text).group(1).split(", ")]


def mean(v):
    return Fraction(sum(v), len(v))


def median(v):
    s = sorted(v)
    n = len(s)
    return Fraction(s[n // 2]) if n % 2 else Fraction(s[n // 2 - 1] + s[n // 2], 2)


def mode(v):
    counts = Counter(v)
    top = max(counts.values())
    (m,) = [x for x, c in counts.items() if c == top]
    return m


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "mcq",
        ["classify-data", "calculate-mean", "calculate-median", "calculate-mode"],
        {"tier": "exam"},
    )


def test_easy_measure_name_follows_the_situation():
    """The named measure is Mode for categories, Median when a few values are far larger, Mean when a total is wanted."""
    seen = Counter()
    for seed in range(400):
        q = sup.tier_question(TOPIC, "easy", seed)
        if q.answer_text not in ("Mode", "Median", "Mean"):
            continue
        t = q.text
        if "sensible measure of centre" in t:
            assert q.answer_text == "Mode"
        elif "far larger than all the others" in t:
            assert q.answer_text == "Median"
        else:
            assert "estimate the total" in t and q.answer_text == "Mean", t
        options = set(mcq_options(q))
        assert options == {"Mean", "Median", "Mode", "Range"}, options
        seen[q.answer_text] += 1
    assert set(seen) == {"Mode", "Median", "Mean"} and min(seen.values()) >= 15, seen


def test_key_word_and_position_vary_in_the_naming_question():
    words, places = Counter(), Counter()
    for seed in range(400):
        q = sup.tier_question(TOPIC, "easy", seed)
        if q.answer_text in ("Mode", "Median", "Mean"):
            words[q.answer_text] += 1
            places[mcq_options(q).index(q.answer_text)] += 1
    assert len(words) == 3 and len(places) == 4, (words, places)


def _tags(topic, tiers, seeds=1500):
    from mathgen.core import REGISTRY

    out = set()
    for tier in tiers:
        for seed in range(seeds):
            q = REGISTRY[topic][1](random.Random(seed), tier=tier)
            for unit in q.get("parts") or [q]:
                out |= {d.misconception for d in unit["distractors"]}
    return out


def test_skill_misconceptions_are_exactly_the_tags_emitted():
    """The YAML lists every tag the generators emit (with the plain near-miss tag) and nothing they never emit."""
    import yaml

    from mathgen.generators.nv_summary_choice_build import MISCONCEPTIONS

    emitted = _tags(TOPIC, ("easy", "medium", "hard")) | _tags(EXAM, ("exam",))
    with open(f"skills/statistics-probability/{SKILL}.yaml", encoding="utf-8") as fh:
        listed_ids = {m["id"] for m in yaml.safe_load(fh)["misconceptions"]}
    assert emitted == listed_ids, (emitted - listed_ids, listed_ids - emitted)
    assert {m[0] for m in MISCONCEPTIONS} | {"near-miss-value"} == listed_ids


def test_no_key_sits_on_a_rounding_boundary():
    """Every number key is exact at its stated places or clear of a rounding tie (0 over 1000 seeds)."""
    from mathgen.generators._exd_r3f_parts import clear_of_boundary

    bad = []
    for seed in range(1000):
        q = sup.exam_question(EXAM, seed)
        v = listed(q.text)
        m, md = mean(v), median(v)
        c, d = q.parts[2], q.parts[3]
        exact = (
            (m - md) / md * 100
            if "percentage" in d.prompt
            else (m - md) * int(re.search(r"for (\d+)", c.prompt).group(1))
        )
        if not clear_of_boundary(exact, 1):
            bad.append(seed)
    assert not bad, bad[:5]


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_numeric_keys_match_the_listed_values(tier):
    checked = Counter()
    for seed in range(400):
        q = sup.tier_question(TOPIC, tier, seed)
        t = q.text
        if not LIST.search(t):
            continue
        v = listed(t)
        key = Fraction(q.answer_text)
        if "mean rating" in t:
            assert key == median(v) and round(mean(v)) != key
            kind = "ordinal"
        elif "does answer the question" in t:
            assert key == mode(v) and key not in (mean(v), median(v))
            kind = "common"
        elif re.search(r"multiplied by|scaled up|estimate the total|scale", t):
            assert key == mean(v) and median(v) != key, t
            kind = "total"
        elif re.search(r"most often|most of|most cars|most households|value that tells", t):
            assert key == mode(v)
            kind = "mode"
        elif "coded" in t:
            counts = Counter(v)
            assert key == max(counts.values())
            kind = "coded"
        else:
            assert key == median(v) and mean(v) != key, t
            kind = "median"
        checked[kind] += 1
    assert sum(checked.values()) >= 100 and len(checked) >= 3, checked


def test_hard_claims_never_accept_the_claimed_summary():
    """The claimed figure (stated in the stem) is a distractor, never the key."""
    seen = 0
    for seed in range(300):
        q = sup.tier_question(TOPIC, "hard", seed)
        claimed = re.search(r"(?:says the typical [a-z ]+ is|used the median,|reports) (\d+\.\d)", q.text)
        if claimed:
            assert q.answer_text != claimed.group(1), q.text
            seen += 1
    assert seen >= 50


def test_hard_coded_count_is_the_largest_tally_and_the_story_agrees():
    checked = 0
    for seed in range(500):
        q = sup.tier_question(TOPIC, "hard", seed)
        if "coded" not in q.text:
            continue
        names = re.findall(r"(\d) = ([A-Za-z]+)", q.text)
        codes = [int(x) for x in re.search(r"are: ((?:\d+, )+\d+)\.", q.text).group(1).split(", ")]
        assert int(q.answer_text) == max(Counter(codes).values())
        shown = float(re.search(r"mean code is (\d\.\d)", q.text).group(1))
        near_name = re.search(r"typical response is (\w+)\.", q.text).group(1)
        near = int([c for c, n in names if n == near_name][0])
        assert abs(shown - near) < 0.5 and abs(Fraction(sum(codes), len(codes)) - near) < Fraction(9, 20), q.text
        checked += 1
    assert checked >= 30


def test_ordinal_claim_mean_read_to_one_place_agrees_with_the_rounded_claim():
    checked = 0
    for seed in range(500):
        q = sup.tier_question(TOPIC, "hard", seed)
        m = re.search(r"mean rating is (\d\.\d), so the typical rating is (\d)", q.text)
        if m:
            assert abs(float(m.group(1)) - int(m.group(2))) < 0.5, q.text
            checked += 1
    assert checked >= 30


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=3, top=0.5)


def test_tiers_differ():
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_totals_are_asked_only_of_quantities_that_add_up():
    """No total of temperatures or heights: every total is of rainfall, distance, money, grocery spending or mass."""
    seen = 0
    for tier in ("easy", "medium", "hard"):
        for seed in range(300):
            t = sup.tier_question(TOPIC, tier, seed).text
            if re.search(r"total|scaled up|multiplied by", t):
                assert not BAD_TOTALS.search(t), t
                seen += 1
    for seed in range(100):
        assert not BAD_TOTALS.search(sup.exam_question(EXAM, seed).text)
    assert seen >= 60


def test_exam_chain():
    kinds = Counter()
    for seed in range(240):
        q = sup.exam_question(EXAM, seed)
        a, b, c, d = q.parts
        v = listed(q.text)
        assert Fraction(a.answer_text) == mean(v) and Fraction(b.answer_text) == median(v)
        if "estimate the total" in q.text:
            k = int(re.search(r"for (\d+) ", q.text).group(1))
            assert Fraction(c.answer_text) == mean(v) * k
            assert Fraction(d.answer_text) == (mean(v) - median(v)) * k
            kinds["total"] += 1
        else:
            assert Fraction(c.answer_text) == mean(v) - median(v)
            assert abs(Fraction(d.answer_text) - (mean(v) - median(v)) / median(v) * 100) <= Fraction(1, 20)
            kinds["skewed"] += 1
        assert c.follow_through and d.follow_through and not a.follow_through and not b.follow_through
    assert min(kinds.values()) >= 80 and len(kinds) == 2
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_answer_space(EXAM)


def test_exam_follow_through():
    from tests.r3d_support import assert_follow_through_credit

    assert_follow_through_credit(sup.skill_object(SKILL), EXAM, range(40), min_checked=10)


def _rank_and_position(unit):
    key, wrong = unit.answer_text, list(unit.distractors)
    if not re.fullmatch(r"-?[\d.]+", key):  # a measure named in words has no numeric order
        return 0, mcq_options(unit).index(key)
    return sum(float(w) < float(key) for w in wrong), mcq_options(unit).index(key)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_key_rank_and_position_vary(tier):
    """Over 240 seeds each part or question has its key at three or more different ranks and in several list places."""
    ranks, places = defaultdict(set), defaultdict(set)
    for seed in range(240):
        q = sup.exam_question(EXAM, seed) if tier == "exam" else sup.tier_question(TOPIC, tier, seed)
        for label, unit in [(p.label, p) for p in q.parts] or [("q", q)]:
            r, pos = _rank_and_position(unit)
            ranks[label].add(r)
            places[label].add(pos)
    for label in ranks:
        assert len(ranks[label]) >= 3, (tier, label, ranks[label])
        assert len(places[label]) >= 2, (tier, label, places[label])


def test_key_rank_is_not_fixed_for_any_question_kind():
    kinds = defaultdict(Counter)
    for tier in ("easy", "medium", "hard"):
        for seed in range(400):
            q = sup.tier_question(TOPIC, tier, seed)
            if not re.fullmatch(r"-?[\d.]+", q.answer_text):
                continue  # a measure named in words has no numeric rank
            kinds[(tier, " ".join(re.sub(r"\d+", "#", q.text).split()[:3]))][_rank_and_position(q)[0]] += 1
    checked = 0
    for kind, counts in kinds.items():
        n = sum(counts.values())
        if n >= 40:
            assert max(counts.values()) / n <= 0.6, (kind, counts)
            checked += 1
    assert checked >= 2


def test_unknown_tier_raises():
    from mathgen.generators.nv_summary_choice_build import exd_x_nv_summary_choice, nv_summary_choice

    with pytest.raises(ValueError):
        nv_summary_choice(random.Random(1), tier="exam")
    with pytest.raises(ValueError):
        exd_x_nv_summary_choice(random.Random(1), tier="hard")
