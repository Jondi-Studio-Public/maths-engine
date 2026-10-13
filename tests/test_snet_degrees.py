"""Tests `graph-vertex-degrees-handshaking` (topic `snet_degrees`) and its exam chain `snet_degrees_chain`.

The oracles read the network or degree list back out of the question TEXT and recount: degrees from the printed edge
list, the handshake sum, the unlisted degree, the odd vertices, and the two hard shapes by search.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
from snet_support import check_graded, check_text, degrees, network_from_text, skill

from mathgen import make
from mathgen.engine import grade

SK = skill("graph-vertex-degrees-handshaking")
TIERS = ("easy", "medium", "hard")
SEEDS = range(80)


def _q(tier, seed):
    return make("snet_degrees", seed=seed, difficulty=tier)


def _exam(seed):
    return make("snet_degrees_chain", seed=seed, difficulty="exam")


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in SEEDS:
        check_graded(SK, _q(tier, seed))


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in SEEDS:
        check_text(_q(tier, seed))


def test_easy_keys_match_a_recount_from_the_text():
    seen = Counter()
    for seed in range(200):
        q = _q("easy", seed)
        if "Find the degree" in q.text or "What is the degree" in q.text:
            n, edges = network_from_text(q.text)
            x = "ABCDEFGH".index(re.search(r"vertex ([A-H])", q.text).group(1))
            assert q.answer == degrees(n, edges)[x]
            seen["degree"] += 1
        elif "odd degree" in q.text or "vertices of odd degree" in q.text:
            n, edges = network_from_text(q.text)
            assert q.answer == sum(d % 2 for d in degrees(n, edges))
            seen["odd"] += 1
        else:
            listed = [int(v) for v in re.findall(r"[A-H]: (\d+)", q.text)]
            assert listed and q.answer * 2 == sum(listed)
            seen["edges"] += 1
    assert min(seen.values()) >= 20, seen


def test_medium_keys_match_the_handshake_sum():
    seen = Counter()
    for seed in range(200):
        q = _q("medium", seed)
        if "other than" in q.text or "except" in q.text:
            listed = [int(v) for v in re.findall(r"[A-H]: (\d+)", q.text)]
            m = _edge_count(q.text)
            assert q.answer == 2 * m - sum(listed) and q.answer >= 1
            seen["missing"] += 1
        else:
            listed = [int(v) for v in re.findall(r"\b(\d+)\b", re.split(r"degrees", q.text)[1].split("?")[0])]
            assert 2 * q.answer == sum(listed)
            seen["edges"] += 1
    assert min(seen.values()) >= 30, seen


def _edge_count(text):
    return int(
        re.search(r"(\d+) (?:ferry routes|roads|rail links|bike paths|walking tracks|doorways|cables)", text).group(1)
    )


def test_hard_keys_match_a_search():
    seen = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        if "degree at least" in q.text:
            m = int(re.search(r"(\d+) (?:edges|ferry|roads|rail|bike|walking|doorways|cables)", q.text).group(1))
            k = int(re.search(r"at least (\d+)", q.text).group(1))
            assert q.answer == max(v for v in range(1, 2 * m + 1) if k * v <= 2 * m)
            seen["least"] += 1
        else:
            v = int(re.search(r"(\d+) vertices", q.text).group(1))
            m = int(re.search(r"and (\d+) ", q.text).group(1))
            d1, d2 = (int(x) for x in re.search(r"degree (\d+) or (?:degree )?(\d+)", q.text).groups())
            solutions = [b for b in range(v + 1) if d1 * (v - b) + d2 * b == 2 * m]
            assert solutions == [q.answer]
            seen["two"] += 1
    assert min(seen.values()) >= 40, seen


def test_tiers_differ_in_shape():
    easy = [_q("easy", s).text for s in range(60)]
    hard = [_q("hard", s).text for s in range(60)]
    assert sum("joins:" in t for t in easy) >= 20  # a stated network
    assert not any("joins:" in t for t in hard)  # no network at all
    assert all("at least" in t or "degree" in t for t in hard)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    answers = [_q(tier, s).answer_text for s in range(200)]
    assert len(set(answers)) >= 4
    assert max(Counter(answers).values()) <= 80


# -- exam chain ---------------------------------------------------------------------------------------------------


def test_exam_shape():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 1, 2]
        assert [p.answer_kind for p in q.parts] == ["number"] * 4
        assert [bool(p.follow_through) for p in q.parts] == [False, True, True, True]
        assert all(p.sensible_range for p in q.parts)
        check_text(q)


def test_exam_keys_match_a_recount_from_the_text():
    for seed in range(200):
        q = _exam(seed)
        listed = [int(v) for v in re.findall(r"[A-H]: (\d+)", q.stem)]
        m = _edge_count(q.stem)
        n = int(re.search(r"(\d+) (?:islands|towns|lookouts|galleries|suburbs|stations|computers)", q.stem).group(1))
        assert len(listed) == n - 1
        total = 2 * m
        last = total - sum(listed)
        assert [int(p.answer) for p in q.parts] == [total, last, sum(d % 2 for d in listed) + last % 2, m - last]
        assert 1 <= last <= n - 1


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(80):
        check_graded(SK, _exam(seed))


def test_exam_follow_through_from_wrong_earlier_parts():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        m = _edge_count(q.stem)
        listed = sum(int(v) for v in re.findall(r"[A-H]: (\d+)", q.stem))
        # A wrong degree sum carried into (b), then (b) carried into (c) and (d).
        wrong_a = 2 * m + 2
        result = grade(SK, q, {**canon, "a": str(wrong_a), "b": str(wrong_a - listed)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, result)
        listed_odd = sum(int(v) % 2 for v in re.findall(r"[A-H]: (\d+)", q.stem))
        # A wrong degree of the opposite parity, inside the sensible range, whose follow-through values differ from it
        # (a reply that merely repeats the earlier value earns no follow-through).
        wrong_b = next(
            v for v in range(1, 7) if v % 2 != int(canon["b"]) % 2 and m - v != v and listed_odd + v % 2 != v
        )
        result = grade(SK, q, {**canon, "b": str(wrong_b), "c": str(listed_odd + wrong_b % 2), "d": str(m - wrong_b)})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True, (seed, result)
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (seed, result)
        checked += 1
    assert checked == 60


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 60
    assert max(Counter(answers).values()) <= 30
    assert len({_exam(s).text for s in range(300)}) >= 295
