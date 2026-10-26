"""Build test for `senior-read-a-piece-wise-linear-cost-graph` (ACMGM047, the piece-wise linear half).

The oracle reads the drawn graph (its segments) and the numbers in the stem, then answers by evaluating the
piece-wise line; it never imports the generator module.
"""

from __future__ import annotations

import collections
import functools
import re
from fractions import Fraction

import pytest

import mathgen
from mathgen import figures
from mathgen.capabilities import load_capability_gates
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

SKILL_ID = "senior-read-a-piece-wise-linear-cost-graph"
TOPIC = "pw1_piecewise_cost_graph"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(120)


@functools.lru_cache(maxsize=1)
def _skill():
    return next(s for s in load_skills()[1] if s.id == SKILL_ID)


@functools.lru_cache(maxsize=None)
def _make(tier, seed):
    return mathgen.make(TOPIC, seed=seed, difficulty=tier)


def _chain(fig):
    return [tuple(Fraction(str(v)) for v in (*s["from"], *s["to"])) for s in fig["segments"]]


def _at(chain, x):
    for x0, y0, x1, y1 in chain:
        if x <= x1:
            return y0 + (y1 - y0) / (x1 - x0) * (x - x0)
    x0, y0, x1, y1 = chain[-1]
    return y1 + (y1 - y0) / (x1 - x0) * (x - x1)


def _oracle(q):
    """The answer by brute force over whole usages, from the figure and the numbers in the stem."""
    chain = _chain(q.figure)
    t = q.text
    n = [int(v) for v in re.findall(r"\d+", t)]
    if "continues" in t:
        return _at(chain, n[0])
    if "$" in t:
        c = Fraction(re.search(r"\$(\d+(?:\.\d+)?)", t).group(1))
        hits = [x for x in range(0, int(chain[-1][2]) + 1) if _at(chain, x) == c]
        assert len(hits) == 1
        return hits[0]
    if "different rate" in t:
        slopes = [(y1 - y0) / (x1 - x0) for x0, y0, x1, y1 in chain]
        return next(chain[i][0] for i in range(1, len(slopes)) if slopes[i] != slopes[i - 1])
    if "each extra" in t:
        return (_at(chain, n[1]) - _at(chain, n[0])) / (n[1] - n[0])
    return _at(chain, n[0])


def _canon(q):
    return {p.label: p.answer_text for p in q.parts}


def test_skill_shape():
    s = _skill()
    assert s.status == "active" and s.needs_figure and s.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert s.answer == "number" and s.placement_eligible is False and s.mental is None
    assert [x.difficulty for x in s.sources] == TIERS
    assert [x.stages for x in s.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert all(x.topic == TOPIC for x in s.sources)


def test_reader_registered():
    assert figures.get_reader(TOPIC) is not None


@pytest.mark.parametrize("tier", TIERS)
def test_unknown_tier_is_rejected(tier):
    with pytest.raises(ValueError):
        mathgen.make(TOPIC, seed=1, tier="bogus")


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_clean(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert q.to_json() == mathgen.make(TOPIC, seed=seed, difficulty=tier).to_json()
        assert find_text_defects(q) == [], q.text


@pytest.mark.parametrize("tier", TIERS)
def test_figure_valid_no_leak_and_shows_no_answer(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert figures.validate(q.figure) == []
        answers = [p.answer_text for p in q.parts] if q.parts else [q.answer_text, *q.distractors]
        figures.assert_no_leak(q.figure, answers)
        assert all("label" not in p for p in q.figure["points"])
        # every vertex of the plan(s) read sits on the gridlines (the right-hand end of Plan A on the exam need not)
        gx, gy = q.figure["x_grid"], q.figure["y_grid"]
        segs = q.figure["segments"]
        for s in segs[1 if q.parts else 0 :]:
            for x, y in (s["from"], s["to"]):
                assert Fraction(str(x)) % gx == 0, (tier, seed, x)
                # on the single-line tiers every vertex is on the cost gridlines; on the exam only Plan B's fixed charge
                # and Plan A's fixed charge (the kink and the right-hand ends need not be)
                if not q.parts or x == 0:
                    assert Fraction(str(y)) % Fraction(str(gy)) == 0, (tier, seed, x, y)
        if q.parts:
            assert Fraction(str(segs[0]["from"][1])) % Fraction(str(gy)) == 0


@pytest.mark.parametrize("tier", TIERS)
def test_reader_agrees_with_key(tier):
    reader = figures.get_reader(TOPIC)
    for seed in range(250):
        q = _make(tier, seed)
        assert reader(q) == q.answer_text, (tier, seed)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_independent_oracle_agrees(tier):
    for seed in range(300):
        q = _make(tier, seed)
        assert _oracle(q) == Fraction(q.answer_text), (tier, seed, q.text)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_key_accepted_wrong_values_rejected(tier):
    sk = _skill()
    for seed in SEEDS:
        q = _make(tier, seed)
        assert grade(sk, q, q.answer_text).correct and grade(sk, q, str(q.answer)).correct
        assert len(q.distractors) >= 2
        for d in q.distractors:
            assert not grade(sk, q, d).correct, (tier, seed, d)
            assert q.option_notes.get(d) is None or q.option_notes[d] in CATALOGUE
        assert not grade(sk, q, str(q.answer + 1)).correct
        assert not grade(sk, q, str(q.answer + 100)).correct


@pytest.mark.parametrize("tier", TIERS)
def test_variety_and_prompt_shapes(tier):
    qs = [_make(tier, s) for s in range(60)]
    assert len({q.text for q in qs}) >= 30
    assert len({prompt_shape(q) for q in qs}) >= 3
    counts = collections.Counter(q.answer_text for q in [_make(tier, s) for s in range(300)])
    assert max(counts.values()) <= 0.25 * 300


def test_each_single_tier_asks_every_kind_and_both_section_counts():
    texts = [_make("easy", s).text for s in range(200)]
    assert any(" is 0 " in t for t in texts) and any(" is 0 " not in t for t in texts)
    mediums = [_make("medium", s).text for s in range(200)]
    assert any("different rate" in t for t in mediums) and any("each extra" in t for t in mediums)
    hards = [_make("hard", s).text for s in range(200)]
    assert any("continues" in t for t in hards) and any("$" in t for t in hards)
    sections = {len(_make(t, s).figure["segments"]) for t in ("easy", "medium", "hard") for s in range(100)}
    assert sections == {2, 3}


def test_hard_usage_is_in_a_later_section_and_unique():
    for seed in range(300):
        q = _make("hard", seed)
        if "$" not in q.text:
            continue
        chain = _chain(q.figure)
        x = int(q.answer_text)
        assert x > chain[0][2]  # beyond the first kink: the first rate alone would be wrong


# ---------------------------------------------------------------- the exam


def test_exam_shape_and_follow_through():
    sk = _skill()
    for seed in range(150):
        q = _make("exam", seed)
        assert [p.label for p in q.parts] == list("abcd")
        assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]
        assert [p.marks for p in q.parts] == [1, 2, 2, 2]
        canon = _canon(q)
        assert grade(sk, q, q.answer_text).correct
        # (d) follows through from the learner's own (b) and (c)
        fa = Fraction(re.search(r"\$(\d+(?:\.\d+)?)", q.text).group(1))
        wrong_b, wrong_c = Fraction(canon["b"]) + 1, int(canon["c"]) + 1
        carried = f"{float(fa + wrong_b * wrong_c):.2f}"
        res = grade(sk, q, {**canon, "b": f"{float(wrong_b):.2f}", "c": str(wrong_c), "d": carried})
        if carried in q.parts[3].distractors or carried == canon["d"]:
            continue
        assert res.parts["b"].marks == 0 and res.parts["c"].marks == 0
        assert res.parts["d"].marks == q.parts[3].marks and res.parts["d"].follow_through is True


def test_exam_options_graded_wrong_and_tagged():
    sk = _skill()
    tagged = 0
    for seed in range(100):
        q = _make("exam", seed)
        canon = _canon(q)
        for p in q.parts:
            assert len(p.distractors) == 3 and len(set(p.distractors)) == 3
            for d in p.distractors:
                assert grade(sk, q, {**canon, p.label: d}).parts[p.label].marks == 0, (seed, p.label, d)
                assert p.option_notes.get(d) is None or p.option_notes[d] in CATALOGUE
                tagged += d in p.option_notes
    assert tagged >= 300


def test_exam_plans_meet_exactly_once_and_a_is_straight():
    for seed in range(300):
        q = _make("exam", seed)
        segs = _chain(q.figure)
        a, b = segs[:1], segs[1:]
        assert len(b) == 2 and b[0][2] == b[1][0]
        xs = range(int(a[0][2]) + 1)
        meet = [x for x in xs if _at(a, x) == _at(b, x)]
        assert meet == [int(_canon(q)["c"])]
        assert _at(b, int(re.search(r"\d+", q.text.split("(a)")[1]).group())) == Fraction(_canon(q)["a"])
        assert (a[0][3] - a[0][1]) / (a[0][2] - a[0][0]) == Fraction(_canon(q)["b"])
        assert _at(a, meet[0]) == Fraction(_canon(q)["d"])
        # both crossing orders occur (the crossing before and after B's kink)
    orders = {int(_canon(_make("exam", s))["c"]) < _chain(_make("exam", s).figure)[2][0] for s in range(200)}
    assert orders == {True, False}


def test_exam_parts_have_many_distinct_answers_over_300_seeds():
    for window in (0, 300):
        qs = [_make("exam", s) for s in range(window, window + 300)]
        for i, label in enumerate("abcd"):
            counts = collections.Counter(q.parts[i].answer_text for q in qs)
            # (a) and (d) are costs on many cost steps; (b) and (c) are rates and usages that must stay realistic
            assert len(counts) >= (100 if label in "ad" else 80 if label == "b" else 70), (window, label, len(counts))
            assert max(counts.values()) <= 0.1 * 300, (window, label)


def test_medium_answers_are_widely_spread():
    counts = collections.Counter(_make("medium", s).answer_text for s in range(300))
    assert len(counts) >= 45 and max(counts.values()) <= 0.08 * 300


def test_exam_contract():
    answers = collections.Counter(mathgen.make(TOPIC, seed=s, difficulty="exam").answer_text for s in range(300))
    assert len(answers) >= 100 and max(answers.values()) <= 0.1 * 300
    assert len({prompt_shape(_make("exam", s)) for s in range(60)}) >= 3


def test_all_wrong_earns_zero():
    sk = _skill()
    for seed in range(40):
        q = _make("exam", seed)
        assert grade(sk, q, {p.label: "999" for p in q.parts}).marks == 0


def test_every_kink_changes_the_slope_by_at_least_a_fifth():
    for tier, seeds in (("exam", 600), ("easy", 200), ("medium", 200), ("hard", 200)):
        for seed in range(seeds):
            segs = _chain(_make(tier, seed).figure)
            segs = segs[1:] if tier == "exam" else segs
            slopes = [(y1 - y0) / (x1 - x0) for x0, y0, x1, y1 in segs]
            for a, b in zip(slopes, slopes[1:]):
                assert abs(a - b) >= max(a, b) / 5, (tier, seed, a, b)
