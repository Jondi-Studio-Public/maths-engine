"""Tests `f10-shape-and-variation-of-a-display` (AC9M4ST02): every answer is recomputed from the drawn figure and the
statements printed in the question, by code that does not share the generator's classes, and every tier's key is
accepted and every distractor rejected by the real grader."""

from __future__ import annotations

import functools
import re
import statistics
from collections import Counter

import pytest

from mathgen import figures, make
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape
from tests import w7b_support as sup

SKILL = "f10-shape-and-variation-of-a-display"
TOPICS = {"easy": "ds_shape_var_read", "medium": "ds_shape_var_shape", "hard": "ds_shape_var_compare"}
SEEDS = range(70)


def groups(fig):
    """[(name, {value: count})] read straight off the figure's spec."""
    if fig["kind"] == "dot_plot":
        return [("", {d["at"]: d["count"] for d in fig["dots"]})]
    first = int(fig["categories"][0].split()[0])
    return [(s["name"], {first + i: n for i, n in enumerate(s["values"]) if n}) for s in fig["series"]]


def facts(freq):
    values = sorted(v for v, n in freq.items() for _ in range(n))
    (mode,) = statistics.multimode(values)
    return dict(values=values, mode=mode, lo=values[0], hi=values[-1], total=len(values), top=freq[mode])


def has_outlier(freq):
    xs = sorted(freq)
    return len(xs) >= 3 and freq[xs[-1]] == 1 and xs[-1] - xs[-2] >= 3 or freq[xs[0]] == 1 and xs[1] - xs[0] >= 3


def heights(freq):
    return [freq.get(v, 0) for v in range(min(freq), max(freq) + 1)]


def symmetric(freq):
    h = heights(freq)
    return h == h[::-1]


def bunched(freq, end):
    h = heights(freq)
    if has_outlier(freq):
        return False
    steps = list(zip(h, h[1:]))
    if end == "low":
        return all(a >= b for a, b in steps) and h[0] > h[1] and h[-1] == 1
    return all(a <= b for a, b in steps) and h[-1] > h[-2] and h[0] == 1


def truth(statement, gs):
    ds = {n: (facts(f), f) for n, f in gs}
    f0, fr0 = next(iter(ds.values()))
    patterns = [
        (r"The mode is (\d+)\.", lambda m: f0["mode"] == int(m[1])),
        (r"The range is (\d+)\.", lambda m: f0["hi"] - f0["lo"] == int(m[1])),
        (r"A total of (\d+) \w+ are shown\.", lambda m: f0["total"] == int(m[1])),
        (r"The data are symmetric\.", lambda m: symmetric(fr0)),
        (r"The values are bunched at the low end\.", lambda m: bunched(fr0, "low")),
        (r"The values are bunched at the high end\.", lambda m: bunched(fr0, "high")),
        (r"One value is an outlier, far from all the others\.", lambda m: has_outlier(fr0)),
        (r"(.+) has the greater range\.", lambda m: rng(ds[m[1]][0]) == max(rng(f) for f, _ in ds.values())),
        (r"(.+) and (.+) have the same mode\.", lambda m: ds[m[1]][0]["mode"] == ds[m[2]][0]["mode"]),
        (
            r"The mode of (.+) is greater than the mode of (.+)\.",
            lambda m: ds[m[1]][0]["mode"] > ds[m[2]][0]["mode"],
        ),
        (r"(.+) has an outlier\.", lambda m: has_outlier(ds[m[1]][1])),
        (r"The data for (.+) are symmetric\.", lambda m: symmetric(ds[m[1]][1])),
        (r"(.+) has more values than (.+)\.", lambda m: ds[m[1]][0]["total"] > ds[m[2]][0]["total"]),
        (r"(.+) and (.+) have the same number of values\.", lambda m: ds[m[1]][0]["total"] == ds[m[2]][0]["total"]),
    ]
    for pattern, fn in patterns:
        if m := re.fullmatch(pattern, statement):
            return fn(m)
    raise AssertionError(statement)


def rng(f):
    return f["hi"] - f["lo"]


@functools.lru_cache(maxsize=1)
def skill_obj():
    return sup.skill_of(SKILL)


def make_q(tier, seed):
    return make(TOPICS[tier], seed=seed, difficulty=tier)


def expected_number(q):
    """The numeric answer from the printed question and the figure alone."""
    gs = groups(q.figure)
    t = q.text
    fs = [facts(f) for _, f in gs]
    if len(gs) == 1:
        f, freq = fs[0], gs[0][1]
        if "range of the rest" in t:
            rest = [
                v
                for v in f["values"]
                if v
                != (
                    f["hi"] if freq[f["hi"]] == 1 and has_outlier(freq) and f["hi"] - sorted(freq)[-2] >= 3 else f["lo"]
                )
            ]
            return max(rest) - min(rest)
        if "What is the mode" in t:
            return f["mode"]
        if "What is the range" in t:
            return rng(f)
        if "shown altogether" in t:
            return f["total"]
        if "have the mode value" in t:
            return f["top"]
        if "far away from all the others" in t:
            xs = sorted(freq)
            return xs[-1] if freq[xs[-1]] == 1 and xs[-1] - xs[-2] >= 3 else xs[0]
        if "greater than the mode" in t:
            return sum(1 for v in f["values"] if v > f["mode"])
    else:
        a, b = fs
        if "One group has the greater spread" in t:
            return abs(rng(a) - rng(b))
        if "How many more" in t:
            return abs(a["total"] - b["total"])
        if "all the values of" in t:
            return max(a["hi"], b["hi"]) - min(a["lo"], b["lo"])
    raise AssertionError(t)


def unit(q):
    """(kind, key text, distractor texts, option notes, grade function) for a question or its single part."""
    if q.parts:
        (p,) = q.parts
        return (
            p.answer_kind,
            p.answer_text,
            p.distractors,
            p.option_notes,
            lambda s: grade(skill_obj(), q, {"a": s}),
        )
    return q.answer_kind, q.answer_text, q.distractors, q.option_notes, lambda s: grade(skill_obj(), q, s)


def test_wired():
    skill = skill_obj()
    assert skill.status == "active" and skill.answer == "number" and "AC9M4ST02".lower() in skill.tags
    assert [(s.topic, s.difficulty) for s in skill.sources] == [(TOPICS[t], t) for t in TOPICS]
    assert "open-response" in skill.description and "rubric" in skill.description
    listed = {m.id for m in skill.misconceptions}
    used = set()
    for tier in TOPICS:
        for seed in SEEDS:
            q = make_q(tier, seed)
            used |= {v for v in unit(q)[3].values()}
    assert used <= listed, used - listed
    assert all(i in CATALOGUE for i in listed)


@pytest.mark.parametrize("tier", list(TOPICS))
def test_every_key_accepted_every_distractor_rejected(tier):
    for seed in SEEDS:
        q = make_q(tier, seed)
        kind, key, distractors, notes, g = unit(q)
        assert g(key).correct, (tier, seed, key)
        assert len(distractors) == 3 and len(set(distractors)) == 3, (tier, seed)
        for text in distractors:
            assert not g(text).correct, (tier, seed, text, key)
            assert notes[text] in CATALOGUE, (tier, seed, text)
        assert find_text_defects(q) == [], (tier, seed, find_text_defects(q))
        assert q.to_json() == make_q(tier, seed).to_json()


@pytest.mark.parametrize("tier", list(TOPICS))
def test_answers_recomputed_from_the_figure_and_the_printed_statements(tier):
    kinds = Counter()
    for seed in SEEDS:
        q = make_q(tier, seed)
        kind, key, _, _, g = unit(q)
        kinds[kind] += 1
        assert not figures.validate(q.figure)
        if kind == "set":
            body = q.text[re.search(r"(?:^|\s)A\. ", q.text).end() :]
            lines = [
                (letter, re.sub(r"^[B-E]\. ", "", piece))
                for letter, piece in zip("ABCDE", re.split(r"(?<=\.) (?=[B-E]\. )", body))
            ]
            assert [letter for letter, _ in lines] == list("ABCDE")
            true = sorted(letter for letter, s in lines if truth(s, groups(q.figure)))
            assert 1 <= len(true) <= 3
            assert g("{" + ", ".join(true) + "}").correct, (tier, seed, true, key)
        else:
            assert g(str(expected_number(q))).correct, (tier, seed, key)
        assert figures.get_reader(TOPICS[tier])(q) is not None
    assert kinds["number"] >= 10
    if tier != "easy":
        assert kinds["set"] >= 10, kinds


def test_the_figure_never_names_a_number_answer():
    for tier in TOPICS:
        for seed in SEEDS:
            q = make_q(tier, seed)
            if not q.parts:
                figures.assert_no_leak(q.figure, [q.answer_text])


@pytest.mark.parametrize("tier", list(TOPICS))
def test_three_prompt_shapes_per_tier(tier):
    shapes = {prompt_shape(make_q(tier, s)) for s in range(120)}
    assert len(shapes) >= 3, shapes


def test_hard_adds_a_shape_over_easy():
    easy = {prompt_shape(make_q("easy", s)) for s in range(80)}
    hard = {prompt_shape(make_q("hard", s)) for s in range(80)}
    assert hard - easy


def test_easy_uses_both_displays_and_every_ask():
    seen, asks = set(), set()
    for seed in range(120):
        q = make_q("easy", seed)
        seen.add(q.figure["kind"])
        asks.add(re.sub(r"^.*?\. (?=[A-Z][^.]*\?$)", "", q.text)[:40])
    assert seen == {"dot_plot", "bar_chart"}
    assert len(asks) >= 5


def test_medium_covers_every_shape():
    shapes = Counter()
    for seed in range(200):
        q = make_q("medium", seed)
        (freq,) = [f for _, f in groups(q.figure)]
        shapes[
            "outlier"
            if has_outlier(freq)
            else "sym"
            if symmetric(freq)
            else "low"
            if bunched(freq, "low")
            else "high"
            if bunched(freq, "high")
            else "other"
        ] += 1
    assert shapes["other"] == 0, shapes
    assert all(shapes[s] >= 5 for s in ("outlier", "sym", "low", "high")), shapes


def test_hard_always_has_two_groups_with_different_ranges():
    for seed in SEEDS:
        q = make_q("hard", seed)
        gs = groups(q.figure)
        assert len(gs) == 2 and gs[0][0] != gs[1][0]
        assert rng(facts(gs[0][1])) != rng(facts(gs[1][1]))
        for _, f in gs:
            assert len(statistics.multimode(facts(f)["values"])) == 1


def test_unknown_tier_is_an_error():
    for topic in TOPICS.values():
        with pytest.raises(ValueError):
            make(topic, seed=1, tier="exam")
