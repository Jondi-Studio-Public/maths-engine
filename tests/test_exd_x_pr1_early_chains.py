"""Exam-tier checks for the four early-years chains (story, sharing, mystery number, repeating pattern).

(a) several prompt shapes, some absent from the hard tier; (b) every part's key is re-derived from the RENDERED
PROMPT with independent code over 300 seeds; (c) the full-marks reply is accepted and every distractor is
rejected; (d) a wrong earlier answer used correctly in a later part earns that part's marks.
"""

import re

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

SKILLS = {s.id: s for s in load_skills()[1]}


def _ints(text):
    return [int(x) for x in re.findall(r"\d+", text)]


# -- independent oracles: (question, override for part a) -> {"a": .., "b": .., "c": ..} ------------------------


def oracle_story(q, a=None):
    stem = q.stem
    if "more" in stem and "passengers" not in stem and "canteen" not in stem:
        A, d = _ints(stem)
        g = _ints(q.parts[2].prompt)[0]
        a = A + d if a is None else a
        b = A + a
        return {"a": a, "b": b, "c": b - g}
    if "passengers" in stem:
        A, u, v, cap = _ints(stem)
        a = A + u if a is None else a
        b = a - v
        return {"a": a, "b": b, "c": cap - b}
    A, s, m, T = _ints(stem)
    a = A - s if a is None else a
    b = a + m
    return {"a": a, "b": b, "c": T - b}


def oracle_share(q, a=None):
    total, n = _ints(q.stem)[:2]
    k = _ints(q.parts[1].prompt)[0]
    a = total // n if a is None else a
    assert total % n == 0
    b = k * a
    return {"a": a, "b": b, "c": total - b}


def oracle_mystery(q, a=None):
    told = q.stem.split("thinks of a number. ")[1]
    nums = _ints(told)
    if "then takes away" in told:
        b, s, r = nums
        x = r - b + s
    elif "takes" in told:
        b, r = nums
        x = r + b
    else:
        b, r = nums
        x = r - b
    e = _ints(q.parts[1].prompt)[0]
    a = x if a is None else a
    y = a + e
    return {"a": a, "b": y, "c": a + y}


def oracle_pattern(q, a=None):
    seq = re.search(r"pattern: (.*), and so on", q.stem).group(1).split(", ")
    count = _ints(q.stem.split("and so on")[1])[0]
    target = re.search(r"How many (\w+) \w+ are on", q.parts[2].prompt).group(1)
    n = len(seq)
    u = next(p for p in range(1, n + 1) if n % p == 0 and seq == seq[:p] * (n // p))
    unit = seq[:u]
    rem = count % u
    per, extra = unit.count(target), unit[:rem].count(target)
    a = u if a is None else a
    b = count // a
    return {"a": a, "b": b, "c": per * b + extra}


CASES = [
    ("exd_x_pr1_story_chain", "f10-add-subtract-practical-situations", oracle_story),
    ("exd_x_pr1_share_chain", "f10-partition-collections-equal-groups", oracle_share),
    ("exd_x_pr1_mystery_chain", "f10-addition-facts-related-subtraction", oracle_mystery),
    ("exd_x_pr1_pattern_chain", "f10-pattern-copy-repeat-unit", oracle_pattern),
]
IDS = [c[0] for c in CASES]
HARD = {
    "f10-add-subtract-practical-situations": ("practical_situation_sum", {"variant": "trap"}),
    "f10-partition-collections-equal-groups": ("p19a_partition_equal_groups", {"tier": "hard"}),
    "f10-addition-facts-related-subtraction": ("p19a_add_sub_fact_family", {"tier": "hard"}),
    "f10-pattern-copy-repeat-unit": (
        "pattern_repeat_unit",
        {"unit_length": 3, "shape": "trap", "full_repeats": [2, 3]},
    ),
}


def _q(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


def _canon(q):
    return {p.label: p.answer_text for p in q.parts}


@pytest.mark.parametrize("topic,skill_id,oracle", CASES, ids=IDS)
def test_source_wired(topic, skill_id, oracle):
    exam = [s for s in SKILLS[skill_id].sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [topic]
    assert exam[0].stages == ["exam"]


@pytest.mark.parametrize("topic,skill_id,oracle", CASES, ids=IDS)
def test_shapes_and_text(topic, skill_id, oracle):
    exam = {prompt_shape(_q(topic, s)) for s in range(60)}
    name, params = HARD[skill_id]
    hard = {prompt_shape(make(name, seed=s, **params)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard
    for s in range(60):
        assert find_text_defects(_q(topic, s)) == []


@pytest.mark.parametrize("topic,skill_id,oracle", CASES, ids=IDS)
def test_chain_shape_and_determinism(topic, skill_id, oracle):
    for s in range(60):
        q = _q(topic, s)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "number" and len(p.distractors) == 3 for p in q.parts)
        assert all(d.isdigit() and int(d) > 0 for p in q.parts for d in p.distractors)
    assert _q(topic, 7).to_json() == _q(topic, 7).to_json()


@pytest.mark.parametrize("topic,skill_id,oracle", CASES, ids=IDS)
def test_keys_match_independent_derivation(topic, skill_id, oracle):
    for s in range(300):
        q = _q(topic, s)
        want = {k: str(v) for k, v in oracle(q).items()}
        assert want == _canon(q), (s, q.text)


@pytest.mark.parametrize("topic,skill_id,oracle", CASES, ids=IDS)
def test_key_accepted_and_every_distractor_rejected(topic, skill_id, oracle):
    skill = SKILLS[skill_id]
    for s in range(120):
        q = _q(topic, s)
        full = grade(skill, q, _canon(q))
        assert full.correct and full.marks == full.max_marks, (s, q.text)
        # the labelled answer_text the library sweeps use
        assert grade(skill, q, q.answer_text).correct
        for p in q.parts:
            for d in p.distractors:
                r = grade(skill, q, {**_canon(q), p.label: d})
                assert r.parts[p.label].marks == 0, (s, p.label, d)


@pytest.mark.parametrize("topic,skill_id,oracle", CASES, ids=IDS)
def test_follow_through_from_a_wrong_first_part(topic, skill_id, oracle):
    skill = SKILLS[skill_id]
    checked = 0
    for s in range(150):
        q = _q(topic, s)
        # the first wrong first answer that keeps every later value a sensible positive count (follow-through is
        # withheld when an earlier reply makes the chain impossible)
        wrongs = [int(d) for d in q.parts[0].distractors]
        usable = [
            w
            for w in wrongs
            if min(oracle(q, a=w).values()) > 0
            # the engine withholds follow-through for a reply that merely repeats the earlier value it uses
            and len(set(oracle(q, a=w).values())) == 3
        ]
        if not usable:
            continue
        a_wrong = usable[0]
        want = oracle(q, a=a_wrong)
        reply = {k: str(v) for k, v in want.items()}
        r = grade(skill, q, reply)
        assert r.parts["a"].marks == 0
        assert r.parts["b"].marks == r.parts["b"].max_marks and r.parts["b"].follow_through, (s, reply)
        assert r.parts["c"].marks == r.parts["c"].max_marks and r.parts["c"].follow_through, (s, reply)
        checked += 1
    assert checked >= 100


@pytest.mark.parametrize("topic,skill_id,oracle", CASES, ids=IDS)
def test_all_wrong_earns_nothing(topic, skill_id, oracle):
    skill = SKILLS[skill_id]
    for s in range(30):
        r = grade(skill, _q(topic, s), {"a": "999", "b": "998", "c": "997"})
        assert r.marks == 0 and not r.correct


@pytest.mark.parametrize("topic,skill_id,oracle", CASES, ids=IDS)
def test_variety(topic, skill_id, oracle):
    qs = [_q(topic, s) for s in range(300)]
    assert len({q.stem for q in qs}) >= 100
    assert len({q.parts[2].answer_text for q in qs}) >= 25
