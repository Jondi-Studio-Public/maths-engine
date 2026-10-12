"""Tests for the two activity-network exam sources in `exd_x_r2e_network_chain`:

* `exd_x_r2e_network_crash` for `senior-use-ests-and-lsts-to-locate-the-critical`;
* `exd_x_r2e_network_missing` for `senior-use-forward-and-backward-scanning-to-determine-the`.

Every key is re-derived from the rendered stem and part prompts with an independent route enumeration."""

from __future__ import annotations

import collections
import re

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

CRASH = "exd_x_r2e_network_crash"
MISSING = "exd_x_r2e_network_missing"
CRASH_SKILL = "senior-use-ests-and-lsts-to-locate-the-critical"
MISSING_SKILL = "senior-use-forward-and-backward-scanning-to-determine-the"
SEEDS = range(300)

_SKILLS = {s.id: s for s in load_skills()[1] if s.id in (CRASH_SKILL, MISSING_SKILL)}
_ACTIVITY = re.compile(
    r"Activity ([A-I]): duration (not known|\d+ days), (?:predecessors: ([A-I](?:, [A-I])*)|no predecessors)\."
)


# -- independent network helpers --------------------------------------------------------------------------------


def parse(stem: str):
    """(predecessors, durations, hidden): durations in days, the hidden one None."""
    preds, dur, hidden = {}, {}, None
    for name, d, ps in _ACTIVITY.findall(stem):
        preds[name] = ps.split(", ") if ps else []
        if d == "not known":
            hidden, dur[name] = name, None
        else:
            dur[name] = int(d.split()[0])
    return preds, dur, hidden


def all_routes(preds):
    nxt = collections.defaultdict(list)
    for a, ps in preds.items():
        for p in ps:
            nxt[p].append(a)
    out = []

    def walk(path):
        if not nxt[path[-1]]:
            out.append(path)
        for s in nxt[path[-1]]:
            walk(path + [s])

    for a in preds:
        if not preds[a]:
            walk([a])
    return out


def route_len(route, dur):
    return sum(dur[a] for a in route)


def crash_oracle(q):
    preds, dur, _ = parse(q.stem)
    y, r = re.search(r"activity ([A-I]) takes (\d+) days less", q.stem).groups()
    r = int(r)
    routes = all_routes(preds)
    ranked = sorted(routes, key=lambda rt: -route_len(rt, dur))
    total = route_len(ranked[0], dur)
    assert route_len(ranked[1], dur) < total  # a unique critical path
    rival = max(route_len(rt, dur) for rt in routes if y not in rt)
    shortened = dict(dur, **{y: dur[y] - r})
    new = max(route_len(rt, shortened) for rt in routes)
    return {"a": ", ".join(ranked[0]), "b": str(total), "c": str(rival), "d": str(new)}, y, r


def passes(preds, dur):
    est, eft = {}, {}
    for a in preds:
        est[a] = max((eft[p] for p in preds[a]), default=0)
        eft[a] = est[a] + dur[a]
    finish = max(eft.values())
    nxt = collections.defaultdict(list)
    for a, ps in preds.items():
        for p in ps:
            nxt[p].append(a)
    lst = {}
    for a in reversed(list(preds)):
        lft = min((lst[s] for s in nxt[a]), default=finish)
        lst[a] = lft - dur[a]
    return est, finish, lst


def missing_oracle(q, d_u=None):
    """Answers a-d from the stem; `d_u` replaces the recovered duration (a wrong (a) carried forward)."""
    preds, dur, hidden = parse(q.stem)
    v, start_v = re.search(r"earliest start time of activity ([A-I]) is (\d+) days", q.stem).groups()
    z = re.search(r"latest start time of activity ([A-I])", q.parts[2].prompt).group(1)
    w = re.search(r"total float of activity ([A-I])", q.parts[3].prompt).group(1)
    probe = dict(dur, **{hidden: 0})
    est0, _, _ = passes(preds, probe)  # the start of the hidden activity does not depend on its own duration
    true_u = int(start_v) - est0[hidden]
    use = true_u if d_u is None else d_u
    full = dict(dur, **{hidden: use})
    est, finish, lst = passes(preds, full)
    float_w = finish - max(route_len(rt, full) for rt in all_routes(preds) if w in rt)
    assert float_w == lst[w] - est[w]
    return {"a": str(true_u), "b": str(finish), "c": str(lst[z]), "d": str(float_w)}


# -- fixtures and shared checks ----------------------------------------------------------------------------------

CASES = [
    pytest.param(CRASH, CRASH_SKILL, "p10c_critical_path_locate", [2, 1, 2, 3], id="crash"),
    pytest.param(
        MISSING, MISSING_SKILL, "use_forward_and_backward_scanning_to_determine_the", [2, 2, 2, 2], id="missing"
    ),
]


@pytest.fixture(scope="module")
def made():
    return {t: [make(t, seed=s) for s in SEEDS] for t in (CRASH, MISSING)}


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


@pytest.mark.parametrize(("topic", "skill_id", "hard_topic", "marks"), CASES)
def test_wired_to_the_skill(topic, skill_id, hard_topic, marks):
    exam = [s for s in _SKILLS[skill_id].sources if s.difficulty == "exam"]
    assert [(s.topic, s.params, s.stages) for s in exam] == [(topic, {"tier": "exam"}, ["exam"])]


@pytest.mark.parametrize(("topic", "skill_id", "hard_topic", "marks"), CASES)
def test_shape_and_variety(made, topic, skill_id, hard_topic, marks):
    for q in made[topic]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == marks
        assert all(
            p.answer_kind == ("ordered_list" if (topic == CRASH and p.label == "a") else "number") for p in q.parts
        )
        assert any(p.follow_through for p in q.parts)
    shapes = {prompt_shape(make(topic, seed=s)) for s in range(60)}
    hard = {prompt_shape(make(hard_topic, seed=s, difficulty="hard")) for s in range(60)}
    assert len(shapes) >= 3, shapes
    assert shapes - hard
    for q in made[topic][:60]:
        assert find_text_defects(q) == []


@pytest.mark.parametrize(("topic", "skill_id", "hard_topic", "marks"), CASES)
def test_answer_space(made, topic, skill_id, hard_topic, marks):
    tuples = collections.Counter(tuple(p.answer_text for p in q.parts) for q in made[topic])
    assert len(tuples) >= 100
    assert max(tuples.values()) / len(made[topic]) <= 0.05


@pytest.mark.parametrize(("topic", "skill_id", "hard_topic", "marks"), CASES)
def test_composed_answer_full_marks(made, topic, skill_id, hard_topic, marks):
    for q in made[topic][:100]:
        result = grade(_SKILLS[skill_id], q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(marks)


@pytest.mark.parametrize(("topic", "skill_id", "hard_topic", "marks"), CASES)
def test_distractors_rejected_and_tagged(made, topic, skill_id, hard_topic, marks):
    for q in made[topic][:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (p.label, d)
                result = grade(_SKILLS[skill_id], q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


@pytest.mark.parametrize(("topic", "skill_id", "hard_topic", "marks"), CASES)
def test_deterministic(topic, skill_id, hard_topic, marks):
    assert make(topic, seed=5).to_dict() == make(topic, seed=5).to_dict()


# -- the crash chain -----------------------------------------------------------------------------------------------


def test_crash_oracle_agrees(made):
    for q in made[CRASH]:
        expected, _y, _r = crash_oracle(q)
        assert expected == _canonical(q), q.text


def test_crash_both_outcomes_occur(made):
    """Sometimes the shortening is used in full and sometimes a rival route takes over: the learner chooses."""
    full, capped = 0, 0
    for q in made[CRASH]:
        expected, _y, r = crash_oracle(q)
        if int(expected["d"]) == int(expected["b"]) - r:
            full += 1
        else:
            capped += 1
    assert full >= 60 and capped >= 60


def test_crash_follow_through(made):
    skill = _SKILLS[CRASH_SKILL]
    for q in made[CRASH][:100]:
        canon = _canonical(q)
        _expected, _y, r = crash_oracle(q)
        total, rival = int(canon["b"]), int(canon["c"])
        # A wrong completion time carried into (d).
        b_wrong = total + 3
        d_wrong = max(rival, b_wrong - r)
        result = grade(skill, q, {**canon, "b": str(b_wrong), "d": str(d_wrong)})
        assert result.parts["b"].marks == 0
        assert result.parts["d"].marks == 3
        # A wrong rival length carried into (d).
        c_wrong = rival + 2
        d_wrong = max(c_wrong, total - r)
        if d_wrong not in (c_wrong, total):
            result = grade(skill, q, {**canon, "c": str(c_wrong), "d": str(d_wrong)})
            assert result.parts["c"].marks == 0
            assert result.parts["d"].marks == 3
            assert result.parts["d"].follow_through or str(d_wrong) == canon["d"]


# -- the missing-duration chain ------------------------------------------------------------------------------------


def test_missing_oracle_agrees(made):
    for q in made[MISSING]:
        assert missing_oracle(q) == _canonical(q), q.text


def test_missing_follow_through(made):
    skill = _SKILLS[MISSING_SKILL]
    checked = 0
    for q in made[MISSING][:120]:
        canon = _canonical(q)
        wrong = missing_oracle(q, d_u=int(canon["a"]) + 2)
        replies = {"a": wrong["a"], "b": wrong["b"], "c": wrong["c"], "d": wrong["d"]}
        replies["a"] = str(int(canon["a"]) + 2)
        earlier = {replies["a"], replies["b"]}
        if replies["c"] in earlier or replies["d"] in earlier:
            continue  # a reply that only repeats an earlier answer it uses earns no follow-through
        result = grade(skill, q, replies)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label, replies)
        checked += 1
    assert checked >= 80


def test_all_wrong_earns_zero(made):
    for topic, skill_id in ((CRASH, CRASH_SKILL), (MISSING, MISSING_SKILL)):
        for q in made[topic][:40]:
            result = grade(_SKILLS[skill_id], q, {"a": "Z", "b": "-5", "c": "-7", "d": "-9"})
            assert result.marks == 0 and not result.correct
