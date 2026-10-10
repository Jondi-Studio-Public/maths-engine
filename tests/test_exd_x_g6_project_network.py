"""Tests for the exam-tier activity-network sources (``mathgen/generators/exd_x_g6_project_network.py``).

Each key is re-derived from the RENDERED PROMPT alone, by enumerating every start-to-finish route (no forward or
backward pass), and every distractor is run through the skill's real grader.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.core import mcq_options
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPICS = {
    "delay": ("exd_x_g6_delay_propagation", "p10c_critical_path_float", "hard", "number"),
    "crash": (
        "exd_x_g6_crash_shortest_time",
        "use_the_critical_path_to_determine_the_minimum",
        "twopath_completion",
        "number",
    ),
    "missing": (
        "exd_x_g6_missing_duration",
        "use_forward_and_backward_scanning_to_determine_the",
        "nonchain_lst",
        "number",
    ),
    "change": ("exd_x_g6_critical_path_after_change", "p10c_critical_path_locate", "locate_count", "mcq"),
}
_ACTIVITY = re.compile(
    r"Activity ([A-Z]): duration (\d+|unknown)(?: days)?, (?:no predecessors|predecessors: ([A-Z, ]+))\."
)


def _exam(key, seed):
    return make(TOPICS[key][0], seed=seed, difficulty="exam")


def _graph(text):
    """({activity: [predecessors]}, {activity: duration or None}) read from the prompt."""
    preds, dur = {}, {}
    for a, d, p in _ACTIVITY.findall(text):
        preds[a] = [x.strip() for x in p.split(",")] if p else []
        dur[a] = None if d == "unknown" else int(d)
    return preds, dur


def _routes(preds):
    """Every start-to-finish route, built backwards from the activities nobody depends on."""
    depended = {p for ps in preds.values() for p in ps}
    out = []

    def back(path):
        first = path[0]
        if not preds[first]:
            out.append(tuple(path))
            return
        for p in preds[first]:
            back([p] + path)

    for a in preds:
        if a not in depended:
            back([a])
    return out


def _longest(preds, dur):
    return max(sum(dur[a] for a in r) for r in _routes(preds))


# -- shapes and text --------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", TOPICS)
def test_variety_and_new_shapes_over_hard_tier(key):
    topic, hard_topic, hard_tier, _kind = TOPICS[key]
    exam = {prompt_shape(_exam(key, s)) for s in range(60)}
    hard = {prompt_shape(make(hard_topic, seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert len(exam - hard) >= 3, (len(exam), len(hard))


@pytest.mark.parametrize("key", TOPICS)
def test_no_text_defects(key):
    for s in range(60):
        q = _exam(key, s)
        assert find_text_defects(q) == [], (s, q.text)


@pytest.mark.parametrize("key", TOPICS)
def test_deterministic(key):
    for s in (0, 1, 7, 99):
        assert _exam(key, s).to_json() == _exam(key, s).to_json()


# -- independent oracles ----------------------------------------------------------------------------------


def test_delay_oracle():
    bad = []
    for s in range(300):
        q = _exam("delay", s)
        preds, dur = _graph(q.text)
        x, t = re.search(r"Activity ([A-Z]) overruns its stated duration by (\d+) days", q.text).groups()
        late = dict(dur)
        late[x] += int(t)
        if _longest(preds, late) - _longest(preds, dur) != q.answer or q.answer <= 0:
            bad.append((s, q.answer))
    assert bad == []


def test_crash_oracle():
    bad = []
    for s in range(300):
        q = _exam("crash", s)
        preds, dur = _graph(q.text)
        x, r = re.search(r"activity ([A-Z]) takes (\d+) days less", q.text).groups()
        new = dict(dur)
        new[x] -= int(r)
        old_len, new_len = _longest(preds, dur), _longest(preds, new)
        if new_len != q.answer or not 0 < old_len - new_len < int(r):
            bad.append((s, q.answer, new_len))
    assert bad == []


def test_missing_duration_oracle():
    bad = []
    for s in range(300):
        q = _exam("missing", s)
        preds, dur = _graph(q.text)
        m, target = re.search(r"earliest start time of activity ([A-Z]) is day (\d+)", q.text).groups()
        (u,) = [a for a, d in dur.items() if d is None]
        sols = []
        for guess in range(1, 60):
            trial = dict(dur)
            trial[u] = guess
            # earliest start of m = longest chain of activities strictly before m
            best = 0
            for r in _routes(preds):
                if m in r:
                    best = max(best, sum(trial[a] for a in r[: r.index(m)]))
            if best == int(target):
                sols.append(guess)
        if sols != [q.answer]:
            bad.append((s, sols, q.answer))
    assert bad == []


def test_critical_path_after_change_oracle():
    bad = []
    for s in range(300):
        q = _exam("change", s)
        preds, dur = _graph(q.text)
        x, r = re.search(r"Activity ([A-Z]) is shortened by (\d+) days", q.text).groups()
        old = sorted(_routes(preds), key=lambda rt: -sum(dur[a] for a in rt))
        new_dur = dict(dur)
        new_dur[x] -= int(r)
        ranked = sorted(_routes(preds), key=lambda rt: -sum(new_dur[a] for a in rt))
        unique = sum(new_dur[a] for a in ranked[0]) > sum(new_dur[a] for a in ranked[1])
        if not unique or "-".join(ranked[0]) != q.answer_text or ranked[0] == old[0]:
            bad.append((s, q.answer_text))
    assert bad == []


# -- the real grader --------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", TOPICS)
def test_grader_accepts_key_and_rejects_distractors(key):
    kind = TOPICS[key][3]
    grade = GRADERS[kind]
    bad = []
    for s in range(120):
        q = _exam(key, s)
        if kind == "mcq":
            opts = mcq_options(q)
            ok = grade(q, str(opts.index(q.answer_text))).correct
            rejected = [d for d in q.distractors if grade(q, str(opts.index(d))).correct]
        else:
            ok = grade(q, q.answer_text).correct
            rejected = [d for d in q.distractors if grade(q, d).correct]
        if not ok or rejected or len(q.distractors) < 2:
            bad.append((s, ok, rejected))
        if kind == "number":
            for wrong in (q.answer + 1, q.answer - 1, 0 if q.answer else 1):
                if wrong != q.answer and grade(q, str(wrong)).correct:
                    bad.append((s, "wrong accepted", wrong))
    assert bad == []
