"""`grade_with_deadline`: the same verdicts as `grade()`, in a worker process that is killed when it runs over.

The pool here uses one or two spawned workers; each takes a second or two to import the engine, so the module
shares one pool and the tests that need their own make small ones.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from mathgen import engine
from mathgen.grading import grade
from mathgen.grading.deadline import DeadlineResult, GradingPool

# One live skill per answer kind, so every grader goes through the worker at least once.
_SKILLS = {
    "number": "cyclic-group-element-order",
    "mcq": "integer-polynomial-quotient-remainder",
    "ordered_list": "permutation-composition",
    "expression": "senior-determine-f-left-x-right-given-f-x",
    "pair": "ucv1-inverse-regularised-solve",
    "fraction": "senior-demonstrate-an-understanding-of-the-elementary-ideas-and",
    "list": "stationary-points",
}
_SLACK = 3.0  # seconds a timed-out call may take beyond its limit: killing and joining a process, a busy runner


def _sleepy_grade(kind, question, given, *, skill=None):
    """A grader that never finishes in time: a worker imports it by name, so it lives at module level."""
    time.sleep(60)


@pytest.fixture(scope="module")
def pool():
    with GradingPool(2, timeout=2.0) as p:
        yield p


@pytest.fixture(scope="module")
def skills():
    by_id = {s.id: s for s in engine.load_skills()[1]}
    return {kind: by_id[sid] for kind, sid in _SKILLS.items()}


def _learner_replies(question) -> list:
    """What people type: the answer as given, with a unicode minus, an `x =` prefix, a unit, extra spaces, and
    each wrong option."""
    answer = question.answer_text
    return [
        answer,
        answer.replace("-", "−"),
        f"x = {answer}",
        f"{answer} cm",
        f"  {answer}  ",
        answer.replace(" ", ""),
        *question.distractors,
        "",
        "I don't know",
    ]


def test_every_kind_gets_the_same_verdict_as_grade(pool, skills):
    checked = 0
    for kind, skill in skills.items():
        for seed in range(3):
            q = engine.make_for_skill(skill, seed=seed)
            replies = [0, 1, 2] if kind == "mcq" else _learner_replies(q)
            for given in replies:
                expected = grade(kind, q, given, skill=skill)
                got = pool.grade(kind, q, given, skill=skill)
                assert isinstance(got, DeadlineResult) and not got.timed_out
                assert (got.correct, got.canonical, got.note) == (
                    expected.correct,
                    expected.canonical,
                    expected.note,
                ), (
                    kind,
                    seed,
                    given,
                )
                checked += 1
    assert checked > 100


def test_a_reply_that_never_finishes_is_graded_wrong_on_time():
    q = engine.make_for_skill({s.id: s for s in engine.load_skills()[1]}[_SKILLS["number"]], seed=0)
    with GradingPool(1, timeout=1.0, grade_fn=_sleepy_grade) as slow:
        slow._idle.queue[0].wait_ready()  # start-up out of the clock
        started = time.monotonic()
        result = slow.grade("number", q, q.answer_text)
        assert time.monotonic() - started < 1.0 + _SLACK
    assert result.timed_out and not result.correct
    assert result.canonical == q.answer_text


def test_a_slow_real_reply_cannot_hang_the_call(pool, skills):
    """`sqrt(acos(x)^2)` sends `simplify` through `Abs` and `arg` and did not finish (safe_parse.py's docstring).
    Whatever the grader makes of it, the call returns within the limit and the pool keeps working."""
    skill = skills["expression"]
    q = engine.make_for_skill(skill, seed=1)
    pool.grade("expression", q, q.answer_text, skill=skill, timeout=60)  # both workers' start-up out of the clock
    started = time.monotonic()
    result = pool.grade("expression", q, "sqrt(acos(x)^2)", skill=skill, timeout=1.0)
    assert time.monotonic() - started < 1.0 + _SLACK
    assert not result.correct
    after = pool.grade("expression", q, q.answer_text, skill=skill, timeout=60)
    assert after.correct and not after.timed_out


def test_a_worker_killed_from_outside_is_replaced(skills):
    skill = skills["number"]
    q = engine.make_for_skill(skill, seed=0)
    with GradingPool(1, timeout=30) as one:
        assert one.grade("number", q, q.answer_text, skill=skill).correct
        worker = one._idle.queue[0]
        worker.process.kill()
        worker.process.join()
        lost = one.grade("number", q, q.answer_text, skill=skill)
        assert not lost.correct and not lost.timed_out
        assert one.grade("number", q, q.answer_text, skill=skill).correct


def test_an_unknown_kind_raises_as_grade_does(pool, skills):
    q = engine.make_for_skill(skills["number"], seed=0)
    with pytest.raises(KeyError, match="unknown answer kind"):
        pool.grade("no-such-kind", q, "1", timeout=60)
    assert pool.grade("number", q, q.answer_text, timeout=60).correct


def test_threads_share_the_pool(pool, skills):
    skill = skills["number"]
    questions = [engine.make_for_skill(skill, seed=s) for s in range(8)]
    with ThreadPoolExecutor(4) as threads:
        results = list(
            threads.map(lambda q: pool.grade("number", q, q.answer_text, skill=skill, timeout=60), questions)
        )
    assert all(r.correct for r in results)


def test_engine_grade_with_deadline_matches_engine_grade(skills):
    skill = skills["expression"]
    q = engine.make_for_skill(skill, seed=2)
    for given in (q.answer_text, "x", *q.distractors):
        assert engine.grade_with_deadline(skill, q, given, timeout=60).correct == engine.grade(skill, q, given).correct


def test_a_closed_pool_refuses_work(skills):
    q = engine.make_for_skill(skills["number"], seed=0)
    p = GradingPool(1)
    p.close()
    with pytest.raises(RuntimeError, match="closed"):
        p.grade("number", q, "1")
