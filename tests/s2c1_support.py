"""Shared checks for the second-year ODE skills built by worker s2c1 (``tests/test_s2c1_*.py``)."""

from __future__ import annotations

import functools

import pytest

import mathgen.generators  # noqa: F401  (registers every generator)
from mathgen import make
from mathgen.grading import grade
from mathgen.grading.parts import follow_through_target, reply_value

SEEDS = range(1, 9)


@pytest.fixture(autouse=True)
def second_route():
    """Switch the generators' numerical second-route checks on, so every drawn seed is still verified here."""
    from mathgen.generators import _s2c1_kit

    _s2c1_kit.SECOND_ROUTE = True
    yield
    _s2c1_kit.SECOND_ROUTE = False


TIERS = ("easy", "medium", "hard")


def tier_topic(name: str, tier: str) -> str:
    return f"exd_x_s2c1_{name}" if tier == "exam" else f"s2c1_{name}"


@functools.lru_cache(maxsize=None)
def _one(name: str, tier: str, seed: int):
    return make(tier_topic(name, tier), seed=seed, difficulty=tier)


def questions(name: str, tier: str, seeds=SEEDS):
    return [_one(name, tier, s) for s in seeds]


def _flat(q):
    """(label, kind, answer text, distractor texts) for a question or each of its parts."""
    if q.parts:
        return [(p.label, p.answer_kind, p.answer_text, [str(d) for d in p.distractors]) for p in q.parts]
    return [("", q.answer_kind, q.answer_text, [str(d) for d in q.distractors])]


def check_grading(q) -> None:
    """The canonical answer is marked right and every distractor wrong, part by part."""
    if q.parts:
        right = {p.label: p.answer_text for p in q.parts}
        result = grade(q.parts[0].answer_kind, q, right)
        assert result.correct, (q.seed, right, result)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.seed, p.label)
            for d in p.distractors:
                wrong = dict(right, **{p.label: str(d)})
                assert not grade(q.parts[0].answer_kind, q, wrong).correct, (q.seed, p.label, str(d))
        return
    assert grade(q.answer_kind, q, q.answer_text).correct, (q.seed, q.answer_text)
    assert len(q.distractors) >= 2, q.seed
    for d in q.distractors:
        assert not grade(q.answer_kind, q, str(d)).correct, (q.seed, str(d), q.answer_text)


def check_follow_through(q) -> None:
    """Each follow-through part reproduces its own canonical answer from the earlier parts' answers."""
    values = {}
    for p in q.parts:
        if p.follow_through:
            target = follow_through_target(p, values)
            assert target is not None, (q.seed, p.label)
            assert grade(p.answer_kind, target, p.answer_text).correct or p.answer_kind == "number", (q.seed, p.label)
        values[p.label] = reply_value(p.answer_kind, p.answer_text)


def check_text(q) -> None:
    from mathgen.text_quality import find_text_defects

    assert not find_text_defects(q, include_working=True), (q.seed, find_text_defects(q, include_working=True))
    texts = [q.text] + [p.prompt for p in q.parts]
    for t in texts:
        assert t and t == t.strip()
        assert "  " not in t and "None" not in t and "{" not in t and "}" not in t, t
        assert "^1 " not in t and "(1t)" not in t, t
        assert t.count("(") == t.count(")"), t
