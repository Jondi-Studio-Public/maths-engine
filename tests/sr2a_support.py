"""Checks shared by the ``tests/test_sr2a_*.py`` files (General Mathematics networks, bivariate data, distributions).

Every check goes through ``mathgen.grading.grade`` (never ``GRADERS[kind]`` directly), so a question with parts is
marked part by part exactly as the engine marks it.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

from mathgen import figures, make
from mathgen.grading import grade
from mathgen.grading.parts import follow_through_target, reply_value, symbol_name

SEEDS = range(1, 41)
TIERS = ("easy", "medium", "hard")


def questions(topic: str, difficulty: str, seeds=SEEDS):
    return [make(topic, seed=s, difficulty=difficulty) for s in seeds]


def check_question_grading(q) -> None:
    """The canonical answer grades correct and every distractor grades wrong, for the question or each part."""
    if q.parts:
        for p in q.parts:
            assert grade(p.answer_kind, p, p.answer_text).correct, (q.seed, p.label, p.answer_text)
            assert len(p.distractors) >= 3, (q.seed, p.label)
            assert len(set(p.distractors)) == len(p.distractors)
            for d in p.distractors:
                assert not grade(p.answer_kind, p, d).correct, (q.seed, p.label, "accepted distractor", d)
        replies = {p.label: p.answer_text for p in q.parts}
        result = grade("number", q, replies)
        assert result.correct and result.marks == result.max_marks == sum(p.marks for p in q.parts)
        return
    kind = q.answer_kind
    assert grade(kind, q, q.answer_text).correct, (q.seed, q.answer_text)
    assert len(q.distractors) >= 3, q.seed
    assert len(set(q.distractors)) == len(q.distractors)
    for d in q.distractors:
        assert not grade(kind, q, d).correct, (q.seed, "accepted distractor", d)


def check_tagged(q) -> None:
    """Every distractor of the question (or of each part) carries a catalogued misconception."""
    targets = q.parts if q.parts else [q]
    for t in targets:
        notes = t.option_notes
        for d in t.distractors:
            assert d in notes, (q.seed, getattr(t, "label", ""), "untagged distractor", d)


def check_follow_through(q) -> int:
    """A wrong earlier answer carried into a later part earns that part's marks, and only that part's.

    The marker withholds follow-through when a carried value is degenerate (a count of 0 or 1 that makes the later
    part trivial), so this counts the carried answers it did honour and returns the count; callers assert it is
    positive over their seeds."""
    honoured = 0
    by_label = {p.label: p for p in q.parts}
    for p in q.parts:
        if not p.follow_through:
            continue
        for name in _named(p):
            source = by_label[name]
            for d in source.distractors:
                values = {name: reply_value(source.answer_kind, d)}
                for other in _named(p):
                    if other != name:
                        values[other] = reply_value(by_label[other].answer_kind, by_label[other].answer_text)
                target = follow_through_target(p, values)
                if target is None or target.answer_text == p.answer_text:
                    continue
                replies = {x.label: x.answer_text for x in q.parts}
                replies[name] = d
                replies[p.label] = target.answer_text
                result = grade("number", q, replies)
                assert not result.parts[name].correct
                honoured += bool(result.parts[p.label].correct)
    return honoured


def _named(part):
    expr = part.follow_through
    return sorted({lab for lab in "abcdefgh" if symbol_name(lab) in expr})


def check_figure(q) -> None:
    assert q.figure is not None
    assert not figures.validate(q.figure)
    reader = figures.get_reader(q.topic)
    if reader is not None:
        assert reader(q) == q.answer_text or q.parts, (q.seed, reader(q), q.answer_text)


def distinct_texts(topic: str, difficulty: str, seeds=range(1, 61)) -> int:
    return len({make(topic, seed=s, difficulty=difficulty).text for s in seeds})


def top_share(topic: str, difficulty: str, n: int = 120) -> float:
    counts = Counter(make(topic, seed=s, difficulty=difficulty).answer_text for s in range(1, n + 1))
    return counts.most_common(1)[0][1] / n


def decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def as_fraction(text: str) -> Fraction:
    return Fraction(text)
