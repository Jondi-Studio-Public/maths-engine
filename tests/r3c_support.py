"""Checks shared by the ``tests/test_r3c_*.py`` files (matrix models: transition, Leslie, dominance, permutation, arithmetic, costing).

Every check goes through ``mathgen.grading.grade`` (never ``GRADERS[kind]`` directly), so a question with parts is
marked part by part exactly as the engine marks it.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

from mathgen import make
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


def distinct_texts(topic: str, difficulty: str, seeds=range(1, 61)) -> int:
    return len({make(topic, seed=s, difficulty=difficulty).text for s in seeds})


def top_share(topic: str, difficulty: str, n: int = 120) -> float:
    counts = Counter(make(topic, seed=s, difficulty=difficulty).answer_text for s in range(1, n + 1))
    return counts.most_common(1)[0][1] / n


def decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def as_fraction(text: str) -> Fraction:
    return Fraction(text)


def literal(text: str):
    """The first bracketed matrix after ``label =`` in ``text``, as a list of lists of ints or floats."""
    import ast
    import re

    return ast.literal_eval(re.search(r"\[\[.*?\]\]", text).group(0))


def matrix_after(text: str, label: str):
    import ast
    import re

    return ast.literal_eval(re.search(rf"{re.escape(label)} = (\[\[.*?\]\])", text).group(1))


def as_rows(answer):
    """A matrix answer (ints or exact dicts) as lists of Fractions."""
    out = []
    for row in answer:
        out.append([Fraction(v["p"], v["q"]) if isinstance(v, dict) else Fraction(v) for v in row])
    return out


# ---------------------------------------------------------------- reading a transition model from its wording

import re

from mathgen.generators.r3c_transition_states_build import _CONTEXTS

_ALL_NAMES = sorted({n for c in _CONTEXTS for n in c["names"]}, key=len, reverse=True)


def names_in(text: str) -> list[str]:
    return [n for n in _ALL_NAMES if re.search(rf"(?<![A-Za-z]){re.escape(n)}(?![A-Za-z])", text)]


def parse_transition(text: str, unknown: Fraction | None = None):
    """``(names, T, rest)``: ``T[dest][source]`` as Fractions read from the "Of the ... " sentences of ``text``.

    An "x%" share is read as ``unknown``. ``names`` are the states, in order of first mention."""
    sentences = re.findall(r"Of the \w+ ([^,]+), ([^.]*)\.", text)
    states = [
        next(n for n in _ALL_NAMES if re.search(rf"(?<![A-Za-z]){re.escape(n)}(?![A-Za-z])", at)) for at, _ in sentences
    ]
    T = {d: {s: Fraction(0) for s in states} for d in states}
    for source, (_at, body) in zip(states, sentences):
        rest_match = re.search(r"the rest [^,;]*", body)
        main = body.split(";")[0]
        known = Fraction(0)
        stay_given = False
        for pct, item in re.findall(r"(\d+|x)% ([^,;]*?)(?=,| and (?:\d+|x)%|$)", main):
            value = unknown if pct == "x" else Fraction(int(pct), 100)
            dest = next((n for n in states if re.search(rf"(?<![A-Za-z]){re.escape(n)}(?![A-Za-z])", item)), None)
            if dest is None:
                dest, stay_given = source, True
            T[dest][source] += value
            known += value
        left = 1 - known
        if rest_match:
            dest = next(
                n for n in states if re.search(rf"(?<![A-Za-z]){re.escape(n)}(?![A-Za-z])", rest_match.group(0))
            )
            T[dest][source] += left
        elif "everyone else stays" in body or not stay_given:
            T[source][source] += left
        else:
            assert left == 0, (source, body)
    return states, T


def step(T, states, vec_by_name, k=1):
    cur = dict(vec_by_name)
    for _ in range(k):
        cur = {d: sum(T[d][s] * cur[s] for s in states) for d in states}
    return cur


def column_sums_are_one(T, states):
    return all(sum(T[d][s] for d in states) == 1 for s in states)
