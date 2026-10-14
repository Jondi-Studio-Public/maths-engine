"""Exam chain of `combinations`: all selections, those meeting a condition, and the percentage chance."""

from __future__ import annotations

import itertools
import math
import re
from fractions import Fraction

import pytest

from tests import exd_x_r3d_support as S

TOPIC = "exd_x_r3d_combinations_selection"
SKILL = S.load_skill("combinations")


def _half_up(x: Fraction, places: int) -> Fraction:
    scale = 10**places
    return Fraction(math.floor(x * scale + Fraction(1, 2)), scale)


def _count(q):
    """Brute-force the number of selections and of those meeting the condition, read from the rendered text."""
    stem, cond = q.stem, q.parts[1].prompt
    r = int(re.search(r"of (\d+) is chosen", stem).group(1))
    both = re.search(r"from (\d+) \w+\. Two of the", stem)
    if both:
        n = int(both.group(1))
        ok = lambda s: 0 in s and 1 in s  # noqa: E731
    else:
        x, y = (int(v) for v in re.search(r"from (\d+) \w+ and (\d+) \w+\.", stem).groups())
        n = x + y  # indices below x are the first group, the rest the second
        exact = re.search(r"have exactly (\d+)", cond)
        atmost = re.search(r"have at most (\d+)", cond)
        if exact:
            ok = lambda s: sum(1 for i in s if i >= x) == int(exact.group(1))  # noqa: E731
        elif atmost:
            ok = lambda s: sum(1 for i in s if i >= x) <= int(atmost.group(1))  # noqa: E731
        else:
            ok = lambda s: 0 < sum(1 for i in s if i >= x) < r  # noqa: E731
    selections = list(itertools.combinations(range(n), r))
    return len(selections), sum(1 for s in selections if ok(s))


def oracle(q):
    total, fav = _count(q)
    pct = _half_up(100 * Fraction(fav, total), 1)
    return {"a": str(total), "b": str(fav), "c": f"{float(pct):.1f}"}


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]
        assert all(p.answer_kind == "number" for p in q.parts)
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_every_condition_occurs(questions):
    prompts = " ".join(q.parts[1].prompt for q in questions)
    for needle in ("exactly", "at most", "at least one of the", "both twins"):
        assert needle in prompts


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_count(questions):
    def chain(q):
        total_wrong = int(q.parts[0].distractors[0])  # the number of ordered selections
        fav = int(q.parts[1].answer_text)
        pct = _half_up(100 * Fraction(fav, total_wrong), 1)
        text = f"{float(pct):.1f}"
        if not 0 < pct < 100 or text == q.parts[2].answer_text:
            return None
        return {"a": str(total_wrong), "b": str(fav), "c": text}, {"b", "c"}

    S.check_follow_through(SKILL, questions, chain, n=200)


def test_follow_through_from_wrong_favourable(questions):
    seen = 0
    for q in questions[:100]:
        total = int(q.parts[0].answer_text)
        wrong = int(q.parts[1].distractors[0])
        pct = _half_up(100 * Fraction(wrong, total), 1)
        if not 0 < pct < 100 or float(pct) == float(q.parts[2].answer_text):
            continue
        result = S.grade(SKILL, q, {"a": str(total), "b": str(wrong), "c": f"{float(pct):.1f}"})
        assert result.parts["a"].marks == q.parts[0].marks
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks
        seen += 1
    assert seen >= 40


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "99999", "b": "99998", "c": "7.7"})


def test_deterministic():
    S.check_deterministic(TOPIC)
