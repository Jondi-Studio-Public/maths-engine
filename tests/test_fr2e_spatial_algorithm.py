"""Tests `f10-spatial-algorithm-choose-test` (topic `fr2e_spatial_algorithm`, exam chain `exd_x_fr2e_spatial_algorithm`).

The oracle holds its own description of each construction (a pattern per step, in the right order) and finds the
construction from the stem; it judges the congruence questions with its own test of the facts, in which two known
angles also give the third, and it recomputes the exam triangle with the Pythagorean theorem.
"""

from __future__ import annotations

import math
import re

import pytest

import mathgen.generators.fr2e_spatial_algorithm_build  # noqa: F401  (registers the topics)
from tests.fr2e_support import (
    SEEDS,
    TIERS,
    assert_chain_shape,
    assert_clean_and_deterministic,
    assert_exam_grading,
    assert_follow_through,
    assert_registered,
    assert_single_grading,
    assert_tiers_differ,
    exam,
    single,
)

SKILL = "f10-spatial-algorithm-choose-test"
TOPIC = "fr2e_spatial_algorithm"
EXAM = "exd_x_fr2e_spatial_algorithm"

# One pattern per step, in the order the steps must be done.
_CONSTRUCTIONS = {
    "construct the perpendicular bisector of a segment": [
        r"Draw the segment",
        r"Open the compass to more than half",
        r"Put the compass point on \w and draw an arc above and below",
        r"Keeping the same width, put the compass point on \w and draw arcs that cross",
        r"Join the two crossing points",
    ],
    "bisect an angle": [
        r"Draw rays",
        r"Put the compass point on \w and draw an arc that crosses both rays",
        r"Keeping the same width, put the compass point on each crossing",
        r"Draw a ray from",
        r"Check with a protractor",
    ],
    "construct a triangle from its three side lengths": [
        r"Draw the side",
        r"Open the compass to \d+ cm, put the point on \w and draw an arc\.",
        r"Open the compass to \d+ cm, put the point on \w and draw an arc that crosses the first arc",
        r"Join",
    ],
    "construct the perpendicular from a point to a line": [
        r"Put the compass point on \w and draw an arc that crosses the line",
        r"Put the compass point on \w and draw an arc on the other side",
        r"Keeping the same width",
        r"Join",
    ],
    "construct a regular hexagon": [
        r"Draw a circle",
        r"Mark a point",
        r"Keep the compass",
        r"Put the compass point on \w and mark the next point",
        r"Join the six points",
    ],
    "construct an angle of 60°": [
        r"Draw a line and mark",
        r"Put the compass point on \w and draw an arc that crosses the line",
        r"Keeping the same width",
        r"Draw the ray from",
    ],
    "construct a triangle from two sides and the angle between them": [
        r"Draw the side",
        r"Put a protractor",
        r"Draw a ray from",
        r"Measure",
        r"Join",
    ],
}
_VERB = r"(?:Draw|Open|Put|Keeping|Join|Mark|Keep|Check|Measure)"
_STEP = re.compile(rf"([A-E])\. ({_VERB}\b.*?)(?= [A-E]\. {_VERB}\b| Which order| Give the letters|$)")


def _steps(text: str) -> dict[str, str]:
    found = dict(_STEP.findall(text))
    assert sorted(found) == list("ABCDE"[: len(found)]) and len(found) >= 4, text
    return found


def _canonical_position(title: str, step: str) -> int:
    hits = [i for i, pattern in enumerate(_CONSTRUCTIONS[title]) if re.match(pattern, step)]
    assert len(hits) == 1, (title, step, hits)
    return hits[0]


def _is_right_order(title: str, steps: dict[str, str], letters: list[str]) -> bool:
    return [_canonical_position(title, steps[letter]) for letter in letters] == list(range(len(steps)))


def _order_cases(tier: str):
    for seed in SEEDS:
        q = single(TOPIC, tier, seed)
        title = re.search(r"These steps to (.+?) have been shuffled", q.text).group(1)
        yield seed, q, title, _steps(q.text)


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_order_questions_have_one_right_order(tier):
    titles = set()
    for seed, q, title, steps in _order_cases(tier):
        titles.add(title)
        assert _is_right_order(title, steps, [s.strip() for s in q.answer_text.split(",")]), (
            seed,
            q.text,
            q.answer_text,
        )
        for wrong in q.distractors:
            assert not _is_right_order(title, steps, [s.strip() for s in wrong.split(",")]), (seed, wrong)
    assert len(titles) >= (3 if tier == "easy" else 4)


# -- congruence ------------------------------------------------------------------------------------------------------


def _parse_fact(text: str, n1: str, n2: str):
    if m := re.fullmatch(r"∠(\w) = ∠(\w)", text):
        assert n2.index(m.group(2)) == n1.index(m.group(1))
        return ("a", n1.index(m.group(1)))
    m = re.fullmatch(r"(\w)(\w) = (\w)(\w)", text)
    first = frozenset({n1.index(m.group(1)), n1.index(m.group(2))})
    assert first == frozenset({n2.index(m.group(3)), n2.index(m.group(4))}), text
    return ("s", first)


def _given(stem: str, n1: str, n2: str):
    known = stem.split("It is known that ")[1].split(". Which one")[0]
    facts = set()
    for item in re.split(r" and (?=∠|\w\w = )", known):
        if item.startswith("each has a right angle") or item.startswith("both have a right angle"):
            facts.add(("r",))
        elif m := re.search(r"(∠\w = ∠\w|\w\w = \w\w)$", item):
            facts.add(_parse_fact(m.group(1), n1, n2))
    return facts


def _completes(test: str, facts: set) -> bool:
    sides = {f[1] for f in facts if f[0] == "s"}
    angles = {f[1] for f in facts if f[0] == "a"}
    if ("r",) in facts:
        angles.add(1)
    if len(angles) == 2:
        angles = {0, 1, 2}
    if test == "SAS":
        return any(sum(1 for s in sides if a in s) == 2 for a in angles)
    if test == "RHS":
        return ("r",) in facts and frozenset({0, 2}) in sides and len(sides) >= 2
    raise AssertionError(test)


def test_hard_only_the_key_completes_the_named_test():
    tests = set()
    for seed in SEEDS:
        q = single(TOPIC, "hard", seed)
        n1, n2 = re.search(r"Triangles (\w{3}) and (\w{3}) are being compared", q.text).groups()
        test = re.search(r"by the (\w+) test", q.text).group(1)
        tests.add(test)
        given = _given(q.text, n1, n2)
        assert given and not _completes(test, given), (seed, q.text)
        assert _completes(test, given | {_parse_fact(q.answer_text, n1, n2)}), (seed, q.text, q.answer_text)
        for wrong in q.distractors:
            fact = _parse_fact(wrong, n1, n2)
            assert fact not in given and not _completes(test, given | {fact}), (seed, q.text, wrong)
        assert len(set(q.distractors)) == len(q.distractors) >= 3
    assert tests == {"SAS", "RHS"}


def test_registered_sources():
    assert_registered(SKILL, TOPIC, EXAM)


@pytest.mark.parametrize("tier", TIERS)
def test_tiers_differ_in_shape(tier):
    assert_tiers_differ(TOPIC)
    questions = [single(TOPIC, tier, seed) for seed in SEEDS]
    if tier == "hard":
        assert all("extra fact" in q.text and "congruent" in q.text for q in questions)
    else:
        assert all("have been shuffled" in q.text for q in questions)
        longest = max(len(_steps(q.text)) for q in questions)
        assert longest == (4 if tier == "easy" else 5)


def test_clean_and_deterministic():
    assert_clean_and_deterministic(TOPIC, TIERS)
    assert_clean_and_deterministic(EXAM, ("exam",))


def test_single_grading():
    assert_single_grading(SKILL, TOPIC)


def test_exam_matches_oracle():
    for seed in range(80):
        q = exam(EXAM, seed)
        leg1, leg2 = (int(v) for v in re.search(r"= (\d+) cm and \w\w = (\d+) cm", q.text).groups())
        a, b, c, d = q.parts
        hyp = math.isqrt(leg1**2 + leg2**2)
        assert hyp * hyp == leg1**2 + leg2**2
        assert int(a.answer_text) == hyp
        assert int(c.answer_text) == leg1 + leg2 + hyp
        # (b): the key is the hypotenuse pair, and no distractor is the hypotenuse
        names = re.search(r"Triangle (\w{3}) has a right angle at (\w)", q.text)
        n1 = names.group(1)
        hypotenuse = n1[2] + n1[0]
        assert b.answer_text.startswith(hypotenuse + " = "), (seed, b.answer_text)
        assert all(not wrong.startswith(hypotenuse + " = ") for wrong in b.distractors)
        # (d): the construction is draw the leg, make the perpendicular, arc of length (a), join
        steps = dict(
            re.findall(
                r"([A-D])\. ((?:Draw|At|Open|Join)\b.*?)(?= [A-D]\. (?:Draw|At|Open|Join)\b| Give the letters)",
                d.prompt,
            )
        )
        order = [s.strip() for s in d.answer_text.split(",")]
        expected = [r"Draw the side", r"At \w, construct a perpendicular", r"Open the compass", r"Join"]
        assert all(re.match(pattern, steps[k]) for pattern, k in zip(expected, order)), (seed, d.prompt)


def test_exam_chain_shape():
    assert_chain_shape(EXAM, ["number", "mcq", "number", "ordered_list"], [2, 2, 1, 3], [False, False, True, False])


def test_exam_grading():
    assert_exam_grading(SKILL, EXAM)


def test_perimeter_follows_through_from_a_wrong_hypotenuse():
    def perimeter(q, canon, wrong_hyp):
        leg1, leg2 = (int(v) for v in re.search(r"= (\d+) cm and \w\w = (\d+) cm", q.text).groups())
        return str(leg1 + leg2 + int(wrong_hyp))

    assert_follow_through(SKILL, EXAM, "c", "a", perimeter)
