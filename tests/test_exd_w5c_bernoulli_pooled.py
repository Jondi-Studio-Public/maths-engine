"""Bernoulli-mle pooled-sample hard tier."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.grading import GRADERS
from mathgen import make
from mathgen.text_quality import prompt_shape

_PATTERN = re.compile(
    r"Batch 1 has (\d+) success(?:es)? in (\d+) independent Bernoulli trials and batch 2 has "
    r"(\d+) success(?:es)? in (\d+) independent trials"
)


def _pooled(seed):
    return make("bernoulli_mle", seed=seed, difficulty="hard", kind="pooled")


def test_pooled_key_matches_recomputation_from_prompt():
    for seed in range(60):
        question = _pooled(seed)
        match = _PATTERN.search(question.text)
        assert match, question.text
        k1, n1, k2, n2 = map(int, match.groups())
        assert "Find the MLE of p from the combined data." in question.text
        assert 1 <= k1 < n1 and 1 <= k2 < n2
        expected = Fraction(k1 + k2, n1 + n2)
        assert expected != Fraction(k1, n1) and expected != Fraction(k2, n2)
        assert expected != (Fraction(k1, n1) + Fraction(k2, n2)) / 2
        assert question.answer == {"p": expected.numerator, "q": expected.denominator}
        assert Fraction(question.answer_text) == expected


def test_default_path_unchanged_and_pooled_text_differs():
    for seed in range(20):
        for difficulty in ("easy", "medium", "hard"):
            a = make("bernoulli_mle", seed=seed, difficulty=difficulty)
            b = make("bernoulli_mle", seed=seed, difficulty=difficulty, kind=None)
            assert (a.text, a.answer, a.distractors) == (b.text, b.answer, b.distractors)
            assert "In " in a.text and "Batch 1" not in a.text
        assert make("bernoulli_mle", seed=seed, difficulty="hard").text != _pooled(seed).text


def test_pooled_distractors_clear_the_real_grader():
    grader = GRADERS["fraction"]
    for seed in range(60):
        question = _pooled(seed)
        assert grader(question, question.answer_text).correct
        assert len(question.distractors) == 3
        for distractor in question.distractors:
            assert not grader(question, distractor).correct, (seed, distractor, question.answer_text)


def test_all_candidates_collision_free_by_construction():
    # Every one of the five misconception values differs from the answer and each other.
    for seed in range(60):
        k1, n1, k2, n2 = map(int, _PATTERN.search(_pooled(seed).text).groups())
        answer = Fraction(k1 + k2, n1 + n2)
        candidates = [
            (Fraction(k1, n1) + Fraction(k2, n2)) / 2,
            Fraction(k1 + k2, n1),
            Fraction(k1 + k2, n2),
            Fraction(n1 + n2 - k1 - k2, n1 + n2),
            Fraction(k1 + k2, n1 + n2 + 1),
        ]
        assert len(set(candidates + [answer])) == 6


def test_pooled_prompt_shape_differs_from_default():
    default = make("bernoulli_mle", seed=1, difficulty="medium")
    assert prompt_shape(default) != prompt_shape(_pooled(1))
