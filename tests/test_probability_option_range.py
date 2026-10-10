"""Every wrong option for a probability-valued question is itself a probability.

An option above 1 or below 0 (a count, a reciprocal, a negated value) is eliminated at sight, so it is a
giveaway rather than a distractor. For each fixed topic and tier, over seeds 0..59 and the skill's real
source params, every distractor must be a value in [0, 1], the option count must be the count the generator
offered before the fix, and no distractor may equal the answer by value.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

import mathgen
from mathgen.skills import load_skills

# (topic, difficulty) -> distractors per question (the count before the range fix; the fix keeps it).
EXPECTED_COUNT = {
    ("probability_complements_compounds", "easy"): 3,
    ("probability_complements_compounds", "medium"): 3,
    ("probability_complements_compounds", "hard"): 3,
    ("finite_sample_space_probability", "easy"): 3,
    ("finite_sample_space_probability", "medium"): 3,
    ("finite_sample_space_probability", "hard"): 3,
    ("markov_two_state_stationary", "easy"): 3,
    ("markov_two_state_stationary", "medium"): 3,
    ("markov_two_state_stationary", "hard"): 3,
    ("markov_two_state_step", "easy"): 3,
    ("markov_two_state_step", "medium"): 3,
    ("markov_two_state_step", "hard"): 3,
    ("traffic_light_probability", "easy"): 2,
    ("probability_complement_union", "medium"): 2,
    ("probability_complement_union", "hard"): 2,
    ("p13c_exponential_pdf_numint", "easy"): 2,
    ("p13c_exponential_pdf_numint", "medium"): 3,
    ("p13c_exponential_pdf_numint", "hard"): 3,
    ("p13c_exponential_pdf_numint", "exam"): 3,
    ("p16a_relative_frequency_estimate", "easy"): 3,
    ("p16a_relative_frequency_estimate", "medium"): 3,
    ("p16a_relative_frequency_estimate", "hard"): 3,
    ("p19b_coefficient_determination", "easy"): 3,
    ("p12f_markov_hitting", "exam"): 4,
    ("p12e_poisson_count", "easy"): 3,
    ("p12e_poisson_count", "medium"): 4,
}
# The poisson count is 3 on easy only when k = 1 drops "omits-factorial"; those seeds offer one fewer.
VARIABLE_COUNT = {("p12e_poisson_count", "easy"), ("p12e_poisson_count", "medium")}

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def _cases():
    seen = set()
    for skill in load_skills()[1]:
        for src in skill.sources:
            key = (src.topic, src.difficulty)
            if key in EXPECTED_COUNT and (key, tuple(sorted(src.params.items()))) not in seen:
                seen.add((key, tuple(sorted(src.params.items()))))
                params = {k: tuple(v) if isinstance(v, list) else v for k, v in src.params.items()}
                yield pytest.param(src.topic, src.difficulty, params, id=f"{src.topic}-{src.difficulty}")


def test_every_fixed_topic_is_covered():
    covered = {(c.values[0], c.values[1]) for c in _cases()}
    assert covered == set(EXPECTED_COUNT)


@pytest.mark.parametrize(("topic", "difficulty", "params"), list(_cases()))
def test_distractors_are_probabilities(topic, difficulty, params):
    expected = EXPECTED_COUNT[(topic, difficulty)]
    for seed in range(60):
        q = mathgen.make(topic, seed=seed, difficulty=difficulty, **params)
        answer = Fraction(q.answer_text)
        assert 0 <= answer <= 1
        if (topic, difficulty) in VARIABLE_COUNT:
            assert len(q.distractors) >= expected - 1, (seed, q.distractors)
        else:
            assert len(q.distractors) == expected, (seed, q.distractors)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.distractors)
        for d in q.distractors:
            assert _NUMBER.fullmatch(d.strip()), (seed, d)
            value = Fraction(d)
            assert 0 <= value <= 1, (topic, difficulty, seed, q.answer_text, d)
            assert value != answer, (topic, difficulty, seed, q.answer_text, d)
