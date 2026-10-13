"""No "cannot be determined" filler in the five skills that carried one (round-2 scope section (e)).

In each, "cannot be determined" was never the key, so it was a free elimination. The replacements are genuine
misconceptions, and every question still offers a key plus at least two distractors.
"""

import pytest

from mathgen import make

CASES = [
    ("propositional_formula_equivalence", ["easy", "medium", "hard"]),
    ("exd_x_s3_interp", ["exam"]),
    ("interpolation_extrapolation_classify", ["easy", "medium", "hard"]),
    ("p12c_estimate_limit_ah", ["easy", "medium", "hard", "exam"]),
    ("implication_transform_identify", ["medium", "hard"]),
    ("quantifier_forall_exists_evaluate", ["easy", "medium"]),
]
FILLER = ("cannot be determined", "not well-defined", "cannot be decided")


@pytest.mark.parametrize("topic,difficulty", [(t, d) for t, ds in CASES for d in ds])
def test_no_filler_option(topic, difficulty):
    for seed in range(80):
        q = make(topic, seed=seed, difficulty=difficulty)
        options = [q.answer_text, *q.distractors]
        assert len(q.distractors) >= 2 and len(set(options)) == len(options), (seed, options)
        for option in options:
            assert not any(f in option.lower() for f in FILLER), (topic, difficulty, seed, option)


def test_distractors_of_each_kind_become_the_key_on_some_seeds():
    """The replacements for the filler are real answers elsewhere: the key must vary across the options' kinds."""
    keys = {make("propositional_formula_equivalence", seed=s, difficulty="hard").answer_text for s in range(80)}
    assert len(keys) >= 3, keys
    verdicts = {make("quantifier_forall_exists_evaluate", seed=s, difficulty="easy").answer_text for s in range(80)}
    assert len(verdicts) >= 2, verdicts


def test_easy_propositional_key_is_not_guessable():
    """The easy tier used to answer "Equivalent" on every seed. No option may be the key on more than half of 100
    seeds, and at least three different keys appear."""
    from collections import Counter

    keys = Counter(make("propositional_formula_equivalence", seed=s, difficulty="easy").answer_text for s in range(100))
    assert len(keys) >= 3, keys
    assert max(keys.values()) <= 50, keys
