"""Option quality: distinct questions per tier and answers that stand out from their options.

Part of the generator sweep, split across `tests/test_mathgen_*.py` so the CI
shards can balance it (`scripts/ci_shard.py` balances files by size). Verification
is built into the generators themselves (each independently re-checks its
answer with SymPy and raises VerificationError on mismatch), so constructing
thousands of questions IS the correctness test; the assertions here cover the
output contract on top of that. Set `MATHS_DEEP_SEEDS` for a deeper run of the
text checks.
"""

import re

import pytest

from mathgen import make, topics
from mathgen.core import PRESETS, REGISTRY
from tests.mathgen_sweep_support import TEXT_SEEDS

TOPICS = [t for _lvl, t in topics()]
# Topics registered by a build module (`mathgen/generators/*_build.py`): these
# must meet the learner-facing text bar. Legacy topics keep their existing
# notation (their `latex` field is what the UI renders) and are out of scope
# until someone owns them.
BUILD_TOPICS = sorted(t for t, (_lvl, fn) in REGISTRY.items() if fn.__module__.rsplit(".", 1)[-1].endswith("_build"))

# Tiers whose question pool is genuinely smaller than _MIN_DISTINCT. These are
# PINNED MINIMUMS, not a ratchet: there are only three inverse trigonometric
# functions to differentiate, and only two ways to halve a unit fraction at the
# easy tier. They will never reach the floor, and pretending they should would
# make the check one nobody trusts. The number is recorded so a REGRESSION
# shows: the assertion is >=, so widening a pool is free and only narrowing one
# fails. A new topic belongs here only with a reason a reviewer would accept.
SMALL_QUESTION_POOL = {
    ("angle_of_inclination_gradient", "easy"): 3,
    ("angle_of_inclination_gradient", "medium"): 6,
    ("angle_of_inclination_gradient", None): 3,
    ("angle_turn_right_reference", "easy"): 4,
    ("angle_turn_right_reference", None): 4,
    ("compare_measurement_attributes", "easy"): 4,
    ("compare_measurement_attributes", "medium"): 4,
    ("compare_measurement_attributes", None): 8,
    ("convert_units_for_rates", "hard"): 4,
    ("find_and_use_the_derivative_of_the_inverse", "easy"): 3,
    ("find_and_use_the_derivative_of_the_inverse", None): 3,
    ("fraction_halving", "easy"): 2,
    ("fraction_halving", "hard"): 4,
    ("fraction_halving", "medium"): 2,
    ("fraction_halving", None): 2,
    ("logarithms_as_indices", "easy"): 9,
    ("logarithms_as_indices", "hard"): 9,
    ("logarithms_as_indices", None): 9,
    ("natural_logarithm_define", "easy"): 4,
    ("natural_logarithm_define", "hard"): 6,
    ("natural_logarithm_define", None): 4,
    ("odd_even_properties", "easy"): 4,
    ("odd_even_properties", "medium"): 8,
    ("odd_even_properties", None): 4,
    ("pascals_triangle_properties", "easy"): 7,
    ("pascals_triangle_properties", None): 7,
    ("percentage_to_fraction", "easy"): 5,
    ("percentage_to_fraction", None): 5,
    ("probability_scale_estimate", "easy"): 5,
    ("probability_scale_estimate", "medium"): 9,
    ("probability_scale_estimate", None): 9,
    # Capped by a real grader-gap defect, not by the bank's own design (the
    # bank held 10 (formula, target) cases until 3 --
    # E=mc^2 for c, P=I^2R for I, y=a(x-h)^2+k for x -- were found to have a
    # canonical answer (sqrt of a quotient with a free symbol denominator)
    # whose natural learner-typed alternative form the real grader rejects
    # (the grader's symbolic equality has no positivity assumptions).
    # Those 3 were removed, leaving 7; raise this pin (or remove it, if the
    # bank grows past _MIN_DISTINCT=10 again) once that grader gap is fixed
    # and a wider bank of safe cases can be re-added.
    ("rearrange_powers_roots", None): 7,
    ("shape_features_classification", "easy"): 4,
    ("shape_features_classification", "hard"): 4,
    ("shape_features_classification", "medium"): 5,
    ("shape_features_classification", None): 5,
    ("tan_shift_complementary_identity", "easy"): 7,
    ("tan_shift_complementary_identity", "hard"): 7,
    ("tan_shift_complementary_identity", "medium"): 7,
    ("tan_shift_complementary_identity", None): 7,
    ("unit_fractions_combine_whole", "easy"): 6,
    ("unit_fractions_combine_whole", None): 6,
}

# Tiers where the answer is sometimes the only option of its shape, so it can
# be picked without doing the arithmetic. Unlike the table above these ARE
# defects and every one is fixable, so this is a ratchet: it may only shrink,
# and the tripwire below forces each entry out once it is fixed. Never add to
# it. The working-text backlog (test_mathgen_text.py) was the same shape and has already reached
# empty, which is where
# this one is headed.
OPTION_SHAPE_BACKLOG = {
    ("fraction_halving", "hard"): 15,
    ("non_uniform_discrete_random_variable", "hard"): 1,
    ("non_uniform_discrete_random_variable", "medium"): 4,
}

_VARIETY_SEEDS = 60
_MIN_DISTINCT = 10


def _distinct_prompts(topic, difficulty):
    return {make(topic, seed=seed, difficulty=difficulty).text for seed in range(_VARIETY_SEEDS)}


@pytest.mark.parametrize("topic", BUILD_TOPICS)
def test_each_tier_offers_enough_distinct_questions(topic):
    for difficulty in [*sorted(PRESETS.get(topic, {})), None]:
        distinct = len(_distinct_prompts(topic, difficulty))
        pinned = SMALL_QUESTION_POOL.get((topic, difficulty))
        if pinned is not None:
            assert distinct >= pinned, (
                f"{topic} [{difficulty}] now yields {distinct} distinct questions in "
                f"{_VARIETY_SEEDS} seeds, down from the pinned {pinned}"
            )
            continue
        assert distinct >= _MIN_DISTINCT, (
            f"{topic} [{difficulty}] yields only {distinct} distinct questions in "
            f"{_VARIETY_SEEDS} seeds; widen the pool, or pin it in "
            f"SMALL_QUESTION_POOL with a reason if it is genuinely this small"
        )


_SHAPE_RES = (
    ("integer", re.compile(r"^-?\d+$")),
    ("decimal", re.compile(r"^-?\d+\.\d+$")),
    ("fraction", re.compile(r"^-?\d+/\d+$")),
)


def _option_shape(value):
    for name, pattern in _SHAPE_RES:
        if pattern.match(value.strip()):
            return name
    return None


def _lone_shape_seeds(topic, difficulty):
    """Seeds where the answer is the only option of its shape."""
    found = []
    for seed in range(TEXT_SEEDS):
        question = make(topic, seed=seed, difficulty=difficulty)
        options = [question.answer_text, *question.distractors]
        if len(options) < 3:
            continue
        shapes = [_option_shape(o) for o in options if isinstance(o, str)]
        if len(shapes) != len(options) or None in shapes:
            continue  # not a numeric option set; shape says nothing here
        if shapes.count(shapes[0]) == 1:
            found.append((seed, question.answer_text, list(question.distractors)))
    return found


@pytest.mark.parametrize("topic", BUILD_TOPICS)
def test_answer_is_not_the_only_option_of_its_shape(topic):
    for difficulty in [*sorted(PRESETS.get(topic, {})), None]:
        if (topic, difficulty) in OPTION_SHAPE_BACKLOG:
            continue
        found = _lone_shape_seeds(topic, difficulty)
        assert not found, (
            f"{topic} [{difficulty}]: the answer is the only option of its shape at "
            f"{len(found)} seeds, so it can be picked without doing the arithmetic. "
            f"First: seed={found[0][0]} answer={found[0][1]!r} "
            f"distractors={found[0][2]!r}"
        )


_LABEL_RE = re.compile(r"^\s*[A-Za-z]\w*\s*=")


def _lone_label_seeds(topic, difficulty):
    """Seeds where only the ANSWER carries a leading label such as "y = ".

    `_option_shape` above compares the numeric shape of an option and skips a
    set that is not numeric, so it cannot see this: one generator shipped the answer
    `'y = -2x^2 + 4x + 1'` among distractors `['-2x^2 - 4x + 1', ...]` at 60/60
    seeds of every tier, pickable without doing any mathematics.
    """
    found = []
    for seed in range(TEXT_SEEDS):
        question = make(topic, seed=seed, difficulty=difficulty)
        options = [question.answer_text, *question.distractors]
        if len(options) < 3 or not all(isinstance(o, str) for o in options):
            continue
        labelled = [bool(_LABEL_RE.match(o)) for o in options]
        if labelled[0] and not any(labelled[1:]):
            found.append((seed, question.answer_text, list(question.distractors)))
    return found


@pytest.mark.parametrize("topic", BUILD_TOPICS)
def test_answer_is_not_the_only_labelled_option(topic):
    for difficulty in [*sorted(PRESETS.get(topic, {})), None]:
        found = _lone_label_seeds(topic, difficulty)
        assert not found, (
            f"{topic} [{difficulty}]: the answer is the only option carrying a "
            f"label at {len(found)} seeds, so it can be picked without doing the "
            f"mathematics. Prefix every option alike, or strip it from the answer. "
            f"First: seed={found[0][0]} answer={found[0][1]!r} "
            f"distractors={found[0][2]!r}"
        )


@pytest.mark.parametrize("topic,difficulty", sorted(OPTION_SHAPE_BACKLOG, key=lambda key: (key[0], key[1] or "~")))
def test_option_shape_backlog_still_dirty(topic, difficulty):
    """Forces the backlog to empty: a fixed tier must be removed from it."""
    assert topic in BUILD_TOPICS, f"{topic} is not a build topic; drop it from OPTION_SHAPE_BACKLOG"
    assert _lone_shape_seeds(topic, difficulty), (
        f"{topic} [{difficulty}] no longer shows a lone-shaped answer: remove it from OPTION_SHAPE_BACKLOG"
    )
