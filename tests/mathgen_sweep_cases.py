"""The generator sweep, one generation per topic, split into parts so CI can spread it over its shards.

Every per-topic check in `tests/test_mathgen_*.py` used to build the same questions for itself:
the build topics' text, answer, option and variety checks each ran
`make(topic, seed=s, difficulty=d)` over every tier, and the default-parameter checks each ran
`make(topic, seed=s)`, so the grid was generated about eight and five times over. Generation is
the cost and the checks are cheap, so each topic's grid is built once here and every check runs
on it. Each check is a plain `check_*` function next to its allowlists, takes the topic and the
questions already built, and slices the seeds it needs, so each keeps its own seed count (the
grid is built to the largest). A failing check never hides another: every check runs and the
test fails with one message naming each failed check. A check that does not apply to a topic
(a grandfathered or exempt topic) returns without running.

The checks only read the questions (`Question.to_dict` serialises a copy), so they share them.

CI shards by file, so `tests/test_mathgen_sweep_build_part_<k>.py` and
`tests/test_mathgen_sweep_default_part_<k>.py` each run the topics whose position in the sorted
topic list is k modulo the part count (like `tests/skill_contract_cases.py`). A case id is still
the topic, so `pytest tests/test_mathgen_sweep_*_part_*.py -k "<topic>"` runs one topic.
`tests/test_mathgen_sweep.py` proves the parts cover every topic once.
"""

import pytest

from tests import test_mathgen_answers as answers
from tests import test_mathgen_contract as contract
from tests import test_mathgen_options as options
from tests import test_mathgen_text as text
from tests import test_mathgen_variety as variety
from tests.mathgen_sweep_support import build_grid, build_questions, TEXT_SEEDS, WORKING_SEEDS

# Topics registered by a build module (`mathgen/generators/*_build.py`) are held to the
# learner-facing text bar; every topic is held to the default-parameter checks.
BUILD_TOPICS = sorted(text.BUILD_TOPICS)
TOPICS = sorted(text.TOPICS)
BUILD_PARTS = 6
DEFAULT_PARTS = 4

# (name, check) over a `{difficulty: questions}` grid, for the build topics.
BUILD_CHECKS = [
    ("no_bad_text_patterns", text.check_no_bad_text_patterns),
    ("no_bad_working_text", text.check_no_bad_working_text),
    ("answer_honours_its_own_stated_precision", answers.check_answer_honours_its_own_stated_precision),
    ("fractions_are_in_lowest_terms", answers.check_fractions_are_in_lowest_terms),
    ("each_tier_offers_enough_distinct_questions", options.check_each_tier_offers_enough_distinct_questions),
    ("answer_is_not_the_only_option_of_its_shape", options.check_answer_is_not_the_only_option_of_its_shape),
    ("answer_is_not_the_only_labelled_option", options.check_answer_is_not_the_only_labelled_option),
    ("prompt_variety", variety.check_prompt_variety),
]
# Seeds per tier the build checks slice, so the grid is built to the largest.
BUILD_GRID_SEEDS = max(TEXT_SEEDS, WORKING_SEEDS, options._VARIETY_SEEDS, variety._SHAPE_SEEDS)

# (name, check, seeds needed for a topic) over `[make(topic, seed=s) ...]`, for every topic.
DEFAULT_CHECKS = [
    ("no_bad_sign_patterns", text.check_no_bad_sign_patterns, text.sign_seeds),
    ("fuzz_generate_and_verify", contract.check_fuzz_generate_and_verify, contract.fuzz_seeds),
    ("distractors_are_valid", contract.check_distractors_are_valid, lambda _topic: contract.DISTRACTOR_SEEDS),
    (
        "all_topics_usually_have_enough_distractors",
        answers.check_all_topics_usually_have_enough_distractors,
        answers.enough_distractor_seeds,
    ),
    (
        "option_notes_absent_from_every_topics_output",
        contract.check_option_notes_absent_from_every_topics_output,
        lambda _topic: contract.OPTION_NOTES_SEEDS,
    ),
    (
        "answer_form_only_present_for_declared_topics",
        contract.check_answer_form_only_present_for_declared_topics,
        lambda _topic: contract.ANSWER_FORM_SEEDS,
    ),
]


def _run(checks):
    """Run every `(name, thunk)`; fail with one message naming each check that failed."""
    failures = []
    for name, thunk in checks:
        try:
            thunk()
        except Exception as exc:  # a check that crashes must not hide the others
            detail = str(exc) if isinstance(exc, AssertionError) else f"{type(exc).__name__}: {exc}"
            failures.append(f"[{name}] {detail}")
    if failures:
        pytest.fail(f"{len(failures)} of {len(checks)} checks failed:\n" + "\n".join(failures), pytrace=False)


def build_part_topics(part):
    return [t for i, t in enumerate(BUILD_TOPICS) if i % BUILD_PARTS == part]


def default_part_topics(part):
    return [t for i, t in enumerate(TOPICS) if i % DEFAULT_PARTS == part]


def build_topic_sweep(topic):
    grid = build_grid(topic, BUILD_GRID_SEEDS)
    _run([(name, lambda check=check: check(topic, grid)) for name, check in BUILD_CHECKS])


def default_topic_sweep(topic):
    questions = build_questions(topic, max(seeds(topic) for _name, _check, seeds in DEFAULT_CHECKS))
    _run([(name, lambda check=check: check(topic, questions)) for name, check, _seeds in DEFAULT_CHECKS])


def build_sweep_test(part):
    """A test function running the build-topic sweep for each build topic in `part`."""

    @pytest.mark.parametrize("topic", build_part_topics(part))
    def test_build_topic_sweep(topic):
        build_topic_sweep(topic)

    return test_build_topic_sweep


def default_sweep_test(part):
    """A test function running the default-parameter sweep for each topic in `part`."""

    @pytest.mark.parametrize("topic", default_part_topics(part))
    def test_default_topic_sweep(topic):
        default_topic_sweep(topic)

    return test_default_topic_sweep
