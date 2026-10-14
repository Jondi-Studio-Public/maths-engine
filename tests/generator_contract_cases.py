"""Shared pieces of the generator contract: thresholds, baselines and the exam-tier checks. The tests are in
`tests/test_generator_contract.py` and `tests/test_generator_contract_exam_part_*.py`; read the former's docstring."""

from __future__ import annotations

import collections
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

from mathgen import make
from mathgen.core import PRESETS, REGISTRY
from mathgen.text_quality import find_text_defects, prompt_shape
from tests.mathgen_sweep_support import take

EXAM = "exam"
EXAM_TOPICS = sorted(t for t in REGISTRY if EXAM in PRESETS.get(t, {}))
EXAM_PREFIXES = ("exd_", "exam_")

# Thresholds. The seed counts and the 100 / 5% bar are the ones the per-generator copies used.
ANSWER_SEEDS = 300
ANSWER_MIN_DISTINCT = 100
ANSWER_MAX_SHARE = 0.05
TEXT_SEEDS = 100
TEXT_INCLUDES_WORKING = False  # find_text_defects(..., include_working=True) would flag the lines below the answer
SHAPE_SEEDS = 60
MIN_SHAPES = 3
DETERMINISM_SEEDS = (0, 1, 7, 99)
# Topics the shared text check skips, with why. Two are legacy generators whose plain text keeps python's `**`
# (the front end renders their `latex`), three trip `singular_count_plural` on correct English: "y = -4x - 1 cross
# at two points" and "the digit 1 repeats forever". The detector is what is wrong there, not the question.
TEXT_EXEMPT: dict[str, str] = {
    "differential_equation": "legacy generator: plain text spells powers with **",
    "quadratic": "legacy generator: plain text spells powers with **",
    "exd_x_m4_area_between_curves": "false positive: '- 1 cross at two points' read as a count of one",
    "exd_x_n4_recurring_mixed": "false positive: 'the digit 1 repeats forever' read as a count of one",
    "exd_x_n4_decimal_sum": "false positive: 'the digit 1 repeats forever' read as a count of one",
}
# Topics that show fewer prompt shapes than MIN_SHAPES, pinned at what they show: topic -> (shapes, why).
SHAPES_PINNED: dict[str, tuple[int, str]] = {
    "linear_equation": (2, "legacy generator, two question forms at the exam preset"),
}
# (topic, seed) pairs where make(topic, seed, difficulty="exam") raises. Each is a latent defect (a learner can
# be dealt that seed), recorded so the rest of the topic is still checked; the test fails when one stops raising.
KNOWN_CRASH_SEEDS: dict[str, tuple[int, ...]] = {
    "exd_x_r4g_back_to_back_stem": (232,),  # VerificationError: two parts would share an answer
}

BASELINE_PATH = Path(__file__).with_name("generator_contract_baseline.yaml")


@lru_cache(maxsize=1)
def baseline() -> dict[str, dict[str, str]]:
    if not BASELINE_PATH.exists():
        return {}
    return yaml.safe_load(BASELINE_PATH.read_text(encoding="utf-8")) or {}


TIER_BASELINE = frozenset(baseline().get("unknown_tier", {}))
UNWIRED_BASELINE = frozenset(baseline().get("unwired", {}))
EXAM_PARAMS_DEVIATIONS: dict[str, dict] = {
    topic: dict(params or {}) for topic, params in baseline().get("exam_params_deviations", {}).items()
}
ANSWER_SPACE_PINNED: dict[str, tuple[int, int]] = {
    topic: (int(pin[0]), int(pin[1])) for topic, pin in baseline().get("answer_space", {}).items()
}


def answer_space_for(topic: str) -> tuple[int, float]:
    """(distinct answers needed, largest share allowed) for `topic` over `ANSWER_SEEDS` seeds."""
    pinned = ANSWER_SPACE_PINNED.get(topic)
    if pinned is None:
        return ANSWER_MIN_DISTINCT, ANSWER_MAX_SHARE
    return pinned[0], pinned[1] / ANSWER_SEEDS


def build_exam_questions(topic: str) -> tuple[list, list[tuple[int, str]]]:
    """Exam-tier questions for seeds 0..ANSWER_SEEDS-1 and every (seed, exception name) where `make()` raised.
    Seeds are not skipped or judged here: `exam_questions` and `scripts/generator_contract_baseline.py` decide
    what a crash means."""
    out, crashes = [], []
    for seed in range(ANSWER_SEEDS):
        try:
            out.append(make(topic, seed=seed, difficulty=EXAM))
        except Exception as exc:
            crashes.append((seed, f"{type(exc).__name__}: {exc}"))
    return out, crashes


@lru_cache(maxsize=2)
def exam_questions(topic: str):
    """Exam-tier questions for seeds 0..ANSWER_SEEDS-1, built once per topic; a seed in KNOWN_CRASH_SEEDS is
    skipped (and must still raise, or the topic's test says so)."""
    known = set(KNOWN_CRASH_SEEDS.get(topic, ()))
    out = []
    for seed in range(ANSWER_SEEDS):
        try:
            out.append(make(topic, seed=seed, difficulty=EXAM))
        except Exception as exc:
            if seed not in known:
                raise AssertionError(f"make({topic!r}, seed={seed}, difficulty='exam') raised {exc!r}") from exc
            known.discard(seed)
    if known:
        raise AssertionError(f"{topic}: seed(s) {sorted(known)} no longer raise: remove them from KNOWN_CRASH_SEEDS")
    return out


def check_answer_space(topic, questions):
    answers = collections.Counter(q.answer_text for q in questions)
    distinct, share = answer_space_for(topic)
    most = max(answers.values())
    allowed = share * ANSWER_SEEDS
    assert len(answers) >= distinct, (
        f"{topic}: only {len(answers)} distinct answers in {len(questions)} seeds, expected {distinct} "
        f"(pin it in {BASELINE_PATH.name} 'answer_space' if its answer space is small by design)"
    )
    assert most <= allowed, (
        f"{topic}: one answer is {most} of {len(questions)} draws, at most {allowed:.0f} allowed; "
        f"commonest {answers.most_common(2)}"
    )
    if topic in ANSWER_SPACE_PINNED:
        beats = len(answers) >= ANSWER_MIN_DISTINCT and most <= ANSWER_MAX_SHARE * ANSWER_SEEDS
        assert not beats, f"{topic}: now meets the default answer-space bar ({len(answers)} distinct); drop its pin"


def check_text(topic, questions):
    if topic in TEXT_EXEMPT:
        return
    found = []
    for seed, q in enumerate(take(questions, TEXT_SEEDS)):
        for defect in find_text_defects(q, include_working=TEXT_INCLUDES_WORKING):
            found.append(f"seed={seed}: {defect}")
    assert not found, f"{topic}: " + "; ".join(found[:5])


def check_shapes(topic, questions):
    shapes = len({prompt_shape(q) for q in take(questions, SHAPE_SEEDS)})
    needed = SHAPES_PINNED[topic][0] if topic in SHAPES_PINNED else MIN_SHAPES
    assert shapes >= needed, f"{topic}: {shapes} prompt shapes in {SHAPE_SEEDS} seeds, expected {needed}"
    if topic in SHAPES_PINNED:
        assert shapes < MIN_SHAPES, f"{topic}: now shows {shapes} shapes; drop it from SHAPES_PINNED"


def check_multipart(topic, questions):
    for seed, q in enumerate(take(questions, SHAPE_SEEDS)):
        parts = q.parts
        if not parts:
            return
        labels = [p.label for p in parts]
        assert labels == [chr(ord("a") + i) for i in range(len(parts))], f"{topic} seed={seed}: labels {labels}"
        assert all(isinstance(p.marks, int) and p.marks >= 1 for p in parts), f"{topic} seed={seed}: marks"
        assert parts[0].follow_through is None, f"{topic} seed={seed}: part (a) cannot follow through from nothing"


def check_deterministic(topic, questions):
    for seed in DETERMINISM_SEEDS:
        if seed in KNOWN_CRASH_SEEDS.get(topic, ()):
            continue
        again = make(topic, seed=seed, difficulty=EXAM)
        assert questions[seed].to_json() == again.to_json(), f"{topic} seed={seed}: two builds differ"


EXAM_CHECKS = [
    ("answer_space", check_answer_space),
    ("text", check_text),
    ("shapes", check_shapes),
    ("multipart", check_multipart),
    ("deterministic", check_deterministic),
]


def exam_topic_contract(topic):
    questions = exam_questions(topic)
    failures = []
    for name, check in EXAM_CHECKS:
        try:
            check(topic, questions)
        except AssertionError as exc:
            failures.append(f"[{name}] {exc}")
    if failures:
        pytest.fail(f"{len(failures)} of {len(EXAM_CHECKS)} checks failed:\n" + "\n".join(failures), pytrace=False)


EXAM_PARTS = 8


def exam_part_topics(part):
    return [t for i, t in enumerate(EXAM_TOPICS) if i % EXAM_PARTS == part]


def exam_contract_test(part):
    """A test function running the exam-tier contract for each exam topic in `part`. The sweep is split over
    EXAM_PARTS files so that CI can spread it over shards (see scripts/ci_shard_weights.txt)."""

    @pytest.mark.parametrize("topic", exam_part_topics(part))
    def test_exam_topic_contract(topic):
        exam_topic_contract(topic)

    return test_exam_topic_contract
