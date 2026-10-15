"""One library-wide contract for the generic checks that per-generator test files used to copy.

Hundreds of `tests/test_*.py` files each re-asserted the same things for their one generator: that it is
deterministic, that its answers are not a handful of values, that its text has no defects, that it shows several
prompt shapes, that its exam source is wired into a skill, that it rejects an unknown tier. This file asserts the
parts of that which no other library-wide test already does, once, for every generator, so those copies can go
(`scripts/prune_generic_tests.py` says which). Oracle tests (answer re-derived from the prompt, the real grader
rejecting a wrong answer, follow-through marks) are topic-specific and stay in their own files.

What the other library-wide tests already cover, and so is NOT repeated here:

- determinism of every skill source at every tier, 60 seeds: `scripts/skill_contract.check_determinism`; the default
  tier of every topic: `tests/test_mathgen_contract.py::test_determinism`;
- text defects, prompt-shape variety and distinct-question counts of every BUILD topic at every tier:
  `tests/mathgen_sweep_cases.py`;
- an exam source adds a prompt shape the lower tiers lack: `tests/test_mathgen_variety.py::test_tier_escalation`
  (the exam source is the last source of its skill, so it is held against the union of every lower tier);
- the real grader accepts the key and rejects every distractor: `scripts/skill_contract.check_grader`.

What this file adds, over the exam tier of every topic that has one (`PRESETS[topic]["exam"]`), built once per topic
(the same shape as the generator sweep: one test per topic, every check on the questions already built, one
message naming each failed check):

- `answer_space`: the composed answer takes at least `ANSWER_MIN_DISTINCT` values in `ANSWER_SEEDS` seeds and no
  value is more than `ANSWER_MAX_SHARE` of them. 302 of the 876 exam topics are narrower than that by design (a
  classification, a yes/no verdict, a one-line proof key), so each is pinned in the baseline file at 80% of the
  distinct count and 125% of the commonest count it measures today: a narrowing fails, and a pin the topic now
  beats must be dropped;
- `text`: no text defect (`find_text_defects`) in the first `TEXT_SEEDS` questions;
- `shapes`: at least `MIN_SHAPES` prompt shapes in `SHAPE_SEEDS` seeds;
- `multipart`: a multi-part question's parts are labelled a, b, c ... in order with marks of at least 1, and the
  first part has no follow-through;
- `deterministic`: seeds 0, 1, 7 and 99 rebuild to the same JSON.

The exam-tier checks run from `tests/test_generator_contract_exam_part_*.py` (split so CI can spread them over
shards) and live in `tests/generator_contract_cases.py` with the thresholds and baselines. A seed on which `make()` raises fails its topic's test, except the three in `KNOWN_CRASH_SEEDS`, which must still
raise (a latent generator defect found by this test; fix the generator and delete the line).

And library-wide, without generating a question:

- every exam source is the only exam source of its skill, comes last, has stages ["exam"], and (for the
  `exd_*`/`exam_*` generators) carries `{"tier": "exam"}` unless it is listed in the baseline's `exam_params_deviations`;
- every `exd_*`/`exam_*` generator is wired into a skill as an exam source, unless it is in `UNWIRED_BASELINE`;
- a generator with a `tier` parameter raises ValueError for an unknown tier, unless it is in `TIER_BASELINE`; a
  topic with presets raises KeyError for an unknown difficulty.

The baselines live in `tests/generator_contract_baseline.yaml`, shrink-only like `skill_contract_baseline.yaml`: an
entry that no longer applies fails until it is removed. `scripts/generator_contract_baseline.py --write` regenerates
the file from the tree (`--check` says whether it is stale); run it after generators merge.
"""

from __future__ import annotations

import inspect
from functools import lru_cache

import pytest

from mathgen import make
from mathgen.core import PRESETS, REGISTRY
from mathgen.skills import load_skills
from tests.generator_contract_cases import (
    ANSWER_SPACE_PINNED,
    BASELINE_PATH,
    EXAM,
    EXAM_PARAMS_DEVIATIONS,
    EXAM_PREFIXES,
    EXAM_TOPICS,
    KNOWN_CRASH_SEEDS,
    SHAPES_PINNED,
    TEXT_EXEMPT,
    TIER_BASELINE,
    UNWIRED_BASELINE,
)

# ---------------------------------------------------------------------------
# The library: skills and their exam sources.


@lru_cache(maxsize=1)
def skills():
    return load_skills()[1]


def exam_sources():
    """[(skill, source)] for every exam source of every skill."""
    return [(s, src) for s in skills() for src in s.sources if src.difficulty == EXAM]


def expected_exam_params(topic: str, skill_id: str = "") -> dict:
    """The params the exam source of an `exd_*`/`exam_*` generator carries: {"tier": "exam"} unless listed."""
    if topic == "exd_x_u5_partial_chain":
        return {"tier": EXAM, "var": "x" if skill_id.endswith("-x") else "y"}
    return EXAM_PARAMS_DEVIATIONS.get(topic, {"tier": EXAM})


def test_every_exam_source_is_wired_alike():
    problems = []
    for skill in skills():
        exam = [src for src in skill.sources if src.difficulty == EXAM]
        if not exam:
            continue
        if len(exam) != 1:
            problems.append(f"{skill.id}: {len(exam)} exam sources, expected 1")
            continue
        src = exam[0]
        if skill.sources[-1] is not src:
            problems.append(f"{skill.id}: the exam source is not the last source")
        if src.stages != ["exam"]:
            problems.append(f"{skill.id}: exam source stages are {src.stages!r}, expected ['exam']")
        if src.topic not in REGISTRY:
            problems.append(f"{skill.id}: exam source topic {src.topic!r} is not registered")
        elif "tier" in (src.params or {}) and src.params["tier"] != EXAM:
            problems.append(f"{skill.id}: exam source has tier={src.params['tier']!r}")
    assert not problems, "\n".join(problems)


def test_exam_source_params_are_the_pinned_ones():
    problems = []
    for skill, src in exam_sources():
        if not src.topic.startswith(EXAM_PREFIXES):
            continue
        expected = expected_exam_params(src.topic, skill.id)
        if dict(src.params or {}) != expected:
            problems.append(
                f"{skill.id}: exam source of {src.topic} has params {dict(src.params or {})}, pinned {expected}"
            )
    stale = [t for t in EXAM_PARAMS_DEVIATIONS if t not in {src.topic for _s, src in exam_sources()}]
    assert not problems and not stale, "\n".join([*problems, *(f"stale deviation: {t}" for t in stale)])


def test_exam_generators_are_wired_into_a_skill():
    wired = {src.topic for _s, src in exam_sources()}
    exam_generators = {t for t in REGISTRY if t.startswith(EXAM_PREFIXES) and EXAM in PRESETS.get(t, {})}
    unwired = exam_generators - wired
    new = sorted(unwired - UNWIRED_BASELINE)
    stale = sorted(UNWIRED_BASELINE - unwired)
    assert not new, f"{len(new)} exam generator(s) no skill uses as its exam source, e.g. {new[:5]}; wire them in"
    assert not stale, f"{stale[:5]} are wired now (or gone): remove them from {BASELINE_PATH.name} 'unwired'"


# ---------------------------------------------------------------------------
# Unknown names are refused.


def _tier_generators():
    return sorted(t for t, (_lvl, fn) in REGISTRY.items() if "tier" in inspect.signature(fn).parameters)


def _rejects_unknown_tier(topic: str) -> bool:
    try:
        make(topic, seed=0, tier="not-a-tier")
    except ValueError:
        return True
    except Exception:  # a different error, or a question, is not a rejection with ValueError
        return False
    return False


@pytest.mark.parametrize("topic", _tier_generators())
def test_unknown_tier_is_rejected(topic):
    rejected = _rejects_unknown_tier(topic)
    if topic in TIER_BASELINE:
        assert not rejected, f"{topic} rejects an unknown tier with ValueError now: remove it from the baseline"
    else:
        assert rejected, f"make({topic!r}, tier='not-a-tier') should raise ValueError"


def test_unknown_difficulty_is_rejected_where_there_are_presets():
    accepted = []
    for topic in sorted(PRESETS):
        if topic not in REGISTRY:
            continue
        try:
            make(topic, seed=0, difficulty="not-a-difficulty")
        except KeyError:
            continue
        accepted.append(topic)
    assert not accepted, f"{len(accepted)} topic(s) with presets accept an unknown difficulty: {accepted[:5]}"


def test_pins_name_real_exam_topics():
    exam = set(EXAM_TOPICS)
    assert set(ANSWER_SPACE_PINNED) <= exam and set(TEXT_EXEMPT) <= exam and set(SHAPES_PINNED) <= exam
    assert set(KNOWN_CRASH_SEEDS) <= exam
