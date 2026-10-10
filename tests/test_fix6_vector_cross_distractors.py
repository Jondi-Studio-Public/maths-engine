"""This file's tests all go through `mathgen.skills.load_skills()` and each skill's
own `sources`, never a bare `make()` call, so the population tested here
matches the population the real skill renders. It is independent
of `tests/test_exd_e6_vectors_kinds.py`.
"""

from __future__ import annotations

from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills

TOPIC = "vectors"

_LIST_GRADER = GRADERS["list"]

SEEDS = 600


def _tuplify(params: dict) -> dict:
    """Mirrors `mathgen.skills._tuplify` / `scripts/regrade_sweep.py::_tuplify`."""
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in (params or {}).items()}


def _find_skill(skill_id: str):
    _domains, skill_list = load_skills()
    for skill in skill_list:
        if skill.id == skill_id:
            return skill
    return None


def _add_sources(skill):
    """Only the skill's sources whose `kinds` param is `["add"]` -- excludes `vector-addition`'s own `hard`
    source, which uses kind `subtract`."""
    out = []
    for src in skill.sources:
        params = src.params or {}
        kinds = params.get("kinds")
        if kinds and list(kinds) == ["add"]:
            out.append(src)
    return out


def test_cross_skill_loads_via_api_skills_vector_cross_product():
    skill = _find_skill("vector-cross-product")
    assert skill is not None
    assert skill.answer == "list"
    assert len(skill.sources) >= 1


def test_add_skill_loads_via_api_skills_vector_addition():
    skill = _find_skill("vector-addition")
    assert skill is not None
    assert skill.answer == "list"
    assert len(_add_sources(skill)) >= 1


def _sweep_source_for_false_accepts(src, seeds):
    """`[(seed, answer_text, distractor), ...]` false accepts against the
    real registered `GRADERS["list"]` grader, for one skill source, over
    `seeds` seeds -- the same method `scripts/regrade_sweep.py::sweep_skill`
    uses (`load_skills()`'s own `Source.topic`/`difficulty`/`params`, never a
    bare `make()` call with a hand-picked `kinds` tuple)."""
    params = _tuplify(src.params)
    bad = []
    for seed in range(seeds):
        q = make(src.topic, seed=seed, difficulty=src.difficulty, **params)
        canonical_result = _LIST_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _LIST_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    return bad


def test_cross_grader_accepts_canonical_and_rejects_every_distractor_over_600_seeds_per_source():
    """The zero-false-accept property test: sweeps the REAL registered grader
    (`mathgen.grading.GRADERS["list"]`, the same grader object the skill actually uses) against every
    distractor over 600 seeds per source, through `vector-cross-product`'s own declared sources
    (`easy`/`medium` kind `cross`, `hard` kind `cross_sum`) rather than a bare `make` call."""
    skill = _find_skill("vector-cross-product")
    bad = []
    for src in skill.sources:
        bad.extend(_sweep_source_for_false_accepts(src, SEEDS))
    assert bad == [], bad


def test_add_grader_accepts_canonical_and_rejects_every_distractor_over_600_seeds_per_source():
    """Proven RED against the pre-fix `add` branch (no `unordered=True`) --."""
    skill = _find_skill("vector-addition")
    bad = []
    for src in _add_sources(skill):
        bad.extend(_sweep_source_for_false_accepts(src, SEEDS))
    assert bad == [], bad


def test_cross_no_question_renders_with_zero_distractors_over_600_seeds_per_source():
    """Matches the repo's own convention for "renders with zero distractors"
    (`tests/test_exd_e6_vectors_kinds.py`'s
    `test_*_distractors_non_empty_and_never_equal_answer_over_300_seeds` functions all assert
    `len(q.distractors) >= 1`, not an exact count of 3) -- `pick_distractors` can legitimately cull fewer than
    3 candidates when some collide with the answer or each other."""
    skill = _find_skill("vector-cross-product")
    for src in skill.sources:
        params = _tuplify(src.params)
        for seed in range(SEEDS):
            q = make(src.topic, seed=seed, difficulty=src.difficulty, **params)
            assert len(q.distractors) >= 1, (src.difficulty, seed, q.text, q.distractors)
