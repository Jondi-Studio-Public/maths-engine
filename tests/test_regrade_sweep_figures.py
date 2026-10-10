"""Tests for the figure-plausibility checks scripts/regrade_sweep.py runs on
every figure-bearing question.

Uses a throwaway fixture topic registered directly against `mathgen.core`'s
live `REGISTRY`/`PRESETS` and `mathgen.figures`' live reader registry, torn
down after each test -- never a committed skill or generator module.
"""

from __future__ import annotations

import itertools

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import Skill, Source
from mathgen import core, figures
from scripts.regrade_sweep import sweep_skill

_counter = itertools.count()


@pytest.fixture
def fixture_topic():
    """Registers a unique number-line-point generator topic for one test,
    with a `mode` kwarg controlling its figure's plausibility:

    - "good": the mark sits at the answer; alt text names the object only.
    - "wrong_key": the mark sits one unit away from the answer, so the
      registered reader (which reads the mark's own position) disagrees
      with the key.
    - "leak": the mark sits at the answer, but the alt text states the
      answer's value outright.

    Torn down (REGISTRY/PRESETS/figures reader) after the test regardless
    of outcome, so topics never leak across tests or pollute the real
    registries other tests rely on.
    """
    topic = f"_fixture_numberline_point_{next(_counter)}"

    def make_question(rng, answer=23, mode="good"):
        # `assert_no_leak` only flags a multi-character match (a single digit
        # like "7" is allowed to appear incidentally in a label); use a
        # two-digit answer so the "leak" mode actually exercises the check.
        mark_at = answer if mode != "wrong_key" else answer + 1
        alt = "a point on a number line" if mode != "leak" else f"a point on a number line at {answer}"
        spec = figures.number_line(
            range_=(0, 30),
            alt=alt,
            tick=5,
            marks=[{"at": mark_at}],
        )
        return dict(
            text=f"What number is marked on the number line?",
            latex=r"\text{What number is marked?}",
            answer=answer,
            answer_text=str(answer),
            figure=spec,
        )

    core.REGISTRY[topic] = (1, make_question)

    def read_marked_point(question):
        return str(question.figure["marks"][0]["at"])

    figures.register_reader(topic, read_marked_point)

    yield topic

    core.REGISTRY.pop(topic, None)
    core.PRESETS.pop(topic, None)
    figures._READERS.pop(topic, None)


def _skill_for(topic: str, **params) -> Skill:
    return Skill(
        id=f"test-{topic}",
        name="fixture",
        domain="test",
        answer="number",
        sources=[Source(topic=topic, params=params)],
    )


def test_good_figure_passes(fixture_topic):
    skill = _skill_for(fixture_topic, mode="good")
    result = sweep_skill(skill, seeds=5, graders=GRADERS)
    assert result.questions == 5
    assert result.figure_checks == 5
    assert result.figure_failures == []
    assert result.clean


def test_figure_disagreeing_with_key_fails(fixture_topic):
    skill = _skill_for(fixture_topic, mode="wrong_key")
    result = sweep_skill(skill, seeds=5, graders=GRADERS)
    assert result.figure_checks == 5
    assert len(result.figure_failures) == 5
    assert not result.clean
    kind, topic, difficulty, seed, message = result.figure_failures[0]
    assert kind == "number_line"
    assert "does not grade correct" in message


def test_leaking_alt_text_fails(fixture_topic):
    skill = _skill_for(fixture_topic, mode="leak")
    result = sweep_skill(skill, seeds=5, graders=GRADERS)
    assert result.figure_checks == 5
    assert len(result.figure_failures) == 5
    assert not result.clean
    kind, topic, difficulty, seed, message = result.figure_failures[0]
    assert kind == "number_line"
    assert "figure shows the answer" in message


def test_topic_without_a_registered_reader_still_gets_validate_and_leak_checks(fixture_topic):
    """A figure-bearing topic with no reader (figures.get_reader returns
    None) still runs `figures.validate` and `assert_no_leak` -- only the
    reader re-derivation step is skipped."""
    figures._READERS.pop(fixture_topic, None)  # simulate "no reader registered"

    good_skill = _skill_for(fixture_topic, mode="good")
    result = sweep_skill(good_skill, seeds=3, graders=GRADERS)
    assert result.figure_checks == 3
    assert result.figure_failures == []

    leak_skill = _skill_for(fixture_topic, mode="leak")
    result = sweep_skill(leak_skill, seeds=3, graders=GRADERS)
    assert len(result.figure_failures) == 3
    assert "figure shows the answer" in result.figure_failures[0][4]


def test_deferred_skill_is_swept_when_ids_pass_it_explicitly(fixture_topic):
    """`select_skills` looks a `--ids` skill up by id with no status filter,
    so a `status: deferred` figure-bearing skill (the shape every
    figure_stimulus row will ship as, until the gate passes) is still swept."""
    from scripts.regrade_sweep import select_skills

    deferred = Skill(
        id=f"test-deferred-{fixture_topic}",
        name="fixture",
        domain="test",
        answer="number",
        status="deferred",
        deferred_reason="capability_gate",
        sources=[Source(topic=fixture_topic, params={"mode": "good"})],
    )
    selected = select_skills([deferred], ids=[deferred.id])
    assert selected == [deferred]

    result = sweep_skill(deferred, seeds=3, graders=GRADERS)
    assert result.figure_checks == 3
    assert result.clean
