"""A unit that describes an earlier value the learner is told to use is not the unit being asked for."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mathgen.grading.units import unit_is_acceptable

RATE_PART = "Using your speed in m/s, how long would it take to ride a further 550 m, to the nearest second?"


@pytest.mark.parametrize("reply", ["100", "100 s", "100 seconds"])
def test_earlier_value_unit_does_not_make_the_reply_wrong(reply):
    question = SimpleNamespace(text=RATE_PART, answer_text="100", answer=100)
    assert unit_is_acceptable(question, reply)


@pytest.mark.parametrize(
    "text, good, bad",
    [
        ("Convert 3 cm to mm.", "30 mm", "30 kg"),
        ("How long is the rope in cm?", "30 cm", "30 kg"),
        ("Using your answer, how many mm is that?", "30 mm", "30 cm"),
    ],
)
def test_a_unit_that_is_asked_for_is_still_enforced(text, good, bad):
    question = SimpleNamespace(text=text, answer_text="30", answer=30)
    assert unit_is_acceptable(question, good)
    assert not unit_is_acceptable(question, bad)


def test_your_value_in_unit_then_asked_unit_uses_the_asked_one():
    question = SimpleNamespace(text="Using your speed in m/s, what is the distance in km?", answer_text="5", answer=5)
    assert unit_is_acceptable(question, "5 km")
    assert not unit_is_acceptable(question, "5 m/s")
