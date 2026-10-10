"""mathgen.grading.spoken: grading a speech-recognition transcript of a spoken answer."""

from mathgen import make
from mathgen.core import Question
from mathgen.grading import spoken


def test_a_spoken_number_is_graded_against_the_answer():
    q = make("addition", seed=42)  # 20 + 98
    assert q.answer == 118
    assert spoken.check_answer(q, "one hundred and eighteen")
    assert spoken.check_answer(q, "one hundred eighteen")
    assert spoken.check_answer(q, "118")
    assert not spoken.check_answer(q, "one hundred and nineteen")
    assert not spoken.check_answer(q, "one eighteen")


def test_the_fallback_is_tried_only_when_the_transcript_does_not_match():
    q = make("addition", seed=42)
    assert spoken.check_answer(q, "um", fallback="118")
    assert not spoken.check_answer(q, "um", fallback="119")


def test_decimals_are_accepted_within_the_transcript_tolerance():
    q = Question(topic="t", level=1, seed=1, text="", latex="", answer=3.5, answer_text="3.5")
    assert spoken.check_answer(q, "three point five")
    assert spoken.check_answer(q, "3.52")
    assert not spoken.check_answer(q, "3.6")
