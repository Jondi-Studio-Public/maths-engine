"""The shape question of the histogram exam chain only offers words its grader accepts."""

from __future__ import annotations

from mathgen import make
from mathgen.grading import grade

TOPIC = "exd_x_r2k_histogram"


def test_prompt_offers_single_words_and_each_is_accepted():
    for seed in range(1, 31):
        q = make(TOPIC, seed=seed, difficulty="exam")
        shape = q.parts[3]
        assert "(skewed" not in shape.prompt, shape.prompt
        offered = ("right", "left", "symmetric", "bimodal")
        assert all(word in shape.prompt for word in offered)
        assert shape.answer_text in offered
        assert grade(shape.answer_kind, shape, shape.answer_text).correct
