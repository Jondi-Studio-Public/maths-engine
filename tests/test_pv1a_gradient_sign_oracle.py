"""Independent oracle for the `pv1a_gradient_sign` questions: it re-derives each answer from the rendered
question text.

Imports neither the generator module nor the shared line helper. It calls
`mathgen.make` through the skill's own sources and derives the sign of the
gradient from `question.figure` alone (the two through points), over 1000
seeds per tier in four chunks, then checks the key, the two distractors and
the real mcq grader.
"""

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "senior-recognise-features-of-the-graph-of-y-mx"
CHUNK = 250


@pytest.fixture(scope="module")
def sources():
    _, skills = load_skills()
    return {s.difficulty: s for s in {k.id: k for k in skills}[SKILL_ID].sources}


def _sign(fig):
    (x1, y1), (x2, y2) = fig["lines"][0]["through"]
    assert x1 != x2
    rise, run = y2 - y1, x2 - x1
    if rise == 0:
        return "zero"
    return "positive" if (rise > 0) == (run > 0) else "negative"


@pytest.mark.parametrize("chunk", range(4))
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "exam"])
def test_key_equals_sign_of_the_drawn_line(sources, difficulty, chunk):
    src = sources[difficulty]
    for seed in range(chunk * CHUNK, (chunk + 1) * CHUNK):
        q = make(src.topic, seed=seed, difficulty=difficulty, **(src.params or {}))
        want = _sign(q.figure)
        assert q.answer_text == want, (difficulty, seed)
        assert sorted([q.answer_text, *q.distractors]) == ["negative", "positive", "zero"]
        options = mcq_options(q)
        assert options.count(want) == 1
        assert grade_mcq(q, options.index(want)).correct
        for i, opt in enumerate(options):
            if opt != want:
                assert not grade_mcq(q, i).correct
        # a flat line is a horizontal line off the x-axis; the picture is a figure, not text
        if want == "zero":
            (x1, y1), (x2, y2) = q.figure["lines"][0]["through"]
            assert y1 == y2 != 0
        assert not any(ch.isdigit() for ch in q.text)
