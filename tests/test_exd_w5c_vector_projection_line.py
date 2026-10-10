"""Vector-projection hard tier: project onto the line through two points."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

from mathgen.grading import GRADERS
from mathgen import make
from mathgen.text_quality import prompt_shape

PATTERN = re.compile(
    r"Project u = \((-?\d+), (-?\d+)\) onto the line through A = \((-?\d+), (-?\d+)\) and B = \((-?\d+), (-?\d+)\)\."
)


def _line(seed):
    return make("vector_projection", seed=seed, difficulty="hard", kind="line_through_points")


def test_line_answer_matches_recomputation_from_rendered_text():
    for seed in range(60):
        q = _line(seed)
        m = PATTERN.fullmatch(q.text)
        assert m, q.text
        ux, uy, ax, ay, bx, by = map(int, m.groups())
        vx, vy = bx - ax, by - ay
        assert (vx, vy) != (0, 0)
        k = Fraction(ux * vx + uy * vy, vx * vx + vy * vy)
        assert q.answer == {"x": k * vx, "y": k * vy}, seed
        assert q.answer_text == f"({k * vx}, {k * vy})"


def test_default_path_unchanged_and_new_kind_differs():
    for seed in range(20):
        default = make("vector_projection", seed=seed, difficulty="hard")
        explicit = make("vector_projection", seed=seed, difficulty="hard", kind=None)
        assert default.text == explicit.text
        assert default.answer_text == explicit.answer_text
        assert re.fullmatch(r"Project u = \(.*\) onto v = \(.*\)\.", default.text)
        assert _line(seed).text != default.text


def test_distractors_rejected_and_answer_accepted_over_200_seeds():
    grader = GRADERS["pair"]
    for seed in range(200):
        q = _line(seed)
        assert grader(q, q.answer_text).correct, seed
        assert q.distractors, seed
        for d in q.distractors:
            assert d != q.answer_text
            assert not grader(q, d).correct, (seed, d, q.answer_text)


def test_deliberately_wrong_answer_rejected():
    q = _line(5)
    wrong = f"({q.answer['x'] + 1}, {q.answer['y']})"
    assert not GRADERS["pair"](q, wrong).correct


def test_prompt_shape_differs_from_easy_and_medium():
    shapes = {prompt_shape(make("vector_projection", seed=1, difficulty=d)) for d in ("easy", "medium")}
    assert prompt_shape(_line(1)) not in shapes


def test_answer_concentration_is_reasonable():
    counts = Counter(_line(s).answer_text for s in range(200))
    assert counts.most_common(1)[0][1] / 200 < 0.25
