"""W5b: coordinates kind=fourth_vertex (hard tier of coordinate-plane-readiness)."""

import re
from collections import Counter

import pytest

from mathgen.grading import GRADERS
from mathgen import make

# Captured before the edit (EXD-001: the default read path must not change).
OLD_TEXTS = {
    (1, "easy"): "A point is 3 units left and 3 units up from the origin. Write its coordinates (x, y).",
    (1, "hard"): "A point is 10 units right and 13 units down from the origin. Write its coordinates (x, y).",
    (2, "easy"): "A point is 3 units right and 0 units up from the origin. Write its coordinates (x, y).",
    (2, "hard"): "A point is 14 units left and 6 units up from the origin. Write its coordinates (x, y).",
    (3, "easy"): "A point is 5 units right and 4 units up from the origin. Write its coordinates (x, y).",
    (3, "hard"): "A point is 3 units right and 3 units down from the origin. Write its coordinates (x, y).",
}

SEEDS = range(300)
PAIR = re.compile(r"[A-C]\((-?\d+), (-?\d+)\)")


def _hard(seed):
    return make("coordinates", seed=seed, difficulty="hard", kind="fourth_vertex")


@pytest.mark.parametrize("key", sorted(OLD_TEXTS))
def test_read_path_unchanged(key):
    seed, diff = key
    assert make("coordinates", seed=seed, difficulty=diff).text == OLD_TEXTS[key]


def test_answer_rederived_from_prompt():
    for seed in SEEDS:
        q = _hard(seed)
        pts = [(int(a), int(b)) for a, b in PAIR.findall(q.text)]
        assert len(pts) == 3
        xs, ys = Counter(p[0] for p in pts), Counter(p[1] for p in pts)
        (ex,) = [v for v, c in xs.items() if c == 1]
        (ey,) = [v for v, c in ys.items() if c == 1]
        assert q.answer == {"x": ex, "y": ey}, seed
        assert q.answer_text == f"({ex}, {ey})"
        assert max(abs(v) for p in pts for v in p) <= 15


def test_distractors_distinct_and_answer_present():
    for seed in SEEDS:
        q = _hard(seed)
        assert len(q.distractors) == 3, seed
        assert len(set(q.distractors)) == 3
        assert q.answer_text not in q.distractors
        vals = []
        for d in q.distractors:
            m = re.fullmatch(r"\((-?\d+), (-?\d+)\)", d)
            vals.append((int(m.group(1)), int(m.group(2))))
        assert (q.answer["x"], q.answer["y"]) not in vals
        assert len(set(vals)) == 3


def test_graded_through_api():
    grader = GRADERS["pair"]
    for seed in range(150):
        q = _hard(seed)
        assert grader(q, q.answer_text).correct, seed
        for d in q.distractors:
            assert not grader(q, d).correct, (seed, d)
