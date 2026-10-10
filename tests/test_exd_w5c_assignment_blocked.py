"""Optimum-assignment hard tier with blocked (not possible) cells."""

from __future__ import annotations

import hashlib
import json
import re
from itertools import permutations

from mathgen.grading import GRADERS
from mathgen import make
from mathgen.text_quality import prompt_shape

TOPIC = "determine_the_optimum_assignment_s_by_inspection_for"
SKILL_ID = "senior-determine-the-optimum-assignment-s-by-inspection-for"
# sha256 of (text, answer_text, distractors) for seeds 0-19 x easy/medium/hard,
# taken from the generator as it stood at HEAD before the blocked option.
DEFAULT_PIN = "c61c44f68fa02f00533bfa5032615bc63c92668ece473b19135b88a8f40d4238"

_ROW = re.compile(r"Worker \d: ([^;.]*(?:\$\d+|not possible)[^;.]*)")
_CELL = re.compile(r"Task [A-D] (?:costs \$(\d+)|is not possible)")


def _parse(text):
    rows = []
    for m in _ROW.finditer(text):
        rows.append([None if c.group(1) is None else int(c.group(1)) for c in _CELL.finditer(m.group(1))])
    return rows


def _dp_min(rows):
    """Bitmask DP over workers: a different route from permutation search."""
    n = len(rows)
    inf = float("inf")
    best = {0: 0}
    for i in range(n):
        nxt = {}
        for mask, cost in best.items():
            for j in range(n):
                if mask >> j & 1 or rows[i][j] is None:
                    continue
                key = mask | 1 << j
                nxt[key] = min(nxt.get(key, inf), cost + rows[i][j])
        best = nxt
    return best[(1 << n) - 1]


def _blocked(seed):
    return make(TOPIC, seed=seed, difficulty="hard", blocked=3)


def test_blocked_prompt_parses_and_key_matches_recomputation():
    for seed in range(60):
        q = _blocked(seed)
        rows = _parse(q.text)
        assert len(rows) == 4 and all(len(r) == 4 for r in rows), q.text
        assert sum(c is None for r in rows for c in r) == 3, q.text
        assert "cannot be assigned a task marked not possible" in q.text
        assert "\n" not in q.text
        assert (
            q.answer
            == _dp_min(rows)
            == min(
                sum(rows[i][p[i]] for i in range(4))
                for p in permutations(range(4))
                if all(rows[i][p[i]] is not None for i in range(4))
            )
        )


def test_blocked_distractors_are_computable_values_distinct_from_the_answer():
    for seed in range(60):
        q = _blocked(seed)
        assert len(q.distractors) >= 3
        assert len(set(q.distractors)) == len(q.distractors)
        assert q.answer_text not in q.distractors


def test_default_path_is_unchanged():
    out = {d: [] for d in ("easy", "medium", "hard")}
    for d in out:
        for seed in range(20):
            q = make(TOPIC, seed=seed, difficulty=d)
            assert "not possible" not in q.text
            out[d].append([q.text, q.answer_text, list(q.distractors)])
    assert hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest() == DEFAULT_PIN
    assert make(TOPIC, seed=1, difficulty="hard").text != _blocked(1).text


def test_blocked_distractors_clear_the_real_grader():
    grader = GRADERS["number"]
    for seed in range(60):
        q = _blocked(seed)
        assert grader(q, q.answer_text).correct
        for distractor in q.distractors:
            assert not grader(q, distractor).correct, (seed, distractor, q.answer_text)
        assert not grader(q, str(q.answer + 1)).correct


def test_blocked_prompt_shape_differs_from_default_hard():
    shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="hard")) for s in range(5)}
    blocked = {prompt_shape(_blocked(s)) for s in range(5)}
    assert not (shapes & blocked)
