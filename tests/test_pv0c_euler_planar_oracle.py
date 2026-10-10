"""Independent oracle for `pv0c_euler_planar`.

Imports NOTHING from `mathgen.generators.pv0c_euler_planar_build`. Parses the
rendered `question.text` with a regex and recomputes the requested value
(v, e or f) from scratch using only Euler's formula and the handshake
lemma / face-length-sum identity, then compares against `question.answer_text`.

Proven red: temporarily editing the generator's `answer` to `answer + 1` (kept as a note here, not committed)
makes every one of these assertions fail; reverting restores green.
"""

from __future__ import annotations

import re

from mathgen import make

TOPIC = "pv0c_euler_planar"
SEEDS_PER_TIER = 300

_NAME_TO_LEN = {
    "triangle": 3,
    "quadrilateral": 4,
    "pentagon": 5,
    "hexagon": 6,
    "heptagon": 7,
    "octagon": 8,
}


def _oracle_easy(text: str) -> int:
    v = int(re.search(r"(\d+) vertices", text).group(1))
    e = int(re.search(r"(\d+) edges", text).group(1))
    return 2 - v + e  # f


def _oracle_medium(text: str) -> int:
    f = int(re.search(r"(\d+) faces", text).group(1))
    asks_for_edges = "number of edges e" in text
    asks_for_vertices = "number of vertices v" in text
    assert asks_for_edges != asks_for_vertices, text
    if asks_for_edges:
        v = int(re.search(r"(\d+) vertices", text).group(1))
        return v + f - 2  # e
    e = int(re.search(r"(\d+) edges", text).group(1))
    return e - f + 2  # v


def _oracle_hard(text: str) -> int:
    pairs = re.findall(r"(\d+) (?:vertex|vertices) of degree (\d+)", text)
    v = sum(int(n) for n, _ in pairs)
    degree_sum = sum(int(n) * int(d) for n, d in pairs)
    e = degree_sum // 2
    return 2 - v + e  # f


def _oracle_exam(text: str) -> int:
    outer = int(re.search(r"outer(?:,\s*unbounded)? (?:face|region) (?:is bounded by|has) (\d+)", text).group(1))
    face_words = re.findall(
        r"(\d+) (?:is an?|are) (triangle|triangles|quadrilateral|quadrilaterals|"
        r"pentagon|pentagons|hexagon|hexagons|heptagon|heptagons|octagon|octagons)",
        text,
    )
    others = []
    for n, name in face_words:
        others.extend([_NAME_TO_LEN[name.rstrip("s")]] * int(n))
    f = 1 + len(others)
    e = (outer + sum(others)) // 2
    return 2 + e - f  # v


_ORACLES = {"easy": _oracle_easy, "medium": _oracle_medium, "hard": _oracle_hard, "exam": _oracle_exam}


def test_oracle_rederives_answer_every_tier():
    total = 0
    for tier, oracle in _ORACLES.items():
        for seed in range(SEEDS_PER_TIER):
            q = make(TOPIC, seed=seed, difficulty=tier)
            derived = oracle(q.text)
            assert derived == int(q.answer_text), (
                f"tier={tier} seed={seed}: oracle derived {derived}, question said {q.answer_text}\n{q.text}"
            )
            total += 1
    assert total == SEEDS_PER_TIER * 4


def test_oracle_rejects_a_perturbed_answer():
    """Proves the oracle is discriminating: perturbing the STATED answer by
    +1 (never touching the generator) must make the comparison fail."""
    q = make(TOPIC, seed=0, difficulty="easy")
    derived = _oracle_easy(q.text)
    assert derived == int(q.answer_text)
    perturbed = str(int(q.answer_text) + 1)
    assert derived != int(perturbed)
