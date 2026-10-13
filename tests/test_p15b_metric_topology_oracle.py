"""Independent oracle for the `p15b_metric_topology` questions: it re-derives each answer from the rendered
question text.

Parses ONLY `question.text`, `question.answer_text` and `question.distractors`
(via `mathgen.grading.mcq_options`) -- never anything from
`mathgen.generators.p15b_metric_topology_build` -- and re-derives each answer
from scratch: the axiom classification via a fresh brute-force check over the
parsed distance table, and the subcover choice via a fresh coverage check
over the parsed sets.
"""

import re

import pytest

from mathgen.grading.typed import mcq_options
from mathgen.skills import load_skills
from types import SimpleNamespace
from mathgen import make

METRIC_ID = "ucv1-metric-axiom-check"
COMPACT_ID = "ucv1-compactness-finite-check"
SEEDS = range(1000)
TIERS = ["easy", "medium", "hard", "exam"]

_METRIC_OPTIONS = {
    "none": "No axiom fails: the table defines a genuine metric.",
    "identity": "The identity axiom fails: some point has nonzero self-distance.",
    "zero": "The identity of indiscernibles fails: two distinct points have distance 0.",
    "triangle": "The triangle inequality fails for some triple of points.",
}


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


METRIC_SOURCES = _skill_sources(METRIC_ID)
COMPACT_SOURCES = _skill_sources(COMPACT_ID)
# The skill's exam tier is now a multi-part chain (tests/test_exd_x_r2g_subcover_count.py). These tests keep covering the
# generator's own earlier exam preset, which is unchanged.
COMPACT_SOURCES["exam"] = SimpleNamespace(
    topic="p15b_compactness_subcover_check", difficulty="exam", params={}, stages=["exam"]
)


# ==========================================================================
# ucv1-metric-axiom-check
# ==========================================================================

_PAIR_RE = re.compile(r"d\(([A-Z]), ([A-Z])\) = (-?\d+)")


def _parse_table(text):
    """Pull every d(X, Y) = v triple out of the rendered text (prose list OR
    the matrix form -- the matrix form ALSO states the point order in
    prose, so parse it from the bmatrix body positionally instead)."""
    pairs = _PAIR_RE.findall(text)
    if pairs:
        labels = sorted({a for a, _b, _v in pairs} | {b for _a, b, _v in pairs})
        idx = {lab: i for i, lab in enumerate(labels)}
        n = len(labels)
        dist = [[0] * n for _ in range(n)]
        for a, b, v in pairs:
            dist[idx[a]][idx[b]] = int(v)
            dist[idx[b]][idx[a]] = int(v)
        return dist, n

    # matrix form: "points (A, B, C, D):\quad \begin{bmatrix}...\end{bmatrix}"
    header_m = re.search(r"points.*?\(([A-Z](?:, [A-Z])*)\)", text)
    assert header_m, text
    labels = header_m.group(1).split(", ")
    n = len(labels)
    body_m = re.search(r"\\begin\{bmatrix\}(.+?)\\end\{bmatrix\}", text)
    assert body_m, text
    rows = body_m.group(1).split(r"\\")
    dist = [[int(x.strip()) for x in row.split("&")] for row in rows]
    assert len(dist) == n and all(len(r) == n for r in dist), text
    return dist, n


def _classify(dist, n):
    identity_ok = all(dist[i][i] == 0 for i in range(n))
    zero_ok = all(dist[i][j] != 0 for i in range(n) for j in range(n) if i != j)
    triangle_ok = all(dist[i][j] <= dist[i][k] + dist[k][j] for i in range(n) for j in range(n) for k in range(n))
    if identity_ok and zero_ok and triangle_ok:
        return "none"
    if not identity_ok and zero_ok and triangle_ok:
        return "identity"
    if identity_ok and not zero_ok and triangle_ok:
        return "zero"
    if identity_ok and zero_ok and not triangle_ok:
        return "triangle"
    return None


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_rederives_metric_axiom_classification(difficulty, seed):
    src = METRIC_SOURCES[difficulty]
    q = make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    dist, n = _parse_table(q.text)
    derived = _classify(dist, n)
    assert derived is not None, (q.text, dist)
    assert _METRIC_OPTIONS[derived] == q.answer_text, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\n"
        f"derived: {_METRIC_OPTIONS[derived]}\nstated: {q.answer_text}"
    )
    options = mcq_options(q)
    assert set(options) == set(_METRIC_OPTIONS.values())


# ==========================================================================
# ucv1-compactness-finite-check
# ==========================================================================

_SET_RE = re.compile(r"U(\d+) = \{([^}]*)\}")
_SUBFAMILY_RE = re.compile(r"U\d+")


def _parse_cover(text):
    universe_m = re.search(r"^X = \{([^}]*)\}", text)
    assert universe_m, text
    universe = frozenset(x.strip() for x in universe_m.group(1).split(",") if x.strip())
    sets = {}
    for num, members in _SET_RE.findall(text):
        sets[int(num)] = frozenset(x.strip() for x in members.split(",") if x.strip())
    assert sets, text
    return universe, sets


def _covers(indices, sets, universe):
    union: set = set()
    for i in indices:
        union |= sets[i]
    return union == universe


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_rederives_the_subcover_choice(difficulty, seed):
    src = COMPACT_SOURCES[difficulty]
    q = make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    universe, sets = _parse_cover(q.text)

    options = mcq_options(q)
    covering_options = []
    for opt in options:
        indices = [int(x[1:]) for x in _SUBFAMILY_RE.findall(opt)]
        if _covers(indices, sets, universe):
            covering_options.append((opt, len(indices)))

    if difficulty == "exam":
        # the smallest COVERING option is the answer
        assert covering_options, (q.text, options)
        smallest = min(size for _opt, size in covering_options)
        winners = [opt for opt, size in covering_options if size == smallest]
        assert len(winners) == 1, (q.text, covering_options)
        assert winners[0] == q.answer_text, (
            f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {winners[0]}\nstated: {q.answer_text}"
        )
    else:
        assert covering_options == [(q.answer_text, len(_SUBFAMILY_RE.findall(q.answer_text)))], (
            f"seed={seed} difficulty={difficulty}\ntext: {q.text}\n"
            f"covering options found: {covering_options}\nstated: {q.answer_text}"
        )


def test_oracle_coverage():
    total = len(TIERS) * len(SEEDS) * 2  # both skills
    assert len(SEEDS) >= 60
    assert total >= 480
    print(f"\nOracle re-derived {total} questions ({len(TIERS)} tiers x {len(SEEDS)} seeds x 2 skills).")
