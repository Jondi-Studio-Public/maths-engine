"""Tests `uc2-partial-order-hasse` (topic `r4j_partial_order_hasse`) and its exam chain.

The oracle reads the set out of the TEXT and applies the definitions directly: minimal and maximal by comparison, covers
by searching for an element in between, longest chains by listing every chain, antichains by trying every subset. The
fewest covering chains is checked against Dilworth's theorem through a maximum matching on the comparability graph.
"""

from __future__ import annotations

import itertools
import re

import mathgen.generators.r4j_partial_order_hasse_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import load_skills
from tests.r4j_support import check_carried, check_parts, check_single

TOPIC = "r4j_partial_order_hasse"
EXAM = "exd_x_r4j_partial_order_hasse"
SKILL = next(s for s in load_skills()[1] if s.id == "uc2-partial-order-hasse")


def _set(text):
    return [int(v) for v in re.search(r"\{([\d, ]+)\}", text).group(1).split(",")]


def _below(s, x):
    return [y for y in s if y != x and x % y == 0]


def _chains(s):
    out = []
    for r in range(1, len(s) + 1):
        for sub in itertools.combinations(sorted(s), r):
            if all(b % a == 0 for a, b in zip(sub, sub[1:])):
                out.append(sub)
    return out


def _antichain(s):
    best = 0
    for r in range(1, len(s) + 1):
        for sub in itertools.combinations(s, r):
            if all(a % b and b % a for a, b in itertools.combinations(sub, 2)):
                best = r
    return best


def _matching_cover(s):
    """Dilworth: |S| minus a maximum matching in the strict-divisibility bipartite graph."""
    left = {x: [y for y in s if y != x and y % x == 0] for x in s}
    match = {}

    def augment(x, seen):
        for y in left[x]:
            if y not in seen:
                seen.add(y)
                if y not in match or augment(match[y], seen):
                    match[y] = x
                    return True
        return False

    return len(s) - sum(augment(x, set()) for x in s)


def test_skill_metadata():
    assert list(SKILL.year_level) == ["university"] and "university-year-2" in SKILL.tags


def test_easy_minimal_and_maximal():
    seen = set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        s = _set(q.text)
        word = "minimal" if "minimal" in q.text else "maximal"
        seen.add(word)
        want = [x for x in s if not any(y != x and (x % y == 0 if word == "minimal" else y % x == 0) for y in s)]
        assert sorted(q.answer) == sorted(want), (seed, q.text)
        check_single(q, "set")
    assert seen == {"minimal", "maximal"}


def test_medium_covers():
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="medium")
        s = _set(q.text)
        x = (
            int(
                re.search(
                    r"(?:does|cover|covered by|that) (\d+)", q.text.split("Which")[-1] if "Which" in q.text else q.text
                ).group(1)
            )
            if False
            else None
        )
        m = re.search(
            r"Which elements of S does (\d+) cover|Which elements of S cover (\d+)\?|that (\d+) covers|that cover (\d+)",
            q.text,
        )
        x = int(next(g for g in m.groups() if g))
        lower = "does" in q.text.split("under divisibility.")[-1] or "that %d covers" % x in q.text
        if lower:
            below = _below(s, x)
            want = [y for y in below if not any(z != y and z % y == 0 for z in below)]
        else:
            above = [y for y in s if y != x and y % x == 0]
            want = [y for y in above if not any(z != y and y % z == 0 for z in above)]
        assert sorted(q.answer) == sorted(want), (seed, q.text)
        check_single(q, "set")


def test_hard_longest_chain_elements():
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="hard")
        s = _set(q.text)
        chains = _chains(s)
        top = max(len(c) for c in chains)
        want = sorted({e for c in chains if len(c) == top for e in c})
        assert sorted(q.answer) == want, (seed, q.text)
        assert want != sorted(s)
        check_single(q, "set")


def test_exam_chain():
    for seed in range(30):
        q = make(EXAM, seed=seed, difficulty="exam")
        check_parts(q)
        s = _set(q.text)
        assert sorted(q.parts[0].answer) == sorted(x for x in s if not any(y != x and y % x == 0 for y in s))
        assert int(q.parts[1].answer_text) == max(len(c) for c in _chains(s))
        width = _antichain(s)
        assert int(q.parts[2].answer_text) == width == _matching_cover(s) == int(q.parts[3].answer_text)
        check_carried(q, "c", str(width + 1), str(width + 1))
