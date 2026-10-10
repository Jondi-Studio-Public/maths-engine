"""Independent oracle for senior-midpoints-of-the-sides-of-a-quadrilateral-join.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p8d_midpoint_parallelogram_build`) and re-derives the
answer from scratch, using a single general solver that works uniformly for
every tier: build whichever of the four midpoints P, Q, R, S are directly
computable or directly given from the parsed text, then solve the one
parallelogram vector identity (P + R = Q + S) for whichever one is still
missing -- the one the question asks for.
"""

import re
from fractions import Fraction

import pytest

import mathgen
from mathgen.skills import load_skills

SKILL_ID = "senior-midpoints-of-the-sides-of-a-quadrilateral-join"
_VERTEX_RE = re.compile(r"([A-D])\((-?\d+),\s*(-?\d+)\)")
_MIDPOINT_GIVEN_RE = re.compile(r"([PQRS]) = \((-?\d+),\s*(-?\d+)\)")
_TARGET_RE = re.compile(r"find the coordinates of ([A-D]|[PQRS])\.")
_ANSWER_RE = re.compile(r"x = (-?\d+), y = (-?\d+)")

_SIDE = {"P": ("A", "B"), "Q": ("B", "C"), "R": ("C", "D"), "S": ("D", "A")}


def _solve(text):
    """Return the independently-derived (x, y) for whatever letter the
    question asks for."""
    points = {}
    for letter, x, y in _VERTEX_RE.findall(text):
        points[letter] = (Fraction(x), Fraction(y))
    for letter, x, y in _MIDPOINT_GIVEN_RE.findall(text):
        points[letter] = (Fraction(x), Fraction(y))

    target_match = _TARGET_RE.search(text)
    assert target_match, f"could not find a target letter in: {text}"
    target = target_match.group(1)

    def resolve(letter):
        if letter in points:
            return points[letter]
        u, v = _SIDE[letter]
        if u in points and v in points:
            pu, pv = points[u], points[v]
            mid = (Fraction(pu[0] + pv[0], 2), Fraction(pu[1] + pv[1], 2))
            points[letter] = mid
            return mid
        return None

    P, Q, R, S = resolve("P"), resolve("Q"), resolve("R"), resolve("S")

    # Solve the single vector identity P + R = Q + S for whichever of the
    # four midpoints is still unresolved (must be exactly the target).
    known = {"P": P, "Q": Q, "R": R, "S": S}
    missing = [k for k, v in known.items() if v is None]
    assert missing == [target], f"expected only {target} missing, got {missing} in: {text}"

    if target == "S":
        Sx = P[0] + R[0] - Q[0]
        Sy = P[1] + R[1] - Q[1]
        return (Sx, Sy)
    if target == "R":
        Rx = Q[0] + S[0] - P[0]
        Ry = Q[1] + S[1] - P[1]
        return (Rx, Ry)
    if target == "Q":
        Qx = P[0] + R[0] - S[0]
        Qy = P[1] + R[1] - S[1]
        return (Qx, Qy)
    if target == "P":
        Px = Q[0] + S[0] - R[0]
        Py = Q[1] + S[1] - R[1]
        return (Px, Py)
    raise AssertionError(f"unexpected target letter {target!r}")


def _parse_answer(answer_text):
    m = _ANSWER_RE.search(answer_text)
    assert m, f"could not parse answer_text: {answer_text}"
    return (Fraction(m.group(1)), Fraction(m.group(2)))


def _skill_sources():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return {src.difficulty: src for src in skill.sources}


SOURCES = _skill_sources()
SEEDS = range(60)


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium", "hard"] for s in SEEDS])
def test_oracle_rederives_the_answer(difficulty, seed):
    src = SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve(q.text)
    stated = _parse_answer(q.answer_text)
    assert derived == stated, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


def test_oracle_coverage():
    tiers = ["easy", "medium", "hard"]
    total = len(tiers) * len(SEEDS)
    assert len(SEEDS) >= 60
    assert total >= 180
    print(f"\nOracle re-derived {total} questions ({len(tiers)} tiers x {len(SEEDS)} seeds).")
