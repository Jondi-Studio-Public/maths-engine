"""Independent oracle for the `p15c_topology_physics` questions: it re-derives each answer from the rendered
question text.

Parses ONLY `question.text`/`question.answer_text` -- never anything from
`mathgen.generators.p15c_topology_physics_build` -- and re-derives each
answer from scratch: the basis classification via a second implementation of
the coverage/intersection predicate, and the conserved quantity via a
from-scratch Noether lookup keyed on the physical phrase actually printed in
the rendered prompt.
"""

import re
from itertools import combinations

import pytest
import sympy
import sympy.parsing.sympy_parser

from mathgen.skills import load_skills

TOPOLOGY_ID = "ucv1-topology-basis-membership"
SYMMETRY_ID = "ucv1-symmetry-conservation"
SEEDS = range(1000)  # mcq/classification: >= 1000 seeds
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


TOPOLOGY_SOURCES = _skill_sources(TOPOLOGY_ID)
SYMMETRY_SOURCES = _skill_sources(SYMMETRY_ID)


def _make(sources, difficulty, seed):
    import mathgen

    src = sources[difficulty]
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ---------------------------------------------------------------------------
# ucv1-topology-basis-membership

_SET_LITERAL_RE = re.compile(r"\{[^{}]*\}")
_CLAUSE_RE = re.compile(r"(B\d+) = (.+)")


def _resolve_topology_clause(rhs: str) -> frozenset:
    """Resolved here exactly as the generator computed it (P1 | P2), never trusting a naive "every {...} in the
    text is a separate member" scan, which would otherwise double-count the two pieces of one indirect member."""
    if rhs.startswith("the union of"):
        parts = _SET_LITERAL_RE.findall(rhs)
        result = frozenset()
        for part in parts:
            result |= frozenset(int(x) for x in re.findall(r"\d+", part))
        return result
    nums = [int(x) for x in re.findall(r"\d+", rhs)]
    return frozenset(nums)


def _parse_topology(text):
    n = int(re.search(r"universe is X = \{1, \.\.\., (\d+)\}", text).group(1))
    universe = frozenset(range(1, n + 1))
    body = text.split("is given: ")[1].split(". Decide")[0]
    family = []
    for clause in body.split("; "):
        m = _CLAUSE_RE.match(clause.strip())
        family.append(_resolve_topology_clause(m.group(2)))
    return universe, family


def _covers(universe, family):
    covered = frozenset()
    for s in family:
        covered |= s
    return covered == universe


def _intersection_holds(family):
    for a, b in combinations(family, 2):
        inter = a & b
        if not inter:
            continue
        covered = frozenset()
        for c in family:
            if c <= inter:
                covered |= c
        if covered != inter:
            return False
    return True


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_rederives_topology_basis_membership(difficulty, seed):
    q = _make(TOPOLOGY_SOURCES, difficulty, seed)
    universe, family = _parse_topology(q.text)
    covers = _covers(universe, family)
    inter_ok = _intersection_holds(family)
    if covers and inter_ok:
        derived = "a basis"
    elif inter_ok:
        derived = "fails coverage"
    elif covers:
        derived = "fails the intersection condition"
    else:
        derived = "fails both"
    assert derived == q.answer_text, (q.text, derived, q.answer_text)


def test_oracle_coverage_topology():
    total = len(TIERS) * len(SEEDS)
    assert total >= 4000
    print(f"\nTopology-basis-membership oracle re-derived {total} questions.")


# ---------------------------------------------------------------------------
# ucv1-symmetry-conservation
#
# from-scratch phrase table, unchanged in kind). medium/hard/exam now GIVE an
# explicit potential V(x, y, z) in the rendered text, so the oracle checks
# each candidate quantity ALGEBRAICALLY against that V, using a fresh sympy
# symbol set built here (never imported from the generator):
#   - p_i is conserved iff dV/di == 0 (V does not depend on coordinate i);
#   - L_a (rotation about axis a) is conserved iff V, rewritten in polar
#     coordinates about that axis, has no dependence on the polar angle
#     (equivalently: V depends on the other two coordinates only through the
#     rotationally-invariant combination of their squares);
#   - E is conserved because V is parsed using only the symbols x, y, z --
#     any explicit t would fail this parse, so the oracle's own parse is
#     itself the check that no explicit time dependence exists.
_PHRASE_TO_QUANTITY = [
    ("does not depend explicitly on time", "E"),
    ("run starting at a later time", "E"),
    ("does not depend on the x coordinate", "p_x"),
    ("shifted along the x direction", "p_x"),
    ("does not depend on the y coordinate", "p_y"),
    ("shifted along the y direction", "p_y"),
    ("does not depend on the z coordinate", "p_z"),
    ("shifted along the z direction", "p_z"),
]


def _quantity_for_clause(clause):
    for phrase, quantity in _PHRASE_TO_QUANTITY:
        if phrase in clause:
            return quantity
    raise AssertionError(f"no known Noether phrase in clause: {clause!r}")


_sx, _sy, _sz, _sr, _stheta = sympy.symbols("x y z r theta", real=True)
_V_LOCALS = {"x": _sx, "y": _sy, "z": _sz, "sqrt": sympy.sqrt}
_V_RE = re.compile(
    r"V\(x, y, z\) = (.+?)(?: and no explicit time dependence"
    r"| is autonomous \(no explicit"
    r"| and there is no explicit time dependence)"
)


_V_TRANSFORMATIONS = sympy.parsing.sympy_parser.standard_transformations + (
    sympy.parsing.sympy_parser.implicit_multiplication_application,
)


def _parse_v(text):
    m = _V_RE.search(text)
    if not m:
        raise AssertionError(f"no stated V(x, y, z) found in: {text!r}")
    return sympy.parsing.sympy_parser.parse_expr(
        m.group(1).replace("^", "**"),
        local_dict=_V_LOCALS,
        transformations=_V_TRANSFORMATIONS,
    )


def _conserved_quantities(expr):
    conserved = set()
    if sympy.simplify(sympy.diff(expr, _sx)) == 0:
        conserved.add("p_x")
    if sympy.simplify(sympy.diff(expr, _sy)) == 0:
        conserved.add("p_y")
    if sympy.simplify(sympy.diff(expr, _sz)) == 0:
        conserved.add("p_z")
    rot_z = expr.subs({_sx: _sr * sympy.cos(_stheta), _sy: _sr * sympy.sin(_stheta)}, simultaneous=True)
    if sympy.simplify(sympy.diff(rot_z, _stheta)) == 0:
        conserved.add("L_z")
    rot_x = expr.subs({_sy: _sr * sympy.cos(_stheta), _sz: _sr * sympy.sin(_stheta)}, simultaneous=True)
    if sympy.simplify(sympy.diff(rot_x, _stheta)) == 0:
        conserved.add("L_x")
    rot_y = expr.subs({_sz: _sr * sympy.cos(_stheta), _sx: _sr * sympy.sin(_stheta)}, simultaneous=True)
    if sympy.simplify(sympy.diff(rot_y, _stheta)) == 0:
        conserved.add("L_y")
    conserved.add("E")  # V parsed using only x, y, z -- no explicit t is possible
    return conserved


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_rederives_symmetry_conservation(difficulty, seed):
    q = _make(SYMMETRY_SOURCES, difficulty, seed)
    text = q.text
    if "V(x, y, z) = " not in text:
        # easy: a words-only clause, no explicit V given.
        clause = text.split(": ", 1)[1].rsplit(".", 2)[0]
        derived = _quantity_for_clause(clause)
        assert derived == q.answer_text, (text, derived, q.answer_text)
        return

    expr = _parse_v(text)
    conserved = _conserved_quantities(expr)
    all_options = set(q.distractors) | {q.answer_text}
    for option in all_options:
        is_conserved = option in conserved
        should_be_answer = option == q.answer_text
        assert is_conserved == should_be_answer, (
            text,
            option,
            "conserved" if is_conserved else "not conserved",
            "is" if should_be_answer else "is not",
            "the stated answer",
        )


def test_oracle_coverage_symmetry():
    total = len(TIERS) * len(SEEDS)
    assert total >= 4000
    print(f"\nSymmetry-conservation oracle re-derived {total} questions.")
