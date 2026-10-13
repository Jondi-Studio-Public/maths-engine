"""Exam sources for the basic-transformations, composition and inverse skills.

Each key is re-derived from the rendered prompt alone, with a small
interpreter for the transformation descriptions that shares no code with the
generator (complex-number rotation and 2*projection-minus-point reflection,
rather than matrices).
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
from sympy import Matrix

from mathgen import make
from mathgen import skills as sm
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

BASIC = ("senior-define-and-use-basic-linear-transformations-dilations-of", "exd_x_g4_lt_basic")
COMPOSITION = ("senior-define-and-use-composition-of-linear-transformations-and", "exd_x_g4_lt_composition")
INVERSE = ("senior-define-and-use-inverses-of-linear-transformations-and", "exd_x_g4_lt_inverse")
ALL = (BASIC, COMPOSITION, INVERSE)


def _skill(skill_id):
    return {s.id: s for s in sm.load_skills()[1]}[skill_id]


def _make(topic, seed):
    return make(topic, seed=seed, difficulty="exam", tier="exam")


@pytest.mark.parametrize("skill_id,topic", ALL)
def test_yaml_has_the_exam_source_last(skill_id, topic):
    sources = _skill(skill_id).sources
    # These three skills' exam sources moved to multi-part chains (tests/test_exd_x_r2d_lt_*.py).
    topic = {
        "exd_x_g4_lt_basic": "exd_x_r2d_lt_reflection_gradient",
        "exd_x_g4_lt_composition": "exd_x_r2d_lt_composition_chain",
        "exd_x_g4_lt_inverse": "exd_x_r2d_lt_inverse_chain",
    }.get(topic, topic)
    assert (sources[-1].topic, sources[-1].difficulty, list(sources[-1].stages)) == (topic, "exam", ["exam"])
    assert len(sources) == 4


@pytest.mark.parametrize("skill_id,topic", ALL)
def test_no_text_defects(skill_id, topic):
    for seed in range(60):
        assert find_text_defects(_make(topic, seed)) == []


@pytest.mark.parametrize("skill_id,topic", ALL)
def test_prompt_shapes_are_varied_and_new_over_hard(skill_id, topic):
    hard = next(s for s in _skill(skill_id).sources if s.difficulty == "hard")
    params = {k: tuple(v) if isinstance(v, list) else v for k, v in hard.params.items()}
    hard_shapes = {prompt_shape(make(hard.topic, seed=s, difficulty="hard", **params)) for s in range(60)}
    shapes = {prompt_shape(_make(topic, s)) for s in range(60)}
    assert len(shapes) >= 3
    assert len(shapes - hard_shapes) >= 3


@pytest.mark.parametrize("skill_id,topic", ALL)
def test_deterministic(skill_id, topic):
    assert _make(topic, 11).to_json() == _make(topic, 11).to_json()


# -- the interpreter ------------------------------------------------------------------------

_DILATION = re.compile(r"the dilation \(x, y\) -> \((-?\d+)x, (-?\d+)y\)")
_ROT_SPECIAL = re.compile(r"a rotation of (\d+) degrees anticlockwise about the origin")
_ROT_GENERAL = re.compile(
    r"a rotation about the origin through an angle theta, where cos\(theta\) = (-?\d+)/(\d+) and sin\(theta\) = (-?\d+)/(\d+)"
)
_REFLECT = re.compile(r"a reflection in (?:the )?(x-axis|y-axis|line y = x|line y = -x|line y = (-?\d*)x(?:/(\d+))?)")


def _apply(desc: str, P):
    x, y = Fraction(P[0]), Fraction(P[1])
    if m := _DILATION.fullmatch(desc):
        return (int(m[1]) * x, int(m[2]) * y)
    if m := _ROT_SPECIAL.fullmatch(desc):
        for _ in range(int(m[1]) // 90):
            x, y = -y, x
        return (x, y)
    if m := _ROT_GENERAL.fullmatch(desc):
        assert m[2] == m[4]
        c, s = Fraction(int(m[1]), int(m[2])), Fraction(int(m[3]), int(m[4]))
        assert c * c + s * s == 1
        return (c * x - s * y, s * x + c * y)
    m = _REFLECT.fullmatch(desc)
    assert m, desc
    d = {"x-axis": (1, 0), "y-axis": (0, 1), "line y = x": (1, 1), "line y = -x": (1, -1)}.get(m[1])
    if d is None:
        d = (int(m[3] or 1), {"": 1, "-": -1}.get(m[2], None) or int(m[2]))
    dd = d[0] ** 2 + d[1] ** 2
    t = (x * d[0] + y * d[1]) / dd
    return (2 * t * d[0] - x, 2 * t * d[1] - y)


def _pt(text):
    return tuple(int(v) for v in text)


def _check_int(P):
    assert all(Fraction(c).denominator == 1 for c in P), P
    return {"x": int(P[0]), "y": int(P[1])}


def _oracle_basic(text):
    m = re.fullmatch(
        r"Find the image of the point \((-?\d+), (-?\d+)\) when it is reflected in the line y = (-?\d*)x(?:/(\d+))?\.",
        text,
    )
    if m:
        P = (int(m[1]), int(m[2]))
        return _check_int(_apply(f"a reflection in the line y = {m[3]}x" + (f"/{m[4]}" if m[4] else ""), P))
    m = re.fullmatch(
        r"The line L passes through the origin and the point \((-?\d+), (-?\d+)\)\. "
        r"Find the image of the point \((-?\d+), (-?\d+)\) when it is reflected in L\.",
        text,
    )
    if m:
        q, p = int(m[1]), int(m[2])
        return _check_int(_apply(f"a reflection in the line y = {p}x/{q}", (int(m[3]), int(m[4]))))
    m = re.fullmatch(r"Find the image of the point \((-?\d+), (-?\d+)\) under (a rotation.*)\.", text)
    assert m, text
    return _check_int(_apply(m[3], (int(m[1]), int(m[2]))))


_DESC = r"(?:the dilation \(x, y\) -> \([^)]*\)|a rotation of \d+ degrees anticlockwise about the origin|a rotation about the origin through an angle theta, where cos\(theta\) = -?\d+/\d+ and sin\(theta\) = -?\d+/\d+|a reflection in [^,.]*?)"


def _oracle_composition(text):
    m = re.fullmatch(
        r"Let (.+?)\. Find the image of the point \((-?\d+), (-?\d+)\) under the composite ([FGH o]+), where \(G o F\)\(P\) means G\(F\(P\)\)\.",
        text,
    )
    if m:
        defs = {}
        for part in re.split(r"(?:, | and )(?=[FGH] be )", m[1]):
            name, desc = part.split(" be ", 1)
            defs[name] = desc
        P = (int(m[2]), int(m[3]))
        for name in reversed(m[4].split(" o ")):  # right-hand map acts first
            P = _apply(defs[name], P)
        return _check_int(P)
    m = re.fullmatch(
        r"A transformation T is (.+?), followed by (.+?)\. T maps a point P to the image \((-?\d+), (-?\d+)\)\. Find P\.",
        text,
    )
    assert m, text
    return _solve_preimage(m[1], m[2], (int(m[3]), int(m[4])))


def _matrix_of(desc):
    cols = [_apply(desc, (1, 0)), _apply(desc, (0, 1))]
    return Matrix([[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]])


def _solve_preimage(first, second, Q):
    M = _matrix_of(second) * _matrix_of(first)
    P = M.LUsolve(Matrix([Q[0], Q[1]]))
    return {"x": int(P[0]), "y": int(P[1])}


def _oracle_inverse(text):
    m = re.fullmatch(
        r"A linear transformation has matrix \[(-?\d+) (-?\d+); (-?\d+) (-?\d+)\] and maps a point P to the image \((-?\d+), (-?\d+)\)\. Find P\.",
        text,
    )
    if m:
        M = Matrix(2, 2, [int(g) for g in m.groups()[:4]])
        P = M.LUsolve(Matrix([int(m[5]), int(m[6])]))
        return {"x": int(P[0]), "y": int(P[1])}
    m = re.fullmatch(
        r"The transformation T is (.+?), followed by (.+?)\. Find the point that T sends to \((-?\d+), (-?\d+)\)\.",
        text,
    )
    if m:
        return _solve_preimage(m[1], m[2], (int(m[3]), int(m[4])))
    m = re.fullmatch(r"(A rotation.*), maps a point P to the image \((-?\d+), (-?\d+)\)\. Find P\.", text)
    assert m, text
    desc = m[1][0].lower() + m[1][1:]
    P = _matrix_of(desc).LUsolve(Matrix([int(m[2]), int(m[3])]))
    return {"x": int(P[0]), "y": int(P[1])}


ORACLES = {BASIC[1]: _oracle_basic, COMPOSITION[1]: _oracle_composition, INVERSE[1]: _oracle_inverse}


@pytest.mark.parametrize("skill_id,topic", ALL)
def test_key_matches_independent_oracle_over_300_seeds(skill_id, topic):
    bad = []
    for seed in range(300):
        q = _make(topic, seed)
        if ORACLES[topic](q.text) != q.answer:
            bad.append((seed, q.text, q.answer))
    assert bad == []


@pytest.mark.parametrize("skill_id,topic", ALL)
def test_grader_accepts_key_and_rejects_every_distractor_over_150_seeds(skill_id, topic):
    grade = GRADERS["pair"]
    bad = []
    for seed in range(150):
        q = _make(topic, seed)
        assert len(q.distractors) >= 2
        if not grade(q, q.answer_text).correct:
            bad.append(("key", seed))
        bad += [("distractor", seed, d) for d in q.distractors if grade(q, d).correct]
    assert bad == []
