"""Tests for the exam tier of seven vector skills (`mathgen/generators/exd_x_g2_vectors_build.py`).

For every topic: (a) prompt variety and a shape the skill's hard tier does not have; (b) the key re-derived from
the RENDERED PROMPT alone with independent code (sympy or exact fractions) over 300 seeds; (c) the skill's real
grader accepts the key and rejects every distractor over 100 seeds. Nothing here imports the generator module.
"""

from __future__ import annotations

import re

import pytest
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

SKILLS = {
    "senior-apply-vector-calculus-to-motion-in-a-plane": ("exd_x_g2_projectile_landing", "number"),
    "senior-consider-position-of-vectors-as-a-function-of": ("exd_x_g2_position_collision", "pair"),
    "senior-define-and-use-multiplication-by-a-scalar-of": ("exd_x_g2_scalar_pair", "number"),
    "senior-determine-equations-of-motion-of-a-particle-travelling": ("exd_x_g2_line_motion_event", "number"),
    "senior-examine-addition-of-complex-numbers-as-vector-addition": ("exd_x_g2_complex_vector_system", "pair"),
    "senior-examine-examples-of-vectors-including-displacement-and-velocity": (
        "exd_x_g2_velocity_composition",
        "number",
    ),
    "senior-examine-properties-of-parallel-and-perpendicular-vectors-and": (
        "exd_x_g2_parallel_perp_condition",
        "number",
    ),
}
TOPICS = [t for t, _ in SKILLS.values()]
SEEDS = range(300)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_t = sp.Symbol("t")


def _q(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


def _expr(text, **local):
    return parse_expr(text, local_dict={"t": _t, **local}, transformations=_TRANSFORMS)


def _ij(text):
    """Parse ``3i - j`` into (3, -1)."""
    out = {"i": 0, "j": 0}
    for sign, num, axis in re.findall(r"([+-]?)\s*(\d*)([ij])\b", text):
        out[axis] += (-1 if sign == "-" else 1) * (int(num) if num else 1)
    return out["i"], out["j"]


def _pair(text):
    a, b = re.fullmatch(r"\((-?\d+), (-?\d+)\)", text.strip()).groups()
    return int(a), int(b)


# -- source wiring, variety, text quality -----------------------------------------------------------------------


@pytest.mark.parametrize("skill_id", SKILLS)
def test_exam_source_is_last_and_in_exam_stage_only(skill_id):
    topic, kind = SKILLS[skill_id]
    # A skill whose exam source has since been replaced by a multi-part chain registers that topic instead.
    topic = {
        "senior-define-and-use-multiplication-by-a-scalar-of": "exd_x_r3e_vector_combination",
        "senior-examine-properties-of-parallel-and-perpendicular-vectors-and": "exd_x_r3f_vector_perpendicular",
    }.get(skill_id, topic)
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    assert skill.answer == kind
    assert len(skill.sources) == 4
    last = skill.sources[-1]
    # Three exam sources moved to multi-part chains (tests/test_exd_x_r2d_velocity_chain.py, tests/test_exd_x_r2c_*.py).
    topic = {
        "exd_x_g2_velocity_composition": "exd_x_r2d_velocity_chain",
        "exd_x_g2_projectile_landing": "exd_x_r2c_projectile",
        "exd_x_g2_position_collision": "exd_x_r2c_collision",
    }.get(topic, topic)
    assert (last.topic, last.difficulty, list(last.stages)) == (topic, "exam", ["exam"])


@pytest.mark.parametrize("skill_id", SKILLS)
def test_prompt_variety_and_new_shapes_over_hard_tier(skill_id):
    topic, _ = SKILLS[skill_id]
    _, skills = load_skills()
    hard = next(s for s in skills if s.id == skill_id).sources[2]
    hard_shapes = {prompt_shape(make(hard.topic, seed=s, **hard.params)) for s in range(60)}
    shapes = {prompt_shape(_q(topic, s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    assert shapes - hard_shapes, "the exam source adds no prompt shape beyond the hard tier"


@pytest.mark.parametrize("topic", TOPICS)
def test_no_text_defects_and_deterministic(topic):
    for seed in range(60):
        q = _q(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == _q(topic, seed).to_json()


# -- (b) independent re-derivation from the rendered prompt -------------------------------------------------------


def _projectile_key(text):
    r0, v0 = re.search(r"r\(0\) = (.*?) m with velocity v\(0\) = (.*?) m/s", text).groups()
    (x0, y0), (vx, vy) = _ij(r0), _ij(v0)
    y = y0 + vy * _t - 5 * _t**2
    if "greatest value" in text:
        return sp.Integer(y.subs(_t, sp.solve(sp.diff(y, _t), _t)[0]))
    land = [r for r in sp.solve(y, _t) if r > 0]
    assert len(land) == 1
    if "i-component" in text:
        return x0 + vx * land[0]
    assert "speed" in text
    return sp.sqrt(vx**2 + (vy - 10 * land[0]) ** 2)


def _collision_key(text):
    p, q = re.search(r"r_P\(t\) = (\(.*?\)) and r_Q\(t\) = (\(.*?\))\. ", text).groups()
    cp = [_expr(c) for c in p[1:-1].split(", ")]
    cq = [_expr(c) for c in q[1:-1].split(", ")]
    sols = sp.solve([cp[0] - cq[0], cp[1] - cq[1]], [_t], dict=True)
    sols = [s for s in sols if s[_t] >= 0]
    assert len(sols) == 1, sols
    t0 = sols[0][_t]
    return {"x": int(cp[0].subs(_t, t0)), "y": int(cp[1].subs(_t, t0))}


def _scalar_key(text):
    r, u, w, ask = re.search(
        r"Write (.*?) in the form p·u \+ q·w, where .* u is the vector (.*?) and w is the vector (.*?)\. "
        r"Find the value of ([pq])\.",
        text,
    ).groups()
    (u1, u2), (w1, w2), (r1, r2) = _ij(u), _ij(w), _ij(r)
    p, q = sp.symbols("p q")
    sol = sp.solve([u1 * p + w1 * q - r1, u2 * p + w2 * q - r2], [p, q])
    return sol[p] if ask == "p" else sol[q]


def _line_key(text):
    d_text = re.search(r"vector d = (.*?)\. ", text).group(1)
    dx, dy = _ij(d_text)
    acc = re.search(r"acceleration vector is (.*?)d m/s²", text).group(1)
    has_t = acc.endswith("t·")
    coeff = acc[:-2] if has_t else acc
    coeff = {"": 1, "-": -1}.get(coeff, None) if coeff in ("", "-") else int(coeff)
    f0_text = re.search(r"velocity vector is (-?\d*)d m/s", text).group(1)
    f0 = {"": 1, "-": -1}.get(f0_text) or int(f0_text)
    accel = coeff * _t if has_t else sp.Integer(coeff)
    v = sp.integrate(accel, _t) + f0
    s = sp.integrate(v, _t)
    axis = re.search(r"Find the ([ij])-component", text).group(1)
    comp = dx if axis == "i" else dy
    if "momentarily at rest" in text:
        t0 = [r for r in sp.solve(v, _t) if r > 0][0]
        r0 = _ij(re.search(r"r\(0\) = (.*?) m\.", text).group(1))
        base = r0[0] if axis == "i" else r0[1]
        return base + s.subs(_t, t0) * comp
    assert "returns to its starting position" in text
    t0 = [r for r in sp.solve(s, _t) if r > 0][0]
    return v.subs(_t, t0) * comp


def _complex_key(text):
    r_txt, eq, s_txt, ask = re.search(
        r"z1 \+ z2 = (.*?), and they also satisfy (.*?) = (.*?)\. Find (z[12]),", text
    ).groups()

    def c(s):
        s = re.sub(r"(\d*)i\b", lambda m: f"({m.group(1) or 1}*I)", s)
        return sp.sympify(s)

    z1, z2 = sp.symbols("z1 z2")
    e2 = _expr(eq, z1=z1, z2=z2)
    sol = sp.solve([z1 + z2 - c(r_txt), e2 - c(s_txt)], [z1, z2])
    z = sp.expand(sol[z1 if ask == "z1" else z2])
    return {"x": int(sp.re(z)), "y": int(sp.im(z))}


def _velocity_key(text):
    if "constant velocity" in text:
        air, n1, w1, n2, w2 = re.search(
            r"constant velocity (.*?) km/h relative to the air\. For (\d+) hours? the wind has velocity (.*?) km/h, "
            r"and for the next (\d+) hours? it has velocity (.*?) km/h",
            text,
        ).groups()
        a, b, c = _ij(air), _ij(w1), _ij(w2)
        n1, n2 = int(n1), int(n2)
        tot = (n1 * (a[0] + b[0]) + n2 * (a[0] + c[0]), n1 * (a[1] + b[1]) + n2 * (a[1] + c[1]))
        return sp.sqrt(tot[0] ** 2 + tot[1] ** 2)
    air, wind = re.search(
        r"velocity (.*?) km/h relative to the air\. The wind blows with velocity (.*?) km/h\.", text
    ).groups()
    a, w = _ij(air), _ij(wind)
    speed = sp.sqrt((a[0] + w[0]) ** 2 + (a[1] + w[1]) ** 2)
    m = re.search(r"after (\d+) hours", text)
    return speed * (int(m.group(1)) if m else 1)


def _parallel_key(text):
    u, v = (
        _pair(re.search(r"u is the vector (\(.*?\)) and v", text).group(1)),
        _pair(re.search(r"v is the vector (\(.*?\))[.] ", text).group(1)),
    )
    k = sp.Symbol("k")
    c = (u[0] + k * v[0], u[1] + k * v[1])
    m = re.search(r"Let w be the vector (\d)u - v", text)
    if m:
        n = int(m.group(1))
        w = (n * u[0] - v[0], n * u[1] - v[1])
    else:
        w = _pair(re.search(r"(?:parallel|perpendicular) to the vector (\(.*?\))\.", text).group(1))
    eq = c[0] * w[0] + c[1] * w[1] if "perpendicular" in text else c[0] * w[1] - c[1] * w[0]
    sol = sp.solve(eq, k)
    assert len(sol) == 1
    return sol[0]


ORACLES = {
    "exd_x_g2_projectile_landing": _projectile_key,
    "exd_x_g2_position_collision": _collision_key,
    "exd_x_g2_scalar_pair": _scalar_key,
    "exd_x_g2_line_motion_event": _line_key,
    "exd_x_g2_complex_vector_system": _complex_key,
    "exd_x_g2_velocity_composition": _velocity_key,
    "exd_x_g2_parallel_perp_condition": _parallel_key,
}


@pytest.mark.parametrize("topic", TOPICS)
def test_key_matches_independent_derivation_from_prompt(topic):
    oracle = ORACLES[topic]
    disagreements = []
    for seed in SEEDS:
        q = _q(topic, seed)
        expected = oracle(q.text)
        if isinstance(expected, dict):
            if q.answer != expected:
                disagreements.append((seed, q.answer, expected))
        else:
            assert sp.nsimplify(expected).is_Integer, (seed, expected, q.text)
            if q.answer != int(expected):
                disagreements.append((seed, q.answer, expected))
    assert disagreements == [], disagreements[:3]


# -- (c) the real grader accepts the key and rejects every distractor -------------------------------------------


@pytest.mark.parametrize("skill_id", SKILLS)
def test_grader_accepts_key_and_rejects_every_distractor(skill_id):
    topic, kind = SKILLS[skill_id]
    grader = GRADERS[kind]
    for seed in range(150):
        q = _q(topic, seed)
        assert grader(q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, (seed, q.distractors)
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert not grader(q, d).correct, (seed, d, q.answer_text)


@pytest.mark.parametrize("topic", ["exd_x_g2_projectile_landing", "exd_x_g2_line_motion_event"])
def test_all_prompt_variants_occur(topic):
    markers = {
        "exd_x_g2_projectile_landing": ("lands", "greatest value", "speed"),
        "exd_x_g2_line_motion_event": ("returns to its starting position", "momentarily at rest"),
    }[topic]
    texts = [_q(topic, s).text for s in range(120)]
    for marker in markers:
        assert any(marker in t for t in texts), marker


_PASTE_SKILLS = (
    "senior-examine-properties-of-parallel-and-perpendicular-vectors-and",
    "senior-define-and-use-multiplication-by-a-scalar-of",
)


def _pasted_fragments(text):
    """What a learner could paste back: the whole prompt, each sentence, every piece around an '=', and
    every vector written in the prompt."""
    pieces = [text]
    pieces += [s.strip() for s in re.split(r"(?<=[.?])\s+", text)]
    for left, right in zip(text.split("=")[:-1], text.split("=")[1:]):
        pieces += [left.strip(), right.strip(), right.split(".")[0].strip(), left.split(".")[-1].strip()]
    pieces += re.findall(r"\(-?\d+, -?\d+\)", text)
    pieces += re.findall(r"-?\d*i [+-] \d*j|-?\d*j|-?\d*i", text)
    return [p for p in pieces if p]


@pytest.mark.parametrize("skill_id", _PASTE_SKILLS)
def test_pasting_the_prompt_or_part_of_it_is_never_graded_correct(skill_id):
    topic, kind = SKILLS[skill_id]
    grader = GRADERS[kind]
    bad = []
    for seed in range(200):
        q = _q(topic, seed)
        for piece in _pasted_fragments(q.text):
            if piece.strip() != q.answer_text and grader(q, piece).correct:
                bad.append((seed, piece, q.answer_text))
    assert bad == [], bad[:5]
