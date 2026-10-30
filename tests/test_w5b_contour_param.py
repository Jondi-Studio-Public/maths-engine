"""``uc2-contour-integrals-parametrisation``: keys grade right, distractors wrong, the integral is recomputed from the stem.

The oracle reads the rendered stem alone (the integrand, the path, the direction), builds the parametrisation for
itself and integrates numerically with Simpson's rule, a route that shares no code with the generator's exact
integrator or its Gauss-Legendre check.
"""

from __future__ import annotations

import cmath
import math
import re

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade as skill_grade
from mathgen.engine import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

from w5a_support import Z, check_grading, check_text, close, numeric, parse

SKILL = "uc2-contour-integrals-parametrisation"
TOPIC = "w5b_contour_param"
EXAM = "exd_x_w5b_contour_param"
TIERS = ("easy", "medium", "hard")
SEEDS = range(1, 41)


def _q(tier: str, seed: int):
    return make(EXAM if tier == "exam" else TOPIC, seed=seed, difficulty=tier)


# -- reading the stem ---------------------------------------------------------------------------------------------------

_END = r"(?=[.,](?:\s|$)|\sand\s|$)"
SEG = re.compile(r"the straight line segment from (.+?) to (.+?)" + _END)
POLY = re.compile(
    r"the path made of the straight segment from (.+?) to (.+?) followed by the straight segment from (.+?) to (.+?)"
    + _END
)
ARC = re.compile(r"the arc of the circle \|(.+?)\| = (\d+) from (.+?) to (.+?), traversed (anticlockwise|clockwise)")
CIRC = re.compile(r"the circle \|(.+?)\| = (\d+), traversed once (anticlockwise|clockwise)")


def _point(text: str) -> complex:
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return complex(sympy.N(parse(text)))


def _centre(inner: str) -> complex:
    return complex(sympy.N(-parse(inner).subs(Z, 0)))


def paths_in(text: str) -> list:
    """Every path named in the stem, in the order it appears, as ('seg', a, b), ('poly', [pts]) or ('arc', c, r, t0, t1)."""
    found = []
    for m in re.finditer(
        f"(?P<poly>{POLY.pattern})|(?P<arc>{ARC.pattern})|(?P<circ>{CIRC.pattern})|(?P<seg>{SEG.pattern})", text
    ):
        if m.group("poly"):
            g = POLY.match(m.group("poly")).groups()
            found.append((m.start(), ("poly", [_point(g[0]), _point(g[1]), _point(g[3])])))
            assert _point(g[1]) == _point(g[2])
        elif m.group("arc"):
            inner, r, p, q, way = ARC.match(m.group("arc")).groups()
            c, r = _centre(inner), float(r)
            t0 = cmath.phase((_point(p) - c) / r)
            t1 = cmath.phase((_point(q) - c) / r)
            if way == "anticlockwise":
                span = (t1 - t0) % (2 * math.pi)
                t1 = t0 + span
            else:
                span = (t0 - t1) % (2 * math.pi)
                t1 = t0 - span
            assert span > 1e-9
            found.append((m.start(), ("arc", c, r, t0, t1)))
        elif m.group("circ"):
            inner, r, way = CIRC.match(m.group("circ")).groups()
            c, r = _centre(inner), float(r)
            found.append((m.start(), ("arc", c, r, 0.0, 2 * math.pi if way == "anticlockwise" else -2 * math.pi)))
        else:
            g = SEG.match(m.group("seg")).groups()
            found.append((m.start(), ("seg", _point(g[0]), _point(g[1]))))
    return [p for _pos, p in sorted(found, key=lambda t: t[0])]


def integrand_in(text: str):
    m = re.search(r"f\(z\) = (.+?)(?:\. |, |\.$)", text)
    if m is None:
        m = re.search(r"integral of (.+?) along", text) or re.search(r"∫_C (.+?) dz", text)
    src = m.group(1)
    src = (
        src.replace("Re(z)", "((z + conj(z))/2)")
        .replace("Im(z)", "((z - conj(z))/(2i))")
        .replace("|z|", "sqrt(z conj(z))")
    )
    expr = parse(src)
    return sympy.lambdify(Z, expr, modules=[{"conjugate": lambda w: w.conjugate()}, "cmath"])


def simpson(fn, a: float, b: float, n: int = 4000) -> complex:
    h = (b - a) / n
    total = fn(a) + fn(b)
    for k in range(1, n):
        total += (4 if k % 2 else 2) * fn(a + k * h)
    return total * h / 3


def integral(f, path) -> complex:
    if path[0] == "seg":
        a, b = path[1], path[2]
        return simpson(lambda t: f(a + (b - a) * t) * (b - a), 0.0, 1.0)
    if path[0] == "poly":
        pts = path[1]
        return sum((integral(f, ("seg", a, b)) for a, b in zip(pts, pts[1:])), 0j)
    _, c, r, t0, t1 = path
    return simpson(lambda t: f(c + r * cmath.exp(1j * t)) * 1j * r * cmath.exp(1j * t), t0, t1)


def answer_of(q) -> complex:
    return numeric(parse(q.answer_text))


# -- tests ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", (*TIERS, "exam"))
def test_answers_and_distractors_grade(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        check_grading(q)
        check_text(q)
        assert not find_text_defects(q)


@pytest.mark.parametrize("tier", (*TIERS, "exam"))
def test_deterministic(tier):
    for seed in (0, 1, 7, 123):
        assert _q(tier, seed).to_json() == _q(tier, seed).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_single_part_with_a_typed_expression_answer(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert not q.parts and q.answer_kind == "expression"
        assert len(q.distractors) >= 3
        assert q.option_notes, "distractors are tagged with the error they stand for"


@pytest.mark.parametrize("tier", TIERS)
def test_integral_is_recomputed_from_the_stem(tier):
    checked = 0
    for seed in range(1, 91):
        q = _q(tier, seed)
        f = integrand_in(q.text)
        ps = paths_in(q.text)
        if len(ps) == 2 and tier == "hard":
            want = integral(f, ps[0]) - integral(f, ps[1])
        else:
            assert len(ps) == 1, (q.text, ps)
            want = integral(f, ps[0])
        assert close(answer_of(q), want, 1e-6), (q.text, q.answer_text, want)
        checked += 1
    assert checked == 90


def test_exam_parts_are_recomputed_from_the_stem():
    for seed in range(1, 91):
        q = _q("exam", seed)
        f = integrand_in(q.text)
        chord, arc = paths_in(q.text)
        assert chord[0] == "seg" and arc[0] == "arc"
        a, b, c, d = (p.answer_text for p in q.parts)
        assert close(numeric(parse(a)), integral(f, chord), 1e-6), (q.text, a)
        i_arc = integral(f, arc)
        assert close(numeric(parse(b)), i_arc, 1e-6), (q.text, b)
        # ML bound: the largest |f| on the arc (sampled densely) times the arc length
        _, centre, r, t0, t1 = arc
        m_max = max(abs(f(centre + r * cmath.exp(1j * (t0 + (t1 - t0) * k / 2000)))) for k in range(2001))
        bound = m_max * r * abs(t1 - t0)
        assert abs(float(c) - bound) <= 0.0051, (q.text, c, bound)
        # part (d) starts from the 2 d.p. M L the stem tells the learner to use, i.e. the answer to part (c)
        assert abs(float(d) - (float(c) - abs(i_arc))) <= 0.0051, (q.text, c, d, abs(i_arc))
        assert bound >= abs(i_arc)
        assert [p.answer_kind for p in q.parts] == ["expression", "expression", "number", "number"]


def test_conj_over_a_closed_circle_is_not_zero():
    """The point of the skill: a closed loop integral of conj(z - c) is 2 pi i r^2, never 0."""
    seen = 0
    for seed in range(1, 200):
        q = _q("hard", seed)
        if "conj(z - " not in q.text and "conj(z)" not in q.text:
            continue
        ps = paths_in(q.text)
        if len(ps) == 1 and ps[0][0] == "arc" and abs(ps[0][4] - ps[0][3]) > 6:
            f = integrand_in(q.text)
            if abs(integral(f, ps[0])) > 1e-6:
                seen += 1
    assert seen >= 5


@pytest.mark.parametrize("tier", (*TIERS, "exam"))
def test_answer_floors(tier):
    answers = {_q(tier, s).answer_text for s in range(60)}
    assert len(answers) >= 6, answers
    if tier == "hard":
        zero = sum(1 for s in range(200) if _q(tier, s).answer_text == "0")
        assert zero <= 70, zero  # a 0 key stays a minority


def test_exam_answer_space_and_shapes():
    answers = [_q("exam", s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    shapes = {prompt_shape(_q("exam", s)) for s in range(60)}
    assert len(shapes) >= 3, shapes


@pytest.mark.parametrize("tier", (*TIERS, "exam"))
def test_three_prompt_shapes(tier):
    assert len({prompt_shape(_q(tier, s)) for s in range(60)}) >= 3


def test_tiers_are_distinct():
    texts = {t: {_q(t, s).text for s in range(1, 25)} for t in TIERS}
    assert not (texts["easy"] & texts["hard"])
    assert not (texts["easy"] & texts["medium"])


def test_unknown_tier_is_an_error():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="nope")
    with pytest.raises(ValueError):
        make(EXAM, seed=1, tier="easy")


def test_wiring():
    skills = load_skills()[1]
    sk = next(s for s in skills if s.id == SKILL)
    assert sk.domain == "complex-analysis" and sk.status == "active"
    by = {s.difficulty: s for s in sk.sources}
    assert set(by) == {"easy", "medium", "hard", "exam"}
    assert by["exam"].topic == EXAM and list(by["exam"].stages) == ["exam"]
    assert all(by[t].topic == TOPIC for t in TIERS)


def test_skill_level_grade_accepts_key_and_rejects_distractor():
    sk = next(s for s in load_skills()[1] if s.id == SKILL)
    for tier in (*TIERS, "exam"):
        for seed in range(1, 15):
            q = _q(tier, seed)
            assert skill_grade(sk, q, q.answer_text).correct
            if not q.parts:
                for dist in q.distractors:
                    assert not skill_grade(sk, q, dist).correct, (tier, seed, dist)


def test_distractor_tags_are_in_the_skill():
    sk = next(s for s in load_skills()[1] if s.id == SKILL)
    listed = {m.id for m in sk.misconceptions}
    used = set()
    for tier in (*TIERS, "exam"):
        for seed in range(1, 200):
            q = _q(tier, seed)
            used |= set(q.option_notes.values())
            for p in q.parts or ():
                used |= set(p.option_notes.values())
    assert used <= listed, used - listed


def test_exam_part_d_follows_the_rounded_bound_of_part_c():
    """The stem says to take M L from part (c) to 2 d.p.; the key and its working must subtract that value."""
    for seed in range(300):
        q = _q("exam", seed)
        c, d = q.parts[2], q.parts[3]
        assert "2 decimal place value from the previous part" in d.prompt
        i_arc = abs(numeric(parse(q.parts[1].answer_text)))
        assert abs(float(d.answer_text) - (float(c.answer_text) - i_arc)) <= 0.00501, (
            seed,
            c.answer_text,
            d.answer_text,
        )
        assert f"{c.answer_text} - " in d.working[0], d.working


def test_exam_decimal_parts_do_not_hand_the_key_a_fixed_rank():
    from collections import Counter

    for idx in (2, 3):
        ranks, steps = Counter(), 0
        for seed in range(300):
            part = _q("exam", seed).parts[idx]
            vals = sorted([float(part.answer_text), *(float(x) for x in part.distractors)])
            ranks[vals.index(float(part.answer_text))] += 1
            gaps = [round(b - a, 2) for a, b in zip(vals, vals[1:])]
            steps += len(set(gaps)) == 1
        assert max(ranks.values()) <= 0.4 * 300, ranks
        assert len(ranks) == 5, ranks
        assert steps <= 0.1 * 300, steps  # evenly spaced options are rare
        structural = sum(
            1
            for s in range(60)
            if len(set(_q("exam", s).parts[idx].option_notes.values()) - {"cx2-arithmetic-slip"}) >= 1
        )
        assert structural >= 55, (
            structural
        )  # a named slip (M L alone, |I| alone, the chord length...) is always among the options


def test_closed_loop_zero_option_is_offered_wherever_it_is_a_slip():
    seen = 0
    for seed in range(1, 300):
        q = _q("hard", seed)
        ps = paths_in(q.text)
        if len(ps) != 1 or ps[0][0] != "arc" or abs(ps[0][4] - ps[0][3]) < 6.2:
            continue
        if q.answer_text == "0":
            continue
        seen += 1
        assert "cx2-param-closed-loop-assumed-zero" in q.option_notes.values(), (seed, q.text)
    assert seen >= 20


def test_unit_steps_read_plus_t_and_values_put_pi_last():
    for seed in range(300):
        for tier in ("easy", "medium", "hard"):
            q = _q(tier, seed)
            assert "1t" not in q.text.replace("11t", "").replace("21t", ""), q.text
            assert "Evaluate the exact value of" not in q.text
            assert not re.search(r"π[a-z]* [+-] \d", q.answer_text), q.answer_text  # rational part comes first
