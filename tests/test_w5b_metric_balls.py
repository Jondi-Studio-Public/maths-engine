"""`uc2-metrics-and-balls-in-rn`: keys graded right, distractors graded wrong, and every answer recomputed from the stem.

The oracle reads the RENDERED text of a question (points, the metric named in words, radii, region inequalities) and
recomputes the answer by brute force, sharing no code with the generator: exact `Fraction` distance comparisons for
membership, a search over small integer vectors for norm-comparison constants, and a numeric grid for areas and
volumes.
"""

from __future__ import annotations

import itertools
import re
from fractions import Fraction

import numpy as np
import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade
from mathgen.text_quality import find_text_defects, prompt_shape
from tests.exd_x_sc2_support import check_all_wrong_earns_nothing, check_deterministic, check_wired, skill, text_of

SKILL = "uc2-metrics-and-balls-in-rn"
TOPIC = "w5b_metric_balls"
EXAM = "exd_x_w5b_metric_balls"
SEEDS = range(60)
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

POINT = re.compile(r"\((-?\d+(?:, -?\d+)+)\)")


def tier_q(tier: str, seed: int):
    return make(EXAM if tier == "exam" else TOPIC, seed=seed, difficulty=tier)


def value_of(text: str):
    """The exact value of an answer as a learner types it: 5√2, 9π/4, 164π(√41)/3, 3/2."""
    s = text.replace("√", "sqrt").replace("π", "pi")
    s = re.sub(r"(\d|\))(?=sqrt|pi)", r"\1*", s)
    s = re.sub(r"pi(?=\()", "pi*", s)
    s = re.sub(r"sqrt(\d+)", r"sqrt(\1)", s)
    return parse_expr(s, transformations=TRANSFORMS)


def points_in(text: str):
    return [tuple(int(v) for v in m.split(", ")) for m in POINT.findall(text)]


def metric_of(text: str):
    """(kind, weights) named in the stem."""
    w = re.search(r"d_w\(x, y\) = (\d+)\|x_1 - y_1\| \+ (\d+)\|x_2 - y_2\|", text)
    if w:
        return "w", (int(w.group(1)), int(w.group(2)))
    if "d_0(x, y) = 0 if" in text:
        return "disc", None
    for word, kind in (("taxicab", "1"), ("Euclidean", "2"), ("maximum", "inf")):
        if word in text:
            return kind, None
    raise AssertionError(text)


def norm_value(kind, w, v):
    """||v|| in the named norm as an exact SymPy number."""
    a = [abs(x) for x in v]
    if kind == "1":
        return sympy.Integer(sum(a))
    if kind == "inf":
        return sympy.Integer(max(a))
    if kind == "2":
        return sympy.sqrt(sum(x * x for x in a))
    if kind == "w":
        return sympy.Integer(sum(wi * x for wi, x in zip(w, a)))
    return sympy.Integer(0 if not any(a) else 1)


def d(kind, w, p, q):
    return norm_value(kind, w, [x - y for x, y in zip(p, q)])


def dim_of(text: str) -> int:
    return 3 if "R³" in text else 2


def fr(text: str) -> Fraction:
    return Fraction(text)


# -- the oracle for every single-part form ------------------------------------------------------------------------------


def oracle_easy(q):
    kind, w = metric_of(q.text)
    p, r = points_in(q.text)[:2]
    return d(kind, w, p, r)


def oracle_count(q):
    kind, w = metric_of(q.text)
    c, *pts = points_in(q.text)
    m = re.search(r"\br = (\d+(?:/\d+)?)", q.text) or re.search(r"radius (\d+(?:/\d+)?) about", q.text)
    rad = sympy.Rational(m.group(1))
    return sympy.Integer(sum(1 for p in pts if d(kind, w, p, c) < rad))


def sup_ratio(outer: str, inner: str, n: int):
    """The largest value of ||x||_outer / ||x||_inner found over the integer vectors with entries in -3..3."""
    best = sympy.Integer(0)
    for v in itertools.product(range(-3, 4), repeat=n):
        if not any(v):
            continue
        ratio = norm_value(outer, None, v) / norm_value(inner, None, v)
        if ratio > best:
            best = ratio
    return sympy.nsimplify(best)


SYMBOL = {"1": "1", "2": "2", "∞": "inf"}


def oracle_compare(q):
    text = q.text
    n = dim_of(text)
    rad = int(re.search(r"\br = (\d+)", text).group(1))
    balls = re.findall(r"B_([12∞])\(c, ([sr])\)", text)
    if "diameter" in text:
        inner = SYMBOL[balls[0][0]]
        outer = SYMBOL[re.search(r"supremum of d_([12∞])\(x, y\)", text).group(1)]
        return 2 * rad * sup_ratio(outer, inner, n)
    (a, _), (b, _) = balls[0], balls[1]
    if "largest s" in text:
        return sympy.Rational(rad) / sup_ratio(SYMBOL[b], SYMBOL[a], n)
    return sympy.Rational(rad) * sup_ratio(SYMBOL[b], SYMBOL[a], n)


def oracle_contain(q):
    kind, w = metric_of(q.text)
    c = points_in(q.text.split("S = ")[0])[0]
    block = re.search(r"S = \{(.*?)\}", q.text).group(1)
    return max(d(kind, w, p, c) for p in points_in(block))


def grid_area(inside, half: float, n: int = 2, cells: int = 1200):
    axes = [np.linspace(-half, half, cells, endpoint=False) + half / cells for _ in range(n)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return float(np.count_nonzero(inside(*mesh))) * (2 * half / cells) ** n


def numeric_measure(q):
    """The area or volume asked for, found by counting grid cells in the region read off the stem."""
    text = q.text
    n = dim_of(text)
    if "Euclidean area of A" in text:
        kind, _ = metric_of(text)
        lo, hi = (int(v) for v in re.search(r"with (\d+) ≤ d_\S+\(x, c\) < (\d+)", text).groups())
        half = hi + 1.0
        f = {
            "1": lambda x, y: abs(x) + abs(y),
            "2": lambda x, y: np.hypot(x, y),
            "inf": lambda x, y: np.maximum(abs(x), abs(y)),
        }[kind]
        return grid_area(lambda x, y: (f(x, y) >= lo) & (f(x, y) < hi), half)
    if "d_w" in text:
        kind, w = metric_of(text)
        rad = float(fr(re.search(r"when r = (\S+?)\.", text).group(1)))
        return grid_area(lambda x, y: w[0] * abs(x) + w[1] * abs(y) < rad, rad / min(w) + 0.5)
    if "region of the plane given by" in text:
        body = re.search(r"given by (.*?)\. Give", text).group(1)
        lhs, rhs = body.split(" < ")
        expr = re.sub(r"\|([^|]*)\|", r"abs(\1)", lhs).replace("²", "**2").replace("max(", "np.maximum(")
        bound = float(fr(rhs.strip()))

        def inside(x, y):
            return eval(expr, {"__builtins__": {}, "np": np, "abs": np.abs}, {"x": x, "y": y}) < bound  # noqa: S307

        return grid_area(inside, 20.0, 2, 1600)
    kind, _ = metric_of(text)
    rad = float(fr(re.search(r"when r = (\S+?)\.", text).group(1)))
    if n == 2:
        f = {
            "1": lambda x, y: abs(x) + abs(y),
            "2": lambda x, y: np.hypot(x, y),
            "inf": lambda x, y: np.maximum(abs(x), abs(y)),
        }[kind]
        return grid_area(lambda x, y: f(x, y) < rad, rad + 0.5)
    f3 = {
        "1": lambda x, y, z: abs(x) + abs(y) + abs(z),
        "2": lambda x, y, z: np.sqrt(x * x + y * y + z * z),
        "inf": lambda x, y, z: np.maximum(np.maximum(abs(x), abs(y)), abs(z)),
    }[kind]
    return grid_area(lambda x, y, z: f3(x, y, z) < rad, rad + 0.5, 3, 110)


def oracle_vertices(q):
    vs = points_in(q.text)
    cx = sum(v[0] for v in vs) / 4
    cy = sum(v[1] for v in vs) / 4
    hits = []
    box = [(cx + i / 4, cy + j / 4) for i in range(-80, 81) for j in range(-80, 81)]

    def in_hull(pt):
        # the four vertices in angular order about the centre; inside when on the inner side of every edge
        order = sorted(vs, key=lambda v: np.arctan2(v[1] - cy, v[0] - cx))
        signs = []
        for a, b in zip(order, order[1:] + order[:1]):
            signs.append((b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]))
        return all(s > 0 for s in signs) or all(s < 0 for s in signs)

    hull = {pt for pt in box if in_hull(pt)}
    for kind in ("1", "2", "inf"):
        for v in vs:
            rad = norm_value(kind, None, (v[0] - cx, v[1] - cy))
            ball = {pt for pt in box if norm_value(kind, None, (pt[0] - cx, pt[1] - cy)) < rad}
            if ball == hull:
                hits.append(rad)
    assert hits, q.text
    assert len(set(hits)) == 1
    return hits[0]


def single_oracle(q):
    t = q.text
    if t.startswith("Find the distance") or "how far apart" in t or "determine the distance" in t:
        return oracle_easy(q)
    if "open ball" in t and ("How many" in t or "how many" in t or "Count the points" in t):
        return oracle_count(q)
    if "largest s" in t or "least s" in t or "supremum of d_" in t:
        return oracle_compare(q)
    if "r0" in t or "infimum" in t:
        return oracle_contain(q)
    if "vertices" in t:
        return oracle_vertices(q)
    return None  # an area or volume: checked numerically


# -- tests -------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_key_accepted_and_distractors_rejected(tier):
    sk = skill(SKILL)
    for seed in SEEDS:
        q = tier_q(tier, seed)
        assert grade(sk, q, q.answer_text).correct, (seed, q.text, q.answer_text)
        assert len(q.distractors) >= 3, (seed, q.text)
        for dstr in q.distractors:
            assert not grade(sk, q, dstr).correct, (seed, q.text, q.answer_text, dstr)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_oracle_recomputes_every_answer(tier):
    seen = measured = 0
    for seed in SEEDS:
        q = tier_q(tier, seed)
        want = single_oracle(q)
        if want is None:
            got = float(value_of(q.answer_text))
            ref = numeric_measure(q)
            assert abs(got - ref) <= (0.03 if dim_of(q.text) == 3 else 0.012) * got + 0.02, (
                seed,
                q.text,
                q.answer_text,
                got,
                ref,
            )
            measured += 1
            continue
        got = value_of(q.answer_text)
        assert sympy.simplify(got - want) == 0, (seed, q.text, q.answer_text, want)
        seen += 1
    assert seen + measured == len(SEEDS)
    if tier == "hard":
        assert measured >= 10 and seen >= 10


def test_oracle_forms_all_appear():
    forms = {"easy": set(), "medium": set(), "hard": set()}
    for tier in forms:
        for seed in range(120):
            t = tier_q(tier, seed).text
            forms[tier].add(
                "count"
                if "how many" in t.lower() or "Count the points" in t
                else "compare"
                if ("largest s" in t or "least s" in t or "supremum" in t)
                else "contain"
                if ("r0" in t or "infimum" in t)
                else "vertices"
                if "vertices" in t
                else "measure"
                if ("area" in t or "volume" in t)
                else "distance"
            )
    assert forms["easy"] == {"distance"}
    assert forms["medium"] == {"count", "compare"}
    assert forms["hard"] == {"contain", "measure", "vertices"}


def test_open_ball_trap_is_present():
    """Many medium counts have a point exactly on the sphere, so the open-versus-closed slip is a live distractor."""
    on_sphere = counts = 0
    for seed in range(200):
        q = tier_q("medium", seed)
        if "How many" not in q.text and "how many" not in q.text and "Count" not in q.text:
            continue
        kind, w = metric_of(q.text)
        c, *pts = points_in(q.text)
        m = re.search(r"\br = (\d+(?:/\d+)?)", q.text) or re.search(r"radius (\d+(?:/\d+)?) about", q.text)
        rad = sympy.Rational(m.group(1))
        counts += 1
        on_sphere += any(d(kind, w, p, c) == rad for p in pts)
        assert all(0 <= int(x) <= len(pts) for x in q.distractors), (seed, q.text, q.distractors)
    assert counts >= 60 and on_sphere == counts, (counts, on_sphere)


def test_exam_count_has_a_boundary_point_and_possible_distractors():
    for seed in range(300):
        q = tier_q("exam", seed)
        part = q.parts[2]
        assert all(0 <= int(x) <= 5 for x in part.distractors), (seed, part.distractors)
        m = re.search(r"B_(1|2|∞)\(P, (\d+)\)", part.prompt)
        kind = {"1": "1", "2": "2", "∞": "inf"}[m.group(1)]
        P = points_in(re.search(r"P = (\([^)]*\))", q.text).group(1))[0]
        pts = points_in(re.search(r"S = \{(.*?)\}", q.text).group(1))
        assert len(pts) == 5
        assert any(d(kind, None, p, P) == int(m.group(2)) for p in pts), (seed, part.prompt, q.text)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_non_whole_keys_say_how_to_type_them(tier):
    for seed in range(150):
        q = tier_q(tier, seed)
        v = sympy.sympify(q.answer.replace("^", "**")) if isinstance(q.answer, str) else sympy.sympify(q.answer)
        if v.is_Integer:
            continue
        lowered = q.text.lower()
        assert "exact" in lowered or "fraction" in lowered, (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_variety_and_shapes(tier):
    qs = [tier_q(tier, s) for s in SEEDS]
    shapes = {prompt_shape(q) for q in qs}
    assert len(shapes) >= 3, shapes
    answers = {q.answer_text for q in qs}
    assert len(answers) >= 6, answers
    top = max(sum(1 for q in qs if q.answer_text == a) for a in answers)
    assert top <= 0.4 * len(qs), (tier, top)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_deterministic_and_no_text_defects(tier):
    for seed in (0, 1, 7, 123):
        assert tier_q(tier, seed).to_json() == tier_q(tier, seed).to_json()
    for seed in SEEDS:
        q = tier_q(tier, seed)
        assert find_text_defects(q, include_working=True) == [], (seed, q.text)
        assert "*" not in " ".join([q.text, *(p.prompt for p in q.parts)])


def test_unknown_tier_is_rejected():
    for topic in (TOPIC, EXAM):
        with pytest.raises(Exception):
            make(topic, seed=1, tier="nonsense")


def test_wired():
    sk = skill(SKILL)
    assert sk.domain == "metric-spaces-and-topology"
    assert {s.difficulty for s in sk.sources} == {"easy", "medium", "hard", "exam"}
    assert {s.topic for s in sk.sources if s.difficulty != "exam"} == {TOPIC}
    check_wired(SKILL, EXAM)


# -- the exam chain -----------------------------------------------------------------------------------------------------------


def exam_oracle(q):
    text = text_of(q)
    P = points_in(re.search(r"P = (\([^)]*\))", q.text).group(1))[0]
    Q = points_in(re.search(r"Q = (\([^)]*\))", q.text).group(1))[0]
    S = points_in(re.search(r"S = \{(.*?)\}", q.text).group(1))
    n = len(P)
    out = {}
    a = re.search(r"Find d_([12∞])\(P, Q\)", q.parts[0].prompt).group(1)
    b = re.search(r"Find d_([12∞])\(P, Q\)", q.parts[1].prompt).group(1)
    out["a"] = d(SYMBOL[a], None, P, Q)
    out["b"] = d(SYMBOL[b], None, P, Q)
    m3, r3 = re.search(r"lie in B_([12∞])\(P, (\d+)\)", q.parts[2].prompt).groups()
    out["c"] = sympy.Integer(sum(1 for s in S if d(SYMBOL[m3], None, s, P) < int(r3)))
    m4 = re.search(r"The ball B_([12∞])\(P, r\)", q.parts[3].prompt).group(1)
    r0 = max(d(SYMBOL[m4], None, s, P) for s in S)
    out["d"] = r0
    # the area or volume of the ball of radius r0, by the numeric grid
    assert ("area" if n == 2 else "volume") in q.parts[4].prompt
    out["e"] = r0
    return out, SYMBOL[m4], n


def ball_measure_numeric(kind: str, n: int, rad: float) -> float:
    if n == 2:
        f = {
            "1": lambda x, y: abs(x) + abs(y),
            "2": lambda x, y: np.hypot(x, y),
            "inf": lambda x, y: np.maximum(abs(x), abs(y)),
        }[kind]
        return grid_area(lambda x, y: f(x, y) < rad, rad + 0.5)
    f3 = {
        "1": lambda x, y, z: abs(x) + abs(y) + abs(z),
        "2": lambda x, y, z: np.sqrt(x * x + y * y + z * z),
        "inf": lambda x, y, z: np.maximum(np.maximum(abs(x), abs(y)), abs(z)),
    }[kind]
    return grid_area(lambda x, y, z: f3(x, y, z) < rad, rad + 0.5, 3, 90)


def test_exam_chain_matches_oracle():
    for seed in range(300):
        q = tier_q("exam", seed)
        assert [p.label for p in q.parts] == list("abcde")
        want, m4, n = exam_oracle(q)
        for label in "abcd":
            part = next(p for p in q.parts if p.label == label)
            assert sympy.simplify(value_of(part.answer_text) - want[label]) == 0, (
                seed,
                label,
                part.answer_text,
                want[label],
            )
        got_e = float(value_of(q.parts[4].answer_text))
        if seed < 60:
            ref = ball_measure_numeric(m4, n, float(want["e"]))
            assert abs(got_e - ref) <= (0.05 if n == 3 else 0.02) * got_e + 0.05, (seed, got_e, ref)
        # the follow-through formula is the measure formula of the ball, in r0
        r0 = want["d"]
        formula = {
            ("1", 2): 2 * r0**2,
            ("inf", 2): 4 * r0**2,
            ("2", 2): sympy.pi * r0**2,
            ("1", 3): sympy.Rational(4, 3) * r0**3,
            ("inf", 3): 8 * r0**3,
            ("2", 3): sympy.Rational(4, 3) * sympy.pi * r0**3,
        }[(m4, n)]
        assert sympy.simplify(value_of(q.parts[4].answer_text) - formula) == 0, seed


def test_exam_keys_accepted_distractors_rejected():
    sk = skill(SKILL)
    for seed in range(120):
        q = tier_q("exam", seed)
        result = grade(sk, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(p.marks for p in q.parts), (seed, q.text)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for dstr in p.distractors:
                part_result = grade(sk, q, {**canon, p.label: dstr}).parts[p.label]
                assert part_result.marks == 0, (seed, p.label, dstr, p.answer_text)


def test_exam_follow_through_from_a_wrong_containment_radius():
    """A wrong (d) earns nothing, and (e) worked out correctly from that wrong value still earns full marks."""
    sk = skill(SKILL)
    checked = 0
    for seed in range(120):
        q = tier_q("exam", seed)
        want, m4, n = exam_oracle(q)
        wrong_d = want["d"] + 3
        formula = {
            ("1", 2): 2 * wrong_d**2,
            ("inf", 2): 4 * wrong_d**2,
            ("2", 2): sympy.pi * wrong_d**2,
            ("1", 3): sympy.Rational(4, 3) * wrong_d**3,
            ("inf", 3): 8 * wrong_d**3,
            ("2", 3): sympy.Rational(4, 3) * sympy.pi * wrong_d**3,
        }[(m4, n)]
        reply = {p.label: p.answer_text for p in q.parts}
        reply["d"] = str(wrong_d).replace("**", "^")
        reply["e"] = str(sympy.simplify(formula)).replace("**", "^").replace("*", "")
        result = grade(sk, q, reply)
        assert result.parts["d"].marks == 0, (seed, reply)
        assert result.parts["e"].marks == next(p.marks for p in q.parts if p.label == "e"), (
            seed,
            reply,
            result.parts["e"],
        )
        assert result.parts["e"].follow_through is True, seed
        checked += 1
    assert checked == 120


def test_exam_answer_space_and_junk():
    answers = {tier_q("exam", s).answer_text for s in range(300)}
    assert len(answers) >= 100, len(answers)
    shapes = {prompt_shape(tier_q("exam", s)) for s in range(60)}
    assert len(shapes) >= 3
    for label in "abcde":
        values = [p.answer_text for s in range(300) for p in tier_q("exam", s).parts if p.label == label]
        assert max(values.count(v) for v in set(values)) <= 0.5 * len(values), label
    check_deterministic(EXAM)
    check_all_wrong_earns_nothing(SKILL, EXAM, {"a": "999", "b": "998", "c": "997", "d": "996", "e": "995"})
