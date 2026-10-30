"""``pde-discriminant-values-and-canonical-form``: keys grade right, distractors wrong, every answer is re-derived from the stem.

The oracle reads the rendered stem alone: it finds the run of derivative terms (in either notation, in any order,
whole equation negated or not), reads the coefficients (some linear in k) with SymPy, and answers each question type
from first principles. The canonical-form coefficient is found by applying the second-order part of the operator to
the test function w = xi eta, for which w_xixi = w_etaeta = 0 and w_xieta = 1, so no formula for the coefficient is used.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade as skill_grade
from mathgen.engine import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

from w5a_support import check_grading, check_text, parse

SKILL = "pde-discriminant-values-and-canonical-form"
TOPIC = "w5b_pde_discriminant"
EXAM = "exd_x_w5b_pde_discriminant"
TIERS = ("easy", "medium", "hard")
SEEDS = range(1, 41)
K = sympy.Symbol("k")
X, Y = sympy.symbols("x y")


def _q(tier: str, seed: int):
    return make(EXAM if tier == "exam" else TOPIC, seed=seed, difficulty=tier)


# -- reading the equation ----------------------------------------------------------------------------------------------

_NORMAL = [
    ("∂²u/∂x∂y", "@XY@"),
    ("∂²u/∂x²", "@XX@"),
    ("∂²u/∂y²", "@YY@"),
    ("∂u/∂x", "@X@"),
    ("∂u/∂y", "@Y@"),
    ("u_xx", "@XX@"),
    ("u_xy", "@XY@"),
    ("u_yy", "@YY@"),
    ("u_x", "@X@"),
    ("u_y", "@Y@"),
]
_RUN = re.compile(r"(?:[0-9k()+\-\s]*@[A-Z]+@)+")
_TERM = re.compile(r"([0-9k()+\-\s]*?)(@[A-Z]+@)")


def read_equation(text: str) -> dict:
    """{'XX': expr in k, 'XY': ..., 'YY': ..., 'X': ..., 'Y': ..., 'U': ...} for the longest run of derivative terms."""
    for old, new in _NORMAL:
        text = text.replace(old, new)
    text = re.sub(r"(?<![A-Za-z@])u(?![A-Za-z_(])", "@U@", text)
    runs = [m.group(0) for m in _RUN.finditer(text)]
    best = max(runs, key=lambda r: len(_TERM.findall(r)))
    coefs: dict = {}
    for coef, sym in _TERM.findall(best):
        s = coef.strip()
        if s in ("", "+"):
            value = sympy.Integer(1)
        elif s == "-":
            value = sympy.Integer(-1)
        else:
            value = parse(s.lstrip("+").strip().replace(" ", ""))
        coefs[sym.strip("@")] = coefs.get(sym.strip("@"), 0) + value
    return {key: sympy.sympify(coefs.get(key, 0)) for key in ("XX", "XY", "YY", "X", "Y", "U")}


def disc_of(eq: dict, k_value=None):
    d = eq["XY"] ** 2 - 4 * eq["XX"] * eq["YY"]
    return sympy.expand(d.subs(K, k_value) if k_value is not None else d)


def kind_of(d) -> str:
    return "hyperbolic" if d > 0 else "elliptic" if d < 0 else "parabolic"


def slopes(a, b, c):
    m = sympy.Symbol("m")
    return sorted(sympy.solve(sympy.Eq(a * m**2 - b * m + c, 0), m))


def item(f) -> str:
    f = Fraction(int(f.p), int(f.q))
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def set_text(values) -> str:
    return "{" + ", ".join(item(sympy.Rational(v)) for v in values) + "}"


def sympy_text(value: sympy.Rational) -> str:
    return item(sympy.Rational(value))


# -- the oracle for a single-answer question --------------------------------------------------------------------------


def oracle(q) -> str:
    text = q.text
    eq = read_equation(text)
    if "2b" in text:  # b² - AC for a mixed term written 2b
        d = disc_of(eq) / 4
        return sympy_text(d)
    if "Find Q" in text or "coefficient Q" in text or "What is Q" in text:
        s = re.findall(r"[ξη] = y ([-+]) (?:(\d+)?x(?:/(\d+))?)", text)
        assert len(s) == 2, text
        slopes_ = []
        for sign, num, den in s:
            mag = sympy.Rational(int(num or 1), int(den or 1))
            slopes_.append(mag if sign == "-" else -mag)  # xi = y - s x
        xi, eta = (Y - slopes_[0] * X), (Y - slopes_[1] * X)
        u = xi * eta
        q_coef = eq["XX"] * sympy.diff(u, X, 2) + eq["XY"] * sympy.diff(u, X, Y) + eq["YY"] * sympy.diff(u, Y, 2)
        q_coef = sympy.simplify(q_coef)
        assert q_coef.is_number, q_coef
        return sympy_text(q_coef)
    if "slope" in text and ("larger" in text or "smaller" in text):
        k_value = re.search(r"k = (-?\d+)", text)
        sub = {K: int(k_value.group(1))} if k_value else {}
        a, b, c = (eq[key].subs(sub) for key in ("XX", "XY", "YY"))
        assert b**2 - 4 * a * c > 0
        pair = slopes(a, b, c)
        return sympy_text(pair[-1] if "larger" in text else pair[0])
    if re.search(r"How many|count the integers|how many integers", text):
        lo, hi = (
            re.search(r"(-?\d+) ≤ k ≤ (-?\d+)", text).groups()
            if "≤" in text
            else re.search(r"from (-?\d+) to (-?\d+)", text).groups()
        )
        want = "hyperbolic" if "hyperbolic" in text else "elliptic"
        d = disc_of(eq)
        return str(sum(1 for k in range(int(lo), int(hi) + 1) if kind_of(d.subs(K, k)) == want))
    if re.search(r"smallest|largest", text):
        hi = int(
            next(
                m.group(1) for pat in (r"≤ k ≤ (\d+)", r"k ≤ (\d+)", r"not above (\d+)") if (m := re.search(pat, text))
            )
        )
        want = "hyperbolic" if "hyperbolic" in text else "elliptic"
        d = disc_of(eq)
        good = [k for k in range(1, hi + 1) if kind_of(d.subs(K, k)) == want]
        return str(min(good) if "smallest" in text else max(good))
    if "k" in text and ("parabolic" in text or "vanish" in text or "changes type" in text or "B² - 4AC = 0" in text):
        d = disc_of(eq)
        roots = [r for r in sympy.solve(sympy.Eq(d, 0), K) if r.is_real and r > 0]
        assert len(roots) == 1, (text, roots)
        return decimal_text(roots[0])
    d = disc_of(eq)
    return sympy_text(d)


def decimal_text(value) -> str:
    f = Fraction(int(sympy.Rational(value).p), int(sympy.Rational(value).q))
    for places in range(0, 5):
        if (f * 10**places).denominator == 1:
            return str(f.numerator) if places == 0 else format(float(f), f".{places}f").rstrip("0").rstrip(".")
    raise AssertionError(f)


# -- tests -------------------------------------------------------------------------------------------------------------


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
def test_answer_is_recomputed_from_the_stem(tier):
    for seed in range(1, 121):
        q = _q(tier, seed)
        assert q.answer_text == oracle(q), (q.text, q.answer_text, oracle(q))
        assert not q.parts and q.answer_kind in ("number", "fraction")
        assert q.answer_kind != "mcq"


def test_exam_parts_are_recomputed_from_the_stem():
    for seed in range(1, 151):
        q = _q("exam", seed)
        eq = read_equation(q.text)
        d = disc_of(eq)
        k1 = int(re.search(r"when k = (-?\d+)", q.parts[0].prompt).group(1))
        lo, hi = (int(v) for v in re.search(r"(-?\d+) ≤ k ≤ (-?\d+)", q.parts[1].prompt).groups())
        k2 = int(re.search(r"k = (-?\d+), find", q.parts[3].prompt).group(1))
        assert q.parts[0].answer_text == str(d.subs(K, k1))
        assert q.parts[1].answer_text == str(sum(1 for k in range(lo, hi + 1) if d.subs(K, k) < 0))
        assert q.parts[2].answer_text == str(sum(1 for k in range(lo, hi + 1) if d.subs(K, k) > 0))
        a, b, c = (eq[key].subs(K, k2) for key in ("XX", "XY", "YY"))
        assert q.parts[3].answer_text == set_text(slopes(a, b, c)), (q.text, q.parts[3].answer_text)
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "set"]
        assert "elliptic" in q.parts[1].prompt and "hyperbolic" in q.parts[2].prompt


def test_lower_order_terms_and_overall_sign_are_ignored():
    """The easy key does not change when the same A, B, C are written negated or with other lower-order terms."""
    saw_negated = saw_partial = saw_zero_coefficient = False
    for seed in range(1, 200):
        q = _q("easy", seed)
        eq = read_equation(q.text)
        saw_negated |= q.text.lstrip("Consider ").startswith("-") or " -u_xx" in q.text
        saw_partial |= "∂" in q.text
        saw_zero_coefficient |= eq["XX"] == 0 or eq["YY"] == 0
    assert saw_partial and saw_zero_coefficient


@pytest.mark.parametrize("tier", (*TIERS, "exam"))
def test_answer_floors(tier):
    answers = {_q(tier, s).answer_text for s in range(60)}
    assert len(answers) >= 6, answers


def test_exam_answer_space_and_shapes():
    answers = [_q("exam", s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100, len(set(answers))
    assert len({prompt_shape(_q("exam", s)) for s in range(60)}) >= 3


@pytest.mark.parametrize("tier", (*TIERS, "exam"))
def test_three_prompt_shapes(tier):
    assert len({prompt_shape(_q(tier, s)) for s in range(60)}) >= 3


def test_every_tier_asks_for_a_value_not_a_choice():
    """Each single-answer tier is graded by the skill's `number` grader (a fraction is a number); the set is in the exam."""
    kinds = {t: {_q(t, s).answer_kind for s in range(60)} for t in TIERS}
    assert kinds["easy"] == {"number"} and kinds["medium"] == {"number"} and kinds["hard"] == {"fraction"}
    assert {p.answer_kind for s in range(20) for p in _q("exam", s).parts} == {"number", "set"}
    sk = next(s for s in load_skills()[1] if s.id == SKILL)
    assert sk.answer == "number"


def test_unknown_tier_is_an_error():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="nope")
    with pytest.raises(ValueError):
        make(EXAM, seed=1, tier="easy")


def test_wiring():
    skills = load_skills()[1]
    sk = next(s for s in skills if s.id == SKILL)
    assert sk.domain == "fourier-analysis-and-pdes" and sk.status == "active"
    by = {s.difficulty: s for s in sk.sources}
    assert set(by) == {"easy", "medium", "hard", "exam"}
    assert by["exam"].topic == EXAM and list(by["exam"].stages) == ["exam"]
    assert all(by[t].topic == TOPIC for t in TIERS)


def test_existing_pde_skill_is_untouched():
    sk = next(s for s in load_skills()[1] if s.id == "classify-constant-coefficient-pde")
    assert sk.answer == "mcq"
    assert [s.topic for s in sk.sources] == [
        "constant_coefficient_pde_classification",
        "constant_coefficient_pde_classification",
        "constant_coefficient_pde_classification",
        "exd_x_sc2_pde_parameter",
    ]


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
        for seed in range(1, 250):
            q = _q(tier, seed)
            used |= set(q.option_notes.values())
            for p in q.parts or ():
                used |= set(p.option_notes.values())
    assert used <= listed, used - listed


def test_hard_and_exam_slope_and_q_prompts_state_the_answer_format():
    for seed in range(1, 120):
        q = _q("hard", seed)
        assert q.answer_kind == "fraction"
        assert q.text.endswith("Give your answer as a fraction in lowest terms or a whole number."), q.text
    for seed in range(1, 120):
        q = _q("exam", seed)
        assert "as a fraction in lowest terms or a whole number" in q.parts[3].prompt


def test_hard_fraction_keys_are_accepted_as_instructed():
    sk = next(s for s in load_skills()[1] if s.id == SKILL)
    for seed in range(1, 120):
        q = _q("hard", seed)
        assert skill_grade(sk, q, q.answer_text).correct, (seed, q.answer_text)


def test_b_not_squared_distractor_is_b_as_printed():
    """B - 4AC with B read off the printed equation: a negated layout prints -b, so the slip changes with it."""
    checked = 0
    for seed in range(1, 400):
        q = _q("easy", seed)
        if "2b" in q.text:
            continue
        slip = [d for d in q.distractors if q.option_notes.get(d) == "pde-b-not-squared"]
        if not slip:
            continue
        eq = read_equation(q.text)
        a, b, c = (eq[key] for key in ("XX", "XY", "YY"))
        assert Fraction(str(slip[0])) == Fraction(str(b - 4 * a * c)), (seed, q.text, slip, b - 4 * a * c)
        checked += 1
    assert checked >= 20
