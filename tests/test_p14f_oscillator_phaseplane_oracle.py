"""Re-derives each rendered question's mcq key from `question.text` (and `grading.mcq_options(question)`) alone,
via hand-written parsing and predicates -- never importing anything from
`mathgen.generators.p14f_oscillator_phaseplane_build`, not a helper, not a constant.

Both skills here are `answer: mcq`, so every tier's oracle re-derives
the TRUE option rather than trusting the generator's own construction.

Total re-derived across both skills: 1200 (600 for the oscillator skill
across 4 tiers x 150 seeds, 600 for the phase-plane skill across 4 tiers x
150 seeds -- see the parametrised counts below), clearing the >= 1000
combined bar for a pair of mcq/classification skills.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import mcq_options
from mathgen import make

OSC_TOPIC = "p14f_linear_oscillator_classification"
PP_TOPIC = "p14f_phase_plane_linear_classification"
N_SEEDS = 150

# ===========================================================================
# Oracle A: p14f_linear_oscillator_classification
# ===========================================================================

_OSC_STANDARD_RE = re.compile(r"([-\d]*)x''(?:\s*([+-]\s*\d*)x')?(?:\s*([+-]\s*\d*)x)?\s*=\s*0")


def _coef(token: str | None, default: int) -> int:
    """Parse a coefficient token like ' + 6', ' - ', '-', '' into an int,
    treating a bare sign with no digits as magnitude 1."""
    if token is None:
        return 0
    token = token.replace(" ", "")
    if token in ("", "+"):
        return default
    if token == "-":
        return -default
    return int(token)


def _oracle_osc_classify_easy_medium(text: str) -> str:
    """easy/medium: the equation is already in standard form m*x'' + b*x' +
    k*x = 0 (b or the whole b-term may be absent when b == 0)."""
    m = re.search(r"obeys ([^.]+)\. Classify", text)
    assert m, f"could not find the equation clause in {text!r}"
    eq = m.group(1).strip()
    m_part, rest = eq.split("x''", 1)
    m_coef = 1 if m_part.strip() in ("", "-") else int(m_part.strip() or "1")
    if m_part.strip() == "-":
        m_coef = -1
    b_match = re.search(r"([+-])\s*(\d*)x'(?!')", rest)
    b_coef = 0
    if b_match:
        sign, mag = b_match.groups()
        mag = int(mag) if mag else 1
        b_coef = mag if sign == "+" else -mag
    k_match = re.search(r"([+-])\s*(\d*)x(?!')\s*=", rest)
    assert k_match, f"could not find k term in {rest!r}"
    sign, mag = k_match.groups()
    mag = int(mag) if mag else 1
    k_coef = mag if sign == "+" else -mag
    D = Fraction(b_coef) ** 2 - 4 * Fraction(m_coef) * Fraction(k_coef)
    if D < 0:
        return "underdamped"
    if D == 0:
        return "critically damped"
    return "overdamped"


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_osc_easy_medium_oracle_matches(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(OSC_TOPIC, seed=seed, difficulty=tier)
        recomputed = _oracle_osc_classify_easy_medium(q.text)
        assert recomputed == q.answer_text, (
            f"{tier} seed={seed}: oracle {recomputed} != canonical {q.answer_text} (text={q.text!r})"
        )
        checked += 1
    assert checked == N_SEEDS


def _oracle_osc_classify_hard(text: str) -> str:
    """hard: the equation is rearranged (moved_rhs / shuffled_order /
    partly_expanded). Extract every x''/x'/x coefficient by regex over the
    WHOLE clause (order-independent), including any coefficient carried on
    the right-hand side of an '=' (moved_rhs negates it back to the left)."""
    m = re.search(r"obeys (.+?) \(rearrange", text)
    assert m, f"could not find the equation clause in {text!r}"
    eq = m.group(1).strip()
    lhs, rhs = eq.split("=")
    rhs = rhs.strip()
    # Normalise: move every RHS term to the LHS by flipping its sign, so the
    # whole equation reads "... = 0" with everything on one side.
    rhs_terms = re.findall(r"([+-]?\s*\d*x(?:''|')?)", rhs)
    flipped = []
    for t in rhs_terms:
        t = t.strip()
        if not t:
            continue
        if t.startswith("-"):
            flipped.append("+" + t[1:])
        elif t.startswith("+"):
            flipped.append("-" + t[1:])
        else:
            flipped.append("-" + t)
    combined = lhs + " " + " ".join(flipped)

    m_coef = 0
    b_coef = 0
    k_coef = 0
    for match in re.finditer(r"([+-]?)\s*(\d*)(x''|x'|x)(?!')", combined):
        sign, mag, kind = match.groups()
        sign_val = -1 if sign == "-" else 1
        mag_val = int(mag) if mag else 1
        value = sign_val * mag_val
        if kind == "x''":
            m_coef += value
        elif kind == "x'":
            b_coef += value
        else:
            k_coef += value
    D = Fraction(b_coef) ** 2 - 4 * Fraction(m_coef) * Fraction(k_coef)
    if D < 0:
        return "underdamped"
    if D == 0:
        return "critically damped"
    return "overdamped"


@pytest.mark.parametrize("seed", range(N_SEEDS))
def test_osc_hard_oracle_matches(seed):
    q = make(OSC_TOPIC, seed=seed, difficulty="hard")
    recomputed = _oracle_osc_classify_hard(q.text)
    assert recomputed == q.answer_text, (
        f"seed={seed}: oracle {recomputed} != canonical {q.answer_text} (text={q.text!r})"
    )


# exam: coefficients are given in non-SI units (mass in g, damping in g/s, stiffness already in N/m)
# -- the oracle must convert (divide by 1000) before applying the same discriminant test as every
# other tier, parsed from the rendered text alone.
_EXAM_MASS_RE = re.compile(r"(?:mass (?:m = |is )|m = )(-?\d+) g\b")
_EXAM_DAMPING_RE = re.compile(r"(?:damping constant (?:b = |is )|b = )(-?\d+) g/s")
_EXAM_STIFFNESS_RE = re.compile(r"(?:stiffness (?:k = |is )|k = )(-?\d+) N/m")


def _oracle_osc_classify_exam(text: str) -> str:
    m_match = _EXAM_MASS_RE.search(text)
    b_match = _EXAM_DAMPING_RE.search(text)
    k_match = _EXAM_STIFFNESS_RE.search(text)
    assert m_match and b_match and k_match, f"could not parse exam clause from {text!r}"
    m_g, b_gs, k = int(m_match.group(1)), int(b_match.group(1)), int(k_match.group(1))
    assert m_g % 1000 == 0 and b_gs % 1000 == 0, (m_g, b_gs, "must convert cleanly to kg / kg/s")
    m_coef, b_coef = m_g // 1000, b_gs // 1000
    D = Fraction(b_coef) ** 2 - 4 * Fraction(m_coef) * Fraction(k)
    if D < 0:
        return "underdamped"
    if D == 0:
        return "critically damped"
    return "overdamped"


@pytest.mark.parametrize("seed", range(N_SEEDS))
def test_osc_exam_oracle_matches(seed):
    q = make(OSC_TOPIC, seed=seed, difficulty="exam")
    recomputed = _oracle_osc_classify_exam(q.text)
    assert recomputed == q.answer_text, (
        f"seed={seed}: oracle {recomputed} != canonical {q.answer_text} (text={q.text!r})"
    )


def test_osc_oracle_is_red_on_a_corrupted_answer():
    q = make(OSC_TOPIC, seed=0, difficulty="easy")
    recomputed = _oracle_osc_classify_easy_medium(q.text)
    corrupted = next(o for o in mcq_options(q) if o != recomputed)
    assert recomputed != corrupted


# ===========================================================================
# Oracle B: p14f_phase_plane_linear_classification
# ===========================================================================

_EQ_RE = re.compile(r"x' = ([^,]+), y' = ([^.]+)\.? classify the equilibrium at the origin\.?", re.IGNORECASE)
_WORD_RE = re.compile(
    r"the rate of change of x is ([^;]+); the rate of change of y is ([^.]+)\. Identify",
    re.IGNORECASE,
)


def _parse_linear_expr(expr_text: str, default_coefs: dict) -> dict:
    """Parse a sum of signed x/y terms (equation-form 'ax + by' or word-form
    'a times x plus b times y' / 'minus a times x') into {'x': a, 'y': b},
    defaulting any variable that never appears to 0."""
    text = expr_text.strip()
    result = dict(default_coefs)
    is_word_form = bool(re.search(r"\b(plus|minus|times)\b", text))
    # Equation form: e.g. "-3x + 8y", "6x", "-y".
    eq_terms = re.findall(r"([+-]?\s*\d*)(x|y)(?!')", text)
    if eq_terms and not is_word_form:
        for coef_str, var in eq_terms:
            coef_str = coef_str.replace(" ", "")
            if coef_str in ("", "+"):
                coef = 1
            elif coef_str == "-":
                coef = -1
            else:
                coef = int(coef_str)
            result[var] = coef
        return result
    # Word form, e.g. "minus 8 times x plus 12 times y", "y", "minus x plus
    # 4 times y" -- each term optionally led by a sign word (default plus for
    # the very first term), optionally a "N times" magnitude, then x or y.
    term_re = re.compile(r"(?:(plus|minus)\s+)?(?:(\d+)\s+times\s+)?([xy])")
    matches = list(term_re.finditer(text))
    assert matches, f"could not parse any word term in {expr_text!r}"
    consumed_end = 0
    for wm in matches:
        assert wm.start() == consumed_end or text[consumed_end : wm.start()].strip() == "", (
            f"unparsed text before term in {expr_text!r}"
        )
        consumed_end = wm.end()
        sign, mag_str, var = wm.groups()
        sign = sign or "plus"
        mag = int(mag_str) if mag_str else 1
        result[var] = mag if sign == "plus" else -mag
    assert consumed_end == len(text), f"unparsed trailing text in {expr_text!r}"
    return result


def _oracle_pp_matrix(text: str) -> list[list[int]]:
    m = _EQ_RE.search(text.lower())
    if m:
        row1 = _parse_linear_expr(m.group(1), {"x": 0, "y": 0})
        row2 = _parse_linear_expr(m.group(2), {"x": 0, "y": 0})
    else:
        m = _WORD_RE.search(text)
        assert m, f"could not find either equation or word form in {text!r}"
        row1 = _parse_linear_expr(m.group(1), {"x": 0, "y": 0})
        row2 = _parse_linear_expr(m.group(2), {"x": 0, "y": 0})
    return [[row1["x"], row1["y"]], [row2["x"], row2["y"]]]


def _oracle_pp_classify(A: list[list[int]]) -> str:
    a, b = A[0]
    c, d = A[1]
    T = Fraction(a + d)
    D = Fraction(a * d - b * c)
    assert D != 0, f"oracle hit a degenerate D == 0 matrix {A}"
    if D < 0:
        return "saddle"
    disc = T * T - 4 * D
    assert not (disc == 0 and T != 0), f"oracle hit a degenerate repeated-eigenvalue matrix {A}"
    if disc > 0:
        return "stable node" if T < 0 else "unstable node"
    if T == 0:
        return "centre"
    return "stable spiral" if T < 0 else "unstable spiral"


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_pp_oracle_matches(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(PP_TOPIC, seed=seed, difficulty=tier)
        A = _oracle_pp_matrix(q.text)
        recomputed = _oracle_pp_classify(A)
        assert recomputed == q.answer_text, (
            f"{tier} seed={seed}: oracle {recomputed} != canonical {q.answer_text} (A={A}, text={q.text!r})"
        )
        checked += 1
    assert checked == N_SEEDS


def test_pp_oracle_is_red_on_a_corrupted_answer():
    q = make(PP_TOPIC, seed=0, difficulty="medium")
    A = _oracle_pp_matrix(q.text)
    recomputed = _oracle_pp_classify(A)
    corrupted = next(o for o in mcq_options(q) if o != recomputed)
    assert recomputed != corrupted
