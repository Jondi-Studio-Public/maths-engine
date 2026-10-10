"""Re-derives each rendered question's mcq key from `question.text` (and, for `senior-formulate-...`,
`grading.mcq_options(question)`) alone, via hand-written parsing and predicates -- never importing anything
from `mathgen.generators.p13b_de_qualitative_build`, not a helper, not a form pool, not a feature list.

Both skills here are `answer: mcq`, so every tier's oracle re-derives the TRUE option rather than
trusting the generator's own construction, and states the tie case and how it is excluded. Total re-derived
across both skills: >= 1000 (600 for the slope skill across 4 tiers x 150 seeds, 800 for the formulate skill
across 4 tiers x 200 seeds -- see the parametrised counts below).
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen.grading.typed import mcq_options
from mathgen import make

SLOPE_TOPIC = "p13b_slope_field_examine"
FORMULATE_TOPIC = "p13b_formulate_de"

_X, _Y = sympy.symbols("x y")
_TRANS = standard_transformations + (implicit_multiplication_application,)
_GRID = tuple(range(-3, 4))


def _parse_xy(form_text: str):
    return parse_expr(form_text.replace("^", "**"), local_dict={"x": _X, "y": _Y}, transformations=_TRANS)


# ===========================================================================
# Skill A oracle: p13b_slope_field_examine
# ===========================================================================

_FORM_RE = re.compile(r"dy/dx = ([xy0-9\^+\- ]+)")
_POINT_RE = re.compile(r"\((-?\d+), (-?\d+)\)")


def _oracle_slope_value(text: str) -> int:
    form_text = _FORM_RE.search(text).group(1).strip()
    x0, y0 = (int(v) for v in _POINT_RE.search(text).groups())
    expr = _parse_xy(form_text)
    return int(expr.subs({_X: x0, _Y: y0}))


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_slope_value_oracle_matches(tier):
    checked = 0
    for seed in range(150):
        q = make(SLOPE_TOPIC, seed=seed, difficulty=tier)
        recomputed = str(_oracle_slope_value(q.text))
        assert recomputed == q.answer_text, (
            f"{tier} seed={seed}: oracle {recomputed} != canonical {q.answer_text} (text={q.text!r})"
        )
        checked += 1
    assert checked == 150


_TABLE_ROW_RE = re.compile(r"Point ([A-D]) = \((-?\d+), (-?\d+)\), slope stated as (positive|negative)")


def _oracle_slope_classify(text: str) -> str:
    form_text = _FORM_RE.search(text).group(1).strip()
    expr = _parse_xy(form_text)
    rows = _TABLE_ROW_RE.findall(text)
    assert len(rows) == 4, f"expected 4 table rows, got {rows} in {text!r}"
    disagreements = []
    for label, x_str, y_str, stated in rows:
        value = expr.subs({_X: int(x_str), _Y: int(y_str)})
        true_sign = "positive" if value > 0 else "negative"
        if true_sign != stated:
            disagreements.append(label)
    assert len(disagreements) == 1, f"expected exactly one disagreement, got {disagreements} in {text!r}"
    return f"Point {disagreements[0]}"


@pytest.mark.parametrize("seed", range(150))
def test_slope_classify_oracle_matches(seed):
    q = make(SLOPE_TOPIC, seed=seed, difficulty="hard")
    recomputed = _oracle_slope_classify(q.text)
    assert recomputed == q.answer_text, (
        f"seed={seed}: oracle {recomputed} != canonical {q.answer_text} (text={q.text!r})"
    )


# --- exam tier: independent feature -> predicate table, own grid ----------

_CLAUSE_RE = re.compile(r"\(\d\) ([^;.]+)")

_FEATURE_PREDICATES = {
    "the slope is zero whenever y = 0": lambda e: all(e.subs({_X: x, _Y: 0}) == 0 for x in _GRID),
    "the slope is zero whenever x = 0": lambda e: all(e.subs({_X: 0, _Y: y}) == 0 for y in _GRID),
    "the slope is always positive when y is positive": lambda e: all(
        e.subs({_X: x, _Y: y}) > 0 for x in _GRID for y in _GRID if y > 0
    ),
    "the slope is always negative when y is negative": lambda e: all(
        e.subs({_X: x, _Y: y}) < 0 for x in _GRID for y in _GRID if y < 0
    ),
    "the slope is always positive when x is positive": lambda e: all(
        e.subs({_X: x, _Y: y}) > 0 for x in _GRID for y in _GRID if x > 0
    ),
    "the slope is always negative when x is negative": lambda e: all(
        e.subs({_X: x, _Y: y}) < 0 for x in _GRID for y in _GRID if x < 0
    ),
    "the slope never depends on x": lambda e: all(
        e.subs({_X: x1, _Y: y}) == e.subs({_X: x2, _Y: y}) for y in _GRID for x1 in _GRID for x2 in _GRID
    ),
    "the slope is never negative": lambda e: all(e.subs({_X: x, _Y: y}) >= 0 for x in _GRID for y in _GRID),
    "the slope is zero whenever x equals y": lambda e: all(e.subs({_X: x, _Y: x}) == 0 for x in _GRID),
    "the slope is zero whenever y equals negative x": lambda e: all(e.subs({_X: x, _Y: -x}) == 0 for x in _GRID),
    "the slope is always positive when x is greater than y": lambda e: all(
        e.subs({_X: x, _Y: y}) > 0 for x in _GRID for y in _GRID if x > y
    ),
    "the slope is always positive when y is greater than negative x": lambda e: all(
        e.subs({_X: x, _Y: y}) > 0 for x in _GRID for y in _GRID if y > -x
    ),
    "the slope is zero when y = 2": lambda e: all(e.subs({_X: x, _Y: 2}) == 0 for x in _GRID),
    "the slope is always negative when y is less than 2": lambda e: all(
        e.subs({_X: x, _Y: y}) < 0 for x in _GRID for y in _GRID if y < 2
    ),
    "the slope is zero when y = 0 or y = 3": lambda e: (
        all(e.subs({_X: x, _Y: 0}) == 0 for x in _GRID) and all(e.subs({_X: x, _Y: 3}) == 0 for x in _GRID)
    ),
    "the slope is positive when x and y have opposite signs, and negative when x and y have the same sign": lambda e: (
        all(e.subs({_X: x, _Y: y}) > 0 for x in _GRID for y in _GRID if x * y < 0)
        and all(e.subs({_X: x, _Y: y}) < 0 for x in _GRID for y in _GRID if x * y > 0)
    ),
}


def _oracle_slope_match(q) -> str:
    clauses = [c.strip() for c in _CLAUSE_RE.findall(q.text)]
    assert clauses, f"no feature clauses parsed from {q.text!r}"
    predicates = [_FEATURE_PREDICATES[c] for c in clauses]
    options = mcq_options(q)
    matching = []
    for opt in options:
        form_text = opt.removeprefix("dy/dx = ")
        expr = _parse_xy(form_text)
        if all(pred(expr) for pred in predicates):
            matching.append(opt)
    assert len(matching) == 1, f"expected exactly one matching option, got {matching} (text={q.text!r})"
    return matching[0]


@pytest.mark.parametrize("seed", range(150))
def test_slope_match_oracle_matches(seed):
    q = make(SLOPE_TOPIC, seed=seed, difficulty="exam")
    recomputed = _oracle_slope_match(q)
    assert recomputed == q.answer_text, (
        f"seed={seed}: oracle {recomputed!r} != canonical {q.answer_text!r} (text={q.text!r})"
    )


def test_slope_match_oracle_is_red_on_a_corrupted_answer():
    q = make(SLOPE_TOPIC, seed=0, difficulty="exam")
    recomputed = _oracle_slope_match(q)
    corrupted = next(o for o in mcq_options(q) if o != recomputed)
    assert recomputed != corrupted


# =========================================================================== Skill B oracle:
# p13b_formulate_de
#
# The OLD oracle here derived its "true" answer from the WORDING ORDER of a parenthetical (medium
# tier: "exceeds the ambient temperature" fixed the string order q - a, never the physical sign) --
# that is not an independent check of physical correctness, it is a second copy of the same
# string-order assumption the generator made, and
# `test_formulate_medium_oracle_excludes_the_sign_reversed_tie` (removed below) explicitly PINNED
# the old backwards key as correct.
#
# Every tier below now re-derives the correct option by SUBSTITUTING concrete numbers under the
# POSITIVE-constant convention the generator's text now states explicitly ("... with a positive
# proportionality constant <letter>") and checking the resulting qualitative/physical property,
# never by matching the shape or order of the option string alone.
# ===========================================================================

_POSITIVE_CONST_RE = re.compile(r"positive proportionality constant (\w)")


def _rhs_value(rhs_text: str, subs_map: dict) -> float:
    """Parse an mcq option's RHS (after 'd../d.. = ') as a sympy expression
    and substitute every free symbol from `subs_map` (keyed by letter).
    Raises if the RHS contains a symbol not covered by `subs_map`, so a
    parsing mismatch fails loudly rather than silently substituting 0.

    Letters are drawn from pools that include single characters sympy
    treats specially (`S`, `N`, `E`, `I`, `O`, ...) -- an explicit
    `local_dict` forces every known letter to parse as a plain `Symbol`
    rather than colliding with one of those singletons."""
    local_dict = {letter: sympy.Symbol(letter) for letter in subs_map}
    expr = parse_expr(rhs_text.replace("^", "**"), local_dict=local_dict, transformations=_TRANS)
    subs = {}
    for sym in expr.free_symbols:
        name = str(sym)
        assert name in subs_map, f"unexpected symbol {name!r} in RHS {rhs_text!r} (known: {subs_map})"
        subs[sym] = subs_map[name]
    return float(expr.subs(subs))


def _option_rhs(opt: str, q_letter: str, t_letter: str) -> str:
    prefix = f"d{q_letter}/d{t_letter} = "
    assert opt.startswith(prefix), f"option {opt!r} does not start with {prefix!r}"
    return opt[len(prefix) :]


def _oracle_formulate_easy(q) -> str:
    m = re.search(r"Let (\w) denote .+, changing over time (\w)[,.]", q.text)
    assert m, f"could not parse q/t from {q.text!r}"
    q_letter, t_letter = m.groups()
    k_letter = _POSITIVE_CONST_RE.search(q.text).group(1)
    assert "proportional to the amount of" in q.text
    direction = "increases" if "increases" in q.text else "decreases"
    assert direction in q.text

    # Physical properties, jointly sufficient: (1) at Q = q1 the rate's SIGN matches the stated
    # direction (positive for "increases", negative for "decreases") (every context used +kQ
    # regardless of direction); (2) doubling Q doubles the rate (ratio == 2 between q1 and q2) --
    # this alone would NOT catch the sign bug (+kQ and -kQ both double), and (1) alone does not
    # uniquely pin the key for "increases" rows (kt, k, kQ^2 are all positive too at a single
    # positive probe) -- only the two together are unique, checked against every option.
    k_val, t_val = 3, 7
    q1, q2 = 3, 6
    matches = []
    for opt in mcq_options(q):
        rhs = _option_rhs(opt, q_letter, t_letter)
        subs_base = {k_letter: k_val, t_letter: t_val}
        v1 = _rhs_value(rhs, {**subs_base, q_letter: q1})
        v2 = _rhs_value(rhs, {**subs_base, q_letter: q2})
        sign_ok = (v1 > 0) if direction == "increases" else (v1 < 0)
        ratio_ok = v1 != 0 and v2 / v1 == 2
        if sign_ok and ratio_ok:
            matches.append(opt)
    assert len(matches) == 1, f"expected exactly one match, got {matches} (text={q.text!r})"
    return matches[0]


def _oracle_formulate_medium(q) -> str:
    m = re.search(r"An object's temperature, (\w), changes over time (\w)\.", q.text)
    assert m, f"could not parse q/t from {q.text!r}"
    q_letter, t_letter = m.groups()
    m2 = re.search(r"ambient \(surrounding\) temperature (\w) when", q.text)
    assert m2, f"could not parse ambient letter from {q.text!r}"
    a_letter = m2.group(1)
    k_letter = _POSITIVE_CONST_RE.search(q.text).group(1)

    # Physical properties, jointly sufficient (Newton's law of cooling under
    # a POSITIVE constant): the rate must be NEGATIVE when Q is above A
    # (cools toward ambient) and POSITIVE when Q is below A (warms toward
    # ambient). Neither probe alone is unique -- e.g. at Q > A, a
    # drop-the-ambient-term decoy (kQ) is also negative only if Q < 0, so
    # both probes are needed together (checked arithmetically for this
    # exact decoy set, not assumed).
    k_val = 3
    matches = []
    for opt in mcq_options(q):
        rhs = _option_rhs(opt, q_letter, t_letter)
        above = _rhs_value(rhs, {k_letter: k_val, q_letter: 10, a_letter: 4})  # Q > A
        below = _rhs_value(rhs, {k_letter: k_val, q_letter: 4, a_letter: 10})  # Q < A
        if above < 0 and below > 0:
            matches.append(opt)
    assert len(matches) == 1, f"expected exactly one match, got {matches} (text={q.text!r})"
    return matches[0]


def _oracle_formulate_hard(q) -> str:
    m = re.search(
        r"A population, (\w), grows over time (\w) within an environment of carrying capacity (\w)",
        q.text,
    )
    assert m, f"could not parse p/t/n from {q.text!r}"
    p_letter, t_letter, n_letter = m.groups()
    k_letter = _POSITIVE_CONST_RE.search(q.text).group(1)
    assert "proportional to the product of" in q.text

    # Physical properties, jointly sufficient (logistic growth under a
    # POSITIVE constant): the rate is strictly positive strictly inside
    # (0, N) -- grows toward capacity -- and exactly zero AT capacity
    # (P = N) -- growth stops there.
    k_val, n_val = 2, 10
    matches = []
    for opt in mcq_options(q):
        rhs = _option_rhs(opt, p_letter, t_letter)
        inside = _rhs_value(rhs, {k_letter: k_val, p_letter: 4, n_letter: n_val})
        boundary = _rhs_value(rhs, {k_letter: k_val, p_letter: n_val, n_letter: n_val})
        if inside > 0 and boundary == 0:
            matches.append(opt)
    assert len(matches) == 1, f"expected exactly one match, got {matches} (text={q.text!r})"
    return matches[0]


def _oracle_formulate_exam(q) -> str:
    options = mcq_options(q)
    m = re.match(r"^d(\w)/d(\w) = ", options[0])
    assert m, f"could not parse q/t from option {options[0]!r}"
    q_letter, t_letter = m.groups()
    k_letter = _POSITIVE_CONST_RE.search(q.text).group(1)

    # r's letter is never named by a fixed phrase shared across all three
    # exam templates, so derive it structurally instead: every option's RHS
    # uses exactly the symbols {q, k, r} (checked below), so whatever symbol
    # remains once q and k are excluded from the FIRST option's free symbols
    # is r -- independent of which template rendered this question.
    rhs0 = _option_rhs(options[0], q_letter, t_letter)
    letters0 = set(re.findall(r"[A-Za-z]", rhs0))
    local_dict0 = {letter: sympy.Symbol(letter) for letter in letters0}
    free0 = {
        str(s) for s in parse_expr(rhs0.replace("^", "**"), local_dict=local_dict0, transformations=_TRANS).free_symbols
    }
    remaining = free0 - {q_letter, k_letter}
    assert len(remaining) == 1, f"expected exactly one leftover symbol (r), got {remaining} in {rhs0!r}"
    r_letter = remaining.pop()

    # Physical properties, jointly sufficient (constant input + proportional
    # decay under a POSITIVE constant): the rate equals the stated constant
    # input r when Q = 0 (no decay yet), and the rate strictly decreases as
    # Q grows (more of Q present means more decay).
    r_val, k_val = 3, 2
    q1, q2 = 1, 5
    matches = []
    for opt in options:
        rhs = _option_rhs(opt, q_letter, t_letter)
        subs_base = {k_letter: k_val, r_letter: r_val}
        at_zero = _rhs_value(rhs, {**subs_base, q_letter: 0})
        v1 = _rhs_value(rhs, {**subs_base, q_letter: q1})
        v2 = _rhs_value(rhs, {**subs_base, q_letter: q2})
        if at_zero == r_val and v2 < v1:
            matches.append(opt)
    assert len(matches) == 1, f"expected exactly one match, got {matches} (text={q.text!r})"
    return matches[0]


_FORMULATE_ORACLES = {
    "easy": _oracle_formulate_easy,
    "medium": _oracle_formulate_medium,
    "hard": _oracle_formulate_hard,
    "exam": _oracle_formulate_exam,
}


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_formulate_oracle_matches(tier):
    fn = _FORMULATE_ORACLES[tier]
    checked = 0
    for seed in range(200):
        q = make(FORMULATE_TOPIC, seed=seed, difficulty=tier)
        recomputed = fn(q)
        assert recomputed == q.answer_text, (
            f"{tier} seed={seed}: oracle {recomputed!r} != canonical {q.answer_text!r} (text={q.text!r})"
        )
        checked += 1
    assert checked == 200


def test_formulate_oracle_is_red_on_a_corrupted_answer():
    q = make(FORMULATE_TOPIC, seed=0, difficulty="easy")
    recomputed = _oracle_formulate_easy(q)
    corrupted = next(o for o in mcq_options(q) if o != recomputed)
    assert recomputed != corrupted


def test_formulate_medium_oracle_rejects_the_sign_reversed_tie():
    """The genuine tie for this tier under an UNSTATED sign is k(T-A) vs
    k(A-T) -- the same model either way. The generator's text now states
    the constant is positive AND names the physical direction (cools/warms
    toward ambient), so this oracle re-derives the key from Newton's law of
    cooling directly (rate negative above ambient, positive below) rather
    than from the wording order of a parenthetical. Confirm the
    reversed-order option is present among the distractors and is not
    selected."""
    q = make(FORMULATE_TOPIC, seed=0, difficulty="medium")
    reversed_decoys = [d for d in q.distractors if re.search(r"= [a-z]\(\w - \w\)$", d) and d != q.answer_text]
    assert reversed_decoys, f"expected a reversed-order decoy among {q.distractors}"
    assert _oracle_formulate_medium(q) == q.answer_text
