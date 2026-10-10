"""Independent oracle for the `p15d` questions: it re-derives each answer from the rendered question text.

Parses ONLY the rendered question (`question.text`, `question.answer_text`, `question.distractors`) and
re-derives the answer from scratch, per every rule this oracle enforces on itself: it imports NOTHING from
`mathgen.generators.p15d_build` -- not a tree, not a rule table, not a formula parser, not a transition-table
helper. Every regex and every re-derivation below is written fresh, reading the rendered text the way a
learner does.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make

TIERS = ["easy", "medium", "hard", "exam"]
# 1000 seeds/tier
# qualify: 4 mcq, 1 list-of-a-classification-derived-value, 1 mcq).
SEEDS = range(1000)


# ---------------------------------------------------------------------------
# ucv1-finite-automaton-acceptance
# ---------------------------------------------------------------------------

_START_RE = re.compile(r"starting state (q\d+)")
_ACCEPT_RE = re.compile(r"accepting states ([^.]+)\.")
_TRANS_RE = re.compile(r"from (q\d+) on (\w) go to (q\d+)")
# At exam tier the word is given COMPACTLY, e.g.
# "(ab)^17bc" or "c^25ca", never spelled out -- so this now captures the
# raw rendered word (letters, parens and ^ all allowed) and expands it
# itself, the way a learner reasoning about the automaton's eventual cycle
# by period must.
_WORD_RE = re.compile(r'the word "([\w^()]*)"')
_COMPACT_RE = re.compile(r"^(?:\(([a-z]+)\)|([a-z]))\^(\d+)([a-z]*)$")


def _oracle_expand_word(word_display: str) -> str:
    m = _COMPACT_RE.match(word_display)
    if not m:
        return word_display  # plain (non-compact) word, unchanged
    block = m.group(1) or m.group(2)
    repeats = int(m.group(3))
    suffix = m.group(4)
    return block * repeats + suffix


def _oracle_automaton_verdict(text: str) -> str:
    start = _START_RE.search(text).group(1)
    accepting = set(re.findall(r"q\d+", _ACCEPT_RE.search(text).group(1)))
    transitions = {(s, sym): dst for s, sym, dst in _TRANS_RE.findall(text)}
    word = _oracle_expand_word(_WORD_RE.search(text).group(1))
    current = start
    for ch in word:
        current = transitions[(current, ch)]
    verdict = "Accepted" if current in accepting else "Rejected"
    return f"{verdict}, ends in {current}"


class TestAutomatonOracle:
    def test_oracle_matches_answer(self):
        for difficulty in TIERS:
            for seed in SEEDS:
                q = make("p15d_automaton_acceptance", seed=seed, difficulty=difficulty)
                assert _oracle_automaton_verdict(q.text) == q.answer_text


# ---------------------------------------------------------------------------
# ucv1-formal-derivation-check
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(r"Line (\d+): (.+?) -- ([\w() ]+?)(?: \(lines? ([\d, ]+)\))?(?:[.|]|$)")


def _oracle_parse_formula(s: str) -> tuple:
    s = s.strip()
    for op, key in ((" -> ", "implies"), (" & ", "and"), (" | ", "or")):
        if op in s:
            left, right = s.split(op, 1)
            return (key, _oracle_parse_formula(left), _oracle_parse_formula(right))
    return ("atom", s)


def _oracle_derivation_answer(text: str) -> str:
    body = text.split("check the following finite derivation line by line: ", 1)[1]
    body = body.split(". Identify the first line")[0]
    lines = {}
    # Split only at the boundary before the next "Line N:" -- a bare " | "
    # inside a chunk is a rendered disjunction (_fmt_formula uses " | " for
    # OR), not a line separator, so a naive `body.split(" | ")` corrupts any
    # line whose own formula contains a disjunction.
    for chunk in re.split(r"\s\|\s(?=Line \d+:)", body):
        m = _LINE_RE.match(chunk.strip())
        assert m, f"oracle could not parse line: {chunk!r}"
        num = int(m.group(1))
        formula = _oracle_parse_formula(m.group(2))
        rule = m.group(3).strip()
        refs = [int(x) for x in m.group(4).split(",")] if m.group(4) else []
        lines[num] = (formula, rule, refs)

    def valid(num):
        formula, rule, refs = lines[num]
        cited = [lines[r][0] for r in refs]
        if rule == "Premise":
            return True
        if rule == "Modus Ponens":
            a, aimpb = cited
            return aimpb[0] == "implies" and aimpb[1] == a and aimpb[2] == formula
        if rule == "Conjunction Introduction":
            a, b = cited
            return formula == ("and", a, b)
        if rule == "Conjunction Elimination (left conjunct)":
            (aandb,) = cited
            return aandb[0] == "and" and formula == aandb[1]
        if rule == "Conjunction Elimination (right conjunct)":
            (aandb,) = cited
            return aandb[0] == "and" and formula == aandb[2]
        if rule == "Disjunction Introduction":
            (a,) = cited
            return formula[0] == "or" and formula[1] == a
        raise AssertionError(f"oracle does not know rule {rule!r}")

    for num in sorted(lines):
        if not valid(num):
            return f"Line {num}"
    return "Every line is valid"


class TestFormalDerivationOracle:
    def test_oracle_matches_answer(self):
        for difficulty in TIERS:
            for seed in SEEDS:
                q = make("p15d_formal_derivation_check", seed=seed, difficulty=difficulty)
                assert _oracle_derivation_answer(q.text) == q.answer_text


# ---------------------------------------------------------------------------
# ucv1-dominated-convergence-check
# ---------------------------------------------------------------------------

_FUNC_VAL_RE = re.compile(r"f(\d+)\((x\d+)\)=(-?\d+(?:/\d+)?)")
_BOUND_VAL_RE = re.compile(r"g\((x\d+)\)=(-?\d+(?:/\d+)?)")


def _oracle_num(s: str) -> Fraction:
    return Fraction(s)


_POINTWISE_CLAUSE_RE = re.compile(r"pointwise-limit test \((.*?)\) and the domination test")


def _oracle_tail_start(text: str) -> int:
    """The 1-based index of the first tail term, read from the STATED
    convention rather than assumed. The
    prompt now always names the tail explicitly (either "stabilises from
    fK onward" at easy/medium/hard, or "constant across at least the last
    N listed values, fK, ..., fN" at exam) -- the smallest f-index named in
    that clause is the tail's own start, at every tier, so the oracle reads
    it from the text instead of assuming a run-length rule."""
    clause = _POINTWISE_CLAUSE_RE.search(text).group(1)
    indices = [int(m) for m in re.findall(r"f(\d+)", clause)]
    assert indices, f"oracle found no stated tail index in: {clause!r}"
    return min(indices)


def _oracle_dominated_convergence(text: str) -> str:
    func_part, bound_part = text.split("A stated bound g is given by:", 1)
    values: dict[str, dict[int, Fraction]] = {}
    for n, point, val in _FUNC_VAL_RE.findall(func_part):
        values.setdefault(point, {})[int(n)] = _oracle_num(val)
    bound: dict[str, Fraction] = {}
    for point, val in _BOUND_VAL_RE.findall(bound_part):
        bound[point] = _oracle_num(val)

    points = sorted(values)
    n_max = max(n for pv in values.values() for n in pv)
    tail_start = _oracle_tail_start(text)

    def stabilises(p: str) -> bool:
        return all(values[p][n] == values[p][tail_start] for n in range(tail_start, n_max + 1))

    converges = all(stabilises(p) for p in points)
    dominates = all(all(abs(values[p][n]) <= bound[p] for n in values[p]) for p in points)
    # On this finite domain every stated bound is automatically integrable
    # (a finite list of finite values is always integrable), so
    # "The bound is not integrable" is never a genuine outcome to derive.

    if not converges:
        return "Pointwise convergence fails"
    if not dominates:
        return "The stated bound is violated at a supplied point"
    return "All hypotheses are met"


class TestDominatedConvergenceOracle:
    def test_oracle_matches_answer(self):
        for difficulty in TIERS:
            for seed in SEEDS:
                q = make("p15d_dominated_convergence", seed=seed, difficulty=difficulty)
                assert _oracle_dominated_convergence(q.text) == q.answer_text


# ---------------------------------------------------------------------------
# ucv1-manifold-chart-jacobian
# ---------------------------------------------------------------------------

_UV_RE = re.compile(r"u\(x,y\) = (.+?) and v\(x,y\) = (.+?)\. At the point")
_POINT_RE = re.compile(r"\(x, y\) = \((-?\d+), (-?\d+)\)")


def _oracle_eval_poly(expr: str, x: int, y: int) -> Fraction:
    """Evaluate a caret-power, juxtaposition-multiplication polynomial in
    x, y (the exact display shape `poly_display` produces: `^` for power,
    no `*`) at (x, y), and separately its two partial derivatives, by
    parsing each additive term with a tiny hand-written tokenizer -- never
    `sympy.sympify` on the display string, and never importing the
    generator's own sympy expressions."""
    terms = re.findall(r"([+-]?\s*\d*x?\^?\d*y?\^?\d*)", expr.replace(" ", ""))
    # Simpler and robust: split on + and - (keeping sign), then parse each
    # monomial "cx^a y^b" (c, x-power, y-power each optional).
    expr_c = expr.replace(" ", "")
    parts = re.findall(r"[+-]?[^+-]+", expr_c)
    total = Fraction(0)
    dtotal_dx = Fraction(0)
    dtotal_dy = Fraction(0)
    for part in parts:
        sign = -1 if part.startswith("-") else 1
        body = part.lstrip("+-")
        m = re.match(r"^(\d*)(?:x\^?(\d*))?(?:y\^?(\d*))?$", body)
        assert m, f"oracle could not parse monomial {body!r} in {expr!r}"
        coeff_s, xp_s, yp_s = m.groups()
        coeff = int(coeff_s) if coeff_s else 1
        has_x = "x" in body
        has_y = "y" in body
        xp = int(xp_s) if xp_s else (1 if has_x else 0)
        yp = int(yp_s) if yp_s else (1 if has_y else 0)
        coeff *= sign
        total += coeff * Fraction(x) ** xp * Fraction(y) ** yp
        if xp >= 1:
            dtotal_dx += coeff * xp * Fraction(x) ** (xp - 1) * Fraction(y) ** yp
        if yp >= 1:
            dtotal_dy += coeff * Fraction(x) ** xp * yp * Fraction(y) ** (yp - 1)
    return total, dtotal_dx, dtotal_dy


def _oracle_chart_jacobian(text: str, x0: int, y0: int) -> list[int]:
    u_expr, v_expr = _UV_RE.search(text).groups()
    _, du_dx, du_dy = _oracle_eval_poly(u_expr, x0, y0)
    _, dv_dx, dv_dy = _oracle_eval_poly(v_expr, x0, y0)
    det = du_dx * dv_dy - du_dy * dv_dx
    return [int(du_dx), int(du_dy), int(dv_dx), int(dv_dy), int(det)]


class TestChartJacobianOracle:
    def test_oracle_matches_answer(self):
        for difficulty in TIERS:
            for seed in SEEDS:
                q = make("p15d_chart_jacobian", seed=seed, difficulty=difficulty)
                x0, y0 = (int(v) for v in _POINT_RE.search(q.text).groups())
                assert _oracle_chart_jacobian(q.text, x0, y0) == q.answer


# ---------------------------------------------------------------------------
# ucv1-validation-calibration-check
# ---------------------------------------------------------------------------

_PRED_RE = re.compile(r"p(\d+)=(-?\d+/\d+|-?\d+)")
_OBS_RE = re.compile(r"o(\d+)=(\d+)")
_DP_RE = re.compile(r"to (\d+) decimal places")
# The exam tier's bins are unequal-width and their
# boundaries are STATED, not computed from a bin count -- so the oracle
# now reads the stated "bin i: [lo, hi)" edges straight out of the prompt
# for every tier (equal-width tiers state them too), instead of assuming
# an even division of [0, 1] by num_bins.
_EDGE_RE = re.compile(r"bin (\d+): \[([^,]+), ([^)]+)\)")


def _oracle_round_half_up(value: Fraction, places: int) -> Fraction:
    scale = 10**places
    import math

    scaled = value * scale
    magnitude = math.floor(scaled + Fraction(1, 2))
    return Fraction(magnitude, scale)


def _oracle_stated_precision_string(value: Fraction, places: int) -> str:
    scaled = value * (10**places)
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled.numerator)).zfill(places + 1)
    whole, frac = (digits[:-places], digits[-places:]) if places else (digits, "")
    return f"{sign}{whole}.{frac}" if places else f"{sign}{whole}"


def _oracle_calibration(text: str) -> str:
    preds = {int(i): Fraction(v) for i, v in _PRED_RE.findall(text)}
    outcomes = {int(i): Fraction(v) for i, v in _OBS_RE.findall(text)}
    dp = int(_DP_RE.search(text).group(1))
    n = len(preds)

    edges = sorted((int(i), Fraction(lo), Fraction(hi)) for i, lo, hi in _EDGE_RE.findall(text))
    assert len(edges) >= 2, f"oracle found no stated bin edges in: {text!r}"

    def bin_of(p: Fraction) -> int:
        for i, lo, hi in edges:
            if lo <= p < hi:
                return i
        return edges[-1][0]  # p == 1, last bin closed at 1

    bins: dict[int, list] = {i: [] for i, _, _ in edges}
    for idx in preds:
        bins[bin_of(preds[idx])].append((preds[idx], outcomes[idx]))
    total_error = Fraction(0)
    for b in bins.values():
        if not b:
            continue
        n_b = len(b)
        avg_p = Fraction(sum(p for p, _ in b), n_b)
        avg_o = Fraction(sum(o for _, o in b), n_b)
        total_error += Fraction(n_b, n) * abs(avg_p - avg_o)
    rounded = _oracle_round_half_up(total_error, dp)

    avg_pred_all = Fraction(sum(preds.values()), n)
    avg_obs_all = Fraction(sum(outcomes.values()), n)
    direction = "over-confident" if avg_pred_all > avg_obs_all else "under-confident"
    return f"ECE = {_oracle_stated_precision_string(rounded, dp)}, {direction}"


class TestValidationCalibrationOracle:
    def test_oracle_matches_answer(self):
        for difficulty in TIERS:
            for seed in SEEDS:
                q = make("p15d_validation_calibration", seed=seed, difficulty=difficulty)
                assert _oracle_calibration(q.text) == q.answer_text


# ---------------------------------------------------------------------------
# f10-shape-sorting-algorithm
# ---------------------------------------------------------------------------

_TEST_LINE_RE = re.compile(
    r"Test (\d+): (.+?) If yes, (go to Test (\d+)|assign (Bin \w+))\. "
    r"If no, (go to Test (\d+)|assign (Bin \w+))\."
)
# The shape is no longer a bare "stated attributes:"
# boolean list -- it is now concrete side lengths, plus a right-angle
# sentence and a parallel-sides sentence only when this tier's tree asks
# about them, all in f10-polygon-classification's own declarative style.
# The oracle now reads those facts directly, the way a learner must, and
# no longer trusts any of the generator's own booleans.
_FACTS_RE = re.compile(
    r"(A (?:triangle|quadrilateral) has sides measuring [^.]+\.(?: [^.]+\.)*) "
    r"Trace the procedure"
)
_SIDES_RE = re.compile(r"has sides measuring ([\d, ]+) units")


def _oracle_shape_facts(facts_text: str) -> dict:
    sides = [int(s) for s in _SIDES_RE.search(facts_text).group(1).split(", ")]
    attrs = {
        "sides3": len(sides) == 3,
        "equal_sides": len(set(sides)) == 1,
    }
    if "One of its interior angles is a right angle" in facts_text:
        attrs["right_angle"] = True
    elif "None of its interior angles is a right angle" in facts_text:
        attrs["right_angle"] = False
    if "Both pairs of its opposite sides are parallel" in facts_text:
        attrs["has_parallel_pair"] = True
        attrs["has_second_parallel_pair"] = True
    elif "Exactly one pair of its opposite sides is parallel" in facts_text:
        attrs["has_parallel_pair"] = True
        attrs["has_second_parallel_pair"] = False
    elif "No pair of its opposite sides is parallel" in facts_text:
        attrs["has_parallel_pair"] = False
        attrs["has_second_parallel_pair"] = False
    return attrs


def _oracle_test_key(question: str) -> str:
    """Which fact a test's wording is asking about, robust to either of
    the two independently-drawn phrasings each test may use."""
    q = question.lower()
    if "3 sides" in q or "exactly 3" in q:
        return "sides3"
    if "equal in length" in q or "same length" in q:
        return "equal_sides"
    if "right angle" in q:
        return "right_angle"
    if "second" in q or "both pairs" in q:
        return "has_second_parallel_pair"
    if "pair" in q:
        return "has_parallel_pair"
    raise AssertionError(f"oracle cannot classify test question: {question!r}")


def _oracle_shape_sorting(text: str) -> str:
    facts_text = _FACTS_RE.search(text).group(1)
    attrs = _oracle_shape_facts(facts_text)
    tests = {}
    for m in _TEST_LINE_RE.finditer(text):
        num = int(m.group(1))
        question = m.group(2)
        yes_next, yes_bin = m.group(4), m.group(5)
        no_next, no_bin = m.group(7), m.group(8)
        tests[num] = (question, yes_next, yes_bin, no_next, no_bin)
    assert tests, f"oracle found no Test lines in: {text!r}"

    current = 1
    for _ in range(20):
        question, yes_next, yes_bin, no_next, no_bin = tests[current]
        taken_yes = attrs[_oracle_test_key(question)]
        if taken_yes:
            if yes_bin:
                return yes_bin
            current = int(yes_next)
        else:
            if no_bin:
                return no_bin
            current = int(no_next)
    raise AssertionError("oracle trace did not terminate")


class TestShapeSortingOracle:
    def test_oracle_matches_answer(self):
        for difficulty in TIERS:
            for seed in SEEDS:
                q = make("p15d_shape_sorting", seed=seed, difficulty=difficulty)
                assert _oracle_shape_sorting(q.text) == q.answer_text
