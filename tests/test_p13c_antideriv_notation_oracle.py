"""Independent oracle for senior-use-the-notation-int-f-left-x-right
(ACMMM115, topic ``p13c_antideriv_notation``).

Parses ONLY ``question.text``/``question.answer_text`` -- never imports
anything from ``mathgen.generators.p13c_antideriv_notation_build`` (not a
helper, not a constant, not a table) -- and re-derives the answer from
scratch with ``sympy.diff``/``sympy.integrate``, a route independent of
however the generator's own construction built the stated facts.

The oracle's own route: read the STATED F(x) fact(s) straight out of the
question text, differentiate each F itself with sympy (never trusting the
question's own stated f(x)/g(x) as ground truth -- that would just be
checking the generator agrees with itself), confirm the differentiated
value matches what the question claims f(x)/g(x) to be (a genuine
consistency check on the premise), then combine per the tier's own rule
(identity for easy/medium/hard, p*F+q*G for exam) and compare the combined
antiderivative plus "C" against ``question.answer_text``.
"""

import re

import pytest
from sympy import Symbol, cos, diff, sympify

from mathgen import make

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")
TOPIC = "p13c_antideriv_notation"

x = Symbol("x")


def _to_sympy(expr_str: str):
    # The generator's PROMPT text writes "^" for powers and "·" (a real multiplication dot, never a
    # literal asterisk) between factors, per `mathgen.text_quality`'s star_operator/double_star bans
    # on `Question.text` -- undo both before handing the string to sympy, which accepts "^" (via
    # `convert_xor`, sympify's own default) but has no idea what "·" means.
    #
    # Undo that FIRST, before the "^" -> power handling below, or sympy would parse "e^(3*x)" as
    # `Symbol('e') ** (3*x)` -- confirmed directly that `_parse_expression` makes exactly this
    # mistake -- rather than `exp(3*x)`, breaking this oracle's independent re-derivation for the
    # exp family.
    expr_str = expr_str.replace("e^(", "exp(")
    return sympify(expr_str.replace("·", "*"), locals={"x": x})


# --- easy/medium/hard: single stated fact -----------------------------------

_SINGLE_FACT_PATTERNS = (
    # The hard tier's third family used to be `ln`, whose templates stated "... for x > 0" between
    # the fact and the sentence boundary (a domain qualifier `cos`/`exp`/`sin` never need) -- those
    # "for x > 0" patterns were matched FIRST and explicitly here so the generic patterns below
    # never greedily swallowed that clause into group "f". `ln` has been removed from the hard tier
    # (see the generator module docstring's history note) and replaced with `cos`, which needs no
    # such qualifier, so only the generic (non-"for x > 0") patterns below are reachable now. Left
    # as plain generic patterns; no family-specific parsing is needed for `cos`.
    re.compile(r"d/dx\((?P<F>.+?)\) = (?P<f>.+?), write down"),
    re.compile(r"derivative of F\(x\) = (?P<F>.+?) is (?P<f>.+?)\. Find"),
    re.compile(r"d/dx\[(?P<F>.+?)\] = (?P<f>.+?)\. Write"),
)


def _parse_single_fact(text: str):
    for pattern in _SINGLE_FACT_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group("F").strip(), m.group("f").strip()
    raise AssertionError(f"could not parse a single F/f fact out of {text!r}")


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_oracle_matches_stated_fact_and_answer(tier):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        F_str, f_str = _parse_single_fact(q.text)
        F = _to_sympy(F_str)
        f_stated = _to_sympy(f_str)
        # Consistency check on the premise itself: the stated fact must be a
        # real differentiation fact, independent of the generator's own
        # construction.
        f_oracle = diff(F, x)
        assert (f_oracle - f_stated).simplify() == 0, (
            f"{tier} seed={seed}: stated fact d/dx({F_str}) = {f_str} does not hold "
            f"(oracle differentiated to {f_oracle})"
        )
        # The learner's expected answer is F(x) + C -- this generator's whole
        # kernel is reading the notation BACKWARDS from the stated fact, so
        # the oracle's expected answer VALUE is simply the stated F
        # re-parsed. answer_text spells powers with "^" (never "**"), so
        # compare by value, not exact string.
        assert q.answer_text.endswith(" + C"), (
            f"{tier} seed={seed}: answer_text {q.answer_text!r} does not end in ' + C'"
        )
        answer_value = sympify(q.answer_text[: -len(" + C")].replace("^", "**"))
        assert (answer_value - F).simplify() == 0, (
            f"{tier} seed={seed}: oracle expected answer value {F}, "
            f"generator said {answer_value} (from answer_text {q.answer_text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


# --- hard tier: dedicated cos-family check ----------------------------------


def test_oracle_matches_cos_family_hard_tier():
    checked_cos_seeds = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        F_str, f_str = _parse_single_fact(q.text)
        F = _to_sympy(F_str)
        if not F.has(cos):
            continue
        checked_cos_seeds += 1
        f_stated = _to_sympy(f_str)
        f_oracle = diff(F, x)
        assert (f_oracle - f_stated).simplify() == 0, (
            f"hard seed={seed}: stated cos-family fact d/dx({F_str}) = {f_str} does not "
            f"hold (oracle differentiated to {f_oracle})"
        )
        answer_value = sympify(q.answer_text[: -len(" + C")].replace("^", "**"))
        assert (answer_value - F).simplify() == 0, (
            f"hard seed={seed}: oracle expected cos-family answer value {F}, "
            f"generator said {answer_value} (from answer_text {q.answer_text!r})"
        )
    assert checked_cos_seeds > 0, "no cos-family hard-tier seed found in range(60)"


# --- exam: two stated facts, combined linearly ------------------------------

_EXAM_PATTERN_A = re.compile(
    r"d/dx\((?P<F1>.+?)\) = (?P<f1>.+?) and d/dx\((?P<G>.+?)\) = (?P<g>.+?), find ∫\((?P<combined>.+?)\) dx"
)
_EXAM_PATTERN_B = re.compile(
    r"derivative of F\(x\) = (?P<F1>.+?) is (?P<f1>.+?), and the derivative of G\(x\) = "
    r"(?P<G>.+?) is (?P<g>.+?)\. Find ∫\((?P<combined>.+?)\) dx"
)


def _parse_exam(text: str):
    for pattern in (_EXAM_PATTERN_A, _EXAM_PATTERN_B):
        m = pattern.search(text)
        if m:
            return m.group("F1").strip(), m.group("f1").strip(), m.group("G").strip(), m.group("g").strip()
    raise AssertionError(f"could not parse the two exam facts out of {text!r}")


def test_oracle_matches_exam_tier_linearity():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        F1_str, f1_str, G_str, g_str = _parse_exam(q.text)
        F1, f1_stated, G, g_stated = (_to_sympy(s) for s in (F1_str, f1_str, G_str, g_str))

        # Both stated facts must independently hold.
        assert (diff(F1, x) - f1_stated).simplify() == 0, (
            f"exam seed={seed}: stated fact d/dx({F1_str}) = {f1_str} does not hold"
        )
        assert (diff(G, x) - g_stated).simplify() == 0, (
            f"exam seed={seed}: stated fact d/dx({G_str}) = {g_str} does not hold"
        )

        # The oracle reads the combined integrand straight out of the
        # question's own displayed text -- never from the generator's
        # internal p/q values.
        combined_match = re.search(r"[Ff]ind ∫\((?P<c>.+?)\) dx", q.text)
        assert combined_match is not None, f"exam seed={seed}: could not find the combined integrand in {q.text!r}"
        combined_str = combined_match.group("c")
        combined_expr = _to_sympy(combined_str)
        # By linearity, integrating combined_expr = p*f1 + q*g term-by-term
        # against the two stated antiderivatives F1/G gives p*F1 + q*G; since
        # f1 and g are linearly independent function families (a polynomial
        # power term vs an exp/sin term) here, sympy's own `integrate` on the
        # combined expression is an independent route to the SAME value.
        from sympy import integrate

        oracle_antiderivative = integrate(combined_expr, x)
        expected_answer_text = f"{oracle_antiderivative} + C"
        # Compare by value (the generator's own p*F1+q*G string form may
        # order/format terms differently from sympy's raw `integrate` output)
        # rather than by exact string.
        generator_value = sympify(q.answer)
        assert (oracle_antiderivative - generator_value).simplify() == 0, (
            f"exam seed={seed}: oracle integrated {combined_expr} to {oracle_antiderivative}, "
            f"generator's answer is {q.answer!r}"
        )
        assert q.answer_text.endswith(" + C"), f"exam seed={seed}: answer_text {q.answer_text!r} does not end in ' + C'"
        answer_text_value = sympify(q.answer_text[: -len(" + C")].replace("^", "**"))
        assert (answer_text_value - generator_value).simplify() == 0, (
            f"exam seed={seed}: answer_text value {answer_text_value} does not match generator's answer {q.answer!r}"
        )
        checked += 1
    assert checked == len(SEEDS)
