"""Tests the function-concept and domain-and-range-intro skills (`w5_function_concept`, `w5_domain_range_intro`).

The oracles read the relation, the rule and the domain back out of the question TEXT and re-derive the key with their
own code (a pair-by-pair function test, SymPy for equations and rules), never the generator's helpers.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

import mathgen.generators.w5_functions_intro_build  # noqa: F401  (registers the topics)
from tests import w5_support as sup

X, Y = sympy.symbols("x y", real=True)
_TR = standard_transformations + (implicit_multiplication_application, convert_xor)
CON, CON_X = "w5_function_concept", "exd_x_w5_function_concept"
DOM, DOM_X = "w5_domain_range_intro", "exd_x_w5_domain_range_intro"
SKILLS = {CON: sup.skill_of("function-concept"), DOM: sup.skill_of("domain-and-range-intro")}
EXAMS = {CON: CON_X, DOM: DOM_X}
KINDS = {CON: "mcq", DOM: "list"}


def _pairs(text: str):
    return [(int(a), int(b)) for a, b in re.findall(r"\((-?\d+), (-?\d+)\)", text)]


def _is_function(pairs) -> bool:
    seen: dict[int, int] = {}
    for a, b in pairs:
        if seen.setdefault(a, b) != b:
            return False
    return True


def _table_pairs(text: str):
    m = re.fullmatch(r"x: ([-\d, ]+)  y: ([-\d, ]+)", text)
    xs = [int(v) for v in m.group(1).split(", ")]
    ys = [int(v) for v in m.group(2).split(", ")]
    assert len(xs) == len(ys)
    return list(zip(xs, ys))


def _expr(text: str):
    text = text.replace("|", "Abs(", 1)
    if "Abs(" in text:
        # the closing bar is the next remaining "|"
        text = text.replace("|", ")", 1)
    return parse_expr(text, local_dict={"x": X, "y": Y, "Abs": sympy.Abs}, transformations=_TR)


def _y_values(equation: str, xv):
    """The real y that satisfy the equation at x = xv, found by SymPy."""
    equation = equation.replace("|y|", "Abs(y)")
    left, right = equation.split("=")
    eq = (_expr(left) - _expr(right)).subs(X, xv)
    if eq.has(sympy.zoo, sympy.nan):
        return []
    sols = sympy.solve(eq, Y) if not eq.has(sympy.Abs) else sympy.solveset(eq, Y, domain=sympy.S.Reals)
    out = set()
    for s in sols:
        s = sympy.simplify(s)
        if s.is_real:
            out.add(s)
    return out


def _equation_is_function(equation: str) -> bool:
    for xv in [sympy.Rational(n, 2) for n in range(-12, 21)]:
        if len(_y_values(equation, xv)) > 1:
            return False
    return True


# -- wiring and the generic checks ------------------------------------------------------------------------------------


@pytest.mark.parametrize("topic", list(SKILLS))
def test_wiring(topic):
    skill = SKILLS[topic]
    sup.check_wiring(skill, topic, EXAMS[topic], KINDS[topic])
    steps = [skill.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and len(set(steps)) == 4


@pytest.mark.parametrize("tier", sup.TIERS)
@pytest.mark.parametrize("topic", list(SKILLS))
def test_key_accepted_and_distractors_rejected(topic, tier):
    sup.check_single_tier(SKILLS[topic], topic, tier)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_typed_lists_are_unguessable(tier):
    assert len(sup.answer_spread(DOM, tier, 60)) >= 12, tier


@pytest.mark.parametrize("tier", sup.TIERS)
def test_mcq_keys_vary(tier):
    assert len(sup.answer_spread(CON, tier, 60)) >= 12, tier


@pytest.mark.parametrize("topic", list(SKILLS))
def test_pools_and_shapes(topic):
    sup.check_pools(topic, min_prompts=20)


@pytest.mark.parametrize("topic", list(SKILLS))
def test_options_not_ranked_and_enough(topic):
    for tier in sup.TIERS:
        for seed in range(60):
            q = sup.single(topic, tier, seed)
            if topic == CON:
                assert len(q.distractors) >= 3, (tier, seed)
                opts = [q.answer_text, *q.distractors]
                assert len(set(opts)) == len(opts) >= 4


@pytest.mark.parametrize("topic", list(SKILLS))
def test_exam_chain_grades_and_distractors(topic):
    q = sup.exam(EXAMS[topic], 0)
    marks = [p.marks for p in q.parts]
    kinds = [{p.answer_kind} for p in q.parts]
    kinds = [{"number", "fraction"} if k <= {"number", "fraction"} else k for k in kinds]
    sup.check_exam_chain_kinds(SKILLS[topic], EXAMS[topic], marks, kinds, seeds=40)


# -- function concept: independent oracles ----------------------------------------------------------------------------


def _asks_function(text: str) -> bool:
    return "NOT" not in text


def test_easy_relations():
    for seed in range(60):
        q = sup.single(CON, "easy", seed)
        if "paired with two" in q.text or "has two different outputs" in q.text:
            pairs = _pairs(q.text)
            conflicting = {x for x, _ in pairs if len({y for xx, y in pairs if xx == x}) > 1}
            assert {int(q.answer_text.removeprefix("x = "))} == conflicting, (seed, q.text)
            for d in q.distractors:
                assert int(d.removeprefix("x = ")) not in conflicting
            continue
        want = _asks_function(q.text)
        assert _is_function(_pairs(q.answer_text)) == want, (seed, q.text, q.answer_text)
        for d in q.distractors:
            assert _is_function(_pairs(d)) != want, (seed, d)


def test_medium_tables_and_equations():
    seen_tables = seen_eq = seen_rows = 0
    for seed in range(80):
        q = sup.single(CON, "medium", seed)
        want = _asks_function(q.text)
        if "extra row" in q.text or "new row" in q.text:
            m = re.search(r"table x: ([-\d, ]+)  y: ([-\d, ]+?)(?: shows|\.)", q.text)
            base = list(zip([int(v) for v in m.group(1).split(", ")], [int(v) for v in m.group(2).split(", ")]))
            assert _is_function(base)
            seen_rows += 1

            def row_pair(opt: str):
                a, b = re.fullmatch(r"x = (-?\d+), y = (-?\d+)", opt).groups()
                return int(a), int(b)

            assert not _is_function([*base, row_pair(q.answer_text)]), (seed, q.text, q.answer_text)
            for d in q.distractors:
                assert _is_function([*base, row_pair(d)]), (seed, q.text, d)
        elif "table" in q.text:
            seen_tables += 1
            assert _is_function(_table_pairs(q.answer_text)) == want, (seed, q.answer_text)
            for d in q.distractors:
                assert _is_function(_table_pairs(d)) != want, (seed, d)
        else:
            seen_eq += 1
            assert _equation_is_function(q.answer_text) == want, (seed, q.text, q.answer_text)
            for d in q.distractors:
                assert _equation_is_function(d) != want, (seed, q.text, d)
    assert seen_tables >= 5 and seen_eq >= 10 and seen_rows >= 10


def test_hard_k_and_mappings():
    seen_k = seen_map = 0
    for seed in range(80):
        q = sup.single(CON, "hard", seed)
        if "Which value of k" in q.text:
            seen_k += 1
            base = _pairs(q.text)
            for opt, breaks in [(q.answer_text, True), *[(d, False) for d in q.distractors]]:
                k = int(opt.removeprefix("k = "))
                # the relation text lists (k, y) as a placeholder: its output is the last number before ")}"
                y_k = int(re.search(r"\(k, (-?\d+)\)", q.text).group(1))
                assert _is_function([*base, (k, y_k)]) != breaks, (seed, opt)
        else:
            seen_map += 1
            a_set = re.search(r"A = \{([^}]*)\}", q.text).group(1).split(", ")
            b_set = re.search(r"B = \{([^}]*)\}", q.text).group(1).split(", ")

            def is_map(option: str) -> bool:
                arrows = [tuple(p.split("→")) for p in option.split(", ")]
                if any(i not in a_set or o not in b_set for i, o in arrows):
                    return False
                ins = [i for i, _ in arrows]
                return len(ins) == len(set(ins)) and set(ins) == set(a_set)

            assert is_map(q.answer_text), (seed, q.answer_text)
            assert not any(is_map(d) for d in q.distractors), (seed, q.distractors)
    assert seen_k >= 10 and seen_map >= 10


def test_exam_function_concept_oracle():
    for seed in range(60):
        q = sup.exam(CON_X, seed)
        pairs = _pairs(q.text)
        inputs = {a for a, _ in pairs}
        a, b, c, d = q.parts
        assert int(a.answer_text) == len(inputs)
        conflicting = [x for x in inputs if len({y for xx, y in pairs if xx == x}) > 1]
        assert b.answer_text == f"x = {conflicting[0]}" or int(b.answer_text.split("= ")[1]) in conflicting
        assert int(b.answer_text.split("= ")[1]) in conflicting
        for text in b.distractors:
            assert int(text.split("= ")[1]) not in conflicting, (seed, text)
        # the fewest pairs to delete so every input keeps one pair
        assert int(c.answer_text) == len(set(pairs)) - len(inputs)
        assert sorted(int(v) for v in d.answer_text.split(", ")) == sorted({y for _, y in pairs})


# -- domain and range: independent oracles -----------------------------------------------------------------------------


def _fx(text: str, values):
    expr = _expr(text.replace("^", "**")) if False else parse_expr(text, local_dict={"x": X}, transformations=_TR)
    return [expr.subs(X, v) for v in values]


def _listed(text: str):
    return [int(v) for v in text.split(", ")]


def _expected_dr(text: str):
    """The sorted list the question asks for, read from the text alone."""
    if m := re.match(r"A function is \{(.*)\}\. List the (domain|range) of this function", text):
        pairs = _pairs(m.group(1) and "{" + m.group(1) + "}")
        idx = 0 if m.group(2) == "domain" else 1
        return sorted({p[idx] for p in pairs})
    if m := re.match(r"A function is f\(x\) = (.+?), with domain \{([^}]*)\}\. List the range", text):
        xs = _listed(m.group(2))
        return sorted({int(v) for v in _fx(m.group(1), xs)})
    if m := re.match(
        r"A function is f\(x\) = (.+?), with domain the whole numbers from (-?\d+) to (-?\d+) inclusive", text
    ):
        xs = range(int(m.group(2)), int(m.group(3)) + 1)
        return sorted({int(v) for v in _fx(m.group(1), xs)})
    if m := re.search(r"\b[VCA] = (.+?) (?:litres|dollars|square (?:centimetres|metres))", text):
        expr = m.group(1)
        if re.search(r"A = w\(\d+ - w\)", text):
            ws = range(1, int(re.search(r"A = w\((\d+) - w\)", text).group(1)))
        else:
            ws = _listed(re.search(r"(?:Orders of|Trips of|t =) ([-\d, ]+)(?: packs| km| minutes)", text).group(1))
        var = next(v for v in "tndw" if re.search(rf"\b\d*{v}\b|\({v}|{v}\)", expr))
        sym = sympy.Symbol(var)
        e = parse_expr(expr, local_dict={var: sym}, transformations=_TR)
        return sorted({int(e.subs(sym, w)) for w in ws})
    if m := re.match(
        r"The function f\(x\) = sqrt\((.+?)\) is only defined .* from (-?\d+) to (-?\d+) that are in its domain", text
    ):
        e = parse_expr(m.group(1), local_dict={"x": X}, transformations=_TR)
        return [v for v in range(int(m.group(2)), int(m.group(3)) + 1) if e.subs(X, v) >= 0]
    if m := re.match(
        r"The function f\(x\) = 1/\((.+)\) cannot .* from (-?\d+) to (-?\d+) that are NOT in its domain", text
    ):
        e = parse_expr(m.group(1), local_dict={"x": X}, transformations=_TR)
        return [v for v in range(int(m.group(2)), int(m.group(3)) + 1) if e.subs(X, v) == 0]
    raise AssertionError(text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_domain_range_keys(tier):
    for seed in range(60):
        q = sup.single(DOM, tier, seed)
        assert _listed(q.answer_text) == _expected_dr(q.text), (seed, q.text, q.answer_text)
        for d in q.distractors:
            assert _listed(d) != _expected_dr(q.text), (seed, d)


def test_exam_domain_range_oracle():
    for seed in range(60):
        q = sup.exam(DOM_X, seed)
        m = re.match(
            r"A function is f\(x\) = (.+?), with domain the whole numbers from (-?\d+) to (-?\d+) inclusive", q.text
        )
        xs = list(range(int(m.group(2)), int(m.group(3)) + 1))
        vals = {x: int(v) for x, v in zip(xs, _fx(m.group(1), xs))}
        a, b, c, d = q.parts
        assert _listed(a.answer_text) == sorted(set(vals.values()))
        assert int(b.answer_text) == len(set(vals.values()))
        assert int(c.answer_text) == max(vals.values())
        target = int(re.search(r"f\(x\) = (-?\d+)\.$", d.prompt).group(1))
        assert _listed(d.answer_text) == sorted(x for x, v in vals.items() if v == target)
