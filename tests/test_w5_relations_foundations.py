"""Tests the five relations-foundation skills (`w5_equivalent_expressions`, `w5_variables_roles`,
`w5_proportional_relations`, `w5_inequality_symbols`, `w5_arithmetic_sequences`) and their exam chains.

The oracles read the expression, table, situation, inequality or sequence back out of the question TEXT and judge every
option with SymPy, Fractions or a hand-written situation table (never with the generators' helpers): a key must be
true, correct or equivalent, and every distractor must be false, wrong or inequivalent.
"""

from __future__ import annotations

import re
from fractions import Fraction as Fr

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

import mathgen.generators.w5_relations_foundations_build  # noqa: F401  (registers the topics)
from tests import w5_support as sup

X, N = sympy.symbols("x n", real=True)
_TR = standard_transformations + (implicit_multiplication_application, convert_xor)

TOPICS = {
    "w5_equivalent_expressions": ("recognise-equivalent-expressions", "exd_x_w5_equivalent_expressions"),
    "w5_variables_roles": ("independent-dependent-variables", "exd_x_w5_variables_roles"),
    "w5_proportional_relations": ("proportional-vs-non-proportional", "exd_x_w5_proportional_relations"),
    "w5_inequality_symbols": ("inequality-symbols", "exd_x_w5_inequality_symbols"),
    "w5_arithmetic_sequences": ("arithmetic-sequences", "exd_x_w5_arithmetic_sequences"),
}
SKILLS = {t: sup.skill_of(sid) for t, (sid, _) in TOPICS.items()}
EXAMS = {t: ex for t, (_, ex) in TOPICS.items()}


def _e(text: str, **local):
    text = text.replace("×", "*").replace("÷", "/").replace("−", "-")
    return parse_expr(text, local_dict={"x": X, "n": N, **local}, transformations=_TR)


def _equiv(a: str, b: str) -> bool:
    return sympy.expand(_e(a) - _e(b)) == 0


def _nums(text: str) -> list[Fr]:
    return [Fr(m) for m in re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)]


# -- wiring, keys and distractors, pools ------------------------------------------------------------------------------


@pytest.mark.parametrize("topic", list(TOPICS))
def test_wiring(topic):
    skill = SKILLS[topic]
    sup.check_wiring(skill, topic, EXAMS[topic], "mcq")
    steps = [skill.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and len(set(steps)) == 4


@pytest.mark.parametrize("tier", sup.TIERS)
@pytest.mark.parametrize("topic", list(TOPICS))
def test_key_accepted_and_distractors_rejected(topic, tier):
    sup.check_single_tier(SKILLS[topic], topic, tier)


@pytest.mark.parametrize("tier", sup.TIERS)
@pytest.mark.parametrize("topic", list(TOPICS))
def test_four_plus_options_and_varied_keys(topic, tier):
    keys = set()
    for seed in range(60):
        q = sup.single(topic, tier, seed)
        opts = [q.answer_text, *q.distractors]
        assert len(set(opts)) == len(opts) >= 5, (seed, opts)
        assert not any(o.strip().lstrip("-").isdigit() for o in opts), (seed, opts)
        keys.add(q.answer_text)
    assert len(keys) >= 12, (topic, tier, len(keys))


@pytest.mark.parametrize("topic", list(TOPICS))
def test_pools_and_shapes(topic):
    sup.check_pools(topic, min_prompts=20)


@pytest.mark.parametrize("topic", list(TOPICS))
def test_key_position_is_not_a_tell(topic):
    longest = shortest = 0
    for tier in sup.TIERS:
        for seed in range(60):
            q = sup.single(topic, tier, seed)
            lens = [len(q.answer_text)] + [len(d) for d in q.distractors]
            longest += lens[0] == max(lens) and lens.count(lens[0]) == 1
            shortest += lens[0] == min(lens) and lens.count(lens[0]) == 1
    assert longest < 0.45 * 180 and shortest < 0.45 * 180, (topic, longest, shortest)


@pytest.mark.parametrize("topic", list(TOPICS))
def test_exam_chain_grades_and_distractors(topic):
    q = sup.exam(EXAMS[topic], 0)
    marks = [p.marks for p in q.parts]
    kinds = [{p.answer_kind} for p in q.parts]
    kinds = [{"number", "fraction"} if k <= {"number", "fraction"} else k for k in kinds]
    sup.check_exam_chain_kinds(SKILLS[topic], EXAMS[topic], marks, kinds, seeds=40)


def _split(q):
    return q.answer_text, list(q.distractors)


# -- equivalent expressions -------------------------------------------------------------------------------------------


def _target(text: str) -> str:
    for pat in (
        r"Expanding (.+?) gives",
        r"Simplify (.+?)\. ",
        r"(?:same value as|same as) (.+?) for every",
        r"equivalent to (.+?)(?:\?|\. Which)",
    ):
        m = re.search(pat, text)
        if m:
            return m.group(1)
    raise AssertionError(text)


def _claim_ok(stmt: str, a: str, b: str) -> bool:
    """Is this evidence statement about a and b entirely correct?"""
    truth = _equiv(a, b)
    if stmt.startswith("Equivalent") != truth or not (
        stmt.startswith("Equivalent") or stmt.startswith("Not equivalent")
    ):
        return False
    m = re.search(r"when x = (-?\d+) they give (-?\d+) and (-?\d+)", stmt)
    if m:
        v, p, q = (int(g) for g in m.groups())
        return _e(a).subs(X, v) == p and _e(b).subs(X, v) == q and p != q
    m = re.search(r"both give (-?\d+) when x = (-?\d+)", stmt)
    if m:
        n, v = int(m.group(1)), int(m.group(2))
        return _e(a).subs(X, v) == n and _e(b).subs(X, v) == n
    m = re.search(r"both expand to (.+)$", stmt)
    if m:
        return _equiv(a, m.group(1)) and _equiv(b, m.group(1))
    return False  # "both have two terms", "look different": not evidence


def test_expressions_easy_medium():
    for tier in ("easy", "medium"):
        for seed in range(60):
            q = sup.single("w5_equivalent_expressions", tier, seed)
            target = _target(q.text)
            key, wrong = _split(q)
            if "NOT" in q.text:
                assert not _equiv(key, target), (seed, q.text, key)
                assert all(_equiv(d, target) for d in wrong), (seed, q.text, wrong)
            else:
                assert _equiv(key, target), (seed, q.text, key)
                assert not any(_equiv(d, target) for d in wrong), (seed, q.text, wrong)


def test_expressions_hard():
    for seed in range(60):
        q = sup.single("w5_equivalent_expressions", "hard", seed)
        m = re.search(r"(?:whether|Are) (.+?) and (.+?) (?:are )?equivalent", q.text)
        assert m, q.text
        a, b = m.groups()
        key, wrong = _split(q)
        assert _claim_ok(key, a, b), (seed, q.text, key)
        assert not any(_claim_ok(d, a, b) for d in wrong), (seed, q.text, wrong)


def test_expressions_exam():
    for seed in range(40):
        q = sup.exam("exd_x_w5_equivalent_expressions", seed)
        target = re.search(r"Consider the expression (.+?)\. \(a\)", q.text).group(1)
        pa, pb, pc = q.parts
        assert _equiv(pa.answer_text, target) and "(" not in pa.answer_text, (seed, pa.answer_text)
        assert not any(_equiv(d, target) for d in pa.distractors), (seed, pa.distractors)
        v = int(re.search(r"when x = (-?\d+)", pb.prompt).group(1))
        want = int(_e(target).subs(X, v))
        assert Fr(pb.answer_text) == want and all(Fr(d) != want for d in pb.distractors), (seed, pb.answer_text)
        assert not _equiv(pc.answer_text, target) and all(_equiv(d, target) for d in pc.distractors), seed


# -- independent and dependent variables --------------------------------------------------------------------------------

# story keyword -> (independent quantity, dependent quantity), written out by hand
ROLES = {
    "taxi company": ("the distance travelled", "the total fare"),
    "phone plan": ("the amount of data used", "the monthly bill"),
    "cafe sells pies": ("the number of pies sold", "the total takings"),
    "tank holds": ("the time the tap has been running", "the volume of water in the tank"),
    "car travels at a steady": ("the time spent travelling", "the distance travelled"),
    "plant is measured": ("the number of weeks", "the height of the plant"),
    "student studies": ("the length of time studied", "the test score"),
    "cyclist has to ride": ("the speed chosen", "the time the ride takes"),
    "shop tries different prices": ("the price of the toy", "the number sold in a week"),
    "farmer uses": ("the amount of fertiliser used", "the mass of the crop"),
    "runner decides": ("the distance chosen", "the time taken"),
    "scientist heats": ("the time the water has been heated", "the temperature of the water"),
    "hardware shop": ("the length of wire bought", "the cost of the wire"),
    "gym charges": ("the number of weeks of membership", "the total cost"),
    "Masses are hung": ("the mass hung from the spring", "the length of the spring"),
    "candle burns": ("the time the candle has burned", "the height of the candle"),
    "bakery decides": ("the number of cakes baked", "the mass of flour used"),
    "ball is dropped": ("the height the ball is dropped from", "the time the ball takes to fall"),
    "student chooses how many hours": ("the hours of sleep", "the reaction test score"),
}


def _roles(text: str):
    hits = [v for k, v in ROLES.items() if k in text]
    assert len(hits) == 1, text
    return hits[0]


def _cap(t: str) -> str:
    return t[0].upper() + t[1:]


def _asked(text: str) -> str:
    if re.search(r"independent variable|chosen first|chosen freely", text):
        return "indep"
    assert re.search(r"dependent variable|depends on the others|changes as a result", text), text
    return "dep"


def test_variables_single_and_pair():
    for tier in sup.TIERS:
        for seed in range(60):
            q = sup.single("w5_variables_roles", tier, seed)
            ind, dep = _roles(q.text)
            key, wrong = _split(q)
            if "Which pair" in q.text or "names the two variables" in q.text:
                good = f"Independent: {ind}; dependent: {dep}"
                assert key == good, (seed, q.text, key)
                assert all(d != good for d in wrong), (seed, wrong)
                for d in wrong:
                    m = re.fullmatch(r"Independent: (.+); dependent: (.+)", d)
                    assert m and (m.group(1), m.group(2)) != (ind, dep), (seed, d)
                continue
            want = _cap(ind if _asked(q.text) == "indep" else dep)
            assert key == want, (seed, q.text, key)
            assert want not in wrong, (seed, wrong)


def test_variables_exam():
    for seed in range(40):
        q = sup.exam("exd_x_w5_variables_roles", seed)
        ind, dep = _roles(q.text)
        pa, pb, pc = q.parts
        assert pa.answer_text == _cap(ind) and _cap(ind) not in pa.distractors
        assert pb.answer_text == _cap(dep) and _cap(dep) not in pb.distractors
        story = q.text.split(" (a)")[0]
        slots = [int(n) for n in re.findall(r"\d+", story)]
        f, r = (slots[0], slots[1]) if len(slots) >= 2 else (0, slots[0])
        v = int(re.findall(r"\d+", pc.prompt)[-1])
        want = f + r * v
        assert Fr(pc.answer_text) == want and all(Fr(d) != want for d in pc.distractors), (seed, story, pc.prompt)


# -- proportional relations -------------------------------------------------------------------------------------------


def _table(text: str):
    m = re.search(r"x: ([-\d, ]+?)\s+y: ([-\d, ]+)", text)
    xs = [int(t) for t in m.group(1).split(",")]
    ys = [int(t) for t in m.group(2).split(",")]
    return list(zip(xs, ys))


def _prop(pairs) -> bool:
    return len({Fr(y, x) for x, y in pairs}) == 1


_FIXED_WORDS = ("fee", "to start", "call-out", "holds", "tall now")


def _situation_prop(text: str) -> bool:
    return not any(w in text for w in _FIXED_WORDS)


def _eq_prop(eq: str) -> bool:
    rhs = _e(eq.split("=")[1])
    return sympy.simplify(rhs / X).free_symbols == set() and rhs.subs(X, 0) == 0


def _line_claim_ok(stmt: str, m: Fr, c: Fr) -> bool:
    mm = re.fullmatch(
        r"(Yes|No): the line is y = (\d+)x(?: ([+-]) (\d+))?, which (?:passes through|does not pass through) the origin",
        stmt,
    )
    if not mm:
        return False  # the reasoning-only distractors
    verdict, gm, sign, gc = mm.groups()
    got_c = 0 if sign is None else (int(gc) if sign == "+" else -int(gc))
    passes = "does not" not in stmt
    return (verdict == "Yes") == (c == 0) and int(gm) == m and got_c == c and passes == (c == 0)


def test_proportional_singles():
    for tier in sup.TIERS:
        for seed in range(60):
            q = sup.single("w5_proportional_relations", tier, seed)
            key, wrong = _split(q)
            t = q.text
            if "straight line" in t or "linear relationship" in t:
                (x1, y1), (x2, y2) = [(int(a), int(b)) for a, b in re.findall(r"\((-?\d+), (-?\d+)\)", t)]
                m = Fr(y2 - y1, x2 - x1)
                c = y1 - m * x1
                assert _line_claim_ok(key, m, c), (seed, t, key)
                assert not any(_line_claim_ok(d, m, c) for d in wrong), (seed, t, wrong)
            elif "equation" in t:
                assert _eq_prop(key), (seed, t, key)
                assert not any(_eq_prop(d) for d in wrong), (seed, t, wrong)
            elif "situation" in t:
                assert _situation_prop(key), (seed, t, key)
                assert not any(_situation_prop(d) for d in wrong), (seed, t, wrong)
            else:
                negated = "NOT" in t or "does not" in t
                assert _prop(_table(key)) != negated, (seed, t, key)
                assert all(_prop(_table(d)) == negated for d in wrong), (seed, t, wrong)


def test_proportional_exam():
    for seed in range(40):
        q = sup.exam("exd_x_w5_proportional_relations", seed)
        pairs = _table(q.text)
        rule = re.search(r"rule y = (.+?)\.", q.text).group(1)
        pa, pb, pc = q.parts
        assert pa.answer_text.startswith("Yes") == _prop(pairs)
        ratios = [Fr(y, x) for x, y in pairs]
        for stmt in [pa.answer_text, *pa.distractors]:
            good = True
            m = re.fullmatch(r"(Yes|No): y ÷ x is (\S+) for every pair", stmt)
            if m:
                good = (m.group(1) == "Yes") == _prop(pairs) and _prop(pairs) and Fr(m.group(2)) == ratios[0]
            else:
                m = re.fullmatch(r"No: y ÷ x is (\S+) for the first pair and (\S+) for the second", stmt)
                if m:
                    good = (not _prop(pairs)) and (Fr(m.group(1)), Fr(m.group(2))) == (ratios[0], ratios[1])
                else:
                    good = False
            assert good == (stmt == pa.answer_text), (seed, stmt, pairs)
        v = int(re.search(r"when x = (-?\d+)", pb.prompt).group(1))
        want = int(_e(rule).subs(X, v))
        assert Fr(pb.answer_text) == want and all(Fr(d) != want for d in pb.distractors), (seed, rule)
        mult = _e(rule).coeff(X)
        assert Fr(pc.answer_text) == Fr(int(mult)) and all(Fr(d) != mult for d in pc.distractors), (seed, rule)


# -- inequality symbols -----------------------------------------------------------------------------------------------

_REL = {"<": lambda a, b: a < b, ">": lambda a, b: a > b, "≤": lambda a, b: a <= b, "≥": lambda a, b: a >= b}
_SPLIT = re.compile(r"\s*([<>≤≥])\s*")


def _holds(stmt: str, **env) -> bool:
    left, sym, right = _SPLIT.split(stmt, maxsplit=1)
    syms = {k: sympy.Symbol(k) for k in env}
    val = lambda t: _e(t, **syms).subs({syms[k]: sympy.Rational(v) for k, v in env.items()})  # noqa: E731
    return bool(_REL[sym](val(left), val(right)))


def _env_of(text: str) -> dict:
    return {k: Fr(v) for k, v in re.findall(r"\b([abx]) = (-?\d+(?:\.\d+)?)", text)}


_LE = ("no more than", "at most", "maximum", "or less", "up to and including")
_GE = ("at least", "minimum", "or older", "or more", "no less")
_GT = ("more than", "exceed", "above", "over ")
_LT = ("fewer than", "under", "less than", "below")


def _word_sym(text: str) -> str:
    for syms, words in (("≤", _LE), ("≥", _GE), (">", _GT), ("<", _LT)):
        if any(w in text for w in words):
            return syms
    raise AssertionError(text)


def _word_key(text: str) -> str:
    var = re.search(r"using ([A-Za-z])\b|Let ([A-Za-z]) be", text)
    var = var.group(1) or var.group(2)
    n = re.search(r"\b(\d+)\b", text).group(1)
    return f"{var} {_word_sym(text)} {n}"


def test_inequality_singles():
    for tier in sup.TIERS:
        for seed in range(60):
            q = sup.single("w5_inequality_symbols", tier, seed)
            key, wrong = _split(q)
            t = q.text
            if "inequality" in t and ("Let " in t or "using " in t):
                assert key == _word_key(t), (seed, t, key)
                assert all(d != key for d in wrong)
                continue
            if "in order from" in t or "puts them in order" in t:
                items = [
                    Fr(s)
                    for s in re.findall(
                        r"-?\d+(?:\.\d+)?",
                        t.split("Which")[0] if "Which list puts" in t else t.split("shows")[1].split("in order")[0],
                    )
                ]
                ascending = "least to greatest" in t
                want = sorted(items, reverse=not ascending)
                parse = lambda s: [Fr(v) for v in s.split(", ")]  # noqa: E731
                assert parse(key) == want, (seed, t, key)
                assert all(parse(d) != want and sorted(parse(d)) == sorted(items) for d in wrong), (seed, t, wrong)
                continue
            m = re.search(r"(-?\d+) ([<≤]) x ([<≤]) (-?\d+)", t)
            if m:
                lo, s1, s2, hi = m.groups()
                ok = lambda opt: (
                    _holds(
                        f"{lo} {s1} {opt.replace('x = ', '')} {s2} {hi}".replace(
                            f"{s1} {opt.replace('x = ', '')} {s2}", f"{s1} X {s2}"
                        )
                    )
                    if False
                    # noqa: E731
                    else (_REL[s1](int(lo), int(opt.split("= ")[1])) and _REL[s2](int(opt.split("= ")[1]), int(hi)))
                )
                assert ok(key), (seed, t, key)
                assert not any(ok(d) for d in wrong), (seed, t, wrong)
                continue
            env = _env_of(t)
            assert _holds(key, **env), (seed, t, key)
            assert not any(_holds(d, **env) for d in wrong), (seed, t, wrong)


def test_inequality_exam():
    for seed in range(40):
        q = sup.exam("exd_x_w5_inequality_symbols", seed)
        head = q.text.split(" Let ")[0]
        pa, pb, pc = q.parts
        assert pa.answer_text == _word_key(q.text), (seed, q.text, pa.answer_text)
        sym = _word_sym(head)
        n = int(re.search(r"\b(\d+)\b", head).group(1))
        want = {"<": n - 1, "≤": n, ">": n + 1, "≥": n}[sym]
        assert Fr(pb.answer_text) == want and all(Fr(d) != want for d in pb.distractors), (seed, head, pb.answer_text)
        env = _env_of(pc.prompt)
        var = next(iter(re.findall(r"^([A-Za-z]) =", pc.prompt)))
        env = {var: Fr(re.search(r"= (-?\d+)", pc.prompt).group(1))}
        assert _holds(pc.answer_text, **env), (seed, pc.prompt, pc.answer_text)
        assert not any(_holds(d, **env) for d in pc.distractors), (seed, pc.prompt, pc.distractors)


# -- arithmetic sequences ---------------------------------------------------------------------------------------------


def _is_ap(terms) -> bool:
    return len(terms) >= 3 and len({b - a for a, b in zip(terms, terms[1:])}) == 1


def _is_gp(terms) -> bool:
    return all(t != 0 for t in terms) and len({b / a for a, b in zip(terms, terms[1:])}) == 1 and not _is_ap(terms)


def _terms(text: str) -> list[Fr]:
    return [Fr(s) for s in re.findall(r"-?\d+(?:/\d+)?", text)]


def _seq_claim_ok(stmt: str, terms) -> bool:
    diffs = [b - a for a, b in zip(terms, terms[1:])]
    m = re.fullmatch(r"Arithmetic: each term is (\d+) (more|less) than the term before", stmt)
    if m:
        d = int(m.group(1)) * (1 if m.group(2) == "more" else -1)
        return _is_ap(terms) and diffs[0] == d
    m = re.fullmatch(r"Not arithmetic: the differences are (.+)", stmt)
    if m:
        return not _is_ap(terms) and [Fr(s) for s in m.group(1).split(", ")] == diffs
    if stmt == "Not arithmetic: each term is multiplied by the same number":
        return _is_gp(terms)
    return False


def _rule_is_ap(rule: str) -> bool:
    vals = [_e(rule).subs(N, k) for k in range(1, 8)]
    return len({b - a for a, b in zip(vals, vals[1:])}) == 1


def test_sequence_singles():
    for tier in sup.TIERS:
        for seed in range(60):
            q = sup.single("w5_arithmetic_sequences", tier, seed)
            key, wrong = _split(q)
            t = q.text
            if "rule" in t:
                assert _rule_is_ap(key), (seed, t, key)
                assert not any(_rule_is_ap(d) for d in wrong), (seed, t, wrong)
            elif "fills the gaps" in t or "Which pair of numbers fills" in t:
                head = t.split("sequence", 1)[1]
                a, b = _terms(head)[0], _terms(head)[-1]
                ok = lambda s: _is_ap([a, *[Fr(v) for v in s.split(" and ")], b])  # noqa: E731
                assert ok(key), (seed, t, key)
                assert not any(ok(d) for d in wrong), (seed, t, wrong)
            elif re.search(r"\bk\b", t):
                a, b = _terms(t.replace("k", ""))[:1][0], _terms(t.replace("k", ""))[-1]
                want = (a + b) / 2
                assert Fr(key.split("= ")[1]) == want, (seed, t, key)
                assert all(Fr(d.split("= ")[1]) != want for d in wrong), (seed, t, wrong)
            elif "next term" in t or "fourth term" in t:
                given = _terms(re.split(r"are |begins ", t)[1].split(".")[0].split(" Which")[0])
                assert _is_ap(given + [given[-1] + (given[-1] - given[-2])])
                want = given[-1] + (given[-1] - given[-2])
                assert Fr(key.split("is ")[1]) == want, (seed, t, key)
                assert all(Fr(d.split("is ")[1]) != want for d in wrong), (seed, t, wrong)
            elif t.startswith(("Which statement about", "The sequence")):
                terms = _terms(t)
                assert _seq_claim_ok(key, terms), (seed, t, key)
                assert not any(_seq_claim_ok(d, terms) for d in wrong), (seed, t, wrong)
            else:  # options are sequences: only the arithmetic one is the key
                assert _is_ap(_terms(key)), (seed, t, key)
                assert not any(_is_ap(_terms(d)) for d in wrong), (seed, t, wrong)


def test_sequence_exam():
    for seed in range(40):
        q = sup.exam("exd_x_w5_arithmetic_sequences", seed)
        terms = _terms(q.text.split("are", 1)[1])
        pa, pb, pc = q.parts
        assert _is_ap(terms)
        assert _seq_claim_ok(pa.answer_text, terms) and not any(_seq_claim_ok(d, terms) for d in pa.distractors), seed
        d = terms[1] - terms[0]
        for part, k in ((pb, 5), (pc, 8)):
            want = terms[0] + d * (k - 1)
            assert Fr(part.answer_text) == want and all(Fr(x) != want for x in part.distractors), (seed, part.label)
