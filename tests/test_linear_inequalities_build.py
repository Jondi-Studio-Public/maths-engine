"""Linear inequalities: the four skills `one-step-inequalities`, `inequalities-negative-mult-div`,
`two-step-inequalities` and `inequality-word-problems`, their practice topics and their exam chains.

The oracle re-derives every key from the RENDERED TEXT alone: it parses the inequality (or the word problem) out of
the question, solves it with `sympy.solve_univariate_inequality`, reads the key's own text into an interval with a
small parser of its own, and compares the two sets. It never imports the generator module.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy
from sympy import Interval, S, Symbol, parse_expr, solve_univariate_inequality

from mathgen import make
from mathgen.engine import grade as skill_grade
from mathgen.grading import GRADERS, grade
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects

SEEDS = range(60)
TIERS = ("easy", "medium", "hard")
SKILLS = {
    "one-step-inequalities": ("lin_ineq_one_step", "exd_x_ineq_one_step"),
    "inequalities-negative-mult-div": ("lin_ineq_negative", "exd_x_ineq_negative"),
    "two-step-inequalities": ("lin_ineq_two_step", "exd_x_ineq_two_step"),
    "inequality-word-problems": ("lin_ineq_word", "exd_x_ineq_word"),
}
PRACTICE = [(t, tier) for t in (v[0] for v in SKILLS.values()) for tier in TIERS]
EXAMS = [v[1] for v in SKILLS.values()]
_SKILLS = {s.id: s for s in load_skills()[1]}


# -- the oracle ------------------------------------------------------------------------------------------------------


def _implicit(s: str) -> str:
    s = s.replace("^", "**")
    return re.sub(r"(?<=[0-9a-z\)])(?=[a-z(])", "*", s)


def _expr(s: str, var: str):
    return parse_expr(_implicit(s.strip()), local_dict={var: Symbol(var)})


_OP = {"<": "<", "≤": "<=", ">": ">", "≥": ">="}


def _rel(lhs, op, rhs):
    return {"<": sympy.Lt, "≤": sympy.Le, ">": sympy.Gt, "≥": sympy.Ge}[op](lhs, rhs)


def _solve_text(expr_text: str, var: str):
    """The solution set of an inequality (or double inequality) printed in the question."""
    pieces = re.split(r"(≤|≥|<|>)", expr_text)
    exprs, ops = pieces[0::2], pieces[1::2]
    x = Symbol(var)
    sides = [_expr(e, var) for e in exprs]
    out = S.Reals
    for i, op in enumerate(ops):
        out = out.intersect(solve_univariate_inequality(_rel(sides[i], op, sides[i + 1]), x, relational=False))
    return out


def _stem_expression(text: str):
    """(var, inequality text) from a non-word question."""
    m = re.search(r"for which (.+?)\. Give", text) or re.search(
        r"(?:Solve the double inequality|Solve the inequality|Solve|Consider the double inequality|Consider the inequality) (.+?)\.(?: Give| \(a\)|$)",
        text,
    )
    expr = m.group(1)
    var = re.search(r"[a-z]", re.sub(r"[^a-z]", "", expr)).group(0)
    return var, expr


def _key_set(answer_text: str):
    """The key's own text as a set, read by a parser that shares nothing with the grader."""
    m = re.fullmatch(r"([a-z]) (<|≤|>|≥) (-?\d+(?:[/.]\d+)?)", answer_text)
    if m:
        b = sympy.Rational(m.group(3))
        return {
            "<": Interval.open(-S.Infinity, b),
            "≤": Interval(-S.Infinity, b),
            ">": Interval.open(b, S.Infinity),
            "≥": Interval(b, S.Infinity),
        }[m.group(2)]
    m = re.fullmatch(r"(-?\d+(?:[/.]\d+)?) (<|≤) ([a-z]) (<|≤) (-?\d+(?:[/.]\d+)?)", answer_text)
    assert m, answer_text
    return Interval(sympy.Rational(m.group(1)), sympy.Rational(m.group(5)), m.group(2) == "<", m.group(4) == "<")


_LIMITS = {
    **{p: "≤" for p in ("no more than", "at most", "not more than", "no greater than")},
    **{p: "<" for p in ("less than", "under", "below")},
    **{p: "≥" for p in ("at least", "no less than", "not less than", "a minimum of")},
    **{p: ">" for p in ("more than", "over", "greater than")},
}
# (regex, groups -> (slope, constant, limit phrase, T))
_FAMILIES = [
    (
        r"Each notebook costs \$(\d+)\. \w+ wants the total cost of the notebooks to be (.+?) \$(\d+)\.",
        lambda g: (g[0], 0, g[1], g[2]),
    ),
    (r"hold (\d+) L each\. .*? to be (.+?) (\d+) L\.", lambda g: (g[0], 0, g[1], g[2])),
    (r"cost \$(\d+) each\. The school wants the ticket sales to be (.+?) \$(\d+)\.", lambda g: (g[0], 0, g[1], g[2])),
    (r"carry a total of (.+?) (\d+) kg\. Each parcel has a mass of (\d+) kg", lambda g: (g[2], 0, g[0], g[1])),
    (
        r"costs \$(\d+) a month plus \$(\d+) for each gigabyte \(GB\) of data used\. \w+ wants the monthly bill to be (.+?) \$(\d+)\.",
        lambda g: (g[1], g[0], g[2], g[3]),
    ),
    (
        r"flag fall of \$(\d+) plus \$(\d+) per kilometre\. \w+ wants the fare to be (.+?) \$(\d+)\.",
        lambda g: (g[1], g[0], g[2], g[3]),
    ),
    (
        r"mass of (\d+) kg when empty and carries crates of (\d+) kg each\. .*? must be (.+?) (\d+) kg\.",
        lambda g: (g[1], g[0], g[2], g[3]),
    ),
    (
        r"saved \$(\d+) and adds \$(\d+) each week\. \w+ wants the total saved to be (.+?) \$(\d+)\.",
        lambda g: (g[1], g[0], g[2], g[3]),
    ),
    (
        r"tickets for \$(\d+) each and has \$(\d+) of costs to cover\. .*? to be (.+?) \$(\d+)\.",
        lambda g: (g[0], "-" + g[1], g[2], g[3]),
    ),
]


def _word_set(text: str):
    """(var, solution set) for a word problem, from its numbers and wording alone."""
    var = re.search(r"Let ([a-z]) be", text).group(1)
    x = Symbol(var)
    for pattern, pick in _FAMILIES:
        m = re.search(pattern, text)
        if m:
            slope, const, limit, total = pick(m.groups())
            return var, solve_univariate_inequality(
                _rel(int(slope) * x + int(const), _LIMITS[limit], int(total)), x, relational=False
            )
    m = re.search(
        r"Plan A costs \$(\d+) a month plus \$(\d+) per gigabyte \(GB\) of data\. Plan B costs \$(\d+) a month plus \$(\d+) per GB\. .*? when Plan ([AB]) costs less",
        text,
    )
    if m:
        fa, ra, fb, rb = (int(v) for v in m.groups()[:4])
        mine, other = (fa + ra * x, fb + rb * x) if m.group(5) == "A" else (fb + rb * x, fa + ra * x)
        return var, solve_univariate_inequality(mine < other, x, relational=False)
    m = re.search(r"scored (.+?) on \w+ tests\..*?mean of at least (\d+) over (\d+) tests", text)
    if m:
        scores = [int(v) for v in re.findall(r"\d+", m.group(1))]
        return var, solve_univariate_inequality(
            (sum(scores) + x) / int(m.group(3)) >= int(m.group(2)), x, relational=False
        )
    m = re.search(
        r"flag fall of \$(\d+) plus \$(\d+) per kilometre\. .*?was (more than|at least) \$(\d+) but (less than|no more than) \$(\d+)\.",
        text,
    )
    assert m, text
    f, r = int(m.group(1)), int(m.group(2))
    o1 = ">" if m.group(3) == "more than" else "≥"
    o2 = "<" if m.group(5) == "less than" else "≤"
    lo = solve_univariate_inequality(_rel(f + r * x, o1, int(m.group(4))), x, relational=False)
    hi = solve_univariate_inequality(_rel(f + r * x, o2, int(m.group(6))), x, relational=False)
    return var, lo.intersect(hi)


def _expected_set(q):
    if re.search(r"Let [a-z] be", q.stem or q.text):
        return _word_set(q.stem or q.text)
    var, expr = _stem_expression(q.stem or q.text)
    return var, _solve_text(expr, var)


# -- single-answer practice ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("topic,tier", PRACTICE)
def test_the_key_is_the_solution_set_of_the_printed_inequality(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        var, expected = _expected_set(q)
        assert _key_set(q.answer_text) == expected, f"{topic}/{tier} seed={seed}: {q.text!r} -> {q.answer_text!r}"
        assert q.answer == q.answer_text
        assert re.search(rf"\b{var}\b", q.answer_text)
        assert f"inequality in {var}" in q.text
        if topic == "lin_ineq_word":
            assert (
                "as a decimal" in q.text
                and "/" not in q.answer_text
                and len(q.answer_text.split(".")[-1].split(" ")[0]) <= 2
            )


@pytest.mark.parametrize("topic,tier", PRACTICE)
def test_distractors_are_real_slips_the_grader_refuses(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        var, expected = _expected_set(q)
        assert len(q.distractors) >= 2
        assert len(set(q.distractors)) == len(q.distractors) and q.answer_text not in q.distractors
        assert grade("inequality", q, q.answer_text).correct
        for d in q.distractors:
            assert not grade("inequality", q, d).correct, f"{topic}/{tier} seed={seed}: {d!r} accepted"
            assert _key_set(d) != expected, f"{topic}/{tier} seed={seed}: distractor {d!r} has the key's set"
            assert q.option_notes[d] in CATALOGUE
            assert q.distractor_values[d] == d


@pytest.mark.parametrize("topic,tier", PRACTICE)
def test_the_key_is_accepted_in_other_spellings(topic, tier):
    for seed in range(15):
        q = make(topic, seed=seed, difficulty=tier)
        m = re.fullmatch(r"([a-z]) (<|≤|>|≥) (-?\d+(?:[/.]\d+)?)", q.answer_text)
        if m:
            var, op, bound = m.groups()
            swapped = {"<": ">", "≤": "≥", ">": "<", "≥": "≤"}[op]
            assert grade("inequality", q, f"{bound} {swapped} {var}").correct
            assert grade("inequality", q, q.answer_text.replace("≤", "<=").replace("≥", ">=")).correct
        else:
            lo, op1, var, op2, hi = re.fullmatch(r"(\S+) (<|≤) ([a-z]) (<|≤) (\S+)", q.answer_text).groups()
            bracket = ("(" if op1 == "<" else "[") + f"{lo}, {hi}" + (")" if op2 == "<" else "]")
            assert grade("inequality", q, bracket).correct


@pytest.mark.parametrize("topic,tier", PRACTICE)
def test_deterministic_and_clean(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.to_json() == make(topic, seed=seed, difficulty=tier).to_json()
        assert find_text_defects(q) == [], (topic, tier, seed, find_text_defects(q))
        assert "**" not in q.text and "sqrt" not in q.text


@pytest.mark.parametrize("topic,tier", PRACTICE)
def test_variety_and_no_dominant_key(topic, tier):
    answers = Counter(make(topic, seed=s, difficulty=tier).answer_text for s in SEEDS)
    assert len(answers) >= 12
    assert max(answers.values()) <= 8
    texts = {make(topic, seed=s, difficulty=tier).text for s in SEEDS}
    assert len(texts) >= 40


# -- the tiers genuinely escalate ------------------------------------------------------------------------------------


def _expr_of(q):
    if re.search(r"Let [a-z] be", q.text):
        return None
    return _stem_expression(q.text)[1]


def _all(topic, tier):
    return [make(topic, seed=s, difficulty=tier) for s in SEEDS]


def test_one_step_escalates():
    easy = [_expr_of(q) for q in _all("lin_ineq_one_step", "easy")]
    medium = [_expr_of(q) for q in _all("lin_ineq_one_step", "medium")]
    hard = [_expr_of(q) for q in _all("lin_ineq_one_step", "hard")]
    assert all(re.search(r"(^|\s)[a-z] [+-] \d+ |^\d+ \+ [a-z] ", e + " ") for e in easy), "easy is x ± b or b + x"
    assert all(re.match(r"^\d+[a-z] ", e) for e in medium), "medium is ax op c"
    assert all("/" in e or "(" in e for e in hard), "hard is x/a or a(x + m)"
    fractional = lambda tier: sum("/" in make("lin_ineq_one_step", seed=s, difficulty=tier).answer_text for s in SEEDS)  # noqa: E731
    assert fractional("easy") == 0 and fractional("medium") >= 20


def test_negative_escalates():
    easy = [_expr_of(q) for q in _all("lin_ineq_negative", "easy")]
    medium = [_expr_of(q) for q in _all("lin_ineq_negative", "medium")]
    hard = [_expr_of(q) for q in _all("lin_ineq_negative", "hard")]
    assert all(re.match(r"^-\d*[a-z] [<>≤≥] ", e) for e in easy)
    assert all(re.match(r"^-[a-z]/\d+ ", e) or re.match(r"^-?\d+ - \d*[a-z] ", e) for e in medium)
    assert all(len(re.findall(r"[a-z]", e)) == 2 for e in hard), "hard has the variable on both sides"
    for tier in TIERS:
        for q in _all("lin_ineq_negative", tier):
            var, expr = _stem_expression(q.text)
            ineq = sympy.sympify(_implicit(expr.replace("≤", "<=").replace("≥", ">=")), locals={var: Symbol(var)})
            # in every tier the net x coefficient of (lhs - rhs) is negative: that is what makes the sign reverse
            lhs, rhs = ineq.lhs, ineq.rhs
            net = sympy.Poly(sympy.expand(lhs - rhs), Symbol(var)).coeff_monomial(Symbol(var))
            assert net < 0, f"{tier}: the net x coefficient should be negative in {expr}"


def test_two_step_escalates():
    easy = [_expr_of(q) for q in _all("lin_ineq_two_step", "easy")]
    medium = [_expr_of(q) for q in _all("lin_ineq_two_step", "medium")]
    hard = [_expr_of(q) for q in _all("lin_ineq_two_step", "hard")]
    assert all(re.match(r"^\d+[a-z] [+-] \d+ [<>≤≥] -?\d+$", e) for e in easy)
    medium_q = _all("lin_ineq_two_step", "medium")
    assert all(
        "(" in e or "/" in e or re.match(r"^-?\d+ - \d*[a-z] ", e) or "/" in q.answer_text
        for e, q in zip(medium, medium_q)
    )
    assert not any("/" in q.answer_text for q in _all("lin_ineq_two_step", "easy"))
    assert sum("(" in e or "/" in e for e in medium) >= 30
    compound = [e for e in hard if len(re.findall(r"[<>≤≥]", e)) == 2]
    both = [e for e in hard if len(re.findall(r"[<>≤≥]", e)) == 1]
    assert compound and both and all(len(re.findall(r"[a-z]", e)) == 2 for e in both)
    assert any(re.search(r"\d - \d*[a-z]", e) for e in compound), "some double inequalities have a negative coefficient"


def test_word_problems_escalate():
    easy = [q.text for q in _all("lin_ineq_word", "easy")]
    medium = [q.text for q in _all("lin_ineq_word", "medium")]
    hard = [q.text for q in _all("lin_ineq_word", "hard")]
    assert not any(
        re.search(r"flag fall|a month plus|saved \$|mass of \d+ kg when empty|costs to cover", t) for t in easy
    )
    assert all(
        re.search(r"flag fall|a month plus|saved \$|mass of \d+ kg when empty|costs to cover", t) for t in medium
    )
    assert all(re.search(r"Plan A|mean of|but (less|no more)", t) for t in hard)
    assert {m for t in hard for m in re.findall(r"Plan A|mean of|but (?:less|no more)", t)} >= {"Plan A", "mean of"}


# -- exam chains -----------------------------------------------------------------------------------------------------


def _skill_for(topic):
    return next(_SKILLS[sid] for sid, (_p, exam) in SKILLS.items() if exam == topic)


def _exam(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


@pytest.mark.parametrize("topic", EXAMS)
def test_exam_chain_shape(topic):
    q = _exam(topic, 3)
    shape = {
        "exd_x_ineq_one_step": ("inequality", "number", "number"),
        "exd_x_ineq_negative": ("inequality", "number", "number"),
        "exd_x_ineq_two_step": ("inequality", "number", "number"),
        "exd_x_ineq_word": ("inequality", "number", "number", "inequality"),
    }[topic]
    assert tuple(p.answer_kind for p in q.parts) == shape
    assert [p.label for p in q.parts] == list("abcd"[: len(shape)])
    assert all(p.marks >= 1 for p in q.parts) and q.parts[0].marks == 2
    assert 4 <= sum(p.marks for p in q.parts) <= 8
    for p in q.parts:
        assert len(p.distractors) >= 2
        assert not any(GRADERS[p.answer_kind](p, d).correct for d in p.distractors), p.distractors
        assert GRADERS[p.answer_kind](p, p.answer_text).correct
        assert all(n in CATALOGUE for n in p.option_notes.values())
    if topic != "exd_x_ineq_two_step":
        assert q.parts[2].follow_through and "part_b" in q.parts[2].follow_through
    else:
        assert all(p.follow_through is None for p in q.parts)


@pytest.mark.parametrize("topic", EXAMS)
def test_exam_answers_rederived_from_the_text(topic):
    for seed in range(60):
        q = _exam(topic, seed)
        var, expected = _expected_set(q)
        assert _key_set(q.parts[0].answer_text) == expected, (q.text, q.parts[0].answer_text)
        ints = [i for i in range(-300, 301) if sympy.sympify(i) in expected]
        if topic == "exd_x_ineq_two_step":
            assert q.parts[1].answer == len(ints) and q.parts[2].answer == sum(ints)
            continue
        which = "greatest" if "greatest" in q.parts[1].prompt else "least"
        n = max(ints) if which == "greatest" else min(ints)
        assert q.parts[1].answer == n
        if topic == "exd_x_ineq_word":
            m = re.search(r"Let [a-z] be", q.text)
            fam = next((pick(re.search(pat, q.text).groups()) for pat, pick in _FAMILIES if re.search(pat, q.text)))
            slope, const = int(fam[0]), int(fam[1])
            assert q.parts[2].answer == slope * n + const
            extra = re.search(r"must be at (least|most) (\d+)", q.parts[3].prompt)
            m_val = int(extra.group(2))
            both = expected.intersect(
                Interval(m_val, S.Infinity) if extra.group(1) == "least" else Interval(-S.Infinity, m_val)
            )
            assert _key_set(q.parts[3].answer_text) == both
        else:
            expr = re.search(r"value of (.+?) when [a-z] is that integer", q.parts[2].prompt).group(1)
            assert q.parts[2].answer == _expr(expr, var).subs(Symbol(var), n)


@pytest.mark.parametrize("topic", EXAMS)
def test_exam_variety_over_300_seeds(topic):
    texts = Counter(_exam(topic, s).answer_text for s in range(300))
    assert len(texts) >= 100
    assert max(texts.values()) <= 6


@pytest.mark.parametrize("topic", EXAMS)
def test_exam_is_deterministic_and_clean(topic):
    for seed in range(40):
        q = _exam(topic, seed)
        assert q.to_json() == _exam(topic, seed).to_json()
        assert find_text_defects(q) == [], (topic, seed, find_text_defects(q))
        assert q.stem and "(a)" in q.text


@pytest.mark.parametrize("topic", ["exd_x_ineq_one_step", "exd_x_ineq_negative", "exd_x_ineq_word"])
def test_follow_through_from_a_wrong_integer_earns_the_later_marks(topic):
    checked = 0
    for seed in range(40):
        q = _exam(topic, seed)
        b, c = q.parts[1], q.parts[2]
        wrong_b = b.answer + 1 if b.answer + 1 != 0 else b.answer - 1
        value = sympy.sympify(c.follow_through).subs(Symbol("part_b"), wrong_b)
        if value == 0 or value == c.answer or wrong_b == b.answer:
            continue
        reply = {"a": q.parts[0].answer_text, "b": str(wrong_b), "c": str(value)}
        result = grade("inequality", q, reply)
        assert not result.correct
        assert result.parts["b"].marks == 0
        assert result.parts["c"].follow_through and result.parts["c"].marks == c.marks
        assert result.parts["a"].marks == q.parts[0].marks
        checked += 1
    assert checked >= 20


@pytest.mark.parametrize("topic", EXAMS)
def test_full_marks_for_the_canonical_reply_through_the_skill_grader(topic):
    skill = _skill_for(topic)
    for seed in range(20):
        q = _exam(topic, seed)
        result = skill_grade(skill, q, {p.label: p.answer_text for p in q.parts})
        assert result.correct and result.marks == result.max_marks == sum(p.marks for p in q.parts)


# -- the skills ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("skill_id", SKILLS)
def test_the_skill_is_active_with_a_tier_ladder_and_an_exam(skill_id):
    skill = _SKILLS[skill_id]
    practice, exam = SKILLS[skill_id]
    assert skill.status == "active" and not getattr(skill, "deferred_reason", None)
    assert skill.answer == "inequality"
    assert [(s.topic, s.difficulty) for s in skill.sources] == [(practice, t) for t in TIERS] + [(exam, "exam")]
    assert skill.sources[-1].stages == ["exam"]
    ids = [m.id for m in skill.misconceptions]
    assert len(ids) >= 4 and all(i in CATALOGUE for i in ids), ids
    assert not any(i.startswith("misconception-") for i in ids)


def test_the_one_step_mental_block_is_untouched():
    skill = _SKILLS["one-step-inequalities"]
    assert [(lv.topic, lv.params) for lv in skill.mental.levels] == [
        ("inequality", {"kind": "one_step", "hi": h}) for h in (6, 9, 12)
    ]


def test_every_distractor_misconception_is_in_the_catalogue_and_documented():
    for skill_id, (practice, exam) in SKILLS.items():
        used = set()
        for topic, tier in [(practice, t) for t in TIERS] + [(exam, "exam")]:
            for seed in range(30):
                q = make(topic, seed=seed, difficulty=tier)
                used |= set(q.option_notes.values())
                for p in q.parts:
                    used |= set(p.option_notes.values())
        assert used <= set(CATALOGUE)
        listed = {m.id for m in _SKILLS[skill_id].misconceptions}
        assert len(used & listed) >= 3, (skill_id, used, listed)


def test_json_roundtrip_of_a_chain():
    q = _exam("exd_x_ineq_word", 5)
    d = json.loads(q.to_json())
    assert [p["answer_kind"] for p in d["parts"]] == ["inequality", "number", "number", "inequality"]
