"""w10: the eight `-explain` open-response skills and the four typed-step proof skills.

Open-response skills: the rubric is well formed, the model answer agrees with the prompt (re-derived here from the
prompt's own numbers, not from the generator), the engine returns pending, and the hand-off round-trips.
Proof skills: the canonical chain and every learner spelling of it grade correct, every wrong chain grades wrong
and carries a catalogue id, and the checks are independent of the generator.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy as sp
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.grading.open_response import (
    build_marking_request,
    fold_marking_result,
    render_marking_prompt,
    rubric_of,
    self_marking_result,
)
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

X = sp.Symbol("x")
T = standard_transformations + (implicit_multiplication_application,)
TIERS = ("easy", "medium", "hard")
SEEDS = 60

EXPLAIN = {
    "meaning-of-equality-explain": "w10_explain_equality",
    "inverse-operations-explain": "w10_explain_inverse",
    "solution-and-solution-set-explain": "w10_explain_solution",
    "identity-and-no-solution-explain": "w10_explain_identity",
    "check-solution-by-substitution-explain": "w10_explain_check",
    "recognise-equivalent-expressions-explain": "w10_explain_equivalent",
    "independent-dependent-variables-explain": "w10_explain_variables",
    "proportional-vs-non-proportional-explain": "w10_explain_proportional",
}
PROOFS = {
    "proof-irrational-surd-steps": ("w10_proof_irrational_surd", "w10_proof_irrational_surd_exam"),
    "proof-parity-steps": ("w10_proof_parity", "w10_proof_parity_exam"),
    "proof-am-gm-steps": ("w10_proof_am_gm", "w10_proof_am_gm_exam"),
    "proof-triangle-inequality-steps": ("w10_proof_triangle", "w10_proof_triangle_exam"),
}


@pytest.fixture(scope="module")
def skills():
    return {s.id: s for s in load_skills()[1]}


def parse(text):
    return parse_expr(
        text.replace("^", "**").replace("×", "*").replace("÷", "/"), local_dict={"x": X}, transformations=T
    )


# -- the explain skills ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("skill_id, topic", EXPLAIN.items())
def test_explain_skill_wiring(skills, skill_id, topic):
    skill = skills[skill_id]
    assert skill.status == "active" and skill.answer == "open_response"
    assert [(s.topic, s.difficulty) for s in skill.sources] == [(topic, t) for t in TIERS]


@pytest.mark.parametrize("topic", EXPLAIN.values())
def test_explain_questions_are_clean_deterministic_and_varied(topic):
    pools = {}
    for tier in TIERS:
        questions = [make(topic, seed=s, difficulty=tier) for s in range(SEEDS)]
        assert len({q.text for q in questions}) >= 20, (topic, tier)
        pools[tier] = {q.text for q in questions}
        for seed, q in enumerate(questions):
            assert q.to_json() == make(topic, seed=seed, difficulty=tier).to_json()
            assert find_text_defects(q, include_working=True) == [], (topic, tier, seed)
            assert q.answer_kind == "open_response" and not q.distractors and not q.parts
            rubric = rubric_of(q.answer)
            assert 3 <= rubric["max_marks"] <= 6
            ids = [c["id"] for c in rubric["criteria"]]
            assert len(set(ids)) == len(ids)
            assert [e["label"] for e in rubric["exemplars"]] == ["full marks", "partly correct", "common weak answer"]
            assert rubric["exemplars"][0]["marks"] == rubric["max_marks"]
            assert rubric["exemplars"][0]["answer"] == q.answer_text == rubric["reference"]
            assert rubric["exemplars"][2]["marks"] == 0 and 0 < rubric["exemplars"][1]["marks"] < rubric["max_marks"]
            assert "\n" not in q.answer_text and "*" not in q.answer_text
            for c in rubric["criteria"]:
                assert c["description"].endswith(".")
    assert pools["easy"] != pools["medium"] != pools["hard"]


@pytest.mark.parametrize("skill_id, topic", EXPLAIN.items())
def test_explain_hand_off_round_trip(skills, skill_id, topic):
    skill = skills[skill_id]
    for tier in TIERS:
        for seed in range(10):
            q = make(topic, seed=seed, difficulty=tier)
            pending = grade(skill, q, q.answer_text)
            assert pending.pending and not pending.correct and pending.marks is None
            assert grade(skill, q, "").marks == 0 and not grade(skill, q, "").pending
            request = build_marking_request(q, "My explanation.", skill=skill)
            assert request["question"]["skill"] == skill_id and request["max_marks"] == pending.max_marks
            prompt = render_marking_prompt(request)
            assert all(c["id"] in prompt for c in request["criteria"])
            ids = [c["id"] for c in request["criteria"]]
            full = fold_marking_result(request, self_marking_result(request, dict.fromkeys(ids, True)))
            none = fold_marking_result(request, self_marking_result(request, dict.fromkeys(ids, False)))
            assert full.correct and full.marks == full.max_marks and none.marks == 0 and not none.correct


# -- the model answers agree with the prompts (oracle: read the numbers back out of the prompt) -----------------------


def test_equality_statement_truth():
    for tier_seed in range(SEEDS):
        q = make("w10_explain_equality", seed=tier_seed, difficulty="easy")
        a, b, c, d = map(int, re.search(r"(\d+) \+ (\d+) = (\d+) \+ (\d+)", q.text).groups())
        assert ("is true" in q.answer_text) == (a + b == c + d)
        assert ("statement is false" in q.answer_text) == (a + b != c + d)
        assert f"{a} + {b} = {a + b}" in q.answer["criteria"][0]["description"]
        assert f"{c} + {d} = {c + d}" in q.answer["criteria"][1]["description"]


def test_equality_medium_and_hard_numbers():
    for seed in range(SEEDS):
        q = make("w10_explain_equality", seed=seed, difficulty="medium")
        m = re.search(r"(\d+) \+ (\d+) = □ \+ (\d+)", q.text)
        if m:
            a, b, c = map(int, m.groups())
            n = a + b - c
            assert f"The box must be {a + b} - {c} = {n}." in q.answer_text and n > 0
        else:
            a, b, t, d = map(int, re.search(r"(\d+) \+ (\d+) = (\d+) \+ (\d+)", q.text).groups())
            assert a + b == t and f"{t} + {d} is {t + d}" in q.answer_text
        q = make("w10_explain_equality", seed=seed, difficulty="hard")
        if not re.search(r"\d+x \+ \d+ = \d+", q.text):
            a, t = map(int, re.search(r"x \+ (\d+) = (\d+)", q.text).groups())
            b = int(re.search(r"(?:adds|when) (\d+)", q.text).group(1))
            assert f"x + {a} + {b} = {t} + {b}, which is {t + b}" in q.answer_text
        else:
            m, b, t = map(int, re.search(r"(\d+)x \+ (\d+) = (\d+)", q.text).groups())
            x = int(re.search(r"[xX] = (\d+)", q.text.replace(f"{m}x + {b} = {t}", "")).group(1))
            assert m * x + b == t and f"{m}x = {t - b}" in q.answer_text and m * x == t - b


def test_inverse_answers_solve_the_equation():
    for seed in range(SEEDS):
        q = make("w10_explain_inverse", seed=seed, difficulty="easy")
        eq = re.search(r"To solve (.+?), which", q.text).group(1)
        left, right = eq.split(" = ")
        x = sp.solve(sp.Eq(parse(left.replace("÷", "/")), int(right)), X)[0]
        assert f"Gives x = {x}." in [c["description"] for c in q.answer["criteria"]], eq
        q = make("w10_explain_inverse", seed=seed, difficulty="medium")
        eq = re.search(r"(\d+x [+-] \d+ = \d+)", q.text).group(1)
        left, right = eq.split(" = ")
        x = sp.solve(sp.Eq(parse(left), int(right)), X)[0]
        assert q.answer_text.endswith(f"x = {x}."), eq
        q = make("w10_explain_inverse", seed=seed, difficulty="hard")
        a, b, c = map(int, re.search(r"(\d+)x - (\d+) = (\d+)", q.text).groups())
        x = sp.solve(sp.Eq(a * X - b, c), X)[0]
        assert x.is_integer and f"x = {x}." in q.answer_text and f"{a}x = {c + b}" in q.answer_text


def test_solution_answers():
    for seed in range(SEEDS):
        q = make("w10_explain_solution", seed=seed, difficulty="easy")
        a, b, c = map(int, re.search(r"(\d+)x \+ (\d+) = (\d+)", q.text).groups())
        v = int(re.search(r"(?:x = |thinks )(\d+)", q.text).group(1))
        assert (f"x = {v} is a solution." in q.answer_text) == (a * v + b == c)
        assert f"{a} × {v} + {b} = {a * v + b}" in q.answer_text
        q = make("w10_explain_solution", seed=seed, difficulty="medium")
        cands = list(map(int, re.search(r"(\d+), (\d+) and (\d+)", q.text).groups()))
        a, c = map(int, re.search(r"x \+ (\d+) = (\d+)", q.text).groups())
        assert [n for n in cands if n + a == c] == [c - a] and f"so {c - a} is a solution" in q.answer_text
        q = make("w10_explain_solution", seed=seed, difficulty="hard")
        m = re.search(r"x \+ (\d+) > (\d+)", q.text)
        if m:
            a, c = map(int, m.groups())
            inside, outside = map(int, re.findall(r"\d+", q.text.replace(m.group(0), ""))[:2])
            assert inside + a > c and not outside + a > c and f"greater than {c - a}." in q.answer_text
        else:
            s = int(re.search(r"x² = (\d+)", q.text).group(1))
            r = int(sp.sqrt(s))
            assert r * r == s and f"{{-{r}, {r}}}" in q.answer_text


def _kind(left, right):
    sols = sp.solve(sp.Eq(left, right), X)
    if sp.expand(left - right) == 0:
        return "all"
    return "none" if not sols else "one"


def test_identity_kinds_match_the_algebra():
    phrase = {"none": "has no solution", "all": "has every number as a solution", "one": "has exactly one solution"}
    for tier in TIERS:
        seen = Counter()
        for seed in range(SEEDS):
            q = make("w10_explain_identity", seed=seed, difficulty=tier)
            eq = re.search(
                r"(?:Does|whether|solutions|claims) (.+?) (?:have no solution|has no solution|has \(none|is true for)",
                q.text,
            ).group(1)
            left, right = eq.split(" = ")
            kind = _kind(parse(left), parse(right))
            assert phrase[kind] in " ".join(c["description"] for c in q.answer["criteria"]), (tier, eq, kind)
            seen[kind] += 1
        if tier != "easy":
            assert set(seen) == {"none", "all", "one"}, (tier, seen)
        else:
            assert set(seen) == {"none", "all"}


def test_check_questions_match_the_algebra():
    for seed in range(SEEDS):
        q = make("w10_explain_check", seed=seed, difficulty="easy")
        a, b, c = map(int, re.search(r"(\d+)x \+ (\d+) = (\d+)", q.text).groups())
        v = int(re.search(r"x = (\d+)", q.text).group(1))
        assert ("is correct" in q.answer_text) == (a * v + b == c)
        q = make("w10_explain_check", seed=seed, difficulty="medium")
        a, b, c, d = map(int, re.search(r"(\d+)x \+ (\d+) = (\d+)x \+ (\d+)", q.text).groups())
        v = int(re.search(r"x = (\d+)", q.text).group(1))
        assert ("Ben is right." in q.answer_text) == (a * v + b == c * v + d)
        q = make("w10_explain_check", seed=seed, difficulty="hard")
        a, b, c = map(int, re.search(r"(\d+)x - (\d+) = (\d+)", q.text).groups())
        w = int(re.search(r"x = (\d+)", q.text).group(1))
        x = sp.solve(sp.Eq(a * X - b, c), X)[0]
        assert a * w - b != c and f"so x = {x}." in q.answer_text and x.is_integer


def test_equivalence_verdicts_match_the_algebra():
    for tier in TIERS:
        seen = Counter()
        for seed in range(SEEDS):
            q = make("w10_explain_equivalent", seed=seed, difficulty=tier)
            m = re.search(r"Try x = \d+: (.+?) gives -?\d+ (?:and|but) (.+?) gives", q.answer_text)
            assert m.group(1) in q.text and m.group(2) in q.text
            e1, e2 = parse(m.group(1)), parse(m.group(2))
            equivalent = sp.expand(e1 - e2) == 0
            assert ("The expressions are equivalent" in q.answer_text) == equivalent, (tier, seed)
            if not equivalent:
                t = int(re.search(r"x = (\d+)", q.answer_text).group(1))
                assert e1.subs(X, t) != e2.subs(X, t)
            seen[equivalent] += 1
        assert set(seen) == {True, False}


def test_variable_formula_values():
    for seed in range(SEEDS):
        q = make("w10_explain_variables", seed=seed, difficulty="medium")
        fee, rate, d1, d2 = map(int, re.search(r"C = (\d+) \+ (\d+)d.*d = (\d+) and d = (\d+)", q.text).groups())
        assert (
            f"C = {fee + rate * d1} when d = {d1} and C = {fee + rate * d2} when d = {d2}"
            in q.answer["criteria"][1]["description"]
        )


def test_proportionality_verdicts():
    for seed in range(SEEDS):
        q = make("w10_explain_proportional", seed=seed, difficulty="easy")
        pairs = [tuple(map(int, p)) for p in re.findall(r"x = (\d+), y = (\d+)", q.text)]
        proportional = len({sp.Rational(y, x) for x, y in pairs}) == 1
        assert ("so y is proportional to x" in q.answer_text) == proportional
        q = make("w10_explain_proportional", seed=seed, difficulty="medium")
        rule = re.search(r"relationship (.+?) proportional", q.text).group(1)
        expr = parse(rule.split("=")[1])
        proportional = expr.subs(X, 0) == 0
        assert (f"So {rule} is proportional." in q.answer_text) == proportional
        q = make("w10_explain_proportional", seed=seed, difficulty="hard")
        m = re.search(r"charges \$(\d+) for each kilometre\. .+? charges a \$(\d+) fee plus \$(\d+) for each", q.text)
        rate, fee, rate_b = map(int, m.groups())
        assert rate == rate_b and fee > 0 and "proportional to the distance for" in q.answer_text


# -- the typed-step proof skills ----------------------------------------------------------------------------------


@pytest.mark.parametrize("skill_id, topics", PROOFS.items())
def test_proof_skill_wiring(skills, skill_id, topics):
    skill = skills[skill_id]
    topic, exam = topics
    assert skill.status == "active" and skill.answer == "step_chain"
    assert [(s.topic, s.difficulty) for s in skill.sources] == [
        (topic, "easy"),
        (topic, "medium"),
        (topic, "hard"),
        (exam, "exam"),
    ]


@pytest.mark.parametrize("skill_id, topics", PROOFS.items())
def test_proof_questions_key_and_distractors(skills, skill_id, topics):
    skill, topic = skills[skill_id], topics[0]
    for tier in TIERS:
        answers = set()
        for seed in range(SEEDS):
            q = make(topic, seed=seed, difficulty=tier)
            assert q.to_json() == make(topic, seed=seed, difficulty=tier).to_json()
            assert grade(skill, q, q.answer_text).correct, (tier, seed, q.answer_text)
            assert len(q.distractors) >= 3 and len(set(q.distractors)) == len(q.distractors)
            for text in q.distractors:
                result = grade(skill, q, text)
                assert not result.correct, (tier, seed, text)
                assert q.option_notes[text] in CATALOGUE and text in q.option_notes
                assert result.misconception == q.option_notes[text]
            assert find_text_defects(q, include_working=True) == []
            answers.add(q.answer_text)
        assert len(answers) >= 12, (topic, tier, len(answers))


@pytest.mark.parametrize("skill_id, topics", PROOFS.items())
def test_proof_reference_chains_are_true_identities(skill_id, topics):
    """Oracle: read start/target/hypothesis out of the marking dict and prove every consecutive pair equal with sympy
    (exact simplification), without the grader or the generator's own probe check."""
    topic = topics[0]
    for tier in TIERS:
        for seed in range(25):
            spec = make(topic, seed=seed, difficulty=tier).answer
            symbols = {n: sp.Symbol(n, real=True) for n in {*spec["symbols"], *spec.get("given", {})}}
            for n, kind in spec["symbols"].items():
                if kind == "positive":
                    symbols[n] = sp.Symbol(n, positive=True)
                if kind == "integer":
                    symbols[n] = sp.Symbol(n, integer=True)
            ns = {**symbols, "sqrt": sp.sqrt, "abs": sp.Abs}
            sub = {symbols[n]: sp.sympify(e, locals=ns) for n, e in spec.get("given", {}).items()}
            exprs = [sp.sympify(line, locals=ns).xreplace(sub) for line in spec["reference_lines"]]
            for first, second in zip(exprs, exprs[1:]):
                diff = first - second
                if sp.simplify(diff) != 0:  # not shown by simplify: evaluate at fixed rational points instead
                    for k, point in enumerate((sp.Rational(13, 7), sp.Rational(-9, 4), sp.Rational(17, 5))):
                        env = {
                            s: abs(point) + i if s.is_positive else point + i
                            for i, s in enumerate(sorted(diff.free_symbols, key=str))
                        }
                        assert abs(sp.N(diff.xreplace(env), 30)) < 1e-25, (tier, seed, spec)
            assert spec["reference_lines"][0] == spec["start"] and spec["reference_lines"][-1] == spec["target"]


def test_learner_spellings_of_a_surd_chain(skills):
    skill = skills["proof-irrational-surd-steps"]
    q = next(
        q
        for q in (make("w10_proof_irrational_surd", seed=s, difficulty="easy") for s in range(60))
        if re.search(r"\(\w √2\)", q.answer_text)
    )
    p, rest = q.answer_text.split(" = ", 1)
    q_letter = re.search(r"\((\w) √2\)", q.answer_text).group(1)
    for reply in (
        f"{p} = ({q_letter}√2)^2 = 2{q_letter}^2",
        f"{p}\n({q_letter}*sqrt(2))^2\n2*{q_letter}^2",
        f"{p} = ({q_letter} sqrt(2))^2 = 2 {q_letter}^2",
        f"{p}\n({q_letter}√(2))^2\n2{q_letter}^2",
    ):
        assert grade(skill, q, reply).correct, reply
    assert not grade(skill, q, f"{p} = 2{q_letter}^2 = 2{q_letter}").correct
    assert not grade(skill, q, f"({q_letter}√2)^2 = 2{q_letter}^2").correct  # does not start from the statement


def test_proof_exam_parts(skills):
    for skill_id, (_topic, exam) in PROOFS.items():
        skill = skills[skill_id]
        for seed in range(30):
            q = make(exam, seed=seed, difficulty="exam")
            assert [p.label for p in q.parts] == ["a", "b"] and [p.marks for p in q.parts] == [2, 3]
            assert [p.answer_kind for p in q.parts] == ["step_chain", "step_chain"]
            full = grade(skill, q, q.answer_text)
            assert full.correct and full.marks == full.max_marks == 5, (skill_id, seed)
            canon = {p.label: p.answer_text for p in q.parts}
            assert grade(skill, q, canon).marks == 5
            for p in q.parts:
                assert len(p.distractors) >= 3
                for text in p.distractors:
                    result = grade(skill, q, {**canon, p.label: text})
                    assert result.parts[p.label].marks == 0 and result.marks == 5 - p.marks, (skill_id, seed, text)
                    assert p.option_notes[text] in CATALOGUE
            assert find_text_defects(q) == []
            assert q.to_json() == make(exam, seed=seed, difficulty="exam").to_json()
