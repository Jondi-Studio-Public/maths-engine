"""Independent oracle for simplify-algebraic-fractions.

Parses each rendered question and derives the answer from `question.text`
alone, using sympy on values parsed out of the rendered text. Never imports
`mathgen.generators.t1_simplify_algebraic_fraction_build`.

Uses `sympy.cancel(numerator_expr / denominator_expr)` -- a sympy entry point
the generator never calls (it uses `sympy.div`) -- so this is a genuinely
different route, not a restatement of the generator's own construction.
Also independently confirms the "expanded" answer_form idea from the
learner's side: `sympy.expand(parsed_answer) == parsed_answer`.
"""

import re

import sympy
from sympy import symbols
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.skills import load_skills
from mathgen import make

x = symbols("x")
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
_PROMPT_RE = re.compile(r"^Simplify \((.+)\) / \((.+)\)\.$")


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMATIONS, local_dict={"x": x})


class TestSimplifyAlgebraicFractionsOracle:
    def test_oracle(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "simplify-algebraic-fractions")

        examined = 0
        challenged = []
        for source in skill.sources:
            for seed in range(60):
                q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **(source.params or {}))
                examined += 1
                text, answer_text = q.text, q.answer_text

                m = _PROMPT_RE.match(text)
                if not m:
                    challenged.append({"seed": seed, "issue": "unparsed question shape", "text": text})
                    continue

                numerator_expr = _parse(m.group(1))
                denominator_expr = _parse(m.group(2))
                parsed_answer = _parse(answer_text)

                reduced = sympy.cancel(numerator_expr / denominator_expr)
                if sympy.simplify(reduced - parsed_answer) != 0:
                    challenged.append(
                        {
                            "issue": f"oracle cancel() gave {reduced}, answer_text stated {answer_text}",
                            "text": text,
                        }
                    )
                    continue

                # From-scratch reimplementation of half of the "expanded" idea:
                # a genuine polynomial answer must already be its own expansion.
                if sympy.expand(parsed_answer) != parsed_answer:
                    challenged.append(
                        {
                            "issue": "answer_text is not already in expanded form",
                            "text": text,
                            "answer_text": answer_text,
                        }
                    )
                    continue

                # The reduced result must be a bare polynomial, never a fraction.
                if reduced.is_rational_function(x) and reduced.as_numer_denom()[1] != 1:
                    challenged.append(
                        {
                            "issue": f"oracle's cancelled result {reduced} is still a fraction",
                            "text": text,
                        }
                    )

        assert not challenged, f"{len(challenged)} of {examined} challenged: {challenged}"
        print(f"simplify-algebraic-fractions oracle: examined {examined}, challenged 0")
