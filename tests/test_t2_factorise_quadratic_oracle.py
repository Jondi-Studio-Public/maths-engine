"""Independent oracle for factorise-quadratic.

Parses the trinomial's a, b, c coefficients out of `question.text` and
independently calls `sympy.factor` on `a*x**2 + b*x + c` (built fresh here,
never imported from the generator), asserting it is sympy-equal to
`question.answer_text` parsed the same way. Never imports
`mathgen.generators.t2_factorise_quadratic_build`.
"""

import re

import sympy
from sympy import symbols
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.skills import load_skills
from mathgen import make

x = symbols("x")
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

# The pinned template is "Factorise {trinomial_text}." where trinomial_text is
# core.join_terms([(a, "x^2"), (b, "x"), (c, "")]) -- join_terms drops any
# zero-coefficient term entirely, so the trinomial may render with 1, 2 or 3
# terms. Each present term is one of: "{coef}x^2" (a != 0, always present
# here), "{coef}x" (b term) or "{coef}" (bare constant, c term). Parse the
# whole thing with sympy directly rather than trying to positionally slice it.
_QUESTION_RE = re.compile(r"^Factorise (.+)\.$")


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMATIONS, local_dict={"x": x})


class TestFactoriseQuadraticOracle:
    def test_oracle(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "factorise-quadratic")

        examined = 0
        challenged = []
        for source in skill.sources:
            if source.difficulty == "exam":
                continue  # the exam source is covered by this skill's tests/test_exd_x_* file
            for seed in range(60):
                q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **(source.params or {}))
                examined += 1
                m = _QUESTION_RE.match(q.text)
                if not m:
                    challenged.append({"seed": seed, "issue": "unparsed question shape", "text": q.text})
                    continue

                trinomial_text = m.group(1)
                trinomial = _parse(trinomial_text)
                poly = sympy.Poly(trinomial, x)
                coeffs = poly.all_coeffs()
                while len(coeffs) < 3:
                    coeffs.insert(0, 0)
                a, b, c = (int(v) for v in coeffs[-3:])

                # Rebuild the quadratic fresh from the parsed integers -- this
                # is the independent construction, not a re-parse of `trinomial`.
                rebuilt = a * x**2 + b * x + c
                factored = sympy.factor(rebuilt)
                given = _parse(q.answer_text)

                if sympy.simplify(factored - given) != 0:
                    challenged.append(
                        {
                            "issue": f"oracle factored {factored}, answer_text stated {q.answer_text}",
                            "text": q.text,
                        }
                    )
                    continue
                # Also confirm the parsed trinomial itself agrees with the
                # rebuilt-from-integers quadratic (catches a regex/parse slip
                # independent of the factoring route above).
                if sympy.simplify(trinomial - rebuilt) != 0:
                    challenged.append(
                        {
                            "issue": f"parsed trinomial {trinomial} != rebuilt {rebuilt}",
                            "text": q.text,
                        }
                    )
                    continue
                # answer_form cross-check, from the learner's side: a
                # "Factorise" prompt's answer_text must carry a bracket and
                # must not already sit in expanded (distributed) form.
                if "(" not in q.answer_text:
                    challenged.append({"issue": "factorise prompt but answer_text has no bracket", "text": q.text})
                elif sympy.expand(given) == given:
                    challenged.append(
                        {
                            "issue": "factorise prompt but answer_text is already fully expanded",
                            "text": q.text,
                        }
                    )

        assert not challenged, f"{len(challenged)} of {examined} challenged: {challenged}"
        print(f"factorise-quadratic oracle: examined {examined}, challenged 0")
