"""Independent oracle for difference-of-squares.

Parses each rendered question and derives the answer from `question.text`
and `question.answer_text` alone, using sympy on values parsed out of the
rendered text. Never imports `mathgen.generators.p8a_difference_of_squares_build`.

Also confirms the `answer_form` requirement from the learner's side: parses
the prompt's own verb ("Expand"/"Factorise") and checks which shape
`answer_text` actually is by applying `sympy.expand`/checking for a bracket
on the parsed `answer_text` itself, independent of what the generator
declared.
"""

import re

import sympy
from sympy import symbols
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.skills import load_skills
from mathgen import make

x = symbols("x")
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

_EXPAND_TWO = re.compile(r"^Expand \((.+?)\)\((.+?)\)\.$")
_EXPAND_SQ = re.compile(r"^Expand \((.+?)\)\^2\.$")
_FACTORISE = re.compile(r"^Factorise (.+?) using a recognised special product\.$")


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMATIONS, local_dict={"x": x})


class TestDifferenceOfSquaresOracle:
    def test_oracle(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "difference-of-squares")

        examined = 0
        challenged = []
        for source in skill.sources:
            for seed in range(60):
                q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **(source.params or {}))
                examined += 1
                text, answer_text = q.text, q.answer_text

                m2 = _EXPAND_TWO.match(text)
                msq = _EXPAND_SQ.match(text)
                mf = _FACTORISE.match(text)

                if m2:
                    f1, f2 = _parse(m2.group(1)), _parse(m2.group(2))
                    expected = sympy.expand(f1 * f2)
                    got = _parse(answer_text)
                    if sympy.simplify(expected - got) != 0:
                        challenged.append(
                            {"issue": f"oracle expanded {expected}, answer_text stated {answer_text}", "text": text}
                        )
                        continue
                    # answer_form cross-check, from the learner's side: the
                    # prompt says "expand", so answer_text's raw text must
                    # carry no bracket, and sympy.expand() of the parsed
                    # answer_text must change nothing (it is already expanded).
                    if "(" in answer_text or sympy.expand(got) != got:
                        challenged.append(
                            {
                                "issue": "expand prompt but answer_text is not already in expanded form",
                                "text": text,
                                "answer_text": answer_text,
                            }
                        )
                    continue

                if msq:
                    f1 = _parse(msq.group(1))
                    expected = sympy.expand(f1**2)
                    got = _parse(answer_text)
                    if sympy.simplify(expected - got) != 0:
                        challenged.append(
                            {"issue": f"oracle expanded {expected}, answer_text stated {answer_text}", "text": text}
                        )
                        continue
                    if "(" in answer_text or sympy.expand(got) != got:
                        challenged.append(
                            {
                                "issue": "expand prompt but answer_text is not already in expanded form",
                                "text": text,
                                "answer_text": answer_text,
                            }
                        )
                    continue

                if mf:
                    trinomial_text = mf.group(1)
                    trinomial = _parse(trinomial_text)
                    factored = sympy.factor(trinomial)
                    given = _parse(answer_text)
                    if sympy.simplify(factored - given) != 0 or sympy.simplify(trinomial - given) != 0:
                        challenged.append(
                            {"issue": f"oracle factored {factored}, answer_text stated {answer_text}", "text": text}
                        )
                        continue
                    # answer_form cross-check, from the learner's side: the
                    # prompt says "factorise", so answer_text's raw text must
                    # carry a bracket, and it must not already sit in expanded
                    # (distributed) form.
                    if "(" not in answer_text:
                        challenged.append(
                            {
                                "issue": "factorise prompt but answer_text has no bracket",
                                "text": text,
                                "answer_text": answer_text,
                            }
                        )
                    elif sympy.expand(given) == given:
                        challenged.append(
                            {
                                "issue": "factorise prompt but answer_text is already fully expanded",
                                "text": text,
                                "answer_text": answer_text,
                            }
                        )
                    continue

                challenged.append({"seed": seed, "issue": "unparsed question shape", "text": text})

        assert not challenged, f"{len(challenged)} of {examined} challenged: {challenged}"
        print(f"difference-of-squares oracle: examined {examined}, challenged 0")
