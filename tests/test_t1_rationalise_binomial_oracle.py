"""Independent oracle for rationalise-binomial-denominator.

Parses each rendered question's own `question.text` for c/a/s/b and
independently recomputes the rationalised value with `sympy.radsimp` -- never
importing `mathgen.generators.t1_rationalise_binomial_build` or reusing any of
its c'/D'/g reduction arithmetic. Confirms the recomputed value agrees with
the parsed `answer_text` and that `answer_text`'s own denominator (via
`sympy.fraction`) carries no surviving radical -- a from-scratch
reimplementation of "denominator has no surviving radical", not a call to
`mathgen.grading._rationalised_denominator_flagged_parsed`.
"""

import re

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.skills import load_skills
from mathgen import make

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

# All three tiers share one prompt shape (unicode '√' in the PROMPT only --
# `answer_text` stays ASCII "sqrt(...)", read directly by `_parse`):
# "Rationalise the denominator of {c}/({a} [+-] √{b}). Give your answer ..."
_QUESTION_RE = re.compile(
    r"^Rationalise the denominator of (\d+)/\((\d+) ([+-]) √(\d+)\)\. "
    r"Give your answer as a single simplified expression with a rational denominator\.$"
)


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMATIONS)


def _denominator_has_surviving_radical(expr) -> bool:
    """True if `sympy.fraction(expr)`'s denominator contains a `Pow` with a
    non-integer rational exponent anywhere in it -- built from scratch here,
    never imported from `mathgen.grading`."""
    _, denominator = sympy.fraction(expr)
    for node in sympy.preorder_traversal(denominator):
        if isinstance(node, sympy.Pow):
            exponent = node.exp
            if getattr(exponent, "is_Rational", False) and not getattr(exponent, "is_Integer", False):
                return True
    return False


class TestRationaliseBinomialDenominatorOracle:
    def test_oracle(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "rationalise-binomial-denominator")

        examined = 0
        challenged = []
        for source in skill.sources:
            # The exam source has its own from-scratch independent oracle in
            # tests/test_exd_e1_rationalise_binomial_exam.py (test_oracle_agrees_with_answer_text),
            # so skipping it here is a scope boundary, not a coverage gap.
            if source.difficulty == "exam":
                continue
            for seed in range(60):
                q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **(source.params or {}))
                examined += 1
                text, answer_text = q.text, q.answer_text

                m = _QUESTION_RE.match(text)
                if not m:
                    challenged.append({"seed": seed, "issue": "unparsed question shape", "text": text})
                    continue

                c, a, sign_text, b = m.group(1), m.group(2), m.group(3), m.group(4)
                c, a, b = int(c), int(a), int(b)
                sign = 1 if sign_text == "+" else -1

                original_fraction = sympy.Rational(c) / (sympy.Integer(a) + sign * sympy.sqrt(b))
                expected = sympy.radsimp(original_fraction)
                given = _parse(answer_text)

                if sympy.simplify(expected - given) != 0:
                    challenged.append(
                        {
                            "issue": f"oracle rationalised {expected}, answer_text stated {answer_text}",
                            "text": text,
                        }
                    )
                    continue

                if _denominator_has_surviving_radical(given):
                    challenged.append(
                        {
                            "issue": "answer_text's denominator still carries a radical",
                            "text": text,
                            "answer_text": answer_text,
                        }
                    )
                    continue

                # Sanity check on the oracle's own recomputation, independent
                # of the generator: a rationalised denominator must itself be
                # a positive integer (no leftover radical, no sign flip).
                _, denom = sympy.fraction(sympy.nsimplify(expected))
                if not denom.is_Integer or denom <= 0:
                    challenged.append(
                        {
                            "issue": f"oracle's own rationalised denominator is not a positive integer: {denom}",
                            "text": text,
                        }
                    )

        assert not challenged, f"{len(challenged)} of {examined} challenged: {challenged}"
        print(f"rationalise-binomial-denominator oracle: examined {examined}, challenged 0")
