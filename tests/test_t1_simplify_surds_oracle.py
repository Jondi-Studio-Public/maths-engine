"""Independent oracle for `simplify-surds`, recomputing each tier's value with sympy.

Independent oracle for simplify-surds.

Parses each rendered question's `question.text` with a regex per tier shape
and recomputes the expected simplified value from those parsed numbers via
`sympy.sqrt`/`sympy.simplify` alone, checked against `answer_text`. Never
imports `mathgen.generators.t1_simplify_surds_build` -- `largest_square_factor`
is reimplemented from scratch here, independently.

Also re-derives the `simplified_surd` requirement independently from the
LEARNER's side: parses every `sqrt(...)`-shaped or bare-integer-coefficient
radical out of `answer_text` itself and confirms none of the parsed radicands
has a perfect-square factor > 1.
"""

import re

import sympy
from sympy import sqrt

from mathgen.skills import load_skills
from mathgen import make

_EASY_RE = re.compile(r"^Simplify √(\d+)\. Give your answer in the form a√b\.$")
_MEDIUM_RE = re.compile(r"^Simplify √(\d+) \+ √(\d+)\. Give your answer in the form a√b\.$")
_HARD_RE = re.compile(r"^Simplify (\d+)√(\d+) - (\d+)√(\d+)\. Give your answer in the form a√b\.$")

# Radicand under a "sqrt(...)" call in answer_text -- balanced-paren-free
# since a fully-simplified surd answer never nests parens inside the call.
_ANSWER_SQRT_RE = re.compile(r"sqrt\((\d+)\)")


def _largest_square_factor(n: int) -> tuple[int, int]:
    """Reimplemented from scratch (never imported from the generator): (k, m)
    with n == k*k*m and m square-free, by trial division."""
    k = 1
    m = n
    i = 2
    while i * i <= m:
        while m % (i * i) == 0:
            m //= i * i
            k *= i
        i += 1
    return k, m


def _answer_form_ok(answer_text: str) -> bool:
    """True if no sqrt(...) radicand in answer_text carries a perfect-square
    factor > 1 -- the learner-side re-derivation of `simplified_surd`."""
    for m in _ANSWER_SQRT_RE.finditer(answer_text):
        radicand = int(m.group(1))
        k, _ = _largest_square_factor(radicand)
        if k > 1:
            return False
    return True


class TestSimplifySurdsOracle:
    def test_oracle(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "simplify-surds")

        examined = 0
        challenged = []
        for source in skill.sources:
            for seed in range(60):
                q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **(source.params or {}))
                examined += 1
                text, answer_text = q.text, q.answer_text

                m_easy = _EASY_RE.match(text)
                m_medium = _MEDIUM_RE.match(text)
                m_hard = _HARD_RE.match(text)

                if m_easy:
                    n = int(m_easy.group(1))
                    expected = sqrt(n)
                elif m_medium:
                    r1, r2 = int(m_medium.group(1)), int(m_medium.group(2))
                    expected = sqrt(r1) + sqrt(r2)
                elif m_hard:
                    c1, r1, c2, r2 = (int(g) for g in m_hard.groups())
                    expected = c1 * sqrt(r1) - c2 * sqrt(r2)
                else:
                    challenged.append({"seed": seed, "issue": "unparsed question shape", "text": text})
                    continue

                given = sympy.sympify(answer_text.replace("^", "**"))
                if sympy.simplify(expected - given) != 0:
                    challenged.append(
                        {
                            "issue": f"oracle recomputed {expected}, answer_text stated {answer_text}",
                            "text": text,
                        }
                    )
                    continue

                if not _answer_form_ok(answer_text):
                    challenged.append(
                        {
                            "issue": "answer_text still carries an un-extracted perfect-square factor",
                            "text": text,
                            "answer_text": answer_text,
                        }
                    )

        assert not challenged, f"{len(challenged)} of {examined} challenged: {challenged}"
        print(f"simplify-surds oracle: examined {examined}, challenged 0")
