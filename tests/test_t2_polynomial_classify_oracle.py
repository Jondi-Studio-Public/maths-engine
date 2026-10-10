"""Independent oracle for classify-polynomials. Computational, not enumerative: the classification is fully
determined by the question text, so this re-derives it from scratch with a regex written from nothing shared
with the generator. Never imports `mathgen.generators.t2_polynomial_classify_build` -- not a helper, not a
constant, not a table.
"""

import re

from mathgen.skills import load_skills
from mathgen import make

_MODE_A_PREFIX = "Classify this polynomial by degree and number of terms: "
_MODE_B_PREFIX = "Is this a polynomial? If not, why not: "
_ANSWER_RE = re.compile(r"^Degree (-?\d+), (monomial|binomial|trinomial)$")
_TYPE_NAMES = {1: "monomial", 2: "binomial", 3: "trinomial"}

_CATEGORY_POLYNOMIAL = "Yes, it is a polynomial"
_CATEGORY_NEGATIVE_EXPONENT = "No - it has a negative exponent"
_CATEGORY_FRACTIONAL_EXPONENT = "No - it has a fractional exponent"
_CATEGORY_VARIABLE_DENOMINATOR = "No - the variable is in the denominator"

_X_RE = re.compile(r"x(?:\^(\d+))?")
_Y_RE = re.compile(r"y(?:\^(\d+))?")
_SPLIT_RE = re.compile(r" \+ | - ")


def _term_exponents(term: str) -> tuple[int, int]:
    """A term's own (x-exponent, y-exponent), read directly off its text
    (never off generator state): 'x' or absent x -> exponent 1 or 0,
    'x^p' -> p; same for y."""
    mx = _X_RE.search(term)
    my = _Y_RE.search(term)
    ex = int(mx.group(1)) if (mx and mx.group(1)) else (1 if mx else 0)
    ey = int(my.group(1)) if (my and my.group(1)) else (1 if my else 0)
    return ex, ey


def _split_terms(expr: str) -> list[str]:
    """Split a join_terms-built sum on its own " + "/" - " joins. A leading
    unary minus (no separating space, e.g. "-x^2 + 5") stays attached to the
    first segment; it is irrelevant here since only exponents/term-count are
    read, never a term's sign."""
    return [t for t in _SPLIT_RE.split(expr) if t]


def _mode_a_expected(expr: str) -> tuple[int, str]:
    terms = _split_terms(expr)
    exponent_pairs = [_term_exponents(t) for t in terms]
    degree = max(ex + ey for ex, ey in exponent_pairs)
    term_count = len(terms)
    return degree, _TYPE_NAMES[term_count]


def _mode_b_expected(expr: str) -> str:
    """Priority order: a negative exponent ('^-') beats
    a parenthesised fractional exponent ('^(') beats the denominator check
    (a '/' immediately followed by a letter is a variable denominator; a
    '/' followed by a letter then a digit -- i.e. landing inside a term
    like 'x/2' -- is a constant denominator, still a polynomial)."""
    if "^-" in expr:
        return _CATEGORY_NEGATIVE_EXPONENT
    if "^(" in expr:
        return _CATEGORY_FRACTIONAL_EXPONENT
    for m in re.finditer(r"/(.)", expr):
        if m.group(1).isalpha():
            return _CATEGORY_VARIABLE_DENOMINATOR
    return _CATEGORY_POLYNOMIAL


class TestClassifyPolynomialsOracle:
    def test_oracle(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "classify-polynomials")

        examined = 0
        challenged = []
        for source in skill.sources:
            for seed in range(80):
                q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **(source.params or {}))
                examined += 1
                text, answer_text = q.text, q.answer_text
                options = [answer_text] + list(q.distractors)

                if text.startswith(_MODE_A_PREFIX):
                    expr = text[len(_MODE_A_PREFIX) :]
                    expected_degree, expected_type = _mode_a_expected(expr)
                    m = _ANSWER_RE.match(answer_text)
                    if not m:
                        challenged.append(
                            {
                                "issue": "answer_text not 'Degree N, type' shape",
                                "text": text,
                                "answer_text": answer_text,
                            }
                        )
                        continue
                    stated_degree, stated_type = int(m.group(1)), m.group(2)
                    if (stated_degree, stated_type) != (expected_degree, expected_type):
                        challenged.append(
                            {
                                "issue": f"oracle derived Degree {expected_degree}, {expected_type}; "
                                f"answer_text stated {answer_text!r}",
                                "text": text,
                            }
                        )
                        continue
                    expected_text = f"Degree {expected_degree}, {expected_type}"
                    if expected_text in q.distractors:
                        challenged.append(
                            {
                                "issue": f"the recomputed correct answer {expected_text!r} is also offered as a distractor",
                                "text": text,
                                "distractors": q.distractors,
                            }
                        )
                    continue

                if text.startswith(_MODE_B_PREFIX):
                    expr = text[len(_MODE_B_PREFIX) :]
                    expected_category = _mode_b_expected(expr)
                    if answer_text != expected_category:
                        challenged.append(
                            {
                                "issue": f"oracle derived {expected_category!r}; answer_text stated {answer_text!r}",
                                "text": text,
                            }
                        )
                        continue
                    if expected_category in q.distractors:
                        challenged.append(
                            {
                                "issue": f"the recomputed correct answer {expected_category!r} is also offered as a distractor",
                                "text": text,
                                "distractors": q.distractors,
                            }
                        )
                    continue

                challenged.append({"seed": seed, "issue": "unparsed question shape", "text": text})

        assert not challenged, f"{len(challenged)} of {examined} challenged: {challenged}"
        print(f"classify-polynomials oracle: examined {examined}, challenged 0")
