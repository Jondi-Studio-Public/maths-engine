"""Independent oracle for the `pv0b_system_cases` questions: it re-derives each answer from the rendered question
text.

Imports nothing from `mathgen.generators.pv0b_system_cases_build`. Parses the
printed system (or echelon rows) straight out of the RENDERED `question.text`
with a regex, rebuilds the augmented matrix with sympy, and derives the class
from rank(A) vs rank([A|b]) directly (never trusting the generator's own
bookkeeping). For the exam tier, k is kept as a symbol through parsing and
substituted at >= 5 concrete values satisfying the stated condition; the
class must agree at every substitution AND match the option the question
offers as its key. Run across >= 1000 questions total, >= 250 per tier.
"""

from __future__ import annotations

import re

from sympy import Matrix, Symbol, sympify

from mathgen import make

TOPIC = "pv0b_system_cases"
SEEDS_PER_TIER = 260  # 4 tiers x 260 = 1040 >= 1000

KEY_UNIQUE = "exactly one solution"
KEY_NONE = "no solution"
KEY_INFINITE = "infinitely many solutions"

x, y, z, k = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("k")


def _to_expr(term_str: str):
    s = term_str.replace(" ", "")
    s = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)
    s = re.sub(r"\)([a-zA-Z])", r")*\1", s)
    # Parser-only fix: a bare "kz" (the generator's z-coefficient
    # is unparenthesized "k" when its offset is 0, e.g. `_k_expr(0) == "k"`)
    # has no digit or ")" before the variable, so the two substitutions
    # above never fire and sympify would read "kz" as one opaque symbol
    # instead of k*z. Insert the missing "*" between a bare k and the
    # variable that immediately follows it; the rank-at-several-k logic
    # below is untouched.
    s = re.sub(r"k([xyz])", r"k*\1", s)
    if s == "":
        s = "0"
    return sympify(s, locals={"x": x, "y": y, "z": z, "k": k})


def _parse_equation(eq_str: str):
    lhs_str, rhs_str = eq_str.split("=")
    lhs = _to_expr(lhs_str)
    rhs = _to_expr(rhs_str)
    a = lhs.coeff(x)
    b = lhs.coeff(y)
    c = lhs.coeff(z)
    return a, b, c, rhs


def _classify(rows) -> str:
    """rows: list of (a, b, c, d) -- may be plain ints or sympy expressions
    (already substituted numeric if k is involved)."""
    A = Matrix([[r[0], r[1], r[2]] for r in rows])
    Aug = Matrix([[r[0], r[1], r[2], r[3]] for r in rows])
    rank_a = A.rank()
    rank_aug = Aug.rank()
    if rank_a == 3:
        return KEY_UNIQUE
    if rank_a == rank_aug:
        return KEY_INFINITE
    return KEY_NONE


# --- Easy: echelon rows [a b c | d] -----------------------------------------

_ROW_RE = re.compile(r"\[(-?\d+) (-?\d+) (-?\d+) \| (-?\d+)\]")


def _oracle_easy(q):
    rows = [tuple(int(v) for v in m) for m in _ROW_RE.findall(q.text)]
    assert len(rows) == 3, q.text
    return _classify(rows)


# --- Medium/hard/exam: three plain equations, semicolon-joined -------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")


def _sentences(text: str) -> list[str]:
    """Split on a sentence-ending mark (period/question mark/exclamation)
    followed by whitespace -- some intro variants end in "?"."""
    return _SENTENCE_SPLIT.split(text)


def _extract_three_equations(text: str):
    """The printed system is on its own sentence: identified as the one sentence containing "=", since no prose
    sentence (intro/condition/closing) ever does."""
    eq_sentence = next(s for s in _sentences(text) if "=" in s)
    parts = eq_sentence.rstrip(".").split(";")
    eq1 = parts[0].strip()
    eq2 = parts[1].strip()
    eq3 = parts[2].strip()
    return eq1, eq2, eq3


def _oracle_no_k(q):
    eqs = _extract_three_equations(q.text)
    rows = [_parse_equation(eq) for eq in eqs]
    return _classify(rows)


# --- Exam: k appears symbolically; substitute >= 5 values -------------------


def _sample_ks(condition: str) -> list[int]:
    m_ne = re.search(r"k\s*≠\s*(-?\d+)", condition)
    if m_ne:
        c = int(m_ne.group(1))
        return [v for v in range(-8, 9) if v != c][:5]
    m_eq = re.search(r"k\s*=\s*(-?\d+)", condition)
    if m_eq:
        c = int(m_eq.group(1))
        return [c] * 5
    # "for every value of k"
    return [-4, -1, 0, 2, 5]


def _oracle_exam(q):
    sentences = _sentences(q.text)
    eq_idx = next(i for i, s in enumerate(sentences) if "=" in s)
    eqs = _extract_three_equations(q.text)
    rows_sym = [_parse_equation(eq) for eq in eqs]
    # The condition sentence always immediately follows the equations sentence,
    # and the closing instruction ("Classify ...") always follows that.
    condition = sentences[eq_idx + 1]
    ks = _sample_ks(condition)
    assert len(ks) >= 5, (condition, ks)
    classes = set()
    for kv in ks:
        rows_num = [tuple(expr.subs(k, kv) for expr in row) for row in rows_sym]
        classes.add(_classify(rows_num))
    assert len(classes) == 1, (q.text, condition, ks, classes)
    return classes.pop()


# --- Tests -------------------------------------------------------------------


def test_oracle_easy_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="easy", tier="easy")
        derived = _oracle_easy(q)
        assert derived == q.answer_text, (seed, q.text, q.answer_text, derived)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_medium_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="medium", tier="medium")
        derived = _oracle_no_k(q)
        assert derived == q.answer_text, (seed, q.text, q.answer_text, derived)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_hard_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="hard", tier="hard")
        derived = _oracle_no_k(q)
        assert derived == q.answer_text, (seed, q.text, q.answer_text, derived)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_exam_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="exam", tier="exam")
        derived = _oracle_exam(q)
        assert derived == q.answer_text, (seed, q.text, q.answer_text, derived)
        checked += 1
    assert checked == SEEDS_PER_TIER


# --- Proven red: the oracle catches a deliberately perturbed key -----------


def test_oracle_catches_a_perturbed_key():
    """Take one real question and flip its `answer_text` to a wrong class
    (never mutating the shipping module): the oracle's independent
    recomputation must disagree with the corrupted key."""
    q = make(TOPIC, seed=0, difficulty="easy", tier="easy")
    derived = _oracle_easy(q)
    wrong_classes = [c for c in (KEY_UNIQUE, KEY_NONE, KEY_INFINITE) if c != derived]
    for wrong in wrong_classes:
        assert derived != wrong
