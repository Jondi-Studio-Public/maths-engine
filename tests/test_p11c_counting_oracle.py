"""Reads only `question.text` / `question.answer_text` / `question.distractors`
via `mathgen.make`. Imports NOTHING from
`mathgen.generators.p11c_counting_build` -- every value is parsed back out of
the rendered prose with a regular expression and re-derived from first
principles: `math.factorial` / a manual term-by-term product for
`p11c_factorial_notation`, a fresh `itertools.permutations` brute-force
enumeration (never a formula) for `p11c_restricted_permutations`.
"""

import re
from itertools import permutations
from math import factorial

import pytest
import sympy

from mathgen import make

SEEDS = range(60)
TIERS = ("easy", "medium", "hard")

FACT_TOPIC = "p11c_factorial_notation"
RESTR_TOPIC = "p11c_restricted_permutations"

_n = sympy.symbols("n")


# ============================================================================
# p11c_factorial_notation
# ============================================================================

_EASY_RE = re.compile(r"^Evaluate (?P<a>\d+)! (?P<op>[+×−]) (?P<b>\d+)! as a single number\.$")
# The medium tier gained 2 more instruction-verb templates ("Write... as",
# "Evaluate... as") alongside the original "Simplify... to" -- same question type, different wording, so the
# oracle regex accepts the verb/preposition pairing loosely rather than needing the original three
# regexes.
_MEDIUM_RE = re.compile(r"^(?:Simplify|Write|Evaluate) (?P<n>\d+)!/(?P<m>\d+)! (?:to|as) a single number\.$")
# The hard tier gained a THIRD shape -- the numerator can now be bare "n!" too (the
# ratio inverts to a reciprocal), so both sides of the ratio, not just the denominator, must accept
# either "n!" or "(n + digits)!".
_HARD_RE = re.compile(
    r"^Simplify (?P<num>n!|\(n \+ \d+\)!)/(?P<den>n!|\(n \+ \d+\)!) as an expression in n, "
    r"where n is a positive integer\.$"
)
_SHIFT_RE = re.compile(r"\(n \+ (\d+)\)!")


def _shift_of(fact_text: str) -> int:
    return 0 if fact_text == "n!" else int(_SHIFT_RE.match(fact_text).group(1))


def _oracle_factorial(question):
    text = question.text
    m = _EASY_RE.match(text)
    if m:
        a, b, op = int(m.group("a")), int(m.group("b")), m.group("op")
        # independent route: a fresh manual term-by-term product for each
        # operand (never math.factorial reused as the only route), combined
        # with the operator parsed straight out of the rendered text.
        fa = 1
        for i in range(1, a + 1):
            fa *= i
        fb = 1
        for i in range(1, b + 1):
            fb *= i
        if op == "+":
            return fa + fb
        if op == "−":
            return fa - fb
        return fa * fb
    m = _MEDIUM_RE.match(text)
    if m:
        n, denom_n = int(m.group("n")), int(m.group("m"))
        return factorial(n) // factorial(denom_n)
    m = _HARD_RE.match(text)
    if m:
        num_shift = _shift_of(m.group("num"))
        den_shift = _shift_of(m.group("den"))
        # sympy builds whichever direction the parsed text actually states
        # -- a product when num_shift > den_shift, a reciprocal when
        # num_shift < den_shift -- with no special-casing needed here: the
        # ratio IS the ratio either way.
        return sympy.factorial(_n + num_shift) / sympy.factorial(_n + den_shift)
    raise AssertionError(f"could not parse question text: {text!r}")


@pytest.mark.parametrize("tier", TIERS)
def test_factorial_notation_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(FACT_TOPIC, seed=seed, difficulty=tier)
        expected = _oracle_factorial(q)
        if tier == "hard":
            rendered = sympy.sympify(q.answer_text)
            assert sympy.simplify(expected - rendered) == 0, (
                f"seed={seed}/hard: oracle {expected} != rendered {rendered}\ntext: {q.text}"
            )
        else:
            assert int(q.answer_text) == expected, (
                f"seed={seed}/{tier}: oracle {expected} != rendered {q.answer_text}\ntext: {q.text}"
            )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_factorial_notation_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(FACT_TOPIC, seed=seed, difficulty=tier)
        expected = _oracle_factorial(q)
        for d in q.distractors:
            candidate = sympy.sympify(d)
            if tier == "hard":
                assert sympy.simplify(expected - candidate) != 0, (
                    f"seed={seed}/hard: distractor {d!r} equals the oracle's own answer"
                )
            else:
                assert int(candidate) != expected, (
                    f"seed={seed}/{tier}: distractor {d!r} equals the oracle's own answer"
                )


# ============================================================================
# p11c_restricted_permutations
# ============================================================================

# Easy/medium now also draw a block of THREE specified objects (k=3), not only the
# original pair, and the hard tier's identical-object count and together/apart constraint both vary
# -- see the generator module's own docstring for the combo pool. Four regexes replace the original
# pair-only pair, plus a single hard-tier regex that captures the repeat count and the
# together/NOT-apart direction instead of assuming both.
_TOGETHER_PAIR_RE = re.compile(
    r"^(?P<n>\d+) different (?P<noun>\w+) are placed in a row\. "
    r"(?P<p>[A-Z]) and (?P<q>[A-Z]) must be next to each other\. "
    r"In how many ways can the (?P=noun) be arranged\?$"
)
_APART_PAIR_RE = re.compile(
    r"^(?P<n>\d+) different (?P<noun>\w+) are placed in a row\. "
    r"(?P<p>[A-Z]) and (?P<q>[A-Z]) must NOT be next to each other\. "
    r"In how many ways can the (?P=noun) be arranged\?$"
)
_TOGETHER_TRIPLE_RE = re.compile(
    r"^(?P<n>\d+) different (?P<noun>\w+) are placed in a row\. "
    r"(?P<p>[A-Z]), (?P<q>[A-Z]) and (?P<r>[A-Z]) must all be next to each other, in any order\. "
    r"In how many ways can the (?P=noun) be arranged\?$"
)
_APART_TRIPLE_RE = re.compile(
    r"^(?P<n>\d+) different (?P<noun>\w+) are placed in a row\. "
    r"(?P<p>[A-Z]), (?P<q>[A-Z]) and (?P<r>[A-Z]) must NOT all be next to each other, in any order\. "
    r"In how many ways can the (?P=noun) be arranged\?$"
)
_HARD_RESTR_RE = re.compile(
    r"^(?P<n>\d+) objects are arranged in a row: (?P<r>\d+) of them are labelled (?P<x>[A-Z]) "
    r"\(identical\), and the rest are labelled (?P<rest>[A-Z](?:, [A-Z])*), "
    r"(?P<p>[A-Z]) and (?P<q>[A-Z]) \(all distinct\)\. (?P=p) must (?P<neg>NOT )?be next to (?P=q)\. "
    r"In how many distinct ways can the objects be arranged\?$"
)


def _contiguous(perm, members) -> bool:
    """True if every symbol in `members` occupies a contiguous run of
    positions in `perm` -- re-implemented here independently of the
    generator's own `_is_contiguous_block` (this oracle imports nothing from
    the generator module)."""
    positions = sorted(perm.index(c) for c in members)
    return positions[-1] - positions[0] == len(members) - 1


def _oracle_restricted(question):
    """Brute-force enumeration from the rendered text alone -- never the
    generator's own block-counting / total-minus-together formulas."""
    text = question.text
    m = _TOGETHER_PAIR_RE.match(text)
    if m:
        n = int(m.group("n"))
        members = [m.group("p"), m.group("q")]
        objs = [chr(ord("A") + i) for i in range(n)]
        return sum(1 for perm in permutations(objs) if _contiguous(perm, members))
    m = _APART_PAIR_RE.match(text)
    if m:
        n = int(m.group("n"))
        members = [m.group("p"), m.group("q")]
        objs = [chr(ord("A") + i) for i in range(n)]
        return sum(1 for perm in permutations(objs) if not _contiguous(perm, members))
    m = _TOGETHER_TRIPLE_RE.match(text)
    if m:
        n = int(m.group("n"))
        members = [m.group("p"), m.group("q"), m.group("r")]
        objs = [chr(ord("A") + i) for i in range(n)]
        return sum(1 for perm in permutations(objs) if _contiguous(perm, members))
    m = _APART_TRIPLE_RE.match(text)
    if m:
        n = int(m.group("n"))
        members = [m.group("p"), m.group("q"), m.group("r")]
        objs = [chr(ord("A") + i) for i in range(n)]
        return sum(1 for perm in permutations(objs) if not _contiguous(perm, members))
    m = _HARD_RESTR_RE.match(text)
    if m:
        n, r = int(m.group("n")), int(m.group("r"))
        x = m.group("x")
        rest = m.group("rest").split(", ")
        p, q = m.group("p"), m.group("q")
        multiset = tuple([x] * r + rest + [p, q])
        assert len(multiset) == n, f"multiset size {len(multiset)} != n {n}: {text!r}"
        negated = m.group("neg") is not None
        if negated:
            return sum(1 for perm in set(permutations(multiset)) if not _contiguous(perm, (p, q)))
        return sum(1 for perm in set(permutations(multiset)) if _contiguous(perm, (p, q)))
    raise AssertionError(f"could not parse question text: {text!r}")


@pytest.mark.parametrize("tier", TIERS)
def test_restricted_permutations_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(RESTR_TOPIC, seed=seed, difficulty=tier)
        expected = _oracle_restricted(q)
        assert int(q.answer_text) == expected, (
            f"seed={seed}/{tier}: oracle {expected} != rendered {q.answer_text}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_restricted_permutations_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(RESTR_TOPIC, seed=seed, difficulty=tier)
        expected = _oracle_restricted(q)
        for d in q.distractors:
            assert int(d) != expected, f"seed={seed}/{tier}: distractor {d!r} equals the oracle's own answer"
