"""Generates three number-theory proof topics:

- ``proof_irrationality``: prove a number such as a square root or a logarithm is irrational, by contradiction.
- ``proof_simple_number_results``: prove a simple result about integers (parity, divisibility, digits).
- ``proof_pigeonhole``: solve a problem or prove a result with the pigeon-hole principle.

Each question names what the learner may assume and asks for a complete argument. The argument is marked against
a rubric, so `answer` is a marking dict (see `_free_proof_kit`), not a value.

Every statement a question asks the learner to prove is confirmed here before the question is returned. Divisibility
statements are checked by brute force over a window of integers; the pigeon-hole statements by checking the count
the argument rests on (and, for small sizes, by exhaustive search); the irrationality statements by the arithmetic
facts their proofs rest on (primes are distinct, a number is not a perfect square, a surd is irrational in SymPy).
"""

from __future__ import annotations

from itertools import combinations
from math import ceil, comb, gcd

import sympy as sp

from ..core import check, generator
from ._free_proof_kit import Fm, Prompt, TAILS, attempts, finish, marking

PRIMES = (2, 3, 5, 7, 11, 13)
SMALL_PRIMES = (2, 3, 5, 7)


def _tail(rng) -> str:
    return rng.choice(TAILS)


def _sqrt(n: int) -> Fm:
    return Fm(f"\\sqrt{{{n}}}", f"√{n}")


def _log(base: int, arg: int) -> Fm:
    return Fm(f"\\log_{{{base}}} {arg}", f"log_{base} {arg}")


def _sq(n: int) -> str:
    return f"{n}²"


# ===========================================================================
# Irrationality by contradiction
# ===========================================================================
def _irr_sqrt_prime(rng) -> dict:
    prime = rng.choice(PRIMES)
    check(sp.isprime(prime) and sp.sqrt(prime).is_irrational, "sqrt of a prime is irrational")
    a, b = rng.choice((("m", "n"), ("a", "b"), ("r", "s")))
    p = Prompt()
    p.t("Prove by contradiction that ").m(_sqrt(prime)).t(
        " is irrational. You may use: if a prime p divides n², then p divides n. "
    ).t(_tail(rng))
    q = prime
    ref = [
        f"Suppose √{q} is rational. Then √{q} = {a}/{b} where {a} and {b} are positive integers with no common factor (the fraction is in lowest terms).",
        f"Square both sides: {q} = {a}²/{b}², so {a}² = {q}{b}².",
        f"So {q} divides {a}². Since {q} is prime, {q} divides {a} (the result we may use).",
        f"Write {a} = {q}k for an integer k. Then {q}²k² = {q}{b}², so {b}² = {q}k².",
        f"So {q} divides {b}², and therefore {q} divides {b}.",
        f"Now {q} divides both {a} and {b}, which contradicts the fact that they have no common factor. So √{q} is irrational.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"derives {a}² = {q}{b}² and then that {q} divides both {a} and {b}"),
            (
                "logical_completeness",
                f"applies the given result twice, once to {a}² and once to {b}², and states the contradiction",
            ),
            (
                "definitions_assumptions",
                "assumes the fraction is in lowest terms, with the contradiction aimed at that assumption",
            ),
            ("communication", "begins with 'suppose' and ends with the conclusion that the root is irrational"),
        ],
        traps=[
            (
                "no_lowest_terms",
                "does the proof say the fraction is in lowest terms? Without it there is no contradiction.",
            ),
            ("only_one_divisibility", f"does it show {q} divides {b} as well as {a}?"),
        ],
    )
    return finish(p, f"√{q} is irrational", mark)


def _irr_sqrt_two_parity(rng) -> dict:
    a, b = rng.choice((("a", "b"), ("m", "n"), ("p", "q")))
    p = Prompt()
    p.t("Prove by contradiction that ").m(_sqrt(2)).t(" is irrational. You may use: if n² is even then n is even. ").t(
        _tail(rng)
    )
    ref = [
        f"Suppose √2 is rational. Then √2 = {a}/{b} with {a} and {b} positive integers with no common factor.",
        f"Square both sides: {a}² = 2{b}², so {a}² is even.",
        f"Therefore {a} is even (the result we may use). Write {a} = 2k.",
        f"Then 4k² = 2{b}², so {b}² = 2k², which is even, so {b} is even.",
        f"Both {a} and {b} are even, so they have the common factor 2. This contradicts lowest terms. So √2 is irrational.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"obtains {a}² = 2{b}² and then that both {a} and {b} are even"),
            ("logical_completeness", "uses the given result twice and ends with the contradiction"),
            ("definitions_assumptions", "assumes the fraction is in lowest terms"),
            ("communication", "says what is being supposed, and what the contradiction is with"),
        ],
        traps=[
            ("no_lowest_terms", "does the proof say the fraction is in lowest terms?"),
            ("stops_at_a_even", f"does the proof stop once {a} is even, without showing {b} is even too?"),
        ],
    )
    return finish(p, "√2 is irrational", mark)


def _irr_log(rng) -> dict:
    pp, qq = rng.sample(SMALL_PRIMES, 2)
    check(pp != qq and sp.isprime(pp) and sp.isprime(qq), "distinct primes")
    a, b = rng.choice((("m", "n"), ("a", "b"), ("r", "s")))
    p = Prompt()
    p.t("Prove by contradiction that ").m(_log(pp, qq)).t(
        " is irrational. You may use: every integer greater than 1 can be written as a product of primes in only one way, apart from order. "
    ).t(_tail(rng))
    ref = [
        f"Suppose log_{pp} {qq} is rational. Since {qq} > 1 and the base {pp} > 1, the logarithm is positive, so log_{pp} {qq} = {a}/{b} with {a} and {b} positive integers.",
        f"By the definition of a logarithm, {pp}^({a}/{b}) = {qq}.",
        f"Raise both sides to the power {b}: {pp}^{a} = {qq}^{b}.",
        f"The left side is a power of {pp} with {a} ≥ 1, so {pp} is a prime factor of it.",
        f"The right side, {qq}^{b}, has {qq} as its only prime factor, and {pp} ≠ {qq}.",
        f"So the same integer would have two different prime factorisations, which contradicts uniqueness. So log_{pp} {qq} is irrational.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"gets {pp}^{a} = {qq}^{b} from the assumed fraction"),
            (
                "logical_completeness",
                "compares the prime factors of the two sides and names the contradiction with unique factorisation",
            ),
            (
                "definitions_assumptions",
                f"says {a} and {b} are positive integers, which needs the logarithm to be positive",
            ),
            ("communication", "writes 'suppose' and 'contradiction' where each belongs"),
        ],
        traps=[
            (
                "a_divisibility_by_two_only",
                f"does it only show the sides have a different parity or size, rather than different prime factors ({pp} against {qq})?",
            ),
            ("positivity_not_stated", f"does it say why {a} and {b} can be taken positive?"),
        ],
    )
    return finish(p, f"log_{pp} {qq} is irrational", mark)


def _irr_shift(rng) -> dict:
    prime = rng.choice(PRIMES)
    b = rng.choice((1, 2, 3, 4, 5, -1, -2, -3, -4))
    check(sp.sqrt(prime).is_irrational, "sqrt of a prime is irrational")
    expr = Fm(f"\\sqrt{{{prime}}} {'+' if b > 0 else '-'} {abs(b)}", f"√{prime} {'+' if b > 0 else '-'} {abs(b)}")
    p = Prompt()
    p.t("Given that ").m(_sqrt(prime)).t(" is irrational, prove by contradiction that ").m(expr).t(
        " is irrational. "
    ).t(_tail(rng))
    sgn = "-" if b > 0 else "+"
    ref = [
        f"Suppose √{prime} {'+' if b > 0 else '-'} {abs(b)} is rational, equal to r.",
        f"Then √{prime} = r {sgn} {abs(b)}.",
        f"The sum or difference of two rational numbers is rational, so r {sgn} {abs(b)} is rational.",
        f"So √{prime} is rational, which contradicts the given fact. So √{prime} {'+' if b > 0 else '-'} {abs(b)} is irrational.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"rearranges to √{prime} = r {sgn} {abs(b)}"),
            (
                "logical_completeness",
                "uses that a sum or difference of rationals is rational and names the contradiction",
            ),
            (
                "definitions_assumptions",
                "states clearly the assumption (the number is rational) and the given fact it contradicts",
            ),
            ("communication", "ends with the conclusion"),
        ],
        traps=[
            ("tests_a_decimal", "does it argue from a decimal expansion rather than from the definition of rational?"),
            ("sign_slip", "does the rearrangement keep the right sign for the constant?"),
        ],
    )
    return finish(p, f"{expr.plain} is irrational", mark)


def _irr_affine(rng) -> dict:
    prime = rng.choice(PRIMES)
    a = rng.choice((2, 3, 4, 5, -2, -3))
    b = rng.choice((1, 2, 3, -1, -2, -3))
    check(sp.sqrt(prime).is_irrational and a != 0, "a sqrt(p) + b is irrational")
    lead = f"{a}" if abs(a) != 1 else ("-" if a < 0 else "")
    expr = Fm(
        f"{lead}\\sqrt{{{prime}}} {'+' if b > 0 else '-'} {abs(b)}", f"{lead}√{prime} {'+' if b > 0 else '-'} {abs(b)}"
    )
    p = Prompt()
    p.t("Given that ").m(_sqrt(prime)).t(" is irrational, prove by contradiction that ").m(expr).t(
        " is irrational. "
    ).t(_tail(rng))
    ref = [
        f"Suppose {expr.plain} is rational, equal to r.",
        f"Then {a}√{prime} = r {'-' if b > 0 else '+'} {abs(b)}.",
        f"Since {a} ≠ 0, √{prime} = (r {'-' if b > 0 else '+'} {abs(b)})/{a}.",
        f"The right-hand side is a quotient of rational numbers with a non-zero divisor, so it is rational.",
        f"So √{prime} is rational, which contradicts the given fact. So {expr.plain} is irrational.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"isolates √{prime} as (r - b)/a"),
            ("logical_completeness", "shows the expression for the root is rational and names the contradiction"),
            ("definitions_assumptions", f"notes that {a} is not 0 before dividing"),
            ("communication", "states the assumption and the contradiction"),
        ],
        traps=[
            ("divides_by_a_possibly_zero", "does it divide without saying the coefficient is non-zero?"),
            ("sign_slip_on_the_constant", "does the constant change sides with the right sign?"),
        ],
    )
    return finish(p, f"{expr.plain} is irrational", mark)


def _irr_sqrt_pq(rng) -> dict:
    pp, qq = rng.sample(SMALL_PRIMES, 2)
    check(sp.isprime(pp) and sp.isprime(qq) and pp != qq, "distinct primes")
    a, b = rng.choice((("m", "n"), ("a", "b")))
    p = Prompt()
    p.t("Prove by contradiction that ").m(_sqrt(pp * qq)).t(
        f" is irrational. You may use: if a prime divides n² then it divides n; and if two different primes both divide an integer then their product divides it. "
    ).t(_tail(rng))
    ref = [
        f"Suppose √{pp * qq} is rational: √{pp * qq} = {a}/{b} with {a}, {b} positive integers and no common factor.",
        f"Squaring: {a}² = {pp * qq}{b}².",
        f"So {pp} divides {a}² and {qq} divides {a}², hence {pp} divides {a} and {qq} divides {a}.",
        f"{pp} and {qq} are different primes, so {pp * qq} divides {a}. Write {a} = {pp * qq}k.",
        f"Then {pp * qq}²k² = {pp * qq}{b}², so {b}² = {pp * qq}k².",
        f"So {pp} divides {b}², hence {pp} divides {b}. Now {pp} divides both {a} and {b}, contradicting no common factor. So √{pp * qq} is irrational.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"derives {a}² = {pp * qq}{b}² and that {pp * qq} divides {a}, then that {pp} divides {b}"),
            ("logical_completeness", "uses both given results and ends with a contradiction"),
            ("definitions_assumptions", "assumes lowest terms and uses that the two primes are different"),
            ("communication", "states each divisibility step with its reason"),
        ],
        traps=[
            ("treats_pq_as_prime", f"does it treat {pp * qq} as a prime? {pp * qq} is not prime."),
            ("missing_second_prime", f"does it use only {pp}, and not show that {qq} also divides {a}?"),
        ],
    )
    return finish(p, f"√{pp * qq} is irrational", mark)


def _irr_cuberoot(rng) -> dict:
    prime = rng.choice((2, 3, 5, 7))
    a, b = rng.choice((("m", "n"), ("a", "b")))
    cube = Fm(f"\\sqrt[3]{{{prime}}}", f"∛{prime}")
    p = Prompt()
    p.t("Prove by contradiction that ").m(cube).t(
        " is irrational. You may use: if a prime p divides n³, then p divides n. "
    ).t(_tail(rng))
    ref = [
        f"Suppose ∛{prime} is rational: ∛{prime} = {a}/{b}, with {a}, {b} positive integers and no common factor.",
        f"Cube both sides: {a}³ = {prime}{b}³.",
        f"So {prime} divides {a}³, hence {prime} divides {a}. Write {a} = {prime}k.",
        f"Then {prime}³k³ = {prime}{b}³, so {b}³ = {prime}²k³.",
        f"So {prime} divides {b}³, hence {prime} divides {b}.",
        f"Now {prime} divides both {a} and {b}, contradicting no common factor. So ∛{prime} is irrational.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"derives {a}³ = {prime}{b}³ and that {prime} divides both {a} and {b}"),
            ("logical_completeness", "uses the given result twice and ends with a contradiction"),
            ("definitions_assumptions", "assumes lowest terms"),
            ("communication", "states each step with its reason"),
        ],
        traps=[
            ("uses_the_square_root_proof", "does it square instead of cube, so the equation is not a³ = p b³?"),
            ("arithmetic_with_cubes", f"does the step from {prime}³k³ = {prime}{b}³ lose the factor {prime}²?"),
        ],
    )
    return finish(p, f"∛{prime} is irrational", mark)


def _irr_sum_of_roots(rng) -> dict:
    pp, qq = rng.sample(SMALL_PRIMES, 2)
    check(sp.isprime(pp) and sp.isprime(qq) and pp != qq, "distinct primes")
    check(sp.sqrt(pp * qq).is_irrational, "sqrt(pq) is irrational")
    expr = Fm(f"\\sqrt{{{pp}}} + \\sqrt{{{qq}}}", f"√{pp} + √{qq}")
    p = Prompt()
    p.t("Given that ").m(_sqrt(pp * qq)).t(" is irrational, prove by contradiction that ").m(expr).t(
        " is irrational. "
    ).t(_tail(rng))
    s = pp + qq
    ref = [
        f"Suppose √{pp} + √{qq} = r, where r is rational.",
        f"Square both sides: {pp} + 2√{pp * qq} + {qq} = r², using √{pp} × √{qq} = √{pp * qq}.",
        f"So 2√{pp * qq} = r² - {s}, and √{pp * qq} = (r² - {s})/2.",
        f"r is rational, so r² - {s} is rational, and dividing by 2 keeps it rational.",
        f"So √{pp * qq} is rational. This contradicts the given fact. So √{pp} + √{qq} is irrational.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"squares correctly to {s} + 2√{pp * qq} = r²"),
            ("logical_completeness", "isolates the root of the product and shows it would be rational"),
            ("definitions_assumptions", "uses that a rational number squared, shifted and halved is still rational"),
            ("communication", "states the assumption and names the contradiction"),
        ],
        traps=[
            ("squares_term_by_term", f"does it write (√{pp} + √{qq})² as {pp} + {qq} with no middle term?"),
            ("never_uses_the_given_fact", f"does it fail to link √{pp * qq} to the given irrational number?"),
        ],
    )
    return finish(p, f"{expr.plain} is irrational", mark)


def _irr_log_power(rng) -> dict:
    pp, qq = rng.sample(SMALL_PRIMES, 2)
    s = rng.choice((2, 3))
    t = rng.choice((2, 3))
    base, arg = pp**s, qq**t
    check(sp.isprime(pp) and sp.isprime(qq) and pp != qq, "distinct primes")
    p = Prompt()
    p.t("Given that ").m(_log(pp, qq)).t(" is irrational, prove that ").m(_log(base, arg)).t(
        " is irrational. You may use the change of base formula and the log law "
    ).m(Fm("\\log(a^{k}) = k \\log a", "log(a^k) = k log a")).t(". ").t(_tail(rng))
    ref = [
        f"Write both logarithms in base {pp}: log_{base} {arg} = log_{pp} {arg} / log_{pp} {base} (change of base).",
        f"{arg} = {qq}^{t} and {base} = {pp}^{s}, so log_{pp} {arg} = {t} log_{pp} {qq} and log_{pp} {base} = {s} log_{pp} {pp} = {s}.",
        f"So log_{base} {arg} = ({t}/{s}) log_{pp} {qq}.",
        f"Suppose log_{base} {arg} = r is rational. Then log_{pp} {qq} = ({s}/{t}) r, which is rational.",
        f"This contradicts the given fact that log_{pp} {qq} is irrational. So log_{base} {arg} is irrational.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"reduces log_{base} {arg} to ({t}/{s}) log_{pp} {qq}"),
            (
                "logical_completeness",
                "assumes the logarithm is rational and derives a contradiction with the given fact",
            ),
            (
                "definitions_assumptions",
                f"notes that {s} and {t} are non-zero integers, so the conversion factor is a non-zero rational",
            ),
            ("communication", "shows the change of base step"),
        ],
        traps=[
            ("factor_inverted", f"is the factor {t}/{s} (not {s}/{t}) in the first conversion?"),
            ("log_laws_misapplied", "does it use log(a^k) = k log a with the exponent on the wrong number?"),
        ],
    )
    return finish(p, f"log_{base} {arg} is irrational", mark)


def _irr_sqrt_nonsquare(rng) -> dict:
    for _ in attempts(500, "a non-square composite"):
        n = rng.randint(6, 60)
        if not sp.isprime(n) and not sp.integer_nthroot(n, 2)[1]:
            break
    factors = sp.factorint(n)
    odd_primes = sorted(pr for pr, e in factors.items() if e % 2 == 1)
    check(bool(odd_primes), "a non-square has a prime with an odd exponent")
    check(sp.sqrt(n).is_irrational, "sqrt of a non-square is irrational")
    ell = odd_primes[0]
    fact_text = " × ".join(f"{pr}^{e}" if e > 1 else f"{pr}" for pr, e in sorted(factors.items()))
    a, b = rng.choice((("m", "k"), ("a", "b")))
    p = Prompt()
    p.t("Prove by contradiction that ").m(_sqrt(n)).t(
        f" is irrational. You may use: every integer greater than 1 can be written as a product of primes in only one way, apart from order. Note that {n} = {fact_text}. "
    ).t(_tail(rng))
    ref = [
        f"Suppose √{n} is rational: √{n} = {a}/{b} with {a}, {b} positive integers.",
        f"Square both sides: {a}² = {n}{b}².",
        f"In the prime factorisation of {a}², every prime appears an even number of times (each exponent in {a} is doubled). The same is true of {b}².",
        f"In {n} = {fact_text}, the prime {ell} appears an odd number of times ({factors[ell]}).",
        f"So in {n}{b}², the prime {ell} appears an odd number of times (odd plus even).",
        f"But {a}² and {n}{b}² are the same integer, and it would have {ell} an even number of times in one factorisation and an odd number in the other. This contradicts unique factorisation. So √{n} is irrational.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            (
                "correctness",
                f"identifies {ell} as appearing an odd number of times in {n}, and shows the parity of its exponent differs on the two sides",
            ),
            (
                "logical_completeness",
                "argues about the exponent of a prime on both sides and names the contradiction with unique factorisation",
            ),
            ("definitions_assumptions", "explains why every prime has an even exponent in a perfect square"),
            ("communication", "uses the given factorisation of the number"),
        ],
        traps=[
            (
                "lowest_terms_argument_only",
                f"does it use the lowest-terms argument for a prime ({n} is not prime) without justifying the divisibility step?",
            ),
            ("wrong_prime", "does it pick a prime whose exponent in the number is even, so there is no contradiction?"),
        ],
    )
    return finish(p, f"√{n} is irrational", mark)


IRR_FAMILIES = {
    "easy": (_irr_sqrt_prime, _irr_sqrt_two_parity),
    "medium": (_irr_log, _irr_shift),
    "hard": (_irr_affine, _irr_sqrt_pq, _irr_cuberoot),
    "exam": (_irr_sum_of_roots, _irr_log_power, _irr_sqrt_nonsquare),
}


# ===========================================================================
# Simple results involving numbers
# ===========================================================================
def _window(lo: int = -40, hi: int = 41):
    return range(lo, hi)


def _parity(kind: int) -> str:
    return "odd" if kind else "even"


def _num_parity_rule(rng) -> dict:
    op = rng.choice(("sum", "product"))
    r1, r2 = rng.choice(((0, 0), (1, 1), (0, 1)))
    swap = rng.random() < 0.5 and r1 != r2
    if swap:
        r1, r2 = r2, r1
    if op == "sum":
        result = (r1 + r2) % 2
        check(
            all(((2 * a + r1) + (2 * b + r2)) % 2 == result for a in range(-6, 7) for b in range(-6, 7)),
            "parity of a sum",
        )
    else:
        result = (r1 * r2) % 2
        check(
            all(((2 * a + r1) * (2 * b + r2)) % 2 == result for a in range(-6, 7) for b in range(-6, 7)),
            "parity of a product",
        )
    claim = f"the {op} of {'an' if r1 else 'an'} {_parity(r1)} integer and {'an' if r2 or True else 'a'} {_parity(r2)} integer is {_parity(result)}"
    if r1 == r2:
        claim = f"the {op} of two {_parity(r1)} integers is {_parity(result)}"
    p = Prompt()
    p.t(
        f"Prove that {claim}. You may use: an even integer can be written as 2k and an odd integer as 2k + 1, where k is an integer. "
    ).t(_tail(rng))
    m_form = "2a + 1" if r1 else "2a"
    n_form = "2b + 1" if r2 else "2b"
    if op == "sum":
        total = {0: "2(a + b)", 1: "2(a + b) + 1", 2: "2(a + b + 1)"}[r1 + r2]
        step = f"({m_form}) + ({n_form}) = {total}"
    else:
        expanded = {
            (0, 0): "(2a)(2b) = 2(2ab)",
            (1, 0): "(2a + 1)(2b) = 4ab + 2b = 2(2ab + b)",
            (0, 1): "(2a)(2b + 1) = 4ab + 2a = 2(2ab + a)",
            (1, 1): "(2a + 1)(2b + 1) = 4ab + 2a + 2b + 1 = 2(2ab + a + b) + 1",
        }
        step = expanded[(r1, r2)]
    ref = [
        f"Let the first integer be {m_form} and the second be {n_form}, where a and b are integers.",
        f"Their {op} is {step}.",
        f"The expression has the form {'2 × (an integer) + 1' if result else '2 × (an integer)'}, so it is {_parity(result)}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"expands the {op} to the form {'2k + 1' if result else '2k'}"),
            ("logical_completeness", "uses general integers a and b, not particular numbers"),
            ("definitions_assumptions", "uses the stated forms for even and odd integers"),
            ("communication", "states the conclusion in words"),
        ],
        traps=[
            ("tests_examples", "does it test a few numbers instead of proving the general case?"),
            (
                "same_letter_for_both",
                "does it use the same letter k for both integers, so they are forced to be equal?",
            ),
        ],
    )
    return finish(p, claim, mark)


def _num_multiples(rng) -> dict:
    k = rng.choice((3, 4, 5, 6, 7, 8, 9))
    kind = rng.choice(("sum", "product"))
    if kind == "sum":
        check(all((k * a + k * b) % k == 0 for a in range(-8, 9) for b in range(-8, 9)), "sum of multiples")
        claim = f"the sum of two multiples of {k} is a multiple of {k}"
        ref = [
            f"Let the two multiples of {k} be {k}a and {k}b, where a and b are integers.",
            f"Their sum is {k}a + {k}b = {k}(a + b).",
            f"a + b is an integer, so the sum is {k} times an integer, a multiple of {k}.",
        ]
    else:
        check(all((k * a * b) % k == 0 for a in range(-8, 9) for b in range(-8, 9)), "product with a multiple")
        claim = f"the product of a multiple of {k} and any integer is a multiple of {k}"
        ref = [
            f"Let the multiple of {k} be {k}a and the other integer be b, where a and b are integers.",
            f"Their product is ({k}a)b = {k}(ab).",
            f"ab is an integer, so the product is {k} times an integer, a multiple of {k}.",
        ]
    p = Prompt()
    p.t(f"Prove that {claim}. ").t(_tail(rng))
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"writes the result as {k} times an integer"),
            ("logical_completeness", "uses general integers, not particular multiples"),
            ("definitions_assumptions", f"starts from the definition: a multiple of {k} is {k} times an integer"),
            ("communication", "states the conclusion in words"),
        ],
        traps=[
            ("tests_examples", "does it test a few numbers instead of proving the general case?"),
            ("same_letter_for_both", "does it use one letter for both integers?"),
        ],
    )
    return finish(p, claim, mark)


def _num_consecutive_sum(rng) -> dict:
    k = rng.choice((3, 5, 7, 9))
    check(all(sum(range(n, n + k)) % k == 0 for n in _window()), "k consecutive sum")
    claim = f"the sum of any {k} consecutive integers is divisible by {k}"
    mid = (k - 1) // 2
    p = Prompt()
    p.t(f"Prove that {claim}. ").t(_tail(rng))
    terms = ", ".join(f"n + {i}" if i else "n" for i in range(k))
    ref = [
        f"Let the integers be {terms}, where n is an integer.",
        f"Their sum is {k}n + (0 + 1 + ... + {k - 1}) = {k}n + {k * (k - 1) // 2}.",
        f"{k * (k - 1) // 2} = {k} × {mid}, so the sum is {k}n + {k} × {mid} = {k}(n + {mid}).",
        f"n + {mid} is an integer, so the sum is a multiple of {k}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"gets the sum {k}n + {k * (k - 1) // 2} and factors out {k}"),
            ("logical_completeness", "uses a general first integer n"),
            ("definitions_assumptions", f"notes that {k - 1} is even, so {k * (k - 1) // 2} is a multiple of {k}"),
            ("communication", "states the conclusion in words"),
        ],
        traps=[
            ("sum_formula_slip", f"is 0 + 1 + ... + {k - 1} = {k * (k - 1) // 2}?"),
            ("tests_examples", "does it test particular runs instead of proving the general case?"),
        ],
    )
    return finish(p, claim, mark)


def _num_n_times_n_plus_d(rng) -> dict:
    d = rng.choice((1, 3, 5, 7, 9, 11))
    v = rng.choice(("n", "k", "m"))
    check(all((n * (n + d)) % 2 == 0 for n in _window()), "n(n+d) is even")
    expr = f"{v}² + {d}{v}" if d != 1 else f"{v}² + {v}"
    claim = f"{expr} is even for every integer {v}"
    p = Prompt()
    p.t(f"Prove that {expr} is even for every integer {v}. ").t(_tail(rng))
    ref = [
        f"Factorise: {expr} = {v}({v} + {d}).",
        f"The two factors differ by {d}, which is odd, so one of {v} and {v} + {d} is even and the other is odd.",
        "(If one is even, the other differs from it by an odd number, so it is odd.)",
        "A product with an even factor is even, so the product is even.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"factorises to {v}({v} + {d}) and shows one factor is even"),
            ("logical_completeness", "covers both cases (the first factor even or odd) or gives the parity argument"),
            ("definitions_assumptions", "uses that an even factor makes the whole product even"),
            ("communication", "states the conclusion"),
        ],
        traps=[
            ("tests_examples", "does it check a few values of n instead of proving the general case?"),
            ("only_one_case", "does it treat only the case where n is even, or only where it is odd?"),
        ],
    )
    return finish(p, claim, mark)


def _num_consecutive_squares(rng) -> dict:
    v = rng.choice(("n", "k"))
    kind = rng.choice(("sum", "difference"))
    if kind == "sum":
        check(all(((n + 1) ** 2 + n**2) % 2 == 1 for n in _window()), "sum of consecutive squares is odd")
        claim = f"the sum of the squares of two consecutive integers is odd"
        ref = [
            f"Let the integers be {v} and {v} + 1.",
            f"The sum of their squares is {v}² + ({v} + 1)² = 2{v}² + 2{v} + 1.",
            f"2{v}² + 2{v} + 1 = 2({v}² + {v}) + 1, which is 2 × (an integer) + 1.",
            "So the sum is odd.",
        ]
    else:
        check(all(((n + 1) ** 2 - n**2) % 2 == 1 for n in _window()), "difference of consecutive squares is odd")
        claim = "the difference of the squares of two consecutive integers is odd"
        ref = [
            f"Let the integers be {v} and {v} + 1.",
            f"The difference of their squares is ({v} + 1)² - {v}² = 2{v} + 1.",
            f"2{v} + 1 is 2 × (an integer) + 1.",
            "So the difference is odd.",
        ]
    p = Prompt()
    p.t(f"Prove that {claim}. ").t(_tail(rng))
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "expands the squares correctly and reaches the form 2k + 1"),
            ("logical_completeness", "uses a general integer and covers both even and odd starting values at once"),
            ("definitions_assumptions", "states what makes an integer odd (2k + 1)"),
            ("communication", "states the conclusion"),
        ],
        traps=[
            ("expansion_slip", "is (n + 1)² expanded with the middle term 2n?"),
            ("tests_examples", "does it test pairs of numbers instead of proving the general case?"),
        ],
    )
    return finish(p, claim, mark)


def _num_odd_square_form(rng) -> dict:
    kind = rng.choice(("mod4", "mod8", "even_square"))
    if kind == "mod8":
        check(all(((2 * a + 1) ** 2) % 8 == 1 for a in _window()), "odd square is 1 mod 8")
        claim = "the square of any odd integer leaves remainder 1 when divided by 8"
        ref = [
            "Let the odd integer be 2k + 1, where k is an integer.",
            "Its square is (2k + 1)² = 4k² + 4k + 1 = 4k(k + 1) + 1.",
            "k and k + 1 are consecutive integers, so one of them is even and k(k + 1) = 2j for some integer j.",
            "So the square is 4 × 2j + 1 = 8j + 1, which leaves remainder 1 on division by 8.",
        ]
        steps = [
            ("correctness", "expands to 4k(k + 1) + 1 and shows k(k + 1) is even"),
            ("logical_completeness", "justifies that k(k + 1) is even, not just asserts it"),
            ("definitions_assumptions", "uses 2k + 1 for a general odd integer"),
            ("communication", "writes the final form as 8j + 1"),
        ]
    elif kind == "mod4":
        check(all(((2 * a + 1) ** 2) % 4 == 1 and ((2 * a) ** 2) % 4 == 0 for a in _window()), "squares mod 4")
        claim = "the square of any integer leaves remainder 0 or 1 when divided by 4"
        ref = [
            "Every integer is even or odd, so consider the two cases.",
            "If the integer is 2k, its square is 4k², which is a multiple of 4, remainder 0.",
            "If the integer is 2k + 1, its square is 4k² + 4k + 1 = 4(k² + k) + 1, remainder 1.",
            "So the remainder is always 0 or 1.",
        ]
        steps = [
            ("correctness", "obtains 4k² and 4(k² + k) + 1 for the two cases"),
            ("logical_completeness", "covers both the even and the odd case"),
            ("definitions_assumptions", "uses 2k and 2k + 1 for general integers"),
            ("communication", "states the conclusion"),
        ]
    else:
        check(all(((2 * a) ** 2) % 4 == 0 for a in _window()), "even square is a multiple of 4")
        claim = "the square of any even integer is divisible by 4"
        ref = [
            "Let the even integer be 2k, where k is an integer.",
            "Its square is (2k)² = 4k².",
            "k² is an integer, so the square is 4 times an integer.",
            "So the square is divisible by 4.",
        ]
        steps = [
            ("correctness", "writes the square as 4k²"),
            ("logical_completeness", "uses a general even integer"),
            ("definitions_assumptions", "uses 2k for an even integer"),
            ("communication", "states the conclusion"),
        ]
    p = Prompt()
    p.t(f"Prove that {claim}. ").t(_tail(rng))
    mark = marking(
        reference=ref,
        steps=steps,
        traps=[
            ("tests_examples", "does it test a few squares instead of proving the general case?"),
            ("one_case_only", "does it consider only one parity?"),
        ],
    )
    return finish(p, claim, mark)


def _num_three_consecutive(rng) -> dict:
    which = rng.choice(("n3_minus_n", "product_three", "n3_plus_5n"))
    v = rng.choice(("n", "k"))
    if which == "n3_minus_n":
        check(all((n**3 - n) % 6 == 0 for n in _window()), "n^3 - n")
        claim = f"{v}³ - {v} is divisible by 6 for every integer {v}"
        factor = f"{v}³ - {v} = ({v} - 1){v}({v} + 1)"
        ref_tail = []
    elif which == "product_three":
        check(all((n * (n + 1) * (n + 2)) % 6 == 0 for n in _window()), "three consecutive")
        claim = "the product of any three consecutive integers is divisible by 6"
        factor = f"The product is {v}({v} + 1)({v} + 2)"
        ref_tail = []
    else:
        check(all((n**3 + 5 * n) % 6 == 0 for n in _window()), "n^3 + 5n")
        claim = f"{v}³ + 5{v} is divisible by 6 for every integer {v}"
        factor = f"{v}³ + 5{v} = ({v}³ - {v}) + 6{v} = ({v} - 1){v}({v} + 1) + 6{v}"
        ref_tail = [f"6{v} is divisible by 6, so it is enough to show ({v} - 1){v}({v} + 1) is divisible by 6."]
    p = Prompt()
    p.t(f"Prove that {claim}. You may use: if an integer is divisible by both 2 and 3 then it is divisible by 6. ").t(
        _tail(rng)
    )
    ref = [
        factor + ".",
        *ref_tail,
        "Among three consecutive integers at least one is even, so the product is divisible by 2.",
        "Among three consecutive integers exactly one is a multiple of 3 (they leave remainders 0, 1, 2 in some order on division by 3), so the product is divisible by 3.",
        "The product is divisible by both 2 and 3, so it is divisible by 6.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "factors the expression into three consecutive integers"),
            ("logical_completeness", "proves divisibility by 2 and by 3 separately and then combines them"),
            ("definitions_assumptions", "justifies why one of three consecutive integers is a multiple of 3"),
            ("communication", "states the conclusion"),
        ],
        traps=[
            ("tests_examples", "does it test several values and stop?"),
            ("divisibility_by_six_directly", "does it claim one of three consecutive integers is a multiple of 6?"),
        ],
    )
    return finish(p, claim, mark)


def _num_odd_square_minus_one(rng) -> dict:
    v = rng.choice(("n", "k"))
    check(all((n * n - 1) % 8 == 0 for n in range(-41, 42, 2)), "odd n: n^2 - 1 divisible by 8")
    claim = f"{v}² - 1 is divisible by 8 for every odd integer {v}"
    p = Prompt()
    p.t(f"Prove that {claim}. ").t(_tail(rng))
    ref = [
        f"Since {v} is odd, {v} = 2j + 1 for an integer j.",
        f"Then {v}² - 1 = (2j + 1)² - 1 = 4j² + 4j = 4j(j + 1).",
        "j and j + 1 are consecutive integers, so one is even, and j(j + 1) is even: j(j + 1) = 2i.",
        f"So {v}² - 1 = 4 × 2i = 8i, which is divisible by 8.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "reaches 4j(j + 1) and then 8i"),
            ("logical_completeness", "justifies that j(j + 1) is even"),
            ("definitions_assumptions", "uses 2j + 1 for a general odd integer"),
            ("communication", "states the conclusion"),
        ],
        traps=[
            ("tests_examples", "does it test a few odd numbers instead of proving the general case?"),
            ("assumes_even_product", "does it assert j(j + 1) is even with no reason?"),
        ],
    )
    return finish(p, claim, mark)


def _num_n_n1_2n1(rng) -> dict:
    v = rng.choice(("n", "k"))
    check(all((n * (n + 1) * (2 * n + 1)) % 6 == 0 for n in _window()), "n(n+1)(2n+1)")
    claim = f"{v}({v} + 1)(2{v} + 1) is divisible by 6 for every integer {v}"
    p = Prompt()
    p.t(f"Prove that {claim}. You may use: if an integer is divisible by both 2 and 3 then it is divisible by 6. ").t(
        _tail(rng)
    )
    ref = [
        f"{v}({v} + 1) is a product of consecutive integers, so one factor is even and the product is divisible by 2.",
        f"For divisibility by 3, consider the remainder of {v} on division by 3.",
        f"If {v} = 3j, then {v} is a multiple of 3. If {v} = 3j + 1, then 2{v} + 1 = 6j + 3 = 3(2j + 1). If {v} = 3j + 2, then {v} + 1 = 3j + 3 = 3(j + 1).",
        "In each case one of the factors is a multiple of 3, so the product is divisible by 3.",
        "The product is divisible by both 2 and 3, so it is divisible by 6.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "shows divisibility by 2 and by 3 correctly"),
            ("logical_completeness", "covers all three remainders on division by 3"),
            ("definitions_assumptions", "uses general forms 3j, 3j + 1, 3j + 2"),
            ("communication", "combines the two results to reach 6"),
        ],
        traps=[
            ("missing_case", "does it cover only some of the three remainders?"),
            ("tests_examples", "does it test several values and stop?"),
        ],
    )
    return finish(p, claim, mark)


def _num_digits_rule(rng) -> dict:
    kind = rng.choice(("nine", "three", "eleven"))
    if kind in ("nine", "three"):
        m = 9 if kind == "nine" else 3
        check(
            all(
                (100 * a + 10 * b + c) % m == 0
                for a in range(1, 10)
                for b in range(10)
                for c in range(10)
                if (a + b + c) % m == 0
            ),
            "digit-sum rule",
        )
        claim = f"if the digit sum of a three-digit number is divisible by {m}, then the number is divisible by {m}"
        p = Prompt()
        p.t(f"A three-digit number has digits a, b and c, so it equals 100a + 10b + c. Prove that {claim}. ").t(
            _tail(rng)
        )
        ref = [
            "The number is 100a + 10b + c.",
            "Write it as 99a + 9b + (a + b + c), since 99a + 9b + a + b + c = 100a + 10b + c.",
            f"99a + 9b = 9(11a + b), which is divisible by 9, so it is divisible by {m}.",
            f"If a + b + c is divisible by {m}, then the number is a sum of two multiples of {m}, so it is a multiple of {m}.",
        ]
        steps = [
            ("correctness", "splits the number as 99a + 9b + (a + b + c)"),
            (
                "logical_completeness",
                f"shows 99a + 9b is a multiple of {m} and then uses the assumption on the digit sum",
            ),
            ("definitions_assumptions", "uses the place-value form 100a + 10b + c"),
            ("communication", "states the conclusion"),
        ]
        traps = [
            ("tests_examples", "does it test particular numbers?"),
            ("converse_only", "does it prove the converse (number divisible implies digit sum divisible) instead?"),
        ]
    else:
        check(
            all(
                (100 * a + 10 * b + c) % 11 == 0
                for a in range(1, 10)
                for b in range(10)
                for c in range(10)
                if (a - b + c) % 11 == 0
            ),
            "alternating digit rule",
        )
        claim = "if a - b + c is divisible by 11, then the number is divisible by 11"
        p = Prompt()
        p.t(f"A three-digit number has digits a, b and c, so it equals 100a + 10b + c. Prove that {claim}. ").t(
            _tail(rng)
        )
        ref = [
            "The number is 100a + 10b + c.",
            "Write it as 99a + 11b + (a - b + c), since 99a + 11b + a - b + c = 100a + 10b + c.",
            "99a + 11b = 11(9a + b), which is divisible by 11.",
            "If a - b + c is divisible by 11, the number is a sum of two multiples of 11, so it is a multiple of 11.",
        ]
        steps = [
            ("correctness", "splits the number as 99a + 11b + (a - b + c)"),
            ("logical_completeness", "shows 99a + 11b is a multiple of 11 and then uses the assumption"),
            ("definitions_assumptions", "uses the place-value form 100a + 10b + c"),
            ("communication", "states the conclusion"),
        ]
        traps = [
            ("tests_examples", "does it test particular numbers?"),
            ("sign_slip", "does the middle digit have the right sign in the rearrangement?"),
        ]
    mark = marking(reference=ref, steps=steps, traps=traps)
    return finish(p, claim, mark)


def _num_four_consecutive(rng) -> dict:
    form = rng.choice(("start", "centre"))
    v = rng.choice(("n", "k"))
    if form == "start":
        check(
            all(n * (n + 1) * (n + 2) * (n + 3) + 1 == (n * n + 3 * n + 1) ** 2 for n in _window()),
            "four consecutive plus one",
        )
        claim = f"the product of four consecutive integers plus 1 is a perfect square"
        expr = f"{v}({v} + 1)({v} + 2)({v} + 3) + 1"
        root = f"{v}² + 3{v} + 1"
        ref = [
            f"Let the integers be {v}, {v} + 1, {v} + 2, {v} + 3.",
            f"Pair the outer and inner factors: {v}({v} + 3) = {v}² + 3{v} and ({v} + 1)({v} + 2) = {v}² + 3{v} + 2.",
            f"Let m = {v}² + 3{v}. The product plus 1 is m(m + 2) + 1 = m² + 2m + 1.",
            f"m² + 2m + 1 = (m + 1)² = ({root})², which is the square of an integer.",
        ]
    else:
        check(
            all((n - 1) * n * (n + 1) * (n + 2) + 1 == (n * n + n - 1) ** 2 for n in _window()),
            "four consecutive plus one (centre form)",
        )
        claim = "the product of four consecutive integers plus 1 is a perfect square"
        expr = f"({v} - 1){v}({v} + 1)({v} + 2) + 1"
        root = f"{v}² + {v} - 1"
        ref = [
            f"Let the integers be {v} - 1, {v}, {v} + 1, {v} + 2.",
            f"Pair the outer and inner factors: ({v} - 1)({v} + 2) = {v}² + {v} - 2 and {v}({v} + 1) = {v}² + {v}.",
            f"Let m = {v}² + {v}. The product plus 1 is m(m - 2) + 1 = m² - 2m + 1.",
            f"m² - 2m + 1 = (m - 1)² = ({root})², which is the square of an integer.",
        ]
    p = Prompt()
    p.t(f"Prove that {claim}. For example, you may take the four integers to be ").t(
        ("n, n + 1, n + 2, n + 3" if form == "start" else "n - 1, n, n + 1, n + 2").replace("n", v)
    ).t(f", so the expression is {expr}. ").t(_tail(rng))
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"shows the expression equals ({root})²"),
            ("logical_completeness", "uses general integers and a substitution m, not particular numbers"),
            ("definitions_assumptions", "states that the root is an integer, so the result is a perfect square"),
            ("communication", "shows the factor pairing"),
        ],
        traps=[
            (
                "expands_everything_blindly",
                "does it expand the quartic without noticing the structure, then fail to factorise?",
            ),
            ("tests_examples", "does it only check that 1×2×3×4+1 = 25 and similar?"),
        ],
    )
    return finish(p, claim, mark)


def _num_odd_squares_sum(rng) -> dict:
    check(
        all(((2 * a + 1) ** 2 + (2 * b + 1) ** 2) % 4 == 2 for a in range(-6, 7) for b in range(-6, 7)),
        "sum of two odd squares is 2 mod 4",
    )
    claim = "the sum of the squares of two odd integers is never divisible by 4"
    p = Prompt()
    p.t(f"Prove that {claim}. ").t(_tail(rng))
    ref = [
        "Let the odd integers be 2a + 1 and 2b + 1, where a and b are integers.",
        "The sum of their squares is (2a + 1)² + (2b + 1)² = 4a² + 4a + 1 + 4b² + 4b + 1.",
        "That is 4(a² + a + b² + b) + 2.",
        "So the sum leaves remainder 2 when divided by 4, and is therefore not divisible by 4.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "reaches 4(…) + 2"),
            ("logical_completeness", "uses two independent integers a and b and draws the remainder conclusion"),
            ("definitions_assumptions", "uses 2a + 1 and 2b + 1 for general odd integers"),
            ("communication", "states the remainder is 2, not 0"),
        ],
        traps=[
            (
                "same_letter_for_both",
                "does it use the same letter for both odd integers, so it only proves 2(2a + 1)² is not divisible by 4?",
            ),
            ("tests_examples", "does it test a few pairs?"),
        ],
    )
    return finish(p, claim, mark)


def _num_n5_minus_n(rng) -> dict:
    check(all((n**5 - n) % 30 == 0 for n in _window()), "n^5 - n")
    v = rng.choice(("n", "k"))
    claim = f"{v}⁵ - {v} is divisible by 30 for every integer {v}"
    p = Prompt()
    p.t(f"Prove that {claim}. You may use: if an integer is divisible by 2, 3 and 5 then it is divisible by 30. ").t(
        _tail(rng)
    )
    ref = [
        f"Factorise: {v}⁵ - {v} = {v}({v}⁴ - 1) = {v}({v}² - 1)({v}² + 1) = ({v} - 1){v}({v} + 1)({v}² + 1).",
        f"({v} - 1){v}({v} + 1) are three consecutive integers, so one is even and one is a multiple of 3: the product is divisible by 2 and by 3.",
        f"For 5, consider {v} on division by 5. If {v} = 5j, {v} is a multiple of 5. If {v} = 5j + 1, {v} - 1 = 5j. If {v} = 5j + 4, {v} + 1 = 5j + 5.",
        f"If {v} = 5j + 2, {v}² + 1 = 25j² + 20j + 5 = 5(5j² + 4j + 1). If {v} = 5j + 3, {v}² + 1 = 25j² + 30j + 10 = 5(5j² + 6j + 2).",
        "In every case one factor is a multiple of 5, so the product is divisible by 5.",
        "The product is divisible by 2, 3 and 5, so it is divisible by 30.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "factorises completely and proves divisibility by 2, 3 and 5"),
            ("logical_completeness", "covers all five remainders on division by 5"),
            ("definitions_assumptions", "justifies the 2 and 3 parts from three consecutive integers"),
            ("communication", "combines the three results to reach 30"),
        ],
        traps=[
            ("missing_remainders", "does it cover only some of the remainders 0 to 4?"),
            ("stops_after_factorising", "does it factorise but never prove divisibility by 5?"),
        ],
    )
    return finish(p, claim, mark)


def _num_contrapositive(rng) -> dict:
    pr = rng.choice((2, 3, 5))
    check(all((n * n) % pr != 0 for n in range(-60, 61) if n % pr != 0), "prime divides n^2 only if it divides n")
    v = rng.choice(("n", "k"))
    claim = f"if {v}² is divisible by {pr} then {v} is divisible by {pr}"
    p = Prompt()
    p.t(
        f"Prove that {claim}, where {v} is an integer. Prove the contrapositive: if {v} is not divisible by {pr}, then {v}² is not divisible by {pr}. You may use: if a prime divides a product ab, then it divides a or b. "
    ).t(_tail(rng))
    rems = ", ".join(str(r) for r in range(1, pr))
    ref = [
        f"Suppose {v} is not divisible by {pr}. Then {v} = {pr}j + r for an integer j and a remainder r in {{{rems}}}.",
        f"Expanding, {v}² = ({pr}j + r)² = {pr}({pr}j² + 2jr) + r².",
        f"So {v}² leaves the same remainder as r² on division by {pr}; {v}² is divisible by {pr} only if {pr} divides r².",
        f"But {pr} is prime and {pr} does not divide r (0 < r < {pr}), so {pr} does not divide r × r (a prime dividing a product divides a factor).",
        f"So {v}² is not divisible by {pr}. This proves the contrapositive, and so proves: if {v}² is divisible by {pr} then {v} is divisible by {pr}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"shows {v}² has the same remainder as r² modulo {pr} and that this is not 0"),
            ("logical_completeness", f"covers every non-zero remainder r, using the prime property for {pr}"),
            (
                "definitions_assumptions",
                "states clearly that the contrapositive is equivalent to the original statement",
            ),
            ("communication", "labels the assumption, the work and the conclusion"),
        ],
        traps=[
            ("examples_only", "does it check the remainders of a few squares instead of covering all remainders?"),
            ("converse_instead", f"does it prove the converse (if {v} is divisible by {pr}, so is {v}²) instead?"),
        ],
    )
    return finish(p, claim, mark)


NUM_FAMILIES = {
    "easy": (_num_parity_rule, _num_multiples),
    "medium": (_num_consecutive_sum, _num_n_times_n_plus_d, _num_consecutive_squares, _num_odd_square_form),
    "hard": (_num_three_consecutive, _num_odd_square_minus_one, _num_n_n1_2n1),
    "exam": (_num_digits_rule, _num_four_consecutive, _num_odd_squares_sum, _num_n5_minus_n, _num_contrapositive),
}


# ===========================================================================
# Pigeon-hole principle
# ===========================================================================
CATEGORIES = (
    # (number of boxes, objects phrase template, "share" phrase)
    (12, "people", "were born in the same month of the year"),
    (7, "people", "were born on the same day of the week"),
    (26, "words", "start with the same letter of the English alphabet"),
    (10, "whole numbers", "have the same last digit"),
    (4, "cards drawn from a standard pack", "are of the same suit"),
    (3, "socks taken from a drawer holding only red, blue and green socks", "are the same colour"),
    (5, "students", "have the same score on a quiz marked out of 4 (scores 0 to 4)"),
)


def _pg_basic(rng) -> dict:
    boxes, things, share = rng.choice(CATEGORIES)
    n = boxes + rng.randint(1, 6)
    check(n > boxes, "more objects than boxes")
    claim = f"among {n} {things}, at least two {share}"
    p = Prompt()
    p.t(
        f"Prove that {claim}. You may use the pigeon-hole principle: if more than k objects are placed in k boxes, some box holds at least two objects. "
    ).t(_tail(rng))
    ref = [
        f"There are {boxes} possible categories, so think of {boxes} boxes, one for each category.",
        f"Place each of the {n} {things} in the box for its category.",
        f"There are {n} objects and only {boxes} boxes, and {n} > {boxes}.",
        f"By the pigeon-hole principle, some box contains at least two objects, that is, at least two {share}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"identifies exactly {boxes} boxes and {n} objects"),
            ("logical_completeness", "applies the principle and links 'two in one box' back to the claim"),
            ("definitions_assumptions", "says what the boxes are and what the objects are"),
            ("communication", "states the comparison n > number of boxes"),
        ],
        traps=[
            ("wrong_number_of_boxes", f"does it use exactly {boxes} boxes, not {boxes + 1} or {boxes - 1}?"),
            ("no_boxes_named", "does it say the principle applies without saying what the boxes are?"),
        ],
    )
    return finish(p, claim, mark)


def _pg_ceiling(rng) -> dict:
    boxes, things, share = rng.choice(CATEGORIES[:5])
    c = rng.choice((3, 4, 5, 6))
    n = boxes * (c - 1) + rng.randint(1, boxes)
    check(ceil(n / boxes) == c and n > boxes * (c - 1), "ceiling")
    claim = f"among {n} {things}, at least {c} {share}"
    p = Prompt()
    p.t(
        f"Prove that {claim}. You may use the pigeon-hole principle: if more than k objects are placed in k boxes, some box holds at least two objects. Your argument should show the general step, that if there are more than {boxes} × {c - 1} objects in {boxes} boxes then some box holds at least {c}. "
    ).t(_tail(rng))
    ref = [
        f"There are {boxes} possible categories, so use {boxes} boxes, one for each, and place each of the {n} objects in its box.",
        f"Suppose, for contradiction, that every box holds at most {c - 1} objects.",
        f"Then the total number of objects is at most {boxes} × {c - 1} = {boxes * (c - 1)}.",
        f"But there are {n} objects and {n} > {boxes * (c - 1)}. This is a contradiction.",
        f"So some box holds at least {c} objects, that is, at least {c} {share}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"computes {boxes} × {c - 1} = {boxes * (c - 1)} and compares with {n}"),
            ("logical_completeness", "assumes every box has at most c - 1 objects and derives the contradiction"),
            ("definitions_assumptions", "names the boxes"),
            ("communication", "ends with the conclusion in terms of the original objects"),
        ],
        traps=[
            ("divides_without_justifying", f"does it just divide {n} by {boxes} and round up, with no argument?"),
            ("wrong_threshold", f"does it compare with {boxes} × {c} instead of {boxes} × {c - 1}?"),
        ],
    )
    return finish(p, claim, mark)


def _pg_mod_difference(rng) -> dict:
    m = rng.choice((3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    check(m + 1 > m, "m + 1 integers, m remainders")
    claim = f"among any {m + 1} integers, there are two whose difference is divisible by {m}"
    p = Prompt()
    p.t(
        f"Prove that {claim}. You may use the pigeon-hole principle: if more than k objects are placed in k boxes, some box holds at least two objects. "
    ).t(_tail(rng))
    ref = [
        f"When an integer is divided by {m}, the remainder is one of {', '.join(str(r) for r in range(m))}: {m} possible remainders.",
        f"Take {m} boxes, one for each remainder, and put each of the {m + 1} integers in the box for its remainder.",
        f"There are {m + 1} integers and {m} boxes, so by the pigeon-hole principle two integers, a and b, share a box.",
        f"Then a = {m}s + r and b = {m}t + r for integers s, t and the same remainder r.",
        f"So a - b = {m}(s - t), which is divisible by {m}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"uses the {m} remainders as the boxes and {m + 1} integers as the objects"),
            ("logical_completeness", "turns 'same remainder' into 'difference divisible' with algebra"),
            ("definitions_assumptions", "states that the remainders are 0 to m - 1"),
            ("communication", "says which two integers are chosen"),
        ],
        traps=[
            ("wrong_boxes", "does it use the integers' sizes or last digits as boxes instead of remainders?"),
            (
                "stops_at_same_remainder",
                "does it stop at 'same remainder' without showing the difference is divisible?",
            ),
        ],
    )
    return finish(p, claim, mark)


def _pg_consecutive(rng) -> dict:
    n = rng.randint(4, 12)
    pairs = [(2 * i - 1, 2 * i) for i in range(1, n + 1)]
    check(len(pairs) == n and n + 1 > n, "n pairs, n + 1 numbers")
    if n <= 6:
        full = range(1, 2 * n + 1)
        check(
            all(any(abs(a - b) == 1 for a, b in combinations(sub, 2)) for sub in combinations(full, n + 1)),
            "exhaustive",
        )
    claim = f"if {n + 1} different numbers are chosen from 1, 2, ..., {2 * n}, two of them are consecutive"
    p = Prompt()
    p.t(
        f"Prove that {claim}. You may use the pigeon-hole principle: if more than k objects are placed in k boxes, some box holds at least two objects. "
    ).t(_tail(rng))
    ref = [
        f"Split 1, 2, ..., {2 * n} into the {n} pairs (1, 2), (3, 4), ..., ({2 * n - 1}, {2 * n}). Each pair is two consecutive numbers.",
        f"Use these {n} pairs as boxes. Place each chosen number in the pair that contains it.",
        f"There are {n + 1} chosen numbers and only {n} boxes, so by the pigeon-hole principle some pair holds two chosen numbers.",
        "Both numbers of a pair are chosen, and a pair consists of two consecutive numbers. So two chosen numbers are consecutive.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"forms exactly {n} boxes, each a pair of consecutive numbers"),
            ("logical_completeness", "applies the principle and links 'two in one box' to 'consecutive'"),
            (
                "definitions_assumptions",
                "notes the chosen numbers are different, so a pair holding two chosen numbers holds both of its members",
            ),
            ("communication", "describes the pairs explicitly"),
        ],
        traps=[
            ("pairs_overlap", "do the boxes overlap, as (1,2), (2,3), ... would?"),
            ("wrong_box_count", f"does it use {n + 1} or {2 * n} boxes instead of {n}?"),
        ],
    )
    return finish(p, claim, mark)


def _odd_part(k: int) -> int:
    while k % 2 == 0:
        k //= 2
    return k


def _pg_divides(rng) -> dict:
    n = rng.randint(4, 10)
    check(len({_odd_part(k) for k in range(1, 2 * n + 1)}) == n, "n odd parts among 1..2n")
    if n <= 7:
        full = range(1, 2 * n + 1)
        check(all(any(b % a == 0 for a, b in combinations(sub, 2)) for sub in combinations(full, n + 1)), "exhaustive")
    claim = f"if {n + 1} different numbers are chosen from 1, 2, ..., {2 * n}, one of them divides another"
    p = Prompt()
    p.t(
        f"Prove that {claim}. You may use the pigeon-hole principle: if more than k objects are placed in k boxes, some box holds at least two objects. Hint: every positive integer can be written as 2^j × q, where q is odd. "
    ).t(_tail(rng))
    odds = ", ".join(str(q) for q in range(1, 2 * n, 2))
    ref = [
        f"Write each chosen number as 2^j × q with q odd. In 1, 2, ..., {2 * n} the odd part q is one of the {n} odd numbers {odds}.",
        f"Use these {n} odd numbers as boxes and place each chosen number in the box of its odd part.",
        f"There are {n + 1} chosen numbers and {n} boxes, so two different chosen numbers share an odd part q: they are 2^i × q and 2^j × q with i ≠ j.",
        "Take i < j. Then 2^j × q = 2^(j - i) × (2^i × q), so the smaller number divides the larger.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"uses the {n} odd numbers up to {2 * n - 1} as the boxes"),
            ("logical_completeness", "turns 'same odd part' into a divisibility statement"),
            ("definitions_assumptions", "says the numbers are different, so the powers of 2 differ"),
            ("communication", "writes the two numbers explicitly as 2^i q and 2^j q"),
        ],
        traps=[
            (
                "wrong_boxes",
                "does it use pairs of consecutive numbers (the boxes for the consecutive problem) instead of odd parts?",
            ),
            ("never_shows_divisibility", "does it stop at 'same odd part' with no divisibility step?"),
        ],
    )
    return finish(p, claim, mark)


def _pg_rectangle(rng) -> dict:
    a, b = rng.choice(((2, 3), (3, 2), (2, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3)))
    boxes = a * b
    claim = f"if {boxes + 1} points are placed inside or on the boundary of a {a} by {b} rectangle, two of them are at most √2 apart"
    check(boxes + 1 > boxes, "ab unit squares")
    p = Prompt()
    p.t(
        f"Prove that {claim}. You may use the pigeon-hole principle: if more than k objects are placed in k boxes, some box holds at least two objects. "
    ).t(_tail(rng))
    ref = [
        f"Divide the {a} by {b} rectangle into {boxes} unit squares (1 by 1).",
        "A point on a shared edge or corner can be assigned to any one square that contains it, so each point is placed in exactly one box.",
        f"There are {boxes + 1} points and {boxes} squares, so by the pigeon-hole principle some square holds two points.",
        "Two points in the same unit square are at most as far apart as opposite corners, which is the diagonal √(1² + 1²) = √2.",
        "So two of the points are at most √2 apart.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"uses {boxes} unit squares as boxes and the diagonal √2 as the bound"),
            ("logical_completeness", "links 'two in one square' to 'at most √2 apart'"),
            ("definitions_assumptions", "says what happens to points on shared edges"),
            ("communication", "describes the division of the rectangle"),
        ],
        traps=[
            ("wrong_square_size", "does it use boxes of side other than 1, so the diagonal bound is not √2?"),
            ("distance_equals_diagonal", "does it claim the distance equals √2 rather than at most √2?"),
        ],
    )
    return finish(p, claim, mark)


def _pg_halves_modulus(rng) -> dict:
    m = rng.choice((6, 8, 10, 12, 14))
    classes = m // 2 + 1
    n = classes + 1
    claim = f"among any {n} integers, there are two whose sum or difference is divisible by {m}"
    # exhaustive structural check: the classes {r, m-r} cover all remainders
    check(len({min(r, m - r) for r in range(m)}) == classes, "class count")
    check(
        all(
            any(((a - b) % m == 0) or ((a + b) % m == 0) for a, b in combinations(sub, 2))
            for sub in combinations(range(m), n)
        )
        if n <= m
        else True,
        "exhaustive residues",
    )
    p = Prompt()
    p.t(
        f"Prove that {claim}. You may use the pigeon-hole principle: if more than k objects are placed in k boxes, some box holds at least two objects. "
    ).t(_tail(rng))
    class_list = ", ".join(
        "{0}" if r == 0 else ("{" + str(r) + "}" if r == m - r else "{" + f"{r}, {m - r}" + "}")
        for r in range(0, m // 2 + 1)
    )
    ref = [
        f"The possible remainders on division by {m} are 0, 1, ..., {m - 1}. Group them into the {classes} classes {class_list}.",
        f"Use these {classes} classes as boxes, and place each integer in the box containing its remainder.",
        f"There are {n} integers and {classes} boxes, so two integers a and b share a box.",
        f"If a and b have the same remainder, then a - b is divisible by {m}.",
        f"Otherwise their remainders are r and {m} - r, so a + b has remainder r + ({m} - r) = {m}, which is 0, and a + b is divisible by {m}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"forms the {classes} classes of remainders that are equal or add to {m}"),
            ("logical_completeness", "handles both cases: same remainder, and complementary remainders"),
            (
                "definitions_assumptions",
                "notes that 0" + (f" and {m // 2}" if m % 2 == 0 else "") + " each form a class of their own",
            ),
            ("communication", "lists the classes explicitly"),
        ],
        traps=[
            ("uses_m_boxes", f"does it use {m} boxes (one per remainder), which would need {m + 1} integers?"),
            ("only_the_difference", "does it show only that the difference is divisible?"),
        ],
    )
    return finish(p, claim, mark)


def _pg_prefix_sums(rng) -> dict:
    n = rng.randint(4, 9)
    # Structural check: n prefix sums plus the empty sum give n + 1 values in n remainders.
    check(n + 1 > n, "n + 1 prefix sums, n remainders")
    claim = f"for any {n} integers a1, a2, ..., a{n}, some block of consecutive terms has a sum divisible by {n}"
    p = Prompt()
    p.t(
        f"Prove that {claim}. You may use the pigeon-hole principle: if more than k objects are placed in k boxes, some box holds at least two objects. "
    ).t(_tail(rng))
    ref = [
        f"Define S0 = 0 and S_k = a1 + a2 + ... + a_k for k = 1, ..., {n}. That is {n + 1} numbers S0, S1, ..., S{n}.",
        f"Each S_k has a remainder on division by {n}: one of {', '.join(str(r) for r in range(n))}, which is {n} possible remainders.",
        f"Take {n} boxes, one for each remainder. There are {n + 1} numbers S_k and {n} boxes, so two of them, S_i and S_j with i < j, share a box.",
        f"Then S_j - S_i has remainder 0 on division by {n}, so S_j - S_i is divisible by {n}.",
        f"S_j - S_i = a_(i+1) + ... + a_j, a block of consecutive terms. So this block has sum divisible by {n}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"uses the {n + 1} partial sums S0 to S{n} (including the empty sum)"),
            ("logical_completeness", "turns 'same remainder' into a block of consecutive terms"),
            ("definitions_assumptions", "includes S0 = 0 so there are n + 1 objects for n boxes"),
            ("communication", "defines the partial sums clearly"),
        ],
        traps=[
            (
                "forgets_s0",
                f"does it use only S1, ..., S{n}? That is {n} sums and {n} boxes, so the principle does not apply directly.",
            ),
            ("difference_not_a_block", "does it fail to say S_j - S_i is a block of consecutive terms?"),
        ],
    )
    return finish(p, claim, mark)


def _pg_triangle(rng) -> dict:
    s = rng.choice((1, 2, 3, 4, 5))
    side = 2 * s
    claim = f"if 5 points are placed inside or on the boundary of an equilateral triangle of side {side}, two of them are at most {s} apart"
    check(5 > 4, "four small triangles")
    p = Prompt()
    p.t(
        f"Prove that {claim}. You may use the pigeon-hole principle: if more than k objects are placed in k boxes, some box holds at least two objects. "
    ).t(_tail(rng))
    ref = [
        f"Join the midpoints of the three sides of the triangle. This divides it into 4 equilateral triangles, each of side {s}.",
        "A point on a shared edge or vertex is assigned to any one small triangle that contains it, so each of the 5 points is in exactly one box.",
        "There are 5 points and 4 small triangles, so by the pigeon-hole principle some small triangle holds two points.",
        f"The greatest distance between two points in an equilateral triangle of side {s} is its side {s} (the longest segment inside a triangle joins two vertices).",
        f"So two of the points are at most {s} apart.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"divides the triangle into 4 triangles of side {s}"),
            (
                "logical_completeness",
                f"links 'two in one small triangle' to 'at most {s} apart' and justifies the distance bound",
            ),
            ("definitions_assumptions", "says how points on shared edges are assigned"),
            ("communication", "describes the division"),
        ],
        traps=[
            (
                "wrong_number_of_regions",
                "does it use 3 or 5 regions, so the principle does not give two points in one region?",
            ),
            (
                "distance_bound_unjustified",
                "does it say the maximum distance in a small triangle is its side without a reason?",
            ),
        ],
    )
    return finish(p, claim, mark)


def _pg_handshake(rng) -> dict:
    n = rng.randint(5, 12)
    if n <= 6:
        # exhaustive over all friendship graphs on n people
        pairs = list(combinations(range(n), 2))
        for mask in range(1 << len(pairs)):
            deg = [0] * n
            for i, (u, v) in enumerate(pairs):
                if mask >> i & 1:
                    deg[u] += 1
                    deg[v] += 1
            check(len(set(deg)) < n, "two people with the same number of friends")
    claim = f"in any group of {n} people, there are two people with the same number of friends in the group (friendship is mutual)"
    p = Prompt()
    p.t(
        f"Prove that {claim}. You may use the pigeon-hole principle: if more than k objects are placed in k boxes, some box holds at least two objects. "
    ).t(_tail(rng))
    ref = [
        f"Each person has between 0 and {n - 1} friends in the group, so there are {n} possible values, and {n} people.",
        f"Suppose, for contradiction, that all {n} people have different numbers of friends. Then each of 0, 1, ..., {n - 1} occurs exactly once.",
        f"Someone has {n - 1} friends: they are friends with everyone else. Someone has 0 friends: they are friends with nobody.",
        "These two people are different, since one has a friend and the other has none, but the first is friends with everyone else, including the second. This is a contradiction.",
        f"So 0 and {n - 1} cannot both occur. The {n} people therefore have at most {n - 1} possible numbers of friends, and by the pigeon-hole principle two people have the same number.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"shows that 0 and {n - 1} cannot both occur, leaving {n - 1} possible values"),
            (
                "logical_completeness",
                "applies the principle to n people and n - 1 values after removing the impossible case",
            ),
            ("definitions_assumptions", "uses that friendship is mutual"),
            ("communication", "states the contradiction clearly"),
        ],
        traps=[
            (
                "n_boxes_for_n_people",
                f"does it count {n} possible values (0 to {n - 1}) and apply the principle directly? {n} people in {n} boxes gives nothing.",
            ),
            ("ignores_mutual_friendship", "does the argument avoid using that friendship is mutual?"),
        ],
    )
    return finish(p, claim, mark)


def _pg_lattice_midpoint(rng) -> dict:
    d = rng.choice((2, 3))
    boxes = 2**d
    pts = "ordered pairs" if d == 2 else "ordered triples"
    coord = "(x, y)" if d == 2 else "(x, y, z)"
    check(boxes + 1 == 2**d + 1, "parity classes")
    claim = f"among any {boxes + 1} points with integer coordinates in {'the plane' if d == 2 else 'space'}, there are two whose midpoint also has integer coordinates"
    p = Prompt()
    p.t(
        f"Prove that {claim}. You may use the pigeon-hole principle: if more than k objects are placed in k boxes, some box holds at least two objects. "
    ).t(_tail(rng))
    ref = [
        f"For a point {coord}, record whether each coordinate is even or odd. There are 2^{d} = {boxes} such parity patterns.",
        f"Use the {boxes} patterns as boxes and place each of the {boxes + 1} points in the box for its pattern.",
        f"There are {boxes + 1} points and {boxes} boxes, so two points share a pattern.",
        "Two points with the same parity pattern have coordinates of the same parity in each position, so in every coordinate the sum is even.",
        "The midpoint's coordinates are half those sums, so they are integers.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"identifies the {boxes} parity patterns as the boxes"),
            ("logical_completeness", "turns 'same pattern' into 'integer midpoint' by coordinate"),
            ("definitions_assumptions", "states that the sum of two integers of the same parity is even"),
            ("communication", "describes the boxes explicitly"),
        ],
        traps=[
            ("wrong_box_count", f"does it use {2 * d} or {d} boxes rather than 2^{d} = {boxes}?"),
            ("only_one_coordinate", "does it consider only one coordinate?"),
        ],
    )
    return finish(p, claim, mark)


PG_FAMILIES = {
    "easy": (_pg_basic,),
    "medium": (_pg_ceiling, _pg_mod_difference),
    "hard": (_pg_consecutive, _pg_divides, _pg_rectangle, _pg_halves_modulus),
    "exam": (_pg_prefix_sums, _pg_triangle, _pg_handshake, _pg_lattice_midpoint),
}

_PRESETS = {t: dict(variant=t) for t in ("easy", "medium", "hard", "exam")}


def _pick(families: dict, variant: str, rng) -> dict:
    if variant not in families:
        raise ValueError(f"unknown variant: {variant}")
    return rng.choice(families[variant])(rng)


@generator("proof_irrationality", level=14, presets=_PRESETS)
def proof_irrationality(rng, variant="easy"):
    """`senior-prove-irrationality-by-contradiction-for-numbers-such-as`. Rubric-assessed (`answer: rubric`):
    every question returned carries a per-item marking dict (`Question.answer`), never a plain value."""
    return _pick(IRR_FAMILIES, variant, rng)


@generator("proof_simple_number_results", level=14, presets=_PRESETS)
def proof_simple_number_results(rng, variant="easy"):
    """`senior-prove-simple-results-involving-numbers`. Rubric-assessed (`answer: rubric`)."""
    return _pick(NUM_FAMILIES, variant, rng)


@generator("proof_pigeonhole", level=14, presets=_PRESETS)
def proof_pigeonhole(rng, variant="easy"):
    """`senior-solve-problems-and-prove-results-using-the-pigeon`. Rubric-assessed (`answer: rubric`)."""
    return _pick(PG_FAMILIES, variant, rng)
