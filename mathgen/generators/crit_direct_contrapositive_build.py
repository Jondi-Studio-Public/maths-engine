"""Generates `proof_direct_contrapositive` items: a short direct or contrapositive proof of a stated implication
about integers.

Every question states an implication P(n) => Q(n) about integers whose truth depends only on residues, and names
the method (direct, contrapositive, or "choose"). The proof is marked against a rubric, so `answer` is a marking
dict (see `_crit_kit`), not a value.

Ladder:

- easy: parity. Direct: n has a stated parity, so a polynomial in n has a stated parity. Contrapositive: a
  linear expression in n has a stated parity, so n has one.
- medium: one modulus m in {3, ..., 7}. Direct: n = mk + r gives f(n) = f(r) + m g(k). Contrapositive: a linear
  expression leaves a stated remainder, so n leaves a stated remainder.
- hard: a quadratic expression in n that leaves a stated remainder forces n into one or two residue classes, or
  a prime divides n^2; the proof is a table of the residues.
- exam: two integers a and b, divisibility of a product or a sum of squares, and the choice of method.

Every implication asked is true: each is checked over the residues modulo m (which decides it for every integer)
and again over the integers -60 to 60.
"""

from __future__ import annotations

from math import gcd

from ..core import check, generator
from ._crit_kit import Fm, Prompt, TAILS, attempts, finish, marking, poly_eval, poly_fm

_PRESETS = {
    "easy": {"variant": "easy"},
    "medium": {"variant": "medium"},
    "hard": {"variant": "hard"},
    "exam": {"variant": "exam"},
}
_RANGE = range(-60, 61)


def _tail(rng) -> str:
    return rng.choice(TAILS)


def _vp(m: int, rs: list[int]) -> str:
    """The verb phrase for 'lies in the residue classes `rs` modulo m': is even, is divisible by 5, has remainder 2 ..."""
    if m == 2 and rs == [0]:
        return "is even"
    if m == 2 and rs == [1]:
        return "is odd"
    if rs == [0]:
        return f"is divisible by {m}"
    return "has remainder " + " or ".join(map(str, rs)) + f" when divided by {m}"


def _plain(subject: Fm, m: int, rs: list[int]) -> str:
    return f"{subject.plain} {_vp(m, rs)}"


N = Fm("n")
A = Fm("a")
B = Fm("b")
NSQ = Fm("n^{2}")


def _implication(p: Prompt, lead: str, subj_p: Fm, mp: int, rp: list[int], subj_q: Fm, mq: int, rq: list[int]) -> None:
    p.t("Let ").m(N).t(lead).m(subj_p).t(" " + _vp(mp, rp) + ", then ").m(subj_q).t(" " + _vp(mq, rq) + ". ")


def _claim(subj_p: Fm, mp: int, rp: list[int], subj_q: Fm, mq: int, rq: list[int]) -> str:
    return f"If {_plain(subj_p, mp, rp)} then {_plain(subj_q, mq, rq)}"


def _eval_mod(coeffs: list[int], x: int, m: int) -> int:
    return poly_eval(coeffs, x) % m


def _solution_set(coeffs: list[int], m: int, c: int) -> list[int]:
    return [x for x in range(m) if _eval_mod(coeffs, x, m) == c]


def _brute_implication(coeffs: list[int], m: int, c: int, rs: list[int]) -> bool:
    return all(n % m in rs for n in _RANGE if poly_eval(coeffs, n) % m == c)


def _brute_direct(coeffs: list[int], m: int, r: int, c: int) -> bool:
    return all(poly_eval(coeffs, n) % m == c for n in _RANGE if n % m == r)


def _lead(method: str) -> str:
    return {
        "direct": " be an integer. Prove directly that if ",
        "contrapositive": " be an integer. Prove by the contrapositive that if ",
        "choose": " be an integer. Choose either a direct proof or a proof by the contrapositive, say which you use, and prove that if ",
    }[method]


def _g_coeffs(coeffs: list[int], m: int, r: int) -> list[int]:
    """Ascending coefficients of g with f(mk + r) = f(r) + m g(k), by expanding the powers of (mk + r)."""
    total = [0] * len(coeffs)
    for i, c in enumerate(coeffs):
        term = [1]
        for _ in range(i):
            nxt = [0] * (len(term) + 1)
            for j, t in enumerate(term):
                nxt[j] += t * r
                nxt[j + 1] += t * m
            term = nxt
        for j, t in enumerate(term):
            total[j] += c * t
    check(total[0] == poly_eval(coeffs, r), "constant term of f(mk + r) is f(r)")
    check(all(t % m == 0 for t in total[1:]), "every k term carries the modulus")
    return [0] + [t // m for t in total[1:]]


# ---------------------------------------------------------------------------
# Direct: n in a residue class, so f(n) in a residue class.
# ---------------------------------------------------------------------------
def _direct_modular(rng, m_choices, degree) -> dict:
    for _ in attempts(500, "a nontrivial polynomial"):
        m = rng.choice(m_choices)
        r = rng.randrange(m)
        coeffs = [rng.randint(-5, 6) for _ in range(degree + 1)]
        coeffs[degree] = rng.choice((1, 2, 3, 4, 5, -1, -2, -3))
        if all(c % m == 0 for c in coeffs[1:]):
            continue
        # the hypothesis must matter: some other residue class gives a different value
        if all(poly_eval(coeffs, t) % m == poly_eval(coeffs, r) % m for t in range(m)):
            continue
        break
    c = poly_eval(coeffs, r) % m
    check(_brute_direct(coeffs, m, r, c), "the direct implication is true")
    f = poly_fm(coeffs, "n")
    g = _g_coeffs(coeffs, m, r)
    check(
        all(poly_eval(coeffs, m * k + r) == poly_eval(coeffs, r) + m * poly_eval(g, k) for k in range(-7, 8)),
        "f(mk + r) = f(r) + m g(k)",
    )
    gf = poly_fm(g, "k")
    p = Prompt()
    _implication(p, _lead("direct"), N, m, [r], f, m, [c])
    p.t(_tail(rng))
    if m == 2:
        form_n = "2k" if r == 0 else "2k + 1"
    else:
        form_n = f"{m}k" if r == 0 else f"{m}k + {r}"
    f0 = poly_eval(coeffs, r)
    ref = [
        f"Suppose {_plain(N, m, [r])}. Then n = {form_n} for some integer k.",
        f"Substitute n = {form_n} into {f.plain}. Expanding, the result is {f0} + {m}({gf.plain}), because every term that contains k carries a factor of {m}.",
        f"The bracket {gf.plain} is an integer, so the expression is {f0} plus a multiple of {m}.",
        f"{f0} leaves remainder {c} when divided by {m}, so {_plain(f, m, [c])}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"writes n = {form_n} and expands the expression correctly, reaching {f0} + {m}(...)"),
            (
                "logical_completeness",
                "shows the remaining terms are a multiple of the modulus, with the bracket an integer",
            ),
            ("definitions_assumptions", "states that k is an integer and uses the definition of the residue class"),
            ("communication", "ends by stating the conclusion about the expression"),
        ],
        traps=[
            (
                "checks_examples_only",
                "does the answer test a few values of n and stop, instead of using n = mk + r for every such n?",
            ),
            ("loses_the_constant", f"does it keep the constant {f0} and reduce it modulo {m}?"),
            ("proves_the_converse", "does it prove the stated direction and not its converse?"),
        ],
    )
    return finish(p, _claim(N, m, [r], f, m, [c]), mark)


# ---------------------------------------------------------------------------
# Contrapositive: f(n) in a class forces n into a class.
# ---------------------------------------------------------------------------
def _contra_poly(rng, m_choices, degree, *, linear_only=False, max_classes=1) -> dict:
    for _ in attempts(3000, "a polynomial whose class forces n"):
        m = rng.choice(m_choices)
        coeffs = [rng.randint(-6, 7) for _ in range(degree + 1)]
        coeffs[degree] = rng.choice((1, 2, 3, 4, 5, 6, -1, -2, -3))
        if linear_only and gcd(coeffs[1] % m, m) != 1:
            continue
        c = rng.randrange(m)
        sol = _solution_set(coeffs, m, c)
        if not sol or len(sol) > max_classes or len(sol) == m:
            continue
        break
    check(_brute_implication(coeffs, m, c, sol), "the implication holds for integers")
    others = [x for x in range(m) if x not in sol]
    check(all(_eval_mod(coeffs, x, m) != c for x in others), "no other residue reaches the target class")
    return dict(m=m, coeffs=coeffs, c=c, sol=sol, f=poly_fm(coeffs, "n"), others=others)


def _contrapositive_item(rng, d: dict, method: str) -> dict:
    m, coeffs, c, sol, f, others = d["m"], d["coeffs"], d["c"], d["sol"], d["f"], d["others"]
    p = Prompt()
    _implication(p, _lead(method), f, m, [c], N, m, sol)
    p.t(_tail(rng))
    table = ", ".join(f"n = {x} gives {_eval_mod(coeffs, x, m)}" for x in others)
    ref = [
        f"The contrapositive is: if it is not true that {_plain(N, m, sol)}, then it is not true that {_plain(f, m, [c])}.",
        f"Suppose n is not in the class. Then n leaves one of the remainders {', '.join(map(str, others))} when divided by {m}.",
        f"The expression {f.plain} leaves a remainder that depends only on the remainder of n, so check each remainder (all modulo {m}): {table}.",
        f"None of these remainders is {c}, so it is not true that {_plain(f, m, [c])}. The contrapositive is proved, so the original statement is true.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            (
                "correctness",
                f"states the contrapositive correctly and finds {f.plain} modulo {m} for each remainder {', '.join(map(str, others))}",
            ),
            ("logical_completeness", f"covers every remainder outside the class, none left out ({len(others)} cases)"),
            ("definitions_assumptions", f"says why the expression modulo {m} depends only on n modulo {m}"),
            ("communication", "labels the cases and states that proving the contrapositive proves the original"),
        ],
        traps=[
            (
                "converse_for_contrapositive",
                "is the statement it proves the converse, 'if n is in the class then the expression leaves that remainder'?",
            ),
            ("missing_case", f"does it deal with all {len(others)} remainders that n can leave outside the class?"),
            (
                "wrong_negation",
                "does it negate the conclusion as 'n is not in the class' and not as 'n is in one other single class'?",
            ),
        ],
    )
    return finish(p, _claim(f, m, [c], N, m, sol), mark)


# ---------------------------------------------------------------------------
# Easy / medium
# ---------------------------------------------------------------------------
def _easy_direct(rng) -> dict:
    return _direct_modular(rng, (2,), rng.choice((1, 2)))


def _easy_contra(rng) -> dict:
    return _contrapositive_item(rng, _contra_poly(rng, (2,), 1, linear_only=True), "contrapositive")


def _medium_direct(rng) -> dict:
    return _direct_modular(rng, (3, 4, 5, 6, 7), rng.choice((1, 2)))


def _medium_contra(rng) -> dict:
    return _contrapositive_item(rng, _contra_poly(rng, (3, 4, 5, 6, 7), 1, linear_only=True), "contrapositive")


# ---------------------------------------------------------------------------
# Hard
# ---------------------------------------------------------------------------
def _hard_contra(rng) -> dict:
    return _contrapositive_item(rng, _contra_poly(rng, (3, 4, 5, 6, 7, 8), 2, max_classes=2), "contrapositive")


def _hard_prime_square(rng) -> dict:
    """If a prime p divides n^2 then p divides n, by the table of squares."""
    prime = rng.choice((3, 5, 7, 11, 13))
    squares = {x: (x * x) % prime for x in range(1, prime)}
    check(all(v != 0 for v in squares.values()), "no nonzero residue squares to zero modulo a prime")
    check(all(n % prime == 0 for n in _RANGE if (n * n) % prime == 0), "p | n^2 implies p | n")
    table = ", ".join(f"{x}^2 leaves {v}" for x, v in squares.items())
    pr = Prompt()
    _implication(pr, _lead("contrapositive"), NSQ, prime, [0], N, prime, [0])
    pr.t(_tail(rng))
    ref = [
        f"The contrapositive is: if n is not divisible by {prime}, then n^2 is not divisible by {prime}.",
        f"Suppose n is not divisible by {prime}. Then n leaves one of the remainders 1 to {prime - 1} when divided by {prime}.",
        f"Squaring each remainder and reducing modulo {prime}: {table}.",
        f"None of these is 0, so n^2 is not divisible by {prime}. The contrapositive holds, so the original statement is true.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            (
                "correctness",
                f"states the contrapositive and shows the square of each remainder 1 to {prime - 1} is not 0 modulo {prime}",
            ),
            ("logical_completeness", f"covers all {prime - 1} nonzero remainders (or argues from {prime} being prime)"),
            ("definitions_assumptions", "says n^2 leaves a remainder that depends only on the remainder of n"),
            ("communication", "concludes that the contrapositive proves the original"),
        ],
        traps=[
            ("only_a_few_remainders", f"does it check every remainder from 1 to {prime - 1}, or only some?"),
            (
                "converse_for_contrapositive",
                "does it prove that if the prime divides n then it divides n^2, which is the converse?",
            ),
        ],
    )
    return finish(pr, _claim(NSQ, prime, [0], N, prime, [0]), mark)


# ---------------------------------------------------------------------------
# Exam
# ---------------------------------------------------------------------------
def _exam_sum_squares(rng) -> dict:
    prime = rng.choice((3, 7))
    sq = {x: (x * x) % prime for x in range(prime)}
    check(
        all(
            a % prime == 0 and b % prime == 0
            for a in range(-16, 17)
            for b in range(-16, 17)
            if (a * a + b * b) % prime == 0
        ),
        "a^2 + b^2 divisible by p forces both divisible",
    )
    nonzero = sorted({v for x, v in sq.items() if x})
    sums = sorted({(u + v) % prime for u in nonzero for v in [0] + nonzero})
    check(0 not in sums, "sums of squares never vanish")
    pr = Prompt()
    pr.t("Let ").m(A).t(" and ").m(B).t(
        " be integers. Choose either a direct proof or a proof by the contrapositive, say which you use, and prove that if "
    )
    pr.m(Fm("a^{2}+b^{2}", "a^2 + b^2")).t(f" is divisible by {prime}, then both ").m(A).t(" and ").m(B).t(
        f" are divisible by {prime}. "
    ).t(_tail(rng))
    ref = [
        f"Use the contrapositive: if a or b is not divisible by {prime}, then a^2 + b^2 is not divisible by {prime}.",
        f"The squares of integers leave these remainders modulo {prime}: "
        + ", ".join(f"{x}^2 leaves {sq[x]}" for x in range(prime))
        + f". So a square leaves 0 (if the integer is a multiple of {prime}) or one of {', '.join(map(str, nonzero))}.",
        f"Suppose a is not divisible by {prime} (the case for b is the same). Then a^2 leaves a remainder from {', '.join(map(str, nonzero))}, and b^2 leaves 0 or a remainder from the same list.",
        f"Adding, a^2 + b^2 leaves one of {', '.join(map(str, sums))} modulo {prime}, never 0.",
        f"So a^2 + b^2 is not divisible by {prime}. The contrapositive is proved.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            (
                "correctness",
                f"lists the squares modulo {prime} and shows a nonzero square plus any square is never 0 modulo {prime}",
            ),
            ("logical_completeness", "handles a nonzero with b zero, b nonzero with a zero, and both nonzero"),
            (
                "definitions_assumptions",
                f"uses that the square of an integer is determined by its remainder modulo {prime}",
            ),
            ("communication", "states which method it chose and why"),
        ],
        traps=[
            (
                "only_both_nonzero",
                "does it also cover the case where exactly one of a and b is a multiple of the modulus?",
            ),
            ("example_for_proof", "does it test a few pairs (a, b) and conclude?"),
            (
                "wrong_negation",
                "does the contrapositive start 'a or b is not divisible' and not 'a and b are both not divisible'?",
            ),
        ],
    )
    return finish(pr, f"If a^2 + b^2 is divisible by {prime} then both a and b are divisible by {prime}", mark)


def _exam_product(rng) -> dict:
    prime = rng.choice((2, 3, 5))
    check(
        all(a % prime == 0 or b % prime == 0 for a in range(-30, 31) for b in range(-30, 31) if (a * b) % prime == 0),
        "p | ab implies p | a or p | b",
    )
    nz = list(range(1, prime))
    check(all((x * y) % prime for x in nz for y in nz), "no product of nonzero residues vanishes")
    pr = Prompt()
    pr.t("Let ").m(A).t(" and ").m(B).t(
        " be integers. Choose either a direct proof or a proof by the contrapositive, say which you use, and prove that if "
    )
    pr.m(Fm("ab")).t(f" is divisible by {prime}, then ").m(A).t(f" is divisible by {prime} or ").m(B).t(
        f" is divisible by {prime}. "
    ).t(_tail(rng))
    ref = [
        f"Use the contrapositive: if neither a nor b is divisible by {prime}, then ab is not divisible by {prime}.",
        f"Suppose neither is divisible by {prime}. Then a and b each leave a remainder from {', '.join(map(str, nz))} when divided by {prime}.",
        "The product leaves the product of the remainders, reduced: "
        + ", ".join(f"{x} x {y} leaves {(x * y) % prime}" for x in nz for y in nz)
        + ".",
        f"No product is 0 modulo {prime}, so ab is not divisible by {prime}. The contrapositive is proved.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "negates the conclusion correctly: neither a nor b is divisible"),
            (
                "logical_completeness",
                f"checks all {len(nz) ** 2} pairs of nonzero remainders (or argues from the modulus being prime)",
            ),
            (
                "definitions_assumptions",
                "uses that the remainder of a product is the reduced product of the remainders",
            ),
            ("communication", "states which method it chose and why"),
        ],
        traps=[
            (
                "or_negated_as_or",
                "does the negation of 'a or b is divisible' become 'neither is divisible', and not 'one of them is not divisible'?",
            ),
            ("one_example", "does it rely on a numerical example as the proof?"),
        ],
    )
    return finish(
        pr, f"If ab is divisible by {prime} then a is divisible by {prime} or b is divisible by {prime}", mark
    )


def _exam_square_minus_one(rng) -> dict:
    prime = rng.choice((3, 5, 7, 11))
    sol = [x for x in range(prime) if (x * x - 1) % prime == 0]
    check(sol == [1, prime - 1], "the roots of x^2 = 1 are +-1 modulo a prime")
    check(all(n % prime in sol for n in _RANGE if (n * n - 1) % prime == 0), "the implication holds for integers")
    others = [x for x in range(prime) if x not in sol]
    pr = Prompt()
    _implication(pr, _lead("choose"), Fm("n^{2}-1", "n^2 - 1"), prime, [0], N, prime, sol)
    pr.t(_tail(rng))
    ref = [
        f"Use the contrapositive: if n leaves a remainder other than 1 and {prime - 1} when divided by {prime}, then n^2 - 1 is not divisible by {prime}.",
        f"The other remainders are {', '.join(map(str, others))}.",
        f"Compute n^2 - 1 modulo {prime} for each: "
        + ", ".join(f"n = {x} gives {(x * x - 1) % prime}" for x in others)
        + ".",
        f"None is 0, so n^2 - 1 is not divisible by {prime}. The contrapositive is proved.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            (
                "correctness",
                f"states the contrapositive and finds n^2 - 1 modulo {prime} for each remainder {', '.join(map(str, others))}",
            ),
            ("logical_completeness", f"covers all {len(others)} remainders, none left out"),
            ("definitions_assumptions", f"says n^2 - 1 depends only on the remainder of n modulo {prime}"),
            ("communication", "states the method chosen and why"),
        ],
        traps=[
            (
                "converse_for_contrapositive",
                f"does it only show that remainders 1 and {prime - 1} give a multiple (the converse)?",
            ),
            ("missing_case", f"does it cover every remainder outside 1 and {prime - 1}?"),
        ],
    )
    return finish(pr, _claim(Fm("n^2 - 1", "n^2 - 1"), prime, [0], N, prime, sol), mark)


def _exam_not_divisible_square(rng) -> dict:
    """If n is not divisible by p then n^2 - 1 is divisible by 3 or 8 (direct, by cases)."""
    prime = rng.choice((3, 5))
    if prime == 3:
        check(all((n * n - 1) % 3 == 0 for n in _RANGE if n % 3), "n not divisible by 3 gives 3 | n^2 - 1")
        pr = Prompt()
        pr.t("Let ").m(N).t(" be an integer. Prove that if ").m(N).t(" is not divisible by 3, then ").m(
            Fm("n^{2}-1", "n^2 - 1")
        ).t(" is divisible by 3. ").t(_tail(rng))
        ref = [
            "Direct proof by cases. Suppose n is not divisible by 3, so n = 3k + 1 or n = 3k + 2 for some integer k.",
            "If n = 3k + 1 then n^2 - 1 = 9k^2 + 6k = 3(3k^2 + 2k), a multiple of 3.",
            "If n = 3k + 2 then n^2 - 1 = 9k^2 + 12k + 3 = 3(3k^2 + 4k + 1), a multiple of 3.",
            "Both cases give a multiple of 3, and these are the only cases, so n^2 - 1 is divisible by 3.",
        ]
        mark = marking(
            reference=ref,
            steps=[
                ("correctness", "expands (3k + 1)^2 - 1 and (3k + 2)^2 - 1 correctly and factors out 3"),
                ("logical_completeness", "covers both remainder cases and says they are the only ones"),
                ("definitions_assumptions", "writes n as 3k + 1 or 3k + 2 with k an integer"),
                ("communication", "states the method (direct, by cases) and the conclusion"),
            ],
            traps=[
                ("one_case_only", "does it treat both n = 3k + 1 and n = 3k + 2?"),
                ("factorisation_shortcut", "if it uses (n - 1)(n + 1), does it say why one factor is a multiple of 3?"),
            ],
        )
        return finish(pr, "If n is not divisible by 3 then n^2 - 1 is divisible by 3", mark)
    check(all((n**4 - 1) % 5 == 0 for n in _RANGE if n % 5), "n not divisible by 5 gives 5 | n^4 - 1")
    pr = Prompt()
    pr.t("Let ").m(N).t(" be an integer. Prove that if ").m(N).t(" is not divisible by 5, then ").m(
        Fm("n^{4}-1", "n^4 - 1")
    ).t(" is divisible by 5. ").t(_tail(rng))
    sq = {x: (x * x) % 5 for x in range(1, 5)}
    ref = [
        "Direct proof by cases on the remainder. Suppose n is not divisible by 5, so n leaves remainder 1, 2, 3 or 4.",
        "Squares modulo 5: " + ", ".join(f"{x}^2 leaves {v}" for x, v in sq.items()) + ", so n^2 leaves 1 or 4.",
        "Then n^4 = (n^2)^2 leaves 1^2 = 1 or 4^2 = 16, which also leaves 1.",
        "So n^4 - 1 leaves remainder 0 in every case and is divisible by 5.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "finds n^4 modulo 5 as 1 for each remainder 1 to 4"),
            (
                "logical_completeness",
                "covers all four nonzero remainders or squares in two stages and covers both square classes",
            ),
            ("definitions_assumptions", "uses that n^4 depends only on n modulo 5"),
            ("communication", "states the method (direct, by cases) and the conclusion"),
        ],
        traps=[
            ("one_case_only", "does it treat all four nonzero remainders?"),
            (
                "quotes_fermat",
                "if it quotes Fermat's little theorem, is that allowed by the question, or does it prove the case?",
            ),
        ],
    )
    return finish(pr, "If n is not divisible by 5 then n^4 - 1 is divisible by 5", mark)


def _exam_poly_choice(rng) -> dict:
    return _contrapositive_item(rng, _contra_poly(rng, (5, 6, 7, 8, 9), 2, max_classes=2), "choose")


FAMILIES = {
    "easy": [_easy_direct, _easy_contra],
    "medium": [_medium_direct, _medium_contra],
    "hard": [_hard_contra, _hard_prime_square],
    "exam": [_exam_sum_squares, _exam_product, _exam_square_minus_one, _exam_not_divisible_square, _exam_poly_choice],
}


@generator("proof_direct_contrapositive", level=14, presets=_PRESETS)
def proof_direct_contrapositive(rng, variant="easy"):
    """`ucv1-direct-contrapositive-proof`. Rubric-assessed (`answer: rubric`): every question returned carries a
    per-item marking dict (`Question.answer`), never a plain value."""
    if variant not in FAMILIES:
        raise ValueError(f"unknown variant: {variant}")
    return rng.choice(FAMILIES[variant])(rng)
