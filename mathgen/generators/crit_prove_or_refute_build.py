"""Generates `proof_prove_or_refute` items: a universal claim over an explicit finite set of integers, to be proved
directly when it holds and refuted by a counterexample when it fails.

The claim is stated, never labelled true or false, and the learner decides. The truth of every claim is computed
here by exhaustive evaluation of the (finite) set, so each question knows whether it is a proof or a refutation,
and in the false case which elements are counterexamples. The answer is marked against a rubric, so `answer` is a
marking dict (see `_crit_kit`), not a value.

Ladder:

- easy: one property of one expression ("n^2 + n is divisible by 3") over a short list; roughly half true.
- medium: the same over a longer list, and Euler-style prime-generating expressions that hold for a run of values
  and then fail.
- hard: a property of pairs from the set ("a^2 - b^2 is divisible by 5 for all a and b in S"), true when every
  element lies in the right residue classes and false when one does not.
- exam: an implication over the set ("if n is divisible by 6 then n^2 is divisible by 8"), where elements that fail
  the hypothesis are vacuously fine and a counterexample must satisfy the hypothesis and break the conclusion.
"""

from __future__ import annotations

from itertools import product

from ..core import check, generator
from ._crit_kit import Fm, Prompt, attempts, finish, marking, poly_eval, poly_fm

_PRESETS = {
    "easy": {"variant": "easy"},
    "medium": {"variant": "medium"},
    "hard": {"variant": "hard"},
    "exam": {"variant": "exam"},
}

TAILS = (
    "Decide whether the claim is true. If it is true, write a short direct proof. If it is false, give a counterexample and show that it is one.",
    "Say whether the claim is true. Prove it directly if it is, or give a counterexample and check it if it is not.",
    "Is the claim true? If so, give a short direct proof; if not, exhibit a counterexample and show it breaks the claim.",
)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def _set_fm(values: list[int]) -> Fm:
    body = ", ".join(map(str, values))
    return Fm(f"S=\\{{{body}\\}}", f"S = {{{body}}}")


def _opening(p: Prompt, values: list[int]) -> None:
    p.t("Let ").m(_set_fm(values)).t(". ")


def _fin(p: Prompt, rng, claim: str, mark: dict) -> dict:
    p.t(rng.choice(TAILS))
    return finish(p, claim, mark)


def _verdict(truth: bool, bad: list[int]) -> str:
    return "The claim is true" if truth else f"The claim is false; counterexample n = {bad[0]}"


# ---------------------------------------------------------------------------
# Unary claims: "for every n in S, f(n) <predicate>"
# ---------------------------------------------------------------------------
def _divisible_claim(rng, size: int, want_true: bool, spread: int) -> dict:
    """f(n) divisible by k for every n in S, with S built to make the claim true or false as asked."""
    for _ in attempts(4000, "a divisibility claim with the wanted truth"):
        k = rng.choice((3, 4, 5, 6, 7, 8, 9))
        deg = rng.choice((1, 2, 2, 3))
        coeffs = [rng.randint(-4, 6) for _ in range(deg + 1)]
        coeffs[deg] = rng.choice((1, 2, 3, -1))
        good = [x for x in range(k) if poly_eval(coeffs, x) % k == 0]
        if not good or len(good) == k:
            continue
        pool = list(range(1, spread))
        in_class = [n for n in pool if n % k in good]
        out_class = [n for n in pool if n % k not in good]
        if len(in_class) < size or len(out_class) < 2:
            continue
        if want_true:
            values = sorted(rng.sample(in_class, size))
        else:
            n_bad = rng.choice((1, 1, 2))
            values = sorted(rng.sample(in_class, size - n_bad) + rng.sample(out_class, n_bad))
        bad = [n for n in values if poly_eval(coeffs, n) % k != 0]
        if (not bad) != want_true:
            continue
        break
    f = poly_fm(coeffs, "n")
    p = Prompt()
    _opening(p, values)
    p.t("Claim: for every ").m(Fm("n")).t(" in ").m(Fm("S")).t(", the number ").m(f).t(f" is divisible by {k}. ")
    ref: list[str]
    if want_true:
        calc = ", ".join(f"n = {n} gives {poly_eval(coeffs, n)} = {k} x {poly_eval(coeffs, n) // k}" for n in values)
        ref = [
            f"The claim is true. The set is finite, so check each element: {calc}.",
            f"Every value is a multiple of {k}, and every element of S has been checked, so the claim holds for all of S.",
        ]
        steps = [
            ("correctness", f"decides that the claim is true and evaluates {f.plain} correctly at every element of S"),
            (
                "logical_completeness",
                f"checks all {len(values)} elements of S (or gives a residue argument that covers them all)",
            ),
            ("definitions_assumptions", f"shows each value is a multiple of {k}, not just 'looks right'"),
            ("communication", "lists the values in a clear table or line per element"),
        ]
        traps = [
            ("checks_some_elements", f"does it check every one of the {len(values)} elements and not a sample?"),
            ("calls_it_false", "does it claim a counterexample exists? None does."),
        ]
    else:
        n0 = bad[0]
        val = poly_eval(coeffs, n0)
        ref = [
            f"The claim is false. Take n = {n0}, which is in S.",
            f"Then {f.plain} = {val}, and {val} = {k} x {val // k} + {val % k}, so it leaves remainder {val % k} and is not divisible by {k}.",
            "One element of S that breaks the claim is enough to show that the claim is false.",
        ]
        steps = [
            ("correctness", f"decides that the claim is false and names an element of S that fails, such as n = {n0}"),
            (
                "logical_completeness",
                f"evaluates {f.plain} at the counterexample ({val}) and shows it is not a multiple of {k}",
            ),
            ("definitions_assumptions", "uses an element that is in S"),
            ("communication", "states that one counterexample is enough"),
        ]
        traps = [
            ("counterexample_outside_the_set", "is the number it gives actually in S?"),
            (
                "arithmetic_slip",
                f"does the value of {f.plain} at its counterexample check out, and is it really not a multiple of {k}?",
            ),
            ("claims_true_from_examples", "does it check a few elements that work and call the claim true?"),
        ]
    mark = marking(reference=ref, steps=steps, traps=traps)
    return _fin(p, rng, _verdict(want_true, bad), mark)


# ---------------------------------------------------------------------------
# Prime-generating claims
# ---------------------------------------------------------------------------
_PRIME_FORMS = (
    # (coefficients ascending, label)
    ([11, 1, 1], "n^2 + n + 11"),
    ([17, 1, 1], "n^2 + n + 17"),
    ([41, 1, 1], "n^2 + n + 41"),
    ([1, 0, 1], "n^2 + 1"),
    ([29, 0, 2], "2n^2 + 29"),
    ([5, 2, 1], "n^2 + 2n + 5"),
    ([-1, 0, 6], "6n^2 - 1"),
    ([3, 0, 2], "2n^2 + 3"),
)


def _prime_claim(rng, size: int, want_true: bool) -> dict:
    for _ in attempts(4000, "a prime-generating claim with the wanted truth"):
        coeffs, _label = rng.choice(_PRIME_FORMS)
        pool = list(range(0 if coeffs[0] != 1 else 1, 45))
        good = [n for n in pool if _is_prime(poly_eval(coeffs, n))]
        badn = [n for n in pool if not _is_prime(poly_eval(coeffs, n))]
        if len(good) < size or not badn:
            continue
        if want_true:
            # a run of consecutive values from the start, then pad with isolated good values
            values = sorted(rng.sample(good[: max(size, min(len(good), 12))], size))
        else:
            first_bad = min(badn)
            run = [n for n in good if n < first_bad]
            if len(run) < size - 1:
                continue
            values = sorted(set(rng.sample(run, size - 1) + [first_bad]))
        bad = [n for n in values if not _is_prime(poly_eval(coeffs, n))]
        if (not bad) != want_true:
            continue
        break
    f = poly_fm(coeffs, "n")
    p = Prompt()
    _opening(p, values)
    p.t("Claim: for every ").m(Fm("n")).t(" in ").m(Fm("S")).t(", the number ").m(f).t(" is prime. ")
    if want_true:
        calc = ", ".join(f"n = {n} gives {poly_eval(coeffs, n)}" for n in values)
        ref = [
            f"The claim is true. The set is finite, so check each element: {calc}.",
            "Each value has no divisor other than 1 and itself (test the primes up to its square root), so each is prime; every element has been checked.",
        ]
        steps = [
            ("correctness", f"decides the claim is true and evaluates {f.plain} correctly at every element"),
            (
                "logical_completeness",
                "shows each value is prime by testing divisors up to its square root, for every element",
            ),
            ("definitions_assumptions", "uses the definition of prime, not just a recognised list"),
            ("communication", "presents one clear line per element"),
        ]
        traps = [
            ("misses_an_element", f"does it deal with all {len(values)} elements?"),
            ("calls_it_false", "does it claim a counterexample? None exists."),
        ]
    else:
        n0 = bad[0]
        v = poly_eval(coeffs, n0)
        d = next(d for d in range(2, v) if v % d == 0)
        ref = [
            f"The claim is false. Take n = {n0}, which is in S.",
            f"Then {f.plain} = {v} = {d} x {v // d}, which has a divisor other than 1 and itself, so it is not prime.",
            "One element of S that breaks the claim is enough.",
        ]
        steps = [
            ("correctness", f"decides the claim is false and names the failing element n = {n0}"),
            ("logical_completeness", f"evaluates {f.plain} at n = {n0} to get {v} and exhibits a factor ({d})"),
            ("definitions_assumptions", "uses an element that is in S"),
            ("communication", "states the counterexample and why it is enough"),
        ]
        traps = [
            (
                "claims_true_from_a_run",
                "does it test the earlier elements, find primes, and wrongly call the claim true?",
            ),
            ("no_factor_shown", f"does it show {v} has a factor, rather than just saying it is not prime?"),
        ]
    mark = marking(reference=ref, steps=steps, traps=traps)
    return _fin(p, rng, _verdict(want_true, bad), mark)


def _mersenne_claim(rng, size: int, want_true: bool) -> dict:
    primes_ok = [2, 3, 5, 7, 13, 17, 19, 31]
    primes_bad = [11, 23, 29, 37, 41]
    if want_true:
        values = sorted(rng.sample(primes_ok, size))
    else:
        values = sorted(rng.sample(primes_ok, size - 1) + [rng.choice(primes_bad)])
    bad = [n for n in values if not _is_prime(2**n - 1)]
    check((not bad) == want_true, "truth of the Mersenne claim")
    p = Prompt()
    _opening(p, values)
    p.t("Claim: for every ").m(Fm("n")).t(" in ").m(Fm("S")).t(", the number ").m(Fm("2^{n}-1", "2^n - 1")).t(
        " is prime. "
    )
    if want_true:
        ref = [
            "The claim is true. The set is finite, so check each element: "
            + ", ".join(f"n = {n} gives {2**n - 1}" for n in values)
            + ".",
            "Each of these has no divisor other than 1 and itself (test the primes up to its square root), so each is prime.",
        ]
        steps = [
            ("correctness", "decides the claim is true and computes 2^n - 1 correctly for each element"),
            (
                "logical_completeness",
                "tests primality of each value, including the larger ones, by checking divisors up to the square root",
            ),
            ("definitions_assumptions", "uses the definition of prime"),
            ("communication", "presents one clear line per element"),
        ]
        traps = [
            (
                "assumes_all_are_prime",
                "does it justify each primality and not assume that primes n give prime 2^n - 1?",
            ),
            ("calls_it_false", "does it claim a counterexample? None exists."),
        ]
    else:
        n0 = bad[0]
        v = 2**n0 - 1
        d = next(d for d in range(2, v) if v % d == 0)
        ref = [
            f"The claim is false. Take n = {n0}, which is in S.",
            f"Then 2^{n0} - 1 = {v} = {d} x {v // d}, so it is not prime.",
            "One element that breaks the claim is enough.",
        ]
        steps = [
            ("correctness", f"decides the claim is false and finds the failing element n = {n0}"),
            ("logical_completeness", f"computes 2^{n0} - 1 = {v} and exhibits a factor ({d})"),
            ("definitions_assumptions", "uses an element in S"),
            ("communication", "states the counterexample and why one is enough"),
        ]
        traps = [
            ("prime_exponent_is_not_enough", "does it argue 'n is prime so 2^n - 1 is prime'?"),
            ("no_factor_shown", f"does it show {v} has a factor?"),
        ]
    mark = marking(reference=ref, steps=steps, traps=traps)
    return _fin(p, rng, _verdict(want_true, bad), mark)


# ---------------------------------------------------------------------------
# Pair claims
# ---------------------------------------------------------------------------
def _pair_claim(rng, size: int, want_true: bool) -> dict:
    k = rng.choice((5, 7, 8, 9, 11, 13))
    # a^2 - b^2 divisible by k for all a, b in S iff all squares agree modulo k
    for _ in attempts(2000, "a pair claim with the wanted truth"):
        r = rng.randrange(1, k)
        class_sq = (r * r) % k
        same = [x for x in range(k) if (x * x) % k == class_sq]
        other = [x for x in range(k) if (x * x) % k != class_sq]
        pool = list(range(1, 4 * k))
        in_cls = [n for n in pool if n % k in same]
        out_cls = [n for n in pool if n % k in other]
        if len(in_cls) < size or not out_cls:
            continue
        if want_true:
            values = sorted(rng.sample(in_cls, size))
        else:
            values = sorted(rng.sample(in_cls, size - 1) + [rng.choice(out_cls)])
        bad = [(a, b) for a, b in product(values, values) if a < b and (a * a - b * b) % k != 0]
        if (not bad) != want_true:
            continue
        break
    p = Prompt()
    _opening(p, values)
    p.t("Claim: for all ").m(Fm("a")).t(" and ").m(Fm("b")).t(" in ").m(Fm("S")).t(", the number ").m(
        Fm("a^{2}-b^{2}", "a^2 - b^2")
    ).t(f" is divisible by {k}. ")
    sq_lines = ", ".join(f"{n}^2 leaves {(n * n) % k}" for n in values)
    if want_true:
        ref = [
            f"The claim is true. Reduce each element: modulo {k}, {sq_lines}.",
            f"Every square leaves the same remainder {class_sq} when divided by {k}, so any two squares differ by a multiple of {k}.",
            f"Hence a^2 - b^2 is divisible by {k} for every pair a, b in S (including a = b, where it is 0).",
        ]
        steps = [
            (
                "correctness",
                f"decides the claim is true and finds that every element's square leaves {class_sq} modulo {k}",
            ),
            (
                "logical_completeness",
                "turns 'all squares have the same remainder' into 'every difference of two squares is a multiple', covering all pairs",
            ),
            ("definitions_assumptions", "includes the case a = b or says it is trivial"),
            ("communication", "gives the remainder of each square in an organised list"),
        ]
        traps = [
            ("checks_only_some_pairs", "does it argue for all pairs, by residues, and not test a handful of pairs?"),
            ("calls_it_false", "does it claim a counterexample pair? None exists."),
        ]
    else:
        a0, b0 = bad[0]
        ref = [
            f"The claim is false. Take a = {a0} and b = {b0}, both in S.",
            f"Then a^2 - b^2 = {a0 * a0} - {b0 * b0} = {a0 * a0 - b0 * b0}, which leaves remainder {(a0 * a0 - b0 * b0) % k} when divided by {k}, so it is not divisible by {k}.",
            "One pair that breaks the claim is enough.",
        ]
        steps = [
            ("correctness", f"decides the claim is false and gives a pair in S that fails, such as ({a0}, {b0})"),
            ("logical_completeness", f"computes a^2 - b^2 = {a0 * a0 - b0 * b0} and shows it is not a multiple of {k}"),
            ("definitions_assumptions", "gives two elements of S, and says a pair is enough"),
            ("communication", "states the pair clearly with the working"),
        ]
        traps = [
            ("one_element_only", "does it give a pair (a, b), not a single number?"),
            ("pair_outside_the_set", "are both numbers in S?"),
            ("claims_true_from_pairs_that_work", "does it check a few pairs that work and call the claim true?"),
        ]
    mark = marking(reference=ref, steps=steps, traps=traps)
    verdict = (
        "The claim is true" if want_true else f"The claim is false; counterexample a = {bad[0][0]}, b = {bad[0][1]}"
    )
    return _fin(p, rng, verdict, mark)


# ---------------------------------------------------------------------------
# Implications over a set (exam)
# ---------------------------------------------------------------------------
def _square_div_implication(rng, size: int, want_true: bool) -> dict:
    """If n is divisible by a then n^2 is divisible by b."""
    for _ in attempts(4000, "an implication claim with the wanted truth"):
        a = rng.choice((4, 6, 10, 12, 14, 15, 18, 20))
        b = rng.choice((8, 9, 12, 16, 18, 24, 25, 27, 36))
        pool = [n for n in range(2, 80)]
        hyp = [n for n in pool if n % a == 0]
        fail = [n for n in hyp if (n * n) % b != 0]
        ok = [n for n in hyp if (n * n) % b == 0]
        non_hyp = [n for n in pool if n % a != 0]
        if len(hyp) < 3 or len(non_hyp) < 3:
            continue
        if want_true:
            if not ok:
                continue
            m_h = rng.randint(1, min(3, len(ok)))
            values = sorted(set(rng.sample(ok, m_h) + rng.sample(non_hyp, size - m_h)))
        else:
            if not fail:
                continue
            values = sorted(set(rng.sample(fail, 1) + rng.sample(ok, min(1, len(ok))) + rng.sample(non_hyp, size - 2)))
        bad = [n for n in values if n % a == 0 and (n * n) % b != 0]
        if (not bad) != want_true or len(values) < 4:
            continue
        break
    p = Prompt()
    _opening(p, values)
    p.t("Claim: for every ").m(Fm("n")).t(" in ").m(Fm("S")).t(", if ").m(Fm("n")).t(f" is divisible by {a}, then ").m(
        Fm("n^{2}", "n^2")
    ).t(f" is divisible by {b}. ")
    sat = [n for n in values if n % a == 0]
    if want_true:
        ref = [
            f"The claim is true. Only elements divisible by {a} matter: here {', '.join(map(str, sat))}. The others satisfy the hypothesis vacuously (the 'if' part is false), so the implication holds for them.",
            "Check the relevant elements: " + ", ".join(f"{n}^2 = {n * n} = {b} x {n * n // b}" for n in sat) + ".",
            f"Every element divisible by {a} has a square divisible by {b}, so the claim holds for all of S.",
        ]
        steps = [
            (
                "correctness",
                f"decides the claim is true and identifies the elements divisible by {a}: {', '.join(map(str, sat))}",
            ),
            (
                "logical_completeness",
                "checks the conclusion for each such element, and says the other elements satisfy the claim because the hypothesis fails",
            ),
            (
                "definitions_assumptions",
                "treats 'if P then Q' correctly: an element where P is false cannot break the claim",
            ),
            ("communication", "separates the elements that meet the hypothesis from those that do not"),
        ]
        traps = [
            (
                "tests_the_conclusion_everywhere",
                "does it wrongly demand n^2 divisible by b for elements that are not divisible by a, and so call the claim false?",
            ),
            ("misses_a_relevant_element", "does it check every element divisible by the hypothesis number?"),
        ]
    else:
        n0 = bad[0]
        ref = [
            f"The claim is false. Take n = {n0}, which is in S.",
            f"It satisfies the hypothesis: {n0} = {a} x {n0 // a}, so {n0} is divisible by {a}.",
            f"But n^2 = {n0 * n0} = {b} x {n0 * n0 // b} + {(n0 * n0) % b}, so n^2 is not divisible by {b}. The conclusion fails, so n = {n0} is a counterexample.",
        ]
        steps = [
            (
                "correctness",
                f"decides the claim is false and names an element of S that is divisible by {a} with a square not divisible by {b}, such as n = {n0}",
            ),
            ("logical_completeness", "shows both parts: the hypothesis holds and the conclusion fails"),
            (
                "definitions_assumptions",
                "uses an element of S that satisfies the hypothesis (an element failing the hypothesis is not a counterexample)",
            ),
            ("communication", "shows the working for the hypothesis and for the conclusion"),
        ]
        traps = [
            (
                "counterexample_fails_the_hypothesis",
                f"is the element it gives actually divisible by {a}? An element where the hypothesis is false cannot be a counterexample.",
            ),
            (
                "shows_only_the_conclusion_failing",
                "does it show the hypothesis holds as well as the conclusion failing?",
            ),
        ]
    mark = marking(reference=ref, steps=steps, traps=traps)
    return _fin(p, rng, _verdict(want_true, bad), mark)


def _prime_square_implication(rng, size: int, want_true: bool) -> dict:
    """If n is a prime greater than 3 then n^2 - 1 is divisible by 24 (or by 48, which fails)."""
    k = 24 if want_true else 48
    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 25, 27, 35]
    if want_true:
        values = sorted(rng.sample(primes, 3) + rng.sample(composites, size - 3))
    else:
        values = sorted(rng.sample(primes, 3) + rng.sample(composites, size - 3))
    bad = [n for n in values if _is_prime(n) and n > 3 and (n * n - 1) % k != 0]
    check((not bad) == want_true, "truth of the prime-square implication")
    p = Prompt()
    _opening(p, values)
    p.t("Claim: for every ").m(Fm("n")).t(" in ").m(Fm("S")).t(", if ").m(Fm("n")).t(
        " is a prime greater than 3, then "
    ).m(Fm("n^{2}-1", "n^2 - 1")).t(f" is divisible by {k}. ")
    sat = [n for n in values if _is_prime(n) and n > 3]
    if want_true:
        ref = [
            f"The claim is true. The primes greater than 3 in S are {', '.join(map(str, sat))}; the other elements satisfy the claim because the hypothesis fails for them.",
            "Check each prime: " + ", ".join(f"{n}^2 - 1 = {n * n - 1} = 24 x {(n * n - 1) // 24}" for n in sat) + ".",
            "Every prime greater than 3 in S gives a multiple of 24, so the claim holds for all of S.",
        ]
        steps = [
            (
                "correctness",
                f"decides the claim is true and picks out the primes greater than 3 in S: {', '.join(map(str, sat))}",
            ),
            (
                "logical_completeness",
                "checks n^2 - 1 for each of those primes and dismisses the other elements because the hypothesis is false",
            ),
            ("definitions_assumptions", "does not count 2 or 3 or composites as relevant (none is in the hypothesis)"),
            ("communication", "separates relevant from irrelevant elements"),
        ]
        traps = [
            ("tests_composites", "does it test composite elements and wrongly call the claim false because of them?"),
            ("misses_a_prime", "does it check every prime greater than 3 in S?"),
        ]
    else:
        n0 = bad[0]
        ref = [
            f"The claim is false. Take n = {n0}, a prime greater than 3 in S.",
            f"n^2 - 1 = {n0 * n0 - 1} = 48 x {(n0 * n0 - 1) // 48} + {(n0 * n0 - 1) % 48}, so it is not divisible by 48.",
            "The hypothesis holds and the conclusion fails, so n is a counterexample.",
        ]
        steps = [
            (
                "correctness",
                f"decides the claim is false and gives a prime greater than 3 in S, such as {n0}, with n^2 - 1 = {n0 * n0 - 1}",
            ),
            (
                "logical_completeness",
                "shows the hypothesis holds (n is a prime greater than 3) and the conclusion fails",
            ),
            ("definitions_assumptions", "uses an element in S that satisfies the hypothesis"),
            ("communication", "shows the working"),
        ]
        traps = [
            ("counterexample_not_prime", "is the element it gives a prime greater than 3 in S?"),
            ("arithmetic_slip", "does n^2 - 1 check out and is it really not a multiple of 48?"),
        ]
    mark = marking(reference=ref, steps=steps, traps=traps)
    return _fin(p, rng, _verdict(want_true, bad), mark)


def _easy(rng) -> dict:
    return _divisible_claim(rng, rng.randint(4, 6), rng.random() < 0.5, 40)


def _medium(rng) -> dict:
    which = rng.random()
    want = rng.random() < 0.5
    if which < 0.4:
        return _divisible_claim(rng, rng.randint(6, 8), want, 90)
    if which < 0.8:
        return _prime_claim(rng, rng.randint(4, 8), want)
    return _mersenne_claim(rng, rng.randint(3, 5), want)


def _hard(rng) -> dict:
    return _pair_claim(rng, rng.randint(4, 6), rng.random() < 0.5)


def _exam(rng) -> dict:
    want = rng.random() < 0.5
    if rng.random() < 0.65:
        return _square_div_implication(rng, rng.randint(5, 7), want)
    return _prime_square_implication(rng, rng.randint(6, 8), want)


FAMILIES = {"easy": [_easy], "medium": [_medium], "hard": [_hard], "exam": [_exam]}


@generator("proof_prove_or_refute", level=14, presets=_PRESETS)
def proof_prove_or_refute(rng, variant="easy"):
    """`ucv1-direct-proof-counterexample`. Rubric-assessed (`answer: rubric`): every question returned carries a
    per-item marking dict (`Question.answer`), never a plain value."""
    if variant not in FAMILIES:
        raise ValueError(f"unknown variant: {variant}")
    return rng.choice(FAMILIES[variant])(rng)
