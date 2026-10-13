"""Generates `proof_exposition` items: write a short proof of a supplied finite theorem so that every claim is
linked to a numbered definition, a numbered earlier result, or an explicit calculation.

Each question states a theorem about a stated finite range of integers (so it can be checked completely), lists
numbered definitions and results the learner may cite (some are not needed, and from the hard tier a cited
result has a condition that the theorem's range must be checked against), and asks for the proof with the
support named for each claim. Every theorem is verified here by exhaustive evaluation of its finite range, and
the cited results are the facts the reference proof uses. The answer is marked against a rubric, so `answer` is
a marking dict (see `_crit_kit`), not a value.

Ladder:

- easy: a short chain (sum of consecutive integers, differences of two squares) in which the support for each step
  is a given result or a single calculation.
- medium: a greatest-common-divisor bound and finite-range inequalities, with an unused result in the list.
- hard: a compositeness and a divisibility theorem whose range must start where the cited factorisation gives
  two factors larger than 1 (or whose proof needs several results), with distractor results.
- exam: theorems whose cited step has a hidden condition (square roots need non-negative numbers, clearing
  denominators needs positive ones, a half of n + 1 needs n odd) that the proof must state and check.
"""

from __future__ import annotations

from math import gcd, isqrt

from ..core import check, generator
from ._crit_kit import attempts, Fm, Prompt, finish, short_marking

_PRESETS = {
    "easy": {"variant": "easy"},
    "medium": {"variant": "medium"},
    "hard": {"variant": "hard"},
    "exam": {"variant": "exam"},
}

TAILS_P = (
    "Write the proof as numbered steps. After each step, name the definition or result you use (for example D1 or R2), or show the calculation.",
    "Set out a short proof in which every claim is followed by the label of the definition or result that supports it, or by the calculation itself.",
    "Give a proof in numbered steps and link each step to a numbered definition, a numbered result or an explicit calculation.",
)


def _is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, isqrt(n) + 1))


class Proof:
    """A theorem, the statements offered, and the steps of the reference proof."""

    def __init__(self, theorem: Fm, givens: list[tuple[str, str]], steps: list[tuple[str, str]], hidden: str = ""):
        self.theorem = theorem
        self.givens = givens  # (kind D/R, text) in the order they will be numbered after shuffling within kind
        self.steps = steps  # (claim, support) where support is the text of a given, or "calculation"
        self.hidden = hidden  # the condition of a cited result that the proof must check ("" if none)


def _number(rng, givens: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], dict]:
    """Shuffle the offered statements and label them D1.. for definitions and R1.. for results."""
    order = list(range(len(givens)))
    rng.shuffle(order)
    labelled, lookup = [], {}
    d_i = r_i = 0
    for i in order:
        kind, text = givens[i]
        if kind == "D":
            d_i += 1
            lab = f"D{d_i}"
        else:
            r_i += 1
            lab = f"R{r_i}"
        labelled.append((lab, text))
        lookup[text] = lab
    labelled.sort(key=lambda x: (x[0][0] == "R", int(x[0][1:])))
    return labelled, lookup


def _finish(rng, pr: Proof, claim: str, *, extra_traps: list[tuple[str, str]] | None = None) -> dict:
    labelled, lookup = _number(rng, pr.givens)
    p = Prompt()
    p.t("Theorem. ").m(pr.theorem).t(" ")
    p.t("You may use: " + "; ".join(f"{lab}: {txt}" for lab, txt in labelled) + ". ")
    p.t(rng.choice(TAILS_P))
    ref, steps_txt = [], []
    for i, (cl, sup) in enumerate(pr.steps, 1):
        lab = lookup.get(sup, "")
        support = lab if lab else (sup if sup != "calculation" else "calculation")
        ref.append(f"Step {i}: {cl} [{support}]")
        steps_txt.append((cl, support))
    if pr.hidden:
        ref.append(f"Check of the cited result's condition: {pr.hidden}.")
    needed = sorted({lookup[sup] for _, sup in pr.steps if sup in lookup})
    steps = [
        (
            "correct_idea",
            "every claim in the proof is true and the algebra is right: "
            + "; ".join(c for c, _ in steps_txt[:3])
            + ("; ..." if len(steps_txt) > 3 else ""),
        ),
        (
            "completeness",
            "each claim is linked to the right support: " + "; ".join(f"{c} -> {s}" for c, s in steps_txt),
        ),
        (
            "use_of_context",
            "the explicit calculation(s) are shown with the numbers: "
            + "; ".join(c for c, s in steps_txt if s == "calculation")
            if any(s == "calculation" for _, s in steps_txt)
            else "cites the needed statements (" + ", ".join(needed) + ") by label and uses each correctly",
        ),
        ("communication", "the steps are numbered in a logical order, each with its support named"),
    ]
    if pr.hidden:
        steps.insert(2, ("completeness", f"states and checks the condition of the cited result: {pr.hidden}"))
    traps = [
        (
            "claim_without_support",
            "does any step state a claim with no definition, result or calculation named beside it?",
        ),
        ("support_does_not_fit", "is any cited label a statement that does not say what the step needs?"),
        ("assumes_the_conclusion", "does the proof use the theorem's conclusion as a step?"),
    ]
    if pr.hidden:
        traps.append(("hidden_condition_unchecked", f"does it check the condition on the cited result: {pr.hidden}?"))
    if extra_traps:
        traps += extra_traps
    mark = short_marking(reference=ref, steps=steps, traps=traps)
    return finish(p, claim, mark)


# ---------------------------------------------------------------------------
# Easy
# ---------------------------------------------------------------------------
def _consecutive_sum(rng) -> dict:
    m = rng.choice((3, 5, 7, 9, 11))
    lo = rng.randint(-6, 4)
    hi = lo + rng.randint(8, 16)
    check(all(sum(range(n, n + m)) % m == 0 for n in range(lo, hi + 1)), "sum of m consecutive integers, m odd")
    half = (m - 1) // 2
    pr = Proof(
        Fm(
            f"\\text{{For every integer }}n\\text{{ from }}{lo}\\text{{ to }}{hi},\\ \\text{{the sum }}n+(n+1)+\\dots+(n+{m - 1})\\text{{ is divisible by }}{m}.",
            f"For every integer n from {lo} to {hi}, the sum n + (n + 1) + ... + (n + {m - 1}) is divisible by {m}.",
        ),
        [
            ("D", f"an integer is divisible by {m} when it equals {m} times an integer"),
            ("R", "the sum 1 + 2 + ... + j equals j(j + 1)/2"),
            ("R", "if a and b are integers then a + b and ab are integers"),
            (
                "R",
                f"{m} is an odd number, so (m - 1)/2 is a whole number"[:0]
                + f"{m - 1} is even, so {m - 1} divided by 2 is the integer {half}",
            ),
        ],
        [
            (
                f"The sum is {m}n + (0 + 1 + ... + {m - 1}), since each of the {m} terms is n plus one of 0, 1, ..., {m - 1}.",
                "calculation",
            ),
            (
                f"0 + 1 + ... + {m - 1} = {m - 1}({m})/2 = {m * (m - 1) // 2}.",
                "the sum 1 + 2 + ... + j equals j(j + 1)/2",
            ),
            (
                f"So the sum is {m}n + {m * (m - 1) // 2} = {m}(n + {half}), because {m * (m - 1) // 2} = {m} × {half}.",
                f"{m - 1} is even, so {m - 1} divided by 2 is the integer {half}",
            ),
            (
                f"n + {half} is an integer because n is an integer and {half} is an integer.",
                "if a and b are integers then a + b and ab are integers",
            ),
            (
                f"So the sum equals {m} times an integer, so it is divisible by {m}.",
                f"an integer is divisible by {m} when it equals {m} times an integer",
            ),
        ],
    )
    return _finish(rng, pr, f"True: the sum of {m} consecutive integers is {m}(n + {half}), a multiple of {m}")


def _difference_of_squares(rng) -> dict:
    pool = [n for n in range(3, 80) if n % 2 == 1 or n % 4 == 0]
    nums = sorted(rng.sample(pool, rng.choice((3, 4))))
    decomp = {}
    for n in nums:
        found = None
        for a in range(1, n + 1):
            r = a * a - n
            if r >= 0 and isqrt(r) ** 2 == r:
                found = (a, isqrt(r))
                break
        check(found is not None, f"{n} is a difference of two squares")
        decomp[n] = found
    steps = []
    for n in nums:
        a, b = decomp[n]
        steps.append((f"{n} = {a}^2 - {b}^2 = {a * a} - {b * b} = {n}.", "calculation"))
    steps.append(
        (
            f"So each listed number is a difference of two squares.",
            "a difference of two squares is a number of the form a^2 - b^2 with a and b integers",
        )
    )
    pr = Proof(
        Fm(
            "\\text{Each of the numbers }" + ",\\ ".join(map(str, nums)) + "\\text{ is a difference of two squares.}",
            "Each of the numbers " + ", ".join(map(str, nums)) + " is a difference of two squares.",
        ),
        [
            ("D", "a difference of two squares is a number of the form a^2 - b^2 with a and b integers"),
            ("R", "a^2 - b^2 = (a - b)(a + b)"),
            ("R", "an odd number 2k + 1 equals (k + 1)^2 - k^2"),
        ],
        steps,
    )
    return _finish(rng, pr, "True: " + "; ".join(f"{n} = {decomp[n][0]}^2 - {decomp[n][1]}^2" for n in nums))


# ---------------------------------------------------------------------------
# Medium
# ---------------------------------------------------------------------------
def _gcd_bound(rng) -> dict:
    k = rng.randint(2, 12)
    hi = rng.randint(14, 30)
    check(all(gcd(n, n + k) <= k for n in range(1, hi + 1)), "gcd(n, n+k) <= k")
    pr = Proof(
        Fm(
            f"\\text{{For every integer }}n\\text{{ from 1 to }}{hi},\\ \\gcd(n,n+{k})\\le {k}.",
            f"For every integer n from 1 to {hi}, gcd(n, n + {k}) ≤ {k}.",
        ),
        [
            ("D", "gcd(a, b) is the greatest positive integer that divides both a and b"),
            ("R", "if d divides x and d divides y then d divides x - y"),
            ("R", "a positive integer that divides a positive integer k is at most k"),
            ("R", "every positive integer divides itself"),
        ],
        [
            (
                "Let d = gcd(n, n + %d). Then d is a positive integer that divides n and divides n + %d." % (k, k),
                "gcd(a, b) is the greatest positive integer that divides both a and b",
            ),
            (f"So d divides (n + {k}) - n = {k}.", "if d divides x and d divides y then d divides x - y"),
            (
                f"Since d is a positive divisor of the positive integer {k}, d ≤ {k}.",
                "a positive integer that divides a positive integer k is at most k",
            ),
            (f"So gcd(n, n + {k}) ≤ {k}.", "calculation"),
        ],
    )
    return _finish(rng, pr, f"True: any common divisor of n and n + {k} divides {k}, so the gcd is at most {k}")


def _inequality(rng) -> dict:
    n = rng.randint(4, 9)
    kind = rng.choice(("two", "square", "three"))
    if kind == "two":
        check(all(a * a + b * b >= 2 * a * b for a in range(1, n + 1) for b in range(1, n + 1)), "a^2 + b^2 >= 2ab")
        th = Fm(
            f"\\text{{For all integers }}a,b\\text{{ from 1 to }}{n},\\ a^{{2}}+b^{{2}}\\ge 2ab.",
            f"For all integers a, b from 1 to {n}, a^2 + b^2 ≥ 2ab.",
        )
        steps = [
            ("(a - b)^2 ≥ 0, since it is the square of an integer.", "the square of any integer is at least 0"),
            ("Expanding, (a - b)^2 = a^2 - 2ab + b^2.", "(x - y)^2 = x^2 - 2xy + y^2"),
            ("So a^2 - 2ab + b^2 ≥ 0, and adding 2ab to both sides gives a^2 + b^2 ≥ 2ab.", "calculation"),
        ]
        givens = [
            ("R", "the square of any integer is at least 0"),
            ("R", "(x - y)^2 = x^2 - 2xy + y^2"),
            ("R", "adding the same number to both sides of an inequality keeps it true"),
        ]
        claim = "True: (a - b)^2 ≥ 0 expands to a^2 + b^2 ≥ 2ab"
    elif kind == "square":
        check(all((a + b) ** 2 >= 4 * a * b for a in range(1, n + 1) for b in range(1, n + 1)), "(a+b)^2 >= 4ab")
        th = Fm(
            f"\\text{{For all integers }}a,b\\text{{ from 1 to }}{n},\\ (a+b)^{{2}}\\ge 4ab.",
            f"For all integers a, b from 1 to {n}, (a + b)^2 ≥ 4ab.",
        )
        steps = [
            ("(a + b)^2 - 4ab = a^2 + 2ab + b^2 - 4ab = a^2 - 2ab + b^2.", "(x + y)^2 = x^2 + 2xy + y^2"),
            ("a^2 - 2ab + b^2 = (a - b)^2.", "(x - y)^2 = x^2 - 2xy + y^2"),
            ("(a - b)^2 ≥ 0, so (a + b)^2 - 4ab ≥ 0, which is the claim.", "the square of any integer is at least 0"),
        ]
        givens = [
            ("R", "the square of any integer is at least 0"),
            ("R", "(x - y)^2 = x^2 - 2xy + y^2"),
            ("R", "(x + y)^2 = x^2 + 2xy + y^2"),
            ("R", "a^2 - b^2 = (a - b)(a + b)"),
        ]
        claim = "True: (a + b)^2 - 4ab = (a - b)^2 ≥ 0"
    else:
        check(
            all(
                a * a + b * b + c * c >= a * b + b * c + c * a
                for a in range(1, 6)
                for b in range(1, 6)
                for c in range(1, 6)
            ),
            "three-variable inequality",
        )
        th = Fm(
            "\\text{For all integers }a,b,c\\text{ from 1 to 5},\\ a^{2}+b^{2}+c^{2}\\ge ab+bc+ca.",
            "For all integers a, b, c from 1 to 5, a^2 + b^2 + c^2 ≥ ab + bc + ca.",
        )
        steps = [
            (
                "(a - b)^2 + (b - c)^2 + (c - a)^2 = 2(a^2 + b^2 + c^2) - 2(ab + bc + ca), by expanding each square.",
                "(x - y)^2 = x^2 - 2xy + y^2",
            ),
            (
                "Each of the three squares is at least 0, so their sum is at least 0.",
                "the square of any integer is at least 0",
            ),
            ("So 2(a^2 + b^2 + c^2) - 2(ab + bc + ca) ≥ 0; dividing by 2 gives the claim.", "calculation"),
        ]
        givens = [
            ("R", "the square of any integer is at least 0"),
            ("R", "(x - y)^2 = x^2 - 2xy + y^2"),
            ("R", "a sum of numbers that are each at least 0 is at least 0"),
        ]
        claim = "True: the sum of the three squares (a - b)^2 + (b - c)^2 + (c - a)^2 is at least 0 and equals twice the difference"
    pr = Proof(th, givens, steps)
    return _finish(rng, pr, claim)


# ---------------------------------------------------------------------------
# Hard
# ---------------------------------------------------------------------------
def _composite(rng) -> dict:
    hi = rng.randint(15, 40)
    check(all(not _is_prime(n * n - 1) for n in range(3, hi + 1)), "n^2 - 1 is composite for n >= 3")
    check(_is_prime(2 * 2 - 1), "n = 2 is the boundary: 3 is prime")
    pr = Proof(
        Fm(
            f"\\text{{For every integer }}n\\text{{ from 3 to }}{hi},\\ n^{{2}}-1\\text{{ is not prime.}}",
            f"For every integer n from 3 to {hi}, n^2 - 1 is not prime.",
        ),
        [
            ("D", "a prime is an integer greater than 1 whose only positive divisors are 1 and itself"),
            ("R", "a^2 - b^2 = (a - b)(a + b)"),
            (
                "R",
                "if an integer m > 1 equals a product xy of positive integers with x > 1 and y > 1, then m is not prime",
            ),
            ("R", "every integer greater than 1 has a prime factor"),
            ("R", "the product of two integers each at least 2 is at least 4"),
        ],
        [
            ("n^2 - 1 = (n - 1)(n + 1).", "a^2 - b^2 = (a - b)(a + b)"),
            ("Since n ≥ 3, n - 1 ≥ 2 and n + 1 ≥ 4, so both factors are greater than 1.", "calculation"),
            (
                "So n^2 - 1 is a product of two positive integers each greater than 1, and n^2 - 1 > 1.",
                "the product of two integers each at least 2 is at least 4",
            ),
            (
                "Hence n^2 - 1 is not prime.",
                "if an integer m > 1 equals a product xy of positive integers with x > 1 and y > 1, then m is not prime",
            ),
        ],
        hidden="the factor n - 1 must be at least 2, which is why the range starts at 3: at n = 2 the factor is 1 and n^2 - 1 = 3 is prime",
    )
    return _finish(
        rng,
        pr,
        "True for n ≥ 3: n^2 - 1 = (n - 1)(n + 1) with both factors at least 2 (at n = 2 it is 3, which is prime)",
    )


def _cube_minus(rng) -> dict:
    hi = rng.randint(12, 40)
    check(all((n**3 - n) % 6 == 0 for n in range(2, hi + 1)), "n^3 - n divisible by 6")
    pr = Proof(
        Fm(
            f"\\text{{For every integer }}n\\text{{ from 2 to }}{hi},\\ n^{{3}}-n\\text{{ is divisible by 6.}}",
            f"For every integer n from 2 to {hi}, n^3 - n is divisible by 6.",
        ),
        [
            ("R", "n^3 - n = (n - 1) n (n + 1)"),
            ("R", "among any two consecutive integers, one is even"),
            ("R", "among any three consecutive integers, one is a multiple of 3"),
            ("R", "if a number is divisible by 2 and by 3 then it is divisible by 6"),
            ("R", "if a divides b then a divides bc, for any integer c"),
        ],
        [
            ("n^3 - n = (n - 1) n (n + 1), a product of three consecutive integers.", "n^3 - n = (n - 1) n (n + 1)"),
            (
                "Among n - 1 and n one is even, so the product is even (it has an even factor).",
                "among any two consecutive integers, one is even",
            ),
            (
                "Among n - 1, n, n + 1 one is a multiple of 3, so the product is divisible by 3.",
                "among any three consecutive integers, one is a multiple of 3",
            ),
            (
                "The product is divisible by both 2 and 3, so it is divisible by 6.",
                "if a number is divisible by 2 and by 3 then it is divisible by 6",
            ),
        ],
    )
    return _finish(rng, pr, "True: n^3 - n = (n - 1)n(n + 1) is even and a multiple of 3, so divisible by 6")


# ---------------------------------------------------------------------------
# Exam: hidden conditions
# ---------------------------------------------------------------------------
def _amgm(rng) -> dict:
    n = rng.randint(5, 9)
    check(all((a + b) / 2 >= (a * b) ** 0.5 - 1e-12 for a in range(1, n + 1) for b in range(1, n + 1)), "AM-GM")
    pr = Proof(
        Fm(
            f"\\text{{For all integers }}a,b\\text{{ from 1 to }}{n},\\ \\frac{{a+b}}{{2}}\\ge\\sqrt{{ab}}.",
            f"For all integers a, b from 1 to {n}, (a + b)/2 ≥ sqrt(ab).",
        ),
        [
            ("R", "the square of any real number is at least 0"),
            ("R", "if x ≥ 0 then sqrt(x) is a real number and (sqrt(x))^2 = x"),
            ("R", "sqrt(x y) = sqrt(x) sqrt(y) when x ≥ 0 and y ≥ 0"),
            ("R", "(u - v)^2 = u^2 - 2uv + v^2"),
        ],
        [
            (
                "a and b are positive integers, so sqrt(a) and sqrt(b) are real numbers and sqrt(ab) = sqrt(a) sqrt(b).",
                "sqrt(x y) = sqrt(x) sqrt(y) when x ≥ 0 and y ≥ 0",
            ),
            ("(sqrt(a) - sqrt(b))^2 ≥ 0.", "the square of any real number is at least 0"),
            (
                "Expanding, (sqrt(a) - sqrt(b))^2 = a - 2 sqrt(a) sqrt(b) + b, using (sqrt(x))^2 = x for x = a and x = b.",
                "(u - v)^2 = u^2 - 2uv + v^2",
            ),
            ("So a + b - 2 sqrt(ab) ≥ 0, and dividing by 2 gives (a + b)/2 ≥ sqrt(ab).", "calculation"),
        ],
        hidden="the square roots of a and b exist as real numbers only because a ≥ 0 and b ≥ 0 (here a, b ≥ 1)",
    )
    return _finish(
        rng, pr, "True: (sqrt(a) - sqrt(b))^2 ≥ 0 gives a + b ≥ 2 sqrt(ab), valid because a and b are non-negative"
    )


def _reciprocals(rng) -> dict:
    n = rng.randint(5, 9)
    check(
        all(1 / a + 1 / b >= 4 / (a + b) - 1e-12 for a in range(1, n + 1) for b in range(1, n + 1)),
        "1/a + 1/b >= 4/(a+b)",
    )
    pr = Proof(
        Fm(
            f"\\text{{For all integers }}a,b\\text{{ from 1 to }}{n},\\ \\frac{{1}}{{a}}+\\frac{{1}}{{b}}\\ge\\frac{{4}}{{a+b}}.",
            f"For all integers a, b from 1 to {n}, 1/a + 1/b ≥ 4/(a + b).",
        ),
        [
            (
                "R",
                "multiplying both sides of an inequality by a positive number keeps the inequality the same way round",
            ),
            ("R", "the square of any real number is at least 0"),
            ("R", "(x - y)^2 = x^2 - 2xy + y^2"),
            ("R", "a negative multiplier would reverse the inequality"),
        ],
        [
            ("1/a + 1/b = (a + b)/(ab).", "calculation"),
            (
                "a, b and a + b are positive, so ab(a + b) > 0, and multiplying both sides of the claim by ab(a + b) keeps the inequality the same way round.",
                "multiplying both sides of an inequality by a positive number keeps the inequality the same way round",
            ),
            ("The claim becomes (a + b)^2 ≥ 4ab.", "calculation"),
            ("(a + b)^2 - 4ab = a^2 - 2ab + b^2 = (a - b)^2 ≥ 0.", "the square of any real number is at least 0"),
        ],
        hidden="ab(a + b) is positive because a and b are positive, so multiplying by it does not reverse the inequality",
    )
    return _finish(
        rng, pr, "True: after multiplying by the positive ab(a + b), it is (a + b)^2 ≥ 4ab, i.e. (a - b)^2 ≥ 0"
    )


def _odd_triangular(rng) -> dict:
    hi = rng.choice((11, 13, 15, 17, 19, 21))
    odds = list(range(3, hi + 1, 2))
    check(all((n * (n + 1) // 2) % n == 0 for n in odds), "triangular number divisible by n for odd n")
    check((4 * 5 // 2) % 4 != 0, "fails for n = 4")
    pr = Proof(
        Fm(
            f"\\text{{For every odd integer }}n\\text{{ from 3 to }}{hi},\\ 1+2+\\dots+n\\text{{ is divisible by }}n.",
            f"For every odd integer n from 3 to {hi}, 1 + 2 + ... + n is divisible by n.",
        ),
        [
            ("D", "an integer is divisible by n when it equals n times an integer"),
            ("R", "the sum 1 + 2 + ... + n equals n(n + 1)/2"),
            ("R", "if n is odd then n + 1 is even"),
            ("R", "an even integer divided by 2 is an integer"),
        ],
        [
            ("1 + 2 + ... + n = n(n + 1)/2.", "the sum 1 + 2 + ... + n equals n(n + 1)/2"),
            ("n is odd, so n + 1 is even.", "if n is odd then n + 1 is even"),
            ("So (n + 1)/2 is an integer.", "an even integer divided by 2 is an integer"),
            (
                "Hence the sum is n × ((n + 1)/2), n times an integer, so it is divisible by n.",
                "an integer is divisible by n when it equals n times an integer",
            ),
        ],
        hidden="n must be odd for (n + 1)/2 to be an integer; for n = 4 the sum is 10, which is not divisible by 4",
    )
    return _finish(
        rng, pr, "True for odd n: the sum is n × (n + 1)/2 and (n + 1)/2 is an integer; false for even n such as 4"
    )


def _easy(rng) -> dict:
    return rng.choice((_consecutive_sum, _difference_of_squares))(rng)


def _medium(rng) -> dict:
    return rng.choice((_gcd_bound, _inequality))(rng)


def _hard(rng) -> dict:
    return rng.choice((_composite, _cube_minus))(rng)


def _exam(rng) -> dict:
    return rng.choice((_amgm, _reciprocals, _odd_triangular))(rng)


FAMILIES = {"easy": [_easy], "medium": [_medium], "hard": [_hard], "exam": [_exam]}


@generator("proof_exposition", level=17, presets=_PRESETS)
def proof_exposition(rng, variant="easy"):
    """`ucv1-proof-exposition-audit`. Rubric-assessed (`answer: rubric`): every question returned carries a
    per-item marking dict (`Question.answer`), never a plain value."""
    if variant not in FAMILIES:
        raise ValueError(f"unknown variant: {variant}")
    return rng.choice(FAMILIES[variant])(rng)
