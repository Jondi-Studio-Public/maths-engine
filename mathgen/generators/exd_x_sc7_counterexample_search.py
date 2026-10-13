"""Exam source for `senior-use-examples-and-counter-examples`: hunt for the smallest counterexample, then use it.

The practice tiers show four supplied values and ask which one disproves a claim. Here nothing is supplied: the
learner must search for the smallest positive integer that breaks a stated claim (typed, so it cannot be guessed from
a list) and then use that counterexample in a second part. Three families of claim, each false:

  * ``prime``   -- "n^2 + b n + c is prime for every positive integer n". (a) the smallest n for which it is not
    prime; (b) the value of the expression there (follow-through from (a)).
  * ``square``  -- "if n^2 is divisible by m then n is divisible by m". (a) the smallest n that satisfies the
    hypothesis but not the conclusion; (b) the quotient n^2 / m (follow-through).
  * ``product`` -- "if n is divisible by a and by b then n is divisible by a x b". (a) the smallest
    counterexample, which is the lowest common multiple; (b) how many counterexamples there are up to a limit
    (follow-through, since it counts multiples of the answer to (a)).

Every key is found by brute force over the positive integers, so (a) is the true smallest counterexample.
"""

from __future__ import annotations

import math
import random

from sympy import isprime

from ..answers import Distractor
from ..core import VerificationError, attempts, check, generator, latex_answer

_FAMILIES = ("prime", "square", "product")
_LIMIT = 2000  # brute-force search bound; every draw has a counterexample far below it


def _smallest(is_counterexample) -> int:
    for n in range(1, _LIMIT):
        if is_counterexample(n):
            return n
    raise VerificationError("no counterexample found in range")


def _next_after(is_counterexample, n: int) -> int:
    for k in range(n + 1, _LIMIT):
        if is_counterexample(k):
            return k
    raise VerificationError("no second counterexample found in range")


def _int_part(label, prompt, answer, marks, distractors, working, follow_through=None, sensible=True):
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(answer),
        answer_text=str(int(answer)),
        answer_latex=latex_answer(str(int(answer))),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if sensible:
        part["sensible_range"] = (0, None)
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _prime_family(rng: random.Random) -> dict:
    for _ in attempts(1000, "a prime-generating polynomial that fails between n = 4 and n = 40"):
        b = rng.randint(1, 5)
        c = rng.choice([3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79])

        def value(n, b=b, c=c):
            return n * n + b * n + c

        def fails(n, b=b, c=c):
            return not isprime(value(n))

        n0 = _smallest(fails)
        if 4 <= n0 <= 40:
            break
    n1 = _next_after(fails, n0)
    expr = f"n^2 + {b}n + {c}" if b > 1 else f"n^2 + n + {c}"
    stem = rng.choice(
        [
            f"A student claims that {expr} is a prime number for every positive integer n. "
            "A counterexample is a positive integer n for which the claim is false.",
            f"Claim: “{expr} is prime for every positive integer n.” "
            "A counterexample is a positive integer n for which the claim is false.",
        ]
    )
    key_b = value(n0)
    a_cands = [
        Distractor(str(n0 - 1), "example-confused-with-counterexample", n0 - 1),
        Distractor(str(key_b), "reports-value-not-input", key_b),
        Distractor(str(n1), "skips-to-a-later-counterexample", n1),
        Distractor(str(n0 + 1), None, n0 + 1),
        Distractor(str(n0 + 2), None, n0 + 2),
    ]
    b_cands = [
        Distractor(str(value(n0 - 1)), "evaluates-at-the-wrong-input", value(n0 - 1)),
        Distractor(str(n0 * n0 + b * n0), "dropped-constant-term", n0 * n0 + b * n0),
        Distractor(str(2 * n0 + b * n0 + c), "squared-as-doubled", 2 * n0 + b * n0 + c),
        Distractor(str(value(n1)), "skips-to-a-later-counterexample", value(n1)),
    ]
    for pool, key in ((a_cands, n0), (b_cands, key_b)):
        check(all(d.value != key for d in pool[:3]), "a distractor equals the key")
    parts = [
        _int_part(
            "a",
            "Find the smallest counterexample.",
            n0,
            2,
            _pick(a_cands, n0),
            [
                f"Test n = 1, 2, 3, ... in order: {expr} is prime until n = {n0}.",
                f"n = {n0} gives {key_b}, which is not prime.",
            ],
        ),
        _int_part(
            "b",
            f"Hence find the value of {expr} at your counterexample from part (a).",
            key_b,
            1,
            _pick(b_cands, key_b),
            [f"Substitute n = {n0}: {n0}^2 + {b}×{n0} + {c} = {key_b}."],
            follow_through=f"part_a**2 + {b}*part_a + {c}",
        ),
    ]
    return dict(text=stem, parts=parts)


def _pick(candidates, key: int, n: int = 3):
    seen, out = {key}, []
    fillers = [Distractor(str(key + d), None, key + d) for d in (1, -1, 2, -2, 3) if key + d >= 1]
    for c in [*candidates, *fillers]:
        if c.value in seen:
            continue
        seen.add(c.value)
        out.append(c)
        if len(out) == n:
            break
    check(len(out) >= 3, "fewer than three distractors")
    return out


def _square_family(rng: random.Random) -> dict:
    for _ in attempts(1000, "a modulus that is not squarefree"):
        m = rng.randint(4, 220)
        if any(m % (p * p) == 0 for p in range(2, int(math.isqrt(m)) + 1)):
            break

    def counter(n, m=m):
        return (n * n) % m == 0 and n % m != 0

    n0 = _smallest(counter)
    n1 = _next_after(counter, n0)
    quotient = n0 * n0 // m
    check(n0 * n0 % m == 0 and quotient >= 1, "quotient is not a positive whole number")
    stem = rng.choice(
        [
            f"A student claims: “For every positive integer n, if n^2 is divisible by {m} then n is divisible by {m}.” "
            "A counterexample is a positive integer n for which n^2 is divisible by "
            f"{m} but n is not.",
            f"Claim: “If n^2 is divisible by {m}, then n is divisible by {m}, for every positive integer n.” "
            f"A counterexample is a positive integer n for which n^2 is divisible by {m} but n is not.",
        ]
    )
    a_cands = [
        Distractor(str(m), "example-confused-with-counterexample", m),
        Distractor(str(n1), "skips-to-a-later-counterexample", n1),
        Distractor(str(math.isqrt(m) + 1), "takes-the-square-root-of-the-modulus", math.isqrt(m) + 1),
        Distractor(str(m // 2), None, m // 2),
        Distractor(str(n0 + 1), None, n0 + 1),
        Distractor(str(n0 + 2), None, n0 + 2),
    ]
    b_cands = [
        Distractor(str(n0 * n0), "stops-before-the-final-division", n0 * n0),
        Distractor(str(n0), "repeats-the-previous-answer", n0),
        Distractor(str(quotient + 1), None, quotient + 1),
        Distractor(str(max(quotient - 1, 0)), None, max(quotient - 1, 0)),
        Distractor(str(m), None, m),
    ]
    parts = [
        _int_part(
            "a",
            "Find the smallest counterexample.",
            n0,
            2,
            _pick(a_cands, n0),
            [
                f"We need {m} to divide n^2 but not n. Test n = 1, 2, 3, ... in order.",
                f"The first such n is {n0}: {n0}^2 = {n0 * n0}.",
            ],
        ),
        _int_part(
            "b",
            f"Hence find the whole number you get when n^2 is divided by {m}, for your counterexample from part (a).",
            quotient,
            1,
            _pick(b_cands, quotient),
            [f"{n0}^2 ÷ {m} = {n0 * n0} ÷ {m} = {quotient}."],
            follow_through=f"part_a**2/{m}",
        ),
    ]
    return dict(text=stem, parts=parts)


def _product_family(rng: random.Random) -> dict:
    for _ in attempts(1000, "a pair with a common factor, neither dividing the other"):
        a, b = rng.randint(4, 24), rng.randint(4, 24)
        if a != b and math.gcd(a, b) > 1 and a % b and b % a and math.lcm(a, b) <= 200:
            break
    prod = a * b

    def counter(n, a=a, b=b, prod=prod):
        return n % a == 0 and n % b == 0 and n % prod != 0

    n0 = _smallest(counter)
    n1 = _next_after(counter, n0)
    for _ in attempts(1000, "a limit with at least two counterexamples"):
        limit = n0 * rng.randint(3, 8) + rng.randrange(n0)
        count = sum(1 for n in range(1, limit + 1) if counter(n))
        if count >= 2:
            break
    all_multiples = limit // n0
    multiples_of_prod = limit // prod
    stem = rng.choice(
        [
            f"A student claims: “For every positive integer n, if n is divisible by {a} and by {b}, "
            f"then n is divisible by {prod}.” A counterexample is a positive integer n that is divisible by "
            f"{a} and by {b} but not by {prod}.",
            f"Claim: “If a positive integer n is divisible by both {a} and {b}, then n is divisible by {prod}.” "
            f"A counterexample is a positive integer n that is divisible by {a} and by {b} but not by {prod}.",
        ]
    )
    a_cands = [
        Distractor(str(prod), "example-confused-with-counterexample", prod),
        Distractor(str(max(a, b)), "ignores-one-of-the-two-divisors", max(a, b)),
        Distractor(str(n1), "skips-to-a-later-counterexample", n1),
        Distractor(str(math.gcd(a, b)), None, math.gcd(a, b)),
    ]
    b_cands = [
        Distractor(str(all_multiples), "forgets-to-exclude-the-non-counterexamples", all_multiples),
        Distractor(str(multiples_of_prod), "counts-the-examples-not-the-counterexamples", multiples_of_prod),
        Distractor(str(count + 1), None, count + 1),
        Distractor(str(max(count - 1, 0)), None, max(count - 1, 0)),
        Distractor(str(count + 2), None, count + 2),
        Distractor(str(count * 2), None, count * 2),
    ]
    parts = [
        _int_part(
            "a",
            "Find the smallest counterexample.",
            n0,
            2,
            _pick(a_cands, n0),
            [
                f"n must be a multiple of both {a} and {b} but not of {prod}.",
                f"The smallest common multiple is {n0}, and {n0} is not a multiple of {prod}.",
            ],
        ),
        _int_part(
            "b",
            f"Hence find how many counterexamples there are among the positive integers from 1 to {limit}.",
            count,
            2,
            _pick(b_cands, count),
            [
                f"Counterexamples are the multiples of {n0} that are not multiples of {prod}.",
                f"{limit // n0} multiples of {n0} up to {limit}, less {limit // prod} multiples of {prod}: {count}.",
            ],
            follow_through=f"floor({limit}/part_a) - {limit // prod}",
        ),
    ]
    return dict(text=stem, parts=parts)


_BUILDERS = {"prime": _prime_family, "square": _square_family, "product": _product_family}


@generator("exd_x_sc7_counterexample_search", level=12, presets={"exam": dict(tier="exam")})
def exd_x_sc7_counterexample_search(rng: random.Random, tier: str = "exam") -> dict:
    """See the module docstring. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _BUILDERS[rng.choice(_FAMILIES)](rng)
