"""Maths for the exponential subject g(x) = a^(2x) - b*a^x + c.

With t = a^x the function is the quadratic t^2 - b t + c = (t - r1)(t - r2), so the facts are fixed by a and the
two positive integer roots r1, r2 of the quadratic. r1 is a power of a (so one solution is a whole number) and
r2 is not (so the other solution is a logarithm). The 2023 VCE Methods Exam 1 question is a=2, b=9, c=20 (r1=4,
r2=5). This module holds only the maths and the wrong-answer candidates; the composer glue lives elsewhere.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from sympy import integer_log

from ..core import attempts, check

BASES = (2, 3)
_MAX_C = 60


@dataclass(frozen=True)
class ExpG:
    a: int
    r1: int  # a power of a
    r2: int  # not a power of a

    @property
    def m(self) -> int:
        """The whole-number solution: a^m = r1."""
        return integer_log(self.r1, self.a)[0]

    @property
    def b(self) -> int:
        return self.r1 + self.r2

    @property
    def c(self) -> int:
        return self.r1 * self.r2

    def g_text(self) -> str:
        return f"g(x) = {self.a}^(2x) - {self.b}*{self.a}^x + {self.c}"

    def g_at_log(self, k: int) -> int:
        """g(log_a k): a^(2 log_a k) = k^2 and a^(log_a k) = k."""
        return k * k - self.b * k + self.c

    def log_root_text(self) -> str:
        return f"log_{self.a}({self.r2})"

    def log_root_value(self) -> float:
        import math

        return math.log(self.r2, self.a)


def _is_power(n: int, a: int) -> bool:
    return integer_log(n, a)[1]


def draw(rng: random.Random) -> ExpG:
    for _ in attempts(200, "an exponential g(x) draw"):
        a = rng.choice(BASES)
        m = rng.choice((1, 2, 3))
        r1 = a**m
        r2 = rng.randint(2, 12)
        if r2 == r1 or _is_power(r2, a):
            continue
        if r1 * r2 > _MAX_C:
            continue
        s = ExpG(a, r1, r2)
        check(s.c == r1 * r2 and s.b == r1 + r2)
        return s
    raise AssertionError("unreachable")


def eval_k(rng: random.Random, s: ExpG) -> int:
    """k for 'evaluate g(log_a k)': not a root, not a power of a, 2..9, and g(log_a k) != 0."""
    ks = [k for k in range(2, 10) if k not in (s.r1, s.r2) and not _is_power(k, s.a)]
    check(bool(ks), "no k available")
    return rng.choice(ks)


def eval_wrong(s: ExpG, k: int) -> list[tuple[int, str]]:
    """(wrong value, slip) candidates for g(log_a k), in priority order. Caller filters against the key."""
    a, b, c = s.a, s.b, s.c
    return [
        (2 * k - b * k + c, "reads a^(2x) as 2*a^x"),
        (k * k - b * k, "drops the constant"),
        (k * k - b + c, "treats the middle term's a^x as 1"),
        (k * k + b * k + c, "sign of the middle term"),
        (a ** (2 * k) - b * a**k + c, "substitutes x = k"),
    ]


def log_root_wrong(s: ExpG) -> list[tuple[str, str]]:
    """Wrong expression texts for the log solution."""
    a, r1, r2, b = s.a, s.r1, s.r2, s.b
    return [
        (str(r2), "gives the root of the quadratic in t, not x"),
        (f"log_{a}({b})", "uses b as the argument"),
        (f"log_{a}({s.c})", "uses c as the argument"),
        (f"-log_{a}({r2})", "sign slip"),
        (f"log_{r2}({a})", "swaps base and argument"),
        (f"log_{a}({r1 + r2 - 1})", "off by one in the root"),
    ]
