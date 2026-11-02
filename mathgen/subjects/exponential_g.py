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

BASES = (2, 3, 5, 7)
_MAX_C = 300
_MAX_R2 = 40


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


def shifted_roots(s: ExpG) -> list[tuple[int, int]]:
    """Pairs (p, q), p < q, with p + q = b, q not a power of a and not r1 or r2, p >= 2, p*q != c.

    Then g(x) = c - p*q has the roots t = p, q of the quadratic in t = a^x, so its larger solution is log_a(q), an
    argument that appears nowhere in the stem."""
    out = []
    for p in range(2, s.b):
        q = s.b - p
        if q <= p or _is_power(q, s.a) or q in (s.r1, s.r2) or p in (s.r1, s.r2) or p * q == s.c:
            continue
        out.append((p, q))
    return out


def draw(rng: random.Random) -> ExpG:
    for _ in attempts(200, "an exponential g(x) draw"):
        a = rng.choice(BASES)
        m = rng.choice((1, 2, 3))
        r1 = a**m
        r2 = rng.randint(2, _MAX_R2)
        if r2 == r1 or _is_power(r2, a):
            continue
        if r1 * r2 > _MAX_C:
            continue
        s = ExpG(a, r1, r2)
        if not shifted_roots(s):
            continue
        check(s.c == r1 * r2 and s.b == r1 + r2)
        return s
    raise AssertionError("unreachable")


def eval_k(rng: random.Random, s: ExpG) -> int:
    """k for 'evaluate g(log_a k)': not a root, not a power of a, 2..9, and g(log_a k) != 0."""
    later = {q for _, q in shifted_roots(s)}  # a later part's answer must not appear in this prompt
    ks = [k for k in range(2, 10) if k not in (s.r1, s.r2) and k not in later and not _is_power(k, s.a)]
    check(bool(ks), "no k available")
    return rng.choice(ks)


def eval_wrong(s: ExpG, k: int) -> list[tuple[int, str]]:
    """(wrong value, slip) candidates for g(log_a k), in priority order. Caller filters against the key."""
    a, b, c = s.a, s.b, s.c
    cands = [
        (2 * k - b * k + c, "squared-as-doubled"),
        (k * k - b * k, "dropped-constant-term"),
        (a ** (2 * k) - b * a**k + c, "exp-log-argument-used-as-x"),
        (k * k + b * k + c, "exp-middle-term-sign"),
        (k * k - b + c, "exp-middle-term-power-dropped"),
    ]
    # A value far larger than the key is not a believable slip; keep the candidates a learner could plausibly write.
    return [(w, t) for w, t in cands if w != 0 and abs(w) <= 20 * max(abs(k * k), s.c)]


def log_root_wrong(s: ExpG) -> list[tuple[str, str]]:
    """Wrong expression texts for the log solution."""
    a, r1, r2, b = s.a, s.r1, s.r2, s.b
    return [
        (str(r2), "exp-quadratic-root-given-as-x"),
        (f"log_{a}({b})", "exp-log-wrong-argument"),
        (f"log_{a}({s.c})", "exp-log-wrong-argument"),
        (f"-log_{a}({r2})", "exp-log-sign-dropped"),
        (f"log_{r2}({a})", "log-base-and-argument-swapped"),
    ]
