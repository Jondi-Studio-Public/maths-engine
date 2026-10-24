"""Pick-the-one forms of four yes/no and either/or skills (task w2).

The older generators for ``f10-factors-multiples-divisibility``, ``f10-odd-even-properties``,
``f10-prime-composite-square-properties`` asked a two-way question ("Does 4 divide 48?", "odd or
even?"), so a blind guess was right half the time. Here the learner picks the one option that has the
property from four, which keeps what is taught and makes a guess right a quarter of the time. The
older generators call these when given ``choose=True``, which the skill files set.

Every question is built answer first (one option has the property, three do not), then each option is
checked a second way, by integer arithmetic, so a wrong key cannot get through.
"""

from __future__ import annotations

import math

from ..core import attempts, check, pick_distractors
from ..slots import NAMES
from .p3a_secondary_number_factors_build import _factored_text, _factored_value, _is_prime_trial_division

_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _spread(rng, key, pool, n=3):
    """n values from `pool`, with the key's rank among the n + 1 values drawn at random (not always smallest)."""
    below = [v for v in dict.fromkeys(pool) if v < key]
    above = [v for v in dict.fromkeys(pool) if v > key]
    ranks = [r for r in range(n + 1) if r <= len(below) and n - r <= len(above)]
    check(bool(ranks), "pool too small to spread the key")
    r = rng.choices(ranks, weights=[(1.0, 1.0, 1.5, 2.0)[k] for k in ranks])[0]  # small keys cannot sit high
    return rng.sample(below, r) + rng.sample(above, n - r)


def _result(text, key, wrong, working, plain=False):
    wrong = pick_distractors(str(key), list(wrong), n=3)
    check(len(wrong) == 3, "need three distinct wrong options")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=str(key),
        answer_latex=f"plain:{key}" if plain else str(key),
        working=working,
        distractors=wrong,
    )


# ---------------------------------------------------------------- divisibility


def _prime_factors(n):
    out, d = [], 2
    while d * d <= n:
        if n % d == 0:
            out.append(d)
            while n % d == 0:
                n //= d
        d += 1
    return out + ([n] if n > 1 else [])


def _non_multiples(rng, a, lo, hi):
    """Wrong options for 'a divides it': a number one off a multiple of a, a multiple of one prime factor of a
    only (for a = 6, a multiple of 3 that is not a multiple of 6), then plain non-multiples."""
    near = [m + off for m in range(a * (lo // a + 1), hi, a) for off in (-1, 1) if lo <= m + off <= hi]
    partial = []
    for p in _prime_factors(a):
        if p < a:
            partial += [v for v in range(p * max(1, lo // p), hi + 1, p) if v >= lo and v % a]
    plain = [v for v in range(lo, hi + 1) if v % a]
    out = []
    for source in (near, partial, plain, plain):
        pool = [v for v in source if v not in out]
        if pool:
            out.append(rng.choice(pool))
    return out


_DIVIDES_STEMS = (
    "Which of these numbers does {a} divide exactly (with no remainder)?",
    "{n} shares a number out in groups of {a} with none left over. Which of these could it be?",
    "Which of these numbers is a multiple of {a}, {n}?",
)


def _divides_number(rng, a_range, b_range):
    a = rng.randint(*a_range)
    lo, hi = b_range
    k = rng.randint(max(2, -(-lo // a)), hi // a)
    key = a * k
    wrong = _non_multiples(rng, a, lo, hi)
    check(all(w % a for w in wrong) and key % a == 0, "divisibility key or distractor wrong")
    text = rng.choice(_DIVIDES_STEMS).format(a=a, n=rng.choice(NAMES))
    return _result(text, key, wrong, [f"{key} ÷ {a} = {key // a} with no remainder; the others leave a remainder."])


def _divides_factored(rng):
    primes = sorted(rng.sample([2, 3, 5, 7], 3))
    b_fac = {p: rng.randint(1, 3) for p in primes}
    b = _factored_value(b_fac)
    key_fac = {p: rng.randint(0, b_fac[p]) for p in primes}
    if sum(1 for e in key_fac.values() if e) < 2:
        key_fac[primes[0]], key_fac[primes[1]] = b_fac[primes[0]], 1
    outsiders = [p for p in (2, 3, 5, 7, 11) if p not in b_fac]
    key = _factored_value({p: e for p, e in key_fac.items() if e})
    pool = []
    for culprit in primes + outsiders:
        for _ in range(4):
            w = {p: rng.randint(0, b_fac[p]) for p in primes}
            w[culprit] = b_fac.get(culprit, 0) + 1  # one prime power too many
            value = _factored_value({p: e for p, e in w.items() if e})
            if b % value:
                pool.append(value)
    wrong = _spread(rng, key, pool)
    check(b % key == 0 and all(b % w for w in wrong), "factored divisibility key or distractor wrong")
    text = f"Which of these divides {_factored_text(b_fac)} exactly (with no remainder)?"
    working = [f"{key} uses only prime powers that {_factored_text(b_fac)} also has; each other option needs more."]
    return _result(text, key, wrong, working)


def divides_which(rng, a_range=(2, 10), b_range=(10, 100), form="number"):
    """Four numbers; exactly one is divisible by a. Hard tier: which number divides a factorised number."""
    if form == "factored":
        return _divides_factored(rng)
    return _divides_number(rng, a_range, b_range)


# ---------------------------------------------------------------- odd and even

_PHRASE = {"+": "plus", "-": "minus", "*": "multiplied by"}
_THEN = {"+": "plus", "-": "minus", "*": "multiplied by"}


def _par(op, p, q):
    """Parity rule as a table, so it is a different route from integer arithmetic."""
    if op == "*":
        return "odd" if (p, q) == ("odd", "odd") else "even"
    return "odd" if p != q else "even"


def _value(parity, rng):
    return 2 * rng.randint(1, 40) + (1 if parity == "odd" else 0)


def _apply(op, x, y):
    return x + y if op == "+" else x - y if op == "-" else x * y


# Question stems. Easy and medium keep a small fixed set; hard names a character and asks the reverse question
# ("which does NOT give ..."), which is a new prompt shape, not a bigger number.
_PARITY_STEMS = {
    "easy": ("Which of these gives an {t} result?", "Which of these combinations makes an {t} number?"),
    "medium": (
        "Which of these gives an {t} result?",
        "Which one of these always gives an {t} answer?",
        "Which of these combinations ends up {t}?",
        "Pick the combination whose result is {t}.",
        "Which of these will give an {t} number?",
    ),
}
_PARITY_STEMS_HARD = (
    "{n} asks which of these does not give an {t} result. Which one?",
    "Which of these is the one that does not give an {t} result, {n}?",
    "{n} has four combinations and only one fails to give an {t} result. Which is it?",
)


def _parity_stem(rng, stem, target):
    if stem == "hard":
        return rng.choice(_PARITY_STEMS_HARD).format(n=rng.choice(NAMES), t=target)
    return rng.choice(_PARITY_STEMS[stem]).format(t=target)


def _parity_pick(rng, ops, chained, stem="easy"):
    parities = ("odd", "even")
    if chained:
        combos = [
            (p1, o1, p2, o2, p3) for p1 in parities for p2 in parities for p3 in parities for o1 in ops for o2 in ops
        ]
        result = lambda c: _par(c[3], _par(c[1], c[0], c[2]), c[4])  # noqa: E731
    else:
        combos = [(p1, o, p2) for p1 in parities for p2 in parities for o in ops]
        result = lambda c: _par(c[1], c[0], c[2])  # noqa: E731
    target = rng.choice(parities)
    other = "even" if target == "odd" else "odd"
    negate = stem == "hard"  # hard asks which one does NOT give the named parity
    key_par, wrong_par = (other, target) if negate else (target, other)
    right = [c for c in combos if result(c) == key_par]
    wrong = [c for c in combos if result(c) == wrong_par]
    key = rng.choice(right)
    picks = rng.sample(wrong, 3)

    def words(c):
        if chained:
            p1, o1, p2, o2, p3 = c
            return f"an {p1} number {_PHRASE[o1]} an {p2} number, then that answer {_THEN[o2]} an {p3} number"
        p1, o, p2 = c
        return f"an {p1} number {_PHRASE[o]} an {p2} number"

    # Second route: carry the parities out on real whole numbers.
    def real(c):
        if chained:
            p1, o1, p2, o2, p3 = c
            return _apply(o2, _apply(o1, _value(p1, rng), _value(p2, rng)), _value(p3, rng)) % 2
        p1, o, p2 = c
        return _apply(o, _value(p1, rng), _value(p2, rng)) % 2

    want = 1 if key_par == "odd" else 0
    check(all(real(c) == want for c in [key]) and all(real(c) != want for c in picks), "parity key or distractor wrong")
    texts = [words(c).capitalize() for c in [key, *picks]]
    check(len(set(texts)) == 4, "options must be distinct")
    text = _parity_stem(rng, stem, target)
    working = [f"{texts[0]} gives an {key_par} result; each of the others gives an {wrong_par} one."]
    return _result(text, texts[0], texts[1:], working, plain=True)


def parity_pick(rng, ops=("+", "-"), chained=False, stem="easy"):
    """Four combinations of odd and even numbers; exactly one gives the odd (or even) result named
    (``stem="hard"``: exactly one does not)."""
    return _parity_pick(rng, tuple(ops), chained, stem)


# ---------------------------------------------------------------- prime, composite, square


def _is_square(n):
    return math.isqrt(n) ** 2 == n


_PROPERTY_STEMS = (
    "Which of these is {noun}?",
    "{n} needs {noun}. Which of these will do?",
    "Which of these numbers could {n} choose as {noun}?",
)


def _pick_property(rng, prop, lo, hi, hard):
    nums = range(lo, hi + 1)
    prime = [n for n in nums if _is_prime_trial_division(n)]
    composite = [n for n in nums if n > 3 and not _is_prime_trial_division(n)]
    if hard:  # composites that look prime: odd, no factor 2, 3 or 5
        composite_odd = [n for n in composite if n % 2 and n % 3 and n % 5]
        composite = composite_odd or composite
    square = [n for n in nums if n > 3 and _is_square(n)]
    non_square = [n for n in nums if n > 3 and not _is_square(n)]
    if prop == "prime":
        key = rng.choice(prime)
        wrong = _spread(rng, key, composite)
        why = f"{key} has no factor other than 1 and itself; each other option has another factor."
    elif prop == "composite":
        key = rng.choice(composite)
        wrong = _spread(rng, key, prime)
        why = f"{key} has a factor other than 1 and itself; each other option is prime."
    else:
        key = rng.choice(square)
        near = [n for n in (key - 1, key + 1, (math.isqrt(key) + 1) ** 2 - 1) if n in non_square and n != key]
        rest = [n for n in non_square if n not in near]
        wrong = rng.sample(near, min(2, len(near))) + rng.sample(rest, 3 - min(2, len(near)))
        why = (
            f"{math.isqrt(key)} × {math.isqrt(key)} = {key}; none of the other options is a whole number times itself."
        )
    # Second route: count divisors directly.
    divisors = lambda n: sum(1 for d in range(1, n + 1) if n % d == 0)  # noqa: E731
    if prop == "prime":
        check(divisors(key) == 2 and all(divisors(w) > 2 for w in wrong), "prime key or distractor wrong")
    elif prop == "composite":
        check(divisors(key) > 2 and all(divisors(w) == 2 for w in wrong), "composite key or distractor wrong")
    else:
        check(
            all(math.isqrt(n) ** 2 != n for n in wrong) and math.isqrt(key) ** 2 == key,
            "square key or distractor wrong",
        )
    noun = "a perfect square" if prop == "perfect square" else f"a {prop} number"
    text = rng.choice(_PROPERTY_STEMS).format(noun=noun, n=rng.choice(NAMES))
    return _result(text, key, wrong, [why])


_FACTORED_STEMS = (
    "{n} writes four numbers as products of primes. Which one is not a perfect square?",
    "Written as products of prime factors, which of these is not a perfect square, {n}?",
    "Only one of {n}'s four prime-factor products is not a perfect square. Which is it?",
)


def _pick_nonsquare_factored(rng):
    """Reverse reasoning: three factorised squares and one that is not -- find the odd exponent."""
    primes = sorted(rng.sample([2, 3, 5, 7, 11], rng.choice([2, 3])))
    squares = []
    for _ in attempts(500, "three factorised squares"):
        if len(squares) == 3:
            break
        f = {p: rng.choice([2, 4]) for p in primes}
        if f not in squares:
            squares.append(f)
    key_fac = {p: rng.choice([2, 4]) for p in primes}
    for p in rng.sample(primes, rng.choice([1, 2][: len(primes)])):
        key_fac[p] = rng.choice([1, 3])
    key_n = _factored_value(key_fac)
    sq_n = [_factored_value(f) for f in squares]
    check(not _is_square(key_n) and all(_is_square(n) for n in sq_n), "factored non-square key or distractor wrong")
    texts = [_factored_text(f) for f in [key_fac, *squares]]
    check(len(set(texts)) == 4, "options must be distinct")
    odd = [str(p) for p in primes if key_fac[p] % 2]
    if len(odd) == 1:
        why = f"The exponent of {odd[0]} in {texts[0]} is odd; every exponent in each other option is even."
    else:
        why = (
            f"The exponents of {' and '.join(odd)} in {texts[0]} are odd; every exponent in each other option is even."
        )
    text = rng.choice(_FACTORED_STEMS).format(n=rng.choice(NAMES))
    return _result(text, texts[0], texts[1:], [why])


def _pick_square_factored(rng):
    if rng.random() < 0.5:
        return _pick_nonsquare_factored(rng)
    primes = sorted(rng.sample([2, 3, 5, 7, 11], rng.choice([2, 3])))
    key_fac = {p: rng.choice([2, 4]) for p in primes}
    wrong_fac = []
    for _ in attempts(500, "three factorised non-squares"):
        if len(wrong_fac) == 3:
            break
        w = {p: rng.choice([2, 4]) for p in primes}
        for p in rng.sample(primes, rng.choice([1, 2][: len(primes)])):
            w[p] = rng.choice([1, 3])
        if w != key_fac and w not in wrong_fac:
            wrong_fac.append(w)
    key_n = _factored_value(key_fac)
    wrong_n = [_factored_value(w) for w in wrong_fac]
    check(_is_square(key_n) and not any(_is_square(n) for n in wrong_n), "factored square key or distractor wrong")
    texts = [_factored_text(f) for f in [key_fac, *wrong_fac]]
    check(len(set(texts)) == 4, "options must be distinct")
    why = "Every exponent in " + texts[0] + " is even; each other option has an odd exponent."
    return _result("Which of these is a perfect square?", texts[0], texts[1:], [why])


def property_pick(rng, lo=2, hi=40, prop=None, form="number"):
    """Four numbers; exactly one is prime (or composite, or a perfect square). Hard: factorised squares."""
    if form == "factored":
        return _pick_square_factored(rng)
    return _pick_property(rng, prop or rng.choice(["prime", "composite", "perfect square"]), lo, hi, hard=False)
