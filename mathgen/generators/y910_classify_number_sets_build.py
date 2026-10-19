"""Classify real numbers by the sets they belong to, past a two-way rational/irrational guess (Year 9-10).

Three topics' worth of reasoning in one module. A number is written in a form that hides its set (a fraction that
is a whole number, a recurring decimal, a root of a perfect square, a product of surds), and the learner has to
simplify or recognise it before classifying.

Ladder (reasoning escalation, not size):

* easy: ONE number, and the learner types every set that contains it, from N (natural numbers 1, 2, 3, ...),
  Z (integers), Q (rational), I (irrational) and R (real). The sets nest, so a number such as -18/3 is in Z, Q and
  R and stopping at the first set is wrong. A typed set of letters, never a choice between two options.
* medium: six labelled numbers; type the letters of the irrational ones. Several rational numbers are written like
  irrational ones (a root of a perfect square, a recurring decimal, a fraction such as 22/7) and some irrational
  ones are written like rational ones (a non-repeating decimal, a surd sum).
* hard: eight expressions to simplify first (for example the square root of 12 divided by the square root of 3, or
  (2 + the square root of 3) times (2 - the square root of 3)); type HOW MANY are irrational.

Every number is built from an exact SymPy value (or an exact Fraction for decimals), and its classification is
decided by SymPy (``is_rational``) as well as by the rule that built it; the two must agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors
from mathgen.grading.typed import set_answer_text

TOPIC = "y910_classify_number_sets"
TIERS = ("easy", "medium", "hard")

_SETS_NOTE = (
    "N is the natural numbers (1, 2, 3, ...), Z the integers, Q the rational numbers, I the irrational numbers "
    "and R the real numbers."
)
_NON_SQUARES = [n for n in range(2, 80) if math.isqrt(n) ** 2 != n]
_SQUARES = [n * n for n in range(2, 16)]
_NON_CUBES = [n for n in range(2, 60) if round(n ** (1 / 3)) ** 3 != n]


@dataclass(frozen=True)
class Num:
    """A number as the learner sees it: its decimal value, whether it is rational (known from how it was built)
    and the kind of disguise it wears."""

    text: str
    value: float
    rational: bool
    form: str
    shows_root: bool = False  # written with a root sign or pi, so it LOOKS irrational at a glance

    @property
    def integer(self) -> bool:
        return self.rational and abs(self.value - round(self.value)) < 1e-9

    @property
    def natural(self) -> bool:
        return self.integer and self.value > 0

    @property
    def exact(self) -> Fraction:
        """The exact value of a rational number."""
        check(self.rational, "only a rational number has an exact value")
        return Fraction(self.value).limit_denominator(100000)


def _num(text, value, rational, form, shows_root=False) -> Num:
    """A number. A number built as rational is checked a second way: its decimal value must be within 1e-12 of a
    fraction with a small denominator (the irrational forms are irrational by construction: roots of non-squares,
    sums with them, multiples of pi, and decimals with no repeating block)."""
    value = float(value)
    if rational:
        close = abs(float(Fraction(value).limit_denominator(20000)) - value) < 1e-12
        check(close, f"{text}: built as rational but its decimal value is not a small fraction")
    return Num(text, value, rational, form, shows_root)


def _decimal_text(f: Fraction, places: int) -> str:
    scaled = f * 10**places
    check(scaled.denominator == 1, "not a terminating decimal at that length")
    sign = "-" if f < 0 else ""
    digits = str(abs(scaled.numerator)).zfill(places + 1)
    return f"{sign}{digits[:-places]}.{digits[-places:]}"


# ---------------------------------------------------------------------------------------------------- builders
# Each takes the rng and returns a Num.


def _int(rng, positive=None) -> Num:
    n = rng.choice([v for v in range(-30, 41) if v and (positive is None or (v > 0) == positive)])
    return _num(str(n), n, True, "int")


def _int_fraction(rng) -> Num:
    q = rng.randint(2, 9)
    k = rng.choice([v for v in range(-9, 10) if v and abs(v) > 1])
    return _num(f"{k * q}/{q}", k, True, "int-fraction")


def _fraction(rng) -> Num:
    p, q = rng.choice([(a, b) for b in range(2, 12) for a in range(1, 24) if math.gcd(a, b) == 1 and a % b])
    sign = rng.choice([1, -1])
    return _num(f"{'-' if sign < 0 else ''}{p}/{q}", sign * p / q, True, "fraction")


def _terminating(rng) -> Num:
    places = rng.choice([1, 2, 3])
    f = Fraction(rng.randint(11, 9999), 10**places)
    f = f if f.denominator != 1 else f + Fraction(1, 10**places)
    sign = rng.choice([1, 1, -1])
    return _num(_decimal_text(sign * f, places), float(sign * f), True, "terminating")


def _recurring(rng) -> Num:
    if rng.random() < 0.5:
        d = rng.randint(1, 8)
        value = Fraction(d, 9)
        text = f"0.{str(d) * 6}... (the repeating digit is {d})"
    else:
        pair = rng.choice([12, 27, 36, 45, 54, 63, 18, 81, 15, 72])
        value = Fraction(pair, 99)
        text = f"0.{str(pair).zfill(2) * 3}... (the repeating block is {str(pair).zfill(2)})"
    return _num(text, float(value), True, "recurring")


def _square_root(rng) -> Num:
    n = rng.choice(_SQUARES)
    return _num(f"√{n}", math.sqrt(n), True, "perfect-root", True)


def _root_of_fraction(rng) -> Num:
    p, q = rng.choice([(a, b) for a in range(1, 9) for b in range(2, 10) if math.gcd(a, b) == 1])
    return _num(f"√({p * p}/{q * q})", p / q, True, "perfect-root", True)


def _cube_root(rng) -> Num:
    n = rng.choice([2, 3, 4, 5, -2, -3, -4])
    return _num(f"∛({n**3})", n, True, "perfect-root", True)


def _surd_product(rng) -> Num:
    # sqrt(a) x sqrt(b) with a*b a perfect square: rational in disguise
    r = rng.choice([2, 3, 5, 6, 7])
    k = rng.choice([2, 3, 4])
    a, b = r, r * k * k
    if rng.random() < 0.5:
        return _num(f"√{a} × √{b}", math.sqrt(a) * math.sqrt(b), True, "surd-product", True)
    return _num(f"√{b} ÷ √{a}", math.sqrt(b) / math.sqrt(a), True, "surd-quotient", True)


def _square_of_root(rng) -> Num:
    n = rng.choice(_NON_SQUARES)
    return _num(f"(√{n})²", math.sqrt(n) ** 2, True, "square-of-root", True)


def _conjugates(rng) -> Num:
    a, n = rng.randint(2, 6), rng.choice([2, 3, 5, 7])
    return _num(f"({a} + √{n})({a} − √{n})", (a + math.sqrt(n)) * (a - math.sqrt(n)), True, "conjugates", True)


def _pi_ratio(rng) -> Num:
    k = rng.randint(2, 5)
    return _num(f"{k}π ÷ π", (k * math.pi) / math.pi, True, "pi-ratio", True)


def _surd(rng) -> Num:
    n = rng.choice(_NON_SQUARES)
    return _num(f"√{n}", math.sqrt(n), False, "surd", True)


def _cube_surd(rng) -> Num:
    n = rng.choice(_NON_CUBES)
    return _num(f"∛{n}", n ** (1 / 3), False, "surd", True)


def _pi_multiple(rng) -> Num:
    k = rng.choice([1, 2, 3, 4])
    return _num("π" if k == 1 else f"{k}π", k * math.pi, False, "pi", True)


def _surd_sum(rng) -> Num:
    a, b = rng.sample([2, 3, 5, 6, 7, 10, 11], 2)
    if rng.random() < 0.5:
        return _num(f"√{a} + √{b}", math.sqrt(a) + math.sqrt(b), False, "surd-sum", True)
    k = rng.randint(1, 6)
    return _num(f"{k} + √{a}", k + math.sqrt(a), False, "surd-sum", True)


def _surd_reciprocal(rng) -> Num:
    n = rng.choice([2, 3, 5, 6, 7])
    return _num(f"1 ÷ √{n}", 1 / math.sqrt(n), False, "surd-reciprocal", True)


def _nonrepeating(rng) -> Num:
    # no exact value: its irrationality is stated by construction (a pattern that never repeats)
    if rng.random() < 0.5:
        return _num("0.1010010001... (one more zero between the ones each time)", 0.1010010001, False, "nonrepeating")
    return _num(
        "0.123456789101112... (the counting numbers written one after another)",
        0.12345678910111213,
        False,
        "nonrepeating",
    )


def _pi_approx(rng) -> Num:
    text, value = rng.choice(
        [
            ("22/7", Fraction(22, 7)),
            ("3.14", Fraction(314, 100)),
            ("3.1416", Fraction(31416, 10000)),
            ("355/113", Fraction(355, 113)),
        ]
    )
    return _num(text, float(value), True, "pi-approximation")


def _is_irrational(n: Num) -> bool:
    return not n.rational


_RATIONAL_BUILDERS = [
    _int, _int_fraction, _fraction, _terminating, _recurring, _square_root, _root_of_fraction, _cube_root,
    _pi_approx,
]  # fmt: skip
_DISGUISED_RATIONAL = [_square_root, _root_of_fraction, _cube_root, _recurring, _int_fraction, _pi_approx, _terminating]
_HARD_DISGUISED = [_surd_product, _square_of_root, _conjugates, _pi_ratio, _square_root, _root_of_fraction, _cube_root]
_IRRATIONAL_BUILDERS = [_surd, _cube_surd, _pi_multiple, _surd_sum, _surd_reciprocal, _nonrepeating]


def _membership(n: Num) -> list[str]:
    """Letters of the sets N, Z, Q, I, R that contain the number, in that order."""
    if _is_irrational(n):
        return ["I", "R"]
    out = []
    if n.natural:
        out.append("N")
    if n.integer:
        out.append("Z")
    return [*out, "Q", "R"]


# ---------------------------------------------------------------------------------------------------- easy

_EASY_NOTES = {
    "int": "{x} is a whole number, so it sits in {chain}.",
    "int-fraction": "{x} simplifies to a whole number, so it sits in {chain}.",
    "fraction": "{x} is a fraction of whole numbers that is not a whole number, so it sits in {chain}.",
    "terminating": "{x} is a terminating decimal, so it is a fraction of whole numbers and sits in {chain}.",
    "recurring": "A repeating decimal is a fraction of whole numbers, so it sits in {chain}.",
    "perfect-root": "{x} simplifies to a rational number, so it sits in {chain}.",
    "pi-approximation": "{x} is a decimal that stops, so it is a fraction of whole numbers and sits in {chain}.",
    "surd": "{x} is a root that never ends or repeats, so it sits in {chain}.",
    "pi": "{x} never ends or repeats, so it sits in {chain}.",
}


def _easy_number(rng) -> Num:
    pool = [
        lambda r: _int(r, positive=True),
        lambda r: _int(r, positive=False),
        _int_fraction,
        _fraction,
        _terminating,
        _recurring,
        _square_root,
        _root_of_fraction,
        _cube_root,
        _surd,
        _cube_surd,
        _pi_multiple,
    ]
    return rng.choice(pool)(rng)


def _set_variants(n: Num, key: list[str]) -> list[tuple[list[str], str]]:
    """Wrong sets, each a list of letters with the error it comes from."""
    out: list[tuple[list[str], str]] = []
    order = "NZQIR"
    irrational = "I" in key

    def add(letters, note):
        letters = [x for x in order if x in set(letters)]
        if letters and letters != key:
            out.append((letters, note))

    if not irrational:
        add([key[0]], "named-only-the-smallest-set")
        add([x for x in key if x != "R"], "left-out-the-real-numbers")
        if n.integer:
            add([x for x in key if x != "Q"], "integers-not-counted-as-rational")
        if n.form in ("perfect-root",):
            add(["I", "R"], "every-root-treated-as-irrational")
        if n.form == "recurring":
            add(["I", "R"], "recurring-decimal-treated-as-irrational")
        if not n.integer:
            add(["Z", "Q", "R"], "fraction-counted-as-an-integer")
        if n.form in ("fraction", "terminating", "pi-approximation"):
            add(["I", "R"], "non-whole-number-treated-as-irrational")
        if n.integer and n.value < 0:
            add(["N", "Z", "Q", "R"], "negative-number-counted-as-natural")
        if n.form == "int-fraction":
            add(["Q", "R"], "fraction-not-simplified-to-an-integer")
    else:
        add(["I"], "left-out-the-real-numbers")
        add(["Q", "R"], "irrational-treated-as-rational")
        add(["Q", "I", "R"], "irrational-also-counted-as-rational")
        add(["Z", "Q", "R"], "irrational-treated-as-an-integer")
    return out


def _easy(rng):
    n = _easy_number(rng)
    key = _membership(n)
    ok = [x for x in _set_variants(n, key)]
    check(len(ok) >= 3, "too few wrong sets for an easy number")
    answer_text = set_answer_text(key)
    candidates = [Distractor(set_answer_text(letters), note, letters) for letters, note in ok]
    distractors = pick_distractors(answer_text, candidates, 3, unordered=True)
    chain = ", ".join(key[:-1]) + " and " + key[-1] if len(key) > 1 else key[0]
    shown = f"the number {n.text}" if not n.text.startswith("-") else f"the number {n.text}"
    text = (
        f"{_SETS_NOTE} Which of these sets contain {shown}? Write the letters of every set that contains it, in braces."
    )
    working = [
        _EASY_NOTES.get(n.form, "{x} sits in {chain}.").format(x=n.text, chain=chain),
        "The sets nest: every natural number is an integer, every integer is rational, and every rational and irrational number is real.",
    ]
    return text, key, answer_text, distractors, working, "set"


# ---------------------------------------------------------------------------------------------------- medium

_LABELS = "ABCDEF"


def _distinct_texts(nums: list[Num]) -> bool:
    return len({n.text for n in nums}) == len(nums) and len({round(n.value, 9) for n in nums}) == len(nums)


def _medium(rng):
    for _ in attempts(500, "six numbers to classify"):
        k = rng.choice([2, 2, 3])
        irr = [rng.choice(_IRRATIONAL_BUILDERS)(rng) for _ in range(k)]
        rat = [rng.choice(_DISGUISED_RATIONAL)(rng) for _ in range(6 - k)]
        nums = irr + rat
        if not _distinct_texts(nums):
            continue
        # at least two rational numbers must look irrational or be a decimal that goes on
        if sum(1 for n in rat if n.shows_root or n.form == "recurring") < 2:
            continue
        if not any(n.form in ("nonrepeating", "surd-sum", "pi") for n in irr):
            continue
        order = list(range(6))
        rng.shuffle(order)
        shuffled = [nums[i] for i in order]
        break
    key = [_LABELS[i] for i, n in enumerate(shuffled) if _is_irrational(n)]
    check(len(key) == k, "the count of irrational numbers moved")
    answer_text = set_answer_text(key)
    wrong: list[tuple[list[str], str]] = []
    for i, n in enumerate(shuffled):
        label = _LABELS[i]
        if _is_irrational(n):
            note = (
                "irrational-written-as-a-sum-treated-as-rational"
                if n.form == "surd-sum"
                else "irrational-with-no-root-sign-treated-as-rational"
            )
            if n.form in ("pi", "nonrepeating"):
                note = "irrational-with-no-root-sign-treated-as-rational"
            wrong.append(([x for x in key if x != label], note))
        else:
            note = {
                "perfect-root": "every-root-treated-as-irrational",
                "recurring": "recurring-decimal-treated-as-irrational",
                "pi-approximation": "pi-approximation-treated-as-irrational",
                "int-fraction": "non-whole-number-treated-as-irrational",
            }.get(n.form, "non-whole-number-treated-as-irrational")
            wrong.append((sorted([*key, label]), note))
    rng.shuffle(wrong)
    candidates = []
    seen_notes: set[str] = set()
    for letters, note in sorted(wrong, key=lambda w: w[1] in seen_notes):
        seen_notes.add(note)
        candidates.append(Distractor(set_answer_text(letters) if letters else "{ }", note, letters))
    distractors = pick_distractors(answer_text, candidates, 3, unordered=True)
    listing = "; ".join(f"{_LABELS[i]}: {n.text}" for i, n in enumerate(shuffled))
    text = (
        f"Six numbers are listed. {listing}. "
        f"Which of them are irrational? Write the letters of the irrational numbers, in braces."
    )
    working = [
        f"{_LABELS[i]} ({n.text}) is {'irrational' if _is_irrational(n) else 'rational'}"
        + ("" if _is_irrational(n) else f": it equals {n.exact}")
        + "."
        for i, n in enumerate(shuffled)
    ]
    return text, key, answer_text, distractors, working, "set"


# ---------------------------------------------------------------------------------------------------- hard


def _hard(rng):
    plain = [_recurring, _fraction, _terminating, _pi_approx]
    for _ in attempts(1000, "eight expressions to classify"):
        k = rng.randint(3, 5)
        irr = [
            rng.choice([_surd, _surd_sum, _surd_reciprocal, _pi_multiple, _cube_surd, _nonrepeating])(rng)
            for _ in range(k)
        ]
        n_plain = rng.choice([1, 2])
        rat = [rng.choice(_HARD_DISGUISED)(rng) for _ in range(8 - k - n_plain)] + [
            rng.choice(plain)(rng) for _ in range(n_plain)
        ]
        nums = irr + rat
        if not _distinct_texts(nums):
            continue
        if (
            sum(
                1
                for n in rat
                if n.form in ("surd-product", "surd-quotient", "square-of-root", "conjugates", "pi-ratio")
            )
            < 2
        ):
            continue
        break
    rng.shuffle(nums)
    letters8 = "ABCDEFGH"
    key = [letters8[i] for i, n in enumerate(nums) if n.rational]
    check(len(key) == 8 - k, "the count of rational numbers moved")
    answer_text = set_answer_text(key)
    by_sign = [letters8[i] for i, n in enumerate(nums) if not (n.shows_root or n.form == "nonrepeating")]
    disguised = [letters8[i] for i, n in enumerate(nums) if n.rational and n.shows_root]
    sums = [letters8[i] for i, n in enumerate(nums) if not n.rational and n.form in ("surd-sum", "surd-reciprocal")]
    wrong = [
        (by_sign, "judged-by-the-root-sign-without-simplifying"),
        ([x for x in letters8 if x not in key], "listed-the-irrational-numbers-instead"),
        (
            [x for x in key if x != disguised[0]] if disguised else [],
            "left-out-a-rational-number-that-needs-simplifying",
        ),
        (
            [x for x in key if x != disguised[-1]] if disguised else [],
            "left-out-a-rational-number-that-needs-simplifying",
        ),
        (sorted([*key, sums[0]]) if sums else [], "irrational-sum-treated-as-rational"),
    ]
    candidates = [Distractor(set_answer_text(v), note, v) for v, note in wrong if v and v != key]
    distractors = pick_distractors(answer_text, candidates, 3, unordered=True)
    listing = "; ".join(f"{letters8[i]}: {n.text}" for i, n in enumerate(nums))
    text = (
        f"Eight numbers are listed. {listing}. Simplify where needed. "
        f"Which of them are rational? Write the letters of the rational numbers, in braces."
    )
    working = [
        f"{letters8[i]} ({n.text}): {'rational, it equals ' + str(n.exact) if n.rational else 'irrational'}."
        for i, n in enumerate(nums)
    ]
    return text, key, answer_text, distractors, working, "set"


_BUILDERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(TOPIC, level=9, presets={t: {"tier": t} for t in TIERS})
def y910_classify_number_sets(rng, tier="easy"):
    """Classify numbers by their sets, simplifying first; see the module docstring."""
    check(tier in _BUILDERS, f"unknown tier {tier!r}")
    text, key, answer_text, distractors, working, kind = _BUILDERS[tier](rng)
    check(len(distractors) >= 2, "too few usable wrong answers")
    return dict(
        text=text,
        answer=key,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True) if kind == "set" else latex_answer(answer_text),
        working=working,
        distractors=distractors,
        answer_kind=kind,
    )
