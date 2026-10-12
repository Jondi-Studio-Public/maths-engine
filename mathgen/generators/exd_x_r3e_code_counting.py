"""Exam-tier source for `multiplication-principle`: counting codes with and without repeats.

A code is a sequence of k symbols taken from N. Three settings share the structure (a padlock dial numbered 1 to N
with N between 20 and 40 and a combination of 3 or 4 numbers, a digit code, a letter code), and the chain is the usual counting ladder:

(a) all codes                                    N^k                   number, 1 mark
(b) codes with no symbol repeated                N (N-1) ... (N-k+1)   number, 2 marks
(c) codes with at least one repeat               N^k - P(N, k)         number, 1 mark    part_a - part_b
(d) no repeats and the first symbol restricted   m (N-1) ... (N-k+1)   number, 2 marks

where m counts the allowed first symbols (the multiples of 3, 4 or 5 on the dial, odd digits, vowels).
"""

from __future__ import annotations

from math import comb, perm

from ..answers import Distractor
from ..core import check, generator, phrase, pick_distractors
from ._exd_r3_distinct import distinct_part_answers

_LIMIT = 10**9


def _options(key: int, candidates, n: int = 3):
    out, seen = [], {key}
    for tag, value in candidates:
        if value in seen or value <= 0 or value >= _LIMIT * 10**3:
            continue
        seen.add(value)
        out.append(Distractor(str(value), tag, value=value))
    return pick_distractors(str(key), out, n)


@generator("exd_x_r3e_code_counting", level=13, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_code_counting(rng, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    setting = rng.choices(["dial", "digits", "letters"], weights=[10, 1, 1])[0]
    if setting == "dial":
        N, k, step = rng.randint(20, 40), rng.choice([3, 4]), rng.choice([3, 4, 5])
        symbol, m, first = "number", N // step, f"a multiple of {step}"
        stem = phrase(
            rng,
            (
                "A padlock dial is numbered 1 to {N}. Its combination is a sequence of {k} numbers.",
                "A combination safe has a dial numbered 1 to {N}, and its combination is a sequence of {k} numbers.",
            ),
            N=N,
            k=k,
        )
    elif setting == "digits":
        N, k = 10, rng.choice([4, 5, 6])
        symbol, m, first = "digit", 5, "an odd digit"
        stem = f"A security code is a sequence of {k} digits, each from 0 to 9."
    else:
        N, k = 26, rng.choice([3, 4, 5])
        symbol, m, first = "letter", 5, "a vowel"
        stem = f"A booking reference is a sequence of {k} letters, each from A to Z."

    total = N**k
    no_repeat = perm(N, k)
    at_least_one = total - no_repeat
    restricted = m * perm(N - 1, k - 1)
    check(total == N**k and no_repeat == _product(N, k) and restricted == m * _product(N - 1, k - 1), "counts disagree")
    check(at_least_one > 0 and restricted < no_repeat, "counts out of order")

    a_opts = _options(
        total,
        [
            ("counts-added-not-multiplied", N * k),
            ("choices-reduced-though-repeats-allowed", no_repeat),
            ("base-and-exponent-swapped", k**N if k**N < _LIMIT else 0),
            ("permutation-replaced-by-combination", comb(N, k)),
        ],
    )
    b_opts = _options(
        no_repeat,
        [
            ("repeats-allowed-used-for-no-repeats", total),
            ("permutation-replaced-by-combination", comb(N, k)),
            ("choices-not-reduced-after-use", N * (N - 1) ** (k - 1)),
            ("excluded-only-the-all-same-codes", total - N),
        ],
    )
    c_opts = _options(
        at_least_one,
        [
            ("complement-reported", no_repeat),
            ("reversed-subtraction", no_repeat - total),
            ("all-codes-reported-for-at-least-one-repeat", total),
            ("permutation-replaced-by-combination", total - comb(N, k)),
        ],
    )
    d_opts = _options(
        restricted,
        [
            ("choices-not-reduced-after-use", m * perm(N, k - 1)),
            ("restriction-ignored", no_repeat),
            ("repeats-allowed-used-for-no-repeats", m * (N - 1) ** (k - 1)),
            ("permutation-replaced-by-combination", m * comb(N - 1, k - 1)),
        ],
    )
    check(min(len(a_opts), len(b_opts), len(c_opts), len(d_opts)) >= 3, "too few distractors")
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="How many different combinations are possible?"
                if setting == "dial"
                else "How many different codes are possible?",
                answer_kind="number",
                answer=total,
                answer_text=str(total),
                marks=1,
                working=[f"Each of the {k} positions has {N} choices: {N}^{k} = {total}."],
                distractors=a_opts,
            ),
            dict(
                prompt=f"How many use no {symbol} more than once?",
                answer_kind="number",
                answer=no_repeat,
                answer_text=str(no_repeat),
                marks=2,
                working=[
                    f"The choices fall by one each time: {' x '.join(str(N - i) for i in range(k))} = {no_repeat}."
                ],
                distractors=b_opts,
            ),
            dict(
                prompt=f"How many use at least one {symbol} more than once?",
                answer_kind="number",
                answer=at_least_one,
                answer_text=str(at_least_one),
                marks=1,
                working=[f"All of them, less those with no repeat: {total} - {no_repeat} = {at_least_one}."],
                distractors=c_opts,
                follow_through="part_a - part_b",
            ),
            dict(
                prompt=f"How many use no {symbol} more than once and begin with {first}?",
                answer_kind="number",
                answer=restricted,
                answer_text=str(restricted),
                marks=2,
                working=[
                    f"{m} choices for the first position, then {' x '.join(str(N - 1 - i) for i in range(k - 1))}: "
                    f"{m} x {no_repeat // N} = {restricted}."
                ],
                distractors=d_opts,
            ),
        ],
    )


def _product(top: int, count: int) -> int:
    out = 1
    for i in range(count):
        out *= top - i
    return out
