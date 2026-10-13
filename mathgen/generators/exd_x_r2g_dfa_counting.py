"""Exam question: running a DFA and counting the words it accepts (`ucv1-finite-automaton-acceptance`).

A deterministic finite automaton on the alphabet {a, b} has four states q0 to q3, start state q0 and a stated set of
accepting states. The parts chain:

* (a) the index of the state the run on a stated word of length 5 ends in;
* (b) how many of the 16 words of length 4 are accepted;
* (c) how many of the 32 words of length 5 are accepted;
* (d) the probability that a uniformly random word of length 5 is accepted, (c)/32 (follow-through from (c)).

Counts come from a dynamic program over the number of words ending in each state, and are re-checked by running
every word.
"""

from __future__ import annotations

import random
from fractions import Fraction
from itertools import product

from ..core import check, generator
from .exd_x_r2g_support import build_part, latex_with_sets, redraws

_STATES = 4
_WORD_LEN = 5


def _run(delta, word: str, start: int = 0) -> int:
    state = start
    for ch in word:
        state = delta[state][ch]
    return state


def _counts(delta, accepting, length: int) -> int:
    ways = [1, 0, 0, 0]
    for _ in range(length):
        nxt = [0] * _STATES
        for s, n in enumerate(ways):
            for ch in "ab":
                nxt[delta[s][ch]] += n
        ways = nxt
    return sum(ways[s] for s in accepting)


def _draw(rng: random.Random) -> dict:
    for _ in range(2000):
        delta = [{"a": rng.randrange(_STATES), "b": rng.randrange(_STATES)} for _ in range(_STATES)]
        accepting = sorted(rng.sample(range(_STATES), rng.randint(1, 2)))
        word = "".join(rng.choice("ab") for _ in range(_WORD_LEN))
        n3, n4, n5 = (_counts(delta, accepting, k) for k in (3, 4, 5))
        end = _run(delta, word)
        early = _run(delta, word[:-1])
        backwards = _run(delta, word[::-1])
        if not (2 <= n4 <= 14 and 4 <= n5 <= 28):
            continue
        if len({end, early, backwards}) < 3 or n4 == n3 or n5 == 2 * n4:
            continue
        reachable = {0}
        for _i in range(_STATES):
            reachable |= {delta[s][ch] for s in reachable for ch in "ab"}
        if len(reachable) < 3:
            continue
        return dict(
            delta=delta, accepting=accepting, word=word, n3=n3, n4=n4, n5=n5, end=end, early=early, back=backwards
        )
    check(False, "no acceptable automaton draw")


_INTRO = [
    "A deterministic finite automaton reads words over the alphabet {{a, b}}. Its states are q0, q1, q2, q3, with "
    "start state q0 and accepting {acc}.",
    "A DFA over the alphabet {{a, b}} has states q0, q1, q2, q3. It starts in q0 and accepts in {acc}.",
]


def _acc_text(accepting) -> str:
    names = [f"q{s}" for s in accepting]
    return f"state {names[0]}" if len(names) == 1 else f"states {' and '.join(names)}"


@generator("exd_x_r2g_dfa_counting", level=15, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_r2g_dfa_counting(rng: random.Random, tier: str = "exam") -> dict:
    """Run a DFA on a word, count accepted words of length 4 and 5, and give an acceptance probability."""
    check(tier == "exam", "exd_x_r2g_dfa_counting: only the exam tier exists")
    d = _draw(rng)
    delta, accepting, word = d["delta"], d["accepting"], d["word"]
    n3, n4, n5 = d["n3"], d["n4"], d["n5"]
    check(
        sum(1 for w in product("ab", repeat=4) if _run(delta, "".join(w)) in accepting) == n4
        and sum(1 for w in product("ab", repeat=5) if _run(delta, "".join(w)) in accepting) == n5,
        "counts disagree with running every word",
    )
    moves = "; ".join(f"from q{s}, a goes to q{delta[s]['a']} and b goes to q{delta[s]['b']}" for s in range(_STATES))
    stem = f"{rng.choice(_INTRO).format(acc=_acc_text(accepting))} Transitions: {moves}."
    prob = Fraction(n5, 32)
    end = d["end"]
    others = [s for s in range(_STATES) if s not in (end, d["early"], d["back"])]
    parts = [
        build_part(
            f'Run the automaton on the word "{word}". Give the number i of the state qi in which the run ends.',
            end,
            kind="number",
            marks=1,
            working=[f"Starting at q0 and reading {', '.join(word)} in turn gives the run ending in q{end}."],
            wrong=[
                (d["early"], "stops-one-symbol-early"),
                (d["back"], "reads-the-word-backwards"),
                *((s, "traces-transition-table-wrongly") for s in others),
            ],
        ),
        build_part(
            "How many of the 16 words of length 4 over {a, b} does the automaton accept?",
            n4,
            kind="number",
            marks=2,
            pad_counts=True,
            working=[
                "Count the words ending in an accepting state, extending the counts one letter at a time.",
                f"There are {n4}.",
            ],
            wrong=[
                (n3, "uses-the-wrong-word-length"),
                (16 - n4, "counts-the-complement"),
                (n5, "uses-the-wrong-word-length"),
                (len(accepting), "counts-accepting-states"),
            ],
        ),
        build_part(
            "How many of the 32 words of length 5 over {a, b} does the automaton accept?",
            n5,
            kind="number",
            marks=2,
            pad_counts=True,
            working=[f"Extending the counts one more letter gives {n5}."],
            wrong=[
                (2 * n4, "assumes-every-accepted-word-extends-twice"),
                (32 - n5, "counts-the-complement"),
                (n4, "uses-the-wrong-word-length"),
                (n5 + n4, "uses-the-wrong-word-length"),
            ],
        ),
        build_part(
            "A word of length 5 is chosen uniformly at random. Hence find the probability that it is accepted, as a "
            "fraction in lowest terms.",
            prob,
            kind="fraction",
            marks=1,
            follow_through="part_c/32",
            working=[f"There are 32 equally likely words and {n5} are accepted: {prob}."],
            wrong=[
                (Fraction(n5, 63), "wrong-denominator-for-proportion"),
                (Fraction(n4, 32), "uses-the-wrong-word-length"),
                (Fraction(32 - n5, 32), "gave-complement-for-event"),
                (Fraction(n5, 31), "wrong-denominator-for-proportion"),
            ],
        ),
    ]
    return dict(text=stem, latex=latex_with_sets(stem), parts=parts)
