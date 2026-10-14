"""Generates `f10-number-algorithm-choose-complete` questions and its exam chain. `answer: mcq` (the exam chain also has
typed number and ordered-list parts).

Scope: Years 3 to 6 algorithms with a stated goal. Where the neighbouring skill asks the learner to trace an algorithm
that is given, this one gives the goal and asks for the algorithm: choose the instructions that produce a stated list,
put scrambled steps in an order that works, or fill in the missing decision of a loop from the numbers it writes.
Writing an algorithm of one's own is out of scope; every question has exactly one answer, checked by running each
candidate.

Ladder (reasoning escalation, not size):
  easy:   choose the instructions. Four step lists differ in one thing each (where the list starts, how far it goes,
          the step size); only one writes exactly the numbers the goal describes.
  medium: choose the order of four scrambled number-machine steps that turns a stated input into a stated output.
          The right order is unique (all 24 orders are run).
  hard:   fill in the missing rule of a loop with a branch. The numbers the loop wrote are shown; every wrong rule
          agrees with the right one on the first odd number, so only checking every step separates them.
  exam:   a four-part chain on one number machine: the order of its steps (2 marks), its output for a new input,
          the output when that is fed back in (follows through), and the input that gives a stated output.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

from ..core import check, generator, join_terms
from ._fr2e_kit import Redraw, draw, letter_order, mcq_part, number_part, single, words_part
from .exd_x_r2b_support import exact_number_part as _exact  # noqa: F401  (typed numbers are exact here)
from ._fr2e_kit import exact_number_part

E_START = "fr2e-algorithm-wrong-start"
E_BOUNDARY = "fr2e-algorithm-stop-test-boundary"
E_STEP = "fr2e-algorithm-wrong-step-size"
E_ONE_STEP = "fr2e-algorithm-rule-fitted-to-one-step"
E_LISTED = "fr2e-algorithm-steps-in-listed-order"
E_REVERSED = "fr2e-algorithm-steps-in-reverse-order"
E_SWAPPED = "fr2e-algorithm-two-steps-swapped"
E_FORWARD = "fr2e-algorithm-steps-run-forwards-to-undo"
E_ONE_RUN = "fr2e-algorithm-one-run-for-two"
E_INPUT = "fr2e-algorithm-input-reported-as-output"


# -- easy: choose the instructions -----------------------------------------------------------------------------------


def run_list(start: int, step: int, limit: int, inclusive: bool) -> list[int]:
    """The numbers written by: write `start`; repeat: add `step` to the last number, and stop if the new number is
    beyond the limit (`limit` or more when exclusive, more than `limit` when inclusive), else write it."""
    out = [start]
    n = start
    for _ in range(200):
        n += step
        if (n > limit) if inclusive else (n >= limit):
            return out
        out.append(n)
    raise Redraw("the list does not end")


def _instruction(start: int, step: int, limit: int, inclusive: bool) -> str:
    stop = f"more than {limit}" if inclusive else f"{limit} or more"
    return f"Write {start}. Then repeat: add {step} to the last number written; if the new number is {stop}, stop; otherwise write it."


def _easy(rng):
    mode = rng.choice(("multiples", "from"))
    inclusive = rng.random() < 0.5
    if mode == "multiples":
        d = rng.choice((3, 4, 5, 6, 7, 8, 9))
        start, step = d, d
        k = rng.randint(5, 9)
        limit = d * k
        goal = (
            f"the positive multiples of {d} that are less than {limit}"
            if not inclusive
            else f"the positive multiples of {d} up to and including {limit}"
        )
    else:
        step = rng.choice((3, 4, 5, 6, 7, 9))
        start = rng.randint(2, 15)
        k = rng.randint(5, 9)
        limit = start + step * k
        goal = (
            f"the numbers that start at {start}, go up by {step} each time and stay below {limit}"
            if not inclusive
            else f"the numbers that start at {start}, go up by {step} each time and never go above {limit}"
        )
    target = run_list(start, step, limit, inclusive)
    if mode == "multiples":
        check(target == [d * i for i in range(1, len(target) + 1)], "the list must be the multiples")
    wrong_step = step + rng.choice((-1, 1)) if step > 3 else step + 1
    if mode == "multiples":
        wrong_start = 0
    else:
        wrong_start = start + step
    candidates = [
        (_instruction(start, step, limit, inclusive), "key", None),
        (_instruction(wrong_start, step, limit, inclusive), E_START, None),
        (_instruction(start, step, limit, not inclusive), E_BOUNDARY, None),
        (_instruction(start, wrong_step, limit, inclusive), E_STEP, None),
    ]
    results = [
        run_list(wrong_start, step, limit, inclusive),
        run_list(start, step, limit, not inclusive),
        run_list(start, wrong_step, limit, inclusive),
    ]
    check(all(r != target for r in results), "each wrong instruction must write a different list")
    check(len({tuple(r) for r in results}) == 3, "the wrong lists must differ from one another")
    key_text = candidates[0][0]
    stem = f"A list of numbers is needed: {goal}. Which instructions write exactly that list?"
    part = mcq_part(
        "Choose the correct instructions.",
        key_text,
        [(t, tag) for t, tag, _ in candidates[1:]],
        1,
        [f"The instructions write {', '.join(str(v) for v in target)}."],
    )
    out = single(stem, part)
    out["text"] = stem
    return out


# -- medium and the exam: a number machine ---------------------------------------------------------------------------

_OPS = (
    ("Add {a}", lambda n, a: n + a, (2, 9)),
    ("Subtract {a}", lambda n, a: n - a, (2, 9)),
    ("Multiply by {a}", lambda n, a: n * a, (2, 5)),
    ("Divide by {a}", lambda n, a: n // a if n % a == 0 else None, (2, 4)),
)


class Op:
    def __init__(self, kind: int, arg: int):
        self.kind, self.arg = kind, arg

    @property
    def text(self) -> str:
        return _OPS[self.kind][0].format(a=self.arg)

    def run(self, n: int):
        if n is None:
            return None
        out = _OPS[self.kind][1](n, self.arg)
        return out if out is not None and 0 <= out <= 400 else None

    def undo(self, n):
        """The input that gives output `n`, or None when there is no whole non-negative one."""
        if self.kind == 0:
            out = n - self.arg
        elif self.kind == 1:
            out = n + self.arg
        elif self.kind == 2:
            out = n // self.arg if n % self.arg == 0 else None
        else:
            out = n * self.arg
        return out if out is not None and 0 <= out <= 400 else None

    def sympy_text(self, v: str) -> str:
        return [f"({v} + {self.arg})", f"({v} - {self.arg})", f"({v} * {self.arg})", f"({v} / {self.arg})"][self.kind]


def exact_run(ops, n):
    """Run the steps in exact arithmetic with no limits, so an order counts as working whatever its middle values are."""
    v = Fraction(n)
    for op in ops:
        v = [v + op.arg, v - op.arg, v * op.arg, v / op.arg][op.kind]
    return v


def float_run(ops, n):
    """A quick screen for `exact_run`: the same steps in floating point."""
    v = float(n)
    for op in ops:
        v = (v + op.arg, v - op.arg, v * op.arg, v / op.arg)[op.kind]
    return v


def run_machine(ops, n):
    for op in ops:
        n = op.run(n)
    return n


def _machine(rng):
    """(ops in their working order, input, output) with a unique working order among all 24."""
    kinds = rng.sample(range(4), 3) + [rng.choice(range(4))]
    for _ in range(80):
        ops, texts = [], set()
        for kind in kinds:
            lo, hi = _OPS[kind][2]
            op = Op(kind, rng.randint(lo, hi))
            ops.append(op)
        if len({o.text for o in ops}) < 4:
            continue
        # Two neighbouring steps of the same family (add/subtract, or multiply/divide) swap without changing the result.
        if any(a.kind // 2 == b.kind // 2 for a, b in zip(ops, ops[1:])):
            continue
        n = rng.randint(2, 15)
        out = run_machine(ops, n)
        if out is None or out == n or not 2 <= out <= 200:
            continue
        works = [
            p
            for p in itertools.permutations(range(4))
            if abs(float_run([ops[i] for i in p], n) - out) < 1e-6 and exact_run([ops[i] for i in p], n) == out
        ]
        if works == [(0, 1, 2, 3)]:
            return ops, n, out
    raise Redraw("no machine with a unique working order")


def _scrambled(rng, ops):
    lines, key = letter_order(rng, [o.text for o in ops])
    return lines, key


def _swap_adjacent(key, i):
    k = list(key)
    k[i], k[i + 1] = k[i + 1], k[i]
    return k


def _order_wrongs(key, letters_sorted):
    wrong = [(letters_sorted, E_LISTED), (list(reversed(key)), E_REVERSED)]
    for i in range(len(key) - 1):
        wrong.append((_swap_adjacent(key, i), E_SWAPPED))
    return wrong


def _medium(rng):
    ops, n, out = draw(_machine, rng)
    lines, key = _scrambled(rng, ops)
    letters = sorted(key)
    wrong = [(w, t) for w, t in _order_wrongs(key, letters) if w != key]
    # every wrong order must fail when run
    stem = (
        f"A number machine has four steps, each used once, in an order that is not shown: {'; '.join(lines)}. "
        f"Starting with {n}, the machine must finish with {out}."
    )
    letter_to_op = {}
    for line in lines:
        letter, text = line.split(". ", 1)
        letter_to_op[letter] = next(o for o in ops if o.text == text)
    check(run_machine([letter_to_op[c] for c in key], n) == out, "the key order must give the output")
    for w, _ in wrong:
        check(exact_run([letter_to_op[c] for c in w], n) != out, "a wrong order must not work")
    part = mcq_part(
        "Which order of the steps works?",
        ", ".join(key),
        [(", ".join(w), t) for w, t in wrong],
        1,
        [
            f"Working forwards from {n}: "
            + ", ".join(f"{o.text} → {run_machine(ops[: i + 1], n)}" for i, o in enumerate(ops))
            + "."
        ],
    )
    return single(stem, part)


# -- hard: complete the decision -------------------------------------------------------------------------------------


def trace(rule, start: int, q: int, limit: int = 40):
    """The numbers n takes: divide by `q` when n is a multiple of q, else apply `rule`; stop at 1.
    None if it does not stop within `limit` steps or grows out of range."""
    seq = [start]
    n = start
    for _ in range(limit):
        if n == 1:
            return seq
        n = n // q if n % q == 0 else rule(n)
        if n < 1 or n > 2000:
            return None
        seq.append(n)
    return None


def _rule_text(a: int, b: int) -> str:
    return "replace n by " + join_terms([(a, "n"), (b, "")])


def _hard(rng):
    for _ in range(600):
        q = rng.choice((2, 2, 3))
        a = rng.choice((1, 2, 3, 4, 5)) if q == 3 else rng.choice((3, 3, 5, 1))
        b = rng.choice((1, 2, 3, 4, 5, -1, -2)) if a != 3 else rng.choice((1, 3, -1))
        start = rng.randint(5, 30) if q == 3 else rng.choice((3, 5, 7, 9, 11, 13, 15))
        rule = lambda n, a=a, b=b: a * n + b  # noqa: E731
        seq = trace(rule, start, q)
        if seq is None or not 6 <= len(seq) <= 11:
            continue
        used = [v for v in seq[:-1] if v % q]
        if len(used) < 2 or len(set(used)) < 2:
            continue
        n1 = used[0]
        fits = []
        for k in (1, 2, 3, 4, 5, 6):
            if k == a:
                continue
            kb = rule(n1) - k * n1
            if abs(kb) <= 30 and all(k * v + kb != a * v + b for v in used[1:]):
                fits.append((k, kb))
        if len(fits) >= 3:
            break
    else:
        raise Redraw("no loop with three near-miss rules")
    picks = rng.sample(fits, 3)
    # every wrong rule must give different numbers from the ones shown
    for k, kb in picks:
        alt = trace(lambda n, k=k, kb=kb: k * n + kb, start, q)
        check(alt != seq, "a wrong rule must change the numbers written")
    test = "n is even" if q == 2 else "n is a multiple of 3"
    stem = (
        f"An algorithm starts with n = {start} and repeats these steps: if {test}, replace n by n ÷ {q}; otherwise "
        "apply a rule that is not shown. It stops when n = 1. The numbers n takes, in order, are "
        f"{', '.join(str(v) for v in seq)}."
    )
    part = mcq_part(
        f"Which rule is the missing step for an n that is not {'even' if q == 2 else 'a multiple of 3'}?",
        _rule_text(a, b),
        [(_rule_text(k, kb), E_ONE_STEP) for k, kb in picks],
        1,
        [
            f"The first time the rule is used n = {used[0]} and the next number is {rule(used[0])}.",
            f"Check each rule on every number it is used on: {', '.join(str(v) for v in used)}.",
        ],
    )
    return single(stem, part)


# -- exam ------------------------------------------------------------------------------------------------------------


def _exam(rng):
    ops, n, out = draw(_machine, rng)
    lines, key = _scrambled(rng, ops)
    letters = sorted(key)
    wrong = [(w, t) for w, t in _order_wrongs(key, letters) if w != key]
    letter_to_op = {}
    for line in lines:
        letter, text = line.split(". ", 1)
        letter_to_op[letter] = next(o for o in ops if o.text == text)
    for w, _ in wrong:
        check(exact_run([letter_to_op[c] for c in w], n) != out, "a wrong order must not work")
    stem = (
        f"A number machine has four steps, each used once: {'; '.join(lines)}. Starting with {n}, using the steps in the "
        f"right order, the machine finishes with {out}."
    )
    part_a = words_part(
        "Give the letters of the steps in the order the machine uses them, separated by commas.",
        key,
        2,
        [f"{n} → " + " → ".join(str(run_machine(ops[: i + 1], n)) for i in range(4)) + "."],
        wrong,
    )
    # (b) a second input
    for _ in range(80):
        m2 = rng.randint(2, 20)
        r1 = run_machine(ops, m2)
        r2 = run_machine(ops, r1) if r1 is not None else None
        if r1 is not None and r2 is not None and len({m2, r1, r2}) == 3 and r1 != out and 2 <= r2 <= 400:
            break
    else:
        raise Redraw("no second input whose output can be fed back in")
    part_b = exact_number_part(
        f"The machine is started again with {m2}. Find its output.",
        r1,
        1,
        [f"{m2} → " + " → ".join(str(run_machine(ops[: i + 1], m2)) for i in range(4)) + "."],
        [
            (run_machine(list(reversed(ops)), m2), E_REVERSED),
            (m2, E_INPUT),
            (run_machine(ops[:3], m2), "fr2e-algorithm-last-step-left-out"),
            (run_machine(ops[1:], m2), "fr2e-algorithm-first-step-left-out"),
        ],
        suffix="",
        sensible_range=(0, None),
    )
    expr = "part_b"
    for op in ops:
        expr = op.sympy_text(expr)
    part_c = exact_number_part(
        "The output from the machine in the last part is put straight back into the machine as a new input. "
        "Find the output of this second run.",
        r2,
        1,
        [f"{r1} → " + " → ".join(str(run_machine(ops[: i + 1], r1)) for i in range(4)) + "."],
        [
            (r1, E_ONE_RUN),
            (run_machine(list(reversed(ops)), r1), E_REVERSED),
            (m2, E_INPUT),
            (run_machine(ops[:3], r1), "fr2e-algorithm-last-step-left-out"),
        ],
        suffix="",
        follow_through=expr,
        sensible_range=(0, None),
    )
    # (d) the input that gives a stated output
    target = None
    for _ in range(80):
        cand = rng.randint(20, 200)
        back = cand
        for op in reversed(ops):
            back = op.undo(back) if back is not None else None
        if back is not None and 1 <= back <= 100 and back not in (n, m2):
            target = (cand, back)
            break
    if target is None:
        raise Redraw("no target output with a whole-number input")
    cand, back = target
    check(run_machine(ops, back) == cand, "the input found by undoing must give the output")
    forward = run_machine(ops, cand)
    wrong_d = [(cand, E_INPUT)]
    if forward is not None:
        wrong_d.append((forward, E_FORWARD))
    partial = cand
    for op in reversed(ops[1:]):
        partial = op.undo(partial) if partial is not None else None
    if partial is not None:
        wrong_d.append((partial, "fr2e-algorithm-first-step-left-out-when-undoing"))
    reorder = cand
    for op in ops:  # undoing the steps in their working order, not in reverse
        reorder = op.undo(reorder) if reorder is not None else None
    if reorder is not None:
        wrong_d.append((reorder, E_FORWARD))
    part_d = exact_number_part(
        f"The machine finishes with {cand}. Find the input it was started with.",
        back,
        2,
        [
            f"Undo the steps from the last to the first: {cand} → "
            + " → ".join(str(v) for v in _undo_chain(ops, cand))
            + "."
        ],
        wrong_d,
        suffix="",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


def _undo_chain(ops, cand):
    vals, v = [], cand
    for op in reversed(ops):
        v = op.undo(v)
        vals.append(v)
    return vals


@generator(
    "fr2e_number_algorithm",
    level=6,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2e_number_algorithm(rng, tier: str = "easy"):
    """Choose instructions for a stated goal, order machine steps, or fill in a loop's missing decision."""
    build = {"easy": _easy, "medium": _medium, "hard": _hard}.get(tier)
    check(build is not None, f"unknown tier {tier!r}")
    return draw(build, rng)


@generator("exd_x_fr2e_number_algorithm", level=6, presets={"exam": dict(tier="exam")})
def exd_x_fr2e_number_algorithm(rng, tier: str = "exam"):
    """A four-part chain on one number machine: the order, a second run, the run fed back in, the input for an output."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return draw(_exam, rng)
