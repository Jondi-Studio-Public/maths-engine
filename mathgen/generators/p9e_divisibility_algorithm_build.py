"""Generates `p9e_factor_multiple_divisibility_algorithm` questions: trace a stated finite algorithm about
factors, multiples and divisibility and report its numeric output.

Registers exactly one topic, ``p9e_factor_multiple_divisibility_algorithm``, with
three difficulty-tier templates that escalate by CONTROL-STRUCTURE complexity, not
range size. Every question traces a stated finite algorithm and reports its final
NUMERIC output (a count or a step count) -- never a yes/no divisibility test (that
is the shipping ``factors_multiples_divisibility`` topic in
``p3a_secondary_number_factors_build.py``, deliberately not reused here) and never
a GCD/LCM/Euclidean-reduction shape (that is ``euclidean_algorithm_gcd`` in
``university_foundations.py``, also deliberately not reused).

- ``easy`` (``single_range``): "for each whole number from 1 to R inclusive, test
  whether it is divisible by d. Count how many pass." Answer = ``R // d``.
- ``medium`` (``compound_range``): the same range shape, but the per-number test is
  a stated AND/OR of two divisibility tests over a COPRIME pair ``d1, d2`` (drawn
  coprime by construction -- gcd checked directly -- so an AND-count is exactly the
  count of multiples of ``d1*d2``; a non-coprime pair would need a true LCM count
  instead, which this generator never draws). Answer = the compound count.
- ``hard`` (``iterated_state``): "Start with N. While the current number is even,
  halve it and count a step. Stop when the number is odd. How many steps?" -- the
  2-adic valuation of N. ``N = odd_part * 2**k``, ``k`` in 1..5, ``odd_part`` odd
  and > 1 (never a pure power of 2). Answer = the step count ``k``, never the final
  odd value.

Every constant (the range, the divisor(s), the starting number) is a literal
number in the rendered ``question.text``, so the independent oracle
 can re-execute each
algorithm from the text alone, with no knowledge of this module.

Every builder below runs the SAME literal re-execution (``range(1, R+1)`` with
``%``, or an explicit while-loop dividing by 2) as an independent second route
against its own closed-form/direct construction, via ``check_equivalent`` -- the
in-generator half of the two-checks discipline. ``check_distinct`` (or an
equivalent direct-value re-draw guard) confirms every misconception distractor is
a genuinely different integer before the question is returned, every seed,
unconditionally.
"""

from __future__ import annotations

import math

from ..core import check, check_distinct, check_equivalent, generator, latex_answer, pick_distractors

_MAX_ATTEMPTS = 50


def _count_noun(n: int, singular: str) -> str:
    """``singular`` pluralised for a count ``n`` -- avoids a "1 multiples"/
    "1 numbers" plural-after-one defect in the worked-solution text."""
    return singular if n == 1 else f"{singular}s"


def _is_are(n: int) -> str:
    """Verb agreement matching ``_count_noun`` for the same count."""
    return "is" if n == 1 else "are"


def _verify_range_count(R: int, d: int, expected: int) -> None:
    """Independent route for the easy tier: literally re-execute the stated
    test over the range (``range(1, R+1)`` with ``%``), never the closed-form
    ``R // d`` shortcut construction used to build ``expected``."""
    loop_count = sum(1 for n in range(1, R + 1) if n % d == 0)
    check_equivalent(loop_count, expected, f"loop count {loop_count} != {expected} for d={d} R={R}")


def _verify_compound_count(R: int, d1: int, d2: int, gate: str, expected: int) -> None:
    """Independent route for the medium tier: literally re-execute the
    compound AND/OR test over the range, never a closed-form lcm/gcd count."""
    if gate == "AND":
        loop_count = sum(1 for n in range(1, R + 1) if n % d1 == 0 and n % d2 == 0)
    else:
        loop_count = sum(1 for n in range(1, R + 1) if n % d1 == 0 or n % d2 == 0)
    check_equivalent(
        loop_count, expected, f"loop count {loop_count} != {expected} for d1={d1} d2={d2} gate={gate} R={R}"
    )


def _verify_halving_steps(N: int, expected_steps: int) -> tuple[int, list[int]]:
    """Independent route for the hard tier: literally run the halving
    while-loop the question describes, never a bit-trick 2-adic valuation
    shortcut. Returns ``(final_odd_value, trace)`` for the caller to reuse in
    the worked-solution text."""
    current = N
    steps = 0
    trace = []
    while current % 2 == 0:
        current //= 2
        steps += 1
        trace.append(current)
    check_equivalent(steps, expected_steps, f"traced {steps} steps, expected {expected_steps} for N={N}")
    return current, trace


# -- easy: single test over a range -------------------------------------------


def _build_single_range(rng, d_range, r_range):
    d_lo, d_hi = d_range
    r_lo, r_hi = r_range
    d = R = correct = include_zero = complement = None
    for _ in range(_MAX_ATTEMPTS):
        d = rng.randint(d_lo, d_hi)
        R = rng.randint(r_lo, r_hi)
        correct = R // d
        include_zero = correct + 1  # misconception: counts 0 as if it were in range
        complement = R - correct  # misconception: counts the numbers that FAIL
        if include_zero == correct or complement == correct:
            continue
        break
    else:
        check(False, f"could not draw a coherent easy (d, R) pair after {_MAX_ATTEMPTS} attempts")

    _verify_range_count(R, d, correct)
    check_distinct(
        correct, include_zero, complement, msg=f"d={d} R={R}: an easy-tier distractor collides with correct={correct}"
    )

    text = (
        f"Follow this rule: for each whole number from 1 to {R} inclusive, test "
        f"whether it is divisible by {d}. Count how many numbers in the range "
        f"pass the test."
    )
    shown = min(correct, 5)
    multiples_preview = ", ".join(str(d * i) for i in range(1, shown + 1))
    tail = ", ..." if correct > shown else ""
    working = [
        f"The multiples of {d} from 1 to {R} are {multiples_preview}{tail} -- "
        f"{correct} {_count_noun(correct, 'number')} in total."
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=correct,
        answer_text=str(correct),
        answer_latex=latex_answer(str(correct)),
        working=working,
        distractors=pick_distractors(str(correct), [str(include_zero), str(complement)], n=2),
    )


# -- medium: compound AND/OR test over a range --------------------------------


# Share of AND questions left on the short range, where no number passes (answer 0).
_NONE_CASE_SHARE = 0.12


def _build_compound_range(rng, d_pool, r_range):
    r_lo, r_hi = r_range
    d1 = d2 = R = gate = correct = other_gate_count = single_condition = None
    and_count = or_count = d1_count = d2_count = None
    for _ in range(_MAX_ATTEMPTS):
        d1 = rng.choice(d_pool)
        d2 = rng.choice(d_pool)
        if d1 == d2 or math.gcd(d1, d2) != 1:
            continue
        R = rng.randint(r_lo, r_hi)
        gate = rng.choice(["AND", "OR"])
        if gate == "AND" and rng.random() >= _NONE_CASE_SHARE:
            # Run the range past a couple of multiples of lcm so the count is 1 or more (capped
            # at 300); the occasional short range keeps an honest "none pass" case.
            lcm = d1 * d2
            R = rng.randint(2 * lcm, min(4 * lcm, 300))
        d1_count = R // d1
        d2_count = R // d2
        and_count = R // (d1 * d2)  # valid because d1, d2 are coprime by the guard above
        or_count = d1_count + d2_count - and_count
        correct = and_count if gate == "AND" else or_count
        other_gate_count = or_count if gate == "AND" else and_count
        single_condition = d1_count
        if other_gate_count == correct or single_condition == correct:
            continue
        if gate == "AND" and and_count == 0:
            continue  # a key of 0 is a constant guess that wins whenever the range ends below the lcm
        break
    else:
        check(False, f"could not draw a coherent medium (d1, d2, R, gate) combination after {_MAX_ATTEMPTS} attempts")

    _verify_compound_count(R, d1, d2, gate, correct)
    check_distinct(
        correct,
        other_gate_count,
        single_condition,
        msg=f"d1={d1} d2={d2} R={R} gate={gate}: a medium-tier distractor collides with correct={correct}",
    )

    text = (
        f"Follow this rule: for each whole number from 1 to {R} inclusive, test "
        f"whether it is divisible by {d1} {gate} divisible by {d2}. Count how "
        f"many numbers in the range pass the test."
    )
    if gate == "AND":
        working = [
            f"{d1} and {d2} share no common factor, so divisible by both means "
            f"divisible by {d1 * d2}. There {_is_are(correct)} {correct} "
            f"{_count_noun(correct, 'multiple')} of {d1 * d2} from 1 to {R}."
        ]
    else:
        working = [
            f"{d1_count} {_count_noun(d1_count, 'number')} {_is_are(d1_count)} divisible "
            f"by {d1}, {d2_count} {_is_are(d2_count)} divisible by {d2}, and {and_count} "
            f"{_is_are(and_count)} divisible by both -- so "
            f"{d1_count} + {d2_count} - {and_count} = {correct} "
            f"{_count_noun(correct, 'number')} pass the test."
        ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=correct,
        answer_text=str(correct),
        answer_latex=latex_answer(str(correct)),
        working=working,
        distractors=pick_distractors(str(correct), [str(other_gate_count), str(single_condition)], n=2),
    )


# -- hard: iterated state, not a range ----------------------------------------


def _build_iterated_state(rng, odd_pool, k_range):
    k_lo, k_hi = k_range
    k = odd_part = N = None
    for _ in range(_MAX_ATTEMPTS):
        k = rng.randint(k_lo, k_hi)
        odd_part = rng.choice(odd_pool)
        if odd_part == k:  # would make the "reports final odd value" distractor collide
            continue
        N = odd_part * (2**k)
        break
    else:
        check(False, f"could not draw a coherent hard (odd_part, k) pair after {_MAX_ATTEMPTS} attempts")

    steps = k
    current, trace = _verify_halving_steps(N, steps)
    check_equivalent(current, odd_part, f"traced final value {current}, expected odd_part {odd_part} for N={N}")

    off_by_one = steps - 1 if steps > 1 else steps + 1
    check_distinct(steps, off_by_one, odd_part, msg=f"N={N}: a hard-tier distractor collides with steps={steps}")

    text = (
        f"Start with {N}. While the current number is even, divide it by 2 and "
        f"count one step. Stop as soon as the number is odd. How many steps does "
        f"this algorithm take, starting from {N}?"
    )
    lines = []
    prev = N
    for value in trace:
        lines.append(f"{prev} / 2 = {value}")
        prev = value
    working = lines + [
        f"After {steps} {_count_noun(steps, 'step')} the number is odd ({odd_part}), so the answer is {steps}."
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=steps,
        answer_text=str(steps),
        answer_latex=latex_answer(str(steps)),
        working=working,
        distractors=pick_distractors(str(steps), [str(off_by_one), str(odd_part)], n=2),
    )


@generator(
    "p9e_factor_multiple_divisibility_algorithm",
    level=5,
    presets={
        "easy": dict(mode="single_range", d_range=(2, 10), r_range=(20, 40)),
        "medium": dict(mode="compound_range", d_pool=(2, 3, 4, 5, 6, 7, 8, 9, 10), r_range=(20, 40)),
        "hard": dict(mode="iterated_state", odd_pool=(3, 5, 7, 9, 11, 13), k_range=(1, 5)),
    },
)
def p9e_factor_multiple_divisibility_algorithm(
    rng,
    mode: str = "single_range",
    d_range=(2, 10),
    r_range=(20, 40),
    d_pool=(2, 3, 4, 5, 6, 7, 8, 9, 10),
    odd_pool=(3, 5, 7, 9, 11, 13),
    k_range=(1, 5),
):
    """Trace a stated finite algorithm over factors/multiples/divisibility and
    report its final NUMERIC output (a count, never a yes/no verdict). The
    defaults (``mode="single_range"``) mirror the ``easy`` preset so a bare
    ``make(TOPIC, seed=seed)`` call with no difficulty is still a
    coherent question."""
    if mode == "single_range":
        return _build_single_range(rng, d_range, r_range)
    if mode == "compound_range":
        return _build_compound_range(rng, d_pool, r_range)
    if mode == "iterated_state":
        return _build_iterated_state(rng, odd_pool, k_range)
    raise ValueError(f"unknown mode {mode!r}")
