"""Exam source of `senior-check-results-of-calculations-for-accuracy`: locate the slip, repair it, carry it forward.

The practice tiers show a three-step worked calculation and ask which step is wrong. Here a four-step calculation
is shown, each step starting from the stated result of the one before, with exactly one local slip in step 1, 2 or 3
(the last step is never the slip, so the repair has to be carried through at least one more step):

    (a) which step contains the error                          (1 mark)
    (b) the correct result of that step                        (2 marks)
    (c) the correct final result of the calculation            (2 marks, follow-through from (b))

Every step after the slip is computed correctly from the stated, wrong, value, so the only error is the slip. The
slip is a dropped or extra carry (an addition or subtraction off by 10 or 100) or a times-table slip (a product off
by one multiple of its smaller factor, or ten of them).
"""

from __future__ import annotations

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    generator,
    latex_answer,
    pick_distractors,
)
from ._exd_distinct_parts import distinct_part_answers

_OPS = {"+": lambda p, q: p + q, "-": lambda p, q: p - q, "×": lambda p, q: p * q}
_STORIES = [
    "A student works out a calculation in four steps.",
    "A tuckshop volunteer works out a total in four steps.",
    "A student checks a four-step calculation from a worksheet.",
    "The working for a four-step calculation is shown below.",
]


def _draw_steps(rng):
    """Four (left, op, right, result) steps; each left operand is the previous result."""
    first = rng.randint(12, 95)
    ops = [rng.choice(["+", "-", "×", "×"]) for _ in range(4)]
    value = first
    steps = []
    for op in ops:
        if op == "×":
            right = rng.randint(3, 9)
        else:
            right = rng.randint(14, 160)
        result = _OPS[op](value, right)
        if result < 10 or result > 3000:
            return None
        steps.append([value, op, right, result])
        value = result
    return steps


def _slip(rng, op, left, right, correct):
    """A stated (wrong) result for one step: a carry slip for + and -, a times-table slip for ×."""
    if op == "×":
        unit = min(left, right)
        delta = rng.choice([1, -1]) * unit * rng.choice([1, 10])
    else:
        delta = rng.choice([1, -1]) * rng.choice([10, 10, 100])
    wrong = correct + delta
    return wrong if wrong > 0 and wrong != correct else None


def _chain_from(steps, start_index, value):
    """Final result when step `start_index` is replaced by `value` and the later steps are redone."""
    for left, op, right, _stated in steps[start_index + 1 :]:
        value = _OPS[op](value, right)
    return value


def _line(step, number):
    left, op, right, result = step
    return f"Step {number}: {left} {op} {right} = {result}"


@generator("exd_x_sc1_check_calculation", level=11, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_sc1_check_calculation(rng, tier="exam"):
    for _ in attempts(1000, "a four-step calculation with one slip"):
        steps = _draw_steps(rng)
        if steps is None:
            continue
        k = rng.randint(0, 2)  # index of the slipped step
        left, op, right, correct = steps[k]
        wrong = _slip(rng, op, left, right, correct)
        if wrong is None:
            continue
        # Re-run the later steps from the stated wrong value, as the student did.
        shown = [list(s) for s in steps]
        shown[k][3] = wrong
        value = wrong
        for j in range(k + 1, 4):
            shown[j][0] = value
            shown[j][3] = _OPS[shown[j][1]](value, shown[j][2])
            value = shown[j][3]
        if any(s[3] < 10 or s[3] > 6000 for s in shown):
            continue
        final_correct = _chain_from(shown, k, correct)
        final_stated = shown[3][3]
        if final_correct == final_stated or not (10 <= final_correct <= 6000):
            continue
        # The later steps must change the repaired value: if they cancel out, (c) would be (b) typed again.
        if final_correct == correct:
            continue
        stem_numbers = {n for s in shown for n in (s[0], s[2], s[3])}
        if correct in stem_numbers or final_correct in stem_numbers:
            continue
        break

    # Independent re-check: exactly one stated step disagrees with its own stated inputs.
    bad = [i for i, (lft, o, rgt, res) in enumerate(shown) if _OPS[o](lft, rgt) != res]
    check(bad == [k], "the worked calculation must have exactly one slipped step")
    redo = steps[0][0]
    for _l, o, rgt, _r in steps:
        redo = _OPS[o](redo, rgt)
    check(redo == final_correct, "the repaired calculation does not reach the final result")

    step_no = k + 1
    # (a) which step: the first step by default, the step after the slip (recomputed from the value it "should" have
    # started from), and the last step (judged from the final result alone).
    a_cands = [
        Distractor("1", "blamed-first-step", 1),
        Distractor(str(step_no + 1), "blamed-step-after-slip", step_no + 1),
        Distractor("4", "blamed-last-step", 4),
        Distractor("3", None, 3),
    ]
    a_dist = pick_distractors(str(step_no), a_cands, 3)
    # (b) the correct step result: the stated value, the other way round for the slip, or the wrong operation.
    other = {"+": left - right, "-": left + right, "×": left + right}[op]
    swapped = str(other)
    digits = str(correct)
    transposed = int(digits[:-2] + digits[-1] + digits[-2]) if len(digits) > 1 else correct
    b_cands = [
        Distractor(str(wrong), "accepted-the-stated-result", wrong),
        Distractor(swapped, "swapped-the-operation", other),
        Distractor(str(transposed), "transposed-digits", transposed),
    ]
    b_dist = pick_distractors(str(correct), b_cands, 3)
    # (c) the final result: the stated final, or the repaired step taken as the final answer.
    c_cands = [
        Distractor(str(final_stated), "accepted-the-stated-result", final_stated),
        Distractor(str(correct), "stopped-after-the-repair", correct),
    ]
    later = [s for s in shown[k + 1 :]]
    # The later steps redone with their operations applied in the wrong order is not a natural slip; instead offer the
    # final result when the repaired value is wrongly fed through only the next step.
    one_step = _OPS[later[0][1]](correct, later[0][2])
    c_cands.append(Distractor(str(one_step), "carried-the-repair-one-step-only", one_step))
    c_dist = pick_distractors(str(final_correct), c_cands, 3)
    for dists, key in ((a_dist, step_no), (b_dist, correct), (c_dist, final_correct)):
        check(all(int(str(d)) != key for d in dists), "a distractor equals the key")

    # Follow-through for (c): redo the steps after the slip from whatever (b) the learner gave.
    expr = "part_b"
    for _l, o, rgt, _r in shown[k + 1 :]:
        expr = f"({expr} {'*' if o == '×' else o} {rgt})"

    lines = "; ".join(_line(s, i + 1) for i, s in enumerate(shown))
    stem = f"{rng.choice(_STORIES)} Exactly one step contains an arithmetic error, and each step starts from the result stated in the step before. {lines}."
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Which step contains the error? Give the step number.",
                answer_kind="number",
                answer=step_no,
                answer_text=str(step_no),
                answer_latex=latex_answer(str(step_no)),
                marks=1,
                sensible_range=(0, 5),
                working=[
                    f"Recompute each step from its own stated inputs: Step {step_no} gives {correct}, not {wrong}."
                ],
                distractors=a_dist,
            ),
            dict(
                prompt="Write down the correct result of that step.",
                answer_kind="number",
                answer=correct,
                answer_text=str(correct),
                answer_latex=latex_answer(str(correct)),
                marks=2,
                sensible_range=(0, None),
                working=[f"{left} {op} {right} = {correct}."],
                distractors=b_dist,
            ),
            dict(
                prompt="Hence find the correct final result of the calculation.",
                answer_kind="number",
                answer=final_correct,
                answer_text=str(final_correct),
                answer_latex=latex_answer(str(final_correct)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Redo the steps after Step {step_no} from {correct}: "
                    + ", ".join(f"{s[1]} {s[2]}" for s in shown[k + 1 :])
                    + f" gives {final_correct}."
                ],
                distractors=c_dist,
                follow_through=expr,
            ),
        ],
    )
