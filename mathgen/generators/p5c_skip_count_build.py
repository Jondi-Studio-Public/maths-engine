"""Generates `skip_count_pattern_sequences` questions: continue a skip-counting sequence by 2, 5 or 10 from three
given terms.

F-10 pattern/sequence archetype pilot.

Skill: ``f10-skip-count-pattern-sequences`` -- given three consecutive terms of an
increasing numeral skip-counting sequence (by 2, 5, or 10), continue the
sequence to the next two or three terms.

**Boundary against the existing prerequisite `skip-count-2-5-10`:**
The existing skill (`mathgen/generators/early_math.py::skip_count`) shows 3
consecutive terms and asks for exactly ONE next term via ``mcq``. This new
skill always asks for **TWO or THREE next terms** (depending on difficulty),
typed as an ``ordered_list``, never collapsing to the one-term/``mcq`` shape.
The independent check never reuses the loop that built the answer list;
instead it recomputes each required value via the closed-form `a + (3 + j) * step`
for `j in range(k)` (direct multiplication, not iterative addition).
"""

from __future__ import annotations

from ..core import check, generator, pick_distractors


def _verify_terms(a: int, step: int, k: int, answer_list: list[int]) -> None:
    """Independent check: recompute each required term via closed-form
    a + (3 + j) * step for j in range(k), never reusing the loop that built
    the answer list. Fires correctly on a corrupted answer (mutate one element)."""
    for j in range(k):
        expected = a + (3 + j) * step
        check(len(answer_list) > j, f"answer_list too short: expected {k} terms, got {len(answer_list)}")
        check(answer_list[j] == expected, f"term {j} is {answer_list[j]}, expected a + (3 + {j}) * {step} = {expected}")


@generator(
    "skip_count_pattern_sequences",
    level=1,
    presets={
        "easy": dict(step_choice="easy"),
        "medium": dict(step_choice="medium"),
        "hard": dict(step_choice="hard"),
    },
)
def skip_count_pattern_sequences(rng, step_choice="easy"):
    """Continue a numeral skip-counting sequence by 2, 5 or 10.

    Difficulty progression:
    - easy: start value `a` is a multiple of `step`, ask for 2 next terms
    - medium: start value `a` is NOT a multiple of `step`, ask for 2 next terms
    - hard: start value `a` is NOT a multiple of `step` AND the extension
      crosses a place-value boundary (multiple of 100), ask for 3 next terms

    All sequences show 3 initial consecutive terms (a, a+step, a+2*step),
    always increasing, always extending past the end.
    """
    step = rng.choice([2, 5, 10])

    if step_choice == "easy":
        # Easy: a is a multiple of step
        # Pick a from friendly starts: 0, 5, 10, 15, ... (for step=5) or 0, 2, 4, ... (for step=2)
        a = rng.randint(0, 10) * step
        k = 2
        crossing_check = False
    elif step_choice == "medium":
        # Medium: a is NOT a multiple of step
        # Pick any a in range [0, 99], filter out multiples of step
        while True:
            a = rng.randint(0, 99)
            if a % step != 0:
                break
        k = 2
        crossing_check = False
    else:  # hard
        # Hard: a is NOT a multiple of step AND the extension crosses a
        # place-value boundary (multiple of 100)
        #
        # Strategy: pick the crossing multiple M, then pick last_shown in
        # [M-3*step, M-1] with last_shown % step != 0 (that's equivalent to
        # a % step != 0 since 2*step ≡ 0 mod step), then compute a.
        #
        # The last shown term is a + 2*step. We want a + 2*step < M <= a + 5*step
        # (so at least one of the three required terms crosses M).
        # Also, a + 2*step % step != 0 ensures a % step != 0.
        #
        # Pick M (multiple of 100). Most of the time M=100, sometimes M=200.
        M = rng.choice([100, 200])

        # last_shown = a + 2*step must be in [M - 3*step, M - 1] and
        # (a + 2*step) % step != 0
        while True:
            last_shown = rng.randint(M - 3 * step, M - 1)
            if last_shown % step != 0:
                # This is equivalent to a % step != 0 since last_shown = a + 2*step
                # and 2*step ≡ 0 mod step
                a = last_shown - 2 * step
                break

        # Verify that crossing actually occurs: last_shown < M <= last_required_term
        last_required_term = a + (3 + 2) * step  # 3 required terms: (3+0), (3+1), (3+2)
        check(a >= 0, f"negative start value a={a}")
        check(last_shown < M, f"last_shown {last_shown} >= M {M}, no crossing")
        check(M <= last_required_term, f"M {M} > last_required_term {last_required_term}, crossing too far")
        check(a % step != 0, f"a % step should be nonzero, but a={a}, step={step}")
        k = 3
        crossing_check = True

    # Build the displayed sequence and answer
    displayed = [a, a + step, a + 2 * step]
    next_terms = [a + (3 + j) * step for j in range(k)]
    _verify_terms(a, step, k, next_terms)

    # Answer
    answer_text = ", ".join(str(term) for term in next_terms)
    display_text = ", ".join(str(term) for term in displayed)

    # Question text
    if k == 2:
        question_text = f"What are the next 2 numbers? {display_text}, __, __"
    else:
        question_text = f"What are the next 3 numbers? {display_text}, __, __, __"

    # Misconception distractors
    distractors_candidates = []

    # 1. wrong-step: uses a different step from {2, 5, 10}
    wrong_steps = [s for s in [2, 5, 10] if s != step]
    wrong_step_value = rng.choice(wrong_steps)
    wrong_step_terms = [a + (3 + j) * wrong_step_value for j in range(k)]
    distractors_candidates.append(", ".join(str(term) for term in wrong_step_terms))

    # 2. stopped-incrementing: repeats the last shown term k times
    stopped_terms = [displayed[-1]] * k
    distractors_candidates.append(", ".join(str(term) for term in stopped_terms))

    # 3. hundred-crossing-drop (hard tier only, exercised every hard seed):
    #    fails to carry the hundreds digit when crossing the boundary.
    #    For hard seeds, we know M is crossed. Compute the "did not cross"
    #    answer by subtracting 100 from each post-crossing term.
    if crossing_check and k == 3:
        M = (a + 5 * step) // 100 * 100  # Recover the crossing multiple
        # Actually, let's compute it more carefully: find the smallest M
        # such that M is a multiple of 100 and M > displayed[-1]
        M = ((displayed[-1] // 100) + 1) * 100
        check(M <= last_required_term, f"hard: crossing multiple M={M} > last_required_term, computation invalid")

        # The "did not cross" answer: continue as if we never crossed
        # This is numerically wrong but plausible
        drop_terms = []
        for j in range(k):
            true_term = a + (3 + j) * step
            if true_term >= M:
                # This term crosses the boundary in the true answer.
                # The misconception answer drops 100.
                drop_terms.append(true_term - 100)
            else:
                drop_terms.append(true_term)
        drop_text = ", ".join(str(term) for term in drop_terms)

        # Verify this distractor is genuinely different from the true answer
        check(drop_text != answer_text, f"hard-tier hundred-crossing-drop distractor equals true answer: {drop_text}")
        distractors_candidates.append(drop_text)

    # Build working/reasoning text
    working = [f"The pattern goes up by {step} each time."]
    if k == 2:
        working.append(
            f"The next 2 numbers are {displayed[-1]} + {step} = {next_terms[0]} and {next_terms[0]} + {step} = {next_terms[1]}."
        )
    else:
        working.append(f"The next 3 numbers are:")
        for i, term in enumerate(next_terms):
            working.append(f"  {i + 1}. {displayed[-1] if i == 0 else next_terms[i - 1]} + {step} = {term}")

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=next_terms,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, distractors_candidates),
    )
