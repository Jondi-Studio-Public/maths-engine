"""Calculate total change by integrating rate of change.

- ``calculate_total_change_by_integrating_instantaneous_or_marginal``
  (``senior-calculate-total-change-by-integrating-instantaneous-or-marginal``,
  ACMMM133, domain integration, answer: expression) -- given bounded, fully
  specified inputs, calculate total change by integrating instantaneous or marginal
  rate of change. Return the resulting value, expression, relation, or classification.
"""

from __future__ import annotations

from sympy import symbols, diff, integrate, simplify

from ..core import check, check_distinct, generator, pick_distractors, symbolically_equal


# Display helpers (^/* convention): convert ** to ^ for typed, ** to ^ and * to · for prose
def _typed(expr) -> str:
    """Convert expression to learner-facing typed form: ** -> ^."""
    return str(expr).replace("**", "^")


def _prose(expr) -> str:
    """Convert expression to prose form: ** -> ^, * to ·."""
    typed_str = str(expr).replace("**", "^")
    return typed_str.replace("*", "·")


# Module-scope symbol for time
t = symbols("t")


@generator(
    "calculate_total_change_by_integrating_instantaneous_or_marginal",
    level=17,
    presets={
        "easy": dict(difficulty="easy"),
        "medium": dict(difficulty="medium"),
        "hard": dict(difficulty="hard"),
    },
)
def calculate_total_change_by_integrating_instantaneous_or_marginal(rng, difficulty="easy"):
    """
    Calculate total change by integrating a rate of change function over a bounded interval.

    Easy: integrate c*t (linear rate) from 0 to b_
    Medium: integrate c*t + d (two-term rate) from a_ to b_ (both nonzero)
    Hard: same as medium but with an initial amount in the prompt as a trap
    """

    if difficulty == "easy":
        # Linear rate: r(t) = c*t, integrated from 0 to b_
        c = rng.randint(2, 6)
        b_ = rng.randint(2, 7)

        r = c * t
        F = c * t**2 / 2
        ans = F.subs(t, b_) - F.subs(t, 0)

        # Independent checks
        check(symbolically_equal(diff(F, t), r), msg="F' must equal r")
        check(symbolically_equal(ans, integrate(r, (t, 0, b_))), msg="ans must equal definite integral")

        # Misconceptions
        sign_error = -ans  # swapped bounds
        wrong_power_rule = c * b_**2  # forgot to divide by 2
        rate_times_duration = c * b_**2  # endpoint rate times duration (equals 2*ans)

        check_distinct(
            ans, sign_error, wrong_power_rule, rate_times_duration, msg="easy tier misconceptions must be distinct"
        )

        candidates = [
            _typed(sign_error),
            _typed(wrong_power_rule),
            _typed(rate_times_duration),
        ]
        distractors = pick_distractors(_typed(ans), candidates)

        return {
            "text": f"Water flows into a tank. The flow rate is r(t) = {_prose(r)} (litres per minute), where t is the number of minutes after the tap is opened. Calculate the total change in the volume of water in the tank during the first {b_} minutes.",
            "latex": f"Water flows into a tank. The flow rate is r(t) = {_prose(r)} (litres per minute), where t is the number of minutes after the tap is opened. Calculate the total change in the volume of water in the tank during the first {b_} minutes.",
            "answer": str(ans),
            "answer_text": _typed(ans),
            "answer_latex": _typed(ans),
            "distractors": distractors,
            "working": [
                f"Total change = ∫ from t = 0 to t = {b_} of {_prose(r)} dt",
                f"= [{_prose(F)}] from 0 to {b_}",
                f"= {_typed(ans)}",
            ],
        }

    elif difficulty == "medium":
        # Two-term linear rate: r(t) = c*t + d, integrated from a_ to b_ (both nonzero)
        c = rng.randint(2, 5)
        d = rng.randint(1, 6)
        a_ = rng.randint(1, 4)
        gap = rng.randint(1, 5)
        b_ = a_ + gap

        r = c * t + d
        F = c * t**2 / 2 + d * t
        ans = F.subs(t, b_) - F.subs(t, a_)

        # Independent checks
        check(symbolically_equal(diff(F, t), r), msg="F' must equal r")
        check(symbolically_equal(ans, integrate(r, (t, a_, b_))), msg="ans must equal definite integral")

        # Misconceptions
        sign_error = -ans
        wrong_power_rule = (c * b_**2 + d * b_**2) - (c * a_**2 + d * a_**2)  # incorrect power rule on both terms
        only_upper_bound = F.subs(t, b_) - F.subs(t, 0)  # forgot lower bound

        check_distinct(
            ans, sign_error, wrong_power_rule, only_upper_bound, msg="medium tier misconceptions must be distinct"
        )

        candidates = [
            _typed(sign_error),
            _typed(wrong_power_rule),
            _typed(only_upper_bound),
        ]
        distractors = pick_distractors(_typed(ans), candidates)

        return {
            "text": f"A company's marginal cost is given by r(t) = {_prose(r)} (dollars per unit), where t is the number of units already produced. Calculate the total change in cost as production increases from {a_} to {b_} units.",
            "latex": f"A company's marginal cost is given by r(t) = {_prose(r)} (dollars per unit), where t is the number of units already produced. Calculate the total change in cost as production increases from {a_} to {b_} units.",
            "answer": str(ans),
            "answer_text": _typed(ans),
            "answer_latex": _typed(ans),
            "distractors": distractors,
            "working": [
                f"Total change = ∫ from t = {a_} to t = {b_} of {_prose(r)} dt",
                f"= [{_prose(F)}] from {a_} to {b_}",
                f"= {_typed(ans)}",
            ],
        }

    else:  # hard
        # Same rate function shape as medium, but with an initial amount stated as a context trap
        c = rng.randint(2, 5)
        d = rng.randint(1, 6)
        a_ = rng.randint(1, 4)
        gap = rng.randint(1, 5)
        b_ = a_ + gap
        I0 = rng.randint(10, 80)  # initial amount (distractor trap)

        r = c * t + d
        F = c * t**2 / 2 + d * t
        ans = F.subs(t, b_) - F.subs(t, a_)

        # Independent checks
        check(symbolically_equal(diff(F, t), r), msg="F' must equal r")
        check(symbolically_equal(ans, integrate(r, (t, a_, b_))), msg="ans must equal definite integral")

        # Misconceptions
        added_initial_amount = I0 + ans  # added initial amount (answers the "final amount" instead)
        sign_error = -ans
        only_upper_bound = F.subs(t, b_) - F.subs(t, 0)  # forgot lower bound

        check_distinct(
            ans, added_initial_amount, sign_error, only_upper_bound, msg="hard tier misconceptions must be distinct"
        )

        candidates = [
            _typed(added_initial_amount),
            _typed(sign_error),
            _typed(only_upper_bound),
        ]
        distractors = pick_distractors(_typed(ans), candidates)

        return {
            "text": f"A tank already contains {I0} litres when it starts being monitored. Water then flows in with rate function r(t) = {_prose(r)} (litres per minute), where t is the number of minutes after monitoring starts. Calculate the TOTAL CHANGE in the volume of water in the tank as t increases from {a_} to {b_} minutes (not the final volume).",
            "latex": f"A tank already contains {I0} litres when it starts being monitored. Water then flows in with rate function r(t) = {_prose(r)} (litres per minute), where t is the number of minutes after monitoring starts. Calculate the TOTAL CHANGE in the volume of water in the tank as t increases from {a_} to {b_} minutes (not the final volume).",
            "answer": str(ans),
            "answer_text": _typed(ans),
            "answer_latex": _typed(ans),
            "distractors": distractors,
            "working": [
                f"Total change = ∫ from t = {a_} to t = {b_} of {_prose(r)} dt",
                f"= [{_prose(F)}] from {a_} to {b_}",
                f"= {_typed(ans)}",
            ],
        }
