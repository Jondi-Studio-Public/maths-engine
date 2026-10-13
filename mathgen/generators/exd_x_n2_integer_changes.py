"""Exam tier for "operate with integers".

The practice tiers add, subtract, multiply or divide two integers. Here a quantity that starts at a possibly
negative value falls by the same amount several times and then rises by the same amount several times, so the
learner multiplies a repeated change, applies it to a signed start, and finally compares the end with the start.
Three settings (a temperature, a depth, a bank balance) share the same arithmetic. Three parts:

(a) the value after the falls (2 marks);
(b) the value after the rises (2 marks, follow-through from (a));
(c) the overall change from the start to the end (2 marks, follow-through from (b)).

Every value is a whole number. The values after each stage are never zero and the overall change is never zero.
"""

from __future__ import annotations

from mathgen.answers import Distractor
from mathgen.core import check, draw_until, generator, latex_answer, phrase, pick_distractors

# Each scene carries the ranges that keep its numbers realistic: start, fall per step and steps, rise per step and
# steps, and the lowest and highest value the quantity may reach.
_SCENES = (
    dict(
        start=(-3, 8),
        fall=(1, 3),
        falls=(3, 8),
        rise=(1, 3),
        rises=(2, 6),
        low=-12,
        high=12,
        stem=(
            "At midnight the temperature at a Snowy Mountains ski resort is {s} °C. It falls by {d} °C every hour "
            "for {n} hours, and then rises by {r} °C every hour for {k} hours.",
            "The temperature on a mountain in Tasmania at 6 pm is {s} °C. Over the next {n} hours it falls by {d} "
            "°C each hour. It then rises by {r} °C each hour for a further {k} hours.",
        ),
        a="What is the temperature, in °C, after the {n} hours of falling?",
        b="What is the temperature, in °C, at the end of the {k} hours of rising?",
        c="By how many degrees Celsius has the temperature changed from its starting value to its final value? "
        "Use a negative number for a fall.",
    ),
    dict(
        start=(-12, 6),
        fall=(2, 6),
        falls=(3, 6),
        rise=(2, 5),
        rises=(2, 6),
        low=-30,
        high=6,
        stem=(
            "A diver's height relative to sea level is {s} m. The diver descends {d} m every minute for {n} "
            "minutes, and then ascends {r} m every minute for {k} minutes.",
            "A scuba diver off the Queensland coast is at {s} m relative to sea level. She sinks {d} m each minute "
            "for {n} minutes, then swims up {r} m each minute for {k} minutes.",
        ),
        a="What is the height relative to sea level, in metres, after the {n} minutes of descending?",
        b="What is the height relative to sea level, in metres, at the end of the {k} minutes of climbing?",
        c="By how many metres has the height changed from the start to the end? Use a negative number for a drop.",
    ),
    dict(
        start=(-120, 150),
        fall=(5, 40),
        falls=(3, 8),
        rise=(10, 60),
        rises=(2, 7),
        low=-300,
        high=400,
        stem=(
            "A bank account has a balance of ${s}, where a negative balance means the account is overdrawn. Fees of "
            "${d} are taken at the end of each week for {n} weeks, and then ${r} is deposited at the end of each "
            "week for {k} weeks.",
            "The balance of an account is ${s} (negative means overdrawn). For {n} weeks a ${d} fee is charged "
            "weekly, and after that ${r} is deposited weekly for {k} weeks.",
        ),
        a="What is the balance, in dollars, after the {n} weeks of fees?",
        b="What is the balance, in dollars, after the {k} weeks of deposits?",
        c="By how many dollars has the balance changed from the start to the end? Use a negative number for a "
        "decrease.",
    ),
)


def _draw(rng, scene):
    s = rng.choice([v for v in range(scene["start"][0], scene["start"][1] + 1) if v])
    return (
        s,
        rng.randint(*scene["fall"]),
        rng.randint(*scene["falls"]),
        rng.randint(*scene["rise"]),
        rng.randint(*scene["rises"]),
    )


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _acceptable(draw, scene) -> bool:
    s, d, n, r, k = draw
    after_fall = s - d * n
    end = after_fall + r * k
    in_range = scene["low"] <= after_fall <= scene["high"] and scene["low"] <= end <= scene["high"]
    crosses_zero = _sign(s) != _sign(after_fall) or _sign(after_fall) != _sign(end)
    return in_range and crosses_zero and after_fall != 0 and end != 0 and end != s and d != r and n != k


@generator("exd_x_n2_integer_changes", level=10, presets={"exam": dict(tier="exam")})
def exd_x_n2_integer_changes(rng, tier="exam"):
    """Repeated falls and rises from a signed start; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    scene = rng.choice(_SCENES)
    s, d, n, r, k = draw_until(
        rng,
        lambda r_: _draw(r_, scene),
        lambda draw: _acceptable(draw, scene),
        what="a signed start with non-zero stages in a realistic range",
    )
    after_fall = s - d * n
    end = after_fall + r * k
    change = end - s
    check(change == r * k - d * n and end == s + change, "overall change disagrees with the stages")
    # In a bank scene, a negative balance is written -$5? Keep the sign in front of the dollar sign off: state "$-5".
    stem = phrase(rng, scene["stem"], s=s, d=d, n=n, r=r, k=k)
    if "${s}" in "".join(scene["stem"]):
        stem = stem.replace(f"${s}", f"-${abs(s)}" if s < 0 else f"${s}")

    def part(prompt, key, marks, work, cands, ft=None):
        shown = [Distractor(str(v), tag, v) if tag else str(v) for v, tag in cands]
        out = dict(
            prompt=prompt,
            answer_kind="number",
            answer=key,
            answer_text=str(key),
            answer_latex=latex_answer(str(key)),
            marks=marks,
            working=work,
            distractors=pick_distractors(str(key), shown, 3),
        )
        if ft:
            out["follow_through"] = ft
        return out

    parts = [
        part(
            scene["a"].format(n=n, k=k),
            after_fall,
            2,
            [f"{d} × {n} = {d * n}; {s} - {d * n} = {after_fall}."],
            [
                (s + d * n, "added-fall-instead-of-subtracting"),
                (s - d, "one-step-for-many"),
                (abs(s) - d * n, "ignored-sign-of-start"),
            ],
        ),
        part(
            scene["b"].format(n=n, k=k),
            end,
            2,
            [f"{r} × {k} = {r * k}; {after_fall} + {r * k} = {end}."],
            [
                (after_fall - r * k, "subtracted-rise"),
                (after_fall + r, "one-step-for-many"),
                (s + r * k, None),
            ],
            ft=f"part_a + {r * k}",
        ),
        part(
            scene["c"].format(n=n, k=k),
            change,
            2,
            [f"Change = final - start = {end} - ({s}) = {change}."],
            [
                (s - end, "change-taken-backwards"),
                (abs(end) - abs(s), "subtracted-sizes-ignoring-signs"),
                (end, None),
                (r * k, None),
                (d * n, None),
            ],
            ft=f"part_b - ({s})",
        ),
    ]
    return dict(text=stem, parts=parts)
