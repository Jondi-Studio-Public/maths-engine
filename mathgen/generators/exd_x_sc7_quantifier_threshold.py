"""Exam source for `senior-use-the-quantifiers-for-all-and-there-exists`: a quantified statement with a parameter.

The practice tiers evaluate a quantified statement over a short listed set. Here the domain is every real number, so
the statement cannot be checked value by value: the learner must find the extreme value of a quadratic and turn it
into a condition on an integer parameter k. Three statements, each about g(x) = a x^2 + b x:

  * ``all_positive``     -- "For every real number x, g(x) + k > 0": true when k > -min g, so the smallest integer
    k is -min g + 1;
  * ``all_nonnegative``  -- "For every real number x, g(x) + k >= 0": the smallest integer k is -min g;
  * ``some_negative``    -- "There is a real number x for which g(x) + k < 0": true when k < -min g, so the largest
    integer k is -min g - 1.

(a) the x at which g is least (an integer, since b = -2 a x0 is drawn from x0); (b) the least value of g
(follow-through from (a)); (c) the integer k asked for (follow-through from (b)).
"""

from __future__ import annotations

import random

from ..answers import Distractor
from ..core import attempts, check, generator, join_terms, latex_answer

_KINDS = {
    # kind: (statement wording, which k is asked, offset from -min g, comparator word for the working)
    "all_positive": ("For every real number x, g(x) + k > 0", "smallest", 1),
    "all_nonnegative": ("For every real number x, g(x) + k ≥ 0", "smallest", 0),
    "some_negative": ("There is a real number x for which g(x) + k < 0", "largest", -1),
}


def _part(prompt, answer, marks, distractors, working, follow_through=None):
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(answer),
        answer_text=str(int(answer)),
        answer_latex=latex_answer(str(int(answer))),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _three(candidates, key):
    seen, out = {key}, []
    fillers = [Distractor(str(key + d), None, key + d) for d in (1, -1, 2, -2, 3) if key + d != 0]
    for c in [*candidates, *fillers]:
        if c.value in seen:
            continue
        seen.add(c.value)
        out.append(c)
        if len(out) == 3:
            break
    check(len(out) == 3, "fewer than three distractors")
    return out


@generator("exd_x_sc7_quantifier_threshold", level=12, presets={"exam": dict(tier="exam")})
def exd_x_sc7_quantifier_threshold(rng: random.Random, tier: str = "exam") -> dict:
    """See the module docstring. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    kind = rng.choice(list(_KINDS))
    statement, which, offset = _KINDS[kind]
    for _ in attempts(1000, "a leading coefficient and a vertex with a sensible spread"):
        a = rng.randint(1, 8)
        x0 = rng.randint(-9, 9)
        if x0 != 0:
            break
    b = -2 * a * x0
    g_text = join_terms([(a, "x^2"), (b, "x")])
    least = -a * x0 * x0  # minimum of g
    key_c = -least + offset
    check(a * x0 * x0 + b * x0 == least, "vertex arithmetic")
    stem = f"Let g(x) = {g_text}, and let k be an integer. Consider the statement S: “{statement}.”"
    a_cands = [
        Distractor(str(-x0), "sign-error-vertex-formula", -x0),
        Distractor(str(2 * x0), "forgot-the-two-in-the-vertex-formula", 2 * x0),
        Distractor(str(a * x0), "ignored-the-leading-coefficient", a * x0),
        Distractor(str(x0 + 1), None, x0 + 1),
    ]
    b_cands = [
        Distractor(str(-least), "sign-error-minimum", -least),
        Distractor(str(3 * a * x0 * x0), "substitution-sign-error", 3 * a * x0 * x0),
        Distractor(str(-x0 * x0), "ignored-the-leading-coefficient", -x0 * x0),
        Distractor(str(-a * x0), "forgot-to-square", -a * x0),
    ]
    boundary = -least  # g(x) + k touches zero here, so S fails for a strict inequality
    if offset == 0:
        c_cands = [
            Distractor(str(boundary + 1), "treats-inclusive-bound-as-strict", boundary + 1),
            Distractor(str(boundary - 1), "reverses-the-inequality-direction", boundary - 1),
        ]
    else:
        c_cands = [
            Distractor(str(boundary), "includes-the-boundary-value", boundary),
            Distractor(str(boundary - offset), "reverses-the-inequality-direction", boundary - offset),
        ]
    c_cands += [
        Distractor(str(least + offset), "sign-error-minimum", least + offset),
        Distractor(str(x0 * x0 + offset), "ignored-the-leading-coefficient", x0 * x0 + offset),
    ]
    ft_c = {"all_positive": "1 - part_b", "all_nonnegative": "-part_b", "some_negative": "-part_b - 1"}[kind]
    parts = [
        _part(
            "Find the value of x at which g(x) is least.",
            x0,
            1,
            _three(a_cands, x0),
            [f"g(x) = {'' if a == 1 else a}({'x' if x0 == 0 else f'x − {x0}' if x0 > 0 else f'x + {-x0}'})^2 − {a * x0 * x0}, so the least value occurs at x = {x0}."],
        ),
        _part(
            "Hence find the least value of g(x).",
            least,
            2,
            _three(b_cands, least),
            [f"g({x0}) = {least}."],
            follow_through=f"{a}*part_a**2 + ({b})*part_a",
        ),
        _part(
            f"Find the {which} integer k for which S is true.",
            key_c,
            2,
            _three(c_cands, key_c),
            [
                f"The least value of g(x) + k is {least} + k.",
                {
                    "all_positive": f"S needs {least} + k > 0, so k > {-least}; the smallest integer is {key_c}.",
                    "all_nonnegative": f"S needs {least} + k ≥ 0, so k ≥ {-least}; the smallest integer is {key_c}.",
                    "some_negative": f"S needs {least} + k < 0, so k < {-least}; the largest integer is {key_c}.",
                }[kind],
            ],
            follow_through=ft_c,
        ),
    ]
    return dict(text=stem, parts=parts)
