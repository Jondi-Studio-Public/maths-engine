"""w10: "explain why" questions for Year 7-8 algebra, answered in the learner's own words and marked by a rubric.

Each question is an `open_response` item (``mathgen.grading.open_response``): the prompt, a model answer worked out
from the question's numbers, an ordered rubric of criteria with marks, and exemplar answers at different marks.
Nothing here marks an answer. The engine returns it as pending; a teacher, a model or the learner marks it against
the rubric and the result is folded back in.

Eight topics, one per skill, each with easy / medium / hard presets:
``w10_explain_equality``, ``w10_explain_inverse``, ``w10_explain_solution``, ``w10_explain_identity``,
``w10_explain_check``, ``w10_explain_equivalent``, ``w10_explain_variables`` and ``w10_explain_proportional``.

A question's numbers are chosen first and the claim (true or false, one solution or none) is decided from them with
exact arithmetic, so the model answer and the rubric never disagree with the prompt. Every rubric criterion names
something the answer must say or show, with the question's own numbers in it, so a marker can tell it was met.
"""

from __future__ import annotations

from fractions import Fraction

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from ..core import check, generator
from ..grading.open_response import rubric_of


_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def pick(rng, items):
    return items[rng.randrange(len(items))]


def item(text, reference, criteria, partial, weak, *, partial_marks=None):
    """The question dict. `criteria` is a list of (id, description, marks). `partial` is an answer that meets only the
    first criterion; `weak` is a plausible answer that meets none."""
    rows = [{"id": cid, "description": desc, "marks": marks} for cid, desc, marks in criteria]
    total = sum(r["marks"] for r in rows)
    first = rows[0]["marks"] if partial_marks is None else partial_marks
    rubric = {
        "reference": reference,
        "criteria": rows,
        # the plain-language checks a marker applies (the `checks` marking-block convention of skills/README.md)
        "checks": [r["description"] for r in rows],
        "exemplars": [
            {"label": "full marks", "marks": total, "answer": reference},
            {"label": "partly correct", "marks": first, "answer": partial},
            {"label": "common weak answer", "marks": 0, "answer": weak},
        ],
    }
    rubric_of(rubric)
    check("*" not in text and "**" not in text, "star in the prompt")
    return dict(text=text, latex="", answer=rubric, answer_text=reference, answer_kind="open_response")


# ---------------------------------------------------------------------------------------------------------------
# 1. The meaning of equality
# ---------------------------------------------------------------------------------------------------------------
def _pair_sum(rng, total, lo=2, hi=18):
    a = rng.randint(max(lo, total - hi), min(hi, total - lo))
    return a, total - a


def equality_statement(rng):
    total = rng.randint(10, 30)
    a, b = _pair_sum(rng, total)
    for _ in range(50):
        c, d = _pair_sum(rng, total)
        if {c, d} != {a, b}:
            break
    true = rng.randrange(2) == 0
    if not true:
        d = d + pick(rng, (1, 2, 3, 4))
    right = c + d
    left = a + b
    check((left == right) == true, "equality: claim and values disagree")
    word = "true" if true else "false"
    if true:
        ref = (
            f"The equals sign says the two sides have the same value. The left side {a} + {b} is {left} and the right "
            f"side {c} + {d} is also {right}. Both sides have the same value, so the statement is true."
        )
        conclude = f"Concludes that the statement is true because both sides have the same value ({left})."
    else:
        ref = (
            f"The equals sign says the two sides must have the same value. The left side {a} + {b} is {left} but the "
            f"right side {c} + {d} is {right}. The sides have different values, so the statement is false."
        )
        conclude = (
            f"Concludes that the statement is false because the sides have different values ({left} and {right})."
        )
    return item(
        pick(
            rng,
            (
                f"Is the statement {a} + {b} = {c} + {d} true or false? Explain, using what the equals sign tells you.",
                f"A worksheet has the line {a} + {b} = {c} + {d}. Decide whether it is correct, and explain what the equals sign means here.",
                f"Jo writes {a} + {b} = {c} + {d}. Using the meaning of the equals sign, explain whether Jo has written something true.",
            ),
        ),
        ref,
        [
            ("left-side", f"Works out the left side: {a} + {b} = {left}.", 1),
            ("right-side", f"Works out the right side: {c} + {d} = {right}.", 1),
            ("meaning", "Says that the equals sign means both sides have the same value.", 1),
            ("conclusion", conclude, 1),
        ],
        partial=f"{a} + {b} is {left}.",
        weak=f"The equals sign means the answer comes next, so it is {word} if {a} + {b} makes the number after it.",
    )


def equality_misreading(rng):
    a, b = _pair_sum(rng, rng.randint(10, 24))
    total = a + b
    d = pick(rng, (2, 3, 4, 5, 6, 7))
    return item(
        pick(
            rng,
            (
                f"Sam says that {a} + {b} = {total} + {d} is true, because {a} + {b} makes {total}. Say what Sam has "
                f"misunderstood about the equals sign, and say whether the statement is true.",
                f"Sam reads {a} + {b} = {total} + {d} as 'the answer is {total}' and calls it correct. What is wrong with "
                f"the way Sam reads the equals sign? Is the statement true?",
                f"Is Sam right that {a} + {b} = {total} + {d} is a true statement because {a} + {b} is {total}? Explain "
                f"Sam's mistake about the equals sign.",
            ),
        ),
        f"Sam read the equals sign as 'the answer comes next' and looked only at the left side. The equals sign means "
        f"that both sides have the same value. The left side {a} + {b} is {total}, but the right side {total} + {d} is "
        f"{total + d}. These are not the same value, so the statement is false.",
        [
            ("misreading", "Says Sam treated the equals sign as 'the answer is' and looked only at the left side.", 1),
            ("right-side", f"Works out the right side: {total} + {d} = {total + d}.", 1),
            ("conclusion", f"Concludes the statement is false because {total} and {total + d} are different.", 1),
            ("meaning", "Says that the equals sign means both sides have the same value.", 1),
        ],
        partial=f"Sam only looked at {a} + {b}, which does make {total}.",
        weak=f"Sam is right, because {a} + {b} is {total}.",
    )


def equality_missing_number(rng):
    a, b = _pair_sum(rng, rng.randint(12, 26))
    total = a + b
    c = rng.randint(2, total - 3)
    n = total - c
    return item(
        pick(
            rng,
            (
                f"What number goes in the box to make {a} + {b} = □ + {c} true? Explain how the equals sign tells you.",
                f"Find the missing number in {a} + {b} = □ + {c}. Explain how you know, using the meaning of the equals sign.",
                f"The box in {a} + {b} = □ + {c} must hold a number that keeps the statement true. Say which number, and "
                f"explain why the equals sign makes it that number.",
            ),
        ),
        f"The equals sign means both sides must have the same value. The left side {a} + {b} is {total}, so the right "
        f"side □ + {c} must also be {total}. The box must be {total} - {c} = {n}. Check: {n} + {c} = {total}, the same as "
        f"the left side.",
        [
            ("left-side", f"Works out the left side: {a} + {b} = {total}.", 1),
            ("same-value", "Says the right side must have the same value as the left side.", 1),
            ("number", f"Gives the number {n} for the box.", 1),
            ("check", f"Checks that {n} + {c} = {total}, matching the left side.", 1),
        ],
        partial=f"{a} + {b} is {total}.",
        weak=f"The box is {total}, because {a} + {b} = {total}.",
    )


def equality_one_side(rng):
    x = rng.randint(2, 9)
    a = rng.randint(1, 9)
    b = rng.randint(2, 9)
    t = x + a
    return item(
        pick(
            rng,
            (
                f"Mia has the equation x + {a} = {t}. She adds {b} to the left side only and writes x + {a} + {b} = {t}. "
                f"Explain why this is no longer a true equation for the same x, and say how to keep the equation balanced.",
                f"Start with x + {a} = {t}. Mia adds {b} to one side only, giving x + {a} + {b} = {t}. Why has she broken "
                f"the equation, and what should she do instead?",
                f"Explain what goes wrong when {b} is added to the left side of x + {a} = {t} but not the right, and "
                f"describe the change that keeps both sides equal.",
            ),
        ),
        f"The equation x + {a} = {t} says both sides have the same value. Adding {b} to the left side only makes the "
        f"left side {b} bigger than the right side, so the two sides are no longer equal. To keep it balanced, add {b} "
        f"to both sides: x + {a} + {b} = {t} + {b}, which is {t + b}.",
        [
            (
                "one-side",
                f"Says that adding {b} to only the left side makes the left side {b} bigger than the right.",
                1,
            ),
            ("not-equal", "Says the two sides no longer have the same value, so the equation is not balanced.", 1),
            ("both-sides", f"Says the same amount ({b}) must be added to both sides.", 1),
            ("fixed", f"Writes the balanced equation x + {a} + {b} = {t + b}.", 1),
        ],
        partial=f"Mia only added {b} to the left side.",
        weak="Adding to one side does not matter, because x is still the same number.",
    )


def equality_keeps_true(rng):
    m = rng.randint(2, 6)
    x = rng.randint(2, 9)
    b = rng.randint(1, 9)
    t = m * x + b
    return item(
        pick(
            rng,
            (
                f"The equation {m}x + {b} = {t} is true when x = {x}. Using what the equals sign means, explain why "
                f"{m}x = {t - b} is also true when x = {x}.",
                f"When x = {x}, the equation {m}x + {b} = {t} holds. Explain why taking {b} from both sides gives "
                f"{m}x = {t - b}, which holds for the same x.",
                f"Explain, using the idea of equal values on both sides, why subtracting {b} from each side of "
                f"{m}x + {b} = {t} leaves a true statement when x = {x}.",
            ),
        ),
        f"When x = {x}, the left side {m}x + {b} is {m * x + b}, which is the same value as the right side {t}. If "
        f"{b} is taken away from both sides, both sides change by the same amount so they are still equal. That gives "
        f"{m}x = {t - b}, and {m} × {x} = {m * x} = {t - b}.",
        [
            ("both-equal", f"Says both sides have the same value ({t}) when x = {x}.", 1),
            ("same-change", f"Says {b} is subtracted from both sides, so they stay equal.", 1),
            ("new-equation", f"Arrives at {m}x = {t - b}.", 1),
            ("numeric-check", f"Checks that {m} × {x} = {m * x} equals {t - b}.", 1),
        ],
        partial=f"When x = {x}, both sides of {m}x + {b} = {t} equal {t}.",
        weak=f"{b} just moves to the other side, so the equation changes.",
    )


_EQ = {
    "easy": (equality_statement,),
    "medium": (equality_misreading, equality_missing_number),
    "hard": (equality_one_side, equality_keeps_true),
}


# ---------------------------------------------------------------------------------------------------------------
# 2. Inverse operations
# ---------------------------------------------------------------------------------------------------------------
def inverse_one_step(rng):
    n = rng.randint(2, 12)
    x = rng.randint(2, 15)
    op = rng.randrange(4)
    if op == 0:
        eq, r = f"x + {n} = {x + n}", x + n
        did, undo, step = f"{n} has been added to x", f"subtracting {n}", f"x = {r} - {n} = {x}"
        inverse_word = "subtracting"
    elif op == 1:
        eq, r = f"x - {n} = {x}", x
        did, undo, step = f"{n} has been subtracted from x", f"adding {n}", f"x = {r} + {n} = {x + n}"
        x = x + n
        inverse_word = "adding"
    elif op == 2:
        eq, r = f"{n}x = {n * x}", n * x
        did, undo, step = f"x has been multiplied by {n}", f"dividing by {n}", f"x = {r} ÷ {n} = {x}"
        inverse_word = "dividing"
    else:
        eq, r = f"x ÷ {n} = {x}", x
        did, undo, step = f"x has been divided by {n}", f"multiplying by {n}", f"x = {r} × {n} = {x * n}"
        x = x * n
        inverse_word = "multiplying"
    return item(
        f"To solve {eq}, which operation undoes what has been done to x? Explain why it works, and give the value of x.",
        f"In {eq}, {did}. The inverse operation, {undo}, undoes it. It is done to both sides so that the sides stay "
        f"equal: {step}.",
        [
            ("what-was-done", f"Says what has been done to x: {did}.", 1),
            ("inverse", f"Names the inverse operation: {undo}.", 1),
            ("both-sides", "Says the inverse operation is done to both sides to keep the equation balanced.", 1),
            ("value", f"Gives x = {x}.", 1),
        ],
        partial=f"The equation tells us that {did}.",
        weak=f"Move the number across and change it, so x is {r}.",
    )


def inverse_two_step(rng):
    a = rng.randint(2, 6)
    b = rng.randint(1, 12)
    x = rng.randint(b // a + 2, b // a + 10)
    plus = rng.randrange(2) == 0
    if plus:
        eq, c = f"{a}x + {b} = {a * x + b}", a * x + b
        first, mid = f"subtract {b}", f"{c} - {b} = {a * x}"
    else:
        eq, c = f"{a}x - {b} = {a * x - b}", a * x - b
        first, mid = f"add {b}", f"{c} + {b} = {a * x}"
    return item(
        pick(
            rng,
            (
                f"List, in order, the steps that undo the operations in {eq}, and explain why the steps go in that order. "
                f"Then give x.",
                f"To solve {eq}, which inverse step comes first and which comes second? Explain the order, then give x.",
                f"Describe how to undo what has been done to x in {eq}, step by step, say why the order matters, and "
                f"find x.",
            ),
        ),
        f"In {eq}, x is first multiplied by {a} and then {b} is {'added' if plus else 'subtracted'}. The steps are "
        f"undone in the reverse order: first {first} on both sides ({mid}), then divide both sides by {a} "
        f"({a * x} ÷ {a} = {x}). The last operation done must be the first one undone. x = {x}.",
        [
            ("order", "Says the operations are undone in the reverse order, so the last one done is undone first.", 1),
            ("first-step", f"Says to {first} on both sides first.", 1),
            ("second-step", f"Says to divide both sides by {a} second.", 1),
            ("value", f"Gives x = {x}.", 1),
        ],
        partial=f"First {first}.",
        weak=f"Divide by {a} first, then {first}, because that is the order they are written.",
    )


def inverse_wrong_order(rng):
    a = rng.randint(2, 6)
    b = rng.randint(1, 9)
    x = rng.randint(b // a + 2, b // a + 10)
    c = a * x - b
    check(c > 0, "inverse: c must be positive")
    return item(
        pick(
            rng,
            (
                f"Leo solves {a}x - {b} = {c} by dividing both sides by {a} first, and writes x - {b} = {c} ÷ {a}. Explain "
                f"what is wrong with Leo's step, and show a correct way to solve the equation.",
                f"To solve {a}x - {b} = {c}, Leo divides only the {a}x by {a} and gets x - {b} = {c} ÷ {a}. Why is that "
                f"not allowed? Show how to solve it correctly.",
                f"Which step in Leo's working of {a}x - {b} = {c} is wrong, and why? Leo's first line is x - {b} = "
                f"{c} ÷ {a}. Give a correct solution.",
            ),
        ),
        f"Dividing both sides by {a} divides every term, so the {b} must be divided too: the left side becomes "
        f"x - {b} ÷ {a}, not x - {b}. Leo divided only the {a}x term. It is simpler to undo the subtraction first: "
        f"add {b} to both sides to get {a}x = {c + b}, then divide both sides by {a} to get x = {x}.",
        [
            ("error", f"Says Leo divided only the {a}x term and left the {b} undivided.", 1),
            ("every-term", "Says that dividing both sides by a number divides every term.", 1),
            ("method", f"Gives a correct method, such as adding {b} to both sides first.", 1),
            ("value", f"Gives x = {x}.", 1),
        ],
        partial=f"Leo did not divide the {b}.",
        weak=f"Leo is right, because dividing by {a} undoes the multiplying.",
    )


_INV = {
    "easy": (inverse_one_step,),
    "medium": (inverse_two_step,),
    "hard": (inverse_wrong_order,),
}


# ---------------------------------------------------------------------------------------------------------------
# 3. Solutions and solution sets
# ---------------------------------------------------------------------------------------------------------------
def solution_test(rng):
    a = rng.randint(2, 7)
    b = rng.randint(1, 9)
    v = rng.randint(2, 9)
    c = a * v + b
    claimed = v if rng.randrange(2) else v + pick(rng, (1, 2, -1))
    left = a * claimed + b
    ok = left == c
    word = "is" if ok else "is not"
    return item(
        pick(
            rng,
            (
                f"Is x = {claimed} a solution of {a}x + {b} = {c}? Explain how you can tell.",
                f"Does x = {claimed} make {a}x + {b} = {c} true? Show how you decide whether it is a solution.",
                f"Kim thinks {claimed} is a solution of the equation {a}x + {b} = {c}. Test Kim's idea and explain what "
                f"your test shows.",
            ),
        ),
        f"A solution is a value of x that makes the equation true. Substitute x = {claimed}: the left side is "
        f"{a} × {claimed} + {b} = {left}, and the right side is {c}. "
        + ("They are equal" if ok else f"{left} and {c} are not equal")
        + f", so x = {claimed} {word} a solution.",
        [
            ("meaning", "Says a solution is a value of x that makes the equation true.", 1),
            ("substitute", f"Substitutes x = {claimed} into the left side.", 1),
            ("evaluate", f"Works out the left side as {left}.", 1),
            ("conclusion", f"Compares {left} with {c} and concludes x = {claimed} {word} a solution.", 1),
        ],
        partial="A solution makes the equation true.",
        weak=f"{claimed} is a solution because it is a number in the equation.",
    )


def solution_pick(rng):
    a = rng.randint(2, 9)
    v = rng.randint(3, 12)
    c = v + a
    others = rng.sample([n for n in range(1, 16) if n != v], 2)
    cands = sorted([v, *others])
    return item(
        pick(
            rng,
            (
                f"Which of the numbers {cands[0]}, {cands[1]} and {cands[2]} is a solution of x + {a} = {c}? Explain how you "
                f"decided, and say why the others are not solutions.",
                f"One of {cands[0]}, {cands[1]} and {cands[2]} satisfies x + {a} = {c}. Find it by testing, and explain "
                f"why each of the other two fails.",
                f"Test {cands[0]}, {cands[1]} and {cands[2]} in the equation x + {a} = {c}. Say which is the solution and "
                f"what the tests show about the rest.",
            ),
        ),
        f"Substitute each number for x. For x = {v}: {v} + {a} = {c}, which matches the right side, so {v} is a "
        + "solution. "
        + " ".join(f"For x = {o}: {o} + {a} = {o + a}, which is not {c}, so {o} is not a solution." for o in others)
        + " Only a number that makes both sides equal is a solution.",
        [
            ("method", "Substitutes each of the numbers into the equation.", 1),
            ("correct", f"Shows {v} + {a} = {c} and picks {v}.", 1),
            (
                "others",
                f"Shows the others fail: {others[0]} + {a} = {others[0] + a} and {others[1]} + {a} = {others[1] + a}.",
                1,
            ),
            ("meaning", "Says a solution must make both sides equal.", 1),
        ],
        partial=f"{v} is the solution.",
        weak=f"The biggest number is the solution because x + {a} should be big.",
    )


def solution_set_inequality(rng):
    a = rng.randint(1, 9)
    c = rng.randint(a + 3, a + 12)
    edge = c - a
    inside = edge + rng.randint(1, 5)
    outside = edge - rng.randint(0, 2)
    return item(
        pick(
            rng,
            (
                f"Explain why x = {inside} is a solution of x + {a} > {c} but x = {outside} is not, and describe the "
                f"whole solution set.",
                f"Describe every value of x that satisfies x + {a} > {c}. Use x = {inside} and x = {outside} to explain "
                f"what separates solutions from non-solutions.",
                f"Substitute {inside} and {outside} for x in x + {a} > {c}. What do you find, and what does the full "
                f"solution set look like?",
            ),
        ),
        f"Substitute: {inside} + {a} = {inside + a}, which is greater than {c}, so {inside} is a solution. "
        f"{outside} + {a} = {outside + a}, which is not greater than {c}, so {outside} is not. The solution set is "
        f"every number greater than {edge}. There are infinitely many solutions, and {edge} itself is not one, "
        f"because {edge} + {a} = {c} is equal to {c}, not greater than it.",
        [
            ("inside", f"Shows {inside} + {a} = {inside + a} is greater than {c}.", 1),
            ("outside", f"Shows {outside} + {a} = {outside + a} is not greater than {c}.", 1),
            ("set", f"Describes the solution set as all numbers greater than {edge}.", 1),
            ("boundary", f"Says {edge} is not included, because {edge} + {a} equals {c}.", 1),
        ],
        partial=f"{inside} + {a} = {inside + a}, which is more than {c}.",
        weak=f"The solution is x = {edge}, one number.",
    )


def solution_set_square(rng):
    s = rng.randint(2, 12)
    return item(
        f"Explain why the equation x² = {s * s} has two solutions, and write its solution set.",
        f"A solution is a value of x that makes the equation true. {s} works because {s} × {s} = {s * s}. -{s} also "
        f"works because (-{s}) × (-{s}) = {s * s}, since a negative times a negative is positive. The solution set is "
        f"all the values that work: {{-{s}, {s}}}.",
        [
            ("positive", f"Shows {s} is a solution: {s} × {s} = {s * s}.", 1),
            ("negative", f"Shows -{s} is a solution: (-{s}) × (-{s}) = {s * s}.", 1),
            ("set", f"Gives the solution set {{-{s}, {s}}}.", 1),
            ("meaning", "Says the solution set is all the values that make the equation true.", 1),
        ],
        partial=f"{s} × {s} = {s * s}.",
        weak=f"x = {s}, because the square root of {s * s} is {s}.",
    )


_SOL = {
    "easy": (solution_test,),
    "medium": (solution_pick,),
    "hard": (solution_set_inequality, solution_set_square),
}


# ---------------------------------------------------------------------------------------------------------------
# 4. Identities and equations with no solution
# ---------------------------------------------------------------------------------------------------------------
def _lin(coef, const):
    """``coef x + const`` as the text a student reads: 3x + 2, x - 4, 2x."""
    head = "x" if coef == 1 else f"{coef}x"
    if const == 0:
        return head
    return f"{head} + {const}" if const > 0 else f"{head} - {-const}"


def _xt(n) -> str:
    """``n`` copies of x as written: x, 2x, 3x (never 1x)."""
    return "x" if n == 1 else f"{n}x"


def _plain(expr) -> str:
    """A sympy expression as a student writes it: 3x + 15, not 3*x + 15."""
    return str(expr).replace("*", "")


def _kind_text(kind):
    return {
        "none": "no solution",
        "all": "every number as a solution",
        "one": "exactly one solution",
    }[kind]


def identity_simple(rng):
    n = rng.randint(2, 6)
    a = rng.randint(1, 7)
    kind = pick(rng, ("none", "all"))
    c = n * a if kind == "all" else n * a + pick(rng, (1, 2, 3, -1, -2))
    eq = f"{n}(x + {a}) = {_lin(n, c)}"
    x_free = sp.Symbol("x")
    diff = sp.expand(n * (x_free + a) - (n * x_free + c))
    check((diff == 0) == (kind == "all"), "identity: kind disagrees with algebra")
    if kind == "all":
        tail = f"This leaves {n * a} = {c}, which is always true, so every number is a solution."
    else:
        tail = f"This leaves {n * a} = {c}, which is never true, so no number is a solution."
    return item(
        pick(
            rng,
            (
                f"Does {eq} have no solution, exactly one solution, or every number as a solution? Explain how you can tell "
                f"without trying lots of numbers.",
                f"Without testing many values, say how many solutions {eq} has (none, one or every number), and explain.",
                f"Lee claims {eq} is true for some numbers only. Is that right, or does it have no solution or every "
                f"number as a solution? Explain.",
            ),
        ),
        f"Expand the left side: {n}(x + {a}) = {n}x + {n * a}. The equation becomes {n}x + {n * a} = {n}x + {c}. "
        f"Take away {n}x from both sides. {tail}",
        [
            ("expand", f"Expands the bracket: {n}(x + {a}) = {n}x + {n * a}.", 1),
            ("cancel", f"Takes {n}x from both sides, leaving only numbers.", 1),
            (
                "statement",
                f"Says the leftover statement {n * a} = {c} is {'always' if kind == 'all' else 'never'} true.",
                1,
            ),
            ("conclusion", f"Concludes the equation has {_kind_text(kind)}.", 1),
        ],
        partial=f"{n}(x + {a}) = {n}x + {n * a}.",
        weak=f"Try x = 1 and x = 2; if they work then x is the solution.",
    )


def identity_three_way(rng):
    kind = pick(rng, ("none", "all", "one"))
    p = rng.randint(1, 4)
    r = rng.randint(2, 5)
    q = rng.randint(1, 9)
    x_free = sp.Symbol("x")
    left = f"{_lin(p, q)} + {r}x"
    total = p + r
    if kind == "one":
        c = pick(rng, [v for v in range(1, 10) if v != total])
        x0 = rng.randint(2, 8)
        s = (total - c) * x0 + q
        right = _lin(c, s)
        check(sp.solve(sp.Eq(p * x_free + q + r * x_free, c * x_free + s), x_free) == [x0], "identity: one")
        reasoning = (
            f"Collect the x-terms on the left: {left} = {_lin(total, q)}. The coefficients of x are different "
            f"({total} and {c}), so the equation can be solved to one value: take {_xt(min(total, c))} from both sides to "
            f"get {_xt(abs(total - c))} = {abs(s - q)}, so x = {x0}."
        )
        crit = (
            "coefficients",
            f"Collects the left side to {_lin(total, q)} and sees the coefficients of x differ ({total} and {c}).",
            1,
        )
        concl = ("value", f"Solves to get x = {x0}.", 1)
    else:
        s = q if kind == "all" else q + pick(rng, (1, 2, 3, 4))
        right = _lin(total, s)
        check((sp.expand(p * x_free + q + r * x_free - (total * x_free + s)) == 0) == (kind == "all"), "identity: kind")
        reasoning = (
            f"Collect the x-terms on the left: {left} = {_lin(total, q)}. The equation is {_lin(total, q)} = "
            f"{right}. Take {total}x from both sides to leave {q} = {s}, which is "
            f"{'always' if kind == 'all' else 'never'} true, so the equation has {_kind_text(kind)}."
        )
        crit = (
            "coefficients",
            f"Collects the left side to {_lin(total, q)} and sees both sides have the same x-term.",
            1,
        )
        concl = ("statement", f"Says the leftover {q} = {s} is {'always' if kind == 'all' else 'never'} true.", 1)
    return item(
        f"Without trying lots of numbers, decide whether {left} = {right} has no solution, exactly one solution, or "
        f"every number as a solution. Explain your decision.",
        reasoning,
        [
            ("collect", "Collects the like terms on the left side.", 1),
            crit,
            concl,
            ("conclusion", f"Concludes the equation has {_kind_text(kind)}.", 1),
        ],
        partial="Collects the like terms on the left side.",
        weak="Every equation has exactly one solution, so substitute until it works.",
    )


def identity_collect(rng):
    kind = pick(rng, ("none", "all", "one"))
    p = rng.randint(2, 4)
    a = rng.randint(1, 5)
    q = rng.randint(1, 4)
    x_free = sp.Symbol("x")
    left_text = f"{p}(x + {a}) + {_xt(q)}"
    left_expr = p * (x_free + a) + q * x_free
    total_coef = p + q
    base = p * a
    if kind == "one":
        r = pick(rng, [v for v in range(1, 9) if v != total_coef])
        x0 = rng.randint(2, 7)
        c = (total_coef - r) * x0 + base
        right_text = _lin(r, c)
        sol = sp.solve(sp.Eq(left_expr, r * x_free + c), x_free)
        check(sol == [x0], "identity: collect one")
        reasoning = (
            f"Expand and collect: {p}(x + {a}) + {_xt(q)} = {total_coef}x + {base}. The coefficients of x are different "
            f"({total_coef} and {r}), so the equation can be solved to one value: x = {x0}."
        )
        crit = ("coefficients", f"Compares the coefficients of x ({total_coef} and {r}) and sees they differ.", 1)
        concl = ("value", f"Solves to get x = {x0}.", 1)
    else:
        c = base if kind == "all" else base + pick(rng, (1, 2, 3, -1))
        right_text = _lin(total_coef, c)
        check((sp.expand(left_expr - (total_coef * x_free + c)) == 0) == (kind == "all"), "identity: collect kind")
        reasoning = (
            f"Expand and collect: {p}(x + {a}) + {_xt(q)} = {total_coef}x + {base}. The equation is "
            f"{total_coef}x + {base} = {total_coef}x + {c}. Take {total_coef}x from both sides to leave {base} = {c}, "
            f"which is {'always' if kind == 'all' else 'never'} true."
        )
        crit = (
            "coefficients",
            f"Collects the x-terms to get {total_coef}x and sees both sides have the same x-term.",
            1,
        )
        concl = ("statement", f"Says the leftover {base} = {c} is {'always' if kind == 'all' else 'never'} true.", 1)
    return item(
        f"Decide whether {left_text} = {right_text} has no solution, exactly one solution, or every number as a "
        f"solution. Show the algebra that tells you, and explain.",
        reasoning,
        [
            ("expand", f"Expands the bracket to get {p}x + {base}.", 1),
            crit,
            concl,
            ("conclusion", f"Concludes the equation has {_kind_text(kind)}.", 1),
        ],
        partial=f"{p}(x + {a}) = {p}x + {base}.",
        weak="Add the numbers on each side; if they are different then it has no solution.",
    )


_IDN = {
    "easy": (identity_simple,),
    "medium": (identity_three_way,),
    "hard": (identity_collect,),
}


# ---------------------------------------------------------------------------------------------------------------
# 5. Checking a solution by substitution
# ---------------------------------------------------------------------------------------------------------------
def check_one_side(rng):
    a = rng.randint(2, 7)
    b = rng.randint(1, 9)
    v = rng.randint(2, 9)
    c = a * v + b
    claimed = v if rng.randrange(3) else v + pick(rng, (1, -1, 2))
    left = a * claimed + b
    ok = left == c
    return item(
        pick(
            rng,
            (
                f"Zoe solved {a}x + {b} = {c} and got x = {claimed}. Show how to check her answer by substitution, and say "
                f"whether it is correct.",
                f"Check by substitution whether x = {claimed} is the right answer to {a}x + {b} = {c}, and explain what "
                f"the check tells you.",
                f"Zoe's answer to {a}x + {b} = {c} is x = {claimed}. Put her value back into the equation. Is she "
                f"correct? Explain.",
            ),
        ),
        f"Put x = {claimed} into the left side: {a} × {claimed} + {b} = {a * claimed} + {b} = {left}. The right side is {c}. "
        + (
            f"The two sides are equal, so x = {claimed} is correct."
            if ok
            else f"{left} is not equal to {c}, so x = {claimed} is not correct."
        ),
        [
            ("substitute", f"Replaces x with {claimed} in {a}x + {b}.", 1),
            ("evaluate", f"Works out the left side as {left}.", 1),
            ("compare", f"Compares {left} with the right side, {c}.", 1),
            ("conclusion", f"Concludes x = {claimed} is {'correct' if ok else 'not correct'}.", 1),
        ],
        partial=f"Replace x with {claimed}.",
        weak=f"Solve it again to see if you get {claimed}.",
    )


def check_both_sides(rng):
    a = rng.randint(4, 8)
    c = rng.randint(2, a - 1)
    x0 = rng.randint(2, 8)
    b = rng.randint(1, 9)
    d = (a - c) * x0 + b
    claimed = x0 if rng.randrange(2) else x0 + pick(rng, (1, -1, 2))
    left, right = a * claimed + b, c * claimed + d
    ok = left == right
    check(ok == (claimed == x0), "check: both sides")
    return item(
        pick(
            rng,
            (
                f"Ben says the solution of {a}x + {b} = {c}x + {d} is x = {claimed}. Explain how to check this by "
                f"substitution and say whether Ben is right.",
                f"Substitute x = {claimed} into both sides of {a}x + {b} = {c}x + {d}. What do the two sides show about "
                f"Ben's claim that it is the solution?",
                f"Is x = {claimed} the solution of {a}x + {b} = {c}x + {d}, as Ben says? Check each side and explain.",
            ),
        ),
        f"Substitute x = {claimed} into both sides. Left side: {a} × {claimed} + {b} = {left}. Right side: "
        f"{c} × {claimed} + {d} = {right}. "
        + (
            "The sides are equal, so Ben is right." if ok else f"{left} and {right} are different, so Ben is not right."
        ),
        [
            ("both", f"Substitutes x = {claimed} into both sides of the equation.", 1),
            ("left", f"Works out the left side as {left}.", 1),
            ("right", f"Works out the right side as {right}.", 1),
            (
                "conclusion",
                f"Concludes Ben is {'right' if ok else 'not right'} because the sides are {'equal' if ok else 'different'}.",
                1,
            ),
        ],
        partial="Put the number in for x on both sides.",
        weak=f"Add the numbers on the left: {a} + {b} is {a + b}, and the right is {c} + {d}.",
    )


def check_find_slip(rng):
    a = pick(rng, (2, 3, 4, 5, 6))
    b = pick(rng, [n for n in range(2, 13) if (2 * n) % a == 0])
    x = 2 * b // a + rng.randint(1, 5)
    c = a * x - b
    wrong = Fraction(c - b, a)
    check(wrong.denominator == 1 and wrong != x and wrong > 0, "check: slip answer")
    w = int(wrong)
    left = a * w - b
    return item(
        pick(
            rng,
            (
                f"Ari solved {a}x - {b} = {c} and got x = {w}. Check Ari's answer by substitution. Explain what the check "
                f"shows and what mistake Ari probably made.",
                f"Ari's answer to {a}x - {b} = {c} is x = {w}. Substitute it, say what happens, and suggest the slip Ari "
                f"most likely made.",
                f"Something is wrong with x = {w} as the solution of {a}x - {b} = {c}. Use substitution to show it, and "
                f"explain the likely error.",
            ),
        ),
        f"Substitute x = {w}: the left side is {a} × {w} - {b} = {left}, but the right side is {c}. {left} is not "
        f"equal to {c}, so x = {w} is not a solution. Ari subtracted {b} from both sides instead of adding it. The "
        f"correct working is {a}x = {c} + {b} = {c + b}, so x = {x}. Check: {a} × {x} - {b} = {a * x - b} = {c}.",
        [
            ("substitute", f"Substitutes x = {w} to get {a} × {w} - {b} = {left}.", 1),
            ("conclusion", f"Says {left} is not {c}, so x = {w} is not a solution.", 1),
            ("slip", f"Says Ari subtracted {b} when {b} should have been added to both sides.", 1),
            ("correct", f"Gives the correct solution x = {x}.", 1),
            ("recheck", f"Checks x = {x}: {a} × {x} - {b} = {c}.", 1),
        ],
        partial=f"{a} × {w} - {b} = {left}, which is not {c}.",
        weak=f"Ari is right; I get {w} too.",
        partial_marks=2,
    )


_CHK = {
    "easy": (check_one_side,),
    "medium": (check_both_sides,),
    "hard": (check_find_slip,),
}


# ---------------------------------------------------------------------------------------------------------------
# 6. Recognising equivalent expressions
# ---------------------------------------------------------------------------------------------------------------
def _expr_pair(rng, tier):
    """(expression 1 text, expression 2 text, sympy 1, sympy 2) with a chosen truth."""
    x = sp.Symbol("x")
    if tier == "easy":
        n, a = rng.randint(2, 6), rng.randint(2, 9)
        e1 = f"{n}(x + {a})"
        good = f"{n}x + {n * a}"
        bad = pick(rng, (f"{n}x + {a}", f"{n + a}x", f"{n}x + {n + a}"))
        s1 = n * (x + a)
        equivalent = rng.randrange(2) == 0
        e2 = good if equivalent else bad
        s2 = _parse(e2, x)
        return e1, e2, s1, s2
    if tier == "medium":
        p, q, r = rng.randint(2, 7), rng.randint(2, 7), rng.randint(1, 9)
        e1 = f"{p}x + {r} + {q}x"
        s1 = p * x + r + q * x
        equivalent = rng.randrange(2) == 0
        e2 = _lin(p + q, r) if equivalent else pick(rng, (_lin(p + q, r + p), _lin(p * q, r), _lin(p + q + 1, r)))
        return e1, e2, s1, _parse(e2, x)
    p, q, a, b = rng.randint(2, 4), rng.randint(2, 4), rng.randint(1, 6), rng.randint(1, 6)
    e1 = f"{p}(x + {a}) + {q}(x - {b})"
    s1 = p * (x + a) + q * (x - b)
    equivalent = rng.randrange(2) == 0
    good_const = p * a - q * b
    e2 = (
        _lin(p + q, good_const)
        if equivalent
        else pick(rng, (_lin(p + q, p * a + q * b), _lin(p + q, good_const + 1), _lin(p * q, good_const)))
    )
    return e1, e2, s1, _parse(e2, x)


def _parse(text, x):
    return parse_expr(text, local_dict={"x": x}, transformations=_TRANSFORMS)


def _simplifies_to(e1, e2, s1):
    """Sentence saying what `e1` simplifies to, without claiming that an expression matches itself."""
    plain = _plain(sp.expand(s1))
    if plain == e2:
        return f"expanding and collecting like terms in {e1} gives {e2}, the second expression exactly."
    return f"simplifying {e1} gives {plain}, which is the same as {e2}."


def equivalent_item(rng, tier):
    x = sp.Symbol("x")
    for _ in range(200):
        e1, e2, s1, s2 = _expr_pair(rng, tier)
        equivalent = sp.expand(s1 - s2) == 0
        t = rng.randint(2, 7)
        v1, v2 = s1.subs(x, t), s2.subs(x, t)
        if equivalent or v1 != v2:
            break
    else:  # pragma: no cover
        raise AssertionError("no pair found")
    t2 = t + 1
    word = "equivalent" if equivalent else "not equivalent"
    if equivalent:
        ref = (
            f"Try x = {t}: {e1} gives {v1} and {e2} gives {v2}, which match. One matching value is not enough on its "
            f"own, so use algebra: {_simplifies_to(e1, e2, s1)} The expressions "
            f"are equivalent because they have the same value for every x."
        )
        crit_alg = (
            "algebra",
            f"Shows by expanding or collecting like terms that {e1} simplifies to {_plain(sp.expand(s1))}.",
            1,
        )
        crit_con = (
            "conclusion",
            "Concludes the expressions are equivalent because they agree for every value of x.",
            1,
        )
    else:
        ref = (
            f"Try x = {t}: {e1} gives {v1} but {e2} gives {v2}. The values are different, so one value of x is enough "
            f"to show the expressions are not equivalent. Simplifying {e1} gives {_plain(sp.expand(s1))}, which is not "
            f"{e2}."
        )
        crit_alg = ("algebra", f"Shows that {e1} simplifies to {_plain(sp.expand(s1))}, which differs from {e2}.", 1)
        crit_con = (
            "conclusion",
            "Concludes the expressions are not equivalent because one value of x gives different results.",
            1,
        )
    prompt = {
        "easy": pick(
            rng,
            (
                f"Are {e1} and {e2} equivalent expressions? Test a value of x, then explain your decision.",
                f"Pick a value for x and use it to compare {e1} with {e2}. Are the two expressions equivalent? Explain.",
                f"Mei says {e1} and {e2} are the same expression written two ways. Is she right? Test a value and explain.",
            ),
        ),
        "medium": pick(
            rng,
            (
                f"Are {e1} and {e2} equivalent? Test x = {t} in both, then explain what your test shows and does not show.",
                f"Put x = {t} into {e1} and into {e2}. What does the result tell you about whether they are equivalent, "
                f"and what can it not tell you?",
                f"A student tests x = {t} on {e1} and {e2} to decide if they are equivalent. Do the test, then say how "
                f"much it proves.",
            ),
        ),
        "hard": f"Decide whether {e1} and {e2} are equivalent. Explain why testing one value of x cannot prove they "
        f"are equivalent, and show the algebra that settles it.",
    }[tier]
    crit_test = ("test", f"Substitutes x = {t} into both expressions and gets {v1} and {v2}.", 1)
    crit_limit = (
        "limit",
        "Says that one matching value does not prove equivalence, but one different value proves they are not equivalent.",
        1,
    )
    if tier == "easy":
        criteria = [crit_test, crit_alg, crit_con]
    else:
        criteria = [crit_test, crit_limit, crit_alg, crit_con]
    return item(
        prompt,
        ref,
        criteria,
        partial=f"At x = {t}, {e1} is {v1} and {e2} is {v2}.",
        weak=f"They look different, so they are {'not ' if equivalent else ''}equivalent.",
    )


def equivalent_easy(rng):
    return equivalent_item(rng, "easy")


def equivalent_medium(rng):
    return equivalent_item(rng, "medium")


def equivalent_hard(rng):
    return equivalent_item(rng, "hard")


_EQV = {"easy": (equivalent_easy,), "medium": (equivalent_medium,), "hard": (equivalent_hard,)}


# ---------------------------------------------------------------------------------------------------------------
# 7. Independent and dependent variables
# ---------------------------------------------------------------------------------------------------------------
_CONTEXTS = (
    (
        lambda f, r: f"A plumber charges a ${f} call-out fee plus ${r} for each hour worked",
        "the number of hours worked",
        "the total charge",
        "hour",
        (50, 60, 80, 90, 100),
        (70, 80, 90, 100, 110),
    ),
    (
        lambda f, r: f"A taxi charges ${f} flag-fall plus ${r} for each kilometre",
        "the distance travelled",
        "the fare",
        "kilometre",
        (3, 4, 5, 6),
        (2, 3),
    ),
    (
        lambda f, r: f"A gym charges a ${f} joining fee plus ${r} for each week",
        "the number of weeks",
        "the total cost",
        "week",
        (20, 30, 40, 50, 60),
        (10, 12, 15, 18, 20),
    ),
    (
        lambda f, r: f"A phone plan costs ${f} a month plus ${r} for each gigabyte of data",
        "the gigabytes used",
        "the monthly bill",
        "gigabyte",
        (20, 30, 40, 50),
        (5, 8, 10, 12),
    ),
    (
        lambda f, r: f"A candle starts {f} cm tall and burns down {r % 5 + 1} cm each hour",
        "the time it has burned",
        "the height of the candle",
        "hour",
        (20, 24, 30, 36, 40),
        (1, 2, 3, 4, 5),
    ),
    (
        lambda f, r: f"A tank already holds {f} litres and a hose adds {r} litres each minute",
        "the time the hose has run",
        "the volume of water in the tank",
        "minute",
        (20, 50, 80, 100, 150),
        (5, 8, 10, 12, 15),
    ),
    (
        lambda f, r: f"A hire company charges a ${f} booking fee plus ${r} for each day of a bike hire",
        "the number of days",
        "the hire cost",
        "day",
        (10, 15, 20, 25),
        (15, 20, 25, 30),
    ),
    (
        lambda f, r: f"A baker sells muffins for ${r} each, and a customer also pays a ${f} delivery fee",
        "the number of muffins bought",
        "the total cost",
        "muffin",
        (3, 4, 5, 6, 8),
        (2, 3, 4, 5),
    ),
)


def _context(rng):
    sentence, ind, dep, per, fees, rates = pick(rng, _CONTEXTS)
    return sentence(pick(rng, fees), pick(rng, rates)), ind, dep, per


def variables_name(rng):
    sit, ind, dep, per = _context(rng)
    return item(
        f"{sit}. Name the independent variable and the dependent variable, and explain how you decided which is which.",
        f"The independent variable is {ind}, because it is the quantity that changes freely or is chosen, and it does "
        f"not rely on anything else here. The dependent variable is {dep}, because its value is worked out from "
        f"{ind}: if {ind} changes then {dep} changes with it.",
        [
            ("independent", f"Names {ind} as the independent variable.", 1),
            ("dependent", f"Names {dep} as the dependent variable.", 1),
            ("reason", f"Says {dep} depends on, or is worked out from, {ind}.", 1),
        ],
        partial=f"The independent variable is {ind}.",
        weak=f"The independent variable is {dep}, because it is the answer we want.",
    )


def variables_formula(rng):
    fee = pick(rng, (10, 15, 20, 25))
    rate = pick(rng, (12, 15, 18, 20, 25))
    t1, t2 = sorted(rng.sample(range(1, 9), 2))
    v1, v2 = fee + rate * t1, fee + rate * t2
    return item(
        pick(
            rng,
            (
                f"The cost C dollars of hiring a bike for d days is C = {fee} + {rate}d. Say which letter is the independent "
                f"variable and which is the dependent variable. Use the values d = {t1} and d = {t2} to explain what it means "
                f"for one variable to depend on the other.",
                f"A bike hire shop uses C = {fee} + {rate}d, where d is the number of days and C the cost in dollars. "
                f"Which letter is independent and which is dependent? Work out C for d = {t1} and d = {t2} to explain.",
                f"In the rule C = {fee} + {rate}d for the cost of hiring a bike, explain which variable depends on the "
                f"other, using the cases d = {t1} and d = {t2}.",
            ),
        ),
        f"d is the independent variable and C is the dependent variable. When d = {t1}, C = {fee} + {rate} × {t1} = {v1}. "
        f"When d = {t2}, C = {fee} + {rate} × {t2} = {v2}. Choosing a different d gives a different C, so C depends on d. "
        f"The value of d is chosen first and C is worked out from it.",
        [
            ("letters", "Says d is the independent variable and C is the dependent variable.", 1),
            ("values", f"Works out C = {v1} when d = {t1} and C = {v2} when d = {t2}.", 1),
            ("depends", "Says that changing d changes C, so C depends on d.", 1),
            ("order", "Says d is chosen or changed first and C is worked out from it.", 1),
        ],
        partial="d is the independent variable.",
        weak="C is the independent variable because it comes first in the formula.",
    )


def variables_axes(rng):
    sit, ind, dep, per = _context(rng)
    k = rng.randint(2, 9)
    t1, t2 = sorted(rng.sample(range(1, 9), 2))
    return item(
        f"{sit}. A student wants to draw a graph of this situation. Which variable goes on the horizontal axis and "
        f"which on the vertical axis? Explain why, and say what would be strange about swapping them.",
        f"{ind[0].upper()}{ind[1:]} is the independent variable, so it goes on the horizontal axis. {dep[0].upper()}{dep[1:]} is "
        f"the dependent variable, so it goes on the vertical axis. We read off {dep} for a chosen value of {ind}, for "
        f"example by choosing {t1} or {t2} {per}s and reading the result. If they were swapped, the graph would show "
        f"{ind} as if it depended on {dep}, which is not how the situation works.",
        [
            ("horizontal", f"Puts {ind} (the independent variable) on the horizontal axis.", 1),
            ("vertical", f"Puts {dep} (the dependent variable) on the vertical axis.", 1),
            ("reason", f"Says we choose {ind} and read off {dep}.", 1),
            ("swap", f"Says swapping would make {ind} look as if it depended on {dep}.", 1),
        ],
        partial=f"{ind[0].upper()}{ind[1:]} goes on the horizontal axis.",
        weak=f"The bigger number goes on the vertical axis.",
    )


_VAR = {"easy": (variables_name,), "medium": (variables_formula,), "hard": (variables_axes,)}


# ---------------------------------------------------------------------------------------------------------------
# 8. Proportional and non-proportional relationships
# ---------------------------------------------------------------------------------------------------------------
def _table(rng, proportional):
    k = rng.randint(2, 9)
    b = 0 if proportional else rng.randint(1, 9)
    xs = sorted(rng.sample((1, 2, 4, 5, 8), 3))
    ys = [k * x + b for x in xs]
    return k, b, xs, ys


def proportional_table(rng):
    proportional = rng.randrange(2) == 0
    k, b, xs, ys = _table(rng, proportional)
    ratios = [Fraction(y, x) for x, y in zip(xs, ys)]
    same = len(set(ratios)) == 1
    check(same == proportional, "proportional: table")
    row = ", ".join(f"x = {x}, y = {y}" for x, y in zip(xs, ys))
    show = " and ".join(f"{y} ÷ {x} = {_frac(r)}" for x, y, r in zip(xs, ys, ratios))
    return item(
        pick(
            rng,
            (
                f"A table gives these pairs of values: {row}. Is y proportional to x? Explain, using the values.",
                f"Look at the pairs {row}. Decide whether y is proportional to x, and justify your answer with the numbers.",
                f"Is the relationship in this table a proportional one? The pairs are {row}. Explain how you can tell.",
            ),
        ),
        f"Work out y ÷ x for each pair: {show}. "
        + (
            f"The ratio is always {k}, so y is proportional to x."
            if proportional
            else f"The ratio is not the same each time, so y is not proportional to x."
        )
        + " In a proportional relationship y ÷ x is constant.",
        [
            ("ratios", "Works out y ÷ x for at least two pairs.", 1),
            ("compare", "Compares the ratios and says whether they are the same.", 1),
            ("conclusion", f"Concludes y is {'proportional' if proportional else 'not proportional'} to x.", 1),
            ("meaning", "Says that in a proportional relationship y ÷ x is always the same.", 1),
        ],
        partial=f"{ys[0]} ÷ {xs[0]} = {_frac(ratios[0])}.",
        weak=f"y goes up each time x goes up, so it is proportional.",
    )


def _frac(r):
    """A ratio with a terminating decimal as the decimal a student writes: 8, 7.75, 6.875."""
    check(all(f in (2, 5) for f in sp.factorint(r.denominator)), "ratio does not terminate")
    if r.denominator == 1:
        return str(r.numerator)
    text = format(sp.Rational(r.numerator, r.denominator).evalf(12), "f").rstrip("0")
    return text.rstrip(".")


def _money(r):
    """A rate in dollars: whole dollars plain, otherwise two decimal places (the ratios here have at most two)."""
    check((r * 100).denominator == 1, "rate has more than two decimal places")
    return str(r.numerator) if r.denominator == 1 else f"{float(r):.2f}"


def proportional_rule(rng):
    k = rng.randint(2, 9)
    form = rng.randrange(3)
    b = rng.randint(1, 9)
    if form == 0:
        rule, prop = f"y = {k}x", True
    elif form == 1:
        rule, prop = f"y = {k}x + {b}", False
    else:
        rule, prop = f"y = {k}(x + {b})", False
    x0 = rng.randint(2, 6)
    y_at = {0: k * x0, 1: k * x0 + b, 2: k * (x0 + b)}[form]
    y_zero = {0: 0, 1: b, 2: k * b}[form]
    return item(
        f"Is the relationship {rule} proportional? Find y when x = 0 and when x = {x0}, and use your answers to "
        f"explain.",
        f"When x = 0, y = {y_zero}. When x = {x0}, y = {y_at}. "
        + (
            f"A proportional relationship gives y = 0 when x = 0, and this does, with y ÷ x = {k} each time. So {rule} is proportional."
            if prop
            else f"A proportional relationship must give y = 0 when x = 0, but here y = {y_zero}, so {rule} is not proportional."
        ),
        [
            ("zero", f"Finds y = {y_zero} when x = 0.", 1),
            ("other", f"Finds y = {y_at} when x = {x0}.", 1),
            ("test", "Says a proportional relationship must give y = 0 when x = 0.", 1),
            ("conclusion", f"Concludes {rule} is {'proportional' if prop else 'not proportional'}.", 1),
        ],
        partial=f"When x = 0, y = {y_zero}.",
        weak=f"Any rule with an x in it is proportional.",
    )


def proportional_context(rng):
    rate = rng.randint(2, 9)
    fee = rng.randint(2, 9)
    d1, d2 = sorted(rng.sample((2, 4, 5, 10), 2))
    a_name, b_name, noun = pick(
        rng, (("Plan A", "Plan B", "plan"), ("Taxi A", "Taxi B", "taxi"), ("Cab A", "Cab B", "cab company"))
    )
    return item(
        f"{a_name} charges ${rate} for each kilometre. {b_name} charges a ${fee} fee plus ${rate} for each kilometre. "
        f"For which {noun} is the cost proportional to the distance? Compare the costs for {d1} km and {d2} km, and "
        f"explain.",
        f"{a_name}: {d1} km costs ${rate * d1} and {d2} km costs ${rate * d2}; cost ÷ distance is ${rate} per km each time, "
        f"and 0 km costs $0. {b_name}: {d1} km costs ${rate * d1 + fee} and {d2} km costs ${rate * d2 + fee}; cost ÷ "
        f"distance is ${_money(Fraction(rate * d1 + fee, d1))} per km for {d1} km but "
        f"${_money(Fraction(rate * d2 + fee, d2))} per km for {d2} km, and 0 km already costs ${fee}. So the cost is proportional to the distance for {a_name} only.",
        [
            (
                "plan-a",
                f"Finds {a_name}'s costs ${rate * d1} and ${rate * d2}, with cost ÷ distance constant at ${rate}.",
                1,
            ),
            (
                "plan-b",
                f"Finds {b_name}'s costs ${rate * d1 + fee} and ${rate * d2 + fee}, with cost ÷ distance not constant.",
                1,
            ),
            ("zero", f"Notices that {b_name} charges ${fee} even for 0 km, so it does not start from zero.", 1),
            ("conclusion", f"Concludes only {a_name} is proportional.", 1),
        ],
        partial=f"{a_name} charges ${rate * d1} for {d1} km.",
        weak=f"Both are proportional because the cost goes up with the distance.",
    )


_PRO = {"easy": (proportional_table,), "medium": (proportional_rule,), "hard": (proportional_context,)}


# ---------------------------------------------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------------------------------------------
def _register(topic, level, table):
    def gen(rng, tier="easy"):
        if tier not in table:
            raise ValueError(f"unknown tier {tier!r}; known: {', '.join(table)}")
        return pick(rng, table[tier])(rng)

    gen.__name__ = topic
    generator(topic, level, presets={t: {"tier": t} for t in ("easy", "medium", "hard")})(gen)


_register("w10_explain_equality", 7, _EQ)
_register("w10_explain_inverse", 7, _INV)
_register("w10_explain_solution", 7, _SOL)
_register("w10_explain_identity", 8, _IDN)
_register("w10_explain_check", 7, _CHK)
_register("w10_explain_equivalent", 8, _EQV)
_register("w10_explain_variables", 8, _VAR)
_register("w10_explain_proportional", 8, _PRO)
