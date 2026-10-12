"""Exam source of `one-step-equations`: three linked one-step equations set in a real situation.

The practice tiers hand the learner an equation. Here every equation is hidden in a story and the learner has to
choose the inverse operation each time, with the earlier answer feeding the next equation:

    (a) n items cost $T, so n x p = T: find the price p            (1 mark, division)
    (b) $B buys m items at that price, so p x m = B: find m        (2 marks, follow-through from (a), division)
    (c) a total F is shared equally among those m, F / m = g       (2 marks, follow-through from (b), multiplication)

Each answer is a whole number, drawn first, and the totals are built from them. The three equations use a product,
the same product solved for the other factor, and a quotient, so no single habit ("divide") survives all three.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors, rounded_decimal_string

# Each theme is one situation: an item bought in a group, the people who get one each, and a total shared among
# them. {n} {T} {B} {g} are filled in by the generator; no text states an answer.
_THEMES = [
    dict(
        stem="A Year 7 class is planning an excursion to the Melbourne Aquarium. Every student needs a ticket, and every ticket costs the same.",
        a="One group of {n} students pays ${T} altogether for tickets.",
        item="ticket",
        b="The class has ${B} to spend on tickets at that price.",
        c="The bus company refunds a total amount, shared equally among the students who go. Each of them gets ${g} back.",
        who="students",
        buyer="class",
    ),
    dict(
        stem="A netball club in Adelaide orders playing shirts. Every shirt costs the same.",
        a="An order of {n} shirts costs ${T} altogether.",
        item="shirt",
        b="The club has ${B} to spend on shirts at that price.",
        c="A sponsor gives a total amount, shared equally among the players who get a shirt. Each of them receives ${g}.",
        who="players",
        buyer="club",
    ),
    dict(
        stem="A Landcare group near Hobart buys lunch packs for its volunteers. Every lunch pack costs the same.",
        a="{n} lunch packs cost ${T} altogether.",
        item="lunch pack",
        b="The group has ${B} to spend on lunch packs at that price.",
        c="A local business donates a total amount, shared equally among the volunteers who get a pack. Each of them is given ${g}.",
        who="volunteers",
        buyer="group",
    ),
    dict(
        stem="A school choir from Brisbane is buying tickets to a concert. Every ticket costs the same.",
        a="{n} tickets cost ${T} altogether.",
        item="ticket",
        b="The choir has ${B} to spend on tickets at that price.",
        c="A parent group donates a total amount, shared equally among the choir members who get a ticket. Each of them is given ${g}.",
        who="members",
        buyer="choir",
    ),
]


def _dist(value, tag):
    """A wrong answer written the way a learner with a calculator would: to at most 2 decimal places."""
    text = rounded_decimal_string(Fraction(value), 2)
    return Distractor(text, tag, Fraction(text))


def _terminating(value: Fraction) -> bool:
    """True for a positive amount: a negative count or price is not an answer a learner would write."""
    return value >= 1


def _part(prompt, key, marks, working, candidates, follow_through=None):
    # The key is a whole number, so a wrong answer is offered only when it is one too: a decimal option would give
    # the key away by its format.
    cands = [c for c in candidates if Fraction(c.value).denominator == 1 and _terminating(Fraction(c.value))]
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(key),
        answer_text=str(key),
        answer_latex=latex_answer(str(key)),
        marks=marks,
        sensible_range=(0, None),
        working=working,
        distractors=pick_distractors(str(key), cands, 3),
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


@generator("exd_x_sc1_one_step_chain", level=7, presets={"exam": dict(tier="exam")})
def exd_x_sc1_one_step_chain(rng, tier="exam"):
    theme = rng.choice(_THEMES)
    for _ in attempts(1000, "a price, a count and a share"):
        n = rng.randint(3, 14)
        p = rng.randint(3, 48)
        m = rng.randint(4, 36)
        g = rng.randint(2, 29)
        if len({n, p, m, g}) < 4 or m == n or g >= p:  # a share (or refund) is less than the price of one item
            continue
        T, B, F = n * p, p * m, g * m
        if len({T, B, F, n, p, m, g}) < 7:
            continue
        break
    # Each equation is solved by the inverse operation, checked the other way.
    check(T // n == p and T % n == 0 and p * m == B and B // p == m and F // m == g, "equation roots disagree")

    a_part = _part(
        f"Write an equation and solve it to find the cost of one {theme['item']}, in dollars.",
        p,
        1,
        [f"{n} × p = {T}.", f"p = {T} ÷ {n} = {p}."],
        [
            _dist(T * n, "wrong-inverse-operation"),
            _dist(T - n, "wrong-operation-family"),
            _dist(T + n, "wrong-operation-family"),
            _dist(Fraction(n, T), "inverted-the-division"),
        ],
    )
    b_part = _part(
        "Use your answer to (a). Write an equation and solve it to find how many "
        f"{theme['item']}s the {theme['buyer']} can buy.",
        m,
        2,
        [f"{p} × m = {B}.", f"m = {B} ÷ {p} = {m}."],
        [
            _dist(B * p, "wrong-inverse-operation"),
            _dist(B - p, "wrong-operation-family"),
            _dist(B + p, "wrong-operation-family"),
            _dist(Fraction(p, B), "inverted-the-division"),
            _dist(Fraction(B, n), "used-the-wrong-earlier-answer"),
        ],
        follow_through=f"{B}/part_a",
    )
    c_part = _part(
        f"Use your answer to (b) as the number of {theme['who']}. Write an equation and solve it to find the total "
        "amount shared, in dollars.",
        F,
        2,
        [f"F ÷ {m} = {g}.", f"F = {g} × {m} = {F}."],
        [
            _dist(Fraction(g, m), "wrong-inverse-operation"),
            _dist(g + m, "wrong-operation-family"),
            _dist(m - g, "wrong-operation-family"),
            _dist(Fraction(m, g), "inverted-the-division"),
            _dist(g * p, "used-the-wrong-earlier-answer"),
            _dist(g * n, "used-the-wrong-earlier-answer"),
        ],
        follow_through=f"{g}*part_b",
    )
    stem = f"{theme['stem']} {theme['a'].format(n=n, T=T)} {theme['b'].format(B=B)} {theme['c'].format(g=g)}"
    return dict(text=stem, parts=[a_part, b_part, c_part])
