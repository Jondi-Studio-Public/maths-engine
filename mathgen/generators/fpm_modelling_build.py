"""Generators for three Years 7-8 modelling skills (`fpm_*`), each with an exam chain.

* ``translate-word-problems-to-equations`` (answer ``expression``): ``fpm_translate_equation``. The story gives a
  quantity and what it totalled; the learner types the expression for the left-hand side of the equation, in the
  named variable, with the total already on the right. Easy is one step (a rate, a gain, a loss, a share), medium is
  two steps (a fixed charge with a rate, a rate less a voucher, a share less a charge), hard needs a second
  quantity written in terms of the first, so brackets or a repeated variable are unavoidable. The exam chain
  ``fpm_translate_equation_exam`` writes the expression, solves it, then uses the solution twice.
* ``linear-models`` (answer ``number``): ``fpm_linear_model``. Easy reads a stated model, medium builds the model
  from two data points and predicts, hard asks where a model stops making sense, where two models agree, or when a
  fitted model reaches a target. The exam chain ``fpm_linear_model_exam`` finds the rate, the starting value, a
  prediction and the input that reaches a target.
* ``model-relationships-with-functions`` (answer ``expression``): ``fpm_function_model``. Easy writes the rule of a
  proportional relationship from a table, medium the rule of a linear one with a fixed amount from a table that skips
  inputs, hard decides between a linear, inverse and squared relationship. The exam chain ``fpm_function_model_exam``
  is an inverse relationship: the constant, the rule, a value, and the input that gives a target.

Every answer is re-derived a second way: an expression is evaluated at the hidden value the story was built from,
and a model's prediction is recomputed from the two data points with exact fractions. Distractors are expressions or
numbers that a named slip would produce; each is checked unequal to the answer by SymPy (expressions) or by value.
"""

from __future__ import annotations

import ast
import functools
import operator
import re
from fractions import Fraction

from ..answers import Distractor
from ..core import (
    GRADER_TOLERANCE,
    VerificationError,
    attempts,
    check,
    generator,
    join_terms,
    paren_negative,
    latex_answer,
    phrase,
    pick_distractors,
)


class _Short(Exception):
    """A draw that cannot give three distinct wrong options; the generator redraws."""


def _an(text: str) -> str:
    """'a $8' -> 'an $8': the article before an amount that is spoken with a vowel sound (8, 11, 18, 80-89)."""
    return re.sub(r"\b([aA]) (?=\$(?:11|18|8\d?)(?!\d))", lambda m: m.group(1) + "n ", text)


def _redraw(fn):
    @functools.wraps(fn)
    def wrapper(rng, *args, **kwargs):
        for _ in attempts(200, "a draw with enough distinct distractors"):
            try:
                return fn(rng, *args, **kwargs)
            except _Short:
                continue

    return wrapper


def _int_d(value, tag):
    return Distractor(str(int(value)), tag, int(value))


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _star(text: str) -> str:
    """The answer text with every implied product written out, so Python's parser can read it."""
    text = text.replace("^", "**")
    return re.sub(r"(?<=\d)(?=[a-z(])|(?<=\))(?=[a-z0-9(])|(?<=[a-z])(?=\()", "*", text)


def _evaluate(node, var: str, value: Fraction) -> Fraction:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, var, value)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return Fraction(node.value)
    if isinstance(node, ast.Name) and node.id == var:
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand, var, value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left, var, value), _evaluate(node.right, var, value)
        if isinstance(node.op, ast.Pow):
            check(right.denominator == 1, "only whole-number powers are written in these rules")
            return left ** int(right)
        return _OPERATORS[type(node.op)](left, right)
    raise VerificationError(f"unexpected syntax in a rule: {ast.dump(node)}")


def _at(text: str, var: str, value) -> Fraction:
    """The expression `text` (one variable, written the way a learner types it) at `var = value`, exactly."""
    return _evaluate(ast.parse(_star(text), mode="eval"), var, Fraction(value))


_PROBES = (2, 3, 5, 7, 11)


def _same(left: str, right: str) -> bool:
    """Whether two one-variable expressions agree at every probe point (enough for a degree-two rule)."""
    letters = set(re.findall(r"[a-z]", left + right))
    if len(letters) != 1:
        return left == right
    (var,) = letters
    return all(_at(left, var, x) == _at(right, var, x) for x in _PROBES)


def _expr_distractors(answer: str, candidates, n: int = 3):
    """`candidates` is a list of (text, tag). Keep those that are a different expression from the answer and from
    each other, as the grader sees it, and wrap them as tagged distractors."""
    kept: list[Distractor] = []
    for text, tag in candidates:
        if _same(answer, text) or any(_same(d.text, text) for d in kept):
            continue
        kept.append(Distractor(text, tag))
        if len(kept) == n:
            break
    if len(kept) < n:
        raise _Short
    return kept


def _verify_expression(text: str, symbol: str, value: int, total: int):
    """The expression takes the total when the variable takes the value the story was built from."""
    check(_at(text, symbol, value) == total, "the expression does not give the stated total")


# -- 1. translate a word problem into an equation ----------------------------------------------------------------

# (story with {a} {b} {T}, variable letter, what the variable counts)
_ONE_STEP_MULT = (
    ("Each ticket to the school fair costs ${b}. Priya buys some tickets and pays ${T} in all.", "t", "tickets"),
    ("Each pack holds {b} pencils. A teacher opens some packs and has {T} pencils altogether.", "p", "packs"),
    ("Each row of the hall has {b} seats. The rows that are filled hold {T} students.", "r", "rows"),
    ("A bakery puts {b} rolls in every bag. It fills some bags with {T} rolls in total.", "g", "bags"),
)
_ONE_STEP_ADD = (
    ("Liam had some football cards. After he was given {b} more, he had {T} cards.", "c", "cards Liam had at first"),
    ("A train had some passengers. {b} more got on, and there were then {T} passengers.", "p", "passengers at first"),
    ("Mia had some savings. After she earned ${b} babysitting, she had ${T}.", "s", "dollars Mia had at first"),
)
_ONE_STEP_SUB = (
    ("A tank held some water. After {b} litres were used, {T} litres were left.", "w", "litres at first"),
    ("Jo had some stickers. After giving away {b}, Jo had {T} left.", "s", "stickers at first"),
    ("A shelf held some books. After {b} were borrowed, {T} were left.", "k", "books at first"),
)
_ONE_STEP_DIV = (
    ("A ribbon was cut into {b} equal pieces. Each piece was {T} cm long.", "l", "length of the ribbon in cm"),
    ("A prize of money was shared equally among {b} winners. Each winner received ${T}.", "m", "dollars in the prize"),
    ("A bag of sand was shared equally between {b} garden beds, with {T} kg going to each.", "k", "kg in the bag"),
)
_TWO_STEP_FIXED = (
    (
        "A function hall charges a ${a} booking fee plus ${b} for each guest. The total bill is ${T}.",
        "g",
        "guests",
        "guest",
    ),
    (
        "A taxi charges a ${a} flag fall plus ${b} for each kilometre. A trip costs ${T}.",
        "d",
        "kilometres",
        "kilometre",
    ),
    (
        "A phone plan costs ${a} a month plus ${b} for each extra GB. This month's bill was ${T}.",
        "n",
        "extra GB",
        "extra GB",
    ),
    ("A gym charges a ${a} joining fee plus ${b} for each visit. Jess has paid ${T} in total.", "v", "visits", "visit"),
    ("A tutor charges ${a} to travel to the house plus ${b} for each hour. The bill was ${T}.", "h", "hours", "hour"),
)
_TWO_STEP_VOUCHER = (
    (
        "Each ticket to a show costs ${b}. Maya buys some tickets and uses a ${a} voucher, so she pays ${T}.",
        "t",
        "tickets",
    ),
    (
        "Notebooks cost ${b} each. With a ${a} coupon taken off the bill, Sam pays ${T} for some notebooks.",
        "n",
        "notebooks",
    ),
    ("Caps cost ${b} each. A ${a} discount is taken off Tom's order, leaving ${T} to pay.", "c", "caps"),
)


@generator(
    "fpm_translate_equation",
    level=6,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
@_redraw
def fpm_translate_equation(rng, tier: str = "easy"):
    if tier == "easy":
        shape = rng.choice(["mult", "add", "sub", "div"])
        story_fmt, v, what = rng.choice(
            {"mult": _ONE_STEP_MULT, "add": _ONE_STEP_ADD, "sub": _ONE_STEP_SUB, "div": _ONE_STEP_DIV}[shape]
        )
        b = rng.randint(3, 9) if shape != "div" else rng.randint(3, 8)
        x0 = rng.randint(4, 14) if shape in ("mult", "div") else rng.randint(12, 60)
        if shape == "mult":
            total, answer = b * x0, f"{b}{v}"
            cands = [
                (f"{v} + {b}", "uses-addition-for-a-per-unit-rate"),
                (f"{v}/{b}", "divides-where-multiplying-is-needed"),
                (f"{v} - {b}", "uses-addition-for-a-per-unit-rate"),
                (f"{total}{v}", "puts-the-total-in-the-expression"),
            ]
        elif shape == "add":
            total, answer = x0 + b, f"{v} + {b}"
            cands = [
                (f"{v} - {b}", "uses-the-wrong-operation-for-the-change"),
                (f"{b}{v}", "multiplies-where-adding-is-needed"),
                (f"{b} - {v}", "uses-the-wrong-operation-for-the-change"),
                (f"{v} + {total}", "puts-the-total-in-the-expression"),
            ]
        elif shape == "sub":
            total, answer = x0 - b, f"{v} - {b}"
            cands = [
                (f"{b} - {v}", "reverses-the-subtraction"),
                (f"{v} + {b}", "uses-the-wrong-operation-for-the-change"),
                (f"{b}{v}", "multiplies-where-subtracting-is-needed"),
                (f"{v} - {total}", "puts-the-total-in-the-expression"),
            ]
        else:
            total, answer = x0, f"{v}/{b}"
            x0 = total * b
            cands = [
                (f"{b}/{v}", "reverses-the-division"),
                (f"{b}{v}", "multiplies-instead-of-dividing"),
                (f"{v} - {b}", "uses-the-wrong-operation-for-the-change"),
                (f"{v} + {b}", "uses-the-wrong-operation-for-the-change"),
            ]
        story = _an(story_fmt.format(a="", b=b, T=total))
        _verify_expression(answer, v, x0, total)
        working = [
            f"Name the unknown {v}, the number of {what}.",
            f"The left-hand side is {answer}, and it equals {total}.",
        ]
    elif tier == "medium":
        kind = rng.choice(["fixed", "voucher", "share"])
        if kind == "fixed":
            story_fmt, v, what, _single = rng.choice(_TWO_STEP_FIXED)
            a, b, x0 = rng.randint(8, 40), rng.randint(2, 9), rng.randint(5, 30)
            total, answer = a + b * x0, join_terms([(b, v), (a, "")])
            cands = [
                (join_terms([(a, v), (b, "")]), "swaps-the-fixed-charge-and-the-rate"),
                (f"{a + b}{v}", "adds-the-fixed-charge-and-the-rate-first"),
                (f"{b}{v}", "leaves-out-the-fixed-charge"),
                (join_terms([(b, v), (-a, "")]), "uses-the-wrong-operation-for-the-change"),
            ]
        elif kind == "voucher":
            story_fmt, v, what = rng.choice(_TWO_STEP_VOUCHER)
            a, b, x0 = rng.randint(5, 20), rng.randint(6, 25), rng.randint(3, 12)
            total, answer = b * x0 - a, join_terms([(b, v), (-a, "")])
            if total <= 0:
                raise _Short
            cands = [
                (join_terms([(a, ""), (-b, v)]), "reverses-the-subtraction"),
                (join_terms([(b, v), (a, "")]), "uses-the-wrong-operation-for-the-change"),
                (f"{b}{v}", "leaves-out-the-voucher"),
                (f"{b - a}{v}", "takes-the-voucher-off-the-price-not-the-bill"),
            ]
        else:
            k, a = rng.randint(2, 6), rng.randint(2, 9)
            x0 = k * rng.randint(3, 12)
            v, what = rng.choice([("n", "the number"), ("m", "the number")])
            story_fmt = "Tom thinks of a number, divides it by {k}, then subtracts {a}. His answer is {T}."
            total, answer = x0 // k - a, f"{v}/{k} - {a}"
            cands = [
                (f"{v} - {a}/{k}", "applies-the-division-to-only-one-part"),
                (f"{k}{v} - {a}", "multiplies-instead-of-dividing"),
                (f"({v} - {a})/{k}", "reverses-the-order-of-the-steps"),
                (f"{v}/{k} + {a}", "uses-the-wrong-operation-for-the-change"),
            ]
            if total <= 0:
                raise _Short
        story = _an(story_fmt.format(a=a, b=b if kind != "share" else 0, T=total, k=k if kind == "share" else 0))
        _verify_expression(answer, v, x0, total)
        working = [
            f"Let {v} be the number of {what}." if kind != "share" else f"Let {v} be Tom's number.",
            f"Left-hand side: {answer}. It equals {total}.",
        ]
    else:
        kind = rng.choice(["tickets", "bracket", "perimeter", "consecutive"])
        if kind == "tickets":
            p, q, k = rng.randint(8, 18), rng.randint(4, 9), rng.randint(2, 6)
            x0 = rng.randint(3, 12)
            total = p * x0 + q * (x0 + k)
            v = "a"
            story = (
                f"Adult tickets cost ${p} and child tickets cost ${q}. A family buys adult tickets (a of them) and {k} more "
                f"child tickets than adult tickets. The total cost is ${total}."
            )
            answer = f"{p}a + {q}(a + {k})"
            cands = [
                (f"{p}a + {q * k}", "treats-the-extra-as-the-whole-number"),
                (f"{p}a + {q}a + {k}", "adds-the-extra-count-instead-of-pricing-it"),
                (f"({p} + {q})(a + {k})", "gives-both-kinds-the-same-count"),
                (f"{p}a + {q}(a - {k})", "uses-the-wrong-operation-for-the-change"),
            ]
            what = "adult tickets"
        elif kind == "bracket":
            k, m = rng.randint(2, 9), rng.randint(2, 7)
            x0 = rng.randint(3, 14)
            total = m * (x0 + k)
            v = "n"
            story = f"Ben thinks of a number, adds {k}, then multiplies the result by {m}. He gets {total}."
            answer = f"{m}(n + {k})"
            cands = [
                (f"{m}n + {k}", "leaves-out-the-bracket"),
                (f"n + {m * k}", "applies-the-multiplication-to-only-one-part"),
                (f"(n + {k})/{m}", "divides-instead-of-multiplying"),
                (f"{m} + n + {k}", "operation-from-keyword-alone"),
            ]
            what = "the number"
        elif kind == "perimeter":
            k = rng.randint(2, 9)
            x0 = rng.randint(4, 15)
            total = 2 * x0 + 2 * (x0 + k)
            v = "w"
            story = f"A rectangle is {k} cm longer than it is wide. Its width is w cm, and its perimeter is {total} cm."
            answer = f"2w + 2(w + {k})"
            cands = [
                (f"w + (w + {k})", "uses-half-the-perimeter"),
                (f"2w + {2 * k}", "treats-the-extra-as-the-whole-length"),
                (f"w(w + {k})", "uses-the-area-formula-for-perimeter"),
                (f"2(w + {k})", "leaves-out-the-width-sides"),
            ]
            what = "the width"
        else:
            x0 = rng.randint(5, 30)
            total = x0 + (x0 + 1) + (x0 + 2)
            v = "n"
            story = f"The sum of three consecutive whole numbers is {total}. The smallest of the three is n."
            answer = "n + (n + 1) + (n + 2)"
            cands = [
                ("3n", "treats-consecutive-numbers-as-equal"),
                ("n + (n + 1)", "leaves-a-number-out"),
                ("n + 1 + 2", "adds-the-steps-to-one-number"),
                ("n(n + 1)(n + 2)", "multiplies-where-adding-is-needed"),
            ]
            what = "the smallest number"
        _verify_expression(answer, v, x0, total)
        working = [
            f"Let {v} be {what}.",
            f"Write the other quantity in terms of {v}, then combine: {answer} = {total}.",
        ]
    distractors = _expr_distractors(answer, cands)
    ask = phrase(
        rng,
        (
            "Write an equation for this situation. The right-hand side is {T}; type the expression in {v} for the left-hand side.",
            "An equation for this situation has {T} on the right-hand side. Type the expression in {v} that belongs on the left-hand side.",
            "Complete the equation ____ = {T} for this situation by typing the expression in {v} for the blank.",
        ),
        T=total,
        v=v,
    )
    return dict(
        text=f"{story} {ask}",
        answer_kind="expression",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer),
        working=working,
        distractors=distractors,
    )


@generator("fpm_translate_equation_exam", level=7, presets={"exam": {}})
@_redraw
def fpm_translate_equation_exam(rng):
    story_fmt, v, what, single = rng.choice(_TWO_STEP_FIXED)
    a, b, x0 = rng.randint(10, 60), rng.randint(3, 9), rng.randint(8, 40)
    total = a + b * x0
    c = rng.randint(2, b - 1)
    rival = c * x0
    saving = total - rival
    check(saving > 0, "the cheaper rate must save money")
    answer = join_terms([(b, v), (a, "")])
    _verify_expression(answer, v, x0, total)
    story = _an(story_fmt.format(a=a, b=b, T=total))
    stem = f"{story} A second supplier charges ${c} for each {single}, with no other charge."
    a_d = _expr_distractors(
        answer,
        [
            (join_terms([(a, v), (b, "")]), "swaps-the-fixed-charge-and-the-rate"),
            (f"{a + b}{v}", "adds-the-fixed-charge-and-the-rate-first"),
            (f"{b}{v}", "leaves-out-the-fixed-charge"),
            (join_terms([(b, v), (-a, "")]), "uses-the-wrong-operation-for-the-change"),
        ],
    )
    b_d = pick_distractors(
        str(x0),
        [
            _int_d(total / b if total % b == 0 else (total + a) // b, "ignores-the-fixed-charge-when-solving"),
            _int_d((total + a) // b, "adds-the-fixed-charge-instead-of-taking-it-off"),
            _int_d(total - a - b, "subtracts-the-rate-instead-of-dividing"),
            _int_d(total - a, "forgets-to-divide-by-the-rate"),
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    c_d = pick_distractors(
        str(rival),
        [
            _int_d(c * total // b, "uses-the-total-in-place-of-the-quantity"),
            _int_d(c + x0, "adds-where-multiplying-is-needed"),
            _int_d(c * (x0 + a // b), "ignores-the-fixed-charge-when-solving"),
            _int_d(total - c, "adds-where-multiplying-is-needed"),
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    d_d = pick_distractors(
        str(saving),
        [
            _int_d(rival, "reports-the-other-total-not-the-difference"),
            _int_d(total + rival, "adds-instead-of-subtracting"),
            _int_d(abs(a - rival), "compares-the-wrong-quantities"),
            _int_d(a, "reports-the-fixed-charge-as-the-saving"),
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    if any(len(x) < 3 for x in (b_d, c_d, d_d)):
        raise _Short
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Let {v} be the number of {what}. Write an expression in {v} for the first supplier's bill, in dollars.",
                answer_kind="expression",
                answer=answer,
                answer_text=answer,
                answer_latex=latex_answer(answer),
                marks=2,
                working=[f"Fixed charge ${a} plus ${b} for each: {answer}."],
                distractors=a_d,
            ),
            dict(
                prompt=f"Write and solve an equation to find the number of {what}.",
                answer_kind="number",
                answer=x0,
                answer_text=str(x0),
                answer_latex=latex_answer(str(x0)),
                marks=2,
                sensible_range=(0, None),
                working=[f"{answer} = {total}", f"{b}{v} = {total - a}", f"{v} = {x0}."],
                distractors=b_d,
            ),
            dict(
                prompt="Using your answer to (b), find what the second supplier would have charged, in dollars.",
                answer_kind="number",
                answer=rival,
                answer_text=str(rival),
                answer_latex=latex_answer(str(rival)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{c}*part_b",
                working=[f"{c} × {x0} = {rival}."],
                distractors=c_d,
            ),
            dict(
                prompt="How many dollars would have been saved by using the second supplier?",
                answer_kind="number",
                answer=saving,
                answer_text=str(saving),
                answer_latex=latex_answer(str(saving)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{total} - part_c",
                working=[f"{total} - {rival} = {saving}."],
                distractors=d_d,
            ),
        ],
    )


# -- 2. linear models ---------------------------------------------------------------------------------------------

# qty: what is modelled; unit; x: the input's name; xs: plural; var: its letter; sign: +1 grows, -1 shrinks
_LM_CONTEXTS = (
    dict(
        qty="the height of a sunflower",
        noun="the height",
        unit="cm",
        x="week",
        xs="weeks",
        var="t",
        y="h",
        sign=1,
        init=(8, 40),
        rate=(3, 9),
    ),
    dict(
        qty="the balance of a savings account",
        noun="the balance",
        unit="dollars",
        x="week",
        xs="weeks",
        var="t",
        y="m",
        sign=1,
        init=(50, 300),
        rate=(10, 45),
    ),
    dict(
        qty="the distance a cyclist has ridden",
        noun="the distance ridden",
        unit="km",
        x="hour",
        xs="hours",
        var="t",
        y="d",
        sign=1,
        init=(4, 30),
        rate=(12, 22),
    ),
    dict(
        qty="the volume of water in a tank",
        noun="the volume",
        unit="litres",
        x="minute",
        xs="minutes",
        var="t",
        y="v",
        sign=-1,
        init=(400, 900),
        rate=(12, 40),
    ),
    dict(
        qty="the length of a burning candle",
        noun="the length",
        unit="cm",
        x="hour",
        xs="hours",
        var="t",
        y="l",
        sign=-1,
        init=(20, 32),
        rate=(2, 4),
    ),
    dict(
        qty="the distance still to walk on a hike",
        noun="the distance left",
        unit="km",
        x="hour",
        xs="hours",
        var="t",
        y="d",
        sign=-1,
        init=(18, 30),
        rate=(2, 5),
    ),
)
_PLAN_CONTEXTS = (
    (
        "A bike shop charges ${a} to join, then ${b} for each hour of hire. A second shop charges ${c} for each hour with no joining fee.",
        "hours",
    ),
    (
        "Phone plan A costs ${a} to connect, then ${b} for each GB. Plan B charges ${c} for each GB with no connection fee.",
        "GB",
    ),
    (
        "A hall hire has a ${a} booking fee plus ${b} for each guest. A caterer-run hall charges ${c} for each guest and no booking fee.",
        "guests",
    ),
)


def _rule(ctx, c, m):
    """The rule text, such as 'h = 12 + 5t' or 'v = 900 - 40t'."""
    return f"{ctx['y']} = {join_terms([(c, ''), (m, ctx['var'])])}"


def _draw_model(rng, ctx):
    lo, hi = ctx["init"]
    c = rng.randint(lo, hi)
    m = ctx["sign"] * rng.randint(*ctx["rate"])
    return c, m


def _num_distractors(answer: int, candidates, positive=True):
    out = pick_distractors(
        str(answer),
        [_int_d(v, tag) for v, tag in candidates if (v > 0 or not positive) and v != answer],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    if len(out) < 3:
        raise _Short
    return out


def _num(value: int, working, distractors, text):
    return dict(
        text=text,
        answer_kind="number",
        answer=int(value),
        answer_text=str(int(value)),
        answer_latex=latex_answer(str(int(value))),
        working=working,
        distractors=distractors,
    )


@generator(
    "fpm_linear_model",
    level=6,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
@_redraw
def fpm_linear_model(rng, tier: str = "easy"):
    if tier == "hard":
        return _lm_hard(rng)
    ctx = rng.choice(_LM_CONTEXTS)
    c, m = _draw_model(rng, ctx)
    if tier == "easy":
        x0 = rng.randint(3, 12)
        y0 = c + m * x0
        if y0 <= 0:
            raise _Short
        text = (
            f"{ctx['qty'].capitalize()} is modelled by the rule {_rule(ctx, c, m)}, where {ctx['y']} is {ctx['noun']} in "
            f"{ctx['unit']} and {ctx['var']} is the number of {ctx['xs']}. Use the model to predict {ctx['qty']} "
            f"after {x0} {ctx['xs']}."
        )
        distractors = _num_distractors(
            y0,
            [
                (c * x0 + abs(m) if m > 0 else c * x0 - abs(m), "swaps-the-starting-value-and-the-rate"),
                ((c + m) * x0, "adds-the-starting-value-and-the-rate-first"),
                (abs(m) * x0, "leaves-out-the-starting-value"),
                (c - m * x0, "uses-the-wrong-operation-for-the-change"),
                (c + m * (x0 + 1), "counts-one-step-too-many"),
            ],
        )
        return _num(
            y0,
            [f"{ctx['y']} = {c} {'+' if m > 0 else '-'} {abs(m)} × {x0}", f"{ctx['y']} = {y0} {ctx['unit']}."],
            distractors,
            text,
        )
    # medium: build the model from two readings
    for _ in range(20):
        x1 = rng.randint(1, 4)
        x2 = x1 + rng.randint(2, 5)
        x3 = x2 + rng.randint(2, 6)
        y1, y2, y3 = c + m * x1, c + m * x2, c + m * x3
        if min(y1, y2, y3) > 0 and y1 != y3:
            break
    else:
        raise _Short
    rate = Fraction(y2 - y1, x2 - x1)
    check(rate == m, "the rate from the readings is not the model's rate")
    check(Fraction(y1) + rate * (x3 - x1) == y3, "prediction differs by the point-slope route")
    text = (
        f"{ctx['qty'].capitalize()} was {y1} {ctx['unit']} after {x1} {ctx['xs'] if x1 != 1 else ctx['x']} and "
        f"{y2} {ctx['unit']} after {x2} {ctx['xs']}. Assuming it changes by the same amount each {ctx['x']}, "
        f"predict {ctx['qty']} after {x3} {ctx['xs']}."
    )
    prop = round(Fraction(y2, x2) * x3)
    distractors = _num_distractors(
        y3,
        [
            (prop, "assumes-the-quantity-is-proportional-to-the-input"),
            (y2 + (y2 - y1), "adds-the-whole-change-not-the-rate"),
            (y2 + m, "applies-the-rate-for-only-one-step"),
            (y2 + (x3 - x2) * (y2 - y1), "uses-the-change-between-readings-as-the-rate"),
            (y2 + abs(m) * (x3 - x2) if m < 0 else y2 - m * (x3 - x2), "uses-the-wrong-operation-for-the-change"),
        ],
    )
    return _num(
        y3,
        [
            f"Rate = ({y2} - {y1}) ÷ ({x2} - {x1}) = {m} {ctx['unit']} per {ctx['x']}.",
            f"After {x3} {ctx['xs']}: {y2} + {paren_negative(m)} × ({x3} - {x2}) = {y3} {ctx['unit']}.",
        ],
        distractors,
        text,
    )


def _lm_hard(rng):
    kind = rng.choice(["target", "plans", "valid"])
    if kind == "target":
        ctx = rng.choice(_LM_CONTEXTS)
        for _ in attempts(500, "readings and a target that stay positive"):
            c, m = _draw_model(rng, ctx)
            x1 = rng.randint(1, 3)
            x2 = x1 + rng.randint(2, 4)
            x_t = x2 + rng.randint(3, 8)
            y1, y2, target = c + m * x1, c + m * x2, c + m * x_t
            if min(y1, y2, target) > 0:
                break
        check(Fraction(target - y2, m) + x2 == x_t, "second route to the input differs")
        text = (
            f"{ctx['qty'].capitalize()} was {y1} {ctx['unit']} after {x1} {ctx['xs'] if x1 != 1 else ctx['x']} and "
            f"{y2} {ctx['unit']} after {x2} {ctx['xs']}, and it changes by the same amount each {ctx['x']}. "
            f"After how many {ctx['xs']} in all will it be {target} {ctx['unit']}?"
        )
        distractors = _num_distractors(
            x_t,
            [
                (x_t - x2, "reports-the-extra-time-since-the-last-reading"),
                (round(Fraction(target, abs(m))), "ignores-the-starting-value"),
                (
                    round(Fraction(target - c, abs(m))) + 1 if (target - c) % m == 0 else x_t + 1,
                    "counts-one-step-too-many",
                ),
                (x_t - x1, "reports-the-extra-time-since-the-first-reading"),
            ],
        )
        return _num(
            x_t,
            [
                f"Rate = ({y2} - {y1}) ÷ ({x2} - {x1}) = {m}.",
                f"{c} {'+' if m > 0 else '-'} {abs(m)}{ctx['var']} = {target}, so {ctx['var']} = {x_t}.",
            ],
            distractors,
            text,
        )
    if kind == "plans":
        story, unit = rng.choice(_PLAN_CONTEXTS)
        for _ in attempts(500, "two plans that agree at a whole number"):
            b = rng.randint(2, 8)
            c = b + rng.randint(2, 7)
            n = rng.randint(4, 25)
            a = (c - b) * n
            if 20 <= a <= 400:
                break
        check(a + b * n == c * n, "the plans do not agree at n")
        text = f"{_an(story.format(a=a, b=b, c=c))} For how many {unit} do the two cost exactly the same?"
        distractors = _num_distractors(
            n,
            [
                (round(Fraction(a, c)), "ignores-the-rate-in-the-first-plan"),
                (round(Fraction(a, b)), "divides-by-one-plans-rate-only"),
                (round(Fraction(a, c + b)), "adds-the-rates-instead-of-subtracting"),
                (a + (c - b), "adds-where-dividing-is-needed"),
                (a * (c - b), "multiplies-where-dividing-is-needed"),
            ],
        )
        return _num(
            n,
            [f"{a} + {b}n = {c}n, so {a} = {c - b}n.", f"n = {a} ÷ {c - b} = {n}."],
            distractors,
            text,
        )
    ctx = rng.choice([c for c in _LM_CONTEXTS if c["sign"] < 0])
    for _ in attempts(500, "a decreasing model that does not reach zero at a whole number"):
        c, m = _draw_model(rng, ctx)
        if c % abs(m):
            break
    last = c // abs(m)
    check(c + m * last >= 0 and c + m * (last + 1) < 0, "last sensible whole input is wrong")
    text = (
        f"{ctx['qty'].capitalize()} is modelled by {_rule(ctx, c, m)}, where {ctx['y']} is in {ctx['unit']} and "
        f"{ctx['var']} is the number of {ctx['xs']}. The model only makes sense while {ctx['y']} is not negative. "
        f"What is the largest whole number of {ctx['xs']} for which the model can be used?"
    )
    distractors = _num_distractors(
        last,
        [
            (last + 1, "rounds-up-where-the-model-stops-making-sense"),
            (c - abs(m), "subtracts-the-rate-from-the-starting-value"),
            (round(Fraction(c, abs(m)) * 10), "slips-a-decimal-place"),
            (last - 1, "stops-one-early"),
            (c // abs(m) + 2, "rounds-up-where-the-model-stops-making-sense"),
        ],
    )
    return _num(
        last,
        [
            f"{ctx['y']} ≥ 0 means {c} - {abs(m)}{ctx['var']} ≥ 0, so {ctx['var']} ≤ {c} ÷ {abs(m)} = {c / abs(m):.2f}.",
            f"The largest whole number is {last}.",
        ],
        distractors,
        text,
    )


@generator("fpm_linear_model_exam", level=7, presets={"exam": {}})
@_redraw
def fpm_linear_model_exam(rng):
    ctx = rng.choice(_LM_CONTEXTS)
    for _ in attempts(500, "readings, a prediction and a target that stay positive"):
        c, m = _draw_model(rng, ctx)
        x1 = rng.randint(1, 3)
        x2 = x1 + rng.randint(2, 4)
        x0 = x2 + rng.randint(2, 5)
        x_t = x0 + rng.randint(2, 6)
        y1, y2, y0, target = c + m * x1, c + m * x2, c + m * x0, c + m * x_t
        if min(y1, y2, y0, target) > 0:
            break
    check(Fraction(y2 - y1, x2 - x1) == m and y1 - m * x1 == c, "model differs from the readings")
    check(Fraction(target - c, m) == x_t, "target input differs by the second route")
    unit, x, xs = ctx["unit"], ctx["x"], ctx["xs"]
    stem = (
        f"{ctx['qty'].capitalize()} was {y1} {unit} after {x1} {xs if x1 != 1 else x} and {y2} {unit} after {x2} {xs}. "
        f"It changes by the same amount each {x}. Let {ctx['y']} be {ctx['noun']} in {unit} after {ctx['var']} {xs}."
    )
    a_d = _num_distractors(
        m,
        [
            (y2 - y1, "uses-the-change-between-readings-as-the-rate"),
            (round(Fraction(y2, x2)), "assumes-the-quantity-is-proportional-to-the-input"),
            (-m, "gets-the-direction-of-change-wrong"),
            (round(Fraction(x2 - x1, y2 - y1) * 10), "inverts-the-rate"),
            (m + 1, "miscounts-by-one"),
        ],
        positive=False,
    )
    b_d = _num_distractors(
        c,
        [
            (y1, "reports-the-first-reading-as-the-starting-value"),
            (y1 + m * x1, "uses-the-wrong-operation-for-the-change"),
            (y2 - y1, "reports-the-change-as-the-starting-value"),
            (round(Fraction(y1, x1)), "assumes-the-quantity-is-proportional-to-the-input"),
            (c + m, "applies-the-rate-for-only-one-step"),
        ],
    )
    c_d = _num_distractors(
        y0,
        [
            (c * x0 + abs(m), "swaps-the-starting-value-and-the-rate"),
            (abs(m) * x0, "leaves-out-the-starting-value"),
            (c - m * x0, "uses-the-wrong-operation-for-the-change"),
            ((c + m) * x0, "adds-the-starting-value-and-the-rate-first"),
        ],
    )
    d_d = _num_distractors(
        x_t,
        [
            (round(Fraction(target, abs(m))), "ignores-the-starting-value"),
            (x_t - x0, "reports-the-extra-time-since-the-last-reading"),
            (x_t + 1, "counts-one-step-too-many"),
        ],
    )
    sign_note = " Use a negative number for a decrease." if m < 0 else ""
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"By how much does {ctx['noun']} change each {x}, in {unit}?{sign_note}",
                answer_kind="number",
                answer=m,
                answer_text=str(m),
                answer_latex=latex_answer(str(m)),
                marks=1,
                working=[f"({y2} - {y1}) ÷ ({x2} - {x1}) = {m}."],
                distractors=a_d,
            ),
            dict(
                prompt=f"Use your answer to (a) to find {ctx['noun']} at the start, when {ctx['var']} = 0, in {unit}.",
                answer_kind="number",
                answer=c,
                answer_text=str(c),
                answer_latex=latex_answer(str(c)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{y1} - part_a*{x1}",
                working=[f"{y1} - {paren_negative(m)} × {x1} = {c}."],
                distractors=b_d,
            ),
            dict(
                prompt=f"Use your model to predict {ctx['noun']} after {x0} {xs}, in {unit}.",
                answer_kind="number",
                answer=y0,
                answer_text=str(y0),
                answer_latex=latex_answer(str(y0)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_b + part_a*{x0}",
                working=[f"{c} + {paren_negative(m)} × {x0} = {y0}."],
                distractors=c_d,
            ),
            dict(
                prompt=f"After how many {xs} in all will {ctx['noun']} be {target} {unit}?",
                answer_kind="number",
                answer=x_t,
                answer_text=str(x_t),
                answer_latex=latex_answer(str(x_t)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"({target} - part_b)/part_a",
                working=[f"{join_terms([(c, ''), (m, ctx['var'])])} = {target}, so {ctx['var']} = {x_t}."],
                distractors=d_d,
            ),
        ],
    )


# -- 3. model relationships with functions -----------------------------------------------------------------------

_PROP_CONTEXTS = (
    ("n", "the number of packs", "C", "the cost in dollars", "packs"),
    ("h", "the hours worked", "P", "the pay in dollars", "hours"),
    ("q", "the litres of fuel", "C", "the cost in dollars", "litres"),
    ("b", "the number of boxes", "m", "the mass in kg", "boxes"),
)
_LIN_CONTEXTS = (
    ("h", "the hours of work", "C", "the charge in dollars", "A plumber's charge"),
    ("d", "the distance in km", "F", "the fare in dollars", "A taxi fare"),
    ("n", "the number of guests", "B", "the bill in dollars", "A function hall's bill"),
    ("w", "the weeks of membership", "T", "the total paid in dollars", "A gym's total charge"),
)
_INVERSE_CONTEXTS = (
    ("w", "the number of workers", "d", "the days to finish the job", "A job is shared equally between workers."),
    ("s", "the speed in km/h", "t", "the time in hours", "A fixed distance is travelled at a steady speed."),
    ("p", "the number of people", "a", "the dollars each person receives", "A fixed prize is shared equally."),
    ("n", "the number of taps", "m", "the minutes to fill the pool", "Taps of the same size fill a pool."),
)


def _table(xv, yv, xn, yn):
    return f"{xn} = {', '.join(map(str, xv))} gives {yn} = {', '.join(map(str, yv))}"


def _rule_prompt(xn, xwhat, yn, ywhat, rng):
    return phrase(
        rng,
        (
            "Write a rule for {yn} in terms of {xn}.",
            "Find a rule that gives {yn} from {xn}.",
            "Write the rule connecting {yn} and {xn}.",
        ),
        xn=xn,
        yn=yn,
    )


@generator(
    "fpm_function_model",
    level=6,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
@_redraw
def fpm_function_model(rng, tier: str = "easy"):
    if tier == "easy":
        xn, xw, yn, yw, things = rng.choice(_PROP_CONTEXTS)
        k = rng.randint(3, 12)
        xv = rng.choice([[1, 2, 3, 4], [2, 3, 4, 5], [1, 3, 5, 6]])
        yv = [k * x for x in xv]
        answer = f"{k}{xn}"
        cands = [
            (f"{xn} + {k}", "adds-the-first-output-to-the-input"),
            (f"{xn} + {yv[0] - xv[0]}", "fits-only-the-first-row-of-the-table"),
            (f"{xn}/{k}", "divides-where-multiplying-is-needed"),
            (f"{k + 1}{xn}", "miscounts-the-multiplier"),
            (f"{k}{xn} + {k}", "adds-a-fixed-amount-to-a-proportional-rule"),
        ]
        stem = f"The table pairs {xw} with {yw}: {_table(xv, yv, xn, yn)}."
    elif tier == "medium":
        xn, xw, yn, yw, who = rng.choice(_LIN_CONTEXTS)
        a, b = rng.randint(6, 40), rng.randint(2, 9)
        xv = rng.choice([[2, 3, 5, 8], [1, 4, 6, 9], [2, 5, 7, 10], [3, 4, 7, 10]])
        yv = [a + b * x for x in xv]
        answer = join_terms([(b, xn), (a, "")])
        gap = xv[1] - xv[0]
        cands = [
            (f"{b}{xn}", "leaves-out-the-fixed-amount"),
            (join_terms([(a, xn), (b, "")]), "swaps-the-fixed-amount-and-the-rate"),
            (join_terms([(b * gap, xn), (a, "")]), "uses-the-change-between-readings-as-the-rate"),
            (f"{a + b}{xn}", "adds-the-fixed-amount-and-the-rate-first"),
            (join_terms([(b, xn), (-a, "")]), "uses-the-wrong-operation-for-the-change"),
        ]
        stem = (
            f"{who} depends on {xw}. The table pairs the values: {_table(xv, yv, xn, yn)}, "
            f"where {yn} is {yw}. The relationship is linear."
        )
    else:
        kind = rng.choice(["inverse", "square", "falling"])
        if kind == "inverse":
            xn, xw, yn, yw, note = rng.choice(_INVERSE_CONTEXTS)
            k = rng.choice([24, 36, 48, 60, 72, 120])
            xv = [x for x in (2, 3, 4, 6, 8, 12) if k % x == 0][:4]
            yv = [k // x for x in xv]
            answer = f"{k}/{xn}"
            cands = [
                (f"{k} - {xn}", "assumes-the-sum-stays-fixed"),
                (f"{k}{xn}", "assumes-the-quantity-is-proportional-to-the-input"),
                (f"{xn}/{k}", "inverts-the-rule"),
                (f"{k}/({xn} + 1)", "shifts-the-input-before-dividing"),
            ]
            stem = f"{note} The table pairs {xw} with {yw}: {_table(xv, yv, xn, yn)}."
        elif kind == "square":
            k = rng.choice([2, 3, 4, 5, 6, 7, 8, 9])
            xv = rng.choice([[1, 2, 3, 4], [2, 3, 4, 5], [1, 3, 4, 6]])
            yv = [k * x * x for x in xv]
            answer = f"{k}s^2"
            cands = [
                (f"{k}s", "assumes-the-quantity-is-proportional-to-the-input"),
                (f"({k}s)^2", "squares-the-multiplier-too"),
                (f"{k}^s", "treats-the-input-as-the-power"),
                (f"{2 * k}s", "doubles-instead-of-squaring"),
            ]
            stem = (
                f"Square panels are painted at ${k} for each square metre. The table pairs the side length in metres "
                f"with the cost in dollars: {_table(xv, yv, 's', 'C')}."
            )
            xn, yn = "s", "C"
        else:
            c0 = rng.choice([60, 80, 100, 120, 150])
            m = rng.randint(3, 9)
            xv = [1, 2, 4, 5]
            yv = [c0 - m * x for x in xv]
            answer = join_terms([(c0, ""), (-m, "t")])
            cands = [
                (join_terms([(c0, ""), (m, "t")]), "gets-the-direction-of-change-wrong"),
                (join_terms([(m, "t"), (-c0, "")]), "swaps-which-quantity-is-subtracted"),
                (f"{m}t", "leaves-out-the-fixed-amount"),
                (join_terms([(-m, "t"), (c0 - m, "")]), "fits-only-the-first-row-of-the-table"),
            ]
            stem = (
                f"A tank drains steadily. The table pairs the minutes with the litres left: {_table(xv, yv, 't', 'V')}."
            )
            xn, yn = "t", "V"
        if kind == "inverse":
            check(all(x * y == k for x, y in zip(xv, yv)), "table is not an inverse relationship")
    check_values = {"easy": "proportional", "medium": "linear", "hard": "hard"}[tier]
    for x, y in zip(xv, yv):
        check(_at(answer, xn, x) == y, f"the {check_values} rule does not fit the table")
    distractors = _expr_distractors(answer, cands)
    rule_ask = _rule_prompt(xn, "", yn, "", rng)
    return dict(
        text=f"{stem} {rule_ask}",
        answer_kind="expression",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer),
        working=[
            f"Check the pattern in the table: {_table(xv, yv, xn, yn)}.",
            f"The rule that fits every pair is {yn} = {answer}.",
        ],
        distractors=distractors,
    )


@generator("fpm_function_model_exam", level=7, presets={"exam": {}})
@_redraw
def fpm_function_model_exam(rng):
    xn, xw, yn, yw, note = rng.choice(_INVERSE_CONTEXTS)
    k = rng.choice([48, 60, 72, 96, 120, 144])
    divisors = [x for x in range(2, 25) if k % x == 0]
    xv = divisors[:4]
    yv = [k // x for x in xv]
    rest = [x for x in divisors if x not in xv and k // x >= 2]
    x0 = rng.choice(rest[:4]) if len(rest) > 1 else rest[0]
    y0 = k // x0
    options = [y for y in range(2, 25) if k % y == 0 and y not in yv and y != y0 and k // y != y0 and y != x0]
    if not options:
        raise _Short
    target = rng.choice(options)
    x_t = k // target
    check(all(x * y == k for x, y in zip(xv, yv)) and x0 * y0 == k and target * x_t == k, "inverse relationship broken")
    stem = f"{note} The table pairs {xw} with {yw}: {_table(xv, yv, xn, yn)}."
    answer = f"{k}/{xn}"
    a_d = _num_distractors(
        k,
        [
            (xv[0] + yv[0], "assumes-the-sum-stays-fixed"),
            (yv[0], "reports-the-first-output-as-the-constant"),
            (yv[0] - yv[1], "reports-a-difference-as-the-constant"),
            (k + xv[0], "miscounts-by-one"),
        ],
    )
    b_d = _expr_distractors(
        answer,
        [
            (f"{k} - {xn}", "assumes-the-sum-stays-fixed"),
            (f"{k}{xn}", "assumes-the-quantity-is-proportional-to-the-input"),
            (f"{xn}/{k}", "inverts-the-rule"),
            (f"{k} + {xn}", "assumes-the-sum-stays-fixed"),
        ],
    )
    c_d = _num_distractors(
        y0,
        [
            (k * x0, "assumes-the-quantity-is-proportional-to-the-input"),
            (k - x0, "assumes-the-sum-stays-fixed"),
            (round(Fraction(x0, k) * 100), "inverts-the-rule"),
            (y0 + x0, "adds-where-dividing-is-needed"),
        ],
    )
    d_d = _num_distractors(
        x_t,
        [
            (k * target, "assumes-the-quantity-is-proportional-to-the-input"),
            (k - target, "assumes-the-sum-stays-fixed"),
            (target + x_t, "adds-where-dividing-is-needed"),
            (round(Fraction(target, k) * 100), "inverts-the-rule"),
        ],
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Multiply each {xn} by its {yn} from the table. What is the value every product gives?",
                answer_kind="number",
                answer=k,
                answer_text=str(k),
                answer_latex=latex_answer(str(k)),
                marks=1,
                sensible_range=(0, None),
                working=[" and ".join(f"{x} × {y} = {k}" for x, y in zip(xv[:2], yv[:2])) + "."],
                distractors=a_d,
            ),
            dict(
                prompt=_rule_prompt(xn, xw, yn, yw, rng),
                answer_kind="expression",
                answer=answer,
                answer_text=answer,
                answer_latex=latex_answer(answer),
                marks=2,
                working=[f"{xn} × {yn} = {k}, so {yn} = {k}/{xn}."],
                distractors=b_d,
            ),
            dict(
                prompt=f"Use your constant from (a) to find {yn} when {xn} = {x0}.",
                answer_kind="number",
                answer=y0,
                answer_text=str(y0),
                answer_latex=latex_answer(str(y0)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_a/{x0}",
                working=[f"{k} ÷ {x0} = {y0}."],
                distractors=c_d,
            ),
            dict(
                prompt=f"Use your constant from (a) to find {xn} when {yn} = {target}.",
                answer_kind="number",
                answer=x_t,
                answer_text=str(x_t),
                answer_latex=latex_answer(str(x_t)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a/{target}",
                working=[f"{k} ÷ {target} = {x_t}."],
                distractors=d_d,
            ),
        ],
    )
