"""Foundations of expressions, relations and sequences for Years 7-8, built as auto-marked multiple choice.

Topics (each with easy, medium and hard presets and an exam chain `exd_x_<topic>`):

* `w5_equivalent_expressions` (`recognise-equivalent-expressions`): which expression is equivalent to a bracketed or
  collected form, which is not, and what a match at one value does and does not show.
* `w5_variables_roles` (`independent-dependent-variables`): the independent and the dependent variable in a described
  situation, including situations where time is not the independent variable and where the axis a variable is drawn on
  is no guide.
* `w5_proportional_relations` (`proportional-vs-non-proportional`): tables, equations, situations and lines through two
  points, decided by whether y is a fixed multiple of x.
* `w5_inequality_symbols` (`inequality-symbols`): which statement is true, which number satisfies an inequality, words to
  symbols, and double inequalities with negative numbers.
* `w5_arithmetic_sequences` (`arithmetic-sequences`): which sequence has a constant difference, the term that continues
  one, and the unknown that keeps three terms in an arithmetic sequence.

Every question is built from values first and re-derived a second way: expressions are evaluated at several test values,
tables and lines are tested pair by pair, inequalities are tested number by number, and each distractor is checked to
be wrong by that same independent evaluation.
"""

from __future__ import annotations

import math
from fractions import Fraction as Fr

from ..core import check, generator, join_terms, paren_negative, phrase
from ._w5_kit import (
    NAMES,
    TooFewDistractors,
    enough,
    length_balanced,
    mc_question,
    mcq_cands,
    mcq_fields,
    number_cands,
    number_fields,
    part,
    retry_draw,
    rt,
    sub_fields,
    tier_pool,
)

_sub, _mc, _tiers = sub_fields, mc_question, tier_pool
_TEST_X = (-3, -1, 0, 2, 5, 7)


def _same(f, g) -> bool:
    return all(f(x) == g(x) for x in _TEST_X)


def _paren(text: str) -> str:
    return f"({text})"


# =====================================================================================================================
# equivalent expressions
# =====================================================================================================================


def _bracket_expr(a, b, c, d) -> tuple[str, object]:
    """The text and function of a(bx + c) + d."""
    inner = join_terms([(b, "x"), (c, "")])
    return join_terms([(a, _paren(inner)), (d, "")]), lambda x: a * (b * x + c) + d


def _forms(a, b, c, d):
    """Equivalent ways to write a(bx + c) + d, as (text, function)."""
    ab, rest = a * b, a * c + d
    f = lambda x: ab * x + rest  # noqa: E731
    out = [
        (join_terms([(ab, "x"), (rest, "")]), f),
        (join_terms([(d, ""), (a, _paren(join_terms([(b, "x"), (c, "")])))]), f),
        (join_terms([(ab, "x"), (a * c, ""), (d, "")]), f),
        (join_terms([(a, _paren(join_terms([(c, ""), (b, "x")]))), (d, "")]), f),
    ]
    g = math.gcd(abs(ab), abs(rest))
    if g > 1 and rest != 0:
        out.append((f"{g}{_paren(join_terms([(ab // g, 'x'), (rest // g, '')]))}", f))
    return out


def _near_misses(a, b, c, d):
    """Plausible but inequivalent forms with the misconception behind each."""
    ab = a * b
    return [
        (join_terms([(ab, "x"), (c, ""), (d, "")]), lambda x: ab * x + c + d, "w5q-distributes-to-first-term"),
        (join_terms([(ab, "x"), (a * c, ""), (-d, "")]), lambda x: ab * x + a * c - d, "w5q-sign-slip"),
        (join_terms([(a + b, "x"), (c + d, "")]), lambda x: (a + b) * x + c + d, "w5q-adds-instead-of-multiplies"),
        (join_terms([(ab, "x"), (a * c, "")]), lambda x: ab * x + a * c, "w5q-drops-a-term"),
        (join_terms([(ab, "x"), (a * c + d + 1, "")]), lambda x: ab * x + a * c + d + 1, "w5q-arithmetic-slip"),
    ]


def _eq_easy_expand(rng):
    a = rng.randint(2, 9)
    b = rng.choice((1, 1, 1, 2, 3))
    c = rng.choice([v for v in range(-9, 10) if v != 0])
    text, f = _bracket_expr(a, b, c, 0)
    forms = _forms(a, b, c, 0)
    key = forms[0][0]
    wrongs = []
    for t, g, tag in _near_misses(a, b, c, 0):
        if not _same(f, g) and not t.lstrip("-").isdigit():
            wrongs.append((t, tag))
    wrongs.append(
        (join_terms([(a, "x"), (b * c, "")]), "w5q-distributes-to-first-term")
        if b != 1
        else (join_terms([(a, "x"), (c, "")]), "w5q-distributes-to-first-term")
    )
    check(_same(f, forms[0][1]), "key is not equivalent")
    work = [
        f"Multiply every term inside the bracket by {a}: {key}.",
        "Expressions are equivalent when they have the same value for every x.",
    ]
    return _mc(
        rng,
        (
            "Which expression is equivalent to {t}?",
            "Expanding {t} gives which expression?",
            "Which of these is the same as {t} for every value of x?",
        ),
        key,
        wrongs,
        work,
        t=text,
    )


def _eq_easy_collect(rng):
    p, q = rng.randint(2, 9), rng.randint(2, 9)
    r = rng.choice([v for v in range(-9, 10) if v != 0])
    text = join_terms([(p, "x"), (q, "x"), (r, "")])
    f = lambda x: (p + q) * x + r  # noqa: E731
    key = join_terms([(p + q, "x"), (r, "")])
    cands = [
        (join_terms([(p * q, "x"), (r, "")]), lambda x: p * q * x + r, "w5q-multiplies-like-terms"),
        (join_terms([(p + q, "x^2"), (r, "")]), lambda x: (p + q) * x * x + r, "w5q-adds-powers"),
        (join_terms([(p + q + r, "x")]), lambda x: (p + q + r) * x, "w5q-combines-unlike-terms"),
        (join_terms([(p + q, "x"), (-r, "")]), lambda x: (p + q) * x - r, "w5q-sign-slip"),
        (join_terms([(p - q, "x"), (r, "")]), lambda x: (p - q) * x + r, "w5q-subtracts-like-terms"),
    ]
    wrongs = [(t, tag) for t, g, tag in cands if not _same(f, g) and not t.lstrip("-").isdigit()]
    work = [f"{p}x and {q}x are like terms, so they combine to {p + q}x.", f"The constant {r} is unchanged: {key}."]
    return _mc(
        rng,
        ("Which expression is equivalent to {t}?", "Simplify {t}. Which expression is equivalent to it?"),
        key,
        wrongs,
        work,
        t=text,
    )


def _eq_medium_not_equivalent(rng):
    a = rng.randint(2, 6)
    b = rng.choice((1, 2, 3))
    c = rng.choice([v for v in range(-6, 7) if v != 0])
    d = rng.choice([v for v in range(-8, 9) if v != 0])
    text, f = _bracket_expr(a, b, c, d)
    good = [t for t, g in _forms(a, b, c, d) if _same(f, g)]
    bad = [(t, tag) for t, g, tag in _near_misses(a, b, c, d) if not _same(f, g)]
    good = [t for t in dict.fromkeys(good) if t != text]
    if len(good) < 4 or not bad:
        raise TooFewDistractors("not enough forms")
    rng.shuffle(good)
    key, key_tag = rng.choice(bad)
    wrongs = [(t, "w5q-appearance-differs") for t in good[:4]]
    work = [
        f"Expand {text}: it equals {_forms(a, b, c, d)[0][0]}.",
        f"{key} does not have the same value as {text} for every x.",
    ]
    return _mc(
        rng,
        (
            "Four of these expressions are equivalent to {t}. Which one is NOT?",
            "Which expression is NOT equivalent to {t}?",
            "Which of these does NOT have the same value as {t} for every x?",
        ),
        key,
        wrongs,
        work,
        t=text,
    )


def _eq_hard_evidence(rng):
    a = rng.randint(2, 5)
    b = rng.choice((1, 2))
    c = rng.choice([v for v in range(-5, 6) if v != 0])
    d = rng.choice([v for v in range(-6, 7) if v != 0])
    ptxt, pf = _bracket_expr(a, b, c, d)
    ab, rest = a * b, a * c + d
    if rng.random() < 0.5:
        # equivalent pair: the second expression is a rearranged form
        qtxt = rng.choice([t for t, _ in _forms(a, b, c, d)[:3] if t != ptxt] or [join_terms([(ab, "x"), (rest, "")])])
        key = f"Equivalent: both expand to {join_terms([(ab, 'x'), (rest, '')])}"
        t = rng.choice((2, 3, 4))
        slip = join_terms([(ab, "x"), (rest + 1, "")])
        wrongs = [
            (
                f"Not equivalent: when x = {t} they give {pf(t)} and {pf(t) + rng.choice((1, 2, 3))}",
                "w5q-arithmetic-slip",
            ),
            ("Not equivalent: the two expressions look different", "w5q-appearance-differs"),
            (f"Equivalent: both expand to {slip}", "w5q-arithmetic-slip"),
            (f"Equivalent: both expand to {join_terms([(ab, 'x'), (c + d, '')])}", "w5q-distributes-to-first-term"),
        ]
        for tt in (qtxt,):
            check(_same(pf, lambda x: ab * x + rest), "the second expression is not equivalent")
    else:
        m = rng.choice([v for v in range(-3, 4) if v not in (0,)])
        t0 = rng.choice((1, 2, 3, 4))
        slope, const = ab + m, rest - m * t0
        qf = lambda x: slope * x + const  # noqa: E731
        qtxt = join_terms([(slope, "x"), (const, "")])
        check(pf(t0) == qf(t0), "the expressions do not agree at the shared value")
        t = rng.choice([v for v in (5, 6, 7, -2, -1, 0, 3) if v != t0 and pf(v) != qf(v)])
        key = f"Not equivalent: when x = {t} they give {pf(t)} and {qf(t)}"
        wrongs = [
            (f"Equivalent: both give {pf(t0)} when x = {t0}", "w5q-one-example-as-proof"),
            (f"Not equivalent: when x = {t} they give {pf(t)} and {qf(t) + m}", "w5q-arithmetic-slip"),
            ("Equivalent: both expressions have two terms", "w5q-appearance-differs"),
            (f"Not equivalent: when x = {t0} they give {pf(t0)} and {pf(t0) + 1}", "w5q-arithmetic-slip"),
        ]
    work = [
        f"Expand {ptxt}: it equals {join_terms([(ab, 'x'), (rest, '')])}.",
        f"Compare it with {qtxt}: {'they are the same for every x' if key.startswith('Equiv') else 'they give different values at x = ' + str(t)}.",
    ]
    return _mc(
        rng,
        (
            "Decide whether {p} and {q} are equivalent. Which statement is correct?",
            "Are {p} and {q} equivalent? Choose the statement that is correct and gives the reason.",
        ),
        key,
        wrongs,
        work,
        p=ptxt,
        q=qtxt,
    )


def _eq_exam(rng):
    a = rng.randint(2, 5)
    b = rng.choice((1, 2))
    c = rng.choice([v for v in range(-5, 6) if v != 0])
    d = rng.choice([v for v in range(-6, 7) if v != 0])
    text, f = _bracket_expr(a, b, c, d)
    forms = _forms(a, b, c, d)
    key = forms[0][0]
    near = [(t, tag) for t, g, tag in _near_misses(a, b, c, d) if not _same(f, g)]
    fa = mcq_fields(key)
    da = mcq_cands(key, near, 4)
    t = rng.choice((2, 3, 4))
    fb = number_fields(f(t))
    db = number_cands(
        f(t),
        [
            (a * b * t + c + d, "w5q-distributes-to-first-term"),
            (a * (b * t) + c + d, "w5q-distributes-to-first-term"),
            (a * b * t + a * c - d, "w5q-sign-slip"),
            (f(t) + a, "w5q-arithmetic-slip"),
            ((a + b) * t + c + d, "w5q-adds-instead-of-multiplies"),
        ],
    )
    good = [x for x, g in forms[1:] if _same(f, g)]
    good = [x for x in dict.fromkeys(good) if x != key and x != text]
    if len(good) < 3 or not near:
        raise TooFewDistractors("few forms")
    odd, odd_tag = near[0]
    fc = mcq_fields(odd)
    dc = mcq_cands(odd, [(g, "w5q-appearance-differs") for g in good], 4)
    for xx in (da, db, dc):
        enough(xx, 2)
    return dict(
        text=f"Consider the expression {text}.",
        parts=[
            part(
                "Which expression is equivalent to it, written without brackets?",
                "mcq",
                _sub(fa),
                2,
                [f"Multiply every term in the bracket by {a} and add the other term: {key}."],
                da,
            ),
            part(
                f"What is the value of the expression when x = {t}?",
                fb["answer_kind"],
                _sub(fb),
                1,
                [f"Substitute x = {t}: {f(t)}."],
                db,
            ),
            part(
                "Which of these expressions is NOT equivalent to the original?",
                "mcq",
                _sub(fc),
                1,
                [f"{odd} has a different value from {text} for some x."],
                dc,
            ),
        ],
    )


@generator(
    "w5_equivalent_expressions",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_equivalent_expressions(rng, tier="easy"):
    """Equivalent expressions; the tiers are described in the module docstring."""
    pools = {
        "easy": (_eq_easy_expand, _eq_easy_collect),
        "medium": (_eq_medium_not_equivalent,),
        "hard": (_eq_hard_evidence,),
    }
    return _tiers(pools, rng, tier)


@generator("exd_x_w5_equivalent_expressions", level=17, presets={"exam": dict(tier="exam")})
def exd_x_w5_equivalent_expressions(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _eq_exam)


# =====================================================================================================================
# independent and dependent variables
# =====================================================================================================================

# Each situation names the quantity that is chosen or changes first (independent), the quantity that follows from it
# (dependent), fixed amounts, and quantities that play no part. `lin` = (f, r) fills "f + r * v" where there is one.
_SITUATIONS = (
    dict(
        story="A taxi company charges ${f} to start a trip and ${r} for each kilometre travelled.",
        indep="the distance travelled",
        dep="the total fare",
        fixed=("the ${f} starting charge", "the ${r} charge for each kilometre"),
        other=("the colour of the taxi", "the name of the driver"),
        lin=((3, 9), (2, 5)),
        vq="What is the total fare, in dollars, for a trip of {v} kilometres?",
    ),
    dict(
        story="A phone plan costs ${f} a month plus ${r} for each gigabyte of data used.",
        indep="the amount of data used",
        dep="the monthly bill",
        fixed=("the ${f} monthly charge", "the ${r} charge for each gigabyte"),
        other=("the brand of the phone", "the colour of the phone case"),
        lin=((10, 30), (2, 6)),
        vq="What is the monthly bill, in dollars, when {v} gigabytes of data are used?",
    ),
    dict(
        story="A cafe sells pies for ${r} each.",
        indep="the number of pies sold",
        dep="the total takings",
        fixed=("the ${r} price of one pie",),
        other=("the name of the cafe", "the colour of the napkins", "the opening hours"),
        lin=((0, 0), (3, 8)),
        vq="What are the total takings, in dollars, when {v} pies are sold?",
    ),
    dict(
        story="A tank holds {f} litres of water and a tap adds {r} litres every minute.",
        indep="the time the tap has been running",
        dep="the volume of water in the tank",
        fixed=("the {f} litres at the start", "the flow of {r} litres each minute"),
        other=("the colour of the tank", "the shape of the tap"),
        lin=((20, 80), (3, 9)),
        vq="How many litres are in the tank after {v} minutes?",
    ),
    dict(
        story="A car travels at a steady speed of {r} kilometres per hour.",
        indep="the time spent travelling",
        dep="the distance travelled",
        fixed=("the speed of {r} kilometres per hour",),
        other=("the colour of the car", "the name of the driver", "the make of the tyres"),
        lin=((0, 0), (40, 90)),
        vq="How far, in kilometres, does the car travel in {v} hours?",
    ),
    dict(
        story="A plant is measured at the end of each week as it grows.",
        indep="the number of weeks",
        dep="the height of the plant",
        fixed=(),
        other=("the colour of the pot", "the name of the plant", "the brand of the soil"),
    ),
    dict(
        story="A student studies for different lengths of time before a test and records the test score each time.",
        indep="the length of time studied",
        dep="the test score",
        fixed=(),
        other=("the colour of the pen", "the name of the teacher"),
    ),
    dict(
        story="A cyclist has to ride 60 kilometres and chooses the speed to ride at.",
        indep="the speed chosen",
        dep="the time the ride takes",
        fixed=("the 60 kilometre distance",),
        other=("the colour of the bike", "the name of the cyclist"),
        unusual=True,
    ),
    dict(
        story="A shop tries different prices for a new toy and counts how many it sells in a week at each price.",
        indep="the price of the toy",
        dep="the number sold in a week",
        fixed=(),
        other=("the colour of the toy", "the name of the shop", "the shop's street address"),
        unusual=True,
    ),
    dict(
        story="A farmer uses different amounts of fertiliser on separate paddocks and weighs the crop from each.",
        indep="the amount of fertiliser used",
        dep="the mass of the crop",
        fixed=(),
        other=("the name of the farmer", "the colour of the tractor"),
        unusual=True,
    ),
    dict(
        story="A runner decides how far to run and a smartwatch records the time it takes her.",
        indep="the distance chosen",
        dep="the time taken",
        fixed=(),
        other=("the colour of the watch", "the name of the runner"),
        unusual=True,
    ),
    dict(
        story="A scientist heats a beaker of water and records its temperature every minute.",
        indep="the time the water has been heated",
        dep="the temperature of the water",
        fixed=(),
        other=("the colour of the beaker", "the name of the scientist", "the brand of the thermometer"),
    ),
    dict(
        story="A hardware shop sells wire at ${r} a metre.",
        indep="the length of wire bought",
        dep="the cost of the wire",
        fixed=("the ${r} price of one metre",),
        other=("the name of the shop", "the colour of the wire", "the width of the aisle"),
        lin=((0, 0), (2, 9)),
        vq="What is the cost, in dollars, of {v} metres of wire?",
    ),
    dict(
        story="A gym charges a joining fee of ${f} and then ${r} for each week of membership.",
        indep="the number of weeks of membership",
        dep="the total cost",
        fixed=("the ${f} joining fee", "the ${r} charge for each week"),
        other=("the name of the gym", "the colour of the towels"),
        lin=((20, 60), (5, 12)),
        vq="What is the total cost, in dollars, after {v} weeks of membership?",
    ),
    dict(
        story="Masses are hung one at a time from a spring and the length of the spring is measured each time.",
        indep="the mass hung from the spring",
        dep="the length of the spring",
        fixed=(),
        other=("the colour of the spring", "the name of the scientist", "the brand of the ruler"),
    ),
    dict(
        story="A candle burns down as time passes and its height is measured every ten minutes.",
        indep="the time the candle has burned",
        dep="the height of the candle",
        fixed=(),
        other=("the colour of the candle", "the shape of the holder", "the brand of the matches"),
    ),
    dict(
        story="A bakery decides how many cakes to bake in a day and records the flour used.",
        indep="the number of cakes baked",
        dep="the mass of flour used",
        fixed=(),
        other=("the name of the baker", "the colour of the oven", "the size of the shop"),
    ),
    dict(
        story="A ball is dropped from different heights and the time it takes to reach the ground is measured.",
        indep="the height the ball is dropped from",
        dep="the time the ball takes to fall",
        fixed=(),
        other=("the colour of the ball", "the name of the student", "the brand of the stopwatch"),
        unusual=True,
    ),
    dict(
        story="A student chooses how many hours to sleep on different nights and records her score on a reaction test the next morning.",
        indep="the hours of sleep",
        dep="the reaction test score",
        fixed=(),
        other=("the colour of the pillow", "the name of the test", "the brand of the alarm clock"),
        unusual=True,
    ),
)


def _draw_situation(rng, unusual=None, numeric=False):
    pool = [
        s for s in _SITUATIONS if (unusual is None or bool(s.get("unusual")) == unusual) and (not numeric or "lin" in s)
    ]
    s = rng.choice(pool)
    slots = {}
    if "lin" in s:
        (f_lo, f_hi), (r_lo, r_hi) = s["lin"]
        slots = dict(f=rng.randint(f_lo, f_hi) if f_hi else 0, r=rng.randint(r_lo, r_hi))
    fill = lambda t: t.replace("${f}", f"${slots.get('f', 0)}").replace("${r}", f"${slots.get('r', 0)}").format(**slots)  # noqa: E731
    return dict(
        story=fill(s["story"]),
        indep=s["indep"],
        dep=s["dep"],
        fixed=tuple(fill(t) for t in s["fixed"]),
        other=tuple(s["other"]),
        slots=slots,
        vq=s.get("vq"),
    )


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def _vr_ask(rng, unusual=None):
    s = _draw_situation(rng, unusual)
    ask_dep = rng.random() < 0.5
    key = s["dep"] if ask_dep else s["indep"]
    swapped = s["indep"] if ask_dep else s["dep"]
    wrongs = [(swapped, "w5v-swaps-the-roles")]
    wrongs += [(t, "w5v-picks-a-fixed-amount") for t in s["fixed"]]
    wrongs += [(t, "w5v-irrelevant-quantity") for t in s["other"]]
    word = "dependent" if ask_dep else "independent"
    meaning = (
        "depends on the other quantity: it is worked out from it"
        if ask_dep
        else "is chosen or changes first, and the other quantity depends on it"
    )
    work = [f"In this situation {key} {meaning}.", f"So {key} is the {word} variable."]
    ask = (
        (
            "Which quantity is the dependent variable?",
            "Which of these is the dependent variable?",
            "Which quantity depends on the others?",
        )
        if ask_dep
        else (
            "Which quantity is the independent variable?",
            "Which of these is the independent variable?",
            "Which quantity is chosen first, so that the others follow from it?",
        )
    )
    stems = tuple("{s} " + a for a in ask)
    return _mc(rng, stems, _cap(key), [(_cap(t), g) for t, g in wrongs], work, s=s["story"])


def _vr_pair(rng):
    s = _draw_situation(rng)
    key = f"Independent: {s['indep']}; dependent: {s['dep']}"
    wrongs = [(f"Independent: {s['dep']}; dependent: {s['indep']}", "w5v-swaps-the-roles")]
    for t in s["fixed"][:2]:
        wrongs.append((f"Independent: {t}; dependent: {s['dep']}", "w5v-picks-a-fixed-amount"))
    for t in s["other"][:2]:
        wrongs.append((f"Independent: {s['indep']}; dependent: {t}", "w5v-irrelevant-quantity"))
    work = [
        f"{_cap(s['dep'])} is worked out from {s['indep']}, so {s['dep']} depends on it.",
        "The independent variable is the one that is chosen or changes first.",
    ]
    return _mc(
        rng,
        (
            "{s} Which pair names the independent and the dependent variable?",
            "Read this situation. {s} Which statement names the two variables correctly?",
        ),
        key,
        wrongs,
        work,
        s=s["story"],
    )


def _vr_axis(rng):
    s = _draw_situation(rng)
    on_horizontal = rng.random() < 0.5
    horiz, vert = (s["dep"], s["indep"]) if on_horizontal else (s["indep"], s["dep"])
    ask_dep = rng.random() < 0.5
    key = s["dep"] if ask_dep else s["indep"]
    axis_pick = horiz if (horiz != key) else vert
    wrongs = [(_cap(axis_pick), "w5v-axis-position-decides")]
    wrongs += [(_cap(t), "w5v-picks-a-fixed-amount") for t in s["fixed"]]
    wrongs += [(_cap(t), "w5v-irrelevant-quantity") for t in s["other"]]
    word = "dependent" if ask_dep else "independent"
    work = [
        f"Which axis a variable is drawn on does not decide its role. {_cap(s['dep'])} depends on {s['indep']}.",
        f"So {key} is the {word} variable.",
    ]
    stems = (
        "{s} A graph of the situation shows {h} on the horizontal axis and {v} on the vertical axis. Which quantity is the "
        + word
        + " variable?",
        "{s} {a} is drawn on the horizontal axis of a graph and {b} on the vertical axis. Which quantity is the "
        + word
        + " variable?",
    )
    return _mc(rng, stems, _cap(key), wrongs, work, s=s["story"], h=horiz, v=vert, a=_cap(horiz), b=vert)


def _vr_change(rng):
    s = _draw_situation(rng)
    ask_dep = rng.random() < 0.5
    key = s["dep"] if ask_dep else s["indep"]
    wrongs = [(_cap(s["indep"] if ask_dep else s["dep"]), "w5v-swaps-the-roles")]
    wrongs += [(_cap(t), "w5v-picks-a-fixed-amount") for t in s["fixed"]]
    wrongs += [(_cap(t), "w5v-irrelevant-quantity") for t in s["other"]]
    if ask_dep:
        stem = "{s} When {i} changes, which quantity changes as a result?"
        work = [
            f"{_cap(s['dep'])} is worked out from {s['indep']}, so it changes when {s['indep']} changes.",
            f"{_cap(s['dep'])} is the dependent variable.",
        ]
    else:
        stem = "{s} Which quantity can be chosen freely, with {d} then following from it?"
        work = [
            f"{_cap(s['indep'])} is the one that is chosen, and {s['dep']} follows from it.",
            f"{_cap(s['indep'])} is the independent variable.",
        ]
    return _mc(rng, (stem,), _cap(key), wrongs, work, s=s["story"], i=s["indep"], d=s["dep"])


def _vr_exam(rng):
    s = _draw_situation(rng, unusual=False, numeric=True)
    f, r = s["slots"]["f"], s["slots"]["r"]
    v = rng.randint(3, 12)
    value = f + r * v
    ka, kb = _cap(s["indep"]), _cap(s["dep"])
    wa = (
        [(_cap(s["dep"]), "w5v-swaps-the-roles")]
        + [(_cap(t), "w5v-picks-a-fixed-amount") for t in s["fixed"]]
        + [(_cap(t), "w5v-irrelevant-quantity") for t in s["other"]]
    )
    wb = (
        [(_cap(s["indep"]), "w5v-swaps-the-roles")]
        + [(_cap(t), "w5v-picks-a-fixed-amount") for t in s["fixed"]]
        + [(_cap(t), "w5v-irrelevant-quantity") for t in s["other"]]
    )
    fa, fb = mcq_fields(ka), mcq_fields(kb)
    da, db = mcq_cands(ka, wa, 4), mcq_cands(kb, wb, 4)
    fc = number_fields(value)
    dc = number_cands(
        value,
        [
            (r * v, "w5v-forgets-the-fixed-amount"),
            (f + r + v, "w5v-adds-instead-of-multiplies"),
            ((f + r) * v, "w5v-multiplies-the-fixed-amount"),
            (value + r, "w5v-off-by-a-step"),
            (f * r * v if f else r + v, "w5v-adds-instead-of-multiplies"),
        ],
    )
    for xx in (da, db, dc):
        enough(xx, 3)
    check(value == f + r * v, "value differs")
    return dict(
        text=s["story"],
        parts=[
            part(
                "Which quantity is the independent variable?",
                "mcq",
                _sub(fa),
                1,
                [f"{ka} is chosen first; the other quantity depends on it."],
                da,
            ),
            part(
                "Which quantity is the dependent variable?",
                "mcq",
                _sub(fb),
                1,
                [f"{kb} is worked out from {s['indep']}."],
                db,
            ),
            part(
                s["vq"].format(v=v),
                fc["answer_kind"],
                _sub(fc),
                2,
                [f"{f} + {r} × {v} = {value}." if f else f"{r} × {v} = {value}."],
                dc,
                sensible_range=(0, None),
            ),
        ],
    )


@generator(
    "w5_variables_roles",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_variables_roles(rng, tier="easy"):
    """Independent and dependent variables; the tiers are described in the module docstring."""
    pools = {
        "easy": (lambda r: _vr_ask(r, unusual=False),),
        "medium": (_vr_pair, lambda r: _vr_ask(r, unusual=True)),
        "hard": (_vr_axis, _vr_change),
    }
    return _tiers(pools, rng, tier)


@generator("exd_x_w5_variables_roles", level=17, presets={"exam": dict(tier="exam")})
def exd_x_w5_variables_roles(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _vr_exam)


# =====================================================================================================================
# proportional and non-proportional relations
# =====================================================================================================================


def _table_text(pairs) -> str:
    return "x: " + ", ".join(str(x) for x, _ in pairs) + "  y: " + ", ".join(str(y) for _, y in pairs)


def _is_proportional(pairs) -> bool:
    """Every y ÷ x is the same (all the x here are positive)."""
    return len({Fr(y, x) for x, y in pairs}) == 1


def _xs(rng, n=4):
    start = rng.choice((1, 1, 2, 3))
    step = rng.choice((1, 1, 2, 3))
    return [start + step * i for i in range(n)]


def _prop_pair(rng, xs=None):
    """(proportional table, [(non-proportional table, misconception id), ...])."""
    xs = xs or _xs(rng)
    k = rng.randint(2, 9)
    c = rng.choice((1, 2, 3, 4, 5, 6, -1, -2, -3))
    prop = [(x, k * x) for x in xs]
    others = [
        ([(x, k * x + c) for x in xs], "w5p-line-means-proportional"),
        ([(x, x + abs(c) + 1) for x in xs], "w5p-constant-difference-not-ratio"),
        ([(x, x * x) for x in xs], "w5p-increasing-means-proportional"),
        ([(x, k * x + (i % 2)) for i, x in enumerate(xs)], "w5p-roughly-constant-ratio"),
    ]
    for pairs, _ in others:
        check(not _is_proportional(pairs), "a non-proportional table is proportional")
    check(_is_proportional(prop), "the proportional table is not proportional")
    return prop, others, k


def _prop_table(rng):
    prop, others, k = _prop_pair(rng)
    key = _table_text(prop)
    wrongs = length_balanced(rng, key, [(_table_text(p), tag) for p, tag in others])
    work = [
        f"In the key table y ÷ x is always {k}, so y is a fixed multiple of x.",
        "In the other tables the value of y ÷ x changes.",
    ]
    stems = (
        "Each table has x = {xs}. Which table shows a proportional relationship between x and y?",
        "The x values are {xs}. In which table is y proportional to x?",
        "Exactly one table, with x = {xs}, shows y proportional to x. Which one?",
    )
    return _mc(rng, stems, key, wrongs, work, xs=", ".join(str(x) for x, _ in prop))


def _prop_table_not(rng):
    """Which table is NOT proportional, among four proportional tables."""
    xs = _xs(rng, rng.choice((3, 4, 5)))
    others = _prop_pair(rng, xs)[1]
    pairs, tag = rng.choice(others)
    tables = []
    for _ in range(4):
        kk = rng.randint(2, 9)
        tables.append([(x, kk * x) for x in xs])
    texts = [_table_text(t) for t in tables]
    if len(set(texts)) < 4 or _table_text(pairs) in texts:
        raise TooFewDistractors("repeated tables")
    key = _table_text(pairs)
    wrongs = [(t, "w5p-ratio-not-checked") for t in texts]
    for t in tables:
        check(_is_proportional(t), "a distractor table is not proportional")
    work = [
        "Divide y by x for each pair. In the key table the results are not all the same.",
        "In the other tables y ÷ x is the same for every pair.",
    ]
    stems = (
        "Each table has {rows} pairs of values, starting at x = {x0}. In which table is y NOT proportional to x?",
        "Four of these tables, all with {rows} pairs of values, show y proportional to x except one. Which one does not?",
    )
    return _mc(rng, stems, key, wrongs, work, x0=pairs[0][0], rows=len(pairs))


def _prop_equation(rng):
    k = rng.randint(2, 9)
    c = rng.randint(1, 9)
    j = rng.choice([v for v in (2, 3, 4, 5) if k % v])
    ct = rng.choice((f"{k}x", f"x/{k}", f"{k}x/{j}"))  # the fixed multiple of x, written three ways
    key = f"y = {ct}"
    wrongs = [
        (f"y = {ct} + {c}", "w5p-line-means-proportional"),
        (f"y = x + {k}", "w5p-constant-difference-not-ratio"),
        (f"y = {k}/x", "w5p-inverse-is-proportional"),
        ("y = x^2" if rng.random() < 0.5 else f"y = x^2 + {k}", "w5p-increasing-means-proportional"),
        (f"y = {ct} - {c}", "w5p-line-means-proportional"),
    ]
    wrongs = length_balanced(rng, key, wrongs)
    work = [
        f"y = {ct} makes y a fixed multiple of x, and x = 0 gives y = 0.",
        "The other equations have an added number or put x in the wrong place, so y is not a fixed multiple of x.",
    ]
    return _mc(
        rng,
        (
            "Which equation shows y proportional to x?",
            "Which of these equations describes a proportional relationship?",
            "In which equation is y a fixed multiple of x?",
        ),
        key,
        wrongs,
        work,
    )


def _prop_situation(rng):
    r = rng.randint(2, 9)
    f = rng.randint(2, 8)
    item = rng.choice(("pens", "tickets", "sandwiches", "stamps", "buns", "bottles"))
    key = f"The cost of {item} bought at ${r} each, for different numbers of {item}"
    wrongs = [
        (
            f"The cost of {item} at ${r} each when a ${f} booking fee is added to every order",
            "w5p-ignores-fixed-charge",
        ),
        (f"A taxi fare when the taxi charges ${f} to start and ${r} for each kilometre", "w5p-ignores-fixed-charge"),
        (
            f"The volume in a tank that holds {f * 10} litres to start and gains {r} litres each minute",
            "w5p-ignores-fixed-charge",
        ),
        (f"A plumber's bill when the call-out is ${f * 5} and each minute costs ${r}", "w5p-ignores-fixed-charge"),
        (
            f"A child's height when the child is {f * 10} centimetres tall now and grows {r} centimetres a year",
            "w5p-ignores-fixed-charge",
        ),
    ]
    work = [
        f"The cost of {item} is ${r} times the number bought, so zero {item} cost $0 and the cost is proportional.",
        "In the other situations there is an amount at the start, so zero input does not give zero output.",
    ]
    stems = (
        "Every situation below involves a rate of {r}. Which one shows a proportional relationship?",
        "Each situation has the number {r} in it as a rate. In which situation is the total proportional to the amount used?",
    )
    return _mc(rng, stems, key, wrongs, work, r=r)


def _prop_two_points(rng):
    """The line through two points: proportional only when it passes through the origin."""
    x1, x2 = sorted(rng.sample(range(1, 9), 2))
    m = rng.randint(2, 8)
    near = rng.sample([v for v in (m - 2, m - 1, m + 1, m + 2, m + 3) if v >= 1], 2)  # wrong multipliers, either side
    proportional = rng.random() < 0.5
    c = 0 if proportional else rng.choice([-6, -4, -3, -2, 2, 3, 4, 6])
    y1, y2 = m * x1 + c, m * x2 + c
    slope = Fr(y2 - y1, x2 - x1)
    intercept = y1 - slope * x1
    check(slope == m and intercept == c, "the line through the points differs from the construction")
    if c == 0:
        key = f"Yes: the line is y = {m}x, which passes through the origin"
        wrongs = [
            (
                f"No: the line is y = {m}x + {rng.choice((1, 2, 3))}, which does not pass through the origin",
                "w5p-intercept-miscalculated",
            ),
            ("No: it is a straight line, and a straight line is never proportional", "w5p-line-not-proportional"),
            (f"Yes: the line is y = {near[0]}x, which passes through the origin", "w5p-gradient-miscalculated"),
            (f"Yes: the line is y = {near[1]}x, which passes through the origin", "w5p-gradient-miscalculated"),
            ("No: a proportional relationship must have a gradient of 1", "w5p-gradient-one-only"),
        ]
    else:
        eq = f"y = {m}x {'+' if c > 0 else '-'} {abs(c)}"
        key = f"No: the line is {eq}, which does not pass through the origin"
        wrongs = [
            (f"Yes: the line is y = {m}x, which passes through the origin", "w5p-intercept-ignored"),
            ("Yes: the gradient is constant, so y is proportional to x", "w5p-line-means-proportional"),
            (
                f"No: the line is y = {near[0]}x {'+' if c > 0 else '-'} {abs(c)}, which does not pass through the origin",
                "w5p-gradient-miscalculated",
            ),
            (
                f"No: the line is y = {m}x {'-' if c > 0 else '+'} {abs(c)}, which does not pass through the origin",
                "w5p-sign-slip",
            ),
        ]
    stems = (
        "A straight line passes through the points ({x1}, {y1}) and ({x2}, {y2}). Is y proportional to x? Choose the correct statement.",
        "The relationship between x and y is a straight line through ({x1}, {y1}) and ({x2}, {y2}). Which statement about it is correct?",
        "The points ({x1}, {y1}) and ({x2}, {y2}) both lie on the graph of a linear relationship. Decide whether y is proportional to x and pick the statement that is correct.",
    )
    work = [
        f"The gradient is ({y2} - {paren_negative(y1)}) ÷ ({x2} - {x1}) = {m}.",
        f"Using a point, the y-intercept is {c}, so the line {'passes' if c == 0 else 'does not pass'} through the origin.",
    ]
    return _mc(rng, stems, key, wrongs, work, x1=x1, y1=y1, x2=x2, y2=y2)


def _prop_exam(rng):
    xs = _xs(rng)
    k = rng.randint(2, 9)
    prop = rng.random() < 0.5
    c = 0 if prop else rng.choice((2, 3, 4, 5, -2, -3))
    pairs = [(x, k * x + c) for x in xs]
    check(prop == _is_proportional(pairs), "proportionality differs from the construction")
    r0, r1 = Fr(pairs[0][1], pairs[0][0]), Fr(pairs[1][1], pairs[1][0])
    if prop:
        ka = f"Yes: y ÷ x is {k} for every pair"
        wa = [
            (f"No: y ÷ x is {k} for the first pair and {k + 1} for the second", "w5p-ratio-miscalculated"),
            ("No: the table does not include x = 0", "w5p-origin-not-in-table"),
            (f"Yes: y - x is {k - 1} for every pair", "w5p-difference-not-ratio"),
            (f"No: y ÷ x is {k + 1} for every pair", "w5p-ratio-miscalculated"),
        ]
    else:
        ka = f"No: y ÷ x is {rt(r0)} for the first pair and {rt(r1)} for the second"
        wa = [
            (f"Yes: y ÷ x is {k} for every pair", "w5p-ignores-fixed-charge"),
            (f"No: y ÷ x is {rt(r1)} for the first pair and {rt(r0)} for the second", "w5p-ratio-miscalculated"),
            ("Yes: y goes up each time x goes up", "w5p-increasing-means-proportional"),
            (f"Yes: y ÷ x is {rt(r0)} for every pair", "w5p-roughly-constant-ratio"),
        ]
    fa = mcq_fields(ka)
    da = mcq_cands(ka, wa, 4)
    t = rng.choice((10, 12, 15, 20))
    rule = f"y = {k}x" + ("" if not c else (f" + {c}" if c > 0 else f" - {-c}"))
    value = k * t + c
    ft = number_fields(value)
    dt = number_cands(
        value,
        [
            (k + t, "w5p-adds-instead-of-multiplies"),
            (k * t, "w5p-ignores-fixed-charge"),
            (value + k, "w5p-off-by-a-step"),
            (t + c, "w5p-adds-instead-of-multiplies"),
            (k * t - c if c else k * t + 1, "w5p-sign-slip"),
        ],
    )
    fk = number_fields(k)
    dk = number_cands(
        k,
        [
            (k + 1, "w5p-off-by-a-step"),
            (k - 1, "w5p-off-by-a-step"),
            (pairs[0][1], "w5p-first-y-as-constant"),
            (pairs[1][1] - pairs[0][1], "w5p-difference-not-ratio"),
            (k * 2, "w5p-adds-instead-of-multiplies"),
        ],
    )
    for xx in (da, dt, dk):
        enough(xx, 2)
    return dict(
        text=f"A table of values has {_table_text(pairs)}. It follows the rule {rule}.",
        parts=[
            part(
                "Is y proportional to x? Choose the correct statement.",
                "mcq",
                _sub(fa),
                2,
                ["y is proportional to x only when y ÷ x is the same for every pair."],
                da,
            ),
            part(
                f"Use the rule to find y when x = {t}.",
                ft["answer_kind"],
                _sub(ft),
                1,
                [f"y = {k} × {t}{'' if not c else (f' + {c}' if c > 0 else f' - {-c}')} = {value}."],
                dt,
            ),
            part(
                "What number is multiplied by x in the rule?",
                fk["answer_kind"],
                _sub(fk),
                1,
                [f"In the rule {rule}, x is multiplied by {k}."],
                dk,
            ),
        ],
    )


@generator(
    "w5_proportional_relations",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_proportional_relations(rng, tier="easy"):
    """Proportional or not; the tiers are described in the module docstring."""
    pools = {
        "easy": (_prop_table, _prop_equation),
        "medium": (_prop_situation, _prop_table_not),
        "hard": (_prop_two_points,),
    }
    return _tiers(pools, rng, tier)


@generator("exd_x_w5_proportional_relations", level=17, presets={"exam": dict(tier="exam")})
def exd_x_w5_proportional_relations(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _prop_exam)


# =====================================================================================================================
# inequality symbols
# =====================================================================================================================

_HOLDS = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "≤": lambda a, b: a <= b,
    "≥": lambda a, b: a >= b,
}
_FLIP = {"<": ">", ">": "<", "≤": "≥", "≥": "≤"}
_STRICT_SWAP = {"<": "≤", "≤": "<", ">": "≥", "≥": ">"}


def _num_text(v) -> str:
    """An integer or one-place decimal as written."""
    v = Fr(v)
    if v.denominator == 1:
        return str(v.numerator)
    return f"{float(v):.1f}"


def _wrong_tag(a, sym, b) -> str:
    """The slip behind the false statement 'a sym b'."""
    if a == b:
        return "w5g-strict-includes-equal" if sym in "<>" else "w5g-symbol-direction-reversed"
    if (a < 0 or b < 0) and _HOLDS[sym](abs(a), abs(b)):
        return "w5g-negative-larger-magnitude-bigger"
    return "w5g-symbol-direction-reversed"


def _statement_pool(rng):
    pairs = []
    for _ in range(40):
        if rng.random() < 0.7:
            a, b = rng.randint(-12, 12), rng.randint(-12, 12)
        else:
            a, b = Fr(rng.randint(-40, 40), 10), Fr(rng.randint(-40, 40), 10)
        pairs.append((a, b))
    return pairs


def _ineq_true_statement(rng):
    for _ in range(80):
        pairs = _statement_pool(rng)
        ts, fs = [], []
        for a, b in pairs:
            sym = rng.choice("<>≤≥")
            (ts if _HOLDS[sym](a, b) else fs).append((a, b, sym))
        if not ts:
            continue
        ka, kb, ksym = rng.choice(ts)
        key = f"{_num_text(ka)} {ksym} {_num_text(kb)}"
        wrongs = []
        for a, b, sym in fs:
            wrongs.append((f"{_num_text(a)} {sym} {_num_text(b)}", _wrong_tag(a, sym, b)))
        for text, _ in wrongs:
            left, sym, right = text.split(" ")
            check(not _HOLDS[sym](Fr(left), Fr(right)), "a distractor statement is true")
        check(_HOLDS[ksym](Fr(ka), Fr(kb)), "the key statement is false")
        if len({t for t, _ in wrongs}) < 5:
            continue
        # a mix of directions among the wrong ones, and the key not the only one with a decimal
        work = [
            f"Compare the numbers on the number line: {key} is true.",
            "In each of the other statements the symbol points the wrong way for the numbers.",
        ]
        return _mc(
            rng,
            (
                "Which of these statements is true?",
                "Exactly one of these inequalities is true. Which one?",
                "Which statement is correct?",
            ),
            key,
            wrongs,
            work,
        )
    raise TooFewDistractors("no statement set")


def _ineq_given_value(rng, var="x", v=None):
    """x has a known value; which of five inequality statements about x is true."""
    v = rng.randint(-9, 9) if v is None else v
    for _ in range(60):
        cands = {(sym, c) for sym in "<>≤≥" for c in range(v - 5, v + 6)}
        true = sorted(t for t in cands if _HOLDS[t[0]](v, t[1]))
        false = sorted(t for t in cands if not _HOLDS[t[0]](v, t[1]))
        ksym, kc = rng.choice(true)
        wrongs = rng.sample([t for t in false if t[1] != kc or t[0] != ksym], 6)
        key = f"{var} {ksym} {kc}"
        wr = [
            (f"{var} {sym} {c}", "w5g-strict-includes-boundary" if c == v and sym in "<>" else _wrong_tag(v, sym, c))
            for sym, c in wrongs
        ]
        check(all(not _HOLDS[sym](v, c) for sym, c in wrongs) and _HOLDS[ksym](v, kc), "statement truth differs")
        work = [
            f"Put {var} = {v} into each statement. {var} {ksym} {kc} is {v} {ksym} {kc}, which is true.",
            "Each of the other statements is false for this value.",
        ]
        return _mc(
            rng,
            (
                "{x} = {v}. Which of these statements is true?",
                "When {x} = {v}, only one of these statements is true. Which one?",
            ),
            key,
            wr,
            work,
            x=var,
            v=v,
        )
    raise TooFewDistractors("no statement set")


def _ineq_order_list(rng):
    for _ in range(80):
        vals = rng.sample(range(-12, 13), 4) if rng.random() < 0.6 else [Fr(rng.randint(-40, 40), 10) for _ in range(4)]
        if len({Fr(v) for v in vals}) < 4 or sum(1 for v in vals if v < 0) < 2:
            continue
        ascending = rng.random() < 0.5
        key_list = sorted(vals, reverse=not ascending)
        by_mag = sorted(vals, key=abs, reverse=not ascending)
        rev = list(reversed(key_list))
        swaps = []
        for i in range(3):
            t = list(key_list)
            t[i], t[i + 1] = t[i + 1], t[i]
            swaps.append(t)
        fmt = lambda xs: ", ".join(_num_text(v) for v in xs)  # noqa: E731
        key = fmt(key_list)
        wrongs = [
            (fmt(by_mag), "w5g-negative-larger-magnitude-bigger"),
            (fmt(rev), "w5g-increasing-decreasing-confused"),
            (fmt(swaps[rng.randrange(3)]), "w5g-neighbours-swapped"),
            (fmt(swaps[rng.randrange(3)]), "w5g-neighbours-swapped"),
            (fmt(list(reversed(by_mag))), "w5g-negative-larger-magnitude-bigger"),
        ]
        shown = list(vals)
        if shown == key_list:
            continue
        check(
            key_list == sorted(vals, reverse=not ascending)
            and all(fmt(list(t)) != key for t in (by_mag, rev) if list(t) != key_list),
            "list order differs",
        )
        word = "least to greatest" if ascending else "greatest to least"
        work = [
            f"On a number line the order from {word} is {key}.",
            "A negative number with larger digits is further left, so it is smaller.",
        ]
        return _mc(
            rng,
            (
                "These numbers are in no particular order: {v}. Which list puts them in order from {w}?",
                "Which list shows {v} in order from {w}?",
            ),
            key,
            wrongs,
            work,
            v=fmt(shown),
            w=word,
        )
    raise TooFewDistractors("no list")


_WORDS = (
    ("A lift can carry no more than {n} people.", "p", "≤", "the number of people"),
    ("A lift is allowed to carry at most {n} people.", "p", "≤", "the number of people"),
    ("A ride needs at least {n} tickets.", "t", "≥", "the number of tickets"),
    ("A minimum of {n} tickets is needed for a group booking.", "t", "≥", "the number of tickets"),
    ("A car park has fewer than {n} free spaces.", "s", "<", "the number of free spaces"),
    ("There are fewer than {n} seats left.", "s", "<", "the number of seats left"),
    ("A parcel must weigh more than {n} kilograms to be sent as freight.", "w", ">", "the mass in kilograms"),
    ("A test score above {n} earns a certificate.", "m", ">", "the score"),
    ("The temperature stays under {n} degrees.", "T", "<", "the temperature"),
    ("A maximum of {n} students may enrol.", "e", "≤", "the number of students"),
    ("A bridge is safe for loads up to and including {n} tonnes.", "L", "≤", "the load in tonnes"),
    ("A swimmer must be {n} years old or older.", "a", "≥", "the age in years"),
    ("The speed must exceed {n} kilometres per hour.", "v", ">", "the speed"),
    ("The bag holds {n} litres or less.", "V", "≤", "the volume in litres"),
)


def _word_options(var, n, sym):
    key = f"{var} {sym} {n}"
    wrongs = [
        (f"{var} {_FLIP[sym]} {n}", "w5g-symbol-direction-reversed"),
        (f"{var} {_STRICT_SWAP[sym]} {n}", "w5g-strict-for-inclusive" if sym in "≤≥" else "w5g-inclusive-for-strict"),
        (f"{var} {_FLIP[_STRICT_SWAP[sym]]} {n}", "w5g-symbol-direction-reversed"),
        (f"{var} = {n}", "w5g-equals-for-inequality"),
    ]
    return key, wrongs


def _ineq_words(rng):
    text, var, sym, qty = rng.choice(_WORDS)
    n = rng.choice([v for v in range(4, 61) if v % 5 != 0 or rng.random() < 0.3])
    key, wrongs = _word_options(var, n, sym)
    work = [f"{text.format(n=n)} Let {var} be {qty}.", f"The words mean {var} {sym} {n}."]
    stems = ("{t} Let {v} be {q}. Which inequality shows this?", "{t} Write this as an inequality for {q}, using {v}.")
    return _mc(rng, stems, key, wrongs, work, t=text.format(n=n), v=var, q=qty)


def _ineq_double(rng):
    lo = rng.randint(-8, 4)
    hi = lo + rng.randint(3, 7)
    s_lo, s_hi = rng.choice(("<", "≤")), rng.choice(("<", "≤"))
    text = f"{lo} {s_lo} x {s_hi} {hi}"

    def ok(v):
        return (_HOLDS[s_lo](lo, v)) and (_HOLDS[s_hi](v, hi))

    passing = [v for v in range(lo - 5, hi + 6) if ok(v)]
    failing = [v for v in range(lo - 5, hi + 6) if not ok(v)]
    key_v = rng.choice(passing)
    below = [v for v in failing if v <= lo]
    above = [v for v in failing if v >= hi]
    inside = [v for v in failing if lo < v < hi]
    n_below = rng.randint(0, 4)
    pick = rng.sample(below, min(n_below, len(below))) + rng.sample(above, min(4 - n_below + 1, len(above)))
    pick = list(dict.fromkeys(pick + inside))[:6]
    wrongs = []
    for v in pick:
        if v == lo and s_lo == "<" or v == hi and s_hi == "<":
            tag = "w5g-strict-includes-boundary"
        else:
            tag = "w5g-reads-only-one-bound"
        wrongs.append((f"x = {v}", tag))
    rng.shuffle(wrongs)
    check(all(not ok(int(t[4:])) for t, _ in wrongs) and ok(key_v), "double inequality options are wrong")
    work = [
        f"x has to satisfy both parts of {text} at the same time.",
        f"Both parts hold for x = {key_v}; each other value fails at least one.",
    ]
    return _mc(
        rng,
        ("Which value of x satisfies {t}?", "Only one of these values makes {t} true. Which one?"),
        f"x = {key_v}",
        wrongs,
        work,
        t=text,
    )


_STATEMENTS = (
    ("a > b", lambda a, b: a > b, "w5g-negative-larger-magnitude-bigger"),
    ("a < b", lambda a, b: a < b, "w5g-negative-larger-magnitude-bigger"),
    ("-a > -b", lambda a, b: -a > -b, "w5g-negative-larger-magnitude-bigger"),
    ("-a < -b", lambda a, b: -a < -b, "w5g-negative-larger-magnitude-bigger"),
    ("a + b < a", lambda a, b: a + b < a, "w5g-adding-a-negative-increases"),
    ("a + b > a", lambda a, b: a + b > a, "w5g-adding-a-negative-increases"),
    ("a - b > 0", lambda a, b: a - b > 0, "w5g-negative-larger-magnitude-bigger"),
    ("a - b < 0", lambda a, b: a - b < 0, "w5g-negative-larger-magnitude-bigger"),
    ("ab > 0", lambda a, b: a * b > 0, "w5g-sign-of-product"),
    ("ab < 0", lambda a, b: a * b < 0, "w5g-sign-of-product"),
    ("a^2 > b^2", lambda a, b: a * a > b * b, "w5g-squaring-keeps-order"),
    ("a^2 < b^2", lambda a, b: a * a < b * b, "w5g-squaring-keeps-order"),
)


def _ineq_substitute(rng):
    for _ in range(60):
        a, b = rng.sample(range(-9, 0), 2)
        true = [s for s in _STATEMENTS if s[1](a, b)]
        false = [s for s in _STATEMENTS if not s[1](a, b)]
        if not true or len(false) < 4:
            continue
        key = rng.choice(true)
        wrongs = rng.sample(false, 4)
        # two statements such as 'a > b' and 'a < b' always split, so exactly one of each pair is true
        check(all(not s[1](a, b) for s in wrongs) and key[1](a, b), "statement truth differs")
        work = [
            f"With a = {a} and b = {b}: a > b is {a > b}, a^2 and b^2 are {a * a} and {b * b}, and ab = {a * b}.",
            f"So {key[0]} is true and the others are false.",
        ]
        return _mc(
            rng,
            (
                "a = {a} and b = {b}. Which statement is true?",
                "When a = {a} and b = {b}, only one of these statements is true. Which one?",
            ),
            key[0],
            [(s[0], s[2]) for s in wrongs],
            work,
            a=a,
            b=b,
        )
    raise TooFewDistractors("no statement set")


def _ineq_exam(rng):
    text, var, sym, qty = rng.choice(_WORDS)
    n = rng.randint(6, 60)
    key_a, wrongs_a = _word_options(var, n, sym)
    fa = mcq_fields(key_a)
    da = mcq_cands(key_a, wrongs_a, 4)
    upper = sym in "<≤"
    strict = sym in "<>"
    if upper:
        extreme = n - 1 if strict else n
        word = "greatest"
        wrong_b = [
            (n, "w5g-strict-includes-boundary") if strict else (n - 1, "w5g-inclusive-for-strict"),
            (n + 1, "w5g-symbol-direction-reversed"),
            (n - 2 if strict else n + 1, "w5g-off-by-one"),
            (0, "w5g-picks-the-lowest"),
        ]
    else:
        extreme = n + 1 if strict else n
        word = "least"
        wrong_b = [
            (n, "w5g-strict-includes-boundary") if strict else (n + 1, "w5g-inclusive-for-strict"),
            (n - 1, "w5g-symbol-direction-reversed"),
            (n + 2 if strict else n - 1, "w5g-off-by-one"),
            (n * 2, "w5g-picks-the-lowest"),
        ]
    fb = number_fields(extreme)
    db = number_cands(extreme, wrong_b)
    ok = _HOLDS[sym]
    check(ok(extreme, n) and (not ok(extreme + 1, n) if upper else not ok(extreme - 1, n)), "extreme value is wrong")
    v_c = extreme
    cq = _ineq_given_value(rng, var, v_c)
    for _ in range(20):
        if cq["answer_text"] != key_a:
            break
        cq = _ineq_given_value(rng, var, v_c)
    else:
        raise TooFewDistractors("the true statement repeats part (a)")
    fc = {k: val for k, val in cq.items() if k in ("answer_kind", "answer", "answer_text", "answer_latex")}
    dc = cq["distractors"]
    for xx in (da, db, dc):
        enough(xx, 2)
    return dict(
        text=f"{text.format(n=n)} Let {var} be {qty}, and suppose {var} is a whole number.",
        parts=[
            part("Which inequality shows this?", "mcq", _sub(fa), 1, [f"The words mean {key_a}."], da),
            part(
                f"What is the {word} whole number {var} can be?",
                fb["answer_kind"],
                _sub(fb),
                2,
                [f"From {key_a}, the {word} whole number is {extreme}."],
                db,
            ),
            part(
                f"{var} = {v_c}. Which of these statements about {var} is true?", "mcq", _sub(fc), 1, cq["working"], dc
            ),
        ],
    )


@generator(
    "w5_inequality_symbols",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_inequality_symbols(rng, tier="easy"):
    """Reading inequality symbols; the tiers are described in the module docstring."""
    pools = {
        "easy": (_ineq_true_statement, _ineq_given_value),
        "medium": (_ineq_words,),
        "hard": (_ineq_double, _ineq_substitute, _ineq_order_list),
    }
    return _tiers(pools, rng, tier)


@generator("exd_x_w5_inequality_symbols", level=17, presets={"exam": dict(tier="exam")})
def exd_x_w5_inequality_symbols(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _ineq_exam)


# =====================================================================================================================
# arithmetic sequences
# =====================================================================================================================


def _plus(x, d) -> str:
    """'x + d', written 'x - |d|' when d is negative."""
    return f"{rt(x)} + {rt(d)}" if d >= 0 else f"{rt(x)} - {rt(-d)}"


def _seq_text(terms) -> str:
    return ", ".join(rt(t) for t in terms)


def _is_arithmetic(terms) -> bool:
    return len({Fr(b) - Fr(a) for a, b in zip(terms, terms[1:])}) == 1


def _arith(rng, n=5, decimals=False, wide=False):
    d = (
        rng.choice([v for v in range(-12 if wide else -9, 13 if wide else 10) if v not in (0,)])
        if not decimals
        else Fr(rng.choice([-3, -2, -1, 1, 2, 3, 5]), 2)
    )
    a = rng.randint(-12, 40) if wide else rng.randint(-5, 20)
    return a, d, [Fr(a) + d * i for i in range(n)]


def _alternating(start, p, q, n):
    """Terms whose differences alternate p, q, p, q, ..."""
    out = [start]
    for i in range(n - 1):
        out.append(out[-1] + (p if i % 2 == 0 else q))
    return out


def _non_arithmetic_sets(rng, a, d, n=5):
    """Plausible non-arithmetic sequences with the misconception behind each."""
    r = rng.choice((2, 3))
    a0 = rng.randint(1, 6)
    p, q = rng.sample(range(2, 9), 2)
    out = [
        ([a0 * r**i for i in range(n)], "w5a-ratio-for-difference"),
        ([(rng.randint(1, 4) + i) ** 2 for i in range(n)], "w5a-increasing-pattern-is-arithmetic"),
        (_alternating(a0, p, q, n), "w5a-repeating-differences"),
        ([Fr(a) + d * i + (3 if i == n - 2 else 0) for i in range(n)], "w5a-checks-one-pair"),
    ]
    fib = [rng.randint(1, 5), rng.randint(2, 6)]
    for _ in range(n - 2):
        fib.append(fib[-1] + fib[-2])
    out.append((fib, "w5a-increasing-pattern-is-arithmetic"))
    return out


def _seq_pick(rng):
    a, d, terms = _arith(rng, 5, decimals=rng.random() < 0.3)
    key = _seq_text(terms)
    wrongs = []
    for t, tag in _non_arithmetic_sets(rng, a, d):
        if not _is_arithmetic(t):
            wrongs.append((_seq_text(t), tag))
    check(_is_arithmetic(terms), "the key sequence is not arithmetic")
    work = [
        f"Subtract each term from the next: the difference is always {rt(d)}.",
        "In each other sequence the differences are not all the same.",
    ]
    return _mc(
        rng,
        (
            "Which of these sequences is an arithmetic sequence?",
            "Exactly one of these sequences has the same difference between every pair of neighbouring terms. Which one?",
            "Which sequence is arithmetic?",
        ),
        key,
        wrongs,
        work,
    )


def _seq_statement(rng):
    n = 5
    if rng.random() < 0.5:
        a, d, terms = _arith(rng, n)
        seq = _seq_text(terms)
        more = "more" if d > 0 else "less"
        key = f"Arithmetic: each term is {abs(d)} {more} than the term before"
        wrongs = [
            (
                f"Arithmetic: each term is {abs(a)} {more} than the term before"
                if abs(a) != abs(d) and a != 0
                else f"Arithmetic: each term is {abs(d) + 2} {more} than the term before",
                "w5a-first-term-as-difference",
            ),
            (f"Arithmetic: each term is {abs(d) + 1} {more} than the term before", "w5a-difference-miscounted"),
            ("Not arithmetic: each term is multiplied by the same number", "w5a-ratio-for-difference"),
            (
                f"Arithmetic: each term is {abs(d)} {'less' if d > 0 else 'more'} than the term before",
                "w5a-direction-reversed",
            ),
        ]
        work = [
            f"Differences: {', '.join(rt(y - x) for x, y in zip(terms, terms[1:]))}.",
            f"They are all {rt(d)}, so the sequence is arithmetic.",
        ]
    else:
        terms, tag = rng.choice(_non_arithmetic_sets(rng, rng.randint(1, 9), rng.randint(2, 5)))
        terms = [Fr(t) for t in terms]
        if _is_arithmetic(terms):
            raise TooFewDistractors("accidentally arithmetic")
        seq = _seq_text(terms)
        diffs = [y - x for x, y in zip(terms, terms[1:])]
        key = f"Not arithmetic: the differences are {', '.join(rt(x) for x in diffs)}"
        bad = list(diffs)
        bad[-1] += 1
        wrongs = [
            (f"Arithmetic: each term is {rt(diffs[0])} more than the term before", "w5a-checks-one-pair"),
            (f"Not arithmetic: the differences are {', '.join(rt(x) for x in bad)}", "w5a-difference-miscounted"),
            (f"Arithmetic: each term is {rt(terms[0])} more than the term before", "w5a-first-term-as-difference"),
            ("Arithmetic: the terms keep getting bigger", "w5a-increasing-pattern-is-arithmetic"),
        ]
        work = [
            f"Differences: {', '.join(rt(x) for x in diffs)}.",
            "The differences are not all the same, so the sequence is not arithmetic.",
        ]
    return _mc(
        rng,
        (
            "Which statement about the sequence {s} is correct?",
            "The sequence {s} is shown. Which description is correct?",
        ),
        key,
        wrongs,
        work,
        s=seq,
    )


def _seq_next(rng):
    a, d, terms = _arith(rng, 3, decimals=False, wide=True)
    nxt = terms[-1] + d
    key = f"The next term is {rt(nxt)}"
    wrongs = [
        (f"The next term is {rt(terms[-1] + a)}", "w5a-first-term-as-difference"),
        (f"The next term is {rt(nxt + 1)}", "w5a-difference-miscounted"),
        (f"The next term is {rt(terms[-1] - d)}", "w5a-direction-reversed"),
        (f"The next term is {rt(terms[-1] * 2)}", "w5a-ratio-for-difference"),
        (f"The next term is {rt(terms[-1] + terms[-2])}", "w5a-adds-previous-terms"),
    ]
    wrongs = [(t, g) for t, g in wrongs if t != key]
    work = [
        f"The common difference is {rt(terms[1] - terms[0])}.",
        f"Add it to the last term: {_plus(terms[-1], d)} = {rt(nxt)}.",
    ]
    return _mc(
        rng,
        (
            "The first three terms of an arithmetic sequence are {s}. Which is the next term?",
            "An arithmetic sequence begins {s}. What is the fourth term?",
            "Three consecutive terms of an arithmetic sequence are {s}. What comes straight after them?",
        ),
        key,
        wrongs,
        work,
        s=_seq_text(terms),
    )


def _seq_find_k(rng):
    for _ in range(80):
        m = rng.choice((1, 2, 3, 5))
        p, q = rng.sample(range(1, 7), 2)
        a, c = m * p * p, m * q * q
        if (a + c) % 2:
            continue
        k = (a + c) // 2
        geo = m * p * q
        if geo == k or c <= a:
            continue
        key = f"k = {k}"
        wrongs = [
            (f"k = {geo}", "w5a-ratio-for-difference"),
            (f"k = {c - a}", "w5a-difference-for-middle"),
            (f"k = {k + 1}", "w5a-difference-miscounted"),
            (f"k = {a + c}", "w5a-adds-previous-terms"),
            (f"k = {k - 1}", "w5a-difference-miscounted"),
        ]
        for t, _ in wrongs:
            v = int(t[4:])
            check(not (v - a == c - v), "a distractor value keeps the sequence arithmetic")
        check(k - a == c - k, "the key does not keep the sequence arithmetic")
        wrongs = [(t, g) for t, g in wrongs if int(t[4:]) != k]
        work = [f"For {a}, k, {c} to be arithmetic, k - {a} = {c} - k.", f"So 2k = {a + c} and k = {k}."]
        return _mc(
            rng,
            (
                "For which value of k are {a}, k, {c} three consecutive terms of an arithmetic sequence?",
                "The numbers {a}, k and {c} are consecutive terms of an arithmetic sequence. What is k?",
            ),
            key,
            wrongs,
            work,
            a=a,
            c=c,
        )
    raise TooFewDistractors("no k")


def _seq_rule(rng):
    a = rng.choice([v for v in range(-5, 8) if v not in (0,)])
    b = rng.randint(-5, 9)
    lin = join_terms([(a, "n"), (b, "")])
    f_lin = lambda n: a * n + b  # noqa: E731
    cands = [
        (
            f"n^2 {'+' if b >= 0 else '-'} {abs(b)}" if b else "n^2",
            lambda n: n * n + b,
            "w5a-increasing-pattern-is-arithmetic",
        ),
        (f"{abs(a) + 1}^n", lambda n: (abs(a) + 1) ** n, "w5a-ratio-for-difference"),
        (f"n(n + {abs(a)})", lambda n: n * (n + abs(a)), "w5a-increasing-pattern-is-arithmetic"),
        (f"{abs(a) + 2}/n", lambda n: Fr(abs(a) + 2, n), "w5a-decreasing-means-arithmetic"),
        (f"2^n + {abs(b) + 1}", lambda n: 2**n + abs(b) + 1, "w5a-ratio-for-difference"),
    ]
    wrongs = []
    for text, f, tag in cands:
        terms = [Fr(f(n)) for n in range(1, 7)]
        if not _is_arithmetic(terms):
            wrongs.append((text, tag))
    check(_is_arithmetic([Fr(f_lin(n)) for n in range(1, 7)]), "the linear rule is not arithmetic")
    work = [
        f"For {lin} the terms are {', '.join(str(f_lin(n)) for n in range(1, 5))}, with a constant difference of {a}.",
        "The other rules give differences that change from term to term.",
    ]
    return _mc(
        rng,
        (
            "The nth term of a sequence is given by a rule. Which rule gives an arithmetic sequence?",
            "Which of these rules for the nth term gives an arithmetic sequence?",
            "For which rule is the difference between neighbouring terms always the same?",
        ),
        lin,
        wrongs,
        work,
    )


def _seq_gaps(rng):
    a = rng.randint(-4, 12)
    d = rng.choice([v for v in range(-6, 8) if v not in (0,)])
    c = a + 3 * d
    key = f"{a + d} and {a + 2 * d}"
    d2 = Fr(c - a, 2)
    d4 = Fr(c - a, 4)
    wrongs = [
        (f"{rt(a + d2)} and {rt(a + 2 * d2)}", "w5a-miscounts-steps"),
        (f"{rt(a + d4)} and {rt(a + 2 * d4)}", "w5a-miscounts-steps"),
        (f"{a + a} and {a + 2 * a}" if a else "1 and 2", "w5a-first-term-as-difference"),
        (f"{a + d + 1} and {a + 2 * d + 2}", "w5a-difference-miscounted"),
        (f"{c - d} and {c - 2 * d}", "w5a-direction-reversed"),
    ]
    wrongs = [(t, g) for t, g in wrongs if t != key]
    check(_is_arithmetic([a, a + d, a + 2 * d, c]), "the filled sequence is not arithmetic")
    work = [
        f"From {a} to {c} there are 3 equal steps, so the common difference is ({c} - {paren_negative(a)}) ÷ 3 = {d}.",
        f"The missing terms are {_plus(a, d)} = {a + d} and {_plus(a + d, d)} = {a + 2 * d}.",
    ]
    return _mc(
        rng,
        (
            "The sequence {a}, __, __, {c} is arithmetic. Which pair of numbers fills the gaps?",
            "Two terms are missing from the arithmetic sequence {a}, __, __, {c}. Which pair fills the gaps?",
        ),
        key,
        wrongs,
        work,
        a=a,
        c=c,
    )


def _seq_exam(rng):
    a, d, terms = _arith(rng, 4, wide=True)
    seq = _seq_text(terms)
    more = "more" if d > 0 else "less"
    ka = f"Arithmetic: each term is {abs(d)} {more} than the term before"
    wa = [
        (
            f"Arithmetic: each term is {abs(a)} {more} than the term before"
            if abs(a) != abs(d) and a != 0
            else f"Arithmetic: each term is {abs(d) + 2} {more} than the term before",
            "w5a-first-term-as-difference",
        ),
        (f"Arithmetic: each term is {abs(d) + 1} {more} than the term before", "w5a-difference-miscounted"),
        ("Not arithmetic: each term is multiplied by the same number", "w5a-ratio-for-difference"),
        (
            f"Arithmetic: each term is {abs(d)} {'less' if d > 0 else 'more'} than the term before",
            "w5a-direction-reversed",
        ),
    ]
    fa = mcq_fields(ka)
    da = mcq_cands(ka, wa, 4)
    nxt = terms[-1] + d
    fb = number_fields(nxt)
    db = number_cands(
        nxt,
        [
            (terms[-1] + a, "w5a-first-term-as-difference"),
            (nxt + 1, "w5a-difference-miscounted"),
            (terms[-1] - d, "w5a-direction-reversed"),
            (terms[-1] * 2, "w5a-ratio-for-difference"),
            (terms[-1] + terms[-2], "w5a-adds-previous-terms"),
        ],
    )
    eighth = a + 7 * d
    fc = number_fields(eighth)
    dc = number_cands(
        eighth,
        [
            (a + 8 * d, "w5a-off-by-one-term"),
            (8 * d, "w5a-forgets-the-first-term"),
            (a + 7 * a, "w5a-first-term-as-difference"),
            (eighth - d * 2, "w5a-off-by-one-term"),
            (nxt + 4 * d, "w5a-off-by-one-term"),
        ],
    )
    for xx in (da, db, dc):
        enough(xx, 2)
    check(_is_arithmetic(terms), "the sequence is not arithmetic")
    return dict(
        text=phrase(
            rng,
            (
                "The first four terms of a sequence are {s}.",
                "A sequence begins {s}.",
                "Here are the first four terms of a sequence: {s}.",
            ),
            s=seq,
        ),
        parts=[
            part(
                "Which statement about the sequence is correct?",
                "mcq",
                _sub(fa),
                1,
                [f"The difference between neighbouring terms is {rt(d)}."],
                da,
            ),
            part(
                "What is the fifth term?",
                fb["answer_kind"],
                _sub(fb),
                1,
                [f"{_plus(terms[-1], d)} = {rt(nxt)}."],
                db,
            ),
            part(
                "What is the eighth term?",
                fc["answer_kind"],
                _sub(fc),
                2,
                [f"The eighth term is the first term plus 7 differences: {a} + 7 × {paren_negative(d)} = {eighth}."],
                dc,
            ),
        ],
    )


@generator(
    "w5_arithmetic_sequences",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_arithmetic_sequences(rng, tier="easy"):
    """Arithmetic sequences; the tiers are described in the module docstring."""
    pools = {
        "easy": (_seq_pick, _seq_statement),
        "medium": (_seq_next, _seq_pick, _seq_statement),
        "hard": (_seq_find_k, _seq_rule, _seq_gaps),
    }
    return _tiers(pools, rng, tier)


@generator("exd_x_w5_arithmetic_sequences", level=17, presets={"exam": dict(tier="exam")})
def exd_x_w5_arithmetic_sequences(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _seq_exam)
