"""Recipe subject "even_odd": symmetry of functions (VCE Methods 2023 Exam 1 Q8d shape).

The maths lives in ``_even_odd_maths`` and does not touch the recipe interface; this module is the registration
glue (subject, stem wording, asks, tagged distractors). Answers are exact whatever ``technology_active`` says,
because parity is a conceptual topic with nothing to round.
"""

from __future__ import annotations

from sympy import Rational, pi, simplify

from ..answers import Distractor
from ..core import latex_answer
from ..recipe_vocabulary import property_check
from ..recipes import Subject, ask, subject
from . import _even_odd_maths as m

_SUP = {2: "²", 3: "³", 4: "⁴", 5: "⁵"}
_EVEN_POW, _ODD_POW = (2, 4), (1, 3, 5)
_COEFFS = (1, 2, 3, -1, -2, Rational(1, 2), Rational(1, 3), Rational(2, 3))
_A_COEFFS = (
    1,
    2,
    3,
    4,
    5,
    Rational(1, 2),
    Rational(1, 3),
    Rational(1, 4),
    Rational(3, 2),
    Rational(2, 3),
    Rational(3, 4),
)
_OPTIONS = ("Even", "Odd", "Neither even nor odd", "Both even and odd")
_WORD = {m.EVEN: "Even", m.ODD: "Odd", m.NEITHER: "Neither even nor odd"}


def _coeff(c, body: str) -> str:
    """``c`` times ``body`` as plain text: 1 -> "x²", -1 -> "-x²", 1/3 -> "x²/3", 2/3 -> "2x²/3", 3 -> "3x²"."""
    c = Rational(c)
    sign = "-" if c < 0 else ""
    c = abs(c)
    if c.q == 1:
        return f"{sign}{body}" if c == 1 else f"{sign}{c.p}{body}"
    return f"{sign}{body}/{c.q}" if c.p == 1 else f"{sign}{c.p}{body}/{c.q}"


def term_text(term) -> str:
    coeff, kind, p = term
    if kind == "pow":
        return _coeff(coeff, "x" if p == 1 else f"x{_SUP[p]}")
    return _coeff(coeff, f"{kind}({'x' if p == 1 else f'{p}x'})")


def _draw_term(rng, parity: str, avoid=None):
    for _ in range(50):
        coeff = rng.choice(_COEFFS)
        if parity == m.EVEN:
            kind = rng.choice(("pow", "cos"))
            p = rng.choice(_EVEN_POW) if kind == "pow" else rng.choice((1, 2, 3))
        else:
            kind = rng.choice(("pow", "sin"))
            p = rng.choice(_ODD_POW) if kind == "pow" else rng.choice((1, 2, 3))
        term = (coeff, kind, p)
        if avoid is None or (term[1], term[2]) != (avoid[1], avoid[2]):
            return term
    raise RuntimeError("could not draw a distinct term")


def _phase_text(b: int, trig: str) -> str:
    return f"{trig}({'x' if b == 1 else f'{b}x'} + c)"


@subject("even_odd")
class EvenOdd(Subject):
    """Two terms p and q (for sum and product parity), and h = a·x^n + trig(bx + c) with c unknown."""

    families = frozenset({"polynomial", "circular", "hybrid"})
    properties = frozenset({"even_candidate", "odd_candidate", "periodic"})

    def __init__(self, p, q, a_term, trig, b, target, j=1):
        self.p, self.q = p, q
        self.a_term, self.trig, self.b, self.target, self.j = a_term, trig, b, target, j

    @classmethod
    def draw(cls, rng, technology_active):
        pair = rng.choice(((m.EVEN, m.EVEN), (m.ODD, m.ODD), (m.EVEN, m.ODD), (m.ODD, m.EVEN)))
        p = _draw_term(rng, pair[0])
        q = _draw_term(rng, pair[1], avoid=p)
        target = rng.choice((m.EVEN, m.ODD))
        a_term = (rng.choice(_A_COEFFS), "pow", rng.choice(_EVEN_POW if target == m.EVEN else _ODD_POW))
        return cls(p, q, a_term, rng.choice(("sin", "cos")), rng.choice((1, 2, 3)), target, rng.choice((1, 2, 3)))

    # facts
    @property
    def sum_parity(self) -> str:
        return m.function_parity([self.p, self.q])

    @property
    def product_parity(self) -> str:
        return m.combine_parity(m.term_parity(self.p[1], self.p[2]), m.term_parity(self.q[1], self.q[2]), "*")

    @property
    def needed_trig_parity(self) -> str:
        # the polynomial term has the target parity, so the trig term must share it
        return self.target

    def stem(self) -> str:
        h = f"{term_text(self.a_term)} + {_phase_text(self.b, self.trig)}"
        return (
            f"Let p(x) = {term_text(self.p)} and q(x) = {term_text(self.q)}. "
            f"Let h(x) = {h}, where c is a real constant."
        )


@property_check("even_candidate")
def _even_candidate(s: EvenOdd) -> bool:
    """The declared parity of p + q and of p * q agree with an independent numeric classification."""
    pe, qe = m.term_expr(s.p), m.term_expr(s.q)
    return m.numeric_parity(pe + qe) == s.sum_parity and m.numeric_parity(pe * qe) == s.product_parity


@property_check("odd_candidate")
def _odd_candidate(s: EvenOdd) -> bool:
    """h with c taken from the declared family has the target parity; with a stray shift it has neither."""
    from sympy import pi as _pi

    trig = m.shifted_expr
    fam = m.phase_family(s.trig, s.target).subs(m.K, 1)
    a = m.term_expr(s.a_term)
    ok = m.numeric_parity(a + trig(s.trig, s.b, fam)) == s.target
    return ok and m.numeric_parity(a + trig(s.trig, s.b, fam + _pi / 3)) == m.NEITHER


@property_check("periodic")
def _periodic(s: EvenOdd) -> bool:
    return m.numeric_parity(m.shifted_expr(s.trig, s.b, 0)) in (m.EVEN, m.ODD) and s.b >= 1


def _mcq(key: str, tags: dict[str, str], prompt: str, working: list[str]) -> dict:
    wrong = [Distractor(o, tags[o]) for o in _OPTIONS if o != key]
    return dict(
        prompt=prompt,
        answer_kind="mcq",
        answer=key,
        answer_text=key,
        answer_latex="plain:" + key,
        working=working,
        distractors=wrong,
    )


def _sum_tags(key: str) -> dict[str, str]:
    tags = {"Both even and odd": "eo-both-even-and-odd"}
    for opt in _OPTIONS[:3]:
        if opt == key:
            continue
        if key == _WORD[m.NEITHER]:
            tags[opt] = "eo-mixed-sum-called-even-or-odd"
        elif opt == _WORD[m.NEITHER]:
            tags[opt] = "eo-same-parity-sum-called-neither"
        else:
            tags[opt] = "eo-sum-parity-misjudged"
    return tags


@ask(None, "sum_parity", needs=("even_candidate",))
def sum_parity(s: EvenOdd, rng, ctx) -> dict:
    key = _WORD[s.sum_parity]
    pp, qp = (m.term_parity(t[1], t[2]) for t in (s.p, s.q))
    working = [
        f"p is {pp} and q is {qp}.",
        "A sum of two even functions is even and a sum of two odd functions is odd; an even plus an odd function is neither.",
        f"So p(x) + q(x) is {key.lower()}.",
    ]
    return _mcq(key, _sum_tags(key), "Is p(x) + q(x) even, odd, neither, or both?", working)


@ask(None, "product_parity", needs=("even_candidate",))
def product_parity(s: EvenOdd, rng, ctx) -> dict:
    key = _WORD[s.product_parity]
    pp, qp = (m.term_parity(t[1], t[2]) for t in (s.p, s.q))
    tags = {"Both even and odd": "eo-both-even-and-odd", "Neither even nor odd": "eo-product-parity-misjudged"}
    for opt in ("Even", "Odd"):
        if opt != key:
            tags[opt] = (
                "eo-product-odd-times-odd-called-odd"
                if (opt == "Odd" and pp == qp == m.ODD)
                else "eo-product-parity-misjudged"
            )
    working = [
        f"p is {pp} and q is {qp}.",
        "p(-x)q(-x) = ±p(x)q(x): the product is even when p and q have the same parity, and odd when they differ.",
        f"So p(x)q(x) is {key.lower()}.",
    ]
    return _mcq(key, tags, "Is p(x)q(x) even, odd, neither, or both?", working)


@ask(None, "polynomial_term_parity", needs=("odd_candidate", "periodic"))
def polynomial_term_parity(s: EvenOdd, rng, ctx) -> dict:
    """The parity of the power term of h. The prompt does not name it, so it has to be worked out from the exponent."""
    key = _WORD[s.target]
    other = _WORD[m.ODD if s.target == m.EVEN else m.EVEN]
    tags = {
        other: "eo-power-parity-misjudged",
        "Neither even nor odd": "eo-power-parity-misjudged",
        "Both even and odd": "eo-both-even-and-odd",
    }
    term = term_text(s.a_term)
    n = s.a_term[2]
    working = [
        f"Replace x by -x: (-x){_SUP[n] if n > 1 else ''} = {'' if n % 2 == 0 else '-'}x{_SUP[n] if n > 1 else ''}.",
        f"So {term} is {s.target}.",
    ]
    return _mcq(key, tags, f"Is the term {term} of h even, odd, neither, or both?", working)


@ask(None, "find_c", needs=("odd_candidate", "periodic"), requires=("polynomial_term_parity",), default_marks=3)
def find_c(s: EvenOdd, rng, ctx) -> dict:
    fam = m.phase_family_text(s.trig, s.target)
    other = m.phase_family_text(s.trig, m.ODD if s.target == m.EVEN else m.EVEN)
    single = "0" if fam == "k*pi" else "pi/2"
    period2 = "2k*pi" if fam == "k*pi" else "pi/2 + 2k*pi"
    n = {"sin": "sin", "cos": "cos"}[s.trig]
    fn = f"{n}(bx + c)"
    key_fam = fam
    if s.trig == "sin" and s.target == m.EVEN:
        reason = "sin(bx + c) = sin(c - bx) for all x needs cos c · sin(bx) = 0, so cos c = 0."
    elif s.trig == "cos" and s.target == m.ODD:
        reason = "cos(bx + c) = -cos(c - bx) for all x needs cos c · cos(bx) = 0, so cos c = 0."
    elif s.trig == "sin":
        reason = "sin(bx + c) = -sin(c - bx) for all x needs sin c · cos(bx) = 0, so sin c = 0."
    else:
        reason = "cos(bx + c) = cos(c - bx) for all x needs sin c · sin(bx) = 0, so sin c = 0."
    working = [
        f"The polynomial term is {s.target}, so {fn} must be {s.target} as well.",
        reason,
        f"So c = {key_fam}, where k is an integer.",
    ]
    wrong = [
        Distractor(other, "eo-phase-family-swapped"),
        Distractor(single, "eo-phase-single-value"),
        Distractor(period2, "eo-phase-period-2pi"),
    ]
    return dict(
        prompt=f"Find all values of c for which h is {s.target}. Give c in terms of the integer k.",
        answer_kind="expression",
        answer=key_fam,
        answer_text=key_fam,
        answer_latex=latex_answer(key_fam),
        working=working,
        distractors=wrong,
    )


def _fmt(expr) -> str:
    return str(simplify(expr)).replace("**", "^")


@ask(None, "evaluate_at_multiple_of_pi", needs=("odd_candidate", "periodic"), requires=("find_c",), default_marks=2)
def evaluate_at_multiple_of_pi(s: EvenOdd, rng, ctx) -> dict:
    trig = sympy_trig(s.trig)
    c0 = pi if m.phase_family_text(s.trig, s.target) == "k*pi" else pi / 2
    x0 = s.j * pi
    coeff, _, n = s.a_term
    poly = coeff * x0**n

    def h(c):
        return poly + trig(s.b * x0 + c)

    key = _fmt(h(c0))
    other_c0 = pi / 2 if c0 == pi else pi
    wrong = [
        (_fmt(poly), "eo-drops-trig-term"),
        (_fmt(h(0)), "eo-evaluates-at-c-zero"),
        (_fmt(h(other_c0)), "eo-phase-family-swapped"),
        (_fmt(poly + 1), "eo-evaluation-slip"),
        (_fmt(poly - 1), "eo-evaluation-slip"),
        (_fmt(poly + 2), "eo-evaluation-slip"),
        (_fmt(poly - 2), "eo-evaluation-slip"),
    ]
    seen, dis = {key}, []
    for text, tag in wrong:
        if text not in seen:
            seen.add(text)
            dis.append(Distractor(text, tag))
    ctx_prompt = (
        f"Using the smallest positive value of c from part ({ctx.label_of('find_c')}), "
        f"find h({'' if s.j == 1 else str(s.j)}\u03c0) exactly."
    )
    working = [
        f"The smallest positive c is {_fmt(c0)}.",
        f"h({s.j}\u03c0) = {_fmt(poly)} + {s.trig}({s.b * s.j}\u03c0 + {_fmt(c0)}) = {key}.",
    ]
    return dict(
        prompt=ctx_prompt,
        answer_kind="expression",
        answer=key,
        answer_text=key,
        answer_latex=latex_answer(key),
        working=working,
        distractors=dis[:3],
    )


def sympy_trig(name: str):
    from sympy import cos, sin

    return sin if name == "sin" else cos
