"""Small helpers shared by the `r3g_*` senior trigonometry and algebra builds (radian trigonometric equations,
compound angles, products and sums of sines, multiple angles from De Moivre's theorem, t-formulae, the ambiguous case,
completing the square with the discriminant, and absolute values).

Not a generator module (the leading underscore keeps the package loader from importing it): it holds tagged
distractors for each answer kind these builds use, set answers whose items are multiples of pi or exact fractions,
and decimal number fields.
"""

from __future__ import annotations

import functools
import re
from fractions import Fraction

import sympy as sp

from ..answers import Distractor, exact_value
from ..core import VerificationError, check, format_pi_multiple, latex_pi_multiple, pick_distractors, rational_text


def typed(expr) -> str:
    """A SymPy value as it is typed: sqrt(3)/2, 5/13, -1."""
    return str(sp.sympify(expr)).replace("**", "^")


def latex(expr) -> str:
    return sp.latex(sp.sympify(expr))


def pi_text(coeff) -> str:
    """A multiple of pi as it is typed: Fraction(5, 6) -> 5pi/6, 1 -> pi, 0 -> 0."""
    return format_pi_multiple(Fraction(coeff))


def pi_latex(coeff) -> str:
    return latex_pi_multiple(Fraction(coeff))


def pi_shown(coeff) -> str:
    """A multiple of pi as the stem and working print it: 5π/6."""
    return pi_text(coeff).replace("pi", "π")


def ex(expr, tag: str) -> Distractor:
    """A wrong typed expression (a SymPy value or its text) as a tagged distractor."""
    text = typed(expr)
    return Distractor(text, tag, text)


def pi_dist(coeff, tag: str) -> Distractor:
    text = pi_text(coeff)
    return Distractor(text, tag, text)


def rat_dist(value, tag: str) -> Distractor:
    value = Fraction(value)
    return Distractor(rational_text(value), tag, exact_value(value))


class TooFewDistractors(VerificationError):
    """A draw left fewer than two distinct wrong values; `retry_short` draws again."""


_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def show(text: str) -> str:
    """A typed expression as a learner reads it in working: sqrt(3) -> √3, 2*pi -> 2π, 4*sqrt(3) -> 4√3, x^2 -> x²."""
    text = re.sub(r"sqrt\((\d+|[a-z])\)", r"√\1", text)
    for _ in range(3):
        text = re.sub(r"sqrt\(([^()]*(?:\([^()]*\)[^()]*)*)\)", r"√(\1)", text)
    text = re.sub(r"(?<![A-Za-z])pi(?![A-Za-z])", "π", text)
    text = text.replace("atan(", "arctan(")
    text = re.sub(r"\^(\d+)", lambda m: m.group(1).translate(_SUPERSCRIPT), text)
    text = re.sub(r"(\d)\*(?=arctan)", r"\1 ", text)
    text = re.sub(r"(\d)\*(?=[√π(a-zA-Z])", r"\1", text)
    text = re.sub(r"(?<=[)√π])\*(?=[√π(a-zA-Z\d])", "", text)
    text = re.sub(r"(?<=\S)\*(?=\S)", " × ", text)
    text = text.replace("+ -", "- ").replace("- -", "+ ")
    return text


def tidy_working(result):
    """Run `show` over every working line of a generated question (and of each of its parts)."""
    if isinstance(result, dict):
        if result.get("working"):
            result["working"] = [show(w) for w in result["working"]]
        for part in result.get("parts", ()):
            if part.get("working"):
                part["working"] = [show(w) for w in part["working"]]
    return result


def quad_text(a, b, c, var: str = "x") -> str:
    """a x² + b x + c as written: x² - 2x, 2x² + x - 3, -x² + 5."""

    def term(coef, body, first):
        if coef == 0:
            return ""
        mag = abs(coef)
        shown = body if mag == 1 and body else f"{mag}{body}"
        return (f"-{shown}" if coef < 0 else shown) if first else f" {'-' if coef < 0 else '+'} {shown}"

    t1 = term(a, f"{var}²", True)
    t2 = term(b, var, not t1)
    t3 = term(c, "", not (t1 or t2))
    return t1 + t2 + t3 or "0"


def factor_text(root, var: str = "x") -> str:
    """(x - 5), (x + 6) or x for the root 5, -6, 0."""
    if root == 0:
        return var
    return f"({var} - {root})" if root > 0 else f"({var} + {-root})"


def rename_symbol(result, old: str, new: str):
    """Rename the variable `old` to `new` in the text, working and part prompts of a generated question (never in answers)."""
    if old == new:
        return result
    pattern = re.compile(rf"(?<![A-Za-z]){re.escape(old)}(?![A-Za-z])")

    def fix(text: str) -> str:
        return pattern.sub(new, text)

    if isinstance(result, dict):
        if "text" in result:
            result["text"] = fix(result["text"])
        if result.get("working"):
            result["working"] = [fix(w) for w in result["working"]]
        for part in result.get("parts", ()):
            part["prompt"] = fix(part["prompt"])
            if part.get("working"):
                part["working"] = [fix(w) for w in part["working"]]
    return result


def vary_symbol(old: str, options):
    """Decorator: a generated question uses a randomly chosen variable name from `options` in place of `old`."""

    def decorate(build):
        @functools.wraps(build)
        def run(rng, *args, **kwargs):
            result = build(rng, *args, **kwargs)
            return rename_symbol(result, old, rng.choice(options))

        return run

    return decorate


def retry_short(build, tries: int = 60):
    """Wrap a generator body so a draw with too few distractors is redrawn (at most `tries` times, deterministically)."""

    @functools.wraps(build)
    def run(rng, *args, **kwargs):
        for _ in range(tries):
            try:
                return tidy_working(build(rng, *args, **kwargs))
            except TooFewDistractors:
                continue
        raise TooFewDistractors(f"{build.__name__}: no draw had enough distractors in {tries} tries")

    return run


def _repeats_a_member(candidate) -> bool:
    """A set written with the same member twice, such as {8.5, 8.5}: not a sensible wrong answer."""
    text = getattr(candidate, "text", candidate)
    if not (isinstance(text, str) and text.startswith("{") and text.endswith("}")):
        return False
    items = [item.strip() for item in text[1:-1].split(",")]
    return len(items) != len(set(items))


def pick(answer_text: str, candidates: list, count: int = 3, *, minimum: int = 2) -> list:
    """The first `count` candidates that are not the answer and not each other, at least `minimum` of them."""
    candidates = [c for c in candidates if c is not None and not _repeats_a_member(c)]
    out = pick_distractors(answer_text, candidates, count)
    if len(out) < minimum:
        raise TooFewDistractors(f"too few distractors for {answer_text}: {[str(c) for c in candidates]}")
    return out


# -- set answers ---------------------------------------------------------------------------------------------------


def item_text(item) -> str:
    """One member of a set answer as typed: a `Fraction` or int as `p/q`, a string as it is."""
    return item if isinstance(item, str) else rational_text(Fraction(item))


def set_text(items) -> str:
    return "{" + ", ".join(items) + "}"


def pi_set_fields(coeffs) -> dict:
    """answer, answer_text and answer_latex for a set of multiples of pi (increasing order)."""
    coeffs = sorted(Fraction(c) for c in coeffs)
    items = [pi_text(c) for c in coeffs]
    return dict(
        answer_kind="set",
        answer=items,
        answer_text=set_text(items),
        answer_latex=r"\left\{" + ",\\ ".join(pi_latex(c) for c in coeffs) + r"\right\}",
    )


def value_set_fields(values) -> dict:
    """answer, answer_text and answer_latex for a set of exact rationals (increasing order)."""
    values = sorted(Fraction(v) for v in values)
    items = [rational_text(v) for v in values]
    return dict(
        answer_kind="set",
        answer=items,
        answer_text=set_text(items),
        answer_latex=r"\left\{" + ",\\ ".join(_frac_latex(v) for v in values) + r"\right\}",
    )


def decimal_set_fields(values, places: int) -> dict:
    """A set of decimal numbers each shown at exactly `places` places (increasing order)."""
    from ..core import numeric_answer, round_half_up, stated_precision_string

    shown = sorted(round_half_up(Fraction(v), places) for v in values)
    items = [stated_precision_string(v, places) for v in shown]
    return dict(
        answer_kind="set",
        answer=list(items),
        answer_text=set_text(items),
        answer_latex=r"\left\{" + ",\\ ".join(items) + r"\right\}",
    )


def _frac_latex(v: Fraction) -> str:
    if v.denominator == 1:
        return str(v.numerator)
    sign = "-" if v < 0 else ""
    return rf"{sign}\frac{{{abs(v.numerator)}}}{{{v.denominator}}}"


def pi_set_dist(coeffs, tag: str) -> Distractor:
    coeffs = sorted(Fraction(c) for c in coeffs)
    items = [pi_text(c) for c in coeffs]
    return Distractor(set_text(items), tag, items)


def value_set_dist(values, tag: str) -> Distractor:
    values = sorted(Fraction(v) for v in values)
    items = [rational_text(v) for v in values]
    return Distractor(set_text(items), tag, items)


def decimal_set_dist(values, places: int, tag: str) -> Distractor:
    from ..core import round_half_up, stated_precision_string

    shown = sorted(round_half_up(Fraction(v), places) for v in values)
    items = [stated_precision_string(v, places) for v in shown]
    return Distractor(set_text(items), tag, items)


# -- decimal number answers ----------------------------------------------------------------------------------------


def number_fields(value, places: int) -> dict:
    """answer, answer_text and answer_latex for a number shown at exactly `places` decimal places."""
    from ..core import numeric_answer, round_half_up, stated_precision_string

    value = round_half_up(Fraction(value), places)
    text = stated_precision_string(value, places) if places else str(int(value))
    return dict(answer_kind="number", answer=numeric_answer(value), answer_text=text, answer_latex=text)


def decimal_dist(value, places: int, tag: str):
    """A wrong number as a tagged distractor at the answer's number of places (rounded half up)."""
    from ..core import numeric_answer, round_half_up, stated_precision_string

    shown = round_half_up(Fraction(value), places)
    text = stated_precision_string(shown, places) if places else str(int(shown))
    return Distractor(text, tag, numeric_answer(shown))


def int_dist(value, tag: str) -> Distractor:
    check(Fraction(value).denominator == 1, f"{value} is not a whole number")
    return Distractor(str(int(value)), tag, int(value))


def in_real_set(region, x) -> bool:
    """Whether the number x lies in `region`, a SymPy interval, finite set or union of them (exact, and much quicker
    than SymPy's own membership test)."""
    if region == sp.S.EmptySet:
        return False
    pieces = region.args if isinstance(region, sp.Union) else (region,)
    for piece in pieces:
        if isinstance(piece, sp.FiniteSet):
            if x in piece.args:
                return True
            continue
        low, high = piece.start, piece.end
        above = low == -sp.oo or x > low or (x == low and not piece.left_open)
        below = high == sp.oo or x < high or (x == high and not piece.right_open)
        if above and below:
            return True
    return False
