"""The `inequality` answer kind: a set of real numbers, written as inequalities or as intervals.

The answer is the solution set, not a spelling of it. `x > 3`, `3 < x`, `(3, ∞)` and `x ∈ (3, inf)` are the same
answer; `x ≥ 3` is not. A reply is read into a SymPy set and compared with the canonical answer's set, so a reply
that hedges ("x > 3 or x < 3", "x ≥ 3") is a different set and is wrong, with no rule needed for it.

What a reply may be:

* a comparison of the variable with numbers: `x < 5`, `-2 ≤ x < 7`, `5 > x`, `x ≠ 4` (the operators `<`, `>`,
  `<=`, `>=`, `≤`, `≥`, `⩽`, `⩾`, `=<`, `=>` and `!=`, `≠`);
* an interval: `(3, 5]`, `[-1, ∞)`, `(-inf, 2)`; a closed bracket at infinity is read as open;
* any of those joined with `or`, `∪` or `U` (a union) and `and` or `∧` (an intersection). A comma between two
  inequalities means "and" unless that would leave nothing, in which case it can only mean "or":
  "x > -2, x < 5" is (-2, 5) and "x < -1, x > 3" is the union;
* any of those after `x ∈` or `x in`, or inside set-builder braces `{x : ...}` / `{x | ...}`;
* "all real numbers" (also `ℝ`, `R`, "all x", "all real x") or "no solution" (also `∅`, `{}`, "none",
  "no real solutions", "empty set").

Bounds are exact: an integer, a decimal, a fraction, or a closed form such as `√2` or `5/3`. A decimal is only
the same bound when it is exactly equal (2.5 for 5/2, never 1.67 for 5/3). A reply that names a different
variable from the answer's is wrong; a reply in interval notation names none and is read for the answer's.
"""

from __future__ import annotations

import re

import sympy

from .typed import GradeResult, _multi_part_aware
from .values import read_value, same_number

#: Longest reply read at all. The longest canonical answer is about 40 characters.
MAX_INEQUALITY_LENGTH = 300

_SYMBOL_SPELLINGS = (
    ("≤", "<="),
    ("⩽", "<="),
    ("≦", "<="),
    ("≥", ">="),
    ("⩾", ">="),
    ("≧", ">="),
    ("=<", "<="),
    ("=>", ">="),
    ("≠", "!="),
    ("∞", "oo"),
    ("∪", " or "),
    ("⋃", " or "),
    ("∩", " and "),
    ("∧", " and "),
    ("∨", " or "),
    ("∈", " in "),
    ("ℝ", " R "),
    ("−", "-"),
    ("–", "-"),
)
_INFINITY_WORD_RE = re.compile(r"\b(?:infinity|infty|inf)\b", re.IGNORECASE)
_LATEX_SPELLINGS = (
    (r"\leq", "<="),
    (r"\le", "<="),
    (r"\geq", ">="),
    (r"\ge", ">="),
    (r"\neq", "!="),
    (r"\ne", "!="),
    (r"\infty", "oo"),
    (r"\cup", " or "),
    (r"\cap", " and "),
    (r"\in", " in "),
    (r"\mathbb{R}", " R "),
)
_LATEX_FRAC_RE = re.compile(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}")
#: A capital U between two intervals is a union: ") U (" or "]U[".
_U_UNION_RE = re.compile(r"(?<=[)\]])\s*U\s*(?=[(\[])")
_ALL_REALS_RE = re.compile(
    r"^(?:R|all\s+(?:real\s+)?(?:numbers|values(?:\s+of\s+[a-z])?|x|[a-z])|all\s+real\s+[a-z]|"
    r"every\s+real\s+number|\(\s*-oo\s*,\s*oo\s*\))$",
    re.IGNORECASE,
)
_EMPTY_RE = re.compile(
    r"^(?:no\s+(?:real\s+)?(?:solutions?|values?)|none|empty(?:\s+set)?|∅|\{\s*\}|\\emptyset|\\varnothing|ø|Ø)$",
    re.IGNORECASE,
)
_SET_BUILDER_RE = re.compile(r"^\{\s*(.+?)\s*[:|]\s*(.+)\}$", re.DOTALL)
_MEMBERSHIP_RE = re.compile(r"^([^\W\d]\w*)\s+in\s+(.+)$", re.DOTALL)
#: "x ∈ ℝ," or "x in R:" in front of the condition.
_DOMAIN_PREFIX_RE = re.compile(r"^([^\W\d]\w*)\s+in\s+R\s*[,:|]\s*(.+)$", re.DOTALL)
_OPERATOR_RE = re.compile(r"(<=|>=|!=|<|>)")
_VARIABLE_RE = re.compile(r"^[^\W\d]\w*$")
_CONSTANT_NAMES = frozenset({"e", "pi", "oo", "π", "R", "and", "or", "in"})
_OR_RE = re.compile(r"\s+or\s+|\s*\|\|\s*", re.IGNORECASE)
_AND_RE = re.compile(r"\s+and\s+|\s*&&?\s*", re.IGNORECASE)


class _Unreadable(Exception):
    pass


def _normalise(text: str) -> str:
    text = text.replace("\\left", "").replace("\\right", "")  # before "\\le" can eat their first letters
    for latex, plain in _LATEX_SPELLINGS:
        text = text.replace(latex, plain)
    text = _LATEX_FRAC_RE.sub(r"(\1)/(\2)", text)
    text = text.replace("\\{", "{").replace("\\}", "}")
    text = text.replace("$", "")
    for symbol, plain in _SYMBOL_SPELLINGS:
        text = text.replace(symbol, plain)
    text = _INFINITY_WORD_RE.sub("oo", text)
    text = _U_UNION_RE.sub(" or ", text)
    text = re.sub(r"-\s+oo", "-oo", text)
    text = re.sub(r"\+\s*oo", "oo", text)
    return re.sub(r"\s+", " ", text).strip().rstrip(".")


def _top_level_split(text: str, separator: str = ",") -> list[str]:
    pieces, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                raise _Unreadable
        elif ch == separator and depth == 0:
            pieces.append(text[start:i])
            start = i + 1
    if depth != 0:
        raise _Unreadable
    pieces.append(text[start:])
    return [p.strip() for p in pieces]


def _split_outside_brackets(text: str, pattern: re.Pattern) -> list[str]:
    """`text` split on `pattern`, but never inside a bracket."""
    pieces, depth, start, i = [], 0, 0, 0
    while i < len(text):
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0:
            match = pattern.match(text, i)
            if match and match.end() > i:
                pieces.append(text[start:i])
                start = i = match.end()
                continue
        i += 1
    pieces.append(text[start:])
    return [p.strip() for p in pieces]


def _bound(text: str, *, allow_infinity: bool):
    text = text.strip()
    if allow_infinity and text in ("oo", "-oo"):
        return sympy.oo if text == "oo" else -sympy.oo
    value = read_value(text)
    if value is None:
        raise _Unreadable
    return value


def _interval(text: str):
    """An interval such as "(3, 5]" or "[-1, oo)", or None when `text` is not one."""
    if len(text) < 5 or text[0] not in "([" or text[-1] not in ")]":
        return None
    try:
        inner = _top_level_split(text[1:-1])
    except _Unreadable:
        return None
    if len(inner) != 2:
        return None
    low, high = _bound(inner[0], allow_infinity=True), _bound(inner[1], allow_infinity=True)
    if low == sympy.oo or high == -sympy.oo or not bool(low < high):
        raise _Unreadable
    return sympy.Interval(
        low, high, left_open=text[0] == "(" or low == -sympy.oo, right_open=text[-1] == ")" or high == sympy.oo
    )


def _is_variable(term: str) -> bool:
    return bool(_VARIABLE_RE.match(term)) and term not in _CONSTANT_NAMES


def _ray(variable_on_left: bool, op: str, value) -> sympy.Set:
    """The set of v with `v op value` (variable on the left) or `value op v`."""
    if not variable_on_left:
        op = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "!=": "!="}[op]
    if op == "<":
        return sympy.Interval.open(-sympy.oo, value)
    if op == "<=":
        return sympy.Interval(-sympy.oo, value, left_open=True)
    if op == ">":
        return sympy.Interval.open(value, sympy.oo)
    if op == ">=":
        return sympy.Interval(value, sympy.oo, right_open=True)
    return sympy.Union(sympy.Interval.open(-sympy.oo, value), sympy.Interval.open(value, sympy.oo))


def _comparison(text: str, variables: set) -> sympy.Set:
    tokens = [t.strip() for t in _OPERATOR_RE.split(text)]
    terms, ops = tokens[0::2], tokens[1::2]
    if not ops or any(not t for t in terms):
        raise _Unreadable
    where = [i for i, t in enumerate(terms) if _is_variable(t)]
    if len(where) != 1:
        raise _Unreadable
    index = where[0]
    variables.add(terms[index])
    if len(terms) == 2:
        value = _bound(terms[1 - index], allow_infinity=False)
        return _ray(index == 0, ops[0], value)
    if len(terms) == 3 and index == 1 and "!=" not in ops:
        rising = {"<", "<="}
        if (ops[0] in rising) != (ops[1] in rising):
            raise _Unreadable  # "1 < x > 3" says nothing a learner means
        low, high = _bound(terms[0], allow_infinity=False), _bound(terms[2], allow_infinity=False)
        left = _ray(False, ops[0], low)  # low < x
        right = _ray(True, ops[1], high)  # x < high
        return left.intersect(right)
    raise _Unreadable


def _atom(text: str, variables: set) -> sympy.Set:
    text = text.strip()
    if not text:
        raise _Unreadable
    while len(text) > 2 and text[0] == "(" and text[-1] == ")" and _interval(text) is None:
        # Brackets around a whole comparison: "(x > 3) or (x < -1)".
        try:
            _top_level_split(text[1:-1])
        except _Unreadable:
            break
        text = text[1:-1].strip()
    if _ALL_REALS_RE.match(text):
        return sympy.S.Reals
    if _EMPTY_RE.match(text):
        return sympy.S.EmptySet
    interval = _interval(text)
    if interval is not None:
        return interval
    return _comparison(text, variables)


def _union_of(text: str, variables: set) -> sympy.Set:
    result = sympy.S.EmptySet
    for disjunct in _split_outside_brackets(text, _OR_RE):
        result = result.union(_intersection_of(disjunct, variables))
    return result


def _intersection_of(text: str, variables: set) -> sympy.Set:
    result = sympy.S.Reals
    for conjunct in _split_outside_brackets(text, _AND_RE):
        result = result.intersect(_comma_joined(conjunct, variables))
    return result


def _comma_joined(text: str, variables: set) -> sympy.Set:
    pieces = _top_level_split(text)
    if len(pieces) == 1:
        return _atom(pieces[0], variables)
    sets = [_atom(piece, variables) for piece in pieces]
    both = sympy.Intersection(*sets)
    if both != sympy.S.EmptySet:
        return both
    return sympy.Union(*sets)


def read_solution_set(text: str) -> tuple[sympy.Set, str | None] | None:
    """The set a reply describes and the variable it names (None for interval notation), or None if unreadable."""
    if not isinstance(text, str) or len(text) > MAX_INEQUALITY_LENGTH:
        return None
    text = _normalise(text)
    if not text:
        return None
    variables: set = set()
    builder = _SET_BUILDER_RE.match(text)
    if builder:
        head, text = builder.group(1), builder.group(2).strip()
        head_variable = head.split(" in ")[0].strip()
        if not _is_variable(head_variable):
            return None
        variables.add(head_variable)
    prefix = _DOMAIN_PREFIX_RE.match(text)
    if prefix:
        variables.add(prefix.group(1))
        text = prefix.group(2).strip()
    membership = _MEMBERSHIP_RE.match(text)
    if membership and _is_variable(membership.group(1)):
        variables.add(membership.group(1))
        text = membership.group(2).strip()
    try:
        result = _union_of(text, variables)
    except (_Unreadable, TypeError, ValueError):
        return None
    except Exception:  # noqa: BLE001 - a set SymPy cannot build is unreadable, never an error
        return None
    if len(variables) > 1:
        return None
    return result, next(iter(variables), None)


def _same_set(left: sympy.Set, right: sympy.Set) -> bool:
    if left == right:
        return True
    try:
        return left.symmetric_difference(right) == sympy.S.EmptySet
    except Exception:  # noqa: BLE001 - undecidable is not equal
        return False


def _endpoint_text(value) -> str:
    if value == sympy.oo:
        return "∞"
    if value == -sympy.oo:
        return "-∞"
    return str(value).replace("sqrt", "√").replace("**", "^")


def set_text(solution: sympy.Set, variable: str | None = "x") -> str:
    """`solution` written as inequalities in `variable` (interval notation when `variable` is None)."""
    if solution == sympy.S.Reals:
        return "all real numbers"
    if solution == sympy.S.EmptySet:
        return "no solution"
    pieces = solution.args if isinstance(solution, sympy.Union) else (solution,)
    out = []
    for piece in pieces:
        if isinstance(piece, sympy.FiniteSet):
            out.extend(f"{variable or 'x'} = {_endpoint_text(v)}" for v in piece)
            continue
        low, high = piece.start, piece.end
        if variable is None:
            left = "(" if piece.left_open else "["
            right = ")" if piece.right_open else "]"
            out.append(f"{left}{_endpoint_text(low)}, {_endpoint_text(high)}{right}")
        elif low == -sympy.oo:
            out.append(f"{variable} {'<' if piece.right_open else '≤'} {_endpoint_text(high)}")
        elif high == sympy.oo:
            out.append(f"{variable} {'>' if piece.left_open else '≥'} {_endpoint_text(low)}")
        else:
            out.append(
                f"{_endpoint_text(low)} {'<' if piece.left_open else '≤'} {variable} "
                f"{'<' if piece.right_open else '≤'} {_endpoint_text(high)}"
            )
    return (" ∪ " if variable is None else " or ").join(out)


def _expected(question) -> tuple[sympy.Set, str | None] | None:
    answer = getattr(question, "answer", None)
    for candidate in (answer, getattr(question, "answer_text", None)):
        if isinstance(candidate, str):
            read = read_solution_set(candidate)
            if read is not None:
                return read
    return None


@_multi_part_aware
def grade_inequality(question, given: str) -> GradeResult:
    canonical = getattr(question, "answer_text", "")
    expected = _expected(question)
    if expected is None:
        raise ValueError(f"inequality answer {canonical!r} is not readable")  # a generator bug: `grade` logs it
    expected_set, expected_variable = expected
    read = read_solution_set(given if isinstance(given, str) else str(given))
    if read is None:
        return GradeResult(correct=False, canonical=canonical)
    given_set, given_variable = read
    normalised = set_text(given_set, given_variable or expected_variable)
    if given_variable is not None and expected_variable is not None and given_variable != expected_variable:
        return GradeResult(correct=False, canonical=canonical, given_normalised=normalised)
    return GradeResult(correct=_same_set(expected_set, given_set), canonical=canonical, given_normalised=normalised)


__all__ = ["grade_inequality", "read_solution_set", "same_number", "set_text"]
