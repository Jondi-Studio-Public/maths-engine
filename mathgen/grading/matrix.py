"""The `matrix` answer kind: a matrix or a vector of exact numbers.

The answer (`question.answer`) is a list of rows, each a list of entries; a vector is one row or one column. An
entry is an int, an exact fraction (`{"p": 3, "q": 5}`, as `mathgen.answers.exact_value` writes it, or a
`Fraction`), or text for a closed form ("sqrt(2)/2"). Entries are compared exactly, so 0.2 is 1/5 but 0.33 is
not 1/3.

What a reply may be:

* nested brackets: `[[1, 2], [3, 4]]`, `((1, 2), (3, 4))`, `[1 2] [3 4]`;
* rows separated by `;` or new lines, entries by commas or spaces: `[1 2; 3 4]`, `1, 2; 3, 4`;
* LaTeX: `\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}` (any matrix environment);
* any of those after a label (`A⁻¹ =`, `A^-1 =`, `P =`, `v =`), and with a number in front or a divisor after:
  `1/5 [[1, -2], [3, 4]]`, `(1/5)[1 -2; 3 4]`, `-1/2 × [...]`, `[[1, -2], [3, 4]]/5`;
* for a vector answer, the entries in one list in either orientation: `(1, -2, 3)`, `[1 -2 3]`, `<1, -2, 3>`.

`question.answer_form` relaxes the comparison where the answer is not unique:

* `scalar_multiple`: any non-zero multiple of the answer is right (an eigenvector, a basis for a line);
* `column_scalar_multiple`: each column may be any non-zero multiple of the answer's column (a matrix of
  eigenvectors with the column order fixed by the question).
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from .typed import GradeResult, _multi_part_aware
from .values import read_value, same_number

#: Longest reply read at all. A 4×4 matrix of fractions is under 200 characters.
MAX_MATRIX_LENGTH = 600
#: Largest matrix read, either way. The library asks for nothing over 4×4.
MAX_DIMENSION = 6

MATRIX_FORMS = frozenset({"scalar_multiple", "column_scalar_multiple"})

_LABEL_RE = re.compile(
    r"^\s*[^\W\d]\w*\s*(?:\^\s*\{?\(?\s*-\s*1\s*\)?\}?|⁻¹|\^\s*\{?T\}?|ᵀ|')?\s*(?:\(\s*\w+\s*\))?\s*=\s*"
)
_LATEX_ENV_RE = re.compile(r"\\(?:begin|end)\{[a-zA-Z]*matrix\*?\}|\\left|\\right")
_LATEX_FRAC_RE = re.compile(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}")
_SCALAR_PREFIX_RE = re.compile(
    r"^\s*(\(\s*-?[\d.]+\s*/\s*[\d.]+\s*\)|-?[\d.]+\s*/\s*[\d.]+|-?[\d.]+|-)\s*[*×·]?\s*(?=[\[(])"
)
_DIVISOR_SUFFIX_RE = re.compile(r"(?<=[\])])\s*/\s*(-?\d+)\s*$")
_ANGLE_BRACKETS = str.maketrans({"<": "(", ">": ")", "⟨": "(", "⟩": ")"})


class _Unreadable(Exception):
    pass


def _entry_value(entry):
    """A canonical answer entry as an exact SymPy number."""
    if isinstance(entry, bool):
        raise ValueError(f"matrix entry {entry!r} is not a number")
    if isinstance(entry, int):
        return sympy.Integer(entry)
    if isinstance(entry, Fraction):
        return sympy.Rational(entry.numerator, entry.denominator)
    if isinstance(entry, dict) and set(entry) == {"p", "q"}:
        return sympy.Rational(entry["p"], entry["q"])
    if isinstance(entry, float):
        return sympy.Rational(repr(entry))
    if isinstance(entry, sympy.Basic) and entry.is_number:
        return entry
    if isinstance(entry, str):
        value = read_value(entry)
        if value is not None:
            return value
    raise ValueError(f"matrix entry {entry!r} is not an exact number")


def answer_rows(answer) -> list[list]:
    """`question.answer` as a rectangular list of rows of SymPy numbers. A flat list is one row."""
    if isinstance(answer, dict) and "rows" in answer:
        answer = answer["rows"]
    if not isinstance(answer, (list, tuple)) or not answer:
        raise ValueError(f"matrix answer {answer!r} is not a list of rows")
    if not isinstance(answer[0], (list, tuple)):
        answer = [answer]
    rows = [[_entry_value(e) for e in row] for row in answer]
    if len({len(r) for r in rows}) != 1 or not rows[0]:
        raise ValueError(f"matrix answer {answer!r} is not rectangular")
    return rows


def _top_level_groups(text: str) -> list[str] | None:
    """The bracketed groups `text` is made of at its top level ("[1, 2], [3, 4]"), or None when it has
    anything but groups and the commas or spaces between them."""
    groups, depth, start, between = [], 0, None, []
    for i, ch in enumerate(text):
        if ch in "([":
            if depth == 0:
                start = i
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth < 0:
                raise _Unreadable
            if depth == 0:
                groups.append(text[start + 1 : i])
        elif depth == 0:
            between.append(ch)
    if depth != 0:
        raise _Unreadable
    if not groups or "".join(between).replace(",", "").replace(";", "").strip():
        return None
    return groups


def _split_entries(row: str) -> list[str]:
    row = re.sub(r"\s*/\s*", "/", row.strip())
    if "," in row:
        pieces = [p.strip() for p in row.split(",")]
    else:
        pieces = row.split()
    if not pieces or any(not p for p in pieces):
        raise _Unreadable
    return pieces


def _strip_enclosing(text: str) -> str:
    """`text` without one pair of brackets that encloses all of it."""
    text = text.strip()
    if len(text) >= 2 and text[0] in "([" and text[-1] in ")]":
        depth = 0
        for i, ch in enumerate(text):
            depth += ch in "(["
            depth -= ch in ")]"
            if depth == 0 and i < len(text) - 1:
                return text
        return text[1:-1].strip()
    return text


def _rows_text(body: str) -> list[list[str]]:
    body = body.strip()
    if not body:
        raise _Unreadable
    inner = _strip_enclosing(body)
    groups = _top_level_groups(inner) if any(c in inner for c in "([") else None
    if groups is not None and len(groups) > 1:
        return [_split_entries(g) for g in groups]
    if groups is not None and len(groups) == 1 and inner.strip()[0] in "([":
        # One bracketed group inside the outer pair: "[[1, 2, 3]]", a single row.
        inner = groups[0]
    lines = [line for line in re.split(r"\s*[;\n]\s*", inner) if line.strip()]
    return [_split_entries(line) for line in lines]


def _scale(text: str) -> tuple[str, sympy.Basic]:
    """`text` without a number written in front of the matrix or a divisor after it, and that factor."""
    factor = sympy.Integer(1)
    prefix = _SCALAR_PREFIX_RE.match(text)
    if prefix:
        written = prefix.group(1).strip()
        if written == "-":
            value = sympy.Integer(-1)
        else:
            value = read_value(written.strip("()"))
        if value is None or value == 0:
            raise _Unreadable
        factor, text = value, text[prefix.end() :]
    suffix = _DIVISOR_SUFFIX_RE.search(text)
    if suffix:
        divisor = int(suffix.group(1))
        if divisor == 0:
            raise _Unreadable
        factor, text = factor / divisor, text[: suffix.start()]
    return text, factor


def read_matrix(text: str) -> list[list] | None:
    """A reply as a rectangular list of rows of exact SymPy numbers, or None if it cannot be read."""
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_MATRIX_LENGTH:
        return None
    text = _LATEX_ENV_RE.sub(" ", text)
    text = _LATEX_FRAC_RE.sub(r"(\1)/(\2)", text)
    text = text.replace("\\\\", ";").replace("&", ",").replace("\\cdot", "*").replace("\\times", "×")
    text = text.translate(_ANGLE_BRACKETS).strip().rstrip(".")
    text = _LABEL_RE.sub("", text, count=1)
    try:
        text, factor = _scale(text)
        rows = _rows_text(text)
    except _Unreadable:
        return None
    if len(rows) > MAX_DIMENSION or any(len(r) > MAX_DIMENSION for r in rows) or len({len(r) for r in rows}) != 1:
        return None
    values = []
    for row in rows:
        read_row = []
        for entry in row:
            value = read_value(entry)
            if value is None:
                return None
            read_row.append(value * factor if factor != 1 else value)
        values.append(read_row)
    return values


def _oriented(given: list[list], expected: list[list]) -> list[list] | None:
    """`given` in `expected`'s shape: as it is, or, for a vector, its entries in the other orientation."""
    if len(given) == len(expected) and len(given[0]) == len(expected[0]):
        return given
    is_vector = len(expected) == 1 or len(expected[0]) == 1
    flat = [v for row in given for v in row]
    if is_vector and (len(given) == 1 or len(given[0]) == 1) and len(flat) == len(expected) * len(expected[0]):
        return [flat] if len(expected) == 1 else [[v] for v in flat]
    return None


def _multiple_of(given: list, expected: list) -> bool:
    """Whether the vector `given` is a non-zero multiple of the non-zero vector `expected`."""
    pivot = next((i for i, e in enumerate(expected) if not same_number(e, 0)), None)
    if pivot is None or same_number(given[pivot], 0):
        return False
    ratio = given[pivot] / expected[pivot]
    return all(same_number(g, ratio * e) for g, e in zip(given, expected))


def matrices_match(given: list[list], expected: list[list], form: str | None = None) -> bool:
    shaped = _oriented(given, expected)
    if shaped is None:
        return False
    if form == "scalar_multiple":
        return _multiple_of([v for row in shaped for v in row], [v for row in expected for v in row])
    if form == "column_scalar_multiple":
        return all(
            _multiple_of([row[c] for row in shaped], [row[c] for row in expected]) for c in range(len(expected[0]))
        )
    return all(same_number(g, e) for grow, erow in zip(shaped, expected) for g, e in zip(grow, erow))


def _entry_text(value) -> str:
    if isinstance(value, sympy.Rational):
        return str(value)
    return str(value).replace("sqrt", "√").replace("**", "^").replace("*", "")


def matrix_text(rows) -> str:
    """The canonical spelling of a matrix answer: "[[1, -2], [3/5, 1]]"."""
    rows = answer_rows(rows)
    return "[" + ", ".join("[" + ", ".join(_entry_text(v) for v in row) + "]" for row in rows) + "]"


def vector_text(entries) -> str:
    """The canonical spelling of a vector answer: "(1, -2, 3)"."""
    values = [v for row in answer_rows(list(entries)) for v in row]
    return "(" + ", ".join(_entry_text(v) for v in values) + ")"


def matrix_latex(rows) -> str:
    """A matrix answer in LaTeX: a pmatrix."""
    rows = answer_rows(rows)
    body = r" \\ ".join(" & ".join(sympy.latex(v) for v in row) for row in rows)
    return r"\begin{pmatrix} " + body + r" \end{pmatrix}"


@_multi_part_aware
def grade_matrix(question, given: str) -> GradeResult:
    canonical = getattr(question, "answer_text", "")
    expected = answer_rows(question.answer)  # a malformed answer is a generator bug: `grade` logs it
    form = getattr(question, "answer_form", None)
    if form is not None and form not in MATRIX_FORMS:
        raise ValueError(f"answer_form {form!r} does not apply to a matrix answer")
    read = read_matrix(given if isinstance(given, str) else str(given))
    if read is None:
        return GradeResult(correct=False, canonical=canonical)
    normalised = "[" + ", ".join("[" + ", ".join(_entry_text(v) for v in row) + "]" for row in read) + "]"
    return GradeResult(correct=matrices_match(read, expected, form), canonical=canonical, given_normalised=normalised)


__all__ = ["grade_matrix", "matrices_match", "matrix_latex", "matrix_text", "read_matrix", "vector_text"]
