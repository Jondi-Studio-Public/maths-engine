"""Normalise ASR transcripts of spoken maths answers and check them against mathgen answers.

A transcript is never turned into values by a model: the raw transcript is parsed here, deterministically,
and compared against the structured `Question.answer` with SymPy where needed.
"""

import re
from fractions import Fraction

UNITS = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000}
DENOMINATORS = {
    "half": 2,
    "halves": 2,
    "third": 3,
    "thirds": 3,
    "quarter": 4,
    "quarters": 4,
    "fourth": 4,
    "fourths": 4,
    "fifth": 5,
    "fifths": 5,
    "sixth": 6,
    "sixths": 6,
    "seventh": 7,
    "sevenths": 7,
    "eighth": 8,
    "eighths": 8,
    "ninth": 9,
    "ninths": 9,
    "tenth": 10,
    "tenths": 10,
    "eleventh": 11,
    "elevenths": 11,
    "twelfth": 12,
    "twelfths": 12,
}
NUMBER_WORDS = set(UNITS) | set(TEENS) | set(TENS) | set(SCALES) | set(DENOMINATORS) | {"point"}

_DIGIT_RE = re.compile(r"^-?\d+(\.\d+)?(/\d+)?$")
#: Longest run of digits read as one number. Nobody says (or answers with) a 300-digit number, and Python
#: refuses to turn 4300 digits into an integer at all: that raised `ValueError` out of the transcript path.
_MAX_NUMBER_DIGITS = 300


def _number_token_value(token: str):
    """The `Fraction` a digit token stands for, or None when it is not a number anyone could mean: too many
    digits, or a zero denominator (`1/0`, `0/0`, `5/00`)."""
    if len(token) > _MAX_NUMBER_DIGITS or _DIGIT_RE.match(token) is None:
        return None
    try:
        return Fraction(token)
    except ZeroDivisionError:
        return None


def _is_number_token(token: str) -> bool:
    return _number_token_value(token) is not None


def _is_zero_denominator(token: str) -> bool:
    """`1/0`, `0/0`, `5/00`: a fraction written over zero."""
    if "/" not in token or len(token) > _MAX_NUMBER_DIGITS or _DIGIT_RE.match(token) is None:
        return False
    return int(token.rsplit("/", 1)[1]) == 0


def _tokenize(text: str) -> list:
    text = text.lower()
    # Keep the sign on digit negatives ("-5" -> "minus 5") before hyphens are
    # split for word numbers ("forty-two" -> "forty two").
    text = re.sub(r"-(?=\d)", " minus ", text)
    text = text.replace("-", " ")
    return re.findall(r"\d+(?:\.\d+)?(?:/\d+)?|[a-z']+", text)


def _consume_word_number(tokens: list, i: int):
    """Parse a run of number words starting at tokens[i]. Returns (Fraction, next_index) or None."""
    total = 0
    current = 0
    consumed = False
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t == "a" and i + 1 < n and tokens[i + 1] in SCALES:
            current = 1
            i += 1
            consumed = True
        elif t in UNITS:
            current += UNITS[t]
            i += 1
            consumed = True
        elif t in TEENS:
            current += TEENS[t]
            i += 1
            consumed = True
        elif t in TENS:
            current += TENS[t]
            i += 1
            consumed = True
        elif t in SCALES:
            if not consumed:
                current = 1  # "hundred and four"
            if t == "hundred":
                current = (current or 1) * 100
            else:
                total += (current or 1) * SCALES[t]
                current = 0
            i += 1
            consumed = True
        elif (
            t == "and"
            and consumed
            and i + 1 < n
            and (tokens[i + 1] in UNITS or tokens[i + 1] in TEENS or tokens[i + 1] in TENS)
        ):
            i += 1  # "a hundred and forty four"
        else:
            break
    if not consumed:
        return None
    value = Fraction(total + current)
    # Decimal part: "thirty six point nine" -> 36.9
    if i < n and tokens[i] == "point":
        j = i + 1
        digits = ""
        while j < n and tokens[j] in UNITS and len(digits) < _MAX_NUMBER_DIGITS:
            digits += str(UNITS[tokens[j]])
            j += 1
        if digits:
            value = value + Fraction(int(digits), 10 ** len(digits))
            i = j
    return value, i


def extract_values(text: str) -> list:
    """Extract all numeric values (as Fractions) from a spoken transcript.

    Handles digit strings ("144", "36.9", "3/5"), word numbers ("a hundred and forty four",
    "thirty six point nine"), negatives ("minus three", "negative three"), and fractions
    ("three over five", "three fifths", "a half").
    """
    tokens = _tokenize(text)
    if any(_is_zero_denominator(t) for t in tokens):
        return []  # "1/0" is not a number, and a reply with one in it has no value to read
    values = []
    negate = False
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t in ("minus", "negative"):
            negate = True
            i += 1
            continue
        value = None
        if _DIGIT_RE.match(t):
            value = _number_token_value(t)
            if value is None:  # too long, or a zero denominator: not a number anyone says, so noise like any other
                negate = False
                i += 1
                continue
            i += 1
        elif t == "a" and i + 1 < n and tokens[i + 1] in DENOMINATORS:
            value = Fraction(1, DENOMINATORS[tokens[i + 1]])
            i += 2
        else:
            parsed = _consume_word_number(tokens, i)
            if parsed is not None:
                value, i = parsed
            else:
                negate = False
                i += 1
                continue
        # Fraction forms following a number: "three over five", "three fifths"
        if i < n and tokens[i] == "over":
            nxt = None
            if i + 1 < n:
                if _is_number_token(tokens[i + 1]):
                    nxt = (_number_token_value(tokens[i + 1]), i + 2)
                else:
                    nxt = _consume_word_number(tokens, i + 1)
            if nxt is not None and nxt[0] != 0:
                value = value / nxt[0]
                i = nxt[1]
        elif i < n and tokens[i] in DENOMINATORS and value == int(value):
            value = Fraction(int(value), DENOMINATORS[tokens[i]])
            i += 1
        if negate:
            value = -value
            negate = False
        values.append(value)
    return values


_SPOKEN_EXPR_MAP = {
    "plus": "+",
    "minus": "-",
    "times": "*",
    "x": "x",
    "y": "y",
    "squared": "**2",
    "cubed": "**3",
}


def spoken_to_expr(text: str):
    """Best-effort parse of a spoken polynomial ("twelve x squared minus four x plus one")
    into a SymPy expression. Returns None if it cannot be parsed."""
    import sympy
    from tokenize import TokenError

    from mathgen.grading.safe_parse import SYMPIFY_TRANSFORMATIONS, parse_expression_safely

    tokens = _tokenize(text)
    parts = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t in ("to", "power") or t == "the" or t == "of":
            # "to the power of five"
            if t == "to" and i + 3 < n and tokens[i + 1] == "the" and tokens[i + 2] == "power":
                j = i + 4 if tokens[i + 3] == "of" else i + 3
                if j < n:
                    exp = None
                    if _is_number_token(tokens[j]):
                        exp, j = int(_number_token_value(tokens[j])), j + 1
                    else:
                        parsed = _consume_word_number(tokens, j)
                        if parsed:
                            exp, j = int(parsed[0]), parsed[1]
                    if exp is not None:
                        parts.append(f"**{exp}")
                        i = j
                        continue
            i += 1
            continue
        if t in _SPOKEN_EXPR_MAP:
            token = _SPOKEN_EXPR_MAP[t]
            if token in ("x", "y") and parts and re.match(r"^-?[\d.]+$", parts[-1]):
                parts.append("*")  # implicit multiplication: "twelve x" -> 12*x
            parts.append(token)
            i += 1
            continue
        if t in ("negative",):
            parts.append("-")
            i += 1
            continue
        if _DIGIT_RE.match(t):
            value = _number_token_value(t)
            if value is not None:  # a number too long to read, or over zero, is skipped
                parts.append(str(value) if "/" in t else t)
            i += 1
            continue
        parsed = _consume_word_number(tokens, i)
        if parsed is not None:
            value, i = parsed
            parts.append(str(value) if value.denominator == 1 else f"({value})")
            continue
        i += 1  # unknown word: skip
    if not parts:
        return None
    expr_str = "".join(p if p.startswith("**") else f" {p} " for p in parts)
    # The text is built from number words, so it is short and plain, but a number can be as large as the
    # speaker likes: "five to the power of fifty million" is `5 **50000000`, which `sympify` would try to
    # compute. It is read by the restricted, cost-checked parser instead, and refused (None) when it is
    # too big to evaluate.
    try:
        return parse_expression_safely(expr_str, transformations=SYMPIFY_TRANSFORMATIONS)
    except (sympy.SympifyError, SyntaxError, TokenError, TypeError, ValueError, IndexError):
        return None  # includes `TooCostly` and anything the token check refuses (both are `ValueError`)


_TOLERANCE = 0.051  # spoken answers are accepted to the same 1 d.p. the questions ask for


def _matches_number(values: list, expected) -> bool:
    expected_f = float(expected)
    for v in values:
        if v == expected:
            return True
        if abs(float(v) - expected_f) <= _TOLERANCE:
            return True
    return False


def check_answer(question, transcript: str, fallback: str = None) -> bool:
    """Check a spoken transcript against a mathgen Question's structured answer.

    `fallback` is an optional second candidate transcript (for example from a second recogniser),
    tried only if the first does not match: the first transcript is authoritative."""
    for text in [t for t in (transcript, fallback) if t]:
        if _check_one(question, text):
            return True
    return False


def _check_one(question, text: str) -> bool:
    answer = question.answer
    values = extract_values(text)

    if isinstance(answer, int):
        return _matches_number(values, answer)
    if isinstance(answer, float):
        return _matches_number(values, answer)
    if isinstance(answer, dict):
        keys = set(answer.keys())
        if keys == {"p", "q"}:
            target = Fraction(answer["p"], answer["q"])
            return _matches_number(values, target)
        # Pairs like {x, y} or {R_A, R_B}: every component must be present.
        remaining = list(values)
        for expected in answer.values():
            hit = next((v for v in remaining if v == expected or abs(float(v) - float(expected)) <= _TOLERANCE), None)
            if hit is None:
                return False
            remaining.remove(hit)
        return True
    if isinstance(answer, list):
        if all(isinstance(a, (int, float)) for a in answer):
            remaining = list(values)
            for expected in answer:
                hit = next(
                    (v for v in remaining if v == expected or abs(float(v) - float(expected)) <= _TOLERANCE), None
                )
                if hit is None:
                    return False
                remaining.remove(hit)
            return True
        return False  # lists of expressions (gradients) are not graded from a transcript yet
    if isinstance(answer, str):
        import sympy

        from mathgen.grading.safe_parse import sympify_safely

        spoken = spoken_to_expr(text)
        if spoken is None:
            return False
        try:
            # `answer` is the generator's own string, but it is read by the restricted parser all the same:
            # nothing in this module hands text to `sympify`.
            expected = sympify_safely(answer)
            return sympy.simplify(spoken - expected) == 0
        except (sympy.SympifyError, TypeError):
            return False
    return False
