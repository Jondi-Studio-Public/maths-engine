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
    "thirteenth": 13,
    "thirteenths": 13,
    "fourteenth": 14,
    "fourteenths": 14,
    "fifteenth": 15,
    "fifteenths": 15,
    "sixteenth": 16,
    "sixteenths": 16,
    "seventeenth": 17,
    "seventeenths": 17,
    "eighteenth": 18,
    "eighteenths": 18,
    "nineteenth": 19,
    "nineteenths": 19,
    "twentieth": 20,
    "twentieths": 20,
    "thirtieth": 30,
    "thirtieths": 30,
    "fortieth": 40,
    "fortieths": 40,
    "fiftieth": 50,
    "fiftieths": 50,
    "sixtieth": 60,
    "sixtieths": 60,
    "seventieth": 70,
    "seventieths": 70,
    "eightieth": 80,
    "eightieths": 80,
    "ninetieth": 90,
    "ninetieths": 90,
    "hundredth": 100,
    "hundredths": 100,
    "thousandth": 1000,
    "thousandths": 1000,
}
NUMBER_WORDS = set(UNITS) | set(TEENS) | set(TENS) | set(SCALES) | set(DENOMINATORS) | {"point"}

_DIGIT_RE = re.compile(r"^-?(?:\d+(?:\.\d+)?|\.\d+)(/\d+)?$")
#: Longest run of digits read as one number. Nobody says (or answers with) a 300-digit number, and Python
#: refuses to turn 4300 digits into an integer at all: that raised `ValueError` out of the transcript path.
_MAX_NUMBER_DIGITS = 300
#: Longest run of scale words ("hundred hundred hundred ...") that is read at all. A longer run is not a number
#: anyone says, so it is dropped before reading rather than turned into an enormous product.
_MAX_SCALE_RUN = 12


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
    # Keep the sign on digit negatives ("-5" -> "minus 5", "-.5" -> "minus .5") before hyphens are
    # split for word numbers ("forty-two" -> "forty two").
    text = re.sub(r"-(?=\.?\d)", " minus ", text)
    text = text.replace("-", " ")
    return re.findall(r"\d+(?:\.\d+)?(?:/\d+)?|\.\d+|[a-z']+", text)


def _drop_long_scale_runs(tokens: list) -> list:
    """`tokens` without any run of more than `_MAX_SCALE_RUN` consecutive scale words, dropped whole, in one pass."""
    out = []
    start = None
    for k, token in enumerate(tokens + [None]):
        if token in SCALES:
            if start is None:
                start = k
            continue
        if start is not None:
            if k - start <= _MAX_SCALE_RUN:
                out.extend(tokens[start:k])
            start = None
        if token is not None:
            out.append(token)
    return out


def _decimal_digits(tokens: list, j: int):
    """The digits after a spoken "point" (tokens[j:]): digit words ("one four", "oh five") or a run of digits.
    Returns (digit string, next index), or None when no digit follows."""
    n = len(tokens)
    digits = ""
    while j < n and len(digits) < _MAX_NUMBER_DIGITS:
        t = tokens[j]
        if t in UNITS:
            digits += str(UNITS[t])
        elif t.isdigit():
            digits += t
        else:
            break
        j += 1
    if not digits:
        return None
    return digits, j


def _consume_word_number(tokens: list, i: int):
    """Parse one spoken number starting at tokens[i]. Returns (Fraction, next_index) or None.

    One number is a well-formed compound and no more: "forty two", "a hundred and forty four", "one thousand
    and five", "twelve hundred", "thirty six point nine", "point five". Two number words that do not make a
    compound ("twelve twelve", "three four", "three and four") are two numbers: the first is returned and the
    reader starts again at the second. "and" joins only straight after hundred/thousand/million."""
    n = len(tokens)
    total = 0
    current = 0  # the part under a hundred, plus hundreds, of the group being built
    consumed = False
    last = None  # "unit", "zero", "teen", "tens", "hundred" or "scale": what the last word was
    last_scale = None  # the smallest of thousand/million seen so far; the next one must be smaller
    if i < n and tokens[i] == "point":
        decimal = _decimal_digits(tokens, i + 1)
        if decimal is None:
            return None
        digits, j = decimal
        return Fraction(int(digits), 10 ** len(digits)), j
    while i < n:
        t = tokens[i]
        if t == "a":
            # "a hundred", "a thousand": only at the start of a number
            if not consumed and i + 1 < n and tokens[i + 1] in SCALES:
                current = 1
                last = "unit"
                i += 1
                consumed = True
                continue
            break
        if t in UNITS:
            value = UNITS[t]
            if last is None:
                last = "zero" if value == 0 else "unit"
            elif value != 0 and last in ("hundred", "scale", "tens"):
                # "one hundred five", "twenty one"; but not "twenty twenty" or "five five"
                if last == "tens" and current % 10 != 0:
                    break
                last = "unit"
            else:
                break
            current += value
        elif t in TEENS:
            if last not in (None, "hundred", "scale"):
                break
            current += TEENS[t]
            last = "teen"
        elif t in TENS:
            if last not in (None, "hundred", "scale"):
                break
            current += TENS[t]
            last = "tens"
        elif t == "hundred":
            if last is None:
                current = 100  # "hundred and four"
            elif last in ("unit", "teen", "tens") and current < 100:
                current *= 100
            else:
                break
            last = "hundred"
        elif t in SCALES:  # thousand, million
            scale = SCALES[t]
            if last_scale is not None and scale >= last_scale:
                break
            if last is None:
                current = 1
            elif last in ("zero", "scale"):
                break
            total += current * scale
            current = 0
            last_scale = scale
            last = "scale"
        elif (
            t == "and"
            and last in ("hundred", "scale")
            and i + 1 < n
            and (
                tokens[i + 1] in UNITS and UNITS[tokens[i + 1]] != 0 or tokens[i + 1] in TEENS or tokens[i + 1] in TENS
            )
        ):
            i += 1  # "a hundred and forty four"
            continue
        else:
            break
        i += 1
        consumed = True
    if not consumed:
        return None
    value = Fraction(total + current)
    # Decimal part: "thirty six point nine" -> 36.9
    if i < n and tokens[i] == "point":
        decimal = _decimal_digits(tokens, i + 1)
        if decimal is not None:
            digits, j = decimal
            value = value + Fraction(int(digits), 10 ** len(digits))
            i = j
    return value, i


def _mixed_fraction_part(tokens: list, j: int):
    """The proper fraction of a mixed number, read from tokens[j:] after its "and": "a half", "a quarter",
    "three quarters", "3/4". Returns (Fraction, next_index) or None."""
    n = len(tokens)
    if j >= n:
        return None
    t = tokens[j]
    if "/" in t:
        frac = _number_token_value(t)
        if frac is not None and 0 < frac < 1:
            return frac, j + 1
        return None
    if t == "a" and j + 1 < n and tokens[j + 1] in DENOMINATORS:
        return Fraction(1, DENOMINATORS[tokens[j + 1]]), j + 2
    if _DIGIT_RE.match(t):
        numerator, k = _number_token_value(t), j + 1
    else:
        parsed = _consume_word_number(tokens, j)
        if parsed is None:
            return None
        numerator, k = parsed
    if numerator is None or numerator.denominator != 1 or k >= n or tokens[k] not in DENOMINATORS:
        return None
    denominator = DENOMINATORS[tokens[k]]
    if not 0 < numerator < denominator:
        return None
    return Fraction(numerator, denominator), k + 1


def extract_values(text: str) -> list:
    """Extract all numeric values (as Fractions) from a spoken transcript.

    Handles digit strings ("144", "36.9", ".5", "3/5"), word numbers ("a hundred and forty four",
    "thirty six point nine", "point five"), negatives ("minus three", "negative three"), fractions
    ("three over five", "three fifths", "a half") and mixed numbers ("three and a half"). Number words that
    are not one number ("three and four", "twelve twelve") are separate values."""
    tokens = _drop_long_scale_runs(_tokenize(text))
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
        # Fraction forms following a number: "three over five", "three fifths", "three and a half"
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
        elif i < n and tokens[i] == "and" and value == int(value):
            part = _mixed_fraction_part(tokens, i + 1)
            if part is not None:
                value = value + part[0]
                i = part[1]
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
                parts.append(str(value) if "/" in t or t.startswith(".") else t)
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
#: The tolerance for a key shown to two or more decimal places, or a small non-terminating one: half a unit in
#: the hundredths place. It is the floor: a recogniser that drops the last digit of "zero point zero three"
#: still passes, but "zero" and "zero point zero eight" no longer do for 0.03.
_FINE_TOLERANCE = Fraction(5, 1000)
_COARSE_TOLERANCE = Fraction(51, 1000)


def _exact(expected) -> Fraction:
    """`expected` as the decimal it is shown as (a float by its shortest repr, so 0.03 is 3/100)."""
    return Fraction(repr(expected)) if isinstance(expected, float) else Fraction(expected)


def _decimal_places(expected):
    """The decimal places `expected` is displayed to, or None when it does not end (1/3, 2/7)."""
    exact = _exact(expected)
    denominator = exact.denominator
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    if denominator != 1:
        return None
    k = 0
    while (exact * 10**k).denominator != 1:
        k += 1
    return k


def _tolerance_for(expected) -> Fraction:
    """The absolute tolerance a spoken value gets against `expected`, relative to the key's displayed places.

    A key of 0 or 1 decimal places (12, 7.5) keeps the flat +/-0.051, which is within half a unit there. A key of
    two or more places (0.03, 12.34) gets half a hundredth, +/-0.005. A key that does not end (1/3, 2/3) keeps
    +/-0.051 from one half upwards and gets +/-0.005 below it, where 0.051 would be a tenth of the key or more."""
    places = _decimal_places(expected)
    if places is None:
        return _COARSE_TOLERANCE if abs(_exact(expected)) >= Fraction(1, 2) else _FINE_TOLERANCE
    return _COARSE_TOLERANCE if places <= 1 else _FINE_TOLERANCE


def _close(value, expected) -> bool:
    """Whether the spoken `value` equals `expected` or is within its tolerance."""
    return value == expected or abs(Fraction(value) - _exact(expected)) <= _tolerance_for(expected)


def _matches_number(values: list, expected) -> bool:
    return any(_close(v, expected) for v in values)


# Words and signs that negate or hedge a value ("not twelve", "isn't twelve", "twelve plus or minus one").
# Kept in step with `mathgen.grading.typed._NEGATION_RE`, which cannot be imported here (typed imports this
# module).
_NEGATION_RE = re.compile(
    r"\b(?:not|never|neither|nor|cannot)\b"
    r"|n['\u2019]t\b"
    r"|[\u2260\u2264\u2265\u00b1\u2213]|!=|<(?!-)|(?<![-=])>"
    r"|\+\s*/\s*-"
    r"|\bplus[\s-]+(?:or[\s-]+)?minus\b",
    re.IGNORECASE,
)


def _matches_one_value(values: list, text: str, expected) -> bool:
    """Whether `text` states exactly one value, not negated or hedged, and it matches `expected`.

    A transcript of a single-number answer is one value: "not twelve", "twelve or thirteen" and "twelve and a
    half" are not "twelve", though each holds a twelve. Fillers that are not values ("um twelve", "it's
    twelve", "the answer is twelve") are not numbers and so do not count; "twelve point five" is 12.5."""
    if len(values) != 1 or _NEGATION_RE.search(text):
        return False
    return _matches_number(values, expected)


def _matches_components(values: list, text: str, expected: list, *, ordered: bool) -> bool:
    """Whether `text` states exactly one value per component of `expected`, not negated or hedged, each within
    tolerance: position by position when `ordered` (a pair, an ordered list), in any order otherwise (a set).
    So "fifteen nine" is not (9, 15), and neither is "not nine not fifteen" or a count up to fifteen."""
    if not expected or len(values) != len(expected) or _NEGATION_RE.search(text):
        return False
    if ordered:
        return all(_close(v, e) for v, e in zip(values, expected))
    remaining = list(values)
    for e in expected:
        hit = next((v for v in remaining if _close(v, e)), None)
        if hit is None:
            return False
        remaining.remove(hit)
    return True


def check_answer(question, transcript: str, fallback: str = None) -> bool:
    """Check a spoken transcript against a mathgen Question's structured answer.

    `fallback` is an optional second candidate transcript (for example from a second recogniser),
    tried only if the first does not match: the first transcript is authoritative.

    For a single-number answer (an int, a float or a {"p", "q"} fraction) a transcript must state exactly one
    value and must not negate or hedge it, so "not twelve", "twelve or thirteen" and "twelve and a half" are
    all wrong for 12. A value is within +/-0.051 of the key, or +/-0.005 when the key is shown to two or more
    decimal places (see `_tolerance_for`). An answer of several values (a dict of components, a list) must be
    stated with exactly one value per component, unnegated: in order for a pair or an `ordered_list`, in any
    order for other lists."""
    for text in [t for t in (transcript, fallback) if t]:
        if _check_one(question, text):
            return True
    return False


def _check_one(question, text: str) -> bool:
    answer = question.answer
    values = extract_values(text)

    if isinstance(answer, int):
        return _matches_one_value(values, text, answer)
    if isinstance(answer, float):
        return _matches_one_value(values, text, answer)
    if isinstance(answer, dict):
        keys = set(answer.keys())
        if keys == {"p", "q"}:
            target = Fraction(answer["p"], answer["q"])
            return _matches_one_value(values, text, target)
        # Pairs like {x, y} or {R_A, R_B}: one value per component, in order, none negated or hedged.
        return _matches_components(values, text, list(answer.values()), ordered=True)
    if isinstance(answer, list):
        if all(isinstance(a, (int, float)) for a in answer):
            ordered = getattr(question, "answer_kind", None) == "ordered_list"
            return _matches_components(values, text, answer, ordered=ordered)
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
