"""Reply checks that read the question's wording as well as its key.

`grade` calls `rewrite_reply` and then `refuses` before the kind's own grader runs. Both are narrow on purpose:
each one turns a reply that a learner plausibly types into the form the grader already understands
(`rewrite_reply`), or refuses a reply that states something the question rules out (`refuses`), and says nothing
otherwise. A reply that is only the key's value, with or without the key's own unit, is never touched.

Refusals, by what the question or key says:

* an angle in degrees: no radians, turns, gradians, pi or any other unit with the number; an angle in radians: no
  degree sign or unit (`_angle_unit_conflicts`);
* a decimal answer: no trailing pi (`113.1π` is 355) (`_pi_after_a_decimal`);
* an amount in cents: no leading dollar sign, which means dollars (`_dollar_sign_for_cents`);
* a key in a named currency: no other currency (`_currency_conflicts`);
* a percent sign on an answer that is not a percentage, judged on the sentences that ask, not the whole stem
  (`_unasked_percent`);
* "mixed numeral" / "mixed number": a whole part and a proper fraction in lowest terms (`_not_a_mixed_numeral`);
* "Simplify" / "Expand" with an expanded polynomial key: the question copied back, terms of one kind left
  uncollected, a term left as arithmetic (`45*1`, `90/2`, `x^(5+2)`, `x^8/x`) (`_unsimplified`).

Rewrites, for replies that are right but typed in a way the graders do not read:

* thousands separators in a plain numeric key's reply on the expression kind;
* a root typed twice for a one-root list key, `3, 3` for `x = 3`;
* `%` on each item of a list when the question asks for percentages;
* spaced thousands inside a list whose key has them;
* `*`, `·` between a number and a letter in a pair-kind equation: `y = -9*x + 4`;
* `{q, s}` for a set of single capital letters;
* "1 and 1/2", "1 + 1/2" for a mixed numeral;
* `pi over 4` and `¼π` for a key with pi.
"""

from __future__ import annotations

import re
from fractions import Fraction
from math import gcd

from . import units

_FLAGS = re.IGNORECASE


def _text(question) -> str:
    return str(getattr(question, "text", "") or "")


def _key(question) -> str:
    return str(getattr(question, "answer_text", "") or "")


def _typed():
    from . import typed  # typed imports this module, so it is read when first needed

    return typed


# -- the part of a reply after its last number -----------------------------------------------------------------


def _after_number(given: str) -> str:
    """What the reply writes after its last number, trailing space and full stop off ("" for a bare number)."""
    found = units._after_last_number(given or "")
    return "" if found is None else found[1]


# -- angles ----------------------------------------------------------------------------------------------------

_ASKS_DEGREES_RE = re.compile(
    r"\b(?:(?:in|to|into)\s+degrees?|nearest\s+degree)\b(?!\s*(?:celsius|fahrenheit|kelvin|c\b|f\b))", _FLAGS
)
_ASKS_RADIANS_RE = re.compile(r"\b(?:in|to|into)\s+radians?\b", _FLAGS)
_NOT_AN_ANGLE_RE = re.compile(r"temperature|celsius|fahrenheit|kelvin", _FLAGS)
_DEGREE_WORD_RE = re.compile(r"^(?:[°º˚]|degs?\.?|degrees?)$", _FLAGS)
_RADIAN_WORD_RE = re.compile(r"^(?:rads?|radians?)$", _FLAGS)
_OTHER_ANGLE_WORD_RE = re.compile(r"(?:grads?|gradians?|gons?|turns?|rads?|radians?)\b", _FLAGS)
_PI_WORD_RE = re.compile(r"π|\bpi\b", _FLAGS)
_LEADING_CURRENCY_RE = re.compile(r"^\s*(?:[$£€¥]|[A-Za-z]{1,3}\$)")


# A key that is one number, perhaps with a unit: not "V = 30 - 12t", where the "t" is a variable.
_PLAIN_VALUE_KEY_RE = re.compile(r"^[\s$£€]*-?\d[\d,]*(?:\.\d+)?\s*[^\d=,+*/^]{0,15}$")


def _asked_angle_unit(question, key: str) -> str | None:
    """ "degrees" or "radians" when the stem asks for an angle in one of them and the key is a bare number or
    states that unit; None for anything else (a temperature, a key in pi, a stem that names neither)."""
    text = _text(question)
    if _NOT_AN_ANGLE_RE.search(text) or _PI_WORD_RE.search(key):
        return None
    degrees = list(_ASKS_DEGREES_RE.finditer(text))
    radians = list(_ASKS_RADIANS_RE.finditer(text))
    if not degrees and not radians:
        return "degrees" if "°" in key else None
    if degrees and radians:
        return "degrees" if degrees[-1].start() > radians[-1].start() else "radians"
    return "degrees" if degrees else "radians"


def _angle_unit_conflicts(question, given: str) -> bool:
    key = _key(question)
    asked = _asked_angle_unit(question, key)
    after = _after_number(given)
    if asked is None:
        # "7 degrees" or "8 t" for an answer in cm or kg: words the unit table has no id for.
        unit_asked = units._asked_unit(question)
        if (
            unit_asked in (None, "celsius", "fahrenheit", "kelvin")
            or unit_asked.startswith("unlisted:")
            or _PLAIN_VALUE_KEY_RE.match(key) is None
        ):
            return False
        return bool(_DEGREE_WORD_RE.match(after) or _RADIAN_WORD_RE.match(after) or after.lower() == "t")
    if _PI_WORD_RE.search(after) and asked == "degrees":
        return True  # "80.3π" for an angle in degrees
    if after:
        if asked == "degrees" and (_RADIAN_WORD_RE.match(after) or _OTHER_ANGLE_WORD_RE.match(after)):
            return True
        if asked == "radians" and (_DEGREE_WORD_RE.match(after) or _OTHER_ANGLE_WORD_RE.match(after)):
            return not _RADIAN_WORD_RE.match(after)
    unit = units.split_unit(given)[1] or units._compose(after)
    if unit is not None and unit != "%":
        return True  # a length, a mass, a currency, a temperature: not an angle
    if after and re.fullmatch(r"°\s*[CFK]", after, _FLAGS):
        return True
    return False


def _pi_after_a_decimal(question, given: str) -> bool:
    """ "113.1π" for the number 113.1: the number is read without the pi, which makes it 355."""
    key = _key(question)
    return bool(_PI_WORD_RE.search(_after_number(given)) and not _PI_WORD_RE.search(key))


# -- money ------------------------------------------------------------------------------------------------------

_CURRENCY_PREFIXES = (
    ("AUD", re.compile(r"^\s*(?:A\$|AU\$|AUD)", re.IGNORECASE)),
    ("USD", re.compile(r"^\s*(?:US\$|USD)", re.IGNORECASE)),
    ("NZD", re.compile(r"^\s*(?:NZ\$|NZD)", re.IGNORECASE)),
    ("GBP", re.compile(r"^\s*(?:£|GBP)", re.IGNORECASE)),
    ("EUR", re.compile(r"^\s*(?:€|EUR)", re.IGNORECASE)),
    ("JPY", re.compile(r"^\s*(?:¥|JPY)", re.IGNORECASE)),
    ("$", re.compile(r"^\s*\$")),
)
_CURRENCY_WORDS = (
    ("AUD", re.compile(r"\b(?:AUD|australian\s+dollars?)\b", re.IGNORECASE)),
    ("USD", re.compile(r"\b(?:USD|us\s+dollars?|american\s+dollars?)\b", re.IGNORECASE)),
    ("NZD", re.compile(r"\b(?:NZD|new\s+zealand\s+dollars?)\b", re.IGNORECASE)),
    ("GBP", re.compile(r"\b(?:GBP|pounds?(?:\s+sterling)?|quid)\b", re.IGNORECASE)),
    ("EUR", re.compile(r"\b(?:EUR|euros?)\b", re.IGNORECASE)),
    ("JPY", re.compile(r"\b(?:JPY|yen)\b", re.IGNORECASE)),
    ("$", re.compile(r"\bdollars?\b", re.IGNORECASE)),
)
_DOLLAR_CURRENCIES = frozenset({"AUD", "USD", "NZD", "$"})


def _currency_of(text: str, *, key: bool) -> str | None:
    """The currency a key or reply writes: a prefix ("A$", "£") or, in a reply, a word after the number."""
    for name, pattern in _CURRENCY_PREFIXES:
        if pattern.match(text or ""):
            return name
    if key:
        return None
    after = _after_number(text)
    for name, pattern in _CURRENCY_WORDS:
        if after and pattern.search(after):
            return name
    return None


def _currency_conflicts(question, given: str) -> bool:
    """A reply that names a currency other than the key's: "US$138" or "138 pounds" for "A$138.00". A bare number,
    and the key's own currency, are fine; so is a plain "$" or "dollars" for any key in a dollar."""
    wanted = _currency_of(_key(question), key=True)
    if wanted is None or wanted == "$":
        return False
    stated = _currency_of(given, key=False)
    if stated is None or stated == wanted:
        return False
    return not (stated == "$" and wanted in _DOLLAR_CURRENCIES)


def _dollar_sign_for_cents(question, given: str) -> bool:
    return bool(units._asked_unit(question) == "cent" and re.match(r"\s*\$", given or ""))


# -- percent ----------------------------------------------------------------------------------------------------

_PERCENT_WORD_RE = re.compile(r"%|per[ \t]?cent", _FLAGS)
_PERCENT_SIGN_END_RE = re.compile(r"(?<=\d)[ \t]*(?:%|per[ \t]?cent)\s*$", _FLAGS)
_ASKING_ANYWHERE_RE = re.compile(
    r"\b(?:find|calculate|determine|work\s+out|convert|express|estimate|state|give|write|what|how|which|"
    r"hence|round)\b",
    _FLAGS,
)


def _asking_text(question) -> str:
    """The sentences of the stem that ask for the answer; the last sentence when none is recognised."""
    sentences = [s.strip() for s in units._SENTENCE_RE.findall(_text(question)) if s.strip()]
    asking = [s for s in sentences if s.endswith("?") or _ASKING_ANYWHERE_RE.search(s)]
    return " ".join(asking) if asking else (sentences[-1] if sentences else "")


def _unasked_percent(question, given: str) -> bool:
    """ "5%" for the number 5 when the sentences that ask for the answer do not ask for a percentage."""
    if _PERCENT_SIGN_END_RE.search(given or "") is None:
        return False
    if _PERCENT_WORD_RE.search(_key(question)):
        return False
    return _PERCENT_WORD_RE.search(_asking_text(question)) is None


# -- a decimal where a fraction is asked ------------------------------------------------------------------------

_FRACTION_KEY_RE = re.compile(r"^\s*-?\d+\s*/\s*\d+\s*$")
_ASKS_FOR_A_FRACTION_RE = re.compile(r"\bas an? (?:common |simple |proper )?fraction\b|\bwhich fraction\b", _FLAGS)
_PLAIN_DECIMAL_REPLY_RE = re.compile(r"^\s*[+-]?\d*\.\d+\s*$")


def _decimal_for_a_fraction(question, given: str) -> bool:
    """A decimal reply (0.89) where the stem asks for a fraction and the key is one (89/100): the skill is the conversion.
    Only reached for a `number` grader, which would otherwise take the decimal by value."""
    if _PLAIN_DECIMAL_REPLY_RE.match(given or "") is None or _FRACTION_KEY_RE.match(_key(question)) is None:
        return False
    return _ASKS_FOR_A_FRACTION_RE.search(_asking_text(question)) is not None


# -- mixed numerals -----------------------------------------------------------------------------------------------

_MIXED_KEY_RE = re.compile(r"^\s*-?\s*\d+\s+\d+\s*/\s*\d+\s*$")
_MIXED_WORDS_RE = re.compile(r"mixed\s+(?:numeral|number)", _FLAGS)
_MIXED_REPLY_RE = re.compile(r"(?<![\d/.])\d+[ \t]+(\d+)[ \t]*/[ \t]*(\d+)(?![\d/.])")
_MIXED_AND_RE = re.compile(r"^(\s*-?\s*\d+)\s*(?:\band\b|\+)\s*(\d+\s*/\s*\d+\s*)$", _FLAGS)


def _asks_for_a_mixed_numeral(question) -> bool:
    return bool(_MIXED_WORDS_RE.search(_asking_text(question)) and _MIXED_KEY_RE.match(_key(question)))


def _mixed_numeral_as_written(given: str) -> str:
    """ "1 and 1/2" and "1 + 1/2" as "1 1/2"; "1½" as "1 1/2"."""
    typed = _typed()
    return typed._VULGAR_FRACTION_RE.sub(typed._vulgar_fraction, _MIXED_AND_RE.sub(r"\1 \2", given or ""))


def _not_a_mixed_numeral(question, given: str) -> bool:
    """For a question that asks for a mixed numeral: a whole part and a proper fraction in lowest terms."""
    if not _asks_for_a_mixed_numeral(question):
        return False
    found = _MIXED_REPLY_RE.search(given or "")
    if found is None:
        return True  # "3/2", "1.5", "2"
    numerator, denominator = int(found.group(1)), int(found.group(2))
    return denominator == 0 or numerator >= denominator or gcd(numerator, denominator) != 1


# -- simplify / expand ------------------------------------------------------------------------------------------

_SIMPLIFY_RE = re.compile(r"\b(?:simplify|expand|collect)\b", _FLAGS)
_FACTORISE_RE = re.compile(r"\bfactori[sz]e", _FLAGS)
_SUPERS = str.maketrans({"²": "^2", "³": "^3", "×": "*", "·": "*", "⋅": "*", "−": "-", "–": "-", "—": "-", "÷": "/"})
_EXPONENT_GROUP_RE = re.compile(r"\^\s*\(([^()]*)\)")
_PLAIN_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_NUMBER_FRACTION_RE = re.compile(r"[+-]?\s*\(?\s*(\d+)\s*/\s*(\d+)\s*\)?")


def _top_level_terms(text: str) -> list[str]:
    """`text` split at the + and - that are not inside a bracket, after a power sign or at the start."""
    terms, depth, current, previous = [], 0, [], ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth == 0 and ch in "+-" and previous and previous not in "^*/(+-":
            terms.append("".join(current))
            current = []
        current.append(ch)
        if not ch.isspace():
            previous = ch
    terms.append("".join(current))
    return [t.strip() for t in terms if t.strip()]


def _is_unreduced_or_arithmetic(inner: str) -> bool:
    """Whether the text of an exponent or of a numeric term is arithmetic left undone: "5+2", "14/2", "2*3"."""
    inner = inner.strip().strip("()").strip()
    if re.search(r"[A-Za-zπ√]", inner):
        return False
    if re.fullmatch(r"[+-]?\s*\d+(?:\.\d+)?", inner):
        return False
    found = re.fullmatch(r"[+-]?\s*(\d+)\s*/\s*(\d+)", inner)
    if found is not None:
        return int(found.group(2)) == 1 or gcd(int(found.group(1)), int(found.group(2))) != 1
    return bool(re.search(r"\d\s*[-+*/^]\s*[\d(]", inner))


#: A longer reply is left to the kind's own grader: the term checks below are for a line of working, not a page.
_MAX_SIMPLIFY_REPLY = 300


def _asks_to_simplify(question) -> bool:
    if getattr(question, "answer_form", None) not in (None, "expanded", "single_fraction"):
        return False
    text = _text(question)
    return _SIMPLIFY_RE.search(text) is not None and _FACTORISE_RE.search(text) is None


def _key_is_expanded_polynomial(question) -> bool:
    """Whether the key, as sympy reads it, is a polynomial that is already expanded."""
    import sympy

    try:
        expected = _typed()._expected_expression(question)
        return (
            expected is not None
            and sympy.expand(expected) == expected
            and expected.is_polynomial(*expected.free_symbols)
        )
    except Exception:  # noqa: BLE001 - a key this cannot read is not judged
        return False


def _normalised_for_copy(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").translate(_SUPERS)).replace("**", "^").lower()


def _unsimplified(question, given: str) -> bool:
    """A reply to "Simplify"/"Expand" that is the stem's own expression, or (for an expanded polynomial key) leaves
    like terms uncollected, or a number or exponent unworked. Only ever a refusal: the grader still has to accept."""
    if len(given) > _MAX_SIMPLIFY_REPLY or not _asks_to_simplify(question):
        return False
    typed = _typed()
    reply = _normalised_for_copy(given)
    key = _normalised_for_copy(_key(question))
    if not reply or reply == key:
        return False
    if len(reply) >= 5 and re.search(r"[-+*/^]", reply) and reply in _normalised_for_copy(_text(question)):
        return True  # the question copied back
    if not _key_is_expanded_polynomial(question):
        return False
    text = given.translate(_SUPERS).replace("**", "^")
    for terms_text in (text,):
        terms = _top_level_terms(typed._strip_label(terms_text))
        if len(terms) > 8:
            return False
        monomials = []
        for term in terms:
            body = term.lstrip("+- ").strip()
            for inner in _EXPONENT_GROUP_RE.findall(body):
                if _is_unreduced_or_arithmetic(inner):
                    return True
            outside = _EXPONENT_GROUP_RE.sub("", body)
            if re.search(r"([A-Za-z])(?:\^[^*/+\-]+)?\s*\*\s*\1(?![A-Za-z])", outside):
                return True  # x^5*x^2
            if re.search(r"/\s*\(?[0-9.]*[A-Za-z]", outside) and not re.search(r"[A-Za-z]\s*\(", outside):
                return True  # x^8/x
            if not re.search(r"[A-Za-zπ√]", body) and _is_unreduced_or_arithmetic(body):
                return True  # 45*1, 90/2
            parsed = typed._parse_expression(term)
            if parsed is None:
                return False
            if parsed == 0 and len(terms) > 1:
                return True  # + 0
            coefficient, monomial = parsed.as_coeff_Mul()
            monomials.append(monomial)
        if len(set(monomials)) != len(monomials):
            return True
    return False


# -- the entry points -------------------------------------------------------------------------------------------

_NUMERIC_KEY_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")


def refuses(kind: str, question, given: str) -> bool:
    """Whether the question's wording rules the reply out, whatever the kind's own grader would say."""
    if not isinstance(given, str):
        return False
    if kind in ("number", "fraction", "expression"):
        if _angle_unit_conflicts(question, given):
            return True
        if _currency_conflicts(question, given) or _dollar_sign_for_cents(question, given):
            return True
        if _unasked_percent(question, given):
            return True
    if kind == "number" and _decimal_for_a_fraction(question, given):
        return True
    if kind in ("number", "fraction") and _pi_after_a_decimal(question, given):
        return True
    if kind in ("fraction", "number") and _not_a_mixed_numeral(question, given):
        return True
    if kind == "expression" and _unsimplified(question, given):
        return True
    return False


def rewrite_reply(kind: str, question, given: str) -> str:
    """`given` in the form its grader reads, where a plausible spelling is one the grader does not know."""
    if not isinstance(given, str) or not given:
        return given
    typed = _typed()
    key = _key(question)
    if kind == "expression" and _NUMERIC_KEY_RE.match(key):
        given = _SPACE_OR_COMMA_THOUSANDS(given)
    if kind == "expression" and _PI_WORD_RE.search(key):
        given = re.sub(r"(?<!\w)(\w+)\s+over\s+(\w+)", r"\1/\2", given, flags=_FLAGS)
        given = re.sub(
            r"([¼½¾⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞])\s*(π|pi)\b",
            lambda m: "(" + str(Fraction(typed._VULGAR_FRACTIONS[m.group(1)])) + ")" + m.group(2),
            given,
            flags=_FLAGS,
        )
        given = re.sub(r"(?<=[\w)])\s+rads?\b\.?\s*$", " rad", given, flags=_FLAGS)
    if kind in ("list", "ordered_list", "pair", "set") and "%" in given:
        if _PERCENT_WORD_RE.search(_asking_text(question)) and not _PERCENT_WORD_RE.search(key):
            given = given.replace("%", "")
    if kind == "list":
        given = _collapse_repeated_root(question, given)
    if kind in ("list", "ordered_list"):
        if re.search(r"\d[, ]\d{3}", key):
            given = typed._SPACE_GROUP_RE.sub(lambda m: re.sub(r"\s", "", m.group(0)), given)
    if kind == "pair" and typed._EQUATION_ANSWER_RE.match(key):
        given = re.sub(r"(?<=[\d)])\s*[*·⋅×]\s*(?=[A-Za-z(])|(?<=[A-Za-z])\s*[*·⋅]\s*(?=[A-Za-z])", "", given)
    if kind == "set":
        expected = getattr(question, "answer", None)
        if (
            isinstance(expected, (list, tuple))
            and expected
            and all(isinstance(e, str) and len(e) == 1 and e.isalpha() for e in expected)
        ):
            given = given.upper() if all(e.isupper() for e in expected) else given
    if kind in ("fraction", "number") and _asks_for_a_mixed_numeral(question):
        given = _mixed_numeral_as_written(given)
    return given


def _SPACE_OR_COMMA_THOUSANDS(given: str) -> str:
    typed = _typed()
    text = typed._strip_comma_groups(given)
    return typed._SPACE_GROUP_RE.sub(lambda m: re.sub(r"[^\d.]", "", m.group(0)), text)


_ROOT_SEPARATOR_RE = re.compile(r"\s*(?:,|;|\bor\b|\band\b)\s*", _FLAGS)


def _collapse_repeated_root(question, given: str) -> str:
    """ "x = 3 or x = 3", "3, 3" as "x = 3" when the key has the one value: a double root typed twice."""
    key = _key(question)
    if _ROOT_SEPARATOR_RE.search(key):
        return given
    typed = _typed()
    items = [typed._strip_label(item) for item in _ROOT_SEPARATOR_RE.split(given) if item.strip()]
    if len(items) < 2:
        return given
    squeezed = {re.sub(r"\s+", "", item).lower() for item in items}
    return items[0] if len(squeezed) == 1 else given
