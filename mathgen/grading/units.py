"""Units on a typed number: "30 kg" is not an answer to "Convert 3 cm to mm".

The number grader reads the value of a reply and used to drop every word after it, so any unit at all was
accepted. A reply may still give a bare number. When it names a unit this module recognises, that unit has to be
the one the question asks for:

* the unit the canonical answer shows ("F = 60.00 N"), or else
* the unit the question asks for by name ("to mm", "in km/h", "how many kilojoules") in a sentence that asks
  for the answer: one ending in "?" or starting "Find", "Convert", "Calculate" and the like. A unit named in a
  sentence that only describes the data ("the height, in metres, of a drone") says nothing about the answer.

When neither says which unit is wanted, the reply's unit is not judged: "24 m/s" is a fair way to state a rate of
change of a height in metres, and nothing in such a question rules it out.

The degree is deliberately absent: a degree sign is read by the grader itself, and many angle questions never
write a unit. Radians, gradians and turns are listed, so that a reply in one of them is refused for a key in
degrees ("31 radians" for "31.0°") and judged against the unit a question asks for ("to radians").
"""

from __future__ import annotations

import re

#: unit id -> spellings, compared case-insensitively. Each id is one unit; spellings are the ways a learner or a
#: question writes it. Single letters that are also common words or variables ("a", "c", "t", "in") are left out.
_UNITS: dict[str, tuple[str, ...]] = {
    # length
    "mm": ("mm", "millimetre", "millimetres", "millimeter", "millimeters"),
    "cm": ("cm", "centimetre", "centimetres", "centimeter", "centimeters"),
    "m": ("m", "metre", "metres", "meter", "meters"),
    "km": ("km", "kilometre", "kilometres", "kilometer", "kilometers"),
    "inch": ("inch", "inches"),
    "foot": ("ft", "foot", "feet"),
    "yard": ("yd", "yds", "yard", "yards"),
    "mile": ("mi", "mile", "miles"),
    "furlong": ("furlong", "furlongs"),
    # mass
    "mg": ("mg", "milligram", "milligrams"),
    "g": ("g", "gram", "grams", "gramme", "grammes"),
    "kg": ("kg", "kilogram", "kilograms", "kilo", "kilos"),
    "tonne": ("tonne", "tonnes", "ton", "tons"),
    "pound": ("lb", "lbs", "pound", "pounds"),
    "ounce": ("oz", "ounce", "ounces"),
    "stone": ("stone", "stones"),
    # capacity
    "ml": ("ml", "millilitre", "millilitres", "milliliter", "milliliters"),
    "l": ("l", "litre", "litres", "liter", "liters"),
    "kl": ("kl", "kilolitre", "kilolitres", "kiloliter", "kiloliters"),
    "gallon": ("gallon", "gallons"),
    "pint": ("pint", "pints"),
    # area
    "mm2": ("mm2", "mm^2", "mm²", "sq mm", "square millimetres", "square millimeters"),
    "cm2": ("cm2", "cm^2", "cm²", "sq cm", "square centimetres", "square centimeters"),
    "m2": ("m2", "m^2", "m²", "sq m", "square metres", "square meters"),
    "km2": ("km2", "km^2", "km²", "sq km", "square kilometres", "square kilometers"),
    "ha": ("ha", "hectare", "hectares"),
    "acre": ("acre", "acres"),
    # volume
    "mm3": ("mm3", "mm^3", "mm³", "cubic millimetres", "cubic millimeters"),
    "cm3": ("cm3", "cm^3", "cm³", "cc", "cubic centimetres", "cubic centimeters"),
    "m3": ("m3", "m^3", "m³", "cubic metres", "cubic meters"),
    # time
    "ms": ("ms", "millisecond", "milliseconds"),
    "s": ("s", "sec", "secs", "second", "seconds"),
    "min": ("min", "mins", "minute", "minutes"),
    "h": ("h", "hr", "hrs", "hour", "hours"),
    "day": ("day", "days"),
    "week": ("week", "weeks", "wk", "wks"),
    "year": ("year", "years", "yr", "yrs"),
    # speed
    "km/h": ("km/h", "kmh", "kph", "km/hr", "kilometres per hour", "kilometers per hour"),
    "m/s": ("m/s", "metres per second", "meters per second"),
    "mph": ("mph", "miles per hour"),
    "m/s2": (
        "m/s²",
        "m/s^2",
        "m/s2",
        "m s^-2",
        "m/s/s",
        "metres per second squared",
        "meters per second squared",
        "metres per second per second",
        "meters per second per second",
    ),
    # density, concentration, stiffness: a compound unit is also found part by part (see `_compose`)
    "g/cm3": ("g/cm³", "g/cm^3", "g/cm3", "g/cc"),
    "kg/m3": ("kg/m³", "kg/m^3", "kg/m3"),
    "g/ml": ("g/ml",),
    "n/m": ("n/m",),
    # energy
    "j": ("joule", "joules"),  # not a bare "j": "-4j" is a vector component
    "kj": ("kj", "kilojoule", "kilojoules"),
    "mj": ("mj", "megajoule", "megajoules"),
    "w": ("w", "watt", "watts"),
    "kw": ("kw", "kilowatt", "kilowatts"),
    "hz": ("hz", "hertz"),
    "khz": ("khz", "kilohertz"),
    "mhz": ("mhz", "megahertz"),
    "volt": ("volt", "volts"),
    "mol": ("mol", "mole", "moles"),
    "cal": ("cal", "cals", "calorie", "calories", "kcal"),
    "kwh": ("kwh", "kilowatt hour", "kilowatt hours", "kilowatt-hour", "kilowatt-hours"),
    # force, pressure
    "n": ("n", "newton", "newtons"),
    "kn": ("kn", "kilonewton", "kilonewtons"),
    "pa": ("pa", "pascal", "pascals"),
    "kpa": ("kpa", "kilopascal", "kilopascals"),
    "mpa": ("mpa", "megapascal", "megapascals"),
    "gpa": ("gpa", "gigapascal", "gigapascals"),
    # temperature
    "celsius": ("°c", "degrees celsius", "degree celsius", "celsius"),
    "fahrenheit": ("°f", "degrees fahrenheit", "degree fahrenheit", "fahrenheit"),
    "kelvin": ("kelvin", "kelvins"),
    # money
    "dollar": ("$", "dollar", "dollars"),
    "cent": ("cent", "cents"),
    # angle units other than the degree: the degree sign is read by the grader itself, so a degree key or a stem
    # asking "in degrees" has no unit id of its own and refuses these as a different unit (see `_asked_unit`)
    "rad": ("rad", "radian", "radians"),
    "grad": ("grad", "grads", "gradian", "gradians", "gon", "gons"),
    "turn": ("turn", "turns"),
}

_SPELLING_TO_UNIT = {spelling: unit for unit, spellings in _UNITS.items() for spelling in spellings}
# Longest first, so "km/h" is found before "km" and "square metres" before "metres".
_SPELLINGS = sorted(_SPELLING_TO_UNIT, key=len, reverse=True)
_SPELLING_PATTERN = "|".join(re.escape(spelling) for spelling in _SPELLINGS)
# Case-folding turns two different SI prefixes into one: a capital M is mega and a small m is milli, so "ML" is a
# megalitre and "mL" a millilitre. A reply whose case says the other prefix than the table's is a different
# unit, not the table's one, so it gets an id no unit shares (see `_resolve`).
# "MM" is left alone: a megametre is all but never written, and a shouted "30 MM" has always meant millimetres.
_MEGA_WRITTEN_AS_MILLI_RE = re.compile(r"(?<![A-Za-z])M(?:[LlGgSs]|m)(?![A-Za-z])")
_MILLI_WRITTEN_AS_MEGA_RE = re.compile(r"(?<![A-Za-z])m(?:[jJ]|[hH][zZ]|[pP][aA])(?![A-Za-z])")


def _prefix_case_clash(spelling: str) -> bool:
    """Whether `spelling` is written with the case of one M prefix but is the table's spelling of the other:
    a capital M on a milli unit ("ML", "Mg", "Mm") or a small m on a mega unit ("mHz", "mPa", "mJ")."""
    return bool(_MEGA_WRITTEN_AS_MILLI_RE.search(spelling) or _MILLI_WRITTEN_AS_MEGA_RE.search(spelling))


def _resolve(spelling: str) -> str:
    """The unit id of a spelling found in a reply, compared case-insensitively except for the M prefix clash."""
    unit = _SPELLING_TO_UNIT[spelling.lower()]
    return "~" + unit if _prefix_case_clash(spelling) else unit


# A spelling standing on its own: not inside a word ("m" in "mass"), and not followed by a letter or digit that
# would make it a longer word or an exponent ("m" in "m2" is found as "m2").
_MENTION_RE = re.compile(rf"(?<![^\W\d_])(?:{_SPELLING_PATTERN})(?![\w²³'’])", re.IGNORECASE)
# Not a digit written straight after a letter, or a letter and "^": the "2" in "m2" or "m^2" is the unit's power, not the value.
_NUMBER_RE = re.compile(r"(?<![A-Za-z])(?<![A-Za-z]\^)(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
# "to mm", "into grams", "in km/h", "how many kilojoules": the unit the question asks for by name. Not "in year 3"
# (a label, not a unit), not "to the nearest cent" (a precision: the answer is still in dollars), and not "in n = 5"
# or "in g(x)" (a variable or a function).
_ASKED_RE = re.compile(
    rf"\b(?:to|into|in|how\s+many(?:\s+(?:more|fewer|less|extra|further|additional))?)\s+({_SPELLING_PATTERN})(?![\w²³'’^])(?!\s*[\d=(])(?!\s*(?:per\b|[/·⋅×*]))",
    re.IGNORECASE,
)
# "in N/m", "in kL per hour", "in m/s²", "how many cents per 100 g": the question asks for a compound unit that the
# table has no spelling for. Its parts are read so that a reply in some other unit ("45 kg") can still be refused.
_ASKED_COMPOUND_RE = re.compile(
    rf"\b(?:to|into|in|how\s+many(?:\s+(?:more|fewer|less|extra|further|additional))?)\s+({_SPELLING_PATTERN})"
    rf"\s*(?:per\b|[/·⋅×*])\s*(?:\d[\d,]*(?:\.\d+)?\s*)?({_SPELLING_PATTERN})?(?:\^?[23²³])?(?![\w'’])",
    re.IGNORECASE,
)
#: Units that are themselves rates: a reply in one of them is not refused for a compound unit asked by its parts.
_RATE_UNITS = frozenset({"km/h", "m/s", "mph"})
# "Using your speed in m/s": the unit describes a value the learner already has, not the unit being asked for.
_EARLIER_VALUE_RE = re.compile(r"\byour\s+\w+(?:\s+\w+)?\s+$", re.IGNORECASE)
# ... but "Give your answer in dollars" asks for the answer's unit: the verb in front of "your" says so.
_ASKING_VERB_BEFORE_RE = re.compile(
    r"\b(?:give|write|state|express|show|leave|round|present|enter|type)\s+$", re.IGNORECASE
)
# A full stop inside a number ("4.5 seconds") does not end a sentence.
_SENTENCE_RE = re.compile(r"(?:[^.?!]|\.(?=\d))+[.?!]?")
_ASKING_RE = re.compile(
    r"^\s*(?:find|convert|calculate|work\s+out|determine|write|express|give|estimate|state|what|how|"
    r"hence|therefore|thus|then|round|using\s+(?:your|the)|use\s+(?:your|the))\b",
    re.IGNORECASE,
)


#: Capital letters that are a unit only when written, in capitals, after a space that follows the number
#: ("20 J", "300 K", "30 F"): lower case is a variable or a vector component ("-4j"), so these stay out of the
#: case-blind table above.
_CAPITAL_UNITS = {"J": "j", "K": "kelvin", "F": "fahrenheit", "C": "celsius", "V": "volt"}
# Longest text after the number that is read as a unit at all; the longest real one is "metres per second squared".
_MAX_UNIT_TEXT = 40
_EXPONENT_WORDS = (("squared", "2"), ("cubed", "3"))
_SUPERSCRIPTS = {"²": "2", "³": "3"}
# Two units that are one unit when written as a ratio of parts.
_COMPOSED_ALIASES = {"mile/h": "mph"}


def _segment_unit(segment: str) -> str | None:
    """The unit id of one part of a compound unit ("km", "cm^3", "s squared"), or None. An exponent is appended
    to the id ("s2"), which is how the table names "m2" and "cm3" too."""
    text = segment.strip().lower()
    if text in _SPELLING_TO_UNIT:
        return _SPELLING_TO_UNIT[text]
    exponent = ""
    for word, digit in _EXPONENT_WORDS:
        if text.endswith(" " + word):
            text, exponent = text[: -len(word)].strip(), digit
    if not exponent and text[-1:] in _SUPERSCRIPTS:
        text, exponent = text[:-1].strip(), _SUPERSCRIPTS[text[-1]]
    if not exponent and text[-2:-1] == "^" and text[-1:] in ("2", "3"):
        text, exponent = text[:-2].strip(), text[-1]
    unit = _SPELLING_TO_UNIT.get(text)
    return None if unit is None or not exponent else unit + exponent


def _compose(text: str) -> str | None:
    """A compound unit written out of parts ("km/s", "kilometres per hour", "g/cm^3", "metres squared",
    "m s"), as one id such as "km/s", "km/h" or "g/cm3"; None when the text is not made wholly of recognised
    units. The id is the table's own id for a unit the table lists ("km/h", "m/s2"), so the two compare equal."""
    text = text.strip().rstrip(".!").strip()
    if not text or len(text) > _MAX_UNIT_TEXT:
        return None
    clash = _prefix_case_clash(text)
    text = text.lower()
    whole = _segment_unit(text)
    if whole is not None:
        return "~" + whole if clash else whole
    segments = text.replace(" per ", "/").split("/")
    if len(segments) == 1:
        segments = text.split()  # "m s": two units written side by side
        joiner = "*"
    else:
        joiner = "/"
    if len(segments) < 2 or len(segments) > 3:
        return None
    ids = [_segment_unit(segment) for segment in segments]
    if None in ids:
        return None
    composed = joiner.join(ids)  # type: ignore[arg-type]
    composed = _COMPOSED_ALIASES.get(composed, composed)
    return "~" + composed if clash else composed


def _after_last_number(text: str) -> tuple[int, str] | None:
    """Where the last number of `text` ends and what is written after it (trailing space and full stop off)."""
    numbers = list(_NUMBER_RE.finditer(text or ""))
    if not numbers:
        return None
    return numbers[-1].end(), text[numbers[-1].end() :].strip().rstrip(".!").strip()


def _capital_unit(text: str) -> str | None:
    """The unit of a trailing capital letter written after a space ("20 J", "30 F"), or None."""
    found = _after_last_number(text)
    if found is None or not text[found[0] : found[0] + 1].isspace():
        return None
    return _CAPITAL_UNITS.get(found[1])


_LEADING_DOLLAR_RE = re.compile(r"^\s*\$\s*(?=\S)")
_PERCENT_RE = re.compile(r"\s*(?:%|percent)\s*$", re.IGNORECASE)


def split_unit(text: str) -> tuple[str, str | None]:
    """`text` as (the number it states, the recognised unit written with it).

    The unit is a trailing one ("1185/8 m^2", "109.5 m/s", "71 dollars") or a leading "$" ("$71"); the number
    is the rest, unit removed. With no recognised unit the text comes back unchanged with None. A percent
    sign or the word "percent" is taken off too and reported as the unit "%", which is not in the unit table
    and so is never judged against what the question asks for."""
    text = text or ""
    numbers = list(_NUMBER_RE.finditer(text))
    if numbers:
        rest = text[numbers[-1].end() :].strip().rstrip(".!")
        match = _MENTION_RE.match(rest)
        if match is not None and match.end() == len(rest.rstrip()):
            return text[: numbers[-1].end()], _resolve(match.group(0))
        capital = _capital_unit(text)
        if capital is not None:
            return text[: numbers[-1].end()], capital
    dollar = _LEADING_DOLLAR_RE.match(text)
    if dollar is not None:
        return text[dollar.end() :], "dollar"
    percent = _PERCENT_RE.search(text)
    if percent is not None and numbers:
        return text[: percent.start()], "%"
    return text, None


def _unit_of(text: str) -> str | None:
    """The recognised unit written straight after the last number in `text`, or None: one the table lists, or a
    compound of listed units ("km/s", `_compose`). A leading "$" is not judged: "$501" for an amount in cents has
    always been read as the number 501."""
    unit = split_unit(text)[1]
    if unit is None:
        found = _after_last_number(text)
        return None if found is None else _compose(found[1])
    return None if unit in ("%", "dollar") and not _trailing_unit(text) else unit


def _trailing_unit(text: str) -> bool:
    """Whether `text` writes a recognised unit straight after its last number."""
    numbers = list(_NUMBER_RE.finditer(text or ""))
    if not numbers:
        return False
    rest = text[numbers[-1].end() :].strip().rstrip(".!")
    match = _MENTION_RE.match(rest)
    return (match is not None and match.end() == len(rest.rstrip())) or _capital_unit(text) is not None


def _describes_earlier_value(before: str) -> bool:
    """Whether the words in front of an "in <unit>" make it describe a value the learner already has."""
    found = _EARLIER_VALUE_RE.search(before)
    return found is not None and _ASKING_VERB_BEFORE_RE.search(before[: found.start()]) is None


def _asking_matches(question, pattern: re.Pattern) -> list[re.Match]:
    """Every match of `pattern` in a sentence of the question that asks for the answer, in order, leaving out
    an "in <unit>" that describes a value the learner already has."""
    return [
        match
        for sentence in _SENTENCE_RE.findall(getattr(question, "text", "") or "")
        if sentence.rstrip().endswith("?") or _ASKING_RE.match(sentence)
        for match in pattern.finditer(sentence)
        if not _describes_earlier_value(sentence[: match.start()])
    ]


def _asked_unit(question) -> str | None:
    """The unit the question asks for: the canonical's own unit, else the last unit it asks for by name.

    The canonical's unit may be one the table does not list ("9.8 L/min"): it is then its text, spelled plainly
    ("l/min"), which no recognised unit equals, so a reply in any recognised unit is refused and one in the
    same spelling is not judged."""
    answer_text = getattr(question, "answer_text", "") or ""
    shown = _unit_of(answer_text)
    if shown is not None:
        return shown
    unlisted = _unlisted_unit_text(answer_text)
    if unlisted is not None:
        return unlisted
    asked = _asking_matches(question, _ASKED_RE)
    return _SPELLING_TO_UNIT[asked[-1].group(1).lower()] if asked else None


# What a unit written with the number looks like: letters, a degree or micro sign, a slash, an exponent, spaces.
_UNIT_TEXT_RE = re.compile(r"[a-zA-Z°µμΩ/^²³·⋅\d ]{1,20}")


def _unlisted_unit_text(answer_text: str) -> str | None:
    """The unit an answer shows after its number when no listed unit makes it up ("L/min" gives "l/min"), or None.
    Only text that looks like a unit counts: a remark such as "(to 2 d.p.)" is not one."""
    found = _after_last_number(answer_text)
    if found is None or not found[1] or not _UNIT_TEXT_RE.fullmatch(found[1]):
        return None
    if not re.search(r"[a-zA-Z°µμΩ]", found[1]) or found[1].lower() in ("th", "st", "nd", "rd", "e", "x", "pi", "i"):
        return None
    return "unlisted:" + found[1].lower().replace(" ", "")


def _asked_compound_parts(question) -> frozenset[str] | None:
    """The recognised parts of the last compound unit the question asks for ("N/m" gives N and m), or None."""
    asked = _asking_matches(question, _ASKED_COMPOUND_RE)
    if not asked:
        return None
    return frozenset(_SPELLING_TO_UNIT[spelling.lower()] for spelling in asked[-1].groups() if spelling)


# A unit written as an SI prefix and a base symbol that the table does not list ("mN", "MN", "dm", "nm", "mA").
# Capital and small letters are different prefixes here (M is mega, m is milli), so it is read as written.
_PREFIXED_UNIT_RE = re.compile(r"(?:da|[YZEPTGMkhdcmµμunpf])(?:Pa|Hz|mol|[NJWVACFHTgmsLl])")


def _prefixed_unit_conflicts(question, given: str) -> bool:
    """Whether the reply ends in a prefixed unit the table does not list ("25 mN") while the question asks for
    another unit: the canonical's own unit, or one the question names. The canonical's own spelling, listed or
    not, never conflicts, and a question that asks for no unit judges none, as for any unit it does not know."""
    found = _after_last_number(given)
    if found is None or _PREFIXED_UNIT_RE.fullmatch(found[1]) is None:
        return False
    shown = _after_last_number(getattr(question, "answer_text", "") or "")
    if shown is not None and shown[1] == found[1]:
        return False
    return _asked_unit(question) is not None


def _same_unlisted_unit(unit: str, asked: str) -> bool:
    """Whether a reply's unit is the key's own unit, spelled with a space where the key has a middle dot: "N m" or
    "kN m" for a key in "N·m" or "kN·m". `asked` is an `_unlisted_unit_text` id, "unlisted:kn·m"."""
    if not asked.startswith("unlisted:") or not any(dot in asked for dot in "·⋅"):
        return False
    return unit == asked.removeprefix("unlisted:").replace("·", "*").replace("⋅", "*")


def unit_is_acceptable(question, given: str) -> bool:
    """Whether the unit a typed number reply gives (if any) is one the question allows. See the module docstring."""
    unit = _unit_of(given or "")
    if unit is None:
        return not _prefixed_unit_conflicts(question, given or "")
    asked = _asked_unit(question)
    if asked is not None:
        return unit == asked or _same_unlisted_unit(unit, asked)
    parts = _asked_compound_parts(question)
    if parts is None or unit in parts or unit in _RATE_UNITS:
        return True
    # "N/s" is not a way to write "N/m", but "N/m" itself, spelled out as parts, is.
    pieces = {piece.rstrip("23") or piece for piece in unit.split("/")} if "/" in unit else None
    return pieces is not None and pieces <= parts


def strip_asked_unit(question, text: str) -> str:
    """`text` without the unit it ends in, when that unit is the one the question asks for ("360 cm³" for "Find its
    volume, in cm³", "π/4 rad" for "Convert 45 degrees to radians"); otherwise `text` as it was. For a graded value
    that is not read as a number-with-unit, such as an expression: a question that names no unit strips nothing."""
    bare, unit = split_unit(text or "")
    if unit is None or unit in ("%", "dollar") or _asked_unit(question) != unit:
        return text
    return bare
