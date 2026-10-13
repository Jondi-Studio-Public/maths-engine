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

Angles are deliberately absent: a degree sign is read by the grader itself, and many angle questions never
write a unit.
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
    "mile": ("mile", "miles"),
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
    # energy
    "j": ("joule", "joules"),  # not a bare "j": "-4j" is a vector component
    "kj": ("kj", "kilojoule", "kilojoules"),
    "mj": ("mj", "megajoule", "megajoules"),
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
    # money
    "dollar": ("$", "dollar", "dollars"),
    "cent": ("cent", "cents"),
}

_SPELLING_TO_UNIT = {spelling: unit for unit, spellings in _UNITS.items() for spelling in spellings}
# Longest first, so "km/h" is found before "km" and "square metres" before "metres".
_SPELLINGS = sorted(_SPELLING_TO_UNIT, key=len, reverse=True)
_SPELLING_PATTERN = "|".join(re.escape(spelling) for spelling in _SPELLINGS)
# A spelling standing on its own: not inside a word ("m" in "mass"), and not followed by a letter or digit that
# would make it a longer word or an exponent ("m" in "m2" is found as "m2").
_MENTION_RE = re.compile(rf"(?<![^\W\d_])(?:{_SPELLING_PATTERN})(?![\w²³'’])", re.IGNORECASE)
# Not a digit written straight after a letter, or a letter and "^": the "2" in "m2" or "m^2" is the unit's power, not the value.
_NUMBER_RE = re.compile(r"(?<![A-Za-z])(?<![A-Za-z]\^)(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
# "to mm", "into grams", "in km/h", "how many kilojoules": the unit the question asks for by name. Not "in year 3"
# (a label, not a unit), not "to the nearest cent" (a precision: the answer is still in dollars), and not "in n = 5"
# or "in g(x)" (a variable or a function).
_ASKED_RE = re.compile(
    rf"\b(?:to|into|in|how\s+many(?:\s+(?:more|fewer|less|extra|further|additional))?)\s+({_SPELLING_PATTERN})(?![\w²³'’])(?!\s*[\d=(])(?!\s*(?:per\b|[/·⋅×*]))",
    re.IGNORECASE,
)
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
            return text[: numbers[-1].end()], _SPELLING_TO_UNIT[match.group(0).lower()]
    dollar = _LEADING_DOLLAR_RE.match(text)
    if dollar is not None:
        return text[dollar.end() :], "dollar"
    percent = _PERCENT_RE.search(text)
    if percent is not None and numbers:
        return text[: percent.start()], "%"
    return text, None


def _unit_of(text: str) -> str | None:
    """The recognised unit written straight after the last number in `text`, or None. A leading "$" is not
    judged: "$501" for an amount in cents has always been read as the number 501."""
    unit = split_unit(text)[1]
    return None if unit in ("%", "dollar") and not _trailing_unit(text) else unit


def _trailing_unit(text: str) -> bool:
    """Whether `text` writes a recognised unit straight after its last number."""
    numbers = list(_NUMBER_RE.finditer(text or ""))
    if not numbers:
        return False
    rest = text[numbers[-1].end() :].strip().rstrip(".!")
    match = _MENTION_RE.match(rest)
    return match is not None and match.end() == len(rest.rstrip())


def _describes_earlier_value(before: str) -> bool:
    """Whether the words in front of an "in <unit>" make it describe a value the learner already has."""
    found = _EARLIER_VALUE_RE.search(before)
    return found is not None and _ASKING_VERB_BEFORE_RE.search(before[: found.start()]) is None


def _asked_unit(question) -> str | None:
    """The unit the question asks for: the canonical's own unit, else the last unit it asks for by name."""
    shown = _unit_of(getattr(question, "answer_text", "") or "")
    if shown is not None:
        return shown
    asked = [
        spelling
        for sentence in _SENTENCE_RE.findall(getattr(question, "text", "") or "")
        if sentence.rstrip().endswith("?") or _ASKING_RE.match(sentence)
        for match in _ASKED_RE.finditer(sentence)
        if not _describes_earlier_value(sentence[: match.start()])
        for spelling in (match.group(1),)
    ]
    return _SPELLING_TO_UNIT[asked[-1].lower()] if asked else None


def unit_is_acceptable(question, given: str) -> bool:
    """Whether the unit a typed number reply gives (if any) is one the question allows. See the module docstring."""
    unit = _unit_of(given or "")
    if unit is None:
        return True
    asked = _asked_unit(question)
    return asked is None or unit == asked
