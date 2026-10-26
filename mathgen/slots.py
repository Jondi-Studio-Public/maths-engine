"""Shared vocabulary banks for phrasing banks and the variety/tier instrument.

A generator that wants a name, a countable object, a unit or a scenario for a
word problem draws from here (via ``mathgen.core.phrase`` and its own
``rng``) so that two questions differing only in "Amir" vs "Priya" or
"apples" vs "marbles" are recognised as the same underlying shape. That
recognition happens in ``mathgen.text_quality.prompt_shape``, which masks
every value below -- so this module and that one must always agree on what
counts as filler. Nothing here is itself a generator: no ``rng``, no
weighting, no i18n -- plain tuples of prose content, reviewed once as a bank
rather than once per generator.

Deliberately imports nothing from ``mathgen.core`` or any generator: this is
raw material other things (``phrase()``, ``prompt_shape()``) mask against or
draw from, never the other way around.
"""

from __future__ import annotations

#: Given names for a word-problem's characters -- a mix of genders and
#: cultural origins, since word problems should not all
#: default to one. Plain first names only: no surname, no title, so a
#: generator can drop one straight into a sentence ("Amir has 5 apples.").
NAMES: tuple[str, ...] = (
    "Amir",
    "Priya",
    "Noah",
    "Olivia",
    "Liam",
    "Mia",
    "Ethan",
    "Sophia",
    "Lucas",
    "Ava",
    "Kai",
    "Zara",
    "Diego",
    "Yuki",
    "Aisha",
    "Mateo",
    "Chloe",
    "Hassan",
    "Ingrid",
    "Tariq",
    "Beatrice",
    "Santiago",
    "Freya",
    "Kenji",
    "Nadia",
    "Oscar",
    "Priyanka",
    "Ravi",
    "Sione",
    "Talia",
)

#: (singular, plural) pairs of everyday countable objects for a word
#: problem, so a generator can pick a pair and pluralise correctly rather
#: than concatenating an "s" by hand (the "1 apples" defect
#: ``text_quality.NOT_PLURAL_NOUNS``/``LABEL_WORDS`` already guards
#: elsewhere). Includes one irregular plural ("box"/"boxes") on purpose, so a
#: consumer that tries to derive the plural instead of storing it is caught
#: immediately.
OBJECTS: tuple[tuple[str, str], ...] = (
    ("apple", "apples"),
    ("marble", "marbles"),
    ("sticker", "stickers"),
    ("pencil", "pencils"),
    ("cupcake", "cupcakes"),
    ("balloon", "balloons"),
    ("ribbon", "ribbons"),
    ("ticket", "tickets"),
    ("coin", "coins"),
    ("button", "buttons"),
    ("biscuit", "biscuits"),
    ("chair", "chairs"),
    ("box", "boxes"),
    ("muffin", "muffins"),
    ("crayon", "crayons"),
    ("bead", "beads"),
    ("shell", "shells"),
    ("bottle", "bottles"),
)

#: Common measurement units a word problem states an answer or a given value
#: in. Plain symbols/words as a learner would read them, not SI-formal forms.
UNITS: tuple[str, ...] = (
    "cm",
    "mm",
    "m",
    "km",
    "g",
    "kg",
    "mL",
    "L",
    "$",
    "minutes",
    "hours",
    "seconds",
)

#: Everyday scenario the problem is set in, written as the phrase would
#: appear mid-sentence ("at a bakery", "on a car trip"), so a generator can
#: drop it straight in after "at"/"during"/"on".
CONTEXTS: tuple[str, ...] = (
    "a bakery",
    "a school fete",
    "a car trip",
    "a farmers market",
    "a birthday party",
    "a school canteen",
    "a road trip",
    "a science fair",
    "a sports carnival",
    "a market stall",
    "a camping trip",
    "a school excursion",
)
