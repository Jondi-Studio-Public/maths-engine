"""Shared pieces for the short written-answer generators (`wa_*_build.py`).

A short written-answer question is marked against a rubric, not against one expected value, so a generator
returns a prompt plus a per-item marking dict (`Question.answer`):

    {"start": None, "target": None, "reference_lines": [...], "key_steps": [...],
     "traps": [...], "checks": []}

`reference_lines` is a model answer in plain sentences. `key_steps` lists the key points a correct answer
contains, each tagged with the criterion it earns (`correct_idea`, `use_of_context`, `completeness`,
`communication`). `traps` lists the common wrong answers, each with the question a marker should ask to catch it.
`checks` is empty: nothing here can be decided by a computer algebra system, so a person or a model marks the
written answer.

The scenarios are written as `Case` records: a setup with slots (names, places, counts) and the fields a question
needs (what is wrong, why, the effect, the fix). A `Tier` is a prompt template with the key points and wrong answers
that go with it; `build` fills one case (or two, for a comparison) into a tier.

Underscore-prefixed, so the generator registry does not import it as a generator module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

from ..core import VerificationError, check, generator

try:  # a bounded stand-in for `while True:`; older cores do not carry it
    from ..core import attempts
except ImportError:  # pragma: no cover - exercised only by a core without `attempts`

    def attempts(limit: int = 1000, what: str = "an acceptable draw"):
        """Yield 0 .. limit - 1, then raise `VerificationError`."""
        yield from range(limit)
        raise VerificationError(f"no acceptable draw for {what} in {limit} tries")


CRITERIA = ("correct_idea", "use_of_context", "completeness", "communication")
TIERS = ("easy", "medium", "hard", "exam")

NAMES = (
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
    "Omar",
    "Hana",
    "Jack",
)

_TEXT_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "_": r"\_",
    "$": r"\$",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
}


def tex_text(s: str) -> str:
    """`s` as upright text inside a LaTeX formula (`\\text{...}`), every special character escaped."""
    return r"\text{" + "".join(_TEXT_ESCAPES.get(ch, ch) for ch in s) + "}"


def tex_lines(text: str) -> str:
    """The prompt as LaTeX: one `\\text{}` per line of the prompt, lines joined by a LaTeX line break."""
    out = [tex_text(ln) if ln.strip() else r"\ " for ln in text.split("\n")]
    return r" \\ ".join(out)


_SPACE = re.compile(r"[ \t]{2,}")


def tidy(s: str) -> str:
    """Collapse double spaces and trim stray space before punctuation."""
    s = _SPACE.sub(" ", s)
    s = re.sub(r" ([,;:?]|\.(?!\.))", r"\1", s)
    return re.sub(r"([?!])\.", r"\1", s).strip()


def tidy_prompt(text: str) -> str:
    """Tidy each line of a multi-line prompt, keeping its line structure."""
    return "\n".join(tidy(ln) if ln.strip() else "" for ln in text.split("\n")).strip()


def pct(x: int | Fraction, y: int | Fraction, places: int = 1) -> str:
    """`x` as a percentage of `y`, rounded to `places` decimal places and written without a trailing `.0`."""
    value = round(100 * Fraction(x) / Fraction(y), places)
    text = f"{value:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text + "%"


def join_and(items: list[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def cap(s: str) -> str:
    """First letter upper-cased, the rest untouched."""
    return s[:1].upper() + s[1:]


def money(x: int | float) -> str:
    return f"${x:,.0f}" if float(x).is_integer() else f"${x:,.2f}"


# ---------------------------------------------------------------------------
# Marking
# ---------------------------------------------------------------------------
def marking(*, reference: list[str], points: list[tuple[str, str]], wrong: list[tuple[str, str]]) -> dict:
    """The per-item marking dict. `points` are (criterion id, key point); `wrong` are (name, question)."""
    check(len(reference) >= 2, "a model answer needs at least two lines")
    ids = [cid for cid, _ in points]
    check(set(ids) <= set(CRITERIA), f"unknown criterion id in {set(ids) - set(CRITERIA)}")
    check({"correct_idea", "use_of_context", "completeness"} <= set(ids), "key points must cover three criteria")
    check(len(points) >= 4, "an item needs at least four key points")
    check(len(wrong) >= 2, "an item needs at least two named wrong answers")
    check(len({n for n, _ in wrong}) == len(wrong), "wrong-answer names must be distinct")
    check(all("?" in q for _, q in wrong), "each wrong answer carries a question for the marker")
    return {
        "start": None,
        "target": None,
        "reference_lines": [tidy(line) for line in reference],
        "key_steps": [{"criterion_id": cid, "description": tidy(desc)} for cid, desc in points],
        "traps": [{"name": name, "question": tidy(q)} for name, q in wrong],
        "checks": [],
    }


_SENTENCE_END = re.compile(r"[.?!:][\"')]*$")


def flatten(text: str) -> str:
    """The prompt as one line of prose: the learner's page does not show line breaks.

    A line that is not a finished sentence is a row of a list or table (a run, a record, a data set); consecutive
    rows are joined with semicolons and the group is closed with a full stop.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    out: list[str] = []
    for i, ln in enumerate(lines):
        row = not _SENTENCE_END.search(ln)
        nxt_row = i + 1 < len(lines) and not _SENTENCE_END.search(lines[i + 1])
        if row and not nxt_row:
            ln += "."
        out.append(ln + ("; " if row and nxt_row else " "))
    return "".join(out).strip()


def finish(text: str, answer_text: str, mark: dict) -> dict:
    """The dict a written-answer generator returns. `answer_text` is the headline of the model answer."""
    text = tidy_prompt(text)
    return dict(text=flatten(text), latex=tex_lines(text), answer=mark, answer_text=tidy(answer_text))


# ---------------------------------------------------------------------------
# Cases and tiers
# ---------------------------------------------------------------------------
@dataclass
class Case:
    """One scenario. `fields` hold templates (`{slot}` placeholders); `slots` say how each slot is drawn.

    A slot is a list/tuple (one element is chosen) or a function `f(rng, drawn) -> value` that may read the slots
    drawn before it, so derived numbers (a percentage, a total) are computed once and stated consistently.
    `wrong` is a list of (name, question) templates.
    """

    id: str
    slots: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)
    wrong: list = field(default_factory=list)

    def draw(self, rng) -> dict:
        values: dict = {}
        for name, spec in self.slots.items():
            values[name] = spec(rng, values) if callable(spec) else rng.choice(spec)
        ctx = dict(values)
        out: dict = {}
        for k, v in self.fields.items():
            out[k] = _fmt(v, ctx) if isinstance(v, str) else v
            ctx[k] = out[k]
        out["wrong"] = [(_fmt(n, ctx), _fmt(q, ctx)) for n, q in self.wrong]
        out["case_id"] = self.id
        for k, v in values.items():
            out.setdefault(k, v)
        return out


def _fmt(template: str, values: dict) -> str:
    try:
        return template.format(**values)
    except (KeyError, IndexError) as exc:
        raise VerificationError(f"template uses an undefined slot {exc}: {template!r}") from exc


@dataclass
class Tier:
    """A prompt template with its key points. Templates are filled from the drawn case(s).

    `two=True` draws two different cases whose fields are available as `{a_field}` and `{b_field}`; `pair_ok`
    (optional) rejects a pair, and `derive` (optional) adds fields computed from the pair.
    `prompt` is the text; `reference`, `points` and `wrong` follow `marking`'s shapes; `answer` is the headline.
    """

    prompt: str
    reference: list
    points: list
    wrong: list = field(default_factory=list)
    answer: str = ""
    two: bool = False
    pair_ok: object = None
    derive: object = None
    cases: list | None = None


def _prefixed(d: dict, p: str) -> dict:
    return {f"{p}_{k}": v for k, v in d.items() if k != "wrong"}


def build(rng, cases: list[Case], tier: Tier, tails: tuple[str, ...]) -> dict:
    cases = tier.cases or cases
    if tier.two:
        for _ in range(200):
            a, b = rng.sample(cases, 2)
            da, db = a.draw(rng), b.draw(rng)
            if tier.pair_ok is None or tier.pair_ok(da, db):
                break
        else:
            raise VerificationError("no acceptable pair of cases")
        values = {**_prefixed(da, "a"), **_prefixed(db, "b")}
        case_wrong = [(f"A: {n}", q) for n, q in da["wrong"][:1]] + [(f"B: {n}", q) for n, q in db["wrong"][:1]]
    else:
        d = rng.choice(cases).draw(rng)
        values = {k: v for k, v in d.items() if k != "wrong"}
        case_wrong = d["wrong"]
    if tier.derive is not None:
        values.update(tier.derive(values))
    text = _fmt(tier.prompt, values) + " " + rng.choice(tails)
    ref = [_fmt(line, values) for line in tier.reference]
    points = [(cid, _fmt(desc, values)) for cid, desc in tier.points]
    wrong = [(_fmt(n, values), _fmt(q, values)) for n, q in tier.wrong]
    taken = {n for n, _ in wrong}
    wrong += [(n, q) for n, q in case_wrong if n not in taken]
    mark = marking(reference=ref, points=points, wrong=wrong)
    return finish(text, _fmt(tier.answer or ref[0], values), mark)


TAILS = (
    "Answer in a few sentences.",
    "Write a short paragraph.",
    "Give a clear written answer of two to four sentences.",
    "Use complete sentences and refer to the details given.",
    "Keep your answer to a short paragraph that someone else could follow.",
)

#: A prompt built around data asks the learner to quote it.
DATA_TAILS = (
    "Use the numbers given to support your answer.",
    "Refer to the data in your answer.",
    "Support your answer with figures from the question.",
    "Quote values from the data to back up what you say.",
)


def register(topic: str, level: int, cases: list[Case], tiers: dict[str, Tier], tails=TAILS, doc: str = ""):
    """Register a generator `topic` whose four difficulties are the given tiers over the given cases."""
    check(set(tiers) == set(TIERS), "a topic defines all four tiers")
    presets = {t: dict(variant=t) for t in TIERS}

    @generator(topic, level=level, presets=presets)
    def _gen(rng, variant="easy"):
        if variant not in tiers:
            raise ValueError(f"unknown variant: {variant}")
        return build(rng, cases, tiers[variant], tails)

    _gen.__doc__ = doc or f"Short written answer `{topic}` (rubric-marked): every question carries a marking dict."
    return _gen


def register_custom(topic: str, level: int, makers: dict, doc: str = ""):
    """Register a generator whose four difficulties are plain functions `f(rng) -> dict`."""
    check(set(makers) == set(TIERS), "a topic defines all four tiers")
    presets = {t: dict(variant=t) for t in TIERS}

    @generator(topic, level=level, presets=presets)
    def _gen(rng, variant="easy"):
        if variant not in makers:
            raise ValueError(f"unknown variant: {variant}")
        return makers[variant](rng)

    _gen.__doc__ = doc or f"Short written answer `{topic}` (rubric-marked): every question carries a marking dict."
    return _gen


#: (organisation, plural for its people, singular, topic a survey could ask about). Shared by the survey topics.
SURVEY_CTX = (
    ("Hillcrest High School", "students", "student", "how they travel to school"),
    ("Maple Grove Primary School", "students", "student", "their favourite canteen lunch"),
    ("Ridgeway College", "students", "student", "a later start time for school"),
    ("Eastside Tennis Club", "members", "member", "the new court booking system"),
    ("Lakeside Fitness Centre", "members", "member", "the class timetable"),
    ("Bayview Library", "borrowers", "borrower", "the opening hours"),
    ("Northgate Council", "ratepayers", "ratepayer", "a proposed bike path"),
    ("Harbour Town Football Club", "members", "member", "the canteen prices"),
    ("Westfield Community Centre", "members", "member", "which evening classes to run"),
    ("Greenacre Retirement Village", "residents", "resident", "the weekend activities"),
    ("Southbank Swimming Club", "swimmers", "swimmer", "the training times"),
    ("Willow Creek Estate", "residents", "resident", "a new playground"),
)


def ctx_slots(prefix: str = "") -> dict:
    """Slots that draw one survey context and split it into `org`, `members`, `member`, `topic`."""
    return {
        f"{prefix}ctx": SURVEY_CTX,
        f"{prefix}org": lambda r, v: v[f"{prefix}ctx"][0],
        f"{prefix}members": lambda r, v: v[f"{prefix}ctx"][1],
        f"{prefix}member": lambda r, v: v[f"{prefix}ctx"][2],
        f"{prefix}topic": lambda r, v: v[f"{prefix}ctx"][3],
    }


def ri(lo: int, hi: int, step: int = 1):
    """A slot that draws an integer from lo..hi in steps."""
    return lambda r, v: r.choice(range(lo, hi + 1, step))


NAME_SLOTS = {"person": NAMES, "other": lambda r, v: r.choice([n for n in NAMES if n != v["person"]])}


# ---------------------------------------------------------------------------
# Small statistics helpers, so every number a prompt states is computed
# ---------------------------------------------------------------------------
def num_text(x, places: int = 1) -> str:
    """A number rounded to `places` decimals, written without a trailing `.0`."""
    value = round(float(Fraction(x)), places)
    text = f"{value:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def mean_of(values) -> Fraction:
    return Fraction(sum(Fraction(str(v)) for v in values), len(values))


def median_of(values) -> Fraction:
    s = sorted(Fraction(str(v)) for v in values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def mode_of(values) -> list:
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    return sorted(v for v, c in counts.items() if c == top)


def values_line(values, sep: str = ", ") -> str:
    return sep.join(str(v) for v in values)
