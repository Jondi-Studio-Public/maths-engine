"""Core framework: Question dataclass, generator registry, seeded factory,
difficulty presets, distractor helper."""

from __future__ import annotations

import difflib
import json
import math
import random
import re
from fractions import Fraction
from functools import lru_cache
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from sympy import (
    Eq,
    Float,
    Integer,
    Rational,
    S,
    Symbol,
    Tuple,
    cancel,
    expand,
    expand_trig,
    radsimp,
    simplify,
    srepr,
    sympify,
)
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy import latex as _sympy_latex
from sympy.core.function import AppliedUndef
from sympy.core.sympify import SympifyError

from .answers import ANSWER_FORMS, ANSWER_KINDS, Distractor, flatten_distractors


@dataclass
class Question:
    topic: str  # e.g. "quadratic"
    level: int  # 1..20 curriculum level
    seed: int  # (topic, seed[, difficulty]) fully reproduces this question
    text: str  # plain-text question
    latex: str  # LaTeX question for rendering
    answer: Any  # structured, JSON-serialisable answer
    answer_text: str  # human-readable answer
    answer_latex: str = ""
    working: list[str] = field(default_factory=list)  # optional worked steps
    difficulty: str | None = None  # preset name, if one was used
    distractors: list[str] = field(default_factory=list)  # misconception-based wrong answers,
    # formatted like answer_text
    answer_alt: Any = None  # optional numeric alternate answer (e.g. a surd's
    # decimal approximation) a numeric grader path can
    # also accept; omitted from JSON when unset so it
    # never perturbs any other topic's output/hash.
    option_notes: dict[str, str] = field(default_factory=dict)  # option text -> a
    # note id into the skill's `misconceptions` or
    # `near_misses`. Empty for every generator
    # that doesn't set it, and omitted from to_dict()/
    # to_json() when empty, exactly like `answer_alt`
    # above -- so it can never perturb an existing
    # topic's pinned output/hash.
    answer_form: str | None = None  # None | "expanded" | "factored" |
    # "simplified_surd" | "rationalised" |
    # "single_fraction" |
    # "line_standard_form".
    # The shape the grader must require of a learner's
    # answer -- a property of what the question ASKS,
    # not of how the canonical answer happens to be
    # spelled (sympy auto-distributes Number * Add, so
    # the canonical string itself cannot be parsed to
    # recover this). Unset for every generator that
    # doesn't declare it, and omitted from to_dict()/
    # to_json() when None, exactly like `answer_alt`
    # above -- so it can never perturb an existing
    # topic's pinned output/hash.
    figure: dict | None = None  # a picture in the question, as data: a spec built
    # by `mathgen.figures`. None for every generator that doesn't draw
    # one, and omitted from to_dict()/to_json() when None,
    # exactly like `answer_alt` above.
    # The typed answer contract (mathgen/answers.py). Each is unset for a generator that doesn't declare it and
    # omitted from to_dict()/to_json() when unset, exactly like `answer_alt` above.
    answer_kind: str | None = None  # grader kind of `answer`, one of answers.ANSWER_KINDS
    distractor_values: dict[str, Any] = field(default_factory=dict)  # option text -> the structured
    # value a learner making that option's error reaches, shaped like `answer`
    distractor_credit: dict[str, str] = field(default_factory=dict)  # option text -> share of the
    # marks that error still earns, as a fraction string ("1/2")
    parts: list[Part] = field(default_factory=list)  # a multi-part question's parts, in order; see `Part`.
    # Empty for a single-answer question, and omitted from to_dict()/to_json() when empty, like `answer_alt`.
    stem: str | None = None  # a multi-part question's shared information, without the parts' prompts
    # (`text` is the stem followed by every part's prompt). None, and omitted, for a single-answer question.

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d.get("parts"):
            d.pop("parts", None)
        if d.get("stem") is None:
            d.pop("stem", None)
        for part in d.get("parts", ()):
            if part.get("sig_figs") is None:
                part.pop("sig_figs", None)  # omitted when unset, like `answer_alt`, so unset parts hash as before
        if d.get("answer_alt") is None:
            d.pop("answer_alt", None)
        if not d.get("option_notes"):
            d.pop("option_notes", None)
        if d.get("answer_form") is None:
            d.pop("answer_form", None)
        if d.get("figure") is None:
            d.pop("figure", None)
        for key in ("answer_kind", "distractor_values", "distractor_credit"):
            if not d.get(key):
                d.pop(key, None)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


@dataclass
class Part:
    """One part of a multi-part question: "(a) find the radius", "(b) hence find the circumference".

    A part has the same answer attributes a single-answer `Question` has (`answer`, `answer_text`,
    `answer_latex`, `answer_form`, `answer_alt`, `distractors`, `option_notes`), so every grader grades a part
    exactly as it grades a question: `grade(part.answer_kind, part, reply)`. See
    `docs/design/multi-part-questions.md`.

    `follow_through` is a SymPy expression for this part's answer in terms of earlier parts' values, named
    `part_a`, `part_b`, ... (for example `"2*pi*part_a"`). When it is set, a reply that is right for the
    learner's OWN earlier answers earns this part's marks, as exam markers give follow-through marks. Only
    parts graded as one value (`number`, `fraction`, `expression`) can be named in it.

    Follow-through is not given for a later part when an earlier reply it uses makes that part trivial or
    impossible, as markers withhold it: a zero where the canonical value is not zero, a later answer of zero
    where the canonical one is not, or a reply outside the earlier part's `sensible_range`. `sensible_range` is
    the open interval a sensible answer to this part lies in, `(0, None)` for a length, an area or an amount of
    money, `(0, 1)` for a probability; `make()` checks the canonical answer lies in it.
    """

    label: str  # "a", "b", ...
    prompt: str  # what this part asks, plain text
    answer_kind: str  # the grader for this part, a key of mathgen.grading.GRADERS
    answer: Any
    answer_text: str
    prompt_latex: str = ""
    answer_latex: str = ""
    marks: int = 1
    follow_through: str | None = None
    working: list[str] = field(default_factory=list)
    distractors: list[str] = field(default_factory=list)
    answer_alt: Any = None
    option_notes: dict[str, str] = field(default_factory=dict)
    answer_form: str | None = None
    distractor_values: dict[str, Any] = field(default_factory=dict)  # as on `Question` (mathgen/answers.py)
    distractor_credit: dict[str, str] = field(default_factory=dict)  # as on `Question`: a share of this part's marks
    sensible_range: tuple | None = None  # (low, high), open, None for no bound: see below
    sig_figs: int | str | None = None  # significant figures the answer is given to when the prompt leaves the count
    # to an earlier answer ("to that number of significant figures"): the label of the part that gives it. Graders
    # refuse a reply with more figures; follow-through uses the learner's own count.
    seed: str = ""  # "<question seed>:<label>"; shuffles this part's options for an mcq part (`mcq_options`)

    @property
    def text(self) -> str:
        """The prompt, under the name graders read a question's wording by (`requested_rounding(text)`)."""
        return self.prompt


#: Keys a generator may give each entry of `parts`; everything else in a part dict is an error.
_PART_KEYS = frozenset(Part.__dataclass_fields__) - {"seed"}
#: Answer keys `make()` fills from the parts, so a multi-part generator must not set them itself.
_FILLED_FROM_PARTS = (
    "answer",
    "answer_text",
    "answer_latex",
    "answer_kind",
    "distractors",
    "answer_alt",
    "option_notes",
    "answer_form",
    "distractor_values",
    "distractor_credit",
)


def _build_parts(d: dict, seed) -> dict:
    """Turn a multi-part generator's dict into the question fields: `Part` objects plus the composed
    question-level text, answer and working that code unaware of parts reads (see `Question.parts`)."""
    for key in _FILLED_FROM_PARTS:
        if key in d:
            raise ValueError(f"a multi-part question sets {key!r} on each part, not on the question")
    raw = d.pop("parts")
    if not raw:
        raise ValueError("parts must not be empty")
    parts = []
    for i, entry in enumerate(raw):
        unknown = set(entry) - _PART_KEYS
        if unknown:
            raise ValueError(f"part {i} has unknown keys {sorted(unknown)}")
        entry = dict(entry)
        entry.setdefault("label", chr(ord("a") + i))
        # A part's tagged distractors are flattened exactly as a question's are (mathgen/answers.py).
        flatten_distractors(entry)
        if entry.get("answer_kind") not in ANSWER_KINDS:
            raise ValueError(f"part {entry['label']!r}: unknown answer_kind {entry.get('answer_kind')!r}")
        if entry.get("answer_form") is not None and entry["answer_form"] not in ANSWER_FORMS:
            raise ValueError(f"part {entry['label']!r}: unknown answer_form {entry['answer_form']!r}")
        if not entry.get("prompt_latex") or entry["prompt_latex"] == entry["prompt"]:
            entry["prompt_latex"] = latex_prompt(entry["prompt"])
        if isinstance(entry.get("working"), str):
            entry["working"] = [entry["working"]]
        part = Part(**entry, seed=f"{seed}:{entry['label']}")
        if not isinstance(part.marks, int) or isinstance(part.marks, bool) or part.marks < 1:
            raise ValueError(f"part {part.label!r}: marks must be a whole number of at least 1")
        parts.append(part)
    labels = [p.label for p in parts]
    if len(set(labels)) != len(labels):
        raise ValueError(f"part labels must be unique: {labels}")

    stem = d.pop("text")
    stem_latex = d.pop("latex", "") or ""
    if not stem_latex or stem_latex == stem:
        stem_latex = latex_prompt(stem)
    d["stem"] = stem
    d["text"] = " ".join([stem, *(f"({p.label}) {p.prompt}" for p in parts)])
    d["latex"] = " ".join([stem_latex, *(rf"\text{{({p.label}) }}{p.prompt_latex}" for p in parts)])
    d["answer"] = {p.label: p.answer for p in parts}
    d["answer_text"] = "; ".join(f"({p.label}) {p.answer_text}" for p in parts)
    d["answer_latex"] = "plain:" + d["answer_text"]
    d["working"] = [*d.get("working", []), *(f"({p.label}) {step}" for p in parts for step in p.working)]
    d["parts"] = parts
    return d


class VerificationError(Exception):
    """Raised when a generator's independent SymPy check fails."""


# topic -> (level, generator function)
REGISTRY: dict[str, tuple[int, Callable]] = {}
# topic -> {preset name -> params dict}
PRESETS: dict[str, dict[str, dict]] = {}


def generator(topic: str, level: int, presets: dict[str, dict] | None = None):
    """Decorator registering a generator.

    A generator takes a random.Random and optional difficulty kwargs and
    returns a dict with keys: text, latex, answer, answer_text and optionally
    answer_latex, working, distractors. Where the answer is derived rather
    than chosen outright (a solved root, an antiderivative, an inverse), the
    generator must re-check it with SymPy independently of how it was
    constructed and raise VerificationError on mismatch.

    `presets` maps difficulty names (e.g. "easy"/"medium"/"hard") to kwargs
    bundles for this generator.
    """

    def deco(fn: Callable) -> Callable:
        if topic in REGISTRY:
            raise ValueError(f"duplicate topic: {topic}")
        REGISTRY[topic] = (level, fn)
        if presets:
            PRESETS[topic] = presets
        return fn

    return deco


def make(topic: str, seed: int | None = None, difficulty: str | None = None, **params) -> Question:
    """Build one verified Question. Same (topic, seed, difficulty) -> same question.

    `difficulty` selects a named preset (see PRESETS); explicit **params
    override preset values. An unknown difficulty name raises; a topic with
    no presets simply uses its defaults for any difficulty name.
    """
    if topic not in REGISTRY:
        # Close matches, not the whole list: there are over a thousand topics, and `topics()` lists them.
        close = difflib.get_close_matches(topic, REGISTRY, n=3) if isinstance(topic, str) else []
        hint = f"; did you mean {', '.join(repr(c) for c in close)}?" if close else "; topics() lists them"
        raise KeyError(f"unknown topic {topic!r}{hint}")
    level, fn = REGISTRY[topic]
    if seed is None:
        seed = random.randrange(2**31)
    if difficulty is not None and topic in PRESETS:
        if difficulty not in PRESETS[topic]:
            raise KeyError(f"topic {topic!r} has no preset {difficulty!r}; known: {', '.join(sorted(PRESETS[topic]))}")
        params = {**PRESETS[topic][difficulty], **params}
    # random.Random seeded with str hashes via sha512 -> stable across runs
    # and machines (unlike hash() of a str). Difficulty is part of the seed
    # string so easy/hard streams are independent.
    rng = random.Random(f"{topic}:{difficulty or ''}:{seed}")
    d = fn(rng, **params)
    # Tagged distractors become plain option text plus the keyed attributes (mathgen/answers.py), so every consumer
    # that reads `distractors` as strings is unchanged.
    flatten_distractors(d)
    if d.get("answer_kind") is not None and d["answer_kind"] not in ANSWER_KINDS:
        raise ValueError(f"{topic}: unknown answer_kind {d['answer_kind']!r}")
    if d.get("answer_form") is not None and d["answer_form"] not in ANSWER_FORMS:
        raise ValueError(f"{topic}: unknown answer_form {d['answer_form']!r}")
    if "parts" in d:
        # A multi-part question: build its parts and the question-level fields they fill, then prove every
        # follow-through expression reproduces its part's answer from the earlier parts' answers.
        d = _build_parts(d, seed)
        from mathgen.grading.parts import verify_follow_through

        verify_follow_through(d["parts"])
    # A generator that never authored its own `latex` (or authored it as a
    # verbatim copy of `text`) gets one for free here -- the single choke
    # point every consumer goes through, so no generator needs touching.
    # One that DID author a distinct `latex` (`\begin{cases}`, `\int`, ...)
    # is untouched.
    if not d.get("latex") or d["latex"] == d["text"]:
        d["latex"] = latex_prompt(d["text"])
    if d.get("figure") is not None:
        # Every figure passes one validator, whichever generator drew it.
        from mathgen import figures

        figures.check(d["figure"])
    return Question(topic=topic, level=level, seed=seed, difficulty=difficulty, **d)


def topics() -> list[tuple[int, str]]:
    """Registered (level, topic) pairs, sorted by level."""
    return sorted((lvl, t) for t, (lvl, _fn) in REGISTRY.items())


def ordinal(n: int) -> str:
    """`n` as an English ordinal numeral: 1st, 2nd, 3rd, 4th, 11th, 12th, 13th, 21st, 112th."""
    if 10 <= abs(n) % 100 <= 20:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(abs(n) % 10, "th")


def check(cond: bool, msg: str = "answer verification failed"):
    if not cond:
        raise VerificationError(msg)


def attempts(limit: int = 1000, what: str = "an acceptable draw"):
    """Yield 0, 1, ... limit - 1, then raise `VerificationError`: a bounded stand-in for `while True:`.

    For rejection loops that draw several values per attempt (use `draw_until` when one
    value is drawn). `break` out of the loop when the draw is acceptable; if the loop
    runs the budget out, the error names `what` and `limit`::

        for _ in attempts(1000, "a coprime pair"):
            a, b = rng.randint(1, 12), rng.randint(1, 12)
            if math.gcd(a, b) == 1:
                break

    It draws nothing itself, so replacing `while True:` with it leaves every seed's
    random stream, and so every question, unchanged.
    """
    yield from range(limit)
    raise VerificationError(f"no acceptable draw for {what} in {limit} tries")


def draw_until(
    rng: random.Random,
    draw: Callable[[random.Random], Any],
    accept: Callable[[Any], bool],
    *,
    tries: int = 1000,
    what: str = "an acceptable draw",
):
    """Return the first `draw(rng)` that `accept` allows; raise `VerificationError` after `tries` draws.

    The bounded form of rejection sampling. It replaces::

        while True:
            v = draw(rng)
            if accept(v):
                break

    and consumes `rng` exactly as that loop does: `draw(rng)` is called once per attempt,
    in order, and nothing else touches `rng`. Converting an existing loop therefore keeps
    every seed's question identical, provided `accept` does not use `rng` and `draw`
    makes the same calls, in the same order, as the loop body did. A value that fails is
    discarded; the one returned is the one that passed. When all `tries` draws are
    rejected the error names `what` and `tries`, so a bad seed or odd params fails loudly
    with a message instead of hanging the process.
    """
    for _ in range(tries):
        value = draw(rng)
        if accept(value):
            return value
    raise VerificationError(f"no acceptable draw for {what} in {tries} tries")


_VALUE_TOKEN_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?")


def _distractor_signature(text: str, *, unordered: bool):
    """``(wording, values)`` for deciding whether two options say the same thing.

    ``wording`` is the text with every numeric token removed and whitespace
    collapsed; ``values`` are those tokens as ``Fraction``. Two options with
    the same wording and the same values are the same answer however they are
    spelled, so "b = 5" is not a wrong answer to "b = 5.0" and "x = 0/4" is
    not a wrong answer to "x = 0".

    Comparing wording as well as values is what keeps a genuine unit-error
    distractor: "12 cm" and "12 m" share a value but not a wording.

    With ``unordered`` the values are compared as a multiset, for answers a
    grader accepts in any order -- a set of eigenvalues, a set of stationary
    points. Sequence order is kept by default, so a transposition stays a
    valid distractor for an ordered answer.
    """
    values = []
    for token in _VALUE_TOKEN_RE.findall(text):
        try:
            values.append(Fraction(token.replace(" ", "")))
        except (ValueError, ZeroDivisionError):
            return None
    wording = " ".join(_VALUE_TOKEN_RE.sub(" ", text).split())
    if not values:
        return None
    return wording, (tuple(sorted(values)) if unordered else tuple(values))


#: The typed-number/pair grader's tolerance (mathgen/grading/typed.py's `_PAIR_TOLERANCE`).
#: Pass it as `tolerance=` from a skill graded that way; see `pick_distractors`.
#: This is the *spoken-transcript* tolerance (`mathgen/grading/spoken.py`'s
#: `_TOLERANCE`). It is deliberately coarser than most typed-decimal answers
#: need -- see `stated_precision_tolerance` below for that path, and do not
#: use this constant for a typed numeric answer's own distractor margin.
GRADER_TOLERANCE = Fraction(51, 1000)


def stated_precision_tolerance(dp: int) -> Fraction:
    """Half a unit in the ``dp``-th decimal place: the tolerance for a typed numeric answer graded to its OWN
    STATED PRECISION -- e.g. an ``answer_text`` stating 2 d.p. tolerates +/-0.005. Use this one for a typed
    decimal answer's distractor margin and for a deliberately-wrong test value; read the generator's own
    `answer_text` formatting first to know what ``dp`` actually is for a given seed/tier.

    ``dp=0`` gives `Fraction(1, 2)`, which is *not* the real tolerance for an
    integer-formatted canonical: `mathgen/grading/typed.py`'s typed-number path
    requires an EXACT match when `answer_text` carries no decimal point at
    all, tighter than this function's own halfway point. Do not call this
    with ``dp=0`` and assume the result is the grader's integer tolerance.
    """
    if dp < 0:
        raise ValueError(f"dp must be >= 0, got {dp!r}")
    return Fraction(1, 2 * 10**dp)


def _same_answer(left, right, tolerance) -> bool:
    """Whether two option signatures are the same answer as the grader sees it."""
    if left is None or right is None:
        return False
    if left[0] != right[0] or len(left[1]) != len(right[1]):
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(left[1], right[1]))


def pick_distractors(
    answer_text: str,
    candidates: list,
    n: int = 3,
    *,
    unordered: bool = False,
    tolerance: Fraction | float = 0,
) -> list[str]:
    """Dedupe misconception candidates and drop any that collide with the
    correct answer. Order is preserved (candidates should be listed in
    misconception-priority order); rendering code shuffles options.

    A candidate is dropped when it is the same *answer* as the correct one and
    not merely the same string. The regrade sweep found 54 distractors that a
    learner could pick and be marked correct -- "1.0" offered against "1",
    "x = 0/4, y = 3" against "x = 0, y = 3" -- because the old check compared
    strings alone. Pass ``unordered=True`` when the skill's grader accepts the
    values in any order, so a reordering is not offered as a wrong answer
    either.

    ``tolerance`` defaults to 0, so only an exactly equal value is dropped. Pass
    ``GRADER_TOLERANCE`` from a skill graded by `grade_number` or `grade_pair`,
    whose comparison allows about 0.051 -- a candidate inside that is accepted by
    the real grader however different it looks. It has to stay opt-in: `fraction`
    skills are graded exactly, and an absolute 0.051 would throw away perfectly
    good distractors for a small probability such as 2/437.

    A `mathgen.answers.Distractor` candidate is judged by its text and kept as it is, tag and all; every other
    candidate comes back as its string.
    """
    tolerance = Fraction(tolerance).limit_denominator(10**6)
    seen = {str(answer_text)}
    answer_signature = _distractor_signature(str(answer_text), unordered=unordered)
    out: list = []
    for c in candidates:
        s = str(c)
        if s in seen:
            continue
        if _same_answer(answer_signature, _distractor_signature(s, unordered=unordered), tolerance):
            continue
        seen.add(s)
        out.append(c if isinstance(c, Distractor) else s)
        if len(out) == n:
            break
    return out


def fmt_signed(coef, var: str = "") -> str:
    """Format one (coefficient, variable) term the way a student would write
    it as the FIRST term of an expression: fmt_signed(-1, "x") -> "-x",
    fmt_signed(1, "x") -> "x", fmt_signed(3, "y") -> "3y",
    fmt_signed(-14) -> "-14". `var` may itself be compound text such as
    "(6x + 10)"; the coefficient is simply prefixed onto it. Shared by
    generators that used to interpolate a coefficient directly into an
    f-string (producing "1x"/"-1x" for a unit coefficient)."""
    if not var:
        return str(coef)
    if coef == 1:
        return var
    if coef == -1:
        return f"-{var}"
    return f"{coef}{var}"


def bracket_compound(text) -> str:
    """Bracket `text` when it cannot stand after a "/" as written: a fraction ("4/25"), a sum ("x + 1") or a
    negative ("-4") -- so ``f"-1/{bracket_compound('4/25')}"`` reads "-1/(4/25)", never the chain "-1/4/25".
    A plain number, letter or surd ("7", "n", "7√2") is returned unchanged."""
    text = str(text)
    return f"({text})" if ("/" in text or " " in text or text.startswith(("-", "−"))) else text


def paren_negative(value) -> str:
    """Bracket a negative value so it can follow an operator: -14 -> "(-14)".

    `join_terms` is the right tool when the expression should be simplified --
    it turns 17 + (-14) into "17 - 14". Use this one instead when the operator
    is the thing being taught, so "add negative fourteen" must stay visible as
    an addition. Either is correct; writing the value straight after the
    operator, as "17 + -14", is not.
    """
    return f"({value})" if value < 0 else str(value)


def join_terms(terms) -> str:
    """Join [(coef, var), ...] into a signed sum/difference the way a student
    would write it, e.g.:
        join_terms([(-1, "x"), (-3, "y")])              -> "-x - 3y"
        join_terms([(17, ""), (-14, "")])                -> "17 - 14"
        join_terms([(13, "x"), (-3, "(6x + 10)"), (5, "")]) -> "13x - 3(6x + 10) + 5"
    Zero-coefficient terms are dropped (renders "0" if every term is zero).
    Shared by generators that used to interpolate a signed value straight
    after a literal "+"/"-", producing "+ -3" or "- -3"."""
    parts = []
    for coef, var in terms:
        if coef == 0:
            continue
        if not parts:
            parts.append(fmt_signed(coef, var))
            continue
        sign = "+" if coef >= 0 else "-"
        mag = abs(coef)
        if var:
            parts.append(f"{sign} {var}" if mag == 1 else f"{sign} {mag}{var}")
        else:
            parts.append(f"{sign} {mag}")
    return " ".join(parts) if parts else "0"


def rational_text(r) -> str:
    """Format an int or Rational the way a student would write it: "3", "3/4",
    "-1/2". Shared by every generator whose answer is an exact fraction."""
    r = Rational(r)
    return str(r.p) if r.q == 1 else f"{r.p}/{r.q}"


def article_for_number(value) -> str:
    """ "a" or "an" for a number written in digits and read aloud: "an 8", "an 85", "an 800", "an 11.5",
    "an 18", but "a 40", "a 110", "a 0.8". Only a leading 8, or exactly eleven or eighteen, starts with a vowel
    sound."""
    whole = str(value).lstrip("-+$").split(".")[0].replace(",", "")
    return "an" if whole.startswith("8") or whole in ("11", "18") else "a"


def equals_chain(*steps) -> str:
    """Steps of a worked line joined with " = ", leaving out a step that repeats the one before it, so an
    already-simplified fraction reads "19/35" and never "19/35 = 19/35"."""
    kept = [str(steps[0])]
    for step in steps[1:]:
        if str(step) != kept[-1]:
            kept.append(str(step))
    return " = ".join(kept)


def phrase(rng: random.Random, patterns, **slots) -> str:
    """Pick ONE wording from `patterns` and fill it with `slots`.

    `patterns` is a non-empty sequence of plain `str.format`-style template
    strings sharing the same `**slots` keys (so text and latex can draw the
    same fill values from one call each). One is chosen with `rng.choice`
    and returned as `pattern.format(**slots)` -- no fallback, no retry: an
    unfilled `{name}` in the chosen pattern raises `KeyError` exactly as
    `str.format` would, so a mismatched pattern/slots pair fails loudly at
    the call site rather than silently at render time.

    Deterministic for a given `rng` state, the same discipline as every other helper in this module: no
    wall-clock, no unseeded randomness, so `(topic, seed[, difficulty])` still fully determines the question
    even when the wording itself varies by seed. `mathgen.text_quality.prompt_shape` is what measures whether
    that variety is real (masks the slot values back out) rather than cosmetic.
    """
    patterns = list(patterns)
    if not patterns:
        raise ValueError("phrase() needs at least one pattern")
    pattern = rng.choice(patterns)
    return pattern.format(**slots)


def compose(rng: random.Random, parts, *, final) -> dict:
    """Chain or bundle several sub-question steps into one composed
    generator dict.

    Contract
    --------
    `parts` is a sequence of callables, each ``part(rng, values) -> (value,
    working)``:

    - ``values`` is the list of every EARLIER part's own ``value``, in part
      order (empty for the first part). This is what lets a **chain** part
      read an earlier part's raw result (``values[0]``, ``values[1]``, ...)
      to compute its own; a **multi-part** generator's parts simply ignore
      ``values`` and compute independently. `compose()` itself does not
      know or care which shape it is being used for -- it only threads the
      growing list of values through, in order, to every later part.
    - ``value`` is the part's own raw computed VALUE -- a ``Fraction``, a
      ``(coefficient, exponent)`` tuple, a sympy object, whatever the next
      part or `final` needs to keep computing with. Never a formatted
      string: passing a part's rendered text here instead of its value is
      write by accident, because `final` below only ever receives this
      list of raw values, never any part's ``answer_text``.
    - ``working`` is one worked-step string (or a list of strings) for THIS
      part, shown to the learner so every intermediate value is checkable,
      not only the final one. Required and non-empty -- `compose` raises
      via `check` if a part returns no working line, because an unshown
      intermediate is exactly the failure this function is built to
      prevent.

    ``final(values, rng) -> dict`` is called once, after every part has
    run, with the complete list of raw part values (in part order, never
    any part's rendered text) and the same `rng` (for a final-step
    distractor draw, if the caller wants fresh randomness there rather than
    reusing a part's own draws). It returns the dict `compose()` augments
    and returns -- the SAME shape every other generator returns (`text`,
    `latex`, `answer`, `answer_text`, optionally `answer_latex`/
    `distractors`), so a generator using `compose()` still ends with its
    own `mathgen.core.check`/`check_equivalent` call proving that dict's
    answer against an independently-recomputed value, exactly like every
    generator that does not use `compose()`. `compose()` itself never calls
    `mathgen.make()`/the topic registry -- it is plumbing a generator's own
    function body calls, not a new entry point alongside `make()`.

    Why `working` is `compose()`'s job, not `final`'s: every other
    generator builds its own `working` because it is the only thing that
    ran. A composed generator has N+1 steps (N parts, one combining step),
    and if each part had to also remember to prepend its own line onto
    whatever `final` returns, one dropped line would silently lose a step.
    `compose()` does that prepending centrally, in part order, ahead of
    whatever `working` (if any) `final` itself returns, so it cannot be
    skipped.
    """
    values: list = []
    part_working: list[str] = []
    for spec in parts:
        value, working = spec(rng, list(values))
        if isinstance(working, str):
            working = [working]
        working = list(working or [])
        check(len(working) > 0, "compose: a part returned no working line")
        values.append(value)
        part_working.extend(working)
    result = dict(final(values, rng))
    result["working"] = [*part_working, *result.get("working", [])]
    return result


#: A plain signed integer/decimal is already valid LaTeX; do not sympify it
#: (that can normalise away a stated precision, e.g. "18.40" -> "18.4").
_PLAIN_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")

#: A bare "=" that is a relation, not part of "==", "<=", ">=" or "!=".
_BARE_EQUALS_RE = re.compile(r"(?<![<>=!])=(?!=)")

# A Python keyword `sympify` would parse as a real operator rather than a
#: word -- "is"/"and"/"or"/"not"/"in"/"if"/"else" -- present as a
#: standalone token. `sympify("a and b")` returns `b`, silently
#: dropping `a`; `sympify("10 is even")` becomes an identity comparison
#: that renders "\\text{False}". Word-boundary matching only: "android",
#: "orthogonal", "isosceles", "notation", "origin" etc. must not trip this
#: (no Python-keyword substring of theirs sits on a word boundary).
_BARE_KEYWORD_RE = re.compile(r"\b(?:is|and|or|not|in|if|else)\b")

#: A Symbol name short enough to be a math variable on its own ("x", "e") or
#: a subscripted one ("x_1", "a_n"). Anything else -- a multi-letter word
#: with no subscript -- is a candidate for `_is_plain_symbol` below.
_MATH_SYMBOL_NAME_RE = re.compile(r"^[A-Za-z]$|^[A-Za-z]_\w+$")


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split `text` on `sep`, but never inside ``()``/``[]``/``{}`` -- so a
    relation list ("x = 3, y = -5") splits on its comma while a pair
    ("(2, -5)") does not split on the one inside its own parens."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _is_plain_symbol(sym: Symbol) -> bool:
    """Whether `sym` is a deliberately non-mathematical word ("Obtuse",
    "Monday") rather than a variable. A short name ("x") or a subscripted one
    ("x_1") is a variable outright; otherwise a symbol whose LaTeX form
    carries no backslash was never going to render as anything but italic
    letters, which is exactly the defect `latex_answer` exists to avoid --
    a Greek-letter name ("theta" -> \\theta) still gets a backslash from
    sympy and so is not plain."""
    if _MATH_SYMBOL_NAME_RE.match(sym.name):
        return False
    return not _sympy_latex(sym).startswith("\\")


def _is_plain_expr(expr) -> bool:
    """Whether `expr` is entirely made of plain words -- a bare word, or a
    sequence of them ("Monday, Tuesday") -- with no actual mathematics."""
    seq = _as_sequence(expr)
    if seq is not None:
        return bool(seq) and all(_is_plain_expr(e) for e in seq)
    return isinstance(expr, Symbol) and _is_plain_symbol(expr)


def _latex_one_relation(segment: str, *, evaluate: bool) -> str:
    """LaTeX for one comma-separated piece of a relation string. A piece
    with no bare "=" is just an expression ("y" in "x = 3, y"); one with a
    bare "=" becomes an unevaluated `Eq` so "x = 3" cannot collapse the way
    `Eq(3, 3)` would to `True` -- the `Eq` wrapper is always built with
    `evaluate=False` regardless of `evaluate`, which only controls how each
    SIDE of the relation is sympified."""
    match = _BARE_EQUALS_RE.search(segment)
    if not match:
        return _sympy_latex(_sympify(segment.strip(), evaluate=evaluate))
    lhs = _sympify(segment[: match.start()].strip(), evaluate=evaluate)
    rhs = _sympify(segment[match.end() :].strip(), evaluate=evaluate)
    return _sympy_latex(Eq(lhs, rhs, evaluate=False))


def latex_answer(value, *, evaluate: bool = True, plain: bool = False) -> str:
    """Render a canonical answer as real LaTeX for a KaTeX-style renderer
    (`katex.renderToString`), or a ``plain:``-prefixed string for a
    deliberately non-mathematical answer ("Obtuse", "Monday, Tuesday") so
    the renderer's existing ``plain:`` branch takes over instead. Why this
    exists: a generator that built `answer_latex` by hand
    from plain text -- `simplify-surds`' `8*sqrt(7)` -- shipped it verbatim,
    and KaTeX does not fail over on plain text; it renders each letter as an
    italic maths variable ("8*sqrt(7)" as if it were five one-letter
    variables multiplied together), not a fallback to plain text.

    **Frozen contract.** Existing and new generators all build against
    this signature and this return shape. A case this cannot express gets a NEW keyword argument with a
    default that preserves today's behaviour -- never a change to what an existing
    call returns. ``evaluate`` is the first such keyword: pass ``evaluate=False`` for a FACTORED `answer_text`
    -- the default ``evaluate=True`` runs sympy's ordinary auto-simplification, which distributes `Number *
    Add` and would silently expand a factored canonical back out ("3*(x-2)*(x+4)" -> "3x^2 + 6x - 24"),
    destroying the very form the question asked for.

    ``plain`` is the second such keyword: pass ``plain=True`` when the CALLER already knows the answer is not
    mathematics, rather than relying on this function's own sympify-based plainness detection. That detection
    has real false negatives -- a word that collides with a live sympy class name ("Triangle", "Circle",
    "Categorical", "Equivalent" all resolve to a sympy type via `sympify` and produce garbage LaTeX, not a
    plain fallback), a word immediately followed by ``(...)`` ("Unstable (repelling)" parses as a function
    call), and a multi-clause relation whose EVERY clause happens to sympify cleanly with a plain-word symbol
    on one side ("left = 1, right = 2" renders as "left = 1,\\; right = 2" -- real LaTeX syntactically, still
    two words rendered as italic maths letters). ``plain=True`` skips all of that and returns
    ``f"plain:{value}"`` outright (idempotent if `value` already carries the prefix), so it is the correct
    choice whenever the generator's own branch already knows its answer is a label, category or sentence
    rather than a computed value.

    Accepts a `str` (an `answer_text`-shaped string: a surd, a fraction, a
    relation ``"x = 3"``, a pair ``"(2, -5)"``, a plain word, or LaTeX
    already) or a `sympy.Expr` / `Fraction` / plain number. A string already
    containing a backslash (real LaTeX) or already starting with ``plain:``
    is returned unchanged, so `latex_answer` is idempotent and never
    double-escapes. A bare decimal string ("18.40") is also returned
    unchanged so a stated precision survives -- pass the formatted STRING,
    not a raw `float`/`Float`, when the number of decimal places is
    load-bearing; `sympify` on a `float` normalises away a trailing zero.
    """
    if plain:
        text = str(value)
        return text if text.startswith("plain:") else f"plain:{text}"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "plain:"
        if "\\" in text or text.startswith("plain:"):
            return value
        if _PLAIN_NUMBER_RE.match(text):
            return text
        if _BARE_EQUALS_RE.search(text):
            try:
                return r",\; ".join(_latex_one_relation(seg, evaluate=evaluate) for seg in _split_top_level(text, ","))
            except (SympifyError, SyntaxError, TypeError, ZeroDivisionError, AttributeError):
                return f"plain:{value}"
        if _BARE_KEYWORD_RE.search(text):
            return f"plain:{value}"
        try:
            expr = _sympify(text, evaluate=evaluate)
        except (SympifyError, SyntaxError, TypeError, ZeroDivisionError, AttributeError):
            return f"plain:{value}"
    else:
        text = None
        try:
            expr = _sympify(value, evaluate=evaluate)
        except (SympifyError, SyntaxError, TypeError, ZeroDivisionError, AttributeError):
            return f"plain:{value}"
    if _is_plain_expr(expr):
        return f"plain:{text if text is not None else value}"
    return _sympy_latex(expr)


# --- latex_prompt -------------------------------------------------------

#: A run of 3+ ASCII letters, used both to decide whether a whole string is
#: pure math (no such run at all -> nothing to do) and to keep a genuine
#: word (never a bare 1-2 letter variable) out of a lifted math span.
_LETTER_RUN_RE = re.compile(r"[A-Za-z]{3,}")

#: Unicode maths symbols `latex_prompt` recognises as unambiguous -- these
#: never appear in ordinary English prose, so (unlike `+ - * / = < >`) they
#: need no adjacency guard to avoid colliding with a hyphenated word.
_UNI_SYMBOLS = "×÷≤≥≠±°π√−∫→"

# One "atom": the smallest unit that can appear inside a lifted math span.
# Tried in this order (first match wins at each position):
#  - `^(-1)` / `^-1` / `^8`      -- a caret exponent, with or without parens
#  - a single letter immediately before `^` and not itself preceded by
#    another letter (`x` in `x^(-1)`, never the trailing `e` of `time^2`)
#  - a bare integer or decimal (`3.59`) -- NOT a trigger by itself (a bare
#    number stays inside `\text{}` unless it sits next to a real trigger)
#  - one of the unicode symbols above
#  - one of `+ - * / = < >`, but ONLY when neither neighbour is a letter --
#    this is what keeps "SI-prefix" and "and/or" out of math mode: a real
#    minus/operator in this corpus is always flanked by digits, spaces,
#    parens or another operator, never glued directly onto a word.
# Deliberately NO bare `\^` fallback: a caret this parser cannot resolve to
# a numeric exponent (an algebraic one, `2^x`, `r^(n-1)`, `x^(1/2)`) must
# NOT become its own atom, or the run stops one atom short with a caret at
# the very end (`2^` before the next `\text{...}`) -- a caret with no
# argument, which is a KaTeX parse error, not "worse than plain text" but
# actually broken. Left unmatched, that `^` simply falls through to the
# surrounding prose and is escaped like any other special character
# (`\textasciicircum{}`) -- exactly the "parser cannot handle it, leave it
# in `\text{}`" fallback this whole function is built on. (Measured: swept
# every active skill's `text` for `\^(?!-?\d)` before adding this rule --
_ATOM = (
    r"(?:\^\(-?\d+\)"
    r"|\^-?\d+"
    r"|(?<![A-Za-z])[A-Za-z](?=\^)"
    r"|\d+(?:\.\d+)?"
    r"|[" + _UNI_SYMBOLS + r"]"
    r"|(?<![A-Za-z])[+\-*/=<>](?![A-Za-z]))"
)
#: A maximal run of atoms, adjacent or separated by whitespace only -- this
#: is the unit `latex_prompt` decides to lift (or not) as a whole, so
#: "3.59 × 10^8" lifts together even though "3.59" alone carries no trigger.
_RUN_RE = re.compile(_ATOM + r"(?:\s*" + _ATOM + r")*")
#: Whether a matched run actually carries something that makes it math --
#: a run of bare numbers alone (no operator/^/symbol) does not.
_TRIGGER_RE = re.compile(r"[+\-*/=<>^" + _UNI_SYMBOLS + r"]")

# `\text{...}` escaping. Order doesn't matter -- `_escape_prose` substitutes in one pass over the
# ORIGINAL string, so an escape's own backslash is never re-escaped.
_PROSE_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "$": r"\$",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_PROSE_ESCAPE_RE = re.compile("[" + re.escape("".join(_PROSE_ESCAPE_MAP)) + "]")


def _escape_prose(s: str) -> str:
    return _PROSE_ESCAPE_RE.sub(lambda m: _PROSE_ESCAPE_MAP[m.group()], s)


def _convert_math_span(s: str) -> str:
    """Turn one already-decided-liftable raw run into real LaTeX. Order
    matters only for `°` vs the exponent regexes: `°` is textually replaced
    with `^\\circ` first and never matches the digit-exponent patterns that
    run after it."""
    s = s.replace("−", "-")  # U+2212 MINUS SIGN -> ASCII hyphen-minus
    s = s.replace("×", r"\times")
    s = s.replace("÷", r"\div")
    s = s.replace("≤", r"\le")
    s = s.replace("≥", r"\ge")
    s = s.replace("≠", r"\ne")
    s = s.replace("±", r"\pm")
    s = s.replace("°", r"^\circ")
    s = s.replace("π", r"\pi")
    s = re.sub(r"\^\((-?\d+)\)", r"^{\1}", s)  # x^(-1) -> x^{-1}
    s = re.sub(r"\^(-?\d+)", r"^{\1}", s)  # 10^8   -> 10^{8}
    return s


_INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)", re.S)


def latex_from_inline(s: str) -> str:
    r"""Convert a prompt written as prose with inline maths (``Use \(x = 0\) on \(y = f(x)\).``) to the
    whole-string form `latex_prompt` produces: each prose run wrapped in ``\text{...}`` (escaped), each
    ``\(...\)`` span left as bare LaTeX in a {...} group between them. A string with no ``\(`` is returned unchanged."""
    if "\\(" not in s:
        return s
    parts, pos = [], 0
    for m in _INLINE_MATH_RE.finditer(s):
        if m.start() > pos:
            parts.append(r"\text{" + _escape_prose(s[pos : m.start()]) + "}")
        parts.append("{" + m.group(1) + "}")  # a group, so neighbouring maths never runs together
        pos = m.end()
    if pos < len(s):
        parts.append(r"\text{" + _escape_prose(s[pos:]) + "}")
    return "".join(parts)


def latex_prompt(text: str) -> str:
    r"""Render a plain-text question stem (`Question.text`) as real LaTeX
    for a KaTeX-style renderer (`katex.renderToString`). This is the prompt-side twin of
    `latex_answer` above -- read that docstring first, the shape is the
    same: idempotent, string in, string out, a case it cannot express is
    left exactly as plain text rather than mangled.

    **Frozen contract**, same terms as `latex_answer`: a case this cannot
    express gets a NEW keyword argument with a default that preserves
    today's behaviour, never a change to what an existing call returns.

    A string already containing a backslash is assumed to already be real
    LaTeX (or this function's own prior output) and is returned unchanged
    -- this is what makes `latex_prompt(latex_prompt(s)) == latex_prompt(s)`
    hold, exactly like `latex_answer`'s idempotency on a string that already
    contains a backslash.

    Pure-math input -- no run of 3+ letters anywhere -- is also returned
    unchanged, byte-identical: `"56 + 71"` must come back exactly as given,
    since `addition`/`subtraction` produce pure-math text and their `latex`
    has always equalled `text`.

    Anything else is split into alternating spans: a prose span is wrapped
    in `\text{...}` with `$ % & # _ { } ~ ^ \` escaped; a span that carries
    an operator (`+ - * / = < >`), a `^`, or a unicode maths symbol (`× ÷ ≤
    ≥ ≠ ± ° π √ − ∫ →`) is lifted out of `\text{}` and converted to real
    LaTeX (`10^8` -> `10^{8}`, `x^(-1)` -> `x^{-1}`, `×` -> `\times`, etc).
    A bare integer or decimal with no such trigger stays inside `\text{}`
    -- digits render upright either way, and lifting one out on its own
    (e.g. the `2` in "to 2 decimal places") is what would make
    `tests/test_question_latex_instructions.py` stop finding "2 decimal
    places" as one phrase; nothing is gained by lifting it.

    Whatever the span parser cannot confidently classify as math is left
    inside `\text{}` -- the worst case is exactly today's plain-text
    appearance, never worse.
    """
    if not text or "\\" in text:
        return text
    if not _LETTER_RUN_RE.search(text):
        return text

    parts: list[str] = []
    pending: list[str] = []
    pos = 0
    for m in _RUN_RE.finditer(text):
        start, end = m.span()
        if start > pos:
            pending.append(text[pos:start])
        run = m.group()
        if _TRIGGER_RE.search(run):
            if pending:
                parts.append(r"\text{" + _escape_prose("".join(pending)) + "}")
                pending = []
            parts.append(_convert_math_span(run))
        else:
            pending.append(run)
        pos = end
    if pos < len(text):
        pending.append(text[pos:])
    if pending:
        parts.append(r"\text{" + _escape_prose("".join(pending)) + "}")
    return "".join(parts)


def xy_solution(x0: int, y0: int) -> dict:
    """The answer/answer_text/answer_latex/distractors block for a solved
    (x, y) pair. Shared by the simultaneous-equations generator and the
    matrix-form linear system, which present the same answer the same way.

    Misconceptions: swapped the pair; sign error on one unknown.
    """
    ans_text = f"x = {x0}, y = {y0}"
    cands = [f"x = {y0}, y = {x0}", f"x = {-x0}, y = {y0}", f"x = {x0}, y = {-y0}"]
    return dict(
        answer={"x": x0, "y": y0},
        answer_text=ans_text,
        answer_latex=rf"x = {x0},\; y = {y0}",
        distractors=pick_distractors(ans_text, cands),
    )


# --- symbolic equivalence, without paying for simplify() every time ---------
#
# `simplify(a - b) == 0` is the correct test and an expensive one: on the
# product/quotient/chain-rule topic it runs trigsimp and costs about 35 ms a
# call. A generator makes four such calls per question -- one to confirm its
# own answer and three to confirm each distractor is genuinely different -- and
# the grader makes one per option. The three "different" calls are the common
# case and the slow one, because simplify only gives up after exhausting its
# rewrites.
#
# A numeric probe settles those in microseconds. It is deliberately one-sided:
# it may only ever report "these are definitely different", and anything else
# falls through to simplify. So it can turn a slow False into a fast False and
# can never turn a True into a False -- which is the property that matters,
# since a wrong answer here marks a correct learner wrong.

_PROBE_POINTS = (
    Rational(3, 7),
    Rational(11, 5),
    Rational(-9, 4),
    Rational(17, 13),
    Rational(5, 2),
    Rational(-7, 3),
)


def _sympify(value, *, evaluate: bool = True):
    """`sympify`, except that text is read by `mathgen.grading.safe_parse`'s restricted parser.

    `symbolically_equal`, `pairs_equal` and `latex_answer` accept strings because generators write answers and
    distractors as text. Plain `sympify(str)` `eval`s it with the builtins in scope, so a string that came from
    anywhere else (a learner's reply) would run as Python. The grader never passes them text -- it parses
    a reply with `parse_expression_safely` first and hands over SymPy objects (`typed._parse_expression`,
    `typed._expected_expression`) -- but that was a convention, not a guarantee, so text is now read with
    the same token allow-list and namespace as a reply: nothing in it can reach `__import__`, an attribute
    or a keyword. Unreadable text raises `SympifyError`, as before. This is the safety half only; it does
    not apply the cost guard (the strings are the generators' own and small), so a reply must still go
    through `parse_expression_safely`, which does.
    """
    if isinstance(value, (str, tuple, list)):  # a tuple or list of strings is text too
        from mathgen.grading.safe_parse import sympify_safely  # lazy: mathgen.grading imports this module

        return sympify_safely(value, evaluate=evaluate)
    return sympify(value, evaluate=evaluate)


def _as_sequence(value):
    """`value` as a plain tuple if it is a sequence answer (a coordinate
    pair, a solved ``(x, y)``) -- a `sympy.Tuple`, Python `tuple` or `list`
    -- else `None`. A raw string is deliberately excluded even though it is
    Python-iterable: "18.4" is a scalar answer, not a two-character
    sequence, and `sympify` of a pair-shaped string ("(2, -5)") already
    returns a plain Python `tuple` (not a `sympy.Tuple`), which is what this
    also catches."""
    if isinstance(value, (tuple, list, Tuple)):
        return tuple(value)
    return None


def pairs_equal(a, b, tol=None) -> bool:
    """Whether two sequences are the same value, element-wise. Same length
    is required; a sequence compared against a non-sequence, one of a
    different length, or one holding an element that fails to sympify, is
    simply not equal -- never an exception.

    Why this exists: `symbolically_equal` handed a pair-valued
    string straight to `simplify(left - right)`, and `sympy.Tuple`/`tuple`
    subtraction itself raises `TypeError` -- there is no way to compare two
    sequences by subtracting them, so they need their own element-wise
    comparison (each element delegated to `symbolically_equal`, which uses
    `.is_zero` on the simplified difference to match this function's own
    idiom -- measured on the current code, `==` and `.is_zero` already agree for
    every equal-Float case actually reachable via subtraction, since sympy's
    Float arithmetic collapses an exact-zero difference to `Integer(0)`
    directly; `.is_zero` is used for consistency and as a defensive margin,
    not because a live disagreement was found). Pass `tol` for a numeric
    within-tolerance comparison instead (e.g. two floats that should agree
    to a stated precision, or a spoken-transcript grade).
    """
    a_seq, b_seq = _as_sequence(a), _as_sequence(b)
    if a_seq is None:
        try:
            a_seq = _as_sequence(_sympify(a))
        except (SympifyError, SyntaxError, TypeError, ZeroDivisionError, AttributeError):
            a_seq = None
    if b_seq is None:
        try:
            b_seq = _as_sequence(_sympify(b))
        except (SympifyError, SyntaxError, TypeError, ZeroDivisionError, AttributeError):
            b_seq = None
    if a_seq is None or b_seq is None or len(a_seq) != len(b_seq):
        return False
    for x, y in zip(a_seq, b_seq):
        if tol is not None:
            try:
                diff = complex(_sympify(x).evalf(30)) - complex(_sympify(y).evalf(30))
            except (TypeError, ValueError, SympifyError):
                return False
            if abs(diff) > float(tol):
                return False
        else:
            try:
                if not symbolically_equal(x, y):
                    return False
            except (SympifyError, SyntaxError, TypeError, ZeroDivisionError, AttributeError):
                return False
    return True


def _probe_differs(left, right) -> bool:
    """True only when a numeric probe PROVES the two expressions differ.

    False means "inconclusive", not "equal": a point where either side is
    undefined is skipped, and a difference is only believed when it is large
    next to the values it came from, so floating-point noise in an identity like
    sin(x)**2 + cos(x)**2 - 1 never counts as a difference.

    The values are compared as SymPy numbers, never converted to a Python
    float. A float tops out near 1e308, and a point where either side was
    bigger (`cosh(4096)`, `(x + 1)**3000`) used to be skipped as "huge";
    with every point skipped the caller went on to `simplify`, which takes
    minutes on such a value. SymPy's own numbers have no such ceiling, so the
    probe still proves the two expressions differ.

    `left`/`right` must already be sympified -- the caller, `symbolically_equal`,
    sympifies once and passes the result down both here and to its own
    `simplify` call. This function used to sympify again into its own local
    names while `symbolically_equal` went on to use the ORIGINAL, un-sympified
    arguments -- so a plain string pair with no free symbols (a bare number,
    "18.4"; a coordinate pair, "(18.4, 2.5)") reached `simplify(str - str)`
    and raised `TypeError` instead of ever being compared.
    """
    difference = left - right
    if not difference.free_symbols:
        # Two constants (log(2)/pi**2 against 0.0001) need no probe points: evaluate them. Without this the
        # caller went straight to `simplify`, which spent minutes in powsimp on a learner's small decimal.
        # 30 digits are certified by evalf, so a gap above 1e-20 of the scale is a real difference.
        try:
            value = abs(difference.evalf(30))
            scale = max(abs(left.evalf(30)), abs(right.evalf(30)), Integer(1))
            if not (value.is_number and scale.is_number and value.is_finite and scale.is_finite):
                return False
            return bool(value > Float("1e-20") * scale)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return False
    # Substitute every symbol either side carries, not only the difference's: an `expr + C` pair
    # cancels C in the difference but not in `left`/`right`, whose scale evaluation then raised on
    # every probe point and made the probe always inconclusive, sending each such distractor check
    # to the slow `simplify` fallback.
    if difference.atoms(AppliedUndef):
        # An unknown function (text such as "k(k + 1)" reads as one) cannot be evaluated, so evalf falls back to a
        # symbolic .subs() that can take minutes on the other side. Leave it to the caller's symbolic check.
        return False
    symbols_used = sorted(left.free_symbols | right.free_symbols, key=str)
    hits = 0
    for offset in range(len(_PROBE_POINTS)):
        substitutions = {
            symbol: _PROBE_POINTS[(offset + index) % len(_PROBE_POINTS)] for index, symbol in enumerate(symbols_used)
        }
        try:
            # evalf(subs=...) evaluates numerically straight away. A symbolic .subs() first auto-evaluates
            # the substituted function exactly, and for sqrt(acos(x)**2) at x = 11/5 that took 40 s.
            value = abs(difference.evalf(30, subs=substitutions))
            if value.is_number and value.is_finite and not value > Float("1e-9"):
                # `scale` is never below 1, so a value this small cannot exceed 1e-9 * scale whatever `scale` is:
                # this point can only be inconclusive or "no difference". Skipping the two scale evaluations
                # leaves the outcome unchanged and saves two thirds of the evalf work on equal pairs.
                continue
            scale = max(
                abs(left.evalf(30, subs=substitutions)),
                abs(right.evalf(30, subs=substitutions)),
                Integer(1),
            )
            if not (value.is_number and scale.is_number and value.is_finite and scale.is_finite):
                continue  # undefined (zoo, nan) or not a number at this point
            differs = bool(value > Float("1e-9") * scale)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            continue
        if differs:
            hits += 1
            if hits >= 2:
                return True
    return False


def _has_undefined(value) -> bool:
    """Whether a sympified value contains zoo, nan or an infinity (anything else, a plain set included, is False)."""
    has = getattr(value, "has", None)
    return bool(has and has(S.ComplexInfinity, S.NaN, S.Infinity, S.NegativeInfinity))


def symbolically_equal(left, right) -> bool:
    """Whether two sympy expressions are equivalent. Exact; just not always slow.

    Sympifies once, here, and passes the sympified values down to
    `_probe_differs` -- see that function's docstring for the bug this
    fixes. A sympified value that is a sequence (a coordinate pair, a solved
    ``(x, y)``) cannot be subtracted (`sympy.Tuple`/`tuple` subtraction
    raises `TypeError`), so those go through `pairs_equal` instead; a
    sequence compared against a non-sequence is not equal, never an
    exception.
    """
    left, right = _sympify(left), _sympify(right)
    left_seq, right_seq = _as_sequence(left), _as_sequence(right)
    if left_seq is not None or right_seq is not None:
        if left_seq is None or right_seq is None:
            return False
        return pairs_equal(left_seq, right_seq)
    if _has_undefined(left) or _has_undefined(right):
        # An infinite or undefined value is only "equal" to the very same expression. Sent to `simplify`, a
        # reply such as 1/0 or log(0) spent seconds there and could never be shown equal to a finite answer.
        return bool(left == right)
    return _compare_sympified(_Pair(left, right))


class _Pair:
    """A sympified (left, right) pair, hashed and compared by `srepr` so `_compare_sympified` can cache on it."""

    __slots__ = ("left", "right", "_key")

    def __init__(self, left, right):
        self.left, self.right = left, right
        self._key = (srepr(left), srepr(right))

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return isinstance(other, _Pair) and self._key == other._key


@lru_cache(maxsize=20000)
def _compare_sympified(pair: _Pair) -> bool:
    """The probe, the cheap exact proofs, then `simplify`, for one sympified pair; the result is cached.

    Generators and the contract tests compare the same expression pairs again and again (a distractor set repeats
    across seeds), and the answer is a pure function of the two expressions' `srepr`. Exceptions are not cached."""
    left, right = pair.left, pair.right
    if _probe_differs(left, right):
        return False
    if _cheap_proof_of_equal(left, right):
        return True
    return bool(simplify(left - right).is_zero)


def _cheap_proof_of_equal(left, right) -> bool:
    """True only when a cheap EXACT step proves `left - right` is zero; False means "not proved", never "different".

    Each step is a SymPy rewrite that must return exactly 0 (structural equality of the two sides, `expand`, `cancel`
    for a rational function, `expand` of `radsimp` for surd denominators, `expand_trig` for multiple-angle identities). Anything else, an exception included, falls
    through to the caller's `simplify`. No numeric evidence is ever used here.
    """
    try:
        if left == right:
            return True
        difference = left - right
        if expand(difference) == 0:
            return True
        if difference.is_rational_function() is True and cancel(difference) == 0:
            return True
        if expand(radsimp(difference)) == 0:
            return True
        if difference.has(TrigonometricFunction) and expand(expand_trig(difference)) == 0:
            return True
    except Exception:  # noqa: BLE001 - any failure just means "not proved here"; simplify decides
        return False
    return False


# A generator has exactly two symbolic things to prove about every question it
# makes: that the answer it worked out matches the answer it derived a second
# way, and that no distractor it offers is secretly the same value. Both were
# being written as `check(simplify(a - b) == 0, ...)` at 49 sites across 11
# generators, which is slow (see symbolically_equal) and easy to get backwards
# -- `!=` where `==` was meant passes silently on every question. Say which of
# the two you mean instead.


def check_equivalent(actual, expected, msg: str = "answer verification failed"):
    """Fail unless the two expressions are the same value.

    For the independent-route check: derive the answer a second way and pass
    both, so a slip in either route is caught rather than confirmed.
    """
    check(symbolically_equal(actual, expected), msg)


def check_distinct(answer, *candidates, msg: str = "a distractor equals the answer"):
    """Fail unless every candidate is a genuinely different value from `answer`.

    A distractor equal to the answer is a question with two correct options, so
    check every candidate unconditionally -- including the ones you are sure
    about, since those are the ones nobody re-reads.

    Compares each candidate against `answer` only -- `n - 1` comparisons. It
    CANNOT see two candidates that equal each other but not the answer (two
    distractors offering the same wrong value). For that, use
    `check_pairwise_distinct` instead, passing the answer as one of its
    `*values` if the answer must also be distinct from every candidate.
    """
    for candidate in candidates:
        check(not symbolically_equal(answer, candidate), msg)


def pairwise_distinct(*values) -> bool:
    """True iff every pair in `values` is genuinely different -- `n choose 2`
    comparisons, not `n - 1`. The non-raising sibling of `check_pairwise_distinct`
    below, sharing its comparison logic exactly (that function is this one plus
    a `check()`), for a caller that wants a boolean to act on -- e.g. a
    redraw-until-distinct guard loop -- rather than an exception."""
    for i, left in enumerate(values):
        for right in values[i + 1 :]:
            if symbolically_equal(left, right):
                return False
    return True


def check_pairwise_distinct(*values, msg: str = "a distractor collides with another option or the answer"):
    """Fail unless every pair in `values` is genuinely different -- `n choose 2`
    comparisons, not `n - 1`.

    Unlike `check_distinct`, there is no privileged `answer` argument: every
    value is checked against every other value, including candidate-vs-candidate
    collisions that `check_distinct` cannot see. `pick_distractors`'s own
    value-equality dedup only compares a candidate against the answer, never
    against another candidate (it only catches a candidate-vs-candidate clash
    when the two are the exact same string) -- so two distractors with equal
    value but different spelling ("3.0" and "3", both wrong) both survive it
    and ship as duplicate wrong options. Pass the answer as one of `*values`
    if it must also be checked against every candidate.

    Use `check_distinct` instead when you only need each candidate checked
    against a single answer and do not care whether candidates collide with
    each other.
    """
    check(pairwise_distinct(*values), msg)


# The eleven helpers below were found hand-rolled, near-verbatim, across ~40
# generator files. They are built and unit-tested in isolation here; the older
# local copies in generators (`_fixed_string`, `_round_half_up`, etc.) are left
# exactly as they are, and new code should call these instead.


def exact_decimal_text(value) -> str:
    """An exact rational as plain decimal text with no trailing zeros ("0.12", "1.7", "12"), never as a fraction
    such as "3/25". The value must terminate."""
    exact = Fraction(value)
    places = 0
    while (exact * 10**places).denominator != 1:
        places += 1
        check(places <= 12, f"{value} does not terminate as a decimal")
    return fixed_point_string(int(exact * 10**places), places)


def fixed_point_string(scaled_int: int, places: int) -> str:
    """Sign-prefixed fixed-point string from an integer already scaled by
    ``10 ** places`` (so ``fixed_point_string(-305, 2)`` is "-3.05"). This
    exact zfill/slice body was hand-copied into ~10 generator files; a caller
    that already has a scaled integer should call this instead of re-deriving
    the arithmetic. ``places=0`` returns a bare integer, no decimal point."""
    sign = "-" if scaled_int < 0 else ""
    scaled_int = abs(scaled_int)
    if places == 0:
        return f"{sign}{scaled_int}"
    digits = str(scaled_int).zfill(places + 1)
    whole, frac = digits[:-places], digits[-places:]
    return f"{sign}{whole}.{frac}"


def exact_decimal_string(value: Fraction, max_places: int) -> str:
    """The shortest terminating decimal string for `value` within
    `max_places`. Raises `ValueError` if `value` does not terminate that
    finely -- that is a signal the CALLER constructed a value it should not
    have (e.g. a `Fraction` with a denominator that cannot terminate in
    decimal within the stated precision), not an expected runtime path to
    catch and continue past.

    This is the SHORTEST form, not a fixed-width one: it renders `0.10` as `0.1`, which is a genuine hazard
    for a graded typed-numeric answer -- `grade_number` derives its comparison tolerance from the number of
    decimal places `answer_text` itself *displays* (`mathgen/grading/typed.py:_decimal_places`), so an answer
    that under-states its own stated precision this way is graded with a tolerance ten times too loose. Use
    `stated_precision_string` instead when the caller has a STATED precision `d` that must display at exactly
    `d` places, never fewer -- e.g. anywhere the question text says "to 2 decimal places" and a distractor
    could otherwise land in an under-displayed answer's now-too-loose tolerance. Use `rounded_decimal_string`
    instead when `value` may not terminate at all and a distractor display (not a graded canonical) is
    acceptable."""
    for places in range(0, max_places + 1):
        scaled = value * (10**places)
        if scaled.denominator == 1:
            return fixed_point_string(scaled.numerator, places)
    raise ValueError(f"{value} does not terminate within {max_places} decimal places")


def rounded_decimal_string(value: Fraction, max_places: int) -> str:
    """Like `exact_decimal_string`, but for a DISTRACTOR display value only:
    rounds half-up to `max_places` instead of raising when `value` does not
    terminate exactly. Keeps the "+infinity" tie-break formula
    (`floor(value * scale + 1/2)`) that every existing duplicate of this
    function uses, deliberately NOT routed through `round_half_up` below --
    the two have different tie-break rules, and every caller in this family formats a non-negative
    physical measurement (length/area/volume) where the two tie-break rules
    coincide anyway, so changing the implementation here would be scope
    creep with no behavioural benefit.

    Like `exact_decimal_string`, this can still under-display a value that
    happens to land on a shorter terminating decimal (e.g. rounding to
    `0.10` renders `0.1`) -- fine for a distractor display, which this
    function is scoped to, but NOT a safe substitute for
    `stated_precision_string` on a graded canonical `answer_text`:
    use that one instead when the caller needs exactly `d`
    places on the value the grader will compare against."""
    try:
        return exact_decimal_string(value, max_places)
    except ValueError:
        scaled = math.floor(value * (10**max_places) + Fraction(1, 2))
        return fixed_point_string(scaled, max_places)


def stated_precision_string(value: Fraction, d: int) -> str:
    """Display `value` at EXACTLY `d` decimal places, never fewer -- unlike `exact_decimal_string`, which returns
    the shortest terminating string within `max_places` and would collapse a discrepancy of "0.10" to "0.1".
    That collapse is not cosmetic: `grade_number` derives its comparison tolerance from the number of decimal
    places `answer_text` itself *displays* (`mathgen/grading/typed.py:_decimal_places`), so a canonical answer
    that under-states its own stated precision is graded with a tolerance ten times too loose, matching (and
    false-accepting) a distractor a whole rounding-unit away. Use `exact_decimal_string` instead when there is
    no single stated precision to hit -- the caller genuinely wants the shortest terminating form within a
    maximum. Use `rounded_decimal_string` instead for a distractor display whose value may not terminate
    exactly at all; that one rounds half-up rather than raising, but calls `exact_decimal_string` first and so
    shares ITS under-display risk (this function does not have that risk itself -- eliminating it is the whole
    point), so never reach for `rounded_decimal_string` on a graded canonical either.

    Every candidate value passed here should already sit on the exact
    `10**-d` grid (e.g. already rounded via `round_half_up(..., d)`); the
    `denominator == 1` check after scaling is a genuine invariant, not a
    defensive no-op, and raises the project's `VerificationError` (via
    `check`) rather than silently rounding a value the caller should not
    have constructed off-grid in the first place."""
    scaled = value * (10**d)
    check(scaled.denominator == 1, f"{value} is not on the {d}-decimal-place grid")
    return fixed_point_string(scaled.numerator, d)


def numeric_answer(value: Fraction):
    """Convert a `Fraction` to the plain `int`/`float` that `question.answer`
    must hold. `grade_number`'s stated-precision tolerance only engages for a
    plain `int`/`float` answer, never a raw `Fraction`; this is the required
    conversion at that boundary, duplicated ~10 times today."""
    return int(value) if value.denominator == 1 else float(value)


def round_half_up(value: Fraction, places: int = 0) -> Fraction:
    """Round `value` to `places` decimal places, ties AWAY FROM ZERO (the
    `Decimal`/`ROUND_HALF_UP` convention) -- unlike the "+infinity" tie-break
    that most of today's hand-rolled duplicates use (`rounded_decimal_string`
    above deliberately keeps that older rule; this is the one new call sites
    should adopt going forward). For non-negative input the two rules
    coincide, so introducing this changes no existing output PROVIDED every
    migrated call site is first verified non-negative -- that verification is
    the caller's job, not this helper's. `Fraction`-typed
    only: a caller holding a `float` converts to `Fraction` at its own call
    site, not here."""
    scale = 10**places
    scaled = value * scale
    sign = -1 if scaled < 0 else 1
    magnitude = math.floor(abs(scaled) + Fraction(1, 2))
    return Fraction(sign * magnitude, scale)


def check_conversion(value, from_unit: str, to_unit: str, expected, factors: dict, label: str = "conversion") -> None:
    """Independently re-derive a unit conversion via a pivot unit (multiply
    by `from_unit`'s factor, divide by `to_unit`'s) so construction and
    verification never share the same arithmetic path. Duplicated
    near-verbatim across measurement-conversion generators, differing only
    in the pivot-unit name and the error label."""
    pivot = value * factors[from_unit]
    recomputed = pivot / factors[to_unit]
    check(recomputed == expected, f"{label} mismatch: {value} {from_unit} -> {to_unit}")


def check_combine(
    values: dict, other_unit: str, target_unit: str, operation: str, expected, factors: dict, label: str = "combine"
) -> None:
    """Independently re-derive a sum/difference of two measurements given in
    different units, by converting `other_unit` onto `target_unit` via a
    pivot and then combining -- the same independent-route idea as
    `check_conversion`, for the "combine two measurements" shape duplicated
    near-verbatim across 4 measurement-conversion generators."""
    other_value = values[other_unit]
    same_value = values[target_unit]
    pivot = other_value * factors[other_unit]
    converted = pivot / factors[target_unit]
    if operation == "sum":
        recomputed = same_value + converted
    elif operation == "difference":
        recomputed = abs(same_value - converted)
    else:
        raise ValueError(f"unknown operation: {operation!r}")
    check(recomputed == expected, f"{label} mismatch")


def format_pi_multiple(coeff) -> str:
    """Format a rational multiple of pi the way a student would write it:
    ``Fraction(5, 6)`` -> "5pi/6", ``Fraction(1, 1)`` -> "pi",
    ``Fraction(-1, 2)`` -> "-pi/2", ``0`` -> "0". The `Fraction(coeff)`
    coercion at the top accepts a bare `int` too, matching the widest of the
    4 existing call sites' own contracts; final logic verified byte-identical
    across all 4."""
    coeff = Fraction(coeff)
    if coeff == 0:
        return "0"
    sign = "-" if coeff < 0 else ""
    coeff = abs(coeff)
    num, den = coeff.numerator, coeff.denominator
    num_part = "" if num == 1 else str(num)
    if den == 1:
        return f"{sign}{num_part}pi"
    return f"{sign}{num_part}pi/{den}"


def latex_pi_multiple(coeff) -> str:
    r"""The LaTeX twin of `format_pi_multiple`, for the `answer_latex` of the
    same answer: ``Fraction(5, 6)`` -> ``\frac{5\pi}{6}``,
    ``Fraction(1, 1)`` -> ``\pi``, ``Fraction(-1, 2)`` -> ``-\frac{\pi}{2}``,
    ``0`` -> "0", ``6`` -> ``6\pi``. Same `Fraction(coeff)` coercion, same
    sign and unit-numerator handling, so the pair can never disagree about
    which value they are describing.

    Exists because `format_pi_multiple`'s output is a *typed* string ("6pi",
    "13pi/12") and KaTeX renders that as italic maths letters, not a pi --
    and `latex_answer` cannot rescue it either, because sympify has no
    implicit multiplication and so reads "6pi" as an unknown symbol and
    falls back to ``plain:``. A generator with the coefficient in hand calls
    this instead of building a sympifiable "6*pi" by hand."""
    coeff = Fraction(coeff)
    if coeff == 0:
        return "0"
    sign = "-" if coeff < 0 else ""
    coeff = abs(coeff)
    num, den = coeff.numerator, coeff.denominator
    num_part = "" if num == 1 else str(num)
    if den == 1:
        return rf"{sign}{num_part}\pi"
    return rf"{sign}\frac{{{num_part}\pi}}{{{den}}}"


def reversed_and_adjacent_swap(ordered: list) -> tuple[list, list]:
    """The two most common ordering-misconception distractors for a list a
    student was asked to put in order: the exact reversal, and the two
    middle elements swapped. Narrowly scoped to just this pair -- NOT the
    surrounding shuffle/join preamble some callers also share, which stays
    local because at least one caller's preamble has its own documented
    divergence from the others."""
    reversed_list = list(reversed(ordered))
    adjacent_swap = list(ordered)
    mid = max(1, len(adjacent_swap) // 2)
    adjacent_swap[mid - 1], adjacent_swap[mid] = adjacent_swap[mid], adjacent_swap[mid - 1]
    return reversed_list, adjacent_swap


def rounds_unambiguously(value: float, places: int, margin: float = 0.05) -> bool:
    """True when rounding `value` to `places` decimals is NOT a coin-flip on
    a ".5" boundary -- duplicated byte-identically across (at time of the
    survey) 4 files, each one's own comment independently flagging it as
    unresolved duplication."""
    scaled = abs(value) * (10**places)
    return abs(scaled - math.floor(scaled) - 0.5) > margin


def currency_from_cents(cents: int, symbol: str = "$", *, allow_negative: bool = True) -> str:
    """Format an integer count of cents as a currency string, e.g.
    ``currency_from_cents(450)`` -> "$4.50", ``currency_from_cents(-450)`` ->
    "-$4.50". ``allow_negative=False`` is not decorative: one caller
    (`p8e_currency_exchange_build.py`) relies on the
    negative-rejecting `check()` to catch a real construction bug, and
    folding the two shapes into one function must not silently delete that
    guard."""
    cents = int(cents)
    if not allow_negative:
        check(cents >= 0, f"a {symbol} amount must never be negative")
        return f"{symbol}{cents // 100}.{cents % 100:02d}"
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{symbol}{cents // 100}.{cents % 100:02d}"


def money_cents_rounded_2dp(cents) -> str:
    """A `Fraction` (or int) count of cents -- possibly NOT an exact integer, e.g. a misconception's own
    inverted-rate arithmetic -- rounded half-up to the cent and formatted as a bare 2dp dollar string (no
    currency symbol), e.g. ``money_cents_rounded_2dp(Fraction(4505, 10))`` -> "4.51" (450.5 cents rounds
    half-up to 451 cents). For a DISTRACTOR display value only: `currency_from_cents` requires an
    already-integer `cents: int` and never rounds, so this composes `round_half_up` +
    `stated_precision_string` to land any `Fraction` cents value onto the cent grid first. Never use this for
    a canonical `answer_text`, which must already be an exact integer cents value formatted via
    `currency_from_cents` or `stated_precision_string` directly."""
    return stated_precision_string(round_half_up(Fraction(cents) / 100, 2), 2)


def prompt_safe_str(expr) -> str:
    """`str(expr)` (a sympy expression, or anything `str`-able) with Python's "**" power operator rendered "^"
    and "*" rendered "·" (a real multiplication dot, never a literal asterisk) -- for PROMPT text only, where
    `mathgen.text_quality`'s `star_operator`/`double_star` checks forbid both anywhere in `Question.text`.
    Matches the convention `p13c_antideriv_notation_build.py`'s own `_disp` and
    `p12c_trig_apply_implicit_diff_build.py`'s hand-built prompt strings already use. Does no other cleanup
    (no `log(` -> `ln(`, no `exp(...)` -> `e^(...)`) -- those remain per-generator, since they are not always
    grader-safe to apply blindly. Use `value_safe_str` instead for a VALUE string (`answer_text`/a
    distractor), where a bare "*" is fine but "**" is still forbidden."""
    return str(expr).replace("**", "^").replace("*", "·")


def value_safe_str(expr) -> str:
    """`str(expr)` with only "**" rendered "^" -- for a VALUE string (`answer_text`/a distractor): `double_star`
    is forbidden in every learner-facing field, but a bare "*" is fine outside the prompt itself. Matches
    `differential_equation`'s own `_ode_exponential` convention, ``str(ans).replace('**', '^')``, and
    `p13c_antideriv_notation_build.py`'s `_value`. Use `prompt_safe_str` instead for `Question.text`, where a
    bare "*" is also forbidden."""
    return str(expr).replace("**", "^")


# --- paper notation ----------------------------------------------------------
#
# Generators build prompts and answers with Python's spelling: ``x**2``, ``2*x``, ``sqrt(21)``, ``exp(-x)``,
# ``pi``. A learner reads (and types) ``x^2``, ``2x``, ``√21``, ``e^(-x)``, ``π``. ``paper_notation`` is the one
# place that rewrites the first into the second. The stored question keeps Python's spelling, which tests and
# oracles parse; a consumer shows ``display_question(q)`` (text, answer, options, working in paper notation).
# The graders read both spellings (``mathgen/grading/typed.py`` normalises ``^``, ``×``, ``√``, ``π`` and implied
# multiplication), and ``tests/test_paper_notation.py`` pins that every rewritten form reads as the same value.

_PAPER_TRIGGER_RE = re.compile(r"\*|(?<![A-Za-z])(?:sqrt|exp)\(|(?<![A-Za-z])pi(?![A-Za-z])")
_PAPER_PI_RE = re.compile(r"(?<![A-Za-z])pi(?![A-Za-z])")
_PAPER_SQRT_RE = re.compile(r"(?<![A-Za-z])sqrt\(")
_PAPER_EXP_RE = re.compile(r"(?<![A-Za-z])exp\(")
# "4*I" is a complex number, written 4i; "lambda*I" is a matrix, so only a number may stand to its left.
_PAPER_IMAGINARY_RE = re.compile(r"(?<=\d)(\s*\*\s*)I(?![A-Za-z0-9_(])")
# A root or a power of e whose argument is one number or one letter needs no brackets ("√21", "e^x"), unless
# something follows that would attach to the argument instead ("√(2)^3", "e^(x)(y)"). A root followed by a
# letter or bracket takes a space instead ("√3 y"): "√3y" does not say which factor is under the root.
_PAPER_SIMPLE_ROOT_RE = re.compile(r"√\((\d+(?:\.\d+)?|[A-Za-z])\)(?![\^\d.])")
_PAPER_SIMPLE_EXP_RE = re.compile(r"(?<![A-Za-z])e\^\((\d+(?:\.\d+)?|[A-Za-z])\)(?![\^\w(.])")
_PAPER_STAR_RE = re.compile(r"[ \t]*\*[ \t]*")
_PAPER_TIMES = " × "


def _paper_matching_open(text: str, close: int) -> int:
    """Index of the ``(`` that closes at ``text[close]``, or -1."""
    depth = 0
    for i in range(close, -1, -1):
        if text[i] == ")":
            depth += 1
        elif text[i] == "(":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _paper_matching_close(text: str, open_at: int) -> int:
    """Index of the ``)`` that closes ``text[open_at]``, or -1."""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _paper_left_class(text: str) -> str:
    """What the operand ending ``text`` is, for choosing how a ``*`` after it is written."""
    if not text:
        return "OTHER"
    last = text[-1]
    if last.isdigit() or last == ".":
        start = len(text)
        while start > 0 and (text[start - 1].isdigit() or text[start - 1] == "."):
            start -= 1
        before = text[start - 1] if start else ""
        if before == "^" or (before == "-" and text[start - 2 : start - 1] == "^"):
            return "EXP"
        if before == "√":
            return "RADICAL"
        if before.isalpha() or before == "_":
            return "WORD"
        return "NUM"
    if last == ")":
        opener = _paper_matching_open(text, len(text) - 1)
        if opener < 0:
            return "OTHER"
        before = text[opener - 1] if opener else ""
        if before == "^":
            return "EXP"
        if before == "√":
            return "RADICAL"
        if before.isalpha():
            return "CLOSE_F"
        return "GROUP"
    if last.isalpha():
        start = len(text)
        while start > 0 and text[start - 1].isalpha():
            start -= 1
        run = text[start:]
        before = text[start - 1] if start else ""
        if before == "^" or (before == "-" and text[start - 2 : start - 1] == "^"):
            return "EXP"
        if run == "π":
            return "PI"
        if len(run) == 1 and before != "_":
            return "VAR1"
        return "WORD"
    return "OTHER"


def _paper_right_class(text: str) -> tuple[str, int]:
    """(class of the operand starting ``text`` after any unary minus, index just past that operand)."""
    i = 1 if text[:1] in ("-", "−") else 0
    if i >= len(text):
        return "OTHER", i
    c = text[i]
    if c.isdigit():
        j = i
        while j < len(text) and (text[j].isdigit() or text[j] == "."):
            j += 1
        return "NUM", j
    if c == "(":
        close = _paper_matching_close(text, i)
        return "GROUP", (close + 1 if close >= 0 else i + 1)
    if c == "√":
        if text[i + 1 : i + 2] == "(":
            close = _paper_matching_close(text, i + 1)
            return "RADICAL", (close + 1 if close >= 0 else i + 2)
        j = i + 1
        while j < len(text) and (text[j].isdigit() or text[j] == "."):
            j += 1
        return "RADICAL", max(j, i + 2)
    if c == "π":
        return "PI", i + 1
    if c.isalpha():
        j = i
        while j < len(text) and text[j].isalpha():
            j += 1
        run = text[i:j]
        if run == "e" and text[j : j + 1] == "^":
            return "EPOW", j
        if text[j : j + 1] == "(":
            close = _paper_matching_close(text, j)
            return "FUNC", (close + 1 if close >= 0 else j + 1)
        if run in ("ln", "log") and text[j : j + 1] == "|" and text.find("|", j + 1) > 0:
            return "FUNC", text.find("|", j + 1) + 1  # ln|x + 1| is a function call, like ln(x + 1)
        if len(run) == 1 and text[j : j + 1] != "_":
            return "VAR1", j
        return "WORD", j
    return "OTHER", i


def _paper_left_operand_follows_slash(text: str) -> bool:
    """True when the operand ending ``text`` is the denominator of a ``/`` (``1/2``, ``x/y``, ``a/(b + 1)``)."""
    if not text:
        return False
    if text[-1] == ")":
        start = _paper_matching_open(text, len(text) - 1)
    elif text[-1].isdigit() or text[-1] == ".":
        start = len(text)
        while start > 0 and (text[start - 1].isdigit() or text[start - 1] == "."):
            start -= 1
    else:
        start = len(text) - 1
    return start > 0 and text[start - 1] == "/"


def _paper_star_separator(left: str, right: str) -> str:
    """What replaces ``*`` between a ``left`` and a ``right`` operand: nothing (``2x``), a space (``x sin(x)``)
    or the multiplication sign (``6 × 8``), whichever reads unambiguously on paper."""
    if right in ("NUM", "WORD", "OTHER") or left in ("WORD", "OTHER"):
        return _PAPER_TIMES
    if left == "NUM":
        return "" if right in ("VAR1", "GROUP", "PI", "RADICAL") else " "
    if left in ("VAR1", "PI", "GROUP"):
        return "" if right in ("VAR1", "GROUP", "PI") else " "
    if left == "CLOSE_F":
        return _PAPER_TIMES if right == "GROUP" else " "
    if left == "EXP":
        return "" if right == "GROUP" else " "
    if left == "RADICAL":
        return " "
    return _PAPER_TIMES


def paper_notation(text):
    """``text`` with Python's spelling of maths written as it is on paper, for anything a learner reads.

    ``x**2`` -> ``x^2``; ``2*x`` -> ``2x``; ``6*8`` -> ``6 × 8``; ``(a + 1)*(a - 1)`` -> ``(a + 1)(a - 1)``;
    ``x*sin(x)`` -> ``x sin(x)``; ``sqrt(21)`` -> ``√21`` and ``sqrt(x + 1)`` -> ``√(x + 1)``;
    ``exp(-2*x)`` -> ``e^(-2x)``; ``5*pi/6`` -> ``5π/6``; ``4*I`` -> ``4i``.
    A multiplication between two numbers, or whose neighbour is a longer name (``lambda*t``), keeps a visible
    sign: ``lambda × t``. A negative factor is bracketed: ``3*-1`` -> ``3 × (-1)``.
    Text with none of these is returned unchanged, and the result is unchanged by a second pass.
    A star touching the operand before it with a space after (``x* is``, ``z* = a - bi``) is a superscript mark
    and stays as written.
    Not a parser: it reads only the operands on each side of a ``*``, so prose passes through untouched.
    The graders read the result as well as the input; ``tests/test_paper_notation.py`` pins that.
    """
    if not isinstance(text, str) or not _PAPER_TRIGGER_RE.search(text):
        return text
    s = text.replace("**", "^")
    s = _PAPER_PI_RE.sub("π", s)
    s = _PAPER_SQRT_RE.sub("√(", s)
    s = _PAPER_EXP_RE.sub("e^(", s)
    s = _PAPER_SIMPLE_ROOT_RE.sub(
        lambda m: (
            "√" + m.group(1) + (" " if s[m.end() : m.end() + 1].isalpha() or s[m.end() : m.end() + 1] == "(" else "")
        ),
        s,
    )
    s = _PAPER_SIMPLE_EXP_RE.sub(r"e^\1", s)
    s = _PAPER_IMAGINARY_RE.sub(r"\1i", s)
    out: list[str] = []
    pos = 0
    for m in _PAPER_STAR_RE.finditer(s):
        if m.start() < pos:
            continue  # inside a negative factor already rewritten
        if m.group()[0] == "*" and m.group()[-1] in " \t":
            continue  # ``x* is``: a star hugging the name before it is a superscript mark, not a product
        left = _paper_left_class(s[: m.start()])
        right, end = _paper_right_class(s[m.end() :])
        if left == "OTHER" or right == "OTHER":
            continue  # a star with no operand on one side is left as written
        negative = s[m.end() : m.end() + 1] in ("-", "−")
        separator = _paper_star_separator(left, right)
        if separator != _PAPER_TIMES and _paper_left_operand_follows_slash(s[: m.start()]):
            separator = _PAPER_TIMES  # ``1/2*x`` is (1/2) × x; ``1/2x`` would read as 1/(2x)
        out.append(s[pos : m.start()])
        out.append(separator)
        operand_end = m.end() + end
        if negative:
            out.append("(" + s[m.end() : operand_end] + ")")
            pos = operand_end
        else:
            pos = m.end()
    out.append(s[pos:])
    return "".join(out)


# "5/9 = 5/9" restated: the second copy is dropped. Conservative: the repeat must end the clause (not feed a longer
# expression "-10 = -10 x (1) + c", a power "10 = 10^1" or a chain "3 = 3 = 4").
_REPEATED_STEP_RE = re.compile(
    r"(?P<pre>(?:^|[=:] |\) |\bso |\bgives ))(?P<t>-?[\d,]*\d(?:\.\d+)?%?(?:[/:]\d+)?) = (?P=t)"
    r"(?![\w/:%^]|\.\d|,\d|\s*[=+\u00d7\u00f7*\u2212(^-]|\s[xX]\s)"
)
# A working line that is only a restated value ("764 = 764." or "(c) 764 = 764."): nothing is left once the repeat goes.
_RESTATED_LINE_RE = re.compile(r"^(?:\([a-z]\) )?(-?[\d,]*\d(?:\.\d+)?%?(?:[/:]\d+)?) = \1\.?$")


def _collapse_repeat(m):
    return m.group("pre") + m.group("t")


_UNIT_POWER_RE = re.compile(
    r"(?<![A-Za-z])(mm|cm|km)\^([23])\b|(?<=\d )(?<!\^\d )(?<!\^\d\d )(mm|cm|km|m)([23])\b|(?<=\d)(mm|cm|km)([23])\b|(?<=\d )(?<!\^\d )(?<!\^\d\d )(m)\^([23])\b|(?<=\()(m)([23])(?=\))|(?<![A-Za-z\d])(mm|cm|km|m)([23])(?= = \d)"
)


def _unit_power(m):
    groups = [g for g in m.groups() if g]
    return groups[0] + {"2": "\u00b2", "3": "\u00b3"}[groups[1]]


_AN_NUMBER_RE = re.compile(
    r"\b([Aa]) (?=(?:8\d*|11|18)(?:\.\d+)?[ -]?[A-Za-z%\u00b0$]"
    r"|\$(?:8[\d,]*|11(?:,?000)?|18(?:,?000)?)(?![\d,]\d)(?:\.\d+)?\b"
    r"|(?:octag\w+|age\b|equilateral|orange|ice\b|egg|elephant|apple|umbrella|isosceles|obtuse|acute|ellipse|ounce|"
    r"engine|exam\b|angle|arc\b|area|oval|item|ice-)\w*)"
)
# A capital "A" is often a label ("A 85c, B 90c"): it takes no "n" when the quantity ends the clause or a list item.
_LABEL_QUANTITY_RE = re.compile(r"\$?(?:8\d*|11|18)(?:\.\d+)?[ -]?[A-Za-z%\u00b0$]?\w*\s*(?:[,;:.]|$)")


_AN_TRIGGER_RE = re.compile(
    r"\b[Aa] (?:\$?(?:8|11|18)|(?:octag|age\b|equilateral|orange|ice|egg|elephant|apple|umbrella|isosceles|obtuse|acute|ellipse|ounce|engine|exam\b|angle|arc\b|area|oval|item))"
)


def _an_number(m):
    if m.group(1) == "A" and _LABEL_QUANTITY_RE.match(m.string, m.end()):
        return m.group(0)
    return m.group(1) + "n "


# "9/-1" and "-5/-3" read as a stray sign after the slash; the divisor takes brackets, "9/(-1)". Only a plain number
# or one letter is rewritten, and only when nothing attaches to it ("1/-3x" is left alone).
_NEGATIVE_DIVISOR_RE = re.compile(r"(?<=[\w)])/-(\d+(?:\.\d+)?|[A-Za-z])(?![\w(^.])")


# "20 x 12" is a product between two plain numbers: the letter x there is the multiplication sign.
_LETTER_X_TIMES_RE = re.compile(r"(?<=[\d)|]) x (?=\d)")
# A coefficient of 1 written out ("1x", "-1n", "3 - 1i", "1 cos(4x)", "1f(x)"): only a letter that is a variable.
_COEFFICIENT_ONE_RE = re.compile(
    r"(?<![\w.^/])1(?=[xyztnui](?![A-Za-z0-9_])|[ab](?=\s[+=\u2212-])|[fgh]\(|e\^|\s(?:sin|cos|tan|ln)\()\s?"
)
# A coefficient of 1 before a bracket ("-1(x - 3)", "1(15/2)²"): the bracket stands alone. A bracket that is only a
# label ("1(a)") or a lone letter is left alone.
_COEFFICIENT_ONE_BRACKET_RE = re.compile(
    r"(?<![\w.^/)\]\u00b2\u00b3])(?<!\d,)([-\u2212]?)1\((?=[^()]*(?:[\d+\u2212/^\u00b2]|\s-\s))"
)
# "+ 0x" and "- 0t" add nothing; "y = 0x" is "y = 0".
_ZERO_TERM_RE = re.compile(r"\s[+\u2212-]\s0[xyztnu](?![A-Za-z0-9_(])")
_ZERO_LEADING_RE = re.compile(r"(?<![\w.^/])0[xyztnu](?=\s[+\u2212-]\s)\s[+\u2212-]\s")
_ZERO_ONLY_RE = re.compile(r"(?<==\s)0[xyztnu]$")
# "1 days", "1 parts": a count of one takes the singular noun (not after an operator, where "n - 1 edges" is right).
_PLURALS = {
    "days": "day",
    "hours": "hour",
    "minutes": "minute",
    "parts": "part",
    "additions": "addition",
    "steps": "step",
    "points": "point",
    "palings": "paling",
    "multiples": "multiple",
    "tenths": "tenth",
    "hundredths": "hundredth",
    "weeks": "week",
    "years": "year",
    "months": "month",
    "terms": "term",
    "items": "item",
    "cents": "cent",
    "leaves": "leaf",
}
_ONE_PLURAL_RE = re.compile(r"(?<![\w.^/])(?<!- )(?<!\+ )(?<!\u2212 )(?<!\u00d7 )1 (" + "|".join(_PLURALS) + r")\b")
# A signed number straight after an operator: "x - -23" is "x - (-23)", "2 × -5" is "2 × (-5)".
_SIGNED_OPERAND_RE = re.compile(r"(?<=[\w)\]]) ([+\u00d7\u00f7\u2212-]) ([-\u2212]\d+(?:\.\d+)?)(?!\d|\.\d)")


# --- Round-2 tidy-ups: typography the generators leave in Python/ASCII spelling -------------------------------------
# Each rule below is a plain regex over the one string, applied by `_learner_text`; none of them reads a number's value,
# and `tests/test_bh2_disp.py` pins that the digits a rule keeps are the digits it was given.

# "20 x (8-3)", "7 symbols x 5", "L x 9.8^x", "y = k x x": the letter x between two operands is the times sign. A
# bracket that holds a letter ("2 x (x+1)") is left as written: there the x may be a variable.
_LETTER_X_WORD_TIMES_RE = re.compile(r"(?<=[A-Za-z]{2}) x (?=\d)|(?<=\b[A-Z]) x (?=\d)")
_LETTER_X_BRACKET_TIMES_RE = re.compile(r"(?<=[\d)]) x (?=\([^A-Za-z]*\))")
_LETTER_X_SIGNED_TIMES_RE = re.compile(r"(?<=\d) x (?=-\d)")
_LETTER_X_VARIABLE_TIMES_RE = re.compile(r"(?<=[\d)]) x (?=x(?![A-Za-z0-9_(]))|(?<=\b[a-z]) x (?=x(?![A-Za-z0-9_(]))")
# "1 x t" and "1 \u00d7 t": a coefficient of one before a lone variable.
_ONE_TIMES_VARIABLE_RE = re.compile(r"(?<![\w.^/])(?<!\d )1 [x\u00d7] (?=[xyztnuk](?![A-Za-z0-9_]))")
# "<=", ">=", "->" as the signs a learner reads.
_ASCII_RELATION_RE = re.compile(r"<=|>=|(?<![-<])->")
_RELATION_SIGNS = {"<=": "\u2264", ">=": "\u2265", "->": "\u2192"}
# "15\u00b7x^2" and "5\u00b7x": a number times a letter is written together, as everywhere else in the library.
_DIGIT_DOT_LETTER_RE = re.compile(
    r"(?<=\d)(?<![A-Za-z_]\d)(?<![A-Za-z_]\d\d)(?<!\^)(?<!\^\d)(?<!\^\d\d)\u00b7(?=[A-Za-z(])"
)
# "log_{6}" and "u_{n}" read "log_6" and "u_n": the braces are LaTeX.
_BRACED_SUBSCRIPT_RE = re.compile(r"(?<=[A-Za-z\)])_\{(\w{1,3})\}")
# "perimeter_mm = 5" and "surface_area = 3": a name with an underscore is a code identifier; a unit suffix goes in brackets.
_UNIT_SUFFIXES = {
    "mm": "mm", "cm": "cm", "m": "m", "km": "km", "mm2": "mm\u00b2", "cm2": "cm\u00b2", "m2": "m\u00b2", "km2": "km\u00b2",
    "mm3": "mm\u00b3", "cm3": "cm\u00b3", "m3": "m\u00b3", "ml": "mL", "l": "L", "g": "g", "kg": "kg",
}  # fmt: skip
_IDENTIFIER_RE = re.compile(r"(?<![\w.])([a-z]{3,})_([a-z]{1,3}\d?|[a-z]{3,})(?![\w])")
# "z-bar" and "c-bar": the conjugate written as its stem writes it.
_BAR_RE = re.compile(r"\b([A-Za-z])-bar\b")
# Greek letters spelled out ("alpha = 28.1", "30 deg").
_GREEK_NAMES = "alpha|beta|gamma|theta|phi|omega|eta|mu|sigma|delta|epsilon|lambda"
_GREEK_RE = re.compile(r"\b(" + _GREEK_NAMES + r")\b(?=\s*(?:=|\)|,|[+\u2212*\u00d7/-]|$))")
_GREEK = {
    "alpha": "\u03b1", "beta": "\u03b2", "gamma": "\u03b3", "theta": "\u03b8", "phi": "\u03c6", "omega": "\u03c9",
    "eta": "\u03b7", "mu": "\u03bc", "sigma": "\u03c3", "delta": "\u03b4", "epsilon": "\u03b5", "lambda": "\u03bb",
}  # fmt: skip
_GREEK_HINT_RE = re.compile(r"\b(?:" + _GREEK_NAMES + r")\b")
_DEGREE_WORD_RE = re.compile(r"(?<=\d) ?\bdeg\b")
# Two spaces between words are one.
_DOUBLE_SPACE_RE = re.compile(r"(?<=\S)  +(?=\S)")
# "$21426.12" and "$11000" take the thousands comma "$21,426.12", so one question does not mix "$18,000" with "$21426.12".
_MONEY_RE = re.compile(r"(?<![\w,.])\$(\d{4,})(?![\d,]\d)")


def _money_commas(m):
    return "$" + f"{int(m.group(1)):,}" if not m.group(1).startswith("0") else m.group(0)


# A bracket round a bare positive number is noise after an operator: "12 \u00d7 (7)", "5 - (6)", "+ (2)x", "\u221a(3)".
_BRACKETED_AFTER_ADD_RE = re.compile(r"(?<= [+\u2212-] )\((\d+(?:\.\d+)?)\)(?![\d.^(\u00b2\u00b3/]|[A-Za-z]{2})")
_BRACKETED_AFTER_TIMES_RE = re.compile(r"(?<= [\u00d7\u00f7] )\((\d+(?:\.\d+)?)\)(?![\w.^(\u00b2\u00b3/])")
_BRACKETED_BEFORE_TIMES_RE = re.compile(r"(?<![\w)\]])\((\d+(?:\.\d+)?)\)(?= [\u00d7\u00f7] )")
_BRACKETED_ROOT_RE = re.compile(r"([\u221a\u221b])\((\d+(?:\.\d+)?)\)(?![\w^.(\u00b2\u00b3])")
# "(x - 0)(x - 5)": a root of zero is "x".
_ZERO_ROOT_RE = re.compile(r"\(([A-Za-z]) [+\u2212-] 0\)")
# A "+ 0" or "- 0" closing a polynomial or recurrence: "f(x) = -x^3 + 15x^2 - x + 0".
_ZERO_CONSTANT_RE = re.compile(
    r"(?<=[A-Za-z)\u00b2\u00b3]) [+\u2212-] 0(?![\d.\w%(])|(?<=\^\d) [+\u2212-] 0(?![\d.\w%(])"
)
# In working: "x = 8/1 = 8" -> "x = 8 = 8" (then the repeat goes); a division by one is not a step.
_DIVIDE_BY_ONE_RE = re.compile(r"(?<=[A-Za-z] = )(\d+)/1(?![\w./^(\d%])")
# "4.0" in working is "4" unless the line is about precision.
_TRAILING_POINT_ZERO_RE = re.compile(r"(?<![\w.,])(\d+)\.0(?![\d\w.])")
_PRECISION_WORDS_RE = re.compile(
    r"d\.p\b|decimal place|significant|sig\.? fig|s\.f\b|nearest|round|precision|\btenth", re.I
)
# "y = mx + b = y = 2x + 12" and "alpha = atan2(b, a) = alpha = 28.1": the repeated "name =" goes.
_REPEATED_LHS_RE = re.compile(r"^(?P<v>[A-Za-z\u03b1-\u03c9][\w\u03b1-\u03c9]{0,10}) = (?P<mid>.+?) = (?P=v) = ")
# A fraction beside another in a product ("1/2 \u00d7 16/9") or the divisor of one ("63 \u00f7 1/567") is bracketed.
_FRACTION_TOKEN_RE = re.compile(r"(?<![\w/.^(])((?:\(-?\d+\)|\d+)/\d+)(?![\w/.^(])")
# "5.5/4%" reads as 5.5 divided by 4%; "(5.5/4)%" says what is meant.
_FRACTION_PERCENT_RE = re.compile(r"(?<![\w/.^(])(\d+(?:\.\d+)?/\d+)%")
# U+2212 beside an ASCII minus in one line: the line takes the ASCII one.
_ASCII_MINUS_USE_RE = re.compile(r"(?<=[\w)\]]) - |(?<![\w)\]])-\d|\(-")


_FRACTION_BEFORE_RE = re.compile(r"(?<![\w/.^])(?:\(-?\d+\)|\d+)/\d+\)? [\u00d7\u00f7] $")
_FRACTION_AFTER_RE = re.compile(r" [\u00d7\u00f7] \(?(?:\(-?\d+\)|\d+)/\d+(?![\w/.^(])")


# "-(-4 - 2)" after a plus is a minus: "+ -(" -> "- (", "- -(" -> "+ (".
_SIGN_BEFORE_BRACKET_RE = re.compile(r"(?<=[\w)\]]) ([+\u2212-]) -(?=\()")
# Money in working is shown to the cent: "$1740.0" -> "$1740.00" (a figure of "$1.5 million" stays).
_MONEY_ONE_DECIMAL_RE = re.compile(
    r"(?<![\w,.])\$(\d[\d,]*\.\d)(?!\d|\.\d|\s*(?:million|billion|thousand|trillion|[kKMB]\b))"
)
# "n = 8 \u00f7 1 = 8", "x = 4 / 1 = 4": dividing by one is not a step.
_DIVIDE_BY_ONE_SPACED_RE = re.compile(
    r"(?<=[A-Za-z] = \d) [\u00f7/] 1(?= = )|(?<=[A-Za-z] = \d\d) [\u00f7/] 1(?= = )|(?<=[A-Za-z] = \d\d\d) [\u00f7/] 1(?= = )"
)
# "-1d m/s", "-1a^9": a unit coefficient before a variable. "(1)x" is handled by the bracket rules first.
_NEGATIVE_ONE_COEFFICIENT_RE = re.compile(
    r"(?<![\w.^/)])(-|\u2212)1(?=[a-fi-kn-rt-zA-Z](?![A-Za-z0-9_(]|\^0)|[a-fi-lnp-zA-Z]\^)"
)
# "0m + 3k", "0i + 23j": a zero multiple of a variable heading a sum with another variable term.
_ZERO_MULTIPLE_HEAD_RE = re.compile(r"(?<![\w.^/])0[ijkm](?=\s[+\u2212-]\s\d*[a-zA-Z](?![A-Za-z]))\s[+\u2212-]\s")
_ZERO_PLUS_HEAD_RE = re.compile(r"(?<== )0 \+ (?=\(?\d+\)?[a-z])")
# Brackets round a bare positive in more places: "(16) + (21)", "/ (120) + 1", "4^(1)", "\u221b(27)".
_BRACKETED_HEAD_RE = re.compile(r"(?<![\w)\]/^.])\((\d+(?:\.\d+)?)\)(?= [+\u2212-] \(?\d)")
_BRACKETED_DIVISOR_RE = re.compile(r"(?<= / )\((\d+(?:\.\d+)?)\)(?= [+\u2212\u00d7\u00f7-]|\)|$)")
_BRACKETED_EXPONENT_RE = re.compile(r"(?<=[\d)]\^)\((\d+)\)(?![\d.\w])")
# "2/3x" and "5/2x" read as 2/(3x): the fraction is bracketed when a variable or a bracket follows it directly.
_FRACTION_COEFFICIENT_RE = re.compile(
    r"(?<![\w/.^(])(\d+/\d+)(?=[xyztnuvwabk](?![A-Za-z0-9_])|[xyztnuvwab]\^|\(|(?:sin|cos|tan|ln|log)\()"
)
# "AM x BM", "1.5 x IQR", "28 x cos(210)": a number or bracket, x, then a function or quantity name.
_LETTER_X_NAME_TIMES_RE = re.compile(
    r"(?<=[\d)]) x (?=(?:sin|cos|tan|cot|sec|csc|ln|log|exp|IQR|SD|side|length|width|height|base|radius|rate|price|cost|speed)\b)"
    r"|(?<=\b[A-Z]{2}) x (?=[A-Z]{2}\b)|(?<=\b[a-z]) x (?=\(-?\d+\))"
)
# "1/2 \u00d7 base x height": the chain carries on.
_LETTER_X_CHAIN_RE = re.compile(r"( \u00d7 [a-z]{3,}) x (?=[a-z]{3})")
# "2\u00d73" run together.
_TIMES_NO_SPACE_RE = re.compile(r"(?<=\d)\u00d7(?=\d)")


def _bracket_fractions(text):
    """Bracket a numeric fraction that sits beside another in a product or quotient, or is a divisor."""
    pieces = []
    pos = 0
    for m in _FRACTION_TOKEN_RE.finditer(text):
        before, after = text[: m.start()], text[m.end() :]
        if before.endswith("(") and after.startswith(")"):
            continue  # already bracketed
        if before.endswith(" \u00f7 ") or _FRACTION_BEFORE_RE.search(before) or _FRACTION_AFTER_RE.match(after):
            pieces.append(text[pos : m.start()])
            pieces.append("(" + m.group(1) + ")")
            pos = m.end()
    pieces.append(text[pos:])
    return "".join(pieces)


def _drop_point_zero(m, keep):
    return m.group(0) if m.group(0) in keep else m.group(1)


def _learner_text(text, *, working=False, keep=()):
    """`paper_notation(text)` made fit for a learner: a negative divisor bracketed ("9/(-1)"), a letter x between
    two operands written \u00d7, a spaced double hyphen written as an en dash, "+-" as \u00b1, "<=" ">=" "->" as signs,
    a coefficient of 1 or a zero term not written out ("1x" -> "x", "+ 0x" dropped, "f(x) + 0" -> "f(x)"), "1 days" ->
    "1 day", a signed number after an operator bracketed ("x - -23" -> "x - (-23)"), a bracket round a bare positive
    number dropped ("12 \u00d7 (7)"), fractions beside fractions bracketed, unit powers as \u00b2, \u00b3, "a 8" as "an 8",
    thousands commas on dollar amounts, one minus glyph per line, and, in working, "x = x" collapsed to "x", "4.0" as "4"
    (unless `keep` has "4.0", the question's own spelling, or the line is about precision). Idempotent; the one
    display pass for every learner-visible string."""
    text = paper_notation(text)
    if not isinstance(text, str):
        return text
    if "<=" in text or ">=" in text or "->" in text:
        text = _ASCII_RELATION_RE.sub(lambda m: _RELATION_SIGNS[m.group()], text)
    if "_{" in text:
        text = _BRACED_SUBSCRIPT_RE.sub(r"_\1", text)
    if "_" in text:
        text = _IDENTIFIER_RE.sub(_identifier_text, text)
    if "-bar" in text:
        text = _BAR_RE.sub("\\1\u0304", text)
    if _GREEK_HINT_RE.search(text):
        spelled = {m.group(1) for m in _GREEK_RE.finditer(text)}
        for name in spelled:
            text = re.sub(r"\b" + name + r"\b", _GREEK[name], text)
    if "deg" in text:
        text = _DEGREE_WORD_RE.sub("\u00b0", text)
    if "  " in text:
        text = _DOUBLE_SPACE_RE.sub(" ", text)
    if "$" in text:
        text = _MONEY_RE.sub(_money_commas, text)
    if "/-" in text:
        text = _NEGATIVE_DIVISOR_RE.sub(r"/(-\1)", text)
    if " x " in text:
        text = _LETTER_X_TIMES_RE.sub(" \u00d7 ", text)
        text = _LETTER_X_WORD_TIMES_RE.sub(" \u00d7 ", text)
        text = _LETTER_X_BRACKET_TIMES_RE.sub(" \u00d7 ", text)
        text = _LETTER_X_SIGNED_TIMES_RE.sub(" \u00d7 ", text)
        text = _LETTER_X_VARIABLE_TIMES_RE.sub(" \u00d7 ", text)
        text = _LETTER_X_NAME_TIMES_RE.sub(" \u00d7 ", text)
        text = _LETTER_X_CHAIN_RE.sub("\\1 \u00d7 ", text)
    if "\u00b7" in text:
        text = _DIGIT_DOT_LETTER_RE.sub("", text)
    if " -- " in text:
        text = text.replace(" -- ", " \u2013 ")
    if "+/-" in text:
        text = text.replace("+/-", "\u00b1")
    if "+-" in text:
        text = re.sub(r"(?<![\w)])\+-(?=\d)", "\u00b1", text)
    if "m" in text:
        text = _UNIT_POWER_RE.sub(_unit_power, text)
    if _AN_TRIGGER_RE.search(text):
        text = _AN_NUMBER_RE.sub(_an_number, text)
    if "(" in text:
        text = _BRACKETED_AFTER_ADD_RE.sub(r"\1", text)
        text = _BRACKETED_AFTER_TIMES_RE.sub(r"\1", text)
        text = _BRACKETED_BEFORE_TIMES_RE.sub(r"\1", text)
        text = _BRACKETED_HEAD_RE.sub(r"\1", text)
        text = _BRACKETED_DIVISOR_RE.sub(r"\1", text)
        if "^(" in text:
            text = _BRACKETED_EXPONENT_RE.sub(r"\1", text)
        if "\u221a(" in text or "\u221b(" in text:
            text = _BRACKETED_ROOT_RE.sub(lambda m: m.group(1) + m.group(2), text)
    if "1" in text:
        text = _ONE_TIMES_VARIABLE_RE.sub("", text)
        text = _COEFFICIENT_ONE_RE.sub("", text)
        text = _NEGATIVE_ONE_COEFFICIENT_RE.sub(r"\1", text)
        if "1(" in text:
            text = _COEFFICIENT_ONE_BRACKET_RE.sub(r"\1(", text)
        text = _ONE_PLURAL_RE.sub(lambda m: "1 " + _PLURALS[m.group(1)], text)
    if "0" in text:
        text = _ZERO_LEADING_RE.sub("", text)
        text = _ZERO_MULTIPLE_HEAD_RE.sub("", text)
        text = _ZERO_PLUS_HEAD_RE.sub("", text)
        text = _ZERO_TERM_RE.sub("", text)
        text = _ZERO_ONLY_RE.sub("0", text)
        text = _ZERO_ROOT_RE.sub(r"\1", text)
        text = _ZERO_CONSTANT_RE.sub("", text)
    if " -" in text or " \u2212" in text:
        text = _SIGNED_OPERAND_RE.sub(lambda m: f" {m.group(1)} ({m.group(2)})", text)
    if "-(" in text:
        text = _SIGN_BEFORE_BRACKET_RE.sub(lambda m: " + " if m.group(1) in "-\u2212" else " - ", text)
    if "\u00d7" in text:
        text = _TIMES_NO_SPACE_RE.sub(" \u00d7 ", text)
    if "/" in text:
        text = _FRACTION_COEFFICIENT_RE.sub(r"(\1)", text)
        if "\u00d7" in text or "\u00f7" in text:
            text = _bracket_fractions(text)
        if "%" in text:
            text = _FRACTION_PERCENT_RE.sub(r"(\1)%", text)
    if "\u2212" in text and _ASCII_MINUS_USE_RE.search(text):
        text = text.replace("\u2212", "-")
    if working:
        text = text.lstrip(" ") if text.strip() else text
        if "$" in text:
            text = _MONEY_ONE_DECIMAL_RE.sub(lambda m: "$" + _money_one_decimal(m.group(1)), text)
        if " \u00f7 1" in text or " / 1" in text:
            text = _DIVIDE_BY_ONE_SPACED_RE.sub("", text)
        if "/1" in text:
            text = _DIVIDE_BY_ONE_RE.sub(r"\1", text)
        if ".0" in text and not _PRECISION_WORDS_RE.search(text):
            text = _TRAILING_POINT_ZERO_RE.sub(lambda m: _drop_point_zero(m, keep), text)
        if " = " in text:
            text = _REPEATED_LHS_RE.sub(
                lambda m: m.group(0)[: m.start("v") - m.start()] + m.group("v") + " = " + m.group("mid") + " = ", text
            )
            text = _REPEATED_STEP_RE.sub(_collapse_repeat, text)
    return text


def _money_one_decimal(figure):
    whole, _, tenth = figure.partition(".")
    whole = f"{int(whole.replace(',', '')):,}"
    return f"{whole}.{tenth}0"


def _identifier_text(m):
    name, suffix = m.group(1), m.group(2)
    if suffix in _UNIT_SUFFIXES:
        return f"{name} ({_UNIT_SUFFIXES[suffix]})"
    if len(suffix) >= 3 and suffix.isalpha():
        return f"{name} {suffix}"
    return m.group(0)


_display_text = _learner_text


_POINT_ZERO_RE = re.compile(r"(?<![\w.,])\d+\.0(?![\d\w.])")


def _precision_spellings(question):
    """The "4.0"-style numbers the question itself writes: working keeps them, since the zero is a stated precision."""
    found = set()
    for field in (question.text, question.answer_text, *question.distractors):
        if isinstance(field, str):
            found.update(_POINT_ZERO_RE.findall(field))
    return found


def _working_lines(working, keep):
    """The working a learner reads. A line that only restates a value ("764 = 764.") is dropped, unless it is all there is."""
    lines = list(working or [])
    shown = [_learner_text(w, working=True, keep=keep) for w in lines]
    kept = [t for w, t in zip(lines, shown) if not (isinstance(w, str) and _RESTATED_LINE_RE.match(w))]
    return kept or shown


def display_question(question) -> dict:
    """The strings a learner reads for `question` (or a `Part`), written as on paper (`paper_notation`).

    The question itself is not changed: its stored `text`, `answer_text`, `distractors` and `working` keep the
    generator's spelling, which the graders and the tests read. A front end or command line that shows a question
    shows this instead. Keys: `text`, `answer_text`, `options`, `distractors` and `working`. `options` is
    `mcq_options(question)` in the same order, so an option picked by index grades the same; `distractors` is the
    wrong options in their stored order. A multi-part question also has `parts`, the same dictionary for each part
    with its `label` added.
    """
    keep = _precision_spellings(question)
    shown = {
        "text": _learner_text(question.text),
        "answer_text": _learner_text(question.answer_text),
        "options": [_learner_text(o) for o in mcq_options(question)],
        "distractors": [_learner_text(d) for d in question.distractors],
        "working": _working_lines(question.working, keep),
    }
    parts = getattr(question, "parts", None)
    if parts:
        shown["parts"] = [{"label": p.label, **display_question(p)} for p in parts]
    return shown


def mcq_options(question) -> list[str]:
    """The shuffled option list for an mcq question: [answer_text] + distractors,
    shuffled deterministically by the question's own seed. Shared with
    whatever presents the question, so client and grader always agree on option order."""
    options = [question.answer_text] + list(question.distractors)
    random.Random(question.seed).shuffle(options)
    return options
