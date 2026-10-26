"""Questions, papers and grading as plain JSON-ready data: the layer a web service, an export or a school's own
code builds on.

    from mathgen import service

    lib = service.Library.load()
    hits = service.find_skills(lib, year="8", text="surd")
    q = service.make_questions(lib, "simplify-surds", count=3, seed=7)     # what a learner sees
    key = service.make_questions(lib, "simplify-surds", count=3, seed=7, answers=True)
    paper = service.build_paper(lib, {"title": "Surds quiz", "seed": 42, "versions": 2,
                                      "sections": [{"items": [{"skill": "simplify-surds", "count": 5}]}]})
    result = service.grade_reply(lib, q[0]["id"], "6√21")

Everything returned is made of dicts, lists, strings, numbers, booleans and None, so `json.dumps` takes it as is.

Nothing is stored. A question is named by its id (`engine.question_id`: skill, source key, seed and a hash of the
content), and grading rebuilds the question from that id, so a reply is always marked against the question that
was shown, never against one a caller sends back. When the library has changed what an id generates, grading
refuses with `QuestionChanged` instead of marking against a different question.

Answers are left out unless asked for (`answers=True`). That matters for a service a learner can reach: give
learners a key that cannot ask for answers (see `mathgen.server`).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
import re
import secrets
from dataclasses import asdict, dataclass, field
from fractions import Fraction

from . import __version__, engine, misconceptions
from .core import _learner_text, display_question, mcq_options, paper_notation
from .engine import QuestionChanged  # noqa: F401  (re-exported for callers of grade_reply)
from .grading.typed import table_prompt
from .skills import Skill

#: The most questions one call may ask for. A paper may hold more, across items (`MAX_PAPER_QUESTIONS`).
MAX_COUNT = 100
MAX_PAPER_QUESTIONS = 400
MAX_VERSIONS = 10
#: The shortest `id_secret` a library accepts.
MIN_ID_SECRET_CHARS = 16
#: How many extra seeds `make_questions` tries to find questions that differ, before it accepts repeats.
_DISTINCT_TRIES = 4
#: One past the largest seed (or nonce) an id can carry; `_regenerate` refuses anything from here up.
MAX_NONCE = 2**63

_V9_CODE = re.compile(r"^AC9M(F|10A|10|[1-9])[A-Z]{1,3}\d{2}$")
_SENIOR_CODE = re.compile(r"^ACM[A-Z]{2}\d{3}$")
_YEAR_TAG = re.compile(r"^acara-v9-year-(f|\d{1,2}a?)$")
_FRAMEWORK_TAGS = {
    "acara-f10-v9": "acara-f10-v9",
    "acara-senior-v8-4": "acara-senior-v8-4",
}

ACARA_ATTRIBUTION = (
    "Derived from the Australian Curriculum (Senior Secondary Version 8.4), (c) Australian Curriculum, "
    "Assessment and Reporting Authority (ACARA) 2010 to present, licensed under CC BY 4.0 "
    "(https://creativecommons.org/licenses/by/4.0/) and modified. ACARA does not endorse this product."
)
SKILLS_ATTRIBUTION = (
    "Skill definitions from maths-engine (https://github.com/Jondi-Studio-Public/maths-engine), "
    "(c) 2026 Jondi Haynes, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)."
)


class NotFound(KeyError):
    """An unknown skill, source or question id."""

    def __str__(self):  # KeyError's own quotes its message
        return str(self.args[0]) if self.args else ""


class BadRequest(ValueError):
    """A call the engine cannot carry out as asked: too many questions, a skill with no questions, and so on."""


@dataclass
class Library:
    """The loaded skill library, indexed for lookups. Load it once and share it: loading takes seconds."""

    skills: list[Skill]
    by_id: dict[str, Skill] = field(default_factory=dict)
    fingerprint: str = ""
    #: Optional server secret. When set, an id's seed field is a public nonce and the generator's real seed is
    #: `HMAC(secret, skill/source/nonce)`, so the public engine cannot rebuild a question from its id alone.
    id_secret: str | None = field(default=None, repr=False)

    @classmethod
    def load(cls, root=None, extra_roots=None, *, id_secret: str | None = None) -> Library:
        if id_secret is not None and len(id_secret) < MIN_ID_SECRET_CHARS:
            raise ValueError(f"the id secret must be at least {MIN_ID_SECRET_CHARS} characters")
        _domains, skills = engine.load_skills(root, extra_roots)
        digest = hashlib.sha256()
        for skill in sorted(skills, key=lambda s: s.id):
            digest.update(skill.id.encode())
            if skill.path is not None and skill.path.is_file():
                digest.update(skill.path.read_bytes())
        return cls(
            skills=skills, by_id={s.id: s for s in skills}, fingerprint=digest.hexdigest()[:12], id_secret=id_secret
        )

    def skill(self, skill_id: str) -> Skill:
        try:
            return self.by_id[skill_id]
        except KeyError:
            raise NotFound(f"unknown skill {skill_id!r}") from None

    def version(self) -> dict:
        """What a caller needs to know which questions an id will give: the engine and the library it loaded."""
        return {"engine": __version__, "library": self.fingerprint, "skills": len(self.skills)}


# --- skills ----------------------------------------------------------------------------------------------------


def curriculum_codes(skill: Skill) -> list[str]:
    """The ACARA content description codes a skill is tagged with, upper-cased (`AC9M8N01`, `ACMMM092`)."""
    codes = []
    for tag in skill.tags:
        up = str(tag).upper()
        if (_V9_CODE.match(up) or _SENIOR_CODE.match(up)) and up not in codes:
            codes.append(up)
    return codes


def year_levels(skill: Skill) -> list[str]:
    """The school years a skill belongs to ("F", "1" ... "10", "10A", or "university"), from its own `year_level` when the
    library has one, else from its ACARA v9 codes and year tags. Empty when nothing says (some senior skills)."""
    own = getattr(skill, "year_level", None)
    if own:
        return [str(y) for y in (own if isinstance(own, (list, tuple)) else [own])]
    years = []
    for code in curriculum_codes(skill):
        m = _V9_CODE.match(code)
        if m and m.group(1) not in years:
            years.append(m.group(1))
    for tag in skill.tags:
        m = _YEAR_TAG.match(str(tag).lower())
        if m and m.group(1).upper() not in years:
            years.append(m.group(1).upper())
    return years


def frameworks(skill: Skill) -> list[str]:
    found = [name for tag, name in _FRAMEWORK_TAGS.items() if tag in skill.tags]
    # `acara-v8-4`, `acara-v8-4-methods` and the like tag a skill drawn from the senior curriculum too.
    if "acara-senior-v8-4" not in found and any(str(tag).startswith("acara-v8-4") for tag in skill.tags):
        found.append("acara-senior-v8-4")
    for alignment in skill.alignments:
        name = getattr(alignment, "framework", None)
        if name and name not in found:
            found.append(name)
    return found


def needs_acara_attribution(skill: Skill) -> bool:
    """Whether a skill's name or description is adapted from ACARA's senior curriculum (CC BY 4.0), which
    must then be attributed wherever it is republished."""
    return (
        "acara-senior-v8-4" in frameworks(skill)
        or skill.id.startswith("senior-")
        or skill.domain == "senior-mathematics"
        or any(_SENIOR_CODE.match(code) for code in curriculum_codes(skill))
    )


def source_keys(skill: Skill) -> list[str]:
    """The names a caller picks a skill's sources by: each source's key, else its difficulty label."""
    keys = []
    for src in skill.sources:
        key = src.effective_key
        if key and key not in keys:
            keys.append(key)
    return keys


def gradeable(skill: Skill) -> bool:
    """Whether the engine marks this skill's answers itself. Rubric skills (proofs, explanations) need a person."""
    return skill.answer != "rubric"


def skill_summary(skill: Skill, *, full: bool = False) -> dict:
    out = {
        "id": skill.id,
        "name": skill.name,
        "domain": skill.domain,
        "status": skill.status,
        "answer_kind": skill.answer,
        "gradeable": gradeable(skill),
        "sources": source_keys(skill),
        "difficulties": sorted({s.difficulty for s in skill.sources if s.difficulty}),
        "years": year_levels(skill),
        "curriculum_codes": curriculum_codes(skill),
        "frameworks": frameworks(skill),
        "learner_band": skill.learner_band,
        "needs_figure": skill.needs_figure,
    }
    if needs_acara_attribution(skill):
        out["attribution"] = ACARA_ATTRIBUTION
    if full:
        out["description"] = skill.description
        out["prerequisites"] = list(skill.prerequisites)
        out["tags"] = list(skill.tags)
        out["misconceptions"] = [{"id": m.id, "description": m.description} for m in skill.misconceptions]
    return out


def find_skills(
    lib: Library,
    *,
    text: str | None = None,
    domain: str | None = None,
    year: str | None = None,
    code: str | None = None,
    framework: str | None = None,
    answer_kind: str | None = None,
    difficulty: str | None = None,
    tag: str | None = None,
    gradeable_only: bool = False,
    include_deferred: bool = False,
) -> list[Skill]:
    """Skills matching every filter given, in library order. `text` matches the id, name or description, ignoring
    case; `code` is an ACARA code or a prefix of one (`AC9M8` for every Year 8 code); `year` is "F", "1" ... "10", "10A" or "university", in any case."""
    needle = text.casefold() if text else None
    code_prefix = code.upper() if code else None
    hits = []
    for skill in lib.skills:
        if not include_deferred and (skill.status != "active" or not skill.sources):
            continue
        if domain and skill.domain != domain:
            continue
        if year and str(year).strip().casefold() not in {y.casefold() for y in year_levels(skill)}:
            continue
        if code_prefix and not any(c.startswith(code_prefix) for c in curriculum_codes(skill)):
            continue
        if framework and framework not in frameworks(skill):
            continue
        if answer_kind and skill.answer != answer_kind:
            continue
        if difficulty and not any(s.difficulty == difficulty for s in skill.sources):
            continue
        if tag and tag not in skill.tags:
            continue
        if gradeable_only and not gradeable(skill):
            continue
        if needle and not any(needle in s.casefold() for s in (skill.id, skill.name, skill.description or "")):
            continue
        hits.append(skill)
    return hits


# --- questions -------------------------------------------------------------------------------------------------


def _latex(value: str | None) -> str | None:
    """LaTeX to typeset, or None when the generator gave plain text only (its `plain:` marker)."""
    if not value or value.startswith("plain:"):
        return None
    return value


def _plain(value):
    """`value` as JSON-ready data: SymPy numbers and tuples become strings and lists."""
    return json.loads(json.dumps(value, default=str))


def _prompt_block(item, shown: dict, kind: str) -> dict:
    """What a learner sees of a question or a part: wording, LaTeX, options, a grid to fill."""
    out = {"text": shown["text"], "latex": _latex(getattr(item, "prompt_latex", None) or getattr(item, "latex", ""))}
    if kind == "mcq":
        out["options"] = shown["options"]
    if kind == "table" and isinstance(getattr(item, "answer", None), dict) and "blanks" in item.answer:
        out["table"] = _plain(table_prompt(item.answer))
    return out


def _answer_block(item, shown: dict, kind: str, skill: Skill | None = None) -> dict:
    out = {
        "text": shown["answer_text"],
        "latex": _latex(getattr(item, "answer_latex", "")),
        "working": shown["working"],
    }
    if kind == "mcq":
        out["option"] = mcq_options(item).index(item.answer_text)
        notes = getattr(item, "option_notes", None) or {}
        if notes:
            out["option_notes"] = {str(i): notes[o] for i, o in enumerate(mcq_options(item)) if o in notes}
            # the same notes as words a learner can read: the skill's own wording first, then the catalogue's
            said = {i: misconceptions.describe(m, skill) for i, m in out["option_notes"].items()}
            if any(said.values()):
                out["option_feedback"] = {i: d[0] for i, d in said.items() if d}
    return out


def _figure_view(spec: dict, answers: bool) -> dict:
    """A figure for a learner is the picture (SVG) and its alt text, never the spec: for a read-the-picture
    question the spec holds the answer (a clock's hour and minute, a protractor's angle, a scale's level). With
    `answers` the spec is given too, for a caller that draws figures itself."""
    from .svg import figure_svg

    try:
        svg = figure_svg(spec)
    except ValueError:  # a kind the SVG renderer does not draw yet: the alt text still describes it
        svg = None
    view = {"alt": spec.get("alt"), "svg": svg}
    if answers:
        view["spec"] = _plain(spec)
    return view


def question_view(skill: Skill, question, qid: str, *, answers: bool = False) -> dict:
    """One question as data. Without `answers` it holds only what a learner may see."""
    shown = display_question(question)
    kind = skill.answer
    view = {
        "id": qid,
        "skill": skill.id,
        "difficulty": question.difficulty,
        "answer_kind": kind,
        "gradeable": gradeable(skill),
        "figure": _figure_view(question.figure, answers) if question.figure else None,
    }
    if question.parts:
        view["stem"] = {"text": paper_notation(question.stem or ""), "latex": None}
        view["parts"] = []
        for part, part_shown in zip(question.parts, shown["parts"]):
            entry = {"label": part.label, "answer_kind": part.answer_kind, "marks": part.marks}
            entry.update(_prompt_block(part, part_shown, part.answer_kind))
            if answers:
                entry["answer"] = _answer_block(part, part_shown, part.answer_kind, skill)
            view["parts"].append(entry)
        view["marks"] = sum(p.marks for p in question.parts)
    else:
        view.update(_prompt_block(question, shown, kind))
        view["marks"] = 1
        if answers:
            view["answer"] = _answer_block(question, shown, kind, skill)
    return view


def _random_seed() -> int:
    return secrets.randbelow(2**31)


def derive_seed(*parts) -> int:
    """A seed from any labels, the same on every machine: `derive_seed(42, "A", 0, 3)`."""
    text = ":".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") >> 1


def _real_seed(lib: Library, skill: Skill, src, nonce: int) -> int:
    """The seed the generator runs with for an id's seed field. Without an `id_secret` it is the field itself;
    with one it is a keyed hash of (skill, source, nonce), 63 bits, which only the server can compute."""
    if lib.id_secret is None:
        return nonce
    message = f"{skill.id}/{engine._key_of(skill, src)}/{nonce}".encode()
    digest = hmac.new(lib.id_secret.encode(), message, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def _make_with_id(lib: Library, skill: Skill, nonce: int, src):
    """`(question, id)` for a nonce. The id carries the nonce, not the generator's real seed."""
    if lib.id_secret is None:
        return engine.make_with_id(skill, nonce, source=src)
    question = engine.make_for_skill(skill, _real_seed(lib, skill, src, nonce), source=src)
    return question, engine.question_id(skill, src, nonce, question)


def _check_count(count: int, limit: int = MAX_COUNT, name: str = "count") -> None:
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= limit:
        raise BadRequest(f"{name} must be a whole number from 1 to {limit}, got {count!r}")


def _pick(skill: Skill, source):
    if isinstance(source, bool) or (isinstance(source, int) and source < 0):
        raise BadRequest(f"a source is a key, a difficulty or a 0-based index, got {source!r}")
    try:
        return engine.pick_source(skill, source)
    except IndexError:
        raise NotFound(f"skill {skill.id!r} has no source number {source!r}") from None
    except KeyError as exc:
        raise NotFound(exc.args[0] if exc.args else str(exc)) from None
    except ValueError as exc:
        raise BadRequest(str(exc)) from None


def make_questions(
    lib: Library,
    skill_id: str,
    *,
    source: str | int | None = None,
    count: int = 1,
    seed: int | None = None,
    answers: bool = False,
) -> list[dict]:
    """`count` questions for one skill. With a `seed` the same call gives the same questions anywhere; without one
    a random seed is used (each question's id still names its own seed). Seeds `seed`, `seed + 1`, ... are tried in
    turn and a question whose wording repeats an earlier one is skipped while fresh ones keep coming, so a small
    pool gives repeats only once it runs dry."""
    _check_count(count)
    skill = lib.skill(skill_id)
    src = _pick(skill, source)
    start = _random_seed() if seed is None else int(seed)
    out, seen = [], set()
    n = start
    budget = count * (1 + _DISTINCT_TRIES)
    if start + budget > MAX_NONCE:
        # Seeds run up to `budget` past `start`, and an id's seed field must stay below 2**63 to be graded.
        raise BadRequest(f"seed {start} is too close to the top of the range for {count} question(s)")
    spare: list[dict] = []
    while len(out) < count and n < start + budget:
        question, qid = _make_with_id(lib, skill, n, src)
        view = question_view(skill, question, qid, answers=answers)
        key = question.text
        (spare if key in seen else out).append(view)
        seen.add(key)
        n += 1
    return out + spare[: count - len(out)]


def question_by_id(lib: Library, qid: str, *, answers: bool = False) -> dict:
    """The question an id names, rebuilt. Raises `QuestionChanged` if the library now generates something else."""
    question, skill = _regenerate(lib, qid)
    return question_view(skill, question, qid, answers=answers)


def _regenerate(lib: Library, qid: str):
    try:
        skill_id, key, nonce, _digest = engine.parse_question_id(qid)
    except ValueError as exc:
        raise BadRequest(str(exc)) from None
    skill = lib.skill(skill_id)
    if lib.id_secret is None:
        try:
            question = engine.regenerate(qid, lib.by_id)
        except KeyError as exc:
            raise NotFound(exc.args[0] if exc.args else str(exc)) from None
        return question, skill
    if not 0 <= nonce < MAX_NONCE:
        raise BadRequest(f"not a question id: seed out of range in {qid!r}")
    matches = [src for src in skill.sources if src.effective_key == key]
    if not matches:
        raise NotFound(f"skill {skill_id!r} has no source with key {key!r}")
    question, new_id = _make_with_id(lib, skill, nonce, matches[0])
    if new_id != qid:
        raise QuestionChanged(qid, new_id, question)
    return question, skill


# --- papers ----------------------------------------------------------------------------------------------------


def _title(value) -> str | None:
    """A caller's title as safe text: at most 200 characters, and nothing that cannot be written as UTF-8 (a lone
    surrogate from JSON's \\ud800 would break every export)."""
    if value is None:
        return None
    text = str(value)[:200]
    return text.encode("utf-8", "replace").decode("utf-8")


def _version_label(i: int) -> str:
    return chr(ord("A") + i)


def build_paper(lib: Library, spec: dict, *, answers: bool = True) -> dict:
    """A worksheet or test, in one or more versions, with its answer key.

    `spec`::

        {"title": "Year 8 surds check",          # optional
         "seed": 42,                             # optional; the same spec and seed give the same paper
         "versions": 2,                          # optional, 1 to 10: versions A, B, ... with different numbers
         "shuffle": false,                       # optional: shuffle question order within each section
         "sections": [{"title": "Part 1",        # optional
                       "items": [{"skill": "simplify-surds", "source": "easy", "count": 4}]}]}

    Every question's seed is derived from the paper seed, the version and the question's place, so version A of
    paper 42 is the same today and next year (until the library changes, which the ids make visible). Without
    `answers` the paper holds no answers; build the key with `answers=True` (the default) and print it apart."""
    if not isinstance(spec, dict):
        raise BadRequest("a paper spec is a JSON object")
    sections = spec.get("sections")
    if not isinstance(sections, list) or not sections:
        raise BadRequest("a paper needs at least one section")
    versions = spec.get("versions", 1)
    _check_count(versions, MAX_VERSIONS, "versions")
    paper_seed = spec.get("seed")
    if paper_seed is not None and (isinstance(paper_seed, bool) or not isinstance(paper_seed, int)):
        raise BadRequest("a paper's seed is a whole number")
    if paper_seed is not None and not 0 <= paper_seed < 2**63:
        raise BadRequest("a paper's seed is from 0 to 2**63 - 1")
    paper_seed = _random_seed() if paper_seed is None else paper_seed
    shuffle = bool(spec.get("shuffle", False))

    total = 0
    for section in sections:
        items = section.get("items") if isinstance(section, dict) else None
        if not isinstance(items, list) or not items:
            raise BadRequest("every section needs at least one item")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("skill"), str):
                raise BadRequest("every item names a skill by its id")
            if item.get("source") is not None and not isinstance(item["source"], (str, int)):
                raise BadRequest("an item's source is a source key, a difficulty or an index")
            count = item.get("count", 1)
            _check_count(count)
            total += count
    if total * versions > MAX_PAPER_QUESTIONS:
        raise BadRequest(f"a paper may hold at most {MAX_PAPER_QUESTIONS} questions across its versions")

    out_versions = []
    for v in range(versions):
        label = _version_label(v)
        out_sections, number = [], 0
        for s_index, section in enumerate(sections):
            questions = []
            for i_index, item in enumerate(section["items"]):
                skill = lib.skill(item["skill"])
                src = _pick(skill, item.get("source"))
                seen: set[str] = set()
                for k in range(item.get("count", 1)):
                    for attempt in range(_DISTINCT_TRIES + 1):
                        seed = derive_seed(paper_seed, label, s_index, i_index, k, attempt)
                        question, qid = _make_with_id(lib, skill, seed, src)
                        if question.text not in seen or attempt == _DISTINCT_TRIES:
                            break
                    seen.add(question.text)
                    questions.append(question_view(skill, question, qid, answers=answers))
            if shuffle:
                random.Random(derive_seed(paper_seed, label, s_index, "order")).shuffle(questions)
            for q in questions:
                number += 1
                q["number"] = number
            out_sections.append({"title": _title(section.get("title")), "questions": questions})
        out_versions.append({"label": label, "sections": out_sections})

    used = {q["skill"] for v in out_versions for s in v["sections"] for q in s["questions"]}
    attributions = [SKILLS_ATTRIBUTION]
    if any(needs_acara_attribution(lib.skill(skill_id)) for skill_id in sorted(used)):
        attributions.append(ACARA_ATTRIBUTION)
    return {
        "title": _title(spec.get("title")) or "Worksheet",
        "seed": paper_seed,
        "answers": answers,
        "versions": out_versions,
        "attribution": attributions,
        "engine": lib.version(),
    }


# --- grading ---------------------------------------------------------------------------------------------------


def _number(value):
    """Marks as JSON: a whole number as an int, a fraction of a mark as text ("3/2"), so nothing is rounded."""
    if isinstance(value, Fraction):
        return value.numerator if value.denominator == 1 else str(value)
    return value


def result_view(result, *, reveal: bool = False) -> dict:
    """A `GradeResult` as data. `canonical` (the right answer) is kept only with `reveal`."""
    raw = asdict(result)
    for key in ("marks", "max_marks"):
        raw[key] = _number(raw.get(key))
    for part in (raw.get("parts") or {}).values():
        if isinstance(part, dict):
            for key in ("marks", "max_marks"):
                if key in part:
                    part[key] = _number(part[key])
    data = _plain(raw)
    if reveal:
        return data
    _drop_notes_naming_the_key(data)
    return _strip_canonical(data)


def _drop_notes_naming_the_key(data) -> None:
    """Without `reveal`, a note that states the right answer ("Turns only a quarter turn when told a half turn")
    is left out: the learner may try again, and a note must not hand over what the key is. Each part of a
    multi-part result is checked against its own key."""
    if not isinstance(data, dict):
        return
    note, key = data.get("note"), data.get("canonical")
    if isinstance(note, str) and isinstance(key, str):
        needle = " ".join(key.split()).casefold()
        if needle and re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", " ".join(note.split()).casefold()):
            data["note"] = None
            data["note_kind"] = None
    for part in (data.get("parts") or {}).values():
        _drop_notes_naming_the_key(part)


def _strip_canonical(data):
    """`data` without any `canonical` key, however deep: a multi-part result nests one per part."""
    if isinstance(data, dict):
        return {k: _strip_canonical(v) for k, v in data.items() if k != "canonical"}
    if isinstance(data, list):
        return [_strip_canonical(v) for v in data]
    return data


def _option_reply(item, reply):
    """A reply to a multiple-choice question or part, as the grader reads it.

    A reply here has a JSON type, so an integer is always the 0-based index of an option, and text is first read as
    an option's text: "8" picks the option 8, wherever it sits, as the command line does. The grader alone reads
    digits as an index, which would mark "8" against the ninth option. Text is matched whole with spacing
    collapsed, then ignoring case if only one option matches that way. Text that names no option goes to the
    grader as it is, which reads a letter ("B") or an index written as text.
    """
    if not isinstance(reply, str):
        return reply
    options = [" ".join(str(option).split()) for option in mcq_options(item)]
    text = " ".join(reply.split())
    if text in options:
        return options.index(text)
    folded = [i for i, option in enumerate(options) if option.casefold() == text.casefold()]
    if len(folded) == 1:
        return folded[0]
    # The text as the page showed it (en dash, ×, superscripts, ≥), when it names exactly one option.
    shown = [" ".join(str(option).split()).casefold() for option in [_learner_text(o) for o in mcq_options(item)]]
    seen = [i for i, option in enumerate(shown) if option == text.casefold()]
    return seen[0] if len(seen) == 1 else reply


def grade_reply(lib: Library, qid: str, reply, *, reveal: bool = False) -> dict:
    """Mark `reply` against the question `qid` names. `reply` is the learner's text; for a multiple-choice question
    the option's text, or the 0-based index of the option picked as an integer; for a multi-part question a dict
    keyed by part label. Raises `QuestionChanged` when the library no longer generates that question, `BadRequest`
    for a rubric skill."""
    question, skill = _regenerate(lib, qid)
    if not gradeable(skill):
        raise BadRequest(f"skill {skill.id!r} is marked against a rubric by a person, not by the engine")
    if isinstance(reply, bool) or not isinstance(reply, (str, int, dict)):
        raise BadRequest("a reply is text, an option index, or an object of replies keyed by part label")
    if isinstance(reply, dict):
        if not all(isinstance(v, str) or (isinstance(v, int) and not isinstance(v, bool)) for v in reply.values()):
            raise BadRequest("each part's reply is text or an option index")
        if not question.parts:
            raise BadRequest("this question has one answer; send its reply as text, not an object of parts")
        labels = [part.label for part in question.parts]
        unknown = [str(k) for k in reply if str(k) not in labels]
        if unknown:
            raise BadRequest(
                f"no such part {', '.join(map(repr, unknown))}; this question's parts are {', '.join(labels)}"
            )
        mcq_parts = {part.label: part for part in question.parts if part.answer_kind == "mcq"}
        reply = {
            str(k): (_option_reply(mcq_parts[str(k)], v) if str(k) in mcq_parts else str(v)) for k, v in reply.items()
        }
    elif skill.answer == "mcq" and not question.parts:
        reply = _option_reply(question, reply)
    elif isinstance(reply, int):
        reply = str(reply)
    return {"id": qid, **result_view(engine.grade(skill, question, reply), reveal=reveal)}
