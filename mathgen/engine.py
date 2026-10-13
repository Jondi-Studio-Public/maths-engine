"""The small public API for working with skills: load, generate a question, grade an answer.

    from mathgen import engine

    skills = {s.id: s for s in engine.load_skills()[1]}
    skill = skills["simplify-surds"]
    q = engine.make_for_skill(skill, seed=7)
    print(q.text)                              # Simplify √756. Give your answer in the form a√b.
    print(q.answer_text)                       # 6*sqrt(21)
    print(engine.grade(skill, q, "6√21").correct)   # True
    print(engine.grade(skill, q, "2√3").correct)    # False

A question has a stable identity. `question_id(skill, source, seed, q)` is `<skill id>/<source key>/<seed>/<hash>`,
where the source key is the source's optional `key` (default: its `difficulty` label) and the hash covers the
question's content (`content_hash`). `make_with_id` builds a question and its id together, and `regenerate(id,
skills)` rebuilds the question an id names, raising `QuestionChanged` if the generator now produces different
content, so a host re-grading an old attempt finds out instead of marking it against a different question.

Nothing here touches a database, the network or any app code.

For a multiple-choice skill (`skill.answer == "mcq"`), `grade` takes the 0-based index of the chosen option in
`mathgen.core.mcq_options(q)`, the seed-shuffled list `[answer, *distractors]`; the command line shows the
options and accepts an option's label or its text instead.
"""

from __future__ import annotations

import hashlib
import json

from . import make
from .core import display_question, paper_notation  # noqa: F401  (re-exported)
from .grading import GradeResult, grade as _grade_kind
from .skills import Domain, Skill, Source, load_skills  # noqa: F401  (re-exported)


def _tuplify(params: dict | None) -> dict:
    """Generators expect e.g. `kinds=(...)`; YAML and JSON only have lists."""
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in (params or {}).items()}


def pick_source(skill: Skill, source: Source | int | str | None = None) -> Source:
    """A skill's source: a `Source`, an index into `skill.sources`, a source key (which defaults to the
    difficulty label), a difficulty label shared by several sources (the first of them), or the first source."""
    if isinstance(source, Source):
        return source
    if not skill.sources:
        raise ValueError(f"skill {skill.id!r} has no question sources (status: {skill.status})")
    if source is None:
        return skill.sources[0]
    if isinstance(source, int):
        return skill.sources[source]
    for src in skill.sources:
        if src.effective_key == source:
            return src
    for src in skill.sources:  # a difficulty label shared by several sources picks the first of them
        if src.difficulty == source:
            return src
    labels = sorted({src.difficulty for src in skill.sources if src.difficulty})
    keys = sorted({src.effective_key for src in skill.sources if src.effective_key})
    raise KeyError(
        f"skill {skill.id!r} has no source with difficulty {source!r}"
        + (f"; its difficulties are {', '.join(labels)}" if labels else "; none of its sources is labelled")
        + (f"; its source keys are {', '.join(keys)}" if keys != labels else "")
    )


def make_for_skill(skill: Skill, seed: int | None = None, *, source: Source | int | str | None = None):
    """Build one verified question for `skill`. Same (skill, source, seed) gives the same question."""
    src = pick_source(skill, source)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **_tuplify(src.params))


def grade(skill: Skill, question, given: str | dict) -> GradeResult:
    """Grade `given` against `question` with the grader the skill's answer kind names.

    A multi-part question (`question.parts`) is marked part by part with each part's own grader, and `given`
    may be a dict of replies keyed by part label; the result's `marks`, `max_marks` and `parts` give the
    marks (see `mathgen.grading.parts`)."""
    return _grade_kind(skill.answer, question, given, skill=skill)


def content_hash(question) -> str:
    """The first 12 hex characters of SHA-256 over the question's content.

    The content is `question.to_dict()` as JSON with sorted keys, compact separators and ASCII escapes, so
    the hash does not depend on dictionary order, the platform or the Python version. It covers everything
    `to_dict()` reports: the prompt and its LaTeX, the answer and its forms, distractors, worked steps,
    figure, parts and the rest, and also `topic`, `level`, `seed` and `difficulty`. Nothing is left out: every
    field is a pure function of (generator, params, difficulty, seed), including presentational ones, because
    a change to what a learner sees is a change to the question. Values JSON cannot represent are written
    with `str()`."""
    canonical = json.dumps(question.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _key_of(skill: Skill, src: Source) -> str:
    key = src.effective_key
    if key is None:
        raise ValueError(f"skill {skill.id!r} has a source (topic {src.topic!r}) with no key and no difficulty")
    return key


def question_id(skill: Skill, source: Source | int | str | None, seed: int, question) -> str:
    """`<skill id>/<source key>/<seed>/<hash>`: names (skill, source, seed) and pins the content they gave.

    `source` is anything `pick_source` accepts. The first three parts say how to rebuild the question and
    stay the same while the library is unchanged; the hash changes when a generator, a source's params or
    difficulty, or the engine changes what the question is. See `regenerate`."""
    src = pick_source(skill, source)
    return f"{skill.id}/{_key_of(skill, src)}/{seed}/{content_hash(question)}"


def make_with_id(skill: Skill, seed: int, *, source: Source | int | str | None = None):
    """Build one question for `skill` and its id: `(question, question_id)`. Same question as `make_for_skill`."""
    src = pick_source(skill, source)
    question = make_for_skill(skill, seed, source=src)
    return question, question_id(skill, src, seed, question)


class QuestionChanged(Exception):
    """`regenerate` rebuilt a question whose content differs from the one the id was issued for.

    `old_id` is the id given, `new_id` the id of what the current library produces, and `question` that
    question. A host should not mark an old attempt against `question`; it can show the attempt as it was
    recorded, or re-ask the new question."""

    def __init__(self, old_id: str, new_id: str, question):
        super().__init__(f"question {old_id} now generates different content ({new_id})")
        self.old_id = old_id
        self.new_id = new_id
        self.question = question


def parse_question_id(qid: str) -> tuple[str, str, int, str]:
    """Split an id into (skill id, source key, seed, hash). Raises ValueError if it is not well formed."""
    parts = qid.split("/") if isinstance(qid, str) else []
    if len(parts) != 4 or not all(parts):
        raise ValueError(f"not a question id (want <skill>/<source key>/<seed>/<hash>): {qid!r}")
    skill_id, key, seed_text, digest = parts
    try:
        seed = int(seed_text)
    except ValueError:
        raise ValueError(f"not a question id: seed {seed_text!r} is not an integer in {qid!r}") from None
    return skill_id, key, seed, digest


def regenerate(qid: str, skills):
    """Rebuild the question a `question_id` names. `skills` is a list of `Skill`s or a dict by skill id.

    Raises `KeyError` for an unknown skill or source key, `ValueError` for a malformed id, and
    `QuestionChanged` (carrying the new question) when the hash no longer matches."""
    skill_id, key, seed, _digest = parse_question_id(qid)
    by_id = skills if isinstance(skills, dict) else {s.id: s for s in skills}
    if skill_id not in by_id:
        raise KeyError(f"unknown skill {skill_id!r} in question id {qid!r}")
    skill = by_id[skill_id]
    matches = [src for src in skill.sources if src.effective_key == key]
    if not matches:
        known = ", ".join(sorted({s.effective_key for s in skill.sources if s.effective_key})) or "none"
        raise KeyError(f"skill {skill_id!r} has no source with key {key!r} (its keys: {known})")
    question, new_id = make_with_id(skill, seed, source=matches[0])
    if new_id != qid:
        raise QuestionChanged(qid, new_id, question)
    return question
