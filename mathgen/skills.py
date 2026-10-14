"""Skill-tree loader and validator.

Scans ``skills/<domain>/<id>.yaml`` (plus optional ``policy.yaml`` and ``domains.yaml``) into plain
dataclasses, then validates the whole tree: unique ids matching filename stems, domains matching folder names,
resolvable acyclic prerequisites (an active skill builds only on active ones, every active senior or university
skill names at least one), registered answer kinds, a difficulty progression whose
tiers never get easier, and (for active skills only) that every source
actually generates a question via ``mathgen.make`` for a handful of seeds. Deferred skills are checked
structurally only -- their sources may reference generators or answer kinds that don't exist yet.

Nothing here talks to the database or HTTP layer; ``load_and_validate`` is the one entry point everything else
calls.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ._paths import default_skills_dir
from .capabilities import canonical_gate_id, gate_passed, skill_gates_satisfied

# The public names: see docs/api.md. Anything else here may change in any release.
__all__ = [
    "load_skills",
    "validate",
    "load_and_validate",
    "extra_skills_roots",
    "skills_root",
    "Skill",
    "Source",
    "Domain",
    "ValidationReport",
    "SkillsNotFoundError",
    "DuplicateSkillError",
]

# `skills/` beside `mathgen/` in a repository checkout, or the copy inside an installed wheel.
DEFAULT_SKILLS_DIR = default_skills_dir()
VALID_STAGES = {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}
READ_ALOUD_MAX_WORDS = 15  # the longest prompt a `read_aloud` skill may ask (tests/test_read_aloud.py)
# The tier a source stands for. An active skill's sources must each name one (see `validate`).
VALID_DIFFICULTIES = ("easy", "medium", "hard", "exam")
SOURCE_KEY_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")  # a source `key`: lower-case words joined by hyphens
VALID_ALIGNMENT_STATUSES = {"verified", "candidate", "unverified"}
# Bands (skills/domains.yaml) whose active skills must name a prerequisite: nothing at that level is a
# true starting point, and an empty list there means the graph has a hole, not that the skill has no past.
PREREQUISITE_REQUIRED_BANDS = ("senior", "university")
VALID_DEFERRED_REASONS = {
    "generator",
    "grader",
    "hard",
    "invalid",
    "visual",
    "assessment_contract",
    "formal_verification",
    "safety_gate",
}


# libyaml's C loader parses the tree ~8x faster than PyYAML's pure-Python one and yields identical
# data for every file in the library. Fall back where PyYAML was built without libyaml.
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def _yaml_load(f):
    return yaml.load(f, Loader=_YAML_LOADER)


def skills_root() -> Path:
    override = os.environ.get("MATHS_SKILLS_DIR")
    return Path(override) if override else DEFAULT_SKILLS_DIR


def extra_skills_roots() -> list[Path]:
    """Folders named by ``MATHS_EXTRA_SKILLS_DIRS`` (separated by ``os.pathsep``), in order.

    An extra folder adds skills that are not in the library -- a site's private skills, say -- laid out the
    same way (``<domain>/<id>.yaml``). Its skills join the main library's domains; the main library's
    ``domains.yaml`` and ``capability-gates.yaml`` still apply, and ``load_skills`` refuses an id that is already
    taken, so an extra folder can add skills but never replace one. ``~`` is expanded.
    """
    raw = os.environ.get("MATHS_EXTRA_SKILLS_DIRS", "")
    return [Path(p).expanduser() for p in raw.split(os.pathsep) if p]


class DuplicateSkillError(ValueError):
    """An extra skills folder holds a skill whose id is already in the library (or in an earlier extra folder).

    Raised by ``load_skills`` rather than left to ``validate``, because callers that only load (and key skills
    by id) would otherwise resolve the id to whichever copy came last.
    """


class SkillsNotFoundError(FileNotFoundError):
    """The skills folder is missing or holds no skill files.

    Raised instead of returning an empty library, so a broken install or a wrong ``MATHS_SKILLS_DIR`` cannot
    pass for "zero skills, no errors".
    """

    def __init__(self, root: Path):
        self.root = root
        super().__init__(
            f"no skills found under {root}. A repository checkout keeps them in skills/<domain>/<id>.yaml; "
            f"an installed package ships them inside the package. "
            f"Run from a checkout, reinstall the package, or set MATHS_SKILLS_DIR to a skills folder."
        )


@dataclass
class Source:
    topic: str
    params: dict = field(default_factory=dict)
    difficulty: str | None = None
    stages: list | None = None  # None means "all stages"
    key: str | None = None  # stable name within the skill; None means "use the difficulty label"

    @property
    def effective_key(self) -> str | None:
        """The name that identifies this source within its skill: `key`, else the `difficulty` label."""
        return self.key if self.key is not None else self.difficulty


@dataclass
class Misconception:
    id: str
    description: str = ""


@dataclass
class NearMiss:
    """A defensible answer that is not what a specific prompt asked for --
    right reasoning, aimed at a different question. Same `id`/`description` shape as `Misconception`,
    kept as a separate list rather than folded into it: a misconception is
    wrong thinking (tone: here is the slip), a near miss is sound thinking
    the wording pinned down (tone: your reasoning is sound, and here is what
    was asked instead). Collapsing them would teach the wrong lesson."""

    id: str
    description: str = ""


@dataclass
class Alignment:
    """Evidence linking a skill to an external curriculum or local framework.

    ``candidate`` and ``unverified`` links are deliberately visible but may
    never be presented as formal alignment. Only ``verified`` links require a
    review date; an external curriculum code is optional because university
    frameworks use stable area ids rather than course codes.
    """

    source_id: str
    status: str = "unverified"
    code: str | None = None
    framework: str | None = None
    area: str | None = None
    source_version: str | None = None
    source_url: str | None = None
    verified_on: str | None = None


@dataclass
class MentalLevel:
    """One rung of a skill's mental fluency ladder. ``target_s``/``limit_s`` are required for every level; the
    validator enforces that separately from dataclass defaults so a missing
    value is reported as a validation error rather than silently becoming 0."""

    topic: str
    params: dict = field(default_factory=dict)
    difficulty: str | None = None
    target_s: float | None = None
    limit_s: float | None = None


@dataclass
class Mental:
    levels: list[MentalLevel] = field(default_factory=list)
    entry_stage: int | None = None  # None = use the global mental.entry_stage


# Coverage labels, in the order a list must be written. year_level: F, 1 to 12, or university.
# naplan_domain: Australian Curriculum v9 mathematics strands. exam_board: senior exam boards.
YEAR_LEVELS = ("F", *(str(n) for n in range(1, 13)), "university")
NAPLAN_DOMAINS = ("number", "algebra", "measurement", "space", "statistics", "probability")
EXAM_BOARDS = ("nesa", "vcaa", "qcaa", "scsa", "sace", "tasc", "ib")
COVERAGE_LABELS = {"year_level": YEAR_LEVELS, "naplan_domain": NAPLAN_DOMAINS, "exam_board": EXAM_BOARDS}


@dataclass
class TierDemand:
    """What one tier asks of a learner: `steps` is the number of reasoning steps they must chain
    (substitute, then simplify, then compare is three), and `demand` says which ones in a sentence."""

    steps: int
    demand: str


@dataclass
class Skill:
    id: str
    name: str
    domain: str
    description: str = ""
    status: str = "active"
    deferred_reason: str | None = None
    prerequisites: list[str] = field(default_factory=list)
    capability_gates: list[str] = field(default_factory=list)
    answer: str = ""
    sources: list[Source] = field(default_factory=list)
    misconceptions: list[Misconception] = field(default_factory=list)
    near_misses: list[NearMiss] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    framework_areas: list[str] = field(default_factory=list)
    alignments: list[Alignment] = field(default_factory=list)
    placement_order: int = 1000
    placement_questions: int = 4
    placement_pass: int = 3
    placement_eligible: bool = True
    learner_band: str = "general"
    # Coverage, all optional (see skills/README.md): year levels the content belongs to, the Australian
    # Curriculum v9 strands NAPLAN numeracy can test it under, and the senior exam boards that cover it.
    year_level: tuple[str, ...] = ()
    naplan_domain: tuple[str, ...] = ()
    exam_board: tuple[str, ...] = ()
    mental: Mental | None = None
    # The skill's questions carry a figure the learner must see
    # it has no spoken form, so the skill may not have a mental block.
    needs_figure: bool = False
    # An early-years skill whose prompts are short enough to be read aloud to a child who cannot yet read
    # them: at most READ_ALOUD_MAX_WORDS words, with the figure carrying the rest (tests/test_read_aloud.py).
    read_aloud: bool = False
    # Rubric-assessed skills: `answer: rubric` names a versioned rubric. `rubric_id` is required and
    # validated below; `rubric_high_stakes` lets a skill (a proof, say) declare that its
    # answers always need the most careful marking route.
    rubric_id: str | None = None
    rubric_high_stakes: bool = False
    # How the tiers escalate in reasoning: tier name -> TierDemand (structured, checked by `validate`), or
    # the older one-paragraph prose, which is kept for authors and checked by nothing.
    difficulty_progression: dict[str, TierDemand] | str | None = None
    path: Path | None = None


@dataclass
class Domain:
    id: str
    name: str
    order: int
    band: str | None = None
    band_name: str | None = None


@dataclass
class Policy:
    stages: list[dict]
    maintenance: dict
    rules: dict
    mastery_caps: list[int]
    new_skill_due_immediately: bool = True
    session_question_cap: int = 15


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    total: int = 0
    active: int = 0
    deferred: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def by_status(self) -> dict:
        return {"active": self.active, "deferred": self.deferred, "total": self.total}


def _source_from_dict(d: dict) -> Source:
    params = d.get("params") or {}
    return Source(
        topic=d["topic"],
        params=params,
        difficulty=d.get("difficulty"),
        stages=d.get("stages"),
        key=d.get("key"),
    )


def _mental_level_from_dict(d: dict) -> MentalLevel:
    params = d.get("params") or {}
    return MentalLevel(
        topic=d.get("topic"),
        params=params,
        difficulty=d.get("difficulty"),
        target_s=d.get("target_s"),
        limit_s=d.get("limit_s"),
    )


def _mental_from_dict(d: dict | None) -> Mental | None:
    """``d`` is the raw ``mental:`` mapping from a skill YAML file, or an
    override's ``mental`` value. ``None``/``{}`` -> no mental block."""
    if not d:
        return None
    levels = [_mental_level_from_dict(lv) for lv in (d.get("levels") or [])]
    return Mental(levels=levels, entry_stage=d.get("entry_stage"))


def _alignment_from_dict(d: dict) -> Alignment:
    return Alignment(
        source_id=str(d.get("source_id", "")),
        status=str(d.get("status", "unverified")),
        code=d.get("code"),
        framework=d.get("framework"),
        area=d.get("area"),
        source_version=d.get("source_version"),
        source_url=d.get("source_url"),
        verified_on=d.get("verified_on"),
    )


def _rubric_block_from_dict(d: dict | None) -> tuple[str | None, bool]:
    """The skill YAML's `rubric:` block -> `(rubric_id, high_stakes)`.
    Structural shape only; whether the id actually resolves is checked in
    `validate()` when it is given a `rubric_loader` (without one it only
    requires that an id is present), so a bad id becomes a per-skill error
    instead of a failure at load time."""
    if not d:
        return None, False
    return d.get("id"), bool(d.get("high_stakes", False))


def _tier_demand_from_dict(d) -> TierDemand:
    if not isinstance(d, dict):
        return TierDemand(steps=0, demand="")
    return TierDemand(steps=d.get("steps", 0), demand=d.get("demand", ""))


def _progression_from_value(value) -> dict[str, TierDemand] | str | None:
    if isinstance(value, dict):
        return {str(tier): _tier_demand_from_dict(entry) for tier, entry in value.items()}
    return value


def _labels(value) -> tuple[str, ...]:
    """A YAML list of labels -> a tuple of strings, so `1` and `'1'` both read as "1"."""
    if value is None:
        return ()
    if isinstance(value, (str, int)):
        value = [value]
    return tuple(str(v) for v in value)


def _skill_from_dict(d: dict, path: Path) -> Skill:
    sources = [_source_from_dict(s) for s in (d.get("sources") or [])]
    misconceptions = [
        Misconception(id=m["id"], description=m.get("description", "")) for m in (d.get("misconceptions") or [])
    ]
    near_misses = [NearMiss(id=nm["id"], description=nm.get("description", "")) for nm in (d.get("near_misses") or [])]
    rubric_id, rubric_high_stakes = _rubric_block_from_dict(d.get("rubric"))
    return Skill(
        id=d.get("id"),
        name=d.get("name", ""),
        domain=d.get("domain", ""),
        description=d.get("description", ""),
        status=d.get("status", "active"),
        deferred_reason=d.get("deferred_reason"),
        prerequisites=list(d.get("prerequisites") or []),
        capability_gates=[canonical_gate_id(g) for g in (d.get("capability_gates") or [])],
        answer=d.get("answer", ""),
        sources=sources,
        misconceptions=misconceptions,
        near_misses=near_misses,
        tags=list(d.get("tags") or []),
        framework_areas=list(d.get("framework_areas") or []),
        alignments=[_alignment_from_dict(a) for a in (d.get("alignments") or [])],
        placement_order=int(d.get("placement_order", 1000)),
        placement_questions=int(d.get("placement_questions", 4)),
        placement_pass=int(d.get("placement_pass", 3)),
        placement_eligible=bool(d.get("placement_eligible", True)),
        learner_band=str(d.get("learner_band", "general")),
        year_level=_labels(d.get("year_level")),
        naplan_domain=_labels(d.get("naplan_domain")),
        exam_board=_labels(d.get("exam_board")),
        mental=_mental_from_dict(d.get("mental")),
        needs_figure=bool(d.get("needs_figure", False)),
        read_aloud=bool(d.get("read_aloud", False)),
        rubric_id=rubric_id,
        rubric_high_stakes=rubric_high_stakes,
        difficulty_progression=_progression_from_value(d.get("difficulty_progression")),
        path=path,
    )


def _invalid_skill(path: Path, domain: str, exc: Exception) -> Skill:
    """`id` and `domain` are set to what the loader can infer purely from the filesystem path so this stand-in
    still passes the id/filename-stem and domain/folder checks in `validate` below; the failure itself is
    surfaced there as both a validation error and this skill's `description`."""
    return Skill(
        id=path.stem,
        name=path.stem,
        domain=domain,
        description=f"failed to load {path.name}: {exc}",
        status="deferred",
        deferred_reason="invalid",
        path=path,
    )


# One loaded tree per root, keyed by every file's (path, mtime, size) so an
# edited, added or removed file reloads. Tests call load_skills at import time
# in ~100 modules and every xdist worker collects them all; re-parsing 700
# files each time was 94% of collection.
_LOAD_CACHE: dict[Path, tuple[tuple, tuple[list[Domain], list[Skill]]]] = {}


def _tree_fingerprint(root: Path) -> tuple:
    if not root.is_dir():
        return ()
    entries = []
    for p in sorted(root.glob("*.yaml")) + sorted(root.glob("*/*.yaml")):
        st = p.stat()
        entries.append((str(p), st.st_mtime_ns, st.st_size))
    return tuple(entries)


def load_skills(root: Path | None = None, extra_roots: list[Path] | None = None) -> tuple[list[Domain], list[Skill]]:
    """Load every ``<domain>/<id>.yaml`` under ``root`` plus ``domains.yaml``, then every skill in
    ``extra_roots`` (default: ``extra_skills_roots()``; pass ``[]`` for none).

    Cached per root and invalidated by any file change; every call returns
    its own deep copy, so callers may mutate the result freely.

    Raises ``SkillsNotFoundError`` when the folder is missing or holds no skill files: an empty library is
    always a mistake (a broken install, a wrong ``MATHS_SKILLS_DIR``), never a result. A missing extra folder
    raises ``FileNotFoundError``; an extra folder that exists but holds no skills is allowed. An extra skill
    whose id is already loaded raises ``DuplicateSkillError``.
    """
    root = Path(root) if root else skills_root()
    extras = [Path(p) for p in extra_roots] if extra_roots is not None else extra_skills_roots()
    domains, skills = _load_cached(root)
    if not skills:
        raise SkillsNotFoundError(root)
    seen = {s.id: s.path for s in skills if s.id}
    for extra in extras:
        extra = extra.expanduser()
        if not extra.is_dir():
            raise FileNotFoundError(
                f"extra skills folder {extra} does not exist (named by MATHS_EXTRA_SKILLS_DIRS or extra_roots)"
            )
        more = _load_cached(extra)[1]
        for s in more:
            if s.id in seen:
                raise DuplicateSkillError(f"skill {s.id!r} in {s.path} is already loaded from {seen[s.id]}")
            if s.id:
                seen[s.id] = s.path
        skills = skills + more
    return copy.deepcopy((domains, skills))


def _load_cached(root: Path) -> tuple[list[Domain], list[Skill]]:
    key = root.resolve()
    fingerprint = _tree_fingerprint(root)
    hit = _LOAD_CACHE.get(key)
    if hit is None or hit[0] != fingerprint:
        hit = (fingerprint, _load_skills_uncached(root))
        _LOAD_CACHE[key] = hit
    return hit[1]


def load_problems(skills: list[Skill]) -> list[str]:
    """One line per skill file that did not load, without generating any questions.

    A file that is not valid YAML or not the shape of a skill loads as a stand-in with
    ``deferred_reason: invalid``; a file with no ``id``, an ``id`` other than its filename, or a ``domain``
    other than its folder is just as unusable. ``validate`` reports all of these too, but it also generates
    questions for every active skill, which takes about half a minute. This is the quick check behind
    ``mathgen --skills``. Valid deferred skills are not problems.
    """
    problems = []
    for s in skills:
        where = f"{s.path.parent.name}/{s.path.name}" if s.path else f"{s.domain}/{s.id}.yaml"
        if s.status == "deferred" and s.deferred_reason == "invalid":
            problems.append(f"{where}: {s.description}")
        elif not s.id:
            problems.append(f"{where}: missing id")
        elif s.path and s.id != s.path.stem:
            problems.append(f"{where}: id {s.id!r} != filename stem {s.path.stem!r}")
        elif s.path and s.domain != s.path.parent.name:
            problems.append(f"{where}: domain {s.domain!r} != folder {s.path.parent.name!r}")
    return problems


def _load_skills_uncached(root: Path) -> tuple[list[Domain], list[Skill]]:
    """Load every ``<domain>/<id>.yaml`` under ``root`` plus ``domains.yaml``.

    Does not validate; a malformed individual file (YAML syntax error, or a
    shape `_skill_from_dict` can't parse) becomes a `status: deferred,
    deferred_reason: invalid` stand-in via `_invalid_skill` rather than
    raising, so one broken file can't crash the whole app at startup. Structural rule-checking of otherwise
    well-formed files happens in ``validate``.
    """
    skills: list[Skill] = []
    if root.is_dir():
        for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for yaml_path in sorted(domain_dir.glob("*.yaml")):
                try:
                    with open(yaml_path, "r", encoding="utf-8") as f:
                        data = _yaml_load(f) or {}
                    skill = _skill_from_dict(data, yaml_path)
                except Exception as exc:  # noqa: BLE001 - report every failure kind uniformly
                    skill = _invalid_skill(yaml_path, domain_dir.name, exc)
                skills.append(skill)

    domains = _load_domains(root, skills)
    return domains, skills


def _load_domains(root: Path, skills: list[Skill]) -> list[Domain]:
    domains_path = root / "domains.yaml"
    seen_ids = {s.domain for s in skills if s.domain}
    if domains_path.is_file():
        with open(domains_path, "r", encoding="utf-8") as f:
            raw = _yaml_load(f) or {}
        entries = raw.get("domains", raw) if isinstance(raw, dict) else raw
        band_names: dict[str, str] = {}
        if isinstance(raw, dict):
            for b in raw.get("bands") or []:
                bid = b.get("id")
                if bid:
                    band_names[bid] = b.get("name", bid)
        domains = []
        for i, e in enumerate(entries):
            did = e["id"]
            band = e.get("band")
            # `band` carries the raw value verbatim (None if absent) so
            # `validate()` below can flag a band naming an undeclared entry
            # in `bands:`. `band_name` is the resolved display name, and is
            # None whenever `band` is missing or names an unknown band --
            # this layer is presentation only and must not be able to fail
            # a load (an unlisted domain folder is already appended silently
            # today); `validate()` is where the typo is reported instead.
            band_name = band_names.get(band) if band is not None else None
            domains.append(
                Domain(
                    id=did,
                    name=e.get("name", did.replace("_", " ").replace("-", " ").title()),
                    order=e.get("order", i),
                    band=band,
                    band_name=band_name,
                )
            )
        listed = {d.id for d in domains}
        # Any domain folder not listed in domains.yaml is appended, alphabetically, after it.
        for did in sorted(seen_ids - listed):
            domains.append(Domain(id=did, name=did.replace("_", " ").replace("-", " ").title(), order=len(domains)))
        return domains
    return [
        Domain(id=did, name=did.replace("_", " ").replace("-", " ").title(), order=i)
        for i, did in enumerate(sorted(seen_ids))
    ]


def _tuplify(params: dict) -> dict:
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in params.items()}


def _has_cycle(skills_by_id: dict[str, Skill]) -> str | None:
    """Returns the id of a skill involved in a cycle, or None."""
    WHITE, GREY, BLACK = 0, 1, 2
    color = {sid: WHITE for sid in skills_by_id}

    def visit(sid: str) -> str | None:
        color[sid] = GREY
        for pre in skills_by_id[sid].prerequisites:
            if pre not in skills_by_id:
                continue  # unresolved prereq reported separately
            if color[pre] == GREY:
                return sid
            if color[pre] == WHITE:
                found = visit(pre)
                if found:
                    return found
        color[sid] = BLACK
        return None

    for sid in skills_by_id:
        if color[sid] == WHITE:
            found = visit(sid)
            if found:
                return found
    return None


def _progression_errors(s: Skill) -> list[str]:
    """A structured difficulty_progression names only tiers the skill's sources use, covers every one of
    them, gives each a whole number of steps (at least 1) and a sentence, and never asks fewer steps of a
    harder tier than of an easier one."""
    errors = []
    progression = s.difficulty_progression
    unknown = [t for t in progression if t not in VALID_DIFFICULTIES]
    if unknown:
        errors.append(f"{s.id}: difficulty_progression names unknown tier(s) {unknown}")
    used = {src.difficulty for src in s.sources if src.difficulty in VALID_DIFFICULTIES}
    missing = [t for t in VALID_DIFFICULTIES if t in used and t not in progression]
    if missing:
        errors.append(f"{s.id}: difficulty_progression does not describe tier(s) {missing}")
    extra = [t for t in VALID_DIFFICULTIES if t in progression and t not in used]
    if extra:
        errors.append(f"{s.id}: difficulty_progression describes tier(s) {extra} that no source uses")
    previous = None
    for tier in VALID_DIFFICULTIES:
        entry = progression.get(tier)
        if entry is None:
            continue
        if isinstance(entry.steps, bool) or not isinstance(entry.steps, int) or entry.steps < 1:
            errors.append(f"{s.id}: difficulty_progression.{tier}.steps must be a whole number of at least 1")
            continue
        if not isinstance(entry.demand, str) or not entry.demand.strip():
            errors.append(f"{s.id}: difficulty_progression.{tier}.demand must say what the tier asks")
        if previous is not None and entry.steps < previous[1]:
            errors.append(
                f"{s.id}: difficulty_progression.{tier} asks {entry.steps} step(s), fewer than "
                f"{previous[0]} ({previous[1]}); a harder tier must not ask less"
            )
        previous = (tier, entry.steps)
    return errors


def validate(
    domains: list[Domain],
    skills: list[Skill],
    graders: dict | None = None,
    gate_registry: dict | None = None,
    rubric_loader: Callable[[str], object] | None = None,
) -> ValidationReport:
    """`graders` defaults to the live registry in mathgen.grading (kept as a parameter for isolated tests).
    `gate_registry` defaults to the live capability-gate registry in skills/capability-gates.yaml (also kept
    as a parameter for isolated tests, e.g. a synthetic registry with a gate marked "passed").
    `rubric_loader` resolves a rubric id for `answer: rubric` skills: it is called with the id and raises
    `FileNotFoundError` or `ValueError` for one that does not exist or is malformed. This package ships no
    rubric library, so without a loader the id is required but not resolved."""
    if graders is None:
        from .grading import GRADERS as graders

    if gate_registry is None:
        from .capabilities import load_capability_gates

        gate_registry = load_capability_gates()

    import mathgen

    report = ValidationReport(total=len(skills))
    if not skills:
        report.errors.append("no skills were loaded: an empty library is not a valid one")
    domain_ids = {d.id for d in domains}

    for d in domains:
        # A domain whose `band` names an entry not declared in `bands:` is a
        # typo in a file we control (unlike an unlisted domain folder, which
        # the loader appends silently) -- `_load_domains` resolves an unknown
        # band to `band_name=None` while keeping the raw `band` value, so
        # this is the one place that combination is flagged.
        if d.band is not None and d.band_name is None:
            report.errors.append(f"domain {d.id!r}: band {d.band!r} is not declared in bands:")

    band_of = {d.id: d.band for d in domains}
    by_id: dict[str, Skill] = {}
    for s in skills:
        if not s.id:
            report.errors.append(f"<{s.path}>: missing id")
            continue
        stem = s.path.stem if s.path else None
        if stem and s.id != stem:
            report.errors.append(f"{s.id}: id != filename stem ({stem!r})")
        if s.id in by_id:
            report.errors.append(f"{s.id}: duplicate id ({by_id[s.id].path} and {s.path})")
            continue
        by_id[s.id] = s

    for s in skills:
        if s.id not in by_id or by_id[s.id] is not s:
            continue  # already reported as a dup
        folder = s.path.parent.name if s.path else s.domain
        if s.domain != folder:
            report.errors.append(f"{s.id}: domain {s.domain!r} != folder {folder!r}")
        if domain_ids and s.domain not in domain_ids:
            report.errors.append(f"{s.id}: domain {s.domain!r} not in domains.yaml")

        if s.status not in ("active", "deferred"):
            report.errors.append(f"{s.id}: unknown status {s.status!r}")
        if s.placement_questions < 1:
            report.errors.append(f"{s.id}: placement_questions must be at least 1")
        if not 1 <= s.placement_pass <= s.placement_questions:
            report.errors.append(f"{s.id}: placement_pass must be between 1 and placement_questions")
        if s.learner_band not in ("early", "primary", "general"):
            report.errors.append(f"{s.id}: learner_band must be early, primary or general")
        for name, allowed in COVERAGE_LABELS.items():
            got = getattr(s, name)
            if len(got) != len(set(got)):
                report.errors.append(f"{s.id}: {name} contains duplicates")
            bad = [g for g in got if g not in allowed]
            if bad:
                report.errors.append(f"{s.id}: {name} has unknown labels {bad} (allowed: {', '.join(allowed)})")
            elif list(got) != [a for a in allowed if a in got]:
                report.errors.append(f"{s.id}: {name} must be listed in this order: {', '.join(allowed)}")
        if s.status == "deferred" and not s.deferred_reason:
            report.errors.append(f"{s.id}: deferred skill missing deferred_reason")
        if s.status == "deferred" and s.deferred_reason not in VALID_DEFERRED_REASONS:
            report.errors.append(f"{s.id}: deferred_reason {s.deferred_reason!r} is not recognised")
        if s.status == "deferred" and s.deferred_reason == "invalid":
            report.errors.append(f"{s.id}: {s.description}")

        for pre in s.prerequisites:
            if pre not in by_id:
                report.errors.append(f"{s.id}: prerequisite {pre!r} does not resolve")
            elif s.status == "active" and by_id[pre].status != "active":
                report.errors.append(
                    f"{s.id}: active skill has deferred prerequisite {pre!r}; a host that waits for "
                    f"prerequisites would never offer it"
                )
        if s.status == "active" and not s.prerequisites and band_of.get(s.domain) in PREREQUISITE_REQUIRED_BANDS:
            report.errors.append(f"{s.id}: active {band_of[s.domain]} skill has no prerequisites")

        if len(s.framework_areas) != len(set(s.framework_areas)):
            report.errors.append(f"{s.id}: framework_areas contains duplicates")

        # `misconceptions` (wrong thinking) and `near_misses` (right thinking, wrong question) are
        # deliberately separate lists so the feedback tone differs -- an id used in both would be
        # ambiguous about which tone to show, so it is a validation error.
        misconception_ids = {m.id for m in s.misconceptions}
        colliding_near_miss_ids = {nm.id for nm in s.near_misses} & misconception_ids
        for note_id in sorted(colliding_near_miss_ids):
            report.errors.append(f"{s.id}: near_misses id {note_id!r} collides with a misconceptions id")
        for alignment in s.alignments:
            if not alignment.source_id:
                report.errors.append(f"{s.id}: alignment missing source_id")
            if alignment.status not in VALID_ALIGNMENT_STATUSES:
                report.errors.append(f"{s.id}: alignment status {alignment.status!r} is not recognised")
            if alignment.status == "verified" and not alignment.verified_on:
                report.errors.append(f"{s.id}: verified alignment missing verified_on")
            if alignment.framework and not alignment.area:
                report.errors.append(f"{s.id}: framework alignment missing area")

        if isinstance(s.difficulty_progression, dict):
            report.errors.extend(_progression_errors(s))
        elif s.difficulty_progression is not None and not isinstance(s.difficulty_progression, str):
            report.errors.append(f"{s.id}: difficulty_progression must be a mapping of tiers or prose")

        seen_keys: dict[str, int] = {}
        for i, src in enumerate(s.sources):
            ekey = src.effective_key
            if src.key is not None and (not isinstance(src.key, str) or not SOURCE_KEY_RE.match(src.key)):
                report.errors.append(
                    f"{s.id}: source {i} key {src.key!r} must be lower-case words joined by hyphens "
                    f"(letters and digits only, matching {SOURCE_KEY_RE.pattern})"
                )
            elif ekey is not None:
                if ekey in seen_keys:
                    report.errors.append(
                        f"{s.id}: sources {seen_keys[ekey]} and {i} share the key {ekey!r}; "
                        f"give one of them an explicit `key` (a source's key defaults to its difficulty)"
                    )
                else:
                    seen_keys[ekey] = i
            if src.stages:
                for st in src.stages:
                    if st not in VALID_STAGES:
                        report.errors.append(f"{s.id}: source stage {st!r} invalid")

        # Capability gates: an unknown gate slug is an error for any skill regardless of status --
        # fail closed, never treat an unrecognised gate as satisfied.
        for gate in s.capability_gates:
            if gate not in gate_registry:
                report.errors.append(f"{s.id}: capability gate {gate!r} is not registered")

        # Rubric wiring:
        # `answer: rubric` is not a deterministic grader, so it is validated
        # against the rubric library instead of `graders`, for active and
        # deferred skills alike -- a mis-typed rubric id is a load error
        # regardless of whether the skill is live yet.
        if s.answer == "rubric":
            if not s.rubric_id:
                report.errors.append(f"{s.id}: answer 'rubric' requires a rubric.id")
            elif rubric_loader is not None:
                try:
                    rubric_loader(s.rubric_id)
                except (FileNotFoundError, ValueError) as exc:
                    report.errors.append(f"{s.id}: rubric {s.rubric_id!r} is invalid: {exc}")

        if s.status == "active":
            report.active += 1
            if not s.sources:
                report.errors.append(f"{s.id}: active skill has no sources")
            if s.answer != "rubric" and s.answer not in graders:
                report.errors.append(f"{s.id}: answer kind {s.answer!r} not registered")
            if s.capability_gates and not skill_gates_satisfied(s.capability_gates, gate_registry):
                blocking = [g for g in s.capability_gates if not gate_passed(g, gate_registry)]
                report.errors.append(f"{s.id}: active skill has unsatisfied capability gate(s) {blocking}")
            for i, src in enumerate(s.sources):
                if src.difficulty is None:
                    report.errors.append(
                        f"{s.id}: source {i} (topic={src.topic!r}) has no difficulty; "
                        f"it must be one of {', '.join(VALID_DIFFICULTIES)}"
                    )
                elif src.difficulty not in VALID_DIFFICULTIES:
                    report.errors.append(
                        f"{s.id}: source {i} (topic={src.topic!r}) difficulty {src.difficulty!r} "
                        f"must be one of {', '.join(VALID_DIFFICULTIES)}"
                    )
            for src in s.sources:
                for seed in range(1, 6):
                    try:
                        params = _tuplify(src.params)
                        q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **params)
                    except Exception as exc:  # noqa: BLE001 - report every failure kind uniformly
                        report.errors.append(
                            f"{s.id}: source topic={src.topic!r} params={src.params!r} "
                            f"difficulty={src.difficulty!r} failed at seed={seed}: {exc}"
                        )
                        break
                    unknown_part_kinds = sorted({p.answer_kind for p in q.parts} - set(graders))
                    if unknown_part_kinds:
                        report.errors.append(
                            f"{s.id}: source topic={src.topic!r} has parts graded as {unknown_part_kinds}, "
                            f"which have no grader"
                        )
                        break
                    if getattr(q, "figure", None) is not None and not s.needs_figure:
                        report.errors.append(
                            f"{s.id}: source topic={src.topic!r} draws a figure; "
                            f"the skill must declare needs_figure: true"
                        )
                        break
        else:
            report.deferred += 1

        # mental block: structural checks
        # always run; generation is only checked for active skills, same as
        # sources -- a deferred skill's mental block may reference generators
        # that don't exist yet.
        if s.needs_figure and s.mental is not None:
            report.errors.append(f"{s.id}: needs_figure skills have no spoken form and may not have a mental block")
        if s.mental is not None:
            if not (1 <= len(s.mental.levels) <= 5):
                report.errors.append(f"{s.id}: mental.levels must have 1 to 5 entries")
            for i, lvl in enumerate(s.mental.levels):
                if not isinstance(lvl.target_s, (int, float)) or isinstance(lvl.target_s, bool) or lvl.target_s <= 0:
                    report.errors.append(f"{s.id}: mental.levels[{i}].target_s must be a positive number")
                if not isinstance(lvl.limit_s, (int, float)) or isinstance(lvl.limit_s, bool) or lvl.limit_s <= 0:
                    report.errors.append(f"{s.id}: mental.levels[{i}].limit_s must be a positive number")
                elif (
                    isinstance(lvl.target_s, (int, float))
                    and not isinstance(lvl.target_s, bool)
                    and lvl.limit_s < lvl.target_s
                ):
                    report.errors.append(f"{s.id}: mental.levels[{i}].limit_s must be >= target_s")
                if s.status == "active":
                    for seed in range(1, 6):
                        try:
                            params = _tuplify(lvl.params)
                            mathgen.make(lvl.topic, seed=seed, difficulty=lvl.difficulty, **params)
                        except Exception as exc:  # noqa: BLE001
                            report.errors.append(
                                f"{s.id}: mental level {i} topic={lvl.topic!r} params={lvl.params!r} "
                                f"difficulty={lvl.difficulty!r} failed at seed={seed}: {exc}"
                            )
                            break

    cycle_at = _has_cycle(by_id)
    if cycle_at:
        report.errors.append(f"prerequisite graph has a cycle reachable from {cycle_at!r}")

    return report


def load_and_validate(
    root: Path | None = None,
    rubric_loader: Callable[[str], object] | None = None,
    extra_roots: list[Path] | None = None,
) -> tuple[list[Domain], list[Skill], ValidationReport]:
    domains, skills = load_skills(root, extra_roots)
    report = validate(domains, skills, rubric_loader=rubric_loader)
    return domains, skills, report
