"""Recipes: connected exam questions built from one subject and an ordered list of asks.

A *subject* is an object drawn once per question with every fact about it known. An *ask* is one part about a
subject: it states the prompt, works out the answer, builds tagged distractors and sets follow-through. A *recipe*
(a YAML file under ``recipes/``) names a subject and the asks, with marks. Each recipe is registered as the
generator topic ``recipe_<id>`` and produces an ordinary multi-part question (``docs/design/multi-part-questions.md``),
so ``make()``, ``grade()`` and skills use it unchanged. See ``docs/design/recipes.md`` for the interface.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import yaml

from . import core

__all__ = [
    "Subject",
    "AskContext",
    "Recipe",
    "RecipeError",
    "subject",
    "ask",
    "compose",
    "load_recipes",
    "register_recipe",
    "register_all",
    "recipes_dir",
    "topic_for",
    "conformance",
]

RECIPE_ID_RE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")
_RECIPE_KEYS = {"id", "title", "level", "subject", "technology_active", "asks"}
_ASK_KEYS = {"ask", "marks"}


class RecipeError(ValueError):
    """A recipe, subject or ask is malformed, or names something that does not exist."""


class Subject:
    """The object a recipe asks about, drawn once per question.

    Subclass, register with ``@subject("name")`` and implement ``draw`` and ``stem``. ``technology_active`` is set by
    the composer before ``stem()`` or any ask runs (and is passed to ``draw``): ``None`` or ``False`` means exact
    forms, ``True`` means decimals. ``properties`` names what the subject can supply, for generic asks.
    """

    name: ClassVar[str] = ""
    properties: ClassVar[frozenset[str]] = frozenset()
    technology_active: bool | None = None

    @classmethod
    def draw(cls, rng, technology_active: bool | None) -> Subject:
        raise NotImplementedError(f"{cls.__name__}.draw")

    def stem(self) -> str:
        """The information every part shares, plain text, placed before part (a)."""
        raise NotImplementedError(f"{type(self).__name__}.stem")

    def figure(self) -> dict | None:
        """An optional figure for the whole question (``mathgen.figures``)."""
        return None


@dataclass
class AskContext:
    """What an ask can see of the part it is building."""

    label: str
    index: int
    earlier: list[dict] = field(default_factory=list)  # the earlier part dicts, with their labels
    _labels: dict[str, str] = field(default_factory=dict)  # ask name -> label of its latest occurrence

    def label_of(self, ask_name: str) -> str:
        """The label of an earlier ask, for writing ``follow_through`` (``f"2*part_{ctx.label_of('x')}"``)."""
        if ask_name not in self._labels:
            raise RecipeError(f"ask {ask_name!r} has not come earlier in this recipe (earlier: {sorted(self._labels)})")
        return self._labels[ask_name]


@dataclass(frozen=True)
class _AskSpec:
    fn: Callable
    name: str
    subject: str | None
    needs: frozenset[str]
    requires: tuple[str, ...]
    repeatable: bool
    default_marks: int


SUBJECTS: dict[str, type[Subject]] = {}
ASKS: dict[tuple[str | None, str], _AskSpec] = {}


def subject(name: str):
    """Register a Subject subclass under ``name``."""

    def deco(cls: type[Subject]) -> type[Subject]:
        if name in SUBJECTS:
            raise RecipeError(f"duplicate subject: {name}")
        cls.name = name
        SUBJECTS[name] = cls
        return cls

    return deco


def ask(
    subject_name: str | None,
    name: str,
    *,
    needs: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    repeatable: bool = False,
    default_marks: int = 1,
):
    """Register ``fn(subject, rng, ctx) -> part dict``.

    ``subject_name`` ties the ask to one subject type; ``None`` registers a generic ask valid for any subject whose
    ``properties`` include every name in ``needs``. A per-subject ask beats a generic one of the same name.
    ``requires`` lists asks that must come earlier in a recipe.
    """

    def deco(fn: Callable) -> Callable:
        key = (subject_name, name)
        if key in ASKS:
            raise RecipeError(f"duplicate ask: {key}")
        ASKS[key] = _AskSpec(fn, name, subject_name, frozenset(needs), tuple(requires), repeatable, default_marks)
        return fn

    return deco


@dataclass(frozen=True)
class Recipe:
    id: str
    title: str
    level: int
    subject: str
    asks: tuple[tuple[str, int | None], ...]  # (ask name, marks or None for the ask's default)
    technology_active: bool | None = None


def topic_for(recipe_id: str) -> str:
    return f"recipe_{recipe_id}"


def _resolve(recipe: Recipe, name: str) -> _AskSpec:
    cls = SUBJECTS.get(recipe.subject)
    if cls is None:
        raise RecipeError(f"recipe {recipe.id!r}: unknown subject {recipe.subject!r}; known: {sorted(SUBJECTS)}")
    spec = ASKS.get((recipe.subject, name))
    if spec is None:
        generic = ASKS.get((None, name))
        if generic is not None and generic.needs <= cls.properties:
            return generic
        if generic is not None:
            missing = sorted(generic.needs - cls.properties)
            raise RecipeError(f"recipe {recipe.id!r}: ask {name!r} needs {missing}, which {recipe.subject!r} lacks")
        known = sorted(
            {n for (s, n), a in ASKS.items() if s == recipe.subject or (s is None and a.needs <= cls.properties)}
        )
        raise RecipeError(f"recipe {recipe.id!r}: unknown ask {name!r} for subject {recipe.subject!r}; known: {known}")
    return spec


def validate(recipe: Recipe) -> None:
    """Raise `RecipeError` unless every ask resolves and comes after the asks it requires."""
    if not RECIPE_ID_RE.match(recipe.id):
        raise RecipeError(f"recipe id {recipe.id!r} must be lower-case words joined by underscores")
    if not recipe.asks:
        raise RecipeError(f"recipe {recipe.id!r} has no asks")
    seen: list[str] = []
    for name, marks in recipe.asks:
        spec = _resolve(recipe, name)
        if name in seen and not spec.repeatable:
            raise RecipeError(f"recipe {recipe.id!r}: ask {name!r} appears twice but is not repeatable")
        late = [r for r in spec.requires if r not in seen]
        if late:
            raise RecipeError(f"recipe {recipe.id!r}: ask {name!r} requires {late} earlier")
        if marks is not None and (not isinstance(marks, int) or isinstance(marks, bool) or marks < 1):
            raise RecipeError(f"recipe {recipe.id!r}: marks for {name!r} must be a whole number of at least 1")
        seen.append(name)


def compose(recipe: Recipe, rng, technology_active: bool | None = None) -> dict:
    """Build the generator dict (``text`` + ``parts``) for one draw of ``recipe``."""
    cls = SUBJECTS.get(recipe.subject)
    if cls is None:
        raise RecipeError(f"recipe {recipe.id!r}: unknown subject {recipe.subject!r}")
    tech = technology_active if technology_active is not None else recipe.technology_active
    subj = cls.draw(rng, tech)
    subj.technology_active = tech
    parts: list[dict] = []
    ctx = AskContext(label="", index=0)
    for i, (name, marks) in enumerate(recipe.asks):
        spec = _resolve(recipe, name)
        label = chr(ord("a") + i)
        ctx = AskContext(label=label, index=i, earlier=[dict(p) for p in parts], _labels=dict(ctx._labels))
        part = dict(spec.fn(subj, rng, ctx))
        if "label" in part and part["label"] != label:
            raise RecipeError(f"ask {name!r} set label {part['label']!r}; the composer assigns {label!r}")
        part["label"] = label
        part["marks"] = marks if marks is not None else part.get("marks", spec.default_marks)
        parts.append(part)
        ctx._labels[name] = label
    out: dict[str, Any] = dict(text=subj.stem(), parts=parts)
    figure = subj.figure()
    if figure is not None:
        out["figure"] = figure
    return out


def register_recipe(recipe: Recipe) -> str:
    """Validate ``recipe`` and register it as a generator topic; returns the topic."""
    validate(recipe)
    topic = topic_for(recipe.id)

    def generate(rng, technology_active=None):
        return compose(recipe, rng, technology_active)

    generate.__name__ = topic
    core.generator(topic, level=recipe.level)(generate)
    return topic


def recipes_dir() -> Path:
    """``MATHS_RECIPES_DIR`` if set, else ``recipes/`` beside the package (a repository checkout)."""
    env = os.environ.get("MATHS_RECIPES_DIR")
    return Path(env) if env else Path(__file__).resolve().parent.parent / "recipes"


def _parse(path: Path) -> Recipe:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise RecipeError(f"{path.name}: not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise RecipeError(f"{path.name}: must be a mapping")
    unknown = set(raw) - _RECIPE_KEYS
    if unknown:
        raise RecipeError(f"{path.name}: unknown keys {sorted(unknown)}")
    for key in ("id", "title", "level", "subject", "asks"):
        if key not in raw:
            raise RecipeError(f"{path.name}: missing {key!r}")
    if raw["id"] != path.stem:
        raise RecipeError(f"{path.name}: id {raw['id']!r} must equal the file name")
    if not isinstance(raw["level"], int) or isinstance(raw["level"], bool):
        raise RecipeError(f"{path.name}: level must be an integer")
    tech = raw.get("technology_active")
    if tech is not None and not isinstance(tech, bool):
        raise RecipeError(f"{path.name}: technology_active must be true, false or null")
    if not isinstance(raw["asks"], list) or not raw["asks"]:
        raise RecipeError(f"{path.name}: asks must be a non-empty list")
    asks = []
    for entry in raw["asks"]:
        if not isinstance(entry, dict) or "ask" not in entry or set(entry) - _ASK_KEYS:
            raise RecipeError(f"{path.name}: each ask is a mapping with 'ask' and optional 'marks', got {entry!r}")
        asks.append((str(entry["ask"]), entry.get("marks")))
    return Recipe(
        id=str(raw["id"]),
        title=str(raw["title"]),
        level=raw["level"],
        subject=str(raw["subject"]),
        asks=tuple(asks),
        technology_active=tech,
    )


def load_recipes(directory: Path | None = None) -> list[Recipe]:
    """Every ``*.yaml`` recipe under ``directory`` (default `recipes_dir()`), sorted by id; none if it is absent."""
    root = directory or recipes_dir()
    if not root.is_dir():
        return []
    return [_parse(p) for p in sorted(root.rglob("*.yaml"))]


def register_all(directory: Path | None = None) -> list[str]:
    """Register every recipe found; called once when ``mathgen.generators`` loads."""
    return [register_recipe(r) for r in load_recipes(directory)]


def _shape(text: str) -> str:
    return re.sub(r"\d+(\.\d+)?", "#", text)


def conformance(recipe_id: str, seeds: int = 300, technology_active: bool | None = None) -> list[str]:
    """Problems found building ``recipe_id`` over ``seeds`` seeds; an empty list means it conforms.

    Checks: every seed builds; every answer is present; no part's stem or prompt contains a later part's answer
    (answers of three or more characters); at least 100 distinct last-part answers per 300 seeds; at least 3
    distinct question shapes (digits masked).
    """
    topic = topic_for(recipe_id)
    problems: list[str] = []
    finals: set[str] = set()
    shapes: set[str] = set()
    for seed in range(seeds):
        try:
            q = core.make(
                topic, seed=seed, **({} if technology_active is None else {"technology_active": technology_active})
            )
        except Exception as e:  # noqa: BLE001 -- report every failure kind rather than stop at the first
            problems.append(f"seed {seed}: {type(e).__name__}: {e}")
            if len(problems) >= 10:
                problems.append("stopping after 10 failures")
                return problems
            continue
        parts = q.parts
        for p in parts:
            if not str(p.answer_text).strip() or str(p.answer_text).strip() in {"None", "nan"}:
                problems.append(f"seed {seed}: part {p.label} has no clean answer_text")
        seen_text = q.stem
        for j, later in enumerate(parts):
            target = str(later.answer_text).strip()
            if j > 0 and len(target) >= 3:
                haystack = seen_text
                if re.search(rf"(?<![\w.]){re.escape(target)}(?!\w|\.\d)", haystack):
                    problems.append(
                        f"seed {seed}: part {later.label}'s answer {target!r} appears in the stem or an earlier prompt"
                    )
            seen_text += " " + later.prompt
        finals.add(str(parts[-1].answer_text))
        shapes.add(_shape(q.text))
    needed = -(-100 * seeds // 300)
    if len(finals) < needed:
        problems.append(f"only {len(finals)} distinct last-part answers over {seeds} seeds (need {needed})")
    if len(shapes) < 3:
        problems.append(f"only {len(shapes)} distinct question shapes (need 3)")
    return problems[:25]
