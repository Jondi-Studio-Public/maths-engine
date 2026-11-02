"""Coverage-grid gap report: which exam cells can the library produce, and which can it not?

A *grid* (see `grids/README.md`) crosses function family x operation x mode (hand or calculator) and lists the
cells a study design makes possible. *Evidence* files say which skill tier or recipe produces each cell. This
script loads both, checks every claim against the repository, and reports the possible cells with no evidence,
so a "gap" is a query rather than an audit. Past papers are held-out data: each paper part is mapped to cells,
and a part that maps to no possible cell means the grid itself is wrong.

Usage:
    python scripts/grid_gap_report.py                      # every grid under grids/, text report
    python scripts/grid_gap_report.py --grid grids/vce-methods-units-3-4 --format markdown
    python scripts/grid_gap_report.py --check              # exit 1 if any claim fails verification
    python scripts/grid_gap_report.py --format json

Read-only. It parses skill YAML directly (no generator is run), so it is fast, and it never writes a file
unless `--out` is given.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRIDS = REPO_ROOT / "grids"
DEFAULT_SKILLS = REPO_ROOT / "skills"
DEFAULT_RECIPES = REPO_ROOT / "recipes"

Cell = tuple[str, str, str]  # (family, operation, mode)
ROW_STATUSES = {"provisional", "verified"}
ROW_KEYS = {"id", "unit", "area", "skill", "status", "auto_markable", "not_auto_reason", "cells"}
EXPANSION_KEYS = {"families", "operations", "modes"}
SPEC_KEYS = EXPANSION_KEYS | {"shapes"}  # `shapes` is optional


class GridError(ValueError):
    """The grid data itself is malformed."""


def cell_id(cell: Cell) -> str:
    return "/".join(cell)


def parse_cell_id(text: str) -> Cell:
    parts = tuple(str(text).split("/"))
    if len(parts) != 3:
        raise GridError(f"cell {text!r} must be family/operation/mode")
    return parts  # type: ignore[return-value]


@dataclass
class Grid:
    id: str
    title: str
    source: dict
    families: list[str]
    operations: dict[str, str]
    modes: list[str]
    ask_operations: dict[str, str]
    rows: list[dict]
    shapes: dict[str, str] = field(default_factory=dict)
    evidence: list[dict] = field(default_factory=list)
    papers: list[dict] = field(default_factory=list)
    path: Path | None = None

    def expand(self, spec: dict, where: str) -> list[Cell]:
        """Cross product of a {families, operations, modes} spec; each name must be in the closed lists."""
        if not isinstance(spec, dict) or set(spec) - SPEC_KEYS or EXPANSION_KEYS - set(spec):
            raise GridError(
                f"{where}: a cell spec needs families, operations and modes (and optionally shapes), got {spec!r}"
            )
        self.shapes_of(spec, where)
        for key, allowed in (("families", self.families), ("operations", self.operations), ("modes", self.modes)):
            names = spec[key]
            if not isinstance(names, list) or not names:
                raise GridError(f"{where}: {key} must be a non-empty list")
            bad = [n for n in names if n not in allowed]
            if bad:
                raise GridError(f"{where}: {key} {bad} are not in the grid's closed list {sorted(allowed)}")
        return list(itertools.product(spec["families"], spec["operations"], spec["modes"]))

    def shapes_of(self, spec: dict, where: str) -> frozenset[str]:
        """The `shapes` of a spec: finer structure of the cell (a quadratic argument, two unknown constants ...)."""
        names = spec.get("shapes") or []
        if not isinstance(names, list):
            raise GridError(f"{where}: shapes must be a list")
        bad = [n for n in names if n not in self.shapes]
        if bad:
            raise GridError(f"{where}: shapes {bad} are not in the grid's closed list {sorted(self.shapes)}")
        return frozenset(names)


def _load_yaml(path: Path):
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise GridError(f"{path}: not valid YAML: {e}") from e


def load_grid(directory: Path) -> Grid:
    """Read `<directory>/grid.yaml`, every `evidence/*.yaml` and every `papers/*.yaml`."""
    main = directory / "grid.yaml"
    if not main.is_file():
        raise GridError(f"{directory}: no grid.yaml")
    raw = _load_yaml(main)
    for key in ("id", "title", "source", "families", "operations", "modes", "rows"):
        if key not in raw:
            raise GridError(f"{main}: missing {key!r}")
    grid = Grid(
        id=raw["id"],
        title=raw["title"],
        source=raw["source"],
        families=list(raw["families"]),
        operations=dict(raw["operations"]),
        modes=list(raw["modes"]),
        ask_operations=dict(raw.get("ask_operations") or {}),
        shapes=dict(raw.get("shapes") or {}),
        rows=list(raw["rows"]),
        path=directory,
    )
    for p in sorted((directory / "evidence").glob("*.yaml")):
        doc = _load_yaml(p) or {}
        for entry in doc.get("evidence", []):
            entry = dict(entry)
            entry["_file"] = p.name
            grid.evidence.append(entry)
    for p in sorted((directory / "papers").glob("*.yaml")):
        doc = _load_yaml(p) or {}
        doc["_file"] = p.name
        grid.papers.append(doc)
    return grid


def possible_cells(grid: Grid) -> tuple[dict[Cell, list[str]], dict[str, str]]:
    """(cell -> ids of the rows that make it possible, row id -> reason for rows that are not auto-markable)."""
    possible, not_auto, _ = possible_with_shapes(grid)
    return possible, not_auto


def possible_with_shapes(grid: Grid) -> tuple[dict[Cell, list[str]], dict[str, str], dict[Cell, set[str]]]:
    """`possible_cells` plus, per cell, the shapes the rows make possible for it."""
    possible: dict[Cell, list[str]] = defaultdict(list)
    shapes: dict[Cell, set[str]] = defaultdict(set)
    not_auto: dict[str, str] = {}
    seen_ids: set[str] = set()
    for row in grid.rows:
        rid = row.get("id", "?")
        extra = set(row) - ROW_KEYS
        if extra:
            raise GridError(f"row {rid}: unknown keys {sorted(extra)}")
        for key in ("id", "unit", "area", "skill", "status"):
            if key not in row:
                raise GridError(f"row {rid}: missing {key!r}")
        if rid in seen_ids:
            raise GridError(f"duplicate row id {rid}")
        seen_ids.add(rid)
        if row["status"] not in ROW_STATUSES:
            raise GridError(f"row {rid}: status must be one of {sorted(ROW_STATUSES)}")
        if not row.get("auto_markable", True):
            if not row.get("not_auto_reason"):
                raise GridError(f"row {rid}: a not-auto-markable row needs not_auto_reason")
            not_auto[rid] = row["not_auto_reason"]
            continue
        specs = row.get("cells")
        if not specs:
            raise GridError(f"row {rid}: an auto-markable row needs cells")
        for i, spec in enumerate(specs):
            for cell in grid.expand(spec, f"row {rid} cells[{i}]"):
                if rid not in possible[cell]:
                    possible[cell].append(rid)
                shapes[cell] |= grid.shapes_of(spec, f"row {rid} cells[{i}]")
    return dict(possible), not_auto, dict(shapes)


# ---------------------------------------------------------------------------------------------------------
# Evidence


def load_skill_index(skills_dir: Path) -> dict[str, dict]:
    """id -> {status, tiers, technology, path} read straight from the skill YAML."""
    index: dict[str, dict] = {}
    for p in sorted(skills_dir.glob("*/*.yaml")):
        if p.name in {"domains.yaml", "capability-gates.yaml", "index.yaml", "skill.schema.yaml"}:
            continue
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if not isinstance(raw, dict) or "id" not in raw:
            continue
        index[raw["id"]] = {
            "status": raw.get("status"),
            "tiers": {s.get("difficulty") for s in raw.get("sources", []) or [] if isinstance(s, dict)},
            "technology": raw.get("technology"),  # the technology-flag change adds this field; absent today
            "answer": raw.get("answer"),
        }
    return index


def recipe_cells(recipe: dict, subject_families: dict[str, list[str]], ask_operations: dict[str, str]):
    """Cells one recipe produces: subject families x the operation each ask maps to, in the recipe's mode.

    Returns (cells, problems). A recipe ask with no entry in `ask_operations` is a problem, never silently
    ignored, so a new ask forces a decision about which operation it counts as.
    """
    problems: list[str] = []
    fams = subject_families.get(recipe["subject"])
    if fams is None:
        return [], [f"recipe {recipe['id']}: subject {recipe['subject']!r} is not registered"]
    mode = "calc" if recipe.get("technology_active") is True else "hand"
    cells: list[Cell] = []
    for ask_name in recipe["asks"]:
        op = ask_operations.get(ask_name)
        if op is None:
            problems.append(f"recipe {recipe['id']}: ask {ask_name!r} has no entry in the grid's ask_operations")
            continue
        cells.extend((f, op, mode) for f in fams)
    return cells, problems


def load_recipes_for_grid(
    recipes_dir: Path,
) -> tuple[list[dict], dict[str, list[str]], dict[str, list[str]], str | None]:
    """Recipe dicts, subject -> families, subject -> properties, and a note if recipes cannot be read here."""
    if not recipes_dir.is_dir() or not list(recipes_dir.rglob("*.yaml")):
        return [], {}, {}, None
    try:
        from mathgen import recipes as rmod  # noqa: PLC0415 -- optional: only present once the composer ships
    except ImportError:
        return [], {}, {}, "recipes/ has files but mathgen.recipes is not importable; recipe evidence skipped"
    import mathgen.generators  # noqa: F401, PLC0415 -- subjects register when the generators load

    out = []
    for r in rmod.load_recipes(recipes_dir):
        out.append(
            {"id": r.id, "subject": r.subject, "asks": [a for a, _ in r.asks], "technology_active": r.technology_active}
        )
    fams = {name: sorted(cls.families) for name, cls in rmod.SUBJECTS.items()}
    props = {name: sorted(getattr(cls, "properties", ())) for name, cls in rmod.SUBJECTS.items()}
    return out, fams, props, None


@dataclass
class Report:
    grid: Grid
    possible: dict[Cell, list[str]]
    not_auto: dict[str, str]
    covered: dict[Cell, list[str]]  # cell -> who produces it
    errors: list[str]
    notes: list[str]
    paper_results: list[dict]
    possible_shapes: dict[Cell, set[str]] = field(default_factory=dict)
    provided: dict[Cell, list[tuple[str, frozenset[str]]]] = field(default_factory=dict)

    @property
    def gaps(self) -> list[Cell]:
        return sorted(c for c in self.possible if c not in self.covered)

    @property
    def shape_gaps(self) -> list[tuple[Cell, str]]:
        """(cell, shape) pairs a row makes possible for a cell that has evidence, but no evidence provides."""
        out = []
        for cell, shapes in sorted(self.possible_shapes.items()):
            if cell not in self.covered:
                continue
            have = set().union(*(s for _, s in self.provided.get(cell, [])))
            out.extend((cell, sh) for sh in sorted(shapes - have))
        return out


def build_report(
    grid: Grid,
    skills: dict[str, dict],
    recipes: list[dict] | None = None,
    subject_families: dict[str, list[str]] | None = None,
    subject_properties: dict[str, list[str]] | None = None,
) -> Report:
    possible, not_auto, possible_shapes = possible_with_shapes(grid)
    covered: dict[Cell, list[str]] = defaultdict(list)
    provided: dict[Cell, list[tuple[str, frozenset[str]]]] = defaultdict(list)
    errors: list[str] = []
    notes: list[str] = []

    for n, entry in enumerate(grid.evidence):
        where = f"{entry.get('_file', '?')} evidence[{n}]"
        by = str(entry.get("by", ""))
        try:
            spec = {k: entry[k] for k in SPEC_KEYS if k in entry}
            cells = grid.expand(spec, where)
            claimed = grid.shapes_of(spec, where)
        except GridError as e:
            errors.append(str(e))
            continue
        kind, _, ref = by.partition(":")
        if kind != "skill" or not ref:
            errors.append(
                f"{where}: 'by' must be skill:<id> (recipes are read from recipes/ automatically), got {by!r}"
            )
            continue
        info = skills.get(ref)
        if info is None:
            errors.append(f"{where}: skill {ref!r} does not exist")
            continue
        if info["status"] != "active":
            errors.append(f"{where}: skill {ref!r} is {info['status']}, not active")
            continue
        tier = entry.get("tier")
        if tier is not None and tier not in info["tiers"]:
            errors.append(
                f"{where}: skill {ref!r} has no {tier!r} source (has {sorted(t for t in info['tiers'] if t)})"
            )
            continue
        for cell in cells:
            if cell[2] == "hand" and info.get("technology") == "required":
                errors.append(f"{where}: {cell_id(cell)} is hand but skill {ref!r} is technology: required")
                continue
            if cell not in possible:
                errors.append(
                    f"{where}: {cell_id(cell)} is produced by {ref!r} but no auto-markable row makes it "
                    "possible (the grid is missing a row, or the claim is wrong)"
                )
                continue
            stray = claimed - possible_shapes.get(cell, set())
            if stray:
                errors.append(f"{where}: {cell_id(cell)} claims shapes {sorted(stray)} that no row makes possible")
                continue
            who = f"{by}" + (f"@{tier}" if tier else "")
            covered[cell].append(who)
            provided[cell].append((who, claimed))

    if recipes:
        for r in recipes:
            cells, problems = recipe_cells(r, subject_families or {}, grid.ask_operations)
            errors.extend(problems)
            for cell in cells:
                if cell not in possible:
                    errors.append(f"recipe {r['id']}: {cell_id(cell)} is not a possible cell of grid {grid.id}")
                else:
                    covered[cell].append(f"recipe:{r['id']}")
                    provided[cell].append(
                        (f"recipe:{r['id']}", frozenset((subject_properties or {}).get(r["subject"], ())))
                    )
    for op in grid.ask_operations.values():
        if op not in grid.operations:
            errors.append(f"ask_operations maps to unknown operation {op!r}")

    paper_results = [check_paper(grid, paper, possible, covered, possible_shapes, provided) for paper in grid.papers]
    return Report(
        grid, possible, not_auto, dict(covered), errors, notes, paper_results, possible_shapes, dict(provided)
    )


def check_paper(
    grid: Grid,
    paper: dict,
    possible: dict[Cell, list[str]],
    covered: dict[Cell, list[str]],
    possible_shapes: dict[Cell, set[str]],
    provided: dict[Cell, list[tuple[str, frozenset[str]]]],
) -> dict:
    """Map each part of a held-out paper to cells. Verdicts per part:

    * `not-auto` -- the part is a sketch, proof or written reason;
    * `grid-miss` -- some cell the part needs is not possible in the grid (the grid is wrong or incomplete);
    * `gap` -- every cell is possible but at least one has no evidence;
    * `shape-gap` -- every cell has evidence, but no single piece of evidence provides the shapes the part needs
      (the cell is covered for a simpler shape of the same family and operation);
    * `covered` -- every cell has evidence with the needed shapes.

    A part's `shapes` are the finer structure it needs (a quadratic argument, two unknown constants ...). This is
    what stops a linear-argument log skill from counting as the logarithm of a quadratic.
    """
    parts = []
    for part in paper.get("parts", []):
        pid = f"{part.get('q')}{part.get('part', '')}"
        if part.get("not_auto"):
            parts.append({"part": pid, "marks": part.get("marks"), "verdict": "not-auto", "detail": part["not_auto"]})
            continue
        try:
            needs: list[tuple[Cell, frozenset[str]]] = []
            for spec in part["cells"]:
                where = f"paper {paper.get('_file')} part {pid}"
                sh = grid.shapes_of(spec, where)
                needs.extend((c, sh) for c in grid.expand(spec, where))
            cells = [c for c, _ in needs]
        except (GridError, KeyError) as e:
            parts.append({"part": pid, "marks": part.get("marks"), "verdict": "error", "detail": str(e)})
            continue
        missing = [cell_id(c) for c, sh in needs if c not in possible or not sh <= possible_shapes.get(c, set())]
        uncovered = [c for c in cells if c in possible and c not in covered]
        thin = [
            f"{cell_id(c)} + {sorted(sh)}"
            for c, sh in needs
            if c in covered and not any(sh <= have for _, have in provided.get(c, []))
        ]
        if missing:
            verdict, detail = "grid-miss", missing
        elif uncovered:
            verdict, detail = "gap", [cell_id(c) for c in uncovered]
        elif thin:
            verdict, detail = "shape-gap", thin
        else:
            verdict, detail = "covered", []
        parts.append(
            {"part": pid, "marks": part.get("marks"), "what": part.get("what"), "verdict": verdict, "detail": detail}
        )
    return {"paper": paper.get("id", paper.get("_file")), "title": paper.get("title"), "parts": parts}


# ---------------------------------------------------------------------------------------------------------
# Output


def summarise(report: Report) -> dict:
    by_op: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for cell in report.possible:
        by_op[cell[1]][1] += 1
        if cell in report.covered:
            by_op[cell[1]][0] += 1
    return {
        "grid": report.grid.id,
        "possible_cells": len(report.possible),
        "covered_cells": sum(1 for c in report.possible if c in report.covered),
        "gap_cells": len(report.gaps),
        "shape_gaps": len(report.shape_gaps),
        "provisional_rows": sum(1 for r in report.grid.rows if r["status"] == "provisional"),
        "rows": len(report.grid.rows),
        "not_auto_rows": len(report.not_auto),
        "by_operation": {op: {"covered": c, "possible": t} for op, (c, t) in sorted(by_op.items())},
        "errors": report.errors,
    }


def render_text(report: Report, markdown: bool = False) -> str:
    s = summarise(report)
    h = (lambda t: f"## {t}") if markdown else (lambda t: f"\n== {t} ==")
    lines = [f"# {report.grid.title}" if markdown else f"{report.grid.title}"]
    lines.append(
        f"{s['covered_cells']} of {s['possible_cells']} possible cells covered; {s['gap_cells']} gaps, {s['shape_gaps']} shape gaps (a covered cell lacking a finer shape). "
        f"{s['rows']} study-design rows ({s['provisional_rows']} provisional), {s['not_auto_rows']} not auto-markable."
    )
    lines.append(h("Coverage by operation"))
    for op, v in s["by_operation"].items():
        lines.append(f"{'- ' if markdown else '  '}{op}: {v['covered']}/{v['possible']}")
    lines.append(h("Shape gaps (covered cells missing a finer structure)"))
    sg: dict[str, list[str]] = defaultdict(list)
    for cell, shape in report.shape_gaps:
        sg[cell_id(cell)].append(shape)
    for cid, shapes in sg.items():
        lines.append(f"{'- ' if markdown else '  '}{cid}: {', '.join(shapes)}")
    if not sg:
        lines.append("none")
    lines.append(h("Gaps (possible cells with no skill or recipe)"))
    by_op: dict[str, list[Cell]] = defaultdict(list)
    for cell in report.gaps:
        by_op[cell[1]].append(cell)
    for op, cells in sorted(by_op.items()):
        lines.append(f"{'- ' if markdown else '  '}{op}: " + ", ".join(f"{f}/{m}" for f, _, m in cells))
    if not by_op:
        lines.append("none")
    for pr in report.paper_results:
        lines.append(h(f"Held-out paper: {pr['title'] or pr['paper']}"))
        counts: dict[str, int] = defaultdict(int)
        for part in pr["parts"]:
            counts[part["verdict"]] += 1
            if part["verdict"] not in ("covered",):
                lines.append(
                    f"{'- ' if markdown else '  '}{part['part']} [{part['verdict']}] {part.get('detail') or ''}"
                )
        lines.append(f"verdicts: {dict(sorted(counts.items()))}")
    lines.append(h("Errors"))
    lines.extend(report.errors or ["errors: []"])
    for n in report.notes:
        lines.append(f"note: {n}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=Path, action="append", help="a grid directory (default: every one under grids/)")
    ap.add_argument("--skills", type=Path, default=DEFAULT_SKILLS)
    ap.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    ap.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    ap.add_argument("--check", action="store_true", help="exit 1 if any evidence claim or recipe fails verification")
    ap.add_argument("--fail-on-gaps", action="store_true", help="exit 1 if any possible cell is uncovered")
    ap.add_argument("--out", type=Path, help="write the report here instead of stdout")
    args = ap.parse_args(argv)

    grids = args.grid or sorted(p for p in DEFAULT_GRIDS.glob("*") if (p / "grid.yaml").is_file())
    if not grids:
        print("no grids found", file=sys.stderr)
        return 2
    skills = load_skill_index(args.skills)
    recipes, subj_fams, subj_props, note = load_recipes_for_grid(args.recipes)
    reports = []
    for g in grids:
        rep = build_report(load_grid(g), skills, recipes, subj_fams, subj_props)
        if note:
            rep.notes.append(note)
        reports.append(rep)

    if args.format == "json":
        text = json.dumps(
            [
                {
                    **summarise(r),
                    "gaps": [cell_id(c) for c in r.gaps],
                    "shape_gaps": [f"{cell_id(c)} + {sh}" for c, sh in r.shape_gaps],
                    "papers": r.paper_results,
                }
                for r in reports
            ],
            indent=2,
        )
    else:
        text = "\n".join(render_text(r, args.format == "markdown") for r in reports)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    failed = any(r.errors for r in reports) and args.check
    if args.fail_on_gaps and any(r.gaps for r in reports):
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
