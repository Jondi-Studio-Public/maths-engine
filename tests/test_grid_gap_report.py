"""scripts/grid_gap_report.py: the grid loader, evidence verification, recipe cells and the held-out paper check."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("grid_gap_report", ROOT / "scripts" / "grid_gap_report.py")
ggr = importlib.util.module_from_spec(_spec)
sys.modules["grid_gap_report"] = ggr
_spec.loader.exec_module(ggr)


def _write(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")


@pytest.fixture
def toy(tmp_path):
    """A two-family grid, one skill, no papers; tests add evidence and papers as they need."""
    grid = tmp_path / "grid"
    _write(
        grid / "grid.yaml",
        {
            "id": "toy",
            "title": "Toy",
            "source": {"name": "none"},
            "families": ["polynomial", "circular"],
            "modes": ["hand", "calc"],
            "operations": {"differentiate": "d", "symmetry": "s", "sketch": "x"},
            "ask_operations": {"deriv": "differentiate"},
            "shapes": {"quadratic_argument": "q", "linear_argument": "l"},
            "rows": [
                {
                    "id": "R1",
                    "unit": 3,
                    "area": "calculus",
                    "skill": "differentiate",
                    "status": "provisional",
                    "cells": [
                        {"families": ["polynomial", "circular"], "operations": ["differentiate"], "modes": ["hand"]},
                        {
                            "families": ["circular"],
                            "operations": ["differentiate"],
                            "modes": ["hand"],
                            "shapes": ["linear_argument", "quadratic_argument"],
                        },
                    ],
                },
                {
                    "id": "R2",
                    "unit": 3,
                    "area": "graphs",
                    "skill": "symmetry",
                    "status": "verified",
                    "cells": [{"families": ["circular"], "operations": ["symmetry"], "modes": ["hand", "calc"]}],
                },
                {
                    "id": "N1",
                    "unit": 3,
                    "area": "graphs",
                    "skill": "sketch",
                    "status": "provisional",
                    "auto_markable": False,
                    "not_auto_reason": "a drawing",
                },
            ],
        },
    )
    skills = {
        "diff-skill": {"status": "active", "tiers": {"exam", "easy"}, "technology": None, "answer": "expression"},
        "old-skill": {"status": "deferred", "tiers": {"exam"}, "technology": None, "answer": "expression"},
        "tech-skill": {"status": "active", "tiers": {"exam"}, "technology": "required", "answer": "number"},
    }
    return grid, skills


def _evidence(grid: Path, entries) -> None:
    _write(grid / "evidence" / "e.yaml", {"evidence": entries})


def _report(grid, skills, **kw):
    return ggr.build_report(ggr.load_grid(grid), skills, **kw)


def test_gaps_are_possible_cells_without_evidence(toy):
    grid, skills = toy
    _evidence(
        grid,
        [
            {
                "by": "skill:diff-skill",
                "tier": "exam",
                "families": ["polynomial"],
                "operations": ["differentiate"],
                "modes": ["hand"],
            }
        ],
    )
    rep = _report(grid, skills)
    assert rep.errors == []
    assert ("polynomial", "differentiate", "hand") in rep.covered
    assert [ggr.cell_id(c) for c in rep.gaps] == [
        "circular/differentiate/hand",
        "circular/symmetry/calc",
        "circular/symmetry/hand",
    ]
    assert rep.not_auto == {"N1": "a drawing"}


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"by": "skill:ghost", "families": ["polynomial"]}, "does not exist"),
        ({"by": "skill:old-skill", "families": ["polynomial"]}, "not active"),
        ({"by": "skill:diff-skill", "tier": "hard", "families": ["polynomial"]}, "no 'hard' source"),
        ({"by": "skill:tech-skill", "tier": "exam", "families": ["polynomial"]}, "technology: required"),
        (
            {"by": "skill:diff-skill", "families": ["polynomial"], "operations": ["symmetry"]},
            "no auto-markable row makes it",
        ),
        ({"by": "recipe:x", "families": ["polynomial"]}, "'by' must be skill"),
    ],
)
def test_bad_evidence_is_reported_not_counted(toy, entry, message):
    grid, skills = toy
    entry = {"operations": ["differentiate"], "modes": ["hand"], **entry}
    _evidence(grid, [entry])
    rep = _report(grid, skills)
    assert any(message in e for e in rep.errors), rep.errors
    assert rep.covered == {}


def test_names_outside_the_closed_lists_are_errors(toy):
    grid, skills = toy
    _evidence(
        grid, [{"by": "skill:diff-skill", "families": ["tangent"], "operations": ["differentiate"], "modes": ["hand"]}]
    )
    assert any("closed list" in e for e in _report(grid, skills).errors)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda g: g["rows"].append(dict(g["rows"][0])), "duplicate row id"),
        (lambda g: g["rows"][0].update(status="maybe"), "status must be"),
        (lambda g: g["rows"][0].pop("cells"), "needs cells"),
        (lambda g: g["rows"][2].pop("not_auto_reason"), "needs not_auto_reason"),
        (lambda g: g["rows"][0].update(surprise=1), "unknown keys"),
    ],
)
def test_malformed_rows_fail_loudly(toy, mutate, message):
    grid, skills = toy
    doc = yaml.safe_load((grid / "grid.yaml").read_text())
    mutate(doc)
    _write(grid / "grid.yaml", doc)
    with pytest.raises(ggr.GridError, match=message):
        _report(grid, skills)


def test_recipe_cells_use_subject_families_ask_operations_and_mode(toy):
    grid, skills = toy
    recipe = {"id": "r1", "subject": "s", "asks": ["deriv"], "technology_active": None}
    cells, problems = ggr.recipe_cells(recipe, {"s": ["polynomial", "circular"]}, {"deriv": "differentiate"})
    assert problems == [] and sorted(cells) == [
        ("circular", "differentiate", "hand"),
        ("polynomial", "differentiate", "hand"),
    ]
    cells, _ = ggr.recipe_cells(
        {**recipe, "technology_active": True}, {"s": ["polynomial"]}, {"deriv": "differentiate"}
    )
    assert cells == [("polynomial", "differentiate", "calc")]
    _, problems = ggr.recipe_cells({**recipe, "asks": ["mystery"]}, {"s": ["polynomial"]}, {})
    assert "no entry in the grid's ask_operations" in problems[0]
    _, problems = ggr.recipe_cells(recipe, {}, {"deriv": "differentiate"})
    assert "not registered" in problems[0]
    rep = _report(grid, skills, recipes=[recipe], subject_families={"s": ["polynomial", "circular"]})
    assert rep.errors == [] and ("circular", "differentiate", "hand") in rep.covered


def test_a_recipe_outside_the_possible_cells_is_an_error(toy):
    grid, skills = toy
    recipe = {"id": "r1", "subject": "s", "asks": ["deriv"], "technology_active": True}
    rep = _report(grid, skills, recipes=[recipe], subject_families={"s": ["polynomial"]})
    assert any("not a possible cell" in e for e in rep.errors)


def test_held_out_paper_verdicts(toy):
    grid, skills = toy
    _evidence(
        grid,
        [
            {
                "by": "skill:diff-skill",
                "tier": "exam",
                "families": ["polynomial"],
                "operations": ["differentiate"],
                "modes": ["hand"],
            }
        ],
    )
    spec = lambda f, o, m: [{"families": [f], "operations": [o], "modes": [m]}]  # noqa: E731
    _write(
        grid / "papers" / "p.yaml",
        {
            "id": "p",
            "title": "P",
            "parts": [
                {"q": 1, "part": "a", "cells": spec("polynomial", "differentiate", "hand")},
                {"q": 1, "part": "b", "cells": spec("circular", "differentiate", "hand")},
                {"q": 2, "part": "", "cells": spec("polynomial", "symmetry", "hand")},
                {"q": 3, "part": "", "not_auto": "proof"},
            ],
        },
    )
    verdicts = {p["part"]: p["verdict"] for p in _report(grid, skills).paper_results[0]["parts"]}
    assert verdicts == {"1a": "covered", "1b": "gap", "2": "grid-miss", "3": "not-auto"}


def test_the_shipped_grids_verify_against_the_real_skills():
    skills = ggr.load_skill_index(ROOT / "skills")
    grids = sorted(p for p in (ROOT / "grids").glob("*") if (p / "grid.yaml").is_file())
    assert grids
    for g in grids:
        rep = ggr.build_report(ggr.load_grid(g), skills)
        assert rep.errors == [], rep.errors
        misses = [p for pr in rep.paper_results for p in pr["parts"] if p["verdict"] in ("grid-miss", "error")]
        assert misses == [], misses


def test_cli_text_and_json(capsys):
    assert ggr.main(["--check"]) == 0
    assert "errors: []" in capsys.readouterr().out
    assert ggr.main(["--format", "json"]) == 0
    assert '"possible_cells"' in capsys.readouterr().out


def test_a_cell_covered_for_a_simple_shape_does_not_cover_the_exam_shape(toy):
    grid, skills = toy
    _evidence(
        grid,
        [
            {
                "by": "skill:diff-skill",
                "tier": "exam",
                "families": ["circular"],
                "operations": ["differentiate"],
                "modes": ["hand"],
                "shapes": ["linear_argument"],
            }
        ],
    )
    rep = _report(grid, skills)
    assert rep.errors == []
    assert [(ggr.cell_id(c), s) for c, s in rep.shape_gaps] == [("circular/differentiate/hand", "quadratic_argument")]
    need = lambda *sh: [  # noqa: E731
        {"families": ["circular"], "operations": ["differentiate"], "modes": ["hand"], "shapes": list(sh)}
    ]
    _write(
        grid / "papers" / "p.yaml",
        {
            "id": "p",
            "parts": [
                {"q": 1, "part": "a", "cells": need("linear_argument")},
                {"q": 1, "part": "b", "cells": need("quadratic_argument")},
                {"q": 1, "part": "c", "cells": need("has_pi_roots")},
            ],
        },
    )
    verdicts = [p["verdict"] for p in ggr.build_report(ggr.load_grid(grid), skills).paper_results[0]["parts"]]
    assert verdicts == ["covered", "shape-gap", "error"]


def test_evidence_claiming_a_shape_no_row_allows_is_an_error(toy):
    grid, skills = toy
    _evidence(
        grid,
        [
            {
                "by": "skill:diff-skill",
                "tier": "exam",
                "families": ["polynomial"],
                "operations": ["differentiate"],
                "modes": ["hand"],
                "shapes": ["quadratic_argument"],
            }
        ],
    )
    assert any("no row makes possible" in e for e in _report(grid, skills).errors)
