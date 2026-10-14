"""mathgen.figures: the figure spec envelope, the cartesian and number_line
kinds, and their path through mathgen.make and the
skill validator."""

from __future__ import annotations

import pytest

import mathgen
from mathgen import core, figures
from mathgen import skills as skills_mod


def _line_plane(**kw):
    base = dict(
        x_range=(-5, 5), y_range=(-5, 5), alt="a straight line on a grid", lines=[{"through": [[0, 1], [1, 3]]}]
    )
    base.update(kw)
    return figures.cartesian(**base)


# ---------------------------------------------------------------- envelope


def test_known_kinds():
    assert {"cartesian", "number_line"} <= set(figures.kinds())


def test_unknown_kind_and_non_dict_are_rejected():
    assert figures.validate("not a dict") == ["figure must be a dict"]
    assert "unknown figure kind" in figures.validate({"kind": "doodle", "alt": "x"})[0]


def test_alt_text_is_required():
    spec = _line_plane()
    spec["alt"] = "  "
    assert "figure needs non-empty alt text" in figures.validate(spec)


def test_register_kind_twice_is_an_error():
    with pytest.raises(ValueError):
        figures.register_kind("cartesian", lambda spec: [])


# ---------------------------------------------------------------- cartesian


def test_cartesian_valid_spec_round_trips():
    spec = _line_plane(
        points=[{"at": [2, 5], "label": "A"}], segments=[{"from": [-2, 0], "to": [0, 1], "open_end": True}]
    )
    assert figures.validate(spec) == []
    assert spec["kind"] == "cartesian" and spec["x_range"] == [-5, 5]


@pytest.mark.parametrize(
    "kw, fragment",
    [
        ({"x_range": (3, 3)}, "x_range"),
        ({"grid": 0}, "grid must be a positive number"),
        ({"points": [{"at": [9, 0]}]}, "outside the plane"),
        ({"segments": [{"from": [0, 0], "to": [0, 9]}]}, "leaves the plane"),
        ({"lines": [{"through": [[1, 1], [1, 1]]}]}, "points must differ"),
        ({"x_range": (-100, 100)}, "more than 60 grid steps"),
    ],
)
def test_cartesian_rejects_bad_specs(kw, fragment):
    with pytest.raises(ValueError, match=fragment):
        _line_plane(**kw)


def test_sample_curve_breaks_at_an_asymptote_and_clips():
    pieces = figures.sample_curve(lambda x: 1 / x, -5, 5, (-5, 5), n=201)
    assert len(pieces) == 2
    for piece in pieces:
        assert all(-5.25 <= y <= 5.25 for _, y in piece)
    assert all(x < 0 for x, _ in pieces[0]) and all(x > 0 for x, _ in pieces[1])


def test_sample_curve_is_capped():
    pieces = figures.sample_curve(lambda x: x, 0, 1, (0, 1), n=10_000)
    assert sum(len(p) for p in pieces) <= figures.MAX_CURVE_POINTS


def test_sample_curve_is_deterministic():
    f = lambda x: x * x - 2  # noqa: E731
    assert figures.sample_curve(f, -3, 3, (-3, 8)) == figures.sample_curve(f, -3, 3, (-3, 8))


# ---------------------------------------------------------------- number line


def test_number_line_valid_and_invalid():
    spec = figures.number_line(
        range_=(-5, 5), alt="a number line with a ray", rays=[{"from": 2, "direction": "right", "open": True}]
    )
    assert figures.validate(spec) == []
    with pytest.raises(ValueError, match="direction"):
        figures.number_line(range_=(-5, 5), alt="x", rays=[{"from": 2, "direction": "up"}])
    with pytest.raises(ValueError, match="intervals"):
        figures.number_line(range_=(-5, 5), alt="x", intervals=[{"from": 3, "to": 1}])


# ---------------------------------------------------------------- leak check


def test_assert_no_leak_checks_labels_and_alt_text():
    spec = _line_plane(points=[{"at": [2, 5], "label": "(2, 5)"}])
    with pytest.raises(AssertionError):
        figures.assert_no_leak(spec, ["(2,5)"])
    leaky_alt = _line_plane(alt="the line y = 2x + 1")
    with pytest.raises(AssertionError):
        figures.assert_no_leak(leaky_alt, ["y = 2x + 1"])
    figures.assert_no_leak(_line_plane(), ["y = 2x + 1", "3"])


# ---------------------------------------------------------------- through make()


@pytest.fixture
def figure_topic(monkeypatch):
    """A throwaway topic that draws a figure, registered only for this test."""

    def gen(rng, bad=False):
        spec = _line_plane()
        if bad:
            spec = dict(spec, grid=-1)
        return {
            "text": "What is the gradient of the line?",
            "latex": "",
            "answer": 2,
            "answer_text": "2",
            "figure": spec,
        }

    monkeypatch.setitem(core.REGISTRY, "zz-figure-probe", (1, gen))
    return "zz-figure-probe"


def test_make_carries_a_valid_figure_and_to_dict_keeps_it(figure_topic):
    q = mathgen.make(figure_topic, seed=1)
    assert q.figure["kind"] == "cartesian"
    assert q.to_dict()["figure"] == q.figure


def test_make_rejects_an_invalid_figure(figure_topic):
    with pytest.raises(ValueError, match="invalid figure"):
        mathgen.make(figure_topic, seed=1, bad=True)


def test_questions_without_a_figure_are_unchanged():
    q = mathgen.make("addition", seed=1)
    assert q.figure is None
    assert "figure" not in q.to_dict()


# ---------------------------------------------------------------- skill validator


def _skill(tmp_path, body: str):
    domain = tmp_path / "algebra"
    domain.mkdir()
    (domain / "fig.yaml").write_text(
        "id: fig\nname: Fig\ndomain: algebra\nstatus: active\nprerequisites: []\nanswer: number\n" + body,
        encoding="utf-8",
    )
    domains, skills = skills_mod.load_skills(tmp_path)
    return skills_mod.validate(domains, skills)


def test_figure_source_needs_the_flag(tmp_path, figure_topic):
    report = _skill(tmp_path, f"sources:\n  - {{topic: {figure_topic}, params: {{}}}}\n")
    assert any("must declare needs_figure" in e for e in report.errors)


def test_flagged_figure_skill_validates(tmp_path, figure_topic):
    report = _skill(tmp_path, f"needs_figure: true\nsources:\n  - {{topic: {figure_topic}, params: {{}}}}\n")
    assert not any("needs_figure" in e for e in report.errors)


def test_figure_skill_may_not_have_a_mental_block(tmp_path, figure_topic):
    report = _skill(
        tmp_path,
        (
            f"needs_figure: true\nsources:\n  - {{topic: {figure_topic}, params: {{}}}}\n"
            f"mental:\n  levels:\n    - {{topic: {figure_topic}, params: {{}}, target_s: 5, limit_s: 10}}\n"
        ),
    )
    assert any("may not have a mental block" in e for e in report.errors)


# ---------------------------------------------------------------- cartesian: per-axis grid,
# equal_scale, x_tick_format (new kwargs; backward compatible with the defaults above)


def test_grid_defaults_x_grid_and_y_grid_to_grid():
    spec = _line_plane(grid=2)
    assert spec["grid"] == 2
    assert spec["x_grid"] == 2
    assert spec["y_grid"] == 2


def test_omitting_the_new_kwargs_renders_the_same_defaults_as_before():
    spec = _line_plane()
    assert spec["x_grid"] == spec["y_grid"] == spec["grid"] == 1
    assert spec["equal_scale"] is True
    assert spec["x_tick_format"] == "number"


def test_x_grid_and_y_grid_can_differ():
    spec = _line_plane(x_grid=0.5, y_grid=5)
    assert figures.validate(spec) == []
    assert spec["x_grid"] == 0.5
    assert spec["y_grid"] == 5


def test_sixty_step_limit_applies_per_axis_independently():
    # x_grid is fine on its own (12 steps); y_grid=0.1 over a 10-unit range
    # is 100 steps, over the limit -- only the y-axis should be flagged.
    with pytest.raises(ValueError, match="y-axis"):
        _line_plane(x_grid=1, y_grid=0.1)
    # The reverse: a fine x_grid over a wide x_range is flagged on the x-axis.
    with pytest.raises(ValueError, match="x-axis"):
        _line_plane(x_range=(-30, 30), x_grid=0.5, y_grid=1)


def test_equal_scale_false_is_accepted_and_recorded():
    spec = _line_plane(equal_scale=False)
    assert figures.validate(spec) == []
    assert spec["equal_scale"] is False


def test_equal_scale_must_be_a_bool():
    spec = _line_plane()
    spec["equal_scale"] = "false"
    assert "equal_scale must be a bool" in figures.validate(spec)


def test_x_tick_format_pi_is_accepted_and_recorded():
    spec = _line_plane(x_tick_format="pi")
    assert figures.validate(spec) == []
    assert spec["x_tick_format"] == "pi"


def test_x_tick_format_rejects_an_unknown_value():
    spec = _line_plane()
    spec["x_tick_format"] = "degrees"
    problems = figures.validate(spec)
    assert "x_tick_format must be 'number' or 'pi'" in problems
