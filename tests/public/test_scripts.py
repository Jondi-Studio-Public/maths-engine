"""The maintainer scripts under scripts/: bad arguments are usage errors, not tracebacks or a false "all clear"."""

import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import regrade_sweep

ROOT = Path(__file__).resolve().parents[2]
TUPLE_SKILL = "senior-understand-the-unit-circle-definition-of-cos-theta"  # answers like (sqrt(2)/2, sqrt(2)/2)


def run_script(name, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=300,
    )


# -- regrade_sweep: a selection that matches nothing is an error, not "0 false rejects" ---------------------------


def test_an_unknown_skill_id_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as stopped:
        regrade_sweep.main(["--ids", "no-such-skill"])
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    assert "unknown skill id" in err and "no-such-skill" in err


def test_one_unknown_id_among_real_ones_is_still_an_error(capsys):
    with pytest.raises(SystemExit) as stopped:
        regrade_sweep.main(["--ids", TUPLE_SKILL, "no-such-skill", "--seeds", "1"])
    assert stopped.value.code == 2
    assert "no-such-skill" in capsys.readouterr().err


def test_a_domain_with_no_skills_is_a_usage_error_that_names_the_real_domains(capsys):
    with pytest.raises(SystemExit) as stopped:
        regrade_sweep.main(["--domain", "no-such-domain"])
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    assert "no skills in domain 'no-such-domain'" in err and "trigonometry" in err


@pytest.mark.parametrize("seeds", ["0", "-3"])
def test_regrade_sweep_rejects_seeds_below_one(seeds, capsys):
    with pytest.raises(SystemExit) as stopped:
        regrade_sweep.main(["--ids", TUPLE_SKILL, "--seeds", seeds])
    assert stopped.value.code == 2
    assert "--seeds" in capsys.readouterr().err


def test_a_skill_with_no_grader_is_named_and_a_sweep_of_only_those_fails(capsys):
    """Sweeping nothing must not look like a pass."""
    assert regrade_sweep.main(["--ids", "one-step-inequalities", "--seeds", "1"]) == 1
    captured = capsys.readouterr()
    assert "skipped one-step-inequalities: no grader for answer kind 'inequality'" in captured.err
    assert "SWEEP: 0 skills" in captured.out


def test_no_new_skills_since_a_ref_is_still_a_clean_pass(monkeypatch, capsys):
    monkeypatch.setattr(regrade_sweep, "_added_skill_ids_since", lambda ref: set())
    assert regrade_sweep.main(["--since", "HEAD"]) == 0
    assert "SWEEP: 0 skills" in capsys.readouterr().out


def test_a_sweep_that_runs_exits_zero(capsys):
    assert regrade_sweep.main(["--ids", TUPLE_SKILL, "--seeds", "2"]) == 0
    assert f"OK {TUPLE_SKILL}" in capsys.readouterr().out


# -- regrade_sweep --since: no git, or a ref git does not know, is a message ---------------------------------------


def test_since_without_git_is_a_clean_error(monkeypatch, capsys):
    def no_git(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.setattr(regrade_sweep.subprocess, "run", no_git)
    with pytest.raises(SystemExit) as stopped:
        regrade_sweep.main(["--since", "main"])
    assert stopped.value.code == 2
    assert "--since needs git" in capsys.readouterr().err


def test_since_a_ref_git_does_not_know_is_a_clean_error(monkeypatch, capsys):
    def bad_ref(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0], stderr="fatal: bad revision 'nope..HEAD'\n")

    monkeypatch.setattr(regrade_sweep.subprocess, "run", bad_ref)
    with pytest.raises(SystemExit) as stopped:
        regrade_sweep.main(["--since", "nope"])
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    assert "--since 'nope'" in err and "bad revision" in err and "Traceback" not in err


# -- regrade_sweep: the wrong answer it makes for a tuple is a tuple ----------------------------------------------


@pytest.mark.parametrize(
    ("text", "parts"),
    [
        ("(a, b)", ["a", "b"]),
        ("(sqrt(2)/2, sqrt(2)/2)", ["sqrt(2)/2", "sqrt(2)/2"]),
        ("(a, (b, c))", ["a", "(b, c)"]),
        ("(f(a, b), c, d)", ["f(a, b)", "c", "d"]),
        ("  (1, 2)  ", ["1", "2"]),
        ("(x)", None),
        ("(x - 3)(x - 9)", None),
        ("(1, 2) + (3, 4)", None),
        ("(a)(b, c)", None),
        ("x + 1", None),
        ("f(a, b)", None),
        ("", None),
    ],
)
def test_tuple_components_splits_a_tuple_and_nothing_else(text, parts):
    assert regrade_sweep._tuple_components(text) == parts


def test_the_wrong_answer_for_a_tuple_changes_one_component_instead_of_adding_to_the_tuple():
    skill = SimpleNamespace(answer="expression")
    question = SimpleNamespace(answer_text="(sqrt(2)/2, sqrt(2)/2)")
    assert regrade_sweep._synthetic_wrong_given(skill, question) == "((sqrt(2)/2)+1, sqrt(2)/2)"


def test_the_wrong_answer_for_a_plain_expression_is_unchanged():
    skill = SimpleNamespace(answer="expression")
    for answer_text in ("x^2 + 3x", "(x - 3)(x - 9)", "(x + 1)"):
        assert regrade_sweep._synthetic_wrong_given(skill, SimpleNamespace(answer_text=answer_text)) == (
            f"{answer_text}+1"
        )


def test_sweeping_a_tuple_answer_raises_no_sympy_deprecation_warning():
    skill = {s.id: s for s in regrade_sweep.load_skills()[1]}[TUPLE_SKILL]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = regrade_sweep.sweep_skill(skill, seeds=5)
    assert result.questions > 0 and result.clean
    assert [str(w.message) for w in caught if "non-Expr" in str(w.message)] == []


# -- the other scripts --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seeds", ["0", "-1", "x"])
def test_the_speed_check_rejects_a_seed_count_it_cannot_take_a_median_of(seeds):
    done = run_script("check_generator_speed.py", "--seeds", seeds)
    assert done.returncode == 2
    assert "--seeds" in done.stderr and "Traceback" not in done.stderr


def test_skill_contract_says_it_is_a_library_when_run_as_a_command():
    done = run_script("skill_contract.py", "--help")
    assert done.returncode != 0
    assert "not a command" in done.stderr and "Traceback" not in done.stderr
