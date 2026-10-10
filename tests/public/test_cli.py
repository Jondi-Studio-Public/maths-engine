"""The `mathgen` command line: what the README promises, and clear errors instead of tracebacks."""

import errno
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

import mathgen
from mathgen import engine
from mathgen.__main__ import _choose_option, _labels, main
from mathgen.core import PRESETS, REGISTRY, mcq_options
from mathgen.grading import GRADERS


@pytest.fixture(scope="module")
def library():
    return engine.load_skills()[1]


def run(capsys, *argv):
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def run_error(capsys, *argv):
    """A call argparse rejects (usage error): the exit status and stderr."""
    with pytest.raises(SystemExit) as exc:
        main(list(argv))
    return exc.value.code, capsys.readouterr().err


# -- the README quick start ---------------------------------------------------------------------


def test_skill_question_and_answer(capsys):
    code, out, err = run(capsys, "--skill", "simplify-surds", "--seed", "7")
    assert code == 0
    assert out.splitlines() == [
        "[simplify-surds #7]",
        "  Q: Simplify √756. Give your answer in the form a√b.",
        "  A: 6*sqrt(21)",
    ]


def test_readme_grading_example_is_graded_as_documented(capsys):
    _, out, _ = run(capsys, "--skill", "simplify-surds", "--seed", "7", "--answer", "6√21")
    assert out.splitlines()[-1] == "  given '6√21': correct"
    _, out, _ = run(capsys, "--skill", "simplify-surds", "--seed", "7", "--answer", "2√3")
    assert out.splitlines()[-1] == "  given '2√3': wrong"


def test_skills_lists_every_skill_with_a_status(capsys, library):
    code, out, err = run(capsys, "--skills")
    assert code == 0 and err == ""
    lines = out.splitlines()
    assert len(lines) == len(library)
    assert {line.split()[1] for line in lines} == {"active", "deferred"}


# -- multiple choice ----------------------------------------------------------------------------


def mcq_skills(library):
    return [s for s in library if s.status == "active" and s.answer == "mcq"]


def test_mcq_options_are_printed_with_labels_and_answer_names_its_label(capsys, library):
    skill = mcq_skills(library)[0]
    code, out, _ = run(capsys, "--skill", skill.id, "--seed", "1")
    assert code == 0
    q = engine.make_for_skill(skill, seed=1)
    options = mcq_options(q)
    lines = out.splitlines()
    assert [line.strip() for line in lines[2 : 2 + len(options)]] == [
        f"{label}) {text}" for label, text in zip(_labels(options), options)
    ]
    label = _labels(options)[options.index(q.answer_text)]
    assert lines[-1] == f"  A: {label}) {q.answer_text}"


def test_mcq_accepts_a_label_or_the_option_text(capsys, library):
    skill = mcq_skills(library)[0]
    q = engine.make_for_skill(skill, seed=1)
    options = mcq_options(q)
    labels = _labels(options)
    right = options.index(q.answer_text)
    wrong = next(i for i in range(len(options)) if i != right)
    for given, verdict in (
        (labels[right], "correct"),
        (labels[right].lower(), "correct"),
        (f"({labels[right]})", "correct"),
        (q.answer_text, "correct"),
        (labels[wrong], "wrong"),
        (options[wrong], "wrong"),
    ):
        _, out, _ = run(capsys, "--skill", skill.id, "--seed", "1", "--answer", given)
        assert out.splitlines()[-1] == f"  given {given!r}: {verdict}", given


def test_mcq_reply_that_names_no_option_is_wrong_and_says_why(capsys, library):
    skill = mcq_skills(library)[0]
    code, out, _ = run(capsys, "--skill", skill.id, "--seed", "1", "--answer", "no such option")
    assert code == 0
    assert out.splitlines()[-1].startswith("  given 'no such option': wrong (not an option label A-")


def test_a_reply_that_fits_two_options_apart_from_case_names_the_one_typed(capsys):
    """Options `dS/ds = bS` (the key) and `dS/ds = bs` differ only in case: each reply must pick its own."""
    skill = "senior-formulate-differential-equations-including-the-logistic-equation-that"
    for given, verdict in (("dS/ds = bS", "correct"), ("dS/ds = bs", "wrong"), ("B", "correct"), ("C", "wrong")):
        _, out, _ = run(capsys, "--skill", skill, "--seed", "13", "--answer", given)
        assert out.splitlines()[-1] == f"  given {given!r}: {verdict}", given
    # Ignoring case is only for a reply that fits one option.
    assert _choose_option(["dS/ds = bS", "dS/ds = bs"], "DS/DS = BS") is None
    assert _choose_option(["Paris", "Berlin"], "paris") == 0


def test_a_printed_label_wins_over_option_text_that_looks_the_same(capsys):
    """Options `3*z`, `a`, `1`, `-8*z**2` are labelled 1 to 4 (one option is a letter); `--answer 1` means the
    option labelled 1, not the option whose text is "1"."""
    _, out, _ = run(capsys, "--skill", "identify-like-terms", "--seed", "307")
    assert [line.strip() for line in out.splitlines()[2:6]] == ["1) 3*z", "2) a", "3) 1", "4) -8*z**2"]
    for given, verdict in (("1", "correct"), ("3*z", "correct"), ("3", "wrong"), ("2", "wrong"), ("-8*z**2", "wrong")):
        _, out, _ = run(capsys, "--skill", "identify-like-terms", "--seed", "307", "--answer", given)
        assert out.splitlines()[-1] == f"  given {given!r}: {verdict}", given
    options = ["3*z", "a", "1", "-8*z**2"]
    assert _labels(options) == ["1", "2", "3", "4"]
    assert [_choose_option(options, given) for given in ("1", "3", "a", "3*z")] == [0, 2, 1, 0]
    assert _choose_option(["x", "y"], "A") == 0  # loose spellings of a label still work
    assert _choose_option(["x", "y"], "(b)") == 1


def test_every_mcq_skill_can_be_answered_by_label_and_by_text(library):
    """For every active mcq skill, over 25 seeds and its first and last source: each option's label picks that
    option, each option's text picks it too (unless the text is itself another option's printed label, which
    wins), and only the option holding the answer grades correct."""
    checked = 0
    for skill in mcq_skills(library):
        for source in sorted({0, len(skill.sources) - 1}):
            for seed in range(1, 26):
                q = engine.make_for_skill(skill, seed=seed, source=source)
                options = mcq_options(q)
                labels = _labels(options)
                key = options.index(q.answer_text)
                for i, (label, text) in enumerate(zip(labels, options)):
                    where = (skill.id, seed, source, options)
                    assert _choose_option(options, label) == i, (label, *where)
                    expected = labels.index(text) if text in labels else i
                    assert _choose_option(options, text) == expected, (text, *where)
                    for given, index in ((label, i), (text, expected)):
                        assert engine.grade(skill, q, str(index)).correct == (index == key), (given, *where)
                    checked += 1
    assert checked > 20000


def test_options_that_are_themselves_letters_are_numbered_so_labels_cannot_clash(capsys, library):
    """'Which of statements A-D ...' questions have options 'A'..'D'; a letter label would name another option."""
    lettered = [s for s in mcq_skills(library) if {"A", "B", "C"} <= set(mcq_options(engine.make_for_skill(s, seed=1)))]
    assert lettered, "the library is expected to have questions whose options are the letters A, B, C, ..."
    skill = lettered[0]
    q = engine.make_for_skill(skill, seed=1)
    options = mcq_options(q)
    assert _labels(options) == [str(i + 1) for i in range(len(options))]
    _, out, _ = run(capsys, "--skill", skill.id, "--seed", "1", "--answer", q.answer_text)
    assert out.splitlines()[-1].endswith(": correct")
    _, out, _ = run(capsys, "--skill", skill.id, "--seed", "1", "--answer", str(options.index(q.answer_text) + 1))
    assert out.splitlines()[-1].endswith(": correct")


# -- deferred skills ----------------------------------------------------------------------------


def test_deferred_skill_without_sources_is_a_clear_error_not_a_traceback(capsys, library):
    skill = next(s for s in library if s.status == "deferred" and not s.sources)
    code, out, err = run(capsys, "--skill", skill.id, "--seed", "1", "--answer", "1")
    assert code == 1
    assert out == ""
    assert "Traceback" not in err
    assert f"skill {skill.id!r} has no question sources" in err
    assert "deferred" in err


def test_deferred_skill_without_a_grader_prints_the_question_then_says_it_cannot_grade(capsys, library):
    skill = next(s for s in library if s.status == "deferred" and s.sources and s.answer not in GRADERS)
    code, out, err = run(capsys, "--skill", skill.id, "--seed", "1", "--answer", "x")
    assert code == 1
    assert out.startswith(f"[{skill.id} #1]\n  Q: ")
    assert "Traceback" not in err
    assert f"no grader for answer kind {skill.answer!r}" in err


def test_deferred_skill_that_works_still_works(capsys, library):
    skill = next(s for s in library if s.status == "deferred" and s.sources and s.answer in GRADERS)
    code, out, err = run(capsys, "--skill", skill.id, "--seed", "1")
    assert code == 0
    assert out.startswith(f"[{skill.id} #1]")
    assert "is deferred" in err  # a note on stderr only


def test_unknown_skill_difficulty_and_topic_preset_are_usage_errors(capsys):
    code, err = run_error(capsys, "--skill", "simplify-surds", "--difficulty", "exam")
    assert code == 2 and "no source with difficulty 'exam'" in err and "easy, hard, medium" in err
    code, err = run_error(capsys, "--topic", "addition", "--difficulty", "exam")
    assert code == 2 and "has no preset 'exam'; known: easy, hard, medium" in err
    code, err = run_error(capsys, "--skill", "no-such-skill")
    assert code == 2 and "unknown skill" in err


def test_answer_without_skill_is_a_usage_error(capsys):
    code, err = run_error(capsys, "--topic", "addition", "--answer", "3")
    assert code == 2 and "--answer needs --skill" in err


# -- bulk generation with --difficulty -----------------------------------------------------------


def topics_without(difficulty):
    return sorted(t for t in PRESETS if difficulty not in PRESETS[t])


def test_bulk_generation_skips_topics_that_lack_the_preset_and_says_so(capsys, tmp_path):
    lacking = topics_without("easy")
    assert lacking, "the library is expected to have topics with presets but no 'easy'"
    level = REGISTRY[lacking[0]][0]
    out_file = tmp_path / "bank.jsonl"
    code, out, err = run(
        capsys, "--level", str(level), "-n", "1", "--seed", "1", "--difficulty", "easy", "--out", str(out_file)
    )
    assert code == 0
    skipped = [t for t in lacking if REGISTRY[t][0] == level]
    assert f"skipped {len(skipped)} topic(s) with no 'easy' preset: {', '.join(skipped)}" in err
    rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert rows and not {r["topic"] for r in rows} & set(skipped)
    assert {r["difficulty"] for r in rows} == {"easy"}


def test_one_topic_without_the_preset_is_a_usage_error(capsys):
    topic = topics_without("easy")[0]
    code, err = run_error(capsys, "--topic", topic, "--difficulty", "easy")
    assert code == 2 and f"topic {topic!r} has no preset 'easy'; known: " in err


# -- the library folder -------------------------------------------------------------------------


GOOD_SKILL = """\
id: add-two
name: Add two
domain: arithmetic
status: {status}
{reason}prerequisites: []
answer: number
sources:
  - topic: addition
    params: {{}}
    difficulty: easy
"""


def write_library(root, **files):
    (root / "arithmetic").mkdir(parents=True)
    for name, text in files.items():
        (root / "arithmetic" / f"{name}.yaml").write_text(text, encoding="utf-8")
    return root


def test_skills_fails_loudly_when_there_is_no_library(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("MATHS_SKILLS_DIR", str(tmp_path / "nothing-here"))
    code, out, err = run(capsys, "--skills")
    assert code == 1 and out == ""
    assert "no skills found" in err and "MATHS_SKILLS_DIR" in err
    monkeypatch.setenv("MATHS_SKILLS_DIR", str(tmp_path))  # exists, but empty
    code, out, err = run(capsys, "--skills")
    assert code == 1 and "no skills found" in err


def test_skills_reports_a_malformed_file_as_an_error(capsys, monkeypatch, tmp_path):
    root = write_library(
        tmp_path,
        **{"add-two": GOOD_SKILL.format(status="active", reason=""), "broken": "id: [unclosed\nname: x\n"},
    )
    monkeypatch.setenv("MATHS_SKILLS_DIR", str(root))
    code, out, err = run(capsys, "--skills")
    assert code == 1
    assert "arithmetic/add-two  active" in out
    assert "arithmetic/broken  invalid" in out
    assert "1 skill file(s) did not load" in err and "arithmetic/broken.yaml" in err
    code, out, err = run(capsys, "--skill", "broken")
    assert code == 1 and "did not load" in err and "Traceback" not in err


def test_skills_catches_a_file_whose_id_does_not_match_its_name(capsys, monkeypatch, tmp_path):
    root = write_library(tmp_path, **{"copy-of-add-two": GOOD_SKILL.format(status="active", reason="")})
    monkeypatch.setenv("MATHS_SKILLS_DIR", str(root))
    code, out, err = run(capsys, "--skills")
    assert code == 1 and "id 'add-two' != filename stem 'copy-of-add-two'" in err


def test_a_valid_deferred_skill_is_not_a_problem(capsys, monkeypatch, tmp_path):
    root = write_library(
        tmp_path, **{"add-two": GOOD_SKILL.format(status="deferred", reason="deferred_reason: hard\n")}
    )
    monkeypatch.setenv("MATHS_SKILLS_DIR", str(root))
    code, out, err = run(capsys, "--skills")
    assert code == 0 and out.splitlines() == ["arithmetic/add-two  deferred"] and err == ""


# -- replies that start with a minus sign --------------------------------------------------------


def test_a_reply_starting_with_a_minus_sign_is_a_value_not_an_option(capsys, library):
    for skill in library:
        if skill.status == "active" and skill.answer == "number":
            q = engine.make_for_skill(skill, seed=1)
            if q.answer_text.startswith("-"):
                break
    else:
        pytest.fail("expected a number skill with a negative answer at seed 1")
    for given, verdict in ((q.answer_text, "correct"), ("-3/4", "wrong")):
        code, out, err = run(capsys, "--skill", skill.id, "--seed", "1", "--answer", given)
        assert code == 0 and err == ""
        assert out.splitlines()[-1] == f"  given {given!r}: {verdict}"
    code, err = run_error(capsys, "--skill", skill.id, "--answer")
    assert code == 2 and "expected one argument" in err


# -- output files and difficulty names -----------------------------------------------------------


def test_an_output_file_that_cannot_be_written_is_a_clean_error(capsys, tmp_path):
    for target in (tmp_path / "missing" / "bank.jsonl", tmp_path):
        code, out, err = run(capsys, "--topic", "addition", "-n", "1", "--out", str(target))
        assert code == 1 and out == ""
        assert err.startswith(f"mathgen: error: cannot write {str(target)!r}: ") and "Traceback" not in err


def test_an_unknown_difficulty_name_is_a_usage_error_that_lists_the_valid_ones(capsys):
    valid = {"easy", "medium", "hard", "exam"}
    for argv in (("--all", "-n", "1"), ("--level", "1"), ("--topic", "addition"), ("--topic", "quadratic")):
        code, err = run_error(capsys, *argv, "--difficulty", "bogus")
        assert code == 2 and "unknown difficulty 'bogus'; valid names are " in err, argv
        names = err.split("valid names are ")[1].strip().split(", ")
        assert valid <= set(names) and names == sorted(names)


def test_valid_difficulty_names_still_work(capsys):
    for name in ("easy", "medium", "hard"):  # topics with no presets use their defaults for these too
        code, out, _ = run(capsys, "--topic", "addition", "-n", "1", "--seed", "1", "--difficulty", name)
        assert code == 0 and json.loads(out)["difficulty"] == name
    exam_topic = next(t for t in PRESETS if "exam" in PRESETS[t])
    code, out, _ = run(capsys, "--topic", exam_topic, "-n", "1", "--seed", "1", "--difficulty", "exam")
    assert code == 0 and json.loads(out)["difficulty"] == "exam"


# -- what the final review found -----------------------------------------------------------------


ROOT = Path(__file__).resolve().parents[2]


def library_with_source(monkeypatch, tmp_path, *, status="active", **replacements):
    """A one-skill library whose only source is GOOD_SKILL's with `replacements` (old -> new) applied."""
    reason = "deferred_reason: generator\n" if status == "deferred" else ""
    text = GOOD_SKILL.format(status=status, reason=reason)
    for old, new in replacements.items():
        assert old in text
        text = text.replace(old, new)
    monkeypatch.setenv("MATHS_SKILLS_DIR", str(write_library(tmp_path, **{"add-two": text})))


def test_an_unknown_topic_in_an_active_skill_is_a_clean_error_with_suggestions(capsys, monkeypatch, tmp_path):
    library_with_source(monkeypatch, tmp_path, **{"topic: addition": "topic: additon"})
    code, out, err = run(capsys, "--skill", "add-two", "--seed", "1")
    assert code == 1 and out == ""
    assert err.startswith(
        "mathgen: error: skill 'add-two' names topic 'additon', which has no generator; did you mean 'addition'"
    )
    assert err.endswith("?\n") and err.count("\n") == 1
    library_with_source(monkeypatch, tmp_path / "other", **{"topic: addition": "topic: qqqqzzzz"})
    code, out, err = run(capsys, "--skill", "add-two", "--seed", "1")
    assert code == 1 and err.endswith("which has no generator (see --list)\n") and "Traceback" not in err


def test_an_unknown_preset_in_a_skill_source_is_a_clean_error_that_lists_the_presets(capsys, monkeypatch, tmp_path):
    library_with_source(monkeypatch, tmp_path, **{"difficulty: easy": "difficulty: esay"})
    code, out, err = run(capsys, "--skill", "add-two", "--seed", "1", "--answer", "3")
    assert code == 1 and out == ""
    assert "skill 'add-two': topic 'addition' has no preset 'esay'; known: easy, hard, medium" in err
    assert "Traceback" not in err


def test_a_deferred_skill_naming_an_unknown_topic_gets_the_same_clean_error(capsys, monkeypatch, tmp_path):
    library_with_source(monkeypatch, tmp_path, status="deferred", **{"topic: addition": "topic: additon"})
    code, out, err = run(capsys, "--skill", "add-two", "--seed", "1")
    assert code == 1 and out == ""
    assert "is deferred" in err and "names topic 'additon', which has no generator; did you mean 'addition'" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("mode", [("--skill", "simplify-surds"), ("--skills",), ("--list",)])
@pytest.mark.parametrize("flag", ["--out", "-n", "--pretty"])
def test_options_a_command_would_ignore_are_usage_errors(capsys, tmp_path, mode, flag):
    extra = {"--out": [flag, str(tmp_path / "bank.jsonl")], "-n": [flag, "3"], "--pretty": [flag]}[flag]
    code, err = run_error(capsys, *mode, *extra)
    shown = "-n/--count" if flag == "-n" else flag
    assert code == 2 and f"{shown} does not apply to {mode[0]}; it is for --topic, --level and --all" in err
    assert not (tmp_path / "bank.jsonl").exists()


def test_an_empty_name_is_an_error_and_not_everything(capsys):
    code, err = run_error(capsys, "--skill", "")
    assert code == 2 and "unknown skill ''" in err
    code, err = run_error(capsys, "--topic", "")
    assert code == 2 and "unknown topic ''" in err
    code, err = run_error(capsys, "--topic", "addition", "--out", "")
    assert code == 2 and "--out needs a file name" in err


@pytest.mark.parametrize("count", ["0", "-5"])
def test_a_count_below_one_is_a_usage_error(capsys, count):
    code, err = run_error(capsys, "--topic", "addition", "-n", count)
    assert code == 2 and "argument -n/--count: must be at least 1" in err


class FullDisk:
    """A file that accepts writes and fails when the data has to reach the disk."""

    closed = False

    def write(self, text):
        return len(text)

    def flush(self):
        raise OSError(errno.ENOSPC, "No space left on device")

    def close(self):
        self.closed = True
        raise OSError(errno.ENOSPC, "No space left on device")


def test_a_file_that_fills_up_is_a_clean_error_and_no_success_message(capsys, monkeypatch):
    monkeypatch.setattr("mathgen.__main__.open", lambda *a, **k: FullDisk(), raising=False)
    code, out, err = run(capsys, "--topic", "addition", "-n", "2", "--out", "bank.jsonl")
    assert code == 1 and out == ""
    assert err == "mathgen: error: cannot write 'bank.jsonl': No space left on device\n"


def test_standard_output_that_fills_up_is_a_clean_error(capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdout", FullDisk())
    code = main(["--topic", "addition", "-n", "2"])
    assert code == 1
    assert capsys.readouterr().err == "mathgen: error: cannot write standard output: No space left on device\n"


@pytest.mark.skipif(not os.path.exists("/dev/full"), reason="needs /dev/full")
def test_dev_full_is_reported_not_a_traceback(capsys):
    code, out, err = run(capsys, "--topic", "addition", "-n", "1", "--out", "/dev/full")
    assert code == 1 and err == "mathgen: error: cannot write '/dev/full': No space left on device\n"


def test_a_closed_pipe_still_ends_quietly():
    """`mathgen ... | head -n 1`: the reader goes away, and nothing is printed on stderr."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "mathgen", "--topic", "addition", "-n", "50000", "--seed", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert proc.stdout.readline().startswith(b'{"topic": "addition"')
    proc.stdout.close()
    _, err = proc.communicate(timeout=120)
    assert proc.returncode == 1 and err == b""


def test_the_version_is_reported_by_the_package_and_the_command(capsys):
    try:
        expected = metadata.version("maths-engine")
    except metadata.PackageNotFoundError:
        expected = "0+unknown"
    assert mathgen.__version__ == expected
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0 and capsys.readouterr().out == f"mathgen {expected}\n"


def test_a_checkout_that_was_never_installed_reports_an_unknown_version(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    assert mathgen._version() == "0+unknown"
