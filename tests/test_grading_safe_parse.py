"""A typed answer is data, never code: the expression grader must not execute what a learner types.

Every payload below would write `marker` (or call `os`) if the answer reached Python's `eval`.
"""

from __future__ import annotations

import logging
import pathlib
import sys

import pytest
import sympy

from mathgen.core import Question
from mathgen.grading import grade, typed
from mathgen.grading.safe_parse import MAX_LENGTH, parse_expression_safely

_HOSTILE = [
    "__import__('pathlib').Path('MARKER').write_text('1')",
    "(__import__('pathlib').Path('MARKER').write_text('1'))",
    "x + __import__('pathlib').Path('MARKER').write_text('1')",
    "eval(\"__import__('pathlib').Path('MARKER').write_text('1')\")",
    "exec(\"open('MARKER','w').write('1')\")",
    "open('MARKER','w').write('1')",
    "getattr(__import__('os'),'system')('touch MARKER')",
    "x.__class__.__mro__[1].__subclasses__()",
    "(1).__class__.__base__.__subclasses__()",
    "[c for c in ().__class__.__base__.__subclasses__()]",
    "lambda: __import__('os')",
    "sin.__globals__",
    "x; import os",
    "compile('1', 'f', 'eval')",
    "globals()",
    "sympify('__import__(\"os\")')",
    "init_session()",
    "var('x')",
    'f\'{open("MARKER","w").write("1")}\'',
]


#: A relative file name: payloads write it into the test's working directory (an absolute tmp path
#: would be mangled by the learner-notation normaliser, whose function-name rewrite splits "exp").
_MARKER = "zz_marker"


@pytest.fixture(autouse=True)
def _in_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _question(answer: str = "6*sqrt(21)", form: str | None = None) -> Question:
    return Question(
        topic="test",
        level=0,
        seed=1,
        text="Simplify.",
        latex="Simplify.",
        answer=answer,
        answer_text=answer,
        answer_latex=answer,
        working=[],
        difficulty="easy",
        distractors=[],
        **({"answer_form": form} if form else {}),
    )


@pytest.mark.parametrize("payload", _HOSTILE)
def test_parse_expression_never_executes_the_answer(payload, tmp_path):
    typed._parse_expression(payload.replace("MARKER", _MARKER))
    assert not (tmp_path / _MARKER).exists()


@pytest.mark.parametrize("payload", _HOSTILE)
def test_expression_grader_never_executes_the_answer(payload, tmp_path):
    result = grade("expression", _question(), payload.replace("MARKER", _MARKER))
    assert result.correct is False
    assert not (tmp_path / _MARKER).exists()


@pytest.mark.parametrize("payload", _HOSTILE)
def test_parse_expression_safely_rejects_or_stays_symbolic(payload, tmp_path):
    try:
        parse_expression_safely(payload.replace("MARKER", _MARKER), transformations=typed._TRANSFORMATIONS)
    except Exception:
        pass
    assert not (tmp_path / _MARKER).exists()


def test_builtins_are_not_in_scope():
    # Before the fix `open`, `__import__` and the rest resolved as Python builtins inside parse_expr.
    for name in ("open", "eval", "exec", "getattr", "__import__", "compile", "input"):
        parsed = typed._parse_expression(name)
        # implicit multiplication reads "open" as o*p*e*n: plain symbols, nothing callable
        assert parsed is None or (isinstance(parsed, sympy.Basic) and not parsed.atoms(sympy.Function)), name


def test_oversized_answer_is_refused_before_tokenising():
    assert typed._parse_expression("1+" * (MAX_LENGTH // 2 + 1) + "1") is None


def test_logged_reply_is_truncated(caplog):
    class Boom:
        answer_text = "1"

    def explode(question, given):
        raise RuntimeError("boom")

    original = typed.GRADERS["number"]
    typed.GRADERS["number"] = explode
    try:
        with caplog.at_level(logging.ERROR, logger="mathgen.grading"):
            grade("number", Boom(), "9" * 50_000)
    finally:
        typed.GRADERS["number"] = original
    assert caplog.records
    assert all(len(record.getMessage()) < 1_000 for record in caplog.records)


@pytest.mark.parametrize(
    "given, expected",
    [
        ("6√21", "6*sqrt(21)"),
        ("6*sqrt(21)", "6*sqrt(21)"),
        ("sqrt(756)", "6*sqrt(21)"),
        ("2x^2 + 3x - 5", "2*x**2 + 3*x - 5"),
        ("sin(x)^2 + cos(x)^2", "1"),
        ("ln|x| + C", "log(Abs(x)) + C"),
        ("e^(2x)", "exp(2*x)"),
        ("π/3", "pi/3"),
        ("arcsin(1/2)", "pi/6"),
        ("cosec(x)", "1/sin(x)"),
        ("3 theta", "3*theta"),
        ("5!", "120"),
        ("abs(-3)", "3"),
        ("1/2", "1/2"),
    ],
)
def test_ordinary_answers_still_parse_and_compare(given, expected):
    parsed = typed._parse_expression(given)
    assert parsed is not None, given
    want = typed._parse_expression(expected)
    assert want is not None
    assert sympy.simplify(parsed - want) == 0 or typed._resolve_constants(parsed, want) == want


def test_letters_named_like_sympy_functions_are_symbols_not_functions():
    # `beta`, `gamma`, `zeta`, `lambda` style names a question may use as variables: plain Symbols.
    parsed = typed._parse_expression("2*beta + gamma")
    assert parsed is not None
    assert {s.name for s in parsed.free_symbols} == {"beta", "gamma"}


def test_module_is_importable_without_the_app():
    # `mathgen.grading.safe_parse` ships in the public package: no api/web import.
    source = pathlib.Path(sys.modules["mathgen.grading.safe_parse"].__file__).read_text()
    assert "from api" not in source and "import api" not in source


def test_unevaluated_parse_keeps_the_cost_guard_working():
    # `evaluate=False` makes parse_expr emit Add/Mul/Pow calls; a namespace without them failed that
    # parse, which `_too_costly_to_parse` reads as "not costly", so a power tower then got evaluated.
    tree = parse_expression_safely("2**2**2**2**2**2**2**2", transformations=typed._TRANSFORMATIONS, evaluate=False)
    assert isinstance(tree, sympy.Pow)
    assert typed._too_costly_to_parse("2**2**2**2**2**2**2**2") is True


def test_payloads_really_do_write_a_file_under_raw_parse_expr(tmp_path):
    # Negative control: the same payload through sympy's own parse_expr writes the file, so the tests
    # above can fail. (This is the hole the module closes.)
    from sympy.parsing.sympy_parser import parse_expr

    parse_expr("open('zz_marker','w').write('1')")
    assert (tmp_path / _MARKER).exists()


@pytest.mark.parametrize("given", ["x2", "a1+b2", "2x3", "H2O", "x\u2082", "\u03b8\u2081"])
def test_names_mixing_letters_and_digits_parse_or_fail_cleanly(given):
    # sympy turns "x2" into Number('2') * x: the namespace must carry what the transformations emit.
    typed._parse_expression(given)  # must not raise NameError


def test_double_factorial_is_unsupported_not_an_exception():
    assert typed._parse_expression("5!!") is None or isinstance(typed._parse_expression("5!!"), sympy.Basic)


def test_leading_nbsp_cannot_hide_a_leading_underscore():
    for text in ("\xa0__builtins__", "\u3000__import__", "\u00a0_x"):
        with pytest.raises(ValueError):
            parse_expression_safely(text, transformations=typed._TRANSFORMATIONS)


def test_oversized_answer_never_reaches_the_spoken_fallback(monkeypatch):
    called = []
    monkeypatch.setattr(typed, "spoken_to_expr", lambda text: called.append(text))
    assert typed._parse_expression("1" * 80_000) is None
    assert typed._parse_expression("2x" * 80_000) is None
    assert called == []


def test_huge_number_literals_are_refused_before_they_are_built():
    for text in ("1e500000", "1e5000000", "9" * 60):
        with pytest.raises(ValueError):
            parse_expression_safely(text, transformations=typed._TRANSFORMATIONS)
