"""`symbolically_equal`, `pairs_equal` and `latex_answer` accept text, and text must never run as Python.

The helpers were written for the generators, which pass answers and distractors as strings, and they called
plain `sympify(str)`, which `eval`s it with the builtins in scope. The grader does not hand them learner
text (it parses a reply with `parse_expression_safely` first and passes SymPy objects), but that was a
convention. Text is now read with the same restricted parser as a reply, so the convention is also a
guarantee; and a second test fails if any code under `mathgen/` (outside the generators) names an evaluator
-- `sympify`, `parse_expr`, `nsimplify`, `S(...)`, `eval`, `exec` and the like, or one of the functions that
`sympify` their argument (`simplify`, `expand`, `factor`, `cancel`, `together`, `solve`, `diff`, `integrate`,
`limit`, `series`), under whatever name it was imported or assigned -- that this file has not reviewed.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import sympy

import mathgen.core as core
from mathgen.core import Question, pairs_equal, symbolically_equal
from mathgen.grading import grade, typed
from mathgen.grading.spoken import check_answer, spoken_to_expr

_HOSTILE = [
    "__import__('pathlib').Path('zz_marker').write_text('1')",
    "(__import__('pathlib').Path('zz_marker').write_text('1'))",
    "x + __import__('pathlib').Path('zz_marker').write_text('1')",
    "open('zz_marker','w').write('1')",
    "exec(\"open('zz_marker','w').write('1')\")",
    "getattr(__import__('os'),'system')('touch zz_marker')",
    "x.__class__.__mro__[1].__subclasses__()",
    "[c for c in ().__class__.__base__.__subclasses__()]",
    "lambda: __import__('os')",
    "sympify('__import__(\"os\")')",
]


@pytest.fixture(autouse=True)
def _in_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("payload", _HOSTILE)
def test_symbolically_equal_never_runs_its_text(payload, tmp_path):
    for left, right in ((payload, "1"), ("1", payload), (payload, payload)):
        with pytest.raises((sympy.SympifyError, TypeError, ValueError, SyntaxError)):
            symbolically_equal(left, right)
    assert not (tmp_path / "zz_marker").exists()


@pytest.mark.parametrize("payload", _HOSTILE)
def test_pairs_equal_never_runs_its_text(payload, tmp_path):
    assert pairs_equal(payload, "(1, 2)") is False
    assert pairs_equal("(1, 2)", payload) is False
    assert pairs_equal((payload, "2"), ("1", "2")) is False
    assert pairs_equal((payload, "2"), ("1", "2"), tol=0.1) is False
    assert not (tmp_path / "zz_marker").exists()


@pytest.mark.parametrize("payload", _HOSTILE)
def test_a_tuple_of_strings_is_text_too(payload, tmp_path):
    with pytest.raises((sympy.SympifyError, TypeError, ValueError, SyntaxError)):
        symbolically_equal((payload, "1"), ("1", "1"))
    assert not (tmp_path / "zz_marker").exists()


def test_negative_control_plain_sympify_runs_the_text(tmp_path):
    # So the tests above can fail: the same text through SymPy's own `sympify` writes the file.
    sympy.sympify("open('zz_marker','w').write('1')")
    assert (tmp_path / "zz_marker").exists()


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ("x^2 - 1", "(x-1)*(x+1)", True),
        ("sin(x)^2 + cos(x)^2", "1", True),
        ("2*x + 3", "3 + 2*x", True),
        ("x^2", "x^3", False),
        ("1/2", "0.5", True),
        ("18.4", "18.4", True),
        ("(2, -5)", "(2, -5)", True),
        ("(2, -5)", "(2, 5)", False),
        ("(2, -5)", "2", False),
        (sympy.Symbol("x") + 1, "x + 1", True),
        (sympy.Rational(1, 3), "1/3", True),
        (3, "3", True),
    ],
)
def test_ordinary_comparisons_are_unchanged(left, right, expected):
    assert symbolically_equal(left, right) is expected


def test_pairs_equal_still_compares_elementwise_and_within_a_tolerance():
    assert pairs_equal("(1, 2)", "(1, 2)") is True
    assert pairs_equal((1, "1/2"), ("1", 0.5)) is True
    assert pairs_equal("(1.0, 2.0)", "(1.04, 2.0)", tol=0.05) is True
    assert pairs_equal("(1.0, 2.0)", "(1.2, 2.0)", tol=0.05) is False
    assert pairs_equal("(1, 2)", "(1, 2, 3)") is False


def test_unreadable_text_is_an_error_or_not_equal_never_a_crash():
    for text in ("distractor 24 must differ from answer 28", "x ===== 2", "{1, 2}", ""):
        try:
            result = symbolically_equal(text, "1")
        except (sympy.SympifyError, TypeError, ValueError, SyntaxError):
            continue
        assert result is False


# -- latex_answer reads text the same way -----------------------------------------------------------


def _latex_payloads():
    for payload in _HOSTILE:
        yield payload
        yield f"x = {payload}"
        yield f"{payload}, y = 2"


@pytest.mark.parametrize("payload", list(_latex_payloads()))
@pytest.mark.parametrize("evaluate", [True, False])
def test_latex_answer_never_runs_its_text(payload, evaluate, tmp_path):
    result = core.latex_answer(payload, evaluate=evaluate)
    assert result.startswith("plain:"), result
    assert not (tmp_path / "zz_marker").exists()


def test_latex_answer_never_runs_text_inside_a_sequence(tmp_path):
    for payload in _HOSTILE:
        assert core.latex_answer((payload, "2")).startswith("plain:")
        assert core.latex_answer([payload]).startswith("plain:")
    assert not (tmp_path / "zz_marker").exists()


#: What `latex_answer` returned for ordinary generator text before it read text with the restricted parser
#: (measured on the tree before the change): the same strings must come out now.
_LATEX_BEFORE = [
    ("x^2 + 3*x - 5", True, "x^{2} + 3 x - 5"),
    ("x^2 - 5*x + 6", False, "x^{2} - 5 x + 6"),
    ("3*(x-2)*(x+4)", True, "\\left(x + 4\\right) \\left(3 x - 6\\right)"),
    ("3*(x-2)*(x+4)", False, "3 \\left(x + 4\\right) \\left(x - 2\\right)"),
    ("sqrt(8)", True, "2 \\sqrt{2}"),
    ("8*sqrt(7)", True, "8 \\sqrt{7}"),
    ("1/2", True, "\\frac{1}{2}"),
    ("(2, -5)", True, "\\left( 2, \\  -5\\right)"),
    ("x = 3, y = -6", True, "x = 3,\\; y = -6"),
    ("y = -x + 7", True, "y = 7 - x"),
    ("{1, 2, 3}", True, "\\left\\{1, 2, 3\\right\\}"),
    ("{U2, U3, U4}", True, "\\left\\{U_{2}, U_{3}, U_{4}\\right\\}"),
    ("sin(x)^2 + cos(x)^2", False, "\\sin^{2}{\\left(x \\right)} + \\cos^{2}{\\left(x \\right)}"),
    ("exp(2*x)", True, "e^{2 x}"),
    ("pi*r^2", True, "\\pi r^{2}"),
    ("18.40", True, "18.40"),
    ("Obtuse", True, "plain:Obtuse"),
    ("Monday, Tuesday", True, "plain:Monday, Tuesday"),
    ("", True, "plain:"),
    ("2*(x + 1)^2", False, "2 \\left(x + 1\\right)^{2}"),
    ("x/2 + 1/3", True, "\\frac{x}{2} + \\frac{1}{3}"),
    ("log(x)/x", True, "\\frac{\\log{\\left(x \\right)}}{x}"),
    ("sqrt(x^2 + 1)", True, "\\sqrt{x^{2} + 1}"),
    ("-3/4", True, "- \\frac{3}{4}"),
    ("(1, 2), (3, 4)", True, "\\left( \\left( 1, \\  2\\right), \\  \\left( 3, \\  4\\right)\\right)"),
    ("sin(x)", True, "\\sin{\\left(x \\right)}"),
    ("3.5*x", True, "3.5 x"),
    ("x_1 + x_2", True, "x_{1} + x_{2}"),
    ("theta = 30", True, "\\theta = 30"),
    ("2 pi", True, "plain:2 pi"),
]


@pytest.mark.parametrize("text, evaluate, expected", _LATEX_BEFORE)
def test_latex_answer_output_for_ordinary_text_is_unchanged(text, evaluate, expected):
    assert core.latex_answer(text, evaluate=evaluate) == expected


def test_latex_answer_still_renders_sympy_objects_and_numbers():
    x = sympy.Symbol("x")
    assert core.latex_answer(x**2 + 1) == "x^{2} + 1"
    assert core.latex_answer(sympy.Rational(3, 4)) == "\\frac{3}{4}"
    assert core.latex_answer(5) == "5"
    assert core.latex_answer((1, 2)) == "\\left( 1, \\  2\\right)"


# -- nothing a learner types reaches `sympify` ------------------------------------------------------

_LEARNER_REPLIES = [
    "12x^2 - 4x + 1",
    "x^2 + (",
    "__import__('os').system('true')",
    "5 to the power of 3 and 2",
    "twelve x squared minus four x plus one",
    "x = 3, y = -6",
    "(28, 24)",
    "y = -x + 7",
    "x**2**2**2**2",
    "10^10^10",
    "3 and 4 or 5",
    "2x + 3y - 6 = 0",
]


def _hand_built(**fields) -> Question:
    base = dict(
        topic="t", level=0, seed=1, text="q", latex="q", answer_latex="", working=[], difficulty="easy",
        distractors=[],
    )  # fmt: skip
    return Question(**{**base, **fields})


def test_no_learner_text_reaches_sympify(monkeypatch):
    seen: list = []
    real_core, real_sympy = core.sympify, sympy.sympify

    def watch_core(value, *args, **kwargs):
        if isinstance(value, (str, tuple, list)):
            seen.append(value)
        return real_core(value, *args, **kwargs)

    def watch_sympy(value, *args, **kwargs):
        if isinstance(value, (str, tuple, list)):
            seen.append(value)
        return real_sympy(value, *args, **kwargs)

    monkeypatch.setattr(core, "sympify", watch_core)
    monkeypatch.setattr(sympy, "sympify", watch_sympy)

    questions = [
        ("expression", _hand_built(answer="12*x**2 - 4*x + 1", answer_text="12x^2 - 4x + 1")),
        ("expression", _hand_built(answer="y = -x + 7", answer_text="y = -x + 7")),
        ("pair", _hand_built(answer={"x": 3, "y": -6}, answer_text="x = 3, y = -6")),
        ("pair", _hand_built(answer={"m": -1, "b": 7}, answer_text="y = -x + 7")),
        ("number", _hand_built(answer=7, answer_text="7")),
        ("list", _hand_built(answer=[3, 6], answer_text="x = 3 and x = 6")),
    ]
    for kind, question in questions:
        for reply in _LEARNER_REPLIES:
            grade(kind, question, reply)
            check_answer(question, reply)
        spoken_to_expr("twelve x squared")
    assert seen == []


# -- and nobody adds a new way to evaluate text without this test noticing -----------------------

_ROOT = pathlib.Path(core.__file__).resolve().parent

#: Names that turn text into a SymPy object or run it. `sympify`, `parse_expr` and `nsimplify` read a string;
#: `eval`, `exec` and `compile` run one; `literal_eval`, `__import__` and `import_module` reach a module by name;
#: `parse_latex` and friends are SymPy's other text readers.
_EVALUATORS = frozenset(
    {
        "sympify", "parse_expr", "nsimplify", "eval", "exec", "compile", "literal_eval", "__import__",
        "import_module", "parse_latex", "parse_mathematica", "latex2sympy", "stringify_expr", "eval_expr",
    }
)  # fmt: skip
#: Names that are an evaluator only when called: `S.One` is a constant, `S("x")` is `sympify`, and a local `N`
#: or `Q` is just a name unless it came from sympy. The second group are the SymPy functions that `sympify`
#: their first argument, so that `simplify("__import__('os')...")` runs it; a local called `diff`, `limit` or
#: `series` is a name like any other. Their uses on SymPy objects are listed in `_REVIEWED`.
_SYMPIFYING_FUNCTIONS = frozenset(
    {"simplify", "expand", "factor", "cancel", "together", "solve", "diff", "integrate", "limit", "series"}
)
_EVALUATORS_WHEN_CALLED = frozenset({"S", "N"}) | _SYMPIFYING_FUNCTIONS
#: What a dynamic lookup needs: `getattr(sympy, "sympify")`, `vars(sympy)["sympify"]`, `globals()["eval"]`.
_DYNAMIC_LOOKUPS = frozenset({"getattr", "vars", "globals", "locals", "__getattribute__"})

#: The calls and references reviewed, as (file under mathgen/, enclosing function, name). Each either hands
#: `safe_parse` text it has already restricted, or hands SymPy something that is not text. Anything else fails
#: the test below until it is read and listed here with its reason.
_REVIEWED = {
    # the one place `parse_expr` runs, behind the token check, the namespace and the cost guard
    ("grading/safe_parse.py", "<module>", "parse_expr"): "imported for `_parse`",
    ("grading/safe_parse.py", "_parse", "parse_expr"): "the restricted parse itself",
    # `sympify_safely` hands a non-text value (a number, a SymPy object) to sympify
    ("grading/safe_parse.py", "sympify_safely", "sympify"): "a value that is not text",
    # `core._sympify` hands the same non-text values over
    ("core.py", "<module>", "sympify"): "imported for `_sympify`",
    ("core.py", "_sympify", "sympify"): "a value that is not text",
    # the functions that `sympify` what they are given, each called on SymPy objects that came out of the
    # restricted parser (a text argument to any of them would run as Python)
    ("core.py", "symbolically_equal", "simplify"): "the difference of two values read by `_sympify`",
    ("grading/spoken.py", "_check_one", "simplify"): "a spoken reading less a `sympify_safely` value",
    ("grading/typed.py", "_expected_expression", "simplify"): "two expressions from `_parse_expression`",
    ("grading/typed.py", "_line_standard_form_coeffs", "expand"): "the two sides of an equation, parsed",
    ("grading/typed.py", "grade_expression", "expand"): "the reply, already parsed by `_parse_expression`",
}  # fmt: skip


def _scope_name(stack: list) -> str:
    return ".".join(stack) if stack else "<module>"


class _Scan(ast.NodeVisitor):
    """Every use of an evaluator in one module: a call, a reference, an import, an alias or a dynamic lookup."""

    def __init__(self, relative: str):
        self.relative = relative
        self.found: set = set()
        self.stack: list = []
        self.aliases: dict = {}  # a local name -> the evaluator it stands for

    def _hit(self, name: str) -> None:
        self.found.add((self.relative, _scope_name(self.stack), name))

    def _evaluator_of(self, node):
        """The evaluator a name or attribute stands for, else None."""
        if isinstance(node, ast.Name):
            return node.id if node.id in _EVALUATORS else self.aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            return node.attr if node.attr in _EVALUATORS else None
        return None

    def _scoped(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.rsplit(".", 1)[-1] in _EVALUATORS:
                self._hit(alias.name.rsplit(".", 1)[-1])
                self.aliases[alias.asname or alias.name] = alias.name.rsplit(".", 1)[-1]

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name in _EVALUATORS:
                self._hit(alias.name)
                self.aliases[alias.asname or alias.name] = alias.name
            elif alias.name in _EVALUATORS_WHEN_CALLED and "sympy" in (node.module or ""):
                self.aliases[alias.asname or alias.name] = alias.name  # reported where it is called

    def visit_Assign(self, node):
        evaluator = self._evaluator_of(node.value)
        if evaluator is not None:  # `run = sympy.sympify`
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.aliases[target.id] = evaluator
        self.generic_visit(node)

    def visit_Name(self, node):
        evaluator = self.aliases.get(node.id, node.id)
        if evaluator in _EVALUATORS:
            self._hit(evaluator)

    def visit_Attribute(self, node):
        # `re.compile` is a regular expression; only the builtin `compile` (a bare name) evaluates text.
        if node.attr in _EVALUATORS and node.attr != "compile":
            self._hit(node.attr)
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Name):
            called = self.aliases.get(func.id)  # only an `S` / `N` that came from sympy counts
        else:
            called = getattr(func, "attr", None)
        if called in _EVALUATORS_WHEN_CALLED:
            self._hit(called)  # `sympy.S("x")`, or an `S` / `N` imported from sympy, under any name
        written = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if written in _DYNAMIC_LOOKUPS:
            names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if any(n in _EVALUATORS | _EVALUATORS_WHEN_CALLED for n in names):
                self._hit(next(n for n in names if n in _EVALUATORS | _EVALUATORS_WHEN_CALLED))
        self.generic_visit(node)

    def visit_Subscript(self, node):
        key = node.slice
        if isinstance(key, ast.Constant) and key.value in _EVALUATORS | _EVALUATORS_WHEN_CALLED:
            self._hit(key.value)  # `vars(sympy)["sympify"]`, `globals()["eval"]`
        self.generic_visit(node)


def _findings(source: str, relative: str = "snippet.py") -> set:
    scan = _Scan(relative)
    scan.visit(ast.parse(source))
    return scan.found


def _guarded_files():
    """Everything under mathgen/ except the generators, which only ever read their own text."""
    for path in sorted(_ROOT.rglob("*.py")):
        relative = path.relative_to(_ROOT).as_posix()
        if not relative.startswith("generators/"):
            yield path, relative


def test_the_scanner_sees_every_way_to_reach_an_evaluator():
    """Each of these must be reported, or the guard below would let it through."""
    snippets = [
        "from sympy import sympify\nsympify('x')",
        "from sympy import sympify as read\nread('x')",
        "import sympy\nsympy.sympify('x')",
        "import sympy as sp\nsp.sympify('x')",
        "import sympy as sp\nsp.S('x')",
        "from sympy import S\nS('x')",
        "from sympy import S as one\none('x')",
        "from sympy import N\nN('1+1')",
        "import sympy as sp\nsp.nsimplify('x')",
        "from sympy import nsimplify as nice\nnice('x')",
        "from sympy.parsing.sympy_parser import parse_expr\nparse_expr('x')",
        "from sympy.parsing.sympy_parser import parse_expr as p\np('x')",
        "import sympy\nrun = sympy.sympify\nrun('x')",
        "import sympy\nlist(map(sympy.sympify, ['x']))",
        "import sympy\ngetattr(sympy, 'sympify')('x')",
        "import sympy\nvars(sympy)['sympify']('x')",
        "eval('1')",
        "exec('x = 1')",
        "compile('1', 'f', 'eval')",
        "import ast\nast.literal_eval('1')",
        "__import__('os')",
        "import importlib\nimportlib.import_module('os')",
        "import sympy.core.sympify as m\nm.sympify('x')",
        "from sympy.parsing.latex import parse_latex\nparse_latex('x')",
        # the functions that sympify their argument, however they are reached
        "from sympy import simplify\nsimplify('x')",
        "from sympy import expand as spread\nspread('x')",
        "import sympy\nsympy.factor('x')",
        "import sympy as sp\nsp.cancel('x')",
        "import sympy\nsympy.together('x')",
        "import sympy\nsympy.solve('x - 1')",
        "from sympy import diff\ndiff('x', 'x')",
        "import sympy\nsympy.integrate('x', 'x')",
        "from sympy import limit\nlimit('x', 'x', 0)",
        "import sympy\nsympy.series('x')",
        "import sympy\ngetattr(sympy, 'simplify')('x')",
        "import sympy\nvars(sympy)['integrate']('x', 'x')",
    ]
    for source in snippets:
        assert _findings(source), f"not reported: {source!r}"


def test_the_scanner_leaves_ordinary_code_alone():
    ordinary = [
        "import re\nre.compile('x')",
        "from sympy import S as one\nvalue = one.One",
        "import sympy\nsympy.S.One + sympy.Symbol('S')",
        "N = 10\nprint(N)",
        "from fractions import Fraction\nFraction('1/2')",
        "def sympify_safely(value):\n    return value",
        "diff = 3 - 1\nprint(diff)",
        "limit = 5\nprint(limit + 1)",
        "def solve(x):\n    return x\nsolve(1)",
        "series = [1, 2]\nprint(series)",
        "import sympy\nsympy.Symbol('simplify')",
        "from fractions import Fraction\nFraction(1, 3).limit_denominator(10)",
    ]
    for source in ordinary:
        assert _findings(source) == set(), f"reported: {source!r}"


def test_the_code_names_no_evaluator_it_did_not_review():
    found: set = set()
    for path, relative in _guarded_files():
        scan = _Scan(relative)
        scan.visit(ast.parse(path.read_text(encoding="utf-8")))
        found |= scan.found
    unreviewed = sorted(found - set(_REVIEWED))
    assert unreviewed == [], (
        "code under mathgen/ names an evaluator that turns text into maths or runs it. Read text through "
        "mathgen.grading.safe_parse (parse_expression_safely for a reply, sympify_safely for the generators' "
        f"text), or review the use and list it in _REVIEWED with its reason: {unreviewed}"
    )
    stale = sorted(set(_REVIEWED) - found)
    assert stale == [], f"_REVIEWED lists a use that is gone; remove it: {stale}"
