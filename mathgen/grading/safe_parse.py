"""Parse a learner's typed expression without ever handing them Python.

SymPy's `parse_expr` compiles its input and `eval`s it, with the Python builtins and the whole of
`sympy` in scope. Typed answers are untrusted text, so an answer such as `__import__('os')...` or
`x.__class__...` would run code on the machine that grades it. `parse_expression_safely` is the
only way grading code should call `parse_expr` on text it did not write:

1. a length cap, so a pasted essay never reaches the tokenizer;
2. a token allow-list: names, numbers and the arithmetic operators only. No strings, no attribute
   access (`.` outside a number), no leading-underscore names, no Python keywords, no `=`/`:`/`;`/`@`
   and the like;
3. a namespace of mathematical functions and constants only, with `__builtins__` empty, so a name
   outside that list is simply a `Symbol` (that is how `x`, `theta` and `kN` are read) and never a
   function that can touch the machine.

4. a cost guard (`TooCostly`, below): before and after the parse the expression is measured, and one
   that SymPy could spend seconds on (a huge power, a tower of powers, a factorial of a big number,
   a deeply nested function, a product that expands to thousands of terms) is refused without ever
   being evaluated.

Refusals raise `ValueError` (`TooCostly` is a subclass), which every caller already treats as "not
parseable".

The guard bounds the shape of an expression, not the time SymPy takes to compare it, so an expression
built to stay inside every limit can still be slow. Measured by grading 87,905 random expressions against
a canonical answer (all the functions the parser knows, numbers up to 40 digits, nesting up to ten deep;
23,322 of up to 400 characters and 64,583 of up to 70, from eleven seeds), nine did not finish in 8 s and
five took over half a second, the slowest 5.4 s. None is a thing anyone answers (`sec((E - 2)^6)`,
`floor(ln(y)^275)`, `Max(ln(floor(x))^1000, pi)`, `sin(I)/e^182`), and each of the nine hangs the same way
under the limits this commit started from. Every fuzz round found hangs that the one before it had not, and each
of those was turned into a limit above; so more are there to be found. A shorter expression can do it too:
`sqrt(acos(x)^2)` never finishes, because the numeric probe puts a number outside [-1, 1] in for `x`, which
makes `acos(x)` a complex number, and `simplify` goes through `Abs` and `arg` for it. (The same with a
constant, `sqrt(log(asin(2) + x)^2)`, is refused: see `_outside_real_domain`.) A service that grades
untrusted replies should still give each call a deadline, for example by grading in a worker process it
can stop.

`sympify_safely` is the same reading for text the *generators* write (a canonical answer, a distractor):
it takes the same tokens and namespace, without the cost guard, and raises `SympifyError` like
`sympify` does, so the equality helpers in `mathgen.core` can accept text and still never run it.
"""

from __future__ import annotations

import io
import keyword
import math
import re
import tokenize

import sympy
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

#: Longest answer text parsed. The longest canonical answer in the library is well under 100.
MAX_LENGTH = 400

#: Operators a typed expression may contain. `^` is legal because some callers parse before
#: `^` -> `**`; `!` is the factorial transformation; `,` separates arguments and tuples.
_ALLOWED_OPS = frozenset({"+", "-", "*", "/", "**", "(", ")", ",", "!", "^"})

#: The only sympy names a typed answer can reach, as functions or constants. Anything else a
#: learner types is a plain `Symbol`.
_SYMPY_NAMES = (
    "pi E I oo "
    "sqrt cbrt root exp log ln Abs sign floor ceiling factorial gcd lcm Max Min "
    "re im conjugate arg "
    "sin cos tan sec csc cot asin acos atan asec acsc acot "
    "sinh cosh tanh sech csch coth asinh acosh atanh Rational"
).split()

#: Names parse_expr's own transformations emit (`Symbol('x')`, `Integer(2)`, `Function('f')(x)`, and
#: `Number('2')` for names like `x2`, and `Add`/`Mul`/`Pow` when it parses with `evaluate=False`), so they must resolve in the namespace.
#: A learner never types them, so typing one is refused.
_PLUMBING = ("Symbol", "Function", "Integer", "Float", "Number", "Add", "Mul", "Pow")
_CONSTANTS = ("True", "False", "None")


def _build_namespace() -> dict:
    space = {name: getattr(sympy, name) for name in (*_SYMPY_NAMES, *_PLUMBING) if hasattr(sympy, name)}
    space["ln"] = sympy.log
    # `abs(x)`, `max(a, b)` and `min(a, b)` were builtins before the builtins were taken away.
    space["abs"] = sympy.Abs
    space["max"] = sympy.Max
    space["min"] = sympy.Min
    space["__builtins__"] = {}
    return space


_NAMESPACE = _build_namespace()

#: Functions that SymPy evaluates the moment they are built, even in an unevaluated parse (it only holds
#: back arithmetic and the functions in `EvaluateFalseTransformer.functions`). The cost check parses them
#: as plain applications instead, so `(9!)!`, `gcd(huge, huge)` and `floor(2^(10^8))` are measured and
#: refused rather than computed.
_LAZY_FUNCTIONS = ("factorial", "root", "floor", "ceiling", "gcd", "lcm", "Max", "Min", "abs", "max", "min")


def _namespace(*, lazy: bool = False) -> dict:
    """A fresh namespace for one parse (`parse_expr` may add to the dict it is given)."""
    space = dict(_NAMESPACE)
    if lazy:
        for name in _LAZY_FUNCTIONS:
            space[name] = sympy.Function(name)
    return space


#: A number literal whose exponent has four or more digits (`1e500000`) takes seconds to build.
_HUGE_EXPONENT_RE = re.compile(r"[eE][+-]?\d{4,}")
_MAX_NUMBER_LENGTH = 40


def _check_tokens(text: str, *, max_length: int = MAX_LENGTH, extra_ops: frozenset = frozenset()) -> None:
    if len(text) > max_length:
        raise ValueError("answer text is too long to parse")
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise ValueError(f"cannot tokenise answer text: {exc}") from exc
    for tok in tokens:
        if tok.type in (tokenize.NAME,):
            name = tok.string
            if name.startswith("_") or name in _PLUMBING or (keyword.iskeyword(name) and name not in _CONSTANTS):
                raise ValueError(f"name {name!r} is not allowed in an answer")
        elif tok.type == tokenize.OP:
            if tok.string not in _ALLOWED_OPS and tok.string not in extra_ops:
                raise ValueError(f"operator {tok.string!r} is not allowed in an answer")
        elif tok.type == tokenize.NUMBER:
            if len(tok.string) > _MAX_NUMBER_LENGTH or _HUGE_EXPONENT_RE.search(tok.string):
                raise ValueError("number literal is too large to parse")
        elif tok.type not in (
            tokenize.NUMBER,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.ENDMARKER,
            tokenize.INDENT,
            tokenize.DEDENT,
        ):
            raise ValueError(f"token {tok.string!r} is not allowed in an answer")


# -- the cost guard -------------------------------------------------------------------------------
#
# A learner's text can be short and still ask SymPy for something that takes minutes or all the memory:
# `x^(10^10)`, `(x+1)^(10^6)`, `2^-2^-2^-...`, `(9!)!`, `((x+1)^2+1)^2...` nested twenty deep, a
# 30-digit radicand, `ln(ln(ln(...)))`. The checks below read the *shape* of the expression, which costs
# nothing, and refuse before anything is evaluated (the measuring parse never computes a power, a
# factorial or a gcd) and again on the parsed result, because the parse itself can turn a small tree
# into a product, and the comparison that follows (`subs`, `evalf`, `simplify`) is where the time goes.
#
# The library's own answers sit far inside every limit (measured over every canonical answer and
# distractor of the expression, pair and list skills: 43 nodes, depth 8, one transcendental function inside
# another, powers up to 72, numbers up to 13 digits, four symbols, at most 4 function calls on a variable).
# Each limit was then raised until the worst expression that fits is still graded in a few tens of
# milliseconds, so that a right answer a little past the library (`299792458t`, `x^300`, `sqrt(10^15)`,
# `(a+b+c+d)^10`) is not turned away, and the limits only refuse what costs SymPy seconds. They bound what
# SymPy is asked to do; they do not make `simplify` fast, and a crafted expression inside every limit can
# still be slow (see the "known limit" paragraph of the module docstring).

#: Longest run of nested operations (a path from the root to a leaf).
MAX_DEPTH = 30
#: Most nodes in the expression tree.
MAX_NODES = 150
#: Powers in exponent position inside one another: `a^b^c^d` is 3, `a^b^c^d^e` is 4.
MAX_POWER_TOWER = 3
#: Functions (and roots) applied inside one another: `sqrt(abs(floor(x)))` is 3, and each root counts as one.
#: Transcendental functions have a lower limit of their own, `MAX_TRANSCENDENTAL_DEPTH`.
MAX_FUNCTION_DEPTH = 4
#: Largest power a symbolic base may be raised to (`x^1000` is fine, `x^100000` is not).
MAX_SYMBOLIC_EXPONENT = 1000
#: Most monomials the expression could expand to. For a polynomial (sums, products and whole powers of symbols and
#: numbers, nothing else) they are counted from its terms as `_Walker` goes: `(a+b+c+d)^15` is 816, `(a+b)^100` is
#: 101, `(a+b)^100 * (c+d)^100` is 10,201, thirteen linear factors in one variable are 14. Anything with a function
#: of a symbol, a root or a variable exponent in it is counted as the monomials its degree allows in all its
#: symbols (`(a+b+c+d)^15` is 3,876), because SymPy's work on those depends on more than the terms.
MAX_EXPANDED_TERMS = 5000
#: Highest degree anything may have. A power of a power of a power multiplies the exponents, and `((x/27)^272)^997`
#: is `x^271184` over a number of 388,000 digits; `x^1000` is the largest a single power may be.
MAX_DEGREE = 5000
#: A number stays below this many digits wherever it is built (the old Python int-to-str limit is 4300).
MAX_NUMERIC_DIGITS = 4000
#: Digits of a number written next to a variable (`299792458t` has nine, `6.022e23*n` has 24). It is also the
#: longest number `_check_tokens` reads at all, so a coefficient is one literal. A coefficient this long makes
#: no difference to the comparison: 40-digit semiprimes beside a variable are graded in tens of milliseconds,
#: equal forms that need `simplify` included. The library's own answers reach four digits. Every limit on
#: "digits" below is on log10 of the number plus one, so 40 allows every number below 10^39.
MAX_COEFFICIENT_DIGITS = 40
#: The same for a number with pi, e, a root or a function in it, whose digits are only an upper bound: `y*(3 - 10*pi)^6`
#: is counted as 22 digits (it is about 6e8), and `sec(exp(y*(3 - 10*pi)^6))` is the secant of a number with 600
#: million digits. This stays at the old limit of 19.
MAX_IRRATIONAL_COEFFICIENT_DIGITS = 19
#: Largest value a number may have as the argument of `exp`, `sinh`, `cosh`, `sech` or `csch`, whose values
#: grow or shrink like `exp` of the argument. `exp(849146628)` is a number with 368 million digits, `floor` of
#: it never finishes, and `cosh(4096)` is too big for a float, so the numeric probe skips it and `simplify`
#: is left with a number it cannot handle (`sech(10^30)` never finishes either). `exp(700)` is about 1e304,
#: the largest that a float holds; `exp(701)`, `exp(-701)` and `cosh(750)` are refused with it.
MAX_TRANSCENDENTAL_ARGUMENT = 700
#: The same for every other trig, hyperbolic and inverse function, whose values stay bounded, repeat or grow
#: like a logarithm: `sin(720)`, `tan(765)`, `atan(1000)`, `tanh(1000)` are ordinary answers. SymPy reduces
#: `sin(10^100)` with pi to a hundred digits: graded in a few tens of milliseconds, as are the others at
#: this size (measured for sin cos tan cot sec csc atan acot tanh coth asinh acosh over a dozen shapes each).
MAX_BOUNDED_ARGUMENT = 10**12
#: Transcendental functions (and powers with a variable in the exponent, which are `exp` of a product) applied
#: inside one another: `sin(ln(x))` and `e^(sin x)` are 2, `ln(sin(e^x))` and `sin(cos(sin(x)))` are 3. The
#: library never goes past two; the chain rule on a composite of three (`cos(sin(sin(x)))*cos(sin(x))*cos(x)`)
#: is the deepest answer anyone writes. A level more is where `simplify` stops finishing over two variables.
MAX_TRANSCENDENTAL_DEPTH = 3
#: Digits of the longest number written anywhere inside a chain of three (log10 plus one, so 5 allows numbers below
#: 10^4). The chain rule has small whole numbers in it; the probe puts a large value through three functions
#: (`cos(sin(864076265/sqrt(z)))` at a negative `z` is the cosine of the sine of a huge imaginary number).
MAX_NESTED_LITERAL_DIGITS = 5.0
#: Digits a number that has pi, e, a root or a function in it may have (the digits of a float: `cosh(700)^2`
#: is too big to evaluate at a probe point, and `simplify` is left with it).
MAX_FLOAT_DIGITS = 306
#: Largest whole power of a sum with an irrational constant in it (`(pi + 11)^50`): SymPy expands it, term by
#: term. The same for a function value (`asin(1)^1000000` is `(pi/2)^1000000`, and the value is not known from
#: here). A single constant or a product is not expanded and is not limited this way (`pi^25`, `sqrt(2)^100`,
#: `(2*pi)^30`; `sqrt(2)^100` is the whole number 2^50).
MAX_IRRATIONAL_POWER = 20
#: Largest factorial argument (1000! has 2568 digits).
MAX_FACTORIAL_ARGUMENT = 1000
#: Highest degree of the argument of a trigonometric function when it has a variable in it and is a polynomial.
#: The numeric probe puts a number of about 2 in for the variable, so `sin(x^1000)` asks for the sine of a 340-digit
#: number, and that is graded in a few milliseconds (so is `sin((99999x)^400)`).
MAX_PERIODIC_DEGREE = 1000
#: The same when the argument has a function or a root in it (`tan(sqrt(P^3))`, `cot(cos(P^30)^10)`): the probe
#: takes the root of a negative number or the cosine of a large one, and the function outside has to reduce that.
MAX_PERIODIC_COMPOSED_DEGREE = 40
#: The same when the argument has a factorial of a variable in it, which grows far faster than its degree:
#: `tan(((12z)!)^16)` asks for the tangent of a number with 540 digits and takes 0.2 s, `^40` 3.5 s, and
#: `tan(((12z)!)^653)` has 19,000 digits.
MAX_PERIODIC_FACTORIAL_DEGREE = 8
#: Highest degree of what `Abs`, `re`, `im`, `arg`, `sign` or `conjugate` may be applied to when it has a
#: variable in it: SymPy splits `(a + b*i)^n` into real and imaginary parts term by term, and
#: `Abs(exp(cos(t)^1500))` never finishes. `Abs(exp(cos(t)^100))` takes 0.2 s.
MAX_PART_DEGREE = 100
#: Highest degree a polynomial gcd or lcm may be asked for (SymPy computes it).
MAX_GCD_DEGREE = 20
#: Digits of an integer allowed under a fractional power: SymPy pulls the roots out of it with a bounded search
#: for factors, so the size hardly matters (`cbrt(10^21)`, `sqrt(2^100)`; semiprimes of 144 digits under
#: `sqrt`, `cbrt` and `root`, alone or in a sum, a product or a nested root, are graded in 15 ms).
MAX_RADICAND_DIGITS = 100
#: Functions applied to something with a variable in it (`sin(7x - 6)` is one).
MAX_SYMBOLIC_CALLS = 12
#: Roots of something with a variable in it (`sqrt(1 - x^2)` is one).
MAX_SYMBOLIC_ROOTS = 6
#: Powers with a variable in the exponent (`2^x`, `e^(2x)`).
MAX_VARIABLE_POWERS = 4
#: Degree of an exponent that has a variable in it (`e^(x^2/2)` is 2; `3^(x*y^50)` is 51).
MAX_EXPONENT_DEGREE = 4
#: Degree of the argument of a factorial of something with a variable in it (`(n+1)!`, `(2n)!`).
MAX_FACTORIAL_DEGREE = 2
#: Factorials of something with a variable in it (`n!/(k!(n-k)!)` is three).
MAX_SYMBOLIC_FACTORIALS = 4
#: Digits of the longest number inside such a factorial: SymPy expands `(n+733)!` into 733 factors.
MAX_FACTORIAL_LITERAL_DIGITS = 3
#: Digits of the longest number written in an exponent that has a variable in it (`2^(12t)`, `e^(-45t)`).
MAX_EXPONENT_LITERAL_DIGITS = 5
#: Digits a number may have beside pi, e, i, a root or a transcendental function in one sum or product.
MAX_IRRATIONAL_DIGITS = 100

# sympy evaluates a factorial even in an unevaluated parse, so a big one is also refused on the text.
# Anchored at the start of a digit run (`(?<!\d)`), so a long run of digits is scanned once, not once per digit.
_HUGE_FACTORIAL_RE = re.compile(r"(?<!\d)\d{4,}\s*!")


#: Functions of whole numbers that give whole numbers (or hold their arguments' kind): not a source of
#: irrational constants. A `Tuple` is how a coordinate pair arrives.
_EXACT_FUNCTIONS = frozenset(
    "factorial floor ceiling gcd lcm Abs abs Max Min max min sign Tuple re im conjugate".split()
)
#: Functions of an angle: SymPy reduces the argument modulo pi, which takes digits of pi as long as the argument.
_PERIODIC_FUNCTIONS = frozenset("sin cos tan cot sec csc".split())
#: Functions that take a complex number apart into its real and imaginary parts.
_PART_FUNCTIONS = frozenset("Abs abs re im arg sign conjugate".split())
_TRANSCENDENTAL_FUNCTIONS = frozenset(
    "exp sin cos tan cot sec csc asin acos atan acot asec acsc sinh cosh tanh coth sech csch asinh acosh atanh".split()
)
#: What counts as one level of `MAX_TRANSCENDENTAL_DEPTH`.
_NESTING_FUNCTIONS = _TRANSCENDENTAL_FUNCTIONS | {"log"}
#: Functions whose value grows like `exp` of the argument, so that a symbol in the argument is the same as
#: a symbol in an exponent, and a number in it sets the size of the result.
_GROWTH_FUNCTIONS = frozenset("exp sinh cosh tanh coth sech csch".split())


#: Functions whose value grows or shrinks like `exp` of the argument, so that a number in the argument sets the
#: size of the result (see `MAX_TRANSCENDENTAL_ARGUMENT`). Every other transcendental function is bounded,
#: repeats or grows like a logarithm (`MAX_BOUNDED_ARGUMENT`).
_EXPONENTIAL_VALUE_FUNCTIONS = frozenset("exp sinh cosh sech csch".split())
#: A real number this close to the edge of a function's domain counts as inside it (`evalf` carries 15 digits).
_DOMAIN_SLACK = 1e-9


def _outside_real_domain(name: str, value: float) -> bool:
    """Whether the inverse function `name` has no real value at the real number `value`. SymPy keeps
    `asin(2)` as it is, a complex number it cannot say the real and imaginary parts of without working them
    out, and `simplify` of anything built on one (`sqrt(log(asin(2) + x)^2)`) never finishes."""
    if name in ("asin", "acos", "atanh"):
        return abs(value) > 1 + _DOMAIN_SLACK
    if name == "acosh":
        return value < 1 - _DOMAIN_SLACK
    if name in ("asec", "acsc"):
        return abs(value) < 1 - _DOMAIN_SLACK
    return False


def _function_name(node) -> str:
    return getattr(getattr(node, "func", None), "__name__", "")


def _has_function(node, name: str) -> bool:
    return any(_function_name(part) == name for part in sympy.preorder_traversal(node))


def _has_transcendental_constant(node) -> bool:
    """Whether a symbol-free subtree has pi, e or a function in it: a number that is not algebraic."""
    for part in sympy.preorder_traversal(node):
        if part is sympy.pi or part is sympy.E:
            return True
        if not part.is_Atom and not isinstance(part, (sympy.Add, sympy.Mul, sympy.Pow)):
            return True
    return False


def _has_function_call(node) -> bool:
    """Whether a subtree has a function applied to something in it (`sin(2)`, `asin(1)`, `floor(7.5)`), as
    opposed to numbers, pi, e and roots, which are measured by their digits."""
    return any(
        not part.is_Atom and not isinstance(part, (sympy.Add, sympy.Mul, sympy.Pow))
        for part in sympy.preorder_traversal(node)
    )


def _expands(node) -> bool:
    """Whether raising `node` to a whole power makes SymPy multiply a sum out: it is a sum, or a product or
    a whole power of one (an integer power of a product is distributed over its factors)."""
    if isinstance(node, sympy.Add):
        return True
    if isinstance(node, sympy.Mul):
        return any(_expands(arg) for arg in node.args)
    if isinstance(node, sympy.Pow) and node.args[1].is_Integer:
        return _expands(node.args[0])
    return False


def _is_irrational_constant(node) -> bool:
    """A symbol-free subtree that is not built from whole numbers and +, -, *, / and whole powers alone:
    it has pi, e, i, a function or a root in it."""
    for part in sympy.preorder_traversal(node):
        if part.is_Atom:
            if not part.is_Number:
                return True
        elif isinstance(part, sympy.Pow):
            exponent = part.args[1]
            if not (exponent.is_Number and exponent.is_Integer) and not (
                exponent.is_Mul and exponent.args[0] == -1 and exponent.args[1].is_Integer
            ):
                return True
        elif not isinstance(part, (sympy.Add, sympy.Mul)):
            return True
    return False


class TooCostly(ValueError):
    """The expression is too big or too deeply nested to evaluate safely."""


class _Summary:
    """What the cost check needs to know about one subtree."""

    __slots__ = (
        "digits",
        "degree",
        "tower",
        "depth",
        "literal",
        "irrational",
        "tdepth",
        "terms",
        "poly",
        "root",
        "powers",
    )

    def __init__(
        self,
        digits=None,
        degree=0,
        tower=0,
        depth=0,
        literal=0.0,
        irrational=False,
        tdepth=0,
        terms=1,
        poly=True,
        root=False,
        powers=0,
    ):
        self.digits = digits  # upper bound on the digits of a symbol-free value, else None
        self.degree = degree  # upper bound on the polynomial degree in the symbols
        self.tower = tower  # powers nested in exponent position
        self.depth = depth  # functions and roots nested inside one another
        self.literal = literal  # digits of the longest number written anywhere inside
        self.irrational = irrational  # has pi, e, i, a root or a transcendental function inside
        self.tdepth = tdepth  # transcendental functions and variable-exponent powers nested inside one another
        self.terms = terms  # upper bound on the monomials it expands to (1 for anything without a variable)
        self.poly = poly  # a polynomial in its symbols: only sums, products and whole powers of them
        self.root = root  # a power whose exponent is not a whole number
        self.powers = powers  # exponentials (`exp`, a power with a variable in the exponent) nested inside one another


def _number_digits(node) -> float:
    try:
        value = float(node)
    except OverflowError:
        raise TooCostly("number is too large") from None
    except (TypeError, ValueError):
        return 1.0
    if value == 0 or not math.isfinite(value):  # zero, oo, nan
        return 1.0
    return math.log10(abs(value)) + 1


def _numeric_value(node) -> complex:
    """The value of a symbol-free subtree, as a complex float."""
    if node.has(AppliedUndef):
        # `factorial`, `floor`, `gcd` and the rest are held back in the measuring parse, as functions with no value.
        # SymPy cannot say what they come to, and tries anyway: `factorial(1)**500` is split into real and
        # imaginary parts by multiplying out (a + b*i)^500, which takes minutes. Such a number is not measured.
        raise TooCostly("number is too large")
    try:
        value = complex(node.evalf(15))
    except (OverflowError, ValueError, TypeError, ZeroDivisionError):
        raise TooCostly("number is too large") from None
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise TooCostly("number is too large")
    return value


def _binomial(n: int, k: int) -> int:
    """`math.comb(n, k)`, except that it stops, saying one more than `MAX_EXPANDED_TERMS`, once it is past it."""
    cap = MAX_EXPANDED_TERMS + 1
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
        if result > cap:
            return cap
    return result


def _monomials(degree: int, symbols: int) -> int:
    """Most monomials a polynomial of this degree in this many symbols can have."""
    return _binomial(min(degree, 10**6) + symbols, symbols)


class _Walker:
    """One pass over an expression tree, raising `TooCostly` at the first limit it passes."""

    def __init__(self):
        self.nodes = 0
        self.calls = 0  # functions applied to something with a variable in it
        self.roots = 0  # roots of something with a variable in it
        self.variable_powers = 0  # powers with a variable in the exponent
        self.factorials = 0  # factorials of something with a variable in it

    def walk(self, node, depth: int = 0) -> _Summary:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise TooCostly("expression has too many parts")
        if depth > MAX_DEPTH:
            raise TooCostly("expression is nested too deeply")
        if node.is_Atom:
            if node.is_Number:
                digits = _number_digits(node)
                return _Summary(digits, literal=digits)
            return _Summary(None, 1) if node.free_symbols else _Summary(1.5, irrational=True)  # pi, E, I, oo
        parts = [self.walk(arg, depth + 1) for arg in node.args]
        summary = self.node(node, parts, depth)
        if summary.terms > MAX_EXPANDED_TERMS:
            raise TooCostly("expression would expand to too many terms")
        if summary.degree > MAX_DEGREE:
            raise TooCostly("expression has too high a degree")
        summary.literal = max([summary.literal, *(p.literal for p in parts)])
        summary.irrational = summary.irrational or any(p.irrational for p in parts)
        summary.tdepth = max((p.tdepth for p in parts), default=0) + self.nests(node)
        if summary.root and parts[0].tdepth:
            # A root of something with a transcendental function in it is one more level: the probe puts a
            # number outside the domain into the function inside (`asin(11/5)`), and the root of that complex
            # number is where SymPy's assumptions queries go round and round
            # (`exp((y*ln(asin(x)))^(1/2))` is four levels, and never finishes).
            summary.tdepth += 1
        if summary.tdepth > MAX_TRANSCENDENTAL_DEPTH:
            raise TooCostly("transcendental functions are nested too deeply")
        summary.powers = max((p.powers for p in parts), default=0) + self.exponentials(node)
        if summary.tdepth > 2 and summary.literal > MAX_NESTED_LITERAL_DIGITS:
            raise TooCostly("large number inside functions nested three deep")
        if summary.tdepth > 2 and summary.powers > 1:
            # `exp(((x^(1/8))^z)^((x - 12)^-3))`: exponentials inside one another (`exp`, or a power with a variable in
            # the exponent) are a double exponential, and the next function out has to take a value with as many digits
            # as that: `sec(exp(y*6e8))` is the secant of a number with 600 million digits.
            raise TooCostly("exponentials inside one another, inside a third function")
        if summary.tdepth > 1 and node.has(sympy.I):
            # `sin(i*x)` is a hyperbolic function: grows like exp, and the next function out has to reduce a
            # number with as many digits as that (`sec(sin((883*i + x)^9))` never finishes).
            raise TooCostly("transcendental functions nested around the imaginary unit")
        if summary.irrational and summary.digits is not None and summary.digits > MAX_FLOAT_DIGITS:
            raise TooCostly("number is too large to evaluate")
        return summary

    @staticmethod
    def exponentials(node) -> int:
        """1 for `exp` of something with a variable in it, or a power with a variable in the exponent, else 0."""
        if isinstance(node, sympy.Pow):
            return 1 if node.args[1].free_symbols else 0
        return 1 if getattr(getattr(node, "func", None), "__name__", "") == "exp" and node.free_symbols else 0

    @staticmethod
    def nests(node) -> int:
        """1 for a node that counts as a level of `MAX_TRANSCENDENTAL_DEPTH`, else 0."""
        if isinstance(node, sympy.Pow):
            return 1 if node.args[1].free_symbols else 0
        return 1 if getattr(getattr(node, "func", None), "__name__", "") in _NESTING_FUNCTIONS else 0

    def node(self, node, parts: list, depth: int) -> _Summary:
        symbolic = bool(node.free_symbols)
        if symbolic and any(
            p.digits is not None
            and p.digits > (MAX_IRRATIONAL_COEFFICIENT_DIGITS if p.irrational else MAX_COEFFICIENT_DIGITS)
            for p in parts
        ):
            raise TooCostly("number is too large next to a variable")
        tower = max((p.tower for p in parts), default=0)
        nested = max((p.depth for p in parts), default=0)
        if isinstance(node, sympy.Pow):
            return self.power(node, parts[0], parts[1])
        if isinstance(node, (sympy.Add, sympy.Mul)) and not symbolic and any(p.irrational for p in parts):
            # 10^200 + pi is a 200-digit sum that has to be carried to the digit: nothing asks for it
            if any(p.digits is not None and p.digits > MAX_IRRATIONAL_DIGITS for p in parts):
                raise TooCostly("large number next to an irrational constant")
        if isinstance(node, sympy.Add):
            degree = max((p.degree for p in parts), default=0)
            summary = _Summary(None if symbolic else max(p.digits for p in parts) + 1, degree, tower, nested)
            return self.expanded(node, parts, summary, sum(p.terms for p in parts))
        if isinstance(node, sympy.Mul):
            digits = None if symbolic else sum(p.digits for p in parts)
            self.bound(digits)
            summary = _Summary(digits, sum(p.degree for p in parts), tower, nested)
            terms = 1
            for p in parts:
                terms = min(terms * p.terms, MAX_EXPANDED_TERMS + 1)
            return self.expanded(node, parts, summary, terms)
        summary = self.function(node, parts, symbolic, tower, nested + 1)
        if getattr(getattr(node, "func", None), "__name__", "") not in _EXACT_FUNCTIONS:
            summary.irrational = True
        # A function is one term. What is inside it is expanded on its own, so it is the largest of those.
        summary.terms = max((p.terms for p in parts), default=1)
        summary.poly = not symbolic  # a function of a variable is a generator of its own, not a polynomial
        return summary

    @staticmethod
    def expanded(node, parts: list, summary: _Summary, terms: int) -> _Summary:
        """Set how many monomials a sum or product of `parts` has once multiplied out. A number is one term
        however it is written, and a polynomial has no more than the monomials of its degree."""
        symbols = len(node.free_symbols)
        if not symbols:
            return summary
        summary.poly = all(p.poly for p in parts)
        summary.terms = min(terms, _monomials(summary.degree, symbols)) if summary.poly else terms
        return summary

    @staticmethod
    def bound(digits) -> None:
        if digits is not None and digits > MAX_NUMERIC_DIGITS:
            raise TooCostly("number is too large")

    def power(self, node, base: _Summary, exponent: _Summary) -> _Summary:
        base_node, exp_node = node.args
        tower = max(base.tower, 1 + exponent.tower)
        if tower > MAX_POWER_TOWER:
            raise TooCostly("powers are stacked too deeply")
        if exp_node.free_symbols:
            if exponent.literal > MAX_EXPONENT_LITERAL_DIGITS:
                raise TooCostly("large number in an exponent with a variable in it")
            self.variable_powers += 1
            if self.variable_powers > MAX_VARIABLE_POWERS:
                raise TooCostly("too many powers with a variable in the exponent")
            if exponent.degree > MAX_EXPONENT_DEGREE:
                raise TooCostly("exponent of too high a degree")
            if not base_node.free_symbols and not base_node.is_Rational and _has_transcendental_constant(base_node):
                # `sec(2)^x` is a negative number to a variable power: complex, and SymPy has to pick the
                # branch of the logarithm to say anything about it (`Abs((sec(2)^x + atan(x))^(1/2))` never
                # finishes). `(-2)^n` and `(1 - sqrt(5))^n` are fine: whole or algebraic.
                base_value = _numeric_value(base_node)
                if base_value.imag == 0 and base_value.real < 0:
                    raise TooCostly("negative irrational base with a variable exponent")
            return _Summary(None, base.degree, tower, max(base.depth, exponent.depth), terms=base.terms, poly=False)
        value = _numeric_value(exp_node)
        magnitude = abs(value)
        integer_exponent = value.imag == 0 and float(value.real).is_integer()
        # A root (a power that is not a whole number) is one more level of nesting, like a function.
        depth = max(base.depth, exponent.depth) + (0 if integer_exponent else 1)
        irrational = base.irrational or not integer_exponent
        if depth > MAX_FUNCTION_DEPTH:
            raise TooCostly("roots are nested too deeply")
        if base_node.free_symbols:
            if magnitude > MAX_SYMBOLIC_EXPONENT:
                raise TooCostly("power is too large")
            if not integer_exponent:
                self.roots += 1
                if self.roots > MAX_SYMBOLIC_ROOTS:
                    raise TooCostly("too many roots of a variable")
            # A number written in the base is raised to the power too: `(733*z)^3` has the coefficient 733^3.
            literal = magnitude * max(base.literal - 1.0, 0.0) + 1.0 if base.literal > 1.0 else 0.0
            degree = base.degree * max(1, math.ceil(magnitude))
            terms, poly = base.terms, False
            if integer_exponent:
                # (a + b + c)^n multiplies out to the ways of choosing n of the terms with repeats; a polynomial
                # never has more monomials than its degree allows in its symbols.
                whole = int(magnitude)
                if base.terms > 1:
                    terms = _binomial(whole + base.terms - 1, base.terms - 1)
                poly = base.poly and value.real >= 0
                if poly:
                    terms = min(terms, _monomials(degree, len(node.free_symbols)))
            return _Summary(
                None,
                degree,
                tower,
                depth,
                literal=literal,
                irrational=irrational,
                terms=terms,
                poly=poly,
                root=not integer_exponent,
            )
        # A sum is multiplied out; a constant or a product is not (`pi^25`, `sqrt(2)^100`). The value of a function
        # is not known from here (`asin(1)` is pi/2, and its millionth power takes seconds to build).
        if (
            magnitude > MAX_IRRATIONAL_POWER
            and (_expands(base_node) or _has_function_call(base_node))
            and _is_irrational_constant(base_node)
        ):
            raise TooCostly("large power of a sum or a function value with an irrational constant")
        # A number to a number: `base_digits` is log10 of the base plus one, so the result has about
        # |exponent| * log10(base) + 1 digits (10^15 has 16).
        base_digits = base.digits if base.digits is not None else 1.0
        size = magnitude * max(base_digits - 1.0, 0.0) + 1.0
        if size > MAX_NUMERIC_DIGITS:
            raise TooCostly("power is too large")
        if not integer_exponent and base_digits > MAX_RADICAND_DIGITS:
            raise TooCostly("root of a large number")
        # A whole number written as a power (`10**7*x`) is a number of that size, as it is once evaluated.
        return _Summary(
            size, 0, tower, depth, literal=0.0 if irrational else size, irrational=irrational, root=not integer_exponent
        )

    def function(self, node, parts: list, symbolic: bool, tower: int, nested: int) -> _Summary:
        """Any node that is not +, * or a power is a function (or a tuple of values), one level deeper."""
        if nested > MAX_FUNCTION_DEPTH:
            raise TooCostly("functions are nested too deeply")
        degree = max((p.degree for p in parts), default=0)
        name = getattr(getattr(node, "func", None), "__name__", "")
        if symbolic:
            self.calls += 1
            if self.calls > MAX_SYMBOLIC_CALLS:
                raise TooCostly("too many functions of a variable")
        if name in _TRANSCENDENTAL_FUNCTIONS and not symbolic and len(parts) == 1:
            argument = _numeric_value(node.args[0])
            if abs(argument) > (
                MAX_TRANSCENDENTAL_ARGUMENT if name in _EXPONENTIAL_VALUE_FUNCTIONS else MAX_BOUNDED_ARGUMENT
            ):
                raise TooCostly("argument of a transcendental function is too large")
            if argument.imag == 0 and _outside_real_domain(name, argument.real):
                raise TooCostly("inverse function of a number outside its real domain")
            if name in ("exp", "sinh", "cosh"):
                # The value has about |argument| / ln(10) digits (exp(700) is 304 digits long), not the digits
                # of the argument: this is what a sum or product of these is measured by.
                return _Summary(abs(argument.real) / math.log(10) + 1.0, degree, tower, nested)
        if name in _GROWTH_FUNCTIONS and symbolic and len(parts) == 1:
            if parts[0].literal > MAX_EXPONENT_LITERAL_DIGITS:
                raise TooCostly("large number in the argument of an exponential function")
            if parts[0].degree > MAX_EXPONENT_DEGREE:  # `exp(x^9)`, as `e^(x^9)` already was
                raise TooCostly("argument of an exponential function of too high a degree")
        if name in _PERIODIC_FUNCTIONS and symbolic and len(parts) == 1:
            if _has_function(node.args[0], "factorial"):
                limit = MAX_PERIODIC_FACTORIAL_DEGREE
            else:
                limit = MAX_PERIODIC_DEGREE if parts[0].poly else MAX_PERIODIC_COMPOSED_DEGREE
            if parts[0].degree > limit:
                raise TooCostly("argument of a trigonometric function of too high a degree")
        if name in _PART_FUNCTIONS and symbolic and parts and parts[0].degree > MAX_PART_DEGREE:
            raise TooCostly("real or imaginary part of an expression of high degree")
        if name == "factorial" and len(parts) == 1:
            if symbolic:
                self.factorials += 1
                if self.factorials > MAX_SYMBOLIC_FACTORIALS:
                    raise TooCostly("too many factorials of a variable")
                if degree > MAX_FACTORIAL_DEGREE:
                    raise TooCostly("factorial of a polynomial of high degree")
                if parts[0].depth > 0 or parts[0].irrational:  # tan(11), sqrt(x) or pi inside: nothing to hold on to
                    raise TooCostly("factorial of a function")
                if parts[0].literal > MAX_FACTORIAL_LITERAL_DIGITS:  # (n+733)! is expanded to 733 factors
                    raise TooCostly("factorial of a variable plus a large number")
                return _Summary(None, degree, tower, nested)
            value = _numeric_value(node.args[0])
            if value.imag:
                raise TooCostly("factorial of a complex number")  # (i)! is a gamma function at a complex point
            n = abs(value)
            if n > MAX_FACTORIAL_ARGUMENT:
                raise TooCostly("factorial of a large number")
            return _Summary(n * math.log10(max(n, 2.0)), 0, tower, nested)
        if name == "root" and len(parts) == 2 and not node.args[0].free_symbols:
            # root(a, n) is a ** (1/n): the same limits as a root written with a power.
            if parts[0].digits is not None and parts[0].digits > MAX_RADICAND_DIGITS:
                raise TooCostly("root of a large number")
        if name in ("gcd", "lcm") and symbolic:
            if degree > MAX_GCD_DEGREE:
                raise TooCostly("polynomial gcd of a large degree")
            if any(
                p.irrational or p.depth > 0 for p in parts
            ):  # lcm(558, exp(c*cot(z)^3)) builds a polynomial of degree c
                raise TooCostly("gcd of something that is not a polynomial")
        digits = None if symbolic else sum(p.digits for p in parts) if parts else 1.0
        self.bound(digits)
        return _Summary(digits, degree, tower, nested)


def _is_bare_infinity(expr) -> bool:
    """`oo` or `-oo` on its own: the one place an infinity is an answer rather than an input.

    Asks only atoms (`is_infinite` on a power or a function is an assumptions query, which can itself take
    seconds: `ln(2*pi/e + sqrt(asin(6)))^2` did)."""
    if expr.is_Atom:
        return bool(expr.is_infinite)
    return (
        isinstance(expr, sympy.Mul)
        and len(expr.args) == 2
        and expr.args[0] == -1
        and expr.args[1].is_Atom
        and bool(expr.args[1].is_infinite)
    )


def check_expression_cost(expr) -> None:
    """Raise `TooCostly` unless `expr` (a SymPy expression, or a tuple/list of them) is small enough to
    evaluate and compare. Looks only at the tree's shape; evaluates nothing."""
    if isinstance(expr, (tuple, list)):
        for item in expr:
            check_expression_cost(item)
        return
    if not isinstance(expr, sympy.Basic):
        return
    try:
        summary = _Walker().walk(expr)
    except RecursionError:
        raise TooCostly("expression is nested too deeply") from None
    symbols = len(expr.free_symbols)
    if symbols and not summary.poly and math.comb(min(summary.degree, 10**6) + symbols, symbols) > MAX_EXPANDED_TERMS:
        raise TooCostly("expression would expand to too many terms")
    # An infinity or nan inside a sum, a product or a function sends trigsimp and friends down paths
    # that do not end (sec(oo*x) ...), and no answer is written that way.
    if not _is_bare_infinity(expr) and any(getattr(a, "is_infinite", False) or a is sympy.nan for a in expr.atoms()):
        raise TooCostly("infinity inside an expression")


def _parse(text: str, transformations, evaluate: bool, *, lazy: bool = False):
    """The raw `parse_expr` call. `text` has already passed `_check_tokens`."""
    try:
        return parse_expr(
            text,
            local_dict={},
            global_dict=_namespace(lazy=lazy),
            transformations=transformations,
            evaluate=evaluate,
        )
    except NameError as exc:  # a name the transformations emit but the namespace lacks (e.g. `!!`)
        raise ValueError(f"cannot parse answer text: {exc}") from exc


def too_costly(text: str, *, transformations) -> bool:
    """Whether evaluating `text` could cost SymPy seconds or a lot of memory.

    Parses it *unevaluated* (so no power or factorial is computed) and measures the tree. Text that does
    not parse at all, or that the token check refuses, is not "costly": the real parse reports that
    itself, and for a number too large to print this says so.
    """
    text = text.strip()
    if len(text) > MAX_LENGTH:
        return False  # too long to read at all: `parse_expression_safely` refuses it on the length
    if _HUGE_FACTORIAL_RE.search(text):
        return True
    try:
        _check_tokens(text)
        tree = _parse(text, transformations, evaluate=False, lazy=True)
    except ValueError as exc:
        return "digits" in str(exc)  # an integer too large to print
    except Exception:  # noqa: BLE001 - the real parse below reports its own failures
        return False
    try:
        check_expression_cost(tree)
    except TooCostly:
        return True
    except (TypeError, ValueError, AttributeError):
        return False
    return False


def parse_expression_safely(
    text: str,
    *,
    transformations,
    evaluate: bool = True,
    max_length: int = MAX_LENGTH,
    extra_ops: frozenset = frozenset(),
    check_cost: bool = True,
):
    """`parse_expr(text, transformations=...)`, restricted as the module docstring describes.

    Raises `ValueError` for text outside the allowed token set, and `TooCostly` (a `ValueError`) for an
    expression `check_expression_cost` refuses, before or after evaluating it. `evaluate=False` is the
    measuring parse and is never cost-checked. Anything `parse_expr` itself raises for malformed maths
    (`SympifyError`, `SyntaxError`, `TypeError`, `IndexError`, ...) propagates unchanged so callers keep
    their existing handling.
    """
    # sympy tokenises the stripped text, so check exactly that: Python 3.12+ keeps a leading NBSP or
    # ideographic space inside a NAME token, which would hide a leading underscore from the check.
    text = text.strip()
    _check_tokens(text, max_length=max_length, extra_ops=extra_ops)
    guard = check_cost and evaluate
    if guard and too_costly(text, transformations=transformations):
        raise TooCostly("expression is too costly to evaluate")
    result = _parse(text, transformations, evaluate)
    if guard:
        check_expression_cost(result)
    return result


#: What `sympify` reads, minus the exotic: `^` is a power, and nothing else changes.
SYMPIFY_TRANSFORMATIONS = standard_transformations + (convert_xor,)
#: Longest text `sympify_safely` reads. Generators write the text, so this is a sanity limit.
_SYMPIFY_MAX_LENGTH = 2000


def sympify_safely(value, *, evaluate: bool = True):
    """`sympy.sympify(value)`, except that text is read by this module's restricted parser.

    For the strings *generators* write (a canonical answer, a distractor): the same token allow-list and
    namespace as a learner's reply, so nothing in the text can be run, but no cost guard, because those
    strings are small and ours. Lists, tuples, dicts and sets are read item by item. Raises `SympifyError`, as
    `sympify` does, for text it cannot read; learner text should go through `parse_expression_safely`.
    """
    if isinstance(value, str):
        try:
            return parse_expression_safely(
                value,
                transformations=SYMPIFY_TRANSFORMATIONS,
                evaluate=evaluate,
                max_length=_SYMPIFY_MAX_LENGTH,
                # `[...]` and `{...}` are lists and sets of numbers and names. A set display cannot call anything,
                # and without a `:` or a `for` (both refused) it cannot be a dict or a comprehension either.
                extra_ops=frozenset({"[", "]", "{", "}"}),
                check_cost=False,
            )
        except sympy.SympifyError:
            raise
        except Exception as exc:  # noqa: BLE001 - sympify reports every unreadable text as a SympifyError
            raise sympy.SympifyError(value) from exc
    if isinstance(value, (tuple, list)):
        return type(value)(sympify_safely(item, evaluate=evaluate) for item in value)
    # SymPy `eval`s every string inside a dict or a set, so those go through the safe reader item by item too.
    if isinstance(value, dict):
        return {sympify_safely(k, evaluate=evaluate): sympify_safely(v, evaluate=evaluate) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return type(value)(sympify_safely(item, evaluate=evaluate) for item in value)
    return sympy.sympify(value, evaluate=evaluate)
