"""Primary-number decimal/fraction generators (group A): two more generators for two genuinely new graded operations in
the ``primary-number`` domain, both framework ``acara-f10-v9``.

- ``fraction_halving`` (AC9M2N03 and AC9M2M02 -- one skill, two target
  areas -- prerequisite ``fraction-arithmetic``) -- partition a whole or a
  fraction into halves, quarters and eighths by repeated halving. Every question produces a fraction answer via
  **halving** -- never an operation on two supplied fractions (general
  fraction operations are out of scope at every tier).

  - ``forward`` (``variant="forward"``, used at ``easy``/``medium``): start
    from a fraction with a simple denominator in ``{1, 2}`` (i.e. ``1`` or
    ``1/2``) and apply ``n_steps`` successive halvings (multiplying by
    ``Fraction(1, 2)`` each step). ``easy`` fixes ``n_steps=1`` (a single
    halving step); ``medium`` fixes ``n_steps=2`` (two *chained* halving
    steps, requiring the learner to track state across steps, not just
    apply the rule once) -- the starting fraction itself is still randomised
    between ``1`` and ``1/2`` at both tiers.
  - ``inverse`` (``variant="inverse"``, ``hard`` only): genuinely different
    reasoning -- run the halving rule **backwards**. A target fraction
    (``1/4`` or ``1/8``) is stated as the *result* of one or two halving
    steps (``n_steps`` randomised over ``{1, 2}``, matching a "one or
    more halving steps" phrasing), and the question
    asks for the fraction *before* it was halved. Construction computes the
    original algebraically (``target * 2 ** n_steps``); the independent
    check never reuses that multiplication -- it forward-simulates halving
    the found original ``n_steps`` times (the same repeated
    multiply-by-``Fraction(1, 2)`` route the ``forward`` variant itself
    uses to *construct* its own answer) and asserts the result lands
    exactly on the stated target.

  Independent check, ``forward``: construction repeatedly multiplies a
  ``Fraction`` by ``Fraction(1, 2)``, ``n_steps`` times. The check never
  re-runs that loop -- since every starting fraction here has numerator 1,
  it derives the expected denominator directly as
  ``start_denominator * 2 ** n_steps`` and asserts the already-reduced
  answer ``Fraction`` matches exactly.

  Misconceptions, each proven a genuinely **different fraction value** (not
  merely a different displayed form of the same value -- Python's
  ``fractions.Fraction`` auto-reduces on construction, so no candidate here
  is ever built or displayed in unreduced form):

  - ``wrong-direction-halving``: halves the **denominator** instead of
    doubling it (e.g. ``1/2 -> 1/1``, going the wrong direction) -- the
    classic literal slip. For the ``forward`` variant this is
    ``Fraction(1, max(1, start_denominator // 2 ** n_steps))``, which is
    *always* exactly ``1`` for every sampled ``(start_denominator,
    n_steps)`` combination this generator produces (proven directly, not
    assumed -- see the test suite), and the ``forward`` variant's correct
    answer is always a proper fraction strictly less than ``1``, so this
    distractor is always distinct. For the ``inverse`` variant it continues
    halving the *target* instead of undoing the halving
    (``target * Fraction(1, 2) ** n_steps``), always smaller than the
    correct (inverted, larger) original, so always distinct there too.
  - ``off-by-one-halving-step``: stops one halving step early or goes one
    step too far. ``forward``: one step *too far*
    (``start * Fraction(1, 2) ** (n_steps + 1)``, always distinct since an
    extra halving always changes the denominator by an exact factor of 2).
    ``inverse``: undoes one *fewer* halving step than actually happened
    (``target * 2 ** max(0, n_steps - 1)``, clamped at zero undone steps --
    "forgot to invert at all" when ``n_steps == 1``), always distinct since
    undoing one fewer step always changes the value by an exact factor of 2
    in the other direction.
  - ``adds-instead-of-multiplies``: treats each halving step as *adding*
    ``Fraction(1, 2)`` rather than *multiplying* by it (a genuine, common
    "halving means subtracting/adding a half" confusion, not a
    display-form quirk). ``forward``: ``start + n_steps * Fraction(1, 2)``.
    ``inverse``: ``target + n_steps * Fraction(1, 2)``. This particular
    formula was chosen over the more literal illustrative
    reading (a bare "1/2 + 1/2" giving a displayed "2/4") specifically
    *because* that literal reading reduces to exactly ``1/2`` -- the same
    value as the ``forward`` variant's own easy-tier, ``start = 1``
    correct answer -- which must never be offered as a distractor.
    The formula actually implemented is checked directly (via a 500-seed
    sweep per tier, not assumed) to differ from the correct answer's exact
    reduced ``Fraction`` value in every case it can produce -- see
    ``test_forward_halving_every_misconception_value_is_distinct_from_the_answer_every_seed``
    and
    ``test_inverse_halving_every_misconception_value_is_distinct_from_the_answer_every_seed``
    in the test suite.

  A genuine, understood coincidence, not a bug, documented rather than
hidden: at the
  ``forward`` variant's ``easy`` tier specifically, when the randomly
  sampled starting fraction is ``1/2`` (denominator 2, the other easy-tier
  possibility being the whole ``1``), ``wrong-direction-halving`` and
  ``adds-instead-of-multiplies`` both evaluate to exactly ``1`` and collide
  -- ``pick_distractors`` correctly dedupes this to 2 distinct distractors
  instead of 3 for that half of easy-tier seeds; the collision never
  produces a distractor equal to the correct answer and never drops the
  count below the required minimum of 1.

- ``decimal_tenths_hundredths_compose`` (AC9M4N01, prerequisites
  ``decimal-place-value``, ``fraction-decimal-percent``) -- **compose** a
  decimal number from stated tenths/hundredths quantities (e.g. "write 3
  tenths and 7 hundredths as a decimal" -> ``0.37``). The skill covers
  naming and writing decimals to hundredths in standard decimal notation.

  **Composition, not decomposition -- the explicit boundary against the
  existing prerequisite skill ``decimal-place-value``** (whose own
  generator, ``decimal_place_value`` in ``mathgen/generators/primary_math.py``,
  answers "What is the value of the digit 3 in 4.35?" -- i.e. it
  *decomposes* a *given* decimal into a named digit's place value). This
  skill's graded operation runs in the opposite direction: the tenths and
  hundredths *quantities* are stated as the question's input, and the
  decimal numeral is the thing to be produced. No question this generator
  builds ever asks "what is the value of digit X in decimal Y" -- that
  shape belongs exclusively to ``decimal_place_value`` and is never
  duplicated here.

  Three variants, chosen by difficulty:

  - ``single_place`` (``easy``): compose from **one** named place only,
    randomly ``tenths`` (e.g. "write 5 tenths as a decimal" -> ``0.5``) or
    ``hundredths`` (e.g. "write 6 hundredths as a decimal" -> ``0.06``) --
    both single-place cases are included.
    Both places draw a single digit, ``1``-``9``.
  - ``both_places`` (``medium``): compose from **two** named places
    together (e.g. "write 3 tenths and 7 hundredths as a decimal" ->
    ``0.37``), both single digits ``1``-``9``, no leading-zero trap forced.
  - ``forced_trap`` (``hard``): two genuine traps, one **forced** every
    single seed (never left to chance) --

    - ``leading_zero`` (hundredths-only, digit ``1``-``9``, e.g. "5
      hundredths" -> ``0.05`` -- the naive error drops the required
      tenths-place zero and writes ``0.5``).
    - ``regrouping`` (tenths-only, but the stated tenths *count* is now
      **two digits and never a multiple of ten** -- drawn as
      ``10 * whole + rem`` with ``whole, rem`` both ``1``-``9``, e.g. "12
      tenths" -> ``1.2`` -- the naive error fails to carry the extra whole
      unit out of the tenths place and writes ``0.12``).

    Which trap fires is recorded directly in the question's own text shape
    (a "hundredths"-only sentence for ``leading_zero``, a "tenths"-only
    sentence with a two-digit count for ``regrouping``) so a test can
    determine, per seed, which trap fired -- asserted directly, not
    assumed.

  Independent check (``_verify_decimal_composition``): construction always
  assembles the answer string directly via place-value string formatting
  (an ``f"0.{tenths}{hundredths}"``-shaped literal, or, for the
  ``regrouping`` trap, an explicit ``divmod`` split of the stated
  two-digit tenths count into whole and remaining-tenths digits). The
  check never touches that formatting -- it converts the *stated*
  tenths/hundredths quantities to an exact ``Fraction``
  (``tenths_count * Fraction(1, 10) + hundredths_count * Fraction(1,
  100)``) and asserts it equals ``Fraction(answer_str)`` (Python's
  ``fractions.Fraction`` parses a decimal literal string exactly, e.g.
  ``Fraction("1.2") == Fraction(6, 5)``) -- an exact rational-value
  comparison, never a tolerant float comparison and never a string
  comparison.

  Misconceptions (every candidate's numeric gap from the correct answer is
  checked directly against ``grade_number``'s own ~0.05 tolerance --
  ``mathgen/grading/spoken.py`` ``_TOLERANCE = 0.051`` -- not merely
  assumed to clear it; see the module-level gap analysis below and the test
  suite's exhaustive sweep):

  - ``leading-zero-drop``: writes a stated hundredths-only quantity into
    the tenths place instead (``0.05`` misread/miswritten as ``0.5``).
    Applicable to the ``single_place``/``hundredths`` case and, forced,
    the ``forced_trap``/``leading_zero`` case. Minimum gap across every
    legal ``(1-9)`` hundredths digit: ``9 * 1 / 100 = 0.09 > 0.051``
    (proven algebraically for every digit ``1``-``9``, gap only grows with
    the digit).
  - ``regrouping-failure``: writes a two-digit tenths count as if its
    digits *were* the decimal's own tenths+hundredths digits directly,
    without carrying (``12`` tenths written as ``0.12`` instead of
    ``1.2``). Applicable only to the ``forced_trap``/``regrouping`` case
    (the only variant with a two-digit tenths count). Minimum gap across
    every legal two-digit count (``11``-``99``, excluding multiples of
    ten): the gap is always ``9 * count / 100``, minimised at
    ``count = 11`` -> ``0.99 > 0.051`` (a huge margin, since the naive
    value is always below ``1`` while the correct value is always at least
    ``1.1``).
  - ``place-swap``: the tenths and hundredths digits transposed (e.g.
    ``0.37`` read as ``0.73``). Applicable only to the ``both_places``
    case, and only when the two stated digits differ (when they are equal
    the transposed string is identical to the answer, so
    ``pick_distractors`` correctly drops it rather than ever offering a
    distractor equal to the answer). Minimum gap when the two digits
    differ by exactly ``1`` (the smallest possible difference for two
    distinct digits ``1``-``9``): ``9 * 1 / 100 = 0.09 > 0.051``.
  - ``drops-decimal-point`` (a necessary fourth
    addition): reports the bare stated digit(s) as an integer, with no
    decimal point at all (e.g. "5 tenths" answered as literal ``5`` instead
    of ``0.5``; "3 tenths and 7 hundredths" answered as the bare
    concatenation ``37`` instead of ``0.37``). Added because the
    ``single_place``/``tenths`` case has **no** other applicable named
    misconception (``leading-zero-drop`` needs a hundredths-only
    composition; ``place-swap`` needs two distinct stated digits;
    ``regrouping-failure`` needs a two-digit tenths count) -- without this
    addition that specific case would offer zero distractors, violating
    the "distractors never empty" requirement. Always trivially distinct
    from the correct answer by a wide margin (the bare integer is always
    at least ``1`` while every composed decimal in this generator is
    always ``< 10``, or, for the bare-integer case itself, the composed
    decimal is always well below the bare integer value).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, pick_distractors

# -- fraction_halving -----------------------------------------------------

_FORWARD_START_DENOMS = (1, 2)
_INVERSE_TARGET_DENOMS = (4, 8)
_INVERSE_STEP_CHOICES = (1, 2)
_STEP_WORDS = {1: "once", 2: "twice"}


def _fraction_text(f: Fraction) -> str:
    """Format a Fraction the way a student would type it: "3" for a whole
    number, "1/2" otherwise. Python's Fraction always auto-reduces on
    construction, so this is never an unreduced display -- no candidate
    built anywhere in this module can ever render as e.g. "2/4"."""
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _verify_forward_halving(start_denom: int, n_steps: int, answer: Fraction) -> None:
    """Independent check for the ``forward`` variant: construction
    repeatedly multiplies a Fraction by Fraction(1, 2), n_steps times. This
    check never re-runs that loop -- since every starting fraction here has
    numerator 1, it derives the expected denominator directly as
    start_denom * 2**n_steps and asserts the already-reduced answer
    Fraction matches exactly."""
    expected = Fraction(1, start_denom * (2**n_steps))
    check(
        answer == expected,
        f"halving 1/{start_denom} {n_steps} time(s) should give {expected}, got {answer}",
    )


def _forward_halving(rng, n_steps: int):
    if n_steps not in _STEP_WORDS:
        raise ValueError(f"unsupported n_steps {n_steps!r}; expected one of {sorted(_STEP_WORDS)}")
    start_denom = rng.choice(_FORWARD_START_DENOMS)
    start = Fraction(1, start_denom)
    answer = start
    for _ in range(n_steps):
        answer = answer * Fraction(1, 2)
    _verify_forward_halving(start_denom, n_steps, answer)

    # Misconceptions -- see the module docstring for the full derivation and
    # the documented wrong-direction/adds-instead-of-multiplies coincidence
    # at easy tier when start_denom == 2.
    wrong_direction = Fraction(1, max(1, start_denom // (2**n_steps)))
    off_by_one = start * Fraction(1, 2) ** (n_steps + 1)
    adds_instead = start + n_steps * Fraction(1, 2)
    candidates = [wrong_direction, off_by_one, adds_instead]

    start_display = _fraction_text(start)
    step_word = _STEP_WORDS[n_steps]
    answer_text = _fraction_text(answer)

    text = f"A fraction starts at {start_display}. If it is halved {step_word}, what fraction results?"
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Halving {start_display} {step_word} gives {answer_text}."],
        distractors=pick_distractors(answer_text, [_fraction_text(c) for c in candidates]),
    )


def _verify_inverse_halving(n_steps: int, target: Fraction, original: Fraction) -> None:
    """Independent check for the ``inverse`` variant: construction computes
    the original fraction directly via original = target * 2**n_steps (an
    algebraic inversion). This check never performs that multiplication --
    it forward-simulates halving `original` n_steps times (the same
    repeated multiply-by-Fraction(1, 2) route the forward variant itself
    uses to construct its own answer) and asserts the result lands exactly
    on `target`."""
    simulated = original
    for _ in range(n_steps):
        simulated = simulated * Fraction(1, 2)
    check(
        simulated == target,
        f"halving {original} {n_steps} time(s) gives {simulated}, expected target {target}",
    )


def _inverse_halving(rng):
    n_steps = rng.choice(_INVERSE_STEP_CHOICES)
    target_denom = rng.choice(_INVERSE_TARGET_DENOMS)
    target = Fraction(1, target_denom)
    original = target * (2**n_steps)
    _verify_inverse_halving(n_steps, target, original)

    # Misconceptions -- see the module docstring for the full derivation.
    wrong_direction = target * Fraction(1, 2) ** n_steps
    off_by_one = target * (2 ** max(0, n_steps - 1))
    adds_instead = target + n_steps * Fraction(1, 2)
    candidates = [wrong_direction, off_by_one, adds_instead]

    target_text = _fraction_text(target)
    step_word = _STEP_WORDS[n_steps]
    answer_text = _fraction_text(original)

    text = (
        f"A fraction was halved {step_word}, and the result is {target_text}. "
        "What was the original fraction, before it was halved?"
    )
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"p": original.numerator, "q": original.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Undoing {step_word} halving from {target_text} gives {answer_text}."],
        distractors=pick_distractors(answer_text, [_fraction_text(c) for c in candidates]),
    )


@generator(
    "fraction_halving",
    level=2,
    presets={
        "easy": dict(variant="forward", n_steps=1),
        "medium": dict(variant="forward", n_steps=2),
        "hard": dict(variant="inverse"),
    },
)
def fraction_halving(rng, variant="forward", n_steps=1, scenes=False):
    """Partition a whole or a fraction into halves, quarters and eighths by
    repeated halving. See the module docstring for the forward/inverse
    difficulty rationale and the misconception derivations.

    ``scenes=True`` (what the skill file sets) tells the same question about a pizza, a ribbon or a
    chocolate bar, with the starting piece shaded on a picture, so a tier holds dozens of questions
    instead of two or three; see ``w2_early_pools``."""
    if scenes:
        from .w2_early_pools import halving_scene

        return halving_scene(rng, variant, n_steps)
    if variant == "forward":
        return _forward_halving(rng, n_steps)
    if variant == "inverse":
        return _inverse_halving(rng)
    raise ValueError(f"unknown variant {variant!r}; expected 'forward' or 'inverse'")


# -- decimal_tenths_hundredths_compose -------------------------------------

_SINGLE_PLACES = ("tenths", "hundredths")


def _place_count(n: int, plural: str) -> str:
    """Agree a place-value count with its noun: "1 tenth" but "0 tenths"/"3 tenths".
    ``plural`` is the plural form as it appears in prose (e.g. "tenths");
    the singular is that word with its trailing "s" dropped."""
    return f"{n} {plural[:-1] if n == 1 else plural}"


def _verify_decimal_composition(tenths_count: int, hundredths_count: int, answer_str: str) -> None:
    """Independent check: construction always assembles the answer string
    directly via place-value string formatting. This check never touches
    that formatting -- it converts the *stated* tenths/hundredths
    quantities to an exact Fraction and asserts it equals Fraction(answer_str)
    (Python's fractions.Fraction parses a decimal literal string exactly),
    an exact rational-value comparison, never a tolerant float comparison
    and never a string comparison."""
    expected = tenths_count * Fraction(1, 10) + hundredths_count * Fraction(1, 100)
    actual = Fraction(answer_str)
    check(
        expected == actual,
        f"{tenths_count} tenths + {hundredths_count} hundredths = {expected}, but answer string "
        f"{answer_str!r} parses to {actual}",
    )


def _single_place(rng):
    place = rng.choice(_SINGLE_PLACES)
    count = rng.randint(1, 9)
    if place == "tenths":
        tenths_count, hundredths_count = count, 0
        answer_str = f"0.{count}"
        # Misconception: drops-decimal-point only -- see module docstring
        # for why leading-zero-drop/place-swap/regrouping-failure do not
        # apply to this specific case.
        candidates = [str(count)]
    else:
        tenths_count, hundredths_count = 0, count
        answer_str = f"0.0{count}"
        # Misconceptions: drops-decimal-point, and the forced-at-hard
        # leading-zero-drop (writes the hundredths digit into the tenths
        # place instead, e.g. "0.6" for "6 hundredths").
        candidates = [str(count), f"0.{count}"]
    _verify_decimal_composition(tenths_count, hundredths_count, answer_str)
    text = f"Write {_place_count(count, place)} as a decimal."
    return _decimal_result(text, tenths_count, hundredths_count, answer_str, candidates)


def _both_places(rng):
    tenths_count = rng.randint(1, 9)
    hundredths_count = rng.randint(1, 9)
    answer_str = f"0.{tenths_count}{hundredths_count}"
    _verify_decimal_composition(tenths_count, hundredths_count, answer_str)

    # Misconceptions: drops-decimal-point always; place-swap only when the
    # two stated digits differ (when equal, the transposed string is
    # identical to the answer and pick_distractors correctly drops it).
    candidates = [f"{tenths_count}{hundredths_count}"]
    if tenths_count != hundredths_count:
        candidates.append(f"0.{hundredths_count}{tenths_count}")

    text = (
        f"Write {_place_count(tenths_count, 'tenths')} and {_place_count(hundredths_count, 'hundredths')} as a decimal."
    )
    return _decimal_result(text, tenths_count, hundredths_count, answer_str, candidates)


def _forced_trap(rng):
    trap = rng.choice(("leading_zero", "regrouping"))
    if trap == "leading_zero":
        count = rng.randint(1, 9)
        tenths_count, hundredths_count = 0, count
        answer_str = f"0.0{count}"
        # Forced: leading-zero-drop, plus drops-decimal-point.
        candidates = [str(count), f"0.{count}"]
        text = f"Write {_place_count(count, 'hundredths')} as a decimal."
    else:
        whole = rng.randint(1, 9)
        rem = rng.randint(1, 9)
        count = whole * 10 + rem  # a two-digit tenths count, never a multiple of ten
        tenths_count, hundredths_count = count, 0
        answer_str = f"{whole}.{rem}"
        # Forced: regrouping-failure, plus drops-decimal-point.
        candidates = [str(count), f"0.{count}"]
        text = f"Write {count} tenths as a decimal."
    _verify_decimal_composition(tenths_count, hundredths_count, answer_str)
    return _decimal_result(text, tenths_count, hundredths_count, answer_str, candidates)


def _decimal_result(
    text: str, tenths_count: int, hundredths_count: int, answer_str: str, candidates: list[str]
) -> dict:
    answer_val = float(answer_str)
    # grade_number's ~0.05 numeric tolerance (mathgen/grading/spoken.py
    # _TOLERANCE = 0.051) -- checked directly here, not just assumed. See
    # the module docstring's per-misconception gap analysis: every
    # distractor this generator can produce clears this by a wide,
    # algebraically-proven margin (minimum 0.09), so no construction-time
    # decimal skills) -- this check exists as a direct, non-optional guard
    # in case that proof is ever invalidated by a future edit.
    for candidate in candidates:
        gap = abs(answer_val - float(candidate))
        check(gap > 0.051, f"distractor {candidate!r} is within grader tolerance of {answer_str!r} (gap {gap})")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_val,
        answer_text=answer_str,
        answer_latex=answer_str,
        working=[
            f"{_place_count(tenths_count, 'tenths')} and {_place_count(hundredths_count, 'hundredths')} = {answer_str}."
        ],
        distractors=pick_distractors(answer_str, candidates),
    )


_DECIMAL_VARIANT_FNS = {
    "single_place": _single_place,
    "both_places": _both_places,
    "forced_trap": _forced_trap,
}


@generator(
    "decimal_tenths_hundredths_compose",
    level=4,
    presets={
        "easy": dict(variant="single_place"),
        "medium": dict(variant="both_places"),
        "hard": dict(variant="forced_trap"),
    },
)
def decimal_tenths_hundredths_compose(rng, variant="both_places"):
    """Compose a decimal number from stated tenths/hundredths quantities.
    See the module docstring for the composition-vs-decomposition boundary
    against decimal_place_value, the difficulty-tier variant shapes, and
    the misconception derivations.

    Bare-call default is deliberately ``both_places`` (the ``medium``
    preset), not ``single_place`` (the ``easy`` preset): every skill source
    passes ``variant`` explicitly (see the three presets above), so this
    default only matters for a bare ``make("decimal_tenths_hundredths_compose")``
    call with no difficulty -- and ``single_place``'s ``tenths``-only
    sub-case only ever offers 1 distractor (``drops-decimal-point``; see the
    module docstring), which would fail
    the usual 85%-of-seeds-need->=2-distractors bar roughly half the time.
    ``both_places`` clears that bar comfortably (only the ~1-in-9 seeds
    where the two sampled digits happen to coincide drop below 2)."""
    if variant not in _DECIMAL_VARIANT_FNS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(_DECIMAL_VARIANT_FNS)}")
    return _DECIMAL_VARIANT_FNS[variant](rng)
