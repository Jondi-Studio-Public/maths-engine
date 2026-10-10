"""Builds the `f10-transformations-invariants` skill: what changes and what stays the same when a shape is
translated, reflected or rotated.

Topic: ``transformations_invariants`` (``answer: mcq``). The transformation is always *named*; the learner
never has to perform it or draw it. Each option is a short profile of the image, and the learner picks the
profile that is true. The facts behind it are the three rigid motions' shared and distinguishing properties:

- side lengths and angles never change (translation, reflection, rotation alike);
- a reflection gives a mirror image (the corners go round the opposite way); a translation or a rotation does not;
- a shape's own symmetry (lines of symmetry, order of rotational symmetry) is the same in the image.

"Mirror image" is defined in every stem as *corners in the opposite order* so that a rotation, which changes the
way a shape points, is never mistaken for a reversal.

Three tiers, escalating in reasoning rather than size:

- easy: one named transformation of a plain shape; pick the profile (side lengths, angles, mirror image or not).
- medium: the same, but the shape carries a stated side length and angle, so the profile quotes numbers and a
  learner who thinks the movement changes them can be caught on the exact value.
- hard: two or three transformations in a row on a shape whose symmetry is stated by its name; the final image is
  a mirror image only when the number of reflections is odd, and its symmetry is that of the original.

Distractors flip one or two components of the profile; the answer is never the profile whose parts are all
nearest the others (the flipped parts are drawn at random). Each distractor is tagged with the misconception
behind its first slip (reversal, then size, then angle, then symmetry).
"""

from __future__ import annotations

from ..core import check, generator, latex_answer, pick_distractors

# -- shapes ---------------------------------------------------------------------------------------------

_PLAIN_SHAPES = ["triangle", "kite", "parallelogram", "trapezium", "pentagon", "hexagon", "arrowhead"]

# (name, lines of symmetry, order of rotational symmetry). Each name pins both numbers down exactly.
_SYMMETRIC_SHAPES = [
    ("scalene triangle", 0, 1),
    ("isosceles triangle that is not equilateral", 1, 1),
    ("equilateral triangle", 3, 3),
    ("rectangle that is not a square", 2, 2),
    ("square", 4, 4),
    ("regular pentagon", 5, 5),
    ("regular hexagon", 6, 6),
    ("regular octagon", 8, 8),
]

_NUMBERED = [("triangle", "ABC"), ("quadrilateral", "ABCD"), ("pentagon", "ABCDE")]

_MIRROR_GLOSS = (
    "A mirror image has its corners in the opposite order: if the corners go round clockwise, "
    "the image's corners go round anticlockwise."
)

MID_MIRROR = "reversal"
MID_SIZE = "size"
MID_ANGLE = "angle"
MID_SYMMETRY = "symmetry"

# misconception ids (also in the skill YAML)
SLIDE_OR_TURN_REVERSES = "slide-or-turn-reverses"
REFLECTION_NOT_REVERSED = "reflection-not-reversed"
MOVEMENT_CHANGES_SIZE = "movement-changes-size"
MOVEMENT_CHANGES_ANGLES = "movement-changes-angles"
ANY_REFLECTION_REVERSES = "any-reflection-reverses"
REFLECTION_EFFECT_LOST = "reflection-effect-lost"
SYMMETRY_CHANGES = "symmetry-changes-with-movement"


def _article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


# -- the movements ----------------------------------------------------------------------------------------


def _step(rng, kind, m=None):
    """A named movement: (kind, words, parameters used by the independent check)."""
    if kind == "translation":
        direction = rng.choice(["right", "left", "up", "down"])
        d = m if m is not None else rng.randint(2, 9)
        dx, dy = {"right": (d, 0), "left": (-d, 0), "up": (0, d), "down": (0, -d)}[direction]
        where = f"to the {direction}" if direction in ("right", "left") else direction
        return dict(kind=kind, words=f"translated {d} cm {where}", dx=dx, dy=dy)
    if kind == "reflection":
        axis = rng.choice(["vertical", "horizontal"])
        return dict(kind=kind, words=f"reflected in a {axis} mirror line", axis=axis)
    turn = rng.choice([90, 180, 270])
    way = rng.choice(["clockwise", "anticlockwise"])
    return dict(kind=kind, words=f"rotated {turn}° {way} about a point O", turn=turn, way=way)


def _apply(step, pts):
    """Apply one movement to integer-coordinate points. Lines and centres sit on grid lines, so every
    coordinate stays an integer."""
    if step["kind"] == "translation":
        return [(x + step["dx"], y + step["dy"]) for x, y in pts]
    if step["kind"] == "reflection":
        if step["axis"] == "vertical":
            return [(40 - x, y) for x, y in pts]
        return [(x, 40 - y) for x, y in pts]
    turns = (step["turn"] // 90) % 4
    if step["way"] == "clockwise":
        turns = (4 - turns) % 4
    out = pts
    for _ in range(turns):  # quarter turn anticlockwise about the origin
        out = [(-y, x) for x, y in out]
    return out


_TEST_SHAPE = [(0, 0), (7, 1), (9, 6), (3, 8), (-2, 4)]  # a scalene, asymmetric, anticlockwise pentagon


def _signed_area2(pts):
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))


def _side_lengths_sq(pts):
    return [(x1 - x2) ** 2 + (y1 - y2) ** 2 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1])]


def _angles_signature(pts):
    """The cosines' numerators of every corner (dot product and squared lengths), which fix each angle."""
    n = len(pts)
    sig = []
    for i in range(n):
        ax, ay = pts[i - 1][0] - pts[i][0], pts[i - 1][1] - pts[i][1]
        bx, by = pts[(i + 1) % n][0] - pts[i][0], pts[(i + 1) % n][1] - pts[i][1]
        sig.append((ax * bx + ay * by, ax * ax + ay * ay, bx * bx + by * by))
    return sig


def _image_reversed(steps) -> bool:
    """Independent route to 'is the image a mirror image': actually move an asymmetric shape and compare
    its signed area, and confirm lengths and angles are untouched on the way."""
    pts = _TEST_SHAPE
    out = pts
    for s in steps:
        out = _apply(s, out)
    check(_side_lengths_sq(out) == _side_lengths_sq(pts), "a rigid motion changed a side length")
    check(_angles_signature(out) == _angles_signature(pts), "a rigid motion changed an angle")
    return (_signed_area2(out) > 0) != (_signed_area2(pts) > 0)


# -- option profiles --------------------------------------------------------------------------------------

_MASKS = [0b001, 0b010, 0b100, 0b011, 0b101, 0b110]


def _build_options(rng, components, render):
    """components: three dicts {correct, wrongs: [(token, note_id), ...], priority}. A distractor flips one or
    two components (3 distinct patterns drawn at random, so the answer is not the 'middle' option). Returns
    (answer_text, distractor texts, option_notes)."""
    answer = render([c["correct"] for c in components])
    while True:  # no component may be wrong in all three, or the answer would be the only option that differs there
        masks = rng.sample(_MASKS, 3)
        if all(sum(m >> i & 1 for m in masks) <= 2 for i in range(3)):
            break
    used = [0, 0, 0]
    texts, notes = [], {}
    for mask in masks:
        tokens, flipped = [], []
        for i, comp in enumerate(components):
            if mask >> i & 1:
                wrongs = comp["wrongs"]
                token, note = wrongs[used[i] % len(wrongs)]
                used[i] += 1
                tokens.append(token)
                flipped.append((comp["priority"], note))
            else:
                tokens.append(comp["correct"])
        text = render(tokens)
        note = min(flipped)[1]
        if text != answer and text not in notes:
            texts.append(text)
            notes[text] = note
    kept = pick_distractors(answer, texts, n=3)
    check(len(kept) == 3, "fewer than three distinct distractors")
    return answer, kept, {t: notes[t] for t in kept}


def _yes_no(flag, yes, no):
    return yes if flag else no


# -- the generator ----------------------------------------------------------------------------------------


@generator(
    "transformations_invariants",
    level=21,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def transformations_invariants(rng, tier="easy"):
    """State which properties of a shape stay the same, and which change, under translation, reflection or
    rotation (a mirror image is the only change among lengths, angles and reversal)."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    return _hard(rng)


def _result(text, answer, distractors, notes, working):
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=distractors,
        option_notes=notes,
    )


def _easy(rng):
    kind = rng.choice(["translation", "reflection", "rotation"])
    step = _step(rng, kind)
    shape = rng.choice(_PLAIN_SHAPES)
    reversed_ = kind == "reflection"
    check(_image_reversed([step]) == reversed_, "reversal rule disagrees with the moved test shape")

    def render(tokens):
        return f"Side lengths: {tokens[1]}. Angles: {tokens[2]}. The image {tokens[0]} a mirror image of the {shape}."

    components = [
        dict(
            correct=_yes_no(reversed_, "is", "is not"),
            wrongs=[
                (_yes_no(reversed_, "is not", "is"), REFLECTION_NOT_REVERSED if reversed_ else SLIDE_OR_TURN_REVERSES)
            ],
            priority=0,
        ),
        dict(correct="the same", wrongs=[("different", MOVEMENT_CHANGES_SIZE)], priority=1),
        dict(correct="the same", wrongs=[("different", MOVEMENT_CHANGES_ANGLES)], priority=2),
    ]
    answer, distractors, notes = _build_options(rng, components, render)
    stem = rng.choice(["Which statement about the image is correct?", "Which of these correctly describes the image?"])
    text = f"{_article(shape).capitalize()} {shape} is {step['words']}. {_MIRROR_GLOSS} {stem}"
    working = [
        "A translation, a reflection and a rotation all keep every side length and every angle.",
        "Only a reflection turns the shape into a mirror image."
        if reversed_
        else "A translation or a rotation never turns the shape into a mirror image.",
    ]
    return _result(text, answer, distractors, notes, working)


def _medium(rng):
    kind = rng.choice(["translation", "reflection", "rotation"])
    shape, labels = rng.choice(_NUMBERED)
    m = rng.randint(2, 9)
    step = _step(rng, kind, m=m)
    side = rng.randint(4, 15)
    angle = rng.randint(35, 145)
    reversed_ = kind == "reflection"
    check(_image_reversed([step]) == reversed_, "reversal rule disagrees with the moved test shape")

    a, b, c = labels[0], labels[1], labels[2]
    if kind == "translation":
        motion = step["words"]
    elif kind == "reflection":
        motion = f"reflected in a mirror line {m} cm from side {a}{b}"
    else:
        motion = f"{step['words']}, where O is {m} cm from corner {a}"

    length_wrongs = []
    for cand in (side + m, abs(side - m), side + 2 * m, side * 2):
        if cand > 0 and cand != side and cand not in [w for w, _ in length_wrongs]:
            length_wrongs.append((cand, MOVEMENT_CHANGES_SIZE))
    turn = step.get("turn", 0)
    angle_wrongs = []
    for cand in (180 - angle, angle + turn, abs(angle - turn), angle + 10, angle - 10, angle + m):
        if 0 < cand < 180 and cand != angle and cand not in [w for w, _ in angle_wrongs]:
            angle_wrongs.append((cand, MOVEMENT_CHANGES_ANGLES))
    check(len(length_wrongs) >= 2 and len(angle_wrongs) >= 2, "not enough wrong values")

    def render(tokens):
        return (
            f"{a}'{b}' = {tokens[1]} cm, angle {a}'{b}'{c}' = {tokens[2]}°, and the image {tokens[0]} a mirror image."
        )

    components = [
        dict(
            correct=_yes_no(reversed_, "is", "is not"),
            wrongs=[
                (_yes_no(reversed_, "is not", "is"), REFLECTION_NOT_REVERSED if reversed_ else SLIDE_OR_TURN_REVERSES)
            ],
            priority=0,
        ),
        dict(correct=str(side), wrongs=[(str(v), n) for v, n in length_wrongs], priority=1),
        dict(correct=str(angle), wrongs=[(str(v), n) for v, n in angle_wrongs], priority=2),
    ]
    answer, distractors, notes = _build_options(rng, components, render)
    text = (
        f"{_article(shape).capitalize()} {shape} {labels} has {a}{b} = {side} cm and angle {a}{b}{c} = {angle}°. "
        f"It is {motion}, and its image is {''.join(ch + chr(39) for ch in labels)}. {_MIRROR_GLOSS} "
        f"Which statement about the image is correct?"
    )
    working = [
        f"A {kind} keeps every side length and angle, so {a}'{b}' = {side} cm and angle {a}'{b}'{c}' = {angle}°.",
        "The image is a mirror image only for a reflection.",
    ]
    return _result(text, answer, distractors, notes, working)


def _hard(rng):
    name, lines, order = rng.choice(_SYMMETRIC_SHAPES)
    n_steps = rng.choice([2, 3])
    n_refl = rng.randint(0, n_steps)
    kinds = ["reflection"] * n_refl + [rng.choice(["translation", "rotation"]) for _ in range(n_steps - n_refl)]
    rng.shuffle(kinds)
    steps = [_step(rng, k) for k in kinds]
    reversed_ = _image_reversed(steps)
    check(reversed_ == (n_refl % 2 == 1), "parity of reflections disagrees with the moved test shape")

    focus = rng.choice(["lines", "order"])
    value = lines if focus == "lines" else order
    if focus == "lines":
        sym_words = lambda v: "no lines of symmetry" if v == 0 else f"{v} line{'s' if v != 1 else ''} of symmetry"
        cands = [value + 1, max(value - 1, 0), value * 2, value + 2]
    else:
        sym_words = lambda v: "no rotational symmetry" if v == 1 else f"rotational symmetry of order {v}"
        cands = [value + 1, value - 1, value * 2, value + 2]
    sym_wrongs = []
    for cand in cands:
        if cand >= (0 if focus == "lines" else 1) and cand != value and cand not in [w for w, _ in sym_wrongs]:
            sym_wrongs.append((cand, SYMMETRY_CHANGES))
    check(len(sym_wrongs) >= 2, "not enough wrong symmetry values")

    if n_refl == 0:
        rev_note = SLIDE_OR_TURN_REVERSES
    elif reversed_:
        rev_note = REFLECTION_EFFECT_LOST
    else:
        rev_note = ANY_REFLECTION_REVERSES

    def render(tokens):
        same = "are the same as" if tokens[1] == "same" else "are different from"
        sym = sym_words(int(tokens[2]))
        return (
            f"Its side lengths and angles {same} the original shape's, it has {sym}, and it {tokens[0]} a mirror image."
        )

    components = [
        dict(
            correct=_yes_no(reversed_, "is", "is not"),
            wrongs=[(_yes_no(reversed_, "is not", "is"), rev_note)],
            priority=0,
        ),
        dict(correct="same", wrongs=[("different", MOVEMENT_CHANGES_SIZE)], priority=1),
        dict(correct=str(value), wrongs=[(str(v), n) for v, n in sym_wrongs], priority=3),
    ]
    answer, distractors, notes = _build_options(rng, components, render)
    sequence = ", then ".join(s["words"] for s in steps)
    text = (
        f"{_article(name).capitalize()} {name} is {sequence}. {_MIRROR_GLOSS} "
        f"Which statement about the final image is correct?"
    )
    working = [
        "Every movement keeps all side lengths and angles, so the final image has the same lengths and angles.",
        f"A rigid motion cannot create or remove symmetry, so the image still has {sym_words(value)}.",
        f"There {'is' if n_refl == 1 else 'are'} {n_refl} reflection{'s' if n_refl != 1 else ''}: "
        + (
            "an odd number gives a mirror image."
            if reversed_
            else "an even number (including none) gives no mirror image."
        ),
    ]
    return _result(text, answer, distractors, notes, working)
