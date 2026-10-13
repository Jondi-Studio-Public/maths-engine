"""Exam-tier source for the surface area of an irregular solid found by adding the areas of its faces.

A rectangular block has a rectangular channel cut along its whole length. The learner must

(a) find the area of the end face (the block's rectangle less the channel's rectangle),
(b) build the total surface area from that end, the channel's extra walls and the long faces, and
(c) price a coat of sealant over the whole surface, converting square centimetres to square metres.

The channel is cut into one of the long faces and runs the full length, so the top face loses a strip and
gains the two walls and the floor of the channel; the surface area is not simply the block's less the channel.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import VerificationError, attempts, generator, latex_answer, numeric_answer

from .exd_x_m5_support import clear_of_boundary, display, keys_absent, rounded, tagged_distractors

_OBJECTS = (
    ("A concrete drainage block", "the block", "sealant"),
    ("A timber bench slab with a groove along its length", "the slab", "varnish"),
    ("A paving block with a drainage channel", "the block", "sealant"),
    ("A plastic cable tray", "the tray", "protective coating"),
    ("A garden edging block with a channel for irrigation pipe", "the block", "sealant"),
)
_FIRST = (
    "{obj} is a rectangular solid {length} cm long, {width} cm wide and {height} cm high. "
    "A rectangular channel {cw} cm wide and {cd} cm deep runs the full length of its top face.",
    "{obj} measures {length} cm by {width} cm by {height} cm. Along the whole length of the top there is "
    "a rectangular channel {cw} cm wide and {cd} cm deep.",
    "{obj} has length {length} cm, width {width} cm and height {height} cm, with a rectangular channel "
    "{cw} cm wide and {cd} cm deep cut along the full length of the top.",
)
_A_PROMPT = (
    "Find the area of one end face of {the}, in cm².",
    "Find the area of the cross-section at one end of {the}, in cm².",
)
_B_PROMPT = (
    "Hence find the total surface area of {the}, including the inside of the channel, in cm².",
    "Using your answer to (a), find the total surface area of {the} in cm², including all surfaces of the channel.",
)
_C_PROMPT = (
    "The whole surface is given one coat of {coat} costing ${rate} per square metre. Using your answer to (b), "
    "find the cost, in dollars, to the nearest cent. (1 m² = 10 000 cm²)",
    "One coat of {coat} costs ${rate} per square metre and covers the whole surface. Use your answer to (b) to "
    "find the cost of one coat, in dollars, to the nearest cent. (1 m² = 10 000 cm²)",
)


@generator("exd_x_m5_grooved_block_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m5_grooved_block_chain(rng, tier: str = "exam") -> dict:
    """Exam tier: end face, total surface area and coating cost of a block with a channel."""
    if tier != "exam":
        raise VerificationError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a block whose answers are clear of rounding boundaries"):
        length = rng.randint(40, 120)
        width = rng.randint(20, 50)
        height = rng.randint(10, 30)
        cw = rng.randint(4, width - 8)
        cd = rng.randint(3, height - 4)
        rate = Fraction(rng.randint(16, 60), 2)
        end = width * height - cw * cd
        total = 2 * end + (2 * width + 2 * height + 2 * cd) * length
        cost = Fraction(total) * rate / 10_000
        if not clear_of_boundary(cost, 2, Fraction(1, 20)):
            continue
        a_text, b_text, c_text = str(end), str(total), display(cost, 2)
        obj, the, coat = rng.choice(_OBJECTS)
        rate_text = display(rate, 2)
        first = rng.choice(_FIRST).format(obj=obj, length=length, width=width, height=height, cw=cw, cd=cd)
        prompts = [
            rng.choice(_A_PROMPT).format(the=the),
            rng.choice(_B_PROMPT).format(the=the),
            rng.choice(_C_PROMPT).format(coat=coat, rate=rate_text),
        ]
        if keys_absent(first, prompts, [a_text, b_text, c_text]):
            break

    plain = 2 * (width * height + width * length + height * length)
    a_dis = tagged_distractors(
        a_text,
        0,
        [
            (Fraction(width * height), None),  # ignored the channel
            (Fraction(width * height + cw * cd), None),  # added the channel instead of removing it
            (Fraction((width - cw) * (height - cd)), None),  # multiplied the remaining widths and heights
            (Fraction(width * height - cw), None),
        ],
    )
    b_dis = tagged_distractors(
        b_text,
        0,
        [
            (Fraction(plain), None),  # surface of the uncut block
            (Fraction(2 * end + (2 * width + 2 * height) * length), None),  # left out the walls of the channel
            (Fraction(end + (2 * width + 2 * height + 2 * cd) * length), None),  # counted one end only
            (Fraction(plain - cw * length), None),  # removed the strip from the top but added nothing
        ],
    )
    c_dis = tagged_distractors(
        c_text,
        2,
        [
            (Fraction(total) * rate / 100, None),  # 100 cm² in a m²
            (Fraction(total) * rate / 1000, None),
            (Fraction(plain) * rate / 10_000, None),  # priced the uncut block
        ],
    )
    return dict(
        text=first,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="number",
                answer=end,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[f"End face = {width} × {height} − {cw} × {cd} = {end} cm²."],
                distractors=a_dis,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=total,
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=3,
                sensible_range=(0, None),
                follow_through=f"2*part_a + {2 * width + 2 * height + 2 * cd}*{length}",
                working=[
                    f"Two ends: 2 × {end} = {2 * end} cm².",
                    f"The strip around the cross-section has perimeter {2 * width + 2 * height + 2 * cd} cm "
                    f"(the channel adds its two walls and floor and removes the strip of the top it replaces).",
                    f"Long surfaces: {2 * width + 2 * height + 2 * cd} × {length} = "
                    f"{(2 * width + 2 * height + 2 * cd) * length} cm².",
                    f"Total = {b_text} cm².",
                ],
                distractors=b_dis,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=numeric_answer(rounded(cost, 2)),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_b*{rate_text}/10000",
                working=[f"{b_text} cm² = {display(Fraction(total, 10_000), 4)} m²; cost = × {rate_text} = ${c_text}."],
                distractors=c_dis,
            ),
        ],
    )
