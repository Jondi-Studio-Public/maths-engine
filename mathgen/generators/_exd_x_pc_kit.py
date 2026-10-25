"""Small shared builder for the whole-number exam chains in the ``exd_x_pc1_*`` and ``exd_x_tg1_*`` modules.

Not a generator module (the leading underscore keeps ``mathgen.generators`` from importing it as one).
"""

from __future__ import annotations

from ..answers import Distractor
from ..core import GRADER_TOLERANCE, latex_answer, pick_distractors
from ._spread_kit import mixed


def count_part(
    prompt,
    key,
    marks,
    working,
    wrong,
    *,
    follow=None,
    want=3,
    signed=False,
    lo=None,
    hi=None,
    exclude=(),
    ok=None,
    step=1,
):
    """A ``number`` part whose key is a whole number: positive, or any non-zero whole number when ``signed``.

    The caller's generator must be wrapped in ``uses_rng`` so the option split varies by seed.
    ``wrong`` lists ``(value, misconception id)`` in priority order. Every option, named slip or plain near miss
    (key plus or minus 1 to 3 steps of ``step``), must be a value the part's stem allows: a whole number from ``lo``
    (default 1, or no floor if ``signed``) to ``hi`` (default none), not in ``exclude``, and accepted by ``ok`` if
    that is given. Returns None when fewer than ``want`` survive, so the caller draws again.
    """
    key = int(key)
    text = str(key)
    floor = lo if lo is not None else (-(10**6) if signed else 1)

    def valid(v):
        return v >= floor and (hi is None or v <= hi) and v not in exclude and (ok is None or ok(v))

    cands = [
        Distractor(str(int(v)), tag, int(v)) for v, tag in wrong if v is not None and v == int(v) and valid(int(v))
    ]
    # Spread the key's rank among the options: a random number of the named slips sit below it, the rest above it, with
    # plain near misses filling a side that has too few.
    near = [
        Distractor(str(key + d * step), None, key + d * step) for d in (-3, -2, -1, 1, 2, 3) if valid(key + d * step)
    ]
    cands = mixed(key, cands, want, near, lo=floor, hi=hi)
    picked = pick_distractors(text, cands, want, tolerance=GRADER_TOLERANCE)
    if len(picked) < want:
        return None
    picked = _keep_round_key_company(key, picked, valid)
    if not all(valid(int(d.value)) for d in picked):
        raise ValueError("retry")
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=key,
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=picked,
    )
    if floor >= 0:
        part["sensible_range"] = (0, None)
    if follow:
        part["follow_through"] = follow
    return part


def _keep_round_key_company(key, picked, valid):
    """A round key (a non-zero multiple of 10) must not be the lone round option: a learner would pick it unseen.
    When no wrong option is round, the option nearest to a round neighbour of the key on its own side is replaced by it."""
    if key == 0 or key % 10 or any(int(d.value) % 10 == 0 and d.value != 0 for d in picked):
        return picked
    taken = {int(d.value) for d in picked}
    for gap in (10, -10, 20, -20):
        value = key + gap
        if value in taken or not valid(value):
            continue
        same_side = [i for i, d in enumerate(picked) if (d.value < key) == (gap < 0)]
        if same_side:
            nearest = min(same_side, key=lambda i: abs(picked[i].value - value))
            return [*picked[:nearest], Distractor(str(value), None, value), *picked[nearest + 1 :]]
    return picked
