"""The ``canvas`` key: how a figure asks the learner to answer by working on the graph.

A ``cartesian`` or ``number_line`` figure (`mathgen.figures`) may carry a ``canvas`` dict, as a ``grid_drawing``
figure carries one for a ``grid_cells`` answer. The figure draws what is *given*; the canvas says what the front end
lets the learner add to it and which reply it sends back. The figure never holds the answer: the whole spec is sent to
clients, so a key outside :data:`CANVAS_KEYS` is refused.

Keys of ``canvas``
    - ``reply`` (required): the answer kind the front end submits, one of :data:`REPLY_KINDS`. The widget writes the
      reply JSON documented for that kind in ``docs/design/interactive-graph-answers.md``.
    - ``tools`` (required): the tools the widget enables, a non-empty list from :data:`TOOLS`. Every tool must
      produce the ``reply`` kind (:data:`TOOLS` says which).
    - ``snap`` (required): the grid step every placed point snaps to, in the figure's own units. The grader refuses a
      reply point off this lattice, so the widget must snap to it exactly and send decimals or ``"a/b"`` text.
    - ``vertex_labels`` (``polygon`` replies): one short name per vertex, in the order the reply lists them (the
      image of A, B, C is A', B', C'), shown beside each point the learner places.
    - ``max_points`` (optional): how many points the widget lets the learner place, at most 60.

The plane's ``x_range`` and ``y_range`` (or a number line's ``range``) are the canvas bounds: the grader refuses a
reply outside them, so the figure and the question's answer carry the same ranges (a generator test checks it).
A platform that cannot run the widget draws the figure as it draws any other and shows the typed form of the reply
(the question's ``answer_text`` is always that form).
"""

from __future__ import annotations

from mathgen.answers import ANSWER_KINDS

#: tool -> the reply kind it produces.
TOOLS: dict[str, str] = {
    "plot_points": "points",
    "line_two_points": "line_graph",
    "polyline": "polyline",
    "half_plane": "half_planes",
    "polygon_points": "polygon",
    "parabola_vertex_point": "curve",
    "curve_points": "curve",
    "number_line_marks": "number_line_set",
}
REPLY_KINDS = frozenset(TOOLS.values())
CANVAS_KEYS = frozenset({"reply", "tools", "snap", "vertex_labels", "max_points"})
PLANE_TOOLS = frozenset(t for t in TOOLS if t != "number_line_marks")
NUMBER_LINE_TOOLS = frozenset({"number_line_marks"})
MAX_POINTS = 60
MAX_LABEL = 8


def canvas(reply: str, tools, snap, *, vertex_labels=None, max_points: int | None = None) -> dict:
    """A ``canvas`` dict for a figure's ``canvas`` argument. Validation happens when the figure is built."""
    out: dict = {"reply": reply, "tools": list(tools), "snap": snap}
    if vertex_labels is not None:
        out["vertex_labels"] = list(vertex_labels)
    if max_points is not None:
        out["max_points"] = max_points
    return out


def validate_canvas(canvas, space: str) -> list[str]:
    """Problems with a figure's ``canvas``; ``space`` is ``"plane"`` or ``"number_line"``."""
    if not isinstance(canvas, dict):
        return ["canvas must be a dict"]
    problems = [f"canvas has an unknown key {k!r}" for k in sorted(canvas, key=repr) if k not in CANVAS_KEYS]
    reply = canvas.get("reply")
    if not isinstance(reply, str) or reply not in REPLY_KINDS or reply not in ANSWER_KINDS:
        problems.append(f"canvas.reply must be one of {', '.join(sorted(REPLY_KINDS))}")
    tools = canvas.get("tools")
    allowed = PLANE_TOOLS if space == "plane" else NUMBER_LINE_TOOLS
    if not isinstance(tools, list) or not tools or len(set(map(repr, tools))) != len(tools):
        problems.append("canvas.tools must be a non-empty list of different tools")
    else:
        for tool in tools:
            if not isinstance(tool, str) or tool not in allowed:
                problems.append(
                    f"canvas tool {tool!r} is not available on a {space} (known: {', '.join(sorted(allowed))})"
                )
            elif isinstance(reply, str) and TOOLS[tool] != reply:
                problems.append(f"canvas tool {tool!r} produces a {TOOLS[tool]!r} reply, not {reply!r}")
    snap = canvas.get("snap")
    if isinstance(snap, bool) or not isinstance(snap, (int, float)) or not snap > 0:
        problems.append("canvas.snap must be a positive number")
    labels = canvas.get("vertex_labels")
    if labels is not None:
        if reply != "polygon":
            problems.append("canvas.vertex_labels belongs to a polygon reply")
        elif (
            not isinstance(labels, list)
            or not 3 <= len(labels) <= 12
            or not all(isinstance(t, str) and 0 < len(t.strip()) <= MAX_LABEL for t in labels)
            or len(set(labels)) != len(labels)
        ):
            problems.append("canvas.vertex_labels is 3 to 12 different names of at most 8 characters")
    cap = canvas.get("max_points")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= MAX_POINTS):
        problems.append(f"canvas.max_points is a whole number from 1 to {MAX_POINTS}")
    return problems
