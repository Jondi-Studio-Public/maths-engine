"""Figure kinds beyond the coordinate plane and the number line.

Each module registers its kinds with :func:`mathgen.figures.register_kind` on
import, and :mod:`mathgen.figures` imports this package once its own helpers
exist, so every kind is known wherever ``figures`` is imported.

- :mod:`.calendar`: a month calendar and a strip of daily-event picture cards.

- :mod:`.compare`: comparing lengths, masses on a pan balance and capacities of containers.
- :mod:`.data`: statistical displays (charts and plots).
- :mod:`.early`: money, the hundred chart, fraction models, counters and base-ten blocks.
- :mod:`.measure`: clocks, scales, money and early-years counting pictures.
- :mod:`.pattern`: repeating-pattern strips of shapes, numerals, letters and actions.
- :mod:`.scene`: pictures of flat shapes, solids and everyday objects for early years.
- :mod:`.space`: angles, two-dimensional diagrams, trees and Venn diagrams.
- :mod:`.tally`: tally tables and scatters of objects to sort.
"""

from mathgen.figure_kinds import calendar, compare, data, early, measure, pattern, scene, space, tally  # noqa: F401
