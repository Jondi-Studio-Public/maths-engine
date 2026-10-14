"""Figure kinds beyond the coordinate plane and the number line.

Each module registers its kinds with :func:`mathgen.figures.register_kind` on
import, and :mod:`mathgen.figures` imports this package once its own helpers
exist, so every kind is known wherever ``figures`` is imported.

- :mod:`.data`: statistical displays (charts and plots).
- :mod:`.early`: money, the hundred chart, fraction models, counters and base-ten blocks.
- :mod:`.measure`: clocks, scales, money and early-years counting pictures.
- :mod:`.pattern`: repeating-pattern strips of shapes, numerals, letters and actions.
- :mod:`.scene`: pictures of flat shapes, solids and everyday objects for early years.
- :mod:`.space`: angles, two-dimensional diagrams, trees and Venn diagrams.
"""

from mathgen.figure_kinds import data, early, measure, pattern, scene, space  # noqa: F401
