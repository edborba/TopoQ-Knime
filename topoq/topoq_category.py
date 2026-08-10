"""Shared node category for the TopoQ extension.

Kept in its own module so every node module can import it without creating a
circular import with the main extension module (topoq.py).
"""

import knime.extension as knext

topoq_category = knext.category(
    path="/community",
    level_id="topoq",
    name="TopoQ",
    description="Nodes to run semiempirical geometry optimizations and post-process the results.",
    icon="icons/topoq.png",
)
