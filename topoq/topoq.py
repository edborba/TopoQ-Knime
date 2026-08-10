"""Entry module of the TopoQ extension (referenced by knime.yml).

KNIME imports this module and registers every @knext.node found through it.
The category must be created before the node modules are imported, which is
guaranteed by importing topoq_category first inside each node module.
"""

import batch_geometry_optimizer  # noqa: F401  Batch Geometry Optimizer
import bond_distance_checker  # noqa: F401  Bond Distance Checker
import structure_correction  # noqa: F401  Structure Correction
