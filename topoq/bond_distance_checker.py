import math
import logging

import knime.extension as knext
from topoq_category import topoq_category
import pandas as pd

KNIME_LOGGER = logging.getLogger(__name__)

# Single-bond covalent radii in Angstroms (Alvarez 2008 / CRC values)
COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76,
    "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.61, "Fe": 1.52,
    "Co": 1.50, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20,
    "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16, "Rb": 2.20, "Sr": 1.95,
    "I": 1.39, "Sn": 1.39, "Pb": 1.46,
}

_DEFAULT_RADIUS = 1.50

# Detailed per-row warnings beyond this count are suppressed; the summary at the end
# still reports the totals.
_MAX_ROW_WARNINGS = 10


def _parse_atom_line(line: str):
    # V2000 fields are fixed width (coordinates in 10 columns, element in columns
    # 32-34). split() would glue neighbouring fields together when a coordinate
    # fills its whole width (e.g. values <= -1000).
    try:
        x = float(line[0:10])
        y = float(line[10:20])
        z = float(line[20:30])
        element = line[31:34].strip()
        if element:
            return element, x, y, z
    except ValueError:
        pass

    parts = line.split()
    if len(parts) >= 4:
        try:
            return parts[3], float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            pass
    return None


def _parse_bond_line(line: str):
    # Atom indices in fixed 3-column fields: bond 100-101 comes out as "100101  1"
    # and split() would silently drop it in large molecules.
    try:
        return int(line[0:3]), int(line[3:6]), int(line[6:9])
    except ValueError:
        pass

    parts = line.split()
    if len(parts) >= 3:
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            pass
    return None


def _parse_sdf_atoms_and_bonds(sdf_text: str):
    """Return (atoms, bonds) from an SDF/MOL block.

    atoms: list of (element, x, y, z)
    bonds: list of (atom1_1indexed, atom2_1indexed, bond_type)
    """
    # Do NOT strip() the whole text: SDF is a fixed-line-position format and
    # the molecule-name line (line 0) is often blank. Stripping would shift
    # every subsequent line up by one and break the counts-line lookup.
    lines = sdf_text.splitlines()
    if len(lines) < 4:
        return [], []

    counts_line = lines[3]
    try:
        n_atoms = int(counts_line[0:3])
        n_bonds = int(counts_line[3:6])
    except (ValueError, IndexError):
        return [], []

    atoms: list[tuple[str, float, float, float]] = []
    for i in range(4, 4 + n_atoms):
        if i >= len(lines):
            break
        atom = _parse_atom_line(lines[i])
        if atom is not None:
            atoms.append(atom)

    bonds: list[tuple[int, int, int]] = []
    for i in range(4 + n_atoms, 4 + n_atoms + n_bonds):
        if i >= len(lines):
            break
        bond = _parse_bond_line(lines[i])
        if bond is not None:
            bonds.append(bond)

    return atoms, bonds


_SDF_WRAPPER_ATTRS = (
    "sdf", "_sdf", "sdf_value", "value", "_value", "data", "_data",
    "text", "_text", "string_value", "raw_value", "molblock", "mol_block",
)


def _is_missing(value) -> bool:
    return (
        value is None
        or value is pd.NA
        or (isinstance(value, float) and math.isnan(value))
    )


def _looks_like_sdf(text: str) -> bool:
    return "V2000" in text or "V3000" in text or "M  END" in text


def _extract_sdf_text(value) -> str:
    """Extract the raw SDF string from whatever type KNIME delivers in pandas.

    Depending on the source node, the column value may be a plain str, a
    knime.types.chemistry.SdfValue object, or another wrapper. We try common
    attribute names, then callables, then str(), and only accept a result
    that actually looks like an SDF/MOL block.

    Probing must never crash the whole node, so any exception raised by a
    wrapper's attribute or method is swallowed and the next candidate is tried.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            pass

    candidates: list[str] = []

    for attr in _SDF_WRAPPER_ATTRS:
        try:
            candidate = getattr(value, attr, None)
        except Exception:
            continue
        if candidate is None:
            continue
        if callable(candidate):
            try:
                candidate = candidate()
            except Exception:
                continue
        if isinstance(candidate, str):
            candidates.append(candidate)

    for method_name in ("get_sdf", "to_sdf", "get_value", "to_string"):
        try:
            method = getattr(value, method_name, None)
        except Exception:
            continue
        if callable(method):
            try:
                result = method()
            except Exception:
                continue
            if isinstance(result, str):
                candidates.append(result)

    try:
        candidates.append(str(value))
    except Exception:
        candidates.append("")

    for candidate in candidates:
        if _looks_like_sdf(candidate):
            return candidate

    # Nothing looked like a real SDF block — return the str() fallback anyway
    # so downstream parsing fails predictably (0 atoms/0 bonds) rather than
    # raising, but log enough to diagnose.
    return candidates[-1]


def _check_bonds(
    sdf_text: str,
    tolerance: float,
    min_tolerance: float = 0.6,
    unknown_elements: set[str] | None = None,
) -> tuple[bool | None, float, str]:
    """Analyse bond distances in a single SDF structure.

    A bond is flagged when longer than (r1 + r2) * tolerance or shorter than
    (r1 + r2) * min_tolerance (collapsed atoms; 0.0 disables the lower check).
    Elements without a tabulated covalent radius use _DEFAULT_RADIUS and are
    collected into unknown_elements when a set is provided.

    Returns (has_warning, max_distance, details_string). has_warning is None when
    the structure could not be parsed — or when every bond referenced atoms
    outside the parsed range — and therefore was NOT verified; callers must not
    treat that as "no problem found".
    """
    atoms, bonds = _parse_sdf_atoms_and_bonds(sdf_text)
    if not atoms:
        return None, float("nan"), ""
    if not bonds:
        # Valid structure without bonds (e.g. an isolated atom): nothing to violate.
        return False, float("nan"), ""

    def _radius(element: str) -> float:
        radius = COVALENT_RADII.get(element)
        if radius is None:
            if unknown_elements is not None:
                unknown_elements.add(element)
            return _DEFAULT_RADIUS
        return radius

    bad_bonds: list[str] = []
    max_dist = 0.0
    checked = 0

    for a1_idx, a2_idx, _btype in bonds:
        if not (1 <= a1_idx <= len(atoms) and 1 <= a2_idx <= len(atoms)):
            continue
        checked += 1

        elem1, x1, y1, z1 = atoms[a1_idx - 1]
        elem2, x2, y2, z2 = atoms[a2_idx - 1]

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        if distance > max_dist:
            max_dist = distance

        r1 = _radius(elem1)
        r2 = _radius(elem2)
        max_expected = (r1 + r2) * tolerance
        min_expected = (r1 + r2) * min_tolerance

        if distance > max_expected:
            bad_bonds.append(
                f"{elem1}{a1_idx}-{elem2}{a2_idx}: {distance:.3f} Å "
                f"(max {max_expected:.3f} Å)"
            )
        elif distance < min_expected:
            bad_bonds.append(
                f"{elem1}{a1_idx}-{elem2}{a2_idx}: {distance:.3f} Å "
                f"(min {min_expected:.3f} Å)"
            )

    if checked == 0:
        # Every bond line referenced atoms outside the parsed range: the block is
        # inconsistent, so nothing was actually verified.
        return None, float("nan"), ""

    return bool(bad_bonds), max_dist if max_dist > 0 else float("nan"), "; ".join(bad_bonds)


@knext.node(
    name="Bond Distance Checker",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/bonds.png",
    category=topoq_category,
)
@knext.input_table(
    name="Molecules",
    description="Table containing a molecule column in SDF format.",
)
@knext.output_table(
    name="Bond Check Results",
    description="Original table with three added columns: Bond Warning, Max Bond Distance, and Long Bonds.",
)
class BondDistanceChecker:
    """Check SDF bond distances against expected covalent radii.

    For each molecule, the node calculates the Euclidean distance between every
    pair of bonded atoms and compares it to the sum of their covalent radii
    multiplied by the configured tolerance factor. Bonds that exceed this
    threshold — or fall below (r1 + r2) × short bond factor, indicating
    collapsed atoms — are reported, and the molecule is flagged with a warning.

    A flagged molecule likely underwent a failed or incomplete MOPAC geometry
    optimisation, resulting in atoms that drifted too far apart.
    """

    molecule_column = knext.ColumnParameter(
        "Molecule column",
        "Column containing the SDF structure to analyse.",
        port_index=0,
    )

    tolerance = knext.DoubleParameter(
        "Tolerance factor",
        "A bond is flagged when its length exceeds (r1 + r2) × tolerance, "
        "where r1 and r2 are the covalent radii of the two bonded atoms. "
        "Use 1.3 for a moderately strict check (default).",
        1.3,
        min_value=1.0,
        max_value=3.0,
    )

    min_tolerance = knext.DoubleParameter(
        "Short bond factor",
        "A bond is also flagged when its length is below (r1 + r2) × factor, "
        "indicating collapsed or overlapping atoms. Use 0.0 to disable this check.",
        0.6,
        min_value=0.0,
        max_value=1.0,
    )

    def configure(self, config_context, input_schema):
        mol_col = self.molecule_column
        if not mol_col:
            # Column not selected yet: keep the input schema unchanged.
            return input_schema

        input_cols = list(input_schema.column_names)
        if mol_col not in input_cols:
            raise ValueError(
                f"Selected column '{mol_col}' is missing from the input table."
            )

        try:
            import knime.types.chemistry as cet

            molecule_type = knext.logical(cet.SdfValue)
        except ImportError:
            KNIME_LOGGER.warning(
                "knime.types.chemistry is not available - the molecule column will be "
                "of type String. Install the KNIME Chemistry Types extension for full "
                "SDF support."
            )
            molecule_type = knext.string()

        # Declare the molecule column as the canonical SdfValue logical type, matching
        # what execute() actually returns. Reusing the input schema's type as-is would
        # keep a legacy adapter type (e.g. SdfAdapterValue from a native SDF Reader),
        # causing a "DataSpec generated by configure does not match spec after
        # execution" warning.
        types = [
            molecule_type if col == mol_col else input_schema[col].ktype
            for col in input_cols
        ]
        extra = knext.Schema(
            [knext.bool_(), knext.double(), knext.string()],
            ["Bond Warning", "Max Bond Distance (Å)", "Long Bonds"],
        )
        return knext.Schema(types, input_cols).append(extra)

    def execute(self, exec_context, input_table):
        mol_col = self.molecule_column
        if not mol_col:
            raise ValueError("Select the 'Molecule column' in the node dialog.")

        df = input_table.to_pandas()
        if mol_col not in df.columns:
            raise ValueError(f"Column '{mol_col}' was not found in the input table.")

        total = len(df)
        warnings: list[bool | None] = []
        max_dists: list[float] = []
        details: list[str] = []
        unknown_elements: set[str] = set()

        row_warnings = 0

        def _log_row_warning(message: str):
            nonlocal row_warnings
            row_warnings += 1
            if row_warnings <= _MAX_ROW_WARNINGS:
                KNIME_LOGGER.warning(message)
            elif row_warnings == _MAX_ROW_WARNINGS + 1:
                KNIME_LOGGER.warning(
                    "Further per-row warnings suppressed; see the summary at the end."
                )

        for i, sdf_value in enumerate(df[mol_col]):
            if _is_missing(sdf_value):
                _log_row_warning(f"Row {i}: molecule cell is empty; bonds not verified.")
                warnings.append(None)
                max_dists.append(float("nan"))
                details.append("")
            else:
                sdf_text = _extract_sdf_text(sdf_value)
                valid = _looks_like_sdf(sdf_text)
                if i == 0:
                    KNIME_LOGGER.debug(
                        f"Row 0: column '{mol_col}' type = {type(sdf_value).__name__}, "
                        f"looks like a valid SDF = {valid}"
                    )
                if not valid:
                    _log_row_warning(
                        f"Row {i}: value in column '{mol_col}' does not look like a "
                        f"valid SDF block (type={type(sdf_value).__name__}). Extracted "
                        f"snippet: {sdf_text[:200]!r}. Bonds not verified for this row."
                    )
                has_warning, max_dist, detail = _check_bonds(
                    sdf_text, self.tolerance, self.min_tolerance, unknown_elements
                )
                if has_warning is None and valid:
                    _log_row_warning(
                        f"Row {i}: the SDF block could not be verified (unsupported "
                        "format, e.g. V3000, or inconsistent atom/bond data). "
                        "'Bond Warning' will be missing for this row."
                    )
                warnings.append(has_warning)
                max_dists.append(max_dist)
                details.append(detail)

            if (i + 1) % max(1, total // 20) == 0 or i + 1 == total:
                exec_context.set_progress(
                    (i + 1) / total,
                    f"Checking bonds: {i + 1}/{total}",
                )

        # Explicit dtypes: with an empty table (or None values in between), pandas
        # inference would produce object columns and the spec would diverge from the
        # one declared in configure. "boolean" (nullable) preserves the None of
        # "not verified".
        df["Bond Warning"] = pd.array(warnings, dtype="boolean")
        df["Max Bond Distance (Å)"] = pd.array(max_dists, dtype="float64")
        df["Long Bonds"] = pd.array(details, dtype="string")

        n_flagged = sum(1 for w in warnings if w)
        n_unchecked = sum(1 for w in warnings if w is None)
        if n_flagged:
            KNIME_LOGGER.warning(
                f"{n_flagged} of {total} molecule(s) have suspicious bonds "
                f"(tolerance {self.tolerance}x)."
            )
        if n_unchecked:
            KNIME_LOGGER.warning(
                f"{n_unchecked} of {total} molecule(s) could not be verified "
                "('Bond Warning' is missing)."
            )
        if unknown_elements:
            KNIME_LOGGER.warning(
                "No covalent radius tabulated for element(s): "
                + ", ".join(sorted(unknown_elements))
                + f". The default radius of {_DEFAULT_RADIUS} Å was used."
            )

        # Re-wrap the molecule column into the canonical SdfValue type before
        # returning. Native KNIME readers can deliver a legacy "adapter" cell (e.g.
        # SdfAdapterValue) that reads fine but does not round-trip correctly through
        # from_pandas() if left untouched.
        try:
            import knime.types.chemistry as cet

            def _rewrap(value):
                if _is_missing(value):
                    return None
                sdf_text = _extract_sdf_text(value)
                return cet.SdfValue(sdf_text) if sdf_text else None

            df[mol_col] = df[mol_col].apply(_rewrap)
        except ImportError:
            pass

        exec_context.set_progress(1.0, "Done.")
        return knext.Table.from_pandas(df)
