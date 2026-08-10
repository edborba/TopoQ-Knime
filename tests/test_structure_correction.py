"""Tests for the pure helpers of structure_correction.

Runs without KNIME: the knime modules are stubbed before importing the extension.
Run with pytest, or directly: python tests/test_structure_correction.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "topoq"))

sys.modules.setdefault("knime", MagicMock())
sys.modules["knime.extension"] = MagicMock()
sys.modules["knime.types"] = MagicMock()
# None in sys.modules makes the import raise ModuleNotFoundError, exercising the
# fallback paths that run without the KNIME Chemistry Types extension.
sys.modules["knime.types.chemistry"] = None

import pandas as pd

import structure_correction as mx

ORIGINAL_SDF = (
    "241\n"
    " OpenBabel\n"
    "\n"
    "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
    "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "    1.5000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  1  2  1  0  0  0  0\n"
    "M  END\n"
    "$$$$\n"
)

MOPAC_SDF = (
    "241\n"
    " OpenBabel\n"
    "\n"
    "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
    "    0.1234    0.5678    0.9012 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "    1.4321   -0.8765    0.2109 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  1  2  1  0  0  0  0\n"
    "M  END\n"
    "$$$$\n"
)


def test_extract_and_apply_atom_positions():
    positions = mx._extract_atom_positions(MOPAC_SDF)
    assert len(positions) == 2

    result = mx._apply_atom_positions(ORIGINAL_SDF, positions)

    assert "    0.1234    0.5678    0.9012 C" in result
    assert "    1.4321   -0.8765    0.2109 O" in result
    # Counts line, trailing atom fields and bond block are untouched.
    assert "  2  1  0  0  0  0  0  0  0  0999 V2000" in result
    assert "  1  2  1  0  0  0  0" in result


def test_output_is_mopac_table_with_corrected_structure():
    # The first table's extra columns (Note) must NOT reach the output; the output
    # keeps exactly the MOPAC table's columns, with the structure corrected.
    original = pd.DataFrame(
        {
            "Name": ["241"],
            "Structure": [ORIGINAL_SDF],
            "Note": ["not carried over"],
        }
    )
    mopac = pd.DataFrame(
        {
            "ID": ["241"],
            "Molecule (MOPAC)": [MOPAC_SDF],
            "Energy": [-23.4],
            "Converged": [True],
        }
    )

    output = mx._correct_structures(
        original, mopac, "Name", "Structure", "ID", "Molecule (MOPAC)"
    )

    assert list(output.columns) == ["ID", "Molecule (MOPAC)", "Energy", "Converged"]
    corrected = output.at[0, "Molecule (MOPAC)"]
    # MOPAC coordinates on the original bond table.
    assert "    0.1234    0.5678    0.9012 C" in corrected
    assert "  1  2  1  0  0  0  0" in corrected
    # Other MOPAC columns untouched.
    assert output.at[0, "Energy"] == -23.4
    assert bool(output.at[0, "Converged"]) is True


def test_shared_column_names_are_no_problem():
    # Both tables using the same column names (e.g. output of the old single node)
    # must work: only table 2's columns are output.
    original = pd.DataFrame({"Name": ["241"], "Structure": [ORIGINAL_SDF]})
    mopac = pd.DataFrame(
        {"Name": ["241"], "Structure": [MOPAC_SDF], "Energy": [-1.0]}
    )

    output = mx._correct_structures(
        original, mopac, "Name", "Structure", "Name", "Structure"
    )

    assert list(output.columns) == ["Name", "Structure", "Energy"]
    assert "    0.1234    0.5678    0.9012 C" in output.at[0, "Structure"]


def test_no_matching_original_leaves_cell_empty():
    original = pd.DataFrame({"Name": ["999"], "Structure": [ORIGINAL_SDF]})
    mopac = pd.DataFrame({"ID": ["241"], "Molecule (MOPAC)": [MOPAC_SDF]})

    output = mx._correct_structures(
        original, mopac, "Name", "Structure", "ID", "Molecule (MOPAC)"
    )

    assert output.at[0, "Molecule (MOPAC)"] is None


def test_ambiguous_original_id_leaves_cell_empty():
    original = pd.DataFrame(
        {"Name": ["241", "241"], "Structure": [ORIGINAL_SDF, ORIGINAL_SDF]}
    )
    mopac = pd.DataFrame({"ID": ["241"], "Molecule (MOPAC)": [MOPAC_SDF]})

    output = mx._correct_structures(
        original, mopac, "Name", "Structure", "ID", "Molecule (MOPAC)"
    )

    assert output.at[0, "Molecule (MOPAC)"] is None


def test_atom_count_mismatch_leaves_cell_empty():
    three_atom_mopac = MOPAC_SDF.replace(
        "  1  2  1  0  0  0  0\n",
        "    2.0000    2.0000    2.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  1  0  0  0  0\n",
    )
    original = pd.DataFrame({"Name": ["241"], "Structure": [ORIGINAL_SDF]})
    mopac = pd.DataFrame({"ID": ["241"], "Molecule (MOPAC)": [three_atom_mopac]})

    output = mx._correct_structures(
        original, mopac, "Name", "Structure", "ID", "Molecule (MOPAC)"
    )

    assert output.at[0, "Molecule (MOPAC)"] is None


def test_missing_mopac_structure_stays_missing():
    original = pd.DataFrame({"Name": ["241"], "Structure": [ORIGINAL_SDF]})
    mopac = pd.DataFrame(
        {"ID": ["241"], "Molecule (MOPAC)": [None], "Energy": [2.0]}
    )

    output = mx._correct_structures(
        original, mopac, "Name", "Structure", "ID", "Molecule (MOPAC)"
    )

    assert output.at[0, "Molecule (MOPAC)"] is None
    assert output.at[0, "Energy"] == 2.0


def test_to_text_and_is_missing():
    assert mx._is_missing(None) is True
    assert mx._is_missing(float("nan")) is True
    assert mx._is_missing(pd.NA) is True
    assert mx._is_missing("V2000") is False
    assert mx._to_text(None) == ""
    assert mx._to_text(float("nan")) == ""
    assert mx._to_text("abc") == "abc"


def main():
    failures = 0
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for name, test in tests:
        try:
            test()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001 - report and keep going
            failures += 1
            print(f"FAIL {name}: {error!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
