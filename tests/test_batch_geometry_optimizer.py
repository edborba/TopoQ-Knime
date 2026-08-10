"""Tests for the pure helpers of batch_geometry_optimizer.

Runs without KNIME: the knime modules are stubbed before importing the extension.
Run with pytest, or directly: python tests/test_batch_geometry_optimizer.py
"""

import math
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

import batch_geometry_optimizer as mx


def _make_results_row(mol_id, molecule="SDF TEXT"):
    return {
        "ID": mol_id,
        "Molecule": molecule,
        "FINAL HEAT OF FORMATION (KCAL/MOL)": "-23.45678",
        "COSMO AREA (SQUARE ANGSTROMS)": "123.45",
        "COSMO VOLUME (CUBIC ANGSTROMS)": "456.78",
        "GRADIENT NORM": "0.04321",
        "IONIZATION POTENTIAL (EV)": "10.123",
        "homo_lumo": "-10.123 1.234",
        "alpha_homo_lumo": "",
        "beta_homo_lumo": "",
    }


def test_build_results_table_order_and_values():
    # Input order must be preserved even when results arrive in another order.
    results = pd.DataFrame(
        [_make_results_row("b", "SDF B"), _make_results_row("a", "SDF A")]
    )
    output = mx._build_results_table(["a", "b"], results, "Name")

    assert list(output.columns) == (
        ["Name", mx.MOPAC_MOLECULE_COLUMN] + mx.NUMERIC_COLUMNS
    )
    assert list(output["Name"]) == ["a", "b"]
    assert list(output[mx.MOPAC_MOLECULE_COLUMN]) == ["SDF A", "SDF B"]
    assert output.at[0, "FINAL HEAT OF FORMATION (KCAL/MOL)"] == -23.45678
    assert output.at[0, "HOMO (EV)"] == -10.123
    assert output.at[0, "LUMO (EV)"] == 1.234


def test_build_results_table_missing_molecule():
    # Molecule "b" has no result: its row must exist with missing values.
    results = pd.DataFrame([_make_results_row("a")])
    output = mx._build_results_table(["a", "b"], results, "Name")

    assert list(output["Name"]) == ["a", "b"]
    assert output.at[1, mx.MOPAC_MOLECULE_COLUMN] is None
    assert math.isnan(output.at[1, "FINAL HEAT OF FORMATION (KCAL/MOL)"])


def test_build_results_table_empty_results():
    output = mx._build_results_table(["a", "b"], pd.DataFrame(), "Name")

    assert list(output.columns) == (
        ["Name", mx.MOPAC_MOLECULE_COLUMN] + mx.NUMERIC_COLUMNS
    )
    assert list(output["Name"]) == ["a", "b"]
    assert all(value is None for value in output[mx.MOPAC_MOLECULE_COLUMN])
    assert all(math.isnan(output.at[i, col]) for i in (0, 1) for col in mx.NUMERIC_COLUMNS)


def test_resolve_homo_lumo_columns_priority():
    df = pd.DataFrame(
        [
            {"ID": "a", "homo_lumo": "-9.1 1.2", "alpha_homo_lumo": "", "beta_homo_lumo": ""},
            {"ID": "b", "homo_lumo": "", "alpha_homo_lumo": "-5.0 0.5", "beta_homo_lumo": "-6.0 0.6"},
        ]
    )
    resolved = mx._resolve_homo_lumo_columns(df)
    assert list(resolved["HOMO (EV)"]) == ["-9.1", "-5.0"]
    assert list(resolved["LUMO (EV)"]) == ["1.2", "0.5"]


def test_validate_molecule_ids():
    mx._validate_molecule_ids(pd.Series(["a", "b"]))  # must not raise
    try:
        mx._validate_molecule_ids(pd.Series(["a", "a"]))
    except ValueError as error:
        assert "duplicated" in str(error)
    else:
        raise AssertionError("ValueError was not raised")


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
