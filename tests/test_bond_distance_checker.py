"""Tests for the pure helpers of bond_distance_checker.

Runs without KNIME: the knime modules are stubbed before importing the extension.
Run with pytest, or directly: python tests/test_bond_distance_checker.py
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

import bond_distance_checker as mx


def _atom_line(element, x, y, z):
    return (
        f"{x:10.4f}{y:10.4f}{z:10.4f} {element:<3} 0  0  0  0  0  0  0  0  0  0  0  0"
    )


def _make_sdf(atoms, bonds):
    lines = ["", " TestMol", ""]
    lines.append(f"{len(atoms):3}{len(bonds):3}  0  0  0  0  0  0  0  0999 V2000")
    for element, x, y, z in atoms:
        lines.append(_atom_line(element, x, y, z))
    for a1, a2, bond_type in bonds:
        lines.append(f"{a1:3}{a2:3}{bond_type:3}  0")
    lines.append("M  END")
    lines.append("$$$$")
    return "\n".join(lines) + "\n"


def test_parse_atom_line_fixed_width_glued_coordinates():
    # -1000.1234 fills all 10 columns, so split() would glue x and y together;
    # only the fixed-width path parses this line.
    line = f"{-1000.1234:10.4f}{-2000.5678:10.4f}{0.0:10.4f} C   0  0  0"
    assert mx._parse_atom_line(line) == ("C", -1000.1234, -2000.5678, 0.0)


def test_parse_atom_line_split_fallback():
    assert mx._parse_atom_line("1.0 2.0 3.0 C") == ("C", 1.0, 2.0, 3.0)
    assert mx._parse_atom_line("not an atom line") is None


def test_parse_bond_line():
    assert mx._parse_bond_line("  1  2  1  0") == (1, 2, 1)
    # Bond 100-101 in fixed 3-column fields, unparseable via split().
    assert mx._parse_bond_line("100101  1  0") == (100, 101, 1)
    assert mx._parse_bond_line("M  END") is None


def test_parse_sdf_atoms_and_bonds():
    sdf = _make_sdf(
        [("C", 0.0, 0.0, 0.0), ("O", 1.5, 0.0, 0.0)],
        [(1, 2, 1)],
    )
    atoms, bonds = mx._parse_sdf_atoms_and_bonds(sdf)
    assert atoms == [("C", 0.0, 0.0, 0.0), ("O", 1.5, 0.0, 0.0)]
    assert bonds == [(1, 2, 1)]

    assert mx._parse_sdf_atoms_and_bonds("too short") == ([], [])
    assert mx._parse_sdf_atoms_and_bonds("") == ([], [])


def test_check_bonds_ok():
    # C-O at 1.5 A vs limit (0.76 + 0.66) * 1.3 = 1.846 A: no warning.
    sdf = _make_sdf([("C", 0.0, 0.0, 0.0), ("O", 1.5, 0.0, 0.0)], [(1, 2, 1)])
    has_warning, max_dist, detail = mx._check_bonds(sdf, 1.3)
    assert has_warning is False
    assert max_dist == 1.5
    assert detail == ""


def test_check_bonds_long_bond():
    sdf = _make_sdf([("C", 0.0, 0.0, 0.0), ("O", 2.0, 0.0, 0.0)], [(1, 2, 1)])
    has_warning, max_dist, detail = mx._check_bonds(sdf, 1.3)
    assert has_warning is True
    assert max_dist == 2.0
    assert "C1-O2" in detail and "max" in detail


def test_check_bonds_short_bond():
    # C-O at 0.3 A, below (0.76 + 0.66) * 0.6 = 0.852 A: collapsed atoms.
    sdf = _make_sdf([("C", 0.0, 0.0, 0.0), ("O", 0.3, 0.0, 0.0)], [(1, 2, 1)])
    has_warning, _max_dist, detail = mx._check_bonds(sdf, 1.3)
    assert has_warning is True
    assert "min" in detail

    # Short check disabled with factor 0.0.
    has_warning, _max_dist, detail = mx._check_bonds(sdf, 1.3, min_tolerance=0.0)
    assert has_warning is False
    assert detail == ""


def test_check_bonds_unparseable_is_none():
    has_warning, max_dist, detail = mx._check_bonds("garbage text", 1.3)
    assert has_warning is None
    assert math.isnan(max_dist)
    assert detail == ""


def test_check_bonds_v3000_is_none():
    v3000 = "\n TestMol\n\n  0  0  0     0  0            999 V3000\nM  END\n"
    has_warning, _max_dist, _detail = mx._check_bonds(v3000, 1.3)
    assert has_warning is None


def test_check_bonds_no_bonds_is_false():
    sdf = _make_sdf([("C", 0.0, 0.0, 0.0)], [])
    has_warning, max_dist, detail = mx._check_bonds(sdf, 1.3)
    assert has_warning is False
    assert math.isnan(max_dist)
    assert detail == ""


def test_check_bonds_all_invalid_indices_is_none():
    # The bond references atoms 5 and 6, which do not exist: nothing was verified,
    # so the result must be None (not verified), not False (no problem found).
    sdf = _make_sdf([("C", 0.0, 0.0, 0.0), ("O", 1.5, 0.0, 0.0)], [(5, 6, 1)])
    has_warning, max_dist, detail = mx._check_bonds(sdf, 1.3)
    assert has_warning is None
    assert math.isnan(max_dist)
    assert detail == ""


def test_check_bonds_collects_unknown_elements():
    sdf = _make_sdf([("Pt", 0.0, 0.0, 0.0), ("C", 2.0, 0.0, 0.0)], [(1, 2, 1)])
    unknown: set[str] = set()
    # Pt is not tabulated: default radius (1.50 + 0.76) * 1.3 = 2.938 > 2.0, no flag.
    has_warning, _max_dist, _detail = mx._check_bonds(sdf, 1.3, unknown_elements=unknown)
    assert has_warning is False
    assert unknown == {"Pt"}


def test_extract_sdf_text_plain_types():
    assert mx._extract_sdf_text("any string") == "any string"
    assert mx._extract_sdf_text(None) == ""
    assert "V2000" in mx._extract_sdf_text(b"header\nV2000\nM  END")


def test_extract_sdf_text_wrapper_attribute():
    sdf = _make_sdf([("C", 0.0, 0.0, 0.0)], [])

    class _Wrapper:
        def __init__(self, text):
            self._text = text

        def __str__(self):
            return "<Wrapper>"

    assert mx._extract_sdf_text(_Wrapper(sdf)) == sdf


def test_extract_sdf_text_survives_raising_wrappers():
    # A wrapper whose attribute access and methods raise must not crash the node;
    # extraction falls back to str().
    class _Hostile:
        @property
        def sdf(self):
            raise RuntimeError("boom")

        def get_sdf(self):
            raise RuntimeError("boom")

        def __str__(self):
            return "not an sdf"

    assert mx._extract_sdf_text(_Hostile()) == "not an sdf"


def test_is_missing():
    assert mx._is_missing(None) is True
    assert mx._is_missing(float("nan")) is True
    assert mx._is_missing(pd.NA) is True
    assert mx._is_missing("") is False
    assert mx._is_missing("V2000") is False
    assert mx._is_missing(0.0) is False


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
