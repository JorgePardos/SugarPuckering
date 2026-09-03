"""CLI tests that do not require mdtraj (the FEL path and argument handling)."""

import os

import numpy as np
import pytest

from src.cli import build_parser, main


@pytest.fixture
def fel_file(tmp_path):
    theta, phi = np.meshgrid(np.linspace(0, 180, 30), np.linspace(0, 360, 40), indexing="ij")
    energy = (theta - 90.0) ** 2 / 400.0
    rows = np.column_stack([theta.ravel(), phi.ravel(), energy.ravel()])
    path = tmp_path / "fes.dat"
    with open(path, "w") as handle:
        handle.write("#! FIELDS theta phi file.free\n")
        np.savetxt(handle, rows, fmt="%.6f")
    return str(path)


def test_fel_mode_writes_the_plot(fel_file, tmp_path, capsys):
    assert main(["fel", "--file", fel_file, "--job", "felrun", "--outdir", str(tmp_path)]) == 0
    assert os.path.exists(tmp_path / "felrun" / "felrun_FEL.png")


def test_fel_mode_reports_the_plumed_header(fel_file, tmp_path, capsys):
    main(["fel", "--file", fel_file, "--job", "felrun", "--outdir", str(tmp_path)])
    assert "PLUMED header fields: theta, phi, file.free" in capsys.readouterr().out


def test_fel_mode_falls_back_to_the_input_name(fel_file, tmp_path):
    main(["fel", "--file", fel_file, "--outdir", str(tmp_path)])
    assert os.path.exists(tmp_path / "fes" / "fes_FEL.png")


def test_fel_mode_reports_a_missing_file(tmp_path, capsys):
    assert main(["fel", "--file", str(tmp_path / "nope.dat"), "--outdir", str(tmp_path)]) == 1
    assert "valid FEL data file" in capsys.readouterr().err


def test_fel_options_are_accepted(fel_file, tmp_path):
    assert main(["fel", "--file", fel_file, "--job", "opts", "--outdir", str(tmp_path),
                 "--angle-units", "deg", "--energy-label", "Free Energy (kJ/mol)",
                 "--contour-step", "2.5", "--cmap", "magma"]) == 0
    assert os.path.exists(tmp_path / "opts" / "opts_FEL.png")


def test_structural_modes_require_indices():
    parser = build_parser()
    for argv in (["pdb", "--files", "a.pdb"], ["md", "--top", "a.prmtop", "--traj", "b.nc"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_a_mode_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
