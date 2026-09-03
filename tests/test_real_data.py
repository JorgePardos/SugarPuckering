"""
Tests against the real research data in tests/data.

Those files are large (hundreds of MB for the trajectory) and are not committed,
so every test here skips when its input is absent. They are the checks that
actually closed "Comprobar funcionamiento FEL" from ToDo.txt.
"""

import os

import numpy as np
import pytest

from src.analysis import AnalysisError, load_fel, load_ring_coordinates, compute_puckering
from src.math_core import get_strict_conformation
from src.plotting import plot_fel_mercator

DATA = os.path.join(os.path.dirname(__file__), "data")
FES = os.path.join(DATA, "FEL", "fes.dat")
PDB_DIR = os.path.join(DATA, "PDBs")
PRMTOP = os.path.join(DATA, "Traj", "mutated.prmtop")
DCD = os.path.join(DATA, "Traj", "qmmm-100ps.dcd")

# Ring atoms differ between the deposited PDBs (residue TRH453) and the
# trajectory topology (TRH452) -- the numbering is not shared.
PDB_RING = "7274 7254 7256 7260 7264 7268"
TRAJ_RING = "7277 7257 7259 7263 7267 7271"

needs_fes = pytest.mark.skipif(not os.path.exists(FES), reason="tests/data/FEL/fes.dat absent")
needs_traj = pytest.mark.skipif(not (os.path.exists(PRMTOP) and os.path.exists(DCD)),
                                reason="tests/data/Traj absent")


def _indices(text):
    return [int(t) - 1 for t in text.split()]


@needs_fes
def test_the_plumed_surface_is_self_describing():
    """The '#! FIELDS' header states the column order, removing the guesswork."""
    data, fields = load_fel(FES)
    assert fields[:3] == ["p.theta", "p.phi", "ff"]
    assert data.shape[1] >= 3


@needs_fes
def test_the_surface_is_in_radians_and_kcal():
    data, _ = load_fel(FES)
    # GRID_MAX=pi,2pi in the PLUMED input
    assert data[:, 0].max() == pytest.approx(np.pi, abs=1e-3)
    assert data[:, 1].max() < 2 * np.pi
    finite = data[np.isfinite(data[:, 2]), 2]
    assert finite.min() == pytest.approx(0.0, abs=1e-6)  # CONVERT_TO_FES MINTOZERO
    assert 5.0 < finite.max() < 50.0  # kcal/mol, not kJ/mol


@needs_fes
def test_unsampled_bins_dominate_and_are_survived(tmp_path):
    """Most of this surface is +inf; rendering must not choke or drop the basins."""
    data, _ = load_fel(FES)
    assert np.isinf(data[:, 2]).sum() > np.isfinite(data[:, 2]).sum()
    path = plot_fel_mercator(data, str(tmp_path / "real"), title="real")
    assert os.path.getsize(path) > 10_000


@needs_fes
def test_auto_unit_detection_agrees_with_explicit_radians(tmp_path):
    data, _ = load_fel(FES)
    auto = plot_fel_mercator(data, str(tmp_path / "auto"))
    explicit = plot_fel_mercator(data, str(tmp_path / "rad"), angle_units="rad")
    with open(auto, "rb") as a, open(explicit, "rb") as b:
        assert a.read() == b.read()


@pytest.mark.skipif(not os.path.exists(os.path.join(PDB_DIR, "4C1.pdb")),
                    reason="tests/data/PDBs absent")
def test_the_chair_structure_is_assigned_as_a_chair():
    pytest.importorskip("mdtraj")
    selection = load_ring_coordinates(
        "PDB", _indices(PDB_RING), pdb_files=[os.path.join(PDB_DIR, "4C1.pdb")])
    assert [n.split("-")[-1] for n in selection.atom_names] == [
        "O5", "C1", "C2", "C3", "C4", "C5"]

    results = compute_puckering(selection.xyz, selection.frames)
    assert get_strict_conformation(results[0, 2], results[0, 3]) == "4C1"
    assert 0.4 < results[0, 1] < 0.8  # a sensible pyranose amplitude, in Angstrom


@needs_traj
def test_trajectory_ring_is_a_real_ring_and_gives_sane_values():
    pytest.importorskip("mdtraj")
    selection = load_ring_coordinates(
        "MD", _indices(TRAJ_RING), topology=PRMTOP, trajectory=DCD)
    assert [n.split("-")[-1] for n in selection.atom_names] == [
        "O5", "C1", "C2", "C3", "C4", "C5"]

    results = compute_puckering(selection.xyz, selection.frames)
    assert len(results) > 100
    assert np.isfinite(results).all()
    # Pyranose puckering amplitude stays near 0.6 A throughout a stable run.
    assert 0.4 < results[:, 1].mean() < 0.8
    assert results[:, 1].std() < 0.2


@needs_traj
def test_the_pdb_indices_are_refused_against_the_trajectory_topology():
    """
    Regression guard for a real mistake: the indices that are correct for the
    deposited PDBs point at unrelated atoms in mutated.prmtop, and the maths
    still returns plausible numbers. The ring check must stop that.
    """
    pytest.importorskip("mdtraj")
    with pytest.raises(AnalysisError, match="do not form a closed 6-membered ring"):
        load_ring_coordinates("MD", _indices(PDB_RING), topology=PRMTOP, trajectory=DCD)
