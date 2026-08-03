"""End-to-end tests driving the CLI, including the mdtraj-backed structural modes."""

import os

import numpy as np
import pytest

from src.cli import main

from conftest import make_ring

mdtraj = pytest.importorskip("mdtraj", reason="mdtraj is only needed for PDB/MD modes")


# Ring atoms are deliberately scattered through the file and interleaved with
# decoys, so the requested order (O5 C1 C2 C3 C4 C5) is neither the file order
# nor the sorted order. 1-based indices: O5=3, C1=1, C2=2, C3=5, C4=6, C5=8.
PDB_ATOM_LAYOUT = ["C1", "C2", "O5", "XX", "C3", "C4", "YY", "C5"]
RING_INDICES_1BASED = "3 1 2 5 6 8"
RING_ORDER_IN_FILE = [2, 0, 1, 4, 5, 7]  # 0-based positions of O5, C1..C5

REFERENCE_Q = 0.57
REFERENCE_THETA = 55.0
REFERENCE_PHI = 120.0


def write_test_pdb(path, ring_coords, with_bonds=False):
    """
    Writes a minimal single-residue PDB with the ring atoms at given coords.

    with_bonds adds CONECT records closing the ring, which is what lets mdtraj
    report bonds for this non-standard residue and therefore what makes the
    ring-connectivity check possible.
    """
    coords = np.zeros((len(PDB_ATOM_LAYOUT), 3))
    coords[RING_ORDER_IN_FILE] = ring_coords + 10.0  # keep the field positive
    # Park the decoys well away from the ring so a mis-selection cannot go unnoticed.
    coords[[3, 6]] = [[40.0, 40.0, 40.0], [45.0, 45.0, 45.0]]

    with open(path, "w") as handle:
        for i, name in enumerate(PDB_ATOM_LAYOUT):
            x, y, z = coords[i]
            handle.write(
                f"ATOM  {i + 1:>5} {name:<4} GLC A   1    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           "
                f"{name[0]}\n"
            )
        if with_bonds:
            serials = [i + 1 for i in RING_ORDER_IN_FILE]
            for k, serial in enumerate(serials):
                previous = serials[(k - 1) % len(serials)]
                nxt = serials[(k + 1) % len(serials)]
                handle.write(f"CONECT{serial:>5}{previous:>5}{nxt:>5}\n")
        handle.write("END\n")
    return str(path)


@pytest.fixture
def reference_pdb(tmp_path):
    ring = make_ring(REFERENCE_Q, REFERENCE_THETA, REFERENCE_PHI)
    return write_test_pdb(tmp_path / "sugar.pdb", ring)


def _read_params(path):
    """Returns (numeric rows, conformation labels) from a *_params.dat file."""
    numbers = np.atleast_2d(np.loadtxt(path, usecols=(0, 1, 2, 3)))
    labels = []
    with open(path) as handle:
        for line in handle:
            if not line.startswith("#") and line.strip():
                labels.append(line.split()[-1])
    return numbers, labels


def test_pdb_mode_recovers_the_known_geometry(reference_pdb, tmp_path, capsys):
    exit_code = main(["pdb", "--files", reference_pdb, "--indices", RING_INDICES_1BASED,
                      "--job", "job", "--outdir", str(tmp_path)])
    assert exit_code == 0

    numbers, labels = _read_params(str(tmp_path / "job" / "job_params.dat"))
    assert numbers.shape == (1, 4)
    # PDB stores three decimals, so allow for that rounding.
    assert numbers[0, 1] == pytest.approx(REFERENCE_Q, abs=5e-3)
    assert numbers[0, 2] == pytest.approx(REFERENCE_THETA, abs=0.5)
    assert numbers[0, 3] == pytest.approx(REFERENCE_PHI, abs=0.5)
    # theta 55 -> northern band; phi 120 -> sector int((120+15)//30) = 4
    assert labels == ["2E"]

    assert os.path.exists(tmp_path / "job" / "job_mercator.png")


def test_ring_atoms_are_taken_in_the_requested_order(reference_pdb, tmp_path, capsys):
    """
    mdtraj does not promise to honour the caller's ordering in atom_indices, and
    Cremer-Pople is order-sensitive. The loader sorts the selection and reorders
    afterwards; this checks the resolved atoms really are O5, C1..C5 in order.
    """
    main(["pdb", "--files", reference_pdb, "--indices", RING_INDICES_1BASED,
          "--job", "job", "--outdir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Ring atoms resolved to:" in out
    resolved = out.split("Ring atoms resolved to:")[1].splitlines()[0]
    # Entries look like "GLC1-O5"; compare the atom name only, since the residue
    # part ("GLC1") contains a "C1" of its own.
    names = [entry.strip().split("-")[-1] for entry in resolved.split(",")]
    assert names == ["O5", "C1", "C2", "C3", "C4", "C5"]


def test_reversed_ring_order_flips_the_chair(reference_pdb, tmp_path):
    """
    Traversing the ring the other way must map theta to 180-theta, i.e. swap the
    two chairs -- the failure mode of getting the atom order wrong.

    The reversal keeps the first atom fixed and reverses the rest
    (O5 C5 C4 C3 C2 C1). Reversing the whole list instead would also shift the
    ring by one position, and an odd cyclic shift flips the sign of q3 a second
    time, leaving theta unchanged.
    """
    main(["pdb", "--files", reference_pdb, "--indices", RING_INDICES_1BASED,
          "--job", "fwd", "--outdir", str(tmp_path)])
    first, *rest = RING_INDICES_1BASED.split()
    reversed_indices = " ".join([first, *reversed(rest)])
    main(["pdb", "--files", reference_pdb, "--indices", reversed_indices,
          "--job", "rev", "--outdir", str(tmp_path)])

    forward, _ = _read_params(str(tmp_path / "fwd" / "fwd_params.dat"))
    backward, _ = _read_params(str(tmp_path / "rev" / "rev_params.dat"))
    assert backward[0, 1] == pytest.approx(forward[0, 1], abs=1e-6)
    assert backward[0, 2] == pytest.approx(180.0 - forward[0, 2], abs=1e-4)


def test_fel_projection_is_produced_on_request(reference_pdb, tmp_path):
    """--fel projects the structure onto its landscape as an extra figure."""
    theta, phi = np.meshgrid(np.linspace(0, 180, 25), np.linspace(0, 360, 30), indexing="ij")
    surface = np.column_stack([theta.ravel(), phi.ravel(), (theta.ravel() - 55.0) ** 2 / 100.0])
    fel_path = tmp_path / "fes.dat"
    np.savetxt(fel_path, surface, fmt="%.6f")

    assert main(["pdb", "--files", reference_pdb, "--indices", RING_INDICES_1BASED,
                 "--fel", str(fel_path), "--angle-units", "deg",
                 "--job", "proj", "--outdir", str(tmp_path)]) == 0

    assert os.path.exists(tmp_path / "proj" / "proj_on_FEL.png")
    # the plain landscape name is reserved for the FEL mode
    assert not os.path.exists(tmp_path / "proj" / "proj_FEL.png")


def test_stoddart_is_produced_for_pyranoses(reference_pdb, tmp_path):
    main(["pdb", "--files", reference_pdb, "--indices", RING_INDICES_1BASED,
          "--job", "st", "--outdir", str(tmp_path)])
    assert os.path.exists(tmp_path / "st" / "st_stoddart.png")


def test_multiple_pdbs_produce_one_row_each(tmp_path):
    paths = []
    # Chairs need theta outside [15, 165]; 20 and 160 would still be half-chairs.
    thetas = [5.0, 90.0, 175.0]
    for i, theta in enumerate(thetas):
        ring = make_ring(REFERENCE_Q, theta, REFERENCE_PHI)
        paths.append(write_test_pdb(tmp_path / f"frame{i}.pdb", ring))

    assert main(["pdb", "--files", *paths, "--indices", RING_INDICES_1BASED,
                 "--job", "multi", "--outdir", str(tmp_path)]) == 0

    numbers, labels = _read_params(str(tmp_path / "multi" / "multi_params.dat"))
    assert numbers.shape == (3, 4)
    assert numbers[:, 2] == pytest.approx(thetas, abs=0.5)
    assert labels == ["4C1", "2,5B", "1C4"]
    # More than one frame also produces the time series
    assert os.path.exists(tmp_path / "multi" / "multi_timeseries.png")


def test_out_of_range_index_is_reported_clearly(reference_pdb, tmp_path, capsys):
    exit_code = main(["pdb", "--files", reference_pdb, "--indices", "3 1 2 5 6 99",
                      "--job", "job", "--outdir", str(tmp_path)])
    assert exit_code == 1
    assert "outside the structure" in capsys.readouterr().err


def test_missing_file_is_reported_clearly(tmp_path, capsys):
    exit_code = main(["pdb", "--files", str(tmp_path / "nope.pdb"),
                      "--indices", RING_INDICES_1BASED, "--outdir", str(tmp_path)])
    assert exit_code == 1
    assert "not found" in capsys.readouterr().err


# --------------------------------------------------------------------------
# ring connectivity check
# --------------------------------------------------------------------------

@pytest.fixture
def bonded_pdb(tmp_path):
    ring = make_ring(REFERENCE_Q, REFERENCE_THETA, REFERENCE_PHI)
    return write_test_pdb(tmp_path / "bonded.pdb", ring, with_bonds=True)


def test_a_real_ring_passes_the_connectivity_check(bonded_pdb, tmp_path):
    assert main(["pdb", "--files", bonded_pdb, "--indices", RING_INDICES_1BASED,
                 "--job", "ok", "--outdir", str(tmp_path)]) == 0


def test_non_ring_atoms_are_refused(bonded_pdb, tmp_path, capsys):
    """
    The decisive real-world failure: indices valid for one topology but pointing
    at unrelated atoms in another. The arithmetic happily produces numbers, so
    without this check a whole trajectory of nonsense passes unnoticed.
    """
    exit_code = main(["pdb", "--files", bonded_pdb, "--indices", "3 1 2 5 6 4",
                      "--job", "bad", "--outdir", str(tmp_path)])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "do not form a closed 6-membered ring" in err
    assert "XX" in err  # the offending atom is named


def test_scrambled_ring_order_is_refused(bonded_pdb, tmp_path, capsys):
    """Right atoms, wrong sequence: the ring never closes through those bonds."""
    exit_code = main(["pdb", "--files", bonded_pdb, "--indices", "3 2 1 5 6 8",
                      "--job", "scrambled", "--outdir", str(tmp_path)])
    assert exit_code == 1
    assert "do not form a closed 6-membered ring" in capsys.readouterr().err


def test_the_check_can_be_skipped(bonded_pdb, tmp_path):
    assert main(["pdb", "--files", bonded_pdb, "--indices", "3 1 2 5 6 4",
                 "--skip-ring-check", "--job", "forced", "--outdir", str(tmp_path)]) == 0


def test_missing_bond_information_does_not_block(reference_pdb, tmp_path):
    """
    mdtraj derives PDB bonds from residue templates, so a non-standard sugar
    residue with no CONECT records has none. The check must then step aside
    rather than reject a perfectly good selection.
    """
    assert main(["pdb", "--files", reference_pdb, "--indices", RING_INDICES_1BASED,
                 "--job", "nobonds", "--outdir", str(tmp_path)]) == 0


# --------------------------------------------------------------------------
# furanose mode
# --------------------------------------------------------------------------

FURANOSE_LAYOUT = ["C1", "O4", "ZZ", "C2", "C3", "C4"]
FURANOSE_RING_IN_FILE = [1, 0, 3, 4, 5]        # O4, C1, C2, C3, C4
FURANOSE_INDICES = "2 1 4 5 6"                 # 1-based


def write_furanose_pdb(path, ring_coords):
    coords = np.zeros((len(FURANOSE_LAYOUT), 3))
    coords[FURANOSE_RING_IN_FILE] = ring_coords + 10.0
    coords[2] = [40.0, 40.0, 40.0]

    with open(path, "w") as handle:
        for i, name in enumerate(FURANOSE_LAYOUT):
            x, y, z = coords[i]
            handle.write(
                f"ATOM  {i + 1:>5} {name:<4} RIB A   1    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           {name[0]}\n"
            )
        handle.write("END\n")
    return str(path)


def test_five_indices_run_the_furanose_pipeline(tmp_path, capsys):
    from test_furanose import envelope

    path = write_furanose_pdb(tmp_path / "ribose.pdb", envelope(3))  # C3-endo
    assert main(["pdb", "--files", path, "--indices", FURANOSE_INDICES,
                 "--job", "fur", "--outdir", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "5-membered ring (furanose)" in out
    assert "Conformation: 3E" in out
    # the furanose branch draws the wheel, not the Stoddart map
    assert os.path.exists(tmp_path / "fur" / "fur_wheel.png")
    assert not os.path.exists(tmp_path / "fur" / "fur_mercator.png")


def test_furanose_trajectory_gets_its_own_time_series(tmp_path):
    from test_furanose import make_furanose

    paths = [write_furanose_pdb(tmp_path / f"f{i}.pdb", make_furanose(0.4, p))
             for i, p in enumerate((0.0, 90.0, 180.0, 270.0))]
    assert main(["pdb", "--files", *paths, "--indices", FURANOSE_INDICES,
                 "--job", "furtraj", "--outdir", str(tmp_path)]) == 0

    numbers, labels = _read_params(str(tmp_path / "furtraj" / "furtraj_params.dat"))
    assert numbers.shape == (4, 4)
    assert len(set(labels)) == 4  # four distinct points around the wheel
    assert os.path.exists(tmp_path / "furtraj" / "furtraj_timeseries.png")
    assert os.path.exists(tmp_path / "furtraj" / "furtraj_wheel.png")


def test_bad_indices_are_rejected_before_touching_the_file(reference_pdb, tmp_path, capsys):
    exit_code = main(["pdb", "--files", reference_pdb,
                      "--indices", "Ex: 11 12 13 14 15 16", "--outdir", str(tmp_path)])
    assert exit_code == 1
    assert "not a whole number" in capsys.readouterr().err
