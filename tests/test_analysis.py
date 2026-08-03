"""Tests for the GUI-free analysis pipeline."""

import os

import numpy as np
import pytest

from src.analysis import (
    AnalysisError,
    COL_FRAME,
    COL_PHI,
    COL_Q,
    COL_THETA,
    compute_puckering,
    load_fel,
    parse_indices,
    prepare_output_dir,
    write_params_dat,
)

from conftest import make_ring


# --------------------------------------------------------------------------
# parse_indices
# --------------------------------------------------------------------------

def test_accepts_six_space_separated_indices():
    assert parse_indices("11 12 13 14 15 16") == [10, 11, 12, 13, 14, 15]


def test_accepts_commas_and_extra_whitespace():
    assert parse_indices("  11, 12 ,13   14,15 16 ") == [10, 11, 12, 13, 14, 15]


def test_order_is_preserved():
    """Ring order drives the sign of theta, so it must survive parsing intact."""
    assert parse_indices("16 15 14 13 12 11") == [15, 14, 13, 12, 11, 10]


def test_rejects_the_gui_example_text():
    """
    Regression guard: the GUI used to prefill the field with
    "Ex: 11 12 13 14 15 16" and filter tokens with str.isdigit(), which silently
    dropped the "Ex:" and analysed atoms 11-16 for anyone who never touched the
    field. Malformed input must be refused loudly instead.
    """
    with pytest.raises(AnalysisError, match="Ex:"):
        parse_indices("Ex: 11 12 13 14 15 16")


@pytest.mark.parametrize("text", ["", "   ", "\t\n"])
def test_rejects_empty_input(text):
    with pytest.raises(AnalysisError, match="No atom indices"):
        parse_indices(text)


@pytest.mark.parametrize("text", ["11 12 13 14 15 12a", "11 12 13 14 15 1.0", "a b c d e f"])
def test_rejects_non_integers(text):
    with pytest.raises(AnalysisError, match="not a whole number"):
        parse_indices(text)


@pytest.mark.parametrize("text", ["11 12 13 14 15 0", "11 12 13 14 15 -3"])
def test_rejects_non_positive_indices(text):
    """Indices are 1-based; 0 and negatives would silently wrap in numpy."""
    with pytest.raises(AnalysisError, match="1-based"):
        parse_indices(text)


def test_rejects_duplicates():
    with pytest.raises(AnalysisError, match="different"):
        parse_indices("11 12 13 14 15 11")


def test_five_indices_select_a_furanose():
    """Ring size is inferred from the count: 5 atoms means a 5-membered ring."""
    assert parse_indices("11 12 13 14 15") == [10, 11, 12, 13, 14]


@pytest.mark.parametrize("text", ["11 12 13 14", "11 12 13 14 15 16 17"])
def test_rejects_counts_that_are_neither_five_nor_six(text):
    with pytest.raises(AnalysisError, match="5 or 6"):
        parse_indices(text)


# --------------------------------------------------------------------------
# furanose pipeline
# --------------------------------------------------------------------------

def test_furanose_frames_report_phase_and_nu_max():
    from test_furanose import envelope

    results = compute_puckering(np.array([envelope(3), envelope(2)]))
    assert results.shape == (2, 4)
    # columns 2 and 3 are P and nu_max here, not theta and phi
    assert results[0, 2] == pytest.approx(18.0, abs=2.0)
    assert results[1, 2] == pytest.approx(162.0, abs=2.0)
    assert (results[:, 3] > 1.0).all()


def test_furanose_table_is_labelled_for_a_five_ring(tmp_path):
    from test_furanose import envelope

    results = compute_puckering(np.array([envelope(3)]))
    path = tmp_path / "fur_params.dat"
    write_params_dat(results, str(path), atom_names=list("ABCDE"), ring_size=5)

    text = path.read_text()
    assert "5-membered ring (furanose)" in text
    assert "P(deg)" in text and "NuMax(deg)" in text
    assert "Theta" not in text
    assert "3E" in text


def test_unsupported_ring_size_is_refused():
    with pytest.raises(AnalysisError, match="Unsupported ring size 4"):
        compute_puckering(np.zeros((2, 4, 3)))


# --------------------------------------------------------------------------
# compute_puckering
# --------------------------------------------------------------------------

def test_computes_one_row_per_frame():
    frames = [make_ring(0.57, t, 120.0) for t in (10.0, 50.0, 90.0)]
    results = compute_puckering(np.array(frames))

    assert results.shape == (3, 4)
    assert list(results[:, COL_FRAME]) == [0.0, 1.0, 2.0]
    assert results[:, COL_THETA] == pytest.approx([10.0, 50.0, 90.0], abs=1e-6)
    assert results[:, COL_Q] == pytest.approx([0.57, 0.57, 0.57], abs=1e-9)
    assert np.isfinite(results).all()


def test_single_frame_is_accepted():
    results = compute_puckering(np.array([make_ring(0.57, 0.0, 0.0)]))
    assert results.shape == (1, 4)


def test_rejects_empty_trajectory():
    with pytest.raises(AnalysisError, match="no frames"):
        compute_puckering(np.zeros((0, 6, 3)))


def test_degenerate_frame_is_reported_with_its_frame_number():
    good = make_ring(0.57, 40.0, 0.0)
    flat = np.column_stack([
        1.5 * np.cos(np.arange(6) * np.pi / 3),
        1.5 * np.sin(np.arange(6) * np.pi / 3),
        np.zeros(6),
    ])
    with pytest.raises(AnalysisError, match="Frame 2"):
        compute_puckering(np.array([good, flat, good]))


# --------------------------------------------------------------------------
# write_params_dat
# --------------------------------------------------------------------------

def test_written_table_is_numerically_reloadable(tmp_path):
    results = compute_puckering(np.array([make_ring(0.6, t, 270.0) for t in (0.0, 90.0)]))
    path = tmp_path / "job_params.dat"
    write_params_dat(results, str(path), atom_names=["O5", "C1", "C2", "C3", "C4", "C5"])

    # Every non-numeric line is '#'-prefixed, so no skiprows juggling is needed.
    reloaded = np.loadtxt(path, usecols=(0, 1, 2, 3))
    assert reloaded[:, 0] == pytest.approx([1, 2])  # 1-based frames in the file
    assert reloaded[:, 2] == pytest.approx([0.0, 90.0], abs=1e-4)


def test_written_table_records_atoms_and_conformations(tmp_path):
    results = compute_puckering(np.array([make_ring(0.6, 90.0, 270.0)]))
    path = tmp_path / "job_params.dat"
    write_params_dat(results, str(path), atom_names=["O5", "C1", "C2", "C3", "C4", "C5"])

    text = path.read_text()
    assert "# Ring atoms (in order): O5, C1, C2, C3, C4, C5" in text
    assert "Angstrom" in text
    assert "1S5" in text  # theta = 90, phi = 270


# --------------------------------------------------------------------------
# load_fel
# --------------------------------------------------------------------------

def _write_fel(path, rows, header=""):
    with open(path, "w") as handle:
        if header:
            handle.write(header)
        np.savetxt(handle, np.asarray(rows), fmt="%.6f")
    return str(path)


def test_loads_a_plain_three_column_table(tmp_path):
    path = _write_fel(tmp_path / "fes.dat", [[0, 0, 1.0], [90, 180, 2.0]])
    data, fields = load_fel(path)
    assert data.shape == (2, 3)
    assert fields is None


def test_reads_the_plumed_fields_header(tmp_path):
    """PLUMED files are self-describing; the column names resolve any ambiguity."""
    path = _write_fel(tmp_path / "fes.dat", [[0, 0, 1.0], [90, 180, 2.0]],
                      header="#! FIELDS theta phi file.free\n#! SET min_theta 0\n")
    data, fields = load_fel(path)
    assert fields == ["theta", "phi", "file.free"]
    assert data.shape == (2, 3)


def test_single_data_row_does_not_crash(tmp_path):
    """np.loadtxt returns 1-D for one row; data.shape[1] used to raise IndexError."""
    path = _write_fel(tmp_path / "fes.dat", [[10.0, 20.0, 30.0]])
    data, _ = load_fel(path)
    assert data.shape == (1, 3)


def test_extra_columns_are_kept(tmp_path):
    path = _write_fel(tmp_path / "fes.dat", [[0, 0, 1.0, 9.9], [90, 180, 2.0, 8.8]])
    data, _ = load_fel(path)
    assert data.shape == (2, 4)


def test_rejects_too_few_columns(tmp_path):
    path = _write_fel(tmp_path / "fes.dat", [[0, 1.0], [90, 2.0]])
    with pytest.raises(AnalysisError, match="at least 3 columns"):
        load_fel(path)


def test_rejects_missing_file(tmp_path):
    with pytest.raises(AnalysisError, match="valid FEL data file"):
        load_fel(str(tmp_path / "nope.dat"))


def test_rejects_unparseable_file(tmp_path):
    path = tmp_path / "fes.dat"
    path.write_text("this is not a table\n")
    with pytest.raises(AnalysisError, match="Could not parse"):
        load_fel(str(path))


# --------------------------------------------------------------------------
# prepare_output_dir
# --------------------------------------------------------------------------

def test_uses_the_job_name_when_given(tmp_path):
    job_dir, prefix, label = prepare_output_dir("myjob", "fallback", str(tmp_path))
    assert label == "myjob"
    assert os.path.isdir(job_dir)
    assert prefix == os.path.join(job_dir, "myjob")


@pytest.mark.parametrize("job_name", ["", "   ", None])
def test_falls_back_to_the_input_name(job_name, tmp_path):
    _, _, label = prepare_output_dir(job_name, "trajectory1", str(tmp_path))
    assert label == "trajectory1"


def test_is_idempotent(tmp_path):
    first = prepare_output_dir("job", "x", str(tmp_path))
    second = prepare_output_dir("job", "x", str(tmp_path))
    assert first == second


def test_label_is_not_a_path(tmp_path):
    """The label is used as a plot title, so it must not carry the directory."""
    _, prefix, label = prepare_output_dir("myjob", "fallback", str(tmp_path))
    assert os.sep not in label
    assert os.sep in prefix
