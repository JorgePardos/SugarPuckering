"""Tests for the GUI-free analysis pipeline."""

import os
import struct

import numpy as np
import pytest

from src.analysis import (
    AnalysisError,
    COL_FRAME,
    COL_PHI,
    COL_Q,
    COL_THETA,
    compute_puckering,
    conformer_order,
    conformer_tex,
    load_fel,
    looks_like_frame_indices,
    make_progress_axis,
    parse_contour_step,
    parse_energy_max,
    parse_indices,
    prepare_output_dir,
    read_dcd_timestep,
    resolve_timestep,
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
# plot option parsing (shared by the GUI and the CLI)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    (None, None), ("", None), ("   ", None),
    ("auto", "auto"), ("AUTO", "auto"), (" auto ", "auto"),
    ("10", 10.0), ("12.5", 12.5), (7, 7.0), (0, 0.0), ("-3", -3.0),
])
def test_energy_max_accepts_numbers_auto_and_blank(value, expected):
    assert parse_energy_max(value) == expected


@pytest.mark.parametrize("value", ["high", "10 kcal", "auto2", "--"])
def test_energy_max_rejects_anything_else(value):
    with pytest.raises(AnalysisError, match="number, 'auto', or left blank"):
        parse_energy_max(value)


@pytest.mark.parametrize("value, expected", [
    (None, 1.0), ("", 1.0), ("5", 5.0), ("0.5", 0.5), (2, 2.0),
])
def test_contour_step_defaults_to_one(value, expected):
    assert parse_contour_step(value) == expected


def test_contour_step_honours_an_explicit_default():
    assert parse_contour_step("", default=5.0) == 5.0


@pytest.mark.parametrize("value", ["0", "-1", "-0.5"])
def test_contour_step_must_be_positive(value):
    """A step of zero would ask numpy.arange for infinitely many contours."""
    with pytest.raises(AnalysisError, match="greater than zero"):
        parse_contour_step(value)


@pytest.mark.parametrize("value", ["wide", "1,5", "five"])
def test_contour_step_must_be_a_number(value):
    with pytest.raises(AnalysisError, match="must be a number"):
        parse_contour_step(value)


# --------------------------------------------------------------------------
# the progress (time) axis
# --------------------------------------------------------------------------

def test_progress_falls_back_to_frame_numbers():
    values, label = make_progress_axis(4)
    assert list(values) == [1.0, 2.0, 3.0, 4.0]
    assert label == "Frame"


def test_timestep_turns_the_axis_into_picoseconds():
    values, label = make_progress_axis(4, timestep_ps=0.25)
    assert list(values) == [0.0, 0.25, 0.5, 0.75]
    assert label == "Time (ps)"


def test_long_runs_switch_to_nanoseconds():
    values, label = make_progress_axis(3, timestep_ps=1000.0)
    assert label == "Time (ns)"
    assert list(values) == [0.0, 1.0, 2.0]


def test_stored_times_are_used_when_present():
    values, label = make_progress_axis(3, times_ps=np.array([0.0, 0.5, 1.0]))
    assert list(values) == [0.0, 0.5, 1.0]
    assert label == "Time (ps)"


def test_timestep_overrides_stored_times():
    values, _ = make_progress_axis(3, times_ps=np.array([9.0, 9.5, 10.0]), timestep_ps=2.0)
    assert list(values) == [0.0, 2.0, 4.0]


def test_constant_times_are_not_a_time_axis():
    """A trajectory reporting the same time for every frame says nothing."""
    _, label = make_progress_axis(3, times_ps=np.array([5.0, 5.0, 5.0]))
    assert label == "Frame"


@pytest.mark.parametrize("times, expected", [
    ([0, 1, 2, 3], True),        # frame indices wearing time units
    ([0.0, 0.25, 0.5], False),   # a real timestep
    ([0, 1], True),
    ([7.0], True),               # too short to tell; assume absent
])
def test_frame_index_times_are_recognised(times, expected):
    assert looks_like_frame_indices(np.array(times, dtype=float)) is expected


# --------------------------------------------------------------------------
# DCD header timing
# --------------------------------------------------------------------------

def write_dcd_header(path, delta, nsavc, charmm_version=35, endian="<"):
    """Writes just enough of a DCD header for read_dcd_timestep() to parse."""
    control = [0] * 20
    control[2] = nsavc
    control[19] = charmm_version

    head = struct.pack(endian + "i", 84) + b"CORD"
    body = bytearray(struct.pack(endian + "20i", *control))
    if charmm_version:
        body[9 * 4:10 * 4] = struct.pack(endian + "f", delta)
    else:
        body[9 * 4:11 * 4] = struct.pack(endian + "d", delta)
    path.write_bytes(head + bytes(body) + b"\x00" * 8)
    return str(path)


def test_charmm_delta_is_converted_from_akma(tmp_path):
    """CHARMM stores the step in AKMA units; 1 fs is 0.020455 AKMA."""
    path = write_dcd_header(tmp_path / "t.dcd", delta=0.020454827696, nsavc=1)
    timestep, info = read_dcd_timestep(path)
    assert timestep == pytest.approx(0.001, rel=1e-4)
    assert info["flavour"] == "CHARMM"


def test_xplor_delta_is_already_picoseconds(tmp_path):
    path = write_dcd_header(tmp_path / "t.dcd", delta=0.5, nsavc=1, charmm_version=0)
    timestep, info = read_dcd_timestep(path)
    assert timestep == pytest.approx(0.5)
    assert info["flavour"] == "X-PLOR/NAMD"


def test_picoseconds_are_recognised_despite_the_charmm_flag(tmp_path):
    """
    Writers disagree about DELTA's units: files carrying the same CHARMM flag have
    been seen storing AKMA and storing picoseconds. Trusting the flag reads one of
    them 20x wrong, so the value is judged on whether it is a possible integration
    step: 0.001 as AKMA would be 0.049 fs, which no integrator uses.
    """
    path = write_dcd_header(tmp_path / "t.dcd", delta=0.001, nsavc=1)
    timestep, info = read_dcd_timestep(path)
    assert timestep == pytest.approx(0.001)      # 1 fs, not 4.9e-5 ps
    assert info["delta_units"] == "ps"
    assert info["step_is_plausible"]


def test_akma_is_still_recognised_when_it_is_the_plausible_one(tmp_path):
    """The same 1 fs step, written the other way round, must resolve the same."""
    path = write_dcd_header(tmp_path / "t.dcd", delta=0.020454827696, nsavc=1)
    timestep, info = read_dcd_timestep(path)
    assert timestep == pytest.approx(0.001, rel=1e-4)
    assert info["delta_units"] == "AKMA"


def test_an_implausible_step_is_flagged_but_still_returned(tmp_path):
    """Neither reading makes sense here; say so rather than silently pick one."""
    path = write_dcd_header(tmp_path / "t.dcd", delta=7.5, nsavc=1, charmm_version=0)
    timestep, info = read_dcd_timestep(path)
    assert timestep == pytest.approx(7.5)
    assert info["step_is_plausible"] is False


def test_save_frequency_multiplies_the_step(tmp_path):
    """A run saved every NSAVC steps has NSAVC times the spacing between frames."""
    path = write_dcd_header(tmp_path / "t.dcd", delta=0.020454827696, nsavc=250)
    timestep, info = read_dcd_timestep(path)
    assert timestep == pytest.approx(0.25, rel=1e-4)
    assert info["nsavc"] == 250


def test_big_endian_headers_are_read(tmp_path):
    path = write_dcd_header(tmp_path / "t.dcd", delta=0.25, nsavc=2,
                            charmm_version=0, endian=">")
    assert read_dcd_timestep(path)[0] == pytest.approx(0.5)


@pytest.mark.parametrize("delta", [0.0, -1.0])
def test_a_meaningless_step_is_refused(tmp_path, delta):
    path = write_dcd_header(tmp_path / "t.dcd", delta=delta, nsavc=1, charmm_version=0)
    assert read_dcd_timestep(path) is None


def test_a_non_dcd_file_is_refused(tmp_path):
    path = tmp_path / "not.dcd"
    path.write_bytes(b"this is not a DCD header at all, not even close" * 3)
    assert read_dcd_timestep(str(path)) is None


def test_a_missing_file_is_refused(tmp_path):
    assert read_dcd_timestep(str(tmp_path / "nope.dcd")) is None


def test_an_explicit_timestep_beats_the_header(tmp_path):
    """The header is frequently wrong, so the user must be able to override it."""
    path = write_dcd_header(tmp_path / "t.dcd", delta=0.020454827696, nsavc=1)
    timestep, source = resolve_timestep(path, timestep_ps=0.5)
    assert timestep == 0.5
    assert "user" in source


def test_the_header_is_used_when_nothing_is_given(tmp_path):
    path = write_dcd_header(tmp_path / "t.dcd", delta=0.020454827696, nsavc=100)
    timestep, source = resolve_timestep(path)
    assert timestep == pytest.approx(0.1, rel=1e-4)
    assert "DCD header" in source and "NSAVC=100" in source


@pytest.mark.parametrize("name", ["run.nc", "run.xtc", "run.trr", None])
def test_only_dcd_is_probed_for_a_header(name):
    """The other formats carry real per-frame times, so there is nothing to guess."""
    assert resolve_timestep(name) == (None, None)

@pytest.mark.parametrize("plain, tex", [
    ("4C1", r"$^4C_1$"), ("1C4", r"$^1C_4$"),
    ("1S3", r"$^1S_3$"), ("5S1", r"$^5S_1$"), ("B2,5", r"$B_{25}$"),
    ("OE", r"$^OE$"), ("2H3", r"$^2H_3$"), ("EO", r"$E_O$"),
    ("3E", r"$^3E$"),            # furanose and pyranose share this one
    ("3T2", r"$^3T_2$"), ("OT4", r"$^OT_4$"),   # furanose only
])
def test_conformer_labels_have_a_latex_form(plain, tex):
    assert conformer_tex(plain) == tex


@pytest.mark.parametrize("label", ["Undefined", "Other", ""])
def test_non_conformers_pass_through_unchanged(label):
    assert conformer_tex(label) == label


def test_every_assignable_conformer_has_a_latex_form():
    """Nothing the assignment can return may fall through as raw text."""
    for ring_size in (5, 6):
        for name in conformer_order(ring_size):
            assert conformer_tex(name).startswith("$"), name


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
