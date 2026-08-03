"""Tests for the Cremer-Pople engine and the IUPAC conformation assignment."""

import numpy as np
import pytest

from src.math_core import (
    EQUATORIAL_LABELS,
    IDEAL_PHI,
    IDEAL_THETA,
    LABELS_TEX,
    NORTH_LABELS,
    SOUTH_LABELS,
    UNDEFINED_LABEL,
    calculate_cremer_pople,
    get_strict_conformation,
)

from conftest import angles_close, cremer_pople_reference, make_ring


# --------------------------------------------------------------------------
# calculate_cremer_pople
# --------------------------------------------------------------------------

ROUND_TRIP_CASES = [
    (0.57, 0.0, 0.0),      # 4C1 chair
    (0.57, 180.0, 0.0),    # 1C4 chair
    (0.60, 90.0, 270.0),   # 1S5 skew
    (0.60, 90.0, 300.0),   # B2,5 boat
    (0.60, 90.0, 330.0),   # OS2 skew
    (0.55, 50.8, 30.0),    # OH1 half-chair
    (0.55, 54.7, 0.0),     # OE envelope
    (0.62, 125.3, 0.0),    # 3E envelope
    (0.48, 129.2, 210.0),  # 1HO half-chair
    (0.50, 45.0, 123.4),   # arbitrary interior point
]


@pytest.mark.parametrize("Q, theta, phi", ROUND_TRIP_CASES)
def test_round_trip_recovers_input_coordinates(Q, theta, phi):
    """A ring built from (Q, theta, phi) must analyse back to (Q, theta, phi)."""
    got_Q, got_theta, got_phi = calculate_cremer_pople(make_ring(Q, theta, phi))
    assert got_Q == pytest.approx(Q, abs=1e-9)
    assert got_theta == pytest.approx(theta, abs=1e-6)
    assert angles_close(got_phi, phi)


@pytest.mark.parametrize("Q, theta, phi", ROUND_TRIP_CASES)
def test_matches_independent_reference_implementation(Q, theta, phi):
    ring = make_ring(Q, theta, phi)
    got = calculate_cremer_pople(ring)
    expected = cremer_pople_reference(ring)
    assert got[0] == pytest.approx(expected[0], abs=1e-12)
    assert got[1] == pytest.approx(expected[1], abs=1e-9)
    # At the poles q2 is zero and phi is undefined: the reference's naive
    # arctan2 there just reads floating-point noise, while the implementation
    # under test detects the singularity and pins phi to 0. Only compare phi
    # where it actually carries information.
    if 1e-6 < theta < 180.0 - 1e-6:
        assert angles_close(got[2], expected[2])


def test_amplitude_survives_a_non_orthogonal_mean_plane():
    """
    Regression guard for the Q bug: normalising R' and R'' before crossing them
    leaves a normal of length sin(angle between them), scaling Q. Distorted rings
    are where R' and R'' are furthest from orthogonal, so Q is checked against an
    independent implementation over many random distortions.
    """
    rng = np.random.default_rng(20240803)
    base = make_ring(0.57, 35.0, 210.0)
    worst = 0.0
    for _ in range(2000):
        ring = base + rng.normal(0.0, 0.5, size=(6, 3))
        got_Q = calculate_cremer_pople(ring)[0]
        expected_Q = cremer_pople_reference(ring)[0]
        worst = max(worst, abs(got_Q - expected_Q) / expected_Q)
    assert worst < 1e-10, f"Q drifts from the canonical definition by {worst:.1%}"


def test_invariant_under_rigid_motion():
    ring = make_ring(0.57, 62.0, 143.0)
    reference = calculate_cremer_pople(ring)

    # Rotation about an arbitrary axis, plus a translation
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    angle = 0.9
    cross = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    rotation = (np.eye(3) * np.cos(angle) + np.sin(angle) * cross
                + (1 - np.cos(angle)) * np.outer(axis, axis))
    moved = ring @ rotation.T + np.array([12.0, -4.0, 7.5])

    got = calculate_cremer_pople(moved)
    assert got[0] == pytest.approx(reference[0], abs=1e-9)
    assert got[1] == pytest.approx(reference[1], abs=1e-6)
    assert angles_close(got[2], reference[2], tol=1e-4)


def test_reversing_the_ring_flips_theta_but_keeps_amplitude():
    """Traversing the ring the other way swaps the two chairs."""
    ring = make_ring(0.57, 35.0, 100.0)
    forward = calculate_cremer_pople(ring)
    backward = calculate_cremer_pople(ring[[0, 5, 4, 3, 2, 1]])
    assert backward[0] == pytest.approx(forward[0], abs=1e-9)
    assert backward[1] == pytest.approx(180.0 - forward[1], abs=1e-6)


def test_scaling_the_geometry_scales_only_the_amplitude():
    ring = make_ring(0.57, 70.0, 45.0)
    reference = calculate_cremer_pople(ring)
    scaled = calculate_cremer_pople(ring * 3.0)
    assert scaled[0] == pytest.approx(3.0 * reference[0], abs=1e-9)
    assert scaled[1] == pytest.approx(reference[1], abs=1e-6)
    assert angles_close(scaled[2], reference[2])


def test_ideal_chair_has_defined_output(ideal_chair):
    """The chair is the q2 = 0 singularity that used to yield phi = NaN."""
    Q, theta, phi = calculate_cremer_pople(ideal_chair)
    assert np.isfinite([Q, theta, phi]).all()
    assert theta == pytest.approx(0.0, abs=1e-9)
    assert phi == 0.0
    assert get_strict_conformation(theta, phi) == "4C1"


@pytest.mark.parametrize("requested_phi", [0.0, 90.0, 217.0])
@pytest.mark.parametrize("pole_theta", [0.0, 180.0])
def test_phi_is_reported_as_zero_at_the_poles(pole_theta, requested_phi):
    """
    At a perfect chair the pseudorotation phase has no meaning, so whatever phi
    the geometry was built with, 0 is reported. Harmless because the label comes
    from theta alone in that band.
    """
    _, theta, phi = calculate_cremer_pople(make_ring(0.57, pole_theta, requested_phi))
    assert phi == 0.0
    assert get_strict_conformation(theta, phi) == ("4C1" if pole_theta == 0.0 else "1C4")


@pytest.mark.parametrize("phi", [0.0, 1e-7, 359.9999999, 90.0, 180.0, 359.0])
def test_phi_stays_inside_the_documented_range(phi):
    """
    Phi is documented as [0, 360). A phase of -1e-13 wraps to 359.9999999999,
    which prints as "360.0000" in the output table -- outside the range.
    """
    _, _, got_phi = calculate_cremer_pople(make_ring(0.57, 60.0, phi))
    assert 0.0 <= got_phi < 360.0


def test_planar_ring_raises_instead_of_returning_nan():
    angles = np.arange(6) * np.pi / 3
    flat = np.column_stack([1.5 * np.cos(angles), 1.5 * np.sin(angles), np.zeros(6)])
    with pytest.raises(ValueError, match="planar"):
        calculate_cremer_pople(flat)


def test_collinear_ring_raises():
    line = np.column_stack([np.arange(6.0), np.zeros(6), np.zeros(6)])
    with pytest.raises(ValueError, match="[Dd]egenerate"):
        calculate_cremer_pople(line)


@pytest.mark.parametrize("bad", [np.zeros((5, 3)), np.zeros((6, 2)), np.zeros((7, 3)), np.zeros(18)])
def test_rejects_wrong_shape(bad):
    with pytest.raises(ValueError, match="6x3"):
        calculate_cremer_pople(bad)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_rejects_non_finite_coordinates(bad_value):
    ring = make_ring(0.57, 30.0, 30.0)
    ring[2, 1] = bad_value
    with pytest.raises(ValueError, match="NaN or infinite"):
        calculate_cremer_pople(ring)


# --------------------------------------------------------------------------
# get_strict_conformation
# --------------------------------------------------------------------------

# The 41 reference points of the Stoddart map, transcribed independently from
# LABELS_TEX. Index-locked to IDEAL_PHI / IDEAL_THETA.
EXPECTED_IDEAL_LABELS = [
    "4C1",
    "OE", "OH1", "E1", "2H1", "2E", "2H3", "E3", "4H3", "4E", "4H5", "E5", "OH5", "OE",
    "3,OB", "3S1", "B1,4", "5S1", "2,5B", "2SO", "B3,O", "1S3", "1,4B", "1S5",
    "B2,5", "OS2", "3,OB",
    "3E", "3H4", "E4", "5H4", "5E", "5HO", "EO", "1HO", "1E", "1H2", "E2", "3H2", "3E",
    "1C4",
]


def test_reference_tables_stay_in_step():
    assert len(IDEAL_PHI) == len(IDEAL_THETA) == len(LABELS_TEX) == 41
    assert len(EXPECTED_IDEAL_LABELS) == 41
    for table in (EQUATORIAL_LABELS, NORTH_LABELS, SOUTH_LABELS):
        assert len(table) == 12


@pytest.mark.parametrize("index", range(41))
def test_every_ideal_point_maps_to_its_own_label(index):
    """Each of the 41 plotted reference points must classify as itself."""
    label = get_strict_conformation(IDEAL_THETA[index], IDEAL_PHI[index])
    assert label == EXPECTED_IDEAL_LABELS[index]


@pytest.mark.parametrize("theta", [0.0, 7.5, 15.0])
def test_north_pole_region_is_the_4c1_chair(theta):
    for phi in range(0, 360, 15):
        assert get_strict_conformation(theta, phi) == "4C1"


@pytest.mark.parametrize("theta", [165.0, 172.0, 180.0])
def test_south_pole_region_is_the_1c4_chair(theta):
    for phi in range(0, 360, 15):
        assert get_strict_conformation(theta, phi) == "1C4"


@pytest.mark.parametrize("theta, expected_table", [
    (15.001, NORTH_LABELS), (45.0, NORTH_LABELS), (74.999, NORTH_LABELS),
    (75.0, EQUATORIAL_LABELS), (90.0, EQUATORIAL_LABELS), (105.0, EQUATORIAL_LABELS),
    (105.001, SOUTH_LABELS), (140.0, SOUTH_LABELS), (164.999, SOUTH_LABELS),
])
def test_latitude_bands_select_the_right_table(theta, expected_table):
    for sector, expected in enumerate(expected_table):
        centre_phi = sector * 30.0
        assert get_strict_conformation(theta, centre_phi) == expected


def test_phi_is_periodic():
    for phi in (0.0, 360.0, 720.0, -360.0):
        assert get_strict_conformation(90.0, phi) == "3,OB"
    assert get_strict_conformation(90.0, -30.0) == get_strict_conformation(90.0, 330.0)


@pytest.mark.parametrize("offset", [-14.9, 0.0, 14.9])
def test_sector_width_is_thirty_degrees_centred_on_the_ideal_value(offset):
    """Anything within +-15 degrees of an ideal phi keeps that ideal's label."""
    for sector, expected in enumerate(EQUATORIAL_LABELS):
        assert get_strict_conformation(90.0, sector * 30.0 + offset) == expected


@pytest.mark.parametrize("theta, phi", [
    (np.nan, 0.0), (0.0, np.nan), (np.nan, np.nan), (np.inf, 0.0), (90.0, np.inf),
])
def test_non_finite_angles_are_reported_as_undefined(theta, phi):
    assert get_strict_conformation(theta, phi) == UNDEFINED_LABEL


@pytest.mark.parametrize("Q, theta, phi", ROUND_TRIP_CASES)
def test_geometry_to_label_end_to_end(Q, theta, phi):
    """Building a ring at an ideal point and classifying it returns that point."""
    _, got_theta, got_phi = calculate_cremer_pople(make_ring(Q, theta, phi))
    assert get_strict_conformation(got_theta, got_phi) == get_strict_conformation(theta, phi)
