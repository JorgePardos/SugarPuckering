"""Tests for the 5-membered ring (furanose) description."""

import numpy as np
import pytest

from src.furanose import (
    ATOM_ORDER,
    FURANOSE_LABELS,
    FURANOSE_LABELS_TEX,
    IDEAL_P,
    UNDEFINED_LABEL,
    calculate_cremer_pople_furanose,
    calculate_pseudorotation,
    endocyclic_torsions,
    get_furanose_conformation,
)

RING_SIZE = 5


def make_furanose(Q, phi_deg, radius=1.2):
    """
    Builds a 5-membered ring with a given Cremer-Pople amplitude and phase.

    A 5-ring has only the m = 2 puckering mode, so
    z_j = sqrt(2/5) Q cos(phi + 4*pi*j/5) satisfies the three mean-plane
    constraints exactly. Atoms are placed clockwise so the normal comes out
    along +z, matching the pyranose helper.
    """
    j = np.arange(RING_SIZE)
    z = np.sqrt(2.0 / RING_SIZE) * Q * np.cos(np.radians(phi_deg) + 4 * np.pi * j / RING_SIZE)
    angle = -2.0 * np.pi * j / RING_SIZE
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle), z])


def envelope(atom_index, Q=0.4):
    """A ring puckered so that exactly `atom_index` sits out of the mean plane."""
    return make_furanose(Q, (-720.0 * atom_index / RING_SIZE) % 360.0)


# --------------------------------------------------------------------------
# reference tables
# --------------------------------------------------------------------------

def test_the_wheel_has_twenty_conformers_every_eighteen_degrees():
    assert len(IDEAL_P) == len(FURANOSE_LABELS) == len(FURANOSE_LABELS_TEX) == 20
    assert list(IDEAL_P) == list(range(0, 360, 18))
    assert len(set(FURANOSE_LABELS)) == 20


def test_envelopes_and_twists_alternate():
    """Ten envelopes (E) and ten twists (T), interleaved around the circle."""
    assert sum("T" in label for label in FURANOSE_LABELS) == 10
    assert sum("T" not in label for label in FURANOSE_LABELS) == 10
    for k, label in enumerate(FURANOSE_LABELS):
        assert ("T" in label) == (k % 2 == 0)


def test_labels_use_the_letter_o_not_a_zero():
    assert "0" not in "".join(FURANOSE_LABELS)


# --------------------------------------------------------------------------
# pseudorotation
# --------------------------------------------------------------------------

# Displacing a single ring atom must land on the envelope named after it. This
# is what anchors the whole 20-conformer wheel, so all five are checked.
ENVELOPE_ANCHORS = [
    (3, 18.0, "3E"),    # C3 out of plane -> C3-endo
    (4, 234.0, "4E"),   # C4 -> C4-endo
    (0, 90.0, "OE"),    # O4 -> O4-endo
    (1, 306.0, "1E"),   # C1 -> C1-endo
    (2, 162.0, "2E"),   # C2 -> C2-endo
]


@pytest.mark.parametrize("atom_index, expected_P, expected_label", ENVELOPE_ANCHORS)
def test_single_atom_pucker_lands_on_its_own_envelope(atom_index, expected_P, expected_label):
    P, nu_max = calculate_pseudorotation(envelope(atom_index))
    assert P == pytest.approx(expected_P, abs=2.0)
    assert nu_max > 1.0
    assert get_furanose_conformation(P) == expected_label
    assert ATOM_ORDER[atom_index] in ("O4", "C1", "C2", "C3", "C4")


@pytest.mark.parametrize("phi_cp", [0.0, 180.0])
def test_nu_max_is_stable_where_nu2_vanishes(phi_cp):
    """
    The O4 envelopes put P at 90 and 270 degrees, where nu2 is zero by symmetry.
    The textbook nu_max = nu2 / cos(P) is 0/0 there and returns nonsense; nu_max
    must stay consistent with its neighbours instead.
    """
    P, nu_max = calculate_pseudorotation(make_furanose(0.4, phi_cp))
    assert abs(np.cos(np.radians(P))) < 1e-6, "this case should be the ill-conditioned one"

    nearby = [calculate_pseudorotation(make_furanose(0.4, phi_cp + d))[1]
              for d in (-6.0, 6.0)]
    assert nu_max == pytest.approx(np.mean(nearby), rel=0.02)


def test_nu_max_is_always_positive():
    for phi in np.arange(0.0, 360.0, 7.0):
        assert calculate_pseudorotation(make_furanose(0.4, phi))[1] > 0.0


def test_the_two_famous_puckers_sit_north_and_south():
    """C3-endo (3E, North) and C2-endo (2E, South) are the nucleic-acid anchors."""
    north, _ = calculate_pseudorotation(envelope(3))
    south, _ = calculate_pseudorotation(envelope(2))
    assert get_furanose_conformation(north) == "3E"
    assert get_furanose_conformation(south) == "2E"
    assert abs(((south - north) % 360.0) - 144.0) < 5.0


@pytest.mark.parametrize("phi", [0.0, 45.0, 137.0, 250.0, 330.0])
def test_pseudorotation_tracks_the_cremer_pople_phase(phi):
    """
    The two formalisms differ by a constant 90 degrees. They are built from
    different quantities -- torsions versus out-of-plane displacements -- so a
    few degrees of disagreement on a real ring is expected, not a bug.
    """
    ring = make_furanose(0.38, phi)
    P, _ = calculate_pseudorotation(ring)
    _, phi_cp = calculate_cremer_pople_furanose(ring)
    offset = (P - phi_cp) % 360.0
    assert min(abs(offset - 90.0), abs(offset - 450.0)) < 5.0


def test_amplitude_scales_with_the_geometry():
    small = calculate_cremer_pople_furanose(make_furanose(0.2, 60.0))[0]
    large = calculate_cremer_pople_furanose(make_furanose(0.6, 60.0))[0]
    assert small == pytest.approx(0.2, abs=1e-9)
    assert large == pytest.approx(0.6, abs=1e-9)


def test_cremer_pople_phase_round_trips():
    for phi in (0.0, 72.0, 144.0, 216.0, 288.0):
        _, got = calculate_cremer_pople_furanose(make_furanose(0.4, phi))
        assert got == pytest.approx(phi, abs=1e-6)


def test_invariant_under_rigid_motion():
    ring = make_furanose(0.4, 123.0)
    reference = calculate_pseudorotation(ring)

    axis = np.array([0.3, 1.0, -0.7])
    axis /= np.linalg.norm(axis)
    angle = 1.1
    cross = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    rotation = (np.eye(3) * np.cos(angle) + np.sin(angle) * cross
                + (1 - np.cos(angle)) * np.outer(axis, axis))
    moved = ring @ rotation.T + np.array([-5.0, 2.0, 9.0])

    got = calculate_pseudorotation(moved)
    assert got[0] == pytest.approx(reference[0], abs=1e-4)
    assert got[1] == pytest.approx(reference[1], abs=1e-4)


def test_torsions_sum_to_about_zero():
    """The five endocyclic torsions of a closed ring very nearly cancel."""
    nu = endocyclic_torsions(make_furanose(0.35, 200.0))
    assert len(nu) == 5
    assert abs(nu.sum()) < 5.0


def test_planar_ring_raises_instead_of_returning_nan():
    j = np.arange(5)
    angle = -2.0 * np.pi * j / 5.0
    flat = np.column_stack([1.2 * np.cos(angle), 1.2 * np.sin(angle), np.zeros(5)])
    with pytest.raises(ValueError, match="planar"):
        calculate_pseudorotation(flat)
    with pytest.raises(ValueError, match="[Pp]lanar|[Dd]egenerate"):
        calculate_cremer_pople_furanose(flat)


@pytest.mark.parametrize("bad", [np.zeros((4, 3)), np.zeros((6, 3)), np.zeros((5, 2))])
def test_rejects_wrong_shape(bad):
    with pytest.raises(ValueError, match="5x3"):
        calculate_pseudorotation(bad)


def test_rejects_non_finite_coordinates():
    ring = make_furanose(0.4, 30.0)
    ring[1, 2] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        calculate_pseudorotation(ring)


# --------------------------------------------------------------------------
# conformer assignment
# --------------------------------------------------------------------------

@pytest.mark.parametrize("index", range(20))
def test_every_ideal_phase_maps_to_its_own_label(index):
    assert get_furanose_conformation(IDEAL_P[index]) == FURANOSE_LABELS[index]


@pytest.mark.parametrize("offset", [-8.9, 0.0, 8.9])
def test_sectors_are_eighteen_degrees_wide(offset):
    for index, label in enumerate(FURANOSE_LABELS):
        assert get_furanose_conformation(IDEAL_P[index] + offset) == label


def test_phase_is_periodic():
    for P in (0.0, 360.0, 720.0, -360.0):
        assert get_furanose_conformation(P) == FURANOSE_LABELS[0]
    assert get_furanose_conformation(-18.0) == get_furanose_conformation(342.0)


@pytest.mark.parametrize("P", [np.nan, np.inf, -np.inf])
def test_non_finite_phase_is_undefined(P):
    assert get_furanose_conformation(P) == UNDEFINED_LABEL
