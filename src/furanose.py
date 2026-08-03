"""
furanose.py
Conformational analysis of 5-membered sugar rings (furanoses).

A 5-membered ring has a single puckering mode, so the Cremer-Pople sphere of the
pyranose case collapses to a circle: there is no Theta, only an amplitude and a
phase. The conventional description is Altona-Sundaralingam pseudorotation --
a phase angle P and a maximum torsion nu_max -- and the 20 canonical conformers
(10 envelopes E, 10 twists T) sit every 18 degrees around that circle.

Reference:
Altona, C., & Sundaralingam, M. (1972). Conformational analysis of the sugar ring
in nucleosides and nucleotides. A new description using the concept of
pseudorotation. Journal of the American Chemical Society, 94(23), 8205-8212.

Atom order is O4, C1, C2, C3, C4 -- the 5-membered analogue of the O5, C1..C5
order used for pyranoses, and just as order-sensitive.
"""

import numpy as np

RING_SIZE = 5

# Below this nu_max (degrees) the ring is flat and the phase P is undefined.
PLANAR_TOL_DEG = 1e-8

# Ring-atom names in the order the coordinates must be supplied.
ATOM_ORDER = ("O4", "C1", "C2", "C3", "C4")

# The 20 canonical conformers, every 18 degrees of P starting at P = 0.
# Anchored by construction: displacing a single ring atom out of the mean plane
# puts P at the envelope position naming that atom -- C3 out gives P = 18 (3E),
# C2 out gives P = 162 (2E), O4 out gives P = 90 (OE), and so on. "O" is the ring
# oxygen (the letter), never a zero.
IDEAL_P = np.arange(0, 360, 18)

FURANOSE_LABELS = [
    "3T2", "3E", "3T4", "E4", "OT4", "OE", "OT1", "E1", "2T1", "2E",
    "2T3", "E3", "4T3", "4E", "4TO", "EO", "1TO", "1E", "1T2", "E2",
]

FURANOSE_LABELS_TEX = [
    r"$^3T_2$", r"$^3E$", r"$^3T_4$", r"$E_4$", r"$^OT_4$",
    r"$^OE$", r"$^OT_1$", r"$E_1$", r"$^2T_1$", r"$^2E$",
    r"$^2T_3$", r"$E_3$", r"$^4T_3$", r"$^4E$", r"$^4T_O$",
    r"$E_O$", r"$^1T_O$", r"$^1E$", r"$^1T_2$", r"$E_2$",
]

assert len(IDEAL_P) == len(FURANOSE_LABELS) == len(FURANOSE_LABELS_TEX) == 20

UNDEFINED_LABEL = "Undefined"


def _torsion(p0, p1, p2, p3):
    """Signed IUPAC dihedral angle p0-p1-p2-p3, in degrees."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def _validate(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (RING_SIZE, 3):
        raise ValueError(
            f"Expected a {RING_SIZE}x3 array of ring coordinates, got shape {coords.shape}."
        )
    if not np.all(np.isfinite(coords)):
        raise ValueError("Ring coordinates contain NaN or infinite values.")
    return coords


def endocyclic_torsions(ring_coords):
    """
    The five endocyclic torsions nu0..nu4, in degrees.

    Args:
        ring_coords: 5x3 array ordered O4, C1, C2, C3, C4.
    """
    c = _validate(ring_coords)
    return np.array([
        _torsion(c[4], c[0], c[1], c[2]),  # nu0: C4-O4-C1-C2
        _torsion(c[0], c[1], c[2], c[3]),  # nu1: O4-C1-C2-C3
        _torsion(c[1], c[2], c[3], c[4]),  # nu2: C1-C2-C3-C4
        _torsion(c[2], c[3], c[4], c[0]),  # nu3: C2-C3-C4-O4
        _torsion(c[3], c[4], c[0], c[1]),  # nu4: C3-C4-O4-C1
    ])


def calculate_pseudorotation(ring_coords):
    """
    Altona-Sundaralingam pseudorotation parameters.

    Args:
        ring_coords: 5x3 array ordered O4, C1, C2, C3, C4.

    Returns:
        tuple: (P_degrees, nu_max_degrees). P is in [0, 360); nu_max is positive.

    Raises:
        ValueError: on bad input, or a ring that is planar to numerical precision.
    """
    nu = endocyclic_torsions(ring_coords)

    scale = 2.0 * (np.sin(np.radians(36.0)) + np.sin(np.radians(72.0)))
    # With nu_j = nu_max cos(P + 4*pi*(j-2)/5), these two combinations are
    # nu_max*sin(P)*scale and nu_max*cos(P)*scale respectively.
    y = (nu[4] + nu[1]) - (nu[3] + nu[0])
    x = 2.0 * nu[2] * (np.sin(np.radians(36.0)) + np.sin(np.radians(72.0)))

    P = np.degrees(np.arctan2(y, x)) % 360.0
    # nu_max from the hypotenuse, not from the textbook nu2 / cos(P): that form
    # is 0/0 exactly at P = 90 and 270 degrees -- the O4 envelopes, where nu2
    # vanishes by symmetry -- and returns garbage there. This form is stable
    # everywhere, always positive, and agrees with the textbook one to 1e-15
    # wherever cos(P) is not small.
    nu_max = float(np.hypot(y, x) / scale)

    if nu_max < PLANAR_TOL_DEG:
        raise ValueError(
            "Degenerate ring: the five atoms are planar (nu_max = 0), so the "
            "pseudorotation phase is undefined. Check the atom selection."
        )

    if P >= 360.0 - 1e-9:
        P = 0.0
    return float(P), float(nu_max)


def calculate_cremer_pople_furanose(ring_coords):
    """
    Cremer-Pople amplitude and phase for a 5-membered ring.

    With N = 5 there is only the m = 2 puckering mode, so Q is q2 and there is no
    polar angle. The phase relates to the Altona-Sundaralingam angle by
    P = phi + 90 degrees (verified by construction; the two formalisms differ by a
    few degrees on distorted rings because one is built from displacements and the
    other from torsions).

    Args:
        ring_coords: 5x3 array ordered O4, C1, C2, C3, C4.

    Returns:
        tuple: (Q_amplitude, phi_degrees). Q carries the input length unit.

    Raises:
        ValueError: on bad input or a planar/collinear ring.
    """
    coords = _validate(ring_coords)
    centred = coords - coords.mean(axis=0)

    j = np.arange(RING_SIZE)
    r_prime = (centred * np.sin(2.0 * np.pi * j / RING_SIZE)[:, None]).sum(axis=0)
    r_second = (centred * np.cos(2.0 * np.pi * j / RING_SIZE)[:, None]).sum(axis=0)

    # As for pyranoses, the cross product is normalised here and not its operands:
    # R' and R'' are not orthogonal in general.
    normal = np.cross(r_prime, r_second)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        raise ValueError(
            "Degenerate ring: the five atoms are collinear or coincident, so the "
            "mean plane is undefined. Check the atom selection."
        )
    normal /= normal_norm

    displ_z = centred @ normal
    sum_cos = np.sum(displ_z * np.cos(4.0 * np.pi * j / RING_SIZE))
    sum_sin = np.sum(-displ_z * np.sin(4.0 * np.pi * j / RING_SIZE))

    Q = float(np.sqrt(2.0 / RING_SIZE) * np.hypot(sum_cos, sum_sin))
    if Q < 1e-12:
        raise ValueError(
            "Degenerate ring: the five atoms are exactly planar (Q = 0), so the "
            "puckering phase is undefined. Check the atom selection."
        )

    phi = float(np.degrees(np.arctan2(sum_sin, sum_cos)) % 360.0)
    if phi >= 360.0 - 1e-9:
        phi = 0.0
    return Q, phi


def get_furanose_conformation(P):
    """
    Assigns one of the 20 canonical furanose conformers from the phase angle.

    Args:
        P (float): Altona-Sundaralingam phase angle in degrees.

    Returns:
        str: conformer label, or UNDEFINED_LABEL if P is not finite.
    """
    if not np.isfinite(P):
        return UNDEFINED_LABEL
    # 18-degree sectors centred on each ideal P, hence the 9-degree shift.
    sector = int(((P + 9.0) % 360.0) // 18.0)
    return FURANOSE_LABELS[sector]
