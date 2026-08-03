"""
math_core.py
Core mathematical operations for calculating Cremer-Pople puckering coordinates.

Theoretical Foundation & Reference:
Cremer, D., & Pople, J. A. (1975). A general definition of ring puckering
coordinates. Journal of the American Chemical Society, 97(6), 1354-1358.

Stoddart mapping and conformational itinerary:
Ardèvol, A., Biarnés, X., Planas, A., & Rovira, C. (2010).
The Conformational Free-Energy Landscape of beta-D-Mannopyranose:
Evidence for a 1S5 -> B2,5 -> OS2 Catalytic Itinerary in beta-Mannosidases.
Journal of the American Chemical Society, 132(45), 16058-16065.
"""

import numpy as np

RING_SIZE = 6

# Below this total amplitude (in the input length unit, normally Angstrom) the
# ring is planar to numerical precision and neither Theta nor Phi is defined.
PLANAR_TOL = 1e-8

# Below this q2/Q ratio the ring is an ideal chair and the pseudorotation phase
# Phi is undefined. Theta is still well defined (~0 or ~180), and since the
# conformation is assigned from Theta alone in that regime, reporting Phi = 0
# is harmless. See get_strict_conformation().
CHAIR_TOL = 1e-9

# ==============================================================================
# REFERENCE DATA (Stoddart Diagram Ideal Coordinates)
# ==============================================================================
# Ideal Phi and Theta values for projection (Mercator/Stoddart)
IDEAL_PHI = np.array([180, 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360,
                      0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360,
                      0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 180])

IDEAL_THETA = np.array([0, 54.7, 50.8, 54.7, 50.8, 54.7, 50.8, 54.7, 50.8, 54.7, 50.8,
                        54.7, 50.8, 54.7, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
                        90, 90, 125.3, 129.2, 125.3, 129.2, 125.3, 129.2, 125.3, 129.2,
                        125.3, 129.2, 125.3, 129.2, 125.3, 180])

# LaTeX rendering of the 41 reference points above, for plot annotations.
# Index-locked to IDEAL_PHI / IDEAL_THETA -- keep the three in step.
LABELS_TEX = [
    r"$^4C_1$",
    r"$^OE$", r"$^OH_1$", r"$E_1$", r"$^2H_1$", r"$^2E$", r"$^2H_3$", r"$E_3$",
    r"$^4H_3$", r"$^4E$", r"$^4H_5$", r"$E_5$", r"$^OH_5$", r"$^OE$",
    r"$^{3O}B$", r"$^3S_1$", r"$B_{14}$", r"$^5S_1$", r"$^{25}B$", r"$^2S_O$",
    r"$B_{3O}$", r"$^1S_3$", r"$^{14}B$", r"$^1S_5$", r"$B_{25}$", r"$^OS_2$", r"$^{3O}B$",
    r"$^3E$", r"$^3H_4$", r"$E_4$", r"$^5H_4$", r"$^5E$", r"$^5H_O$", r"$E_O$",
    r"$^1H_O$", r"$^1E$", r"$^1H_2$", r"$E_2$", r"$^3H_2$", r"$^3E$",
    r"$^1C_4$",
]

assert len(IDEAL_PHI) == len(IDEAL_THETA) == len(LABELS_TEX), (
    "IDEAL_PHI, IDEAL_THETA and LABELS_TEX describe the same 41 reference points "
    "and must stay the same length."
)

# Ordered labels for the twelve 30-degree Phi sectors, starting at phi = 0.
# "O" is always the letter (the ring oxygen O5), never a zero -- the two were
# previously mixed within these tables, e.g. "5HO" next to "1H0".
EQUATORIAL_LABELS = ["3,OB", "3S1", "B1,4", "5S1", "2,5B", "2SO",
                     "B3,O", "1S3", "1,4B", "1S5", "B2,5", "OS2"]
NORTH_LABELS      = ["OE",   "OH1", "E1",   "2H1", "2E",   "2H3",
                     "E3",   "4H3", "4E",   "4H5", "E5",   "OH5"]
SOUTH_LABELS      = ["3E",   "3H4", "E4",   "5H4", "5E",   "5HO",
                     "EO",   "1HO", "1E",   "1H2", "E2",   "3H2"]

# Returned when Theta or Phi is not a finite number, i.e. when the upstream
# geometry was degenerate. Never returned for a well-formed ring.
UNDEFINED_LABEL = "Undefined"


def get_strict_conformation(theta, phi):
    """
    Assigns the sugar ring conformation using strict IUPAC angular boundaries.

    Args:
        theta (float): Cremer-Pople Theta angle in degrees (0 to 180).
        phi (float): Cremer-Pople Phi angle in degrees (0 to 360).

    Returns:
        str: Conformation label, or UNDEFINED_LABEL if the angles are not finite.
    """
    if not (np.isfinite(theta) and np.isfinite(phi)):
        return UNDEFINED_LABEL

    # 1. Chairs are strictly defined by Theta boundaries
    if theta <= 15.0:
        return "4C1"
    if theta >= 165.0:
        return "1C4"

    # 2. Slice the 360-degree Phi space into 12 sectors of 30 degrees.
    # Adding 15 degrees shifts the boundary so the ideal value is in the center.
    sector_phi = int(((phi + 15.0) % 360.0) // 30.0)

    # 3. Assign specific label based on latitude (Theta), which is now
    # strictly inside (15, 165) and therefore always falls in one of the three.
    if 75.0 <= theta <= 105.0:
        return EQUATORIAL_LABELS[sector_phi]
    if theta < 75.0:
        return NORTH_LABELS[sector_phi]
    return SOUTH_LABELS[sector_phi]


def calculate_cremer_pople(ring_coords):
    """
    Calculates the Cremer-Pople puckering parameters (Q, Theta, Phi).

    Args:
        ring_coords (numpy.ndarray): 6x3 array containing the 3D coordinates
                                     of the 6 ring atoms, in ring connectivity
                                     order (O5, C1, C2, C3, C4, C5).

    Returns:
        tuple: (Q_amplitude, Theta_degrees, Phi_degrees). Q carries the length
               unit of the input; Theta and Phi are in degrees.

    Raises:
        ValueError: if the input is not a 6x3 array of finite numbers, or if the
                    ring is planar/collinear to numerical precision (which in
                    practice means the wrong atoms were selected).
    """
    coords = np.asarray(ring_coords, dtype=float)
    if coords.shape != (RING_SIZE, 3):
        raise ValueError(
            f"Expected a {RING_SIZE}x3 array of ring coordinates, got shape {coords.shape}."
        )
    if not np.all(np.isfinite(coords)):
        raise ValueError("Ring coordinates contain NaN or infinite values.")

    # Translate geometric center to origin
    g_coord = coords - coords.mean(axis=0)

    # Cremer-Pople mean plane: spanned by R' and R''
    j = np.arange(RING_SIZE)
    r_prime = (g_coord * np.sin(2.0 * np.pi * j / RING_SIZE)[:, None]).sum(axis=0)
    r_second = (g_coord * np.cos(2.0 * np.pi * j / RING_SIZE)[:, None]).sum(axis=0)

    # Unit normal to the mean plane. R' and R'' are NOT orthogonal in general,
    # so the cross product must be normalised here -- normalising R' and R''
    # individually instead would scale every displacement (and hence Q) by
    # sin(angle between them).
    normal = np.cross(r_prime, r_second)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < PLANAR_TOL:
        raise ValueError(
            "Degenerate ring: the six atoms are collinear or coincident, so the "
            "mean plane is undefined. Check that the atom indices describe a ring "
            "and are given in connectivity order."
        )
    normal /= normal_norm

    # Perpendicular displacement of each atom from the mean plane
    displ_z = g_coord @ normal

    # Puckering amplitudes.
    # q2*cos(phi2) = sqrt(2/N) * sum z_j cos(4*pi*j/N)
    # q2*sin(phi2) = -sqrt(2/N) * sum z_j sin(4*pi*j/N),  with sqrt(2/6) = sqrt(1/3)
    sum_cos = np.sum(displ_z * np.cos(4.0 * np.pi * j / RING_SIZE))
    sum_sin = np.sum(-displ_z * np.sin(4.0 * np.pi * j / RING_SIZE))

    q2 = np.sqrt(1.0 / 3.0) * np.hypot(sum_cos, sum_sin)
    q3 = np.sqrt(1.0 / 6.0) * np.sum(displ_z * (-1.0) ** j)
    Q = float(np.hypot(q2, q3))

    if Q < PLANAR_TOL:
        raise ValueError(
            "Degenerate ring: the six atoms are exactly planar (Q = 0), so neither "
            "Theta nor Phi is defined. Check the atom selection."
        )

    # Theta: polar angle on the Cremer-Pople sphere. Clipped because rounding can
    # push the ratio marginally outside [-1, 1] for a near-perfect chair.
    theta = float(np.degrees(np.arccos(np.clip(q3 / Q, -1.0, 1.0))))

    # Phi: pseudorotation phase, undefined for an ideal chair (q2 = 0). arctan2
    # already returns 0.0 there, but we branch explicitly so the convention is
    # visible rather than incidental.
    if q2 / Q < CHAIR_TOL:
        phi = 0.0
    else:
        phi = float(np.degrees(np.arctan2(sum_sin, sum_cos)) % 360.0)
        # A phase of -1e-13 wraps to 359.9999999999, which reports as "360.0000".
        # Snap it back so the documented [0, 360) range actually holds.
        if phi >= 360.0 - 1e-9:
            phi = 0.0

    return Q, theta, phi
