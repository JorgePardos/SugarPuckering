"""Shared test helpers."""

import matplotlib

matplotlib.use("Agg")  # never open a window during the test run

import numpy as np
import pytest

RING_RADIUS = 1.5


def make_ring(Q, theta_deg, phi_deg, radius=RING_RADIUS):
    """
    Builds a 6-membered ring with exactly the requested Cremer-Pople coordinates.

    This inverts the Cremer-Pople definition: atoms are placed on a circle of the
    given radius, traversed clockwise so the mean-plane normal comes out along
    +z, and displaced perpendicular by

        z_j = sqrt(2/N) q2 cos(phi + 4*pi*j/N) + sqrt(1/N) q3 (-1)^j

    with q2 = Q sin(theta) and q3 = Q cos(theta). Feeding the result back through
    calculate_cremer_pople() must return (Q, theta, phi), which makes this an
    exact reference generator rather than an approximate one.
    """
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    q2 = Q * np.sin(theta)
    q3 = Q * np.cos(theta)

    j = np.arange(6)
    z = np.sqrt(1.0 / 3.0) * q2 * np.cos(phi + 4.0 * np.pi * j / 6.0)
    z += np.sqrt(1.0 / 6.0) * q3 * (-1.0) ** j

    angle = -2.0 * np.pi * j / 6.0  # clockwise
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle), z])


def cremer_pople_reference(coords):
    """
    Independent, textbook-literal Cremer-Pople implementation used to cross-check
    the production one. Deliberately written from the 1975 definition rather than
    by copying src.math_core.
    """
    n_atoms = 6
    centred = coords - coords.mean(axis=0)
    j = np.arange(n_atoms)

    r_prime = (centred * np.sin(2 * np.pi * j / n_atoms)[:, None]).sum(axis=0)
    r_second = (centred * np.cos(2 * np.pi * j / n_atoms)[:, None]).sum(axis=0)
    normal = np.cross(r_prime, r_second)
    normal = normal / np.linalg.norm(normal)

    z = centred @ normal
    sum_cos = np.sum(z * np.cos(4 * np.pi * j / n_atoms))
    sum_sin = np.sum(-z * np.sin(4 * np.pi * j / n_atoms))

    q2 = np.sqrt(1 / 3) * np.hypot(sum_cos, sum_sin)
    q3 = np.sqrt(1 / 6) * np.sum(z * (-1.0) ** j)
    Q = np.hypot(q2, q3)

    theta = np.degrees(np.arccos(np.clip(q3 / Q, -1, 1)))
    phi = np.degrees(np.arctan2(sum_sin, sum_cos)) % 360
    return Q, theta, phi


def angles_close(a, b, tol=1e-6):
    """Compares two angles in degrees modulo 360."""
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff) < tol


@pytest.fixture
def ideal_chair():
    """A 4C1 chair: Q = 0.57 A, theta = 0."""
    return make_ring(0.57, 0.0, 0.0)
