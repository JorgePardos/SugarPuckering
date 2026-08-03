"""
plotting.py
Visualization tools for generating Stoddart/Mercator maps, Time Series,
and Free Energy Landscapes (FEL).

No matplotlib backend is selected here: importing this module must stay safe in a
headless process (tests, CLI). The Tkinter GUI selects "TkAgg" before importing.
"""

from collections import Counter

import numpy as np
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Import reference map and labels from the local module
from .math_core import IDEAL_PHI, IDEAL_THETA, LABELS_TEX
from .furanose import FURANOSE_LABELS_TEX, IDEAL_P
from .analysis import COL_FRAME, COL_NU_MAX, COL_P, COL_Q, COL_THETA, COL_PHI

# Colour of the reference Stoddart grid drawn behind user data.
REFERENCE_COLOR = "#333333"

# Single hue for one-series marks (bars, histograms). Taken from the middle of
# viridis so the summary plots read as the same family as the FEL maps.
PRIMARY_COLOR = "#31688E"

# Fill for grid cells the sampling never reached. Deliberately an off-white grey
# rather than a colour from the ramp, so "no data" cannot be mistaken for "high
# energy" -- the convention used in the published landscapes.
NO_DATA_COLOR = "#EDEDED"


def _finish(fig, out_path, show):
    """Saves the figure, optionally shows it, and always releases it."""
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _draw_reference_map(ax, color=REFERENCE_COLOR, alpha=0.75, fontsize=11):
    """Draws the 41 ideal conformers and formats the Stoddart axes."""
    ax.scatter(IDEAL_PHI, IDEAL_THETA, marker=".", color=color, alpha=alpha)
    for i, label in enumerate(LABELS_TEX):
        ax.annotate(label, (IDEAL_PHI[i], IDEAL_THETA[i]),
                    fontsize=fontsize, alpha=alpha, color=color)

    ax.axis((0, 360, 180, 0))  # Y-axis inverted for standard Stoddart representation
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_yticks(np.arange(0, 181, 45))
    ax.set_xlabel("Phi (φ) [°]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Theta (θ) [°]", fontsize=12, fontweight="bold")


def _split_at_phi_wrap(phi, theta, threshold=180.0):
    """
    Splits a trajectory into segments that never cross the phi = 0/360 seam.

    Phi is periodic, so a step from 355 to 5 degrees is a small move, but drawn
    literally it streaks a line right across the map. Breaking the path there
    keeps the itinerary readable.
    """
    if len(phi) < 2:
        return []
    breaks = np.flatnonzero(np.abs(np.diff(phi)) > threshold) + 1
    return [(phi[s], theta[s]) for s in np.split(np.arange(len(phi)), breaks)
            if len(s) > 1]


def plot_mercator(results_data, path_prefix, title=None, show=False, connect=True):
    """
    Plots a Mercator projection of the Cremer-Pople sphere showing the
    trajectory points over the ideal Stoddart conformational map.

    Args:
        results_data (numpy.ndarray): (n_frames, 4) array from compute_puckering().
        path_prefix (str): output path prefix; "_mercator.png" is appended.
        title (str): plot title. Defaults to a generic one -- pass the job name,
                     not the path prefix.
        show (bool): open an interactive window. Leave False inside a GUI, where
                     plt.show() would block the Tk event loop.
        connect (bool): join consecutive frames, so the itinerary reads as a path
                        instead of a cloud whose order is only implied by colour.

    Returns:
        str: the path written.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    _draw_reference_map(ax)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_title(title or "Conformational Map", fontsize=14)

    if len(results_data) == 1:
        ax.scatter(results_data[:, COL_PHI], results_data[:, COL_THETA],
                   s=200, color="red", marker="*", edgecolor="black", zorder=5)
    else:
        if connect:
            for seg_phi, seg_theta in _split_at_phi_wrap(results_data[:, COL_PHI],
                                                         results_data[:, COL_THETA]):
                ax.plot(seg_phi, seg_theta, color="#555555", linewidth=0.8,
                        alpha=0.55, zorder=4)
        frames = results_data[:, COL_FRAME] + 1  # 1-based frame indexing
        scatter = ax.scatter(results_data[:, COL_PHI], results_data[:, COL_THETA],
                             s=30, c=frames, cmap="plasma", zorder=5,
                             edgecolor="black", linewidths=0.5)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.85)
        cbar.set_label("Frame", rotation=270, labelpad=20)

    fig.tight_layout()
    return _finish(fig, f"{path_prefix}_mercator.png", show)


def _draw_stoddart_hemisphere(ax, north, points, frames, indices=None, connect=True):
    """Draws one hemisphere of the polar Stoddart diagram onto a polar axis."""
    # Radius is the angular distance from the pole, so the pole sits at the centre
    # and the equator on the rim; both hemispheres then share the same rim.
    if north:
        keep = IDEAL_THETA <= 90.0
        radius = IDEAL_THETA[keep]
        pole_label, hemisphere = r"$^4C_1$", "Northern"
    else:
        keep = IDEAL_THETA >= 90.0
        radius = 180.0 - IDEAL_THETA[keep]
        pole_label, hemisphere = r"$^1C_4$", "Southern"

    angles = np.radians(IDEAL_PHI[keep])
    labels = [LABELS_TEX[i] for i in np.flatnonzero(keep)]

    ax.set_theta_zero_location("E")
    ax.set_ylim(0, 92)
    ax.set_yticks([30, 60, 90])
    # The conformer names *are* the reference here, as on the printed diagram, so
    # numeric radial labels would only collide with them. Rings stay as guides.
    ax.set_yticklabels([])
    ax.set_xticks(np.radians(np.arange(0, 360, 30)))
    ax.set_xticklabels([f"{d}°" for d in range(0, 360, 30)],
                       fontsize=9, color=REFERENCE_COLOR)
    ax.tick_params(axis="x", pad=14)
    ax.grid(True, linestyle=":", alpha=0.45)

    ax.scatter(angles, radius, marker=".", color=REFERENCE_COLOR, alpha=0.65, s=18)
    for angle, r, label in zip(angles, radius, labels):
        ax.annotate(label, (angle, r), fontsize=9, alpha=0.85,
                    color=REFERENCE_COLOR, ha="center",
                    textcoords="offset points", xytext=(0, 6))
    # The pole already appears in the reference table above (it is the theta = 0
    # / theta = 180 entry), so it needs no separate annotation -- adding one drew
    # the chair label twice on top of itself.

    if len(points):
        point_theta, point_phi = points[:, 0], points[:, 1]
        point_radius = point_theta if north else 180.0 - point_theta
        if frames is None:
            ax.scatter(np.radians(point_phi), point_radius, s=180, color="red",
                       marker="*", edgecolor="black", zorder=5)
        else:
            if connect and indices is not None and len(indices) > 1:
                # Only join frames that really were consecutive: a gap means the
                # trajectory crossed the equator into the other dial.
                runs = np.split(np.arange(len(indices)),
                                np.flatnonzero(np.diff(indices) != 1) + 1)
                for run in runs:
                    if len(run) > 1:
                        ax.plot(np.radians(point_phi[run]), point_radius[run],
                                color="#555555", linewidth=0.8, alpha=0.55, zorder=4)
            ax.scatter(np.radians(point_phi), point_radius, s=26, c=frames,
                       cmap="plasma", zorder=5, edgecolor="black", linewidths=0.4,
                       vmin=frames.min() if len(frames) else 0,
                       vmax=frames.max() if len(frames) else 1)

    ax.set_title(f"{hemisphere} hemisphere", fontsize=12, pad=18)


def plot_stoddart(results_data, path_prefix, title=None, show=False, connect=True):
    """
    Plots the polar Stoddart diagram: both hemispheres of the Cremer-Pople sphere
    seen from their poles.

    This is the complement of the Mercator map, not a replacement. Mercator keeps
    every conformer on one rectangle but stretches the poles across the whole
    width; the polar view puts each chair at a single point and shows the
    pseudorotational ring around it undistorted, which is where the boat and skew
    conformers actually live.

    Args and Returns: see plot_mercator(); "_stoddart.png" is appended.
    """
    theta = results_data[:, COL_THETA]
    phi = results_data[:, COL_PHI]
    frame_numbers = results_data[:, COL_FRAME] + 1
    single = len(results_data) == 1

    north_mask = theta <= 90.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             subplot_kw={"projection": "polar"})
    # Without extra room the 0 degree label of one dial lands on the 180 degree
    # label of the other.
    fig.subplots_adjust(wspace=0.35)

    for ax, is_north in zip(axes, (True, False)):
        mask = north_mask if is_north else ~north_mask
        points = np.column_stack([theta[mask], phi[mask]])
        _draw_stoddart_hemisphere(
            ax, is_north, points,
            None if single else frame_numbers[mask],
            indices=np.flatnonzero(mask), connect=connect,
        )

    if not single:
        mappable = plt.cm.ScalarMappable(
            cmap="plasma",
            norm=plt.Normalize(vmin=frame_numbers.min(), vmax=frame_numbers.max()))
        cbar = fig.colorbar(mappable, ax=axes, shrink=0.7, pad=0.08)
        cbar.set_label("Frame", rotation=270, labelpad=20)

    fig.suptitle(title or "Stoddart diagram", fontsize=15)
    return _finish(fig, f"{path_prefix}_stoddart.png", show)


def plot_time_series(results_data, path_prefix, title=None, show=False):
    """
    Plots the temporal evolution of Theta and Phi parameters along a MD trajectory.

    Args and Returns: see plot_mercator(); "_timeseries.png" is appended.
    """
    frames = results_data[:, COL_FRAME] + 1  # 1-based indexing
    theta = results_data[:, COL_THETA]
    phi = results_data[:, COL_PHI]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Theta subplot (Chair vs Boat progression)
    ax1.plot(frames, theta, color="royalblue", linewidth=1.5, alpha=0.8)
    ax1.scatter(frames, theta, color="royalblue", s=10)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.4, label="Chair 4C1 (~0°)")
    ax1.axhline(y=90, color="green", linestyle="--", alpha=0.4, label="Boats/Skews (~90°)")
    ax1.axhline(y=180, color="purple", linestyle="--", alpha=0.4, label="Chair 1C4 (~180°)")

    ax1.set_ylabel("Theta (θ) [°]", fontsize=12, fontweight="bold")
    ax1.set_title(title or "Conformational Evolution", fontsize=14)
    ax1.set_ylim(-10, 190)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Phi subplot (Equatorial pseudorotation)
    ax2.plot(frames, phi, color="darkorange", linewidth=1.5, alpha=0.8)
    ax2.scatter(frames, phi, color="darkorange", s=10)
    ax2.set_ylabel("Phi (φ) [°]", fontsize=12, fontweight="bold")
    ax2.set_ylim(-10, 370)
    ax2.set_yticks(np.arange(0, 361, 60))
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Amplitude subplot. Theta and Phi say which conformation; only Q says how
    # puckered it is, and a ring flattening towards an oxocarbenium-like
    # transition state shows up here and nowhere else on these plots.
    _draw_amplitude_track(ax3, frames, results_data[:, COL_Q])
    ax3.set_xlabel("Frame", fontsize=12, fontweight="bold")

    fig.tight_layout()
    return _finish(fig, f"{path_prefix}_timeseries.png", show)


def _draw_amplitude_track(ax, progress, Q):
    """Shared Q-versus-progress panel for the pyranose and furanose time series."""
    ax.plot(progress, Q, color="#2E7D5B", linewidth=1.5, alpha=0.85)
    ax.scatter(progress, Q, color="#2E7D5B", s=10)
    mean_Q = float(np.mean(Q))
    ax.axhline(mean_Q, color="#B5411F", linestyle="--", alpha=0.6,
               label=f"mean = {mean_Q:.3f} Å")
    ax.set_ylabel("Amplitude Q [Å]", fontsize=12, fontweight="bold")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)


def plot_pseudorotation_wheel(results_data, path_prefix, title=None, show=False):
    """
    Plots furanose conformers on the pseudorotation circle.

    A 5-membered ring has a single puckering mode, so the Stoddart sphere of the
    pyranose case collapses to a circle: the phase angle P is the polar angle and
    nu_max the radius, with the 20 canonical conformers marked around the rim.

    Args:
        results_data (numpy.ndarray): (n_frames, 4) from compute_puckering() on a
                                      5-membered ring, i.e. [frame, Q, P, nu_max].
        path_prefix (str): output path prefix; "_wheel.png" is appended.
        title (str): plot title; pass the job name, not the path prefix.
        show (bool): open an interactive window.

    Returns:
        str: the path written.
    """
    phase = np.radians(results_data[:, COL_P])
    nu_max = results_data[:, COL_NU_MAX]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # P increases clockwise, as drawn in the literature

    outer = max(float(np.max(nu_max)) * 1.25, 1e-6)
    ax.set_ylim(0, outer)

    # Reference conformers around the rim
    ax.set_xticks(np.radians(IDEAL_P))
    ax.set_xticklabels(FURANOSE_LABELS_TEX, fontsize=12, color=REFERENCE_COLOR)
    ax.tick_params(axis="x", pad=8)
    for angle in np.radians(IDEAL_P):
        ax.plot([angle, angle], [0, outer], color=REFERENCE_COLOR,
                alpha=0.25, linewidth=0.7, zorder=1)

    if len(results_data) == 1:
        ax.scatter(phase, nu_max, s=250, color="red", marker="*",
                   edgecolor="black", zorder=5)
    else:
        frames = results_data[:, COL_FRAME] + 1
        scatter = ax.scatter(phase, nu_max, s=30, c=frames, cmap="plasma",
                             zorder=5, edgecolor="black", linewidths=0.5)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.12)
        cbar.set_label("Frame", rotation=270, labelpad=20)

    ax.set_title(title or "Pseudorotation Wheel", fontsize=14, pad=28)
    ax.set_rlabel_position(0)
    ax.text(np.radians(4), outer * 1.02, r"$\nu_{max}$ [°]",
            fontsize=11, color=REFERENCE_COLOR)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    return _finish(fig, f"{path_prefix}_wheel.png", show)


def plot_furanose_time_series(results_data, path_prefix, title=None, show=False):
    """
    Plots the evolution of P and nu_max along a furanose trajectory.

    Args and Returns: see plot_mercator(); "_timeseries.png" is appended.
    """
    frames = results_data[:, COL_FRAME] + 1
    phase = results_data[:, COL_P]
    nu_max = results_data[:, COL_NU_MAX]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax1.plot(frames, phase, color="royalblue", linewidth=1.5, alpha=0.8)
    ax1.scatter(frames, phase, color="royalblue", s=10)
    ax1.axhline(y=18, color="red", linestyle="--", alpha=0.4, label="North $^3E$ (P~18°)")
    ax1.axhline(y=162, color="green", linestyle="--", alpha=0.4, label="South $^2E$ (P~162°)")
    ax1.set_ylabel("Phase P [°]", fontsize=12, fontweight="bold")
    ax1.set_title(title or "Pseudorotation Evolution", fontsize=14)
    ax1.set_ylim(-10, 370)
    ax1.set_yticks(np.arange(0, 361, 60))
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(frames, nu_max, color="darkorange", linewidth=1.5, alpha=0.8)
    ax2.scatter(frames, nu_max, color="darkorange", s=10)
    ax2.set_ylabel(r"$\nu_{max}$ [°]", fontsize=12, fontweight="bold")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle=":", alpha=0.6)

    _draw_amplitude_track(ax3, frames, results_data[:, COL_Q])
    ax3.set_xlabel("Frame", fontsize=12, fontweight="bold")

    fig.tight_layout()
    return _finish(fig, f"{path_prefix}_timeseries.png", show)


def plot_conformer_timeline(results_data, path_prefix, ring_size,
                            title=None, show=False, progress_label="Frame"):
    """
    Which conformer the ring occupies at each step, as a timeline.

    Conformers are stacked in their canonical order (pole, northern band,
    equator, southern band, pole for a pyranose; the pseudorotation wheel for a
    furanose), so the vertical axis behaves like a latitude and a reaction
    itinerary reads as a descent or climb rather than an arbitrary jumble. The
    populations chart says how much time went where; this says when, and in what
    sequence.

    Args:
        results_data (numpy.ndarray): (n_frames, 4) from compute_puckering().
        path_prefix (str): output path prefix; "_timeline.png" is appended.
        ring_size (int): 6 for a pyranose, 5 for a furanose.
        title (str): plot title; pass the job name, not the path prefix.
        show (bool): open an interactive window.
        progress_label (str): x-axis label, i.e. what "progress" means here.

    Returns:
        str: the path written.
    """
    from .analysis import conformer_order, describe_conformation

    labels = [describe_conformation(row, ring_size) for row in results_data]
    canonical = conformer_order(ring_size)

    # Keep only the conformers actually visited, in canonical order, so the axis
    # does not carry 30 empty rows.
    seen = set(labels)
    axis = [name for name in canonical if name in seen]
    axis += sorted(seen - set(axis))  # anything unexpected, e.g. "Undefined"
    row_of = {name: i for i, name in enumerate(axis)}

    progress = results_data[:, COL_FRAME] + 1
    track = np.array([row_of[name] for name in labels], dtype=float)

    fig, ax = plt.subplots(figsize=(11, max(3.0, 0.32 * len(axis) + 2.0)))
    ax.step(progress, track, where="post", color=PRIMARY_COLOR, linewidth=1.4)
    ax.scatter(progress, track, s=12, color=PRIMARY_COLOR, zorder=3)

    ax.set_yticks(np.arange(len(axis)))
    ax.set_yticklabels(axis, fontsize=10)
    ax.set_ylim(-0.6, len(axis) - 0.4)
    ax.set_xlabel(progress_label, fontsize=12, fontweight="bold")
    ax.set_title(title or "Conformational itinerary", fontsize=14)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    return _finish(fig, f"{path_prefix}_timeline.png", show)


def plot_conformer_populations(results_data, path_prefix, ring_size,
                               title=None, show=False, max_bars=20):
    """
    Bar chart of how much of the trajectory sits in each conformer.

    This is the summary a puckering run is usually reduced to in print, and it
    answers the question the scatter plots only imply: which conformations were
    actually populated, and by how much.

    Args:
        results_data (numpy.ndarray): (n_frames, 4) from compute_puckering().
        path_prefix (str): output path prefix; "_populations.png" is appended.
        ring_size (int): 6 for a pyranose, 5 for a furanose.
        title (str): plot title; pass the job name, not the path prefix.
        show (bool): open an interactive window.
        max_bars (int): conformers beyond this are folded into "Other".

    Returns:
        str: the path written.
    """
    from .analysis import describe_conformation  # local: avoids a circular import

    labels = [describe_conformation(row, ring_size) for row in results_data]
    counts = Counter(labels)
    ordered = counts.most_common()
    if len(ordered) > max_bars:
        head, tail = ordered[:max_bars - 1], ordered[max_bars - 1:]
        ordered = head + [("Other", sum(count for _, count in tail))]

    names = [name for name, _ in ordered]
    percent = np.array([count for _, count in ordered], dtype=float)
    percent = 100.0 * percent / len(labels)

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.42 * len(names) + 1.6)))
    positions = np.arange(len(names))[::-1]  # most populated at the top
    ax.barh(positions, percent, height=0.68, color=PRIMARY_COLOR)

    for position, value, (_, count) in zip(positions, percent, ordered):
        ax.text(value + max(percent) * 0.015, position, f"{value:.1f}%  (n={count})",
                va="center", fontsize=9, color=REFERENCE_COLOR)

    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Population [% of frames]", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(percent) * 1.28)
    ax.set_title(title or "Conformer populations", fontsize=14)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    return _finish(fig, f"{path_prefix}_populations.png", show)


def plot_amplitude_histogram(results_data, path_prefix, title=None, show=False, bins=40):
    """
    Distribution of the puckering amplitude Q over the trajectory.

    Q is the one parameter the tool computes for every frame and never drew. It is
    worth a look on its own: the amplitude distribution of a pyranose is reported
    as bimodal when the ring visits both a chair and a non-chair basin, which the
    angular plots do not show directly.

    Args and Returns: see plot_mercator(); "_amplitude.png" is appended.
    """
    Q = results_data[:, COL_Q]

    # A rigid ring gives an amplitude range of zero width, which numpy cannot
    # split into bins. Widen it slightly so the degenerate case draws one bar
    # instead of raising.
    low, high = float(np.min(Q)), float(np.max(Q))
    span = high - low
    hist_range = None if span > 1e-9 else (low - max(abs(low) * 0.01, 1e-3),
                                           high + max(abs(high) * 0.01, 1e-3))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(Q, bins=bins, range=hist_range, color=PRIMARY_COLOR)

    mean_Q = float(np.mean(Q))
    ax.axvline(mean_Q, color="#B5411F", linestyle="--", linewidth=1.6,
               label=f"mean = {mean_Q:.3f} Å")

    ax.set_xlabel("Puckering amplitude Q [Å]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frames", fontsize=12, fontweight="bold")
    ax.set_title(title or "Puckering amplitude distribution", fontsize=14, pad=14)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    return _finish(fig, f"{path_prefix}_amplitude.png", show)


def _clean_fel_energies(data, unsampled="mask"):
    """
    Resolves the non-finite entries in the energy column.

    A converted free energy surface marks never-visited bins with +inf. That is
    real information -- "the sampling never came here" -- and there are two honest
    ways to draw it:

      "mask": leave those bins out of the colour scale entirely, so the ramp spans
              only energies that were actually measured and unexplored ground
              reads as blank. This is what the published landscapes do, and it
              matters because unsampled bins routinely outnumber sampled ones
              several to one; folding them into the ramp costs the basins nearly
              all of the available colour range.
      "max":  clamp them to the highest sampled energy, so they saturate the top
              of the scale.

    NaN and -inf carry no meaning on a free energy surface and are always dropped.

    Returns:
        tuple: (data, sampled_mask) with sampled_mask True for measured bins.
    """
    if unsampled not in ("mask", "max"):
        raise ValueError(f"unsampled must be 'mask' or 'max', got {unsampled!r}")

    keep = np.isfinite(data[:, :2]).all(axis=1)
    keep &= ~np.isnan(data[:, 2])
    keep &= data[:, 2] != -np.inf
    data = data[keep]

    sampled = np.isfinite(data[:, 2]) if len(data) else np.zeros(0, dtype=bool)
    if not sampled.any():
        raise ValueError("The FEL file has no rows with finite angles and energy.")

    if unsampled == "max":
        data[:, 2] = np.where(sampled, data[:, 2], data[sampled, 2].max())
        sampled = np.ones(len(data), dtype=bool)

    return data, sampled


def _resolve_energy_max(energies, energy_max):
    """
    Turns the energy_max option into a number.

    None leaves the scale alone. "auto" trims the top 1% of the sampled energies,
    dropping the poorly converged spikes at the edge of the explored region
    without inventing a threshold in the data's own energy unit -- a fixed number
    only ever suits the one surface it was chosen for.
    """
    if energy_max is None:
        return None
    if isinstance(energy_max, str):
        if energy_max != "auto":
            raise ValueError(
                f"energy_max must be a number, 'auto' or None, got {energy_max!r}")
        return float(np.percentile(energies, 99.0))
    return float(energy_max)


def _to_degrees(data, angle_units):
    """
    Converts the angle columns to degrees.

    Args:
        angle_units (str): "deg", "rad", or "auto" to guess. The guess requires
                           *both* angle columns to fit inside [-2pi, 2pi], which
                           a degree-valued surface only does if it spans a few
                           degrees in total.
    """
    if angle_units == "auto":
        limit = 2.0 * np.pi + 1e-6
        angle_units = "rad" if np.nanmax(np.abs(data[:, :2])) <= limit else "deg"

    if angle_units == "rad":
        data[:, 0] = np.degrees(data[:, 0])
        data[:, 1] = np.degrees(data[:, 1])
    elif angle_units != "deg":
        raise ValueError(f"angle_units must be 'deg', 'rad' or 'auto', got {angle_units!r}")

    return data


def plot_fel_mercator(data, path_prefix, title=None, show=False,
                      angle_units="auto", energy_label="Free Energy (kcal/mol)",
                      contour_step=1.0, cmap="viridis", energy_max=None,
                      unsampled="mask", overlay=None, suffix="_FEL"):
    """
    Generates a Free Energy Landscape (FEL) contour map from energy data arrays.

    Args:
        data (numpy.ndarray): columns [Theta, Phi, Energy].
        path_prefix (str): output path prefix; "_FEL.png" is appended.
        title (str): plot title; pass the job name, not the path prefix.
        show (bool): open an interactive window.
        angle_units (str): "deg", "rad" or "auto".
        energy_label (str): colourbar label -- set the unit your data is in.
                            PLUMED writes kJ/mol by default.
        contour_step (float): spacing of the labelled contour lines, in the
                              energy unit of the data. The published landscapes
                              use 1 kcal/mol, which is the default here.
        cmap (str): matplotlib colormap name.
        energy_max (float|"auto"|None): saturate the colour scale above this
                            energy. "auto" uses the 99th percentile of the sampled
                            energies; None (default) keeps the full sampled range.
        unsampled (str): "mask" to leave never-visited bins blank (default), or
                         "max" to clamp them to the top of the colour scale.
        overlay (numpy.ndarray): optional (n_frames, 4) array from
                         compute_puckering(); its (phi, theta) points are drawn on
                         top, showing which part of the landscape a structure or
                         trajectory actually visits.
        suffix (str): appended to path_prefix before ".png".

    Returns:
        str: the path written.
    """
    d = np.array(data[:, :3], dtype=float, copy=True)
    d, sampled = _clean_fel_energies(d, unsampled)
    d = _to_degrees(d, angle_units)

    cap = _resolve_energy_max(d[sampled, 2], energy_max)
    if cap is not None:
        d[:, 2] = np.minimum(d[:, 2], cap)

    d[:, 1] = d[:, 1] % 360

    # Apply periodic boundary conditions so interpolation does not tear at 0/360
    margin = 15
    wrap_left = d[:, 1] < margin
    wrap_right = d[:, 1] > 360 - margin
    left_margin = d[wrap_left].copy()
    left_margin[:, 1] += 360
    right_margin = d[wrap_right].copy()
    right_margin[:, 1] -= 360
    d = np.vstack([d, left_margin, right_margin])
    sampled = np.concatenate([sampled, sampled[wrap_left], sampled[wrap_right]])

    theta, phi, energy = d[:, 0], d[:, 1], d[:, 2]

    X_grid, Y_grid = np.meshgrid(np.linspace(0, 360, 500), np.linspace(0, 180, 500))

    # Interpolate the energy from the sampled bins only. Linear leaves NaN outside
    # the convex hull, so nearest-neighbour fills the remainder.
    Z_grid = griddata((phi[sampled], theta[sampled]), energy[sampled],
                      (X_grid, Y_grid), method="linear")
    gaps = np.isnan(Z_grid)
    if gaps.any():
        Z_nearest = griddata((phi[sampled], theta[sampled]), energy[sampled],
                             (X_grid, Y_grid), method="nearest")
        Z_grid[gaps] = Z_nearest[gaps]

    # Decide which grid cells count as explored by asking, for every cell, whether
    # the *nearest original bin* was sampled. Without this the interpolation would
    # happily extend the basins across ground the simulation never visited.
    if not sampled.all():
        coverage = griddata((phi, theta), sampled.astype(float),
                            (X_grid, Y_grid), method="nearest")
        Z_grid = np.ma.masked_where(coverage < 0.5, Z_grid)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor(NO_DATA_COLOR)

    colormap = plt.get_cmap(cmap)
    if hasattr(colormap, "with_extremes"):  # matplotlib >= 3.4
        colormap = colormap.with_extremes(bad=NO_DATA_COLOR)
    else:
        colormap = colormap.copy()
        colormap.set_bad(NO_DATA_COLOR)

    # extend="both" fills the open ends of the level range; without it, cells
    # sitting exactly on the outermost level are left unpainted (white).
    contourf = ax.contourf(X_grid, Y_grid, Z_grid, levels=100, cmap=colormap, extend="both")
    cbar = fig.colorbar(contourf, ax=ax, shrink=0.95, aspect=16)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(energy_label, fontsize=16, rotation=270, labelpad=20)

    # Bold contour lines every contour_step energy units. Levels are kept strictly
    # inside the range: a level sitting exactly on the flat top (which is where a
    # capped scale or an unsampled plateau puts most of the map) makes the contour
    # algorithm chase floating-point noise and scribble closed loops everywhere.
    min_e, max_e = float(np.nanmin(Z_grid)), float(np.nanmax(Z_grid))
    margin = 1e-6 * (max_e - min_e)
    levels = np.arange(np.floor(min_e / contour_step) * contour_step,
                       max_e + contour_step, contour_step)
    levels = levels[(levels > min_e + margin) & (levels < max_e - margin)]
    if levels.size > 0:
        contour = ax.contour(X_grid, Y_grid, Z_grid, levels=levels,
                             colors="black", linewidths=0.4)
        # Draw every contour but label only a few of them: at a 1 kcal/mol
        # spacing a label on each line buries the map in numbers.
        stride = max(1, int(np.ceil(levels.size / 6.0)))
        ax.clabel(contour, levels=levels[::stride], inline=True, fontsize=9, fmt="%.0f")

    # Overlay ideal conformational reference points
    ax.scatter(IDEAL_PHI, IDEAL_THETA, marker=".", color="white",
               edgecolor="black", s=50, zorder=3)
    # A white halo instead of a filled box: the labels have to stay legible over
    # both the dark basins and the pale unsampled ground, and a box on each of
    # the 41 points clutters the map.
    halo = [path_effects.withStroke(linewidth=2.5, foreground="white")]
    for i, label in enumerate(LABELS_TEX):
        ax.annotate(label, (IDEAL_PHI[i], IDEAL_THETA[i]), fontsize=13,
                    color="black", weight="bold", zorder=4,
                    path_effects=halo)

    ax.axis((0, 360, 180, 0))
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_yticks(np.arange(0, 181, 45))
    ax.set_xlabel("Phi (φ) [°]", fontsize=16, fontweight="bold")
    ax.set_ylabel("Theta (θ) [°]", fontsize=16, fontweight="bold")
    ax.set_title(title or "Free Energy Landscape", fontsize=20, pad=20)

    if overlay is not None and len(overlay):
        # White edges so the markers stay legible over both the dark basins and
        # the light unsampled ground.
        if len(overlay) == 1:
            ax.scatter(overlay[:, COL_PHI], overlay[:, COL_THETA], s=260,
                       color="red", marker="*", edgecolor="white",
                       linewidths=1.2, zorder=6, label="structure")
        else:
            ax.scatter(overlay[:, COL_PHI], overlay[:, COL_THETA], s=18,
                       color="red", edgecolor="white", linewidths=0.4,
                       alpha=0.85, zorder=6, label=f"trajectory ({len(overlay)} frames)")
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)

    fig.tight_layout()
    return _finish(fig, f"{path_prefix}{suffix}.png", show)
