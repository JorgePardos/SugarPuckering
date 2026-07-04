"""
plotting.py
Visualization tools for generating Stoddart/Mercator maps, Time Series, 
and Free Energy Landscapes (FEL).
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Ensure compatibility with Tkinter GUI in Linux
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Import mathematical constants from local module
from src.math_core import IDEAL_PHI, IDEAL_THETA

# LaTeX formatted labels for high-quality plots
LABELS_TEX = [r"$^4C_1$", r"$^OE$", r"$^OH_1$", r"$E_1$", r"$^2H_1$", r"$^2E$", r"$^2H_3$", r"$E_3$", r"$^4H_3$", r"$^4E$", r"$^4H_5$", r"$E_5$", r"$^OH_5$", r"$^OE$",
              r"$^{3O}B$", r"$^3S_1$", r"$B_{14}$", r"$^5S_1$", r"$^{25}B$", r"$^2S_O$", r"$B_{3O}$", r"$^1S_3$", r"$^{14}B$", r"$^1S_5$", r"$B_{25}$", r"$^OS_2$", r"$^{3O}B$",
              r"$^3E$", r"$^3H_4$", r"$E_4$", r"$^5H_4$", r"$^5E$", r"$^5H_O$", r"$E_O$", r"$^1H_O$", r"$^1E$", r"$^1H_2$", r"$E_2$", r"$^3H_2$", r"$^3E$", r"$^1C_4$"]

def plot_mercator(results_data, name_img="sugar_plot"):
    """
    Plots a Mercator projection of the Cremer-Pople sphere showing the 
    trajectory points over the ideal Stoddart conformational map.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot reference background map
    ax.scatter(IDEAL_PHI, IDEAL_THETA, marker=".", color='gray', alpha=0.5)
    for i, txt in enumerate(LABELS_TEX):
        ax.annotate(txt, (IDEAL_PHI[i], IDEAL_THETA[i]), fontsize=11, alpha=0.7, color='gray')

    # Setup axes
    ax.axis((0, 360, 180, 0)) # Y-axis inverted for standard Stoddart representation
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_yticks(np.arange(0, 181, 45))
    ax.grid(True, linestyle="--", alpha=0.6)
    
    ax.set_xlabel("Phi (φ) [°]", fontsize=12, fontweight='bold')
    ax.set_ylabel("Theta (θ) [°]", fontsize=12, fontweight='bold')
    ax.set_title(f"Conformational Map - {name_img}", fontsize=14)

    # Plot specific user data
    if len(results_data) == 1:
        ax.scatter(results_data[:, 3], results_data[:, 2], s=200, color='red', marker='*', edgecolor='black', zorder=5)
    else:
        # 1-based frame indexing for the colorbar
        frames = results_data[:, 0] + 1
        scatter = ax.scatter(results_data[:, 3], results_data[:, 2], s=30, c=frames, cmap="plasma", zorder=5, edgecolor='black', linewidths=0.5)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.85)
        cbar.set_label("Frame", rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(f"{name_img}_mercator.png", dpi=300)
    plt.show()

def plot_time_series(results_data, name_img="sugar_plot"):
    """
    Plots the temporal evolution of Theta and Phi parameters along a MD trajectory.
    """
    frames = results_data[:, 0] + 1  # 1-based indexing
    theta = results_data[:, 2]
    phi = results_data[:, 3]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Theta subplot (Chair vs Boat progression)
    ax1.plot(frames, theta, color='royalblue', linewidth=1.5, alpha=0.8)
    ax1.scatter(frames, theta, color='royalblue', s=10)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3, label='Chair 4C1 (~0°)')
    ax1.axhline(y=90, color='green', linestyle='--', alpha=0.3, label='Boats/Skews (~90°)')
    ax1.axhline(y=180, color='purple', linestyle='--', alpha=0.3, label='Chair 1C4 (~180°)')
    
    ax1.set_ylabel("Theta (θ) [°]", fontsize=12, fontweight='bold')
    ax1.set_title(f"Conformational Evolution - {name_img}", fontsize=14)
    ax1.set_ylim(-10, 190)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Phi subplot (Equatorial pseudorotation)
    ax2.plot(frames, phi, color='darkorange', linewidth=1.5, alpha=0.8)
    ax2.scatter(frames, phi, color='darkorange', s=10)
    ax2.set_ylabel("Phi (φ) [°]", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Frame", fontsize=12, fontweight='bold')
    ax2.set_ylim(-10, 370)
    ax2.set_yticks(np.arange(0, 361, 60))
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{name_img}_timeseries.png", dpi=300)
    plt.show()

def plot_fel_mercator(data, name_img="FEL_plot"):
    """
    Generates a Free Energy Landscape (FEL) contour map from energy data arrays.
    Assumes data format: [Theta, Phi, Energy].
    """
    d = data.copy()
    
    # 1. Handle Infinite energy values (replace with max finite value)
    max_energy = np.max(d[np.isfinite(d[:, 2]), 2])
    d[:, 2] = np.where(np.isinf(d[:, 2]), max_energy, d[:, 2])

    # 2. Unit detection (Convert radians to degrees if necessary)
    if np.max(d[:, 0]) <= 4.0:
        d[:, 0] = np.degrees(d[:, 0])
        d[:, 1] = np.degrees(d[:, 1])
        
    d[:, 1] = d[:, 1] % 360
    
    # 3. Apply periodic boundary conditions to avoid interpolation artifacts at 0/360 edges
    margin = 15
    left_margin = d[d[:, 1] < margin].copy()
    left_margin[:, 1] += 360
    right_margin = d[d[:, 1] > 360 - margin].copy()
    right_margin[:, 1] -= 360
    d = np.vstack([d, left_margin, right_margin])

    theta = d[:, 0]
    phi = d[:, 1]
    energy = d[:, 2]

    # Create grid for SciPy interpolation
    X_grid, Y_grid = np.meshgrid(np.linspace(0, 360, 500), np.linspace(0, 180, 500))
    Z_grid = griddata((phi, theta), energy, (X_grid, Y_grid), method='linear')

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot filled contours
    contourf = ax.contourf(X_grid, Y_grid, Z_grid, levels=100, cmap='viridis')
    cbar = fig.colorbar(contourf, ax=ax, shrink=0.95, aspect=16)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Free Energy (kcal/mol)', fontsize=16, rotation=270, labelpad=20)
    
    # Plot bold contour lines every 5 kcal/mol
    min_e, max_e = np.nanmin(Z_grid), np.nanmax(Z_grid)
    if max_e > min_e:
        contour = ax.contour(X_grid, Y_grid, Z_grid, levels=np.arange(np.floor(min_e), np.ceil(max_e), 5), colors='black', linewidths=0.3)
        ax.clabel(contour, inline=True, fontsize=9, fmt="%.1f")

    # Overlay ideal conformational reference points
    ax.scatter(IDEAL_PHI, IDEAL_THETA, marker=".", color='white', edgecolor='black', s=50, zorder=3)
    for i, txt in enumerate(LABELS_TEX):
        ax.annotate(txt, (IDEAL_PHI[i], IDEAL_THETA[i]), fontsize=13, color='black', weight='bold', 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    # Format Axes
    ax.axis((0, 360, 180, 0))
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_yticks(np.arange(0, 181, 45))
    ax.set_xlabel("Phi (φ) [°]", fontsize=16, fontweight='bold')
    ax.set_ylabel("Theta (θ) [°]", fontsize=16, fontweight='bold')
    ax.set_title(f"Free Energy Landscape - {name_img}", fontsize=20, pad=20)

    plt.tight_layout()
    plt.savefig(f"{name_img}_FEL.png", dpi=300)
    plt.show()