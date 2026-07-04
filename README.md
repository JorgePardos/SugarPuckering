# 🍩 Sugar Puckering Analyzer (MD & FEL)

⚠️⚠️ UNFINISHED VERSION ⚠️⚠️

A robust, Python-based GUI application for the conformational analysis of 6-membered sugar rings using Cremer-Pople puckering coordinates (Q, θ, φ). 

This tool is designed to analyze static PDB structures, Molecular Dynamics (MD) trajectories, and pre-calculated Free Energy Landscapes (FEL). It automatically classifies conformations using strict IUPAC angular boundaries and generates publication-ready plots.

## ✨ Features
* **Multi-Format Support:** Reads PDB files and common MD trajectory formats (NetCDF, DCD, XTC) through `mdtraj`.
* **Strict Conformational Assignment:** Uses rigorous angular boundaries (Chairs: θ < 15º, Equatorials: 75º < θ < 105, etc.) avoiding Euclidean distance ambiguities.
* **Automated Plotting:** Generates Mercator/Stoddart conformational maps, Time-Series evolution plots, and FEL contour maps.
* **Tabular Output:** Exports nicely formatted `.dat` files with the geometric parameters for each frame.
* **Intuitive GUI:** Built-in Tkinter interface for seamless file selection and execution.

## 📚 Theoretical Background
The mathematical implementation and the Stoddart mapping are based on:
> *Ardèvol, A., Biarnés, X., Planas, A., & Rovira, C. (2010). The Conformational Free-Energy Landscape of beta-D-Mannopyranose: Evidence for a 1S5 -> B2,5 -> OS2 Catalytic Itinerary in beta-Mannosidases. Journal of the American Chemical Society, 132(45), 16058-16065.*

## 🛠️ Installation Guide

You can run **Sugar Puckering Analyzer** either from the Python source code or by creating a standalone executable.

### Option A: From Source (For Developers/Researchers)

**Prerequisites:** [Python 3.8+](https://www.python.org/) and `git`.

1. **Clone the repository:**
```bash
git clone [https://github.com/JorgePardos/SugarPuckering.git](https://github.com/JorgePardos/SugarPuckering.git)
cd SugarPuckering

2. Install 
pip install pyinstaller
pyinstaller --onefile --windowed --collect-all mdtraj --hidden-import PIL._tkinter_finder --hidden-import scipy.interpolate sugar_puckering.py

