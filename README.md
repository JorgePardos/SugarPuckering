# 🍩 Sugar Puckering Analyzer (MD & FEL)

A robust, Python-based GUI application for the conformational analysis of 6-membered sugar rings using Cremer-Pople puckering coordinates (Q, θ, φ). 

This tool is designed to analyze static PDB structures, Molecular Dynamics (MD) trajectories, and pre-calculated Free Energy Landscapes (FEL). It automatically classifies conformations using strict IUPAC angular boundaries and generates publication-ready plots.

## ✨ Features
* **Multi-Format Support:** Reads PDB files and common MD trajectory formats (NetCDF, DCD, XTC) through `mdtraj`.
* **Strict Conformational Assignment:** Uses rigorous angular boundaries (Chairs: $\theta \le 15^\circ$, Equatorials: $75^\circ \le \theta \le 105^\circ$, etc.) avoiding Euclidean distance ambiguities.
* **Automated Plotting:** Generates Mercator/Stoddart conformational maps, Time-Series evolution plots, and FEL contour maps.
* **Tabular Output:** Exports nicely formatted `.dat` files with the geometric parameters for each frame.
* **Intuitive GUI:** Built-in Tkinter interface for seamless file selection and execution.

## 📚 Theoretical Background
The mathematical implementation and the Stoddart mapping are based on:
> *Ardèvol, A., Biarnés, X., Planas, A., & Rovira, C. (2010). The Conformational Free-Energy Landscape of $\beta$-D-Mannopyranose: Evidence for a 1S5 -> B2,5 -> OS2 Catalytic Itinerary in $\beta$-Mannosidases. Journal of the American Chemical Society, 132(45), 16058-16065.*

## 🛠️ Installation (Development)

Clone this repository and install the dependencies:

```bash
git clone [https://github.com/tu-usuario/SugarPuckering.git](https://github.com/tu-usuario/SugarPuckering.git)
cd SugarPuckering
pip install -r requirements.txt