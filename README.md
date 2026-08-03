# 🍩 Sugar Puckering Analyzer (MD & FEL)

⚠️⚠️ UNFINISHED VERSION ⚠️⚠️

A robust, Python-based GUI application for the conformational analysis of sugar rings — 6-membered
pyranoses via Cremer-Pople puckering coordinates (Q, θ, φ) and 5-membered furanoses via
Altona-Sundaralingam pseudorotation (Q, P, ν<sub>max</sub>).

This tool is designed to analyze static PDB structures, Molecular Dynamics (MD) trajectories, and pre-calculated Free Energy Landscapes (FEL). It automatically classifies conformations using strict IUPAC angular boundaries and generates publication-ready plots.

## ✨ Features
* **Multi-Format Support:** Reads PDB files and common MD trajectory formats (NetCDF, DCD, XTC) through `mdtraj`.
* **Pyranoses and furanoses:** the ring size is inferred from how many atom indices you give — six for a
  pyranose (Stoddart map, 41 reference conformers), five for a furanose (pseudorotation wheel,
  20 reference conformers).
* **Strict Conformational Assignment:** Uses rigorous angular boundaries (Chairs: θ < 15º, Equatorials: 75º < θ < 105, etc.) avoiding Euclidean distance ambiguities.
* **Automated Plotting:** Mercator maps and polar Stoddart diagrams (with the itinerary drawn as a
  connected path), a conformer timeline, time-series of θ/φ/Q, conformer population bars,
  puckering-amplitude histograms, pseudorotation wheels, and FEL contour maps — optionally with the
  structure or trajectory projected onto the landscape.
* **Ring sanity check:** refuses atom selections that are not a closed ring, when the topology says so.
* **Tabular Output:** Exports nicely formatted `.dat` files with the geometric parameters for each frame.
* **Intuitive GUI:** Built-in Tkinter interface for seamless file selection and execution.

## 📚 Theoretical Background

The puckering coordinates follow the original definition:
> *Cremer, D., & Pople, J. A. (1975). A general definition of ring puckering coordinates. Journal of the American Chemical Society, 97(6), 1354-1358.*

Furanose pseudorotation follows:
> *Altona, C., & Sundaralingam, M. (1972). Conformational analysis of the sugar ring in nucleosides and nucleotides. A new description using the concept of pseudorotation. Journal of the American Chemical Society, 94(23), 8205-8212.*

The Stoddart mapping and the reference conformational itinerary are based on:
> *Ardèvol, A., Biarnés, X., Planas, A., & Rovira, C. (2010). The Conformational Free-Energy Landscape of beta-D-Mannopyranose: Evidence for a 1S5 -> B2,5 -> OS2 Catalytic Itinerary in beta-Mannosidases. Journal of the American Chemical Society, 132(45), 16058-16065.*

## 🛠️ Installation

**Prerequisites:** [Python 3.8+](https://www.python.org/) and `git`.

### Option A: From source (for developers/researchers)

```bash
git clone https://github.com/JorgePardos/SugarPuckering.git
cd SugarPuckering
python -m venv .venv
.venv\Scripts\activate      # Windows;  on Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

Then launch the GUI:

```bash
python sugar_puckering.py
```

### Option B: Standalone executable

Build a self-contained `.exe` (no Python installation needed on the target machine).
This requires the source checkout and dependencies from Option A first:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --collect-all mdtraj --hidden-import PIL._tkinter_finder --hidden-import scipy.interpolate sugar_puckering.py
```

The executable is written to `dist/sugar_puckering.exe`.

## ⌨️ Command line interface

The same analysis runs headless, which is handy for scripting and for batch jobs:

```bash
python -m src.cli pdb --files structure.pdb --indices "11 12 13 14 15 16" --job myjob
python -m src.cli md  --top system.prmtop --traj run.nc --indices "11 12 13 14 15 16"
python -m src.cli md  --top system.prmtop --traj run.nc --indices "11 12 13 14 15 16" --fel fes.dat
python -m src.cli fel --file fes.dat
```

Run `python -m src.cli <mode> --help` for the full option list. FEL rendering accepts
`--angle-units`, `--energy-label`, `--contour-step`, `--cmap`, `--energy-max` and `--unsampled`.
The same options are available in the GUI's *Plot options* panel, each labelled with its flag.

**See [MANUAL.md](MANUAL.md)** for what every option does, how to read the output, and recipes.

### Reading a free energy surface

Never-visited bins arrive as `+inf`. By default they are **masked**, so the colour scale is spent
entirely on energies that were actually measured and unexplored ground reads as blank — on a typical
metadynamics surface the unsampled bins outnumber the sampled ones several to one, and folding them
into the ramp leaves the basins almost no contrast. `--unsampled max` restores the older look, where
they saturate the top of the scale.

`--energy-max` caps the scale; `--energy-max auto` uses the 99th percentile of the *sampled* energies,
so it adapts to whatever unit and range the file is in instead of assuming one.

Contour lines default to 1 kcal/mol, matching the published landscapes, and only a few are labelled.

## 🧪 Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests that need `mdtraj` are skipped automatically if it is not installed.

`tests/test_real_data.py` runs against the research files under `tests/data/` (the PLUMED
`fes.dat`, the reference PDBs and the trajectory). Those are hundreds of megabytes and are
not committed, so each of those tests skips when its input is absent.

## 📋 Input formats

| Mode | Input |
|---|---|
| **Static PDB(s)** | One or more `.pdb` files sharing the same topology, plus the 1-based ring atom indices in connectivity order — six for a pyranose (O5, C1, C2, C3, C4, C5) or five for a furanose (O4, C1, C2, C3, C4). |
| **MD Trajectory** | Topology (`.prmtop`, `.parm7`, `.pdb`) + trajectory (`.nc`, `.dcd`, `.xtc`, `.trr`, `.crd`), plus the same indices. |
| **FEL** | A pre-computed free energy surface (e.g. the `fes.dat` produced by `plumed sum_hills`), as a whitespace-delimited table of θ, φ and energy. PLUMED's `#! FIELDS` header is read when present. |

The atom **order matters**: puckering coordinates are defined on the ring traversed in connectivity order, and reversing it maps θ to 180−θ (i.e. swaps ⁴C₁ and ¹C₄).

Note that atom numbering is **not** generally shared between a PDB and its topology file — check the
`Ring atoms resolved to:` line, which the tool prints and records in the output header.

## 📝 Changelog

### Unreleased

* **Fixed — puckering amplitude Q was systematically under-estimated.** The mean-plane normal was
  built by normalising R′ and R″ *before* taking their cross product; since those two vectors are not
  orthogonal in general, the resulting normal had length `sin∠(R′,R″)` instead of 1, scaling every
  atomic displacement and therefore Q. The error is small for near-ideal rings (~0.1 %) but grows with
  ring distortion — up to ~45 % in synthetic stress tests.
  **θ and φ are unaffected** (they are ratios in which the scale factor cancels), so previously assigned
  conformations remain valid; only the reported Q values change. Re-run any analysis where Q was used
  quantitatively.
* **Fixed — silent `NaN` results on degenerate geometries.** An ideal chair produced `φ = NaN` and a
  planar ring produced `θ = NaN`; both propagated unnoticed into the output table and the plots. φ is now
  reported as 0 for an ideal chair (where it is genuinely undefined and unused, as the conformation comes
  from θ alone), and a truly planar or collinear ring raises an explicit error instead of returning `NaN`.
* Input validation added: the coordinate array must be 6×3 and finite.
* **Fixed — the atom index field could be left untouched and still "work".** The box was
  pre-filled with `Ex: 11 12 13 14 15 16` and the parser kept only tokens passing `str.isdigit()`,
  which dropped the `Ex:` and left exactly six valid indices. Anyone who never edited the field got
  a silent analysis of atoms 11–16. Malformed input is now rejected with an explicit message.
* **Fixed — unsampled FEL bins rendered as speckle.** `+inf` bins were clamped to the highest finite
  energy, turning each one into a spike among its sampled neighbours; they are now dropped so the
  surrounding data closes the gap. `NaN` bins were not handled at all and poisoned the interpolation.
* Fixed: a plotting failure could be preceded by a "Success" dialog; the GUI froze with no feedback
  during long trajectories; `plt.show()` blocked the Tk event loop and figures were never released.
* Fixed: plot titles showed the output path (`Conformational Map - myjob\myjob`) instead of the job name.
* Fixed: ring atoms are now reordered explicitly after loading, rather than trusting `mdtraj` to honour
  the requested order in `atom_indices=`. Out-of-range indices report an error instead of an `IndexError`.
* Fixed: the conformation tables mixed the letter O and the digit 0 (`5HO` alongside `1H0`). The ring
  oxygen is always the letter O now.
* **Added — the six atoms are now checked to form a closed ring.** Cremer-Pople arithmetic succeeds on
  any six points and returns plausible-looking numbers, so indices that are valid for one topology but
  point at unrelated atoms in another produced a whole trajectory of nonsense in silence. When the
  topology carries bonds this is now refused outright (`--skip-ring-check` overrides). The resolved atom
  names are also printed and recorded in the output header so the selection can be eyeballed.
* **Added — 5-membered rings (furanoses).** Give five indices instead of six and the tool reports the
  Altona-Sundaralingam phase P and ν<sub>max</sub>, assigns one of the 20 canonical conformers, and draws a
  pseudorotation wheel. The wheel is anchored by construction: puckering a single ring atom out of the
  mean plane lands on the envelope named after it, checked for all five atoms.
* **Changed — unsampled FEL bins are masked instead of painted.** They used to be clamped to the
  highest sampled energy, which on a real surface means most of the map sits at the top of the colour
  scale and the basins share what little is left. They are now left blank, as in the published
  landscapes; `--unsampled max` restores the previous behaviour.
* **Added — FEL colour scale can be capped** (`--energy-max`), with `auto` deriving the cap from the
  sampled energies rather than assuming a fixed number of kcal/mol.
* **Changed — contour lines default to 1 kcal/mol** instead of 5, matching the published landscapes,
  and only a subset are labelled so the numbers do not bury the map.
* **Added — three more figures:** a polar Stoddart diagram (both hemispheres seen from their poles,
  complementing the Mercator map), a conformer population bar chart, and a puckering-amplitude
  histogram — Q was computed for every frame and previously never drawn anywhere.
* **Added — a structure or trajectory can be projected onto a free energy surface** (`--fel` in the
  pdb/md modes), showing which part of the landscape the simulation actually visits.
* **Added — the conformational itinerary is now visible as a sequence.** Consecutive frames are joined
  on the Mercator and Stoddart plots (the path is broken at the φ = 0/360 seam, where a small periodic
  step would otherwise streak a line across the map), and a conformer timeline shows which conformation
  is occupied at each step, stacked in canonical order so the vertical axis behaves like a latitude.
* **Added — Q now has a panel in the time series.** θ and φ say *which* conformation the ring is in;
  only Q says *how puckered* it is. A ⁴C₁-labelled frame at Q = 0.6 Å and one at Q = 0.2 Å plot at the
  same point on the Stoddart map but are chemically very different — ring flattening towards an
  oxocarbenium-like transition state appears in Q and nowhere else. Q also bounds how far the labels
  can be trusted: as Q → 0 both angles become ill-conditioned.
* Fixed: a contour level landing exactly on a flat plateau made the contour algorithm chase
  floating-point noise and scribble thousands of tiny loops across it.
* Added: a headless CLI (`python -m src.cli`), and a pytest suite covering the maths, the pipeline and
  the plots.
