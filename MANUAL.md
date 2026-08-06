# Sugar Puckering Analyzer — Manual

How to run the program, what every option does, and how to read what comes out.

For installation see [README.md](README.md). This document assumes the program already runs.

---

## Contents

1. [The three modes](#1-the-three-modes)
2. [Choosing the ring atoms](#2-choosing-the-ring-atoms)
3. [What the numbers mean](#3-what-the-numbers-mean)
4. [Reading the output files](#4-reading-the-output-files)
5. [The figures](#5-the-figures)
6. [Options reference](#6-options-reference)
7. [Free energy surfaces in detail](#7-free-energy-surfaces-in-detail)
8. [Recipes](#8-recipes)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. The three modes

| Mode | Input | Answers |
|---|---|---|
| **Static PDB(s)** | one or more `.pdb` | what conformation is this structure in? |
| **(QMMM) MD Trajectory** | topology + trajectory | how does the conformation evolve? |
| **Free Energy (FEL)** | a pre-computed surface | what does the landscape look like? |

Both front ends do the same work and share the same defaults.

**GUI** — pick a mode with the radio buttons at the top, fill in the files, press *Run Analysis*:

```bash
python sugar_puckering.py
```

**Command line** — one sub-command per mode:

```bash
python -m src.cli pdb --files structure.pdb --indices "11 12 13 14 15 16"
python -m src.cli md  --top system.prmtop --traj run.dcd --indices "11 12 13 14 15 16"
python -m src.cli fel --file fes.dat
```

`python -m src.cli <mode> --help` prints every flag. Every GUI option shows its command-line
equivalent in grey next to it, so a run set up by hand can be turned into a script.

Results go to a folder named after `--job` (`Output Folder / Job Name` in the GUI), falling back to
the input file name.

---

## 2. Choosing the ring atoms

Give the ring atoms as **1-based indices, in ring connectivity order**:

| Ring | Count | Order |
|---|---|---|
| Pyranose (6) | 6 indices | O5, C1, C2, C3, C4, C5 |
| Furanose (5) | 5 indices | O4, C1, C2, C3, C4 |

The ring size is inferred from how many indices you give. Separators can be spaces or commas:
`11 12 13 14 15 16` and `11,12,13,14,15,16` are the same.

### Order matters

Puckering coordinates are defined on the ring *traversed in order*. Reversing the traversal maps
θ to 180−θ, which swaps ⁴C₁ and ¹C₄ — a wrong order silently gives you the mirror-image answer.

Two safeguards run automatically:

- **The resolved atoms are printed and written into the output header.** Always read that line:
  ```
  Ring atoms resolved to: TRH452-O5, TRH452-C1, TRH452-C2, TRH452-C3, TRH452-C4, TRH452-C5
  ```
- **The six atoms are checked to form a closed ring**, when the topology carries bond information.
  Puckering arithmetic succeeds on *any* six points and returns plausible-looking numbers, so
  without this a wrong index yields a whole trajectory of nonsense in silence.

> ⚠️ **Atom numbering is not shared between a PDB and its topology file.** The same residue can sit
> at different indices in `structure.pdb` and in `system.prmtop`. Never reuse indices across the two
> without checking the resolved-atoms line.

Finding the indices with mdtraj:

```bash
python -c "import mdtraj; t=mdtraj.load_topology('system.prmtop'); print([(a.index+1, str(a)) for a in t.residue(451).atoms])"
```

---

## 3. What the numbers mean

### Pyranoses — Cremer-Pople (Q, θ, φ)

Think of a sphere:

- **Q** (Å) — the radius: how far the ring is from planar. A relaxed pyranose sits near 0.5–0.6 Å.
- **θ** (0–180°) — the latitude: ⁴C₁ chair at the north pole, boats and skews around the equator,
  ¹C₄ chair at the south pole.
- **φ** (0–360°) — the longitude: which boat/skew.

Conformations are assigned by **strict IUPAC angular boundaries**, not by nearest-neighbour distance:

| Band | Range | Conformers |
|---|---|---|
| North pole | θ ≤ 15° | ⁴C₁ |
| Northern | 15° < θ < 75° | half-chairs and envelopes |
| Equator | 75° ≤ θ ≤ 105° | boats and skews |
| Southern | 105° < θ < 165° | half-chairs and envelopes |
| South pole | θ ≥ 165° | ¹C₄ |

Within a band, φ is split into twelve 30° sectors centred on each ideal conformer.

### Furanoses — Altona-Sundaralingam (Q, P, ν_max)

A 5-membered ring has a single puckering mode, so the sphere collapses to a circle: there is no θ.

- **Q** (Å) — the puckering amplitude, as above.
- **P** (0–360°) — the pseudorotation phase: which of the 20 conformers (10 envelopes E, 10 twists T,
  every 18°). ³E sits at P ≈ 18° (north), ²E at P ≈ 162° (south).
- **ν_max** (°) — the largest endocyclic torsion, the amplitude in the torsional description.

### Why Q deserves attention

The Mercator and Stoddart maps project onto the surface of the sphere and therefore **discard Q**.
θ and φ say *which* conformation; only Q says *how puckered*. That matters:

- **Chemistry.** An oxocarbenium-like transition state requires C5-O5-C1-C2 coplanarity — the ring
  flattens, and Q falls. A ¹S₅ at Q = 0.6 Å and a "¹S₅" at Q = 0.25 Å plot at the same point on the
  map but are very different states.
- **Trusting the labels.** As Q → 0 both angles become ill-conditioned. A frame with a tiny Q and a
  confident-looking conformer label is noise.
- **Quality control.** A ring broken by a simulation artefact shows up as an anomalous Q before
  anything else gives it away.

---

## 4. Reading the output files

`<job>_params.dat` — one row per frame. Every non-numeric line is `#`-prefixed, so the whole file
loads with a bare `numpy.loadtxt`:

```
# 6-membered ring (pyranose)
# Ring atoms (in order): TRH452-O5, TRH452-C1, TRH452-C2, TRH452-C3, TRH452-C4, TRH452-C5
# Q in Angstrom; Theta and Phi in degrees
#  Frame      Q(A)    Theta(deg)      Phi(deg)    Conformation
# ------------------------------------------------------------
       1    0.6621       88.9251       84.3092             5S1
```

Frame numbers are 1-based. For a furanose the two angle columns become `P(deg)` and `NuMax(deg)`.

```python
import numpy as np
data = np.loadtxt("job/job_params.dat", usecols=(0, 1, 2, 3))   # numbers
labels = [l.split()[-1] for l in open("job/job_params.dat")
          if not l.startswith("#") and l.strip()]               # conformers
```

---

## 5. The figures

| File | Shows | When |
|---|---|---|
| `_mercator.png` | conformation on the Stoddart/Mercator rectangle | pyranose |
| `_stoddart.png` | both hemispheres seen from their poles | pyranose |
| `_wheel.png` | the pseudorotation circle | furanose |
| `_timeseries.png` | θ, φ and Q against frame | > 1 frame |
| `_timeline.png` | which conformer at each step | > 1 frame |
| `_populations.png` | how much of the run sat in each conformer | > 1 frame |
| `_amplitude.png` | distribution of Q | > 1 frame |
| `_FEL.png` | the free energy landscape | FEL mode |
| `_on_FEL.png` | structure/trajectory projected onto a landscape | with `--fel` |

All are 300 dpi PNG.

**Mercator vs Stoddart** — the same data, two projections. Mercator fits every conformer on one
rectangle but stretches each pole across the whole width; the polar Stoddart diagram puts each chair
at a single point and shows the ring of boats and skews undistorted. They complement each other.

**Timeline vs populations** — the populations chart says *how much* time went where; the timeline
says *when*, and in what order. For a reaction path the timeline is usually the one you want: its
vertical axis follows the canonical conformer sequence (⁴C₁ → northern band → equator → southern band
→ ¹C₄), so an itinerary reads as a path rather than as an arbitrary reshuffling.

Points are drawn without connecting lines; the colour carries the order. Passing `connect=True` to
`plot_mercator` or `plot_stoddart` joins consecutive frames when a short path is easier to follow than
a colour ramp — the line is broken where φ crosses 0/360, since a step from 355° to 5° is a small move
on a periodic axis but a line straight across the map if drawn literally.

---

## 6. Options reference

### Every mode

| GUI | CLI | Default | What it does |
|---|---|---|---|
| Output Folder / Job Name | `--job` | input file name | names the output folder and the figure titles |
| — | `--outdir` | `.` | where the job folder is created |

### PDB and MD modes

| GUI | CLI | Default | What it does |
|---|---|---|---|
| Ring Atom Indices | `--indices` | — | 6 indices for a pyranose, 5 for a furanose |
| Timestep (ps/frame) | `--timestep` | blank | puts the per-frame plots on a time axis instead of frame numbers |
| Skip the closed-ring check | `--skip-ring-check` | off | proceed even if the atoms are not bonded as a ring |
| Project onto FEL | `--fel` | none | also draw the result on this free energy surface |
| Angle units | `--angle-units` | `auto` | units of the projected surface |
| Energy label | `--energy-label` | `Free Energy (kcal/mol)` | colourbar label of the projected surface |

### Putting the plots on a time axis

By default the per-frame plots are drawn against the frame number. Give `--timestep` (picoseconds
between frames) and they switch to simulation time, automatically labelled in ps or, past a thousand,
in ns.

Whether you need it depends on the format:

| Format | Per-frame time | Need `--timestep`? |
|---|---|---|
| `.nc` (AMBER NetCDF) | stored, in ps | no |
| `.xtc`, `.trr` | stored | no |
| `.dcd` | not preserved; header timing read instead | check the printed value |

DCD does not round-trip per-frame times — mdtraj returns 0, 1, 2, …, i.e. frame indices wearing time
units — but its header carries `DELTA` (the integration step) and `NSAVC` (the save frequency). The
program reads them and uses `DELTA × NSAVC` as the spacing. `DELTA` may be stored in AKMA units or
already in picoseconds, and the header flag that is supposed to say which is not reliable — files
carrying the same flag have been seen storing each — so both readings are tried and the one that is a
possible integration step (roughly 0.1 to 10 fs) wins. It always prints what it found:

```
Timestep: 0.001 ps/frame, DCD header (CHARMM: DELTA=0.001 ps, NSAVC=1) -> 0.381 ps over 382 frames
```

**Read that line.** Those header fields are often wrong: many writers, QM/MM codes especially, store
the integration step and leave `NSAVC` at 1 regardless of how often frames were actually written, so
a run saved every 250 steps still claims 1 fs per frame and the header describes a run hundreds of
times shorter than the one on disk. If the total does not match your simulation, pass `--timestep`
— an explicit value always wins.

Formats other than DCD are left alone: their stored times are used directly, and a 0, 1, 2, …
sequence is treated as "no time information" so the plots stay on frame numbers rather than
labelling indices as picoseconds.

```bash
# 382 frames over 100 ps
python -m src.cli md --top system.prmtop --traj run.dcd --indices "..." --timestep 0.2618
```

`--skip-ring-check` exists for topologies with missing or wrong bond records. If the check fires,
the far more likely explanation is a wrong index — read the error, it names the atoms it found.

### FEL mode

| GUI | CLI | Default | What it does |
|---|---|---|---|
| Angle units | `--angle-units` | `auto` | `deg`, `rad`, or guess |
| Unsampled bins | `--unsampled` | `mask` | `mask` leaves them blank, `max` clamps them to the top of the scale |
| Colormap | `--cmap` | `viridis` | any matplotlib colormap |
| Contour spacing | `--contour-step` | `1` | gap between contour lines, in the data's energy unit |
| Max energy | `--energy-max` | blank | cap the colour scale: a number, `auto`, or blank for the full range |
| Energy label | `--energy-label` | `Free Energy (kcal/mol)` | colourbar label — **set this to match your data** |

### How to change them

**In the GUI**, the *Plot options* panel below the file selectors rebuilds itself for the current
mode. Each control shows its CLI flag in grey. Text fields are validated when you press *Run
Analysis*, so a typo is reported before any file is read.

**On the command line**, pass the flag:

```bash
python -m src.cli fel --file fes.dat --contour-step 5 --unsampled max --cmap cividis
```

**In a script**, call the plotting functions directly — they take the same names as keyword
arguments:

```python
from src.analysis import load_fel
from src.plotting import plot_fel_mercator

data, fields = load_fel("fes.dat")
plot_fel_mercator(data, "myjob", title="myjob",
                  contour_step=5.0, unsampled="max", cmap="cividis")
```

### A note on colormaps

`viridis` is the default because it is perceptually uniform and readable with colour-vision
deficiency. `cividis` is tuned specifically for deuteranopia and protanopia; `inferno` and `magma`
keep a warm "hot = high energy" reading without sacrificing either property.

Avoid green-to-red ramps such as `RdYlGn`, and avoid `jet`. Around 8 % of men cannot separate red
from green, and `jet` additionally invents boundaries that are not in the data.

---

## 7. Free energy surfaces in detail

### Input format

A whitespace-delimited table with at least three columns, in the order **θ, φ, energy**. The
`fes.dat` written by `plumed sum_hills` works directly.

**Keep the PLUMED header.** Lines like

```
#! FIELDS p.theta p.phi ff dff_p.theta dff_p.phi
#! SET max_p.theta pi
```

are read and echoed back, which is how you confirm the column order is what the program assumes.

### Angle units

PLUMED grids are usually in radians, degrees are also common. `auto` guesses, and only calls it
radians when **both** angle columns fit inside [−2π, 2π]. If your surface genuinely spans just a few
degrees the guess is wrong — pass `--angle-units deg` explicitly.

### Energy units

The program does not convert energies; it only labels them. **PLUMED writes kJ/mol by default**, so
unless your input declares `UNITS ENERGY=kcal/mol`, pass:

```bash
--energy-label "Free Energy (kJ/mol)"
```

and remember that `--contour-step` and `--energy-max` are then in kJ/mol too.

### Unsampled bins — the one option worth understanding

A converted surface marks never-visited bins with `+inf`. Two ways to draw them:

- **`mask` (default)** — left blank in a neutral grey. The colour scale is then spent entirely on
  energies that were actually measured.
- **`max`** — clamped to the highest sampled energy, so they saturate the top of the scale.

This is not cosmetic. On a real surface the unsampled bins can outnumber the sampled ones several to
one; folding them into the ramp leaves the basins almost no contrast, and — more importantly — an
unexplored region then looks exactly like a high barrier. `mask` says "no data" and means it.

Use `max` when you need to reproduce an older figure.

### Capping the scale

`--energy-max` saturates the colour scale above a threshold, giving the interesting range more of the
ramp. `auto` uses the 99th percentile of the *sampled* energies, which adapts to whatever unit and
range the file happens to be in — a fixed number only ever suits the one surface it was chosen for.

Contour lines are always drawn strictly inside the range, never on a flat plateau: a level sitting
exactly on a constant region makes the contour algorithm chase floating-point noise and scribble
thousands of tiny loops across it.

---

## 8. Recipes

**A single structure**

```bash
python -m src.cli pdb --files sugar.pdb --indices "7274 7254 7256 7260 7264 7268"
```

**A trajectory, projected onto its landscape**

```bash
python -m src.cli md --top system.prmtop --traj run.dcd \
    --indices "7277 7257 7259 7263 7267 7271" \
    --fel fes.dat --job myrun
```

**A ribose or deoxyribose (5-membered)** — just give five indices:

```bash
python -m src.cli pdb --files ribose.pdb --indices "12 8 14 18 22"
```

**A publication-style landscape in kJ/mol**

```bash
python -m src.cli fel --file fes.dat \
    --energy-label "Free Energy (kJ/mol)" --contour-step 5 --energy-max auto
```

**Comparing several structures on one map** — pass them all at once; each becomes a frame:

```bash
python -m src.cli pdb --files conf1.pdb conf2.pdb conf3.pdb --indices "..." --job compare
```

---

## 9. Troubleshooting

**`'Ex:' is not a whole number`** — the atom-index field needs only the numbers.

**`The selected atoms do not form a closed 6-membered ring`** — the indices point at atoms that are
not bonded in a cycle. Almost always a wrong index; the message names the atoms it found. Check them
against the topology you are actually using, not the PDB. Override with `--skip-ring-check` only if
you are sure the bond records are wrong.

**`Atom index/indices [...] are outside the structure`** — the index exceeds the atom count. Remember
indices are 1-based.

**`Degenerate ring: the six atoms are exactly planar`** — the selected atoms are coplanar, so no
puckering is defined. In practice this means the wrong atoms.

**`mdtraj is required for PDB and MD modes but is not installed`** — `pip install mdtraj`. The FEL
mode and the test suite work without it.

**`Could not parse '...' as a numeric table`** — the FEL file has a non-numeric row that is not
`#`-commented.

**`File must contain at least 3 columns`** — the surface needs θ, φ and energy.

**The landscape is mostly grey** — most of the surface was never sampled. That is the data, not the
plot. Confirm with the `inf` count:

```bash
python -c "import numpy as np; d=np.loadtxt('fes.dat'); print(np.isinf(d[:,2]).sum(), 'of', len(d))"
```

**The conformations look mirrored** (⁴C₁ where you expect ¹C₄, ¹S₅ where you expect ⁵S₁) — the ring
order is reversed. Check the resolved-atoms line reads O5, C1, C2, C3, C4, C5 in that order.

**The GUI freezes** — it should not; the analysis runs off the UI thread with a status line under the
Run button. If it does, the traceback appears in an error dialog rather than being swallowed.
