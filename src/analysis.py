"""
analysis.py
GUI-free analysis pipeline: input parsing, structure loading, per-frame
Cremer-Pople evaluation and tabular output.

Everything here is importable and testable without Tkinter, and without mdtraj
unless a structural mode is actually used (mdtraj is imported lazily).
"""

import os
import struct

import numpy as np

from . import furanose
from .math_core import (
    EQUATORIAL_LABELS,
    LABEL_TO_TEX,
    NORTH_LABELS,
    RING_SIZE,
    SOUTH_LABELS,
    calculate_cremer_pople,
    get_strict_conformation,
)

# Ring sizes the tool understands: 6 (pyranose) and 5 (furanose).
PYRANOSE_SIZE = RING_SIZE
FURANOSE_SIZE = furanose.RING_SIZE
RING_SIZES = (FURANOSE_SIZE, PYRANOSE_SIZE)

RING_ATOM_ORDER = {
    PYRANOSE_SIZE: ("O5", "C1", "C2", "C3", "C4", "C5"),
    FURANOSE_SIZE: furanose.ATOM_ORDER,
}


class AnalysisError(Exception):
    """Raised for invalid user input or unusable data files.

    Carries a message meant to be shown directly to the user, so the GUI and the
    CLI can render it without inspecting the exception type.
    """


# Columns of the array produced by compute_puckering(). The last two carry
# (Theta, Phi) for a pyranose and (P, nu_max) for a furanose.
COL_FRAME, COL_Q, COL_THETA, COL_PHI = range(4)
COL_P, COL_NU_MAX = COL_THETA, COL_PHI

# mdtraj works in nanometres; Cremer-Pople amplitudes are conventionally Angstrom.
NM_TO_ANGSTROM = 10.0


def parse_indices(text):
    """
    Parses the 1-based ring atom indices typed by the user.

    Six indices select a pyranose (O5, C1..C5), five a furanose (O4, C1..C4);
    the ring size is inferred from how many are given.

    Args:
        text (str): whitespace- and/or comma-separated indices.

    Returns:
        list[int]: the indices converted to 0-based, order preserved.

    Raises:
        AnalysisError: on any malformed, duplicated, out-of-range or missing index.
    """
    expected = " or ".join(str(n) for n in RING_SIZES)
    tokens = text.replace(",", " ").split()
    if not tokens:
        raise AnalysisError(
            f"No atom indices given. Enter {expected} ring atoms as 1-based indices: "
            "6 for a pyranose (O5, C1, C2, C3, C4, C5), e.g. '11 12 13 14 15 16', "
            "or 5 for a furanose (O4, C1, C2, C3, C4)."
        )

    indices = []
    for token in tokens:
        try:
            value = int(token)
        except ValueError:
            raise AnalysisError(
                f"'{token}' is not a whole number. Enter {expected} 1-based atom "
                "indices separated by spaces, e.g. '11 12 13 14 15 16'."
            ) from None
        if value < 1:
            raise AnalysisError(
                f"Atom index {value} is not valid: indices are 1-based, so the "
                "smallest allowed value is 1."
            )
        indices.append(value)

    if len(indices) not in RING_SIZES:
        raise AnalysisError(
            f"Expected exactly {expected} atom indices -- 6 for a pyranose "
            f"(O5, C1, C2, C3, C4, C5) or 5 for a furanose (O4, C1, C2, C3, C4) -- "
            f"got {len(indices)}."
        )
    if len(set(indices)) != len(indices):
        raise AnalysisError(
            f"The {len(indices)} atom indices must all be different; got {indices}."
        )

    return [i - 1 for i in indices]


def check_ring_connectivity(structure_topology, indices, atom_names):
    """
    Verifies the six chosen atoms really form a closed ring.

    Cremer-Pople coordinates are meaningless for atoms that are not a cycle, but
    the arithmetic succeeds anyway and yields plausible-looking numbers -- a wrong
    index silently produces a whole trajectory of nonsense. When the topology
    carries bonds we can catch that outright.

    Returns None when the topology has no bond information for these atoms (mdtraj
    builds PDB bonds from residue templates, so non-standard sugar residues often
    have none) -- the check is then simply unavailable, not failed.

    Raises:
        AnalysisError: if the atoms are bonded but not as a six-membered ring.
    """
    size = len(indices)
    bonded_pairs = set()
    bonded_atoms = set()
    for first, second in structure_topology.bonds:
        bonded_pairs.add(frozenset((first.index, second.index)))
        bonded_atoms.update((first.index, second.index))

    if not any(index in bonded_atoms for index in indices):
        return None

    missing = [
        f"{atom_names[k]}-{atom_names[(k + 1) % size]}"
        for k in range(size)
        if frozenset((indices[k], indices[(k + 1) % size])) not in bonded_pairs
    ]
    if missing:
        raise AnalysisError(
            f"The selected atoms do not form a closed {size}-membered ring: no bond "
            f"between {', '.join(missing)}.\n"
            f"Selected atoms were: {', '.join(atom_names)}.\n"
            "Check the indices and that they are listed in ring connectivity "
            f"order ({', '.join(RING_ATOM_ORDER[size])})."
        )
    return True


def load_ring_coordinates(mode, indices, pdb_files=None, topology=None, trajectory=None,
                          check_ring=True):
    """
    Loads the trajectory and returns only the ring atoms, in the requested order.

    mdtraj's `atom_indices=` does not promise to preserve the caller's ordering,
    and Cremer-Pople coordinates are order-sensitive (reversing the ring maps
    theta to 180-theta). We therefore pass a *sorted* selection -- for which
    "preserve the given order" and "sort by original index" are the same thing --
    and reorder the columns ourselves afterwards.

    Args:
        mode (str): "PDB" or "MD".
        indices (list[int]): six 0-based atom indices, in ring connectivity order.
        pdb_files (list[str]): PDB paths, for mode "PDB".
        topology (str), trajectory (str): paths, for mode "MD".

    Returns:
        tuple: (ring_xyz, atom_names, base_name, times_ps) where ring_xyz has
               shape (n_frames, n_ring_atoms, 3) in Angstrom, and times_ps holds
               the per-frame times or None when the trajectory does not really
               carry them.

    Raises:
        AnalysisError: on missing files, missing mdtraj, or out-of-range indices.
    """
    try:
        import mdtraj as md
    except ImportError:
        raise AnalysisError(
            "mdtraj is required for PDB and MD modes but is not installed.\n"
            "Install it with:  pip install mdtraj"
        ) from None

    sorted_indices = sorted(indices)

    if mode == "PDB":
        if not pdb_files:
            raise AnalysisError("Please select at least one PDB file.")
        missing = [p for p in pdb_files if not os.path.exists(p)]
        if missing:
            raise AnalysisError("PDB file(s) not found:\n" + "\n".join(missing))
        topology_source = pdb_files[0]
        base_name = os.path.splitext(os.path.basename(pdb_files[0]))[0]
        if len(pdb_files) > 1:
            base_name += "_multi"

    elif mode == "MD":
        for label, path in (("Topology", topology), ("Trajectory", trajectory)):
            if not path:
                raise AnalysisError(f"{label} file not selected.")
            if not os.path.exists(path):
                raise AnalysisError(f"{label} file not found:\n{path}")
        topology_source = topology
        base_name = os.path.splitext(os.path.basename(trajectory))[0]

    else:
        raise AnalysisError(f"Unknown structural mode '{mode}'.")

    # Read the topology first: it is cheap, and it lets an out-of-range index be
    # reported properly instead of surfacing as a bare IndexError from deep
    # inside mdtraj's coordinate slicing.
    try:
        structure_topology = md.load_topology(topology_source)
    except Exception as exc:
        raise AnalysisError(
            f"Could not read the topology from '{topology_source}':\n{exc}"
        ) from None

    n_atoms = structure_topology.n_atoms
    out_of_range = [i + 1 for i in indices if i >= n_atoms]
    if out_of_range:
        raise AnalysisError(
            f"Atom index/indices {out_of_range} are outside the structure, which "
            f"has {n_atoms} atoms. Indices are 1-based."
        )

    if mode == "PDB":
        traj = md.load(list(pdb_files), atom_indices=sorted_indices)
    else:
        traj = md.load(trajectory, top=topology, atom_indices=sorted_indices)

    if traj.n_atoms != len(indices):
        raise AnalysisError(
            f"Expected to load {len(indices)} ring atoms but got {traj.n_atoms}."
        )

    # Map each requested index to its position in the sorted selection.
    order = [sorted_indices.index(i) for i in indices]
    ring_xyz = traj.xyz[:, order, :] * NM_TO_ANGSTROM
    atom_names = [str(traj.topology.atom(k)) for k in order]

    if check_ring:
        check_ring_connectivity(structure_topology, indices, atom_names)

    # NetCDF, XTC and TRR carry a real per-frame time and round-trip it faithfully.
    # DCD does not: mdtraj ignores the header's DELTA/NSAVC fields and hands back
    # 0, 1, 2, ... -- frame indices wearing time units. (Those fields are also
    # routinely wrong; QM/MM writers tend to store the integration step and leave
    # NSAVC at 1 regardless of how often frames were actually saved.) So only pass
    # along times that say something a frame number does not.
    times_ps = None
    if getattr(traj, "time", None) is not None and not looks_like_frame_indices(traj.time):
        times_ps = np.asarray(traj.time, dtype=float)

    return ring_xyz, atom_names, base_name, times_ps


def compute_puckering(ring_xyz):
    """
    Evaluates the puckering coordinates for every frame.

    The ring size decides which description is used, and therefore what the last
    two columns mean:
      6 atoms (pyranose): Cremer-Pople  -> [frame, Q, theta, phi]
      5 atoms (furanose): Altona-Sundaralingam pseudorotation, with the amplitude
                          from Cremer-Pople -> [frame, Q, P, nu_max]

    Args:
        ring_xyz (numpy.ndarray): (n_frames, n_ring_atoms, 3) coordinates in Angstrom.

    Returns:
        numpy.ndarray: (n_frames, 4), frame_index 0-based.

    Raises:
        AnalysisError: if there are no frames, the ring size is unsupported, or a
                       frame has a degenerate ring.
    """
    ring_xyz = np.asarray(ring_xyz, dtype=float)
    n_frames = ring_xyz.shape[0]
    if n_frames == 0:
        raise AnalysisError("The selected structure/trajectory contains no frames.")

    ring_size = ring_xyz.shape[1]
    if ring_size not in RING_SIZES:
        raise AnalysisError(
            f"Unsupported ring size {ring_size}; expected "
            f"{' or '.join(str(n) for n in RING_SIZES)} atoms."
        )

    output = np.zeros((n_frames, 4), dtype=float)
    for frame in range(n_frames):
        try:
            if ring_size == PYRANOSE_SIZE:
                Q, second, third = calculate_cremer_pople(ring_xyz[frame])
            else:
                Q, _phi = furanose.calculate_cremer_pople_furanose(ring_xyz[frame])
                second, third = furanose.calculate_pseudorotation(ring_xyz[frame])
        except ValueError as exc:
            raise AnalysisError(f"Frame {frame + 1}: {exc}") from None
        output[frame, :] = [frame, Q, second, third]

    return output


def describe_conformation(row, ring_size):
    """Conformer label for one row of compute_puckering() output."""
    if ring_size == PYRANOSE_SIZE:
        return get_strict_conformation(row[COL_THETA], row[COL_PHI])
    return furanose.get_furanose_conformation(row[COL_P])


def conformer_tex(label):
    """
    LaTeX form of a conformer label, e.g. "1S3" -> "$^1S_3$".

    Falls back to the plain string for anything that is not a conformer, such as
    "Undefined" or the "Other" bucket of the populations chart.
    """
    return LABEL_TO_TEX.get(label, furanose.LABEL_TO_TEX.get(label, label))


def make_progress_axis(n_frames, times_ps=None, timestep_ps=None):
    """
    Builds the x-axis shared by every per-frame plot.

    Args:
        n_frames (int): number of frames.
        times_ps (numpy.ndarray): per-frame times in ps, or None.
        timestep_ps (float): ps between frames; overrides times_ps when given.

    Returns:
        tuple: (values, label). Falls back to 1-based frame numbers labelled
               "Frame" when there is no trustworthy time information -- a
               trajectory whose stored times are just 0, 1, 2, ... is reporting
               frame indices, not picoseconds, and labelling those "ps" would be
               a quiet lie. NetCDF, XTC and TRR round-trip real times and are
               used automatically; DCD does not (see load_ring_coordinates).
    """
    frames = np.arange(n_frames, dtype=float)

    if timestep_ps:
        values = frames * float(timestep_ps)
    elif times_ps is not None and len(times_ps) == n_frames:
        values = np.asarray(times_ps, dtype=float)
    else:
        return frames + 1.0, "Frame"

    if not np.all(np.isfinite(values)) or (n_frames > 1 and np.ptp(values) <= 0):
        return frames + 1.0, "Frame"

    # Nanoseconds once picoseconds would run into the thousands.
    if np.max(np.abs(values)) >= 1000.0:
        return values / 1000.0, "Time (ns)"
    return values, "Time (ps)"


def looks_like_frame_indices(times):
    """True when a trajectory's stored times are just 0, 1, 2, ... -- i.e. absent."""
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        return True
    return np.allclose(times, np.arange(times.size, dtype=float))


def conformer_order(ring_size):
    """
    Every conformer label, in the order they occupy the conformational space.

    For a pyranose that is north pole, northern band, equator, southern band,
    south pole -- so a list index behaves like a latitude and an itinerary drawn
    against it reads as a path rather than as an arbitrary reshuffling. For a
    furanose it is simply the pseudorotation wheel in order of P.
    """
    if ring_size == PYRANOSE_SIZE:
        return (["4C1"] + list(NORTH_LABELS) + list(EQUATORIAL_LABELS)
                + list(SOUTH_LABELS) + ["1C4"])
    return list(furanose.FURANOSE_LABELS)


def write_params_dat(results, path, atom_names=None, ring_size=PYRANOSE_SIZE):
    """
    Writes the per-frame parameter table.

    Metadata lines are prefixed with '#' so the numeric block stays loadable with
    numpy.loadtxt.

    Args:
        results (numpy.ndarray): (n_frames, 4) array from compute_puckering().
        path (str): destination file.
        atom_names (list[str]): resolved ring atom names, recorded in the header
                                so the selection can be checked after the fact.
        ring_size (int): 6 for a pyranose, 5 for a furanose; sets the column names.
    """
    if ring_size == PYRANOSE_SIZE:
        columns, units = ("Theta(deg)", "Phi(deg)"), "Theta and Phi in degrees"
    else:
        columns, units = ("P(deg)", "NuMax(deg)"), "P and NuMax in degrees"

    with open(path, "w") as handle:
        ring_type = "pyranose" if ring_size == PYRANOSE_SIZE else "furanose"
        handle.write(f"# {ring_size}-membered ring ({ring_type})\n")
        if atom_names:
            handle.write(f"# Ring atoms (in order): {', '.join(atom_names)}\n")
        handle.write(f"# Q in Angstrom; {units}\n")
        # The column header is commented too, so the whole file loads with a bare
        # numpy.loadtxt(). The leading '#' takes one of the 8 header columns so
        # the titles stay aligned over the data.
        handle.write(
            f"#{'Frame':>7}  {'Q(A)':>8}  {columns[0]:>12}  "
            f"{columns[1]:>12}  {'Conformation':>14}\n"
        )
        handle.write("# " + "-" * 60 + "\n")
        for row in results:
            conformation = describe_conformation(row, ring_size)
            handle.write(
                f"{int(row[COL_FRAME]) + 1:>8}  {row[COL_Q]:>8.4f}  "
                f"{row[COL_THETA]:>12.4f}  {row[COL_PHI]:>12.4f}  {conformation:>14}\n"
            )


def load_fel(path):
    """
    Loads a pre-computed free energy surface.

    Args:
        path (str): whitespace-delimited table, e.g. the fes.dat written by
                    `plumed sum_hills`. '#' comment lines are skipped.

    Returns:
        tuple: (data, field_names) where data is a 2-D array with at least three
               columns and field_names are the column names declared by a PLUMED
               '#! FIELDS ...' header, or None when the file has no such header.

    Raises:
        AnalysisError: if the file is missing, unreadable, empty or too narrow.
    """
    if not path or not os.path.exists(path):
        raise AnalysisError("Please select a valid FEL data file.")

    field_names = None
    with open(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            # PLUMED self-describing header:  #! FIELDS phi psi file.free ...
            tokens = line.lstrip("#!").split()
            if tokens and tokens[0] == "FIELDS":
                field_names = tokens[1:]
                break

    try:
        data = np.loadtxt(path)
    except ValueError as exc:
        raise AnalysisError(f"Could not parse '{path}' as a numeric table:\n{exc}") from None

    # A single data row loads as 1-D; normalise so data.shape[1] is always valid.
    data = np.atleast_2d(data)
    if data.size == 0:
        raise AnalysisError(f"'{path}' contains no data rows.")
    if data.shape[1] < 3:
        raise AnalysisError(
            "File must contain at least 3 columns: Theta, Phi, Energy "
            f"(found {data.shape[1]})."
        )

    return data, field_names


def parse_energy_max(value):
    """
    Parses the FEL colour-scale cap typed by a user.

    Accepts a number, the word "auto", or blank/None for "use the whole sampled
    range". Shared by the GUI and the CLI so both reject the same things with the
    same message.

    Raises:
        AnalysisError: on anything else.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() == "auto":
        return "auto"
    try:
        return float(text)
    except ValueError:
        raise AnalysisError(
            f"Max energy must be a number, 'auto', or left blank; got {value!r}."
        ) from None


def parse_contour_step(value, default=1.0):
    """
    Parses the contour spacing, in the energy unit of the data.

    Raises:
        AnalysisError: if it is not a positive number.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        return default
    try:
        step = float(text)
    except ValueError:
        raise AnalysisError(f"Contour spacing must be a number; got {value!r}.") from None
    if step <= 0:
        raise AnalysisError(f"Contour spacing must be greater than zero; got {step}.")
    return step


# A DCD header block is 84 bytes and starts with the magic "CORD".
DCD_HEADER_BLOCK = 84
# CHARMM stores its integration step in AKMA time units.
AKMA_TO_PS = 4.88882129e-2
# An MD integration step is physically about 0.1 to 10 fs. Used to tell an AKMA
# DELTA from one already in picoseconds, which the header flag does not settle.
PLAUSIBLE_STEP_PS = (1e-4, 1e-2)


def read_dcd_timestep(path):
    """
    Recovers the frame spacing from a DCD header.

    mdtraj does not use the DCD timing fields, but they are there: the spacing is
    DELTA * NSAVC, with DELTA in AKMA units for a CHARMM-flavoured file and
    already in picoseconds for the X-PLOR/NAMD flavour.

    Treat the result with suspicion. Many writers store the integration step and
    leave NSAVC at 1 whatever the real output frequency was, which makes the
    header describe a much shorter run than the one on disk -- so callers should
    report what they read rather than apply it silently.

    Returns:
        tuple: (timestep_ps, details) or None if the file is not a readable DCD
               or the header carries no usable spacing.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(92)
    except OSError:
        return None
    if len(head) < 92 or head[4:8] != b"CORD":
        return None

    for endian in ("<", ">"):
        if struct.unpack(endian + "i", head[0:4])[0] == DCD_HEADER_BLOCK:
            break
    else:
        return None

    control = struct.unpack(endian + "20i", head[8:88])
    nsavc = control[2] or 1
    charmm_version = control[19]

    delta_offset = 8 + 9 * 4
    if charmm_version:
        raw = struct.unpack(endian + "f", head[delta_offset:delta_offset + 4])[0]
    else:
        raw = struct.unpack(endian + "d", head[delta_offset:delta_offset + 8])[0]
    if not np.isfinite(raw) or raw <= 0.0:
        return None

    # The CHARMM version flag is supposed to say whether DELTA is in AKMA units
    # or already in picoseconds, but writers disagree: files carrying the same
    # flag have been seen storing each. Deciding by units alone therefore gets one
    # of them wrong by a factor of 20. Since an integration step is physically
    # confined to roughly 0.1-10 fs, try both readings and keep whichever lands
    # in that range.
    order = [(raw * AKMA_TO_PS, "AKMA"), (raw, "ps")]
    if not charmm_version:
        order.reverse()
    plausible = [(value, units) for value, units in order
                 if PLAUSIBLE_STEP_PS[0] <= value <= PLAUSIBLE_STEP_PS[1]]

    delta_ps, delta_units = plausible[0] if plausible else order[0]

    timestep = delta_ps * nsavc
    if not np.isfinite(timestep) or timestep <= 0.0:
        return None
    return timestep, {"delta_ps": delta_ps, "nsavc": nsavc,
                      "flavour": "CHARMM" if charmm_version else "X-PLOR/NAMD",
                      "delta_units": delta_units,
                      "step_is_plausible": bool(plausible)}


def resolve_timestep(trajectory, timestep_ps=None, n_frames=None):
    """
    Decides the frame spacing, and describes what the file claims.

    Only an explicit value is ever used. The DCD header is read and reported but
    deliberately not applied: DELTA (the integration step) is usually right, but
    NSAVC (the save frequency) is routinely left at 1 no matter how often frames
    were really written. Both reference trajectories here do exactly that -- one
    was saved every 100 steps and still claims 1 -- which makes the header
    underestimate the run by that same factor. A wrong time axis is worse than an
    honest frame-number axis, so the header only ever becomes a printed hint.

    Returns:
        tuple: (timestep_ps or None, note). The note describes the source when a
               timestep was given, or what the header claims when one was not.
    """
    if timestep_ps:
        return timestep_ps, "given by the user"

    if trajectory and str(trajectory).lower().endswith(".dcd"):
        found = read_dcd_timestep(trajectory)
        if found:
            header_step, info = found
            total = f", {header_step * n_frames:g} ps total" if n_frames else ""
            return None, (
                f"the DCD header claims {header_step:g} ps/frame "
                f"(DELTA={info['delta_ps']:.6g} ps read as {info['delta_units']}, "
                f"NSAVC={info['nsavc']}){total} -- NSAVC is often left at 1 "
                f"regardless of the real output frequency, so this is only a hint; "
                f"pass --timestep to set the axis"
            )
    return None, None


def parse_timestep(value):
    """
    Parses the picoseconds-per-frame typed by a user.

    Blank/None means "no time information": the plots then use frame numbers.

    Raises:
        AnalysisError: if it is not a positive number.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        step = float(text)
    except ValueError:
        raise AnalysisError(
            f"Timestep must be a number of picoseconds per frame; got {value!r}."
        ) from None
    if step <= 0:
        raise AnalysisError(f"Timestep must be greater than zero; got {step}.")
    return step


def prepare_output_dir(job_name, base_name, output_root="."):
    """
    Creates the job directory and returns the prefix shared by every output file.

    Args:
        job_name (str): user-supplied name; falls back to base_name when blank.
        base_name (str): name derived from the input file.
        output_root (str): directory the job folder is created in.

    Returns:
        tuple: (job_directory, path_prefix, job_label)
    """
    label = (job_name or "").strip() or base_name
    job_dir = os.path.join(output_root, label)
    os.makedirs(job_dir, exist_ok=True)
    return job_dir, os.path.join(job_dir, label), label
