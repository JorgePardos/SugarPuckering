"""
cli.py
Headless command line interface. Run with:  python -m src.cli <mode> [options]

Useful for scripting and for reproducing a GUI run without a display.
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window

from .analysis import (  # noqa: E402  (must follow the backend selection)
    AnalysisError,
    PYRANOSE_SIZE,
    compute_puckering,
    describe_conformation,
    load_fel,
    load_ring_coordinates,
    make_progress_axis,
    parse_contour_step,
    parse_energy_max,
    parse_indices,
    prepare_output_dir,
    write_params_dat,
)
from .plotting import (  # noqa: E402
    plot_amplitude_histogram,
    plot_conformer_populations,
    plot_conformer_timeline,
    plot_fel_mercator,
    plot_furanose_time_series,
    plot_mercator,
    plot_pseudorotation_wheel,
    plot_stoddart,
    plot_time_series,
)


def _run_structural(args, mode):
    indices = parse_indices(args.indices)
    ring_xyz, atom_names, base_name, times_ps = load_ring_coordinates(
        mode,
        indices,
        pdb_files=getattr(args, "files", None),
        topology=getattr(args, "top", None),
        trajectory=getattr(args, "traj", None),
        check_ring=not args.skip_ring_check,
    )
    ring_size = len(indices)
    ring_type = "pyranose" if ring_size == PYRANOSE_SIZE else "furanose"
    print(f"{ring_size}-membered ring ({ring_type})")
    print(f"Ring atoms resolved to: {', '.join(atom_names)}")

    results = compute_puckering(ring_xyz)
    progress, progress_label = make_progress_axis(len(results), times_ps, args.timestep)
    print(f"Progress axis: {progress_label}")
    axis = {"progress": progress, "progress_label": progress_label}
    job_dir, prefix, label = prepare_output_dir(args.job, base_name, args.outdir)

    dat_path = f"{prefix}_params.dat"
    write_params_dat(results, dat_path, atom_names, ring_size)
    print(f"Wrote {dat_path}  ({len(results)} frame(s))")

    written = []
    if ring_size == PYRANOSE_SIZE:
        if len(results) > 1:
            written.append(plot_time_series(results, prefix, title=label, **axis))
        written.append(plot_mercator(results, prefix, title=label, **axis))
        written.append(plot_stoddart(results, prefix, title=label, **axis))
        if args.fel:
            fel_data, _ = load_fel(args.fel)
            written.append(plot_fel_mercator(
                fel_data, prefix, title=label, overlay=results,
                angle_units=args.angle_units, energy_label=args.energy_label,
                suffix="_on_FEL"))
    else:
        if len(results) > 1:
            written.append(plot_furanose_time_series(results, prefix, title=label, **axis))
        written.append(plot_pseudorotation_wheel(results, prefix, title=label))

    # Summaries only say something once there is a distribution to summarise.
    if len(results) > 1:
        written.append(plot_conformer_timeline(results, prefix, ring_size,
                                               title=label, **axis))
        written.append(plot_conformer_populations(results, prefix, ring_size, title=label))
        written.append(plot_amplitude_histogram(results, prefix, title=label))

    for path in written:
        print(f"Wrote {path}")

    if len(results) == 1:
        print(f"Conformation: {describe_conformation(results[0], ring_size)}")
    return 0


def _run_fel(args):
    data, field_names = load_fel(args.file)
    if field_names:
        print(f"PLUMED header fields: {', '.join(field_names)}")
    print(f"Loaded {data.shape[0]} rows x {data.shape[1]} columns from {args.file}")

    base_name = args.file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
    job_dir, prefix, label = prepare_output_dir(args.job, base_name, args.outdir)

    path = plot_fel_mercator(
        data,
        prefix,
        title=label,
        angle_units=args.angle_units,
        energy_label=args.energy_label,
        contour_step=parse_contour_step(args.contour_step),
        cmap=args.cmap,
        energy_max=parse_energy_max(args.energy_max),
        unsampled=args.unsampled,
    )
    print(f"Wrote {path}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Cremer-Pople conformational analysis of 6-membered sugar rings.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common(p):
        p.add_argument("--job", default="", help="output folder / job name")
        p.add_argument("--outdir", default=".", help="where the job folder is created")

    indices_help = (
        "the six ring atoms as 1-based indices in connectivity order "
        "(O5 C1 C2 C3 C4 C5), e.g. '11 12 13 14 15 16'"
    )

    def add_ring_check(p):
        p.add_argument("--skip-ring-check", action="store_true",
                       help="do not verify the six atoms are bonded as a ring")
        # Projecting a structure onto its own landscape shows which part of the
        # conformational space the simulation actually visited.
        p.add_argument("--fel", default=None,
                       help="also project the result onto this free energy surface "
                            "(pyranoses only)")
        p.add_argument("--angle-units", default="auto", choices=["auto", "deg", "rad"],
                       help="angle units of the --fel file")
        p.add_argument("--energy-label", default="Free Energy (kcal/mol)",
                       help="colourbar label for the --fel file")
        # Trajectory files routinely store 0, 1, 2, ... as their times, so the
        # spacing has to be stated rather than guessed.
        p.add_argument("--timestep", type=float, default=None,
                       help="picoseconds between frames; switches the time axes "
                            "from frame number to simulation time")

    p_pdb = sub.add_parser("pdb", help="analyse one or more static PDB structures")
    p_pdb.add_argument("--files", nargs="+", required=True, help="PDB file(s)")
    p_pdb.add_argument("--indices", required=True, help=indices_help)
    add_ring_check(p_pdb)
    add_common(p_pdb)

    p_md = sub.add_parser("md", help="analyse an MD trajectory")
    p_md.add_argument("--top", required=True, help="topology (.prmtop, .parm7, .pdb)")
    p_md.add_argument("--traj", required=True, help="trajectory (.nc, .dcd, .xtc, ...)")
    p_md.add_argument("--indices", required=True, help=indices_help)
    add_ring_check(p_md)
    add_common(p_md)

    p_fel = sub.add_parser("fel", help="render a pre-computed free energy landscape")
    p_fel.add_argument("--file", required=True, help="FEL table, e.g. plumed's fes.dat")
    p_fel.add_argument("--angle-units", default="auto", choices=["auto", "deg", "rad"])
    p_fel.add_argument("--energy-label", default="Free Energy (kcal/mol)",
                       help="colourbar label; PLUMED writes kJ/mol by default")
    p_fel.add_argument("--contour-step", type=float, default=1.0,
                       help="spacing of labelled contour lines, in the data's energy "
                            "unit (default 1, as in the published landscapes)")
    p_fel.add_argument("--cmap", default="viridis", help="matplotlib colormap")
    p_fel.add_argument("--energy-max", default=None,
                       help="saturate the colour scale above this energy; a number, "
                            "or 'auto' for the 99th percentile of sampled energies")
    p_fel.add_argument("--unsampled", default="mask", choices=["mask", "max"],
                       help="draw never-visited bins blank (default) or clamped to "
                            "the top of the colour scale")
    add_common(p_fel)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "fel":
            return _run_fel(args)
        return _run_structural(args, args.mode.upper())
    except AnalysisError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
