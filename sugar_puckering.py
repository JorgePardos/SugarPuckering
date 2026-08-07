"""
sugar_puckering.py
Main GUI application for Sugar Puckering Analyzer.

Thin Tkinter layer over src/analysis.py: it collects paths and options, runs the
analysis off the UI thread, and reports the result. All computation lives in
src/analysis.py so it can be tested and scripted without a display
(see `python -m src.cli --help`).
"""

import os
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib

matplotlib.use("TkAgg")  # must precede the plotting import

from src.analysis import (  # noqa: E402
    AnalysisError,
    PYRANOSE_SIZE,
    compute_puckering,
    describe_conformation,
    describe_frame_range,
    load_fel,
    load_ring_coordinates,
    make_progress_axis,
    parse_contour_step,
    parse_energy_max,
    parse_frame_number,
    parse_indices,
    parse_timestep,
    prepare_output_dir,
    resolve_timestep,
    write_params_dat,
)
from src.plotting import (  # noqa: E402
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


class PuckeringApp:
    def __init__(self, root):
        """Initializes the Tkinter main window and GUI layout."""
        self.root = root
        self.root.title("Sugar Puckering Analyzer")
        self.root.geometry("620x760")

        # Internal variables
        self.mode_var = tk.StringVar(value="PDB")
        self.single_pdb_path = tk.StringVar()
        self.pdb_files_list = []
        # Text shown in the entry when several PDBs were picked at once. The
        # entry cannot hold N paths, so this marker is how we tell "use the
        # remembered list" apart from "the user typed a single path".
        self.multi_select_marker = None
        self.top_path = tk.StringVar()
        self.traj_path = tk.StringVar()
        self.fel_path = tk.StringVar()
        self.job_name_var = tk.StringVar()
        self.status_var = tk.StringVar(value="")
        self.is_running = False

        # Plot options. Defaults deliberately mirror the CLI defaults, so the two
        # front ends produce the same figure from the same inputs.
        self.angle_units_var = tk.StringVar(value="auto")
        self.energy_label_var = tk.StringVar(value="Free Energy (kcal/mol)")
        self.contour_step_var = tk.StringVar(value="1")
        self.cmap_var = tk.StringVar(value="viridis")
        self.energy_max_var = tk.StringVar(value="")
        self.unsampled_var = tk.StringVar(value="mask")
        self.overlay_fel_path = tk.StringVar()
        self.skip_ring_check_var = tk.BooleanVar(value=False)
        self.timestep_var = tk.StringVar(value="")
        self.start_var = tk.StringVar(value="")
        self.stop_var = tk.StringVar(value="")
        self.stride_var = tk.StringVar(value="")

        # Header
        tk.Label(root, text="Conformational Analysis",
                 font=("Helvetica", 14, "bold")).pack(pady=10)

        # Mode Selectors
        frame_mode = tk.Frame(root)
        frame_mode.pack(pady=5)
        for text, value in (("Static PDB(s)", "PDB"),
                            ("MD Trajectory", "MD"),
                            ("Free Energy (FEL)", "FEL")):
            tk.Radiobutton(frame_mode, text=text, variable=self.mode_var,
                           value=value, command=self.update_gui).pack(side="left", padx=10)

        # Dynamic File Selection Frame
        self.frame_files = tk.Frame(root)
        self.frame_files.pack(pady=10, fill="x", padx=20)

        # Atom Indices Frame (Hidden during FEL mode).
        # The example lives in the hint label, never as prefilled entry text: a
        # placeholder inside the entry would be read back as real input by anyone
        # who never touches the field.
        self.frame_idx = tk.Frame(root)
        tk.Label(self.frame_idx, text="Ring Atom Indices:").pack(anchor="w")
        tk.Label(self.frame_idx,
                 text="* 6 for a pyranose (O5, C1, C2, C3, C4, C5)  ·  "
                      "5 for a furanose (O4, C1, C2, C3, C4)",
                 fg="gray", font=("Helvetica", 9, "italic")).pack(anchor="w")
        tk.Label(self.frame_idx,
                 text="* 1-Based indexing, in ring order.  Example:  11 12 13 14 15 16",
                 fg="gray", font=("Helvetica", 9, "italic")).pack(anchor="w")
        self.entry_idx = tk.Entry(self.frame_idx, width=40)
        self.entry_idx.pack(anchor="w", pady=5)

        # Plot options, rebuilt per mode by update_gui()
        self.frame_options = tk.LabelFrame(root, text="Plot options", padx=10, pady=6)

        # Output Folder / Job Name Frame
        self.frame_output = tk.Frame(root)
        self.frame_output.pack(pady=15, fill="x", padx=20)
        tk.Label(self.frame_output,
                 text="Output Folder / Job Name (leave blank for default):").pack(anchor="w")
        tk.Entry(self.frame_output, textvariable=self.job_name_var,
                 width=40).pack(anchor="w", pady=5)

        # Execute Button + status line
        self.run_button = tk.Button(root, text="Run Analysis", bg="darkblue", fg="white",
                                    font=("Helvetica", 12, "bold"), command=self.run_analysis)
        self.run_button.pack(pady=10)
        tk.Label(root, textvariable=self.status_var, fg="gray").pack()

        # Initialize default view
        self.update_gui()

    def update_gui(self):
        """Updates the input fields based on the selected mode (PDB/MD/FEL)."""
        for widget in self.frame_files.winfo_children():
            widget.destroy()

        # Leaving a stale multi-file selection behind would silently override a
        # path typed after switching modes.
        self.pdb_files_list = []
        self.multi_select_marker = None
        self.single_pdb_path.set("")
        self.status_var.set("")

        mode = self.mode_var.get()

        if mode == "PDB":
            self.frame_idx.pack(before=self.frame_output, fill="x", padx=20)
            self._add_file_row(0, "PDB File(s):", self.single_pdb_path, "pdb")

        elif mode == "MD":
            self.frame_idx.pack(before=self.frame_output, fill="x", padx=20)
            self._add_file_row(0, "Topology (.prmtop, .pdb):", self.top_path, "top")
            self._add_file_row(1, "Trajectory (.nc, .dcd...):", self.traj_path, "traj")

        elif mode == "FEL":
            self.frame_idx.pack_forget()  # indices are not used by a pre-computed FEL
            self._add_file_row(0, "FEL Data (.dat, .txt):", self.fel_path, "fel")

        self._build_options(mode)

    def _add_file_row(self, row, label, variable, file_type):
        tk.Label(self.frame_files, text=label).grid(row=row, column=0, sticky="w", pady=5)
        tk.Entry(self.frame_files, textvariable=variable, width=35).grid(row=row, column=1, padx=5)
        tk.Button(self.frame_files, text="Browse",
                  command=lambda: self.browse_file(variable, file_type)).grid(row=row, column=2)

    # -- plot options ------------------------------------------------------

    def _build_options(self, mode):
        """
        Rebuilds the options panel for the current mode.

        Everything here has a command-line equivalent; the flag name is shown in
        the tooltip-style hint so a GUI run can be reproduced from a script.
        """
        for widget in self.frame_options.winfo_children():
            widget.destroy()
        self.frame_options.pack(before=self.frame_output, fill="x", padx=20, pady=(0, 4))

        row = 0
        if mode == "FEL":
            row = self._add_fel_options(row)
        else:
            tk.Checkbutton(self.frame_options,
                           text="Skip the closed-ring check  (--skip-ring-check)",
                           variable=self.skip_ring_check_var).grid(
                row=row, column=0, columnspan=3, sticky="w")
            row += 1
            row = self._add_entry(row, "Timestep (ps/frame):", self.timestep_var,
                                  "--timestep   (blank = x axis in frames)", width=10)
            row = self._add_frame_range(row)

            tk.Label(self.frame_options, text="Project onto FEL (optional):").grid(
                row=row, column=0, sticky="w", pady=2)
            tk.Entry(self.frame_options, textvariable=self.overlay_fel_path,
                     width=24).grid(row=row, column=1, sticky="w", padx=4)
            tk.Button(self.frame_options, text="Browse",
                      command=lambda: self.browse_file(self.overlay_fel_path, "fel")).grid(
                row=row, column=2, sticky="w")
            row += 1
            # Those two describe the projected surface, so they only matter here
            # once a surface has been chosen.
            row = self._add_option_menu(row, "Angle units:", self.angle_units_var,
                                        ["auto", "deg", "rad"], "--angle-units")
            row = self._add_entry(row, "Energy label:", self.energy_label_var,
                                  "--energy-label", width=26)

    def _add_fel_options(self, row):
        row = self._add_option_menu(row, "Angle units:", self.angle_units_var,
                                    ["auto", "deg", "rad"], "--angle-units")
        row = self._add_option_menu(row, "Unsampled bins:", self.unsampled_var,
                                    ["mask", "max"], "--unsampled")
        row = self._add_option_menu(row, "Colormap:", self.cmap_var,
                                    ["viridis", "cividis", "inferno", "magma", "plasma"],
                                    "--cmap")
        row = self._add_entry(row, "Contour spacing:", self.contour_step_var,
                              "--contour-step", width=10)
        row = self._add_entry(row, "Max energy:", self.energy_max_var,
                              "--energy-max   (blank = full range, or 'auto')", width=10)
        row = self._add_entry(row, "Energy label:", self.energy_label_var,
                              "--energy-label", width=26)
        return row

    def _add_frame_range(self, row):
        """One compact row for the frame window; blank fields mean the whole run."""
        tk.Label(self.frame_options, text="Frames:").grid(row=row, column=0, sticky="w",
                                                          pady=2)
        holder = tk.Frame(self.frame_options)
        holder.grid(row=row, column=1, sticky="w", padx=4)
        for label, variable in (("from", self.start_var), ("to", self.stop_var),
                                ("every", self.stride_var)):
            tk.Label(holder, text=label).pack(side="left")
            tk.Entry(holder, textvariable=variable, width=6).pack(side="left", padx=(2, 6))
        tk.Label(self.frame_options, text="--start / --stop / --stride   (blank = all)",
                 fg="gray", font=("Helvetica", 8, "italic")).grid(row=row, column=2,
                                                                  sticky="w")
        return row + 1

    def _frame_range(self):
        """Validated frame window, as keyword arguments for load_ring_coordinates()."""
        return {
            "start": parse_frame_number(self.start_var.get(), "Start frame"),
            "stop": parse_frame_number(self.stop_var.get(), "Stop frame"),
            "stride": parse_frame_number(self.stride_var.get(), "Stride"),
        }

    def _add_option_menu(self, row, label, variable, choices, flag):
        tk.Label(self.frame_options, text=label).grid(row=row, column=0, sticky="w", pady=2)
        tk.OptionMenu(self.frame_options, variable, *choices).grid(
            row=row, column=1, sticky="w", padx=4)
        tk.Label(self.frame_options, text=flag, fg="gray",
                 font=("Helvetica", 8, "italic")).grid(row=row, column=2, sticky="w")
        return row + 1

    def _add_entry(self, row, label, variable, flag, width=12):
        tk.Label(self.frame_options, text=label).grid(row=row, column=0, sticky="w", pady=2)
        tk.Entry(self.frame_options, textvariable=variable, width=width).grid(
            row=row, column=1, sticky="w", padx=4)
        tk.Label(self.frame_options, text=flag, fg="gray",
                 font=("Helvetica", 8, "italic")).grid(row=row, column=2, sticky="w")
        return row + 1

    def _plot_options(self):
        """Validated plot options, as keyword arguments for plot_fel_mercator()."""
        return {
            "angle_units": self.angle_units_var.get(),
            "energy_label": self.energy_label_var.get(),
            "contour_step": parse_contour_step(self.contour_step_var.get()),
            "cmap": self.cmap_var.get(),
            "energy_max": parse_energy_max(self.energy_max_var.get()),
            "unsampled": self.unsampled_var.get(),
        }

    def browse_file(self, var_to_update, file_type):
        """Handles native OS file explorer dialogs."""
        if file_type == "pdb":
            filenames = filedialog.askopenfilenames(
                filetypes=[("PDB", "*.pdb"), ("All files", "*.*")])
            if filenames:
                self.pdb_files_list = list(filenames)
                if len(filenames) == 1:
                    self.multi_select_marker = None
                    var_to_update.set(filenames[0])
                else:
                    self.multi_select_marker = f"{len(filenames)} files selected"
                    var_to_update.set(self.multi_select_marker)
            return

        filters = {
            "top": [("Topology", "*.prmtop *.pdb *.parm7")],
            "traj": [("Trajectory", "*.nc *.dcd *.xtc *.trr *.crd")],
            "fel": [("Data files", "*.dat *.txt *.out")],
        }[file_type]
        filename = filedialog.askopenfilename(filetypes=filters + [("All files", "*.*")])
        if filename:
            var_to_update.set(filename)

    def _selected_pdb_files(self):
        """Resolves the PDB selection, preferring a hand-typed path over the list."""
        text = self.single_pdb_path.get().strip()
        if self.pdb_files_list and text == self.multi_select_marker:
            return self.pdb_files_list
        if text:
            return [text]
        raise AnalysisError("Please select at least one PDB file.")

    # -- execution ---------------------------------------------------------

    def run_analysis(self):
        """Validates input on the UI thread, then runs the analysis off it."""
        if self.is_running:
            return

        mode = self.mode_var.get()
        try:
            # Validate the options here, on the UI thread, so a typo in a text
            # field is reported before any file is read.
            self._plot_options()
            parse_timestep(self.timestep_var.get())
            self._frame_range()
            if mode == "FEL":
                job = ("FEL", {"path": self.fel_path.get().strip()})
            else:
                indices = parse_indices(self.entry_idx.get())
                params = {"indices": indices,
                          "check_ring": not self.skip_ring_check_var.get(),
                          **self._frame_range()}
                if mode == "PDB":
                    params["pdb_files"] = self._selected_pdb_files()
                else:
                    params["topology"] = self.top_path.get().strip()
                    params["trajectory"] = self.traj_path.get().strip()
                job = (mode, params)
        except AnalysisError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self._set_running(True, "Loading...")
        threading.Thread(target=self._worker, args=job, daemon=True).start()

    def _set_running(self, running, status=""):
        self.is_running = running
        self.status_var.set(status)
        self.run_button.config(state="disabled" if running else "normal")

    def _worker(self, mode, params):
        """Runs off the UI thread. Only computation and file I/O happen here;
        plotting is handed back to the main thread, as matplotlib is not
        thread-safe."""
        try:
            job_name = self.job_name_var.get()
            if mode == "FEL":
                data, _fields = load_fel(params["path"])
                base_name = os.path.splitext(os.path.basename(params["path"]))[0]
                _dir, prefix, label = prepare_output_dir(job_name, base_name)
                payload = {"mode": "FEL", "data": data, "prefix": prefix, "label": label}
            else:
                ring_size = len(params["indices"])
                selection = load_ring_coordinates(mode, **params)
                atom_names = selection.atom_names
                base_name = selection.base_name
                self.root.after(0, self.status_var.set,
                                f"Computing {len(selection.frames)} frame(s)...")
                results = compute_puckering(selection.xyz, selection.frames)
                frame_range = describe_frame_range(selection.frames,
                                                   selection.total_frames)
                timestep, _note = resolve_timestep(
                    params.get("trajectory"), parse_timestep(self.timestep_var.get()),
                    len(results))
                progress, progress_label = make_progress_axis(
                    selection.frames, selection.times_ps, timestep)
                _dir, prefix, label = prepare_output_dir(job_name, base_name)
                write_params_dat(results, f"{prefix}_params.dat", atom_names,
                                 ring_size, frame_range)
                payload = {"mode": mode, "results": results, "prefix": prefix,
                           "label": label, "atom_names": atom_names,
                           "ring_size": ring_size, "overlay_surface": None,
                           "axis": {"progress": progress,
                                    "progress_label": progress_label}}

                overlay_path = self.overlay_fel_path.get().strip()
                if overlay_path and ring_size == PYRANOSE_SIZE:
                    payload["overlay_surface"] = load_fel(overlay_path)[0]
        except AnalysisError as exc:
            self.root.after(0, self._on_failure, str(exc))
            return
        except Exception:
            self.root.after(0, self._on_failure, traceback.format_exc())
            return

        self.root.after(0, self._on_computed, payload)

    def _on_computed(self, payload):
        """Draws the plots, then reports success -- in that order, so a plotting
        failure cannot be preceded by a 'Success' dialog."""
        try:
            self.status_var.set("Plotting...")
            label, prefix = payload["label"], payload["prefix"]
            options = self._plot_options()

            if payload["mode"] == "FEL":
                plot_fel_mercator(payload["data"], prefix, title=label, **options)
                summary = f"FEL plot generated in folder '{label}'."
            else:
                results = payload["results"]
                ring_size = payload["ring_size"]
                axis = payload["axis"]
                if ring_size == PYRANOSE_SIZE:
                    if len(results) > 1:
                        plot_time_series(results, prefix, title=label, **axis)
                    plot_mercator(results, prefix, title=label, **axis)
                    plot_stoddart(results, prefix, title=label, **axis)
                    ring_type = "pyranose"
                    if payload["overlay_surface"] is not None:
                        plot_fel_mercator(payload["overlay_surface"], prefix,
                                          title=label, overlay=results,
                                          suffix="_on_FEL", **options)
                else:
                    if len(results) > 1:
                        plot_furanose_time_series(results, prefix, title=label, **axis)
                    plot_pseudorotation_wheel(results, prefix, title=label)
                    ring_type = "furanose"

                if len(results) > 1:
                    plot_conformer_timeline(results, prefix, ring_size, title=label, **axis)
                    plot_conformer_populations(results, prefix, ring_size, title=label)
                    plot_amplitude_histogram(results, prefix, title=label)

                atoms = ", ".join(payload["atom_names"])
                header = f"{ring_size}-membered ring ({ring_type})\nRing atoms: {atoms}"
                if len(results) == 1:
                    conformation = describe_conformation(results[0], ring_size)
                    summary = (f"Analysis complete.\n{header}\n"
                               f"Conformation: {conformation}")
                else:
                    summary = (f"Trajectory processed ({len(results)} frames).\n"
                               f"{header}")
                summary += f"\n\nResults saved in folder '{label}'."
        except Exception:
            self._on_failure(traceback.format_exc())
            return

        self._set_running(False, "Done.")
        messagebox.showinfo("Success", summary)

    def _on_failure(self, message):
        self._set_running(False, "Failed.")
        messagebox.showerror("Execution Error", message)


if __name__ == "__main__":
    root = tk.Tk()
    app = PuckeringApp(root)
    root.mainloop()
