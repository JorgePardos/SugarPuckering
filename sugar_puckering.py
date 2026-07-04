"""
sugar_puckering.py
Main GUI application for Sugar Puckering Analyzer.
Bridges the mathematical engine and the visualization tools using Tkinter.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import mdtraj as md
import os

# Import local modules from the 'src' directory
from src.math_core import calculate_cremer_pople_mdtraj, get_strict_conformation
from src.plotting import plot_mercator, plot_time_series, plot_fel_mercator

class PuckeringApp:
    def __init__(self, root):
        """Initializes the Tkinter main window and GUI layout."""
        self.root = root
        self.root.title("Sugar Puckering Analyzer")
        self.root.geometry("550x550") # Slightly taller to fit the new field
        
        # Internal variables
        self.mode_var = tk.StringVar(value="PDB")
        self.single_pdb_path = tk.StringVar()
        self.pdb_files_list = [] 
        self.top_path = tk.StringVar()
        self.traj_path = tk.StringVar()
        self.fel_path = tk.StringVar() 
        self.job_name_var = tk.StringVar() # New variable for Output Folder Name
        
        # Header
        tk.Label(root, text="Conformational Analysis", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        # Mode Selectors
        frame_mode = tk.Frame(root)
        frame_mode.pack(pady=5)
        tk.Radiobutton(frame_mode, text="Static PDB(s)", variable=self.mode_var, value="PDB", command=self.update_gui).pack(side="left", padx=10)
        tk.Radiobutton(frame_mode, text="MD Trajectory", variable=self.mode_var, value="MD", command=self.update_gui).pack(side="left", padx=10)
        tk.Radiobutton(frame_mode, text="Free Energy (FEL)", variable=self.mode_var, value="FEL", command=self.update_gui).pack(side="left", padx=10)
        
        # Dynamic File Selection Frame
        self.frame_files = tk.Frame(root)
        self.frame_files.pack(pady=10, fill="x", padx=20)
        
        # Atom Indices Frame (Hidden during FEL mode)
        self.frame_idx = tk.Frame(root)
        tk.Label(self.frame_idx, text="Atom Indices (O5, C1, C2, C3, C4, C5):").pack(anchor="w")
        tk.Label(self.frame_idx, text="* 1-Based indexing (as seen in PDB/Topology files)", fg="gray", font=("Helvetica", 9, "italic")).pack(anchor="w")
        self.entry_idx = tk.Entry(self.frame_idx, width=40)
        self.entry_idx.pack(anchor="w", pady=5)
        self.entry_idx.insert(0, "Ex: 11 12 13 14 15 16")

        # Output Folder / Job Name Frame
        self.frame_output = tk.Frame(root)
        self.frame_output.pack(pady=15, fill="x", padx=20)
        tk.Label(self.frame_output, text="Output Folder / Job Name (leave blank for default):").pack(anchor="w")
        self.entry_job = tk.Entry(self.frame_output, textvariable=self.job_name_var, width=40)
        self.entry_job.pack(anchor="w", pady=5)
        
        # Execute Button
        tk.Button(root, text="Run Analysis", bg="darkblue", fg="white", font=("Helvetica", 12, "bold"), command=self.run_analysis).pack(pady=10)
        
        # Initialize default view
        self.update_gui()

    def update_gui(self):
        """Updates the input fields based on the selected mode (PDB/MD/FEL)."""
        # Clear existing file frames
        for widget in self.frame_files.winfo_children():
            widget.destroy()
            
        mode = self.mode_var.get()
        
        if mode == "PDB":
            self.frame_idx.pack(before=self.frame_output, fill="x", padx=20) 
            tk.Label(self.frame_files, text="PDB File(s):").grid(row=0, column=0, sticky="w", pady=5)
            tk.Entry(self.frame_files, textvariable=self.single_pdb_path, width=35).grid(row=0, column=1, padx=5)
            tk.Button(self.frame_files, text="Browse", command=lambda: self.browse_file(self.single_pdb_path, "pdb")).grid(row=0, column=2)
            
        elif mode == "MD":
            self.frame_idx.pack(before=self.frame_output, fill="x", padx=20) 
            tk.Label(self.frame_files, text="Topology (.prmtop, .pdb):").grid(row=0, column=0, sticky="w", pady=5)
            tk.Entry(self.frame_files, textvariable=self.top_path, width=35).grid(row=0, column=1, padx=5)
            tk.Button(self.frame_files, text="Browse", command=lambda: self.browse_file(self.top_path, "top")).grid(row=0, column=2)
            
            tk.Label(self.frame_files, text="Trajectory (.nc, .dcd...):").grid(row=1, column=0, sticky="w", pady=5)
            tk.Entry(self.frame_files, textvariable=self.traj_path, width=35).grid(row=1, column=1, padx=5)
            tk.Button(self.frame_files, text="Browse", command=lambda: self.browse_file(self.traj_path, "traj")).grid(row=1, column=2)
            
        elif mode == "FEL":
            self.frame_idx.pack_forget() # Indices are not required for pre-calculated FEL
            tk.Label(self.frame_files, text="FEL Data (.dat, .txt):").grid(row=0, column=0, sticky="w", pady=5)
            tk.Entry(self.frame_files, textvariable=self.fel_path, width=35).grid(row=0, column=1, padx=5)
            tk.Button(self.frame_files, text="Browse", command=lambda: self.browse_file(self.fel_path, "fel")).grid(row=0, column=2)

    def browse_file(self, var_to_update, file_type):
        """Handles native OS file explorer dialogs."""
        if file_type == "pdb":
            filenames = filedialog.askopenfilenames(filetypes=[("PDB", "*.pdb"), ("All files", "*.*")])
            if filenames:
                self.pdb_files_list = list(filenames)
                if len(filenames) == 1: var_to_update.set(filenames[0])
                else: var_to_update.set(f"{len(filenames)} files selected")
        elif file_type == "top":
            filename = filedialog.askopenfilename(filetypes=[("Topology", "*.prmtop *.pdb *.parm7"), ("All files", "*.*")])
            if filename: var_to_update.set(filename)
        elif file_type == "traj":
            filename = filedialog.askopenfilename(filetypes=[("Trajectory", "*.nc *.dcd *.xtc *.trr *.crd"), ("All files", "*.*")])
            if filename: var_to_update.set(filename)
        elif file_type == "fel":
            filename = filedialog.askopenfilename(filetypes=[("Data files", "*.dat *.txt *.out"), ("All files", "*.*")])
            if filename: var_to_update.set(filename)

    def run_analysis(self):
        """Main execution block triggered by the Run Analysis button."""
        mode = self.mode_var.get()
        
        try:
            # === FREE ENERGY LANDSCAPE MODE ===
            if mode == "FEL":
                fel_file = self.fel_path.get()
                if not os.path.exists(fel_file):
                    messagebox.showerror("Error", "Please select a valid FEL data file.")
                    return
                data = np.loadtxt(fel_file)
                if data.shape[1] < 3:
                    messagebox.showerror("Format Error", "File must contain at least 3 columns: Theta, Phi, Energy")
                    return
                
                # Setup output names
                base_name = os.path.splitext(os.path.basename(fel_file))[0]
                user_job_name = self.job_name_var.get().strip()
                final_job_name = user_job_name if user_job_name else base_name
                
                os.makedirs(final_job_name, exist_ok=True)
                save_prefix = os.path.join(final_job_name, final_job_name)
                
                plot_fel_mercator(data, save_prefix)
                messagebox.showinfo("Success", f"FEL plot successfully generated in folder '{final_job_name}'")
                return

            # === STRUCTURAL MODES (PDB & MD) ===
            idx_str = self.entry_idx.get()
            raw_indices = idx_str.replace(',', ' ').split()
            pdb_indices = [int(i) for i in raw_indices if i.isdigit()]
            
            if len(pdb_indices) != 6:
                messagebox.showerror("Error", "You must provide exactly 6 integer indices (O5, C1, C2, C3, C4, C5).")
                return
                
            # Convert 1-based PDB input to 0-based MDTraj internal indexing
            mdtraj_indices = [i - 1 for i in pdb_indices]
            
            # Load appropriate file structures
            if mode == "PDB":
                entry_val = self.single_pdb_path.get()
                if os.path.exists(entry_val) and len(self.pdb_files_list) <= 1:
                    self.pdb_files_list = [entry_val]
                if not self.pdb_files_list:
                    messagebox.showerror("Error", "Please select at least one PDB file.")
                    return
                
                print(f"Loading {len(self.pdb_files_list)} PDB file(s)...")
                sugar_traj = md.load(self.pdb_files_list, atom_indices=mdtraj_indices)
                
                base_name = os.path.splitext(os.path.basename(self.pdb_files_list[0]))[0]
                if len(self.pdb_files_list) > 1: 
                    base_name += "_multi"

            elif mode == "MD":
                top_file = self.top_path.get()
                traj_file = self.traj_path.get()
                if not os.path.exists(top_file) or not os.path.exists(traj_file):
                    messagebox.showerror("Error", "Invalid file paths.")
                    return
                
                print(f"Loading trajectory {traj_file}...")
                sugar_traj = md.load(traj_file, top=top_file, atom_indices=mdtraj_indices)
                base_name = os.path.splitext(os.path.basename(traj_file))[0]
            
            # Setup output names and directories
            user_job_name = self.job_name_var.get().strip()
            final_job_name = user_job_name if user_job_name else base_name
            
            os.makedirs(final_job_name, exist_ok=True)
            save_prefix = os.path.join(final_job_name, final_job_name)
            
            # Trajectory calculations
            output_data = np.zeros((sugar_traj.n_frames, 4), dtype=float)
            for frame in range(sugar_traj.n_frames):
                # MDtraj uses nanometers, converting to Angstroms (* 10)
                ring_coords = sugar_traj.xyz[frame, :, :] * 10.0 
                Q, theta, phi = calculate_cremer_pople_mdtraj(ring_coords)
                output_data[frame, :] = [frame, Q, theta, phi]
            
            # Write tabulated parameters to output .dat file
            dat_filename = f"{save_prefix}_params.dat"
            with open(dat_filename, "w") as dat_file:
                dat_file.write(f"{'Frame':>8}  {'Q(A)':>8}  {'Theta(deg)':>12}  {'Phi(deg)':>12}  {'Conformation':>14}\n")
                dat_file.write("-" * 62 + "\n")
                for i in range(sugar_traj.n_frames):
                    frame_val = int(output_data[i, 0]) + 1
                    q_val = output_data[i, 1]
                    theta_val = output_data[i, 2]
                    phi_val = output_data[i, 3]
                    
                    # Using strict IUPAC assignments
                    conf_texto, _ = get_strict_conformation(theta_val, phi_val)
                    dat_file.write(f"{frame_val:>8}  {q_val:>8.4f}  {theta_val:>12.4f}  {phi_val:>12.4f}  {conf_texto:>14}\n")

            # Provide UI Feedback
            if sugar_traj.n_frames == 1:
                msg = f"Analysis complete.\nData saved in folder '{final_job_name}'\n\nConformation: {conf_texto}"
                messagebox.showinfo("Success", msg)
            else:
                msg = f"Trajectory processed ({sugar_traj.n_frames} frames).\nData saved in folder '{final_job_name}'.\nPlots will launch next."
                messagebox.showinfo("Success", msg)
            
            # Trigger plotting functions
            if sugar_traj.n_frames > 1:
                plot_time_series(output_data, save_prefix)
            plot_mercator(output_data, save_prefix)
            
        except Exception as e:
            messagebox.showerror("Execution Error", f"An error occurred:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PuckeringApp(root)
    root.mainloop()