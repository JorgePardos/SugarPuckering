"""
math_core.py
Core mathematical operations for calculating Cremer-Pople puckering coordinates.

Theoretical Foundation & Reference:
Ardèvol, A., Biarnés, X., Planas, A., & Rovira, C. (2010). 
The Conformational Free-Energy Landscape of beta-D-Mannopyranose: 
Evidence for a 1S5 -> B2,5 -> OS2 Catalytic Itinerary in beta-Mannosidases. 
Journal of the American Chemical Society, 132(45), 16058-16065.
"""

import numpy as np

# ==============================================================================
# REFERENCE DATA (Stoddart Diagram Ideal Coordinates)
# ==============================================================================
# Ideal Phi and Theta values for projection (Mercator/Stoddart)
IDEAL_PHI = np.array([180, 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 
                      0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 
                      0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 180])

IDEAL_THETA = np.array([0, 54.7, 50.8, 54.7, 50.8, 54.7, 50.8, 54.7, 50.8, 54.7, 50.8, 
                        54.7, 50.8, 54.7, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 
                        90, 90, 125.3, 129.2, 125.3, 129.2, 125.3, 129.2, 125.3, 129.2, 
                        125.3, 129.2, 125.3, 129.2, 125.3, 180])

def get_strict_conformation(theta, phi):
    """
    Assigns the sugar ring conformation using strict IUPAC angular boundaries.
    
    Args:
        theta (float): Cremer-Pople Theta angle in degrees (0 to 180).
        phi (float): Cremer-Pople Phi angle in degrees (0 to 360).
        
    Returns:
        tuple: (Conformation_Label (str), 0.0 (float, for backward compatibility))
    """
    # Ordered labels for 30-degree slices starting at phi = 0
    equatorial_lbl = ["3,OB", "3S1", "B1,4", "5S1", "2,5B", "2SO", 
                      "B3,O", "1S3", "1,4B", "1S5", "B2,5", "OS2"]
    north_lbl      = ["0E",   "0H1", "E1",   "2H1", "2E",   "2H3", 
                      "E3",   "4H3", "4E",   "4H5", "E5",   "0H5"]
    south_lbl      = ["3E",   "3H4", "E4",   "5H4", "5E",   "5HO", 
                      "E0",   "1H0", "1E",   "1H2", "E2",   "3H2"]

    # 1. Chairs are strictly defined by Theta boundaries
    if theta <= 15.0:
        return "4C1", 0.0
    elif theta >= 165.0:
        return "1C4", 0.0
        
    # 2. Slice the 360-degree Phi space into 12 sectors of 30 degrees.
    # Adding 15 degrees shifts the boundary so the ideal value is in the center.
    sector_phi = int(((phi + 15.0) % 360) // 30)
    
    # 3. Assign specific label based on latitude (Theta)
    if 75.0 <= theta <= 105.0:
        return equatorial_lbl[sector_phi], 0.0
    elif 15.0 < theta < 75.0:
        return north_lbl[sector_phi], 0.0
    elif 105.0 < theta < 165.0:
        return south_lbl[sector_phi], 0.0
    
    return "Unknown", 0.0

def calculate_cremer_pople_mdtraj(ring_coords):
    """
    Calculates the Cremer-Pople puckering parameters (Q, Theta, Phi).
    
    Args:
        ring_coords (numpy.ndarray): 6x3 array containing the 3D coordinates 
                                     of the 6 ring atoms (O5, C1, C2, C3, C4, C5).
                                     
    Returns:
        tuple: (Q_amplitude, Theta_degrees, Phi_degrees)
    """
    atoms_ring = 6
    
    # Translate geometric center to origin
    center = np.mean(ring_coords, axis=0)
    g_coord = ring_coords - center
    
    # Calculate principal axes
    vect_x = np.zeros(3)
    vect_y = np.zeros(3)
    for j in range(atoms_ring):
        vect_x += g_coord[j, :] * np.sin(2.0 * np.pi * j / atoms_ring)
        vect_y += g_coord[j, :] * np.cos(2.0 * np.pi * j / atoms_ring)
    
    vect_x /= np.linalg.norm(vect_x)
    vect_y /= np.linalg.norm(vect_y)
    
    # Calculate Z vector (normal to mean plane) and atomic displacements
    vect_z = np.cross(vect_x, vect_y)
    displ_z = np.dot(g_coord, vect_z)
    
    # Summation for q2 and q3 parameters
    sum1 = np.sum([displ_z[j] * np.cos(4.0 * np.pi * j / atoms_ring) for j in range(atoms_ring)])
    sum2 = np.sum([-displ_z[j] * np.sin(4.0 * np.pi * j / atoms_ring) for j in range(atoms_ring)])
    
    q2 = np.sqrt(1.0/3.0) * np.sqrt(sum1**2 + sum2**2)
    q3 = np.sqrt(1.0/6.0) * (displ_z[0] - displ_z[1] + displ_z[2] - displ_z[3] + displ_z[4] - displ_z[5])
    
    # Calculate Phi phase angle
    fi2 = np.arcsin(sum2 / np.sqrt(sum1**2 + sum2**2))
    if sum1 < 0:
        fi2 = np.pi - fi2
    elif fi2 < 0:
        fi2 = 2.0 * np.pi + fi2
    phi = np.degrees(fi2)
    
    # Calculate Total Amplitude (Q) and Theta
    Q = np.sqrt(q2**2 + q3**2)
    theta = np.degrees(np.arccos(q3 / Q))
    
    return Q, theta, phi