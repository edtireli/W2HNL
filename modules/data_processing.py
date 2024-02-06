# Handling of data by converting to correct units (if need be) and performing the cuts as per the experimental parameters

import numpy as np
from parameters.data_parameters import *
from parameters.experimental_parameters import *
from utils.hnl import *


def nat_to_s():
    return 6.5823*10**-25

def nat_to_m():
    return 1.9733*10**-16

def light_speed():
    return 299792458

def unit_converter(initial_unit):
    """
    Converts input from mm, cm, or m to meters.
    """
    # Define conversion factors
    conversion_factors = {'mm': 1000, 'cm': 100, 'm': 1}
    
    # Extract numbers and units from the input
    number = [float(s) for s in initial_unit.split() if s.replace('.', '', 1).isdigit()]
    unit = [u for u in conversion_factors.keys() if u in initial_unit]
    
    # Validate input
    if not number or not unit:
        raise ValueError("Invalid input: Please specify the value and unit correctly (mm, cm, m).")
    
    # Perform conversion
    return number[0] / conversion_factors[unit[0]]

def invmass_rdv_efficiency(r_dv, m_dv):
    y1 = (-7/110) * r_dv * nat_to_m() * 1e3 + 180/11
    y2 = 3
    return int(m_dv >= y1 and m_dv >= y2)


class ParticleBatch:
    def __init__(self, momenta):
        self.momenta_dict = {
            'boson': momenta[0],
            'hnl': momenta[1],
            'prompt': momenta[2],
            'displaced_minus': momenta[3],
            'displaced_plus': momenta[4],
            'neutrino': momenta[5],
        }
        self.mass_hnl = mass_hnl 
        self.selected_mass_index = None
        self.selected_momenta = None
        self.current_particle_type = None

    def mass(self, mass):
        """
        Sets the mass for the particle batch. Accepts both numeric and string inputs.

        :param mass: Can be an int, float, or string representing the mass in GeV.
        :return: Self for method chaining.
        """
        # Check if mass is already a numeric type, use directly; if string, parse it
        if isinstance(mass, (int, float)):
            mass_value = mass
        elif isinstance(mass, str):
            mass_value = float(mass.split()[0])
        else:
            raise ValueError("Mass must be an int, float, or string.")

        if mass_value in self.mass_hnl:
            self.selected_mass_index = self.mass_hnl.index(mass_value)
            return self
        else:
            raise ValueError("Mass not found in the dataset.")


    def particle(self, particle_type):
        if self.selected_mass_index is None:
            raise ValueError("HNL mass must be selected before retrieving momenta.")
        if particle_type not in self.momenta_dict:
            raise ValueError(f"Particle type '{particle_type}' not recognized.")
        
        self.current_particle_type = particle_type
        self.selected_momenta = self.momenta_dict[particle_type][self.selected_mass_index]
        return self

    def get_particle_count(self):
        if self.selected_momenta is not None:
            return len(self.selected_momenta)
        else:
            return 0
    
    def E(self):
        return self.selected_momenta[:, 0]

    def px(self):
        return self.selected_momenta[:, 1]

    def py(self):
        return self.selected_momenta[:, 2]

    def pz(self):
        return self.selected_momenta[:, 3]
    
    def pT(self):
        return np.sqrt(self.px()**2 + self.py()**2)
    
    def cut_pT(self, threshold):
        threshold_value = float(threshold.split()[0])
        pT_values = self.pT()  # Calculate pT values for all particles
        survival_mask = pT_values >= threshold_value  # Boolean array indicating survival
        
        # Instead of filtering selected_momenta, return the survival_mask
        return survival_mask
    
    def find_tau(self, m_HNL, angle_sq, flavour):
        """
        Calculate the lifetime of an HNL in its rest frame.

        :param m_HNL: Mass of the HNL.
        :param angle_sq: Mixing angle squared. Can be a single value or a tuple of three values for each flavour.
        :param flavour: The flavour of neutrino (1 = electron, 2 = muon, 3 = tau).
        :return: The lifetime of the HNL.
        """
        if np.shape(angle_sq) == ():
            if flavour == 1:
                angles = [angle_sq, 0, 0]
            elif flavour == 2:
                angles = [0, angle_sq, 0]
            elif flavour == 3:
                angles = [0, 0, angle_sq]
        elif np.shape(angle_sq) == (3,):
            angles = angle_sq
        
        # Assuming HNL is a class or function you have defined elsewhere for calculating the HNL's lifetime
        tau = HNL(m_HNL, angles, False).computeNLifetime() * (1/nat_to_s())

        return tau
    
    def cut_dv(self, angle_sq, cut_type, dv_min=0, dv_max_long=0, dv_max_trans=0):
        if self.selected_mass_index is None or self.current_particle_type is None:
            raise ValueError("Both HNL mass and particle type must be selected before applying DV cut.")
        
        p_lab = self.selected_momenta
        HNLMass = self.mass_hnl[self.selected_mass_index]

        # Step 1: Compute Lorentz factor for HNLs
        g_lab = p_lab[:, 0] / HNLMass

        # Step 2: Find tau and generate decay times in the lab frame
        tau = self.find_tau(HNLMass, angle_sq, flavour_hnl)
        td = np.random.exponential(tau, size=len(p_lab))

        # Multiply decay times by Lorentz factor to get decay times in lab frame
        td_lab = g_lab * td

        # Step 3: Calculate decay positions in 3D space
        rd_lab = (p_lab[:, 1:4].T * (td_lab / p_lab[:, 0])).T
        
        # Calculate the norm of decay positions to get distances
        rd_lab_norm = np.linalg.norm(rd_lab, axis=1)

        # Create a boolean mask initially marking all particles as not surviving
        survival_mask = np.zeros(len(p_lab), dtype=bool)

        # Step 4: Apply cuts based on geometry and update the survival_mask
        if cut_type == 'sphere':
            survival_mask[(rd_lab_norm >= dv_min * nat_to_m()**-1) & (rd_lab_norm <= dv_max_long * nat_to_m()**-1)] = True
        elif cut_type == 'cylinder':
            rho = np.sqrt(rd_lab[:, 0]**2 + rd_lab[:, 1]**2)
            z = np.abs(rd_lab[:, 2])
            survival_mask[(rd_lab_norm >= dv_min * nat_to_m()**-1) & (rho <= dv_max_trans * nat_to_m()**-1) & (z <= dv_max_long * nat_to_m()**-1)] = True
        else:
            raise ValueError("Invalid cut type specified.")

        return survival_mask

    
    def cut_rap(self):
        """
        Compute a rapidity cut based on the maximum rapidity and the detector's length, and return a survival mask.

        :param rap_max: Maximum allowed rapidity.
        :param detector_length: Length of the detector along the beam axis in meters.
        :return: A boolean mask indicating survival of particles.
        """
        if self.selected_mass_index is None or self.current_particle_type is None:
            raise ValueError("Both HNL mass and particle type must be selected before applying rapidity cut.")
        
        # Convert detector length to natural units
        A = unit_converter(r_max_l) * nat_to_m()**-1  # Assuming unit_converter is already applied
        a = 2 * np.arctan(np.exp(-pseudorapidity_maximum))
        B = math.tan(a) * A  # Detector exit-hole radius
        
        p = self.selected_momenta[:, 1:4]  # px, py, pz components
        vProd = self.selected_momenta[:, 4:7] if self.selected_momenta.shape[1] == 7 else np.zeros_like(p)

        # Calculate the exit points
        z_dir = np.sign(p[:, 2])
        t = (A * z_dir - vProd[:, 2]) / p[:, 2]
        exit_points = vProd + p * t[:, np.newaxis]
        
        # Compute survival mask based on the exit hole radius
        survival_mask = (exit_points[:, 0]**2 + exit_points[:, 1]**2) > B**2

        # Return the survival mask instead of modifying selected_momenta
        return survival_mask

    
    def cut_invmass(self, invmass_threshold):
        """
        Apply a cut based on the invariant mass of the displaced_minus and displaced_plus pairs.

        :param invmass_threshold: Invariant mass threshold as a string (e.g., '5 GeV') or a number (interpreted as GeV).
        """
        # Convert invmass_threshold to a float if it's a string
        if isinstance(invmass_threshold, str):
            invmass_threshold = float(invmass_threshold.split()[0])
        
        # Ensure we have the momenta for displaced_minus and displaced_plus
        if 'displaced_minus' not in self.momenta_dict or 'displaced_plus' not in self.momenta_dict:
            raise ValueError("Both displaced_minus and displaced_plus momenta must be present.")

        # Retrieve momenta
        p_minus = self.momenta_dict['displaced_minus'][self.selected_mass_index]
        p_plus = self.momenta_dict['displaced_plus'][self.selected_mass_index]

        # Minkowski metric
        minkowski = np.diag([1, -1, -1, -1])

        # Calculate invariant mass for each pair
        p_sum = p_minus + p_plus  # Sum of 4-momenta
        invariant_masses = np.sqrt(np.einsum('ij,ij->i', np.einsum('ij,jk->ik', p_sum, minkowski), p_sum))

        # Apply the cut
        survival_mask = invariant_masses >= invmass_threshold

        return survival_mask
    
    def cut_deltaR(self, deltaR_threshold):
        """
        Apply a cut based on the angular separation (Delta R) between pairs of particles.
        
        :param deltaR_threshold: Angular separation threshold as a float.
        """
        if 'displaced_minus' not in self.momenta_dict or 'displaced_plus' not in self.momenta_dict:
            raise ValueError("Momenta for both displaced_minus and displaced_plus must be present.")
        
        p_minus = self.momenta_dict['displaced_minus'][self.selected_mass_index]
        p_plus = self.momenta_dict['displaced_plus'][self.selected_mass_index]

        # Calculate Delta Eta and Delta Phi
        delta_eta = self._calculate_delta_eta(p_minus, p_plus)
        delta_phi = self._calculate_delta_phi(p_minus, p_plus)

        # Calculate Delta R
        delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

        # Apply the cut
        survival_mask = delta_R >= deltaR_threshold

        return survival_mask

    def _calculate_delta_eta(self, p1, p2):
        """
        Calculate the difference in pseudorapidity (Delta Eta) between two particles.
        """
        eta1 = -np.log(np.tan(np.arctan2(np.linalg.norm(p1[:, 1:3], axis=1), p1[:, 3])/2))
        eta2 = -np.log(np.tan(np.arctan2(np.linalg.norm(p2[:, 1:3], axis=1), p2[:, 3])/2))
        return eta1 - eta2

    def _calculate_delta_phi(self, p1, p2):
        """
        Calculate the difference in azimuthal angle (Delta Phi) between two particles.
        """
        phi1 = np.arctan2(p1[:, 2], p1[:, 1])
        phi2 = np.arctan2(p2[:, 2], p2[:, 1])
        delta_phi = phi1 - phi2
        # Correct for the periodic boundary condition
        delta_phi = np.where(delta_phi > np.pi, delta_phi - 2*np.pi, delta_phi)
        delta_phi = np.where(delta_phi < -np.pi, delta_phi + 2*np.pi, delta_phi)
        return delta_phi
    
    def cut_invmass_nontrivial(self, angle_sq, cut_type, dv_min, dv_max_long, dv_max_trans):
        if self.selected_mass_index is None or self.current_particle_type is None:
            raise ValueError("Both HNL mass and particle type must be selected before applying non-trivial invariant mass cut.")

        p_lab = self.selected_momenta
        HNLMass = self.mass_hnl[self.selected_mass_index]
        g_lab = p_lab[:, 0] / HNLMass
        tau = self.find_tau(HNLMass, angle_sq, flavour_hnl)
        td_lab = g_lab * np.random.exponential(tau, size=len(p_lab))
        rd_lab = (p_lab[:, 1:4].T * (td_lab / p_lab[:, 0])).T
        rd_lab_norm = np.linalg.norm(rd_lab, axis=1)
        
        p_minus = self.momenta_dict['displaced_minus'][self.selected_mass_index]
        p_plus = self.momenta_dict['displaced_plus'][self.selected_mass_index]
        p_sum = p_minus + p_plus
        invariant_masses = np.sqrt(np.einsum('ij,ij->i', np.einsum('ij,jk->ik', p_sum, np.diag([1, -1, -1, -1])), p_sum))
        
        # Apply non-trivial invariant mass cut based on rd_lab_norm (distance) and invariant_masses
        survival_mask = np.array([invmass_rdv_efficiency(rd, m) for rd, m in zip(rd_lab_norm, invariant_masses)], dtype=bool)

        return survival_mask


def survival_pT(particle_type, momentum):
    # Initialize a list to hold the survival boolean arrays for each mass
    survival_bool_all_masses = []

    for mass in mass_hnl:
        batch = ParticleBatch(momentum)
        # Select the mass and particle type
        batch.mass(mass).particle(particle_type)
        # Apply the pT cut and retrieve the survival mask for all particles
        survival_mask = batch.cut_pT(pT_minimum)
        # Append the survival mask to the list
        survival_bool_all_masses.append(survival_mask)

    # Convert the list of arrays into a single numpy array for easier handling
    # This assumes that each mass has the same number of particles, resulting in a (19, 9999) array if there are 9999 particles per mass
    survival_bool_all_masses = np.array(survival_bool_all_masses)

    return survival_bool_all_masses



def survival_dv(momentum=1):
    # Initialize an array to hold the survival status for each particle across all masses and mixings
    survival_bool = np.zeros((len(mass_hnl), len(mixing), batch_size), dtype=bool)

    for i, mass in enumerate(mass_hnl):
        for j, mix in enumerate(mixing):
            batch = ParticleBatch(momentum)
            # Apply the DV cut for each mass and mixing scenario
            survival_mask = batch.mass(mass).particle('hnl').cut_dv(mix, 'sphere', unit_converter(r_min), unit_converter(r_max_l), unit_converter(r_max_t))
            survival_bool[i, j, :] = survival_mask

    return survival_bool

def survival_invmass_nontrivial(momentum=1):
    # Initialize an array to hold the survival status for each particle across all masses and mixings
    survival_bool = np.zeros((len(mass_hnl), len(mixing), batch_size), dtype=bool)

    for i, mass in enumerate(mass_hnl):
        for j, mix in enumerate(mixing):
            batch = ParticleBatch(momentum)
            # Apply the DV cut for each mass and mixing scenario
            survival_mask = batch.mass(mass).particle('hnl').cut_invmass_nontrivial(mix, 'sphere', unit_converter(r_min), unit_converter(r_max_l), unit_converter(r_max_t))
            survival_bool[i, j, :] = survival_mask

    return survival_bool


def survival_rap(particle_type, momentum):
    # Initialize a list to hold the survival boolean arrays for each mass
    survival_bool_all_masses = []

    for mass in mass_hnl:
        batch = ParticleBatch(momentum)
        # Select the mass and particle type, then apply the rapidity cut
        survival_mask = batch.mass(mass).particle(particle_type).cut_rap()
        # Append the survival mask to the list
        survival_bool_all_masses.append(survival_mask)

    # Convert the list of arrays into a single numpy array for easier handling
    survival_bool_all_masses = np.array(survival_bool_all_masses)

    return survival_bool_all_masses

def survival_invmass(cut_condition, momentum):
    # Initialize a list to hold the survival boolean arrays for each mass
    survival_bool_all_masses = []

    for mass in mass_hnl:
        batch = ParticleBatch(momentum)
        # Select the mass, apply the invariant mass cut to the 'displaced_minus' and 'displaced_plus' pair
        survival_mask = batch.mass(mass).cut_invmass(cut_condition)
        # Append the survival mask to the list
        survival_bool_all_masses.append(survival_mask)

    # Convert the list of arrays into a single numpy array
    survival_bool_all_masses = np.array(survival_bool_all_masses)

    return survival_bool_all_masses


def survival_deltaR(cut_condition, momentum):
    # Initialize a list to hold the survival boolean arrays for each mass
    survival_bool_all_masses = []

    for mass in mass_hnl:
        batch = ParticleBatch(momentum)
        # Select the mass, apply the Delta R cut to the 'displaced_minus' and 'displaced_plus' pair
        survival_mask = batch.mass(mass).cut_deltaR(cut_condition)
        # Append the survival mask to the list
        survival_bool_all_masses.append(survival_mask)

    # Convert the list of arrays into a single numpy array
    survival_bool_all_masses = np.array(survival_bool_all_masses)

    return survival_bool_all_masses


def data_processing(momenta):
    print('--------------------------- Data processing --------------------------')
    batch = ParticleBatch(momenta)
    
    print('Computing cut: Transverse momentum')
    survival_pT_displaced = survival_pT(particle_type='displaced_minus', momentum=momenta) * survival_pT(particle_type='displaced_plus', momentum=momenta)
    
    print('Computing cut: Pseudorapidity')
    survival_rap_displaced = survival_rap(particle_type='displaced_minus', momentum=momenta) * survival_rap(particle_type='displaced_plus', momentum=momenta)
    
    if invmass_cut_type == 'nontrivial':
        print('Computing cut: Invariant mass (nontrivial)')
        survival_invmass_displaced = survival_invmass_nontrivial(momentum=momenta)
    else:
        print('Computing cut: Invariant mass')
        survival_invmass_displaced = survival_invmass(invmass_minimum, momentum=momenta)
        
    print('Computing cut: Angular seperation')
    survival_deltaR_displaced = survival_deltaR(deltaR_minimum, momentum=momenta)
    
    print('Computing cut: Displaced vertex')
    survival_dv_displaced = survival_dv(momentum=momenta)
    
    arrays = (np.array(survival_dv_displaced), np.array(survival_pT_displaced), 
              np.array(survival_rap_displaced), np.array(survival_invmass_displaced), 
              np.array(survival_deltaR_displaced), 
     ) # defining a tuple for easier management of survival arrays on main
    return batch, arrays