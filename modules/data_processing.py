# Handling of data by converting to correct units (if need be) and performing the cuts as per the experimental parameters

import numpy as np
from parameters.data_parameters import *
from parameters.experimental_parameters import *
from utils.hnl import *
import copy


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
        # Deep copy the momenta array to ensure independence
        self.momenta_dict = {
            'boson': copy.deepcopy(momenta[0]),
            'hnl': copy.deepcopy(momenta[1]),
            'prompt': copy.deepcopy(momenta[2]),
            'displaced_minus': copy.deepcopy(momenta[3]),
            'displaced_plus': copy.deepcopy(momenta[4]),
            'neutrino': copy.deepcopy(momenta[5]),
        }
        self.mass_hnl = mass_hnl 
        self.selected_mass_index = None
        self.selected_momenta = None
        self.current_particle_type = None
        

    def mass(self, mass):
        # This method should remain largely unchanged but ensure mass_hnl is correctly initialized
        if isinstance(mass, (int, float)):
            mass_value = mass
        elif isinstance(mass, str):
            mass_value = float(mass.split()[0])
        else:
            raise ValueError("Mass must be an int, float, or string.")

        if mass_value in self.mass_hnl:
            self.selected_mass_index = self.mass_hnl.index(mass_value)
        else:
            raise ValueError("Mass not found in the dataset.")
        return self


    def particle(self, particle_type):
        # This method is crucial for selecting the particle type and its momenta based on the previously set mass
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
    
    def eta(self):
        """
        Calculate the pseudorapidity of the selected particles.

        :return: A numpy array containing the pseudorapidity values of the selected particles.
        """
        if self.selected_momenta is None:
            raise ValueError("Particle type and mass must be selected before computing pseudorapidity.")
        pz = self.pz()  # Assuming self.pz() returns the z-component of momentum
        pt = self.pT()  # Assuming self.pT() returns the transverse momentum
        # Avoid division by zero or log of zero by adding a small number epsilon
        epsilon = 1e-9
        eta = -np.log(np.tan(np.arctan2(pt, pz) / 2) + epsilon)
        return eta
    
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
        tau = HNL(m_HNL, angles, False).computeNLifetime()

        return tau
    
    def cut_dv(self, angle_sq, cut_type, dv_min=0, dv_max_long=0, dv_max_trans=0):
        if self.selected_mass_index is None or self.current_particle_type is None:
            raise ValueError("Both HNL mass and particle type must be selected before applying DV cut.")
        
        p_lab = self.selected_momenta
        HNLMass = self.mass_hnl[self.selected_mass_index]

        # Step 1: Compute Lorentz factor for HNLs
        g_lab = p_lab[:, 0] / HNLMass

        # Step 2: Find tau and generate decay times in the lab frame
        tau_ = self.find_tau(HNLMass, [0,0,1], flavour_hnl)
        tau_ = tau_ / angle_sq
        td = np.random.exponential(tau_, size=len(p_lab))
        # Multiply decay times by Lorentz factor to get decay times in lab frame
        td_lab = g_lab * td

        decay_length_lab = td_lab * light_speed() 
        
        # Step 3: Calculate decay positions in 3D space
        rd_lab = (p_lab[:, 1:4].T * (td_lab / p_lab[:, 0])).T
        
        # Calculate the norm of decay positions to get distances
        rd_lab_norm = np.linalg.norm(rd_lab, axis=1)

        # Create a boolean mask initially marking all particles as not surviving
        survival_mask_dv = np.zeros(len(p_lab), dtype=bool)

        # Step 4: Apply cuts based on geometry and update the survival_mask
        if cut_type == 'sphere':
            survival_mask_dv[(decay_length_lab >= dv_min) & (decay_length_lab <= dv_max_long)] = True
        elif cut_type == 'cylinder':
            rho = np.sqrt(rd_lab[:, 0]**2 + rd_lab[:, 1]**2)
            z = np.abs(rd_lab[:, 2])
            survival_mask_dv[(rd_lab_norm * light_speed() >= dv_min) & (rho <= dv_max_trans) & (z * light_speed() <= dv_max_long)] = True
        else:
            raise ValueError("Invalid cut type specified.")

        return survival_mask_dv, rd_lab, td, g_lab
    
    def cut_dv_2(self, angle_sq, tau, cut_type, dv_min=0, dv_max_long=0, dv_max_trans=0):
        if self.selected_mass_index is None or self.current_particle_type is None:
            raise ValueError("Both HNL mass and particle type must be selected before applying DV cut.")
        
        p_lab = self.selected_momenta
        HNLMass = self.mass_hnl[self.selected_mass_index]

        # Step 1: Compute Lorentz factor for HNLs
        g_lab = p_lab[:, 0] / HNLMass

        # Multiply decay times by Lorentz factor to get decay times in lab frame
        td_lab = g_lab * tau

        decay_length_lab = td_lab * light_speed() 
        
        # Step 3: Calculate decay positions in 3D space
        rd_lab = (p_lab[:, 1:4].T * (td_lab / p_lab[:, 0])).T
        
        # Calculate the norm of decay positions to get distances
        rd_lab_norm = np.linalg.norm(rd_lab, axis=1)

        # Create a boolean mask initially marking all particles as not surviving
        survival_mask_dv = np.zeros(len(p_lab), dtype=bool)

        # Step 4: Apply cuts based on geometry and update the survival_mask
        if cut_type == 'sphere':
            survival_mask_dv[(decay_length_lab >= dv_min) & (decay_length_lab <= dv_max_long)] = True
        elif cut_type == 'cylinder':
            rho = np.sqrt(rd_lab[:, 0]**2 + rd_lab[:, 1]**2)
            z = np.abs(rd_lab[:, 2])
            survival_mask_dv[(rd_lab_norm * light_speed() >= dv_min) & (rho <= dv_max_trans) & (z * light_speed() <= dv_max_long)] = True
        else:
            raise ValueError("Invalid cut type specified.")

        return survival_mask_dv, rd_lab, tau, g_lab

    
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
        survival_mask_rap = (exit_points[:, 0]**2 + exit_points[:, 1]**2) > B**2

        # Return the survival mask instead of modifying selected_momenta
        return survival_mask_rap

    def phi(self):
        """
        Calculate the azimuthal angle of the selected particles.

        :return: A numpy array containing the azimuthal angles of the selected particles.
        """
        return np.arctan2(self.py(), self.px())
    
    def cut_invmass(self, invmass_threshold, experimental=False):
        """
        Apply a cut based on the invariant mass of the displaced_minus and displaced_plus pairs and return a survival mask,
        with an option to use the experimental calculation method.

        :param invmass_threshold: Invariant mass threshold as a string (e.g., '5 GeV') or a number (interpreted as GeV).
        :param experimental: Boolean flag to use the experimental method for invariant mass calculation.
        :return: A boolean numpy array (mask) indicating which particles survive the cut.
        """
        # Convert invmass_threshold to a float if it's a string
        invmass_threshold = float(invmass_threshold.split()[0]) if isinstance(invmass_threshold, str) else invmass_threshold
        
        if not experimental:
            # Original method
            # Ensure we have the momenta for displaced_minus and displaced_plus
            if 'displaced_minus' not in self.momenta_dict or 'displaced_plus' not in self.momenta_dict:
                raise ValueError("Both displaced_minus and displaced_plus momenta must be present.")

            # Retrieve momenta
            p_minus = self.momenta_dict['displaced_minus'][self.selected_mass_index]
            p_plus = self.momenta_dict['displaced_plus'][self.selected_mass_index]

            # Minkowski metric for invariant mass calculation
            minkowski_metric = np.diag([1, -1, -1, -1])

            # Calculate invariant mass for each pair
            p_sum = p_minus + p_plus  # Element-wise sum of 4-momenta vectors
            invariant_masses = np.sqrt(np.einsum('ij,ij->i', np.einsum('ij,jk->ik', p_sum, minkowski_metric), p_sum))
        else:
            # Experimental method
            # Ensure we have the necessary particle data
            if 'displaced_minus' not in self.momenta_dict or 'displaced_plus' not in self.momenta_dict:
                raise ValueError("Both displaced_minus and displaced_plus data must be present for experimental calculation.")

            # Retrieve transverse momentum, pseudorapidity, and azimuthal angles
            self.particle('displaced_minus')
            eta1 = self.eta()
            phi1 = self.phi()
            pT1 = self.pT()

            self.particle('displaced_plus')
            eta2 = self.eta()
            phi2 = self.phi()
            pT2 = self.pT()

            # Calculate invariant mass squared for each pair
            m_squared = 2 * pT1 * pT2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2))

            # Take the square root of m_squared to get the invariant mass
            invariant_masses = np.sqrt(np.maximum(m_squared, 0))  # Ensure non-negative under sqrt

        # Apply the cut and generate a survival mask
        survival_mask_invmass = invariant_masses >= invmass_threshold

        return survival_mask_invmass
    
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
        survival_mask_deltaR = delta_R >= deltaR_threshold

        return survival_mask_deltaR

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
        survival_mask_invmass_nt = np.array([invmass_rdv_efficiency(rd, m) for rd, m in zip(rd_lab_norm, invariant_masses)], dtype=bool)

        return survival_mask_invmass_nt
    
    def __deepcopy__(self, memo):
        # Create a new instance without calling __init__
        cls = self.__class__
        new_instance = cls.__new__(cls)
        memo[id(self)] = new_instance
        
        # Copy each attribute using deepcopy for deep copy behavior
        for k, v in self.__dict__.items():
            setattr(new_instance, k, copy.deepcopy(v, memo))
        
        return new_instance


def survival_pT(particle_type, momentum):
    # Initialize a list to hold the survival boolean arrays for each mass
    survival_bool_all_masses_pT = []
    momentum_pT = copy.deepcopy(momentum)
    for mass in mass_hnl:
        original_batch = ParticleBatch(momentum_pT)
        # Make a deep copy of the batch for this particular cut application
        batch_pT = copy.deepcopy(original_batch)
        # Select the mass and particle type
        batch_pT.mass(mass).particle(particle_type)
        # Apply the pT cut and retrieve the survival mask for all particles
        survival_mask_pT = batch_pT.cut_pT(pT_minimum)
        # Append the survival mask to the list
        survival_bool_all_masses_pT.append(survival_mask_pT)

    # Convert the list of arrays into a single numpy array for easier handling
    # This assumes that each mass has the same number of particles, resulting in a (19, 9999) array if there are 9999 particles per mass
    survival_bool_all_masses_pT = np.array(survival_bool_all_masses_pT)

    return survival_bool_all_masses_pT


def survival_dv(momentum=1, rng_type=1):
    if rng_type == 1: 
        momentum_dv = copy.deepcopy(momentum)
        # Initialize arrays to hold the survival status and decay vertices for each particle across all masses and mixings
        survival_bool_dv = np.zeros((len(mass_hnl), len(mixing), batch_size), dtype=bool)
        rd_labs = np.zeros((len(mass_hnl), len(mixing), batch_size, 3))  # rd_lab is a 3D vector
        lifetimes_rest = np.zeros((len(mass_hnl), len(mixing), batch_size))
        lorentz_factors = np.zeros((len(mass_hnl), len(mixing), batch_size))
        for i, mass in enumerate(mass_hnl):
            for j, mix in enumerate(mixing):
                original_batch = ParticleBatch(momentum_dv)
                # Make a deep copy of the batch for this particular cut application
                batch_dv = copy.deepcopy(original_batch)
                # Apply the DV cut for each mass and mixing scenario and get decay vertices
                survival_mask_dv, rd_lab, td, g_lab = batch_dv.mass(mass).particle('hnl').cut_dv(mix, cut_type_dv, unit_converter(r_min), unit_converter(r_max_l), unit_converter(r_max_t))
                survival_bool_dv[i, j, :] = survival_mask_dv
                rd_labs[i, j, :, :] = rd_lab  # Store the decay vertices
                lifetimes_rest[i, j, :] = td
                lorentz_factors[i, j, :] = g_lab
    elif rng_type==2:
        momentum_dv = copy.deepcopy(momentum)
        # Initialize arrays to hold the survival status and decay vertices for each particle across all masses and mixings
        survival_bool_dv = np.zeros((len(mass_hnl), len(mixing), batch_size), dtype=bool)
        rd_labs = np.zeros((len(mass_hnl), len(mixing), batch_size, 3))  # rd_lab is a 3D vector
        lifetimes_rest = np.zeros((len(mass_hnl), len(mixing), batch_size))
        lorentz_factors = np.zeros((len(mass_hnl), len(mixing), batch_size))
        for i, mass in enumerate(mass_hnl):
            lifetime = HNL(mass, [0,0,1], False).computeNLifetime()
            lifetime_spread = np.random.exponential(lifetime, size=(batch_size))
            for j, mix in enumerate(mixing):
                lifetime_spread_mixing = lifetime_spread / mix
                original_batch = ParticleBatch(momentum_dv)
                # Make a deep copy of the batch for this particular cut application
                batch_dv = copy.deepcopy(original_batch)
                # Apply the DV cut for each mass and mixing scenario and get decay vertices
                survival_mask_dv, rd_lab, td, g_lab = batch_dv.mass(mass).particle('hnl').cut_dv_2(mix, lifetime_spread_mixing, cut_type_dv, unit_converter(r_min), unit_converter(r_max_l), unit_converter(r_max_t))
                survival_bool_dv[i, j, :] = survival_mask_dv
                rd_labs[i, j, :, :] = rd_lab  # Store the decay vertices
                lifetimes_rest[i, j, :] = td
                lorentz_factors[i, j, :] = g_lab                
    return survival_bool_dv, rd_labs,lifetimes_rest, lorentz_factors


def survival_invmass_nontrivial(momentum=1):
    # Initialize an array to hold the survival status for each particle across all masses and mixings
    survival_bool_invmass_nt = np.zeros((len(mass_hnl), len(mixing), batch_size), dtype=bool)
    momentum_invmass_nt = copy.deepcopy(momentum)

    for i, mass in enumerate(mass_hnl):
        for j, mix in enumerate(mixing):
            original_batch = ParticleBatch(momentum_invmass_nt)
            # Make a deep copy of the batch for this particular cut application
            batch_invmass_nt = copy.deepcopy(original_batch)
            # Apply the DV cut for each mass and mixing scenario
            survival_mask_invmass_nt = batch_invmass_nt.mass(mass).particle('hnl').cut_invmass_nontrivial(mix, 'sphere', unit_converter(r_min), unit_converter(r_max_l), unit_converter(r_max_t))
            survival_bool_invmass_nt[i, j, :] = survival_mask_invmass_nt

    return survival_bool_invmass_nt


def survival_rap(particle_type, momentum):
    # Initialize a list to hold the survival boolean arrays for each mass
    survival_bool_all_masses_rap = []
    momentum_rap = copy.deepcopy(momentum)

    for mass in mass_hnl:
        original_batch = ParticleBatch(momentum_rap)
        # Make a deep copy of the batch for this particular cut application
        batch_rap = copy.deepcopy(original_batch)
        # Select the mass and particle type, then apply the rapidity cut
        survival_mask = batch_rap.mass(mass).particle(particle_type).cut_rap()
        # Append the survival mask to the list
        survival_bool_all_masses_rap.append(survival_mask)

    # Convert the list of arrays into a single numpy array for easier handling
    survival_bool_all_masses_rap = np.array(survival_bool_all_masses_rap)

    return survival_bool_all_masses_rap

def survival_invmass(cut_condition, momentum, experimental_trigger = False):
    # Initialize a list to hold the survival boolean arrays for each mass
    survival_bool_all_masses_invmass = []
    momentum_invmass = copy.deepcopy(momentum)
    for mass in mass_hnl:
        original_batch = ParticleBatch(momentum_invmass)
        # Make a deep copy of the batch for this particular cut application
        batch_invmass = copy.deepcopy(original_batch)
        # Select the mass, apply the invariant mass cut to the 'displaced_minus' and 'displaced_plus' pair
        survival_mask_invmass = batch_invmass.mass(mass).cut_invmass(cut_condition, experimental=experimental_trigger)
        # Append the survival mask to the list
        survival_bool_all_masses_invmass.append(survival_mask_invmass)

    # Convert the list of arrays into a single numpy array
    survival_bool_all_masses_invmass = np.array(survival_bool_all_masses_invmass)

    return survival_bool_all_masses_invmass


def survival_deltaR(cut_condition, momentum):
    # Initialize a list to hold the survival boolean arrays for each mass
    survival_bool_all_masses_deltaR = []
    momentum_deltaR = copy.deepcopy(momentum)
    for mass in mass_hnl:
        original_batch = ParticleBatch(momentum_deltaR)
        # Make a deep copy of the batch for this particular cut application
        batch_deltaR = copy.deepcopy(original_batch)
        # Select the mass, apply the Delta R cut to the 'displaced_minus' and 'displaced_plus' pair
        survival_mask = batch_deltaR.mass(mass).cut_deltaR(cut_condition)
        # Append the survival mask to the list
        survival_bool_all_masses_deltaR.append(survival_mask)

    # Convert the list of arrays into a single numpy array
    survival_bool_all_masses_deltaR = np.array(survival_bool_all_masses_deltaR)

    return survival_bool_all_masses_deltaR

def save_array(array, name=''):
    current_directory = os.getcwd()                                         # Current path
    data_path         = os.path.join(current_directory,'data', data_folder) # Data folder path
    array_path         = os.path.join(data_path, 'Plots', 'Plot data', f'{name}.npy')
    os.makedirs(os.path.join(data_path, 'Plots'), exist_ok=True) # Making directory if not already exists

    np.save(array_path, array)


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
        survival_invmass_displaced = survival_invmass(invmass_minimum, momentum=momenta, experimental_trigger=invmass_experimental)
        
    print('Computing cut: Angular seperation')
    survival_deltaR_displaced = survival_deltaR(deltaR_minimum, momentum=momenta)
    
    print('Computing cut: Displaced vertex')
    survival_dv_displaced, r_lab, lifetimes_rest_, lorentz_factors = survival_dv(momentum=momenta, rng_type=1)
    print(np.mean(lifetimes_rest_))
    arrays = (np.array(survival_dv_displaced), np.array(survival_pT_displaced), 
              np.array(survival_rap_displaced), np.array(survival_invmass_displaced), 
              np.array(survival_deltaR_displaced), 
              np.array(r_lab), np.array(lifetimes_rest_), np.array(lorentz_factors)
     ) # defining a tuple for easier management of survival arrays on main
    save_array(survival_dv_displaced)
    return batch, arrays