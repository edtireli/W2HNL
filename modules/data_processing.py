# Handling of data by converting to correct units (if need be) and performing the cuts as per the experimental parameters

import numpy as np
from parameters.data_parameters import *
from parameters.experimental_parameters import *
from utils.hnl import *
from utils.random_w2hnl import string_to_seed
import copy
from tqdm import tqdm
import shutil

def nat_to_s():
    return 6.5823*10**-25

def nat_to_m():
    return 1.9733*10**-16

def light_speed():
    return 299792458


def _closest_index(values, target):
    values_arr = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(values_arr - float(target))))


def atlas_track_reco_eff_d0(d0_mm):
    """ATLAS-inspired track reconstruction efficiency vs |d0| in mm."""
    xp = np.asarray(atlas_track_reco_d0_points_mm, dtype=float)
    fp = np.asarray(atlas_track_reco_d0_eff, dtype=float)
    if xp.size == 0:
        raise ValueError("atlas_track_reco_d0_points_mm is empty")
    if xp.shape != fp.shape:
        raise ValueError("atlas_track_reco_d0_points_mm and atlas_track_reco_d0_eff must have same length")
    if np.any(np.diff(xp) <= 0):
        raise ValueError("atlas_track_reco_d0_points_mm must be strictly increasing")

    d0 = np.asarray(d0_mm, dtype=float)
    d0 = np.maximum(np.abs(d0), 1e-6)
    d0 = np.clip(d0, xp[0], xp[-1])
    out = np.interp(np.log10(d0), np.log10(xp), fp, left=fp[0], right=fp[-1])
    return np.clip(out, 0.0, 1.0)


def atlas_track_reco_eff_rprod(rprod_mm):
    """ATLAS-inspired track reconstruction efficiency vs R_prod in mm."""
    xp = np.asarray(atlas_track_reco_rprod_points_mm, dtype=float)
    fp = np.asarray(atlas_track_reco_rprod_eff, dtype=float)
    if xp.size == 0:
        raise ValueError("atlas_track_reco_rprod_points_mm is empty")
    if xp.shape != fp.shape:
        raise ValueError("atlas_track_reco_rprod_points_mm and atlas_track_reco_rprod_eff must have same length")
    if np.any(np.diff(xp) < 0):
        raise ValueError("atlas_track_reco_rprod_points_mm must be non-decreasing")

    r = np.asarray(rprod_mm, dtype=float)
    r = np.clip(r, xp[0], xp[-1])
    out = np.interp(r, xp, fp, left=fp[0], right=fp[-1])
    return np.clip(out, 0.0, 1.0)


def survival_atlas_track_reco(momentum, r_labs):
    """Compute per-event survival mask for ATLAS track reconstruction efficiency.

    Returns a boolean array with shape (masses, mixings, batch_size).
    Also saves validation arrays (for one representative mass/mixing).
    """
    if not apply_atlas_track_reco:
        return None
    if abs(pid_boson) != 24:
        return None
    if large_data:
        raise ValueError("apply_atlas_track_reco requires large_data=False (needs r_labs to compute d0 and R_prod)")

    array_name = "survival_atlas_trackreco"
    loaded = load_cut_array_(array_name)
    if loaded is not None:
        print('[Loaded cut] ATLAS track reco')
        return loaded.astype(bool)

    # Dedicated RNG so results don't depend on whether other cuts were loaded from disk.
    rng = np.random.default_rng(string_to_seed(str(rng_seed) + "_atlas_trackreco"))

    idx_mass_val = _closest_index(mass_hnl, atlas_track_reco_validation_mass_GeV)
    idx_mix_val = _closest_index(mixing, atlas_track_reco_validation_mixing)

    survival_bool = np.zeros((len(mass_hnl), len(mixing), batch_size), dtype=bool)

    # For validation plots (only for one mass/mix point)
    val = {
        'd0_minus_mm': None,
        'd0_plus_mm': None,
        'rprod_mm': None,
        'mask_d0_minus': None,
        'mask_d0_plus': None,
        'mask_r_minus': None,
        'mask_r_plus': None,
        'mask_track_minus': None,
        'mask_track_plus': None,
    }

    c_mm_per_s = light_speed() * 1e3

    momentum_local = copy.deepcopy(momentum)
    for i, mass in tqdm(enumerate(mass_hnl), total=len(mass_hnl), desc="[Computing cut] ATLAS track reco  "):
        original_batch = ParticleBatch(momentum_local)
        batch_minus = copy.deepcopy(original_batch).mass(mass).particle('displaced_minus')
        batch_plus = copy.deepcopy(original_batch).mass(mass).particle('displaced_plus')

        px_minus = batch_minus.px()
        py_minus = batch_minus.py()
        pt_minus = batch_minus.pT()

        px_plus = batch_plus.px()
        py_plus = batch_plus.py()
        pt_plus = batch_plus.pT()

        for j in range(len(mixing)):
            rd = r_labs[i, j]  # shape (batch_size, 3), stored in seconds
            x_mm = rd[:, 0] * c_mm_per_s
            y_mm = rd[:, 1] * c_mm_per_s

            rprod_mm = np.sqrt(x_mm**2 + y_mm**2)

            # d0 = |r x p_T| / |p_T|
            with np.errstate(divide='ignore', invalid='ignore'):
                d0_minus = np.abs(x_mm * py_minus - y_mm * px_minus) / np.maximum(pt_minus, 1e-12)
                d0_plus = np.abs(x_mm * py_plus - y_mm * px_plus) / np.maximum(pt_plus, 1e-12)

            p_d0_minus = np.clip(atlas_track_reco_eff_d0(d0_minus), 0.0, 1.0)
            p_d0_plus = np.clip(atlas_track_reco_eff_d0(d0_plus), 0.0, 1.0)
            p_r = np.clip(atlas_track_reco_eff_rprod(rprod_mm), 0.0, 1.0)

            # Independent survival decisions for d0 and R per track
            u_d0_minus = rng.random(batch_size)
            u_d0_plus = rng.random(batch_size)
            u_r_minus = rng.random(batch_size)
            u_r_plus = rng.random(batch_size)

            mask_d0_minus = u_d0_minus < p_d0_minus
            mask_d0_plus = u_d0_plus < p_d0_plus
            mask_r_minus = u_r_minus < p_r
            mask_r_plus = u_r_plus < p_r

            mask_track_minus = mask_d0_minus & mask_r_minus
            mask_track_plus = mask_d0_plus & mask_r_plus

            survival_bool[i, j, :] = mask_track_minus & mask_track_plus

            if i == idx_mass_val and j == idx_mix_val:
                val['d0_minus_mm'] = d0_minus.astype(float)
                val['d0_plus_mm'] = d0_plus.astype(float)
                val['rprod_mm'] = rprod_mm.astype(float)
                val['mask_d0_minus'] = mask_d0_minus
                val['mask_d0_plus'] = mask_d0_plus
                val['mask_r_minus'] = mask_r_minus
                val['mask_r_plus'] = mask_r_plus
                val['mask_track_minus'] = mask_track_minus
                val['mask_track_plus'] = mask_track_plus

    # Save survival mask
    save_cut_array1(survival_bool, array_name)

    # Save validation arrays if captured
    if val['d0_minus_mm'] is not None:
        save_array(val['d0_minus_mm'], 'atlas_trackreco_val_d0_minus_mm')
        save_array(val['d0_plus_mm'], 'atlas_trackreco_val_d0_plus_mm')
        save_array(val['rprod_mm'], 'atlas_trackreco_val_rprod_mm')
        save_array(val['mask_d0_minus'].astype(int), 'atlas_trackreco_val_mask_d0_minus')
        save_array(val['mask_d0_plus'].astype(int), 'atlas_trackreco_val_mask_d0_plus')
        save_array(val['mask_r_minus'].astype(int), 'atlas_trackreco_val_mask_r_minus')
        save_array(val['mask_r_plus'].astype(int), 'atlas_trackreco_val_mask_r_plus')
        save_array(val['mask_track_minus'].astype(int), 'atlas_trackreco_val_mask_track_minus')
        save_array(val['mask_track_plus'].astype(int), 'atlas_trackreco_val_mask_track_plus')

    return survival_bool

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

def invmass_rdv_efficiency(r_dv_array, m_dv_array):
    r_dv_m = r_dv_array * light_speed() * 1e3  # Convert rd_lab_norm from GeV^-1 to mm
    c1 = -20
    c2 = 3
    
    y1 = np.where(
        r_dv_m <= 50,
        10,
        ((c1 - 10) / (700 - 50)) * (r_dv_m - 50) + 10
    )
    y2 = c2
    
    # Apply the conditions vectorized
    survival_mask = (m_dv_array >= y1) & (m_dv_array >= y2)
    return survival_mask.astype(bool)


def invmass_rdv_efficiency_ee(r_dv_array, m_dv_array):
    r_dv_m = r_dv_array * light_speed() * 1e3  # Convert rd_lab_norm from GeV^-1 to mm
    c1 = -127
    c2 = 1.5
    
    y1 = np.where(
        r_dv_m <= 50,
        10,
        ((c1 - 10) / (700 - 50)) * (r_dv_m - 50) + 10
    )
    y2 = c2
    
    # Apply the conditions vectorized
    survival_mask = (m_dv_array >= y1) & (m_dv_array >= y2)
    return survival_mask.astype(bool)



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
        pz = self.pz() 
        pt = self.pT() 
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
            survival_mask_dv[(rd_lab_norm * light_speed() >= dv_min) & (rho * light_speed()<= dv_max_trans) & (z * light_speed() <= dv_max_long)] = True
        else:
            raise ValueError("Invalid cut type specified.")

        return survival_mask_dv, rd_lab, td, g_lab
    
    def cut_dv_2(self, angle_sq, tau, cut_type, dv_min=0, dv_max_long=0, dv_max_trans=0):
        if self.selected_mass_index is None or self.current_particle_type is None:
            raise ValueError("Both HNL mass and particle type must be selected before applying DV cut.")
        
        p_lab = self.selected_momenta.astype(np.float16)  # Convert momenta to float16

        HNLMass = np.float16(self.mass_hnl[self.selected_mass_index])  # Ensure mass is float16

        # Step 1: Compute Lorentz factor for HNLs as float16
        g_lab = np.divide(p_lab[:, 0], HNLMass).astype(np.float16)

        # Step 2: Find tau and generate decay times in the lab frame, ensuring operations are in float16
        tau_ = np.float16(self.find_tau(HNLMass, [0,0,1], flavour_hnl))
        tau_ = np.divide(tau_, np.float16(angle_sq))
        td = np.random.exponential(tau_, size=len(p_lab)).astype(np.float16)
        # Multiply decay times by Lorentz factor to get decay times in lab frame, as float16
        td_lab = np.multiply(g_lab, td).astype(np.float16)

        decay_length_lab = np.multiply(td_lab, np.float16(light_speed())).astype(np.float16)
        
        # Step 3: Calculate decay positions in 3D space as float16
        rd_lab = np.multiply((p_lab[:, 1:4].T, td_lab / p_lab[:, 0])).T.astype(np.float16)
        
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
            with np.errstate(invalid='ignore'):
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
        phi1 = np.arctan2(p1[:, 1], p1[:, 2])
        phi2 = np.arctan2(p2[:, 1], p2[:, 2])
        delta_phi = phi1 - phi2
        # Correct for the periodic boundary condition
        delta_phi = np.where(delta_phi > np.pi, delta_phi - 2*np.pi, delta_phi)
        delta_phi = np.where(delta_phi < -np.pi, delta_phi + 2*np.pi, delta_phi)
        return delta_phi
    
    def cut_invmass_nontrivial(self, r_lab):
        if self.selected_mass_index is None or self.current_particle_type is None:
            raise ValueError("Both HNL mass and particle type must be selected before applying non-trivial invariant mass cut.")

        rd_lab = r_lab
        rd_lab_norm = np.linalg.norm(rd_lab, axis=1)

        p_minus = self.momenta_dict['displaced_minus'][self.selected_mass_index]
        p_plus = self.momenta_dict['displaced_plus'][self.selected_mass_index]
        p_sum = p_minus + p_plus

        with np.errstate(invalid='ignore'):
            m2 = p_sum[:, 0]**2 - p_sum[:, 1]**2 - p_sum[:, 2]**2 - p_sum[:, 3]**2
            invariant_masses = np.sqrt(
                np.maximum(0, p_sum[:, 0]**2 - p_sum[:, 1]**2 - p_sum[:, 2]**2 - p_sum[:, 3]**2)
            )
        # Apply non-trivial invariant mass cut based on rd_lab_norm (distance) and invariant_masses
        if pid_displaced_lepton == 13:
            survival_mask_invmass_nt = invmass_rdv_efficiency(rd_lab_norm, invariant_masses)
        elif pid_displaced_lepton == 11:
            survival_mask_invmass_nt = invmass_rdv_efficiency_ee(rd_lab_norm, invariant_masses)
        else:
            raise ValueError("Unsupported pid_displaced_lepton. Use 11 for electron or 13 for muon.")

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


def survival_pT(momentum):
    array_name = f"survival_pT_{pT_minimum}"
    loaded_array = load_cut_array_(array_name)
    if loaded_array is not None:
        return loaded_array
    
    pt_boson = []
    pt_minus = []
    pt_plus = []
    pt_hnl = []
    pt_vt = []

    survival_bool_all_masses_pT_minus = []
    survival_bool_all_masses_pT_plus = []
    momentum_pT = copy.deepcopy(momentum)
    
    for mass in tqdm(mass_hnl, desc=f"[Computing cut] Transverse momentum "):
        original_batch = ParticleBatch(momentum_pT)
        batch_pT_minus = copy.deepcopy(original_batch)
        batch_pT_plus = copy.deepcopy(original_batch)
        batch_pT_parents = copy.deepcopy(original_batch)
        
        batch_pT_minus.mass(mass).particle('displaced_minus')
        pT_values_minus = batch_pT_minus.pT()  # Get pT values
        survival_mask_pT_minus = batch_pT_minus.cut_pT(pT_minimum)
        survival_bool_all_masses_pT_minus.append(survival_mask_pT_minus)
        pt_minus.append(pT_values_minus)  # Save actual pT values

        batch_pT_plus.mass(mass).particle('displaced_plus')
        pT_values_plus = batch_pT_plus.pT()  # Get pT values
        survival_mask_pT_plus = batch_pT_plus.cut_pT(pT_minimum)
        survival_bool_all_masses_pT_plus.append(survival_mask_pT_plus)
        pt_plus.append(pT_values_plus)  # Save actual pT values

        batch_pT_parents.mass(mass).particle('hnl')
        pT_values_hnl = batch_pT_parents.pT()  # Get pT values
        pt_hnl.append(pT_values_hnl)  # Save actual pT values

        batch_pT_parents.mass(mass).particle('neutrino')
        pT_values_neutrino = batch_pT_parents.pT()  # Get pT values
        pt_vt.append(pT_values_neutrino)  # Save actual pT values

        batch_pT_parents.mass(mass).particle('boson')
        pT_values_boson = batch_pT_parents.pT()  # Get pT values
        pt_boson.append(pT_values_boson)  # Save actual pT values

    survival_bool_all_masses_pT_minus = np.array(survival_bool_all_masses_pT_minus)
    survival_bool_all_masses_pT_plus = np.array(survival_bool_all_masses_pT_plus)
    
    combined_survival_pT = survival_bool_all_masses_pT_minus * survival_bool_all_masses_pT_plus
    
    # Save pT arrays
    save_array(pt_minus, 'pT_minus_values')
    save_array(pt_plus, 'pT_plus_values')
    save_array(pt_hnl, 'pT_HNL_values')
    save_array(pt_vt, 'pT_nu_values')
    
    save_cut_array1(combined_survival_pT, array_name)
    
    return combined_survival_pT


from multiprocessing import Pool

def survival_dv(momentum=1, rng_type=1):
    array_name_base = f"survival_dv_{cut_type_dv}_{r_min}_{r_max_l}_{r_max_t}"
    
    # Attempt to load each array separately
    loaded_bool_dv = load_cut_array(f"{array_name_base}_survival_bool_dv")
    loaded_rd_labs = load_cut_array(f"{array_name_base}_rd_labs")
    loaded_lifetimes_rest = load_cut_array(f"{array_name_base}_lifetimes_rest")
    loaded_lorentz_factors = load_cut_array(f"{array_name_base}_lorentz_factors")

    if all(v is not None for v in [loaded_bool_dv, loaded_rd_labs, loaded_lifetimes_rest, loaded_lorentz_factors]):
        print('[Loaded cut] Displaced vertex')
        return loaded_bool_dv, loaded_rd_labs, loaded_lifetimes_rest, loaded_lorentz_factors

    momentum_dv = copy.deepcopy(momentum)
    survival_bool_dv = np.zeros((len(mass_hnl), len(mixing), batch_size), dtype=bool)
    rd_labs = np.zeros((len(mass_hnl), len(mixing), batch_size, 3))
    lifetimes_rest = np.zeros((len(mass_hnl), len(mixing), batch_size))
    lorentz_factors = np.zeros((len(mass_hnl), len(mixing), batch_size))

    for i, mass in tqdm(enumerate(mass_hnl), total=len(mass_hnl), desc="[Computing cut] Displaced vertex    "):
        original_batch = ParticleBatch(momentum_dv)
        # Make a deep copy of the batch for this particular cut application
        batch_dv = copy.deepcopy(original_batch)
        for j, mix in enumerate(mixing):
            if rng_type == 1:
                # Apply the DV cut for each mass and mixing scenario and get decay vertices
                survival_mask_dv, rd_lab, td, g_lab = batch_dv.mass(mass).particle('hnl').cut_dv(mix, cut_type_dv, unit_converter(r_min), unit_converter(r_max_l), unit_converter(r_max_t))
            elif rng_type == 2:
                lifetime = HNL(mass, [0,0,1], False).computeNLifetime()
                lifetime_spread = np.random.exponential(lifetime, size=(batch_size))
                lifetime_spread_mixing = lifetime_spread / mix
                original_batch = ParticleBatch(momentum_dv)
                batch_dv = copy.deepcopy(original_batch)
                # Apply the DV cut for each mass and mixing scenario and get decay vertices with the second method
                survival_mask_dv, rd_lab, td, g_lab = batch_dv.mass(mass).particle('hnl').cut_dv_2(mix, lifetime_spread_mixing, cut_type_dv, unit_converter(r_min), unit_converter(r_max_l), unit_converter(r_max_t))
            
            survival_bool_dv[i, j, :] = survival_mask_dv

            if not large_data:
                rd_labs[i, j] = rd_lab  # Store the decay vertices
                lifetimes_rest[i, j, :] = td
                lorentz_factors[i, j, :] = g_lab

    # Save each array separately, specifying 'float16' where appropriate.
    save_cut_array(survival_bool_dv, f"{array_name_base}_survival_bool_dv")
    save_cut_array(rd_labs, f"{array_name_base}_rd_labs")
    save_cut_array(lifetimes_rest, f"{array_name_base}_lifetimes_rest")
    save_cut_array(lorentz_factors, f"{array_name_base}_lorentz_factors")
    
    if not large_data:
        return survival_bool_dv, rd_labs, lifetimes_rest, lorentz_factors
    else:
        return survival_bool_dv

def survival_invmass_nontrivial(momentum=1, r_labs=None):
    array_name = f"survival_invmass_nontrivial_{cut_type_dv}_{r_min}_{r_max_l}_{r_max_t}"
    loaded_array = load_cut_array_(array_name)
    if loaded_array is not None:
        print('[Loaded cut] Invariant mass')
        return loaded_array
    
    invmass_nontrivial_values = []  # To store actual invariant mass values
    survival_bool_invmass_nt = np.zeros((len(mass_hnl), len(mixing), batch_size), dtype=bool)
    momentum_invmass_nt = copy.deepcopy(momentum)
    total = len(mass_hnl) * len(mixing)

    with tqdm(total=total, desc="[Computing cut] Invariant mass      ") as pbar:
        for i, mass in enumerate(mass_hnl):
            # Compute invariant masses once per mass (since they don't depend on mixing)
            original_batch = ParticleBatch(momentum_invmass_nt)
            batch_invmass_nt = copy.deepcopy(original_batch)
            batch_invmass_nt.batch_size = batch_size
            batch_invmass_nt.mass(mass).particle('hnl')
            
            # Compute the actual invariant masses
            p_minus = batch_invmass_nt.momenta_dict['displaced_minus'][batch_invmass_nt.selected_mass_index]
            p_plus = batch_invmass_nt.momenta_dict['displaced_plus'][batch_invmass_nt.selected_mass_index]
            p_sum = p_minus + p_plus
            with np.errstate(invalid='ignore'):
                invariant_masses = np.sqrt(
                    np.maximum(0, p_sum[:, 0]**2 - p_sum[:, 1]**2 - p_sum[:, 2]**2 - p_sum[:, 3]**2)
                )
            # Append the computed invariant mass values
            if i == 0:
                # Initialize the array on the first iteration
                invmass_nontrivial_values = invariant_masses
            else:
                invmass_nontrivial_values = np.vstack((invmass_nontrivial_values, invariant_masses))

            for j, mix in enumerate(mixing):
                rd_lab = r_labs[i, j]  # Shape: (batch_size, 3)
                rd_lab_norm = np.linalg.norm(rd_lab, axis=1)  # Shape: (batch_size,)
                
                # Apply non-trivial invariant mass cut
                if pid_displaced_lepton == 13:
                    survival_bool_invmass_nt[i, j, :] = invmass_rdv_efficiency(rd_lab_norm, invariant_masses)
                elif pid_displaced_lepton == 11:
                    survival_bool_invmass_nt[i, j, :] = invmass_rdv_efficiency_ee(rd_lab_norm, invariant_masses)
                else:
                    raise ValueError("Unsupported pid_displaced_lepton. Use 11 for electron or 13 for muon.")
                
                pbar.update(1)
    
    # Save the computed invariant mass values
    save_array(invmass_nontrivial_values, 'invmass_nontrivial_values')
    
    save_cut_array1(survival_bool_invmass_nt, array_name)
    return survival_bool_invmass_nt



def survival_rap(momentum):
    array_name = f"survival_rap_{pseudorapidity_maximum}"
    loaded_array = load_cut_array_(array_name)
    if loaded_array is not None:
        print('[Loaded cut] Pseudorapidity')
        return loaded_array
    
    eta_minus = []
    eta_plus = []

    survival_bool_all_masses_rap_minus = []
    survival_bool_all_masses_rap_plus = []
    momentum_rap = copy.deepcopy(momentum)
    momentum_rap_2 = copy.deepcopy(momentum)

    for mass in tqdm(mass_hnl, desc="[Computing cut] Pseudorapidity      "):
        original_batch_minus = ParticleBatch(momentum_rap)
        original_batch_plus = ParticleBatch(momentum_rap_2)

        # Process 'displaced_minus'
        batch_rap_minus = original_batch_minus.mass(mass).particle('displaced_minus')
        eta_values_minus = batch_rap_minus.eta()  # Get eta values
        survival_mask_minus = batch_rap_minus.cut_rap()
        survival_bool_all_masses_rap_minus.append(survival_mask_minus)
        eta_minus.append(eta_values_minus)  # Save actual eta values

        # Process 'displaced_plus'
        batch_rap_plus = original_batch_plus.mass(mass).particle('displaced_plus')
        eta_values_plus = batch_rap_plus.eta()  # Get eta values
        survival_mask_plus = batch_rap_plus.cut_rap()
        survival_bool_all_masses_rap_plus.append(survival_mask_plus)
        eta_plus.append(eta_values_plus)  # Save actual eta values

    survival_bool_all_masses_rap_minus = np.array(survival_bool_all_masses_rap_minus)
    survival_bool_all_masses_rap_plus = np.array(survival_bool_all_masses_rap_plus)

    combined_survival_rap = survival_bool_all_masses_rap_minus * survival_bool_all_masses_rap_plus

    # Save eta arrays
    save_array(eta_minus, 'eta_minus_values')
    save_array(eta_plus, 'eta_plus_values')

    save_cut_array1(combined_survival_rap, array_name)
    
    return combined_survival_rap

def survival_invmass(cut_condition, momentum, experimental_trigger=False):
    array_name = f"survival_invmass_{cut_condition}_trivial"
    loaded_array = load_cut_array_(array_name)
    if loaded_array is not None:
        print('[Loaded cut] Invariant mass')
        return loaded_array
    
    # Extract the numeric value from the cut_condition string (e.g., '5 GeV')
    if isinstance(cut_condition, str):
        cut_value = float(cut_condition.split()[0])  # Extract numerical value
    else:
        cut_value = float(cut_condition)

    invmass_values = []  # To store actual invariant mass values
    survival_bool_all_masses_invmass = []
    momentum_invmass = copy.deepcopy(momentum)
    
    for mass in tqdm(mass_hnl, desc="[Computing cut] Invariant mass      "):
        original_batch = ParticleBatch(momentum_invmass)
        batch_invmass = copy.deepcopy(original_batch)
        
        # Retrieve momenta of 'displaced_minus' and 'displaced_plus' particles
        p_minus = batch_invmass.mass(mass).particle('displaced_minus').selected_momenta
        p_plus = batch_invmass.mass(mass).particle('displaced_plus').selected_momenta

        # Compute the sum of the 4-momenta for the invariant mass calculation
        p_sum = p_minus + p_plus

        # Compute the actual invariant mass (not boolean)
        with np.errstate(invalid='ignore'):
            invmass_values_batch = np.sqrt(np.maximum(0, p_sum[:, 0]**2 - p_sum[:, 1]**2 - p_sum[:, 2]**2 - p_sum[:, 3]**2))
        
        # Append the computed invariant mass values
        invmass_values.append(invmass_values_batch)
        
        # Apply the cut condition to create a boolean mask
        survival_mask_invmass = invmass_values_batch >= cut_value
        survival_bool_all_masses_invmass.append(survival_mask_invmass)

    survival_bool_all_masses_invmass = np.array(survival_bool_all_masses_invmass)
    
    # Save the actual invariant mass values
    save_array(invmass_values, 'invmass_values')

    # Save the boolean survival mask as usual
    save_cut_array1(survival_bool_all_masses_invmass, array_name)
    
    return survival_bool_all_masses_invmass



def survival_deltaR(cut_condition, momentum):
    array_name = f"survival_dR_{cut_condition}"
    loaded_array = load_cut_array_(array_name)
    if loaded_array is not None:
        print('[Loaded cut] Angular seperation')
        return loaded_array
    
    survival_bool_all_masses_deltaR = []
    momentum_deltaR = copy.deepcopy(momentum)
    for mass in tqdm(mass_hnl, desc="[Computing cut] Angular separation  "):
        original_batch = ParticleBatch(momentum_deltaR)
        batch_deltaR = copy.deepcopy(original_batch)
        survival_mask = batch_deltaR.mass(mass).cut_deltaR(cut_condition)
        survival_bool_all_masses_deltaR.append(survival_mask)
    
    survival_bool_all_masses_deltaR = np.array(survival_bool_all_masses_deltaR)
    save_cut_array1(survival_bool_all_masses_deltaR, array_name)
    return survival_bool_all_masses_deltaR


def save_array(array, name=''):
    current_directory = os.getcwd()                                         # Current path
    out_folder = output_folder if ('output_folder' in globals() and output_folder) else data_folder
    data_path         = os.path.join(current_directory,'data', out_folder) # Output folder path
    array_path         = os.path.join(data_path, 'Plots', 'PlotData', f'{name}.npy')
    os.makedirs(os.path.join(data_path, 'Plots'), exist_ok=True) # Making directory if not already exists
    os.makedirs(os.path.join(data_path, 'Plots', 'PlotData'), exist_ok=True)
    np.save(array_path, array)

def print_dashes(text, char='-'):
    width = shutil.get_terminal_size().columns
    side = (width - len(text) - 2) // 2
    print(f"{char * side} {text} {char * (width - side - len(text) - 2)}")

def save_cut_array1(array, name=''):
    current_directory = os.getcwd()
    out_folder = output_folder if ('output_folder' in globals() and output_folder) else data_folder
    data_path = os.path.join(current_directory, 'data', out_folder, 'Cut computations')
    os.makedirs(data_path, exist_ok=True)
    array_path = os.path.join(data_path, f'{name}.npz')
    
    np.savez_compressed(array_path, array=array)
    

def load_cut_array1(name=''):
    current_directory = os.getcwd()
    out_folder = output_folder if ('output_folder' in globals() and output_folder) else data_folder
    data_path = os.path.join(current_directory, 'data', out_folder)
    array_path = os.path.join(data_path, 'Cut computations', f'{name}.npz')
    if os.path.exists(array_path):
        data = np.load(array_path)
        return (data['survival_bool_dv'], data['rd_labs'], data['lifetimes_rest'], data['lorentz_factors'])
    return None

def save_cut_array(array, name=''):
    current_directory = os.getcwd()
    out_folder = output_folder if ('output_folder' in globals() and output_folder) else data_folder
    data_path = os.path.join(current_directory, 'data', out_folder, 'Cut computations')
    os.makedirs(data_path, exist_ok=True)
    array_path = os.path.join(data_path, f'{name}.npz')
    
    # Save the array with a specified name within the .npz file
    if name.endswith('survival_bool_dv'):
        np.savez_compressed(array_path, survival_bool_dv=array)
    elif name.endswith('rd_labs'):
        np.savez_compressed(array_path, rd_labs=array)
    elif name.endswith('lifetimes_rest'):
        np.savez_compressed(array_path, lifetimes_rest=array)
    elif name.endswith('lorentz_factors'):
        np.savez_compressed(array_path, lorentz_factors=array)


def load_cut_array_(name=''):
    current_directory = os.getcwd()
    out_folder = output_folder if ('output_folder' in globals() and output_folder) else data_folder
    data_path = os.path.join(current_directory, 'data', out_folder, 'Cut computations')
    array_path = os.path.join(data_path, f'{name}.npz')
    if os.path.exists(array_path):
        data = np.load(array_path)
        return data['array']  # Extract the array using the key 'array'
    return None


def load_cut_array(name=''):
    current_directory = os.getcwd()
    out_folder = output_folder if ('output_folder' in globals() and output_folder) else data_folder
    data_path = os.path.join(current_directory, 'data', out_folder, 'Cut computations')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    array_path = os.path.join(data_path, f'{name}.npz')
    if os.path.exists(array_path):
        data = np.load(array_path)
        if name.endswith('survival_bool_dv'):
            return data['survival_bool_dv']
        elif name.endswith('rd_labs'):
            return data['rd_labs']
        elif name.endswith('lifetimes_rest'):
            return data['lifetimes_rest']
        elif name.endswith('lorentz_factors'):
            return data['lorentz_factors']
    return None
    
def data_processing(momenta):
    print_dashes("Data Processing")

    batch = ParticleBatch(momenta)
    
    survival_pT_displaced = survival_pT(momentum=momenta)
    
    survival_rap_displaced = survival_rap(momentum=momenta)
    
    survival_deltaR_displaced = survival_deltaR(deltaR_minimum, momentum=momenta)
    
    if large_data:
        # In large_data mode, survival_dv() only returns the DV survival mask (no r_lab/td/gamma).
        # Cuts requiring r_lab (nontrivial invmass, ATLAS track reco) are therefore unsupported.
        if apply_atlas_track_reco:
            raise ValueError("apply_atlas_track_reco requires large_data=False (needs r_lab to compute d0 and R_prod)")
        if invmass_cut_type == 'nontrivial':
            raise ValueError("invmass_cut_type='nontrivial' requires large_data=False (needs r_lab)")

        survival_dv_displaced = survival_dv(momentum=momenta, rng_type=1)
        survival_invmass_displaced = survival_invmass(invmass_minimum, momentum=momenta, experimental_trigger=invmass_experimental)

        arrays = (
            np.array(survival_dv_displaced),
            np.array(survival_pT_displaced),
            np.array(survival_rap_displaced),
            np.array(survival_invmass_displaced),
            np.array(survival_deltaR_displaced),
        )
        return batch, arrays

    # full mode (keeps DV positions, lifetimes, boosts)
    survival_dv_displaced, r_lab, lifetimes_rest_, lorentz_factors = survival_dv(momentum=momenta, rng_type=1)

    survival_atlas_trackreco = None
    if apply_atlas_track_reco:
        survival_atlas_trackreco = survival_atlas_track_reco(momentum=momenta, r_labs=r_lab)

    if invmass_cut_type == 'nontrivial':
        survival_invmass_displaced = survival_invmass_nontrivial(momentum=momenta, r_labs=r_lab)
    else:
        survival_invmass_displaced = survival_invmass(invmass_minimum, momentum=momenta, experimental_trigger=invmass_experimental)

    arrays = (
        np.array(survival_dv_displaced),
        np.array(survival_pT_displaced),
        np.array(survival_rap_displaced),
        np.array(survival_invmass_displaced),
        np.array(survival_deltaR_displaced),
        np.array(r_lab),
        np.array(lifetimes_rest_),
        np.array(lorentz_factors),
        None if survival_atlas_trackreco is None else np.array(survival_atlas_trackreco),
    ) # defining a tuple for easier management of survival arrays on main

    return batch, arrays
    