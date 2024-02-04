# Handling of data by converting to correct units (if need be) and performing the cuts as per the experimental parameters

import numpy as np
from parameters.data_parameters import *
import matplotlib.pyplot as plt

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

class MomentumComponents:
    def __init__(self, momenta):
        """
        Initialize with selected momentum array.
        :param momenta: A numpy array of shape (events, 4) where each row contains (E, px, py, pz).
        """
        self.momenta = momenta

    def E(self):
        return self.momenta[:, 0]

    def px(self):
        return self.momenta[:, 1]

    def py(self):
        return self.momenta[:, 2]

    def pz(self):
        return self.momenta[:, 3]

class particle_batch:
    def __init__(self, momenta):
        self.momenta_dict = {
            'boson': momenta[0],
            'hnl': momenta[1],
            'prompt': momenta[2],
            'displaced_minus': momenta[3],
            'displaced_plus': momenta[4],
            'neutrino': momenta[5],
        }
        self.mass_hnl = mass_hnl  # Ensure this is defined elsewhere, e.g., a list of HNL mass values
        self.selected_mass_index = None

    def mass(self, mass):
        mass_value = float(mass.split()[0])
        if mass_value in self.mass_hnl:
            self.selected_mass_index = self.mass_hnl.index(mass_value)
            return self
        else:
            raise ValueError("Mass not found in the dataset.")

    def momentum(self, particle_type):
        if self.selected_mass_index is None:
            raise ValueError("HNL mass must be selected before retrieving momenta.")
        if particle_type not in self.momenta_dict:
            raise ValueError(f"Particle type '{particle_type}' not recognized.")
        
        # Retrieve and return momentum components for the specified particle
        selected_momenta = self.momenta_dict[particle_type][self.selected_mass_index]
        return MomentumComponents(selected_momenta)

def data_processing(momenta):

    (momentum_boson, momentum_HNL, momentum_prompt, 
    momentum_displaced_minus, momentum_displaced_plus, 
    momentum_neutrino) = momenta

    batch = particle_batch(momenta)
    
    return batch