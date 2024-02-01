from parameters.data_parameters import *
# Handling of data by converting to correct units (if need be) and performing the cuts as per the experimental parameters

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


def data_processing(momenta):

    (momentum_boson, momentum_HNL, momentum_prompt, 
    momentum_displaced_minus, momentum_displaced_plus, 
    momentum_neutrino) = momenta

    HNL_mass
    
    return 0