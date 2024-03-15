import numpy as np
from modules.data_processing import *
from parameters.data_parameters import *
from parameters.experimental_parameters import *

import numpy as np

def compute_production_efficiency(production_nocuts, survivals):
    """
    Compute production rates adjusted by efficiencies for both 2D and 3D survival arrays.

    :param production_nocuts: Array of initial production rates with shape (masses, mixings).
    :param survivals: List of survival arrays, each with shape (masses, particles) for 2D,
                      or (masses, mixings, particles) for 3D.
    :return: Production rates adjusted by combined survival efficiencies, shape (masses, mixings).
    """
    # Create a combined survival array initialized to ones
    combined_survival = np.ones(production_nocuts.shape + (survivals[0].shape[-1],))

    for survival in survivals:
        survival_copy = np.copy(survival)  # Explicit copy to ensure original is not modified
        if survival_copy.ndim == 2:
            # Expand survival across the mixing dimension and use copy for operation
            survival_expanded = np.expand_dims(survival_copy, axis=1)
            combined_survival *= survival_expanded
        elif survival_copy.ndim == 3:
            # Use copy directly if survival includes the mixing dimension
            combined_survival *= survival_copy

    # Compute efficiency by averaging over the particle dimension
    efficiency = np.mean(combined_survival, axis=-1)  # Shape becomes (masses, mixings)

    # Scale initial production rates by efficiency
    production_adjusted = production_nocuts * efficiency

    return production_adjusted



def computations(momenta, arrays):
    print('----------------------- Computing HNL production --------------------')
    survival_dv_displaced, survival_pT_displaced, survival_rap_displaced, survival_invmass_displaced, survival_deltaR_displaced, r_lab, lifetimes_rest, lorentz_factors = arrays

    print('     Invariant mass survival: ', np.mean(survival_invmass_displaced)) # Validated for trivial case
    print('     Pseudorapidity survival: ', np.mean(survival_rap_displaced)) # A bit high (because we use simple pseudorapidity)
    print('   Displaced vertex survival: ', np.mean(survival_dv_displaced)) # Validated
    print(' Angular seperation survival: ', np.mean(survival_deltaR_displaced)) # Validated
    print('Transverse momentum survival: ', np.mean(survival_pT_displaced)) # Validated

    # Ensure all survival arrays are expanded to match shapes for broadcasting
    if invmass_cut_type == 'nontrivial':
        expanded_survivals = [
        survival[:, np.newaxis, :] for survival in [
            survival_pT_displaced, 
            survival_rap_displaced, 
            survival_deltaR_displaced
        ]
    ]
    else:    
        expanded_survivals = [
            survival[:, np.newaxis, :] for survival in [
                survival_pT_displaced, 
                survival_rap_displaced, 
                survival_invmass_displaced, 
                survival_deltaR_displaced
            ]
        ]

    # Combine survivals with proper broadcasting
    if invmass_cut_type == 'nontrivial':
        combined_survival = combined_survival = survival_dv_displaced * np.copy(survival_invmass_displaced)
    else:
        combined_survival = survival_dv_displaced

    for survival in expanded_survivals:
        combined_survival = np.copy(survival_dv_displaced)
    
    # Calculate efficiency by averaging over the particles dimension
    efficiency = np.mean(combined_survival, axis=-1)  # Averaging over the third dimension (particles)

    # Calculate initial production rates before any cuts
    experimental_sigma = 2.064e7  # fb corresponding to 20.5nb from https://arxiv.org/pdf/1603.09222.pdf
    cross_sections = experimental_sigma * np.array([HNL(m, [0,0,1], False).findBranchingRatio('N -> mu- mu+ nu_tau') for m in mass_hnl])
    production_nocuts = luminosity * cross_sections[:, np.newaxis] * np.array(mixing)[np.newaxis, :]  # Shape now (mass, mixing)

    # Apply efficiency to scale production rates
    production_allcuts = production_nocuts * efficiency 
    print('  ...................................................')
    print('    Total mean survival rate: ', np.mean(efficiency))


    # Compute production rates adjusted for different cuts
    production_pT      = compute_production_efficiency(production_nocuts, [survival_pT_displaced])
    production_rap     = compute_production_efficiency(production_nocuts, [survival_rap_displaced])
    production_invmass = compute_production_efficiency(production_nocuts, [survival_invmass_displaced])
    production_dv      = compute_production_efficiency(production_nocuts, [survival_dv_displaced])

    survival_pT_rap    = [survival_pT_displaced, survival_rap_displaced] 
    production__pT_rap = compute_production_efficiency(production_nocuts, survival_pT_rap)

    survival_pT_rap_invmass    = [survival_pT_displaced, survival_rap_displaced, survival_invmass_displaced] 
    production__pT_rap_invmass = compute_production_efficiency(production_nocuts, survival_pT_rap_invmass)

    survival_allcuts   = [survival_pT_displaced, survival_rap_displaced, survival_invmass_displaced, survival_dv_displaced, survival_deltaR_displaced] 
    production_allcuts = compute_production_efficiency(production_nocuts, survival_allcuts)

    production_arrays = production_nocuts, production_allcuts, production_pT, production_rap, production_invmass, production_dv, production__pT_rap, production__pT_rap_invmass
    return production_arrays