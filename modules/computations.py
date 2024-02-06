import numpy as np
from modules.data_processing import *
from parameters.data_parameters import *
from parameters.experimental_parameters import *

def compute_production_efficiency(production_nocuts, survivals, survival_dv_displaced=None):
    """
    Compute production rates adjusted by efficiencies derived from combinations of survival bools.

    :param production_nocuts: Array of initial production rates with shape (masses, mixings).
    :param survivals: List of survival arrays. Each survival array should have shape (masses, particles)
                      or (masses, mixings, particles) if including 'survival_dv_displaced'.
    :param survival_dv_displaced: Optional, an array of survival bools including mixing dependence with shape (masses, mixings, particles).
    :return: Production rates adjusted by combined survival efficiencies, shape (masses, mixings).
    """

    # Handle the special case where survival_dv_displaced is directly provided
    if survival_dv_displaced is not None:
        combined_survival = survival_dv_displaced
    else:
        # Assume all survivals initially affect all particles equally across all mixings
        # This creates an array of shape (masses, mixings, particles) filled with ones
        combined_survival = np.ones((production_nocuts.shape[0], production_nocuts.shape[1], survivals[0].shape[-1]))

    # Adjust combined survival for each survival bool in survivals
    for survival in survivals:
        if len(survival.shape) == 3 and survival.shape[1] > 1:
            # Survival array includes mixing dimension, shape (masses, mixings, particles)
            combined_survival *= survival
        else:
            # Expand survival across the mixing dimension, shape becomes (masses, 1, particles)
            survival_expanded = np.expand_dims(survival, axis=1)
            # Broadcast to match combined_survival's mixing dimension
            combined_survival *= survival_expanded

    # Compute efficiency by averaging over the particle dimension
    efficiency = np.mean(combined_survival, axis=-1)  # Shape becomes (masses, mixings)

    # Scale initial production rates by efficiency
    production_allcuts = production_nocuts * efficiency

    return production_allcuts


def computations(momenta, batch, arrays):
    print('----------------------- Computing HNL production --------------------')
    survival_dv_displaced, survival_pT_displaced, survival_rap_displaced, survival_invmass_displaced, survival_deltaR_displaced = arrays

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
        print(np.shape(survival_dv_displaced))
        print(np.shape(survival_invmass_displaced))
        combined_survival = survival_dv_displaced * survival_invmass_displaced
    else:
        combined_survival = survival_dv_displaced

    for survival in expanded_survivals:
        combined_survival *= survival

    # Calculate efficiency by averaging over the particles dimension
    efficiency = np.mean(combined_survival, axis=-1)  # Averaging over the third dimension (particles)

    # Calculate initial production rates before any cuts
    theoretical_sigma = 2.05e7  # fb corresponding to 20.5nb from https://arxiv.org/pdf/1603.09222.pdf
    cross_sections = theoretical_sigma * np.array([HNL(m, [0,0,1], False).findBranchingRatio('N -> mu- mu+ nu_tau') for m in mass_hnl])
    production_nocuts = luminosity * cross_sections[:, np.newaxis] * np.array(mixing)[np.newaxis, :]  # Shape now (19, 100)

    # Apply efficiency to scale production rates
    production_allcuts = production_nocuts * efficiency 
    print('  ...................................................')
    print('    Total mean survival rate: ', np.mean(efficiency))





    # Compute production rates adjusted for different cuts
    production_pT      = compute_production_efficiency(production_nocuts, [survival_pT_displaced])
    production_rap     = compute_production_efficiency(production_nocuts, [survival_rap_displaced])
    production_invmass = compute_production_efficiency(production_nocuts, [survival_invmass_displaced])

    survival_pT_rap = [survival_pT_displaced, survival_rap_displaced] 
    production__pT_rap = compute_production_efficiency(production_nocuts, survival_pT_rap)

    survival_pT_rap_invmass = [survival_pT_displaced, survival_rap_displaced, survival_invmass_displaced] 
    production__pT_rap_invmass = compute_production_efficiency(production_nocuts, survival_pT_rap_invmass)

    survival_allcuts = [survival_pT_displaced, survival_rap_displaced, survival_invmass_displaced, survival_dv_displaced, survival_deltaR_displaced] 
    production_allcuts = compute_production_efficiency(production_nocuts, survival_allcuts)

    production_arrays = production_nocuts, production_allcuts, production_pT, production_rap, production_invmass, production__pT_rap, production__pT_rap_invmass
    return production_arrays