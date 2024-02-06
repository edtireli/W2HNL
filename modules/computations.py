import numpy as np
from modules.data_processing import *
from parameters.data_parameters import *
from parameters.experimental_parameters import *


def computations(momenta, batch, arrays):
    print('----------------------- Computing HNL production --------------------')
    survival_dv_displaced, survival_pT_displaced, survival_rap_displaced, survival_invmass_displaced, survival_deltaR_displaced = arrays

    print('     Invariant mass survival: ', np.mean(survival_invmass_displaced)) # Validated
    print('     Pseudorapidity survival: ', np.mean(survival_rap_displaced)) # A bit high?
    print('   Displaced vertex survival: ', np.mean(survival_dv_displaced)) # A bit low
    print(' Angular seperation survival: ', np.mean(survival_deltaR_displaced)) # Validated
    print('Transverse momentum survival: ', np.mean(survival_pT_displaced)) # Validated

    # Ensure all survival arrays are expanded to match shapes for broadcasting
    expanded_survivals = [
        survival[:, np.newaxis, :] for survival in [
            survival_pT_displaced, 
            survival_rap_displaced, 
            survival_invmass_displaced, 
            survival_deltaR_displaced
        ]
    ]

    # Combine survivals with proper broadcasting
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

    print('    Total mean survival rate: ', np.mean(efficiency))

    return production_allcuts