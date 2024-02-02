import os
from utils.LHE_processing import *
from utils.HEPMC_processing import *
from parameters.data_parameters import *

def data_loading():
    
    current_directory = os.getcwd()                                         # Current path
    data_path         = os.path.join(current_directory,'data', data_folder) # Data folder path

    if data_type == 'LHE':
        (momentum_boson, momentum_HNL, momentum_prompt, 
        momentum_displaced_minus, momentum_displaced_plus, 
        momentum_neutrino) = LHE_data_processing(data_path, batch_size, prompt_lepton_flavour)
    
    elif data_type == 'HEPMC':
        (momentum_boson, momentum_HNL, momentum_prompt, 
        momentum_displaced_minus, momentum_displaced_plus, 
        momentum_neutrino) = HEPMC_data_processing(data_path, pid_HNL)

    return momentum_boson, momentum_HNL, momentum_prompt, momentum_displaced_minus, momentum_displaced_plus, momentum_neutrino

