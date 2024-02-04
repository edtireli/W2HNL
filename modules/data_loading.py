import os
from utils.LHE_processing import *
from utils.HEPMC_processing import *
from parameters.data_parameters import *

def data_loading():
    
    current_directory = os.getcwd()                                         # Current path
    data_path         = os.path.join(current_directory,'data', data_folder) # Data folder path

    if data_type == 'LHE':
        momenta = LHE_data_processing(data_path, batch_size, prompt_lepton_flavour)
    
    elif data_type == 'HEPMC':
        momenta = HEPMC_data_processing(data_path)

    return momenta

