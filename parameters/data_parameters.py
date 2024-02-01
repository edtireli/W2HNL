# A parameter file for various dimensions and variables to do with the raw data previously generated
import os

data_folder = 'w2tau_@1$' # Folder name containing MadGraph data
prompt_lepton_flavour = 3 # 1 = electron, 2 = muon, 3 = tau

data_type = 'HEPMC' # Either "HEPMC" or "LHE" format
HNL_id = 9900016 # Only required in the case of HEPMC

HNL_mass = [i*0.5 for i in range(2,21)] # HNL masses from data
batch_size = 9999 # Number of events generated per run (mass) in MadGraph

