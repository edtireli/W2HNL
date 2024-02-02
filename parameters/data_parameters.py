# A parameter file for various dimensions and variables to do with the raw data previously generated
import os

data_folder = 'w2tau_@1$' # Folder name containing MadGraph data
prompt_lepton_flavour = 3 # 1 = electron, 2 = muon, 3 = tau

data_type = 'HEPMC' # Either "HEPMC" or "LHE" format

HNL_mass = [i*0.5 for i in range(2,21)] # HNL masses from data
batch_size = 9999 # Number of events generated per run (mass) in MadGraph


# HEPMC PIDs
pid_boson            = 24 # 24 = W boson
pid_prompt_lepton    = 15 # 11=e, 13 = muon, 15 = tau (convention is positive pid = anti-particle)
pid_displaced_lepton = 13 # 11=e, 13 = muon, 15 = tau, here we assume a symmetric decay to lepton+ and lepton- of the same flavour
pid_neutrino         = 16 # 12 = ve, 14 = vm, 16 = vt
pid_HNL              = 9900016 # HNL of SM flavour tau