#data_folder = 'fakedata' # Folder name containing MadGraph data
data_folder = 'w2tau-ee-moremass'
prompt_lepton_flavour = 3 # 1 = electron, 2 = muon, 3 = tau

#data_type = 'ROOT' # Either "HEPMC" or "LHE" or "ROOT" format
data_type = 'LHE' # Either "HEPMC" or "LHE" or "ROOT" format

#mass_hnl = [6,7,8]#[i*0.25 for i in range(2,4)] # HNL masses from data in GeV
mass_hnl = [i*0.25 for i in range(2,81)] # HNL masses from data in GeV

flavour_hnl = 3 # 1=e, 2=mu, 3=tau
batch_size = 25000 # Number of events generated per run (mass) in MadGraph
MG_N_Width = 1e-5 # The HNL decay width within MG (parameter card)

# HEPMC PIDs (positive lepton PID = anti-particle)
pid_boson            = -24 # 24 = W boson
pid_prompt_lepton    = 15 # 11=e, 13 = muon, 15 = tau
pid_displaced_lepton = 11 # 11=e, 13 = muon, 15 = tau, here we assume a symmetric decay to lepton+ and lepton- of the same flavour
pid_neutrino         = 16 # 12 = ve, 14 = vm, 16 = vt
pid_HNL              = 9900016 # HNL of SM flavour tau

large_data = False # Special handling of large data files to avoid RAM overload by neglecting some DV arrays (lifetimes, positions, lorentz factors)
