#data_folder = 'w2tau_@1$' # Folder name containing MadGraph data
data_folder = 'w2tau-moremass'
prompt_lepton_flavour = 3 # 1 = electron, 2 = muon, 3 = tau

#data_type = 'HEPMC' # Either "HEPMC" or "LHE" format
data_type = 'LHE' # Either "HEPMC" or "LHE" format

#mass_hnl = [i*0.5 for i in range(2,21)] # HNL masses from data in GeV
mass_hnl = [i*0.1 for i in range(2,101)] # HNL masses from data in GeV

flavour_hnl = 3 # 1=e, 2=mu, 3=tau
batch_size = 9999 # Number of events generated per run (mass) in MadGraph
MG_N_Width = 1e-5 # The HNL decay width within MG (parameter card)

# HEPMC PIDs (positive lepton PID = anti-particle)
pid_boson            = -24 # 24 = W boson
pid_prompt_lepton    = 15 # 11=e, 13 = muon, 15 = tau
pid_displaced_lepton = 13 # 11=e, 13 = muon, 15 = tau, here we assume a symmetric decay to lepton+ and lepton- of the same flavour
pid_neutrino         = 16 # 12 = ve, 14 = vm, 16 = vt
pid_HNL              = 9900016 # HNL of SM flavour tau
