#data_folder = 'fakedata' # Folder name containing MadGraph data
data_folder = 'w2tau-moremass'
prompt_lepton_flavour = 3 # 1 = electron, 2 = muon, 3 = tau

#data_type = 'ROOT' # Either "HEPMC" or "LHE" or "ROOT" format
data_type = 'LHE' # Either "HEPMC" or "LHE" or "ROOT" format

#mass_hnl = [6,7,8]#[i*0.25 for i in range(2,4)] # HNL masses from data in GeV
mass_hnl = [i*0.1 for i in range(2,101)] # HNL masses from data in GeV

flavour_hnl = 3 # 1=e, 2=mu, 3=tau
batch_size = 10000 # Number of events generated per run (mass) in MadGraph
MG_N_Width = 1e-5 # The HNL decay width within MG (parameter card)

# HEPMC PIDs (positive lepton PID = anti-particle)
pid_boson            = -24 # 24 = W boson
pid_prompt_lepton    = 15 # 11=e, 13 = muon, 15 = tau
pid_displaced_lepton = 13 # 11=e, 13 = muon, 15 = tau, here we assume a symmetric decay to lepton+ and lepton- of the same flavour
pid_neutrino         = 16 # 12 = ve, 14 = vm, 16 = vt
pid_HNL              = 9900016 # HNL of SM flavour tau

large_data = False # Special handling of large data files to avoid RAM overload by neglecting some DV arrays (lifetimes, positions, lorentz factors)

# Optional output folder override.
# If set, plots/cut caches are written under data/<output_folder>/...,
# while input data is still loaded from data/<data_folder>/...
output_folder = None

# ------------------------------------------------------------
# ATLAS-derived track reconstruction efficiency (W process)
#　From https://arxiv.org/pdf/2304.12867 Fig. 7
# When enabled, applies an additional per-event survival mask that
# emulates ATLAS track reconstruction efficiency as a function of:
#  - |d0| (transverse impact parameter) [mm]
#  - R_prod (transverse production radius) [mm]
#
# The efficiency is factorized per track as:
#   eps_track = eps_d0(|d0|) * eps_R(R_prod)
# and the event survives if BOTH displaced tracks are reconstructed.
#
# NOTE: The default points below are a digitization of the W→Nl (purple)
# markers in the provided plot. If you have the underlying table (or the
# vector PDF), we can replace these with exact values.
# ------------------------------------------------------------
apply_atlas_track_reco = False

# |d0| efficiency parameterization (mm -> efficiency)
atlas_track_reco_d0_points_mm = [
	0.440849,
	0.750004,
	2.24999,
	4.00002,
	6.00004,
	8.5,
	12.5,
	17.4999,
	25.0001,
	40.0,
	62.5003,
	87.5001,
	125.001,
	175.001,
]
atlas_track_reco_d0_eff = [
	1.07089,
	0.934775,
	0.92284,
	0.907845,
	0.825546,
	0.823589,
	0.804646,
	0.76969,
	0.749444,
	0.700954,
	0.623784,
	0.550713,
	0.406451,
	0.383833,
]

# R_prod efficiency parameterization (mm -> efficiency)
atlas_track_reco_rprod_points_mm = [
	4.94069,
	14.8222,
	24.7034,
	34.5848,
	44.4662,
	61.7591,
	86.4623,
	111.166,
	135.87,
	172.925,
	222.332,
	271.739,
]
atlas_track_reco_rprod_eff = [
	0.936755,
	0.911044,
	0.886173,
	0.868147,
	0.836178,
	0.81978,
	0.771877,
	0.708359,
	0.571463,
	0.583806,
	0.594192,
	0.487514,
]

# Validation plots will be produced for the closest (mass, mixing)
# point to these targets.
atlas_track_reco_validation_mass_GeV = 10.0
atlas_track_reco_validation_mixing = 1e-6