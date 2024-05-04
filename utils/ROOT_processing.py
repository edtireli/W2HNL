import os
import uproot
import awkward as ak
import numpy as np
from tqdm import tqdm
from parameters.data_parameters import *
from parameters.experimental_parameters import *

def root_data_processing(base_folder):
    # Initialize the data structure with an array of zeros for each particle type
    data_structure = {
        'W_boson': np.zeros((len(mass_hnl), batch_size, 4)),
        'HNL': np.zeros((len(mass_hnl), batch_size, 4)),
        'prompt_lepton': np.zeros((len(mass_hnl), batch_size, 4)),
        'dilepton_minus': np.zeros((len(mass_hnl), batch_size, 4)),
        'dilepton_plus': np.zeros((len(mass_hnl), batch_size, 4)),
        'neutrino': np.zeros((len(mass_hnl), batch_size, 4))
    }

    # Iterate over each run folder
    for i in tqdm(range(1, len(mass_hnl) + 1), desc="Processing ROOT files"):
        folder_name = f"{base_folder}/Events/run_{i:02d}"
        file_path = os.path.join(folder_name, "unweighted_events.root")

        with uproot.open(file_path)["Delphes;1"]["Particle"] as particles:
            # Load particle data
            pids = particles["Particle.PID"].array(library="ak")
            px = particles["Particle.Px"].array(library="ak")
            py = particles["Particle.Py"].array(library="ak")
            pz = particles["Particle.Pz"].array(library="ak")
            energy = particles["Particle.E"].array(library="ak")

            # Process each particle type
            for particle_type, pid in [('W_boson', pid_boson), ('HNL', pid_HNL), 
                                       ('prompt_lepton', pid_prompt_lepton), 
                                       ('dilepton_minus', pid_displaced_lepton),
                                       ('dilepton_plus', pid_displaced_lepton), 
                                       ('neutrino', pid_neutrino)]:
                indices = ak.where(abs(pids) == abs(pid))
                four_momenta = {
                    "E": energy[indices],
                    "px": px[indices],
                    "py": py[indices],
                    "pz": pz[indices]
                }

                # Create a numpy array from the four momentum components
                if ak.count_nonzero(indices) > 0:
                    momenta_array = np.stack([ak.to_numpy(four_momenta[key]) for key in ['E', 'px', 'py', 'pz']], axis=-1)
                    n_events = min(momenta_array.shape[0], batch_size)
                    data_structure[particle_type][i-1, :n_events, :] = momenta_array[:n_events]
                else:
                    continue

    return (data_structure['W_boson'], data_structure['HNL'], data_structure['prompt_lepton'],
            data_structure['dilepton_minus'], data_structure['dilepton_plus'], data_structure['neutrino'])
