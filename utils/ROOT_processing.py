import uproot
import awkward as ak
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from parameters.data_parameters import *
from parameters.experimental_parameters import *

import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

pid_codes = {
    "W_boson": pid_boson,
    "HNL": pid_HNL,
    "prompt_lepton": pid_prompt_lepton,
    "dilepton_minus": pid_displaced_lepton,  # Assuming the negative charge
    "dilepton_plus": pid_displaced_lepton,    # Assuming the positive charge
    "neutrino": pid_neutrino,
}

def root_data_processing(data_folder):
    # Prepare to collect the data for each particle type
    particle_data = {key: [] for key in pid_codes}

    # Create a path pattern to match all relevant ROOT files
    path_pattern = os.path.join(data_folder, "Events/run_*/unweighted_events.root")
    root_files = glob(path_pattern)
    
    # Debug: print the files found
    print("Files found:", root_files)
    if not root_files:
        print("No files found, check the path pattern and directory structure.")

    # Process each file
    for file_path in tqdm(root_files):
        with uproot.open(file_path)["Delphes;1"]["Particle"] as particles:
            pids = particles["Particle.PID"].array(library="ak")
            px = particles["Particle.Px"].array(library="ak")
            py = particles["Particle.Py"].array(library="ak")
            pz = particles["Particle.Pz"].array(library="ak")
            energy = particles["Particle.E"].array(library="ak")

            # Process each particle type
            for name, pid in pid_codes.items():
                indices = ak.where(abs(pids) == pid)
                if len(indices[0]) > 0:  # Ensure there are entries
                    momentum = ak.zip({
                        "E": energy[indices],
                        "px": px[indices],
                        "py": py[indices],
                        "pz": pz[indices]
                    })
                    particle_data[name].append(momentum)

    # Concatenate the data across all files, avoiding empty concatenation
    concatenated_data = {key: ak.concatenate(values) if values else ak.Array([]) for key, values in particle_data.items()}
    
    return (
        concatenated_data['W_boson'],
        concatenated_data['HNL'],
        concatenated_data['prompt_lepton'],
        concatenated_data['dilepton_minus'],
        concatenated_data['dilepton_plus'],
        concatenated_data['neutrino']
    )
