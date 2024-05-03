import os
import uproot
import awkward as ak
import numpy as np
from tqdm import tqdm
from parameters.data_parameters import *
from parameters.experimental_parameters import *

def root_data_processing(base_folder):
    try:
        # Initialize the data structure with nested lists
        data_structure = {
            'W_boson': [],
            'HNL': [],
            'prompt_lepton': [],
            'dilepton_minus': [],
            'dilepton_plus': [],
            'neutrino': []
        }
        
        # Iterate over each run folder with tqdm progress bar
        for i in tqdm(range(1, len(mass_hnl) + 1), desc="Processing runs"):
            folder_name = f"{base_folder}/Events/run_{i:02d}"
            file_path = os.path.join(folder_name, "unweighted_events.root")
            
            with uproot.open(file_path)["Delphes;1"]["Particle"] as particles:
                # Load PID and momentum data
                pids = particles["Particle.PID"].array(library="ak")
                px = particles["Particle.Px"].array(library="ak")
                py = particles["Particle.Py"].array(library="ak")
                pz = particles["Particle.Pz"].array(library="ak")
                energy = particles["Particle.E"].array(library="ak")
                
                # Temporary storage for the current run
                temp_data = {
                    'W_boson': [],
                    'HNL': [],
                    'prompt_lepton': [],
                    'dilepton_minus': [],
                    'dilepton_plus': [],
                    'neutrino': []
                }
                
                # Process each particle type
                for particle_type, pid in [('W_boson', pid_boson), ('HNL', pid_HNL), 
                                           ('prompt_lepton', pid_prompt_lepton), 
                                           ('dilepton_minus', pid_displaced_lepton),
                                           ('dilepton_plus', pid_displaced_lepton), 
                                           ('neutrino', pid_neutrino)]:
                    indices = ak.where(abs(pids) == abs(pid))
                    four_momenta = ak.zip({
                        "E": energy[indices],
                        "px": px[indices],
                        "py": py[indices],
                        "pz": pz[indices]
                    })
                    
                    if 'dilepton' in particle_type:
                        charge_indices = ak.where(pids[indices] == pid)
                        if 'minus' in particle_type:
                            charge_indices = ak.where(pids[indices] == -pid)
                        four_momenta = four_momenta[charge_indices]
                    
                    temp_data[particle_type].append(four_momenta)
                
                # Store all four-momenta data for this run into the main data structure
                for key in data_structure:
                    data_structure[key].append(temp_data[key])
        
        # Convert lists to arrays for better handling in downstream analysis
        print("Converting lists to structured arrays...")
        for key in data_structure:
            # Convert each run's data into a single array per run, maintaining separation between runs
            structured_data = [ak.to_numpy(ak.concatenate(run_data)) for run_data in data_structure[key]]
            data_structure[key] = np.array(structured_data)
        
        print("Processing complete.")
        print({key: np.shape(data_structure[key]) for key in data_structure})  # Print shapes to verify structure
        return (data_structure['W_boson'], data_structure['HNL'], data_structure['prompt_lepton'],
                data_structure['dilepton_minus'], data_structure['dilepton_plus'], data_structure['neutrino'])
    
    except Exception as e:
        print(f"An error occurred: {e}")
        raise