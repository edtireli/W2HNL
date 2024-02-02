import os
import pickle 
import pyhepmc 
from modules.data_loading import * 
from parameters.data_parameters import * 
from parameters.experimental_parameters import *

def load_pickle_files(data_folder):
    files_to_load = [
        "momentum_prompt.pkl",
        "momentum_displaced_plus.pkl",
        "momentum_displaced_minus.pkl",
        "momentum_boson.pkl",
        "momentum_neutrino.pkl"
        "momentum_HNL.pkl"
    ]

    loaded_data = {}
    for file_name in files_to_load:
        file_path = os.path.join(data_folder, 'HEPMC', file_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                # Extract the variable name from the file name by removing the extension
                var_name = os.path.splitext(file_name)[0]
                # Load the pickle file and assign it to the corresponding key in the dictionary
                loaded_data[var_name] = pickle.load(file)
        else:
            print(f"File {file_name} does not exist. Skipping...")

    return loaded_data


def check_and_load_files(data_path):
    # Define the files to check based on the pid_HNL
    files_to_check = [
        f"{data_path}/HEPMC/momentum_prompt.pkl",
        f"{data_path}/HEPMC/momentum_displaced_plus.pkl",
        f"{data_path}/HEPMC/momentum_displaced_minus.pkl",
        f"{data_path}/HEPMC/momentum_boson.pkl",
        f"{data_path}/HEPMC/momentum_neutrino.pkl",
        f"{data_path}/HEPMC/momentum_HNL.pkl"
    ]
    
    # Check if all files exist
    all_files_exist = all(os.path.exists(file) for file in files_to_check)
    
    # If all files exist, load them
    if all_files_exist:
        print("All required files exist. Loading...")
        loaded_data = {}
        for file in files_to_check:
            with open(file, 'rb') as f:
                # The key is the basename of the file without extension for easy reference
                key = os.path.splitext(os.path.basename(file))[0]
                loaded_data[key] = pickle.load(f)
        return True, loaded_data  # Return a flag indicating success and the loaded data
    
    return False, None  # If not all files exist, return a flag indicating failure and no data

def save_to_pickle(data, name, data_folder):
    # Construct the full path for the directory where the pickle file will be saved
    directory_path = os.path.join(data_folder, 'HEPMC')
    
    # Ensure the directory exists
    os.makedirs(directory_path, exist_ok=True)
    
    # Construct the full path for the pickle file
    file_path = os.path.join(directory_path, name + '.pkl')
    
    # Open the file and save the data
    with open(file_path, 'wb') as output_file:
        pickle.dump(data, output_file)
    
    print(f"Data saved to {file_path}")

def HEPMC_data_processing(folder):
    print('---------------------------- HEPMC Data processing --------------------------')
    files_exist, loaded_data = check_and_load_files(folder)
    if files_exist:
        print('Using previously stored pre-analysed momentum data...')
        momentum_prompt = loaded_data['momentum_prompt']
        momentum_displaced_plus = loaded_data['momentum_displaced_plus']
        momentum_displaced_minus = loaded_data['momentum_displaced_minus']
        momentum_boson = loaded_data['momentum_boson']
        momentum_neutrino = loaded_data['momentum_neutrino']
        momentum_HNL = loaded_data['momentum_HNL']
    else:
        print('No previously stored pre-analysed momentum data - running analysis:')
        def has_W_children(particle):
            return any(child.abs_pid == pid_boson and any(grandchild.abs_pid == pid_HNL for grandchild in child.end_vertex.particles_out) for child in particle.children)
    
        def has_particle_parent(particle, pid):
            return any(parent.abs_pid == pid for parent in particle.parents)
        
        def filter_particles(event, conditions):
            """Filter particles based on a set of conditions (functions)."""
            return [particle for particle in event.particles if all(condition(particle) for condition in conditions)]
        
        event_files = sorted([f for f in os.listdir(os.path.join(folder, 'Events')) if f.endswith('.txt') and not f.startswith('.')])
        number_of_events = len(HNL_mass)
        if number_of_events > 0:
            selected_file = event_files[0]
            with open(os.path.join(folder, 'Events', selected_file), 'r') as f:
                lines = f.readlines()
                momentum_boson, momentum_HNL, momentum_prompt, momentum_displaced_minus, momentum_displaced_plus, momentum_neutrino = [], [], [], [], [], []

        # Assuming the event files dictate which HEPMC files to load
        for i in range(number_of_events):  # number_of_events needs to be defined or replaced with actual logic
            filename_hepmc = f"{folder}/Events/run_{i+1:02}/tag_1_pythia8_events.hepmc.gz"
            print(f'Reading HEPMC: {i+1}/{number_of_events}')
            
            with pyhepmc.open(filename_hepmc) as f:
                events = [event for event in f]
        
            # Process events
            for event in events:
                filtered_HNL             = filter_particles(event, [lambda p: p.abs_pid == pid_HNL, has_W_children])
                filtered_boson           = filter_particles(event, [lambda p: p.abs_pid == pid_boson, lambda p: has_particle_parent(p, pid_HNL)])
                filtered_displaced_plus  = filter_particles(event, [lambda p: p.pid     == -pid_displaced_lepton, lambda p: has_particle_parent(p, pid_HNL)])
                filtered_displaced_minus = filter_particles(event, [lambda p: p.pid     == pid_displaced_lepton, lambda p: has_particle_parent(p, pid_HNL)])
                filtered_prompt          = filter_particles(event, [lambda p: p.abs_pid == pid_prompt_lepton, lambda p: has_particle_parent(p, pid_boson)])
                filtered_neutrinos       = filter_particles(event, [lambda p: p.abs_pid == pid_neutrino, lambda p: has_particle_parent(p, pid_HNL)])

                # Append filtered particles to their respective lists
                momentum_HNL.append(filtered_HNL)
                momentum_prompt.append(filtered_prompt)
                momentum_displaced_minus.append(filtered_displaced_minus)
                momentum_displaced_plus.append(filtered_displaced_plus)
                momentum_boson.append(filtered_boson)
                momentum_neutrino.append(filtered_neutrinos)
        
        # Saving the files as pickle files so that HEPMC no longer is used
        save_to_pickle(momentum_boson, 'momentum_boson', folder)
        save_to_pickle(momentum_HNL, 'momentum_HNL', folder)
        save_to_pickle(momentum_prompt, 'momentum_prompt', folder)
        save_to_pickle(momentum_displaced_minus, 'momentum_displaced_minus', folder)
        save_to_pickle(momentum_displaced_plus, 'momentum_displaced_plus', folder)
        save_to_pickle(momentum_neutrino, 'momentum_neutrino', folder)

        
    return momentum_boson, momentum_HNL, momentum_prompt, momentum_displaced_minus, momentum_displaced_plus, momentum_neutrino