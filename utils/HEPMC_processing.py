import os
import pyhepmc 
import pickle 
from parameters.data_parameters import * 
from modules.data_loading import * 


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
        file_path = os.path.join(data_folder, '/HEPMC', file_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                # Extract the variable name from the file name by removing the extension
                var_name = os.path.splitext(file_name)[0]
                # Load the pickle file and assign it to the corresponding key in the dictionary
                loaded_data[var_name] = pickle.load(file)
        else:
            print(f"File {file_name} does not exist. Skipping...")

    return loaded_data


def check_and_load_files(data_path, HNL_id):
    # Define the files to check based on the HNL_id
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
        print("[+] All required files exist. Loading...")
        loaded_data = {}
        for file in files_to_check:
            with open(file, 'rb') as f:
                # The key is the basename of the file without extension for easy reference
                key = os.path.splitext(os.path.basename(file))[0]
                loaded_data[key] = pickle.load(f)
        return True, loaded_data  # Return a flag indicating success and the loaded data
    
    return False, None  # If not all files exist, return a flag indicating failure and no data

def save_to_pickle(data, name, data_folder):
    # Construct the full path for the pickle file
    file_path = os.path.join(data_folder, name + '.pkl')
    
    # Open the file and save the data
    with open(file_path, 'wb') as output_file:
        pickle.dump(data, output_file)
    
    print(f"Data saved to {file_path}")

def HEPMC_data_processing(folder, HNL_id):
    print('---------------------------- HEPMC Data processing --------------------------')
    files_exist, loaded_data = check_and_load_files(folder, HNL_id)
    if files_exist:
        print('Using previously stored momentum data...')
        momentum_prompt = loaded_data['momentum_prompt']
        momentum_displaced_plus = loaded_data['momentum_displaced_plus']
        momentum_displaced_minus = loaded_data['momentum_displaced_minus']
        momentum_boson = loaded_data['momentum_boson']
        momentum_neutrino = loaded_data['momentum_neutrino']
        momentum_HNL = loaded_data['momentum_HNL']
    else:
        print('No previously stored momentum data - running analysis:')
        def has_W_children(particle):
            return any(child.abs_pid == 24 and any(grandchild.abs_pid == HNL_id for grandchild in child.end_vertex.particles_out) for child in particle.children)
    
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
                HNLs_temp = filter_particles(event, [lambda p: p.abs_pid == HNL_id, has_W_children])
                W_HNL = filter_particles(event, [lambda p: p.abs_pid == 24, lambda p: has_particle_parent(p, HNL_id)])
                mus_temp_plus = filter_particles(event, [lambda p: p.pid == -13, lambda p: has_particle_parent(p, HNL_id)])
                mus_temp_minus = filter_particles(event, [lambda p: p.pid == 13, lambda p: has_particle_parent(p, HNL_id)])
                tau_temp = filter_particles(event, [lambda p: p.abs_pid == 15, lambda p: has_particle_parent(p, 24)])
                nus_temp = filter_particles(event, [lambda p: p.abs_pid == 16, lambda p: has_particle_parent(p, HNL_id)])

                # Append filtered particles to their respective lists
                momentum_HNL.append(HNLs_temp)
                momentum_prompt.append(tau_temp)
                momentum_displaced_minus.append(mus_temp_minus)
                momentum_displaced_plus.append(mus_temp_plus)
                momentum_boson.append(W_HNL)
                momentum_neutrino.append(nus_temp)
        
        # Saving the files as pickle files so that HEPMC no longer is used
        save_to_pickle(momentum_boson, 'momentum_boson', data_folder)
        save_to_pickle(momentum_HNL, 'momentum_HNL', data_folder)
        save_to_pickle(momentum_prompt, 'momentum_prompt', data_folder)
        save_to_pickle(momentum_displaced_minus, 'momentum_displaced_minus', data_folder)
        save_to_pickle(momentum_displaced_plus, 'momentum_displaced_plus', data_folder)
        save_to_pickle(momentum_neutrino, 'momentum_neutrino', data_folder)

        
    return momentum_boson, momentum_HNL, momentum_prompt, momentum_displaced_minus, momentum_displaced_plus, momentum_neutrino