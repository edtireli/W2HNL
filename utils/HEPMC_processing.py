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
    files_exist, loaded_data = check_and_load_files(folder, HNL_id)
    if files_exist:
        momentum_prompt = loaded_data['momentum_prompt']
        momentum_displaced_plus = loaded_data['momentum_displaced_plus']
        momentum_displaced_minus = loaded_data['momentum_displaced_minus']
        momentum_boson = loaded_data['momentum_boson']
        momentum_neutrino = loaded_data['momentum_neutrino']
        momentum_HNL = loaded_data['momentum_HNL']
    else:
        foo = os.listdir(folder + '/Events')
        if '.DS_Store' in foo:
            foo.remove('.DS_Store')
        if len(foo)-1<10:
            digits=1
            # less=0
            filename = 'scan_run_0[1-' + f'{len(foo)-1:{digits}}'.format(digits=digits) + '].txt'
            # out_of=f'{{}:01}'.format(n_mass_scans)
        else:
            digits=2
            # less=''
            filename = 'scan_run_[01-' + f'{len(foo)-1:{digits}}'.format(digits=digits) + '].txt'
            # out_of=f'{n_ mass_scans:02}'.format(n_mass_scans)
        with open(folder + '/Events/' + filename, 'r') as f:
            lines=f.readlines()
            data=[elm.split() for elm in lines[1:]]
        n_mass_scans = len(data)
        HNL_mass = [float(elm[1]) for elm in data] 
        momentum_boson, momentum_HNL, momentum_prompt, momentum_displaced_minus, momentum_displaced_plus, momentum_neutrino = [], [], [], [], [], []
        for i in range(n_mass_scans):
            print(f'[!] Loading HEPMC files: {i+1}/{n_mass_scans}')
            filename_hepmc = folder + '/Events/' + '/run_' +f'{i+1:02}' +'/tag_1_pythia8_events.hepmc.gz'
            events=[]
            with pyhepmc.open(filename_hepmc) as f:
                for event in f: 
                    #event = f.read()
                    events.append(event)
            event
            def has_W_children(p): #this could be problematic, if multiple W's are created from quarks, not necessarily decaying further to HNL
                for q in p.children:
                    if q.abs_pid == 24:
                        for i in q.end_vertex.particles_out:
                            if i==HNL_id:
                                return any(q)
                return any(q.abs_pid == 24 for q in p.children)
            def has_p_parent(p):
                return any(q.abs_pid == 2212 for q in p.children)
            def has_HNL_children(p):
                return any(q.abs_pid == HNL_id for q in p.children)
            def has_mu_children(p):
                return any(q.pid == 13 for q in p.children)
            def has_antimu_children(p):
                return any(q.pid == -13 for q in p.children)
            def has_vt_children(p):
                return any(q.abs_pid == 16 for q in p.children)
            def has_W_parents(p):
                return any(q.abs_pid == 24 for q in p.parents)
            def hasnt_W_parents(p):
                return any(q.abs_pid != 24 for q in p.parents)
            def has_HNL_parents(p):
                return any(q.abs_pid == HNL_id for q in p.parents)
            def find_end2(arr, status_code):
                momenta=[]
                if type(status_code)==list: 
                    for i in range(len(arr)):
                        momenta.append([])
                        for j in arr[i]:
                            for l in status_code:
                                #if (j.status!=l):
                                    #for k in j.end_vertex.particles_out:
                                        #momenta[i].append(k)
                                if j.status==l:
                                    momenta[i].append(j)
                else: 
                    for i in range(len(arr)):
                        momenta.append([])
                        for j in arr[i]:
                            if (j.status!=status_code):
                                for k in j.end_vertex.particles_out:
                                    if k.status==status_code:
                                        momenta[i].append(k)
                            if j.status==status_code:
                                momenta[i].append(j)                
                return momenta
            def find_end(arr, pid, status_code):
                momenta=[]
                if type(status_code)==list: 
                    for i in range(len(arr)):
                        momenta.append([])
                        for j in arr[i]:
                            for l in status_code:
                                if (j.status!=l):
                                    for k in j.end_vertex.particles_out:
                                        if k.abs_pid == pid:
                                            momenta[i].append(k)
                                if j.status==l:
                                    momenta[i].append(j)
                else: 
                    for i in range(len(arr)):
                        momenta.append([])
                        for j in arr[i]:
                            if (j.status!=status_code):
                                for k in j.end_vertex.particles_out:
                                    if k.abs_pid == pid:
                                        momenta[i].append(k)
                            if j.status==status_code:
                                momenta[i].append(j)                
                return momenta
            def find_end_HNL(arr, HNL_id):
                momenta=[]
                for i in range(len(arr)):
                    momenta.append([])
                    for j in arr[i]:
                        for k in j.end_vertex.particles_out:
                            if k.abs_pid == HNL_id:
                                find_end_HNL(k,HNL_id)
                            else:
                                momenta[i].append(k)               
                return momenta   
            protons = []
            w_bosons_hnl = []
            w_bosons = []
            HNLs_temp = []
            mus_temp_plus = []
            mus_temp_minus = []
            tau_temp = []
            nus_temp = []
            quarkons_temp = []
            w_bosons_hnl = []
            for evts in events:
                filterHNL = [p for p in evts.particles if (p.abs_pid == HNL_id and has_mu_children(p) and has_antimu_children(p) and has_vt_children(p))]
                filterW_HNL = [p for p in evts.particles if (has_HNL_children(p) and p.abs_pid == 24)]
                filtermu_plus = [p for p in evts.particles if (has_HNL_parents(p) and p.pid == -13)]
                filtermu_minus = [p for p in evts.particles if (has_HNL_parents(p) and p.pid == 13)]
                filtertau = [p for p in evts.particles if (has_W_parents(p) and p.abs_pid == 15)]
                filternu = [p for p in evts.particles if (p.abs_pid == 16 and has_HNL_parents(p))]
                filterW_HNL = [p for p in evts.particles if has_HNL_children(p) if p.abs_pid == 24]
                tau_temp.append(filtertau)
                nus_temp.append(filternu)
                w_bosons_hnl.append(filterW_HNL)
                HNLs_temp.append(filterHNL)
                mus_temp_plus.append(filtermu_plus)   
                mus_temp_minus.append(filtermu_minus)   
                w_bosons_hnl.append(filterW_HNL)
    
            momentum_HNL.append(HNLs_temp)
            taus = tau_temp# find_end(tau_temp, 15, 2)
            momentum_prompt.append(taus)
            momentum_displaced_minus.append(mus_temp_minus)
            momentum_displaced_plus.append(mus_temp_plus)
            momentum_boson.append(w_bosons_hnl)
            momentum_neutrino.append(nus_temp)
    
    # Saving the files as pickle files so that HEPMC no longer is used
    save_to_pickle(momentum_boson, 'momentum_boson', data_folder)
    save_to_pickle(momentum_HNL, 'momentum_HNL', data_folder)
    save_to_pickle(momentum_prompt, 'momentum_prompt', data_folder)
    save_to_pickle(momentum_displaced_minus, 'momentum_displaced_minus', data_folder)
    save_to_pickle(momentum_displaced_plus, 'momentum_displaced_plus', data_folder)
    save_to_pickle(momentum_neutrino, 'momentum_neutrino', data_folder)

        
    return momentum_boson, momentum_HNL, momentum_prompt, momentum_displaced_minus, momentum_displaced_plus, momentum_neutrino