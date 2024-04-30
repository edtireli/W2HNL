import uproot
import awkward as ak
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from parameters.data_parameters import *
from parameters.experimental_parameters import *

class Particle:
    def __init__(self, pid, energy, px, py, pz):
        self.pid = pid
        self.energy = energy
        self.px = px
        self.py = py
        self.pz = pz

class Event:
    def __init__(self):
        self.particles = []

    def add_particle(self, particle):
        self.particles.append(particle)

def process_event(event):
    data = {
        'W_boson': [],
        'HNL': [],
        'prompt_lepton': [],
        'dilepton_minus': [],
        'dilepton_plus': [],
        'neutrino': []
    }
    for particle in event.particles:
        pid = particle.pid
        vector = [particle.energy, particle.px, particle.py, particle.pz]
        if abs(pid) == abs(pid_boson):
            data['W_boson'].append(vector)
        elif abs(pid) == pid_HNL:
            data['HNL'].append(vector)
        elif abs(pid) == pid_prompt_lepton:
            data['prompt_lepton'].append(vector)
        elif abs(pid) == pid_displaced_lepton:
            if pid > 0:
                data['dilepton_plus'].append(vector)
            else:
                data['dilepton_minus'].append(vector)
        elif abs(pid) == pid_neutrino:
            data['neutrino'].append(vector)
    return data

def root_data_processing(main_folder_path):
    mass_dirs = sorted(glob(os.path.join(main_folder_path, 'run_*')))
    mass_data = {key: [] for key in ['W_boson', 'HNL', 'prompt_lepton', 'dilepton_minus', 'dilepton_plus', 'neutrino']}

    for run_dir in mass_dirs:
        file_path = os.path.join(run_dir, 'unweighted_events.root')
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with uproot.open(file_path)["Delphes;1"]["Particle"] as particles:
            pids = particles["Particle.PID"].array(library="np")
            px = particles["Particle.Px"].array(library="np")
            py = particles["Particle.Py"].array(library="np")
            pz = particles["Particle.Pz"].array(library="np")
            energy = particles["Particle.E"].array(library="np")

            event = Event()
            for i in range(len(pids)):
                particle = Particle(pids[i], energy[i], px[i], py[i], pz[i])
                event.add_particle(particle)

            event_data = process_event(event)
            for key in mass_data:
                mass_data[key].append(event_data[key])

    for key in mass_data:
        mass_data[key] = np.array(mass_data[key])  # Convert each list to numpy array for better handling

    return mass_data