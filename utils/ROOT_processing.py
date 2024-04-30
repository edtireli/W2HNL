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

class ROOTReader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = glob(os.path.join(folder_path, '*.root'))
        if not self.files:
            raise ValueError("No ROOT files found in the specified directory.")
        self.file_iter = iter(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        file_path = next(self.file_iter)
        return self.process_file(file_path)

    def process_file(self, file_path):
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
            return event

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
        # Debugging output to verify PID filtering
        print(f"Processing PID: {pid}")
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

def root_data_processing(folder_path):
    print(folder_path)
    reader = ROOTReader(folder_path)
    data_structure = {
        'W_boson': [],
        'HNL': [],
        'prompt_lepton': [],
        'dilepton_minus': [],
        'dilepton_plus': [],
        'neutrino': []
    }
    total_events_processed = 0
    for event in tqdm(reader, desc='Processing ROOT Files'):
        event_data = process_event(event)
        for key in data_structure:
            data_structure[key].extend(event_data[key])
        total_events_processed += 1

    # Final diagnostic to confirm data loading
    print(f"Total events processed: {total_events_processed}")
    for key, values in data_structure.items():
        print(f"Total {key}: {len(values)}")
        
    return (data_structure['W_boson'], data_structure['HNL'], data_structure['prompt_lepton'],
            data_structure['dilepton_minus'], data_structure['dilepton_plus'], data_structure['neutrino'])
