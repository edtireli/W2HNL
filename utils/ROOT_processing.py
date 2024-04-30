import uproot
import awkward as ak
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# Particle Data and Event Classes
class Particle:
    def __init__(self, energy, px, py, pz):
        self.energy = energy
        self.px = px
        self.py = py
        self.pz = pz

class Event:
    def __init__(self):
        self.particles = []

    def add_particle(self, particle):
        self.particles.append(particle)

# ROOT File Reader Class
class ROOTReader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = glob(os.path.join(folder_path, '*.root'))

    def __iter__(self):
        return self

    def __next__(self):
        if not self.files:
            raise StopIteration
        file_path = self.files.pop(0)
        return self.process_file(file_path)

    def process_file(self, file_path):
        with uproot.open(file_path)["Delphes;1"]["Particle"] as particles:
            pids = particles["Particle.PID"].array(library="ak")
            px = particles["Particle.Px"].array(library="ak")
            py = particles["Particle.Py"].array(library="ak")
            pz = particles["Particle.Pz"].array(library="ak")
            energy = particles["Particle.E"].array(library="ak")
            event = Event()
            for i in range(len(pids)):
                particle = Particle(energy[i], px[i], py[i], pz[i])
                event.add_particle(particle)
            return event

# Processing Function
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
        pid = particle.pdgid
        vector = [particle.energy, particle.px, particle.py, particle.pz]
        if abs(pid) == 24:
            data['W_boson'].append(vector)
        elif abs(pid) == 9900016:
            data['HNL'].append(vector)
        # Add similar conditions for other particle types as defined
    return data

# Main Processing Loop
def root_data_processing(folder_path):
    reader = ROOTReader(folder_path)
    all_data = {
        'W_boson': [],
        'HNL': [],
        'prompt_lepton': [],
        'dilepton_minus': [],
        'dilepton_plus': [],
        'neutrino': []
    }
    for event in tqdm(reader, desc='Processing ROOT Files'):
        event_data = process_event(event)
        for key in all_data:
            all_data[key].extend(event_data[key])

    return (all_data['W_boson'], all_data['HNL'], all_data['prompt_lepton'], 
            all_data['dilepton_minus'], all_data['dilepton_plus'], all_data['neutrino'])

