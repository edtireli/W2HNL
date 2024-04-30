import uproot
import awkward as ak
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from parameters.data_parameters import *
from parameters.experimental_parameters import *

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
        self.file_iter = iter(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            file_path = next(self.file_iter)
            return self.process_file(file_path)
        except StopIteration:
            raise StopIteration

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
        pid = abs(particle.pdgid)
        vector = [particle.energy, particle.px, particle.py, particle.pz]
        if pid == 24:
            data['W_boson'].append(vector)
        elif pid == 9900016:
            data['HNL'].append(vector)
        # Implement conditions for other particle types as defined in your project
    return data

# Main Processing Loop
def root_data_processing(folder):
    event_dirs = sorted(glob(os.path.join(folder, 'run_*')), key=lambda x: int(x.split('_')[-1]))
    total_events_to_process = min(len(event_dirs), len(mass_hnl)) * prompt_length
    progress_bar = tqdm(total=total_events_to_process, desc='Processing ROOT Files')

    data_structure = {
        'W_boson': [],
        'HNL': [],
        'prompt_lepton': [],
        'dilepton_minus': [],
        'dilepton_plus': [],
        'neutrino': []
    }

    for dir_path in event_dirs:
        reader = ROOTReader(dir_path)
        for event_index, event in enumerate(reader):
            if event_index >= prompt_length:
                break
            event_data = process_event(event)
            for key in data_structure:
                data_structure[key].extend(event_data[key])
            progress_bar.update(1)

    progress_bar.close()

    return (data_structure['W_boson'], data_structure['HNL'], data_structure['prompt_lepton'],
            data_structure['dilepton_minus'], data_structure['dilepton_plus'], data_structure['neutrino'])
