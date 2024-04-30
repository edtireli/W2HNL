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
        self.file_iter = iter(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        file_path = next(self.file_iter)
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
        if abs(pid) == abs(pid_boson):  # W boson
            data['W_boson'].append(vector)
        elif abs(pid) == pid_HNL:  # HNL
            data['HNL'].append(vector)
        elif abs(pid) == pid_prompt_lepton:  # Prompt lepton
            data['prompt_lepton'].append(vector)
        elif abs(pid) == pid_displaced_lepton:  # Displaced lepton
            if pid > 0:
                data['dilepton_plus'].append(vector)
            else:
                data['dilepton_minus'].append(vector)
        elif abs(pid) == pid_neutrino:  # Neutrino
            data['neutrino'].append(vector)
    return data

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

    return all_data
