#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
from os import path
from parameters.data_parameters import *
import re
from dataclasses import dataclass, field
from xml.etree import ElementTree

from skhep.math import LorentzVector
import gzip
import shutil
from tqdm import tqdm

@dataclass
class Particle:
    pdgid: int
    px: float
    py: float
    pz: float
    energy: float
    mass: float
    spin: float
    status: int
    vtau: float
    parent1: int
    parent2: int

    def p4(self):
        return LorentzVector(self.energy, self.px, self.py, self.pz)


@dataclass
class Event:
    particles: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    scale: float = -1

    def add_particle(self, particle):
        self.particles.append(particle)


class LHEReader():
    def __init__(self, file_path, weight_mode='list', weight_regex = '.*'):
        '''
        Constructor.

        :param file_path: Path to input LHE file
        :type file_path: string
        :param weight_mode: Format to return weights as. Can be dict or list. If dict, weight IDs are used as keys.
        :type weight_mode: string
        :param weight_regex: Regular expression to select weights to be read. Defaults to reading all.
        :type weight_regex: string
        '''
        self.file_path = file_path
        self.iterator = ElementTree.iterparse(self.file_path,events=('start','end'))
        self.current = None
        self.current_weights = None

        assert(weight_mode in ['list','dict'])
        self.weight_mode = weight_mode
        self.weight_regex = re.compile(weight_regex)

    def unpack_from_iterator(self):
        # Read the lines for this event
        lines = self.current[1].text.strip().split("\n")

        # Create a new event
        event = Event()
        event.scale = float(lines[0].strip().split()[3])
        event.weights = self.current_weights

        # Read header
        event_header = lines[0].strip()
        num_part = int(event_header.split()[0].strip())

        # Iterate over particle lines and push back
        for ipart in range(1, num_part+1):
            part_data = lines[ipart].strip().split()
            p = Particle(pdgid = int(part_data[0]),
                        status = int(part_data[1]),
                        parent1 = int(part_data[2]),
                        parent2 = int(part_data[3]),
                        px = float(part_data[6]),
                        py = float(part_data[7]),
                        pz = float(part_data[8]),
                        energy = float(part_data[9]),
                        mass = float(part_data[10]),
                        vtau = float(part_data[11]),
                        spin = int(float(part_data[12])))
            event.add_particle(p)

        return event

    def __iter__(self):
        return self

    def __next__(self):
        # Clear XML iterator
        if(self.current):
            self.current[1].clear()

        # Find beginning of new event in XML
        element = next(self.iterator)
        while element[1].tag != "event":
            element = next(self.iterator)

        # Loop over tags in this event
        element = next(self.iterator)

        if self.weight_mode == 'list':
            self.current_weights = []
        elif self.weight_mode == 'dict':
            self.current_weights = {}

        while not (element[0]=='end' and element[1].tag == "event"):
            if element[0]=='end' and element[1].tag == 'wgt':

                # If available, use "id" identifier as
                # 1. filter which events to read
                # 2. key for output dict
                weight_id = element[1].attrib.get('id','')

                if self.weight_regex.match(weight_id):
                    value = float(element[1].text)
                    if self.weight_mode == 'list':
                        self.current_weights.append(value)
                    elif self.weight_mode == 'dict':
                        self.current_weights[weight_id] = value
            element = next(self.iterator)

        # Find end up this event in XML
        # use it to construct particles, etc
        while not (element[0]=='end' and element[1].tag == "event"):
            element = next(self.iterator)
        self.current = element

        return self.unpack_from_iterator()
    
import numpy as np
import os
from glob import glob

def unzip_lhe_file(lhe_gz_path):
    """
    Unzips an LHE file given its .gz path.
    """
    with gzip.open(lhe_gz_path, 'rb') as f_in:
        with open(lhe_gz_path[:-3], 'wb') as f_out:  # Removes the .gz extension for the output file
            shutil.copyfileobj(f_in, f_out)
    os.remove(lhe_gz_path)  # Optionally remove the gz file after extraction

# Define a helper function to process and extract particle data from each event
def process_event(event, prompt_lepton_flavour):
    # Initialize the structure for holding extracted data
    extracted_data = {
        'HNL': [],
        'prompt_lepton': [],
        'dilepton_minus': [],
        'dilepton_plus': [],
        'W_boson': [],
        'neutrino': []  # Assuming interest in neutrinos as well
    }
    
    # Iterate over each particle in the event
    for particle in event.particles:
        # Extract HNL particles
        if abs(particle.pdgid) == pid_HNL:
            extracted_data['HNL'].append([particle.energy, particle.px, particle.py, particle.pz])
        
        # Extract W bosons
        if abs(particle.pdgid) == abs(pid_boson):
            extracted_data['W_boson'].append([particle.energy, particle.px, particle.py, particle.pz])

        # Extract prompt leptons
        if abs(particle.pdgid) == pid_prompt_lepton:
            extracted_data['prompt_lepton'].append([particle.energy, particle.px, particle.py, particle.pz])

        # Extract displaced leptons (assuming symmetry in PID for + and -)
        if abs(particle.pdgid) == pid_displaced_lepton:
            print('hello')
            if particle.pdgid > 0:
                extracted_data['dilepton_plus'].append([particle.energy, particle.px, particle.py, particle.pz])
            else:
                extracted_data['dilepton_minus'].append([particle.energy, particle.px, particle.py, particle.pz])

        # Extract neutrinos
        if abs(particle.pdgid) == pid_neutrino:
            extracted_data['neutrino'].append([particle.energy, particle.px, particle.py, particle.pz])

    return extracted_data
    
def save_processed_data(data, file_path):
    np.savez_compressed(os.path.join(file_path, data_folder), **data)

def load_processed_data(file_path):
    if os.path.exists(file_path):
        print('[Loaded files] Preprocessed LHE data')
        with np.load(file_path) as data:
            return {key: data[key] for key in data}
    return None

def LHE_data_processing(folder, prompt_length, prompt_lepton_flavour):
    event_dirs = sorted(glob(os.path.join(folder, 'Events', 'run_*')), key=lambda x: int(x.split('_')[-1]))
    total_events_to_process = min(len(event_dirs), len(mass_hnl)) * prompt_length
    progress_bar = tqdm(total=total_events_to_process, desc='Processing LHE Files')

    # Initialize data structure for processed data
    data_structure = {
        'HNL': [],
        'prompt_lepton': [],
        'dilepton_minus': [],
        'dilepton_plus': [],
        'W_boson': [],
        'neutrino': []
    }

    # Check if processed data already exists
    current_directory = os.getcwd()                                         # Current path
    data_path         = os.path.join(current_directory,'data', data_folder) # Data folder path
    name = 'processed_LHE_data.npz'
    processed_data = load_processed_data(os.path.join(data_path, name))
    if processed_data:
        progress_bar.close()
        return processed_data

    for dir_path in event_dirs:
        lhe_gz_path = os.path.join(dir_path, 'unweighted_events.lhe.gz')
        if path.exists(lhe_gz_path):
            unzip_lhe_file(lhe_gz_path)
    
        lhe_file_path = os.path.join(dir_path, 'unweighted_events.lhe')
        reader = LHEReader(lhe_file_path)
        scan_data = {key: [] for key in data_structure.keys()}

        for event_index, event in enumerate(reader):
            if event_index >= prompt_length:
                break
            event_data = process_event(event, prompt_lepton_flavour)
            for key in scan_data:
                scan_data[key].append(event_data[key])
            progress_bar.update(1)

        for key in data_structure:
            data_structure[key].append(np.array(scan_data[key]))

    progress_bar.close()

    for key in data_structure:
        flat_data = [np.concatenate(scan, axis=0) for scan in data_structure[key] if len(scan) > 0]
        data_structure[key] = np.stack(flat_data) if flat_data else np.empty((0, prompt_length, 4))

    # Save processed data before returning
    #save_processed_data(data_structure, name) # currently bugged implementation
    print('    prompt lepton shape: ', np.shape(data_structure['prompt_lepton']))
    print('di-lepton (minus) shape: ',data_structure['dilepton_minus'])
    print('di-lepton (plus) shape: ',data_structure['dilepton_plus'])
    print('HNL              shape: ',data_structure['HNL'])



    return (data_structure['W_boson'], data_structure['HNL'], data_structure['prompt_lepton'],
            data_structure['dilepton_minus'], data_structure['dilepton_plus'], data_structure['neutrino'])

