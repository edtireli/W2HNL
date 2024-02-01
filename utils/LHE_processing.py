#!/usr/bin/env python3
# Importing useful packages
import matplotlib.pyplot as plt
import pandas as pd
from termcolor import colored
import numpy as np
import random as rd
from sys import exit
import math
import os
import shutil
import os.path
from os import path
import time

import re
from dataclasses import dataclass, field
from xml.etree import ElementTree

from skhep.math import LorentzVector

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
        return LorentzVector(self.px, self.py, self.pz, self.energy)


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
    
def LHE_data_processing(folder, prompt_length, prompt_lepton_flavour):     
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
    
    if path.exists( folder + '/HNL_4mom/HNL_4mom_' + f'{n_mass_scans:02}' + '.txt')==True:
        p_HNL=[]
        p_mu_prompt=[]
        p_mu_dimu_minus=[]
        p_mu_dimu_plus=[]
        for i in range(n_mass_scans):
            print('----> reading MG scan .txt file' + str(i+1), end='\r')
            p_HNL.append(np.loadtxt( folder + '/HNL_4mom/HNL_4mom_' + f'{i+1:02}' + '.txt',max_rows=prompt_length))
            p_mu_prompt.append(np.loadtxt( folder + '/promt_mu_4mom/prompt_mu_4mom_' + f'{i+1:02}' + '.txt',max_rows=prompt_length))
            p_mu_dimu_minus.append(np.loadtxt( folder + '/dimu_minus_4mom/dimu_minus_4mom_' + f'{i+1:02}' + '.txt',max_rows=prompt_length))
            p_mu_dimu_plus.append(np.loadtxt( folder + '/dimu_plus_4mom/dimu_plus_4mom_' + f'{i+1:02}' + '.txt',max_rows=prompt_length))
    else:
        p_HNL=[]
        p_W=[]
        p_mu_prompt=[]
        p_mu_dimu_minus=[]
        p_mu_dimu_plus=[]
        
        def flatten(l):
            while len(l) == 1 and type(l[0]) == list:
                l = l.pop()
            return l
    
        if prompt_lepton_flavour==2:
            for i in range(n_mass_scans):
                print('reading MG scan LHE file' + str(i+1), end='\r')
                p_HNL.append([])
                p_mu_prompt.append([])
                p_mu_dimu_minus.append([])
                p_mu_dimu_plus.append([])
                if path.exists( folder + '/Events/run_' + f'{i+1:02}' + '/unweighted_events.lhe.gz')==True:
                    os.system('gzip -d ' +folder+ '/Events/run_' + f'{i+1:02}' + '/unweighted_events.lhe.gz')
                reader = LHEReader( folder + '/Events/run_' + f'{i+1:02}' + '/unweighted_events.lhe')
                for iev, event in enumerate(reader):
                    if iev <= prompt_length:
                        for j in event.particles:
                            if (j.pdgid==9990014 or j.pdgid==9900014 or j.pdgid==9900012) and j.parent1==1 and j.parent2==2:
                                p_HNL[i].append(np.array([j.energy,j.px, j.py,j.pz]))
                                for k in event.particles:
                                    if k.parent1==1 and k.parent2==2 and abs(k.pdgid)==13:
                                        p_mu_prompt[i].append(np.array([k.energy,k.px, k.py,k.pz]))
                                    if (k.parent1==3 and k.parent2==3) and k.pdgid==13:
                                        p_mu_dimu_minus[i].append(np.array([k.energy,k.px, k.py,k.pz]))
                                    if (k.parent1==3 and k.parent2==3) and k.pdgid==13:
                                        p_mu_dimu_minus[i].append(np.array([k.energy,k.px, k.py,k.pz]))    
                                    if (k.parent1==3 and k.parent2==3) and k.pdgid==-13:
                                        p_mu_dimu_plus[i].append(np.array([k.energy,k.px, k.py,k.pz]))
                                    elif (k.parent1==4 and k.parent2==4) and k.pdgid==-13:
                                        p_mu_dimu_plus[i].append(np.array([k.energy,k.px, k.py,k.pz]))    
                            elif (j.pdgid==9990014 or j.pdgid==9900014 or j.pdgid==9900012) and j.parent1==3 and j.parent2==3:
                                p_HNL[i].append(np.array([j.energy,j.px, j.py,j.pz]))
                                for k in event.particles:
                                    if k.parent1==3 and k.parent2==3 and abs(k.pdgid)==13:
                                        p_mu_prompt[i].append(np.array([k.energy,k.px, k.py,k.pz]))
                                    if k.pdgid==13 and k.parent1==4 and k.parent2==4:
                                        p_mu_dimu_minus[i].append(np.array([k.energy,k.px, k.py,k.pz]))   
                                    if k.parent1==4 and k.parent2==4 and k.pdgid==-13:
                                        p_mu_dimu_plus[i].append(np.array([k.energy,k.px, k.py,k.pz])) 
                                    elif abs(k.pdgid)==24 and k.parent1==4 and k.parent2==4:
                                        for l in event.particles:
                                            if l.parent1==5 and l.parent2==5 and l.pdgid==-13:
                                                p_mu_dimu_plus[i].append(np.array([l.energy,l.px, l.py,l.pz]))
                                    elif abs(k.pdgid)==23 and k.parent1==4 and k.parent2==4:
                                        for l in event.particles:
                                            if l.parent1==5 and l.parent2==5 and l.pdgid==-13:
                                                p_mu_dimu_plus[i].append(np.array([l.energy,l.px, l.py,l.pz])) 
                if i==0:
                     os.mkdir( folder + '/HNL_4mom')
                     os.mkdir( folder + '/promt_mu_4mom')
                     os.mkdir( folder + '/dimu_minus_4mom')
                     os.mkdir( folder + '/dimu_plus_4mom')
                np.savetxt( folder + '/HNL_4mom/HNL_4mom_' + f'{i+1:02}' + '.txt', p_HNL[i])
                np.savetxt( folder + '/promt_mu_4mom/prompt_mu_4mom_' + f'{i+1:02}' + '.txt', p_mu_prompt[i])
                np.savetxt( folder + '/dimu_minus_4mom/dimu_minus_4mom_' + f'{i+1:02}' + '.txt', p_mu_dimu_minus[i])
                np.savetxt( folder + '/dimu_plus_4mom/dimu_plus_4mom_' + f'{i+1:02}' + '.txt', p_mu_dimu_plus[i])                                                
        else:                                    
            for i in range(n_mass_scans):
                print('reading MG scan LHE file' + str(i+1), end='\r')
                p_HNL.append([])
                p_W.append([])
                p_mu_prompt.append([])
                p_mu_dimu_minus.append([])
                p_mu_dimu_plus.append([])
                if path.exists( folder + '/Events/run_' + f'{i+1:02}' + '/unweighted_events.lhe.gz')==True:
                    os.system('gzip -d ' +folder+ '/Events/run_' + f'{i+1:02}' + '/unweighted_events.lhe.gz')
                reader = LHEReader( folder + '/Events/run_' + f'{i+1:02}' + '/unweighted_events.lhe')
                for iev, event in enumerate(reader):
                    if iev <= prompt_length:
                        pHNL_ = filter(lambda x: abs(x.pdgid)== 9990012 or abs(x.pdgid)== 9990014 or abs(x.pdgid)== 9990016 or abs(x.pdgid)==9900012 or abs(x.pdgid)==9900014 or abs(x.pdgid)==9900016, event.particles)
                        pmu1_dimu_minus = filter(lambda x: x.pdgid== pdg_di, event.particles)
                        pmu1_dimu_plus = filter(lambda x: x.pdgid== -pdg_di, event.particles)
                        pmu1_prompt = filter(lambda x: abs(x.pdgid)== pdg_prompt, event.particles)
                        W_filter = filter(lambda x: abs(x.pdgid)== 24, event.particles)
                        for p4hnl in map(lambda x: x.p4(), pHNL_):
                            p_HNL[i].append(np.array([p4hnl[3],p4hnl[0],p4hnl[1],p4hnl[2]]))
                        for p4muprompt in map(lambda x: x.p4(), pmu1_prompt):
                            p_mu_prompt[i].append(np.array([p4muprompt[3],p4muprompt[0],p4muprompt[1],p4muprompt[2]]))
                        for p4dimu in map(lambda x: x.p4(), pmu1_dimu_minus):
                            p_mu_dimu_minus[i].append(np.array([p4dimu[3],p4dimu[0],p4dimu[1],p4dimu[2]]))  
                        for p4dimu in map(lambda x: x.p4(), pmu1_dimu_plus):
                            p_mu_dimu_plus[i].append(np.array([p4dimu[3],p4dimu[0],p4dimu[1],p4dimu[2]]))
   
        
                if i==0:
                    os.mkdir( folder + '/HNL_4mom')
                    os.mkdir( folder + '/promt_mu_4mom')
                    os.mkdir( folder + '/dimu_minus_4mom')
                    os.mkdir( folder + '/dimu_plus_4mom')
                np.savetxt( folder + '/HNL_4mom/HNL_4mom_' + f'{i+1:02}' + '.txt', p_HNL[i])
                np.savetxt( folder + '/promt_mu_4mom/prompt_mu_4mom_' + f'{i+1:02}' + '.txt', p_mu_prompt[i])
                np.savetxt( folder + '/dimu_minus_4mom/dimu_minus_4mom_' + f'{i+1:02}' + '.txt', p_mu_dimu_minus[i])
                np.savetxt( folder + '/dimu_plus_4mom/dimu_plus_4mom_' + f'{i+1:02}' + '.txt', p_mu_dimu_plus[i])     
                         
    p_HNL = np.array(p_HNL) 
    p_mu_dimu_minus = np.array(p_mu_dimu_minus) 
    p_mu_dimu_plus = np.array(p_mu_dimu_plus) 
    p_mu_prompt = np.array(p_mu_prompt) 
    #p_W = np.array(p_W)
    return n_mass_scans,HNL_mass,  p_HNL, p_mu_prompt, p_mu_dimu_minus, p_mu_dimu_plus  