# A parameter file for various dimensions and variables to do with the experimental constraints, such as cut conditions, efficiencies etc.
import numpy as np 

pT_minimum = '0 GeV' # Minimum transverse momentum
invmass_minimum = '5 GeV'
invmass_cut_type = 'nontrivial' # nontrivial to consider piecewise function, trivial to consider a flat value that uses the above invmass_minimum regardless of r_dv
invmass_experimental = False # Changes the way the invariant mass is calculated. True uses HEP method with eta and phi, False uses theory version with full 4-mom. from data. 

deltaR_minimum = 0.05 # Minimum angular seperation

pseudorapidity_minimum = 0   # Minimum pseudorapidity criteria for HNL decay 
pseudorapidity_maximum = 2.4 # Maximum pseudorapidity criteria for HNL decay

cut_type_dv = 'cylinder' # Either cylinder or sphere
r_min = '0 mm' # Minimum distance from interaction point (IP) to consider HNL decays/decay volume
r_max_t = '1000 m' # Maximum distance from interaction point (IP) to consider HNL decays/decay volume in the transverse direction
r_max_l = '1000 m' # Maximum distance from interaction point (IP) to consider HNL decays/decay volume in the longitudinal (z) direction along beamline

luminosity = 300 # 1/fb
production_minimum   = 3 # The minimum number of HNLs produced within the parameter region to consider it a success (production_minimum = 3 for 95% CL)
production_minimum_secondary = 10 # A secondary production minimum

mixing = np.logspace(0,-8,200)

rng_seed = 'honeydew' # RNG seed

rescaling = 1 # A factor used to rescale production uniformly accross the parameter space