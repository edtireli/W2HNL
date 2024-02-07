# A parameter file for various dimensions and variables to do with the experimental constraints, such as cut conditions, efficiencies etc.

pT_minimum = '5 GeV' # Minimum transverse momentum
invmass_minimum = '5 GeV' # Needs implementation for more complex approach
invmass_cut_type = 'trivial' # nontrivial to consider piecewise function, trivial to consider a flat value that uses the above invmass_minimum regardless of r_dv
deltaR_minimum = 0.05 # Minimum angular seperation

pseudorapidity_minimum = 0   # Minimum pseudorapidity criteria for HNL decay 
pseudorapidity_maximum = 2.4 # Maximum pseudorapidity criteria for HNL decay

r_min = '120 mm' # Minimum distance from interaction point (IP) to consider HNL decays/decay volume
r_max_t = '5 m' # Maximum distance from interaction point (IP) to consider HNL decays/decay volume in the transverse direction
r_max_l = '6.5 m' # Maximum distance from interaction point (IP) to consider HNL decays/decay volume in the longitudinal (z) direction along beamline

luminosity = 300 # 1/fb
production_minimum = 3 # The minimum number of HNLs produced within the parameter region to consider it a success (production_minimum = 3 for 95% CL)

