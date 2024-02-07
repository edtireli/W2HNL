from modules import *

def main():
    # Loading of simulated data (LHE/HEPMC)
    momenta = data_loading()
    
    # Setting the RNG seed
    randomness(rng_seed) 

    # Processing of data and perfoming experimental cuts
    batch, arrays = data_processing(momenta)

    # Computing N_events 
    production_arrays = computations(momenta, batch, arrays)

    # Plots
    plotting(momenta, batch, production_arrays, arrays)

if __name__ == '__main__':
    main()