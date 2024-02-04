from modules import *

def main():

    # Loading of simulated data (LHE/HEPMC)
    momenta = data_loading()
    
    # Processing of data and perfoming experimental cuts
    batch_data = data_processing(momenta)

    # Computing N_events 
    computations()

    # Plots
    plotting(batch_data)

if __name__ == '__main__':
    main()