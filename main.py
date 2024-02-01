from modules import *
import os

def main():

    # Loading of simulated data (LHE)
    data_loading(data_folder)

    # Processing of data and perfoming experimental cuts
    data_processing()

    # Computing N_events 
    computations()

    # Plots
    plotting()

if __name__ == '__main__':
    main()