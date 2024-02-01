from modules import *
import os


# Global parameters
data_folder = 'w2hnl_@1'


current_directory = os.getcwd() # Current path
data_path = os.path.join(current_directory,'data', data_folder) # Data folder path

def main():
    # Loading of simulated data (LHE)
    data_loading()

if __name__ == '__main__':
    main()