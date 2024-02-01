# A parameter file for various dimensions and variables to do with the raw data previously generated
import os
data_folder = 'w2hnl_@1'

batch_size = 9999 # Number of events generated per run (mass) in MadGraph






current_directory = os.getcwd() # Current path
data_path = os.path.join(current_directory,'data', data_folder) # Data folder path
