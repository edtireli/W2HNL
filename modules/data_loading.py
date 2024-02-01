import os


def data_loading(data_folder):
    
    current_directory = os.getcwd()                                         # Current path
    data_path         = os.path.join(current_directory,'data', data_folder) # Data folder path

    if data_type == 'LHE':
        LHE_data_processing(data_path)
    return 0

