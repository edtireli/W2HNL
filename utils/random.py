import hashlib
import numpy as np
from termcolor import colored

def string_to_seed(s):
    """Convert a string to an integer seed."""
    value = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (2**32)
    # print(value) # Print actual seed instead of user string
    return value

def randomness(seed_string):
    seed = string_to_seed(seed_string)
    print('Randomness seed set to: ', colored(seed_string, 'cyan'))
    
    return np.random.seed(seed)
