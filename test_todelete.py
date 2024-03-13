import numpy as np
from modules import *
from utils import * 

numpy_file = '/Users/edt/W2HNL/data/w2tau_@1$/Plots/Plot data/survival_dv_array.npy'

numpy_file2 = '/Users/edt/W2HNL/data/w2tau_@1$/Plots/Plot data/average_lorentz_factor.npy'

light_speed = 2.997e8

survival_array = np.load(numpy_file)

lorentz_values_avg = np.load(numpy_file2)
print(np.shape(lorentz_values_avg))

mass_hnl = [i*0.5 for i in range(2,21)]
mixing = np.logspace(0,-8,200)

i=2
j=100

print('mixing: ', mixing[j])
print('mass:   ', mass_hnl[i])

print('P = ', survival_array[i,j])


print(HNL(mass_hnl[i], [0,0,mixing[j]], False).computeNLifetime() * light_speed * 160)

print(survival_array[10,143])