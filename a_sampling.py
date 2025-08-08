import numpy as np
import math, torch, time, os
import scipy.io as io
from fourier_coefficients import *



np.random.seed(42)
N = 128
N_COEFFS = 50  # number of samples
x = np.linspace(-np.pi, np.pi, N, endpoint=False)


a0 = 2.0  
b_coeffs = np.random.rand(N_COEFFS)
a_coeffs = np.zeros(N_COEFFS)  

reconstructed_signals = np.empty((N_COEFFS, N))


for idx,  (b_val) in enumerate(zip( b_coeffs)):
    coeffs = {
        "a0": a0,
        "a": np.array([0.0]),  
        "b": np.array([b_val])
    }    
    
    reconstructed_signals[idx] = reconstruct_signal_from_coefficients(x, coeffs)

folder = './dataset_produce'    
os.makedirs(folder, exist_ok=True)
np.savetxt(f"{folder}/n_train.csv", reconstructed_signals)
np.savetxt(f"{folder}/f_coeffs.csv", b_coeffs)
    