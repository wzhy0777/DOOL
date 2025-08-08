# Import custom modules
from b_train import *  # Physics Informed Neural Networks utilities
import pandas as pd
import torch.nn as nn# Import standard libraries
import os  # Miscellaneous operating system interfaces
import time  # Time access and conversions
import warnings  # Warning control
import argparse
import math, torch, time, os
import numpy as np  # NumPy library for numerical operations
from torch.autograd import grad
from deeponet import *  
from fourier_coefficients import *
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt



def errorFun(output, target, params):
    error = output - target
    error = math.sqrt(torch.mean(error * error))
    ref = math.sqrt(torch.mean(target * target))
    return error / (ref + params["minimal"])

def errorFun_np(output, target, params):
    error = output - target
    error = math.sqrt(np.mean(error * error))
    ref = math.sqrt(np.mean(target * target))
    return error / (ref + params["minimal"])

def fft_derivative(f):
    f = f.flatten()
    F = np.fft.fft(f)                          
    N = len(f)
    k = (np.arange(N) + N // 2) % N - N // 2  
    F_deriv = (1j * k) * F                      
    f_deriv = np.fft.ifft(F_deriv)              
    return f_deriv.real                         


deeponet0 = torch.load(f'DeepONet_training_d4_n70_nx128_k120/Phase_new.pth')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='hyper parameters')
parser.add_argument('--leftx', type=int, default=-1*np.pi, help='the space interval is [leftx, rightx]')
parser.add_argument('--rightx', type=int, default=np.pi, help='the space interval is [leftx, rightx]')
parser.add_argument('--nx', type=int, default=1024, help='Sampling')
parser.add_argument('--seed', type=int, default=111, help='Random seed')
parser.add_argument('--dt', type=float, default=0.001, help='Tau')

args = parser.parse_args()
set_seed(args.seed)

params = dict()
params["leftx"] = args.leftx
params["rightx"] = args.rightx
params["nt"] = 1024
params["nx"] = args.nx
params["minimal"] = 10 ** (-14)
params["seed"] = args.seed
params["t_now"] = 0
params["dt"] = args.dt
params["step"] = 0
Total_T = 1
dt = params["dt"]
steps = 1001
error_l2 = 0



N_test = np.empty((params["nt"], 0)) 
J_test = np.empty((params["nt"], 0))
Test_error_n = np.zeros(steps)

x_train = np.linspace(params["leftx"], params["rightx"], params["nx"] + 1)[0:-1, None]
X_train = torch.from_numpy(x_train).float().to(device) 
X_train =  X_train.requires_grad_(True)
x_test = np.linspace(params["leftx"], params["rightx"], params["nt"] + 1)[0:-1, None]
X_test = torch.from_numpy(x_test).float().to(device) 
X_test =  X_test.requires_grad_(True)
dx_test = x_test[1,0] - x_test[0,0]

n_test = torch.sin(X_test) + 2
n_test = n_test.detach()

branch_input = np.array(np.exp(-1*params["t_now"])).reshape(-1,1)
Branch_input = torch.from_numpy(branch_input).float().to(device)

for step in range(steps):   
    Trunk_input = X_test
    j_pred_test = deeponet0( Branch_input,  Trunk_input).T
    J_test = np.hstack((J_test, j_pred_test.cpu().detach().numpy()))
    
    jx_test = np.gradient(j_pred_test.cpu().detach().numpy().squeeze(),dx_test)
    # jx_test = fft_derivative(j_pred_test.cpu().detach().numpy())
    jx_test = torch.tensor(jx_test).unsqueeze(1).to(device)
    
    Phi_besp = np.exp(-1*params["t_now"])*torch.sin(X_test) + 2
    Phi_besp_j = -1*np.exp(-1*params["t_now"])*torch.cos(X_test)
    test_error_n = errorFun(n_test, Phi_besp, params)
    Test_error_n[step] = test_error_n
    test_error_j = errorFun(j_pred_test, Phi_besp_j, params)
    print('nt: %.2e, test: %.4e,  --------- jt: %.2e, test: %.4e' %
          (params["t_now"], test_error_n,  params["t_now"], test_error_j))
    N_test = np.hstack((N_test, n_test.cpu().numpy()))

    n_test = -1*dt*jx_test + n_test
    n_test[0, 0] = 2
    n_test[-1, 0] = 2
    
    params["t_now"] = params["t_now"] + dt
    coeffs_f, error_l2 = compute_fourier_coefficients(x_test, n_test.cpu().detach().numpy().squeeze(), 1)
    branch_input = np.array(coeffs_f['b'][0]).reshape(-1,1)     
    Branch_input = torch.from_numpy(branch_input).float().to(device)
    
    
t = np.linspace(0, Total_T, steps)
x = np.linspace(-1*np.pi, np.pi, params["nx"]+1)[0:-1, None]
[T, X] = np.meshgrid(t, x)
x_test = np.concatenate([T.flatten()[:, None], X.flatten()[:, None]], axis=1)
t = x_test[:, 0:1]
x = x_test[:, 1:2]
exact_n = np.exp(-1*t)*np.sin(x)+2
exact_j = -1*np.exp(-1*t)*np.cos(x)
test_error_N = errorFun_np(N_test.reshape(-1, 1), exact_n, params)
test_error_J = errorFun_np(J_test.reshape(-1, 1), exact_j, params)
print('nt: %.2e, total_time_test: %.4e, ---------jt: %.2e, total_time_test: %.4e,' %
      (params["t_now"], test_error_N, params["t_now"], test_error_J))

    
folderj = './result_1d_T{T}_dt{dt}'.format(T = Total_T, dt=params["dt"])
os.makedirs(folderj, exist_ok=True)
np.savetxt(f"{folderj}/test_err_N.csv", np.array(test_error_N).reshape(-1,1))
np.savetxt(f"{folderj}/test_err_J.csv", np.array(test_error_J).reshape(-1,1))
np.savetxt(f"{folderj}/j_solution_test.csv", J_test)
np.savetxt(f"{folderj}/n_solution_test.csv", N_test)









