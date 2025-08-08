# Import standard libraries
import os  # Miscellaneous operating system interfaces
import time  # Time access and conversions
import warnings  # Warning control
import argparse

import numpy as np  # NumPy library for numerical operations
from torch.autograd import grad

# Import custom modules
from deeponet import *  # Physics Informed Neural Networks utilities
import pandas as pd
import torch.nn as nn

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='hyper parameters')
parser.add_argument('--d', type=int, default=4, help='depth')
parser.add_argument('--n', type=int, default=70, help='width')
parser.add_argument('--k', type=int, default=120, help='width of the output layer')
parser.add_argument('--nx', type=int, default=128, help='Sampling')

args = parser.parse_args()



def dis_values_to_fft(n):
    c_k = np.fft.fft(n) / len(n)
    return c_k

def fft_derivative(f):
    f = f.flatten()
    F = np.fft.fft(f)  
    N = len(f)
    k = (np.arange(N) + N // 2) % N - N // 2  
    F_deriv = (1j * k) * F  
    f_deriv = np.fft.ifft(F_deriv)  
    return f_deriv.real  


def GetGradients(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True, allow_unused=True)[0]


def errorFun(output, target):
    error = output - target
    error = np.sqrt(np.mean(error * error))
    ref = np.sqrt(np.mean(target * target))
    return error / (ref + 1E-14)


def loss_function(deeponet, f_coeffs, n_train, nx_train, params, device):
    N_train = torch.tensor(n_train).float().to(device)
    Nx_train = torch.tensor(nx_train).float().to(device)
    x = np.linspace(-params["L"], params["L"], params["nx"] + 1)[0:-1, None]
    trunk_input = x
    Branch_input = f_coeffs.float()
    Trunk_input = torch.from_numpy(trunk_input).float().to(device)
    J = deeponet(Branch_input, Trunk_input)
        
    Res = 0.5 * J**2 / N_train + J * Nx_train / N_train
    R = torch.sum(((2*params["L"]) / params["nx"]) * Res, dim=1)
    loss = torch.sum(R)
 
    return loss, J


def train_adam(deeponet, f_coeffs,  n_test, n_train, nx_train, j_star_test, params, device, num_iter=20000):
    x_train = np.linspace(-params["L"], params["L"], params["nx"] + 1)[0:-1, None]
    X_train = torch.from_numpy(x_train).float().to(device)
    x_test = np.linspace(-params["L"], params["L"], params["nt"] + 1)[0:-1, None]
    X_test = torch.from_numpy(x_test).float().to(device)
    optimizer = torch.optim.Adam(deeponet.parameters(), lr=0.001)
    global iter

    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        loss, _ = loss_function(deeponet, f_coeffs,  n_train, nx_train,  params, device)
        loss.backward(retain_graph=True)
        optimizer.step()

        n_test_t1 = n_test[0:1,:].float()
        n_test_t2 = n_test[1:2,:].float()
        n_test_t3 = n_test[2:3,:].float()
        
        J_test_t1 = deeponet(n_test_t1, X_test)
        J_test_t2 = deeponet(n_test_t2, X_test)
        J_test_t3 = deeponet(n_test_t3, X_test)

        j_test_t1 = J_test_t1.cpu().detach().numpy()
        j_test_t2 = J_test_t2.cpu().detach().numpy()
        j_test_t3 = J_test_t3.cpu().detach().numpy()

        j_star_t1 = j_star_test[0:1, : ]
        j_star_t2 = j_star_test[1:2, : ]
        j_star_t3 = j_star_test[2:3, : ]

        error_1 = errorFun(j_test_t1, j_star_t1)
        error_2 = errorFun(j_test_t2, j_star_t2)
        error_3 = errorFun(j_test_t3, j_star_t3)

        results.append([iter, loss.item(), error_1, error_2 , error_3])
        iter += 1
        if iter == 1 or iter % 10000 == 0:
            print(
                f"Train adam - Iter: {iter:.6e} - Loss: {loss.item():.6e} - test t=0: {error_1:.6e} - test t=0.5: {error_2:.6e} - test t=1: {error_3:.6e}")
        
        


if __name__ == "__main__":

    
    set_seed(42)
    np.random.seed(42)
    device = torch.device('cuda')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if not os.path.exists(
            'DeepONet_training_d{}_n{}_nx{}_k{}'.format(args.d, args.n, args.nx, args.k)):
        os.makedirs(
            'DeepONet_training_d{}_n{}_nx{}_k{}'.format(args.d, args.n, args.nx, args.k))

    # Initialize 
    results = []
    iter = 0 
    # set parameters
    params = dict()
    params["k"] = args.k
    params["width"] = args.n
    params["depth"] = args.d
    params["L"] = np.pi
    params["nx"] = args.nx
    params["nt"] = 1024
    
    x = np.linspace(-params["L"], params["L"], params["nx"] + 1)[0:-1, None]
    x_test = np.linspace(-params["L"], params["L"], params["nt"] + 1)[0:-1, None].T
    f_coeffs = pd.read_csv("./dataset_produce/f_coeffs.csv", header=None, delimiter="\s+")
    f_coeffs = f_coeffs.values
    f_coeffs = torch.tensor(f_coeffs).to(device)
       
    n_train = pd.read_csv("./dataset_produce/n_train.csv", header=None, delimiter="\s+")
    n_train = n_train.values
    nx_train = np.stack([fft_derivative(row) for row in n_train])
    
    # The values at the three time points are used for testing
    n_test_1 = np.exp(-0.21)
    n_test_2 = np.exp(-0.61)
    n_test_3 = np.exp(-0.92)
    n_test = n_test = np.array([n_test_1, n_test_2, n_test_3])
    n_test = torch.tensor(n_test).to(device).unsqueeze(1)

    j_star_1 = -1 * n_test_1 * np.cos(x_test)
    j_star_2 = -1 * n_test_2 * np.cos(x_test)
    j_star_3 = -1 * n_test_3 * np.cos(x_test)
    j_star_test = np.concatenate([j_star_1, j_star_2, j_star_3], axis=0)

    branch = MLP(1, params["k"], params["width"], params["depth"])
    branch.apply(init_weights)
    trunk = MLP(1, params["k"], params["width"], params["depth"])
    trunk.apply(init_weights)
    model = DeepONet(branch, trunk).to(device)

    # Adam optimizer
    start_time_adam = time.time()
    train_adam(model, f_coeffs, n_test, n_train, nx_train, j_star_test, params, device, num_iter=50000)
    end_time_adam = time.time()
    training_time = end_time_adam - start_time_adam
    print(f"training time: {training_time:.6e} seconds")

    # Final loss and L2 error
    final_loss = results[-1][1]
    print(f"Final Loss: {final_loss:.6e}")
    final_l2_t1 = results[-1][2]
    print(f"Final error: {final_l2_t1:.6e}")
    final_l2_t2 = results[-1][3]
    print(f"Final error: {final_l2_t2:.6e}")

    # Save training summary
    with open('DeepONet_training_d{}_n{}_nx{}_k{}/Phase_training_summary.txt'.format(args.d, args.n,
                                                                                     args.nx,
                                                                                     args.k),
              'w') as file:
        file.write(f"pre-training training time: {training_time:.6e} seconds\n")
        file.write(f"Total iterations: {iter:.6e}\n")
        file.write(f"Final Loss: {final_loss:.6e}\n")
        file.write(f"Final t=0.5: {final_l2_t1:.6e}\n")
        file.write(f"Final t=1: {final_l2_t2:.6e}\n")

    # Save training data and model state
    results = np.array(results)
    np.savetxt(
        "DeepONet_training_d{}_n{}_nx{}_k{}/Phase_training_data.csv".format(args.d, args.n, args.nx, args.k), results,
        delimiter=",",
        header="Iter,Loss,energy", comments="")
    torch.save(model,
               'DeepONet_training_d{}_n{}_nx{}_k{}/Phase_new.pth'.format(args.d, args.n, args.nx, args.k))
    
    

    
    
    
    
    
